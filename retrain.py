"""
retrain.py - fold timestamped live ground readings into the forecast models.

The server appends one real WAQI reading per hour to Live/ground_log.csv.  This
script turns those timestamped rows into the same 17 features the forecaster
uses, combines them with the historical 2017-2023 table, and evaluates the
candidate on a recent *live* holdout.  It never interpolates across the
2023 -> 2026 gap, and it never promotes a model that loses to persistence.

Typical use
-----------
    python retrain.py --if-due          # scheduled / occasional run
    python retrain.py --dry-run         # train and report, write nothing
    python retrain.py --no-promote      # write candidates, keep live models
    python retrain.py --force-promote   # bypass the quality gate (careful)

What --if-due means
-------------------
It waits until at least --min-live-rows hourly readings exist (default 720,
about 30 days), and then at most once every --every-days days, provided at
least --min-new-rows new readings arrived since the last attempt.

What a run does
---------------
1. Builds live features per contiguous hourly segment (gaps > 3 h split a
   segment; small gaps are interpolated exactly like ground_feed.build_window).
2. Trains the six +4h...+24h regressors and the severe-day classifier on
   historical data plus the early part of the live record, weighting live rows
   more heavily so 2026 conditions are not drowned out by 54k older rows.
3. Validates on the next slice of live data and tests on the most recent slice,
   with a 24-hour purge between slices so targets cannot leak across them.
4. Calibrates the 80% interval and the severe probability on the live
   validation slice.
5. Applies a promotion gate (+24h skill over persistence, interval coverage,
   severe recall when severe hours exist), backs up Models/, and only then
   replaces the live model files.
6. Tries to call POST /api/reload_models so a running server picks up the new
   models without a manual restart.

The historical file is never rewritten.  The live log is only read.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

import ground_feed as gf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_HISTORY = os.path.join(BASE_DIR, "Dataset", "ml_ready_data.csv")
DEFAULT_LIVE_LOG = os.path.join(BASE_DIR, "Live", "ground_log.csv")
DEFAULT_MODELS_DIR = os.path.join(BASE_DIR, "Models")

FEATURE_COLS = [
    "AQI", "AQI_Lag_1", "AQI_Lag_3", "AQI_Lag_6", "AQI_Lag_24", "AQI_Lag_48",
    "Rolling_Mean_6", "Rolling_Mean_24", "Rolling_Std_24", "AQI_Diff_24",
    "Temp_C", "Humidity", "Wind_Speed",
    "Hour", "Month", "DayOfWeek", "Is_Weekend",
]
HORIZONS = [4, 8, 12, 16, 20, 24]
SEVERE_CUTOFF = 400.0
STATE_FILE = "retrain_state.json"
CANDIDATE_MODEL_FILES = [f"xgb_horizon_{h}h.json" for h in HORIZONS] + ["xgb_severe_clf.json"]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def utc_now():
    return datetime.now(timezone.utc)


def utc_text(dt=None):
    return (dt or utc_now()).strftime("%Y-%m-%d %H:%M:%S UTC")


def stamp_text(dt=None):
    return (dt or utc_now()).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def num(value, digits=2):
    """JSON-safe rounded float (None instead of NaN/inf)."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return round(value, digits)


def pct(value, digits=1):
    value = num(value, digits)
    return None if value is None else round(value * 100.0, digits)


def range_label(frame):
    if frame is None or len(frame) == 0:
        return "empty"
    first = frame["Datetime"].min().strftime("%d %b %Y")
    last = frame["Datetime"].max().strftime("%d %b %Y")
    return first if first == last else f"{first} - {last}"


def read_state(models_dir):
    path = os.path.join(models_dir, STATE_FILE)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def write_state(models_dir, state):
    path = os.path.join(models_dir, STATE_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, allow_nan=False)


def print_header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def load_historical(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Historical dataset not found: {path}")

    df = pd.read_csv(path, parse_dates=["Datetime"])
    required = {"Datetime", "AQI", "Temp_C", "Humidity", "Wind_Speed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{os.path.basename(path)} is missing columns: {sorted(missing)}")

    if "is_real" not in df.columns:
        df["is_real"] = True
    df["is_real"] = df["is_real"].fillna(False).astype(bool)
    df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce").clip(0.0, 500.0)

    df = (df.dropna(subset=["Datetime", "AQI"])
            .drop_duplicates("Datetime", keep="last")
            .sort_values("Datetime")
            .reset_index(drop=True))

    for h in HORIZONS:
        df[f"Target_{h}h"] = df["AQI"].shift(-h)
        df[f"Real_{h}h"] = df["is_real"].shift(-h).eq(True)

    df["source"] = "historical"
    return df


def _segment_live_log(log_df, max_gap):
    """Split the log wherever more than max_gap hours are missing."""
    log_df = (log_df.dropna(subset=["Datetime", "AQI"])
                    .drop_duplicates("Datetime", keep="last")
                    .sort_values("Datetime")
                    .reset_index(drop=True))
    if log_df.empty:
        return []

    step_hours = log_df["Datetime"].diff() / pd.Timedelta(hours=1)
    segment_id = (step_hours.isna() | (step_hours > max_gap + 1)).cumsum()
    return [part for _, part in log_df.groupby(segment_id, sort=True)]


def _features_for_segment(part, max_gap):
    """Turn one gap-free-enough hourly slice into model rows."""
    idx = pd.date_range(part["Datetime"].min(), part["Datetime"].max(), freq="h")
    s = part.set_index("Datetime").reindex(idx)
    observed = s["AQI"].notna()

    s["AQI"] = pd.to_numeric(s["AQI"], errors="coerce").interpolate(
        method="linear", limit=max_gap, limit_area="inside"
    )

    for col in ("Temp_C", "Humidity", "Wind_Speed"):
        s[col] = pd.to_numeric(s[col], errors="coerce").ffill(limit=6).bfill(limit=6)
        s[col] = s[col].fillna(gf.WEATHER_DEFAULTS[col])

    aqi = s["AQI"].clip(0.0, 500.0)
    f = pd.DataFrame({
        "Datetime": idx,
        "AQI": aqi.to_numpy(),
        "AQI_Lag_1": aqi.shift(1).to_numpy(),
        "AQI_Lag_3": aqi.shift(3).to_numpy(),
        "AQI_Lag_6": aqi.shift(6).to_numpy(),
        "AQI_Lag_24": aqi.shift(24).to_numpy(),
        "AQI_Lag_48": aqi.shift(48).to_numpy(),
        "Rolling_Mean_6": aqi.rolling(6).mean().to_numpy(),
        "Rolling_Mean_24": aqi.rolling(24).mean().to_numpy(),
        "Rolling_Std_24": aqi.rolling(24).std().to_numpy(),
        "AQI_Diff_24": (aqi - aqi.shift(24)).to_numpy(),
        "Temp_C": s["Temp_C"].to_numpy(),
        "Humidity": s["Humidity"].to_numpy(),
        "Wind_Speed": s["Wind_Speed"].to_numpy(),
        "Hour": np.asarray(idx.hour),
        "Month": np.asarray(idx.month),
        "DayOfWeek": np.asarray(idx.dayofweek),
        "Is_Weekend": (np.asarray(idx.dayofweek) >= 5).astype(int),
        "is_real": observed.to_numpy(),
        "source": "live",
    })

    for h in HORIZONS:
        f[f"Target_{h}h"] = aqi.shift(-h).to_numpy()
        f[f"Real_{h}h"] = observed.shift(-h).eq(True).to_numpy()

    f = f.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return f


def build_live_features(log_df, max_gap=gf.MAX_INTERP_GAP):
    """Build feature rows from the timestamped live log, one segment at a time."""
    required = {"Datetime", "AQI", "Temp_C", "Humidity", "Wind_Speed"}
    missing = required - set(log_df.columns)
    if missing:
        raise ValueError(f"Live log is missing columns: {sorted(missing)}")

    segments = _segment_live_log(log_df, max_gap)
    frames = []
    for part in segments:
        f = _features_for_segment(part, max_gap)
        if len(f):
            frames.append(f)
    if not frames:
        return pd.DataFrame(columns=["Datetime"] + FEATURE_COLS + ["source"]), len(segments)

    live = pd.concat(frames, ignore_index=True).sort_values("Datetime").reset_index(drop=True)
    return live, len(segments)


def split_live(live, train_frac, val_frac, purge_hours):
    """Chronological live train/val/test slices with a purge between them."""
    live = live.dropna(subset=["Target_24h"]).sort_values("Datetime").reset_index(drop=True)
    n = len(live)
    if n == 0:
        raise ValueError("No usable live rows (need 48 h of history and a +24 h target).")

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_train = max(0, n_train)
    n_val = max(0, n_val)

    train = live.iloc[:max(0, n_train - purge_hours)]
    val = live.iloc[n_train:max(n_train, n_train + n_val - purge_hours)]
    test = live.iloc[n_train + n_val:]
    if len(test) == 0:
        raise ValueError(f"Live split left no test rows (usable rows: {n}).")
    return train, val, test


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def regressor_params(args, seed):
    return {
        "n_estimators": args.n_estimators,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "n_jobs": -1,
    }


def classifier_params(args, seed, scale_pos_weight):
    return {
        "n_estimators": args.clf_estimators,
        "learning_rate": 0.04,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "random_state": seed,
        "n_jobs": -1,
        "eval_metric": "logloss",
    }


def fit_regressor(X, y, w, args, seed):
    model = xgb.XGBRegressor(**regressor_params(args, seed))
    model.fit(X, y, sample_weight=w)
    return model


def fit_classifier(X, y, w, args, seed, scale_pos_weight):
    model = xgb.XGBClassifier(**classifier_params(args, seed, scale_pos_weight))
    model.fit(X, y, sample_weight=w)
    return model


def regression_row(name, y_all, p_all, y_real, p_real, spread="single fit"):
    return {
        "Model": name,
        "RMSE (All)": num(np.sqrt(mean_squared_error(y_all, p_all)), 2),
        "RMSE (Real Only)": num(np.sqrt(mean_squared_error(y_real, p_real)), 2) if len(y_real) else None,
        "MAE (Real Only)": num(mean_absolute_error(y_real, p_real), 2) if len(y_real) else None,
        "R2 (Real Only)": num(r2_score(y_real, p_real), 3) if len(y_real) > 1 else None,
        "Seed Spread (SD)": spread,
    }


def severe_row(name, y_true, y_pred, split_name):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Method": name,
        "Eval Split": split_name,
        "Precision": num(precision_score(y_true, y_pred, zero_division=0), 3),
        "Recall (Sensitivity)": num(recall_score(y_true, y_pred, zero_division=0), 3),
        "F1-Score": num(f1_score(y_true, y_pred, zero_division=0), 3),
        "Missed Severe Hours (FN)": int(fn),
        "False Alarms (FP)": int(fp),
        "Severe Hours": int(tp + fn),
    }


def calibrate_interval(val, val_pred_24, val_real_mask, q_low=10, q_high=90):
    y = val.loc[val_real_mask, "Target_24h"].to_numpy(dtype=float)
    p = val_pred_24[val_real_mask]
    if len(y) < 2:
        raise ValueError("Not enough real validation rows to calibrate the interval.")
    return {
        "q10_err": num(np.percentile(y - p, q_low), 2),
        "q90_err": num(np.percentile(y - p, q_high), 2),
        "mae_val": num(mean_absolute_error(y, p), 2),
        "val_rows": int(len(y)),
    }


def coverage_pct(y, pred, q10, q90):
    if y is None or len(y) == 0 or q10 is None or q90 is None:
        return None
    low = np.clip(pred + q10, 40.0, 500.0)
    high = np.clip(pred + q90, 40.0, 500.0)
    return num(float(((y >= low) & (y <= high)).mean() * 100.0), 1)


def tune_severe(train, val, args, live_weight, val_real_mask):
    """Tune class weight and raw probability threshold on real live validation rows."""
    mask_train = train["Target_24h"].notna().to_numpy()
    X_train = train.loc[mask_train, FEATURE_COLS]
    y_train = (train.loc[mask_train, "Target_24h"] > SEVERE_CUTOFF).astype(int)
    w_train = np.where(train.loc[mask_train, "source"].eq("live"), live_weight, 1.0)

    X_val = val.loc[val_real_mask, FEATURE_COLS]
    y_val = (val.loc[val_real_mask, "Target_24h"] > SEVERE_CUTOFF).astype(int)

    if y_train.nunique() < 2:
        raise ValueError("Training data contains only one severe/non-severe class.")

    base_neg_pos = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    best = {"f1": -1.0, "weight_mult": 0.65, "threshold": 0.50, "model": None}

    can_tune = y_val.nunique() >= 2 and len(y_val) >= 8
    weight_grid = [0.45, 0.55, 0.65, 0.75, 0.85, 1.0] if can_tune else [0.65]
    threshold_grid = [0.40, 0.45, 0.50, 0.55] if can_tune else [0.50]

    for mult in weight_grid:
        clf = fit_classifier(
            X_train, y_train, w_train, args, args.seed, base_neg_pos * mult
        )
        raw = clf.predict_proba(X_val)[:, 1] if len(X_val) else np.array([])
        for threshold in threshold_grid:
            score = f1_score(y_val, (raw >= threshold).astype(int), zero_division=0) if len(raw) else 0.0
            if score > best["f1"]:
                best.update({"f1": float(score), "weight_mult": mult,
                             "threshold": threshold, "model": clf})

    if best["model"] is None:
        best["model"] = fit_classifier(
            X_train, y_train, w_train, args, args.seed, base_neg_pos * best["weight_mult"]
        )

    return best, X_train, y_train, w_train


def reload_server(url):
    if not url:
        return False, "reload disabled"
    body = b"{}"
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return True, payload.get("trained_at_utc", "reloaded")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Retrain the Anand Vihar forecast models on historical + timestamped live data."
    )
    parser.add_argument("--history", default=DEFAULT_HISTORY, help="Historical feature table (read only).")
    parser.add_argument("--live-log", default=DEFAULT_LIVE_LOG, help="Timestamped live readings CSV.")
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR, help="Where live models and backups live.")
    parser.add_argument("--if-due", action="store_true",
                        help="Only run when enough new live rows have accumulated since the last attempt.")
    parser.add_argument("--dry-run", action="store_true", help="Train and report, write nothing.")
    parser.add_argument("--no-promote", action="store_true",
                        help="Write candidates but keep the current live model files.")
    parser.add_argument("--force-promote", action="store_true",
                        help="Promote even if the quality gate fails (not recommended).")
    parser.add_argument("--min-live-rows", type=int, default=720,
                        help="Minimum saved live rows before a run is allowed (default 720, about 30 days).")
    parser.add_argument("--every-days", type=float, default=7.0,
                        help="Minimum days between --if-due attempts (default 7).")
    parser.add_argument("--min-new-rows", type=int, default=168,
                        help="Minimum new live rows between --if-due attempts (default 168).")
    parser.add_argument("--live-weight", type=float, default=5.0,
                        help="Sample weight for live rows during training (default 5).")
    parser.add_argument("--train-frac", type=float, default=0.60,
                        help="Fraction of live rows used for training (default 0.60).")
    parser.add_argument("--val-frac", type=float, default=0.20,
                        help="Fraction of live rows used for validation (default 0.20).")
    parser.add_argument("--purge-hours", type=int, default=max(HORIZONS),
                        help="Hours dropped between splits so targets cannot leak (default 24).")
    parser.add_argument("--n-estimators", type=int, default=250, help="Trees per horizon model.")
    parser.add_argument("--clf-estimators", type=int, default=200, help="Trees for the severe classifier.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--reload-url", default="http://127.0.0.1:8000/api/reload_models",
                        help="Server endpoint to reload after promotion.")
    parser.add_argument("--no-reload", action="store_true",
                        help="Do not ask a running server to reload after promotion.")
    return parser.parse_args(argv)


def due_check(args, raw_rows, state):
    """Return (is_due, reason)."""
    if raw_rows < args.min_live_rows:
        return False, f"{raw_rows} live rows saved; need {args.min_live_rows} before retraining"

    last_attempt = state.get("last_attempt_utc")
    if not last_attempt:
        return True, "first retrain with enough live rows"

    try:
        last_dt = datetime.strptime(last_attempt, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
    except ValueError:
        return True, "previous state has an unreadable timestamp"

    age_days = (utc_now() - last_dt).total_seconds() / 86400.0
    new_rows = raw_rows - int(state.get("last_live_rows", 0))
    if age_days < args.every_days:
        return False, f"last attempt was {age_days:.1f} days ago; wait {args.every_days:.0f} days"
    if new_rows < args.min_new_rows:
        return False, f"only {new_rows} new rows since the last attempt; need {args.min_new_rows}"
    return True, f"{new_rows} new rows, {age_days:.1f} days since the last attempt"


def main(argv=None):
    args = parse_args(argv)
    started = utc_now()
    print_header("ANAND VIHAR RETRAIN")
    print(f"Started: {utc_text(started)}")

    if not os.path.isdir(args.models_dir):
        raise FileNotFoundError(f"Models directory not found: {args.models_dir}")

    live_log = gf.load_log(args.live_log)
    raw_rows = int(len(live_log))
    print(f"Live log: {args.live_log}")
    print(f"Saved live readings: {raw_rows}")

    state = read_state(args.models_dir)
    if args.if_due:
        due, reason = due_check(args, raw_rows, state)
        if not due:
            print(f"Not due yet: {reason}.")
            return 0
        print(f"Due: {reason}.")

    print(f"Loading historical features: {args.history}")
    historical = load_historical(args.history)
    print(f"Historical rows: {len(historical):,} ({historical['Datetime'].min()} to {historical['Datetime'].max()})")

    live_features, n_segments = build_live_features(live_log)
    print(f"Live feature rows: {len(live_features):,} from {n_segments} contiguous segment(s)")
    if len(live_features) < 3 * max(24, args.purge_hours):
        raise ValueError(
            f"Only {len(live_features)} usable live feature rows. Need at least "
            f"{3 * max(24, args.purge_hours)} (48 h of history + 24 h target + three slices)."
        )

    live_train, live_val, live_test = split_live(
        live_features, args.train_frac, args.val_frac, args.purge_hours
    )
    print(f"Live split: train {len(live_train)} | validation {len(live_val)} | test {len(live_test)}")
    if len(live_val) < 24 or len(live_test) < 24:
        raise ValueError("Live validation/test slices are too small; save more hourly rows.")

    train = pd.concat([historical, live_train], ignore_index=True).sort_values("Datetime").reset_index(drop=True)
    val = live_val.reset_index(drop=True)
    test = live_test.reset_index(drop=True)

    weights_train = np.where(train["source"].eq("live"), args.live_weight, 1.0).astype(float)
    val_real_mask = val["Real_24h"].to_numpy(dtype=bool)
    test_real_mask = test["Real_24h"].to_numpy(dtype=bool)
    print(f"Training rows: {len(train):,} (live weight {args.live_weight}x)")
    print(f"Validation real +24h targets: {int(val_real_mask.sum())} of {len(val)}")
    print(f"Test real +24h targets: {int(test_real_mask.sum())} of {len(test)}")

    # ---- per-horizon regressors -------------------------------------------
    print_header("TRAINING HORIZON MODELS")
    models = {}
    horizon_predictions = {}
    horizon_metrics = []
    naive_test = test["AQI"].to_numpy(dtype=float)
    real_test_y = test.loc[test_real_mask, "Target_24h"].to_numpy(dtype=float)

    for h in HORIZONS:
        target_col = f"Target_{h}h"
        real_col = f"Real_{h}h"
        mask_train = train[target_col].notna().to_numpy()
        X_tr = train.loc[mask_train, FEATURE_COLS]
        y_tr = train.loc[mask_train, target_col].to_numpy(dtype=float)
        w_tr = weights_train[mask_train]

        model = fit_regressor(X_tr, y_tr, w_tr, args, args.seed)
        models[h] = model

        pred_val = np.clip(model.predict(val[FEATURE_COLS]), 40.0, 500.0)
        pred_test = np.clip(model.predict(test[FEATURE_COLS]), 40.0, 500.0)
        horizon_predictions[h] = (pred_val, pred_test)

        real_mask = test[real_col].to_numpy(dtype=bool)
        y_te = test.loc[real_mask, target_col].to_numpy(dtype=float)
        p_te = pred_test[real_mask]
        p_pers = naive_test[real_mask]

        rmse_model = np.sqrt(mean_squared_error(y_te, p_te)) if len(y_te) else None
        rmse_pers = np.sqrt(mean_squared_error(y_te, p_pers)) if len(y_te) else None
        skill = None
        if rmse_model is not None and rmse_pers not in (None, 0):
            skill = (rmse_pers - rmse_model) / rmse_pers * 100.0

        horizon_metrics.append({
            "Horizon": f"+{h}h Ahead",
            "XGBoost RMSE (Real)": num(rmse_model, 2),
            "Persistence RMSE (Real)": num(rmse_pers, 2),
            "XGBoost MAE (Real)": num(mean_absolute_error(y_te, p_te), 2) if len(y_te) else None,
            "Skill vs Persistence": None if skill is None else f"{skill:+.1f}%",
            "Real Test Rows": int(len(y_te)),
        })
        print(f"  +{h:>2}h  RMSE {rmse_model:7.2f}  persistence {rmse_pers:7.2f}  "
              f"skill {skill:+6.1f}%  n={len(y_te)}")

    # ---- calibration and comparison --------------------------------------
    pred_val_24, pred_test_24 = horizon_predictions[24]
    calibration = calibrate_interval(val, pred_val_24, val_real_mask)
    coverage = coverage_pct(
        real_test_y,
        pred_test_24[test_real_mask],
        calibration["q10_err"],
        calibration["q90_err"],
    )

    mask_lr = train["Target_24h"].notna().to_numpy()
    lr = LinearRegression()
    lr.fit(
        train.loc[mask_lr, FEATURE_COLS],
        train.loc[mask_lr, "Target_24h"],
        sample_weight=weights_train[mask_lr],
    )
    lr_test = np.clip(lr.predict(test[FEATURE_COLS]), 40.0, 500.0)

    regression_metrics = [
        regression_row(
            "Naive Baseline (24h Persistence)",
            test["Target_24h"].to_numpy(dtype=float),
            naive_test,
            real_test_y,
            naive_test[test_real_mask],
            "deterministic",
        ),
        regression_row(
            "Linear Regression",
            test["Target_24h"].to_numpy(dtype=float),
            lr_test,
            real_test_y,
            lr_test[test_real_mask],
            "deterministic",
        ),
        regression_row(
            "XGBoost Regressor (+24h)",
            test["Target_24h"].to_numpy(dtype=float),
            pred_test_24,
            real_test_y,
            pred_test_24[test_real_mask],
            "single fit",
        ),
    ]

    # ---- severe classifier -------------------------------------------------
    print_header("TRAINING SEVERE-DAY CLASSIFIER")
    severe_result, X_tr, y_tr, w_tr = tune_severe(
        train, val, args, args.live_weight, val_real_mask
    )
    severe_clf = severe_result["model"]
    val_eval = val.loc[val_real_mask].reset_index(drop=True)
    test_eval = test.loc[test_real_mask].reset_index(drop=True)
    val_raw = severe_clf.predict_proba(val_eval[FEATURE_COLS])[:, 1]
    test_raw = severe_clf.predict_proba(test_eval[FEATURE_COLS])[:, 1]

    threshold = float(severe_result["threshold"])
    y_val_sev = (val_eval["Target_24h"] > SEVERE_CUTOFF).astype(int).to_numpy()
    y_test_sev = (test_eval["Target_24h"] > SEVERE_CUTOFF).astype(int).to_numpy()
    val_sev_pred = (val_raw >= threshold).astype(int)
    test_sev_pred = (test_raw >= threshold).astype(int)

    iso_x, iso_y = None, None
    brier_raw = brier_cal = None
    if y_val_sev.min() != y_val_sev.max():
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(val_raw, y_val_sev)
        iso_x = [round(float(x), 5) for x in iso.X_thresholds_]
        iso_y = [round(float(y), 5) for y in iso.y_thresholds_]
        test_cal = iso.transform(test_raw)
        brier_raw = brier_score_loss(y_test_sev, test_raw)
        brier_cal = brier_score_loss(y_test_sev, test_cal)
    else:
        brier_raw = brier_score_loss(y_test_sev, test_raw)

    severe_classification_metrics = [
        severe_row("Naive Persistence (AQI > 400)", y_val_sev,
                   (val_eval["AQI"] > SEVERE_CUTOFF).astype(int).to_numpy(), "Validation"),
        severe_row("Thresholded XGBRegressor (> 400)", y_val_sev,
                   (pred_val_24[val_real_mask] > SEVERE_CUTOFF).astype(int), "Validation"),
        severe_row("Dedicated XGBClassifier (Tuned)", y_val_sev, val_sev_pred, "Validation"),
        severe_row("Naive Persistence (AQI > 400)", y_test_sev,
                   (test_eval["AQI"] > SEVERE_CUTOFF).astype(int).to_numpy(), "Test"),
        severe_row("Thresholded XGBRegressor (> 400)", y_test_sev,
                   (pred_test_24[test_real_mask] > SEVERE_CUTOFF).astype(int), "Test"),
        severe_row("Dedicated XGBClassifier (Tuned)", y_test_sev, test_sev_pred, "Test"),
    ]

    dedicated_test = severe_classification_metrics[-1]
    severe_recall = dedicated_test["Recall (Sensitivity)"]
    severe_precision = dedicated_test["Precision"]
    print(f"  weight_mult={severe_result['weight_mult']} threshold={threshold:.2f}")
    print(f"  Test severe recall {severe_recall} precision {severe_precision} "
          f"(severe hours: {dedicated_test['Severe Hours']})")

    # ---- promotion gate ----------------------------------------------------
    rmse_24 = next((r["RMSE (Real Only)"] for r in regression_metrics if "XGBoost" in r["Model"]), None)
    pers_rmse_24 = next((r["RMSE (Real Only)"] for r in regression_metrics if "Persistence" in r["Model"]), None)
    skill_24 = None
    if rmse_24 is not None and pers_rmse_24 not in (None, 0):
        skill_24 = (pers_rmse_24 - rmse_24) / pers_rmse_24 * 100.0

    gate_checks = []

    def gate(name, passed, detail):
        gate_checks.append({"check": name, "passed": bool(passed), "detail": detail})

    gate("live_test_rows", int(test_real_mask.sum()) >= 24,
         f"{int(test_real_mask.sum())} real +24h test rows (need 24)")
    gate("beats_persistence", skill_24 is not None and skill_24 > 0,
         "no skill computed" if skill_24 is None else f"{skill_24:+.1f}% RMSE skill vs persistence")
    gate("interval_coverage", coverage is not None and 60.0 <= coverage <= 95.0,
         "coverage unavailable" if coverage is None else f"{coverage:.1f}% empirical 80% coverage (want 60-95%)")
    if dedicated_test["Severe Hours"] >= 5:
        gate("severe_recall", severe_recall is not None and severe_recall >= 0.45,
             f"{pct(severe_recall)}% recall on {dedicated_test['Severe Hours']} severe test hours")
    else:
        gate("severe_recall", True,
             f"only {dedicated_test['Severe Hours']} severe test hours; recall not gated")
    gate("all_horizons", len(models) == len(HORIZONS), f"{len(models)} of {len(HORIZONS)} horizon models")
    gate_passed = all(item["passed"] for item in gate_checks)
    promoted = gate_passed or args.force_promote
    if args.no_promote:
        promoted = False

    eval_label = f"live test ({range_label(test)})"
    val_label = f"live validation ({range_label(val)})"

    metadata = {
        "trained_at_utc": utc_text(),
        "training_mode": "historical_plus_live",
        "dataset_file": os.path.basename(args.history),
        "dataset_sha256": sha256_file(args.history),
        "live_log_file": os.path.basename(args.live_log),
        "live_log_sha256": sha256_file(args.live_log) if os.path.exists(args.live_log) else None,
        "data_start": historical["Datetime"].min().strftime("%Y-%m-%d %H:%M"),
        "data_end": live_features["Datetime"].max().strftime("%Y-%m-%d %H:%M"),
        "data_note": (
            "Historical 2017-2023 plus timestamped live readings. The 2023-2026 gap "
            "is not interpolated; live features are built per contiguous segment."
        ),
        "row_counts": {
            "total": int(len(train) + len(val) + len(test)),
            "train": int(len(train)),
            "val_total": int(len(val)),
            "val_real": int(val_real_mask.sum()),
            "test_total": int(len(test)),
            "test_real": int(test_real_mask.sum()),
            "historical": int(len(historical)),
            "live_feature_rows": int(len(live_features)),
        },
        "split": {
            "train": f"historical + live through {range_label(live_train).split(' - ')[-1]} ({len(train):,} rows)",
            "val": f"{val_label}, {int(val_real_mask.sum())} real of {len(val)}",
            "test": f"{eval_label}, {int(test_real_mask.sum())} real of {len(test)}",
            "purge_hours": int(args.purge_hours),
        },
        "eval_label": eval_label,
        "val_label": val_label,
        "eval_split": "Test",
        "val_split": "Validation",
        "validation_period": range_label(pd.concat([val, test], ignore_index=True)),
        "live_data": {
            "rows": int(raw_rows),
            "feature_rows": int(len(live_features)),
            "segments": int(n_segments),
            "start": live_log["Datetime"].min().strftime("%Y-%m-%d %H:%M"),
            "end": live_log["Datetime"].max().strftime("%Y-%m-%d %H:%M"),
            "weight": float(args.live_weight),
        },
        "library_versions": {
            "xgboost": xgb.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "feature_cols": FEATURE_COLS,
        "horizons": HORIZONS,
        "val_calibration": {
            **calibration,
            "severe_weight_mult": float(severe_result["weight_mult"]),
            "severe_prob_threshold": float(threshold),
            "test_empirical_coverage_pct": coverage,
            "isotonic_x_thresholds": iso_x,
            "isotonic_y_thresholds": iso_y,
            "brier_score_test_uncalibrated": num(brier_raw, 4),
            "brier_score_test_calibrated": num(brier_cal, 4),
        },
        "regression_metrics": regression_metrics,
        "severe_classification_metrics": severe_classification_metrics,
        "horizon_metrics": horizon_metrics,
        "promotion": {
            "gate_passed": bool(gate_passed),
            "promoted": bool(promoted),
            "forced": bool(args.force_promote and not gate_passed),
            "gate_checks": gate_checks,
        },
    }

    print_header("RETRAIN REPORT")
    print(f"+24h MAE           {regression_metrics[-1]['MAE (Real Only)']}")
    print(f"+24h RMSE          {rmse_24}  (persistence {pers_rmse_24})")
    print(f"+24h skill         {None if skill_24 is None else f'{skill_24:+.1f}%'}")
    print(f"80% coverage       {coverage}%")
    print(f"Severe recall      {pct(severe_recall)}%  (precision {pct(severe_precision)}%)")
    print(f"Gate               {'PASS' if gate_passed else 'FAIL'}")
    for item in gate_checks:
        print(f"  [{'x' if item['passed'] else ' '}] {item['check']}: {item['detail']}")

    if args.dry_run:
        print("\nDry run: no files written, no state updated.")
        return 0 if gate_passed else 2

    stamp = stamp_text()
    candidate_dir = os.path.join(args.models_dir, "candidates", stamp)
    os.makedirs(candidate_dir, exist_ok=True)
    for h in HORIZONS:
        models[h].get_booster().save_model(os.path.join(candidate_dir, f"xgb_horizon_{h}h.json"))
    severe_clf.get_booster().save_model(os.path.join(candidate_dir, "xgb_severe_clf.json"))

    backup_dir = None
    if promoted:
        backup_dir = os.path.join(args.models_dir, "backups", stamp)
        os.makedirs(backup_dir, exist_ok=True)
        for name in os.listdir(args.models_dir):
            src = os.path.join(args.models_dir, name)
            if os.path.isfile(src) and name.endswith(".json"):
                shutil.copy2(src, os.path.join(backup_dir, name))

    metadata["promotion"]["candidate_dir"] = os.path.relpath(candidate_dir, args.models_dir)
    metadata["promotion"]["backup_dir"] = (
        None if backup_dir is None else os.path.relpath(backup_dir, args.models_dir)
    )
    with open(os.path.join(candidate_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, allow_nan=False)

    if promoted:
        for name in CANDIDATE_MODEL_FILES + ["model_metadata.json"]:
            shutil.copy2(os.path.join(candidate_dir, name), os.path.join(args.models_dir, name))

    # Keep only the five newest backups.
    backups_root = os.path.join(args.models_dir, "backups")
    if os.path.isdir(backups_root):
        backups = sorted(d for d in os.listdir(backups_root)
                         if os.path.isdir(os.path.join(backups_root, d)))
        for old in backups[:-5]:
            shutil.rmtree(os.path.join(backups_root, old), ignore_errors=True)

    write_state(args.models_dir, {
        "last_attempt_utc": utc_text(),
        "last_live_rows": int(raw_rows),
        "last_live_feature_rows": int(len(live_features)),
        "gate_passed": bool(gate_passed),
        "promoted": bool(promoted),
        "test_mae_24h": regression_metrics[-1]["MAE (Real Only)"],
        "test_rmse_24h": rmse_24,
        "test_skill_pct": num(skill_24, 1),
        "test_coverage_pct": coverage,
        "eval_label": eval_label,
        "trained_at_utc": metadata["trained_at_utc"],
    })

    print(f"\nCandidate: {candidate_dir}")
    if promoted:
        print(f"Backup:    {backup_dir}")
        print(f"Promoted models to: {args.models_dir}")
        ok, detail = reload_server("" if args.no_reload else args.reload_url)
        print(f"Server reload: {'ok - ' + str(detail) if ok else 'not reloaded (' + str(detail) + ')'}")
        print("If the server did not reload, restart it to use the new models.")
    else:
        print("Models were NOT promoted. Inspect the candidate and the gate above.")
        print("Use --force-promote only if you understand why the gate failed.")

    if not gate_passed and not args.no_promote and not args.force_promote:
        return 2
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # keep scheduled runs readable
        print(f"\nERROR: {exc}")
        sys.exit(1)
