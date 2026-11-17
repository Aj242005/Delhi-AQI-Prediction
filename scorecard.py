"""
scorecard.py - record every issued live forecast and grade it once the hour is real.

The API writes the first issue of each (target hour, horizon) pair to
Live/forecast_log.csv, so every row is a forecast at its true lead time: the
+4h row was issued four hours early, the +24h row a full day early.  When that
hour arrives in Live/ground_log.csv, score() joins the two and reports rolling
error, 80% interval coverage, the severe-probability Brier score and skill
against 24-hour persistence.

Three things keep the log honest and small:

* A module lock covers the read-dedupe-append-prune sequence, so two requests
  handled at the same time cannot both append the same target hour.
* The parsed file and its dedupe keys are cached by file signature.  A request
  that issues a known target touches only memory; an unchanged file is never
  re-parsed.
* Rows older than FORECAST_LOG_RETENTION_DAYS (default 45) are dropped when a
  new issue is written, so the file stays at a few thousand rows instead of
  growing for as long as the server lives.

Grading splits the ungraded rows in two: `pending` hours are still in the
future, `missing` hours have already passed without a saved reading (the
server or the feed was down), and those can never be scored.

Everything stays in the two local CSV files; there is no database.
"""
import os
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import ground_feed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.environ.get("FORECAST_LOG_PATH", os.path.join(BASE_DIR, "Live", "forecast_log.csv"))
RETENTION_DAYS = float(os.environ.get("FORECAST_LOG_RETENTION_DAYS", "45"))

COLUMNS = [
    "issued_utc", "mode", "horizon", "lead_hours", "target_iso",
    "base_aqi", "pred", "pi_low", "pi_high", "severe_pct", "trained_at",
]

_LOCK = threading.Lock()
_CACHE = {}  # path -> {"sig": (size, mtime_ns) | None, "df": DataFrame, "keys": set}


def _signature(path):
    try:
        stat = os.stat(path)
        return (stat.st_size, stat.st_mtime_ns)
    except OSError:
        return None


def _empty():
    return pd.DataFrame(columns=COLUMNS)


def _parse(path):
    if _signature(path) is None:
        return _empty()
    try:
        df = pd.read_csv(path, on_bad_lines="skip")
    except Exception:  # an unreadable log must never break scoring
        return _empty()
    if df.empty:
        return _empty()
    df["target_iso"] = pd.to_datetime(df["target_iso"], errors="coerce")
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce")
    df = df.dropna(subset=["target_iso", "pred"])
    # A duplicated pair would double-count an hour in every rolling metric.
    df = df.drop_duplicates(subset=["horizon", "target_iso"], keep="first")
    return df.sort_values("target_iso").reset_index(drop=True)


def _read(path):
    """Parsed log for `path`, re-read only when the file changed on disk."""
    cached = _CACHE.get(path)
    sig = _signature(path)
    if cached is not None and cached["sig"] == sig:
        return cached["df"]
    df = _parse(path)
    _CACHE[path] = {"sig": sig, "df": df,
                    "keys": set(zip(df["horizon"].astype(str), df["target_iso"]))}
    return df


def load(path=None):
    """Snapshot of the log. Treat the frame as read-only."""
    path = path or LOG_PATH
    with _LOCK:
        return _read(path)


def _lead(horizon):
    digits = "".join(ch for ch in str(horizon) if ch.isdigit())
    return int(digits) if digits else 0


def _prune(df, path):
    """Drop rows past the retention window, rewriting the file only when needed."""
    if df.empty or RETENTION_DAYS <= 0:
        return df
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    cutoff = pd.Timestamp(now) - pd.Timedelta(days=RETENTION_DAYS)
    kept = df[df["target_iso"] >= cutoff].reset_index(drop=True)
    if len(kept) == len(df):
        return df
    tmp = path + ".tmp"
    kept.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return kept


def log_issue(wave, base_aqi, pi_low=None, pi_high=None, severe_pct=None,
              trained_at="", path=None, mode="live_ground"):
    """Append the first issue of each (target hour, horizon). Returns rows written."""
    if mode != "live_ground" or not wave:
        return 0
    path = path or LOG_PATH
    last = str(wave[-1].get("horizon", ""))
    issued = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    for point in wave:
        iso = point.get("iso")
        if not iso:
            continue
        horizon = str(point.get("horizon", ""))
        is_last = horizon == last
        rows.append({
            "issued_utc": issued,
            "mode": mode,
            "horizon": horizon,
            "lead_hours": _lead(horizon),
            "target_iso": pd.Timestamp(iso).floor("h").isoformat(),
            "base_aqi": base_aqi,
            "pred": point.get("aqi"),
            "pi_low": pi_low if is_last else np.nan,
            "pi_high": pi_high if is_last else np.nan,
            "severe_pct": severe_pct if is_last and severe_pct is not None else np.nan,
            "trained_at": trained_at,
        })

    with _LOCK:
        df = _read(path)
        keys = _CACHE[path]["keys"]
        fresh = [row for row in rows
                 if (row["horizon"], pd.Timestamp(row["target_iso"])) not in keys]
        if not fresh:
            return 0

        new_df = pd.DataFrame(fresh, columns=COLUMNS)
        # Match the dtypes _parse produces so concat and pruning stay comparable.
        new_df["target_iso"] = pd.to_datetime(new_df["target_iso"])
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        new_file = _signature(path) is None
        new_df.to_csv(path, mode="a", header=new_file, index=False)

        merged = new_df if df.empty else pd.concat([df, new_df], ignore_index=True)
        merged = _prune(merged, path)
        _CACHE[path] = {
            "sig": _signature(path),
            "df": merged,
            "keys": set(zip(merged["horizon"].astype(str), merged["target_iso"])),
        }
        return len(fresh)


def score(days=7, path=None, readings=None, now=None):
    """Join logged forecasts with the live readings that have since arrived."""
    path = path or LOG_PATH
    with _LOCK:
        forecast = _read(path)
        forecast = forecast.copy() if len(forecast) else forecast
    readings = ground_feed.load_log() if readings is None else readings
    now = pd.Timestamp(ground_feed.now_ist()) if now is None else pd.Timestamp(now)

    out = {
        "window_days": int(days), "logged": 0, "scored": 0, "pending": 0, "missing": 0,
        "horizons": [], "coverage_pct": None, "coverage_rows": 0,
        "brier": None, "brier_rows": 0, "severe_hours": 0,
        "reliability": [], "latest_target": None, "note": "",
    }
    if forecast.empty or readings is None or len(readings) == 0:
        out["note"] = "No live forecasts have been issued yet."
        return out

    latest = readings["Datetime"].max()
    window = forecast[forecast["target_iso"] > latest - pd.Timedelta(days=days)]
    out["logged"] = int(len(window))
    if window.empty:
        out["note"] = f"No live forecasts in the last {days} days."
        return out

    merged = window.merge(
        readings[["Datetime", "AQI"]], left_on="target_iso", right_on="Datetime", how="left"
    )
    out["latest_target"] = merged["target_iso"].max().isoformat()

    ungraded = merged[merged["AQI"].isna()]
    # The hour just gone may still be saved a few minutes late; only older
    # targets are treated as permanently missing.
    still_due = ungraded["target_iso"] > now - pd.Timedelta(hours=1)
    out["pending"] = int(still_due.sum())
    out["missing"] = int((~still_due).sum())

    done = merged.dropna(subset=["AQI"]).copy()
    if done.empty:
        out["note"] = (f"{len(merged)} forecast rows logged; {out['pending']} still waiting "
                       f"for their hour and {out['missing']} passed with no saved reading.")
        return out

    done["error"] = done["pred"] - done["AQI"]
    done["abs_error"] = done["error"].abs()
    done["persist_error"] = (done["base_aqi"] - done["AQI"]).abs()
    out["scored"] = int(len(done))

    for lead in sorted(done["lead_hours"].unique()):
        sub = done[done["lead_hours"] == lead]
        mae = float(sub["abs_error"].mean())
        persistence_mae = float(sub["persist_error"].mean())
        skill = None
        if persistence_mae > 0:
            skill = (persistence_mae - mae) / persistence_mae * 100.0
        out["horizons"].append({
            "horizon": f"+{int(lead)}h",
            "rows": int(len(sub)),
            "mae": round(mae, 1),
            "rmse": round(float(np.sqrt((sub["error"] ** 2).mean())), 1),
            "bias": round(float(sub["error"].mean()), 1),
            "persistence_mae": round(persistence_mae, 1),
            "skill_pct": None if skill is None else round(skill, 1),
        })

    band = done.dropna(subset=["pi_low", "pi_high"])
    if len(band):
        inside = (band["AQI"] >= band["pi_low"]) & (band["AQI"] <= band["pi_high"])
        out["coverage_pct"] = round(float(inside.mean() * 100.0), 1)
        out["coverage_rows"] = int(len(band))

    prob = done.dropna(subset=["severe_pct"])
    if len(prob):
        observed = (prob["AQI"] > 400).astype(float)
        predicted = prob["severe_pct"].astype(float) / 100.0
        out["brier"] = round(float(((predicted - observed) ** 2).mean()), 3)
        out["brier_rows"] = int(len(prob))
        out["severe_hours"] = int(observed.sum())
        cut = pd.cut(predicted, bins=[-0.001, 0.2, 0.4, 0.6, 0.8, 1.0],
                     labels=["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"])
        for label, group in prob.groupby(cut, observed=False):
            if not len(group):
                continue
            out["reliability"].append({
                "bin": str(label),
                "predicted_pct": round(float(group["severe_pct"].mean()), 1),
                "observed_pct": round(float((group["AQI"] > 400).mean() * 100.0), 1),
                "rows": int(len(group)),
            })
    return out
