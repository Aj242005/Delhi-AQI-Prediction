"""
scorecard.py - record every issued live forecast and grade it once the hour is real.

The API writes the first issue of each (target hour, horizon) pair to
Live/forecast_log.csv, so every row is a forecast at its true lead time: the
+4h row was issued four hours early, the +24h row a full day early.  When that
hour arrives in Live/ground_log.csv, score() joins the two and reports rolling
error, 80% interval coverage, the severe-probability Brier score and skill
against 24-hour persistence.

Everything stays in the two local CSV files; there is no database and no
network call.  Replay forecasts are never logged, because recorded 2023 hours
can never be graded against today's sensor.
"""
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import ground_feed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.environ.get("FORECAST_LOG_PATH", os.path.join(BASE_DIR, "Live", "forecast_log.csv"))

COLUMNS = [
    "issued_utc", "mode", "horizon", "lead_hours", "target_iso",
    "base_aqi", "pred", "pi_low", "pi_high", "severe_pct", "trained_at",
]


def _lead(horizon):
    digits = "".join(ch for ch in str(horizon) if ch.isdigit())
    return int(digits) if digits else 0


def load(path=None):
    path = path or LOG_PATH
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLUMNS)
    try:
        df = pd.read_csv(path, on_bad_lines="skip")
    except Exception:  # an unreadable log must never break scoring
        return pd.DataFrame(columns=COLUMNS)
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df["target_iso"] = pd.to_datetime(df["target_iso"], errors="coerce")
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce")
    df = df.dropna(subset=["target_iso", "pred"])
    return df.sort_values("target_iso").reset_index(drop=True)


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
            "pi_low": pi_low if is_last else "",
            "pi_high": pi_high if is_last else "",
            "severe_pct": severe_pct if is_last and severe_pct is not None else "",
            "trained_at": trained_at,
        })

    existing = load(path)
    if len(existing):
        seen = set(zip(existing["horizon"].astype(str), existing["target_iso"]))
        rows = [row for row in rows
                if (row["horizon"], pd.Timestamp(row["target_iso"])) not in seen]
    if not rows:
        return 0

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new_file = not os.path.exists(path)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, mode="a", header=new_file, index=False)
    return len(rows)


def score(days=7, path=None, readings=None):
    """Join logged forecasts with the live readings that have since arrived."""
    forecast = load(path)
    readings = ground_feed.load_log() if readings is None else readings
    out = {
        "window_days": int(days), "logged": 0, "scored": 0, "pending": 0,
        "horizons": [], "coverage_pct": None, "coverage_rows": 0,
        "brier": None, "brier_rows": 0, "severe_hours": 0,
        "reliability": [], "latest_target": None, "note": "",
    }
    if forecast.empty or readings is None or len(readings) == 0:
        out["note"] = "No live forecasts have been issued yet."
        return out

    latest = readings["Datetime"].max()
    window = forecast[forecast["target_iso"] > latest - pd.Timedelta(days=days)].copy()
    out["logged"] = int(len(window))
    if window.empty:
        out["note"] = f"No live forecasts in the last {days} days."
        return out

    merged = window.merge(
        readings[["Datetime", "AQI"]], left_on="target_iso", right_on="Datetime", how="left"
    )
    done = merged.dropna(subset=["AQI"]).copy()
    out["pending"] = int(len(merged) - len(done))
    out["latest_target"] = merged["target_iso"].max().isoformat()
    if done.empty:
        out["note"] = (f"{len(merged)} forecast rows logged; none of their target "
                       "hours has a live reading yet.")
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
