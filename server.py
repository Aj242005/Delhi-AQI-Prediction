import os
import sys
import json
import logging
import subprocess
import threading
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import xgboost as xgb
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import ground_feed  # live ground-sensor feed (ground_feed.py)
import scorecard  # forecast scorecard (scorecard.py)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("anand_vihar_server")

app = FastAPI(title="Anand Vihar AQI Forecast & Advisory Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANAND_VIHAR_LAT = 28.6469
ANAND_VIHAR_LON = 77.3160

# The WAQI token is read from the environment or a local .env file - never commit it.
#   .env file:        WAQI_TOKEN=your-token
#   PowerShell:       $env:WAQI_TOKEN = "your-token"
#   Anaconda Prompt:  set WAQI_TOKEN=your-token
WAQI_TOKEN = os.environ.get("WAQI_TOKEN", "").strip()

# Saves one real ground-sensor reading per hour while the server is running.
ground_feed.start_background_logger(WAQI_TOKEN)

# ---------------------------------------------------------
# 1. CPCB CATEGORY HELPERS
# ---------------------------------------------------------
def get_cpcb_meta(aqi):
    if aqi <= 50:
        return "Good (0–50)", "#10b981", "Clean Atmospheric Dispersion"
    elif aqi <= 100:
        return "Satisfactory (51–100)", "#84cc16", "Satisfactory Air Quality"
    elif aqi <= 200:
        return "Moderate (101–200)", "#eab308", "Moderate Particulate Accumulation"
    elif aqi <= 300:
        return "Poor (201–300)", "#f97316", "Poor Air Quality • Boundary Trap"
    elif aqi <= 400:
        return "Very Poor (301–400)", "#ef4444", "Very Poor • Thermal Inversion Smog"
    else:
        return "Severe (401–500)", "#dc2626", "Severe Smog Emergency (>400)"


def calibration_dict(meta):
    """Support both the original 2023 metadata and retrained live metadata."""
    return meta.get("val_calibration") or meta.get("val_2022_calibration") or {}


def calibration_coverage(cal):
    value = cal.get("test_empirical_coverage_pct")
    if value is None:
        value = cal.get("test_2023_empirical_coverage_pct")
    return value


def format_chance_phrase(cal_prob_pct, pred_24h, is_triggered):
    """Plain-language severe-day phrase. The wording always matches the calibrated percentage shown."""
    p = float(cal_prob_pct)
    if p >= 95.0:
        chance = f"near-certain ({p:.0f}%)"
    else:
        chance = f"about {int(round(p / 10.0))} in 10 chance ({p:.0f}%)"
    if p >= 70.0:
        return f"High severe-day risk, {chance}"
    elif p >= 45.0:
        return f"Elevated severe-day risk, {chance}"
    elif is_triggered or p >= 25.0:
        return f"Possible severe day, {chance}"
    elif p >= 10.0:
        return f"Low severe-day risk, {chance}"
    else:
        return f"Unlikely severe day (<10% chance; {p:.1f}%)"


def likely_driver(aqi, wind, hour, month, pm25=None, pm10=None):
    """Heuristic only: what usually explains an hour like this at Anand Vihar.

    Nothing here is measured source apportionment; it reads the wind, the clock,
    the season and the coarse/fine particle split and names the usual suspect.
    """
    fine_share = None
    try:
        if pm25 is not None and pm10 is not None and float(pm10) > 0 and not pd.isna(pm10):
            fine_share = float(pm25) / float(pm10)
    except (TypeError, ValueError):
        fine_share = None
    still = wind is not None and not pd.isna(wind) and float(wind) < 8
    rush = hour in (8, 9, 10, 11, 18, 19, 20, 21)
    winter = month in (10, 11, 12, 1)

    if aqi >= 200 and still and (hour >= 21 or hour <= 7):
        return {"label": "Night inversion",
                "detail": "Still, cool air under a shallow boundary layer is holding particles near the ground."}
    if winter and aqi >= 250 and fine_share is not None and fine_share >= 0.55:
        return {"label": "Smoke, likely stubble and local burning",
                "detail": "Fine particles dominate; from October to January that points to crop-residue and waste burning."}
    if rush and aqi >= 150:
        return {"label": "Traffic build-up",
                "detail": "The reading lands in a commute peak, when vehicle exhaust and road dust stack up."}
    if wind is not None and not pd.isna(wind) and float(wind) >= 18:
        return {"label": "Wind-driven dust",
                "detail": "Stronger wind is lifting coarse dust, so PM10 usually leads this hour."}
    if aqi >= 200:
        return {"label": "Regional accumulation",
                "detail": "Slow dispersion over the Indo-Gangetic plain is letting pollution build up."}
    return {"label": "Mixed sources",
            "detail": "No single driver stands out at this hour."}


def drift_summary(days=7):
    """Compare the last few days of live readings with the training distribution."""
    if df_hist is None or df_hist.empty:
        return {"available": False, "note": "Replay dataset is missing, so there is no training baseline."}
    log_df = ground_feed.load_log()
    have = 0 if log_df is None else int(len(log_df))
    if have < 24:
        return {"available": False, "note": f"Needs 24 saved live hours ({have} so far)."}

    latest = log_df["Datetime"].max()
    live = log_df[log_df["Datetime"] > latest - pd.Timedelta(days=days)]
    out = {"available": True, "hours": int(len(live)), "window_days": int(days), "vars": []}
    for col, label in (("AQI", "AQI"), ("Temp_C", "Temperature (°C)"),
                       ("Humidity", "Humidity (%)"), ("Wind_Speed", "Wind (km/h)")):
        train = pd.to_numeric(df_hist[col], errors="coerce").dropna()
        now = pd.to_numeric(live[col], errors="coerce").dropna()
        if len(train) < 100 or not len(now):
            continue
        live_mean = float(now.mean())
        train_mean, train_std = float(train.mean()), float(train.std())
        z = None if train_std == 0 else (live_mean - train_mean) / train_std
        out["vars"].append({
            "label": label,
            "live_mean": round(live_mean, 1),
            "train_mean": round(train_mean, 1),
            "z": None if z is None else round(z, 2),
            "in_range": bool(train.min() <= live_mean <= train.max()),
        })
    return out

# ---------------------------------------------------------
# 2. LOAD NATIVE XGBOOST JSON MODELS & METADATA
# ---------------------------------------------------------
MODELS_DIR = os.path.join(BASE_DIR, "Models")
META_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
DATA_PATH = os.path.join(BASE_DIR, "Dataset", "ml_ready_data.csv")


def load_history():
    """Replay data is optional at import time: live mode can run without it."""
    if not os.path.exists(DATA_PATH):
        logger.warning("Replay dataset not found at %s - run 'python build_dataset.py' to create it.", DATA_PATH)
        return None
    df = pd.read_csv(DATA_PATH, parse_dates=["Datetime"])
    logger.info("Loaded replay history: %d rows (%s to %s)", len(df),
                df["Datetime"].min(), df["Datetime"].max())
    return df


df_hist = load_history()


def load_bundle_state():
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    h_boosters = {}
    for h in meta["horizons"]:
        jpath = os.path.join(MODELS_DIR, f"xgb_horizon_{h}h.json")
        if os.path.exists(jpath):
            b = xgb.Booster()
            b.load_model(jpath)
            h_boosters[h] = b
    sev_path = os.path.join(MODELS_DIR, "xgb_severe_clf.json")
    sev_b = None
    if os.path.exists(sev_path):
        sev_b = xgb.Booster()
        sev_b.load_model(sev_path)
    return meta, h_boosters, sev_b


metadata, horizon_boosters, severe_booster = load_bundle_state()

# ---------------------------------------------------------
# 2b. AUTO-RETRAIN - fold timestamped live readings back in while the server runs
# ---------------------------------------------------------
RETRAIN_SCRIPT = os.path.join(BASE_DIR, "retrain.py")
RETRAIN_CHECK_HOURS = float(os.environ.get("RETRAIN_CHECK_HOURS", "6"))
RETRAIN_FIRST_DELAY = 5 * 60      # let startup and the first live poll settle
_retrain_thread = None


def swap_bundle():
    """Re-read Models/ into the running process after a promotion."""
    global metadata, horizon_boosters, severe_booster
    metadata, horizon_boosters, severe_booster = load_bundle_state()
    logger.info("Reloaded models trained at %s", metadata.get("trained_at_utc"))


def _metadata_stamp():
    try:
        return os.path.getmtime(META_PATH)
    except OSError:
        return None


def retrain_once():
    """Run 'retrain.py --if-due'. True when a promoted bundle was swapped in."""
    before = _metadata_stamp()
    proc = subprocess.run(
        [sys.executable, RETRAIN_SCRIPT, "--if-due"], cwd=BASE_DIR,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=6 * 3600,
    )
    output = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if proc.returncode != 0:
        logger.warning("Auto-retrain stopped (exit %s): %s", proc.returncode, output[-1500:])
        return False
    # retrain.py rewrites Models/model_metadata.json only when it promotes.
    if _metadata_stamp() == before:
        return False
    swap_bundle()
    logger.info("Auto-retrain promoted a new bundle: %s", metadata.get("trained_at_utc"))
    return True


def start_auto_retrain():
    """Background due-check, so retraining needs no external scheduler."""
    global _retrain_thread
    if os.environ.get("RETRAIN_AUTO", "1") == "0":
        logger.info("Auto-retrain disabled (RETRAIN_AUTO=0)")
        return None
    if _retrain_thread is not None and _retrain_thread.is_alive():
        return _retrain_thread

    def loop():
        time.sleep(RETRAIN_FIRST_DELAY)
        while True:
            try:
                retrain_once()
            except Exception as exc:  # a failed retrain must never stop the server
                logger.warning("Auto-retrain failed: %s", exc)
            time.sleep(max(300.0, RETRAIN_CHECK_HOURS * 3600.0))

    _retrain_thread = threading.Thread(target=loop, name="auto-retrain", daemon=True)
    _retrain_thread.start()
    logger.info("Auto-retrain: checking every %.1f h (disable with RETRAIN_AUTO=0)", RETRAIN_CHECK_HOURS)
    return _retrain_thread


start_auto_retrain()

# ---------------------------------------------------------
# 3. PROFILE-AWARE CPCB HEALTH ENGINE (ENGLISH ONLY)
# ---------------------------------------------------------
def get_health_advisory(aqi_val, profile):
    if aqi_val <= 100:
        band = "0_100"
    elif aqi_val <= 200:
        band = "101_200"
    elif aqi_val <= 300:
        band = "201_300"
    elif aqi_val <= 400:
        band = "301_400"
    else:
        band = "401_500"

    profile_protocols = {
        "Healthy Adult": {
            "0_100": ("No mask needed outdoors", "Open windows freely for natural ventilation"),
            "101_200": ("Optional near heavy road dust", "Ventilate between 12:00 PM – 4:00 PM"),
            "201_300": ("N95 / FFP2 recommended for long commutes", "Keep windows closed during morning/evening rush hours"),
            "301_400": ("Fitted N95 respirator required outdoors", "Keep windows closed; run HEPA filter if available"),
            "401_500": ("Mandatory tight-seal N95/N99 respirator", "Seal windows completely + continuous HEPA filtration")
        },
        "Child (Under 12)": {
            "0_100": ("No mask required for play", "Open school/home windows normally"),
            "101_200": ("Pediatric surgical/cloth mask near traffic", "Ventilate classrooms only during mid-day sun (1–3 PM)"),
            "201_300": ("Pediatric N95 mask during school transit", "Keep nursery/bedroom windows closed; use indoor play areas"),
            "301_400": ("Mandatory child-fitted N95 for any transit", "Seal bedroom windows; operate HEPA purifier in sleeping areas"),
            "401_500": ("Avoid outdoor transit; strict pediatric N95 if emergency", "Strictly sealed indoor environment with HEPA filtration")
        },
        "Elderly (60+ Years)": {
            "0_100": ("No mask required", "Normal household ventilation"),
            "101_200": ("Light mask recommended near bus corridors", "Open windows only between 12:00 PM – 3:30 PM"),
            "201_300": ("Fitted N95 mask (avoid valved masks if cardiac issue)", "Keep bedroom windows closed overnight & early morning"),
            "301_400": ("Mandatory N95 outdoors; minimize walking effort", "Keep home sealed; maintain warm, filtered indoor air"),
            "401_500": ("Remain indoors; N95 required even for brief courtyard exposure", "Full room sealing + continuous HEPA air purification")
        },
        "Asthma Patient": {
            "0_100": ("No mask needed; carry rescue inhaler", "Normal ventilation (avoid dust sweeping indoors)"),
            "101_200": ("N95 respirator recommended near traffic + rescue inhaler", "Keep bedroom windows closed during night inversion"),
            "201_300": ("Mandatory N95 outdoors + spacer/rescue inhaler on person", "Keep windows sealed; run HEPA filter in bedroom"),
            "301_400": ("Strict N95/FFP2 + controller medication compliance", "Continuous HEPA filtration; zero outside air exchange"),
            "401_500": ("Emergency indoor isolation; N95/N99 + nebulizer/inhaler ready", "Hermetically sealed room with continuous HEPA purification")
        },
        "Outdoor Worker": {
            "0_100": ("Standard dust mask only if sweeping/digging", "Rest in shaded open areas"),
            "101_200": ("Particulate mask (FFP1/N95) at highway intersections", "Take breaks away from idling diesel vehicles"),
            "201_300": ("Valved N95 respirator (reduces heat buildup during labor)", "Take 10-min sheltered indoor breaks every 2 hours"),
            "301_400": ("Mandatory industrial N95/FFP2 respirator with tight seal", "Mandatory 15-min filtered indoor shelter breaks every 90 mins"),
            "401_500": ("Mandatory N99 cartridge/respirator; suspend heavy manual labor", "Relocate shift tasks indoors or to enclosed booths")
        }
    }

    rules = {
        "Healthy Adult": {
            "0_100": "CPCB Good/Satisfactory (≤100): Minimal health impact. Safe for normal outdoor exercise and commuting.",
            "101_200": "CPCB Moderate (101–200): Acceptable for general activity, but reduce prolonged endurance workouts near Anand Vihar ISBT traffic.",
            "201_300": "CPCB Poor (201–300): Breathing discomfort possible on prolonged exposure. Replace outdoor running with indoor exercise.",
            "301_400": "CPCB Very Poor (301–400): Respiratory illness risk on prolonged exposure. Avoid outdoor cardio and wear an N95 mask during transit.",
            "401_500": "CPCB Severe (401–500): Affects healthy individuals and seriously impacts those with existing diseases. Eliminate non-essential outdoor exposure."
        },
        "Child (Under 12)": {
            "0_100": "CPCB ≤100: Safe for school playground activities and outdoor sports.",
            "101_200": "CPCB Moderate (101–200): Limit continuous high-exertion outdoor sports to under 60 minutes.",
            "201_300": "CPCB Poor (201–300): Children inhale more air per kg of body weight (WHO, 2021). Shift intense sports indoors.",
            "301_400": "CPCB Very Poor (301–400): Suspend all outdoor playground sessions. Use a pediatric N95 mask during school commute.",
            "401_500": "CPCB Severe (401–500): High risk of acute airway inflammation. Keep children strictly indoors in filtered rooms."
        },
        "Elderly (60+ Years)": {
            "0_100": "CPCB ≤100: Safe for routine morning and evening walks.",
            "101_200": "CPCB Moderate (101–200): Prefer late-morning walks (after 10:00 AM) when surface haze disperses.",
            "201_300": "CPCB Poor (201–300): Avoid early-morning walks during thermal inversion. Monitor blood pressure if hypertensive.",
            "301_400": "CPCB Very Poor (301–400): Remain indoors and avoid outdoor physical exertion.",
            "401_500": "CPCB Severe (401–500): Strictly remain indoors. Seek clinical care immediately if experiencing chest tightness."
        },
        "Asthma Patient": {
            "0_100": "CPCB ≤100: Low risk, though rescue bronchodilator inhaler should remain accessible.",
            "101_200": "CPCB Moderate (101–200): Lungs of asthma patients may experience mild constriction. Keep rescue inhaler on person.",
            "201_300": "CPCB Poor (201–300): High risk of particulate-induced wheezing. Wear an N95 mask outdoors and avoid traffic corridors.",
            "301_400": "CPCB Very Poor (301–400): Pronounced respiratory symptoms likely. Adhere to controller medication and stay indoors.",
            "401_500": "CPCB Severe (401–500): Critical risk of acute bronchospasm. Avoid all outdoor exposure and keep emergency medical contact ready."
        },
        "Outdoor Worker": {
            "0_100": "CPCB ≤100: Normal occupational conditions. Maintain routine hydration.",
            "101_200": "CPCB Moderate (101–200): Use a basic particulate mask if working directly adjacent to road dust or construction.",
            "201_300": "CPCB Poor (201–300): Wear a valved N95 respirator during peak traffic hours (08:00–11:00 and 18:00–21:00).",
            "301_400": "CPCB Very Poor (301–400): Continuous N95 protection required. Schedule sheltered indoor rest breaks every 90 minutes.",
            "401_500": "CPCB Severe (401–500): Severe occupational hazard. Mandatory N99 respirator; suspend heavy manual labor outdoors."
        }
    }

    prof_key = profile if profile in rules else "Healthy Adult"
    mask_rec, vent_rec = profile_protocols[prof_key][band]
    return {
        "advice": rules[prof_key][band],
        "mask": mask_rec,
        "ventilation": vent_rec
    }

# ---------------------------------------------------------
# 4. LIVE MODE - "NOT READY YET" ANSWER (no forecast, honest message)
# ---------------------------------------------------------
def _not_ready_payload(state, profile):
    """Shown while the 49-hour live history is missing: current reading only, no forecast."""
    have, need, reason = state["have"], state["need"], state["reason"]
    token_ok = ground_feed.has_token(WAQI_TOKEN)
    latest = state.get("latest_row")
    latest_str = "" if state.get("latest") is None else state["latest"].strftime("%d %b %Y, %H:%M IST")
    age = state.get("age_hours") or 0.0

    if not token_ok and reason in ("no_data", "stale"):
        level, headline = "no_token", "Live ground feed is not set up"
        banner = ("⚙️ LIVE FEED NOT SET UP: add your free WAQI token (environment variable WAQI_TOKEN) and "
                  "restart the server. Until then, choose Replay mode to see how the forecast works.")
        note = "No live readings are being saved because no WAQI token was found."
    elif reason == "no_data":
        level, headline = "collecting", "Waiting for the first live reading"
        banner = "⏳ WAITING FOR THE FIRST LIVE READING from DPCC Anand Vihar. The server saves one reading per hour."
        note = "The first reading should appear within a minute or two of starting the server."
    elif reason == "stale":
        level, headline = "stale", "Live feed is not updating"
        banner = ("⚠️ LIVE FEED NOT UPDATING: the newest saved reading is about %d h old (%s). "
                  "Forecast paused so it is not built on old data." % (round(age), latest_str))
        note = "Check that the server (or ground_logger.py) is running and the internet is on."
    elif reason == "gaps":
        level, headline = "collecting", "Too many missing hours in the last 48 h"
        banner = ("⚠️ GAPS IN LIVE HISTORY: only %d of the last %d hours are saved. "
                  "Forecast paused until the gaps leave the 48-hour window." % (have, need))
        note = "The server must stay running so it can save a reading every hour."
    else:
        level, headline = "collecting", "Forecast warming up"
        banner = ("⏳ COLLECTING LIVE GROUND READINGS: %d of %d hours saved. The forecast unlocks once %d hours "
                  "are saved; until then only the current reading is shown." % (have, need, need))
        note = ("The model needs the last 48 hours of real sensor readings. Keep the server running - "
                "it saves one reading per hour.")

    show_cur = reason in ("collecting", "gaps") and latest is not None
    if show_cur:
        cur_aqi = int(round(float(latest["AQI"])))
        cur_cat, cur_col, _ = get_cpcb_meta(cur_aqi)
        advisory = get_health_advisory(cur_aqi, profile)
        wx = {"temp": latest.get("Temp_C"), "humidity": latest.get("Humidity"), "wind": latest.get("Wind_Speed")}
        wx = {k: (None if pd.isna(v) else round(float(v), 1)) for k, v in wx.items()}
    else:
        cur_aqi, cur_cat, cur_col = None, "No fresh reading", "#64748b"
        advisory = {"advice": "Advice appears here as soon as a fresh live reading is available.",
                    "mask": "—", "ventilation": "—"}
        wx = {"temp": None, "humidity": None, "wind": None}

    return {
        "source_warning_level": level, "source_banner": banner, "source_mode": "live_ground",
        "forecast_available": False, "collect_have": have, "collect_need": need, "collect_note": note,
        "is_replay": False, "replay_label": "", "reading_time": latest_str if show_cur else "",
        "current_time_iso": (state["latest"].isoformat() if show_cur else ""),
        "current_aqi": cur_aqi, "cur_cat": cur_cat, "cur_col": cur_col, "headline": headline,
        "dual_sub": "", "advisory": advisory,
        "temp": wx["temp"], "humidity": wx["humidity"], "wind": wx["wind"],
        "coverage_test_pct": calibration_coverage(calibration_dict(metadata)),
        "metadata": metadata,
    }

# ---------------------------------------------------------
# 5. REST API ENDPOINT
#    mode = "live_ground"  : real DPCC Anand Vihar sensor readings saved by this server (default)
#    mode = "archive_2023" : replay of a recorded 2023 window (NOT live)
# ---------------------------------------------------------
@app.get("/api/predict")
def predict_aqi(
    profile: str = Query("Healthy Adult"),
    mode: str = Query("live_ground"),
    archive_idx: int = Query(7400)
):
    meta = metadata
    feature_cols = meta["feature_cols"]
    horizons = meta["horizons"]
    val_cal = calibration_dict(meta)

    replay_label = ""
    reading_time_str = ""
    dual_sub = ""
    pm25_v = pm10_v = None

    if mode == "live_ground":
        log_df = ground_feed.load_log()
        state = ground_feed.build_window(log_df, ground_feed.now_ist())
        if not state["ready"]:
            return JSONResponse(_not_ready_payload(state, profile))

        df_past = state["df_past"]
        source_warning_level = "live_ground"
        reading_time_str = df_past["Datetime"].iloc[-1].strftime("%d %b %Y, %H:%M IST")
        notes = []
        if state["n_interpolated"]:
            notes.append("%d missing hour(s) filled with a straight line" % state["n_interpolated"])
        if state["weather_filled"]:
            notes.append("weather was unavailable, typical values used")
        data_source_banner = (
            "🟢 LIVE GROUND SENSOR: DPCC Anand Vihar via WAQI, reading of " + reading_time_str + ". "
            "The model is fed this latest hourly reading (it was trained on hourly values); the previous 48 hours "
            "are real sensor readings saved by this server. Unvalidated live data - can differ a little from "
            "CPCB's official bulletin."
            + ((" Notes: " + "; ".join(notes) + ".") if notes else "")
        )

        # Display-only extras: raw particle readings and an approximate 24-hour average.
        # They are NOT given to the model.
        lr = state["latest_row"]
        avg24, n_avg = ground_feed.avg_24h(log_df, state["latest"])
        parts = []
        if avg24 is not None:
            parts.append(f"24h avg ≈ {avg24:.0f} AQI ({n_avg} h saved)")
        else:
            parts.append(f"24h avg: needs 16 saved hours ({n_avg} so far)")
        pm25_v, pm10_v = lr.get("pm25_ugm3"), lr.get("pm10_ugm3")
        if pm25_v is not None and not pd.isna(pm25_v):
            parts.append(f"PM2.5 {float(pm25_v):.0f} µg/m³")
        if pm10_v is not None and not pd.isna(pm10_v):
            parts.append(f"PM10 {float(pm10_v):.0f} µg/m³")
        dual_sub = " • ".join(parts)
    else:
        mode = "archive_2023"
        if df_hist is None or df_hist.empty:
            return JSONResponse(status_code=503, content={
                "detail": ("Replay dataset is missing. Run 'python build_dataset.py' in the project "
                           "folder (it rebuilds Dataset/ml_ready_data.csv), then restart the server.")
            })
        df_2023 = df_hist[df_hist["Datetime"].dt.year == 2023].reset_index(drop=True)
        safe_idx = max(50, min(int(archive_idx), len(df_2023) - 25))
        df_past = df_2023.iloc[safe_idx - 48: safe_idx + 1][["Datetime", "AQI", "Temp_C", "Humidity", "Wind_Speed"]].reset_index(drop=True)
        hist_dt_str = df_past["Datetime"].iloc[-1].strftime("%d %b %Y, %H:%M IST")
        replay_label = f"REPLAY · {hist_dt_str}"
        source_warning_level = "replay"
        data_source_banner = (
            f"🕘 REPLAY - NOT LIVE: a recorded DPCC Anand Vihar reading from {hist_dt_str} (2023 test set), "
            "shown to demonstrate how the model works. This is not today's air quality."
        )

    current_row = df_past.iloc[-1]
    current_aqi = float(current_row["AQI"])
    current_dt = current_row["Datetime"]

    feat_dict = {
        "AQI": current_aqi,
        "AQI_Lag_1": float(df_past["AQI"].iloc[-2]),
        "AQI_Lag_3": float(df_past["AQI"].iloc[-4]),
        "AQI_Lag_6": float(df_past["AQI"].iloc[-7]),
        "AQI_Lag_24": float(df_past["AQI"].iloc[-25]),
        "AQI_Lag_48": float(df_past["AQI"].iloc[-49]),
        "Rolling_Mean_6": float(df_past["AQI"].tail(6).mean()),
        "Rolling_Mean_24": float(df_past["AQI"].tail(24).mean()),
        "Rolling_Std_24": float(df_past["AQI"].tail(24).std()),
        "AQI_Diff_24": float(current_aqi - df_past["AQI"].iloc[-25]),
        "Temp_C": float(current_row["Temp_C"]),
        "Humidity": float(current_row["Humidity"]),
        "Wind_Speed": float(current_row["Wind_Speed"]),
        "Hour": int(current_dt.hour),
        "Month": int(current_dt.month),
        "DayOfWeek": int(current_dt.dayofweek),
        "Is_Weekend": int(current_dt.dayofweek >= 5)
    }

    X_now = pd.DataFrame([feat_dict])[feature_cols]
    dmat_now = xgb.DMatrix(X_now, feature_names=feature_cols)

    wave_points = []
    for h in horizons:
        booster_h = horizon_boosters.get(h)
        pred_h = float(np.clip(booster_h.predict(dmat_now)[0], 40.0, 500.0))
        target_time = current_dt + timedelta(hours=h)
        wave_points.append({
            "horizon": f"+{h}h",
            "time": target_time.strftime("%I %p"),
            "iso": target_time.isoformat(),
            "aqi": int(round(pred_h))
        })

    pred_24h = float(wave_points[-1]["aqi"])
    horizon_aqis = [p["aqi"] for p in wave_points]
    peak_24h = max(horizon_aqis)
    min_24h = min(horizon_aqis)
    best_hour = wave_points[int(np.argmin(horizon_aqis))]["time"] + f" ({wave_points[int(np.argmin(horizon_aqis))]['horizon']})"
    worst_hour = wave_points[int(np.argmax(horizon_aqis))]["time"] + f" ({wave_points[int(np.argmax(horizon_aqis))]['horizon']})"

    severe_prob_str = "Not Available"
    severe_prob_pct = None
    severe_triggered = False
    if severe_booster is not None:
        raw_score = float(severe_booster.predict(dmat_now)[0])
        thresh = float(val_cal["severe_prob_threshold"])
        is_triggered = (raw_score >= thresh)
        severe_triggered = bool(is_triggered)
        iso_x = val_cal.get("isotonic_x_thresholds")
        iso_y = val_cal.get("isotonic_y_thresholds")
        if iso_x and iso_y:
            cal_prob = float(np.interp(raw_score, iso_x, iso_y)) * 100.0
            severe_prob_pct = round(cal_prob, 1)
            severe_prob_str = format_chance_phrase(cal_prob, pred_24h, is_triggered)
        else:
            severe_prob_pct = round(raw_score * 100.0, 1)
            severe_prob_str = f"Risk Score: {raw_score:.2f} (Cutoff: {thresh:.2f})"

    q10_err = float(val_cal["q10_err"])
    q90_err = float(val_cal["q90_err"])
    mae_val = float(val_cal["mae_val"])

    uncapped_pi80_high = round(pred_24h + q90_err)
    pi80_low = round(max(40.0, pred_24h + q10_err))
    pi80_high = round(min(500.0, pred_24h + q90_err))
    pi80_label = (
        f"{pi80_low}–500 AQI (capped at 500; uncapped: {uncapped_pi80_high})"
        if uncapped_pi80_high > 500 else f"{pi80_low}–{pi80_high} AQI"
    )

    uncapped_spread_high = round(pred_24h + mae_val)
    spread_low = round(max(40.0, pred_24h - mae_val))
    spread_high = round(min(500.0, pred_24h + mae_val))
    spread_label = (
        f"{spread_low}–500 (capped at 500)"
        if uncapped_spread_high > 500 else f"{spread_low}–{spread_high}"
    )

    cur_cat, cur_col, headline = get_cpcb_meta(current_aqi)
    pred_cat, pred_col, _ = get_cpcb_meta(pred_24h)
    adv = get_health_advisory(max(pred_24h, peak_24h), profile)
    driver = likely_driver(max(pred_24h, peak_24h), feat_dict["Wind_Speed"],
                           int(current_dt.hour), int(current_dt.month), pm25_v, pm10_v)

    if mode == "live_ground":
        try:
            scorecard.log_issue(wave_points, base_aqi=round(current_aqi),
                                pi_low=pi80_low, pi_high=pi80_high,
                                severe_pct=severe_prob_pct,
                                trained_at=meta.get("trained_at_utc", ""))
        except Exception as exc:  # never let bookkeeping break a forecast
            logger.warning("Could not record this forecast: %s", exc)

    return JSONResponse({
        "source_warning_level": source_warning_level,
        "source_banner": data_source_banner,
        "source_mode": mode,
        "forecast_available": True,
        "is_replay": bool(replay_label),
        "replay_label": replay_label,
        "reading_time": reading_time_str or current_dt.strftime("%d %b %Y, %H:%M IST"),
        "current_time_iso": current_dt.isoformat(),
        "current_aqi": round(current_aqi),
        "dual_sub": dual_sub,
        "cur_cat": cur_cat,
        "cur_col": cur_col,
        "headline": headline,
        "pred_24h": round(pred_24h),
        "pred_cat": pred_cat,
        "pred_col": pred_col,
        "severe_prob_str": severe_prob_str,
        "severe_prob_pct": severe_prob_pct,
        "severe_triggered": severe_triggered,
        "peak_24h": round(peak_24h),
        "min_24h": round(min_24h),
        "best_hour": best_hour,
        "worst_hour": worst_hour,
        "pi80_label": pi80_label,
        "pi80_low": pi80_low,
        "pi80_high": pi80_high,
        "spread_label": spread_label,
        "spread_low": spread_low,
        "spread_high": spread_high,
        "mae_val": round(mae_val, 1),
        "coverage_test_pct": calibration_coverage(val_cal),
        "eval_label": meta.get("eval_label", "2023 test"),
        "rolling_24": round(feat_dict["Rolling_Mean_24"]),
        "temp": round(feat_dict["Temp_C"], 1),
        "humidity": round(feat_dict["Humidity"]),
        "wind": round(feat_dict["Wind_Speed"], 1),
        "advisory": adv,
        "driver": driver,
        "pm25": None if pm25_v is None or pd.isna(pm25_v) else round(float(pm25_v)),
        "pm10": None if pm10_v is None or pd.isna(pm10_v) else round(float(pm10_v)),
        "wave": wave_points,
        "metadata": meta
    })


@app.get("/api/scorecard")
def forecast_scorecard(days: int = Query(7, ge=1, le=90)):
    """Rolling accuracy of the live forecasts actually issued, plus live-vs-training drift."""
    payload = scorecard.score(days=days)
    payload["drift"] = drift_summary(days)
    return JSONResponse(payload)


@app.get("/api/live_status")
def live_status():
    """Quick health check for the live feed - open http://127.0.0.1:8000/api/live_status"""
    return JSONResponse(ground_feed.status(WAQI_TOKEN))


@app.get("/health")
def health():
    """Liveness/readiness summary used by the UI and local checks."""
    return JSONResponse({
        "status": "ok",
        "replay_dataset_loaded": df_hist is not None and not df_hist.empty,
        "horizon_models_loaded": len(horizon_boosters),
        "severe_model_loaded": severe_booster is not None,
        "waqi_token_set": ground_feed.has_token(WAQI_TOKEN),
        "trained_at_utc": metadata.get("trained_at_utc"),
        "training_data_end": metadata.get("data_end"),
        "live_training_rows": (metadata.get("live_data") or {}).get("rows"),
        "forecast_log_rows": int(len(scorecard.load())),
    })


@app.post("/api/reload_models")
def reload_models():
    """Reload Models/ after a manual retrain promotes a candidate (local use only)."""
    swap_bundle()
    return JSONResponse({
        "reloaded": True,
        "trained_at_utc": metadata.get("trained_at_utc"),
        "horizon_models_loaded": len(horizon_boosters),
        "severe_model_loaded": severe_booster is not None,
    })


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    with open(os.path.join(BASE_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    logger.info("Starting Anand Vihar AQI Forecast Server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)