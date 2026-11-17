"""
ground_feed.py - live ground-sensor feed for DPCC Anand Vihar (WAQI station @2553)

What this file does, in plain words
-----------------------------------
1. Every few minutes it asks the WAQI service for the latest reading of the
   Anand Vihar ground station; duplicate hours are ignored.
2. It converts that reading into the Indian CPCB AQI - the same scale the
   forecasting model was trained on - and saves it as one row in
   Live/ground_log.csv.
3. After 49 hourly rows exist (the model needs the last 48 hours), the server
   can build a forecast from REAL ground readings instead of satellite-grid
   estimates.

The WAQI feed reports each pollutant as a US-style sub-index, not as
micrograms per cubic metre.  We convert it back to micrograms (the table
that WAQI uses for PM2.5 is selectable, see PM25_SCHEME) and then apply the
official CPCB formula.  Run `python ground_logger.py --check` once to see the
numbers side by side and compare them with a trusted page.

No third-party libraries beyond pandas / numpy are needed.
"""
import csv
import json
import logging
import os
import threading
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

logger = logging.getLogger("ground_feed")

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
STATION_ID = "@2553"                      # DPCC Anand Vihar, Delhi on WAQI
LAT, LON = 28.6469, 77.3160
_BASE = os.path.dirname(os.path.abspath(__file__))


def _load_env_file(path):
    """Read KEY=VALUE pairs from a local .env file without overriding real env vars."""
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if key:
                    os.environ.setdefault(key, value)
    except OSError:
        pass


_load_env_file(os.path.join(_BASE, ".env"))
LOG_PATH = os.environ.get("GROUND_LOG_PATH", os.path.join(_BASE, "Live", "ground_log.csv"))
PM25_SCHEME = os.environ.get("WAQI_PM25_SCHEME", "epa2012").strip().lower()

WINDOW_HOURS = 49        # the model looks back 48 h -> needs 49 hourly rows
MAX_INTERP_GAP = 3       # fill up to 3 missing hours in a row (straight line)
MAX_STALE_HOURS = 3      # newest reading older than this = feed not updating
POLL_SECONDS = 5 * 60

# The local log only has the hours this server was running.  Hours the feed
# never saw are filled from Open-Meteo's CAMS air-quality reanalysis, which
# publishes hourly PM2.5/PM10 (and weather) for this grid point.  They are
# modelled, not measured, so they live in their own file and are reported
# separately - Live/ground_log.csv stays pure station readings.
SEED_PATH = os.path.join(_BASE, "Live", "reanalysis_seed.csv")
SEED_PAST_DAYS = 3
SEED_MAX_AGE_MIN = 60
SEED_COLUMNS = ["Datetime", "seed_aqi", "seed_pm25", "seed_pm10",
                "Temp_C", "Humidity", "Wind_Speed", "fetched_utc"]

# Used only if no weather could be fetched at all (training-set averages).
WEATHER_DEFAULTS = {"Temp_C": 24.0, "Humidity": 65.0, "Wind_Speed": 9.3}

COLUMNS = [
    "Datetime", "AQI", "pm25_ugm3", "pm10_ugm3", "pm25_sub", "pm10_sub",
    "waqi_overall", "Temp_C", "Humidity", "Wind_Speed", "weather_src",
    "pm_both", "scheme", "fetched_utc",
]

# ----------------------------------------------------------------------------
# 1. Official CPCB sub-index (concentration in ug/m3 -> AQI)
#    Source: CPCB National AQI (2014) breakpoints; above the top breakpoint the
#    line keeps the same slope (PM2.5 reaches 500 at 380 ug/m3, PM10 at 510).
# ----------------------------------------------------------------------------
def pm25_to_cpcb_aqi(c):
    if c is None or pd.isna(c) or c < 0:
        return float("nan")
    c = float(c)
    if c <= 30:
        v = c * 50.0 / 30.0
    elif c <= 60:
        v = 50.0 + (c - 30.0) * 50.0 / 30.0
    elif c <= 90:
        v = 100.0 + (c - 60.0) * 100.0 / 30.0
    elif c <= 120:
        v = 200.0 + (c - 90.0) * 100.0 / 30.0
    elif c <= 250:
        v = 300.0 + (c - 120.0) * 100.0 / 130.0
    else:
        v = 400.0 + (c - 250.0) * 100.0 / 130.0
    return round(min(500.0, v), 1)


def pm10_to_cpcb_aqi(c):
    if c is None or pd.isna(c) or c < 0:
        return float("nan")
    c = float(c)
    if c <= 100:
        v = c
    elif c <= 250:
        v = 100.0 + (c - 100.0) * 100.0 / 150.0
    elif c <= 350:
        v = 200.0 + (c - 250.0)
    elif c <= 430:
        v = 300.0 + (c - 350.0) * 100.0 / 80.0
    else:
        v = 400.0 + (c - 430.0) * 100.0 / 80.0
    return round(min(500.0, v), 1)


def cpcb_aqi_from_pm(pm25, pm10):
    """Overall AQI = the worse of the PM2.5 and PM10 sub-indices."""
    vals = [v for v in (pm25_to_cpcb_aqi(pm25), pm10_to_cpcb_aqi(pm10)) if not pd.isna(v)]
    return max(vals) if vals else float("nan")


# ----------------------------------------------------------------------------
# 2. WAQI reports US-style sub-indices.  Convert them back to ug/m3.
# ----------------------------------------------------------------------------
# (conc_low, conc_high, index_low, index_high)
US_PM25 = {
    # The table WAQI documented (2012 US EPA standard)
    "epa2012": [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
                (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
                (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)],
    # US EPA revision of 6 May 2024 (lower "very unhealthy" / "hazardous" cut-offs)
    "epa2024": [(0.0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
                (55.5, 125.4, 151, 200), (125.5, 225.4, 201, 300),
                (225.5, 325.4, 301, 500)],
}
US_PM10 = [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
           (255, 354, 151, 200), (355, 424, 201, 300),
           (425, 504, 301, 400), (505, 604, 401, 500)]


def us_subindex_to_ugm3(sub, pollutant="pm25", scheme=None):
    if sub is None or pd.isna(sub):
        return float("nan")
    scheme = (scheme or PM25_SCHEME)
    table = US_PM10 if pollutant == "pm10" else US_PM25.get(scheme, US_PM25["epa2012"])
    s = max(0.0, min(500.0, float(sub)))
    for c_lo, c_hi, i_lo, i_hi in table:
        if s <= i_hi:
            s = max(s, i_lo)
            return round(c_lo + (s - i_lo) * (c_hi - c_lo) / float(i_hi - i_lo), 1)
    return float(table[-1][1])


def ugm3_to_us_subindex(c, pollutant="pm25", scheme=None):
    """Forward direction - only used by the self-tests."""
    scheme = (scheme or PM25_SCHEME)
    table = US_PM10 if pollutant == "pm10" else US_PM25.get(scheme, US_PM25["epa2012"])
    for c_lo, c_hi, i_lo, i_hi in table:
        if c <= c_hi:
            c = max(c, c_lo)
            return i_lo + (c - c_lo) * (i_hi - i_lo) / (c_hi - c_lo)
    return 500.0


# ----------------------------------------------------------------------------
# 3. Fetching
# ----------------------------------------------------------------------------
def _get_json(url, timeout=10):
    req = urllib.request.Request(url, headers={"User-Agent": "anand-vihar-aqi-project/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_waqi(token, station=STATION_ID, timeout=10):
    url = "https://api.waqi.info/feed/%s/?token=%s" % (station, urllib.parse.quote(token))
    payload = _get_json(url, timeout)
    if payload.get("status") != "ok":
        raise RuntimeError("WAQI replied with status=%r (%s)" % (payload.get("status"), payload.get("data")))
    return payload["data"]


def fetch_open_meteo_current(timeout=10):
    url = ("https://api.open-meteo.com/v1/forecast?latitude=%s&longitude=%s"
           "&current=temperature_2m,relative_humidity_2m,wind_speed_10m&timezone=Asia%%2FKolkata"
           % (LAT, LON))
    cur = _get_json(url, timeout)["current"]
    return {"Temp_C": cur.get("temperature_2m"), "Humidity": cur.get("relative_humidity_2m"),
            "Wind_Speed": cur.get("wind_speed_10m")}   # km/h, same unit as the training data


def fetch_open_meteo_air_history(past_days=SEED_PAST_DAYS, timeout=20):
    """Hourly CAMS PM2.5/PM10 for this grid point, converted to CPCB AQI."""
    url = ("https://air-quality-api.open-meteo.com/v1/air-quality"
           "?latitude=%s&longitude=%s&hourly=pm2_5,pm10&past_days=%d&forecast_days=1"
           "&timezone=Asia%%2FKolkata" % (LAT, LON, past_days))
    h = _get_json(url, timeout)["hourly"]
    rows = []
    for t, p25, p10 in zip(h.get("time") or [], h.get("pm2_5") or [], h.get("pm10") or []):
        if p25 is None and p10 is None:
            continue
        rows.append({"Datetime": pd.Timestamp(t), "seed_pm25": p25, "seed_pm10": p10,
                     "seed_aqi": cpcb_aqi_from_pm(p25, p10)})
    return pd.DataFrame(rows, columns=["Datetime", "seed_aqi", "seed_pm25", "seed_pm10"])


def fetch_open_meteo_weather_history(past_days=SEED_PAST_DAYS, timeout=20):
    """Hourly weather for the same hours, so seeded rows carry real conditions."""
    url = ("https://api.open-meteo.com/v1/forecast"
           "?latitude=%s&longitude=%s&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
           "&past_days=%d&forecast_days=1&timezone=Asia%%2FKolkata" % (LAT, LON, past_days))
    h = _get_json(url, timeout)["hourly"]
    return pd.DataFrame({"Datetime": pd.to_datetime(h["time"]),
                         "Temp_C": h["temperature_2m"],
                         "Humidity": h["relative_humidity_2m"],
                         "Wind_Speed": h["wind_speed_10m"]})


def load_seed(path=None):
    """The cached reanalysis hours (no network)."""
    path = path or SEED_PATH
    if not os.path.exists(path):
        return pd.DataFrame(columns=SEED_COLUMNS)
    try:
        df = pd.read_csv(path, parse_dates=["Datetime"])
    except Exception as exc:  # a broken cache must never stop a forecast
        logger.warning("Could not read the reanalysis seed %s: %s", path, exc)
        return pd.DataFrame(columns=SEED_COLUMNS)
    if "seed_aqi" not in df.columns:
        return pd.DataFrame(columns=SEED_COLUMNS)
    df["seed_aqi"] = pd.to_numeric(df["seed_aqi"], errors="coerce")
    df = df.dropna(subset=["Datetime", "seed_aqi"])
    return df.drop_duplicates("Datetime", keep="last").sort_values("Datetime").reset_index(drop=True)


def refresh_seed(path=None, force=False, now=None, max_age_minutes=SEED_MAX_AGE_MIN,
                 air_fetch=None, weather_fetch=None):
    """Reanalysis hours, refetched only when the cached copy has gone stale."""
    path = path or SEED_PATH
    cached = load_seed(path)
    now = now if now is not None else now_ist()
    if not force and len(cached):
        try:
            fetched = pd.to_datetime(cached["fetched_utc"], errors="coerce").max()
            if pd.notna(fetched) and (now - fetched) < pd.Timedelta(minutes=max_age_minutes):
                return cached
        except Exception:
            pass
    try:
        air_df = (air_fetch or fetch_open_meteo_air_history)()
        wx_df = (weather_fetch or fetch_open_meteo_weather_history)()
    except Exception as exc:
        logger.warning("Reanalysis seed unavailable (%s) - using the cached hours", exc)
        return cached
    if air_df is None or air_df.empty:
        return cached
    merged = air_df.merge(wx_df, on="Datetime", how="left")
    merged["fetched_utc"] = now.strftime("%Y-%m-%d %H:%M:%S")
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        merged.to_csv(tmp, index=False)
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning("Could not save the reanalysis seed: %s", exc)
    return load_seed(path)


def parse_waqi(data, scheme=None):
    """Turn one WAQI 'feed' answer into a plain dict (no network, easy to test)."""
    scheme = scheme or PM25_SCHEME
    iaqi = data.get("iaqi") or {}

    def val(key):
        v = (iaqi.get(key) or {}).get("v")
        try:
            return None if v is None else float(v)
        except (TypeError, ValueError):
            return None

    pm25_sub, pm10_sub = val("pm25"), val("pm10")
    if pm25_sub is None and pm10_sub is None:
        raise ValueError("WAQI answer has neither PM2.5 nor PM10 for this station")

    t = data.get("time") or {}
    stamp = t.get("iso") or t.get("s")
    if not stamp:
        raise ValueError("WAQI answer has no timestamp")
    ts = pd.Timestamp(stamp)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("Asia/Kolkata").tz_localize(None)
    hour = ts.floor("h")

    pm25 = us_subindex_to_ugm3(pm25_sub, "pm25", scheme)
    pm10 = us_subindex_to_ugm3(pm10_sub, "pm10", scheme)
    wind_ms = val("w")
    return {
        "Datetime": hour,
        "pm25_sub": pm25_sub, "pm10_sub": pm10_sub,
        "pm25_ugm3": pm25, "pm10_ugm3": pm10,
        "AQI": cpcb_aqi_from_pm(pm25, pm10),
        "waqi_overall": data.get("aqi"),
        "pm_both": int(pm25_sub is not None and pm10_sub is not None),
        "station": (data.get("city") or {}).get("name", ""),
        "wx_waqi": {"Temp_C": val("t"), "Humidity": val("h"),
                    "Wind_Speed": None if wind_ms is None else round(wind_ms * 3.6, 1)},  # m/s -> km/h
    }


# ----------------------------------------------------------------------------
# 4. The log file
# ----------------------------------------------------------------------------
_LOCK = threading.Lock()
LAST_POLL = {"time": None, "status": "not run yet", "detail": ""}


def load_log(path=None):
    path = path or LOG_PATH
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLUMNS)
    try:
        df = pd.read_csv(path, parse_dates=["Datetime"], on_bad_lines="skip")
    except Exception as exc:  # unreadable file should never crash the server
        logger.warning("Could not read %s: %s", path, exc)
        return pd.DataFrame(columns=COLUMNS)
    df = df.dropna(subset=["Datetime", "AQI"])
    return df.drop_duplicates("Datetime", keep="last").sort_values("Datetime").reset_index(drop=True)


def _append(row, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow(row)


def poll_once(token, log_path=None, fetch=fetch_waqi, weather=fetch_open_meteo_current, scheme=None):
    """Fetch the newest reading and append it to the log if that hour is new."""
    log_path = log_path or LOG_PATH
    scheme = scheme or PM25_SCHEME
    rec = parse_waqi(fetch(token), scheme)

    # weather: Open-Meteo first, WAQI's own numbers as backup
    wx, src = {}, "none"
    try:
        wx = weather() or {}
        if all(wx.get(k) is not None for k in WEATHER_DEFAULTS):
            src = "open-meteo"
    except Exception as exc:
        logger.warning("Open-Meteo weather unavailable (%s) - trying WAQI weather", exc)
    if src == "none":
        wx_w = rec["wx_waqi"]
        merged = {k: (wx.get(k) if wx.get(k) is not None else wx_w.get(k)) for k in WEATHER_DEFAULTS}
        if any(v is not None for v in merged.values()):
            wx, src = merged, "waqi" if all(v is not None for v in merged.values()) else "partial"

    row = {
        "Datetime": rec["Datetime"].strftime("%Y-%m-%d %H:%M:%S"),
        "AQI": rec["AQI"], "pm25_ugm3": rec["pm25_ugm3"], "pm10_ugm3": rec["pm10_ugm3"],
        "pm25_sub": rec["pm25_sub"], "pm10_sub": rec["pm10_sub"],
        "waqi_overall": rec["waqi_overall"],
        "Temp_C": wx.get("Temp_C"), "Humidity": wx.get("Humidity"), "Wind_Speed": wx.get("Wind_Speed"),
        "weather_src": src, "pm_both": rec["pm_both"], "scheme": scheme,
        "fetched_utc": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M:%S"),
    }
    with _LOCK:
        existing = load_log(log_path)
        if (existing["Datetime"] == rec["Datetime"]).any():
            return {"status": "duplicate", "hour": row["Datetime"], "AQI": row["AQI"]}
        _append(row, log_path)
    return {"status": "logged", "hour": row["Datetime"], "AQI": row["AQI"], "weather": src,
            "partial_pm": not rec["pm_both"]}


# ----------------------------------------------------------------------------
# 5. Background saving (runs inside the server)
# ----------------------------------------------------------------------------
_thread = None


def has_token(token):
    return bool(token) and token.strip().lower() not in ("", "demo")


def start_background_logger(token, interval=POLL_SECONDS, log_path=None):
    global _thread
    if not has_token(token):
        logger.warning("WAQI_TOKEN is not set - live ground logging is OFF. "
                       "Replay mode still works.")
        return None
    if _thread is not None and _thread.is_alive():
        return _thread

    def loop():
        while True:
            try:
                res = poll_once(token, log_path)
                LAST_POLL.update(time=pd.Timestamp.now().isoformat(timespec="seconds"),
                                 status=res["status"], detail=str(res))
                if res["status"] == "logged":
                    logger.info("Saved ground reading: %s", res)
            except Exception as exc:
                LAST_POLL.update(time=pd.Timestamp.now().isoformat(timespec="seconds"),
                                 status="error", detail=str(exc))
                logger.warning("Ground poll failed: %s", exc)
            time.sleep(interval)

    _thread = threading.Thread(target=loop, name="ground-logger", daemon=True)
    _thread.start()
    logger.info("Ground-sensor logger started (every %d min).", interval // 60)
    return _thread


# ----------------------------------------------------------------------------
# 6. Turn the log into the 49-hour table the forecaster needs
# ----------------------------------------------------------------------------
def now_ist():
    return pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)


def build_window(log_df, now=None, need=WINDOW_HOURS, max_gap=MAX_INTERP_GAP,
                 max_stale=MAX_STALE_HOURS, seed_df=None):
    """
    Returns a dict. ready=True means df_past (49 hourly rows) can go straight
    into the forecaster. Otherwise `reason` says why not:
      no_data    - nothing logged yet and no reanalysis hours either
      stale      - newest reading is older than max_stale hours
      collecting - fewer than `need` hours saved so far
      gaps       - enough time has passed but too many hours are missing

    Hours with no station reading are filled from `seed_df` (CAMS reanalysis)
    when it covers them.  `have` always counts measured hours only, `n_seeded`
    counts the reanalysis hours that stand in for the rest.
    """
    now = now if now is not None else now_ist()
    out = {"ready": False, "reason": "no_data", "have": 0, "need": need, "latest": None,
           "latest_row": None, "age_hours": None, "df_past": None, "n_interpolated": 0,
           "n_seeded": 0, "weather_filled": False}
    has_measured = log_df is not None and len(log_df) > 0

    # The window ends at the newest measured hour.  On a fresh install there is
    # none yet, so it ends at the newest reanalysis hour that has already
    # happened - the API's past days are enough to forecast immediately.
    if has_measured:
        latest = log_df["Datetime"].max()
        latest_row = log_df.loc[log_df["Datetime"].idxmax()]
    else:
        done = (seed_df[seed_df["Datetime"] <= now.floor("h")]
                if seed_df is not None and len(seed_df) else None)
        if done is None or len(done) == 0:
            return out
        latest = done["Datetime"].max()
        latest_row = done.loc[done["Datetime"].idxmax()].rename({"seed_aqi": "AQI"})

    age = (now - latest) / pd.Timedelta(hours=1)
    out.update(latest=latest, latest_row=latest_row, age_hours=float(age))
    if age > max_stale:
        out["reason"] = "stale"
        return out

    idx = pd.date_range(end=latest, periods=need, freq="h")
    if has_measured:
        s = log_df.set_index("Datetime").reindex(idx)
        out["have"] = int(s["AQI"].notna().sum())
    else:
        s = pd.DataFrame(index=idx, columns=["AQI"] + ["Temp_C", "Humidity", "Wind_Speed"],
                         dtype=float)
    span_hours = int((latest - log_df["Datetime"].min()) / pd.Timedelta(hours=1)) + 1 if has_measured else 1

    wx_cols = ["Temp_C", "Humidity", "Wind_Speed"]
    if seed_df is not None and len(seed_df):
        seed = seed_df.set_index("Datetime").reindex(idx)
        gap = s["AQI"].isna() & seed["seed_aqi"].notna()
        if gap.any():
            s.loc[gap, "AQI"] = seed.loc[gap, "seed_aqi"]
            out["n_seeded"] = int(gap.sum())
        for c in wx_cols:
            if c in seed:
                missing = s[c].isna() & seed[c].notna()
                if missing.any():
                    s.loc[missing, c] = seed.loc[missing, c]

    before = int(s["AQI"].isna().sum())
    s["AQI"] = pd.to_numeric(s["AQI"], errors="coerce").interpolate(limit=max_gap, limit_area="inside")
    if s["AQI"].isna().any():
        out["reason"] = "collecting" if span_hours < need else "gaps"
        return out
    out["n_interpolated"] = before

    for c in wx_cols:
        s[c] = pd.to_numeric(s[c], errors="coerce").ffill(limit=6).bfill(limit=6)
    if s[wx_cols].isna().any().any():
        out["weather_filled"] = True
        for c in wx_cols:
            s[c] = s[c].fillna(WEATHER_DEFAULTS[c])

    s["AQI"] = s["AQI"].clip(0.0, 500.0)
    df_past = s[["AQI"] + wx_cols].reset_index().rename(columns={"index": "Datetime"})
    out.update(ready=True, reason="ok", df_past=df_past[["Datetime", "AQI", "Temp_C", "Humidity", "Wind_Speed"]])
    return out


def avg_24h(log_df, latest, min_hours=16):
    """Average of the saved hourly AQI values over the last 24 hours (approximates the
    24-hour running average CPCB publishes). Needs at least `min_hours` saved hours."""
    if log_df is None or len(log_df) == 0 or latest is None:
        return None, 0
    w = log_df[(log_df["Datetime"] > latest - pd.Timedelta(hours=24)) & (log_df["Datetime"] <= latest)]
    n = int(w["AQI"].notna().sum())
    if n < min_hours:
        return None, n
    return float(w["AQI"].mean()), n


def status(token, log_path=None, now=None, seed_df=None):
    """Small summary used by /api/live_status."""
    df = load_log(log_path)
    w = build_window(df, now, seed_df=seed_df)
    return {
        "token_set": has_token(token), "pm25_scheme": PM25_SCHEME,
        "rows_saved": int(len(df)), "hours_in_window": w["have"], "hours_needed": w["need"],
        "seeded_hours": w["n_seeded"],
        "latest_reading": None if w["latest"] is None else str(w["latest"]),
        "age_hours": None if w["age_hours"] is None else round(w["age_hours"], 1),
        "forecast_ready": w["ready"], "reason": w["reason"], "last_poll": LAST_POLL,
    }