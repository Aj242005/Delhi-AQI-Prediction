# Delhi Air Quality Prediction & Personalized Health Risk Advisor

## Overview
This project forecasts the Air Quality Index (AQI) for Anand Vihar, Delhi/NCR, and translates those predictions into actionable, personalized health advisories.

Most air quality models stop at a raw AQI number. This project adds a human-centric layer: it evaluates the forecast against specific health profiles (parents of young children, asthma patients, the elderly, outdoor workers, and healthy adults) and returns targeted, risk-aware guidance instead of generic warnings.

The project is built for real-world civic impact, aligning with e-governance themes by addressing one of the most critical public health challenges in the region.

## Key Features
* **AQI Forecasting:** Multi-horizon XGBoost models (+4h, +8h, +12h, +16h, +20h, +24h) built from historical pollution and meteorological data.
* **Personalized Health Advisory Layer:** A rules engine that maps predicted AQI bands to specific demographic and health profiles.
* **Actionable Insights:** Concrete guidance such as masks, ventilation windows, outdoor-activity limits, and severe-day risk.
* **Calibrated Uncertainty:** 2022-calibrated residual intervals, isotonic-calibrated severe-day probabilities, and a 2023 test-set audit shown in the UI.
* **Forecast Scorecard:** every live forecast is logged at its target hour and graded once that hour is real, so rolling error, interval coverage, Brier score and skill against persistence reflect today rather than 2023.
* **Two Data Modes:** Real live ground-sensor readings (WAQI/DPCC Anand Vihar) or a recorded 2023 replay that works offline.
* **Commute Planner:** the six horizon points are reused to find the cleanest departure window for a 30-minute to 2-hour trip.
* **Likely Driver:** a labelled heuristic that names the usual suspect (night inversion, smoke, traffic, dust) from wind, hour, season and the fine/coarse particle split.

## Data Strategy & Sources
The pipeline is split into historical training data and live inference data:

* **Historical AQI Data (Training):** Official CPCB/DPCC Anand Vihar hourly records (2017–2023) via [OpenCity.in](https://data.opencity.in/dataset/delhi-hourly-air-quality-reports).
* **Historical Weather Data (Training):** Hourly temperature, humidity, and wind speed from the [Open-Meteo archive API](https://open-meteo.com/), cached in `Dataset/anand_vihar_weather_2017_2023.csv`.
* **Live AQI Data (Inference):** The [WAQI API](https://aqicn.org/data-platform/token/) station `@2553` (DPCC Anand Vihar), converted from US sub-indices to the Indian CPCB AQI scale.
* **Live Weather Data (Inference):** Open-Meteo current conditions, with WAQI weather as a fallback.

## Repository Layout
```
server.py                 FastAPI backend, model loading, health-advisory engine
index.html                Single-page frontend (served by server.py)
ground_feed.py            Live WAQI polling, unit conversion, 49-hour window builder
ground_logger.py          Optional stand-alone live logger / diagnostics
scorecard.py              Forecast log + rolling scoring against saved live hours
retrain.py                Occasionally refit the models on timestamped live data
tests/                    Contract tests (payload, scorecard, retrain rules)
build_dataset.py          Reproducible data pipeline (raw table -> ml_ready_data.csv)
Dataset/                  Raw AQI table, cached weather, model-ready features
Models/                   Native XGBoost JSON models + model_metadata.json
Notebooks/                Cleaning, missing-data, feature-engineering, training
Live/ground_log.csv       Created at runtime; ignored by git
```

## Quick Start
```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the app
python server.py

# 3. Open http://127.0.0.1:8000
```

`Dataset/ml_ready_data.csv` is included, so replay mode works immediately. If it is ever missing or you want to rebuild it from the raw table:

```powershell
python build_dataset.py
```

The script performs the same four stages as the notebooks and prints row counts plus the output SHA-256.

### Live mode (optional)
Live mode needs a free WAQI token. You can keep it in the gitignored `.env` file or set it in the shell:

```powershell
# Option A: WAQI_TOKEN=your-token in .env, then:
python server.py

# Option B: current shell only
$env:WAQI_TOKEN = "your-token"
python server.py
```

Verify the token and unit conversion first:

```powershell
python ground_logger.py --check
python ground_logger.py --status
```

The server polls WAQI every 5 minutes and saves one real reading per hour while it runs. The forecast unlocks after **49 hourly rows** exist (48 hours of history plus the current hour); until then the UI shows the current reading and a collection progress bar. This cold start is inherent: the model uses 48-hour lag and rolling features.

Once ready, every new hourly reading moves the 49-hour input window forward by one hour and the model recomputes the full +4h to +24h line from the latest reading. This is a **sliding window**, not a once-a-day batch: there is no day boundary or "window expired" wait. While the page is open in live mode it refetches every 5 minutes, so a newly saved hour appears without a reload.

Without a token the UI automatically starts in **Replay** mode, so the full forecast and advisory flow can still be demonstrated.

### Filling the first two days of live history
A fresh install starts with an empty history: the server can only save the hours it was running, so the model would need 49 hours before drawing anything. Hours the local feed never saw are completed from Open-Meteo's **CAMS** air-quality service (hourly PM2.5/PM10 for the Anand Vihar grid point, converted with the same CPCB formula, plus the matching temperature, humidity and wind). Those hours are cached in `Live/reanalysis_seed.csv` for an hour at a time and kept out of `Live/ground_log.csv`, so the station log stays pure measured readings. That cache file is committed to the repo (public CAMS data, no credentials), so a host that cannot reach Open-Meteo - a free Render instance, for example - still boots with a complete window and draws the forecast instead of waiting two days; the file is refreshed from the API whenever it is reachable. A failed fetch is not retried for ten minutes, so such a host answers in milliseconds rather than paying two HTTP timeouts per request. The request asks for the past two days, so the window is full the first time the page is opened - even with an empty ground log, where it simply ends at the newest hour that has already happened.

Measured hours always win: CAMS only fills a slot the ground log does not have. The board says how the window was assembled (`window: 2 measured + 47 CAMS`) and the source banner spells it out, because a window that is mostly modelled is not the same evidence as one built from two days of station readings. The newest reading must still be a real station reading no more than three hours old once the log has any rows at all, so a dead feed still pauses the forecast instead of hiding behind the model. The single exception is the very first start: with no measured row yet, the CAMS hour that has just passed is used as the anchor and the board says `0 measured + 49 CAMS` until the logger saves its first reading.

### Forecast scorecard
Every live forecast the API issues is written to `Live/forecast_log.csv`: one row per target hour and horizon, stored the first time that target is served, so the row keeps its true lead time (the +4h row really was issued four hours early). When the hour arrives in `Live/ground_log.csv`, `scorecard.py` joins the two and the control room shows:

* rolling MAE, RMSE and bias per horizon over the last 7 days,
* skill against 24-hour persistence (holding the issue-hour reading flat),
* how often the actual reading landed inside the 80% interval,
* the Brier score and a reliability strip for the severe-day probability,
* how many forecast hours are still waiting for their reading, and how many hour slots passed with no reading at all,
* live-vs-training drift (mean and z-score for AQI, temperature, humidity, wind).

An ungraded row is reported as **pending** while its target hour is still ahead (or within the last hour), and as **missing** once that hour has passed without a saved reading. Missing rows are never scored against a guess, and the board says so: a server or feed outage shows up as missing hours instead of a silent gap in the error numbers.

Writes are safe under concurrent requests: `log_issue` holds a module lock across read, dedupe, append and prune, so two overlapping `/api/predict` calls can never write the same `(horizon, target hour)` twice. The parsed log is cached by `(size, mtime)`, so unchanged files are not re-read on every request, and rows older than `FORECAST_LOG_RETENTION_DAYS` (default 45) are pruned on append with an atomic replace. Scoring dedupes `(horizon, target hour)` as well, so a log that was ever edited by hand still cannot double-count.

Replay forecasts are never logged, because recorded 2023 hours can never be graded against today's sensor. Nothing leaves the two CSV files; `Live/` is gitignored.

Run the tests with:

```powershell
python -m unittest discover -s tests -v
```

### Retraining on live data
Every saved live reading carries its own timestamp in `Live/ground_log.csv`, so the live record can be folded back into training without guessing when a row was captured.

It runs by itself while the server is up. A background thread calls `retrain.py --if-due` five minutes after startup and then every six hours, and swaps the promoted models and metadata into the running process, so nothing needs to be scheduled and the server does not restart. `RETRAIN_CHECK_HOURS=12` changes the interval and `RETRAIN_AUTO=0` turns the loop off.

The tool is deliberately conservative, and manual runs still work:

```powershell
python retrain.py --if-due          # what the server's auto-retrain calls
python retrain.py --dry-run         # train and report, write nothing
python retrain.py --force-promote   # bypass the quality gate (not recommended)
```

`--if-due` waits for **720 hourly readings** (about 30 days) and then runs at most once every 7 days, provided at least 168 new readings arrived. A run:

1. Builds features per contiguous live segment. Gaps larger than 3 hours split a segment; the 2023 to 2026 gap is never interpolated.
2. Trains the six horizon models and the severe-day classifier on historical data plus the early live record, giving live rows a 5x sample weight.
3. Validates on the next live slice and tests on the most recent live slice, with a 24-hour purge between slices.
4. Calibrates the 80% interval and severe probability on live validation data.
5. Promotes only if the candidate beats 24-hour persistence, keeps 80% interval coverage between 60% and 95%, and (when severe hours exist in the test slice) reaches at least 45% severe recall.
6. Backs up the previous `Models/*.json`, writes the candidate under `Models/candidates/`, and tells a running server to reload (`POST /api/reload_models`); the auto-retrain loop also reloads in-process as soon as the metadata changes.

The first meaningful retrain cannot happen until roughly a month of live readings exists; the forecast itself only needs 49 hourly rows. Because an idle check costs a few seconds, the server can keep asking until it is due.

Candidate and backup folders are local and ignored by git. The promoted `model_metadata.json` records the live data range, sample weight, split labels, and gate result, so the UI evidence section describes the actual bundle in use.

## API
| Endpoint | Purpose |
| --- | --- |
| `GET /api/predict?mode=live_ground&profile=Asthma%20Patient` | Live forecast + advisory |
| `GET /api/predict?mode=archive_2023&archive_idx=7400&profile=Child%20(Under%2012)` | Replay forecast from a recorded 2023 window |
| `GET /api/scorecard?days=7` | Rolling accuracy of the forecasts actually issued, plus live-vs-training drift |
| `GET /api/live_status` | Token status, saved rows, window readiness, last poll |
| `GET /health` | Liveness + model/dataset status |
| `POST /api/reload_models` | Reload promoted model files without restarting the server |

Profiles: `Healthy Adult`, `Child (Under 12)`, `Elderly (60+ Years)`, `Asthma Patient`, `Outdoor Worker`.

## Model & Validation Snapshot
The current shipped bundle (`Models/model_metadata.json`) used a strict chronological split. It was trained before live 2026 readings existed; `retrain.py` replaces it only after a candidate passes the promotion gate.

* Train: 2017–2021 (36,838 rows)
* Validation/calibration: 2022 (8,760 rows, 8,153 real)
* Test: 2023 (8,712 rows, 7,630 real)

Selected results on real 2023 hours:

| Horizon | XGBoost RMSE | Persistence RMSE | Skill |
| --- | --- | --- | --- |
| +4h | 74.57 | 101.94 | +26.9% |
| +12h | 86.05 | 133.99 | +35.8% |
| +24h | 87.72 | 105.89 | +17.2% |

Severe-day classifier (2023 test): recall 0.812, precision 0.512, F1 0.628. The 80% interval achieved 83.1% empirical coverage on 2023 verified hours.

**Note on dataset drift:** the upstream OpenCity resource has since added real 2023-12-31 readings, so a fresh rebuild produces 54,334 rows instead of the 54,310 recorded in `model_metadata.json`. The extra day extends replay coverage only; the trained JSON models and their reported metrics are unchanged.

## Limitations
* One station (Anand Vihar); this is not a city-wide forecast.
* Live WAQI values are converted to CPCB AQI and can differ slightly from the official CPCB bulletin.
* Live expectations assume the server stays running so hourly readings are saved; gaps pause the forecast rather than silently filling long periods.
* The live forecast unlocks after 49 hourly readings; retraining waits for about 30 days of live data. Neither can be backfilled from the free WAQI feed.
* Advisories are informational, not medical advice.

## Current Status
The full pipeline is implemented end to end: reproducible dataset build, trained multi-horizon models, calibrated severe-day risk, FastAPI backend, live sensor logger, automatic timestamped live retraining, replay mode, and the responsive advisory UI. The shipped models are still the 2023-trained bundle until enough live readings accumulate and a retrained candidate passes the promotion gate.
