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
* **Two Data Modes:** Real live ground-sensor readings (WAQI/DPCC Anand Vihar) or a recorded 2023 replay that works offline.

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
Live mode needs a free WAQI token:

```powershell
$env:WAQI_TOKEN = "your-token"     # PowerShell
python server.py
```

Verify the token and unit conversion first:

```powershell
python ground_logger.py --check
python ground_logger.py --status
```

The server saves one real reading per hour while it runs. The forecast unlocks after **49 hourly rows** exist (48 hours of history plus the current hour); until then the UI shows the current reading and a collection progress bar. This cold start is inherent: the model uses 48-hour lag and rolling features.

Without a token the UI automatically starts in **Replay** mode, so the full forecast and advisory flow can still be demonstrated.

## API
| Endpoint | Purpose |
| --- | --- |
| `GET /api/predict?mode=live_ground&profile=Asthma%20Patient` | Live forecast + advisory |
| `GET /api/predict?mode=archive_2023&archive_idx=7400&profile=Child%20(Under%2012)` | Replay forecast from a recorded 2023 window |
| `GET /api/live_status` | Token status, saved rows, window readiness, last poll |
| `GET /health` | Liveness + model/dataset status |

Profiles: `Healthy Adult`, `Child (Under 12)`, `Elderly (60+ Years)`, `Asthma Patient`, `Outdoor Worker`.

## Model & Validation Snapshot
The current bundle (`Models/model_metadata.json`) used a strict chronological split:

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
* Advisories are informational, not medical advice.

## Current Status
The full pipeline is implemented end to end: reproducible dataset build, trained multi-horizon models, calibrated severe-day risk, FastAPI backend, live sensor logger, replay mode, and the responsive advisory UI. The remaining external dependency for live operation is a WAQI API token.
