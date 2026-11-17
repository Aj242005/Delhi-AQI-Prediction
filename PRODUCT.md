# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

A mixed audience, confirmed by the project owner:

- **Delhi/NCR residents**, especially people deciding whether to go outside, exercise, send children to school, or ventilate a home.
- **Health workers and advisers** explaining risk and protective actions to patients.
- **Project evaluators and judges** checking whether the forecast is rigorous, calibrated, and honestly presented.

The public resident is the first-time reader; the other two audiences must be able to go deeper without cluttering the primary experience.

## Product Purpose

Forecast the Air Quality Index for DPCC Anand Vihar across six horizons (+4h to +24h) and turn the forecast into a personal action plan for a selected health profile. Success means a resident understands the next 24 hours and knows what to do within seconds, while an evaluator can verify the model's evidence without leaving the page.

## Positioning

Most AQI products stop at a current number or a generic warning. This product combines:

- multi-horizon point forecasts,
- a calibrated 80% uncertainty interval,
- a calibrated severe-day probability,
- a single-station CPCB-scale forecast,
- profile-specific mask, ventilation, and activity guidance.

The forecast and the action plan are one product, not a number plus a disclaimer.

## Operating Context

- Runs as a local FastAPI app with a static single-page frontend.
- Two data modes: **live ground sensor** (WAQI/DPCC Anand Vihar, one reading saved per hour, forecast unlocks after 49 hourly rows) and **replay** (recorded 2023 windows for demos and offline use).
- Live mode requires a WAQI token and a server that stays running; replay needs no network.
- Used on phones and laptops, often in the morning or evening while planning the day.
- The model was trained on 2017-2023 hourly CPCB data; live WAQI values are converted to the CPCB scale.

## Capabilities and Constraints

- Six XGBoost horizon models (JSON), one severe-day classifier, one replay dataset.
- Profiles: Healthy Adult, Child (Under 12), Elderly (60+ Years), Asthma Patient, Outdoor Worker.
- Single station only; not city-wide. Not medical advice.
- Live readings can differ from the official CPCB bulletin and may pause when the feed is stale or gappy.
- API returns current AQI, six horizon points, +24h interval, severe-day risk, best/worst horizon, weather, advisory, and model metadata.
- The UI must work without a WAQI token by falling back to replay.

## Evidence on Hand

- `Models/model_metadata.json`: 2022 calibration and 2023 test metrics, severe-day classifier scores, horizon skill table, model provenance.
- `Dataset/ml_ready_data.csv`: 54,334 rows after the upstream source added 2023-12-31 (the recorded training snapshot was 54,310 rows).
- `Models/xgb_horizon_*.json`, `Models/xgb_severe_clf.json`: native JSON models.
- `Notebooks/`: cleaning, missing-data handling, feature engineering, training and validation.
- `Dataset/eda_*.png` and `Models/model_evaluation_plots.png` exist but are not required by the UI.
- No real user testimonials, press, or commercial claims exist; do not invent them.

## Product Principles

1. **The next 24 hours lead.** The forecast is the first thing a resident sees.
2. **Action over numbers.** Every forecast state resolves into something a person can do.
3. **Source honesty.** Live, replay, collecting, stale, and gappy states are unmistakable and never disguised.
4. **Evidence on demand.** Calibration and validation are easy to reach for evaluators but subordinate for residents.
5. **Warm, human language.** Serious air quality does not require cold, bureaucratic delivery.

## Accessibility & Inclusion

- Target WCAG 2.1 AA contrast for text and controls.
- Never use color alone for AQI categories: pair every band with a text label and value.
- Full keyboard access to mode/profile controls and forecast points.
- Respect `prefers-reduced-motion`.
- Plain-language summaries alongside technical labels.
