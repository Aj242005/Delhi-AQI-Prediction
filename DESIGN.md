# Design system — Anand Vihar Air Bulletin

## Intent

A night-console **Metro wayfinding board** for Delhi air quality. The forecast is a transit line, each hourly forecast is a stop, and the resident's next action is a passenger plan beside the line. Research evidence sits in a separate **control room** below the forecast, so evaluators can inspect the method without slowing the resident down.

The interface must stay honest in four dimensions at once: live vs replay, measured vs CAMS-filled, point forecast vs calibrated interval, and historical validation vs live grading.

## Visual world

- **Surface:** near-black night console (`--paper #050811`) over a dim atmospheric Delhi photo. Panels are translucent navy-blue with 1px light borders; no light mode.
- **Accent:** platform orange (`--yellow #f97316`) for the active line, selected stop, primary actions, and current status. Orange is a signal, never decoration.
- **Route:** a heavy blue rail, circular station nodes, orange time chips, and stop codes (`NOW`, `+4H`…`+24H`). The selected stop is an orange-tinted panel with a yellow time chip and visible band text.
- **Type:** Space Grotesk for display and numeric readouts; Plus Jakarta Sans for prose and controls; JetBrains Mono for timestamps, provenance, and data labels; Noto Sans Devanagari for the bilingual station name.
- **Layout:** 1240px max content width, full-bleed utility bar and footer, 14px panel radius, 16–40px responsive gutters. Desktop uses a line + side rail; mobile keeps the line horizontally snap-scrollable and moves inspection directly below it.

## Type and density

- Hero result: `clamp` display size, tabular numbers, `→` between reading and forecast.
- Section headings: large display face, not tiny uppercase eyebrows, except the intentional metro signage kickers.
- Data labels, provenance, and timestamps: 9.5–11px mono/uppercase, letterspaced, reserved for short signage and table headers.
- Explanatory prose: 12–13px minimum; table body 12.5px. Small caps are never used for paragraphs.
- All numerals in readouts, tables, and ledgers use tabular figures.

## Color semantics

| Token | Meaning |
| --- | --- |
| `--band-good` … `--band-severe` | CPCB AQI bands, used on nodes, band tags, and legend |
| `--data-measured` | Sensor-measured input hours |
| `--data-cams` | CAMS reanalysis fill; always labelled, never called measured |
| `--green` | Live service healthy |
| `--amber` | Collecting, waiting, stale |
| `--red` | Board offline, feed failure |
| `--accent-ink` | Text on orange/yellow surfaces; contrast-safe |

## Interaction contract

- Stops are buttons: click, `ArrowLeft`/`ArrowRight`, and focus-visible outlines. Selection updates the adjacent stop inspection and the trip planner.
- Mode (**Live sensor / Replay 2023**) is always visible and always labelled. Replay is never styled as live.
- Profile buttons are a `role="group"` of pressed-state buttons; the plan is explicitly set for the worst forecast stop, not the best.
- Trip length changes the cleanest departure window, computed in IST with day-shift labels (`tomorrow`, `+1d`).
- Evidence is a native `<details>` disclosure with a status pill. The ledger loads independently; failure of the board does not erase the model card.
- Tables are keyboard-scrollable `role="region"` containers with visible mobile swipe hints.

## Evidence rules

- **Live ledger:** issued forecasts graded only against saved measured hours; pending and missing are separate states; CAMS-filled hours never grade a forecast.
- **Input window composition:** a measured / CAMS / still-missing bar with a text key, so the model's inputs are auditable at a glance.
- **Validation record:** 2022 validation and 2023 holdout tables, labelled historical, never presented as a live score.
- **Model card:** shipped bundle, training rows, point error, interval coverage, severe-day recall, and the promotion requirement (720 measured hours) stated plainly.
- **Provenance:** training timestamp, dataset SHA-256 prefix, split sizes, library versions, service readiness, and limitations.
- **No synthetic evidence:** retrain candidate metrics and CAMS-filled hours are not shown as measured production evidence.

## States

Every state keeps the same board metaphor and an explicit status:

- **Live:** reading timestamp, measured/CAMS window, full line and plan.
- **Replay:** `REPLAY · NOT LIVE`, recorded window date, no live-log writes.
- **Collecting / no token:** forecast hidden, route locked, plan explains that guidance unlocks with the forecast, window shows saved vs missing hours.
- **Failure:** service notice, retry and replay actions, evidence replaced by an offline message.
- **Empty ledger:** issued/pending/missing counts, with an explanation of why no errors exist yet.

## Accessibility

- WCAG AA contrast on all text, including orange surfaces (`--accent-ink`).
- 44px minimum control height on mobile; visible focus rings; skip link.
- Live status is conveyed in text, not color alone. Bands always pair color with a name.
- Charts and composition bars carry text equivalents and `aria-label`s.
- Lighthouse: Accessibility 1.0, Best Practices 1.0, SEO 1.0 in live, evidence-open, and replay states (verified 19 Nov 2026).

## Constraints

- Single station (DPCC Anand Vihar), not a city average.
- AQI capped at 500; uncapped output is noted where relevant.
- WAQI readings can differ from the official CPCB bulletin.
- The production bundle is trained through 2023; live retraining needs ~720 measured hours and must beat persistence before promotion.
- Not medical advice.
