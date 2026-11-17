"""Contract tests for the two things a wrong change would silently break:
the /api/predict payload the board renders, and the retrain scheduling rules.

Run with:  python -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Importing the server starts its threads; keep the tests off the network.
os.environ.setdefault("WAQI_TOKEN", "")
os.environ.setdefault("RETRAIN_AUTO", "0")

import threading

import numpy as np
import pandas as pd

import scorecard


class PredictPayloadContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import server
        cls.server = server
        response = server.predict_aqi(profile="Asthma Patient", mode="archive_2023", archive_idx=7400)
        cls.payload = response.body
        import json
        cls.data = json.loads(response.body)

    def test_core_keys_survive(self):
        for key in ("source_mode", "forecast_available", "current_aqi", "pred_24h",
                    "peak_24h", "min_24h", "pi80_low", "pi80_high", "severe_prob_pct",
                    "advisory", "wave", "metadata", "eval_label", "driver"):
            self.assertIn(key, self.data, key)

    def test_six_horizons_with_iso_targets(self):
        wave = self.data["wave"]
        self.assertEqual(len(wave), 6)
        self.assertEqual([point["horizon"] for point in wave],
                         ["+4h", "+8h", "+12h", "+16h", "+20h", "+24h"])
        for point in wave:
            self.assertTrue(point["iso"])

    def test_advisory_has_three_fields(self):
        for key in ("advice", "mask", "ventilation"):
            self.assertTrue(self.data["advisory"][key])

    def test_driver_shape(self):
        self.assertIn("label", self.data["driver"])
        self.assertIn("detail", self.data["driver"])


class ScorecardLog(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "forecast_log.csv")
        base = pd.Timestamp("2026-11-17T00:00:00")
        self.wave = [
            {"horizon": f"+{h}h", "iso": (base + pd.Timedelta(hours=h)).isoformat(), "aqi": 300 + h}
            for h in (4, 8, 12, 16, 20, 24)
        ]

    def test_first_issue_only_is_recorded(self):
        wrote = scorecard.log_issue(self.wave, 320, pi_low=200, pi_high=450,
                                    severe_pct=12.5, trained_at="t0", path=self.path)
        self.assertEqual(wrote, 6)
        again = scorecard.log_issue(self.wave, 320, pi_low=200, pi_high=450,
                                    severe_pct=12.5, trained_at="t0", path=self.path)
        self.assertEqual(again, 0)
        self.assertEqual(len(scorecard.load(self.path)), 6)

    def test_replay_is_never_logged(self):
        wrote = scorecard.log_issue(self.wave, 320, path=self.path, mode="archive_2023")
        self.assertEqual(wrote, 0)
        self.assertFalse(os.path.exists(self.path))

    def test_interval_and_probability_attach_to_the_last_horizon(self):
        scorecard.log_issue(self.wave, 320, pi_low=200, pi_high=450,
                            severe_pct=12.5, trained_at="t0", path=self.path)
        rows = scorecard.load(self.path)
        band = rows.dropna(subset=["pi_low"])
        self.assertEqual(len(band), 1)
        self.assertEqual(band.iloc[0]["horizon"], "+24h")
        self.assertEqual(rows["pi_low"].notna().sum(), 1)


class ScorecardScoring(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "forecast_log.csv")
        rows = []
        for hour in range(1, 25):
            target = pd.Timestamp("2026-11-18") + pd.Timedelta(hours=hour)
            rows.append({
                "issued_utc": "2026-11-18 00:00:00", "mode": "live_ground",
                "horizon": "+4h", "lead_hours": 4, "target_iso": target.isoformat(),
                "base_aqi": 300, "pred": 320, "pi_low": "", "pi_high": "",
                "severe_pct": "", "trained_at": "t0",
            })
            rows.append({
                "issued_utc": "2026-11-18 00:00:00", "mode": "live_ground",
                "horizon": "+24h", "lead_hours": 24, "target_iso": target.isoformat(),
                "base_aqi": 300, "pred": 420, "pi_low": 380, "pi_high": 470,
                "severe_pct": 70.0, "trained_at": "t0",
            })
        pd.DataFrame(rows, columns=scorecard.COLUMNS).to_csv(self.path, index=False)
        # Every target hour really read 400: inside the +24h band, not severe.
        readings = pd.DataFrame({
            "Datetime": [pd.Timestamp("2026-11-18") + pd.Timedelta(hours=h) for h in range(1, 25)],
            "AQI": [400.0] * 24,
        })
        self.readings = readings

    def test_horizon_error_and_skill(self):
        result = scorecard.score(days=7, path=self.path, readings=self.readings)
        self.assertEqual(result["scored"], 48)
        by_horizon = {row["horizon"]: row for row in result["horizons"]}
        self.assertAlmostEqual(by_horizon["+4h"]["mae"], 80.0, places=1)
        self.assertAlmostEqual(by_horizon["+24h"]["mae"], 20.0, places=1)
        # Persistence holds the issue-hour reading of 300, so it errs by 100.
        self.assertAlmostEqual(by_horizon["+4h"]["skill_pct"], 20.0, places=1)
        self.assertAlmostEqual(by_horizon["+24h"]["skill_pct"], 80.0, places=1)

    def test_coverage_and_brier(self):
        result = scorecard.score(days=7, path=self.path, readings=self.readings)
        self.assertEqual(result["coverage_pct"], 100.0)
        self.assertEqual(result["coverage_rows"], 24)
        self.assertEqual(result["severe_hours"], 0)
        # Predicted 0.7, nothing observed severe: Brier = 0.49.
        self.assertAlmostEqual(result["brier"], 0.49, places=2)

    def test_pending_and_missing_are_reported_separately(self):
        # Hours 01-04 were saved. At 06:00 hour 05 has passed unread while
        # hours 06-24 are still ahead, so neither may be scored.
        readings = self.readings.iloc[:4]
        result = scorecard.score(days=7, path=self.path, readings=readings,
                                 now=pd.Timestamp("2026-11-18 06:00:00"))
        self.assertEqual(result["scored"], 8)
        self.assertEqual(result["missing"], 2)    # hour 05, two horizons
        self.assertEqual(result["pending"], 38)   # hours 06-24, two horizons each

    def test_long_outage_becomes_missing_not_pending(self):
        # The feed died after hour 04; a day later the unread hours are
        # permanently missing rather than still due.
        readings = self.readings.iloc[:4]
        result = scorecard.score(days=7, path=self.path, readings=readings,
                                 now=pd.Timestamp("2026-11-19 01:00:00"))
        self.assertEqual(result["scored"], 8)
        self.assertEqual(result["pending"], 0)
        self.assertEqual(result["missing"], 40)

    def test_missing_rows_are_excluded_from_errors(self):
        readings = self.readings.iloc[:12]
        result = scorecard.score(days=7, path=self.path,
                                 readings=readings, now=pd.Timestamp("2026-11-18 23:00:00"))
        self.assertEqual(result["scored"], 24)
        self.assertEqual(result["pending"], 4)    # hours 23 and 24
        self.assertEqual(result["missing"], 20)   # hours 13-22

    def test_empty_log_is_honest(self):
        empty = os.path.join(self.dir, "none.csv")
        result = scorecard.score(days=7, path=empty, readings=self.readings)
        self.assertEqual(result["scored"], 0)
        self.assertTrue(result["note"])


class ForecastLogHygiene(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "forecast_log.csv")
        base = pd.Timestamp("2026-11-17T00:00:00")
        self.wave = [
            {"horizon": f"+{h}h", "iso": (base + pd.Timedelta(hours=h)).isoformat(), "aqi": 300 + h}
            for h in (4, 8, 12, 16, 20, 24)
        ]

    def test_concurrent_requests_write_each_target_once(self):
        threads = [threading.Thread(target=scorecard.log_issue,
                                    args=(self.wave, 320),
                                    kwargs={"path": self.path}) for _ in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(scorecard.load(self.path)), 6)

    def test_repeated_issue_does_not_double_count(self):
        for _ in range(5):
            scorecard.log_issue(self.wave, 320, path=self.path)
        rows = scorecard.load(self.path)
        self.assertEqual(len(rows), 6)
        self.assertEqual(len(rows.drop_duplicates(subset=["horizon", "target_iso"])), 6)

    def test_file_is_not_reparsed_while_unchanged(self):
        scorecard.log_issue(self.wave, 320, path=self.path)
        first = scorecard.load(self.path)
        second = scorecard.load(self.path)
        self.assertIs(first, second)  # same cached frame, no second read

    def test_rows_past_retention_are_pruned(self):
        original = scorecard.RETENTION_DAYS
        scorecard.RETENTION_DAYS = 2
        try:
            old = pd.DataFrame([{
                "issued_utc": "2026-01-01 00:00:00", "mode": "live_ground",
                "horizon": "+4h", "lead_hours": 4,
                "target_iso": "2026-01-01T04:00:00", "base_aqi": 300, "pred": 320,
                "pi_low": "", "pi_high": "", "severe_pct": "", "trained_at": "t0",
            }], columns=scorecard.COLUMNS)
            old.to_csv(self.path, index=False)
            scorecard._CACHE.pop(self.path, None)
            scorecard.log_issue(self.wave, 320, path=self.path)
            rows = scorecard.load(self.path)
            self.assertEqual(len(rows), 6)
            self.assertGreater(rows["target_iso"].min(), pd.Timestamp("2026-11-01"))
        finally:
            scorecard.RETENTION_DAYS = original


class RetrainDueRules(unittest.TestCase):
    def setUp(self):
        import retrain
        self.retrain = retrain
        self.args = retrain.parse_args([])

    def _state(self, days_ago, rows):
        from datetime import datetime, timedelta, timezone
        when = datetime.now(timezone.utc) - timedelta(days=days_ago)
        return {"last_attempt_utc": when.strftime("%Y-%m-%d %H:%M:%S UTC"), "last_live_rows": rows}

    def test_never_due_before_the_minimum(self):
        due, reason = self.retrain.due_check(self.args, 100, {})
        self.assertFalse(due)
        self.assertIn("need 720", reason)

    def test_due_first_time_with_enough_rows(self):
        due, _ = self.retrain.due_check(self.args, 800, {})
        self.assertTrue(due)

    def test_not_due_too_soon_or_too_few_new_rows(self):
        due, _ = self.retrain.due_check(self.args, 900, self._state(1, 800))
        self.assertFalse(due)
        due, _ = self.retrain.due_check(self.args, 850, self._state(10, 800))
        self.assertFalse(due)
        due, _ = self.retrain.due_check(self.args, 1000, self._state(10, 800))
        self.assertTrue(due)


class LiveSegmentation(unittest.TestCase):
    def test_gap_splits_segments(self):
        import ground_feed
        import retrain
        hours = list(pd.date_range("2026-11-01", periods=10, freq="h")) + \
                list(pd.date_range("2026-11-02", periods=6, freq="h"))
        log = pd.DataFrame({
            "Datetime": hours,
            "AQI": np.linspace(200, 260, len(hours)),
            "Temp_C": 20.0, "Humidity": 60.0, "Wind_Speed": 5.0,
        })
        segments = retrain._segment_live_log(log, ground_feed.MAX_INTERP_GAP)
        self.assertEqual(len(segments), 2)


if __name__ == "__main__":
    unittest.main()
