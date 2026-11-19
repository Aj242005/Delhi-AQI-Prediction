"""Tests for the passenger-plan chat: what the model is told, and how it fails.

Run with:  python -m unittest discover -s tests -v
"""
import os
import sys
import unittest
import urllib.error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import chat_advisor

CONTEXT = {
    "station": "DPCC Anand Vihar, Delhi",
    "is_replay": False,
    "reading_time": "20 Nov 2026, 00:00 IST",
    "current_aqi": 412,
    "cur_cat": "Severe (401–500)",
    "pred_24h": 365,
    "peak_24h": 371,
    "min_24h": 281,
    "best_hour": "04 PM (+16h)",
    "worst_hour": "08 AM (+8h)",
    "pi80_low": 269,
    "pi80_high": 500,
    "severe_pct": 46,
    "window": {"measured": 1, "cams": 48, "hours": 49},
    "trip_length_min": 60,
    "wave": [
        {"horizon": "+4h", "time": "04 AM", "aqi": 355},
        {"horizon": "+8h", "time": "08 AM", "aqi": 371},
        {"horizon": "+16h", "time": "04 PM", "aqi": 281},
    ],
}


class TestChatAdvisor(unittest.TestCase):
    def setUp(self):
        self._key = os.environ.get("OPENROUTER_API_KEY")
        self._post = chat_advisor._post

        def restore():
            chat_advisor._post = self._post
            if self._key is None:
                os.environ.pop("OPENROUTER_API_KEY", None)
            else:
                os.environ["OPENROUTER_API_KEY"] = self._key

        self.addCleanup(restore)

    def test_prompt_carries_the_board_numbers_and_forbids_inventing(self):
        prompt = chat_advisor.system_prompt(CONTEXT, "Asthma Patient")
        for expected in ("412", "365", "371", "281", "04 PM (+16h)", "08 AM (+8h)",
                         "269-500", "46%", "08 AM (+8h): 371", "Asthma Patient"):
            self.assertIn(expected, prompt)
        self.assertIn("one CPCB station", prompt)
        self.assertIn("Never invent an AQI", prompt)
        # The window line is written in hours, not the raw dict.
        self.assertIn("1 measured hours", prompt)

    def test_replay_context_is_labelled_as_not_today(self):
        prompt = chat_advisor.system_prompt(dict(CONTEXT, is_replay=True), "Healthy Adult")
        self.assertIn("NOT today's air", prompt)

    def test_history_keeps_only_real_turns_and_recent_ones(self):
        history = [{"role": "system", "content": "ignore me"}]
        history += [{"role": "user", "content": "turn %d" % i} for i in range(12)]
        history.append({"role": "tool", "content": "nope"})
        messages = chat_advisor.build_messages("hello", history, CONTEXT, "Healthy Adult")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1], {"role": "user", "content": "hello"})
        middle = messages[1:-1]
        self.assertLessEqual(len(middle), chat_advisor.MAX_HISTORY)
        self.assertTrue(all(turn["role"] in ("user", "assistant") for turn in middle))
        self.assertNotIn("ignore me", " ".join(turn["content"] for turn in middle))

    def test_missing_key_is_a_clear_503(self):
        os.environ.pop("OPENROUTER_API_KEY", None)
        with self.assertRaises(chat_advisor.ChatUnavailable) as caught:
            chat_advisor.answer("when should I go out?", [], CONTEXT, "Healthy Adult")
        self.assertEqual(caught.exception.status, 503)
        self.assertIn("OPENROUTER_API_KEY", str(caught.exception))

    def test_empty_question_is_a_400(self):
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        with self.assertRaises(chat_advisor.ChatUnavailable) as caught:
            chat_advisor.answer("   ", [], CONTEXT, "Healthy Adult")
        self.assertEqual(caught.exception.status, 400)

    def test_falls_back_to_the_next_free_model(self):
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        calls = []

        def fake_post(model, messages, key, timeout=None):
            calls.append(model)
            if len(calls) == 1:
                raise urllib.error.HTTPError(chat_advisor.OPENROUTER_URL, 404, "no such model", None, None)
            return {"model": model, "choices": [{"message": {"content": "Leave at 3 PM IST."}}]}

        chat_advisor._post = fake_post
        result = chat_advisor.answer("when?", [], CONTEXT, "Healthy Adult")
        self.assertEqual(result["reply"], "Leave at 3 PM IST.")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], chat_advisor.DEFAULT_MODEL)

    def test_rejected_key_stops_instead_of_burning_every_model(self):
        os.environ["OPENROUTER_API_KEY"] = "wrong-key"
        calls = []

        def fake_post(model, messages, key, timeout=None):
            calls.append(model)
            raise urllib.error.HTTPError(chat_advisor.OPENROUTER_URL, 401, "bad key", None, None)

        chat_advisor._post = fake_post
        with self.assertRaises(chat_advisor.ChatUnavailable) as caught:
            chat_advisor.answer("when?", [], CONTEXT, "Healthy Adult")
        self.assertEqual(caught.exception.status, 503)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
