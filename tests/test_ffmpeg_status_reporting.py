"""RM-324: the interface must report the FFmpeg verdict, not an exit code.

The startup status said "FFmpeg ready" whenever `ffmpeg -version` exited 0,
so a build below the enforced 9.0.1 floor looked healthy at launch and then
failed hard once processing reached the security check. The backend already
classified the banner; nothing carried the answer to the user.
"""

from __future__ import annotations

import unittest
from unittest import mock

from backend.ffmpeg_profiles import (
    classify_ffmpeg_security,
    ffmpeg_security_floor_str,
)
from gui.utils import detect_ffmpeg, detect_ffmpeg_state, ffmpeg_status_summary


def _state(banner: str, *, available: bool = True) -> dict:
    payload = classify_ffmpeg_security(banner)
    payload["available"] = available
    return payload


class FfmpegStatusSummaryTests(unittest.TestCase):
    FLOOR = ffmpeg_security_floor_str()

    def test_a_build_at_the_floor_reads_as_ready(self):
        summary = ffmpeg_status_summary(
            _state("ffmpeg version 9.0.1 Copyright (c) 2000-2026"))
        self.assertTrue(summary["safe"])
        self.assertTrue(summary["available"])
        self.assertEqual(summary["tone"], "success")
        self.assertEqual(summary["warning"], "")
        self.assertIn("9.0.1", summary["status"])

    def test_a_build_below_the_floor_is_a_warning_that_names_the_version(self):
        summary = ffmpeg_status_summary(
            _state("ffmpeg version 7.1 Copyright (c) 2000-2024"))
        self.assertFalse(summary["safe"])
        self.assertTrue(summary["available"])
        self.assertEqual(summary["tone"], "warning")
        self.assertIn("7.1", summary["status"])
        self.assertIn("7.1", summary["warning"])
        self.assertIn(self.FLOOR, summary["warning"])
        self.assertNotIn("ready", summary["status"].lower())

    def test_a_snapshot_banner_is_unclassified_rather_than_ready(self):
        summary = ffmpeg_status_summary(
            _state("ffmpeg version N-125875-gabc1234 Copyright (c) 2000-2026"))
        self.assertFalse(summary["safe"])
        self.assertTrue(summary["available"])
        self.assertEqual(summary["tone"], "warning")
        self.assertIn(self.FLOOR, summary["warning"])
        self.assertNotIn("ready", summary["status"].lower())

    def test_a_missing_binary_says_audio_will_be_dropped(self):
        summary = ffmpeg_status_summary(_state("", available=False))
        self.assertFalse(summary["safe"])
        self.assertFalse(summary["available"])
        self.assertEqual(summary["tone"], "warning")
        self.assertIn("without original audio", summary["warning"])

    def test_an_empty_probe_result_is_treated_as_missing(self):
        for value in (None, {}):
            with self.subTest(value=value):
                summary = ffmpeg_status_summary(value)
                self.assertFalse(summary["available"])
                self.assertFalse(summary["safe"])
                self.assertTrue(summary["warning"])


class FfmpegProbeTests(unittest.TestCase):
    def test_the_probe_returns_the_classification_not_a_boolean(self):
        with mock.patch(
            "backend.ffmpeg_profiles.probe_ffmpeg_security",
            return_value=_state("ffmpeg version 7.1 Copyright (c) 2000-2024"),
        ):
            state = detect_ffmpeg_state()
        self.assertTrue(state["available"])
        self.assertEqual(state["version"], "7.1.0")
        self.assertFalse(state["safe"])

    def test_a_failed_probe_degrades_to_unavailable_rather_than_ready(self):
        with mock.patch(
            "backend.ffmpeg_profiles.probe_ffmpeg_security",
            side_effect=OSError("boom"),
        ):
            state = detect_ffmpeg_state()
        self.assertFalse(state["available"])
        self.assertFalse(ffmpeg_status_summary(state)["safe"])

    def test_availability_stays_the_audio_merge_question(self):
        """A below-floor build still merges audio, so it is still available."""
        with mock.patch(
            "gui.utils.detect_ffmpeg_state",
            return_value=_state("ffmpeg version 7.1 Copyright (c) 2000-2024"),
        ):
            self.assertTrue(detect_ffmpeg())


if __name__ == "__main__":
    unittest.main()
