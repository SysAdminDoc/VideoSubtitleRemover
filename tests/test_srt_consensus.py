from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from backend._srt_mixin import (
    SrtFrameObservation,
    SrtTextObservation,
)
from backend.config import ProcessingConfig
from backend.detection_geometry import DetectionGeometry
from backend.io import VideoFrameTiming
from backend.processor import SubtitleRemover
from backend.tracking import (
    SubtitleTracker,
    _group_horizontal_geometry,
)


def _observation(
    frame_idx: int,
    text: str,
    confidence: float = 0.9,
    track_id: int = 7,
) -> SrtFrameObservation:
    return SrtFrameObservation(
        frame_idx,
        (SrtTextObservation(
            track_id,
            text,
            confidence,
            (10, 80, 190, 110),
        ),),
    )


class TrackedOcrObservationTests(unittest.TestCase):
    @staticmethod
    def _remover() -> SubtitleRemover:
        remover = SubtitleRemover.__new__(SubtitleRemover)
        remover.config = ProcessingConfig(ocr_fix_enable=False)
        remover._srt_entries = []
        return remover

    def test_tracker_keeps_text_confidence_on_stable_identity(self):
        tracker = SubtitleTracker(iou_threshold=0.2, max_age=2)
        first = tracker.update_with_geometry([
            DetectionGeometry(
                (10, 20, 110, 50),
                confidence=0.55,
                text="Helo",
            )
        ])[0]
        missing_text = tracker.update_with_geometry([
            DetectionGeometry(
                (12, 20, 112, 50),
                confidence=0.2,
                text="",
            )
        ])[0]
        corrected = tracker.update_with_geometry([
            DetectionGeometry(
                (14, 20, 114, 50),
                confidence=0.96,
                text="Hello",
            )
        ])[0]

        self.assertEqual(first.track_id, missing_text.track_id)
        self.assertEqual(first.track_id, corrected.track_id)
        self.assertEqual(missing_text.text, "Helo")
        self.assertEqual(missing_text.confidence, 0.55)
        self.assertEqual(corrected.text, "Hello")
        self.assertEqual(corrected.confidence, 0.96)

    def test_new_spatial_track_gets_a_distinct_identity(self):
        tracker = SubtitleTracker(iou_threshold=0.3, max_age=2)
        first = tracker.update_with_geometry([
            DetectionGeometry((10, 10, 60, 30), text="first")
        ])[0]
        records = tracker.update_with_geometry([
            DetectionGeometry((200, 10, 260, 30), text="second")
        ])

        self.assertEqual(len({item.track_id for item in records}), 2)
        self.assertIn(first.track_id, {item.track_id for item in records})

    def test_karaoke_grouping_preserves_text_and_confidence(self):
        grouped = _group_horizontal_geometry([
            DetectionGeometry((10, 10, 50, 30), confidence=0.8, text="Hello"),
            DetectionGeometry((55, 10, 100, 30), confidence=0.6, text="world"),
        ])

        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0].text, "Hello world")
        self.assertAlmostEqual(grouped[0].confidence, 0.7)

    def test_collector_uses_tracked_text_without_second_ocr_call(self):
        remover = self._remover()
        detection = DetectionGeometry(
            (10, 20, 110, 50),
            confidence=0.94,
            text="Already recognized",
            track_id=12,
        )

        with mock.patch.object(
            remover,
            "_read_text_for_boxes",
            side_effect=AssertionError("unexpected second OCR call"),
        ):
            remover._collect_srt_entry(
                np.zeros((80, 160, 3), dtype=np.uint8),
                3,
                [detection],
            )

        entry = remover._srt_entries[0]
        self.assertEqual(entry.text, "Already recognized")
        self.assertEqual(entry.track_ids, frozenset({12}))
        self.assertAlmostEqual(entry.confidence, 0.94)

    def test_collector_falls_back_when_recognized_text_is_missing(self):
        remover = self._remover()
        detection = DetectionGeometry(
            (10, 20, 110, 50),
            confidence=0.7,
            text="",
            track_id=4,
        )

        with mock.patch.object(
            remover,
            "_read_text_for_boxes",
            return_value="Fallback text",
        ) as read_text:
            remover._collect_srt_entry(
                np.zeros((80, 160, 3), dtype=np.uint8),
                5,
                [detection],
            )

        read_text.assert_called_once()
        self.assertEqual(remover._srt_entries[0].text, "Fallback text")

    def test_rtl_parts_follow_visual_reading_order(self):
        remover = self._remover()
        right = DetectionGeometry(
            (80, 20, 140, 50),
            text="\u0645\u0631\u062d\u0628\u0627",
            track_id=1,
        )
        left = DetectionGeometry(
            (20, 20, 70, 50),
            text="\u0628\u0643\u0645",
            track_id=2,
        )

        remover._collect_srt_entry(
            np.zeros((80, 160, 3), dtype=np.uint8),
            0,
            [left, right],
        )

        self.assertEqual(
            remover._srt_entries[0].text,
            "\u0645\u0631\u062d\u0628\u0627 \u0628\u0643\u0645",
        )


class SrtConsensusTests(unittest.TestCase):
    @staticmethod
    def _write(entries, *, fps=10.0, frame_timing=None) -> str:
        remover = SubtitleRemover.__new__(SubtitleRemover)
        remover.config = ProcessingConfig()
        remover._srt_entries = entries
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "captions.srt"
            remover._write_srt(
                str(path),
                fps,
                frame_timing=frame_timing,
            )
            return path.read_text(encoding="utf-8")

    def test_latin_typo_uses_confidence_weighted_consensus(self):
        payload = self._write([
            _observation(0, "Hello wor1d", 0.35),
            _observation(1, "Hello world", 0.96),
            _observation(2, "Hello wor1d", 0.30),
        ])

        self.assertEqual(payload.count("-->"), 1)
        self.assertIn("\nHello world\n", payload)

    def test_cjk_low_confidence_fluctuation_stays_one_cue(self):
        correct = "\u4f60\u597d\u4e16\u754c"
        fluctuation = "\u4f60\u59a4\u4e16\u754c"
        payload = self._write([
            _observation(0, fluctuation, 0.30),
            _observation(1, correct, 0.97),
        ])

        self.assertEqual(payload.count("-->"), 1)
        self.assertIn(correct, payload)

    def test_high_confidence_cjk_caption_change_starts_new_cue(self):
        first = "\u4f60\u597d\u4e16\u754c"
        second = "\u4f60\u597d\u4e16\u4ee3"
        payload = self._write([
            _observation(0, first, 0.96),
            _observation(1, second, 0.95),
        ])

        self.assertEqual(payload.count("-->"), 2)
        self.assertIn(first, payload)
        self.assertIn(second, payload)

    def test_combining_mark_forms_merge_without_losing_text(self):
        payload = self._write([
            _observation(0, "Cafe\u0301", 0.7),
            _observation(1, "Caf\u00e9", 0.95),
        ])

        self.assertEqual(payload.count("-->"), 1)
        self.assertIn("Caf\u00e9", payload)

    def test_rtl_near_equivalent_readings_use_consensus(self):
        correct = (
            "\u0645\u0631\u062d\u0628\u0627 "
            "\u0628\u0643\u0645"
        )
        fluctuation = (
            "\u0645\u0631\u062d\u0628\u0627 "
            "\u0628\u0643\u0646"
        )
        payload = self._write([
            _observation(0, fluctuation, 0.25),
            _observation(1, correct, 0.98),
        ])

        self.assertEqual(payload.count("-->"), 1)
        self.assertIn(correct, payload)

    def test_genuinely_changed_latin_caption_is_not_merged(self):
        payload = self._write([
            _observation(0, "Open the door", 0.97),
            _observation(1, "Close the window", 0.96),
        ])

        self.assertEqual(payload.count("-->"), 2)

    def test_tracked_consensus_retains_exact_vfr_frame_timing(self):
        timing = VideoFrameTiming(
            timestamps=[0.0, 0.04, 0.12],
            durations=[0.04, 0.08, 0.05],
            time_base=0.001,
            average_fps=17.647058,
            source_start=0.0,
            is_vfr=True,
        )
        payload = self._write(
            [
                _observation(0, "Exact clock", 0.9),
                _observation(1, "Exact c1ock", 0.3),
            ],
            fps=25.0,
            frame_timing=timing,
        )

        self.assertIn(
            "00:00:00,000 --> 00:00:00,120",
            payload,
        )

    def test_legacy_tuple_entries_remain_supported(self):
        payload = self._write([
            (0, "Legacy"),
            (1, "Legacy"),
        ])

        self.assertEqual(payload.count("-->"), 1)
        self.assertIn("\nLegacy\n", payload)


if __name__ == "__main__":
    unittest.main()
