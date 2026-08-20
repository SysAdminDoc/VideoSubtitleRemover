"""RM-275: pre-run track plans -- grouping, exclusion, and round-trip."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.track_plan import (
    TRACK_PLAN_SCHEMA,
    group_detections_into_tracks,
    load_track_plan,
    plan_to_mask_corrections,
    save_track_plan,
    scan_track_plan,
)


class GroupingTests(unittest.TestCase):
    def _samples(self):
        samples = []
        for index, frame in enumerate(range(0, 100, 5)):
            detections = []
            if index != 4:  # one flickered miss mid-track
                detections.append((100, 400, 300, 440, 0.9, "HELLO"))
            if frame >= 50:
                detections.append((500, 50, 700, 90, 0.8, "NEWS"))
            samples.append((frame, detections))
        return samples

    def test_flicker_does_not_split_a_track(self):
        tracks = group_detections_into_tracks(self._samples())

        self.assertEqual(len(tracks), 2)
        self.assertEqual(tracks[0]["start_frame"], 0)
        self.assertEqual(tracks[0]["end_frame"], 95)
        self.assertEqual(tracks[0]["sample_text"], "HELLO")
        self.assertEqual(tracks[1]["start_frame"], 50)
        self.assertEqual(tracks[1]["sample_text"], "NEWS")

    def test_a_long_gap_produces_two_tracks(self):
        samples = [(f, [(10, 10, 60, 40)]) for f in range(0, 30, 5)]
        samples += [(f, []) for f in range(30, 60, 5)]
        samples += [(f, [(10, 10, 60, 40)]) for f in range(60, 90, 5)]

        tracks = group_detections_into_tracks(samples, gap_samples=2)

        self.assertEqual(len(tracks), 2)
        self.assertEqual(
            [track["start_frame"] for track in tracks], [0, 60])

    def test_grouping_is_deterministic(self):
        first = group_detections_into_tracks(self._samples())
        second = group_detections_into_tracks(self._samples())
        self.assertEqual(first, second)

    def test_degenerate_boxes_are_ignored(self):
        tracks = group_detections_into_tracks(
            [(0, [(10, 10, 10, 40), (5, 5, 4, 4)])])
        self.assertEqual(tracks, [])


class ExclusionTests(unittest.TestCase):
    def _plan(self, keep):
        return {
            "schema": TRACK_PLAN_SCHEMA,
            "tracks": [{
                "id": 1, "start_frame": 100, "end_frame": 200,
                "bbox": [10, 10, 60, 40], "sample_text": "KEEP",
                "keep": keep,
            }],
        }

    def test_a_kept_track_is_excluded_for_exactly_its_span(self):
        from backend.mask_corrections import apply_mask_corrections

        corrections = plan_to_mask_corrections(self._plan(keep=True))

        def cleared_at(frame_index):
            mask = np.full((80, 80), 255, dtype=np.uint8)
            out = apply_mask_corrections(
                mask, corrections, frame_index / 30.0,
                frame_index=frame_index)
            return int(out[20, 20]) == 0

        self.assertFalse(cleared_at(99))
        self.assertTrue(cleared_at(100))
        self.assertTrue(cleared_at(200))
        self.assertFalse(cleared_at(201))

    def test_the_exclusion_covers_only_the_padded_box(self):
        from backend.mask_corrections import apply_mask_corrections

        corrections = plan_to_mask_corrections(self._plan(keep=True))
        mask = np.full((80, 80), 255, dtype=np.uint8)
        out = apply_mask_corrections(mask, corrections, 5.0, frame_index=150)

        self.assertEqual(int(out[20, 20]), 0)
        self.assertEqual(int(out[70, 70]), 255)

    def test_a_removed_track_contributes_nothing(self):
        self.assertEqual(plan_to_mask_corrections(self._plan(keep=False)), [])


class RoundTripTests(unittest.TestCase):
    def test_save_load_preserves_keep_flags_and_spans(self):
        plan = {
            "schema": TRACK_PLAN_SCHEMA,
            "source": "clip.mkv", "fps": 30.0, "frame_count": 300,
            "sample_stride": 8, "detector": "RapidOCR",
            "tracks": [
                {"id": 1, "start_frame": 0, "end_frame": 90,
                 "bbox": [5, 5, 50, 25], "sample_text": "A", "keep": True},
                {"id": 2, "start_frame": 100, "end_frame": 200,
                 "bbox": [5, 40, 50, 60], "sample_text": "B", "keep": False},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            save_track_plan(plan, path)
            loaded = load_track_plan(path)

        self.assertEqual(loaded, plan)
        # Deterministic rerun: the same plan yields the same corrections.
        self.assertEqual(plan_to_mask_corrections(loaded),
                         plan_to_mask_corrections(plan))

    def test_load_rejects_wrong_schema_and_malformed_tracks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            path.write_text(json.dumps({"schema": "nope", "tracks": []}),
                            encoding="utf-8")
            with self.assertRaises(ValueError):
                load_track_plan(path)
            path.write_text(json.dumps({
                "schema": TRACK_PLAN_SCHEMA,
                "tracks": [{"id": 1, "bbox": [0, 0, 5, 5],
                            "start_frame": 9, "end_frame": 3}],
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_track_plan(path)


class _StubDetector:
    """Deterministic detector: text in the bottom band of every frame."""

    _engine_name = "stub"

    def detect_with_text(self, frame, threshold):
        height, width = frame.shape[:2]
        # The synthetic clip draws its caption in the bottom quarter.
        band = frame[int(height * 0.75):, :]
        if int((band > 200).sum()) < 20:
            return []
        return [(10, int(height * 0.75), width - 10, height - 2,
                 0.9, "CAPTION")]

    def detect(self, frame, threshold):
        return [row[:4] for row in self.detect_with_text(frame, threshold)]


class ScanTests(unittest.TestCase):
    def _write_clip(self, path):
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (160, 96))
        self.assertTrue(writer.isOpened())
        try:
            for frame_idx in range(60):
                frame = np.full((96, 160, 3), 40, dtype=np.uint8)
                # Caption on frames 10-39 only.
                if 10 <= frame_idx < 40:
                    cv2.rectangle(frame, (20, 80), (140, 92),
                                  (255, 255, 255), -1)
                writer.write(frame)
        finally:
            writer.release()

    def test_scan_produces_one_track_with_span_and_thumbnail(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "clip.mp4"
            self._write_clip(clip)
            plan = scan_track_plan(
                clip, detector=_StubDetector(), sample_fps=5.0)

        self.assertEqual(plan["schema"], TRACK_PLAN_SCHEMA)
        self.assertEqual(len(plan["tracks"]), 1)
        track = plan["tracks"][0]
        self.assertEqual(track["sample_text"], "CAPTION")
        # The caption lives on frames 10-39; sampling every 4th frame finds
        # it by frame 12 and the end extends by up to one stride.
        self.assertLessEqual(track["start_frame"], 12)
        self.assertGreaterEqual(track["end_frame"], 36)
        self.assertLessEqual(track["end_frame"], 45)
        self.assertTrue(track["thumbnail_png_base64"])

    def test_two_scans_of_the_same_file_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "clip.mp4"
            self._write_clip(clip)
            first = scan_track_plan(
                clip, detector=_StubDetector(), thumbnails=False)
            second = scan_track_plan(
                clip, detector=_StubDetector(), thumbnails=False)

        self.assertEqual(first["tracks"], second["tracks"])


if __name__ == "__main__":
    unittest.main()
