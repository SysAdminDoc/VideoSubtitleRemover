"""RM-301: polygon geometry remains precise without breaking box callers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2

from backend.detection_geometry import DetectionGeometry
from backend.processor import ProcessingConfig, SubtitleRemover
from backend.track_plan import (
    group_detections_into_tracks,
    load_track_plan,
    plan_to_mask_corrections,
)
from backend.tracking import SubtitleTracker


class PolygonMaskTests(unittest.TestCase):
    def test_rotated_text_uses_polygon_not_bounding_rectangle(self):
        points = cv2.boxPoints(((80.0, 70.0), (56.0, 12.0), 45.0))
        detection = DetectionGeometry.from_polygon(points, (160, 180, 3))
        self.assertIsNotNone(detection)
        assert detection is not None

        remover = SubtitleRemover.__new__(SubtitleRemover)
        remover.config = ProcessingConfig(mask_dilate_px=0)
        polygon_mask = remover._create_mask(
            (160, 180, 3),
            [detection.bbox],
            padding=0,
            detections=[detection],
        )
        rectangle_mask = remover._create_mask(
            (160, 180, 3),
            [detection.bbox],
            padding=0,
        )

        self.assertGreater(int(polygon_mask.sum()), 0)
        self.assertLess(
            int(polygon_mask.sum()), int(rectangle_mask.sum()) * 0.7)
        for x, y in detection.polygon or ():
            self.assertEqual(int(polygon_mask[y, x]), 255)


class TrackingGeometryTests(unittest.TestCase):
    def test_tracker_carries_polygon_through_box_smoothing(self):
        first = DetectionGeometry.from_polygon(
            [(20, 30), (60, 20), (70, 40), (30, 50)],
            (100, 120, 3),
        )
        second = DetectionGeometry.from_polygon(
            [(22, 31), (63, 21), (73, 42), (32, 52)],
            (100, 120, 3),
        )
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        tracker = SubtitleTracker()
        first_output = tracker.update_with_geometry([first])
        second_output = tracker.update_with_geometry([second])
        self.assertTrue(first_output[0].polygon)
        self.assertTrue(second_output[0].polygon)
        self.assertEqual(len(second_output[0].polygon or ()), 4)


class TrackPlanGeometryTests(unittest.TestCase):
    def test_polygon_is_serialized_and_old_plans_stay_unchanged(self):
        polygon = [(10, 20), (50, 10), (60, 30), (20, 40)]
        tracks = group_detections_into_tracks([
            (0, [DetectionGeometry.from_polygon(polygon, (80, 100, 3))]),
            (5, [DetectionGeometry.from_polygon(polygon, (80, 100, 3))]),
        ])
        self.assertEqual(tracks[0]["polygon"], [[x, y] for x, y in polygon])
        self.assertEqual(len(tracks[0]["polygon_history"]), 2)

        plan = {
            "schema": "vsr.track_plan.v1",
            "tracks": tracks,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "polygon-plan.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            loaded = load_track_plan(path)
        self.assertEqual(loaded["tracks"][0]["polygon"], tracks[0]["polygon"])
        corrections = plan_to_mask_corrections({
            **plan,
            "tracks": [{**tracks[0], "keep": True}],
        })
        self.assertEqual(len(corrections[0]["polygons"][0]), 8)

    def test_legacy_rectangle_group_has_no_polygon_field(self):
        tracks = group_detections_into_tracks([
            (0, [(10, 10, 40, 30)]),
            (5, [(10, 10, 40, 30)]),
        ])
        self.assertNotIn("polygon", tracks[0])


if __name__ == "__main__":
    unittest.main()
