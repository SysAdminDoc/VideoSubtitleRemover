from __future__ import annotations

import unittest

from backend.region_editing import (
    RegionEditHistory,
    format_polygon_vertices,
    frame_to_seconds,
    parse_polygon_vertices,
    rect_from_xywh,
    seconds_to_frame,
    transform_region_shape,
)


class RegionEditingTests(unittest.TestCase):
    def test_numeric_rectangle_validation_preserves_exclusive_edges(self):
        self.assertEqual(
            rect_from_xywh("10", "20", "100", "30", 320, 180),
            (10, 20, 110, 50),
        )
        with self.assertRaisesRegex(ValueError, "frame width"):
            rect_from_xywh("300", "20", "100", "30", 320, 180)
        with self.assertRaisesRegex(ValueError, "whole number"):
            rect_from_xywh("10.5", "20", "100", "30", 320, 180)

    def test_polygon_round_trip_and_bounds(self):
        coords = parse_polygon_vertices("10,20; 100,20; 90,60", 320, 180)
        self.assertEqual(coords, [10, 20, 100, 20, 90, 60])
        self.assertEqual(format_polygon_vertices(coords), "10,20; 100,20; 90,60")
        with self.assertRaisesRegex(ValueError, "within"):
            parse_polygon_vertices("10,20; 400,20; 90,60", 320, 180)

    def test_nudge_and_resize_are_bounded_for_both_shape_types(self):
        self.assertEqual(
            transform_region_shape(
                {"rect": [280, 140, 320, 180]},
                frame_width=320,
                frame_height=180,
                dx=10,
                dy=10,
                dw=-4,
                dh=-5,
            ),
            {"rect": [280, 140, 316, 175]},
        )
        polygon = transform_region_shape(
            {"polygon": [300, 160, 320, 160, 320, 180]},
            frame_width=320,
            frame_height=180,
            dx=10,
            dy=10,
            dw=-5,
            dh=-5,
        )
        self.assertEqual(polygon, {"polygon": [300, 160, 315, 160, 315, 175]})

    def test_time_conversion_and_history_round_trip(self):
        self.assertEqual(seconds_to_frame("1.4", 10.0), 14)
        self.assertEqual(frame_to_seconds("14", 10.0), 1.4)
        history = RegionEditHistory(limit=2)
        original = {"rects": [(1, 2, 3, 4)]}
        changed = {"rects": [(2, 2, 4, 4)]}
        history.record(original)
        restored = history.undo(changed)
        self.assertEqual(restored, original)
        restored["rects"].append((5, 6, 7, 8))
        self.assertEqual(history.redo(restored), changed)


if __name__ == "__main__":
    unittest.main()
