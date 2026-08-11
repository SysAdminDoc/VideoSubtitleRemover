from __future__ import annotations

import unittest
from types import SimpleNamespace

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


class RegionSaveGuardTests(unittest.TestCase):
    """RM-170: Save must not silently discard a polygon or drawn rects.

    The config carries rects, spans and keyframe tracks -- there is no field
    for a static polygon. Saving with only a polygon fell past the
    `if not self.rects` early return, landed in the else branch, cleared
    every manual-region field and reported "Cleared manual subtitle regions",
    so the shape vanished and detection silently reverted to automatic.
    """

    def _editor(self, **state):
        from gui.region_controller import RegionSelectorWindow

        editor = RegionSelectorWindow.__new__(RegionSelectorWindow)
        editor.pending_keyframes = []
        editor.polygon_points = []
        editor.polygon_shapes = []
        editor.rects = []
        editor.region_spans = []
        editor.keyframe_tracks = []
        editor.is_video = False
        editor.config = SimpleNamespace(
            subtitle_area=None, subtitle_areas=None,
            subtitle_region_spans=None, subtitle_region_keyframes=None,
        )
        editor.messages = []
        editor._update_status = lambda text, tone="info": (
            editor.messages.append((text, tone))
        )
        editor._commit_motion_track = lambda: True
        editor.closed = False
        editor._close = lambda: setattr(editor, "closed", True)
        for key, value in state.items():
            setattr(editor, key, value)
        return editor

    def test_unfinished_polygon_blocks_save(self):
        from gui.region_controller import RegionSelectorWindow

        editor = self._editor(polygon_points=[(1, 1), (5, 5), (9, 1)])
        RegionSelectorWindow._save_and_close(editor)

        self.assertTrue(editor.messages)
        text, tone = editor.messages[-1]
        self.assertEqual(tone, "warning")
        self.assertIn("polygon", text.lower())
        self.assertIsNone(editor.config.subtitle_areas)
        self.assertFalse(editor.closed)

    def test_finished_polygon_is_not_silently_dropped(self):
        from gui.region_controller import RegionSelectorWindow

        editor = self._editor(polygon_shapes=[[1, 1, 5, 5, 9, 1]])
        RegionSelectorWindow._save_and_close(editor)

        text, tone = editor.messages[-1]
        self.assertEqual(tone, "warning")
        # The old behavior reported a *success* that cleared the regions.
        self.assertNotIn("cleared", text.lower())
        self.assertIsNone(editor.config.subtitle_areas)
        self.assertFalse(editor.closed)
