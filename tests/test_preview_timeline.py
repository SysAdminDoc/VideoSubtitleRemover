"""RM-300 preview planning and timestamp regression tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class PreviewTimelineTests(unittest.TestCase):
    @staticmethod
    def _scene_cut_clip(path: Path) -> Path:
        import cv2
        import numpy as np

        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), 4.0, (80, 48))
        if not writer.isOpened():
            raise unittest.SkipTest("OpenCV MJPG video writer unavailable")
        try:
            for value in (20, 20, 20, 230, 230, 230):
                writer.write(np.full((48, 80, 3), value, dtype=np.uint8))
        finally:
            writer.release()
        return path

    def test_scene_bounded_window_stops_at_cut_after_selected_frame(self):
        from backend.proxy_workflow import probe_proxy_window

        with tempfile.TemporaryDirectory() as tmpdir:
            clip = self._scene_cut_clip(Path(tmpdir) / "cut.avi")
            plan = probe_proxy_window(str(clip), 0.5, radius_frames=1)

        self.assertGreater(plan["timestamp"], 0.0)
        self.assertEqual(plan["target_frame"], 2)
        self.assertEqual(plan["frame_indices"], (1, 2))
        self.assertEqual(plan["frame_start"], 1)
        self.assertEqual(plan["frame_end"], 2)
        self.assertTrue(plan["scene_cut_after"])
        self.assertEqual(plan["proxy_resolution"], "80x48")

    def test_scene_bounded_window_stops_at_cut_before_selected_frame(self):
        from backend.proxy_workflow import probe_proxy_window

        with tempfile.TemporaryDirectory() as tmpdir:
            clip = self._scene_cut_clip(Path(tmpdir) / "cut.avi")
            plan = probe_proxy_window(str(clip), 1.0, radius_frames=2)

        self.assertEqual(plan["target_frame"], 4)
        self.assertEqual(plan["frame_indices"], (3, 4, 5))
        self.assertTrue(plan["scene_cut_before"])

    def test_selected_timestamp_is_clamped_to_video_duration(self):
        from gui.preview_controller import PreviewControllerMixin

        controller = PreviewControllerMixin.__new__(PreviewControllerMixin)
        controller._preview_time_by_item = {"clip": 9.0}
        controller._preview_video_info = {"clip": {"duration": 3.25}}

        self.assertEqual(controller._preview_timestamp_for_item("clip"), 3.25)
        self.assertEqual(
            controller._format_preview_timestamp(1.25), "00:00:01.250")

    def test_full_resolution_reader_uses_source_frames_not_proxy_pixels(self):
        from gui.preview_controller import _read_video_frames_at_indices

        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "source.avi"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"MJPG"), 4.0, (80, 48))
            if not writer.isOpened():
                self.skipTest("OpenCV MJPG video writer unavailable")
            try:
                for value in (30, 80, 140, 200):
                    writer.write(np.full((48, 80, 3), value, dtype=np.uint8))
            finally:
                writer.release()

            frames = _read_video_frames_at_indices(str(path), (2,))

        self.assertEqual(len(frames), 1)
        self.assertGreater(float(frames[0].mean()), 100.0)
        self.assertLess(float(frames[0].mean()), 180.0)


if __name__ == "__main__":
    unittest.main()
