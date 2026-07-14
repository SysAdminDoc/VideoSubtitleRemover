from __future__ import annotations

import shutil
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from backend import processor


class SelectiveMaskRerunTests(unittest.TestCase):
    @staticmethod
    def _write_clip(path: Path, values: list[int], fps: float = 10.0):
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
        if not writer.isOpened():
            raise unittest.SkipTest("OpenCV MJPG writer unavailable")
        try:
            for value in values:
                writer.write(np.full((48, 64, 3), value, np.uint8))
        finally:
            writer.release()

    @staticmethod
    def _stub_remover(config, inpainter):
        remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        remover.config = processor.normalize_processing_config(config)
        remover.detector = processor.SubtitleDetector.__new__(processor.SubtitleDetector)
        remover.detector.device = "cpu"
        remover.detector.lang = "en"
        remover.detector._engine_name = "skip"
        remover.detector._rapid_model = None
        remover.detector._paddle_model = None
        remover.detector._surya_det = None
        remover.detector._easyocr_reader = None
        remover.inpainter = inpainter
        remover.on_progress = None
        remover.on_preview_frame = None
        remover.live_preview_stride = 6
        remover._hw_encoder = None
        remover._active_writer = None
        remover._active_subprocess = None
        remover._teardown_requested = False
        remover.last_quality_report = None
        remover.last_error_message = None
        remover.last_error_reason = None
        remover.last_resume_warning = None
        remover.last_pause_checkpoint = None
        remover.last_pause_checkpoint_path = None
        return remover

    def test_only_affected_frames_are_inpainted_and_others_are_reused(self):
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not on PATH")

        class MarkInpainter:
            def __init__(self):
                self.frame_count = 0

            def inpaint(self, frames, _masks):
                self.frame_count += len(frames)
                return [np.full_like(frame, 245) for frame in frames]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.avi"
            previous = root / "previous.avi"
            output = previous
            self._write_clip(source, [10, 20, 30, 40, 50, 60])
            self._write_clip(previous, [100] * 6)
            inpainter = MarkInpainter()
            config = processor.ProcessingConfig(
                mode=processor.InpaintMode.STTN,
                device="cpu",
                sttn_skip_detection=True,
                subtitle_area=(8, 30, 56, 44),
                preserve_audio=False,
                adaptive_batch=False,
                use_hw_encode=False,
                prefetch_decode=False,
                sttn_max_load_num=4,
            )
            remover = self._stub_remover(config, inpainter)

            self.assertTrue(remover.process_video(
                str(source), str(output),
                selective_rerun_from=str(previous),
                selective_rerun_ranges=[(2, 4)],
            ))
            self.assertEqual(inpainter.frame_count, 2)
            self.assertEqual(remover.last_selective_rerun["rerun_frames"], 2)
            self.assertEqual(remover.last_selective_rerun["reused_frames"], 4)
            sidecar = json.loads(
                Path(str(output) + ".vsr.json").read_text(encoding="utf-8"))
            self.assertEqual(
                sidecar["selectiveMaskRerun"]["ranges"], [[2, 4]])

            capture = cv2.VideoCapture(str(output))
            means = []
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    means.append(float(frame.mean()))
            finally:
                capture.release()
            self.assertEqual(len(means), 6)
            self.assertTrue(all(80 < means[index] < 120 for index in (0, 1, 4, 5)))
            self.assertTrue(all(means[index] > 220 for index in (2, 3)))


if __name__ == "__main__":
    unittest.main()
