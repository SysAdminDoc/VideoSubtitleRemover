"""RM-317: image and mask file I/O on paths that are not pure ASCII.

`cv2.imwrite` hands the filename to OpenCV's C++ layer as a narrow byte
string, so on Windows any path holding CJK, Cyrillic, or accented Latin
characters fails by returning False, and `cv2.imread` returns None for the
same path. These tests drive the real product paths through such a
directory and assert the artefacts land with the right dimensions.
"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

import backend.processor as processor
from backend.matte_interchange import (
    MaskInterchangeReader,
    MaskInterchangeWriter,
    mask_interchange_paths,
)
from backend.safe_image import safe_imread, safe_imwrite


# CJK, Cyrillic, and accented Latin in one name, written with escapes so
# this source file stays pure ASCII like the rest of the tree.
NON_ASCII_DIR = "\u4e2d\u6587_\u0420\u0443\u0441\u0441\u043a\u0438\u0439_caf\u00e9"
NON_ASCII_STEM = "\u5b57\u5e55_\u0442\u0435\u0441\u0442_r\u00e9sum\u00e9"


class _FakeDetector:
    def detect(self, *_args, **_kwargs):
        return [(4, 4, 28, 18)]


class _FakeInpainter:
    def inpaint(self, frames, masks):
        output = []
        for frame, mask in zip(frames, masks, strict=True):
            cleaned = frame.copy()
            cleaned[np.asarray(mask) != 0] = 0
            output.append(cleaned)
        return output


class NonAsciiPathTests(unittest.TestCase):
    def _work_dir(self, root: str) -> Path:
        work = Path(root) / NON_ASCII_DIR
        work.mkdir()
        return work

    def test_safe_helpers_round_trip_every_supported_extension(self):
        image = np.zeros((6, 8, 3), dtype=np.uint8)
        image[:, :, 2] = 200
        with tempfile.TemporaryDirectory() as tmpdir:
            work = self._work_dir(tmpdir)
            for ext in (".png", ".jpg", ".webp", ".bmp"):
                with self.subTest(ext=ext):
                    path = work / (NON_ASCII_STEM + ext)
                    self.assertTrue(safe_imwrite(path, image))
                    self.assertTrue(path.is_file())
                    self.assertGreater(path.stat().st_size, 0)
                    decoded = safe_imread(path)
                    self.assertIsNotNone(decoded)
                    self.assertEqual(decoded.shape, image.shape)
                    self.assertEqual(decoded[0, 0].tolist(), [0, 0, 200])

    def test_grayscale_flag_survives_a_non_ascii_path(self):
        import cv2

        image = np.zeros((6, 8, 3), dtype=np.uint8)
        image[:, :, 1] = 180
        with tempfile.TemporaryDirectory() as tmpdir:
            work = self._work_dir(tmpdir)
            path = work / (NON_ASCII_STEM + ".png")
            self.assertTrue(safe_imwrite(path, image))
            gray = safe_imread(path, cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(gray)
            self.assertEqual(gray.shape, (6, 8))

    def test_image_cleanup_writes_its_output_under_a_non_ascii_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            work = self._work_dir(tmpdir)
            source = work / (NON_ASCII_STEM + ".png")
            output = work / (NON_ASCII_STEM + "_clean.png")
            image = np.full((32, 48, 3), 128, dtype=np.uint8)
            self.assertTrue(safe_imwrite(source, image))

            remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
            remover.config = processor.ProcessingConfig()
            remover.detector = _FakeDetector()
            remover.inpainter = _FakeInpainter()
            remover.on_progress = None
            remover.last_stage_timings = remover._empty_stage_timings()

            self.assertTrue(remover.process_image(str(source), str(output)))
            self.assertTrue(output.is_file())
            decoded = safe_imread(output)
            self.assertIsNotNone(decoded)
            self.assertEqual(decoded.shape, image.shape)

    def test_mask_export_and_import_round_trip_under_a_non_ascii_path(self):
        timestamps = [0.0, 0.04, 0.08]
        durations = [0.04, 0.04, 0.04]
        with tempfile.TemporaryDirectory() as tmpdir:
            work = self._work_dir(tmpdir)
            output = work / (NON_ASCII_STEM + ".mp4")
            writer = MaskInterchangeWriter(
                output, "png", width=16, height=12, fps=25.0,
                start_frame=0, end_frame=3,
                timestamps=timestamps, durations=durations,
                is_vfr=False, source_time_base=0.001,
            )
            for value in (0, 96, 255):
                writer.write(np.full((12, 16), value, dtype=np.uint8))
            writer.finalize()

            artifact, manifest = mask_interchange_paths(output, "png")
            self.assertTrue(manifest.is_file())
            frames = sorted(artifact.glob("frame_*.png"))
            self.assertEqual(len(frames), 3)
            for frame_path in frames:
                self.assertGreater(frame_path.stat().st_size, 0)

            edited = np.full((12, 16), 173, dtype=np.uint8)
            self.assertTrue(safe_imwrite(artifact / "frame_00000001.png", edited))

            reader = MaskInterchangeReader(
                manifest, width=16, height=12, start_frame=0, end_frame=3,
                timestamps=timestamps, durations=durations, is_vfr=False,
                source_time_base=0.001, mode="replace",
            )
            try:
                restored = reader.read(1)
                self.assertEqual(restored.shape, (12, 16))
                self.assertTrue(np.all(restored == 173))
            finally:
                reader.close()

    def test_clean_reference_image_loads_from_a_non_ascii_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            work = self._work_dir(tmpdir)
            plate = work / (NON_ASCII_STEM + "_plate.png")
            reference = np.full((12, 16, 3), 77, dtype=np.uint8)
            self.assertTrue(safe_imwrite(plate, reference))

            remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
            remover.config = processor.ProcessingConfig(
                subtitle_region_spans=[{
                    "rect": (2, 2, 12, 8),
                    "start": 0.0,
                    "end": 1.0,
                    "clean_reference": {
                        "path": str(plate),
                        "alignment": "auto",
                        "color_match": True,
                        "min_confidence": 0.65,
                    },
                }],
            )
            remover._initialize_clean_references(16, 12)

            cached = remover._clean_reference_cache[0]
            self.assertEqual(cached.shape, (12, 16, 3))
            self.assertEqual(cached[0, 0].tolist(), [77, 77, 77])

    def test_frame_sequence_writer_works_under_a_non_ascii_temp_dir(self):
        from backend.io import _FrameSequenceWriter

        previous = {key: os.environ.get(key) for key in ("TMPDIR", "TEMP", "TMP")}
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                work = self._work_dir(tmpdir)
                for key in previous:
                    os.environ[key] = str(work)
                tempfile.tempdir = None
                staged = Path(tempfile.mkdtemp(prefix="vsr_"))
                self.assertIn(NON_ASCII_DIR, str(staged))

                writer = _FrameSequenceWriter(str(staged / NON_ASCII_STEM))
                frame = np.full((8, 10, 3), 64, dtype=np.uint8)
                writer.write(frame)
                writer.write(frame)
                writer.release()

                written = sorted((staged / NON_ASCII_STEM).glob("frame_*.png"))
                self.assertEqual(len(written), 2)
                for path in written:
                    decoded = safe_imread(path)
                    self.assertIsNotNone(decoded)
                    self.assertEqual(decoded.shape, frame.shape)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            tempfile.tempdir = None


if __name__ == "__main__":
    unittest.main()
