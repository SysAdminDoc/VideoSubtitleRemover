from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from backend import processor
from backend.matte_interchange import (
    MASK_INTERCHANGE_SCHEMA,
    MaskInterchangeReader,
    MaskInterchangeWriter,
    compose_imported_matte,
    mask_interchange_paths,
)


class MatteInterchangeTests(unittest.TestCase):
    @staticmethod
    def _write_masks(writer: MaskInterchangeWriter, values: list[int]):
        for value in values:
            writer.write(np.full((12, 16), value, dtype=np.uint8))

    def test_png_round_trip_accepts_edits_and_validates_exact_timing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "cleaned.mp4"
            timestamps = [0.08, 0.12, 0.20]
            durations = [0.04, 0.08, 0.04]
            writer = MaskInterchangeWriter(
                output, "png", width=16, height=12, fps=25.0,
                start_frame=2, end_frame=5,
                timestamps=timestamps, durations=durations,
                is_vfr=True, source_time_base=0.001,
            )
            self._write_masks(writer, [0, 96, 255])
            evidence = writer.finalize()
            artifact, manifest = mask_interchange_paths(output, "png")

            self.assertEqual(evidence["schema"], MASK_INTERCHANGE_SCHEMA)
            self.assertTrue(manifest.is_file())
            self.assertEqual(len(list(artifact.glob("frame_*.png"))), 3)
            edited = np.full((12, 16), 173, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(
                str(artifact / "frame_00000001.png"), edited))

            reader = MaskInterchangeReader(
                manifest, width=16, height=12, start_frame=2, end_frame=5,
                timestamps=timestamps, durations=durations, is_vfr=True,
                source_time_base=0.001, mode="replace",
            )
            try:
                self.assertTrue(reader.evidence["edited_since_export"])
                self.assertTrue(np.all(reader.read(1) == 173))
            finally:
                reader.close()

            with self.assertRaisesRegex(ValueError, "timestamp mismatch"):
                MaskInterchangeReader(
                    manifest, width=16, height=12,
                    start_frame=2, end_frame=5,
                    timestamps=[0.08, 0.13, 0.20],
                    durations=durations, is_vfr=True,
                    source_time_base=0.001, mode="add",
                )
            with self.assertRaisesRegex(ValueError, "width"):
                MaskInterchangeReader(
                    manifest, width=17, height=12,
                    start_frame=2, end_frame=5,
                    timestamps=timestamps,
                    durations=durations, is_vfr=True,
                    source_time_base=0.001, mode="add",
                )

    def test_ffv1_round_trip_is_exact_gray8(self):
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not on PATH")
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "cleaned.mp4"
            writer = MaskInterchangeWriter(
                output, "ffv1", width=16, height=12, fps=10.0,
                start_frame=0, end_frame=3,
                timestamps=[0.0, 0.1, 0.2],
                durations=[0.1, 0.1, 0.1],
                is_vfr=False, source_time_base=0.1,
            )
            self._write_masks(writer, [0, 127, 255])
            evidence = writer.finalize()
            reader = MaskInterchangeReader(
                evidence["manifest"], width=16, height=12,
                start_frame=0, end_frame=3,
                timestamps=[0.0, 0.1, 0.2],
                durations=[0.1, 0.1, 0.1], is_vfr=False,
                source_time_base=0.1,
                mode="add",
            )
            try:
                self.assertEqual(
                    [int(reader.read(index)[0, 0]) for index in range(3)],
                    [0, 127, 255],
                )
            finally:
                reader.close()

    def test_composition_modes_preserve_soft_alpha(self):
        base = np.asarray([[0, 50, 200, 255, 200]], dtype=np.uint8)
        imported = np.asarray([[10, 100, 0, 255, 75]], dtype=np.uint8)
        self.assertEqual(
            compose_imported_matte(base, imported, "replace").tolist(),
            imported.tolist(),
        )
        self.assertEqual(
            compose_imported_matte(base, imported, "add").tolist(),
            [[10, 100, 200, 255, 200]],
        )
        self.assertEqual(
            compose_imported_matte(base, imported, "subtract").tolist(),
            [[0, 0, 200, 0, 125]],
        )


class MattePipelineIntegrationTests(unittest.TestCase):
    @staticmethod
    def _write_clip(path: Path):
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
        if not writer.isOpened():
            raise unittest.SkipTest("OpenCV MJPG writer unavailable")
        try:
            for value in (30, 60, 90, 120):
                writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
        finally:
            writer.release()

    @staticmethod
    def _stub_remover(config, inpainter):
        remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        remover.config = processor.normalize_processing_config(config)
        remover.detector = processor.SubtitleDetector.__new__(
            processor.SubtitleDetector)
        remover.detector.device = "cpu"
        remover.detector.lang = "en"
        remover.detector.vertical = False
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

    def test_edited_manifest_matte_reaches_inpainter_and_sidecar(self):
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not on PATH")

        class PassthroughInpainter:
            def inpaint(self, frames, _masks):
                return frames

        class RecordingInpainter:
            def __init__(self):
                self.mask_sums = []

            def inpaint(self, frames, masks):
                self.mask_sums.extend(int(mask.sum()) for mask in masks)
                return frames

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.avi"
            exported_output = root / "exported.mp4"
            imported_output = root / "imported.mp4"
            self._write_clip(source)
            export_config = processor.ProcessingConfig(
                mode=processor.InpaintMode.STTN,
                device="cpu",
                sttn_skip_detection=True,
                preserve_audio=False,
                adaptive_batch=False,
                use_hw_encode=False,
                prefetch_decode=False,
                sttn_max_load_num=4,
                subtitle_area=(8, 32, 56, 44),
                export_mask_video=True,
                mask_export_format="png",
            )
            exporter = self._stub_remover(
                export_config, PassthroughInpainter())
            self.assertTrue(exporter.process_video(
                str(source), str(exported_output)))
            manifest = Path(exporter.last_mask_export["manifest"])
            artifact = Path(exporter.last_mask_export["path"])
            for index in range(4):
                edited = np.full(
                    (48, 64), 211 if index == 1 else 0, dtype=np.uint8)
                self.assertTrue(cv2.imwrite(
                    str(artifact / f"frame_{index:08d}.png"), edited))

            recorder = RecordingInpainter()
            import_config = processor.ProcessingConfig(
                mode=processor.InpaintMode.STTN,
                device="cpu",
                sttn_skip_detection=True,
                preserve_audio=False,
                adaptive_batch=False,
                use_hw_encode=False,
                prefetch_decode=False,
                sttn_max_load_num=4,
                subtitle_area=(8, 32, 56, 44),
                mask_import_path=str(manifest),
                mask_import_mode="replace",
            )
            importer = self._stub_remover(import_config, recorder)
            self.assertTrue(importer.process_video(
                str(source), str(imported_output)))
            self.assertEqual(recorder.mask_sums, [0, 211 * 64 * 48, 0, 0])
            self.assertTrue(importer.last_mask_import["edited_since_export"])
            sidecar = json.loads(Path(
                str(imported_output) + ".vsr.json").read_text(encoding="utf-8"))
            self.assertEqual(sidecar["maskImport"]["mode"], "replace")
            self.assertIn("artifact_sha256", sidecar["maskImport"])
            self.assertEqual(
                sidecar["maskImport"]["composition_order"][-1],
                "imported_matte_replace",
            )


if __name__ == "__main__":
    unittest.main()
