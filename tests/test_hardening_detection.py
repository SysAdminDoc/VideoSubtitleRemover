import os
import sys
import tempfile
import types
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from backend import processor


def _has_display() -> bool:
    """Return True if a GUI display is available."""
    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class RapidOcrOutputCompatibilityTests(unittest.TestCase):
    def _detector_with_output(self, output):
        det = processor.SubtitleDetector.__new__(processor.SubtitleDetector)
        det._rapid_model = lambda frame: output
        det._fallback_detection = lambda frame: [(-1, -1, -1, -1)]
        return det

    @staticmethod
    def _frame():
        import numpy as _np
        return _np.zeros((12, 12, 3), dtype=_np.uint8)

    def test_detect_rapid_accepts_legacy_tuple_output(self):
        output = (
            [
                (
                    [[1, 2], [5, 2], [5, 6], [1, 6]],
                    "subtitle",
                    0.9,
                )
            ],
            {"det": 0.01},
        )
        det = self._detector_with_output(output)

        boxes = det._detect_rapid(self._frame(), threshold=0.5)

        self.assertEqual(boxes, [(1, 2, 5, 6)])

    def test_detect_rapid_accepts_structured_object_output(self):
        output = SimpleNamespace(
            dt_polys=[[[1, 1], [7, 1], [7, 4], [1, 4]]],
            rec_scores=[0.8],
            rec_texts=["subtitle"],
        )
        det = self._detector_with_output(output)

        boxes = det._detect_rapid(self._frame(), threshold=0.5)

        self.assertEqual(boxes, [(1, 1, 7, 4)])

    def test_detect_rapid_accepts_structured_dict_output(self):
        output = {
            "dt_polys": [
                [[1, 1], [7, 1], [7, 4], [1, 4]],
                [[2, 5], [9, 5], [9, 8], [2, 8]],
            ],
            "rec_scores": [0.4, 0.95],
        }
        det = self._detector_with_output(output)

        boxes = det._detect_rapid(self._frame(), threshold=0.5)

        self.assertEqual(boxes, [(2, 5, 9, 8)])

    def test_detect_rapid_ignores_malformed_polygons_without_fallback(self):
        output = {"dt_polys": [["bad"], [[3, 3], [3, 3]]], "rec_scores": [1.0, 1.0]}
        det = self._detector_with_output(output)

        boxes = det._detect_rapid(self._frame(), threshold=0.5)

        self.assertEqual(boxes, [])

    def test_rapidocr_config_load_failures_fall_through_cascade(self):
        from unittest import mock

        class MissingConfigRapidOCR:
            def __init__(self, **kwargs):
                raise FileNotFoundError("default_models.yaml")

        absent = {
            name: None
            for name in (
                "rapidocr_onnxruntime",
                "paddleocr",
                "surya",
                "surya.detection",
                "easyocr",
            )
        }
        fake_rapid = SimpleNamespace(RapidOCR=MissingConfigRapidOCR)

        with mock.patch.dict(sys.modules, {"rapidocr": fake_rapid, **absent}):
            det = processor.SubtitleDetector(device="cpu")

        self.assertEqual(det._engine_name, "OpenCV fallback")


class KaraokeGroupingTests(unittest.TestCase):
    """_group_horizontal_line must fuse same-line boxes within the gap
    threshold and leave separate-line boxes alone."""

    def test_empty_input_returns_empty(self):
        self.assertEqual(processor._group_horizontal_line([]), [])

    def test_single_box_passes_through(self):
        self.assertEqual(
            processor._group_horizontal_line([(10, 10, 50, 30)]),
            [(10, 10, 50, 30)],
        )

    def test_five_close_syllables_fuse_into_one(self):
        # Five 30-px-wide syllables on the same line, 10-px gap each.
        syllables = [(i * 40, 100, i * 40 + 30, 140) for i in range(5)]
        merged = processor._group_horizontal_line(
            syllables, x_gap_px=20, y_overlap_ratio=0.5,
        )
        self.assertEqual(len(merged), 1)
        x1, y1, x2, y2 = merged[0]
        self.assertEqual(x1, 0)
        self.assertEqual(x2, 4 * 40 + 30)
        self.assertEqual(y1, 100)
        self.assertEqual(y2, 140)

    def test_boxes_on_separate_lines_are_not_fused(self):
        # Same x range, totally non-overlapping y.
        a = (10, 10, 100, 40)
        b = (10, 200, 100, 240)
        merged = processor._group_horizontal_line(
            [a, b], x_gap_px=20, y_overlap_ratio=0.5,
        )
        self.assertEqual(set(merged), {a, b})

    def test_boxes_with_gap_larger_than_threshold_stay_separate(self):
        a = (0, 100, 30, 140)
        b = (100, 100, 130, 140)   # gap = 70 px
        merged = processor._group_horizontal_line(
            [a, b], x_gap_px=20, y_overlap_ratio=0.5,
        )
        self.assertEqual(set(merged), {a, b})


class ChyronClassifierTests(unittest.TestCase):
    """_KalmanBox.is_chyron + SubtitleTracker.categorize must classify a
    long-lived track as 'chyron' and a short-lived one as 'subtitle'."""

    def test_is_chyron_below_threshold(self):
        box = processor._KalmanBox((10, 10, 50, 30))
        # Fresh box: 1 hit
        self.assertFalse(box.is_chyron(min_hits=90))

    def test_is_chyron_above_threshold(self):
        box = processor._KalmanBox((10, 10, 50, 30))
        for _ in range(120):
            box.update((10, 10, 50, 30))
        self.assertTrue(box.is_chyron(min_hits=90))

    def test_tracker_categorizes_persistent_track_as_chyron(self):
        tr = processor.SubtitleTracker(iou_threshold=0.3, max_age=2)
        for _ in range(120):
            tr.update([(100, 100, 200, 130)])
        cats = tr.categorize(min_chyron_hits=90)
        # Should have exactly one persistent track classified as chyron.
        self.assertEqual(len(cats), 1)
        self.assertEqual(cats[0], "chyron")

    def test_tracker_categorizes_brief_track_as_subtitle(self):
        tr = processor.SubtitleTracker(iou_threshold=0.3, max_age=2)
        for _ in range(10):
            tr.update([(100, 100, 200, 130)])
        cats = tr.categorize(min_chyron_hits=90)
        self.assertEqual(cats, ["subtitle"])


class SuryaOptInTests(unittest.TestCase):
    """B-2: Surya is GPL; the detector cascade must skip it unless the user
    explicitly opts in via VSR_ALLOW_GPL."""

    def setUp(self):
        self._saved = os.environ.pop("VSR_ALLOW_GPL", None)

    def tearDown(self):
        os.environ.pop("VSR_ALLOW_GPL", None)
        if self._saved is not None:
            os.environ["VSR_ALLOW_GPL"] = self._saved

    def test_surya_disallowed_by_default(self):
        self.assertFalse(processor._surya_allowed())

    def test_surya_allowed_when_env_set(self):
        for token in ("1", "true", "yes", "on", "TRUE"):
            os.environ["VSR_ALLOW_GPL"] = token
            self.assertTrue(processor._surya_allowed(), f"token={token}")

    def test_surya_disallowed_for_unknown_tokens(self):
        for token in ("0", "false", "no", "off", "", "maybe"):
            os.environ["VSR_ALLOW_GPL"] = token
            self.assertFalse(processor._surya_allowed(), f"token={token}")


class KaraokeFlowTests(unittest.TestCase):
    """RM-43 / RM-45: optical-flow mask warp and WhisperX availability
    probe behave deterministically."""

    def test_warp_mask_with_flow_preserves_shape(self):
        import numpy as _np
        from backend.karaoke_flow import warp_mask_with_flow
        prev = _np.zeros((32, 32, 3), dtype=_np.uint8)
        nxt = _np.zeros((32, 32, 3), dtype=_np.uint8)
        mask = _np.zeros((32, 32), dtype=_np.uint8)
        mask[10:20, 10:20] = 255
        warped = warp_mask_with_flow(prev, nxt, mask)
        self.assertEqual(warped.shape, mask.shape)

    def test_whisperx_availability_safe(self):
        from backend.karaoke_flow import is_whisperx_available, run_whisperx
        # Don't crash regardless of whether the package is installed.
        self.assertIsInstance(is_whisperx_available(), bool)
        if not is_whisperx_available():
            self.assertIsNone(run_whisperx("/nonexistent.wav"))


class WhisperFallbackTests(unittest.TestCase):
    """RM-27: Whisper fallback adapter must degrade gracefully when the
    optional dep is missing, and the segments_to_frame_spans helper must
    merge overlapping spans deterministically."""

    def test_is_available_handles_missing_dep(self):
        # Force faster-whisper to look absent even on developer machines
        # where optional packages are installed.
        from unittest import mock
        from backend import whisper_fallback as _wf
        with mock.patch.dict(sys.modules, {"faster_whisper": None}):
            self.assertFalse(_wf.is_available())

    def test_run_whisper_segments_returns_none_without_dep(self):
        from backend import whisper_fallback as _wf
        result = _wf.run_whisper_segments("/nonexistent.wav")
        self.assertIsNone(result)

    def test_extract_audio_returns_none_for_missing_file(self):
        from backend import whisper_fallback as _wf
        # ffmpeg may not be on PATH in CI; either branch should return
        # None for a non-existent source.
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _wf.extract_audio_to_temp("/nonexistent.mp4", tmpdir)
            self.assertIsNone(result)

    def test_ffmpeg_whisper_available_detects_filter(self):
        from unittest import mock
        from backend import whisper_fallback as _wf
        completed = SimpleNamespace(
            returncode=0,
            stdout=" .. whisper           A->A       Transcribe audio using whisper.cpp.\n",
            stderr="",
        )
        with mock.patch.object(_wf.shutil, "which", return_value="ffmpeg"):
            with mock.patch.object(_wf, "run_process", return_value=completed):
                self.assertTrue(_wf.ffmpeg_whisper_available())

    def test_ffmpeg_whisper_filter_escapes_windows_paths(self):
        from backend import whisper_fallback as _wf
        expr = _wf._build_ffmpeg_whisper_filter(
            r"C:\models\ggml-base.en.bin",
            r"C:\Temp\whisper.srt",
            language="en",
            queue_seconds=3.0,
        )
        self.assertIn(r"model=C\:\\models\\ggml-base.en.bin", expr)
        self.assertIn(r"destination=C\:\\Temp\\whisper.srt", expr)

    def test_ffmpeg_whisper_filter_includes_vad_options(self):
        from backend import whisper_fallback as _wf
        expr = _wf._build_ffmpeg_whisper_filter(
            "/models/ggml-base.bin",
            "/tmp/whisper.srt",
            vad_model="/models/silero_vad.onnx",
            vad_threshold=0.6,
            min_speech_duration=0.25,
        )
        self.assertIn("vad_model=", expr)
        self.assertIn("vad_threshold=0.6", expr)
        self.assertIn("min_speech_duration=0.25", expr)

    def test_ffmpeg_whisper_filter_omits_vad_when_empty(self):
        from backend import whisper_fallback as _wf
        expr = _wf._build_ffmpeg_whisper_filter(
            "/models/ggml-base.bin",
            "/tmp/whisper.srt",
        )
        self.assertNotIn("vad_model", expr)
        self.assertNotIn("vad_threshold", expr)

    def test_parse_srt_segments(self):
        from backend import whisper_fallback as _wf
        srt = (
            "1\n"
            "00:00:01,250 --> 00:00:02,500\n"
            "Hello world\n\n"
            "2\n"
            "00:00:03,000 --> 00:00:04,000\n"
            "Second line\n"
        )
        self.assertEqual(
            _wf.parse_srt_segments(srt),
            [(1.25, 2.5, "Hello world"), (3.0, 4.0, "Second line")],
        )

    def test_run_ffmpeg_whisper_segments_parses_mocked_srt(self):
        from unittest import mock
        from backend import whisper_fallback as _wf

        def fake_run(cmd, check, capture_output, timeout):
            filt = cmd[cmd.index("-af") + 1]
            dest = filt.split(":destination=", 1)[1]
            dest_path = Path(_wf._unescape_filter_value(dest))
            dest_path.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nspeech\n",
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

        with tempfile.TemporaryDirectory() as tmpdir:
            media = Path(tmpdir) / "input.mp4"
            media.write_bytes(b"fake")
            model = Path(tmpdir) / "ggml-base.en.bin"
            model.write_bytes(b"fake")
            with mock.patch.object(_wf, "ffmpeg_whisper_available", return_value=True):
                with mock.patch.object(_wf, "run_process", side_effect=fake_run):
                    self.assertEqual(
                        _wf.run_ffmpeg_whisper_segments(
                            str(media), str(model), language="en"
                        ),
                        [(0.0, 1.0, "speech")],
                    )

    def test_processing_config_normalizes_whisper_backend_fields(self):
        cfg = processor.normalize_processing_config(
            processor.ProcessingConfig(
                whisper_backend="faster_whisper",
                whisper_model_path="x" * 600,
                whisper_queue_seconds=0.0,
            )
        )
        self.assertEqual(cfg.whisper_backend, "faster-whisper")
        self.assertEqual(len(cfg.whisper_model_path), 512)
        self.assertEqual(cfg.whisper_queue_seconds, 0.02)

        cfg = processor.normalize_processing_config(
            processor.ProcessingConfig(whisper_backend="not-real")
        )
        self.assertEqual(cfg.whisper_backend, "faster-whisper")

    def test_segments_to_frame_spans_merges_overlaps(self):
        from backend import whisper_fallback as _wf
        segments = [
            (0.0, 1.0, "a"),
            (0.5, 2.0, "b"),  # overlaps with previous
            (5.0, 6.0, "c"),
        ]
        spans = _wf.segments_to_frame_spans(segments, fps=24.0)
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0], (0, 48))  # 0..2s at 24 fps
        self.assertEqual(spans[1], (120, 144))

    def test_segments_to_frame_spans_handles_invalid_fps(self):
        from backend import whisper_fallback as _wf
        self.assertEqual(_wf.segments_to_frame_spans([(0.0, 1.0, "a")], fps=0.0), [])


class PySceneDetectAdapterTests(unittest.TestCase):
    """RM-32: PySceneDetect adapter must return None when the optional
    dep is absent, and the histogram path stays the default."""

    def test_adapter_returns_none_without_dep(self):
        import numpy as _np
        from unittest import mock
        frames = [_np.zeros((10, 10, 3), dtype=_np.uint8) for _ in range(3)]
        # Force scenedetect to look absent even on developer machines where
        # optional packages are installed.
        absent = {"scenedetect": None, "scenedetect.detectors": None}
        with mock.patch.dict(sys.modules, absent):
            result = processor._detect_scene_cuts_pyscenedetect(frames)
        self.assertIsNone(result)

    def test_default_path_is_histogram(self):
        import numpy as _np
        frames = [_np.full((10, 10, 3), v, dtype=_np.uint8) for v in (50, 50, 200, 200)]
        cuts = processor._detect_scene_cuts(frames, threshold=0.5)
        # The 50 -> 200 step is a cut; cuts must include index 0 and 2.
        self.assertIn(0, cuts)
        self.assertIn(2, cuts)


class VerticalTextDetectionTests(unittest.TestCase):
    """RM-24: vertical-text mode wraps the detector with a rotate-detect-
    rotate-back layer. Boxes from the rotated frame must come back in
    the original frame's coordinate space."""

    def _make_detector(self, vertical: bool):
        det = processor.SubtitleDetector.__new__(processor.SubtitleDetector)
        det.device = "cpu"
        det.lang = "en"
        det.vertical = vertical
        det._engine_name = "stub"
        det._rapid_model = None
        det._paddle_model = None
        det._surya_det = None
        det._surya_processor = None
        det._easyocr_reader = None
        return det

    def test_vertical_false_short_circuits(self):
        import numpy as _np
        det = self._make_detector(vertical=False)
        det._detect_axis_aligned = lambda f, t: [(10, 20, 30, 40)]
        frame = _np.zeros((100, 200, 3), dtype=_np.uint8)
        self.assertEqual(det.detect(frame, 0.5), [(10, 20, 30, 40)])

    def test_vertical_rotates_boxes_back(self):
        import numpy as _np
        det = self._make_detector(vertical=True)
        # Original frame 200w x 100h. After 90 CCW the rotated frame is
        # 100w x 200h. A box at (rx1=10, ry1=20, rx2=40, ry2=60) in the
        # rotated frame maps back using the inverse rotation:
        # ox = w - ry -> x in [200-60, 200-20] = [140, 180]
        # oy = rx -> y in [10, 40]
        det._detect_axis_aligned = lambda f, t: [(10, 20, 40, 60)]
        frame = _np.zeros((100, 200, 3), dtype=_np.uint8)
        result = det.detect(frame, 0.5)
        self.assertEqual(result, [(140, 10, 180, 40)])


class OcrCascadeOrderTests(unittest.TestCase):
    """T-3: the OCR loader must follow the documented priority order
    (RapidOCR > PaddleOCR > Surya > EasyOCR > OpenCV fallback). We
    patch importlib so the test does not require any optional OCR
    engine to be installed."""

    def _make_detector(self):
        det = processor.SubtitleDetector.__new__(processor.SubtitleDetector)
        det.device = "cpu"
        det.lang = "en"
        det._engine_name = "none"
        det._rapid_model = None
        det._paddle_model = None
        det._surya_det = None
        det._easyocr_reader = None
        return det

    def test_falls_back_to_opencv_when_no_engine_installed(self):
        from unittest import mock

        det = self._make_detector()
        # Honour the class docstring: force every optional OCR engine to
        # look absent so the cascade deterministically reaches the OpenCV
        # fallback. Without this the test passes only on a bare machine and
        # fails in CI/release builds, which install rapidocr/easyocr/etc.
        # (a None entry in sys.modules makes the import raise ImportError).
        absent = {
            name: None
            for name in (
                "rapidocr",
                "rapidocr_onnxruntime",
                "paddleocr",
                "surya",
                "surya.detection",
                "easyocr",
            )
        }
        with mock.patch.dict(sys.modules, absent):
            det._load_model()
        self.assertEqual(det._engine_name, "OpenCV fallback")

    def test_surya_skipped_unless_env_set(self):
        from unittest import mock

        # The cascade should not pick Surya even when its module is
        # importable; gating is via VSR_ALLOW_GPL.
        os.environ.pop("VSR_ALLOW_GPL", None)
        det = self._make_detector()
        absent = {
            name: None
            for name in (
                "rapidocr",
                "rapidocr_onnxruntime",
                "paddleocr",
                "backend.paddle_compat",
                "easyocr",
            )
        }
        surya_pkg = types.ModuleType("surya")
        surya_detection = types.ModuleType("surya.detection")
        surya_detection.DetectionPredictor = lambda: object()
        with mock.patch.dict(
            sys.modules,
            {**absent, "surya": surya_pkg, "surya.detection": surya_detection},
        ):
            det._load_model()
        self.assertNotEqual(det._engine_name, "Surya")


class ManualMaskCorrectionTests(unittest.TestCase):
    def test_coerce_mask_correction_validates_polygon(self):
        from backend.config import _coerce_mask_correction
        valid = {"polygons": [[10, 20, 30, 40, 50, 60]], "start": 0.0, "end": 5.0}
        result = _coerce_mask_correction(valid)
        self.assertIsNotNone(result)
        self.assertEqual(len(result["polygons"]), 1)
        self.assertEqual(result["start"], 0.0)
        self.assertEqual(result["end"], 5.0)

    def test_coerce_rejects_too_few_points(self):
        from backend.config import _coerce_mask_correction
        short = {"polygons": [[10, 20, 30, 40]], "start": 0.0, "end": 0.0}
        result = _coerce_mask_correction(short)
        self.assertIsNone(result)

    def test_manual_corrections_apply_to_mask(self):
        from backend.config import ProcessingConfig
        from backend.processor import SubtitleRemover

        config = ProcessingConfig(
            manual_mask_corrections=[
                {"polygons": [[5, 5, 25, 5, 25, 25, 5, 25]], "start": 0.0, "end": 0.0}
            ],
        )
        remover = SubtitleRemover.__new__(SubtitleRemover)
        remover.config = config
        mask = np.zeros((32, 32), dtype=np.uint8)
        result = remover._apply_manual_mask_corrections(mask, 0.5)
        self.assertGreater(result.sum(), 0)

    def test_manual_corrections_respect_time_range(self):
        from backend.config import ProcessingConfig
        from backend.processor import SubtitleRemover

        config = ProcessingConfig(
            manual_mask_corrections=[
                {"polygons": [[5, 5, 25, 5, 25, 25, 5, 25]], "start": 2.0, "end": 5.0}
            ],
        )
        remover = SubtitleRemover.__new__(SubtitleRemover)
        remover.config = config
        mask = np.zeros((32, 32), dtype=np.uint8)
        before = remover._apply_manual_mask_corrections(mask.copy(), 1.0)
        during = remover._apply_manual_mask_corrections(mask.copy(), 3.0)
        after = remover._apply_manual_mask_corrections(mask.copy(), 6.0)
        self.assertEqual(before.sum(), 0)
        self.assertGreater(during.sum(), 0)
        self.assertEqual(after.sum(), 0)

    def test_config_round_trip_with_corrections(self):
        from gui.config import ProcessingConfig as GuiConfig
        config = GuiConfig(
            manual_mask_corrections=[
                {"polygons": [[10, 20, 30, 40, 50, 60]], "start": 0.0, "end": 10.0},
            ],
        )
        serialized = config.to_dict()
        self.assertIsNotNone(serialized.get("manual_mask_corrections"))
        restored = GuiConfig.from_dict(serialized)
        self.assertIsNotNone(restored.manual_mask_corrections)
        self.assertEqual(len(restored.manual_mask_corrections), 1)
        self.assertEqual(
            restored.manual_mask_corrections[0]["polygons"],
            [[10, 20, 30, 40, 50, 60]],
        )


    def test_mask_corrections_do_not_corrupt_cached_mask(self):
        from backend.config import ProcessingConfig
        from backend.processor import SubtitleRemover

        config = ProcessingConfig(
            manual_mask_corrections=[
                {"polygons": [[5, 5, 25, 5, 25, 25, 5, 25]], "start": 0.0, "end": 0.0}
            ],
        )
        remover = SubtitleRemover.__new__(SubtitleRemover)
        remover.config = config
        original = np.zeros((32, 32), dtype=np.uint8)
        copy1 = remover._apply_manual_mask_corrections(original.copy(), 0.5)
        self.assertGreater(copy1.sum(), 0)
        self.assertEqual(original.sum(), 0)

    def test_coerce_nan_start_end_falls_back_to_zero(self):
        from backend.config import _coerce_mask_correction
        result = _coerce_mask_correction({
            "polygons": [[10, 20, 30, 40, 50, 60]],
            "start": float("nan"),
            "end": float("inf"),
        })
        self.assertIsNotNone(result)
        self.assertEqual(result["start"], 0.0)
        self.assertEqual(result["end"], 0.0)

    def test_sidecar_config_snapshot_includes_mask_corrections(self):
        from backend.batch_report import _config_snapshot
        from backend.config import ProcessingConfig
        config = ProcessingConfig(
            manual_mask_corrections=[
                {"polygons": [[1, 2, 3, 4, 5, 6]], "start": 0.0, "end": 5.0},
            ],
        )
        snap = _config_snapshot(config)
        self.assertIn("manual_mask_corrections", snap)



if __name__ == "__main__":
    unittest.main()
