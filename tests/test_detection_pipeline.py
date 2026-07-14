import builtins
import importlib
import os
import sys
import types
import unittest
from contextlib import contextmanager
from unittest import mock

import cv2
import numpy as np


@contextmanager
def _fresh_detection_module():
    old_detection = sys.modules.pop("backend.detection", None)
    old_cv2 = sys.modules.get("cv2")
    had_cv2 = "cv2" in sys.modules
    fake_cv2 = types.SimpleNamespace(
        ROTATE_90_COUNTERCLOCKWISE=0,
        rotate=lambda frame, code: frame,
    )
    sys.modules["cv2"] = fake_cv2
    try:
        yield importlib.import_module("backend.detection")
    finally:
        sys.modules.pop("backend.detection", None)
        if old_detection is not None:
            sys.modules["backend.detection"] = old_detection
        if had_cv2:
            sys.modules["cv2"] = old_cv2
        else:
            sys.modules.pop("cv2", None)


class DetectionCascadeTests(unittest.TestCase):
    def _vlm_disabled_module(self):
        module = types.ModuleType("backend.ocr_vlm")
        module.maybe_build_vlm_detector = lambda device, lang: None
        return module

    def test_cascade_falls_back_through_optional_engines_in_order(self):
        blocked = {
            "rapidocr",
            "rapidocr_onnxruntime",
            "backend.paddle_compat",
            "surya.detection",
            "easyocr",
        }
        attempts = []
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in blocked:
                attempts.append(name)
                raise ImportError(name)
            return real_import(name, globals, locals, fromlist, level)

        def fake_can_import(name, **kwargs):
            return name in {"rapidocr", "easyocr"}

        with _fresh_detection_module() as detection:
            with mock.patch.dict(
                sys.modules,
                {"backend.ocr_vlm": self._vlm_disabled_module()},
            ):
                with mock.patch.dict(os.environ, {"VSR_ALLOW_GPL": ""}):
                    with mock.patch.object(
                        detection,
                        "_module_can_import",
                        side_effect=fake_can_import,
                    ):
                        with mock.patch("builtins.__import__", side_effect=fake_import):
                            detector = detection.SubtitleDetector(device="cpu", lang="en")

        self.assertEqual(detector._engine_name, "OpenCV fallback")
        self.assertEqual(
            attempts,
            [
                "rapidocr",
                "backend.paddle_compat",
                "surya.detection",
                "easyocr",
            ],
        )

    def test_rapidocr_wins_before_later_engines(self):
        class FakeRapidOCR:
            pass

        rapid = types.ModuleType("rapidocr")
        rapid.RapidOCR = FakeRapidOCR

        with _fresh_detection_module() as detection:
            with mock.patch.dict(
                sys.modules,
                {
                    "backend.ocr_vlm": self._vlm_disabled_module(),
                    "rapidocr": rapid,
                },
            ):
                detector = detection.SubtitleDetector(device="cpu", lang="en")

        self.assertEqual(detector._engine_name, "RapidOCR")
        self.assertIsInstance(detector._rapid_model, FakeRapidOCR)
        self.assertIsNone(detector._paddle_model)

    def test_rapidocr_openvino_preferred_for_cpu_when_available(self):
        calls = []

        class FakeRapidOCR:
            def __init__(self, **kwargs):
                calls.append(kwargs)

        rapid = types.ModuleType("rapidocr")
        rapid.RapidOCR = FakeRapidOCR
        rapid.EngineType = types.SimpleNamespace(OPENVINO="openvino")

        with _fresh_detection_module() as detection:
            with mock.patch.dict(
                sys.modules,
                {
                    "backend.ocr_vlm": self._vlm_disabled_module(),
                    "rapidocr": rapid,
                    "openvino": types.ModuleType("openvino"),
                },
            ):
                with mock.patch.dict(
                    os.environ,
                    {"VSR_RAPIDOCR_ENGINE": "auto"},
                    clear=False,
                ):
                    detector = detection.SubtitleDetector(device="cpu", lang="en")

        self.assertEqual(detector._engine_name, "RapidOCR (OpenVINO)")
        self.assertEqual(calls[0]["params"]["Det.engine_type"], "openvino")
        self.assertEqual(calls[0]["params"]["Rec.engine_type"], "openvino")

    def test_rapidocr_openvino_falls_back_to_onnxruntime(self):
        calls = []

        class FakeRapidOCR:
            def __init__(self, **kwargs):
                calls.append(kwargs)
                if kwargs.get("params", {}).get("Det.engine_type") == "openvino":
                    raise ImportError("openvino unavailable")

        rapid = types.ModuleType("rapidocr")
        rapid.RapidOCR = FakeRapidOCR
        rapid.EngineType = types.SimpleNamespace(OPENVINO="openvino")

        with _fresh_detection_module() as detection:
            with mock.patch.dict(
                sys.modules,
                {
                    "backend.ocr_vlm": self._vlm_disabled_module(),
                    "rapidocr": rapid,
                    "openvino": types.ModuleType("openvino"),
                },
            ):
                with mock.patch.dict(
                    os.environ,
                    {"VSR_RAPIDOCR_ENGINE": "auto"},
                    clear=False,
                ):
                    detector = detection.SubtitleDetector(device="cpu", lang="en")

        self.assertEqual(detector._engine_name, "RapidOCR")
        self.assertIn("params", calls[0])
        self.assertEqual(calls[1], {})

    def test_paddle_used_when_rapidocr_is_unavailable(self):
        blocked = {"rapidocr", "rapidocr_onnxruntime"}
        real_import = builtins.__import__
        paddle_model = object()
        paddle = types.ModuleType("backend.paddle_compat")
        paddle.build_paddleocr = lambda lang, device: paddle_model

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in blocked:
                raise ImportError(name)
            return real_import(name, globals, locals, fromlist, level)

        with _fresh_detection_module() as detection:
            with mock.patch.dict(
                sys.modules,
                {
                    "backend.ocr_vlm": self._vlm_disabled_module(),
                    "backend.paddle_compat": paddle,
                },
            ):
                with mock.patch("builtins.__import__", side_effect=fake_import):
                    detector = detection.SubtitleDetector(device="cpu", lang="en")

        self.assertEqual(detector._engine_name, "PaddleOCR")
        self.assertIs(detector._paddle_model, paddle_model)

    def test_paddleocr_vl_flag_falls_back_to_paddle_when_server_absent(self):
        blocked = {"rapidocr", "rapidocr_onnxruntime"}
        real_import = builtins.__import__
        paddle_model = object()
        paddle = types.ModuleType("backend.paddle_compat")
        paddle.build_paddleocr = lambda lang, device: paddle_model

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in blocked:
                raise ImportError(name)
            return real_import(name, globals, locals, fromlist, level)

        from backend import ocr_vlm

        with _fresh_detection_module() as detection:
            with mock.patch.dict(
                sys.modules,
                {"backend.paddle_compat": paddle},
            ):
                with mock.patch.dict(os.environ, {
                    "VSR_PADDLEOCR_VL": "1",
                    "VSR_ALLOW_GPL": "",
                }):
                    with mock.patch.object(
                        ocr_vlm,
                        "_llama_cpp_server_reachable",
                        return_value=False,
                    ):
                        with mock.patch("builtins.__import__",
                                        side_effect=fake_import):
                            detector = detection.SubtitleDetector(
                                device="cpu", lang="en")

        self.assertEqual(detector._engine_name, "PaddleOCR")
        self.assertIs(detector._paddle_model, paddle_model)

    def test_opencv_fallback_detect_returns_list(self):
        from backend.detection import SubtitleDetector
        detector = SubtitleDetector.__new__(SubtitleDetector)
        detector.device = "cpu"
        detector.lang = "en"
        detector.vertical = False
        detector._engine_name = "OpenCV fallback"
        detector._rapid_model = None
        detector._paddle_model = None
        detector._surya_det = None
        detector._surya_processor = None
        detector._easyocr_reader = None
        detector._vlm_detector = None
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "SUBTITLE TEXT", (100, 440),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        boxes = detector.detect(frame, threshold=0.3)
        self.assertIsInstance(boxes, list)


class VerticalTextRotationTests(unittest.TestCase):
    """Verify that 90-CCW rotation coordinate mapping is correct."""

    def test_box_round_trips_through_rotation(self):
        import numpy as np
        with _fresh_detection_module() as detection:
            h, w = 1080, 1920
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            detector = detection.SubtitleDetector.__new__(
                detection.SubtitleDetector)
            detector.vertical = True
            detector._rapid_model = None
            detector._paddle_model = None
            detector._surya_det = None
            detector._easyocr_reader = None
            detector._vlm_detector = None
            detector._engine_name = "test"

            rotated_box = (400, 19, 500, 119)

            def fake_detect(f, t):
                return [rotated_box]

            detector._detect_axis_aligned = fake_detect
            result = detector.detect(frame, 0.5)

            self.assertEqual(len(result), 1)
            rx1, ry1, rx2, ry2 = result[0]
            self.assertTrue(1799 <= rx1 <= 1801,
                            f"ox1={rx1} expected ~1800")
            self.assertTrue(399 <= ry1 <= 401,
                            f"oy1={ry1} expected ~400")
            self.assertTrue(1900 <= rx2 <= 1921,
                            f"ox2={rx2} expected ~1901")
            self.assertTrue(499 <= ry2 <= 501,
                            f"oy2={ry2} expected ~500")


class RapidOCROutputParsingTests(unittest.TestCase):
    """Verify that the output parser handles v1.x, v2.x, and v3.x shapes."""

    def _get_detector_cls(self):
        with _fresh_detection_module() as detection:
            return detection.SubtitleDetector

    def test_v1_tuple_output(self):
        cls = self._get_detector_cls()
        output = ([
            [[[10, 20], [100, 20], [100, 40], [10, 40]], "hello", 0.95],
        ], 0.1)
        boxes = cls._rapid_output_to_boxes(output, 0.5)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0], (10, 20, 100, 40))

    def test_v2_structured_object(self):
        cls = self._get_detector_cls()
        import numpy as np
        output = types.SimpleNamespace(
            boxes=np.array([
                [[10, 20], [100, 20], [100, 40], [10, 40]],
            ]),
            scores=[0.92],
        )
        boxes = cls._rapid_output_to_boxes(output, 0.5)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0], (10, 20, 100, 40))

    def test_v3_dict_output(self):
        cls = self._get_detector_cls()
        import numpy as np
        output = {
            "dt_polys": [
                np.array([[15, 25], [110, 25], [110, 45], [15, 45]]),
            ],
            "rec_scores": [0.88],
        }
        boxes = cls._rapid_output_to_boxes(output, 0.5)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0], (15, 25, 110, 45))

    def test_low_confidence_filtered(self):
        cls = self._get_detector_cls()
        output = ([
            [[[10, 20], [100, 20], [100, 40], [10, 40]], "hello", 0.2],
        ], 0.1)
        boxes = cls._rapid_output_to_boxes(output, 0.5)
        self.assertEqual(len(boxes), 0)

    def test_none_output(self):
        cls = self._get_detector_cls()
        boxes = cls._rapid_output_to_boxes(None, 0.5)
        self.assertEqual(boxes, [])

    def test_v3_list_of_dicts(self):
        cls = self._get_detector_cls()
        import numpy as np
        output = [
            {"box": np.array([[5, 10], [50, 10], [50, 30], [5, 30]]),
             "score": 0.91, "text": "test"},
        ]
        boxes = cls._rapid_output_to_boxes(output, 0.5)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0], (5, 10, 50, 30))

    @unittest.skipUnless(
        importlib.util.find_spec("rapidocr"),
        "rapidocr not installed",
    )
    def test_installed_rapidocr_ppocrv6_detects_source_image(self):
        from backend.detection import SubtitleDetector

        frame = np.full((96, 320, 3), 255, dtype=np.uint8)
        cv2.putText(
            frame,
            "TEST",
            (20, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        detector = SubtitleDetector(device="cpu", lang="en")
        if detector._engine_name != "RapidOCR":
            self.skipTest(f"RapidOCR unavailable, got {detector._engine_name}")

        boxes = detector.detect(frame, threshold=0.1)

        self.assertTrue(boxes)
        x1, y1, x2, y2 = boxes[0]
        self.assertLess(x1, x2)
        self.assertLess(y1, y2)


class CliCommandBuilderTests(unittest.TestCase):
    """Verify that _build_cli_command produces correct CLI strings."""

    def test_default_config_produces_minimal_command(self):
        from gui.config import ProcessingConfig, QueueItem
        from gui.widgets import _build_cli_command
        item = QueueItem(
            id="test",
            file_path="input.mp4",
            output_path="output.mp4",
            config=ProcessingConfig(),
        )
        cmd = _build_cli_command(item)
        self.assertIn('-i "input.mp4"', cmd)
        self.assertIn('-o "output.mp4"', cmd)
        self.assertNotIn("--crf", cmd)
        self.assertNotIn("--codec", cmd)

    def test_custom_config_includes_flags(self):
        from gui.config import ProcessingConfig, QueueItem
        from gui.widgets import _build_cli_command
        cfg = ProcessingConfig()
        cfg.detection_lang = "ja"
        cfg.output_quality = 18
        cfg.output_codec = "h265"
        cfg.mask_dilate_px = 12
        cfg.lama_super_fast = True
        item = QueueItem(
            id="test",
            file_path="video.mkv",
            output_path="clean.mkv",
            config=cfg,
        )
        cmd = _build_cli_command(item)
        self.assertIn("-l ja", cmd)
        self.assertIn("--crf 18", cmd)
        self.assertIn("--codec h265", cmd)
        self.assertIn("--mask-dilate 12", cmd)
        self.assertIn("--fast", cmd)

    def test_expanded_fields_round_trip(self):
        from gui.config import ProcessingConfig, QueueItem
        from gui.widgets import _build_cli_command
        cfg = ProcessingConfig()
        cfg.mask_feather_px = 6
        cfg.edge_ring_px = 0
        cfg.temporal_smooth_radius = 2
        cfg.detection_vertical = True
        cfg.time_start = 10.0
        cfg.time_end = 120.5
        cfg.loudnorm_target = -16.0
        cfg.multi_audio_passthrough = False
        cfg.tbe_flow_warp = True
        cfg.colour_tune_enable = True
        cfg.colour_tune_tolerance = 30
        cfg.kalman_tracking = False
        cfg.phash_skip_enable = False
        cfg.whisper_fallback = True
        cfg.whisper_model_size = "base"
        cfg.remove_subtitles = False
        cfg.karaoke_grouping = True
        cfg.export_srt = True
        cfg.export_mask_video = True
        cfg.batch_max_retries = 2
        cfg.batch_retry_backoff_seconds = 0.25
        cfg.quality_report_sheet = True
        cfg.nle_sidecar = "edl"
        item = QueueItem(
            id="test",
            file_path="input.mp4",
            output_path="output.mp4",
            config=cfg,
        )
        cmd = _build_cli_command(item)
        self.assertIn("--mask-feather 6", cmd)
        self.assertIn("--edge-ring 0", cmd)
        self.assertIn("--temporal-smooth 2", cmd)
        self.assertIn("--vertical", cmd)
        self.assertIn("--start 10.0", cmd)
        self.assertIn("--end 120.5", cmd)
        self.assertIn("--loudnorm -16.0", cmd)
        self.assertIn("--single-audio", cmd)
        self.assertIn("--flow-warp", cmd)
        self.assertIn("--colour-tune", cmd)
        self.assertIn("--colour-tolerance 30", cmd)
        self.assertIn("--no-kalman", cmd)
        self.assertIn("--no-phash", cmd)
        self.assertIn("--whisper-fallback", cmd)
        self.assertIn("--whisper-model base", cmd)
        self.assertIn("--keep-subtitles", cmd)
        self.assertIn("--karaoke-grouping", cmd)
        self.assertIn("--export-srt", cmd)
        self.assertIn("--export-mask", cmd)
        self.assertIn("--max-retries 2", cmd)
        self.assertIn("--retry-backoff 0.25", cmd)
        self.assertIn("--quality-sheet", cmd)
        self.assertIn("--nle-sidecar edl", cmd)

    def test_default_config_omits_expanded_fields(self):
        from gui.config import ProcessingConfig, QueueItem
        from gui.widgets import _build_cli_command
        item = QueueItem(
            id="test",
            file_path="input.mp4",
            output_path="output.mp4",
            config=ProcessingConfig(),
        )
        cmd = _build_cli_command(item)
        for flag in ("--mask-feather", "--edge-ring", "--temporal-smooth",
                     "--vertical", "--start", "--end", "--loudnorm",
                     "--single-audio", "--flow-warp", "--colour-tune",
                     "--no-kalman", "--no-phash", "--whisper-fallback",
                     "--keep-subtitles", "--keep-chyrons", "--karaoke",
                     "--export-srt", "--export-mask", "--nle-sidecar",
                     "--quality-sheet", "--output-frames"):
            self.assertNotIn(flag, cmd, f"Default config should not emit {flag}")


class ScriptClassificationTests(unittest.TestCase):
    """Verify _classify_script maps characters to correct script families."""

    def test_latin(self):
        with _fresh_detection_module() as detection:
            self.assertEqual(detection._classify_script("Hello world"), "latin")

    def test_cjk(self):
        with _fresh_detection_module() as detection:
            self.assertEqual(
                detection._classify_script(chr(0x4F60) + chr(0x597D) + chr(0x4E16) + chr(0x754C)), "cjk")

    def test_hangul(self):
        with _fresh_detection_module() as detection:
            self.assertEqual(
                detection._classify_script(chr(0xAC00) + chr(0xB155) + chr(0xD558) + chr(0xC138) + chr(0xC694)), "hangul")

    def test_cyrillic(self):
        with _fresh_detection_module() as detection:
            self.assertEqual(
                detection._classify_script(chr(0x41F) + chr(0x440) + chr(0x438) + chr(0x432) + chr(0x435) + chr(0x442)), "cyrillic")

    def test_empty(self):
        with _fresh_detection_module() as detection:
            self.assertEqual(detection._classify_script(""), "unknown")

    def test_mixed_cjk_dominant(self):
        with _fresh_detection_module() as detection:
            self.assertEqual(
                detection._classify_script(chr(0x4F60) + chr(0x597D) + "Hi"), "cjk")


class OcrBenchmarkHarnessTests(unittest.TestCase):
    """P2: OCR detection benchmark harness (fixtures + scoring)."""

    def test_perfect_detector_meets_recall_floor(self):
        from backend import ocr_benchmark

        # Build a true oracle: map each fixture image to its ground-truth box.
        gt_by_shape = {}
        for image, box in ocr_benchmark.iter_fixtures():
            gt_by_shape[image.tobytes()] = box

        class OracleDetector:
            _engine_name = "oracle"

            def detect(self, image, threshold):
                return [gt_by_shape[image.tobytes()]]

        result = ocr_benchmark.run_ocr_detection_benchmark(OracleDetector())
        self.assertEqual(result["schema"], ocr_benchmark.OCR_BENCHMARK_SCHEMA)
        self.assertEqual(result["engine"], "oracle")
        self.assertEqual(result["recall"], 1.0)
        self.assertTrue(result["meets_floors"])
        self.assertIn("memory", result)
        self.assertIn("peakRssBytes", result["memory"])

    def test_blind_detector_fails_recall_floor(self):
        from backend import ocr_benchmark

        class BlindDetector:
            _engine_name = "blind"

            def detect(self, image, threshold):
                return []

        result = ocr_benchmark.run_ocr_detection_benchmark(BlindDetector())
        self.assertEqual(result["recall"], 0.0)
        self.assertFalse(result["meets_floors"])

    def test_detector_exception_is_recorded_not_raised(self):
        from backend import ocr_benchmark

        class BrokenDetector:
            _engine_name = "broken"

            def detect(self, image, threshold):
                raise RuntimeError("model not loaded")

        result = ocr_benchmark.run_ocr_detection_benchmark(BrokenDetector())
        self.assertFalse(result["meets_floors"])
        self.assertTrue(all("error" in r for r in result["results"]))

    def test_recognition_text_is_scored_when_provider_exposes_it(self):
        from backend import ocr_benchmark

        expected = {}
        for text, (image, box) in zip(
            ocr_benchmark._FIXTURE_TEXTS,
            ocr_benchmark.iter_fixtures(),
        ):
            expected[image.tobytes()] = (box, text)

        class TextDetector:
            _engine_name = "text-provider"

            def benchmark_detect(self, image, threshold):
                box, text = expected[image.tobytes()]
                return [box], [text]

        result = ocr_benchmark.run_ocr_detection_benchmark(TextDetector())
        self.assertTrue(result["recognition_ran"])
        self.assertEqual(result["recognition_mean"], 1.0)
        self.assertTrue(result["recognition_meets_floor"])
        self.assertTrue(result["meets_floors"])


if __name__ == "__main__":
    unittest.main()
