import builtins
import importlib
import os
import sys
import types
import unittest
from contextlib import contextmanager
from unittest import mock


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

        with _fresh_detection_module() as detection:
            with mock.patch.dict(
                sys.modules,
                {"backend.ocr_vlm": self._vlm_disabled_module()},
            ):
                with mock.patch.dict(os.environ, {"VSR_ALLOW_GPL": ""}):
                    with mock.patch("builtins.__import__", side_effect=fake_import):
                        detector = detection.SubtitleDetector(device="cpu", lang="en")

        self.assertEqual(detector._engine_name, "OpenCV fallback")
        self.assertEqual(
            attempts,
            [
                "rapidocr",
                "rapidocr_onnxruntime",
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


class VerticalTextRotationTests(unittest.TestCase):
    """Verify that 90-CCW rotation coordinate mapping is correct."""

    def test_box_round_trips_through_rotation(self):
        import numpy as np
        with _fresh_detection_module() as detection:
            h, w = 1080, 1920
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            original_box = (1800, 400, 1900, 500)
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


if __name__ == "__main__":
    unittest.main()
