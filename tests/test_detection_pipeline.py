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


if __name__ == "__main__":
    unittest.main()
