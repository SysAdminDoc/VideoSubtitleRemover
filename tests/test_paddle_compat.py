import types
import unittest
from unittest import mock

import numpy as np

from backend.paddle_compat import (
    build_paddleocr,
    extract_paddle_boxes,
    extract_paddle_text_boxes,
    normalize_paddleocr_variant,
)


class _JsonMethodResult:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _JsonPropertyResult:
    def __init__(self, payload):
        self.json = payload


class PaddleCompatTests(unittest.TestCase):
    def test_normalize_variant_defaults_to_mobile(self):
        self.assertEqual(normalize_paddleocr_variant("server"), "server")
        self.assertEqual(normalize_paddleocr_variant("PP-OCRv5_mobile"), "mobile")
        self.assertEqual(normalize_paddleocr_variant("unknown"), "mobile")

    def test_v3_builder_names_selected_models_explicitly(self):
        calls = []

        class PaddleOCR:
            def __init__(self, **kwargs):
                calls.append(kwargs)

        paddle = types.ModuleType("paddleocr")
        paddle.PaddleOCR = PaddleOCR
        paddle.__version__ = "3.6.0"
        with mock.patch.dict("sys.modules", {"paddleocr": paddle}):
            build_paddleocr("en", "cpu", variant="server")

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["ocr_version"], "PP-OCRv5")
        self.assertEqual(
            calls[0]["text_detection_model_name"], "PP-OCRv5_server_det")
        self.assertEqual(
            calls[0]["text_recognition_model_name"], "PP-OCRv5_server_rec")

    def test_v2_fallback_drops_v3_model_keywords(self):
        calls = []

        class PaddleOCR:
            def __init__(self, **kwargs):
                calls.append(kwargs)
                if "device" in kwargs:
                    raise TypeError("2.x constructor")

        paddle = types.ModuleType("paddleocr")
        paddle.PaddleOCR = PaddleOCR
        paddle.__version__ = "2.7.0"
        with mock.patch.dict("sys.modules", {"paddleocr": paddle}):
            build_paddleocr("en", "cpu", variant="server")

        self.assertEqual(len(calls), 2)
        self.assertNotIn("ocr_version", calls[1])
        self.assertNotIn("text_detection_model_name", calls[1])
        self.assertNotIn("text_recognition_model_name", calls[1])

    def test_ppocrv6_predict_payload_extracts_rec_polys(self):
        class Model:
            def predict(self, frame):
                return [
                    _JsonMethodResult({
                        "res": {
                            "rec_texts": ["subtitle", "low"],
                            "rec_scores": np.array([0.91, 0.2]),
                            "rec_polys": np.array([
                                [[10, 20], [40, 20], [40, 32], [10, 32]],
                                [[2, 2], [5, 2], [5, 4], [2, 4]],
                            ], dtype=np.int16),
                        }
                    })
                ]

        boxes = extract_paddle_boxes(Model(), np.zeros((4, 4, 3)), 0.5)

        self.assertEqual(boxes, [(10, 20, 40, 32)])

        text_boxes = extract_paddle_text_boxes(
            Model(), np.zeros((4, 4, 3)), 0.5)
        self.assertEqual(
            text_boxes, [(10, 20, 40, 32, 0.91, "subtitle")])

    def test_ppocrv6_predict_payload_extracts_rec_boxes_fallback(self):
        class Model:
            def predict(self, frame):
                return [
                    _JsonPropertyResult({
                        "res": {
                            "rec_texts": ["subtitle"],
                            "rec_scores": [0.88],
                            "rec_boxes": [[5, 7, 55, 18]],
                        }
                    })
                ]

        boxes = extract_paddle_boxes(Model(), np.zeros((4, 4, 3)), 0.5)

        self.assertEqual(boxes, [(5, 7, 55, 18)])

        text_boxes = extract_paddle_text_boxes(
            Model(), np.zeros((4, 4, 3)), 0.5)
        self.assertEqual(
            text_boxes, [(5, 7, 55, 18, 0.88, "subtitle")])


    def test_v3_dt_polys_key(self):
        class Model:
            def predict(self, frame):
                return [{
                    "dt_polys": np.array([
                        [[8, 12], [80, 12], [80, 26], [8, 26]],
                    ], dtype=np.float32),
                    "rec_scores": [0.95],
                }]

        boxes = extract_paddle_boxes(Model(), np.zeros((4, 4, 3)), 0.5)
        self.assertEqual(boxes, [(8, 12, 80, 26)])

    def test_low_confidence_filtered_out(self):
        class Model:
            def predict(self, frame):
                return [
                    _JsonMethodResult({
                        "res": {
                            "rec_polys": np.array([
                                [[10, 20], [40, 20], [40, 32], [10, 32]],
                            ]),
                            "rec_scores": [0.15],
                        }
                    })
                ]

        boxes = extract_paddle_boxes(Model(), np.zeros((4, 4, 3)), 0.5)
        self.assertEqual(boxes, [])


if __name__ == "__main__":
    unittest.main()
