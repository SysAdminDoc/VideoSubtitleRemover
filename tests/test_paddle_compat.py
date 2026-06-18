import unittest

import numpy as np

from backend.paddle_compat import extract_paddle_boxes


class _JsonMethodResult:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _JsonPropertyResult:
    def __init__(self, payload):
        self.json = payload


class PaddleCompatTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
