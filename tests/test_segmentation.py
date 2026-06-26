from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class Sam2RefinementTests(unittest.TestCase):
    def test_sam2_replaces_coarse_box_with_prompted_mask(self):
        from backend import segmentation as seg

        calls = []

        class FakePredictor:
            def set_image(self, rgb):
                calls.append(("set_image", rgb.shape))

            def predict(self, **kwargs):
                calls.append(("predict", kwargs))
                mask = np.zeros((32, 32), dtype=np.uint8)
                mask[13:17, 13:17] = 1
                return np.asarray([mask]), np.asarray([0.99]), None

        saved = dict(seg._SAM2_STATE)
        try:
            seg._SAM2_STATE.update({"probed": True, "predictor": FakePredictor()})
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            base = np.zeros((32, 32), dtype=np.uint8)
            base[10:20, 10:20] = 255
            base[1:3, 1:3] = 255

            out = seg.refine_mask_with_sam2(frame, [(10, 10, 20, 20)], base)
        finally:
            seg._SAM2_STATE.clear()
            seg._SAM2_STATE.update(saved)

        self.assertEqual(int(out[1:3, 1:3].min()), 255)
        self.assertEqual(int(out[10:13, 10:20].max()), 0)
        self.assertEqual(int(out[13:17, 13:17].min()), 255)
        predict_kwargs = calls[1][1]
        self.assertIn("point_coords", predict_kwargs)
        self.assertIn("point_labels", predict_kwargs)


if __name__ == "__main__":
    unittest.main()
