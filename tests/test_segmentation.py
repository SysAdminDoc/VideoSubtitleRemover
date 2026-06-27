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


class MatAnyoneRefinementTests(unittest.TestCase):
    def test_matte_frame_normalizes_float_alpha(self):
        from backend import segmentation as seg

        class FakeModel:
            def matte(self, frame, hint_mask):
                alpha = np.zeros((16, 16), dtype=np.float32)
                alpha[4:8, 5:9] = 0.75
                return alpha

        saved = dict(seg._MATANYONE_STATE)
        try:
            seg._MATANYONE_STATE.update({"probed": True, "model": FakeModel()})
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            hint = np.zeros((32, 32), dtype=np.uint8)
            hint[8:16, 10:18] = 255

            out = seg.matte_frame(frame, hint)
        finally:
            seg._MATANYONE_STATE.clear()
            seg._MATANYONE_STATE.update(saved)

        self.assertIsNotNone(out)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, (32, 32))
        self.assertGreaterEqual(int(out.max()), 190)

    def test_refine_masks_preserves_empty_hints(self):
        from backend import segmentation as seg

        class FakeModel:
            def matte_frames(self, frames, masks):
                return [np.full(frame.shape[:2], 255, dtype=np.uint8) for frame in frames]

        saved = dict(seg._MATANYONE_STATE)
        try:
            seg._MATANYONE_STATE.update({"probed": True, "model": FakeModel()})
            frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(2)]
            empty = np.zeros((16, 16), dtype=np.uint8)
            hint = np.zeros((16, 16), dtype=np.uint8)
            hint[4:8, 4:8] = 255

            out = seg.refine_masks_with_matanyone(frames, [empty, hint])
        finally:
            seg._MATANYONE_STATE.clear()
            seg._MATANYONE_STATE.update(saved)

        self.assertEqual(int(out[0].max()), 0)
        self.assertEqual(int(out[1].min()), 255)

    def test_processor_batch_refinement_uses_matanyone_flag(self):
        from backend import processor
        from backend import segmentation as seg

        class FakeModel:
            def matte_frames(self, frames, masks):
                alpha = np.zeros(frames[0].shape[:2], dtype=np.uint8)
                alpha[6:10, 6:10] = 255
                return [alpha]

        remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        remover.config = processor.ProcessingConfig(
            matanyone_refine=True,
            device="cpu",
        )
        saved = dict(seg._MATANYONE_STATE)
        try:
            seg._MATANYONE_STATE.update({"probed": True, "model": FakeModel()})
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            mask = np.zeros((16, 16), dtype=np.uint8)
            mask[2:14, 2:14] = 255

            [out] = remover._refine_masks_with_matanyone([frame], [mask])
        finally:
            seg._MATANYONE_STATE.clear()
            seg._MATANYONE_STATE.update(saved)

        self.assertEqual(int(out[2:6, 2:14].max()), 0)
        self.assertEqual(int(out[6:10, 6:10].min()), 255)


if __name__ == "__main__":
    unittest.main()
