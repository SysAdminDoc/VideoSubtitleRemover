"""RM-359: tiled repair must not size its buffers to the whole picture.

The three tiled LaMa paths each allocated two full-frame float32 accumulators
and blended across the entire frame to repair a caption strip. The fix bounds
both to the tiles that actually run.

Correctness is checked against an oracle: ``_reference_blend`` below is the
full-frame algorithm the shipped code used to run, written out here so the new
implementation is compared with the behaviour it replaced rather than with
itself. The memory claim is measured with tracemalloc, not asserted.
"""

from __future__ import annotations

import tracemalloc
import unittest

import numpy as np

from backend.inpainters.lama import _blend_tiles, _tile_rects, _tile_window


def _reference_blend(frame, mask, tile_size, overlap, inpaint_tile):
    """The pre-RM-359 full-frame accumulation, kept as the oracle."""
    h, w = frame.shape[:2]
    ys = mask.any(axis=1)
    xs = mask.any(axis=0)
    if not ys.any():
        return frame.copy()
    y_indices = np.where(ys)[0]
    x_indices = np.where(xs)[0]
    roi_y1 = max(0, int(y_indices[0]) - overlap)
    roi_y2 = min(h, int(y_indices[-1]) + 1 + overlap)
    roi_x1 = max(0, int(x_indices[0]) - overlap)
    roi_x2 = min(w, int(x_indices[-1]) + 1 + overlap)
    step = max(1, tile_size - overlap)
    result = frame.copy()
    weight_acc = np.zeros((h, w), dtype=np.float32)
    color_acc = np.zeros_like(frame, dtype=np.float32)
    tile_count = 0
    for ty in range(roi_y1, roi_y2, step):
        for tx in range(roi_x1, roi_x2, step):
            ty2 = min(ty + tile_size, h)
            tx2 = min(tx + tile_size, w)
            ty1 = max(0, ty2 - tile_size)
            tx1 = max(0, tx2 - tile_size)
            tile_mask = mask[ty1:ty2, tx1:tx2]
            if tile_mask.max() == 0:
                continue
            tile_out = inpaint_tile(frame[ty1:ty2, tx1:tx2], tile_mask)
            th, tw = tile_out.shape[:2]
            win = _tile_window(th, tw, overlap)
            color_acc[ty1:ty2, tx1:tx2] += (
                tile_out.astype(np.float32) * win[..., None])
            weight_acc[ty1:ty2, tx1:tx2] += win
            tile_count += 1
    if tile_count > 0:
        blend_mask = weight_acc > 0
        for c in range(3):
            result[:, :, c] = np.where(
                blend_mask,
                (color_acc[:, :, c] / np.maximum(weight_acc, 1e-6)).clip(0, 255),
                frame[:, :, c],
            )
        result = result.astype(np.uint8)
    return result


def _fake_inpaint(tile_frame, tile_mask):
    """Deterministic, content-dependent, and different from the input."""
    out = tile_frame.astype(np.int32)
    out = (out * 3 + 17) % 256
    out[tile_mask > 0] = (out[tile_mask > 0] + 40) % 256
    return out.astype(np.uint8)


def _frame(h, w, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _strip_mask(h, w, top, bottom, left=None, right=None):
    mask = np.zeros((h, w), dtype=np.uint8)
    left = 0 if left is None else left
    right = w if right is None else right
    mask[top:bottom, left:right] = 255
    return mask


class TileBlendEquivalenceTests(unittest.TestCase):
    """The bounded version must reproduce the full-frame version exactly."""

    CASES = {
        "caption strip near the bottom": (480, 640, 400, 450, 80, 560),
        "mask touching the top left corner": (480, 640, 0, 40, 0, 40),
        "mask touching the bottom right corner": (480, 640, 440, 480, 600, 640),
        "mask spanning the full width": (480, 640, 200, 260, 0, 640),
        "single pixel mask": (480, 640, 240, 241, 320, 321),
        "mask larger than one tile": (480, 640, 60, 420, 40, 600),
    }

    def test_output_matches_the_full_frame_algorithm(self):
        for name, (h, w, top, bottom, left, right) in self.CASES.items():
            for tile_size, overlap in ((128, 32), (256, 64), (64, 0)):
                with self.subTest(case=name, tile=tile_size, overlap=overlap):
                    frame = _frame(h, w, seed=len(name))
                    mask = _strip_mask(h, w, top, bottom, left, right)
                    expected = _reference_blend(
                        frame, mask, tile_size, overlap, _fake_inpaint)
                    actual = _blend_tiles(
                        frame, mask, tile_size, overlap, _fake_inpaint)
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertEqual(actual.dtype, np.uint8)
                    np.testing.assert_array_equal(actual, expected)

    def test_an_empty_mask_returns_the_source_untouched(self):
        frame = _frame(120, 160)
        mask = np.zeros((120, 160), dtype=np.uint8)
        out = _blend_tiles(frame, mask, 64, 16, _fake_inpaint)
        np.testing.assert_array_equal(out, frame)

    def test_pixels_no_tile_covered_keep_their_source_value(self):
        frame = _frame(480, 640, seed=7)
        mask = _strip_mask(480, 640, 400, 450, 80, 560)
        out = _blend_tiles(frame, mask, 128, 32, _fake_inpaint)
        rects = _tile_rects(frame.shape, mask, 128, 32)
        covered = np.zeros((480, 640), dtype=bool)
        for ty1, ty2, tx1, tx2 in rects:
            covered[ty1:ty2, tx1:tx2] = True
        np.testing.assert_array_equal(out[~covered], frame[~covered])
        self.assertTrue(
            (out[covered] != frame[covered]).any(),
            "the fake inpainter changes pixels, so the covered region must "
            "differ or this test is asserting nothing",
        )


class TileBlendAllocationTests(unittest.TestCase):
    def test_accumulators_are_sized_to_the_tiles_not_the_frame(self):
        rects = _tile_rects((2160, 3840, 3),
                            _strip_mask(2160, 3840, 1900, 1990, 600, 3200),
                            512, 64)
        self.assertTrue(rects)
        acc_h = max(r[1] for r in rects) - min(r[0] for r in rects)
        acc_w = max(r[3] for r in rects) - min(r[2] for r in rects)
        self.assertLess(
            acc_h * acc_w, 2160 * 3840 * 0.25,
            "a caption strip must not need a quarter of the frame in buffers",
        )

    def test_peak_memory_scales_with_the_masked_region(self):
        # Allocated before the measurement so only the call under test counts.
        frame = _frame(2160, 3840, seed=3)
        mask = _strip_mask(2160, 3840, 1900, 1990, 600, 3200)

        tracemalloc.start()
        try:
            _blend_tiles(frame, mask, 512, 64, _fake_inpaint)
            _, bounded_peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        tracemalloc.start()
        try:
            _reference_blend(frame, mask, 512, 64, _fake_inpaint)
            _, full_frame_peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        # The frame copy alone is 3840*2160*3 = 24.9 MB and both versions pay
        # it, so the comparison is against the old version rather than zero.
        self.assertLess(
            bounded_peak, full_frame_peak * 0.6,
            f"bounded peak {bounded_peak / 1e6:.1f} MB against full-frame "
            f"{full_frame_peak / 1e6:.1f} MB: the buffers are still tracking "
            f"the picture rather than the mask",
        )


class TileRectTests(unittest.TestCase):
    def test_every_rect_stays_inside_the_frame(self):
        mask = _strip_mask(300, 400, 0, 20, 0, 20)
        for ty1, ty2, tx1, tx2 in _tile_rects((300, 400, 3), mask, 128, 32):
            self.assertGreaterEqual(ty1, 0)
            self.assertGreaterEqual(tx1, 0)
            self.assertLessEqual(ty2, 300)
            self.assertLessEqual(tx2, 400)

    def test_only_tiles_that_carry_mask_are_returned(self):
        mask = _strip_mask(300, 400, 100, 130, 100, 130)
        for ty1, ty2, tx1, tx2 in _tile_rects((300, 400, 3), mask, 64, 16):
            self.assertGreater(
                mask[ty1:ty2, tx1:tx2].max(), 0,
                "an empty tile costs an inference call for nothing",
            )

    def test_an_empty_mask_yields_no_tiles(self):
        mask = np.zeros((300, 400), dtype=np.uint8)
        self.assertEqual(_tile_rects((300, 400, 3), mask, 64, 16), [])


if __name__ == "__main__":
    unittest.main()
