"""Tests for the shared post-inpaint finishing step.

``apply_finishing`` is the single edge-ring + feather routine used by the
ONNX, diffusion, and built-in inpainter families, replacing three separate
re-implementations of the same loop.
"""

from __future__ import annotations

import numpy as np

from backend.inpainters import apply_finishing


class _Config:
    def __init__(self, feather=4, ring=2):
        self.mask_feather_px = feather
        self.edge_ring_px = ring


def _frame(value):
    return np.full((32, 32, 3), value, dtype=np.uint8)


def _center_mask():
    m = np.zeros((32, 32), dtype=np.uint8)
    m[10:22, 10:22] = 255
    return m


def test_none_config_passes_through():
    original = [_frame(50)]
    filled = [_frame(200)]
    out = apply_finishing(original, filled, [_center_mask()], None)
    assert np.array_equal(out[0], filled[0])


def test_feather_blends_masked_region():
    # edge_ring disabled so the flat fill is not colour-matched to the
    # (flat) background; this isolates the feather blend.
    original = [_frame(50)]
    filled = [_frame(200)]
    out = apply_finishing(original, filled, [_center_mask()], _Config(),
                          edge_ring=False)
    # Mask core takes the filled value; far outside stays original.
    assert out[0][16, 16, 0] == 200
    assert out[0][0, 0, 0] == 50


def test_edge_ring_toggle_is_respected():
    # A gradient background so edge-ring color match would shift the fill.
    grad = np.tile(np.linspace(0, 255, 32, dtype=np.uint8), (32, 1))
    original = np.dstack([grad] * 3)
    filled = np.full((32, 32, 3), 128, dtype=np.uint8)
    mask = _center_mask()

    with_ring = apply_finishing([original.copy()], [filled.copy()], [mask],
                                _Config(feather=0, ring=3), edge_ring=True)
    without_ring = apply_finishing([original.copy()], [filled.copy()], [mask],
                                   _Config(feather=0, ring=3), edge_ring=False)
    # Disabling edge-ring must leave the fill at its flat value in the core.
    assert without_ring[0][16, 16, 0] == 128
    # Enabling it shifts the core toward the surrounding gradient.
    assert not np.array_equal(with_ring[0], without_ring[0])


def test_explicit_px_overrides_config():
    original = [_frame(50)]
    filled = [_frame(200)]
    out = apply_finishing(original, filled, [_center_mask()], None,
                          feather_px=0, edge_ring_px=0)
    # With feather 0 and no ring, the masked core is exactly the fill.
    assert out[0][16, 16, 0] == 200
