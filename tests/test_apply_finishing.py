"""Tests for the shared post-inpaint finishing step.

``apply_finishing`` is the single edge-ring + feather routine used by the
ONNX, diffusion, and built-in inpainter families, replacing three separate
re-implementations of the same loop.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.inpainters import apply_finishing
from backend import temporal_profile


class _Config:
    def __init__(self, feather=4, ring=2, poisson=False, grain=0.0):
        self.mask_feather_px = feather
        self.edge_ring_px = ring
        self.poisson_seam_enable = poisson
        self.film_grain_strength = grain


def _frame(value):
    return np.full((32, 32, 3), value, dtype=np.uint8)


def _center_mask():
    m = np.zeros((32, 32), dtype=np.uint8)
    m[10:22, 10:22] = 255
    return m


def _gradient_fixture():
    height, width = 96, 128
    x = np.arange(width, dtype=np.float32)[None, :]
    y = np.arange(height, dtype=np.float32)[:, None]
    gradient = np.clip(30 + 0.9 * x + 0.35 * y, 0, 255).astype(np.uint8)
    clean = np.dstack([gradient] * 3)
    original = clean.copy()
    cv2.putText(
        original, "SUB", (41, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
        (255, 255, 255), 2, cv2.LINE_AA)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[24:72, 34:94] = 255
    filled = np.full_like(original, 128)
    return clean, original, filled, mask


def _grain_fixture():
    height = width = 96
    rng = np.random.default_rng(5)
    base = np.full((height, width, 3), 120, dtype=np.float32)
    source = np.clip(
        base + rng.normal(0, 7, (height, width, 1)), 0, 255
    ).astype(np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[24:72, 24:72] = 255
    source[mask > 0] = 235
    filled = np.full_like(source, 120)
    outer = cv2.dilate(
        mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) > 0
    inner = cv2.dilate(
        mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    return source, filled, mask, outer & ~inner


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


def test_poisson_seam_reduces_gradient_residual_vs_edge_ring():
    clean, original, filled, mask = _gradient_fixture()
    ring = apply_finishing(
        [original], [filled.copy()], [mask], _Config(feather=0, ring=3))
    poisson = apply_finishing(
        [original], [filled.copy()], [mask],
        _Config(feather=0, ring=3, poisson=True),
    )
    core = mask > 0
    ring_error = np.abs(
        ring[0][core].astype(np.int16) - clean[core].astype(np.int16)
    ).mean()
    poisson_error = np.abs(
        poisson[0][core].astype(np.int16) - clean[core].astype(np.int16)
    ).mean()
    assert poisson_error < ring_error


def test_poisson_seam_skips_edge_and_degenerate_masks():
    original = _frame(50)
    filled = _frame(200)
    edge_mask = np.zeros((32, 32), dtype=np.uint8)
    edge_mask[0:4, 10:20] = 255
    tiny_mask = np.zeros((32, 32), dtype=np.uint8)
    tiny_mask[15:18, 15:18] = 255
    output = apply_finishing(
        [original, original], [filled, filled], [edge_mask, tiny_mask],
        _Config(feather=0, ring=0, poisson=True), edge_ring=False,
    )
    np.testing.assert_array_equal(output[0], filled)
    np.testing.assert_array_equal(output[1], filled)


def test_poisson_seam_keeps_temporal_flicker_under_profile_floor():
    _clean, subtitled, masks = temporal_profile.synthetic_clip(
        frames=8, background_motion=1.5, seed=7)
    filled = []
    for frame, mask in zip(subtitled, masks):
        result = frame.copy()
        result[mask > 0] = 128
        filled.append(result)

    poisson = apply_finishing(
        subtitled, filled, masks, _Config(feather=0, ring=3, poisson=True))
    flicker = temporal_profile.masked_flicker(poisson, masks)
    assert flicker is not None
    assert flicker <= temporal_profile.DEFAULT_THRESHOLDS.masked_flicker


def test_grain_restore_matches_adjacent_noise_variance():
    source, filled, mask, ring = _grain_fixture()
    output = apply_finishing(
        [source], [filled], [mask],
        _Config(feather=0, ring=0, grain=0.04),
    )[0]
    outside_std = float(source[ring].astype(np.float32).std(axis=0).mean())
    inside_std = float(output[mask > 0].astype(np.float32).std(axis=0).mean())
    assert output.dtype == np.uint8
    assert inside_std == pytest.approx(outside_std, rel=0.3, abs=0.5)


def test_grain_restore_decorrelates_frames_and_skips_clean_source():
    source, filled, mask, _ring = _grain_fixture()
    outputs = apply_finishing(
        [source, source, source], [filled.copy() for _ in range(3)],
        [mask, mask, mask], _Config(feather=0, ring=0, grain=0.04),
    )
    first = outputs[0][mask > 0].astype(np.float32).ravel()
    second = outputs[1][mask > 0].astype(np.float32).ravel()
    assert abs(float(np.corrcoef(first, second)[0, 1])) < 0.2

    clean = np.full_like(source, 120)
    clean_output = apply_finishing(
        [clean], [filled], [mask], _Config(feather=0, ring=0, grain=0.04)
    )[0]
    np.testing.assert_array_equal(clean_output, filled)
