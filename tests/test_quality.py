"""Tests for the SSIM primitive backing the quality report."""

from __future__ import annotations

import math

import numpy as np

from backend.hdr import linear_to_hdr_signal
from backend.quality import (
    _ssim,
    mask_local_temporal_pair,
    outside_mask_color_drift,
    scene_cut_pair,
)
from backend.temporal_profile import synthetic_clip


def _frame(seed: int, h: int = 48, w: int = 48) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_identical_frames_score_one():
    frame = _frame(1)
    assert _ssim(frame, frame) == 1.0


def test_identical_grayscale_frames_score_one():
    rng = np.random.RandomState(2)
    gray = rng.randint(0, 256, size=(48, 48), dtype=np.uint8)
    assert _ssim(gray, gray) == 1.0


def test_shape_mismatch_and_none_return_zero():
    assert _ssim(_frame(3), None) == 0.0
    assert _ssim(_frame(3), _frame(3, h=32)) == 0.0


def test_different_frames_score_below_identical_and_stay_bounded():
    score = _ssim(_frame(4), _frame(5))
    assert 0.0 <= score < 1.0
    assert math.isfinite(score)


def test_flat_regions_never_nan():
    black = np.zeros((48, 48, 3), dtype=np.uint8)
    white = np.full((48, 48, 3), 255, dtype=np.uint8)
    # Identical flat frames -> perfect; opposite flats -> finite, bounded.
    assert _ssim(black, black) == 1.0
    score = _ssim(black, white)
    assert math.isfinite(score)
    assert 0.0 <= score <= 1.0


def test_mask_local_temporal_score_is_stable_for_static_and_camera_motion():
    for kwargs in ({}, {"camera_motion": 6.0}):
        clean, _subtitled, masks = synthetic_clip(frames=8, **kwargs)
        scores = []
        for index in range(len(clean) - 1):
            result = mask_local_temporal_pair(
                clean[index], clean[index + 1], masks[index], masks[index + 1],
                reference_previous=clean[index],
                reference_current=clean[index + 1],
            )
            assert result is not None
            assert not result["scene_cut"]
            scores.append(result["score"])
        assert max(scores) < 0.08


def test_mask_local_temporal_score_flags_flicker_but_credits_occlusion():
    clean, _subtitled, masks = synthetic_clip(
        frames=8, camera_motion=6.0,
    )
    flickering = [frame.copy() for frame in clean]
    flickering[3][masks[3] > 0] = 0
    flicker = mask_local_temporal_pair(
        flickering[2], flickering[3], masks[2], masks[3],
        reference_previous=clean[2],
        reference_current=clean[3],
    )
    assert flicker is not None
    assert flicker["score"] > 0.08

    occluded = [frame.copy() for frame in clean]
    occluded[3][masks[3] > 0] = 255
    valid_occlusion = mask_local_temporal_pair(
        occluded[2], occluded[3], masks[2], masks[3],
        reference_previous=occluded[2],
        reference_current=occluded[3],
    )
    assert valid_occlusion is not None
    assert valid_occlusion["score"] < 0.08


def test_scene_cut_is_excluded_from_the_temporal_score():
    clean, _subtitled, masks = synthetic_clip(frames=4)
    cut = np.zeros_like(clean[1])
    assert scene_cut_pair(clean[0], cut)
    result = mask_local_temporal_pair(clean[0], cut, masks[0], masks[1])
    assert result == {"scene_cut": True}


def test_outside_mask_drift_uses_cielab_for_sdr_and_catches_global_cast():
    reference = np.full((64, 96, 3), 100, dtype=np.uint8)
    mask = np.zeros((64, 96), dtype=np.uint8)
    mask[20:40, 20:60] = 255
    unchanged = outside_mask_color_drift(reference, reference, mask)
    assert unchanged is not None
    assert unchanged["metric"] == "cielab_delta_e"
    assert unchanged["score"] == 0.0

    cast = np.clip(reference.astype(np.int16) + 20, 0, 255).astype(np.uint8)
    drift = outside_mask_color_drift(reference, cast, mask)
    assert drift is not None
    assert drift["metric"] == "cielab_delta_e"
    assert drift["score"] > 1.0


def test_outside_mask_drift_uses_linear_light_for_tagged_hdr():
    linear = np.full((48, 64, 3), 0.2, dtype=np.float32)
    reference = linear_to_hdr_signal(linear, "smpte2084")
    output = linear_to_hdr_signal(linear * 1.02, "smpte2084")
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[16:32, 16:48] = 255
    drift = outside_mask_color_drift(
        reference, output, mask, hdr_transfer="smpte2084",
    )
    assert drift is not None
    assert drift["metric"] == "linear_rgb_mae"
    assert 0.0 < drift["score"] < 0.01
