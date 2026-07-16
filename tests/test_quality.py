"""Tests for the SSIM primitive backing the quality report."""

from __future__ import annotations

import math

import numpy as np

from backend.quality import _ssim


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
