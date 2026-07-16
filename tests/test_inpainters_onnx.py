"""Tests for the ONNX inpainter mask contract.

LaMa / MI-GAN ONNX sessions consume ``mask / 255.0`` as a binary inpaint
indicator. A greyscale or anti-aliased mask reaching these paths must be
thresholded to a strict binary mask first, otherwise the model receives
partial-strength hints (values strictly between 0 and 1) and degrades the
fill quality.
"""

from __future__ import annotations

import numpy as np

from backend.inpainters_onnx import (
    LamaOnnxInpainter,
    MiGanInpainter,
    _binarize_mask,
)


class _CaptureSession:
    """Minimal stub that records the inputs passed to ``run`` and returns a
    zero-valued output tensor of the expected shape."""

    def __init__(self, channels_out: int = 3):
        self.captured = None
        self._channels_out = channels_out

    def get_inputs(self):
        class _Inp:
            def __init__(self, name):
                self.name = name

        return [_Inp("image"), _Inp("mask")]

    def run(self, _outputs, feed):
        self.captured = feed
        # Infer spatial size from the image tensor: (1, 3, H, W).
        img = feed["image"]
        _, _, h, w = img.shape
        return [np.zeros((1, self._channels_out, h, w), dtype=np.float32)]


def _greyscale_mask(h: int = 64, w: int = 64) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    # A soft, anti-aliased blob: a solid core surrounded by partial values.
    mask[20:40, 20:40] = 255
    mask[18:20, 18:42] = 90     # sub-threshold halo -> must drop to 0
    mask[40:42, 18:42] = 200    # above-threshold halo -> must become 255
    return mask


def test_binarize_helper_is_strict_binary():
    out = _binarize_mask(_greyscale_mask())
    assert set(np.unique(out)).issubset({0, 255})
    # Midpoint threshold: 200 -> 255, 90 -> 0.
    assert out[41, 30] == 255
    assert out[19, 30] == 0


def test_binarize_reduces_multichannel_mask():
    mask = np.dstack([_greyscale_mask()] * 3)
    out = _binarize_mask(mask)
    assert out.ndim == 2
    assert set(np.unique(out)).issubset({0, 255})


def test_lama_onnx_receives_binary_mask():
    inp = LamaOnnxInpainter.__new__(LamaOnnxInpainter)
    inp.config = None
    session = _CaptureSession()
    inp._session = session

    frame = np.full((64, 64, 3), 120, dtype=np.uint8)
    inp._inpaint_one(frame, _greyscale_mask())

    fed_mask = session.captured["mask"]
    uniques = set(np.unique(fed_mask).tolist())
    assert uniques.issubset({0.0, 1.0}), uniques


def test_migan_onnx_receives_binary_mask():
    inp = MiGanInpainter.__new__(MiGanInpainter)
    inp.config = None
    session = _CaptureSession()
    inp._session = session

    frame = np.full((64, 64, 3), 120, dtype=np.uint8)
    inp._inpaint_one(frame, _greyscale_mask())

    fed_mask = session.captured["mask"]
    uniques = set(np.unique(fed_mask).tolist())
    assert uniques.issubset({0.0, 1.0}), uniques
