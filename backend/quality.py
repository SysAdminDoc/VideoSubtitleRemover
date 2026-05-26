"""SSIM and quality-metric primitives.

Extracted from processor.py as part of RFP-L-1. ``_compute_quality_report``
and ``_write_quality_sheet`` are methods on ``SubtitleRemover`` (they read
``self.config`` + ``self._quality_mask_bbox``) so they stay there; only
the pure-numpy SSIM helper lives here.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural Similarity between two BGR frames. Mean over the three
    channels. Standard formulation (C1, C2 = (0.01*255)^2, (0.03*255)^2).
    Flat-colour regions where the variance and covariance are all zero
    can still drive (num/den) close to 0/0; we wrap in errstate +
    nan_to_num so the report never yields NaN or inf.
    """
    if a is None or b is None or a.shape != b.shape or a.ndim < 2:
        return 0.0
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    if a.ndim == 2:
        a32 = a32[..., None]
        b32 = b32[..., None]
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    channels = a32.shape[2]
    ssims: List[float] = []
    with np.errstate(invalid='ignore', divide='ignore'):
        for c in range(channels):
            x = a32[..., c]
            y = b32[..., c]
            mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
            mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
            mu_x2 = mu_x * mu_x
            mu_y2 = mu_y * mu_y
            mu_xy = mu_x * mu_y
            sig_x2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2
            sig_y2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2
            sig_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy
            num = (2 * mu_xy + C1) * (2 * sig_xy + C2)
            den = (mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2)
            ratio = np.where(den > 0, num / np.maximum(den, 1e-12), 1.0)
            ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=0.0)
            ssims.append(float(np.mean(ratio)))
    if not ssims:
        return 0.0
    return float(np.clip(np.mean(ssims), 0.0, 1.0))
