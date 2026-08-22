"""SSIM and quality-metric primitives.

Extracted from processor.py as part of RFP-L-1. ``_compute_quality_report``
and ``_write_quality_sheet`` are methods on ``SubtitleRemover`` (they read
``self.config`` + ``self._quality_mask_bbox``) so they stay there; only
the pure-numpy SSIM helper and optional ffmpeg-backed metrics live here.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from backend.subprocess_policy import run_process

logger = logging.getLogger(__name__)


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


def temporal_flicker_score(
    samples: Sequence[Tuple[int, np.ndarray]],
    *,
    max_frame_gap: int = 1,
) -> Optional[float]:
    """Mean adjacent-frame absolute delta for sampled ROI frames.

    The score is normalized to 0..1. Only adjacent sample indices are compared
    by default so sparse random quality samples do not confuse ordinary motion
    with flicker.
    """
    if len(samples) < 2:
        return None
    prepared: list[Tuple[int, np.ndarray]] = []
    for idx, frame in samples:
        arr = _prepare_flicker_frame(frame)
        if arr is not None:
            prepared.append((int(idx), arr))
    if len(prepared) < 2:
        return None
    diffs: List[float] = []
    last_idx, last = prepared[0]
    for idx, cur in prepared[1:]:
        if idx - last_idx <= max_frame_gap:
            diffs.append(float(np.mean(np.abs(cur - last)) / 255.0))
        last_idx, last = idx, cur
    if not diffs:
        return None
    return float(np.mean(diffs))


def residual_text_score(frame: np.ndarray) -> Optional[float]:
    """Return a cheap 0..1 text-residue score for a cleaned ROI.

    This is intentionally dependency-free. It is not a replacement for OCR; it
    flags sharp high-contrast strokes that commonly remain when subtitle text
    was under-masked or when an inpaint fallback left outlines behind.
    """
    if frame is None or frame.size == 0:
        return None
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 2:
        gray = frame
    else:
        return None
    h, w = gray.shape[:2]
    if h < 8 or w < 8:
        return None
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    median = float(np.median(blurred))
    contrast = cv2.absdiff(blurred, np.full_like(blurred, int(round(median))))
    _, mask = cv2.threshold(
        contrast, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(1, w // 96), max(1, h // 96)),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(blurred, 50, 150)
    contour_area = 0.0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if cw < 3 or ch < 3:
            continue
        if ch > h * 0.65:
            continue
        aspect = cw / max(1.0, float(ch))
        if aspect < 0.15 or aspect > 30.0:
            continue
        contour_area += float(cw * ch)
    area = float(max(1, h * w))
    edge_density = float(np.count_nonzero(edges)) / area
    contour_density = contour_area / area
    return float(min(1.0, max(edge_density, contour_density)))


def mask_boundary_seam_score(
    original: np.ndarray,
    filled: np.ndarray,
    mask: np.ndarray,
    band_px: int = 3,
) -> Optional[float]:
    """Return a 0..1 seam score for the boundary of an inpainted region.

    A clean fill blends into the surrounding background, so gradient energy in
    a thin band straddling the mask edge should look like the rest of the
    image. A visible seam (hard box, colour step, halo) shows up as excess
    gradient in that band relative to the same band in the original frame.
    The score is the clipped relative gradient increase; 0 means no visible
    seam. Returns None when there is no boundary to measure.
    """
    if original is None or filled is None or mask is None:
        return None
    if original.shape[:2] != filled.shape[:2] or original.shape[:2] != mask.shape[:2]:
        return None
    if mask.max() == 0 or mask.min() == mask.max():
        return None
    band = max(1, int(band_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band * 2 + 1,) * 2)
    binary = (mask > 0).astype(np.uint8) * 255
    # Thin ring straddling the mask edge -- a visible seam concentrates its
    # extra gradient here, so a narrow band keeps the edge step from being
    # diluted by flat interior or textured exterior pixels.
    boundary = cv2.subtract(cv2.dilate(binary, kernel), cv2.erode(binary, kernel))
    sel = boundary > 0
    if int(np.count_nonzero(sel)) == 0:
        return None

    def _grad(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        gray = gray.astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    filled_grad = _grad(filled)
    original_grad = _grad(original)
    # Excess boundary gradient the fill introduced, over what the original
    # frame already had along the same contour, normalised by the overall
    # image texture scale. A seamless fill leaves ~0; a hard box or blur
    # step raises the boundary gradient well above the original.
    boundary_excess = float(np.mean(filled_grad[sel]) - np.mean(original_grad[sel]))
    texture_scale = max(4.0, float(np.mean(original_grad)))
    return float(min(1.0, max(0.0, boundary_excess / texture_scale)))


def _pyiqa_available() -> bool:
    """True when pyiqa is importable (opt-in advanced metrics)."""
    try:
        import pyiqa  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def compute_extended_metrics(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    metric_names: Sequence[str] = ("lpips", "dists"),
    device: str = "cpu",
) -> dict:
    """Compute perceptual quality metrics via pyiqa on (original, cleaned)
    frame pairs. Returns {metric_name: mean_score} for each available
    metric, or an empty dict when pyiqa is not installed.

    Metrics are computed on the ROI crop when provided; pairs should
    already be cropped by the caller. All scores are lower-is-better
    (distance metrics); pyiqa handles the convention internally.
    """
    try:
        import pyiqa  # type: ignore
    except ImportError:
        return {}
    if not pairs:
        return {}
    try:
        import torch
    except ImportError:
        return {}
    results: dict = {}
    for name in metric_names:
        try:
            metric_fn = pyiqa.create_metric(name, device=device)
        except Exception:
            logger.debug(f"pyiqa metric {name!r} unavailable")
            continue
        scores: List[float] = []
        for orig, cleaned in pairs:
            try:
                a = _frame_to_tensor(orig, device)
                b = _frame_to_tensor(cleaned, device)
                with torch.no_grad():
                    score = metric_fn(a, b)
                scores.append(float(score.item()))
            except Exception:
                continue
        if scores:
            results[name] = float(np.mean(scores))
    return results


def temporal_consistency_score(
    frames: Sequence[np.ndarray],
) -> Optional[float]:
    """Mean pairwise SSIM between consecutive cleaned ROI frames.

    High values (close to 1.0) indicate temporally stable inpainting;
    low values indicate flicker or per-frame inconsistency. Complements
    the simpler absolute-delta flicker score with a structural measure.
    """
    if len(frames) < 2:
        return None
    scores: List[float] = []
    for i in range(len(frames) - 1):
        s = _ssim(frames[i], frames[i + 1])
        if s > 0:
            scores.append(s)
    return float(np.mean(scores)) if scores else None


def _quality_gray_u8(frame: np.ndarray) -> np.ndarray:
    """Return a stable 8-bit luminance view for motion probing."""
    values = np.asarray(frame)
    if values.ndim == 3:
        if values.shape[2] == 1:
            values = values[..., 0]
        else:
            values = cv2.cvtColor(values, cv2.COLOR_BGR2GRAY)
    if values.dtype == np.uint8:
        return values
    if np.issubdtype(values.dtype, np.integer):
        scale = float(np.iinfo(values.dtype).max)
        return np.clip(
            np.rint(values.astype(np.float32) * 255.0 / scale), 0, 255
        ).astype(np.uint8)
    return np.clip(np.rint(values.astype(np.float32) * 255.0), 0, 255).astype(
        np.uint8
    )


def _quality_mask(mask: Optional[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    """Normalize a quality mask to a boolean array matching ``shape``."""
    if mask is None:
        return np.zeros(shape, dtype=bool)
    values = np.asarray(mask)
    if values.ndim == 3:
        values = cv2.cvtColor(values, cv2.COLOR_BGR2GRAY)
    if values.ndim != 2:
        return np.zeros(shape, dtype=bool)
    if values.shape != shape:
        values = cv2.resize(
            values.astype(np.uint8), (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return values > 0


def _quality_histogram(frame: np.ndarray) -> np.ndarray:
    gray = _quality_gray_u8(frame)
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    cv2.normalize(hist, hist)
    return hist


def scene_cut_pair(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    threshold: float = 0.35,
) -> bool:
    """Return whether a pair is too discontinuous for temporal scoring.

    This intentionally follows the inexpensive histogram leg of the existing
    scene-cut cascade. A cut is excluded from the temporal average rather than
    being treated as a failed repair.
    """
    if previous is None or current is None:
        return False
    if previous.shape[:2] != current.shape[:2]:
        return True
    try:
        correlation = cv2.compareHist(
            _quality_histogram(previous),
            _quality_histogram(current),
            cv2.HISTCMP_CORREL,
        )
    except Exception:
        return False
    return bool(correlation < (1.0 - float(threshold)))


def _estimate_quality_motion(
    previous: np.ndarray,
    current: np.ndarray,
    excluded: np.ndarray,
) -> Tuple[Optional[np.ndarray], float]:
    """Estimate dominant affine motion using only untouched pixels."""
    previous_gray = _quality_gray_u8(previous)
    current_gray = _quality_gray_u8(current)
    feature_mask = np.where(
        cv2.dilate(
            excluded.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        ) > 0,
        0,
        255,
    ).astype(np.uint8)
    corners = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=240,
        qualityLevel=0.01,
        minDistance=7,
        mask=feature_mask,
    )
    if corners is None or len(corners) < 6:
        return None, 0.0
    tracked, status, _error = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, corners.astype(np.float32), None,
    )
    if tracked is None or status is None:
        return None, 0.0
    keep = status.ravel() == 1
    source = corners[keep].reshape(-1, 2)
    target = tracked[keep].reshape(-1, 2)
    if len(source) < 6:
        return None, 0.0
    matrix, inliers = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0,
    )
    if matrix is None or inliers is None:
        return None, 0.0
    ratio = float(np.count_nonzero(inliers)) / float(len(source))
    if ratio < 0.35:
        return None, ratio
    return matrix.astype(np.float32), ratio


def mask_local_temporal_pair(
    previous: np.ndarray,
    current: np.ndarray,
    previous_mask: Optional[np.ndarray],
    current_mask: Optional[np.ndarray],
    *,
    reference_previous: Optional[np.ndarray] = None,
    reference_current: Optional[np.ndarray] = None,
    scene_cut_threshold: float = 0.35,
) -> Optional[dict]:
    """Score one cleaned-frame pair inside the active mask.

    The score is the motion-compensated excess error inside the union of the
    two masks over the untouched background error, normalized to 0..1. A
    camera pan that is visible in both regions therefore contributes little,
    while a localized fill jump contributes strongly. Scene cuts return a
    record marked ``scene_cut`` so callers can count the exclusion explicitly.
    """
    if previous is None or current is None:
        return None
    if previous.shape[:2] != current.shape[:2]:
        return None
    shape = previous.shape[:2]
    previous_binary = _quality_mask(previous_mask, shape)
    current_binary = _quality_mask(current_mask, shape)
    if not np.any(previous_binary | current_binary):
        return None
    if scene_cut_pair(
        previous, current, threshold=scene_cut_threshold,
    ):
        return {"scene_cut": True}

    excluded = previous_binary | current_binary
    matrix, inlier_ratio = _estimate_quality_motion(
        previous, current, excluded,
    )
    height, width = shape
    if matrix is None:
        matrix = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    warped_previous = cv2.warpAffine(
        previous, matrix, (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    warped_mask = cv2.warpAffine(
        previous_binary.astype(np.uint8), matrix, (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    union = current_binary | warped_mask
    valid = cv2.warpAffine(
        np.ones(shape, dtype=np.uint8), matrix, (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    outside = (~cv2.dilate(
        union.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    ).astype(bool)) & valid
    inside = union & valid
    if int(np.count_nonzero(inside)) < 16:
        return None
    previous_gray = _quality_gray_u8(warped_previous).astype(np.float32)
    current_gray = _quality_gray_u8(current).astype(np.float32)
    delta = np.abs(current_gray - previous_gray)
    inside_error = float(np.mean(delta[inside]))
    outside_error = (
        float(np.mean(delta[outside]))
        if int(np.count_nonzero(outside))
        else inside_error
    )
    reference_inside_error = None
    if (
        reference_previous is not None
        and reference_current is not None
        and reference_previous.shape[:2] == shape
        and reference_current.shape[:2] == shape
    ):
        warped_reference = cv2.warpAffine(
            reference_previous, matrix, (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        reference_delta = np.abs(
            _quality_gray_u8(reference_current).astype(np.float32)
            - _quality_gray_u8(warped_reference).astype(np.float32)
        )
        reference_inside_error = float(np.mean(reference_delta[inside]))
    baseline_error = max(
        outside_error,
        reference_inside_error if reference_inside_error is not None else 0.0,
    )
    score = float(np.clip(
        max(0.0, inside_error - baseline_error) / 255.0,
        0.0,
        1.0,
    ))
    return {
        "scene_cut": False,
        "score": score,
        "inside_error": inside_error,
        "outside_error": outside_error,
        "reference_inside_error": reference_inside_error,
        "motion_inlier_ratio": float(inlier_ratio),
        "pixels": int(np.count_nonzero(inside)),
    }


def outside_mask_color_drift(
    reference: np.ndarray,
    output: np.ndarray,
    mask: Optional[np.ndarray],
    *,
    hdr_transfer: str = "",
) -> Optional[dict]:
    """Measure changes outside the repair mask in SDR or HDR light space."""
    if reference is None or output is None:
        return None
    if reference.shape[:2] != output.shape[:2]:
        return None
    shape = reference.shape[:2]
    outside = ~_quality_mask(mask, shape)
    if int(np.count_nonzero(outside)) == 0:
        return None
    if hdr_transfer:
        from backend.hdr import hdr_signal_to_linear

        reference_values = hdr_signal_to_linear(reference, hdr_transfer)
        output_values = hdr_signal_to_linear(output, hdr_transfer)
        delta = np.abs(
            output_values.astype(np.float32)
            - reference_values.astype(np.float32)
        )
        if delta.ndim == 3:
            delta = np.mean(delta, axis=2)
        metric = "linear_rgb_mae"
    else:
        reference_u8 = _quality_gray_u8(reference)
        output_u8 = _quality_gray_u8(output)
        if reference.ndim == 3 and output.ndim == 3:
            reference_u8 = cv2.cvtColor(
                _quality_bgr_u8(reference), cv2.COLOR_BGR2LAB,
            ).astype(np.float32)
            output_u8 = cv2.cvtColor(
                _quality_bgr_u8(output), cv2.COLOR_BGR2LAB,
            ).astype(np.float32)
            delta = np.sqrt(np.sum(
                np.square((output_u8 - reference_u8) / 2.55), axis=2,
            ))
        else:
            delta = np.abs(
                output_u8.astype(np.float32)
                - reference_u8.astype(np.float32)
            ) / 255.0 * 100.0
        metric = "cielab_delta_e"
    values = np.asarray(delta, dtype=np.float32)[outside]
    if values.size == 0:
        return None
    return {
        "score": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "metric": metric,
        "pixels": int(values.size),
    }


def _quality_bgr_u8(frame: np.ndarray) -> np.ndarray:
    values = np.asarray(frame)
    if values.dtype == np.uint8:
        return values
    if np.issubdtype(values.dtype, np.integer):
        scale = float(np.iinfo(values.dtype).max)
        return np.clip(
            np.rint(values.astype(np.float32) * 255.0 / scale), 0, 255
        ).astype(np.uint8)
    return np.clip(np.rint(values.astype(np.float32) * 255.0), 0, 255).astype(
        np.uint8
    )


def _mask_bbox_from_masks(
    masks: Sequence[np.ndarray],
    *,
    padding: int = 4,
) -> Optional[Tuple[int, int, int, int]]:
    if not masks:
        return None
    height = width = None
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for mask in masks:
        if mask is None:
            continue
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        if arr.ndim != 2:
            continue
        if height is None or width is None:
            height, width = arr.shape[:2]
        y, x = np.where(arr > 0)
        if x.size:
            xs.append(x)
            ys.append(y)
    if not xs or height is None or width is None:
        return None
    all_x = np.concatenate(xs)
    all_y = np.concatenate(ys)
    x1 = max(0, int(all_x.min()) - padding)
    y1 = max(0, int(all_y.min()) - padding)
    x2 = min(width, int(all_x.max()) + padding + 1)
    y2 = min(height, int(all_y.max()) + padding + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _crop_bbox(frame: np.ndarray,
               bbox: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    if bbox is None:
        return frame
    x1, y1, x2, y2 = bbox
    return frame[y1:y2, x1:x2]


def _mean_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(clean)) if clean else None


def harmonic_mean(values: Sequence[Optional[float]]) -> Optional[float]:
    """Harmonic mean of the finite positive samples, or None.

    RM-281: a handful of badly filled frames on an otherwise sharp clip is
    this product's characteristic defect, and an arithmetic mean is
    structurally blind to it. The harmonic mean weights the low samples
    much more heavily, which is why libvmaf pools it alongside the mean.

    A zero sample pulls the harmonic mean to zero, which is both the
    correct arithmetic and the correct signal: an SSIM of exactly 0 is the
    worst frame this metric can describe, and dropping it would push the
    result ABOVE the arithmetic mean and paint the run as healthy.
    Non-finite samples are dropped instead -- they carry no information.
    """
    clean = [
        float(v) for v in values
        if v is not None and np.isfinite(v)
    ]
    if not clean:
        return None
    if any(value <= 0.0 for value in clean):
        return 0.0
    return float(len(clean) / np.sum(1.0 / np.asarray(clean, dtype=np.float64)))


def worst_sample(
    frames: Sequence[int],
    psnrs: Sequence[float],
    ssims: Sequence[float],
) -> Optional[dict]:
    """Return the lowest-SSIM sample as ``{frame, psnr, ssim}``.

    Ties break on PSNR so the returned frame is the worst on both axes when
    SSIM cannot separate them. Mismatched input lengths return None.
    """
    count = len(frames)
    if count <= 0:
        return None
    if len(psnrs) != count or len(ssims) != count:
        # Trimming to the shortest list would pair every later score with the
        # wrong frame, which is worse than reporting no worst frame at all.
        logger.warning(
            "Quality samples desynchronised (%d frames, %d PSNR, %d SSIM); "
            "skipping the worst-frame report",
            count, len(psnrs), len(ssims),
        )
        return None
    best_index = min(
        range(count),
        key=lambda i: (float(ssims[i]), float(psnrs[i])),
    )
    return {
        "frame": int(frames[best_index]),
        "psnr": float(psnrs[best_index]),
        "ssim": float(ssims[best_index]),
    }


def _static_logo_mask_coverage(masks: Sequence[np.ndarray]) -> float:
    total_pixels = 0
    masked_pixels = 0
    for mask in masks:
        if mask is None:
            continue
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        if arr.ndim != 2:
            continue
        total_pixels += int(arr.shape[0] * arr.shape[1])
        masked_pixels += int(np.count_nonzero(arr))
    if total_pixels <= 0:
        return 0.0
    return float(masked_pixels / total_pixels)


def _best_method(
    method_metrics: dict,
    key: str,
    *,
    higher_is_better: bool = False,
) -> Optional[str]:
    candidates = []
    for name, metrics in method_metrics.items():
        value = metrics.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            candidates.append((float(value), name))
    if not candidates:
        return None
    candidates.sort(reverse=higher_is_better)
    return candidates[0][1]


def compare_static_logo_cleanup(
    original_frames: Sequence[np.ndarray],
    method_outputs: dict,
    masks: Sequence[np.ndarray],
    *,
    reference_frames: Optional[Sequence[np.ndarray]] = None,
) -> dict:
    """Compare static-logo cleanup outputs using dependency-free ROI metrics.

    Lower residual/flicker is better; higher temporal consistency and reference
    SSIM are better. This is a benchmark primitive, not a production quality
    gate, so it returns all available metrics and omits unavailable optional
    values instead of trying to judge pass/fail.
    """
    originals = list(original_frames)
    masks = list(masks)
    if len(originals) != len(masks):
        raise ValueError("original frames and masks must have the same length")
    if not originals:
        raise ValueError("static-logo comparison needs at least one frame")
    bbox = _mask_bbox_from_masks(masks)
    refs = list(reference_frames) if reference_frames is not None else None
    if refs is not None and len(refs) != len(originals):
        raise ValueError("reference frames must match original frame count")

    metrics_by_method = {}
    for name, frames in sorted(method_outputs.items()):
        output_frames = list(frames)
        if len(output_frames) != len(originals):
            raise ValueError(f"method {name!r} frame count does not match input")
        roi_frames = [_crop_bbox(frame, bbox) for frame in output_frames]
        residuals = [residual_text_score(frame) for frame in roi_frames]
        samples = list(enumerate(roi_frames))
        method_metrics = {
            "roiFrameCount": len(roi_frames),
            "residualTextScoreMean": _mean_optional(residuals),
            "temporalFlickerScore": temporal_flicker_score(samples),
            "temporalConsistency": temporal_consistency_score(roi_frames),
        }
        if refs is not None:
            ref_scores: List[Optional[float]] = []
            for ref, out in zip(refs, output_frames):
                ref_roi = _crop_bbox(ref, bbox)
                out_roi = _crop_bbox(out, bbox)
                ref_scores.append(_ssim(ref_roi, out_roi))
            method_metrics["ssimVsReferenceMean"] = _mean_optional(ref_scores)
        metrics_by_method[str(name)] = method_metrics

    return {
        "maskCoverage": _static_logo_mask_coverage(masks),
        "roiBbox": list(bbox) if bbox is not None else None,
        "methods": metrics_by_method,
        "winners": {
            "lowestResidualText": _best_method(
                metrics_by_method, "residualTextScoreMean"),
            "lowestFlicker": _best_method(
                metrics_by_method, "temporalFlickerScore"),
            "highestTemporalConsistency": _best_method(
                metrics_by_method, "temporalConsistency",
                higher_is_better=True),
            "highestReferenceSsim": _best_method(
                metrics_by_method, "ssimVsReferenceMean",
                higher_is_better=True),
        },
    }


def _frame_to_tensor(frame: np.ndarray, device: str = "cpu"):
    """BGR uint8 frame -> torch float32 NCHW tensor in [0, 1]."""
    import torch
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return t.to(device)


def _prepare_flicker_frame(frame: np.ndarray) -> Optional[np.ndarray]:
    if frame is None or frame.size == 0:
        return None
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 2:
        gray = frame
    else:
        return None
    h, w = gray.shape[:2]
    if h <= 0 or w <= 0:
        return None
    scale = min(1.0, 96.0 / max(h, w))
    if scale < 1.0:
        gray = cv2.resize(
            gray,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return gray.astype(np.float32)


def ffmpeg_libvmaf_available(ffmpeg: str = "ffmpeg") -> bool:
    """Return True when `ffmpeg -filters` reports libvmaf."""
    if shutil.which(ffmpeg) is None:
        return False
    try:
        result = run_process(
            [ffmpeg, "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    output = f"{result.stdout}\n{result.stderr}"
    return result.returncode == 0 and bool(
        re.search(r"(?m)^\s*[.A-Z| ]+\s+libvmaf\s+", output)
    )


def _escape_filter_value(value: str) -> str:
    escaped = str(value).replace("\\", "\\\\")
    for char in (":", "'", ",", "[", "]"):
        escaped = escaped.replace(char, f"\\{char}")
    return escaped


def _unescape_filter_value(value: str) -> str:
    out = []
    escaped = False
    for char in value:
        if escaped:
            out.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        else:
            out.append(char)
    if escaped:
        out.append("\\")
    return "".join(out)


def _vmaf_filter(log_path: str,
                 roi: Optional[Tuple[int, int, int, int]] = None) -> str:
    ref = "[0:v]settb=AVTB,setpts=PTS-STARTPTS"
    dist = "[1:v]settb=AVTB,setpts=PTS-STARTPTS"
    if roi is not None:
        x1, y1, x2, y2 = roi
        width = max(1, int(x2) - int(x1))
        height = max(1, int(y2) - int(y1))
        ref += f",crop={width}:{height}:{int(x1)}:{int(y1)}"
        dist += f",crop={width}:{height}:{int(x1)}:{int(y1)}"
    return (
        f"{ref}[ref];{dist}[dist];"
        f"[dist][ref]libvmaf=log_fmt=json:log_path={_escape_filter_value(log_path)}"
    )


def _read_vmaf_score(log_path: Path) -> Optional[float]:
    try:
        payload = json.loads(log_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    pooled = payload.get("pooled_metrics", {})
    score = pooled.get("vmaf", {}).get("mean")
    if isinstance(score, (int, float)):
        return float(score)
    frame_scores = []
    for frame in payload.get("frames", []) or []:
        value = (frame.get("metrics") or {}).get("vmaf")
        if isinstance(value, (int, float)):
            frame_scores.append(float(value))
    if frame_scores:
        return float(np.mean(frame_scores))
    return None


def compute_vmaf(
    reference_path: str,
    distorted_path: str,
    *,
    start_seconds: float = 0.0,
    duration_seconds: Optional[float] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
    ffmpeg: str = "ffmpeg",
) -> Optional[float]:
    """Compute VMAF via ffmpeg's libvmaf filter.

    Returns None when ffmpeg/libvmaf is unavailable or the invocation
    fails. The first input is the reference/original, the second is the
    distorted/cleaned output.
    """
    if not Path(reference_path).is_file() or not Path(distorted_path).is_file():
        return None
    if not ffmpeg_libvmaf_available(ffmpeg):
        logger.info("ffmpeg libvmaf filter unavailable; VMAF omitted.")
        return None
    duration = None if duration_seconds is None else max(0.1, float(duration_seconds))
    start = max(0.0, float(start_seconds))
    with tempfile.TemporaryDirectory(prefix="vsr_vmaf_") as tmpdir:
        log_path = Path(tmpdir) / "vmaf.json"
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats"]
        for path in (reference_path, distorted_path):
            if start > 0:
                cmd += ["-ss", f"{start:.3f}"]
            if duration is not None:
                cmd += ["-t", f"{duration:.3f}"]
            cmd += ["-i", path]
        cmd += [
            # Run from the private temp directory and give libvmaf a plain
            # relative filename. This avoids FFmpeg's Windows filter parser
            # interpreting a drive-letter colon/backslash as filter syntax.
            "-lavfi", _vmaf_filter(log_path.name, roi),
            "-f", "null", "-",
        ]
        timeout = 600.0 if duration is None else max(600.0, duration * 20.0)
        try:
            run_process(
                cmd,
                check=True,
                capture_output=True,
                timeout=timeout,
                cwd=tmpdir,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr or b""
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", "replace")
            logger.info(f"ffmpeg libvmaf failed: {stderr[:400]}")
            return None
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg libvmaf timed out")
            return None
        except FileNotFoundError:
            return None
        return _read_vmaf_score(log_path)
