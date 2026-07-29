"""Mask-aware temporal regression profile.

RM-150: per-frame image scores (PSNR/SSIM on sampled frames) miss the failures
that actually dominate video inpainting -- a fill that is sharp in every single
frame but does not move with the scene, ghosts the removed text back in, leaks
past the mask boundary, or flickers between frames.

The metrics here are all mask-aware and motion-compensated, so ordinary camera
or background motion is not mistaken for a defect:

* ``masked_warp_residual`` -- after compensating global (camera) motion, how
  much more does the filled region disagree frame-to-frame than the untouched
  background does? A fill that is pinned to the frame instead of the scene
  scores high.
* ``mask_edge_leakage`` -- gradient energy in a ring just *outside* the mask,
  relative to the same ring in the source. A fill that bleeds past its mask
  raises it.
* ``masked_flicker`` -- motion-compensated frame-to-frame delta restricted to
  the filled region, normalised by the background's own delta.

Everything is computed with NumPy and OpenCV: no learned metric is downloaded
and no licensed real media is required. The synthetic fixtures in
``synthetic_clip`` independently vary camera motion, background motion, and
mask motion so a regression can be attributed to the right axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np


TEMPORAL_PROFILE_SCHEMA = "vsr.temporal_profile.v1"


@dataclass(frozen=True)
class TemporalThresholds:
    """Fail bars for the mask-aware temporal metrics (lower is better)."""

    warp_residual: float = 2.5
    # A fill must not touch a pixel outside its own mask, so a clean run
    # measures 0.0 here. The allowance only covers re-encode noise in the
    # outer ring, not visible bleed.
    edge_leakage: float = 0.03
    masked_flicker: float = 2.5


DEFAULT_THRESHOLDS = TemporalThresholds()


def _gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def _binary(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask) > 0).astype(np.uint8)


def estimate_global_motion(previous: np.ndarray, current: np.ndarray):
    """Estimate the dominant (camera) motion between two frames.

    Returns a 2x3 affine matrix, or None when the estimate is not reliable.
    Uses sparse feature tracking so a moving foreground object cannot dominate
    the estimate the way dense whole-frame correlation would.
    """
    prev_gray = _gray(previous)
    cur_gray = _gray(current)
    corners = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=8)
    if corners is None or len(corners) < 6:
        return None
    tracked, status, _err = cv2.calcOpticalFlowPyrLK(
        prev_gray, cur_gray, corners.astype(np.float32), None)
    if tracked is None or status is None:
        return None
    keep = status.ravel() == 1
    src = corners[keep].reshape(-1, 2)
    dst = tracked[keep].reshape(-1, 2)
    if len(src) < 6:
        return None
    matrix, _inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    return matrix


def _warp(frame: np.ndarray, matrix) -> np.ndarray:
    height, width = frame.shape[:2]
    if matrix is None:
        return frame
    return cv2.warpAffine(
        frame, matrix, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _region_means(residual: np.ndarray,
                  inside: np.ndarray,
                  outside: np.ndarray):
    if int(inside.sum()) == 0 or int(outside.sum()) == 0:
        return None, None
    return (
        float(np.mean(residual[inside.astype(bool)])),
        float(np.mean(residual[outside.astype(bool)])),
    )


def _ratio(inside: Optional[float], outside: Optional[float]) -> Optional[float]:
    if inside is None or outside is None:
        return None
    # A small floor keeps a perfectly static background from producing an
    # infinite ratio for any nonzero fill residual.
    return float(inside / max(outside, 0.75))


def masked_warp_residual(
    frames: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
) -> Optional[float]:
    """Motion-compensated residual inside the fill, relative to outside it.

    ~1.0 means the filled region is as temporally coherent as the untouched
    background. Values well above 1 mean the fill does not follow the scene.
    """
    if len(frames) < 2 or len(masks) < len(frames):
        return None
    scores = []
    for index in range(len(frames) - 1):
        previous, current = frames[index], frames[index + 1]
        if previous.shape != current.shape:
            continue
        matrix = estimate_global_motion(previous, current)
        warped = _warp(previous, matrix)
        residual = np.abs(
            _gray(current).astype(np.float32) - _gray(warped).astype(np.float32))
        # Union of both frames' masks: a moving subtitle box covers different
        # pixels in each frame and both are "inside the fill" for this pair.
        inside = np.clip(
            _binary(masks[index]) + _binary(masks[index + 1]), 0, 1)
        outside = 1 - cv2.dilate(
            inside, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        in_mean, out_mean = _region_means(residual, inside, outside)
        ratio = _ratio(in_mean, out_mean)
        if ratio is not None:
            scores.append(ratio)
    return float(np.mean(scores)) if scores else None


def mask_edge_leakage(
    sources: Sequence[np.ndarray],
    filled: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    *,
    band_px: int = 4,
) -> Optional[float]:
    """Excess gradient just outside the mask, relative to the source.

    0 means the fill stopped at its mask. Positive values mean it bled into
    pixels it was never supposed to touch.
    """
    if not filled or len(sources) < len(filled) or len(masks) < len(filled):
        return None
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (band_px * 2 + 1,) * 2)
    scores = []
    for source, output, mask in zip(sources, filled, masks):
        binary = _binary(mask) * 255
        if binary.max() == 0:
            continue
        outer = cv2.subtract(cv2.dilate(binary, kernel), binary)
        sel = outer > 0
        if int(np.count_nonzero(sel)) == 0:
            continue
        delta = np.abs(
            output.astype(np.float32) - source.astype(np.float32))
        if delta.ndim == 3:
            delta = delta.mean(axis=2)
        # The fill must not change pixels outside its own mask at all, so any
        # difference in the outer ring is leakage. Normalise by the dynamic
        # range so the score stays 0..1.
        scores.append(float(np.mean(delta[sel]) / 255.0))
    return float(np.mean(scores)) if scores else None


def masked_flicker(
    frames: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
) -> Optional[float]:
    """Motion-compensated flicker inside the fill, relative to the background.

    Complements the warp residual by weighting sudden brightness swings that a
    mean residual can average away.
    """
    if len(frames) < 3 or len(masks) < len(frames):
        return None
    inside_deltas = []
    outside_deltas = []
    for index in range(len(frames) - 1):
        previous, current = frames[index], frames[index + 1]
        if previous.shape != current.shape:
            continue
        matrix = estimate_global_motion(previous, current)
        warped = _gray(_warp(previous, matrix)).astype(np.float32)
        delta = np.abs(_gray(current).astype(np.float32) - warped)
        inside = np.clip(
            _binary(masks[index]) + _binary(masks[index + 1]), 0, 1)
        outside = 1 - cv2.dilate(
            inside, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        in_mean, out_mean = _region_means(delta, inside, outside)
        if in_mean is None:
            continue
        inside_deltas.append(in_mean)
        outside_deltas.append(out_mean)
    if len(inside_deltas) < 2:
        return None
    # Standard deviation of the per-pair delta captures a fill that jumps
    # between frames even when its mean residual looks reasonable.
    inside_spread = float(np.std(inside_deltas))
    outside_spread = float(np.std(outside_deltas))
    return float(inside_spread / max(outside_spread, 0.4))


def evaluate_temporal_profile(
    sources: Sequence[np.ndarray],
    filled: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    *,
    thresholds: TemporalThresholds = DEFAULT_THRESHOLDS,
    label: str = "",
) -> dict:
    """Score a cleaned clip against the mask-aware temporal fail bars."""
    warp = masked_warp_residual(filled, masks)
    leakage = mask_edge_leakage(sources, filled, masks)
    flicker = masked_flicker(filled, masks)
    failures = []
    if warp is not None and warp > thresholds.warp_residual:
        failures.append(
            f"masked warp residual {warp:.3f} exceeds "
            f"{thresholds.warp_residual}")
    if leakage is not None and leakage > thresholds.edge_leakage:
        failures.append(
            f"mask edge leakage {leakage:.3f} exceeds {thresholds.edge_leakage}")
    if flicker is not None and flicker > thresholds.masked_flicker:
        failures.append(
            f"masked flicker {flicker:.3f} exceeds {thresholds.masked_flicker}")
    measured = [warp, leakage, flicker]
    return {
        "schema": TEMPORAL_PROFILE_SCHEMA,
        "label": label,
        "frames": len(filled),
        "maskedWarpResidual": None if warp is None else round(warp, 6),
        "maskEdgeLeakage": None if leakage is None else round(leakage, 6),
        "maskedFlicker": None if flicker is None else round(flicker, 6),
        "thresholds": {
            "maskedWarpResidual": thresholds.warp_residual,
            "maskEdgeLeakage": thresholds.edge_leakage,
            "maskedFlicker": thresholds.masked_flicker,
        },
        "measured": any(item is not None for item in measured),
        "passed": not failures,
        "failures": failures,
    }


# --------------------------------------------------------------------------
# Deterministic synthetic fixtures
# --------------------------------------------------------------------------


def _background(width: int, height: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    base = rng.randint(40, 210, size=(height // 8 + 2, width // 8 + 2, 3))
    return cv2.resize(
        base.astype(np.uint8), (width, height), interpolation=cv2.INTER_CUBIC)


def synthetic_clip(
    *,
    frames: int = 8,
    width: int = 192,
    height: int = 128,
    camera_motion: float = 0.0,
    background_motion: float = 0.0,
    mask_motion: float = 0.0,
    seed: int = 7,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Build (clean, subtitled, masks) with independently varied motion axes.

    ``camera_motion`` pans the whole plate, ``background_motion`` moves a
    foreground object across it, and ``mask_motion`` moves the subtitle box.
    Each axis can be exercised on its own so a regression is attributable.
    """
    plate = _background(width * 2, height * 2, seed)
    clean: list[np.ndarray] = []
    subtitled: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    box_w, box_h = width // 3, height // 6
    for index in range(frames):
        offset = int(round(camera_motion * index))
        view = plate[
            height // 2:height // 2 + height,
            width // 2 + offset:width // 2 + offset + width,
        ].copy()
        if background_motion:
            cx = int(round(10 + background_motion * index)) % max(1, width - 20)
            cv2.circle(view, (cx, height // 2), 12, (250, 40, 40), -1)
        clean.append(view.copy())

        mx = int(round(mask_motion * index))
        x0 = max(0, min(width - box_w, width // 3 + mx))
        y0 = height - box_h - 8
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y0:y0 + box_h, x0:x0 + box_w] = 255
        masks.append(mask)

        burned = view.copy()
        burned[y0:y0 + box_h, x0:x0 + box_w] = 235
        subtitled.append(burned)
    return clean, subtitled, masks


def inject_regression(
    clean: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    kind: str,
    *,
    seed: int = 11,
) -> list[np.ndarray]:
    """Seed a known temporal defect into an otherwise perfect fill.

    ``kind`` is one of ``"none"``, ``"frozen"`` (the fill is pinned to the
    first frame instead of following the scene), ``"flicker"`` (the fill jumps
    in brightness between frames), or ``"leak"`` (the fill bleeds past its
    mask).
    """
    rng = np.random.RandomState(seed)
    out: list[np.ndarray] = []
    first = np.asarray(clean[0])
    for index, (frame, mask) in enumerate(zip(clean, masks)):
        result = np.asarray(frame).copy()
        binary = _binary(mask).astype(bool)
        if kind == "frozen":
            result[binary] = first[binary]
        elif kind == "flicker":
            shift = 60 if index % 2 else -60
            patch = result[binary].astype(np.int16) + shift
            result[binary] = np.clip(patch, 0, 255).astype(np.uint8)
        elif kind == "leak":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            grown = cv2.dilate(_binary(mask) * 255, kernel) > 0
            ring = grown & ~binary
            noise = rng.randint(0, 255, size=(int(ring.sum()), 3))
            result[ring] = noise.astype(np.uint8)
        elif kind != "none":
            raise ValueError(f"unknown regression kind: {kind!r}")
        out.append(result)
    return out


def run_temporal_regression_profile(
    *,
    thresholds: TemporalThresholds = DEFAULT_THRESHOLDS,
) -> dict:
    """Run the whole synthetic profile; used as release evidence.

    Each axis is exercised on its own with a perfect fill (which must pass) so
    the profile proves ordinary camera, background, and mask motion do not
    trip the fail bars.
    """
    cases = (
        ("static", {}),
        ("camera-motion", {"camera_motion": 3.0}),
        ("background-motion", {"background_motion": 9.0}),
        ("mask-motion", {"mask_motion": 5.0}),
        ("all-motion", {
            "camera_motion": 2.0, "background_motion": 7.0, "mask_motion": 4.0}),
    )
    results = []
    passed = True
    for label, kwargs in cases:
        clean, subtitled, masks = synthetic_clip(**kwargs)
        report = evaluate_temporal_profile(
            subtitled, clean, masks, thresholds=thresholds, label=label)
        passed = passed and bool(report["passed"])
        results.append(report)
    return {
        "schema": TEMPORAL_PROFILE_SCHEMA,
        "ran": True,
        "passed": passed,
        "cases": results,
        "failures": [
            f"{item['label']}: {reason}"
            for item in results for reason in item["failures"]
        ],
    }
