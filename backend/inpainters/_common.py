"""Shared inpainter primitives: BaseInpainter ABC, mask conditioning,
dense-flow warp helpers, the TBE primitive, and the scene-cut detector
cascade used by STTN / ProPainter / AUTO.
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import cv2
import numpy as np

from backend.temporal_profile import (
    GLOBAL_MOTION_MIN_INLIER_RATIO,
    estimate_global_motion_quality,
)

logger = logging.getLogger(__name__)

_TBE_MAD_K = 3.0
_TBE_MAD_MIN_TOLERANCE = 1.0
_TBE_MIN_SURVIVORS = 3
_POISSON_SEAM_DILATE_PX = 3
_GRAIN_RING_INNER_PX = 2
_GRAIN_RING_OUTER_PX = 6
_GRAIN_MIN_RING_PIXELS = 32
_GRAIN_MIN_STD = 1.0
_GRAIN_SEED = 0x47524149
_GRAIN_TEXTURE_SIGMA = 0.65
_GRAIN_DITHER_STD = 0.2
_TRANSLUCENCY_MIN_ALPHA = 0.15
_TRANSLUCENCY_MAX_ALPHA = 0.92
_TRANSLUCENCY_RESIDUAL_TOLERANCE = 2.0


class BaseInpainter(ABC):
    """Abstract base class for inpainting models."""

    @abstractmethod
    def inpaint(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        """Inpaint the masked regions in the frames."""
        pass

    @property
    def backend_name(self) -> str:
        """RM-147: which implementation actually ran, not the requested mode.

        Subclasses that can degrade (LaMa -> ONNX / OpenCV DNN / cv2) override
        this; the default reports the class name so every job records
        something concrete.
        """
        return type(self).__name__


def _cv2_inpaint(frame: np.ndarray, mask: np.ndarray, radius: int = 5,
                 method: int = cv2.INPAINT_TELEA) -> np.ndarray:
    """OpenCV inpainting fallback."""
    if mask.max() > 0:
        return cv2.inpaint(frame, mask, radius, method)
    return frame.copy()


_OOM_MARKERS = (
    "out of memory",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "cuda_error_out_of_memory",
    "failed to allocate memory",
    "hiperroroutofmemory",
    "bad_alloc",
)


def is_oom_error(exc: BaseException) -> bool:
    """Best-effort detection of a GPU/host out-of-memory failure.

    Covers torch.cuda.OutOfMemoryError, Python MemoryError, and the runtime
    error strings raised by torch, CUDA/cuBLAS/cuDNN, ROCm, and ONNX Runtime
    allocators. Kept string-based so it works whether or not torch is present.
    """
    if isinstance(exc, MemoryError):
        return True
    try:
        import torch
        oom_cls = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
        if oom_cls is not None and isinstance(exc, oom_cls):
            return True
    except Exception:
        pass
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _OOM_MARKERS)


def free_inference_memory() -> None:
    """Release cached GPU allocations and run a GC pass. Best-effort, no-raise."""
    import gc
    gc.collect()
    try:
        import torch
        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def _binarize_mask(mask: np.ndarray) -> np.ndarray:
    """Return a strict {0, 255} uint8 mask for neural inpaint model input.

    LaMa / MI-GAN models consume ``mask / 255.0`` as a binary inpaint
    indicator; feathering is applied *after* inpainting. A soft alpha
    matte (e.g. from MatAnyone refinement) carries intermediate values
    1-254 that would feed partial-strength hints and silently degrade the
    fill, so threshold at the midpoint. A properly dilated binary mask is
    left unchanged.
    """
    if mask.ndim == 3:
        mask = mask[..., 0]
    return np.where(mask >= 128, np.uint8(255), np.uint8(0))


def _feather_blend(original: np.ndarray, filled: np.ndarray,
                   mask: np.ndarray, feather_px: int = 4,
                   continuous: bool = False) -> np.ndarray:
    """Alpha-blend the inpainted `filled` result back onto `original`
    using a Gaussian-softened mask so the boundary of the removed
    region is seamless. Auto-dilate masks already contain a continuous
    distance-transform alpha and must not be blurred a second time."""
    if continuous and np.any((mask > 0) & (mask < 255)):
        soft = mask.astype(np.float32) / 255.0
        soft = soft[..., None]
        out = filled.astype(np.float32) * soft + original.astype(np.float32) * (1.0 - soft)
        return np.clip(out, 0, 255).astype(np.uint8)
    if feather_px <= 0 or mask.max() == 0:
        if filled.dtype == np.uint8:
            return filled
        return np.clip(filled, 0, 255).astype(np.uint8)
    k = feather_px * 2 + 1
    soft = cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32) / 255.0
    if soft.ndim == 2:
        soft = soft[..., None]
    out = filled.astype(np.float32) * soft + original.astype(np.float32) * (1.0 - soft)
    return np.clip(out, 0, 255).astype(np.uint8)


def _expand_mask_by_color(frame: np.ndarray, mask: np.ndarray,
                           boxes: List[Tuple[int, int, int, int]],
                           tolerance: int = 25,
                           padding: int = 4) -> np.ndarray:
    """Grow the mask to cover Lab-similar pixels inside each detected
    box. Catches serifs / drop shadows the OCR bbox clips."""
    if not boxes or mask.max() == 0:
        return mask
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    out = mask.copy()
    h, w = mask.shape[:2]
    for (x1, y1, x2, y2) in boxes:
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = lab[y1:y2, x1:x2].reshape(-1, 3).astype(np.int16)
        fg = _lab_two_cluster_foreground(roi)
        if fg is None:
            continue
        diff = roi - fg
        dist = np.sqrt((diff * diff).sum(axis=1))
        match = (dist < tolerance).reshape(y2 - y1, x2 - x1).astype(np.uint8) * 255
        out[y1:y2, x1:x2] = np.maximum(out[y1:y2, x1:x2], match)
    return out


def _lab_two_cluster_foreground(roi: np.ndarray) -> Optional[np.ndarray]:
    """Return the lower-variance Lab cluster used by colour-tune.

    The existing colour-tune mask grower uses a deterministic two-cluster
    split at the median lightness. Keeping that estimate in one helper lets
    the translucency detector use the same foreground endpoint instead of
    introducing a second, incompatible colour model.
    """
    values = np.asarray(roi)
    if values.ndim != 2 or values.shape[1] != 3 or values.size == 0:
        return None
    lightness = values[:, 0]
    midpoint = np.median(lightness)
    low = values[lightness < midpoint]
    high = values[lightness >= midpoint]
    if low.size == 0 or high.size == 0:
        return None
    low_var = float(low.var())
    high_var = float(high.var())
    return low.mean(axis=0) if low_var < high_var else high.mean(axis=0)


def _lab_to_bgr(color: np.ndarray) -> np.ndarray:
    value = np.clip(np.rint(np.asarray(color)), 0, 255).astype(np.uint8)
    return cv2.cvtColor(value.reshape(1, 1, 3), cv2.COLOR_LAB2BGR)[0, 0]


def _translucency_foreground_candidates(
    frame: np.ndarray,
    background: np.ndarray,
    component_mask: np.ndarray,
    *,
    padding: int = 4,
) -> list[np.ndarray]:
    """Build deterministic foreground candidates from the colour-tune split.

    A translucent overlay may not contain a pure foreground pixel. In that
    case the cluster is an observed mixture, so the same endpoint is also
    extrapolated from the local TBE background at conservative opacity priors.
    Candidates are scored by the closed-form fit in
    ``_unmix_translucent_regions``; opaque regions are rejected there.
    """
    ys, xs = np.where(component_mask > 0)
    if ys.size == 0:
        return []
    height, width = component_mask.shape[:2]
    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(width, int(xs.max()) + padding + 1)
    y2 = min(height, int(ys.max()) + padding + 1)
    roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2LAB)
    cluster = _lab_two_cluster_foreground(roi.reshape(-1, 3).astype(np.int16))
    if cluster is None:
        return []
    observed = _lab_to_bgr(cluster).astype(np.float32)
    pixels = component_mask > 0
    local_background = background[pixels].astype(np.float32)
    if local_background.size == 0:
        return [observed]
    background_reference = np.median(local_background, axis=0)
    direction = observed - background_reference
    candidates = [observed]
    if float(np.linalg.norm(direction)) >= 8.0:
        for prior in (0.35, 0.5, 0.65):
            candidates.append(
                np.clip(background_reference + direction / prior, 0, 255)
            )
        if float(direction.mean()) > 0:
            candidates.append(np.full(3, 255.0, dtype=np.float32))
        elif float(direction.mean()) < 0:
            candidates.append(np.zeros(3, dtype=np.float32))
    unique: list[np.ndarray] = []
    for candidate in candidates:
        if not any(np.allclose(candidate, prior, atol=0.5) for prior in unique):
            unique.append(candidate.astype(np.float32))
    return unique


def _unmix_translucent_regions(
    frame: np.ndarray,
    background: np.ndarray,
    filled: np.ndarray,
    mask: np.ndarray,
    eligible: np.ndarray,
    *,
    residual_tolerance: float = _TRANSLUCENCY_RESIDUAL_TOLERANCE,
) -> np.ndarray:
    """Recover fitted translucent regions with closed-form alpha unmixing.

    For each connected mask component, solve
    ``alpha = dot(I-B, F-B) / dot(F-B, F-B)`` and recover
    ``B = (I - alpha*F) / (1-alpha)`` where the region has a stable
    intermediate alpha and a low two-endpoint residual. ``B`` is already
    independently estimated by TBE, so the recovered result is blended with
    that endpoint to remain bounded near alpha=1. Regions that look opaque,
    have no temporal exposure, or fail the fit remain on the existing binary
    path and are logged as rejected.
    """
    if (
        frame is None
        or background is None
        or filled is None
        or mask is None
        or eligible is None
        or frame.ndim != 3
        or background.shape != frame.shape
        or filled.shape != frame.shape
        or mask.ndim != 2
        or eligible.shape != mask.shape
    ):
        return filled
    binary = _binarize_mask(mask)
    if binary.max() == 0:
        return filled
    component_count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    output = filled.copy()
    for component_index in range(1, component_count):
        component = labels == component_index
        pixel_count = int(component.sum())
        if pixel_count < 16:
            logger.debug(
                "Translucency region %d rejected: only %d pixels",
                component_index,
                pixel_count,
            )
            continue
        exposed = component & (np.asarray(eligible) > 0)
        if int(exposed.sum()) < 16:
            logger.debug(
                "Translucency region %d rejected: no temporal exposure",
                component_index,
            )
            continue
        candidates = _translucency_foreground_candidates(
            frame, background, component)
        if not candidates:
            logger.debug(
                "Translucency region %d rejected: no foreground cluster",
                component_index,
            )
            continue
        observed = frame[exposed].astype(np.float32)
        local_background = background[exposed].astype(np.float32)
        best = None
        for foreground in candidates:
            delta = foreground[None, :] - local_background
            denominator = np.sum(delta * delta, axis=1)
            valid = denominator >= 64.0
            if int(valid.sum()) < 16:
                continue
            observed_delta = observed - local_background
            with np.errstate(divide="ignore", invalid="ignore"):
                alpha = np.sum(observed_delta * delta, axis=1) / denominator
            alpha = np.clip(alpha, 0.0, 1.0)
            predicted = local_background + alpha[:, None] * delta
            residual = np.linalg.norm(observed - predicted, axis=1)
            valid &= np.isfinite(alpha) & np.isfinite(residual)
            if int(valid.sum()) < 16:
                continue
            alpha_valid = alpha[valid]
            residual_valid = residual[valid]
            intermediate = (
                (alpha_valid >= _TRANSLUCENCY_MIN_ALPHA)
                & (alpha_valid <= _TRANSLUCENCY_MAX_ALPHA)
            )
            intermediate_fraction = float(intermediate.mean())
            median_alpha = float(np.median(alpha_valid))
            median_residual = float(np.median(residual_valid))
            if (
                intermediate_fraction < 0.65
                or not (
                    _TRANSLUCENCY_MIN_ALPHA
                    <= median_alpha
                    <= _TRANSLUCENCY_MAX_ALPHA
                )
                or median_residual > residual_tolerance
            ):
                continue
            score = median_residual + abs(median_alpha - 0.5) * 2.0
            if best is None or score < best[0]:
                best = (
                    score,
                    foreground,
                    alpha,
                    valid,
                    median_alpha,
                    median_residual,
                )
        if best is None:
            logger.debug(
                "Translucency region %d rejected: endpoints do not fit",
                component_index,
            )
            continue
        _score, foreground, alpha, valid, median_alpha, median_residual = best
        safe_denominator = np.maximum(1.0 - alpha, 0.08)
        recovered = (
            observed - alpha[:, None] * foreground[None, :]
        ) / safe_denominator[:, None]
        # TBE is the independently measured clean endpoint. Averaging it with
        # the algebraic recovery damps quantisation noise without reintroducing
        # any subtitle pixels into the accepted region.
        recovered = 0.5 * recovered + 0.5 * local_background
        recovered = np.clip(recovered, 0, 255)
        indices = np.flatnonzero(exposed)
        accepted_indices = indices[valid]
        output.reshape(-1, 3)[
            np.flatnonzero(exposed)[valid]
        ] = recovered[valid].astype(np.uint8)
        logger.info(
            "Translucency region %d accepted: alpha=%.3f residual=%.3f pixels=%d",
            component_index,
            median_alpha,
            median_residual,
            int(accepted_indices.size),
        )
    return output


def _edge_ring_color_correct(original: np.ndarray, filled: np.ndarray,
                              mask: np.ndarray, ring_px: int = 2) -> np.ndarray:
    """Sample a thin ring just outside the mask in both original and
    filled, and shift the filled mask region by the mean delta to kill
    the colour seam on gradient backgrounds."""
    if filled is None or mask is None or ring_px <= 0:
        return filled
    if mask.size == 0 or mask.max() == 0:
        return filled
    mask_bool = mask > 0
    k = ring_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    dilated = cv2.dilate(mask, kernel, iterations=1) > 0
    ring = dilated & ~mask_bool
    ring_count = int(ring.sum())
    if ring_count < 16:
        return filled
    orig_mean = original[ring].astype(np.float32).mean(axis=0)
    fill_mean = filled[ring].astype(np.float32).mean(axis=0)
    delta = orig_mean - fill_mean
    if not np.all(np.isfinite(delta)):
        return filled
    if np.abs(delta).max() < 0.5:
        return filled
    out = filled.astype(np.float32)
    out[mask_bool] = np.clip(out[mask_bool] + delta, 0, 255)
    return out.astype(np.uint8)


def _poisson_seam_correct(
    original: np.ndarray,
    filled: np.ndarray,
    mask: np.ndarray,
    dilate_px: int = _POISSON_SEAM_DILATE_PX,
) -> np.ndarray:
    """Blend a filled region through a boundary-aware Poisson solve.

    The dilated mask is the solve domain, while only the original mask is
    copied back. This lets the solver see clean source and destination pixels
    on both sides of the seam without modifying the untouched ring. Masks
    touching the image edge and tiny regions are intentionally skipped because
    they do not provide a closed boundary for a stable solve.
    """
    if original is None or filled is None or mask is None:
        return filled
    if mask.size == 0 or mask.max() == 0:
        return filled
    if original.ndim != 3 or filled.shape != original.shape:
        return filled
    binary = _binarize_mask(mask)
    if binary.ndim != 2 or binary.shape != original.shape[:2]:
        return filled
    mask_bool = binary > 0
    height, width = binary.shape[:2]
    if (
        height < 3
        or width < 3
        or int(mask_bool.sum()) < 16
        or bool(mask_bool[0].any())
        or bool(mask_bool[-1].any())
        or bool(mask_bool[:, 0].any())
        or bool(mask_bool[:, -1].any())
    ):
        return filled

    radius = max(1, int(dilate_px))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    solve_mask = cv2.dilate(binary, kernel, iterations=1)
    ys, xs = np.where(solve_mask > 0)
    if ys.size == 0:
        return filled
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    if x1 <= 0 or y1 <= 0 or x2 >= width - 1 or y2 >= height - 1:
        return filled
    source_roi = filled[y1:y2 + 1, x1:x2 + 1]
    destination_roi = original[y1:y2 + 1, x1:x2 + 1]
    mask_roi = solve_mask[y1:y2 + 1, x1:x2 + 1]
    center = ((x2 - x1) // 2, (y2 - y1) // 2)
    try:
        cloned = cv2.seamlessClone(
            source_roi,
            destination_roi,
            mask_roi,
            center,
            cv2.NORMAL_CLONE,
        )
    except Exception as exc:
        logger.debug("Poisson seam correction fell back: %s", exc)
        return filled
    out = filled.copy()
    original_mask_roi = mask_bool[y1:y2 + 1, x1:x2 + 1]
    output_roi = out[y1:y2 + 1, x1:x2 + 1].copy()
    output_roi[original_mask_roi] = cloned[original_mask_roi]
    out[y1:y2 + 1, x1:x2 + 1] = output_roi
    return out


def _restore_mask_grain(
    original: np.ndarray,
    filled: np.ndarray,
    mask: np.ndarray,
    strength: float = 0.0,
    *,
    frame_index: int = 0,
) -> np.ndarray:
    """Re-synthesize local texture inside a filled mask.

    A high-pass residual from the unmasked ring estimates the nearby grain
    scale without importing subtitle pixels or large image edges. A fresh,
    lightly correlated noise field is generated for each frame, and a small
    high-pass dither is added before the shared float-to-uint8 feather blend.
    Clean or texture-poor rings return the original fill unchanged.
    """
    if original is None or filled is None or mask is None:
        return filled
    try:
        strength = float(strength)
    except (TypeError, ValueError):
        return filled
    if strength <= 0.0 or mask.size == 0 or mask.max() == 0:
        return filled
    if original.ndim != 3 or original.shape[2] < 3:
        return filled
    if filled.shape != original.shape:
        return filled
    binary = _binarize_mask(mask)
    if binary.ndim != 2 or binary.shape != original.shape[:2]:
        return filled
    mask_bool = binary > 0
    outer_size = _GRAIN_RING_OUTER_PX * 2 + 1
    inner_size = _GRAIN_RING_INNER_PX * 2 + 1
    outer_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (outer_size, outer_size))
    inner_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (inner_size, inner_size))
    outer = cv2.dilate(binary, outer_kernel, iterations=1) > 0
    inner = cv2.dilate(binary, inner_kernel, iterations=1) > 0
    ring = outer & ~inner
    if int(ring.sum()) < _GRAIN_MIN_RING_PIXELS:
        return filled

    source = original[..., :3].astype(np.float32)
    smooth = cv2.GaussianBlur(source, (0, 0), 1.0)
    residual = source - smooth
    ring_values = residual[ring]
    centered = ring_values - np.median(ring_values, axis=0)
    robust_std = (
        np.percentile(centered, 84, axis=0)
        - np.percentile(centered, 16, axis=0)
    ) / 2.0
    measured_std = np.std(centered, axis=0)
    grain_std = np.maximum(robust_std, measured_std)
    grain_std = np.where(grain_std >= _GRAIN_MIN_STD, grain_std, 0.0)
    target_std = np.minimum(grain_std, strength * 255.0)
    if float(target_std.max()) < 0.5:
        return filled

    height, width = mask_bool.shape
    try:
        frame_seed = _GRAIN_SEED + int(frame_index)
    except (TypeError, ValueError, OverflowError):
        frame_seed = _GRAIN_SEED
    rng = np.random.default_rng(frame_seed)
    noise = np.zeros((height, width, 3), dtype=np.float32)
    for channel in range(3):
        white = rng.standard_normal((height, width)).astype(np.float32)
        correlated = cv2.GaussianBlur(
            white, (0, 0), _GRAIN_TEXTURE_SIGMA)
        correlated -= float(correlated.mean())
        correlated_std = float(correlated.std())
        if correlated_std > 1e-6:
            noise[..., channel] = (
                correlated * (float(target_std[channel]) / correlated_std)
            )

    white = rng.random((height, width), dtype=np.float32)
    blue_noise = white - cv2.GaussianBlur(white, (0, 0), 0.8)
    blue_std = float(blue_noise.std())
    if blue_std > 1e-6:
        blue_noise = blue_noise * (_GRAIN_DITHER_STD / blue_std)
    else:
        blue_noise.fill(0.0)

    out = filled.astype(np.float32)
    out[mask_bool] += noise[mask_bool] + blue_noise[mask_bool, None]
    return out


def apply_finishing(original, filled, masks, config=None, *,
                    edge_ring: bool = True,
                    feather_px: Optional[int] = None,
                    edge_ring_px: Optional[int] = None):
    """Single post-inpaint finishing step shared by every inpainter family.

    Applies the opt-in Poisson seam correction or shared edge-ring colour match,
    restores local grain when configured, and then runs the feather blend at
    each mask boundary. Poisson mode uses the dilated solve domain but only
    replaces pixels inside the original mask; edge-ring correction is skipped
    when Poisson mode is enabled.
    ``feather_px``/``edge_ring_px`` default to ``config.mask_feather_px`` and
    ``config.edge_ring_px``. When ``config`` is ``None`` and no explicit
    ``feather_px`` is given, the filled frames pass through unchanged so a
    session without config never crashes.

    Centralizing this here keeps the ONNX, diffusion, and built-in backends on
    identical boundary handling instead of each re-implementing the loop.
    """
    if feather_px is None:
        if config is None:
            return list(filled)
        feather_px = getattr(config, "mask_feather_px", 4)
    if edge_ring_px is None:
        edge_ring_px = (
            getattr(config, "edge_ring_px", 2) if config is not None else 0)
    poisson_seam = bool(
        getattr(config, "poisson_seam_enable", False)
        if config is not None else False
    )
    grain_strength = (
        getattr(config, "film_grain_strength", 0.0)
        if config is not None else 0.0
    )
    out = []
    for index, (f, r, m) in enumerate(zip(original, filled, masks)):
        if poisson_seam:
            r = _poisson_seam_correct(f, r, m)
        elif edge_ring and edge_ring_px > 0 and m.max() > 0:
            r = _edge_ring_color_correct(f, r, m, edge_ring_px)
        r = _restore_mask_grain(
            f, r, m, grain_strength, frame_index=index)
        out.append(_feather_blend(
            f,
            r,
            m,
            feather_px,
            continuous=bool(
                getattr(config, "auto_dilate_enable", False)
                if config is not None else False
            ),
        ))
    return out


# ---------------------------------------------------------------------------
# Scene-cut detector cascade
# ---------------------------------------------------------------------------


def _detect_scene_cuts_pyscenedetect(frames: List[np.ndarray]) -> Optional[List[int]]:
    """RM-32: optional PySceneDetect-backed scene cut detection."""
    try:
        from scenedetect import SceneManager  # type: ignore
        from scenedetect.detectors import AdaptiveDetector  # type: ignore
    except ImportError:
        return None
    try:
        sm = SceneManager()
        sm.add_detector(AdaptiveDetector())
        for i, f in enumerate(frames):
            sm._process_frame(i, f, callback=None)  # type: ignore[attr-defined]
        scene_list = sm.get_scene_list()
        if not scene_list:
            return [0]
        cuts = [0]
        for entry, _exit in scene_list:
            idx = int(entry.get_frames())
            if idx > 0 and idx < len(frames):
                cuts.append(idx)
        return sorted(set(cuts))
    except Exception as exc:
        logger.debug(f"PySceneDetect path failed: {exc}")
        return None


def _detect_scene_cuts(frames: List[np.ndarray],
                        threshold: float = 0.35,
                        prefer_pyscenedetect: bool = False,
                        prefer_transnetv2: bool = False) -> List[int]:
    """Cascade: TransNetV2 (RM-21) -> PySceneDetect (RM-32) -> histogram."""
    if len(frames) <= 1:
        return [0]
    if prefer_transnetv2:
        try:
            from backend.preprocess import transnetv2_scene_cuts
            tn = transnetv2_scene_cuts(frames)
            if tn is not None:
                return tn
        except Exception as exc:
            logger.debug(f"TransNetV2 cascade failed: {exc}")
    if prefer_pyscenedetect:
        psd = _detect_scene_cuts_pyscenedetect(frames)
        if psd is not None:
            return psd
    cuts = [0]
    prev_hist = None
    for i, f in enumerate(frames):
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)
        if prev_hist is not None:
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if corr < (1.0 - threshold):
                cuts.append(i)
        prev_hist = hist
    return cuts


def stabilize_masks_rolling_union(
    masks: List[np.ndarray],
    scene_starts: Optional[List[int]] = None,
    window: int = 3,
) -> List[np.ndarray]:
    """Retain recently observed target pixels via a scene-bounded mask union.

    For each frame, OR together the masks in a short trailing window that never
    crosses a scene boundary, so a single-frame OCR miss or a moving/dissolving
    overlay keeps the pixels its neighbours saw. Resetting at every scene start
    guarantees a mask is never carried into an adjacent scene. Pure function --
    the caller decides when it is safe to apply (e.g. only in automatic
    full-frame detection, never with user-fixed timed regions).
    """
    if not masks or window <= 1:
        return masks
    import bisect
    n = len(masks)
    starts = sorted({int(s) for s in (scene_starts or [0]) if 0 <= int(s) < n})
    if not starts or starts[0] != 0:
        starts = [0] + starts
    out: List[np.ndarray] = []
    for i in range(n):
        scene_start = starts[bisect.bisect_right(starts, i) - 1]
        lo = max(scene_start, i - window + 1)
        acc = masks[i].copy()
        for j in range(lo, i):
            if masks[j] is not None and masks[j].shape == acc.shape:
                acc = cv2.bitwise_or(acc, masks[j])
        out.append(acc)
    return out


# ---------------------------------------------------------------------------
# Dense optical-flow warp helpers + TBE primitive
# ---------------------------------------------------------------------------


def _farneback_winsize(h: int, w: int) -> int:
    """Pick a Farneback window size scaled by short edge."""
    short_edge = max(1, min(h, w))
    return int(max(9, min(33, short_edge // 24)))


def _calc_farneback_flow(
    ref_gray: np.ndarray,
    src_gray: np.ndarray,
) -> np.ndarray:
    h, w = ref_gray.shape[:2]
    winsize = _farneback_winsize(h, w)
    return cv2.calcOpticalFlowFarneback(
        ref_gray, src_gray, None,
        pyr_scale=0.5, levels=3, winsize=winsize, iterations=3,
        poly_n=7, poly_sigma=1.5, flags=0,
    )


def _calc_dense_flow(
    ref_gray: np.ndarray,
    src_gray: np.ndarray,
    flow_estimator: str = "dis",
) -> np.ndarray:
    """Estimate flow from reference coordinates into source coordinates.

    DIS is the default because its FAST preset handles discontinuities without
    adding a dependency. Farneback remains an explicit option and is also the
    closed-loop fallback when a build lacks DIS or DIS fails at runtime.
    """
    estimator = str(flow_estimator or "dis").strip().lower()
    if estimator not in {"dis", "farneback"}:
        logger.debug(
            "Unknown optical-flow estimator %r; using DIS",
            flow_estimator,
        )
        estimator = "dis"

    if estimator == "dis":
        factory = getattr(cv2, "DISOpticalFlow_create", None)
        preset = getattr(cv2, "DISOPTICAL_FLOW_PRESET_FAST", None)
        if callable(factory) and preset is not None:
            try:
                flow = factory(preset).calc(ref_gray, src_gray, None)
                if flow is None or flow.shape[:2] != ref_gray.shape[:2]:
                    raise ValueError("DIS returned an invalid flow shape")
                return flow
            except Exception as exc:
                logger.debug(
                    "DIS optical flow failed; falling back to Farneback: %s",
                    exc,
                )
        else:
            logger.debug(
                "DIS optical flow is unavailable; falling back to Farneback"
            )

    return _calc_farneback_flow(ref_gray, src_gray)


def _warp_to_reference(
    src: np.ndarray,
    ref: np.ndarray,
    flow_estimator: str = "dis",
) -> np.ndarray:
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    h, w = src.shape[:2]
    flow = _calc_dense_flow(ref_gray, src_gray, flow_estimator)
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                  np.arange(h, dtype=np.float32))
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    return cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def _warp_mask_to_reference(
    src_mask: np.ndarray,
    src_frame: np.ndarray,
    ref_frame: np.ndarray,
    flow_estimator: str = "dis",
) -> np.ndarray:
    src_gray = cv2.cvtColor(src_frame, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    h, w = src_mask.shape[:2]
    flow = _calc_dense_flow(ref_gray, src_gray, flow_estimator)
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                  np.arange(h, dtype=np.float32))
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    return cv2.remap(src_mask, map_x, map_y, cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def _warp_affine(
    source: np.ndarray,
    matrix: np.ndarray,
    *,
    interpolation: int,
    border_mode: int,
    border_value: int = 0,
) -> np.ndarray:
    """Warp a frame or mask with a source-to-reference affine matrix."""
    height, width = source.shape[:2]
    return cv2.warpAffine(
        source,
        matrix,
        (width, height),
        flags=interpolation,
        borderMode=border_mode,
        borderValue=border_value,
    )


def _identity_affine() -> np.ndarray:
    return np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def _tbe_aggregate_background(
    frame_stack: np.ndarray,
    unmasked: np.ndarray,
    use_median: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a robust background and return it with the exposure coverage.

    The median/MAD pass identifies per-channel temporal outliers. Surviving
    samples are averaged so stable detail is retained instead of forcing the
    final background to a median. Sparse pixels use the historical median for
    batches up to 64 frames and mean for larger batches, which keeps the old
    low-coverage behavior intact.
    """
    n = frame_stack.shape[0]
    coverage = unmasked.sum(axis=0).astype(np.int32)
    weighted = np.where(unmasked[..., None], frame_stack, np.nan)
    with np.errstate(all="ignore"):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="All-NaN slice encountered",
                category=RuntimeWarning,
            )
            median = np.nanmedian(weighted, axis=0)
            mad = np.nanmedian(
                np.abs(weighted - median[None, ...]), axis=0)
    median = np.nan_to_num(median, nan=0.0)
    mad = np.nan_to_num(mad, nan=0.0)

    if use_median and n <= 64:
        legacy_bg = median
    else:
        sum_vals = (frame_stack * unmasked[..., None]).sum(axis=0)
        count = np.maximum(coverage, 1).astype(np.float32)
        legacy_bg = sum_vals / count[..., None]

    tolerance = np.maximum(_TBE_MAD_K * mad, _TBE_MAD_MIN_TOLERANCE)
    survivors = unmasked[..., None] & (
        np.abs(frame_stack - median[None, ...]) <= tolerance
    )
    survivor_count = survivors.sum(axis=0).astype(np.int32)
    survivor_sum = np.where(survivors, frame_stack, 0.0).sum(axis=0)
    survivor_denominator = np.maximum(survivor_count, 1).astype(np.float32)
    robust_bg = survivor_sum / survivor_denominator
    enough_survivors = survivor_count >= _TBE_MIN_SURVIVORS
    bg = np.where(enough_survivors, robust_bg, legacy_bg)
    return bg, coverage


def _tbe_single_segment(frames: List[np.ndarray], masks: List[np.ndarray],
                         min_coverage: int, use_median: bool,
                         feather_px: int, edge_ring_px: int,
                         flow_warp: bool,
                         global_motion_align: bool = True,
                         flow_estimator: str = "dis",
                         grain_strength: float = 0.0,
                         grain_frame_offset: int = 0,
                         translucency_enable: bool = True) -> List[np.ndarray]:
    """Aggregate one scene-contiguous segment via Temporal Background Exposure.

    Global affine registration puts every frame and mask in the middle-frame
    reference coordinates before aggregation. The optional dense DIS/Farneback
    pass then refines residual parallax on those already registered frames.
    """
    n = len(frames)
    if n == 0:
        return []
    if n == 1:
        filled = _cv2_inpaint(frames[0], masks[0], 7, cv2.INPAINT_NS)
        if edge_ring_px > 0:
            filled = _edge_ring_color_correct(frames[0], filled, masks[0], edge_ring_px)
        filled = _restore_mask_grain(
            frames[0], filled, masks[0], grain_strength,
            frame_index=grain_frame_offset)
        return [_feather_blend(frames[0], filled, masks[0], feather_px)]

    ref_idx = n // 2
    ref_frame = frames[ref_idx]
    frame_to_ref: List[Optional[np.ndarray]] = [None] * n
    aligned_frames: List[np.ndarray] = []
    aligned_masks: List[np.ndarray] = []
    identity = _identity_affine()
    for i, (frame, mask) in enumerate(zip(frames, masks)):
        matrix = None
        if global_motion_align and i != ref_idx:
            try:
                matrix, inlier_ratio = estimate_global_motion_quality(
                    frame, ref_frame)
                if matrix is None:
                    logger.debug(
                        "TBE global-motion alignment frame %d used identity: "
                        "no affine fit (RANSAC inlier ratio %.3f)",
                        i, inlier_ratio,
                    )
                    matrix = None
                elif inlier_ratio < GLOBAL_MOTION_MIN_INLIER_RATIO:
                    logger.debug(
                        "TBE global-motion alignment frame %d used identity: "
                        "RANSAC inlier ratio %.3f below %.3f",
                        i, inlier_ratio, GLOBAL_MOTION_MIN_INLIER_RATIO,
                    )
                    matrix = None
            except Exception as exc:
                logger.debug(
                    "TBE global-motion alignment frame %d used identity: %s",
                    i, exc,
                )
                matrix = None
        elif i == ref_idx:
            matrix = identity

        frame_to_ref[i] = matrix
        if matrix is None:
            aligned_frames.append(frame)
            aligned_masks.append(mask)
        else:
            aligned_frames.append(_warp_affine(
                frame, matrix, interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REPLICATE,
            ))
            aligned_masks.append(_warp_affine(
                mask, matrix, interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT, border_value=255,
            ))

    if flow_warp:
        warped_frames: List[np.ndarray] = []
        warped_masks: List[np.ndarray] = []
        for i, (frame, mask) in enumerate(zip(aligned_frames, aligned_masks)):
            if i == ref_idx:
                warped_frames.append(frame)
                warped_masks.append(mask)
                continue
            try:
                wf = _warp_to_reference(
                    frame, ref_frame, flow_estimator)
                wm = _warp_mask_to_reference(
                    mask, frame, ref_frame, flow_estimator)
                warped_frames.append(wf)
                warped_masks.append(wm)
            except Exception as exc:
                logger.debug("Flow warp fell back for frame %d: %s", i, exc)
                warped_frames.append(frame)
                warped_masks.append(mask)
        agg_frames = warped_frames
        agg_masks = warped_masks
    else:
        agg_frames = aligned_frames
        agg_masks = aligned_masks

    frame_stack = np.stack(agg_frames, axis=0).astype(np.float32)
    mask_stack = np.stack(agg_masks, axis=0)
    unmasked = (mask_stack == 0)
    bg, coverage = _tbe_aggregate_background(
        frame_stack, unmasked, use_median)
    bg = np.clip(bg, 0, 255).astype(np.uint8)

    results = []
    for t in range(n):
        frame = frames[t]
        mask = masks[t]
        if mask.max() == 0:
            results.append(frame.copy())
            continue

        matrix = frame_to_ref[t]
        if flow_warp and t != ref_idx:
            try:
                bg_for_t = _warp_to_reference(bg, frame)
            except Exception as exc:
                logger.debug(f"Flow back-warp fell back for frame {t}: {exc}")
                bg_for_t = bg
        elif matrix is not None and t != ref_idx:
            try:
                bg_for_t = _warp_affine(
                    bg,
                    cv2.invertAffineTransform(matrix),
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_REPLICATE,
                )
            except Exception as exc:
                logger.debug("Global-motion back-warp fell back for frame %d: %s", t, exc)
                bg_for_t = bg
        else:
            bg_for_t = bg

        mask_bool = mask > 0
        coverage_for_frame = coverage
        if matrix is not None and t != ref_idx:
            try:
                coverage_for_frame = _warp_affine(
                    coverage,
                    cv2.invertAffineTransform(matrix),
                    interpolation=cv2.INTER_NEAREST,
                    border_mode=cv2.BORDER_CONSTANT,
                )
            except Exception as exc:
                logger.debug(
                    "Global-motion coverage back-warp fell back for frame %d: %s",
                    t, exc,
                )
        has_exposure = mask_bool & (coverage_for_frame >= min_coverage)
        no_exposure = mask_bool & (coverage_for_frame < min_coverage)

        filled = frame.copy()
        if has_exposure.any():
            filled[has_exposure] = bg_for_t[has_exposure]

        if no_exposure.any():
            residual = np.zeros_like(mask)
            residual[no_exposure] = 255
            filled = _cv2_inpaint(filled, residual, 5, cv2.INPAINT_TELEA)

        if translucency_enable and has_exposure.any():
            filled = _unmix_translucent_regions(
                frame,
                bg_for_t,
                filled,
                mask,
                has_exposure,
            )

        if edge_ring_px > 0:
            filled = _edge_ring_color_correct(frame, filled, mask, edge_ring_px)
        filled = _restore_mask_grain(
            frame, filled, mask, grain_strength,
            frame_index=grain_frame_offset + t)
        results.append(_feather_blend(frame, filled, mask, feather_px))
    return results


def _temporal_background_expose(frames: List[np.ndarray], masks: List[np.ndarray],
                                 min_coverage: int = 3,
                                 use_median: bool = True,
                                 feather_px: int = 4,
                                 edge_ring_px: int = 2,
                                 flow_warp: bool = False,
                                 global_motion_align: bool = True,
                                 scene_cut_split: bool = True,
                                 scene_cut_threshold: float = 0.35,
                                 scene_cut_use_pyscenedetect: bool = False,
                                 scene_cut_use_transnetv2: bool = False,
                                 flow_estimator: str = "dis",
                                 grain_strength: float = 0.0,
                                 translucency_enable: bool = True) -> List[np.ndarray]:
    """Video-inpainting primitive: reconstruct masked pixels from
    temporally exposed neighbours with optional scene splitting, global
    alignment, and residual flow refinement."""
    if not scene_cut_split or len(frames) <= 1:
        segments = [(0, len(frames))]
    else:
        cuts = _detect_scene_cuts(
            frames, scene_cut_threshold,
            prefer_pyscenedetect=scene_cut_use_pyscenedetect,
            prefer_transnetv2=scene_cut_use_transnetv2,
        )
        segments = []
        for i, start in enumerate(cuts):
            end = cuts[i + 1] if i + 1 < len(cuts) else len(frames)
            segments.append((start, end))

    out: List[np.ndarray] = []
    for start, end in segments:
        sub_frames = frames[start:end]
        sub_masks = masks[start:end]
        out.extend(_tbe_single_segment(
            sub_frames, sub_masks,
            min_coverage=min_coverage,
            use_median=use_median,
            feather_px=feather_px,
            edge_ring_px=edge_ring_px,
            flow_warp=flow_warp,
            global_motion_align=global_motion_align,
            flow_estimator=flow_estimator,
            grain_strength=grain_strength,
            grain_frame_offset=start,
            translucency_enable=translucency_enable,
        ))
    return out


def _temporal_smooth_inpainted(
    frames: List[np.ndarray],
    masks: List[np.ndarray],
    radius: int = 2,
    scene_cuts: Optional[List[int]] = None,
) -> List[np.ndarray]:
    """Weighted-average blend of the inpainted region across a sliding
    window of 2*radius+1 frames. Only the masked pixels are blended;
    unmasked pixels are untouched. Scene-cut boundaries gate the window
    so no cross-scene ghosting occurs."""
    n = len(frames)
    if n <= 1 or radius <= 0:
        return list(frames)
    cut_set = set(scene_cuts) if scene_cuts else set()
    out: List[np.ndarray] = []
    for i in range(n):
        mask = masks[i]
        if mask.max() == 0:
            out.append(frames[i].copy())
            continue
        weights = np.zeros(mask.shape, dtype=np.float32)
        accum = np.zeros_like(frames[i], dtype=np.float32)
        for j in range(max(0, i - radius), min(n, i + radius + 1)):
            if j != i:
                crosses_cut = False
                lo, hi = min(i, j), max(i, j)
                for c in range(lo + 1, hi + 1):
                    if c in cut_set:
                        crosses_cut = True
                        break
                if crosses_cut:
                    continue
            dist = abs(i - j)
            w = 1.0 / (1.0 + dist)
            m_j = masks[j].astype(np.float32) / 255.0
            combined = m_j * (mask.astype(np.float32) / 255.0)
            weights += combined * w
            accum += frames[j].astype(np.float32) * combined[..., None] * w
        result = frames[i].copy()
        valid = weights > 0
        if valid.any():
            safe_w = np.maximum(weights, 1e-6)
            for c in range(3):
                result[:, :, c] = np.where(
                    valid,
                    (accum[:, :, c] / safe_w).clip(0, 255),
                    frames[i][:, :, c],
                )
        out.append(result.astype(np.uint8))
    return out
