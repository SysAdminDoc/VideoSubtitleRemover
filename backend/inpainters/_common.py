"""Shared inpainter primitives: BaseInpainter ABC, mask conditioning,
Farneback warp helpers, the TBE primitive, and the scene-cut detector
cascade used by STTN / ProPainter / AUTO.
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BaseInpainter(ABC):
    """Abstract base class for inpainting models."""

    @abstractmethod
    def inpaint(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        """Inpaint the masked regions in the frames."""
        pass


def _cv2_inpaint(frame: np.ndarray, mask: np.ndarray, radius: int = 5,
                 method: int = cv2.INPAINT_TELEA) -> np.ndarray:
    """OpenCV inpainting fallback."""
    if mask.max() > 0:
        return cv2.inpaint(frame, mask, radius, method)
    return frame.copy()


def _feather_blend(original: np.ndarray, filled: np.ndarray,
                   mask: np.ndarray, feather_px: int = 4) -> np.ndarray:
    """Alpha-blend the inpainted `filled` result back onto `original`
    using a Gaussian-softened mask so the boundary of the removed
    region is seamless."""
    if feather_px <= 0 or mask.max() == 0:
        return filled
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
        x1 = max(0, x1 - padding); y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding); y2 = min(h, y2 + padding)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = lab[y1:y2, x1:x2].reshape(-1, 3).astype(np.int16)
        if roi.size == 0:
            continue
        L = roi[:, 0]
        low = roi[L < np.median(L)]
        high = roi[L >= np.median(L)]
        if low.size == 0 or high.size == 0:
            continue
        low_var = float(low.var())
        high_var = float(high.var())
        fg = low.mean(axis=0) if low_var < high_var else high.mean(axis=0)
        diff = roi - fg
        dist = np.sqrt((diff * diff).sum(axis=1))
        match = (dist < tolerance).reshape(y2 - y1, x2 - x1).astype(np.uint8) * 255
        out[y1:y2, x1:x2] = np.maximum(out[y1:y2, x1:x2], match)
    return out


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


# ---------------------------------------------------------------------------
# Farneback warp helpers + TBE primitive
# ---------------------------------------------------------------------------


def _farneback_winsize(h: int, w: int) -> int:
    """Pick a Farneback window size scaled by short edge."""
    short_edge = max(1, min(h, w))
    return int(max(9, min(33, short_edge // 24)))


def _warp_to_reference(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    h, w = src.shape[:2]
    winsize = _farneback_winsize(h, w)
    flow = cv2.calcOpticalFlowFarneback(
        ref_gray, src_gray, None,
        pyr_scale=0.5, levels=3, winsize=winsize, iterations=3,
        poly_n=7, poly_sigma=1.5, flags=0,
    )
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                  np.arange(h, dtype=np.float32))
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    return cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def _warp_mask_to_reference(src_mask: np.ndarray, src_frame: np.ndarray,
                              ref_frame: np.ndarray) -> np.ndarray:
    src_gray = cv2.cvtColor(src_frame, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    h, w = src_mask.shape[:2]
    winsize = _farneback_winsize(h, w)
    flow = cv2.calcOpticalFlowFarneback(
        ref_gray, src_gray, None,
        pyr_scale=0.5, levels=3, winsize=winsize, iterations=3,
        poly_n=7, poly_sigma=1.5, flags=0,
    )
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                  np.arange(h, dtype=np.float32))
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    return cv2.remap(src_mask, map_x, map_y, cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def _tbe_single_segment(frames: List[np.ndarray], masks: List[np.ndarray],
                         min_coverage: int, use_median: bool,
                         feather_px: int, edge_ring_px: int,
                         flow_warp: bool) -> List[np.ndarray]:
    """Aggregate one scene-contiguous segment via Temporal Background Exposure."""
    n = len(frames)
    if n == 0:
        return []
    if n == 1:
        filled = _cv2_inpaint(frames[0], masks[0], 7, cv2.INPAINT_NS)
        if edge_ring_px > 0:
            filled = _edge_ring_color_correct(frames[0], filled, masks[0], edge_ring_px)
        return [_feather_blend(frames[0], filled, masks[0], feather_px)]

    if flow_warp:
        ref_idx = n // 2
        ref_frame = frames[ref_idx]
        warped_frames: List[np.ndarray] = []
        warped_masks: List[np.ndarray] = []
        for i, (f, m) in enumerate(zip(frames, masks)):
            if i == ref_idx:
                warped_frames.append(f)
                warped_masks.append(m)
            else:
                try:
                    wf = _warp_to_reference(f, ref_frame)
                    wm = _warp_mask_to_reference(m, f, ref_frame)
                    warped_frames.append(wf)
                    warped_masks.append(wm)
                except Exception as exc:
                    logger.debug(f"Flow warp fell back for frame {i}: {exc}")
                    warped_frames.append(f)
                    warped_masks.append(m)
        agg_frames = warped_frames
        agg_masks = warped_masks
    else:
        agg_frames = list(frames)
        agg_masks = list(masks)

    frame_stack = np.stack(agg_frames, axis=0).astype(np.float32)
    mask_stack = np.stack(agg_masks, axis=0)
    unmasked = (mask_stack == 0)
    coverage = unmasked.sum(axis=0).astype(np.int32)

    if use_median and n <= 64:
        weighted = np.where(unmasked[..., None], frame_stack, np.nan)
        with np.errstate(all='ignore'):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="All-NaN slice encountered",
                    category=RuntimeWarning,
                )
                bg = np.nanmedian(weighted, axis=0)
        bg = np.nan_to_num(bg, nan=0.0)
    else:
        sum_vals = (frame_stack * unmasked[..., None]).sum(axis=0)
        count = np.maximum(coverage, 1).astype(np.float32)
        bg = sum_vals / count[..., None]
    bg = np.clip(bg, 0, 255).astype(np.uint8)

    results = []
    for t in range(n):
        frame = frames[t]
        mask = masks[t]
        if mask.max() == 0:
            results.append(frame.copy())
            continue

        if flow_warp and t != (n // 2):
            try:
                bg_for_t = _warp_to_reference(bg, frame)
            except Exception as exc:
                logger.debug(f"Flow back-warp fell back for frame {t}: {exc}")
                bg_for_t = bg
        else:
            bg_for_t = bg

        mask_bool = mask > 0
        has_exposure = mask_bool & (coverage >= min_coverage)
        no_exposure = mask_bool & (coverage < min_coverage)

        filled = frame.copy()
        if has_exposure.any():
            filled[has_exposure] = bg_for_t[has_exposure]

        if no_exposure.any():
            residual = np.zeros_like(mask)
            residual[no_exposure] = 255
            filled = _cv2_inpaint(filled, residual, 5, cv2.INPAINT_TELEA)

        if edge_ring_px > 0:
            filled = _edge_ring_color_correct(frame, filled, mask, edge_ring_px)
        results.append(_feather_blend(frame, filled, mask, feather_px))
    return results


def _temporal_background_expose(frames: List[np.ndarray], masks: List[np.ndarray],
                                 min_coverage: int = 3,
                                 use_median: bool = True,
                                 feather_px: int = 4,
                                 edge_ring_px: int = 2,
                                 flow_warp: bool = False,
                                 scene_cut_split: bool = True,
                                 scene_cut_threshold: float = 0.35,
                                 scene_cut_use_pyscenedetect: bool = False,
                                 scene_cut_use_transnetv2: bool = False) -> List[np.ndarray]:
    """Video-inpainting primitive: reconstruct masked pixels from
    temporally exposed neighbours. Optional scene-cut split + flow warp."""
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
