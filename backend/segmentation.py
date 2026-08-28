"""Opt-in mask-refinement adapters.

RM-66 SAM 2 mask refinement -- take an already-detected subtitle bbox,
promote it to a clean text-shaped mask via SAM 2 prompted segmentation.
Eliminates the aggressive dilation we currently need to catch serifs
and drop shadows.

RM-67 SAM 3 text-prompt segmentation -- one-click "segment all
burned-in text" without any bounding boxes. SAM 3 accepts natural-
language prompts.

RM-68 MatAnyone 2 -- video matting; alternative mask generator for
thin moving subtitle lines that OCR + SAM both struggle with.

RM-69 CoTracker3 -- point tracking helper; lighter than SAM 2 memory.
Useful for confirming a karaoke caret stays on the same line across a
clip without engaging SAM's memory-propagation cost.

Each adapter imports lazily. A requested adapter that cannot execute
raises a classified error instead of reporting unchanged input as a
successful refinement. Mask refinement runs AFTER the OCR cascade and
BEFORE _create_mask widens the boxes.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from backend.execution_provenance import (
    FAILURE_DEPENDENCY_MISSING,
    FAILURE_INITIALIZATION,
    FAILURE_OUTPUT_INVALID,
    FAILURE_POLICY_BLOCKED,
    FAILURE_RUNTIME,
    RequestedStageError,
)

from backend.remote_model_policy import resolve_remote_model_source
from backend.safe_image import safe_imread, safe_imwrite

logger = logging.getLogger(__name__)


AUTO_DILATE_MAX_PX = 20


def _auto_dilate_roi(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    max_radius: int,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return Lab contrast, box gate, and context ring for one OCR box."""
    if not isinstance(frame, np.ndarray) or frame.ndim != 3:
        return None
    height, width = frame.shape[:2]
    clipped = _clip_box(box, width, height)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    context = max(4, min(32, int(max_radius) + 4))
    rx1 = max(0, x1 - context)
    ry1 = max(0, y1 - context)
    rx2 = min(width, x2 + context)
    ry2 = min(height, y2 + context)
    roi = frame[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return None
    try:
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    except (cv2.error, ValueError):
        return None
    box_gate = np.zeros(lab.shape[:2], dtype=bool)
    box_gate[y1 - ry1:y2 - ry1, x1 - rx1:x2 - rx1] = True
    ring = ~box_gate
    if int(ring.sum()) < 16:
        return None
    background = np.median(lab[ring], axis=0)
    contrast = np.linalg.norm(lab - background, axis=2)
    return contrast, box_gate, ring


def estimate_auto_dilation_radius(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    max_radius: int = AUTO_DILATE_MAX_PX,
) -> int:
    """Estimate an outlined/drop-shadow halo around one detected text box.

    The detected glyph is separated from its local background in Lab space at
    two thresholds.  The distance from the high-contrast core to the softer
    outer component is the measured intensity falloff, rather than a fixed
    confidence heuristic.  Invalid or ambiguous boxes deliberately return
    zero so the caller can keep the existing rectangular mask path.
    """
    try:
        limit = max(0, min(AUTO_DILATE_MAX_PX, int(max_radius)))
    except (TypeError, ValueError, OverflowError):
        limit = AUTO_DILATE_MAX_PX
    if limit <= 0:
        return 0
    prepared = _auto_dilate_roi(frame, box, limit)
    if prepared is None:
        return 0
    contrast, box_gate, ring = prepared
    inside = contrast[box_gate]
    outside = contrast[ring]
    finite_inside = inside[np.isfinite(inside)]
    finite_outside = outside[np.isfinite(outside)]
    if finite_inside.size < 4 or finite_outside.size < 16:
        return 0
    peak = float(np.percentile(finite_inside, 99.5))
    if peak < 4.0:
        return 0
    background_mad = float(
        np.median(np.abs(finite_outside - np.median(finite_outside)))
    )
    low = max(4.0, background_mad * 3.0, peak * 0.10)
    low = min(low, peak * 0.45)
    high = min(max(low + 5.0, peak * 0.55), peak * 0.90)
    if high <= low:
        return 0

    outer = contrast >= low
    core = (contrast >= high) & box_gate
    if int(core.sum()) < 4:
        return 0
    # Keep only outer components that actually touch a glyph core. This
    # rejects textured-background speckles in the generous context window.
    near_core = cv2.dilate(
        core.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    ) > 0
    component_count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(
        outer.astype(np.uint8), connectivity=8)
    halo = np.zeros_like(outer)
    for component_index in range(1, component_count):
        component = labels == component_index
        if bool(np.any(component & near_core)):
            halo |= component
    halo &= ~core
    if not np.any(halo):
        return 0
    distance = cv2.distanceTransform(
        (~core).astype(np.uint8), cv2.DIST_L2, 3)
    distances = distance[halo]
    if distances.size == 0:
        return 0
    measured = int(np.ceil(float(np.percentile(distances, 90))))
    return max(0, min(limit, measured))


def soft_dilate_mask(
    binary_mask: np.ndarray,
    radius: int,
) -> np.ndarray:
    """Grow a binary mask with a continuous distance-transform edge."""
    if not isinstance(binary_mask, np.ndarray) or binary_mask.ndim != 2:
        return binary_mask
    binary = (binary_mask > 0).astype(np.uint8)
    if binary.size == 0 or not np.any(binary):
        return np.zeros_like(binary_mask, dtype=np.uint8)
    try:
        radius = max(0, min(AUTO_DILATE_MAX_PX, int(radius)))
    except (TypeError, ValueError, OverflowError):
        radius = 0
    if radius <= 0:
        return np.where(binary > 0, np.uint8(255), np.uint8(0))
    distance = cv2.distanceTransform((binary == 0).astype(np.uint8), cv2.DIST_L2, 3)
    alpha = np.zeros(distance.shape, dtype=np.float32)
    alpha[binary > 0] = 1.0
    outside = binary == 0
    alpha[outside] = np.clip(
        (radius + 0.5 - distance[outside]) / (radius + 0.5),
        0.0,
        1.0,
    )
    return np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)


def _env_set(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# RM-66 -- SAM 2 mask refinement
# ---------------------------------------------------------------------------


_SAM2_STATE: dict = {"probed": False, "predictor": None, "load_error": None}


def _clip_box(box: Tuple[int, int, int, int],
              width: int,
              height: int) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width, int(x1)))
    x2 = max(0, min(width, int(x2)))
    y1 = max(0, min(height, int(y1)))
    y2 = max(0, min(height, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _positive_point_for_box(base_mask: np.ndarray,
                            box: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    region = base_mask[y1:y2, x1:x2]
    ys, xs = np.where(region > 0)
    if xs.size and ys.size:
        cx = float(x1 + np.median(xs))
        cy = float(y1 + np.median(ys))
    else:
        cx = float((x1 + x2) / 2.0)
        cy = float((y1 + y2) / 2.0)
    return np.array([[cx, cy]], dtype=np.float32)


def _maybe_load_sam2(device: str):
    if _SAM2_STATE["probed"]:
        return _SAM2_STATE["predictor"]
    _SAM2_STATE["probed"] = True
    _SAM2_STATE["load_error"] = None
    weight_path = os.environ.get("VSR_SAM2_CHECKPOINT", "")
    config_path = os.environ.get("VSR_SAM2_CONFIG", "")
    if not weight_path:
        logger.info(
            "SAM 2 refinement opt-in: set VSR_SAM2_CHECKPOINT (and "
            "VSR_SAM2_CONFIG) to enable. See facebookresearch/sam2."
        )
        _SAM2_STATE["load_error"] = RequestedStageError(
            stage="sam2",
            requested_implementation="sam2",
            failure_class=FAILURE_DEPENDENCY_MISSING,
            detail="VSR_SAM2_CHECKPOINT is not configured",
            recovery_hint=(
                "Install SAM 2 and set VSR_SAM2_CHECKPOINT plus "
                "VSR_SAM2_CONFIG, then retry or disable SAM 2 refinement."
            ),
        )
        return None
    try:
        from sam2.build_sam import build_sam2  # type: ignore
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
    except ImportError as exc:
        logger.info(
            "sam2 package not importable; install via "
            "`pip install git+https://github.com/facebookresearch/sam2`."
        )
        _SAM2_STATE["load_error"] = RequestedStageError(
            stage="sam2",
            requested_implementation="sam2",
            failure_class=FAILURE_DEPENDENCY_MISSING,
            detail=str(exc),
            recovery_hint=(
                "Install SAM 2 and its reviewed checkpoint, then retry or "
                "disable SAM 2 refinement."
            ),
            cause=exc,
        )
        return None
    try:
        model = build_sam2(config_path, weight_path)
        predictor = SAM2ImagePredictor(model)
        _SAM2_STATE["predictor"] = predictor
        return predictor
    except Exception as exc:
        logger.warning(f"SAM 2 load failed: {exc}")
        _SAM2_STATE["load_error"] = RequestedStageError(
            stage="sam2",
            requested_implementation="sam2",
            actual_implementation="sam2",
            provider="sam2",
            failure_class=FAILURE_INITIALIZATION,
            detail=str(exc),
            recovery_hint=(
                "Verify the SAM 2 checkpoint and runtime compatibility, then "
                "retry or disable SAM 2 refinement."
            ),
            cause=exc,
        )
        return None


def refine_mask_with_sam2(frame: np.ndarray,
                          boxes: List[Tuple[int, int, int, int]],
                          base_mask: np.ndarray,
                          device: str = "cpu") -> np.ndarray:
    """RM-66: replace each axis-aligned box in `base_mask` with the
    SAM 2 segmentation prompted by that box. Pixels outside the
    detected boxes carry over from `base_mask` so callers can compose
    SAM-refined regions on top of OpenCV detections.
    """
    if not boxes:
        return base_mask
    predictor = _maybe_load_sam2(device)
    if predictor is None:
        load_error = _SAM2_STATE.get("load_error")
        if isinstance(load_error, RequestedStageError):
            raise load_error
        raise RequestedStageError(
            stage="sam2",
            requested_implementation="sam2",
            failure_class=FAILURE_DEPENDENCY_MISSING,
            detail="SAM 2 checkpoint, configuration, or package is unavailable",
            recovery_hint=(
                "Install SAM 2 and set VSR_SAM2_CHECKPOINT plus "
                "VSR_SAM2_CONFIG, then retry or disable SAM 2 refinement."
            ),
        )
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb)
        refined = base_mask.copy()
        height, width = base_mask.shape[:2]
        clipped_boxes = [
            clipped for raw_box in boxes
            if (clipped := _clip_box(raw_box, width, height)) is not None
        ]
        # Clear all prompted regions before adding any predictor output. If a
        # later box overlaps an earlier one, clearing inside the loop would
        # erase the earlier SAM mask and make the result order-dependent.
        for x1, y1, x2, y2 in clipped_boxes:
            refined[y1:y2, x1:x2] = 0
        for clipped in clipped_boxes:
            x1, y1, x2, y2 = clipped
            box_t = np.array([x1, y1, x2, y2], dtype=np.float32)[None, :]
            point = _positive_point_for_box(base_mask, clipped)
            labels = np.array([1], dtype=np.int32)
            try:
                masks, _scores, _logits = predictor.predict(
                    point_coords=point,
                    point_labels=labels,
                    box=box_t,
                    multimask_output=False,
                )
            except TypeError:
                masks, _scores, _logits = predictor.predict(
                    box=box_t,
                    multimask_output=False,
                )
            sam_mask = (np.asarray(masks[0]) > 0).astype(np.uint8) * 255
            if sam_mask.shape != base_mask.shape:
                sam_mask = cv2.resize(
                    sam_mask,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
            region_gate = np.zeros_like(base_mask)
            region_gate[y1:y2, x1:x2] = 255
            sam_mask = cv2.bitwise_and(sam_mask, region_gate)
            refined = np.maximum(refined, sam_mask)
        return refined
    except Exception as exc:
        raise RequestedStageError(
            stage="sam2",
            requested_implementation="sam2",
            actual_implementation="sam2",
            provider=type(predictor).__name__,
            failure_class=FAILURE_RUNTIME,
            detail=str(exc),
            recovery_hint=(
                "Verify the SAM 2 checkpoint and runtime compatibility, then "
                "retry or disable SAM 2 refinement."
            ),
        ) from exc


# ---------------------------------------------------------------------------
# RM-67 -- SAM 3 text-prompt segmentation
# ---------------------------------------------------------------------------


_SAM3_STATE: dict = {"probed": False, "predictor": None, "load_error": None}


def _maybe_load_sam3():
    if _SAM3_STATE["probed"]:
        return _SAM3_STATE["predictor"]
    _SAM3_STATE["probed"] = True
    _SAM3_STATE["load_error"] = None
    if not _env_set("VSR_SAM3"):
        _SAM3_STATE["load_error"] = RequestedStageError(
            stage="sam3",
            requested_implementation="sam3",
            failure_class=FAILURE_DEPENDENCY_MISSING,
            detail="VSR_SAM3 is not enabled",
            recovery_hint="Install and enable SAM 3, then retry.",
        )
        return None
    try:
        from sam3 import SAM3Predictor  # type: ignore
    except ImportError as exc:
        logger.info(
            "sam3 package not importable; install via "
            "`pip install git+https://github.com/facebookresearch/sam3`."
        )
        _SAM3_STATE["load_error"] = RequestedStageError(
            stage="sam3",
            requested_implementation="sam3",
            failure_class=FAILURE_DEPENDENCY_MISSING,
            detail=str(exc),
            recovery_hint="Install and enable SAM 3, then retry.",
            cause=exc,
        )
        return None
    try:
        predictor = SAM3Predictor()
        _SAM3_STATE["predictor"] = predictor
        return predictor
    except Exception as exc:
        logger.warning(f"SAM 3 load failed: {exc}")
        _SAM3_STATE["load_error"] = RequestedStageError(
            stage="sam3",
            requested_implementation="sam3",
            actual_implementation="sam3",
            provider="sam3",
            failure_class=FAILURE_INITIALIZATION,
            detail=str(exc),
            recovery_hint="Verify the SAM 3 model and runtime, then retry.",
            cause=exc,
        )
        return None


def segment_text_with_sam3(frame: np.ndarray) -> Optional[np.ndarray]:
    """RM-67: ask SAM 3 "segment all burned-in text in this frame".
    Returns a single uint8 mask or raises a classified stage error."""
    predictor = _maybe_load_sam3()
    if predictor is None:
        load_error = _SAM3_STATE.get("load_error")
        if isinstance(load_error, RequestedStageError):
            raise load_error
        raise RequestedStageError(
            stage="sam3",
            requested_implementation="sam3",
            failure_class=FAILURE_DEPENDENCY_MISSING,
            detail="SAM 3 is unavailable",
            recovery_hint="Install and enable SAM 3, then retry.",
        )
    try:
        mask = predictor.segment(frame, prompt="burned-in subtitle text")
        result = (np.asarray(mask) > 0).astype(np.uint8) * 255
        if result.shape != frame.shape[:2]:
            raise RequestedStageError(
                stage="sam3",
                requested_implementation="sam3",
                actual_implementation="sam3",
                provider=type(predictor).__name__,
                failure_class=FAILURE_OUTPUT_INVALID,
                detail="SAM 3 returned a mask with invalid dimensions",
                recovery_hint="Verify the SAM 3 adapter version, then retry.",
            )
        return result
    except Exception as exc:
        if isinstance(exc, RequestedStageError):
            raise
        raise RequestedStageError(
            stage="sam3",
            requested_implementation="sam3",
            actual_implementation="sam3",
            provider=type(predictor).__name__,
            failure_class=FAILURE_RUNTIME,
            detail=str(exc),
            recovery_hint="Verify the SAM 3 model and runtime, then retry.",
            cause=exc,
        ) from exc


# ---------------------------------------------------------------------------
# RM-68 -- MatAnyone 2 video matting
# ---------------------------------------------------------------------------


_MATANYONE_STATE: dict = {"probed": False, "model": None, "load_error": None}
_MATANYONE_MODEL_ID = "PeiqingYang/MatAnyone2"


def _verify_matanyone_path(path: str) -> bool:
    try:
        from backend.adapter_manifest import (
            log_adapter_verification,
            verify_adapter_path,
        )
        result = verify_adapter_path("matanyone2", path)
        log_adapter_verification(result)
        return bool(result.allowed)
    except KeyError:
        return True
    except Exception as exc:
        logger.warning(f"MatAnyone 2 checkpoint verification failed: {exc}")
        return False


def _call_frame_api(target, frame: np.ndarray, hint_mask: np.ndarray):
    for name in ("matte", "predict", "process_frame", "infer"):
        fn = getattr(target, name, None)
        if fn is None:
            continue
        for args, kwargs in (
            ((), {"frame": frame, "mask": hint_mask}),
            ((), {"image": frame, "mask": hint_mask}),
            ((), {"frame": frame, "hint_mask": hint_mask}),
            ((frame, hint_mask), {}),
        ):
            try:
                return fn(*args, **kwargs)
            except TypeError:
                continue
    raise AttributeError("missing MatAnyone frame API")


def _call_sequence_api(target,
                       frames: List[np.ndarray],
                       masks: List[np.ndarray]):
    for name in ("matte_frames", "matte_video", "process_frames", "run"):
        fn = getattr(target, name, None)
        if fn is None:
            continue
        for args, kwargs in (
            ((), {"frames": frames, "masks": masks}),
            ((), {"images": frames, "masks": masks}),
            ((frames, masks), {}),
        ):
            try:
                return fn(*args, **kwargs)
            except TypeError:
                continue
    raise AttributeError("missing MatAnyone sequence API")


def _unwrap_alpha_payload(value):
    if isinstance(value, dict):
        for key in ("alpha", "alphas", "matte", "mattes", "mask", "masks", "output"):
            if key in value:
                return value[key]
    if isinstance(value, (list, tuple)) and len(value) == 2:
        first, second = value
        if isinstance(first, np.ndarray):
            return first
        return second
    return value


def _normalize_alpha_matte(alpha, frame_shape) -> Optional[np.ndarray]:
    alpha = _unwrap_alpha_payload(alpha)
    if alpha is None:
        return None
    arr = np.asarray(alpha)
    if arr.size == 0:
        return None
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[:, :, 3]
        elif arr.shape[2] == 1:
            arr = arr[:, :, 0]
        else:
            arr = cv2.cvtColor(arr.astype(np.float32), cv2.COLOR_BGR2GRAY)
    if arr.ndim != 2:
        return None
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8) * 255
    elif np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
        if float(np.nanmax(arr)) <= 1.0:
            arr *= 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    height, width = frame_shape[:2]
    if arr.shape[:2] != (height, width):
        arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_LINEAR)
    return arr if int(arr.max()) > 0 else None


def _is_explicit_empty_alpha(alpha) -> bool:
    """Return whether an adapter explicitly emitted a valid all-zero matte."""
    alpha = _unwrap_alpha_payload(alpha)
    if alpha is None:
        return False
    try:
        arr = np.asarray(alpha)
    except Exception:
        return False
    if arr.size == 0:
        return False
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[2] not in (1, 3, 4):
        return False
    if arr.ndim not in (2, 3):
        return False
    try:
        return not bool(np.any(np.nan_to_num(arr)))
    except (TypeError, ValueError):
        return False


def _normalize_alpha_sequence(value,
                              frames: List[np.ndarray],
                              masks: List[np.ndarray]) -> Optional[List[np.ndarray]]:
    if isinstance(value, dict):
        value = _unwrap_alpha_payload(value)
    if isinstance(value, np.ndarray):
        if value.ndim == 2 or (value.ndim == 3 and value.shape[-1] in (1, 3, 4)):
            value = [value]
        elif value.ndim == 3:
            value = [value[i] for i in range(value.shape[0])]
        elif value.ndim == 4:
            value = [value[i] for i in range(value.shape[0])]
    if not isinstance(value, (list, tuple)):
        return None
    if len(value) != len(frames):
        return None
    out: List[np.ndarray] = []
    for alpha, frame, hint in zip(value, frames, masks, strict=True):
        normalized = _normalize_alpha_matte(alpha, frame.shape)
        if int(np.asarray(hint).max()) == 0:
            out.append(np.asarray(hint).astype(np.uint8))
        elif normalized is not None:
            out.append(normalized)
        elif _is_explicit_empty_alpha(alpha):
            out.append(np.asarray(hint).astype(np.uint8))
        else:
            return None
    return out


def _write_matanyone_video(path: Path, frames: List[np.ndarray]) -> None:
    if not frames:
        raise RuntimeError("MatAnyone input frame list is empty")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        8.0,
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create MatAnyone input video: {path}")
    try:
        for frame in frames:
            item = np.asarray(frame).astype(np.uint8)
            if item.ndim == 2:
                item = cv2.cvtColor(item, cv2.COLOR_GRAY2BGR)
            if item.shape[:2] != (height, width):
                item = cv2.resize(item, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(item)
    finally:
        writer.release()


def _empty_alpha_fallback(frame: np.ndarray,
                          hint: Optional[np.ndarray]) -> np.ndarray:
    """RM-136: an all-transparent alpha frame is a subtitle gap, not a failure.

    ``_normalize_alpha_matte`` returns None for an all-zero frame, which is
    exactly what a frame with no subtitle produces. Discarding the whole
    refinement for that made the opt-in MatAnyone path a no-op on nearly every
    real clip. Fall back to the per-frame hint (like ``_normalize_alpha_sequence``
    already does), or to an explicit empty matte when no hint is available.
    """
    if hint is not None:
        hint_arr = np.asarray(hint)
        if hint_arr.size:
            return hint_arr.astype(np.uint8)
    height, width = frame.shape[:2]
    return np.zeros((height, width), dtype=np.uint8)


def _hint_at(hint_masks: Optional[List[np.ndarray]],
             index: int) -> Optional[np.ndarray]:
    if not hint_masks or index >= len(hint_masks):
        return None
    return hint_masks[index]


def _read_alpha_video(path: Path,
                      expected_count: int,
                      target_frames: List[np.ndarray],
                      hint_masks: Optional[List[np.ndarray]] = None,
                      ) -> Optional[List[np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    out: List[np.ndarray] = []
    try:
        while len(out) < expected_count:
            ok, frame = cap.read()
            if not ok:
                break
            index = len(out)
            target = target_frames[index]
            normalized = _normalize_alpha_matte(frame, target.shape)
            if normalized is None:
                normalized = _empty_alpha_fallback(
                    target, _hint_at(hint_masks, index))
            out.append(normalized)
    finally:
        cap.release()
    if len(out) != expected_count:
        return None
    return out


def _read_alpha_image_dir(path: Path,
                          expected_count: int,
                          target_frames: List[np.ndarray],
                          hint_masks: Optional[List[np.ndarray]] = None,
                          ) -> Optional[List[np.ndarray]]:
    files = [
        item for item in sorted(path.rglob("*"))
        if item.is_file() and item.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]
    if len(files) < expected_count:
        return None
    out: List[np.ndarray] = []
    for index, (item, frame) in enumerate(
            zip(files[:expected_count], target_frames, strict=True)):
        alpha = safe_imread(item, cv2.IMREAD_UNCHANGED)
        normalized = _normalize_alpha_matte(alpha, frame.shape)
        if normalized is None:
            # RM-136: a fully transparent frame is a subtitle gap.
            normalized = _empty_alpha_fallback(
                frame, _hint_at(hint_masks, index))
        out.append(normalized)
    if len(out) != expected_count:
        return None
    return out


def _read_matanyone_output(output_path: Path,
                           expected_count: int,
                           target_frames: List[np.ndarray],
                           hint_masks: Optional[List[np.ndarray]] = None,
                           ) -> Optional[List[np.ndarray]]:
    candidates: List[Path] = []
    if output_path.is_file():
        candidates.append(output_path)
    elif output_path.is_dir():
        candidates.extend(
            item for item in output_path.rglob("*")
            if item.is_file() and "alpha" in item.name.lower()
        )
        candidates.extend(
            item for item in output_path.rglob("*")
            if item.is_dir() and "alpha" in item.name.lower()
        )
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
            out = _read_alpha_video(
                candidate, expected_count, target_frames, hint_masks)
            if out is not None:
                return out
        if candidate.is_dir():
            out = _read_alpha_image_dir(
                candidate, expected_count, target_frames, hint_masks)
            if out is not None:
                return out
    return None


class _MatAnyone2Adapter:
    def __init__(self, model, processor=None):
        self._model = model
        self._processor = processor

    def matte(self, frame: np.ndarray, hint_mask: np.ndarray):
        for target in (self._processor, self._model):
            if target is None:
                continue
            try:
                return _call_frame_api(target, frame, hint_mask)
            except AttributeError:
                continue
        frames = [frame]
        masks = [hint_mask]
        result = self.matte_frames(frames, masks)
        if result and len(result) == 1:
            return result[0]
        raise AttributeError("missing MatAnyone frame API")

    def matte_frames(self, frames: List[np.ndarray], masks: List[np.ndarray]):
        for target in (self._processor, self._model):
            if target is None:
                continue
            try:
                return _call_sequence_api(target, frames, masks)
            except AttributeError:
                pass
            process_video = getattr(target, "process_video", None)
            if process_video is None:
                continue
            return _run_matanyone_process_video(process_video, frames, masks)
        return [_call_frame_api(self._model, frame, mask) for frame, mask in zip(frames, masks, strict=True)]


def _run_matanyone_process_video(process_video,
                                 frames: List[np.ndarray],
                                 masks: List[np.ndarray]):
    if not frames or not masks:
        return []
    frame_count = min(len(frames), len(masks))
    frames = frames[:frame_count]
    masks = masks[:frame_count]
    first_mask_idx = next(
        (idx for idx, mask in enumerate(masks) if int(np.asarray(mask).max()) > 0),
        None,
    )
    if first_mask_idx is None:
        return list(masks)
    active_frames = frames[first_mask_idx:]
    active_masks = masks[first_mask_idx:]
    with tempfile.TemporaryDirectory(prefix="vsr_matanyone_") as tmpdir:
        work = Path(tmpdir)
        input_video = work / "input.mp4"
        first_mask = work / "mask.png"
        output_dir = work / "results"
        output_dir.mkdir()
        _write_matanyone_video(input_video, active_frames)
        first_alpha = np.where(active_masks[0] > 0, 255, 0).astype(np.uint8)
        if not safe_imwrite(first_mask, first_alpha):
            raise RuntimeError("could not write MatAnyone first-frame mask")
        try:
            result = process_video(
                input_path=str(input_video),
                mask_path=str(first_mask),
                output_path=str(output_dir),
            )
        except TypeError:
            result = process_video(str(input_video), str(first_mask), str(output_dir))
        candidate = output_dir
        if isinstance(result, dict):
            for key in ("alpha", "alpha_path", "alphaVideo", "alpha_video", "output_path"):
                if result.get(key):
                    candidate = Path(str(result[key]))
                    break
        active_out = _read_matanyone_output(
            candidate, len(active_frames), active_frames, active_masks)
        if active_out is None:
            raise RuntimeError("MatAnyone alpha output was not found")
    return list(masks[:first_mask_idx]) + active_out


def _maybe_load_matanyone(device: str = "cpu"):
    if _MATANYONE_STATE["probed"]:
        return _MATANYONE_STATE["model"]
    _MATANYONE_STATE["probed"] = True
    _MATANYONE_STATE["load_error"] = None
    if not _env_set("VSR_MATANYONE"):
        _MATANYONE_STATE["load_error"] = RequestedStageError(
            stage="matanyone",
            requested_implementation="matanyone2",
            failure_class=FAILURE_DEPENDENCY_MISSING,
            detail="VSR_MATANYONE is not enabled",
            recovery_hint=(
                "Enable VSR_MATANYONE with an approved MatAnyone 2 source and "
                "checkpoint, then retry or disable MatAnyone refinement."
            ),
        )
        return None
    source = resolve_remote_model_source("matanyone")
    if not source.allowed:
        logger.warning("MatAnyone 2 disabled: %s", source.reason)
        _MATANYONE_STATE["load_error"] = RequestedStageError(
            stage="matanyone",
            requested_implementation="matanyone2",
            failure_class=FAILURE_POLICY_BLOCKED,
            detail=str(source.reason),
            recovery_hint=(
                "Configure an approved pinned MatAnyone 2 source, then retry "
                "or disable MatAnyone refinement."
            ),
        )
        return None
    if source.source_type == "local" and source.source:
        if not _verify_matanyone_path(source.source):
            _MATANYONE_STATE["load_error"] = RequestedStageError(
                stage="matanyone",
                requested_implementation="matanyone2",
                failure_class=FAILURE_POLICY_BLOCKED,
                detail="the MatAnyone 2 checkpoint failed verification",
                recovery_hint=(
                    "Use an approved MatAnyone 2 checkpoint, then retry or "
                    "disable MatAnyone refinement."
                ),
            )
            return None
    try:
        from matanyone2 import InferenceCore, MatAnyone2  # type: ignore
    except ImportError as exc:
        _MATANYONE_STATE["load_error"] = RequestedStageError(
            stage="matanyone",
            requested_implementation="matanyone2",
            failure_class=FAILURE_DEPENDENCY_MISSING,
            detail=str(exc),
            recovery_hint=(
                "Install MatAnyone 2 from the approved source, then retry or "
                "disable MatAnyone refinement."
            ),
            cause=exc,
        )
        return None
    try:
        model_id = os.environ.get("VSR_MATANYONE_MODEL_ID", "").strip()
        if not model_id:
            model_id = source.source if source.source_type == "local" else _MATANYONE_MODEL_ID
        if hasattr(MatAnyone2, "from_pretrained"):
            kwargs = {"revision": source.revision} if source.revision else {}
            model = MatAnyone2.from_pretrained(model_id, **kwargs)
        else:
            model = MatAnyone2(model_id)
        processor = InferenceCore(model, device=os.environ.get("VSR_MATANYONE_DEVICE", device))
        wrapped = _MatAnyone2Adapter(model, processor)
        _MATANYONE_STATE["model"] = wrapped
        return wrapped
    except Exception as exc:
        logger.warning(f"MatAnyone 2 load failed: {exc}")
        _MATANYONE_STATE["load_error"] = RequestedStageError(
            stage="matanyone",
            requested_implementation="matanyone2",
            actual_implementation="matanyone2",
            provider="matanyone2",
            failure_class=FAILURE_INITIALIZATION,
            detail=str(exc),
            recovery_hint=(
                "Verify the MatAnyone 2 adapter and checkpoint, then retry or "
                "disable MatAnyone refinement."
            ),
            cause=exc,
        )
        return None


def matte_frame(frame: np.ndarray,
                hint_mask: np.ndarray,
                device: str = "cpu") -> Optional[np.ndarray]:
    """RM-68: produce a soft alpha matte for the hinted region. Useful
    for thin moving subtitle lines that OCR + SAM both struggle with.
    Returns the alpha matte as uint8 or raises when the requested adapter fails."""
    if int(np.asarray(hint_mask).max()) == 0:
        return None
    model = _maybe_load_matanyone(device)
    if model is None:
        load_error = _MATANYONE_STATE.get("load_error")
        if isinstance(load_error, RequestedStageError):
            raise load_error
        raise RequestedStageError(
            stage="matanyone",
            requested_implementation="matanyone2",
            failure_class=FAILURE_DEPENDENCY_MISSING,
            detail="MatAnyone 2 is unavailable",
            recovery_hint="Install and enable MatAnyone 2, then retry.",
        )
    try:
        alpha = model.matte(frame, hint_mask)
        normalized = _normalize_alpha_matte(alpha, frame.shape)
        if normalized is None:
            raise RequestedStageError(
                stage="matanyone",
                requested_implementation="matanyone2",
                actual_implementation="matanyone2",
                provider=type(model).__name__,
                failure_class=FAILURE_OUTPUT_INVALID,
                detail="MatAnyone 2 returned no valid alpha matte",
                recovery_hint="Verify the MatAnyone 2 adapter version, then retry.",
            )
        return normalized
    except Exception as exc:
        if isinstance(exc, RequestedStageError):
            raise
        raise RequestedStageError(
            stage="matanyone",
            requested_implementation="matanyone2",
            actual_implementation="matanyone2",
            provider=type(model).__name__,
            failure_class=FAILURE_RUNTIME,
            detail=str(exc),
            recovery_hint=(
                "Verify the MatAnyone 2 adapter and checkpoint, then retry."
            ),
            cause=exc,
        ) from exc


def refine_masks_with_matanyone(frames: List[np.ndarray],
                                masks: List[np.ndarray],
                                device: str = "cpu") -> List[np.ndarray]:
    """Refine an aligned frame/mask batch with MatAnyone 2 when available.

    The adapter may return a temporal alpha sequence, but VSR only replaces
    frames that already had a positive subtitle mask. That keeps this path a
    mask refiner, not a destructive object tracker that can invent new removal
    regions on OCR-empty frames.
    """
    if not frames or not masks:
        return list(masks)
    frame_count = min(len(frames), len(masks))
    frames = list(frames[:frame_count])
    original = [np.asarray(mask).astype(np.uint8) for mask in masks[:frame_count]]
    if not any(int(mask.max()) > 0 for mask in original):
        return original
    model = _maybe_load_matanyone(device)
    if model is None:
        load_error = _MATANYONE_STATE.get("load_error")
        if isinstance(load_error, RequestedStageError):
            raise load_error
        raise RequestedStageError(
            stage="matanyone",
            requested_implementation="matanyone2",
            failure_class=FAILURE_DEPENDENCY_MISSING,
            detail="MatAnyone 2 package or approved checkpoint is unavailable",
            recovery_hint=(
                "Enable VSR_MATANYONE with an approved MatAnyone 2 source and "
                "checkpoint, then retry or disable MatAnyone refinement."
            ),
        )
    try:
        if hasattr(model, "matte_frames"):
            refined = _normalize_alpha_sequence(
                model.matte_frames(frames, original),
                frames,
                original,
            )
        else:
            raw_refined = [
                model.matte(frame, mask) if int(mask.max()) > 0 else None
                for frame, mask in zip(frames, original, strict=True)
            ]
            refined = _normalize_alpha_sequence(
                raw_refined, frames, original
            )
        if refined is None:
            raise RequestedStageError(
                stage="matanyone",
                requested_implementation="matanyone2",
                actual_implementation="matanyone2",
                provider=type(model).__name__,
                failure_class=FAILURE_OUTPUT_INVALID,
                detail="MatAnyone 2 returned no valid alpha sequence",
                recovery_hint=(
                    "Verify the MatAnyone 2 adapter API and checkpoint, then "
                    "retry or disable MatAnyone refinement."
                ),
            )
        out: List[np.ndarray] = []
        for source, refined_mask in zip(original, refined, strict=True):
            out.append(source if int(source.max()) == 0 else refined_mask)
        active_pairs = [
            (source, refined_mask)
            for source, refined_mask in zip(original, out, strict=True)
            if int(source.max()) > 0
        ]
        if active_pairs and all(
            np.array_equal(source, refined_mask)
            for source, refined_mask in active_pairs
        ):
            raise RequestedStageError(
                stage="matanyone",
                requested_implementation="matanyone2",
                actual_implementation="matanyone2",
                provider=type(model).__name__,
                failure_class=FAILURE_OUTPUT_INVALID,
                detail="MatAnyone 2 returned an unchanged alpha sequence",
                recovery_hint=(
                    "Verify the MatAnyone 2 adapter and checkpoint, then retry "
                    "or disable MatAnyone refinement."
                ),
            )
        return out
    except Exception as exc:
        if isinstance(exc, RequestedStageError):
            raise
        raise RequestedStageError(
            stage="matanyone",
            requested_implementation="matanyone2",
            actual_implementation="matanyone2",
            provider=type(model).__name__,
            failure_class=FAILURE_RUNTIME,
            detail=str(exc),
            recovery_hint=(
                "Verify the MatAnyone 2 adapter API and checkpoint, then "
                "retry or disable MatAnyone refinement."
            ),
        ) from exc


# ---------------------------------------------------------------------------
# RM-69 -- CoTracker3 point tracking
# ---------------------------------------------------------------------------


_COTRACKER_STATE: dict = {"probed": False, "model": None, "load_error": None}


def _tensor_to_numpy(value):
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
    except Exception:
        pass
    return np.asarray(value)


def _to_device(value, device: str, *, strict: bool = False):
    try:
        return value.to(device)
    except Exception:
        if strict:
            raise
        return value


def _cotracker_entrypoint() -> str:
    mode = os.environ.get("VSR_COTRACKER_MODE", "offline").strip().lower()
    if mode not in {"offline", "online"}:
        raise RequestedStageError(
            stage="cotracker",
            requested_implementation="cotracker3",
            failure_class=FAILURE_POLICY_BLOCKED,
            detail=f"unsupported CoTracker mode {mode!r}",
            recovery_hint="Set VSR_COTRACKER_MODE to 'offline' or 'online'.",
        )
    return "cotracker3_online" if mode == "online" else "cotracker3_offline"


def _maybe_load_cotracker(device: str = "cpu"):
    if _COTRACKER_STATE["probed"]:
        return _COTRACKER_STATE["model"]
    _COTRACKER_STATE["probed"] = True
    _COTRACKER_STATE["load_error"] = None
    if not _env_set("VSR_COTRACKER"):
        _COTRACKER_STATE["load_error"] = RequestedStageError(
            stage="cotracker",
            requested_implementation="cotracker3",
            failure_class=FAILURE_DEPENDENCY_MISSING,
            detail="VSR_COTRACKER is not enabled",
            recovery_hint=(
                "Enable VSR_COTRACKER with an approved pinned source, then "
                "retry or disable CoTracker propagation."
            ),
        )
        return None
    try:
        source = resolve_remote_model_source("cotracker3")
        if not source.allowed:
            logger.warning("CoTracker3 disabled: %s", source.reason)
            _COTRACKER_STATE["load_error"] = RequestedStageError(
                stage="cotracker",
                requested_implementation="cotracker3",
                failure_class=FAILURE_POLICY_BLOCKED,
                detail=str(source.reason),
                recovery_hint=(
                    "Configure an approved pinned CoTracker3 source, then "
                    "retry or disable CoTracker propagation."
                ),
            )
            return None
        try:
            import torch  # type: ignore
        except ImportError as exc:
            _COTRACKER_STATE["load_error"] = RequestedStageError(
                stage="cotracker",
                requested_implementation="cotracker3",
                failure_class=FAILURE_DEPENDENCY_MISSING,
                detail=str(exc),
                recovery_hint="Install the approved CoTracker3 runtime, then retry.",
                cause=exc,
            )
            return None
        entrypoint = _cotracker_entrypoint()
        if source.source_type == "local":
            model = torch.hub.load(
                source.source,
                entrypoint,
                source="local",
                trust_repo=True,
            )
        else:
            model = torch.hub.load(
                f"{source.policy.repo}:{source.revision}",
                entrypoint,
                trust_repo=True,
            )
        if hasattr(model, "to"):
            model = model.to(device)
        if hasattr(model, "eval"):
            model.eval()
        _COTRACKER_STATE["model"] = model
        return model
    except RequestedStageError as exc:
        _COTRACKER_STATE["load_error"] = exc
        raise
    except Exception as exc:
        logger.warning(f"CoTracker3 load failed: {exc}")
        _COTRACKER_STATE["load_error"] = RequestedStageError(
            stage="cotracker",
            requested_implementation="cotracker3",
            actual_implementation="cotracker3",
            provider="cotracker3",
            failure_class=FAILURE_INITIALIZATION,
            detail=str(exc),
            recovery_hint=(
                "Verify the CoTracker3 source, model, and runtime, then retry."
            ),
            cause=exc,
        )
        return None


def _prepare_cotracker_video(frames: List[np.ndarray]):
    rgb_frames = []
    for frame in frames:
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGRA2RGB)
        elif arr.ndim == 3:
            arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2RGB)
        else:
            return None
        rgb_frames.append(arr)
    return np.stack(rgb_frames, axis=0)


def track_points_with_visibility(
    frames: List[np.ndarray],
    points: List[Tuple[int, int]],
    *,
    device: str = "cpu",
    query_frame: int = 0,
    strict: bool = False,
) -> Optional[Tuple[List[List[Tuple[int, int]]], List[List[float]]]]:
    """Track points and return `(tracks, visibility)` per frame.

    CoTracker expects query rows as `(frame_idx, x, y)`. The default offline
    model can propagate from an anchor frame across the whole batch. Online
    mode is still accepted for users who opt into it via VSR_COTRACKER_MODE,
    but the function returns None if that model does not expose the same call
    signature.
    """
    if not frames or not points:
        return None
    model = _maybe_load_cotracker(device)
    if model is None:
        if strict:
            load_error = _COTRACKER_STATE.get("load_error")
            if isinstance(load_error, RequestedStageError):
                raise load_error
            raise RequestedStageError(
                stage="cotracker",
                requested_implementation="cotracker3",
                failure_class=FAILURE_DEPENDENCY_MISSING,
                detail="CoTracker3 package or approved model source is unavailable",
                recovery_hint=(
                    "Enable VSR_COTRACKER with an approved pinned source, "
                    "then retry or disable CoTracker propagation."
                ),
            )
        return None
    try:
        import torch  # type: ignore

        video_np = _prepare_cotracker_video(frames)
        if video_np is None:
            if strict:
                raise RequestedStageError(
                    stage="cotracker",
                    requested_implementation="cotracker3",
                    actual_implementation="cotracker3",
                    provider=type(model).__name__,
                    failure_class=FAILURE_OUTPUT_INVALID,
                    detail="input frames could not be normalized for CoTracker3",
                    recovery_hint=(
                        "Use supported image frames or disable CoTracker propagation."
                    ),
                )
            return None
        video = torch.from_numpy(video_np).permute(0, 3, 1, 2).unsqueeze(0).float()
        video = _to_device(video, device, strict=strict)
        query_idx = max(0, min(len(frames) - 1, int(query_frame)))
        query = torch.tensor(
            [[query_idx, float(x), float(y)] for (x, y) in points],
            dtype=torch.float32,
        ).unsqueeze(0)
        query = _to_device(query, device, strict=strict)
        result = None
        for kwargs in (
            {"queries": query, "backward_tracking": True},
            {"queries": query},
        ):
            try:
                result = model(video, **kwargs)
                break
            except TypeError:
                continue
        if result is None:
            if strict:
                raise RequestedStageError(
                    stage="cotracker",
                    requested_implementation="cotracker3",
                    actual_implementation="cotracker3",
                    provider=type(model).__name__,
                    failure_class=FAILURE_OUTPUT_INVALID,
                    detail="CoTracker3 returned no track result",
                    recovery_hint=(
                        "Verify the CoTracker3 adapter version, then retry or "
                        "disable CoTracker propagation."
                    ),
                )
            return None
        pred_tracks, pred_visibility = result
        tracks_np = _tensor_to_numpy(pred_tracks)
        vis_np = _tensor_to_numpy(pred_visibility)
        if tracks_np is None:
            if strict:
                raise RequestedStageError(
                    stage="cotracker",
                    requested_implementation="cotracker3",
                    actual_implementation="cotracker3",
                    provider=type(model).__name__,
                    failure_class=FAILURE_OUTPUT_INVALID,
                    detail="CoTracker3 returned no coordinate tensor",
                    recovery_hint=(
                        "Verify the CoTracker3 adapter version, then retry or "
                        "disable CoTracker propagation."
                    ),
                )
            return None
        if tracks_np.ndim == 3:
            tracks_np = tracks_np[None, ...]
        if tracks_np.ndim != 4 or tracks_np.shape[-1] < 2:
            if strict:
                raise RequestedStageError(
                    stage="cotracker",
                    requested_implementation="cotracker3",
                    actual_implementation="cotracker3",
                    provider=type(model).__name__,
                    failure_class=FAILURE_OUTPUT_INVALID,
                    detail="CoTracker3 returned a malformed coordinate tensor",
                    recovery_hint=(
                        "Verify the CoTracker3 adapter version, then retry or "
                        "disable CoTracker propagation."
                    ),
                )
            return None
        tracks_np = tracks_np[0]
        if tracks_np.shape[0] != len(frames):
            if strict:
                raise RequestedStageError(
                    stage="cotracker",
                    requested_implementation="cotracker3",
                    actual_implementation="cotracker3",
                    provider=type(model).__name__,
                    failure_class=FAILURE_OUTPUT_INVALID,
                    detail="CoTracker3 returned a different frame count",
                    recovery_hint=(
                        "Verify the CoTracker3 adapter version, then retry or "
                        "disable CoTracker propagation."
                    ),
                )
            return None
        if vis_np is None:
            if strict:
                raise RequestedStageError(
                    stage="cotracker",
                    requested_implementation="cotracker3",
                    actual_implementation="cotracker3",
                    provider=type(model).__name__,
                    failure_class=FAILURE_OUTPUT_INVALID,
                    detail="CoTracker3 returned no visibility tensor",
                    recovery_hint=(
                        "Verify the CoTracker3 adapter version, then retry or "
                        "disable CoTracker propagation."
                    ),
                )
            vis_np = np.ones(tracks_np.shape[:2], dtype=np.float32)
        else:
            if vis_np.ndim == 4 and vis_np.shape[-1] == 1:
                vis_np = vis_np[..., 0]
            if vis_np.ndim == 3:
                vis_np = vis_np[0]
            if vis_np.shape != tracks_np.shape[:2]:
                if strict:
                    raise RequestedStageError(
                        stage="cotracker",
                        requested_implementation="cotracker3",
                        actual_implementation="cotracker3",
                        provider=type(model).__name__,
                        failure_class=FAILURE_OUTPUT_INVALID,
                        detail="CoTracker3 returned a malformed visibility tensor",
                        recovery_hint=(
                            "Verify the CoTracker3 adapter version, then retry "
                            "or disable CoTracker propagation."
                        ),
                    )
                vis_np = np.ones(tracks_np.shape[:2], dtype=np.float32)
        out_tracks: List[List[Tuple[int, int]]] = []
        out_vis: List[List[float]] = []
        height, width = frames[0].shape[:2]
        for t in range(tracks_np.shape[0]):
            frame_points: List[Tuple[int, int]] = []
            frame_vis: List[float] = []
            for idx, point in enumerate(tracks_np[t]):
                x = int(round(float(point[0])))
                y = int(round(float(point[1])))
                frame_points.append((
                    max(0, min(width - 1, x)),
                    max(0, min(height - 1, y)),
                ))
                frame_vis.append(float(vis_np[t, idx]))
            out_tracks.append(frame_points)
            out_vis.append(frame_vis)
        return out_tracks, out_vis
    except Exception as exc:
        if isinstance(exc, RequestedStageError):
            raise
        if strict:
            raise RequestedStageError(
                stage="cotracker",
                requested_implementation="cotracker3",
                actual_implementation="cotracker3",
                provider=type(model).__name__,
                failure_class=FAILURE_RUNTIME,
                detail=str(exc),
                recovery_hint=(
                    "Verify the CoTracker3 model and Torch runtime, then retry "
                    "or disable CoTracker propagation."
                ),
            ) from exc
        logger.warning(f"CoTracker3 inference failed: {exc}")
        return None


def track_points(frames: List[np.ndarray],
                  points: List[Tuple[int, int]],
                  *,
                  device: str = "cpu",
                  query_frame: int = 0) -> Optional[List[List[Tuple[int, int]]]]:
    """RM-69: track the named pixel points across the frame list.
    Returns one (T, len(points)) coordinate list. A requested CoTracker3
    execution fails with a classified error when it cannot run."""
    result = track_points_with_visibility(
        frames,
        points,
        device=device,
        query_frame=query_frame,
        strict=True,
    )
    if result is None:
        return None
    tracks, _visibility = result
    return tracks


def _sample_mask_points(mask: np.ndarray, max_points: int = 8) -> List[Tuple[int, int]]:
    ys, xs = np.where(np.asarray(mask) > 0)
    if xs.size == 0 or ys.size == 0:
        return []
    coords = np.column_stack((xs, ys))
    order = np.argsort(coords[:, 0] + coords[:, 1])
    coords = coords[order]
    if coords.shape[0] > max_points:
        idx = np.linspace(0, coords.shape[0] - 1, max_points).astype(int)
        coords = coords[idx]
    points = {(int(x), int(y)) for x, y in coords}
    points.add((int(np.median(xs)), int(np.median(ys))))
    return sorted(points)[:max_points]


def _translate_mask(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    height, width = mask.shape[:2]
    matrix = np.float32([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]])
    return cv2.warpAffine(
        np.asarray(mask).astype(np.uint8),
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def propagate_masks_with_cotracker(
    frames: List[np.ndarray],
    masks: List[np.ndarray],
    *,
    device: str = "cpu",
    visibility_threshold: float = 0.5,
) -> List[np.ndarray]:
    """Fill OCR-empty masks by translating a nearby positive mask.

    The function is intentionally conservative: it never replaces a non-empty
    OCR/SAM mask, and it skips propagation when too few tracked anchor points
    remain visible or the estimated translation is implausibly large.
    """
    if not frames or not masks:
        return list(masks)
    frame_count = min(len(frames), len(masks))
    frames = list(frames[:frame_count])
    original = [np.asarray(mask).astype(np.uint8) for mask in masks[:frame_count]]
    if frame_count < 2:
        return original
    positive = [idx for idx, mask in enumerate(original) if int(mask.max()) > 0]
    empty = [idx for idx, mask in enumerate(original) if int(mask.max()) == 0]
    if not positive or not empty:
        return original

    anchor_idx = positive[0]
    anchor_mask = original[anchor_idx]
    points = _sample_mask_points(anchor_mask)
    if not points:
        raise RequestedStageError(
            stage="cotracker",
            requested_implementation="cotracker3",
            failure_class=FAILURE_OUTPUT_INVALID,
            detail="the positive anchor mask produced no tracking points",
            recovery_hint=(
                "Use a larger valid mask or disable CoTracker propagation."
            ),
        )
    result = track_points_with_visibility(
        frames,
        points,
        device=device,
        query_frame=anchor_idx,
        strict=True,
    )
    if result is None:
        raise RequestedStageError(
            stage="cotracker",
            requested_implementation="cotracker3",
            failure_class=FAILURE_OUTPUT_INVALID,
            detail="CoTracker3 returned no usable tracks",
            recovery_hint=(
                "Verify the CoTracker3 adapter and model, then retry or disable "
                "CoTracker propagation."
            ),
        )
    tracks, visibility = result
    if len(tracks) != frame_count:
        raise RequestedStageError(
            stage="cotracker",
            requested_implementation="cotracker3",
            actual_implementation="cotracker3",
            provider=selected_segmentation_provider("cotracker3"),
            failure_class=FAILURE_OUTPUT_INVALID,
            detail="CoTracker3 returned the wrong number of frame tracks",
            recovery_hint="Verify the CoTracker3 adapter version, then retry.",
        )
    base_points = np.asarray(tracks[anchor_idx], dtype=np.float32)
    if base_points.shape[0] != len(points):
        raise RequestedStageError(
            stage="cotracker",
            requested_implementation="cotracker3",
            actual_implementation="cotracker3",
            provider=selected_segmentation_provider("cotracker3"),
            failure_class=FAILURE_OUTPUT_INVALID,
            detail="CoTracker3 returned the wrong number of tracked points",
            recovery_hint="Verify the CoTracker3 adapter version, then retry.",
        )

    out = list(original)
    height, width = frames[0].shape[:2]
    max_dx = width * 0.35
    max_dy = height * 0.35
    for idx in empty:
        frame_points = np.asarray(tracks[idx], dtype=np.float32)
        frame_vis = np.asarray(visibility[idx], dtype=np.float32)
        valid = frame_vis >= float(visibility_threshold)
        if frame_points.shape != base_points.shape or int(valid.sum()) < 2:
            continue
        deltas = frame_points[valid] - base_points[valid]
        if deltas.size == 0:
            continue
        dx, dy = np.median(deltas, axis=0)
        if not np.isfinite(dx) or not np.isfinite(dy):
            continue
        if abs(float(dx)) > max_dx or abs(float(dy)) > max_dy:
            continue
        shifted = _translate_mask(anchor_mask, float(dx), float(dy))
        if int(shifted.max()) > 0:
            out[idx] = shifted
    return out


def selected_segmentation_provider(implementation: str) -> str:
    """Return the concrete cached provider for an executed optional stage."""
    key = str(implementation or "").strip().lower()
    state = {
        "sam2": _SAM2_STATE,
        "matanyone2": _MATANYONE_STATE,
        "cotracker3": _COTRACKER_STATE,
    }.get(key)
    if not isinstance(state, dict):
        return key or "unknown"
    model = state.get("predictor")
    if model is None:
        model = state.get("model")
    if model is None:
        return key or "unknown"
    model_type = type(model)
    return f"{model_type.__module__}.{model_type.__name__}"
