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

Each adapter imports lazily and returns the input mask unchanged when
its dep is unavailable so the pipeline stays correct. Mask refinement
runs AFTER the OCR cascade and BEFORE _create_mask widens the boxes.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _env_set(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# RM-66 -- SAM 2 mask refinement
# ---------------------------------------------------------------------------


_SAM2_STATE: dict = {"probed": False, "predictor": None}


def _maybe_load_sam2(device: str):
    if _SAM2_STATE["probed"]:
        return _SAM2_STATE["predictor"]
    _SAM2_STATE["probed"] = True
    weight_path = os.environ.get("VSR_SAM2_CHECKPOINT", "")
    config_path = os.environ.get("VSR_SAM2_CONFIG", "")
    if not weight_path:
        logger.info(
            "SAM 2 refinement opt-in: set VSR_SAM2_CHECKPOINT (and "
            "VSR_SAM2_CONFIG) to enable. See facebookresearch/sam2."
        )
        return None
    try:
        from sam2.build_sam import build_sam2  # type: ignore
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
    except ImportError:
        logger.info(
            "sam2 package not importable; install via "
            "`pip install git+https://github.com/facebookresearch/sam2`."
        )
        return None
    try:
        model = build_sam2(config_path, weight_path)
        predictor = SAM2ImagePredictor(model)
        _SAM2_STATE["predictor"] = predictor
        return predictor
    except Exception as exc:
        logger.warning(f"SAM 2 load failed: {exc}")
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
        return base_mask
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb)
        refined = base_mask.copy()
        for (x1, y1, x2, y2) in boxes:
            box_t = np.array([x1, y1, x2, y2], dtype=np.float32)[None, :]
            masks, _scores, _logits = predictor.predict(box=box_t, multimask_output=False)
            sam_mask = (masks[0] > 0).astype(np.uint8) * 255
            refined = np.maximum(refined, sam_mask)
        return refined
    except Exception as exc:
        logger.warning(f"SAM 2 inference failed: {exc}")
        return base_mask


# ---------------------------------------------------------------------------
# RM-67 -- SAM 3 text-prompt segmentation
# ---------------------------------------------------------------------------


_SAM3_STATE: dict = {"probed": False, "predictor": None}


def _maybe_load_sam3():
    if _SAM3_STATE["probed"]:
        return _SAM3_STATE["predictor"]
    _SAM3_STATE["probed"] = True
    if not _env_set("VSR_SAM3"):
        return None
    try:
        from sam3 import SAM3Predictor  # type: ignore
    except ImportError:
        logger.info(
            "sam3 package not importable; install via "
            "`pip install git+https://github.com/facebookresearch/sam3`."
        )
        return None
    try:
        predictor = SAM3Predictor()
        _SAM3_STATE["predictor"] = predictor
        return predictor
    except Exception as exc:
        logger.warning(f"SAM 3 load failed: {exc}")
        return None


def segment_text_with_sam3(frame: np.ndarray) -> Optional[np.ndarray]:
    """RM-67: ask SAM 3 "segment all burned-in text in this frame".
    Returns a single uint8 mask or None when the dep is missing."""
    predictor = _maybe_load_sam3()
    if predictor is None:
        return None
    try:
        mask = predictor.segment(frame, prompt="burned-in subtitle text")
        return (np.asarray(mask) > 0).astype(np.uint8) * 255
    except Exception as exc:
        logger.warning(f"SAM 3 inference failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# RM-68 -- MatAnyone 2 video matting
# ---------------------------------------------------------------------------


_MATANYONE_STATE: dict = {"probed": False, "model": None}


def _maybe_load_matanyone():
    if _MATANYONE_STATE["probed"]:
        return _MATANYONE_STATE["model"]
    _MATANYONE_STATE["probed"] = True
    if not _env_set("VSR_MATANYONE"):
        return None
    try:
        from matanyone import MatAnyone  # type: ignore
        model = MatAnyone()
        _MATANYONE_STATE["model"] = model
        return model
    except Exception:
        return None


def matte_frame(frame: np.ndarray, hint_mask: np.ndarray) -> Optional[np.ndarray]:
    """RM-68: produce a soft alpha matte for the hinted region. Useful
    for thin moving subtitle lines that OCR + SAM both struggle with.
    Returns the alpha matte as uint8 or None on missing dep / error."""
    model = _maybe_load_matanyone()
    if model is None:
        return None
    try:
        alpha = model.matte(frame, hint_mask)
        return (np.asarray(alpha)).astype(np.uint8)
    except Exception as exc:
        logger.warning(f"MatAnyone inference failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# RM-69 -- CoTracker3 point tracking
# ---------------------------------------------------------------------------


_COTRACKER_STATE: dict = {"probed": False, "model": None}


def _maybe_load_cotracker():
    if _COTRACKER_STATE["probed"]:
        return _COTRACKER_STATE["model"]
    _COTRACKER_STATE["probed"] = True
    if not _env_set("VSR_COTRACKER"):
        return None
    try:
        import torch  # type: ignore
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
        _COTRACKER_STATE["model"] = model
        return model
    except Exception:
        return None


def track_points(frames: List[np.ndarray],
                  points: List[Tuple[int, int]]) -> Optional[List[List[Tuple[int, int]]]]:
    """RM-69: track the named pixel points across the frame list.
    Returns one (T, len(points)) coord list, or None when CoTracker3
    is unavailable. Used by callers that need to confirm a karaoke
    caret stays on the same line across a clip."""
    model = _maybe_load_cotracker()
    if model is None:
        return None
    try:
        import torch  # type: ignore
        video = torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2).unsqueeze(0).float()
        query = torch.tensor(
            [[0, x, y] for (x, y) in points],
            dtype=torch.float32,
        ).unsqueeze(0)
        pred_tracks, _vis = model(video, queries=query)
        out: List[List[Tuple[int, int]]] = []
        for t in range(pred_tracks.shape[1]):
            frame_points = [
                (int(pt[0]), int(pt[1])) for pt in pred_tracks[0, t]
            ]
            out.append(frame_points)
        return out
    except Exception as exc:
        logger.warning(f"CoTracker3 inference failed: {exc}")
        return None
