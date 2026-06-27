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
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from backend.remote_model_policy import resolve_remote_model_source
from backend.safe_image import safe_imread

logger = logging.getLogger(__name__)


def _env_set(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# RM-66 -- SAM 2 mask refinement
# ---------------------------------------------------------------------------


_SAM2_STATE: dict = {"probed": False, "predictor": None}


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
        height, width = base_mask.shape[:2]
        for raw_box in boxes:
            clipped = _clip_box(raw_box, width, height)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            refined[y1:y2, x1:x2] = 0
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
    for alpha, frame, hint in zip(value, frames, masks):
        normalized = _normalize_alpha_matte(alpha, frame.shape)
        if normalized is None or int(np.asarray(hint).max()) == 0:
            out.append(np.asarray(hint).astype(np.uint8))
        else:
            out.append(normalized)
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


def _read_alpha_video(path: Path,
                      expected_count: int,
                      target_frames: List[np.ndarray]) -> Optional[List[np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    out: List[np.ndarray] = []
    try:
        while len(out) < expected_count:
            ok, frame = cap.read()
            if not ok:
                break
            out.append(_normalize_alpha_matte(frame, target_frames[len(out)].shape))
    finally:
        cap.release()
    if len(out) != expected_count or any(item is None for item in out):
        return None
    return [item for item in out if item is not None]


def _read_alpha_image_dir(path: Path,
                          expected_count: int,
                          target_frames: List[np.ndarray]) -> Optional[List[np.ndarray]]:
    files = [
        item for item in sorted(path.rglob("*"))
        if item.is_file() and item.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]
    if len(files) < expected_count:
        return None
    out: List[np.ndarray] = []
    for item, frame in zip(files[:expected_count], target_frames):
        alpha = safe_imread(item, cv2.IMREAD_UNCHANGED)
        normalized = _normalize_alpha_matte(alpha, frame.shape)
        if normalized is None:
            return None
        out.append(normalized)
    return out


def _read_matanyone_output(output_path: Path,
                           expected_count: int,
                           target_frames: List[np.ndarray]) -> Optional[List[np.ndarray]]:
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
            out = _read_alpha_video(candidate, expected_count, target_frames)
            if out is not None:
                return out
        if candidate.is_dir():
            out = _read_alpha_image_dir(candidate, expected_count, target_frames)
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
        return [_call_frame_api(self._model, frame, mask) for frame, mask in zip(frames, masks)]


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
        if not cv2.imwrite(str(first_mask), first_alpha):
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
        active_out = _read_matanyone_output(candidate, len(active_frames), active_frames)
        if active_out is None:
            raise RuntimeError("MatAnyone alpha output was not found")
    return list(masks[:first_mask_idx]) + active_out


def _maybe_load_matanyone(device: str = "cpu"):
    if _MATANYONE_STATE["probed"]:
        return _MATANYONE_STATE["model"]
    _MATANYONE_STATE["probed"] = True
    if not _env_set("VSR_MATANYONE"):
        return None
    source = resolve_remote_model_source("matanyone")
    if not source.allowed:
        logger.warning("MatAnyone 2 disabled: %s", source.reason)
        return None
    if source.source_type == "local" and source.source:
        if not _verify_matanyone_path(source.source):
            return None
    try:
        from matanyone2 import InferenceCore, MatAnyone2  # type: ignore
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
        logger.debug(f"MatAnyone 2 package load failed: {exc}")
    try:
        from matanyone import MatAnyone  # type: ignore
        kwargs = {"device": device}
        if source.source_type == "local" and source.source:
            kwargs["checkpoint"] = source.source
        try:
            model = MatAnyone(**kwargs)
        except TypeError:
            model = MatAnyone()
        _MATANYONE_STATE["model"] = model
        return model
    except Exception as exc:
        logger.warning(f"MatAnyone 2 load failed: {exc}")
        return None


def matte_frame(frame: np.ndarray,
                hint_mask: np.ndarray,
                device: str = "cpu") -> Optional[np.ndarray]:
    """RM-68: produce a soft alpha matte for the hinted region. Useful
    for thin moving subtitle lines that OCR + SAM both struggle with.
    Returns the alpha matte as uint8 or None on missing dep / error."""
    if int(np.asarray(hint_mask).max()) == 0:
        return None
    model = _maybe_load_matanyone(device)
    if model is None:
        return None
    try:
        alpha = model.matte(frame, hint_mask)
        return _normalize_alpha_matte(alpha, frame.shape)
    except Exception as exc:
        logger.warning(f"MatAnyone inference failed: {exc}")
        return None


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
        return original
    try:
        if hasattr(model, "matte_frames"):
            refined = _normalize_alpha_sequence(
                model.matte_frames(frames, original),
                frames,
                original,
            )
        else:
            refined = [
                _normalize_alpha_matte(model.matte(frame, mask), frame.shape)
                if int(mask.max()) > 0 else None
                for frame, mask in zip(frames, original)
            ]
            refined = [
                mask if alpha is None else alpha
                for alpha, mask in zip(refined, original)
            ]
        if refined is None:
            return original
        out: List[np.ndarray] = []
        for source, refined_mask in zip(original, refined):
            out.append(source if int(source.max()) == 0 else refined_mask)
        return out
    except Exception as exc:
        logger.warning(f"MatAnyone sequence refinement failed: {exc}")
        return original


# ---------------------------------------------------------------------------
# RM-69 -- CoTracker3 point tracking
# ---------------------------------------------------------------------------


_COTRACKER_STATE: dict = {"probed": False, "model": None}


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


def _to_device(value, device: str):
    try:
        return value.to(device)
    except Exception:
        return value


def _cotracker_entrypoint() -> str:
    mode = os.environ.get("VSR_COTRACKER_MODE", "offline").strip().lower()
    return "cotracker3_online" if mode == "online" else "cotracker3_offline"


def _maybe_load_cotracker(device: str = "cpu"):
    if _COTRACKER_STATE["probed"]:
        return _COTRACKER_STATE["model"]
    _COTRACKER_STATE["probed"] = True
    if not _env_set("VSR_COTRACKER"):
        return None
    try:
        source = resolve_remote_model_source("cotracker3")
        if not source.allowed:
            logger.warning("CoTracker3 disabled: %s", source.reason)
            return None
        import torch  # type: ignore
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
    except Exception as exc:
        logger.warning(f"CoTracker3 load failed: {exc}")
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
        return None
    try:
        import torch  # type: ignore

        video_np = _prepare_cotracker_video(frames)
        if video_np is None:
            return None
        video = torch.from_numpy(video_np).permute(0, 3, 1, 2).unsqueeze(0).float()
        video = _to_device(video, device)
        query_idx = max(0, min(len(frames) - 1, int(query_frame)))
        query = torch.tensor(
            [[query_idx, float(x), float(y)] for (x, y) in points],
            dtype=torch.float32,
        ).unsqueeze(0)
        query = _to_device(query, device)
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
            return None
        pred_tracks, pred_visibility = result
        tracks_np = _tensor_to_numpy(pred_tracks)
        vis_np = _tensor_to_numpy(pred_visibility)
        if tracks_np is None:
            return None
        if tracks_np.ndim == 3:
            tracks_np = tracks_np[None, ...]
        if tracks_np.ndim != 4 or tracks_np.shape[-1] < 2:
            return None
        tracks_np = tracks_np[0]
        if tracks_np.shape[0] != len(frames):
            return None
        if vis_np is None:
            vis_np = np.ones(tracks_np.shape[:2], dtype=np.float32)
        else:
            if vis_np.ndim == 4 and vis_np.shape[-1] == 1:
                vis_np = vis_np[..., 0]
            if vis_np.ndim == 3:
                vis_np = vis_np[0]
            if vis_np.shape != tracks_np.shape[:2]:
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
        logger.warning(f"CoTracker3 inference failed: {exc}")
        return None


def track_points(frames: List[np.ndarray],
                  points: List[Tuple[int, int]],
                  *,
                  device: str = "cpu",
                  query_frame: int = 0) -> Optional[List[List[Tuple[int, int]]]]:
    """RM-69: track the named pixel points across the frame list.
    Returns one (T, len(points)) coord list, or None when CoTracker3
    is unavailable. Used by callers that need to confirm a karaoke
    caret stays on the same line across a clip."""
    result = track_points_with_visibility(
        frames,
        points,
        device=device,
        query_frame=query_frame,
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
        return original
    result = track_points_with_visibility(
        frames,
        points,
        device=device,
        query_frame=anchor_idx,
    )
    if result is None:
        return original
    tracks, visibility = result
    if len(tracks) != frame_count:
        return original
    base_points = np.asarray(tracks[anchor_idx], dtype=np.float32)
    if base_points.shape[0] != len(points):
        return original

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
