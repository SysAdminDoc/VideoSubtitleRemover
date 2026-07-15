"""Deterministic clean-reference alignment and final-mask composition."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


CLEAN_REFERENCE_SCHEMA = "vsr.clean_reference.v1"
ALIGNMENT_MODES = ("auto", "translation", "homography")


def _finite_float(value: object, default: float, low: float, high: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return max(low, min(high, result))


def _boolean(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def normalize_clean_reference(value: object) -> Optional[dict]:
    """Normalize the clean plate attached to one timed manual region."""
    if not isinstance(value, Mapping):
        return None
    path = str(value.get("path", value.get("image", "")) or "").strip()
    if not path:
        return None
    path = path[:2048]
    alignment = str(value.get("alignment", "auto") or "auto").strip().lower()
    if alignment not in ALIGNMENT_MODES:
        alignment = "auto"
    return {
        "path": path,
        "alignment": alignment,
        "color_match": _boolean(value.get("color_match", True), True),
        "min_confidence": _finite_float(
            value.get("min_confidence", 0.75), 0.75, 0.05, 0.99),
    }


def clean_reference_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_reference_source_evidence(path: str | Path) -> dict:
    source = Path(path)
    stat = source.stat()
    return {
        "name": source.name,
        "bytes": int(stat.st_size),
        "sha256": clean_reference_sha256(source),
    }


@dataclass
class ReferenceFillResult:
    composite: Any
    aligned: Any
    accepted: bool
    method: str
    confidence: float
    color_delta: tuple[float, float, float]
    matrix: list[list[float]]
    reason: str = ""


def _alignment_scale(width: int, height: int, maximum: int = 960) -> float:
    largest = max(width, height)
    if largest <= maximum:
        return 1.0
    return maximum / float(largest)


def _centered_similarity(target: Any, candidate: Any, valid: Any) -> float:
    import cv2
    import numpy as np

    left = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype(np.float32)
    right = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY).astype(np.float32)
    pixels = valid > 0
    if int(np.count_nonzero(pixels)) < 64:
        return 0.0
    left_values = left[pixels]
    right_values = right[pixels]
    left_values = left_values - float(np.median(left_values))
    right_values = right_values - float(np.median(right_values))
    difference = np.abs(left_values - right_values)
    robust_error = float(np.percentile(difference, 75))
    error_score = max(0.0, min(1.0, 1.0 - robust_error / 64.0))
    left_std = float(left_values.std())
    right_std = float(right_values.std())
    if left_std < 1.0 or right_std < 1.0:
        return error_score
    correlation = float(np.corrcoef(left_values, right_values)[0, 1])
    if not math.isfinite(correlation):
        correlation = 0.0
    correlation = max(0.0, min(1.0, correlation))
    return 0.55 * correlation + 0.45 * error_score


def _full_resolution_warp(warp: Any, scale: float, homography: bool) -> Any:
    import numpy as np

    if scale >= 0.999999:
        return warp.astype(np.float32)
    if not homography:
        result = warp.astype(np.float32).copy()
        result[:, 2] /= scale
        return result
    transform = np.array(
        [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    inverse = np.array(
        [[1.0 / scale, 0.0, 0.0],
         [0.0, 1.0 / scale, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return inverse @ warp.astype(np.float32) @ transform


def _ecc_candidate(
    frame: Any,
    reference: Any,
    valid_mask: Any,
    method: str,
) -> Optional[tuple[Any, float, Any]]:
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    scale = _alignment_scale(width, height)
    if scale < 1.0:
        size = (max(32, int(round(width * scale))),
                max(32, int(round(height * scale))))
        target = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        source = cv2.resize(reference, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(valid_mask, size, interpolation=cv2.INTER_NEAREST)
    else:
        target, source, mask = frame, reference, valid_mask
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    homography = method == "homography"
    motion = cv2.MOTION_HOMOGRAPHY if homography else cv2.MOTION_TRANSLATION
    warp = np.eye(3 if homography else 2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        80,
        1e-5,
    )
    try:
        correlation, warp = cv2.findTransformECC(
            target_gray,
            source_gray,
            warp,
            motion,
            criteria,
            inputMask=mask,
            gaussFiltSize=5,
        )
    except cv2.error:
        return None
    if not np.all(np.isfinite(warp)) or not math.isfinite(float(correlation)):
        return None
    warp = _full_resolution_warp(warp, scale, homography)
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    if homography:
        aligned = cv2.warpPerspective(
            reference, warp, (width, height), flags=flags,
            borderMode=cv2.BORDER_REFLECT101)
    else:
        aligned = cv2.warpAffine(
            reference, warp, (width, height), flags=flags,
            borderMode=cv2.BORDER_REFLECT101)
    appearance = _centered_similarity(frame, aligned, valid_mask)
    ecc_score = max(0.0, min(1.0, float(correlation)))
    confidence = 0.6 * ecc_score + 0.4 * appearance
    return aligned, confidence, warp


def _identity_candidate(frame: Any, reference: Any, valid_mask: Any):
    import numpy as np

    confidence = _centered_similarity(frame, reference, valid_mask)
    return reference.copy(), confidence, np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def _match_frame_color(frame: Any, aligned: Any, valid_mask: Any):
    import numpy as np

    valid = valid_mask > 0
    if int(np.count_nonzero(valid)) < 64:
        return aligned, (0.0, 0.0, 0.0)
    target = frame[valid].astype(np.float32)
    source = aligned[valid].astype(np.float32)
    delta = np.median(target - source, axis=0)
    delta = np.clip(delta, -64.0, 64.0)
    matched = np.clip(aligned.astype(np.float32) + delta, 0, 255).astype(np.uint8)
    return matched, tuple(round(float(value), 3) for value in delta)


def apply_clean_reference(
    frame: Any,
    reference: Any,
    final_mask: Any,
    spec: Mapping[str, object],
    *,
    alignment_mask: Any = None,
) -> ReferenceFillResult:
    """Align one clean plate and composite it strictly inside ``final_mask``."""
    import cv2
    import numpy as np

    normalized = normalize_clean_reference(spec)
    if normalized is None:
        raise ValueError("clean reference configuration is invalid")
    if frame is None or reference is None or final_mask is None:
        raise ValueError("clean reference alignment needs frame, plate, and mask")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("clean reference target must be a BGR frame")
    if reference.ndim == 2:
        reference = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
    if reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("clean reference image must have three color channels")
    if reference.shape[:2] != frame.shape[:2]:
        raise ValueError("clean reference dimensions must match the source video")
    mask = np.asarray(final_mask, dtype=np.uint8)
    if mask.shape != frame.shape[:2]:
        raise ValueError("clean reference mask dimensions do not match the frame")
    if not np.any(mask > 0):
        return ReferenceFillResult(
            frame.copy(), reference.copy(), False, normalized["alignment"],
            0.0, (0.0, 0.0, 0.0), [], "final mask is empty")
    excluded = mask
    if alignment_mask is not None:
        excluded = np.asarray(alignment_mask, dtype=np.uint8)
        if excluded.shape != frame.shape[:2]:
            raise ValueError(
                "clean reference alignment mask dimensions do not match the frame")
    valid_mask = np.where(excluded > 0, 0, 255).astype(np.uint8)
    if int(np.count_nonzero(valid_mask)) < 64:
        return ReferenceFillResult(
            frame.copy(), reference.copy(), False, normalized["alignment"],
            0.0, (0.0, 0.0, 0.0), [],
            "too little unmasked image remains for alignment")

    requested = normalized["alignment"]
    methods = [requested] if requested != "auto" else ["translation", "homography"]
    candidates = []
    for method in methods:
        candidate = _ecc_candidate(frame, reference, valid_mask, method)
        if candidate is not None:
            candidates.append((method, *candidate))
    if "translation" in methods:
        identity = _identity_candidate(frame, reference, valid_mask)
        candidates.append(("translation", *identity))
    if not candidates:
        return ReferenceFillResult(
            frame.copy(), reference.copy(), False, requested, 0.0,
            (0.0, 0.0, 0.0), [], "alignment did not converge")
    candidates.sort(key=lambda item: item[2], reverse=True)
    method, aligned, confidence, warp = candidates[0]
    if requested == "auto" and method == "homography":
        translation = next(
            (item for item in candidates if item[0] == "translation"), None)
        if translation is not None and confidence < translation[2] + 0.02:
            method, aligned, confidence, warp = translation
    accepted = confidence >= float(normalized["min_confidence"])
    if not accepted:
        return ReferenceFillResult(
            frame.copy(), aligned, False, method, round(confidence, 6),
            (0.0, 0.0, 0.0), warp.astype(float).round(6).tolist(),
            "alignment confidence is below the configured floor")
    color_delta = (0.0, 0.0, 0.0)
    if normalized["color_match"]:
        aligned, color_delta = _match_frame_color(frame, aligned, valid_mask)
    composite = frame.copy()
    selected = mask > 0
    composite[selected] = aligned[selected]
    return ReferenceFillResult(
        composite,
        aligned,
        True,
        method,
        round(confidence, 6),
        color_delta,
        warp.astype(float).round(6).tolist(),
    )
