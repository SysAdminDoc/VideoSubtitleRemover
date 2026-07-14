"""Versioned mask-correction, review-span, and selective-rerun helpers."""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional

import cv2
import numpy as np


MASK_CORRECTION_SCHEMA = "vsr.mask_correction.v2"
MASK_REVIEW_SCHEMA = "vsr.mask_review_spans.v1"
SELECTIVE_RERUN_SCHEMA = "vsr.selective_mask_rerun.v1"
CORRECTION_MODES = {"add", "subtract"}
REVIEW_KINDS = {"residual", "flicker", "low-confidence"}


def _finite_nonnegative(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result) or result < 0.0:
        return default
    return result


def _optional_frame(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        frame = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, frame)


def normalize_mask_correction(value: Any) -> Optional[dict]:
    """Normalize legacy add-only and v2 ordered correction operations."""
    if not isinstance(value, dict):
        return None
    polygons = value.get("polygons")
    if not isinstance(polygons, (list, tuple)) or not polygons:
        return None
    coerced_polygons = []
    for polygon in polygons:
        if (
            not isinstance(polygon, (list, tuple))
            or len(polygon) < 6
            or len(polygon) % 2
        ):
            continue
        try:
            coords = [int(round(float(coord))) for coord in polygon]
        except (TypeError, ValueError):
            continue
        points = list(zip(coords[::2], coords[1::2]))
        if len(set(points)) < 3:
            continue
        coerced_polygons.append(coords)
    if not coerced_polygons:
        return None

    start = _finite_nonnegative(value.get("start", 0.0))
    end = _finite_nonnegative(value.get("end", 0.0))
    if end and end <= start:
        end = 0.0
    result = {
        "polygons": coerced_polygons,
        "start": start,
        "end": end,
    }
    if "mode" in value or "schema" in value:
        mode = str(value.get("mode") or "add").strip().lower()
        result["schema"] = MASK_CORRECTION_SCHEMA
        result["mode"] = mode if mode in CORRECTION_MODES else "add"
    start_frame = _optional_frame(value.get("start_frame"))
    end_frame = _optional_frame(value.get("end_frame"))
    if start_frame is not None:
        result["start_frame"] = start_frame
    if end_frame is not None and (start_frame is None or end_frame > start_frame):
        result["end_frame"] = end_frame
    propagation = str(value.get("propagation") or "").strip().lower()
    if propagation in {"frame", "span"}:
        result["propagation"] = propagation
    source = str(value.get("source") or "").strip().lower()
    if source in REVIEW_KINDS | {"manual"}:
        result["source"] = source
    return result


def normalize_mask_correction_list(value: Any) -> Optional[list[dict]]:
    if not isinstance(value, (list, tuple)):
        return None
    corrections = [
        correction
        for item in value
        if (correction := normalize_mask_correction(item)) is not None
    ]
    return corrections or None


def correction_is_active(
    correction: dict,
    frame_seconds: float,
    frame_index: Optional[int] = None,
) -> bool:
    start_frame = _optional_frame(correction.get("start_frame"))
    end_frame = _optional_frame(correction.get("end_frame"))
    if frame_index is not None and start_frame is not None:
        return int(frame_index) >= start_frame and (
            end_frame is None or int(frame_index) < end_frame
        )
    start = _finite_nonnegative(correction.get("start", 0.0))
    end = _finite_nonnegative(correction.get("end", 0.0))
    current = _finite_nonnegative(frame_seconds)
    return current >= start and (end <= 0.0 or current < end)


def apply_mask_corrections(
    mask: np.ndarray,
    corrections: Any,
    frame_seconds: float,
    frame_index: Optional[int] = None,
) -> np.ndarray:
    """Compose ordered add/subtract polygons into ``mask`` deterministically."""
    normalized = normalize_mask_correction_list(corrections) or []
    if not normalized:
        return mask
    height, width = mask.shape[:2]
    for correction in normalized:
        if not correction_is_active(correction, frame_seconds, frame_index):
            continue
        fill_value = 0 if correction.get("mode", "add") == "subtract" else 255
        for coords in correction["polygons"]:
            points = np.asarray(
                list(zip(coords[::2], coords[1::2])), dtype=np.int32)
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
            cv2.fillPoly(mask, [points], fill_value)
    return mask


def brush_polygon(
    x: int,
    y: int,
    radius: int,
    frame_width: int,
    frame_height: int,
    vertices: int = 12,
) -> list[int]:
    """Return a bounded polygonal brush dab in source-pixel coordinates."""
    radius = max(1, int(radius))
    count = max(6, int(vertices))
    coords: list[int] = []
    for index in range(count):
        angle = 2.0 * math.pi * index / count
        px = max(0, min(frame_width - 1, int(round(x + math.cos(angle) * radius))))
        py = max(0, min(frame_height - 1, int(round(y + math.sin(angle) * radius))))
        coords.extend((px, py))
    return coords


def make_review_span(
    kind: str,
    start_frame: int,
    end_frame: int,
    *,
    fps: float,
    score: Optional[float] = None,
    threshold: Optional[float] = None,
    reason: str = "",
) -> dict:
    kind = kind if kind in REVIEW_KINDS else "residual"
    start_frame = max(0, int(start_frame))
    end_frame = max(start_frame + 1, int(end_frame))
    safe_fps = max(float(fps), 1e-9)
    span = {
        "schema": MASK_REVIEW_SCHEMA,
        "kind": kind,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start": start_frame / safe_fps,
        "end": end_frame / safe_fps,
        "reason": str(reason or kind),
        "suggested_mode": "subtract" if kind == "flicker" else "add",
    }
    if score is not None and math.isfinite(float(score)):
        span["score"] = round(float(score), 6)
    if threshold is not None and math.isfinite(float(threshold)):
        span["threshold"] = round(float(threshold), 6)
    return span


def merge_review_spans(spans: Iterable[dict], *, gap_frames: int = 1) -> list[dict]:
    """Merge adjacent signals of the same kind into a compact review queue."""
    ordered = sorted(
        (dict(span) for span in spans if isinstance(span, dict)),
        key=lambda span: (
            int(span.get("start_frame", 0)),
            str(span.get("kind", "")),
        ),
    )
    merged: list[dict] = []
    for span in ordered:
        if not merged:
            merged.append(span)
            continue
        previous = merged[-1]
        if (
            previous.get("kind") == span.get("kind")
            and int(span.get("start_frame", 0))
            <= int(previous.get("end_frame", 0)) + max(0, int(gap_frames))
        ):
            previous["end_frame"] = max(
                int(previous.get("end_frame", 0)),
                int(span.get("end_frame", 0)),
            )
            previous["end"] = max(
                float(previous.get("end", 0.0)),
                float(span.get("end", 0.0)),
            )
            scores = [
                float(item["score"])
                for item in (previous, span)
                if item.get("score") is not None
            ]
            if scores:
                previous["score"] = round(max(scores), 6)
            continue
        merged.append(span)
    return merged


def merge_frame_ranges(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted(
        (max(0, int(start)), max(0, int(end)))
        for start, end in ranges
        if int(end) > int(start)
    )
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def frame_is_in_ranges(frame_index: int, ranges: Iterable[tuple[int, int]]) -> bool:
    index = int(frame_index)
    return any(start <= index < end for start, end in ranges)
