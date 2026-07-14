"""Validation and interpolation for moving manual-region keyframes."""

from __future__ import annotations

import math
from typing import Iterable, Optional


def _finite_nonnegative(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result) or result < 0.0:
        return default
    return result


def _coerce_rect(value) -> Optional[list[int]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(part))) for part in value]
    except (TypeError, ValueError):
        return None
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = max(0, x2), max(0, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _coerce_polygon(value) -> Optional[list[int]]:
    if not isinstance(value, (list, tuple)) or len(value) < 6 or len(value) % 2:
        return None
    try:
        coords = [max(0, int(round(float(part)))) for part in value]
    except (TypeError, ValueError):
        return None
    points = list(zip(coords[::2], coords[1::2]))
    if len(set(points)) < 3:
        return None
    return coords


def _coerce_shape(value) -> Optional[dict]:
    if not isinstance(value, dict):
        return None
    rect = _coerce_rect(value.get("rect"))
    if rect is not None:
        return {"rect": rect}
    polygon = _coerce_polygon(value.get("polygon"))
    if polygon is not None:
        return {"polygon": polygon}
    return None


def normalize_region_keyframe_track(value) -> Optional[dict]:
    """Return a deterministic keyframe-track payload or ``None``.

    A track contains two or more unique-time keyframes. All keyframes use the
    same shape kind; polygons also keep the same vertex count so interpolation
    cannot silently change topology.
    """
    if not isinstance(value, dict):
        return None
    raw_keyframes = value.get("keyframes")
    if not isinstance(raw_keyframes, (list, tuple)):
        return None

    by_time: dict[float, dict] = {}
    for raw in raw_keyframes:
        if not isinstance(raw, dict):
            continue
        shape = _coerce_shape(raw)
        if shape is None:
            continue
        seconds = _finite_nonnegative(
            raw.get("time", raw.get("seconds", raw.get("at", 0.0))))
        by_time[seconds] = {"time": seconds, **shape}
    keyframes = [by_time[key] for key in sorted(by_time)]
    if len(keyframes) < 2:
        return None

    shape_kind = "rect" if "rect" in keyframes[0] else "polygon"
    shape_size = len(keyframes[0][shape_kind])
    if any(
        shape_kind not in keyframe
        or len(keyframe[shape_kind]) != shape_size
        for keyframe in keyframes
    ):
        return None

    first_time = keyframes[0]["time"]
    last_time = keyframes[-1]["time"]
    start = _finite_nonnegative(value.get("start", first_time), first_time)
    end = _finite_nonnegative(value.get("end", last_time), last_time)
    start = min(start, first_time)
    end = max(end, last_time)
    if end <= start:
        return None
    return {
        "start": start,
        "end": end,
        "keyframes": keyframes,
    }


def normalize_region_keyframe_tracks(value) -> Optional[list[dict]]:
    if not isinstance(value, (list, tuple)):
        return None
    tracks = []
    for item in value:
        track = normalize_region_keyframe_track(item)
        if track is not None:
            tracks.append(track)
    return tracks or None


def _interpolate_values(first: Iterable[int], second: Iterable[int], ratio: float) -> list[int]:
    return [
        int(round(left + (right - left) * ratio))
        for left, right in zip(first, second)
    ]


def region_shapes_at(value, seconds: float) -> list[dict]:
    """Interpolate every active moving-region track at ``seconds``."""
    tracks = normalize_region_keyframe_tracks(value) or []
    current = _finite_nonnegative(seconds)
    shapes: list[dict] = []
    for track in tracks:
        start = float(track["start"])
        end = float(track["end"])
        if current < start or current > end:
            continue
        keyframes = track["keyframes"]
        if current <= keyframes[0]["time"]:
            source = keyframes[0]
            kind = "rect" if "rect" in source else "polygon"
            shapes.append({kind: list(source[kind])})
            continue
        if current >= keyframes[-1]["time"]:
            source = keyframes[-1]
            kind = "rect" if "rect" in source else "polygon"
            shapes.append({kind: list(source[kind])})
            continue
        for left, right in zip(keyframes, keyframes[1:]):
            if left["time"] <= current <= right["time"]:
                duration = max(1e-12, right["time"] - left["time"])
                ratio = (current - left["time"]) / duration
                kind = "rect" if "rect" in left else "polygon"
                shapes.append({
                    kind: _interpolate_values(left[kind], right[kind], ratio),
                })
                break
    return shapes


def shape_bounds(shape: dict) -> Optional[tuple[int, int, int, int]]:
    """Return an axis-aligned source-pixel bound for a normalized shape."""
    if not isinstance(shape, dict):
        return None
    rect = _coerce_rect(shape.get("rect"))
    if rect is not None:
        return tuple(rect)
    polygon = _coerce_polygon(shape.get("polygon"))
    if polygon is None:
        return None
    xs = polygon[::2]
    ys = polygon[1::2]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2
