"""Bounded geometry and undo helpers for manual-region editors."""

from __future__ import annotations

import copy
import math
import re
from typing import Any, Optional


def _integer(value: Any, label: str) -> int:
    """Parse an integer field without silently truncating decimal input."""
    text = str(value).strip()
    if not re.fullmatch(r"[+-]?\d+", text):
        raise ValueError(f"{label} must be a whole number")
    return int(text)


def rect_from_xywh(
    x: Any,
    y: Any,
    width: Any,
    height: Any,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    """Validate numeric rectangle fields and return exclusive-edge bounds."""
    x_value = _integer(x, "X")
    y_value = _integer(y, "Y")
    width_value = _integer(width, "Width")
    height_value = _integer(height, "Height")
    if x_value < 0 or y_value < 0:
        raise ValueError("X and Y must be zero or greater")
    if width_value < 1 or height_value < 1:
        raise ValueError("Width and height must be at least 1 pixel")
    if x_value + width_value > frame_width:
        raise ValueError(f"Rectangle exceeds the {frame_width}-pixel frame width")
    if y_value + height_value > frame_height:
        raise ValueError(f"Rectangle exceeds the {frame_height}-pixel frame height")
    return (
        x_value,
        y_value,
        x_value + width_value,
        y_value + height_value,
    )


def parse_polygon_vertices(
    value: Any,
    frame_width: int,
    frame_height: int,
) -> list[int]:
    """Parse ``x,y; x,y`` vertices and enforce source-frame bounds."""
    text = str(value).strip()
    if not text:
        raise ValueError("Enter at least three polygon vertices")
    coords: list[int] = []
    for index, pair in enumerate(text.split(";"), start=1):
        parts = [part.strip() for part in pair.split(",")]
        if len(parts) != 2:
            raise ValueError(
                f"Vertex {index} must use x,y format; separate vertices with semicolons"
            )
        px = _integer(parts[0], f"Vertex {index} X")
        py = _integer(parts[1], f"Vertex {index} Y")
        if not 0 <= px <= frame_width or not 0 <= py <= frame_height:
            raise ValueError(
                f"Vertex {index} must stay within 0,{frame_width} by 0,{frame_height}"
            )
        coords.extend((px, py))
    points = list(zip(coords[::2], coords[1::2]))
    if len(points) < 3 or len(set(points)) < 3:
        raise ValueError("A polygon needs at least three distinct vertices")
    if max(coords[::2]) <= min(coords[::2]) or max(coords[1::2]) <= min(coords[1::2]):
        raise ValueError("Polygon vertices must enclose a non-zero area")
    return coords


def format_polygon_vertices(coords: list[int] | tuple[int, ...]) -> str:
    """Format a flattened polygon for deterministic numeric editing."""
    return "; ".join(
        f"{int(x)},{int(y)}" for x, y in zip(coords[::2], coords[1::2])
    )


def _translate_axis(values: list[int], delta: int, maximum: int) -> list[int]:
    low = min(values)
    high = max(values)
    bounded_delta = max(-low, min(delta, maximum - high))
    return [value + bounded_delta for value in values]


def transform_region_shape(
    shape: dict,
    *,
    frame_width: int,
    frame_height: int,
    dx: int = 0,
    dy: int = 0,
    dw: int = 0,
    dh: int = 0,
) -> dict:
    """Nudge or resize a rectangle/polygon while preserving valid bounds."""
    if "rect" in shape:
        x1, y1, x2, y2 = [int(value) for value in shape["rect"]]
        width = x2 - x1
        height = y2 - y1
        x1 = max(0, min(x1 + int(dx), frame_width - width))
        y1 = max(0, min(y1 + int(dy), frame_height - height))
        width = max(1, min(width + int(dw), frame_width - x1))
        height = max(1, min(height + int(dh), frame_height - y1))
        return {"rect": [x1, y1, x1 + width, y1 + height]}

    coords = [int(value) for value in shape.get("polygon", [])]
    if len(coords) < 6 or len(coords) % 2:
        raise ValueError("Polygon shape is invalid")
    xs = _translate_axis(coords[::2], int(dx), frame_width)
    ys = _translate_axis(coords[1::2], int(dy), frame_height)
    if dw:
        left, right = min(xs), max(xs)
        old_width = max(1, right - left)
        new_width = max(1, min(old_width + int(dw), frame_width - left))
        xs = [left + int(round((x - left) * new_width / old_width)) for x in xs]
    if dh:
        top, bottom = min(ys), max(ys)
        old_height = max(1, bottom - top)
        new_height = max(1, min(old_height + int(dh), frame_height - top))
        ys = [top + int(round((y - top) * new_height / old_height)) for y in ys]
    return {"polygon": [coord for point in zip(xs, ys) for coord in point]}


def seconds_to_frame(value: Any, fps: float) -> int:
    """Convert a finite, nonnegative second value to its nearest frame."""
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Time must be a number") from exc
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Time must be zero or greater")
    return int(round(seconds * max(float(fps), 1e-9)))


def frame_to_seconds(value: Any, fps: float) -> float:
    """Convert a nonnegative integer frame index to seconds."""
    frame = _integer(value, "Frame")
    if frame < 0:
        raise ValueError("Frame must be zero or greater")
    return frame / max(float(fps), 1e-9)


class RegionEditHistory:
    """Small snapshot history shared by pointer, numeric, and keyboard edits."""

    def __init__(self, limit: int = 100):
        self.limit = max(1, int(limit))
        self._undo: list[Any] = []
        self._redo: list[Any] = []

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def record(self, snapshot: Any) -> None:
        state = copy.deepcopy(snapshot)
        if self._undo and self._undo[-1] == state:
            return
        self._undo.append(state)
        del self._undo[:-self.limit]
        self._redo.clear()

    def undo(self, current: Any) -> Optional[Any]:
        if not self._undo:
            return None
        self._redo.append(copy.deepcopy(current))
        return copy.deepcopy(self._undo.pop())

    def redo(self, current: Any) -> Optional[Any]:
        if not self._redo:
            return None
        self._undo.append(copy.deepcopy(current))
        return copy.deepcopy(self._redo.pop())
