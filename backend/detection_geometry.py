"""Normalized OCR geometry shared by detection, tracking, and masks.

The public detector methods still return the historical axis-aligned boxes.
This module carries an optional polygon beside that box so callers can opt
into tighter geometry without invalidating old integrations or saved plans.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]
Point = Tuple[int, int]
Polygon = Tuple[Point, ...]


def normalize_box(
    values: Sequence[float],
    frame_shape: Optional[Sequence[int]] = None,
) -> Optional[Box]:
    """Return a clipped, non-degenerate integer box."""
    try:
        raw = [float(value) for value in values[:4]]
    except (TypeError, ValueError, IndexError):
        return None
    if len(raw) != 4 or not all(np.isfinite(value) for value in raw):
        return None
    x1, y1, x2, y2 = [int(round(value)) for value in raw]
    if frame_shape is not None:
        height, width = int(frame_shape[0]), int(frame_shape[1])
        x1 = max(0, min(width, x1))
        y1 = max(0, min(height, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _point_array(values: Any) -> Optional[np.ndarray]:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.size == 0:
        return None
    if array.ndim == 1:
        if array.size == 4:
            return None
        if array.size < 6 or array.size % 2:
            return None
        return array.reshape(-1, 2)
    if array.ndim == 2 and array.shape[1] == 2:
        return array
    if array.ndim == 2 and array.shape[0] == 1:
        flat = array.reshape(-1)
        if flat.size == 4 or flat.size < 6 or flat.size % 2:
            return None
        return flat.reshape(-1, 2)
    try:
        last = int(array.shape[-1])
        flattened = array.reshape(-1, last)
    except (AttributeError, ValueError):
        return None
    if flattened.shape[1] < 2:
        return None
    return flattened[:, :2]


def normalize_polygon(
    values: Any,
    frame_shape: Optional[Sequence[int]] = None,
) -> Optional[Polygon]:
    """Normalize polygon vertices and clip them to an optional frame."""
    points = _point_array(values)
    if points is None or points.shape[0] < 3:
        return None
    if not np.isfinite(points).all():
        return None
    if frame_shape is not None:
        height, width = int(frame_shape[0]), int(frame_shape[1])
        points[:, 0] = np.clip(points[:, 0], 0, max(0, width - 1))
        points[:, 1] = np.clip(points[:, 1], 0, max(0, height - 1))
    integer_points = [
        (int(round(point[0])), int(round(point[1])))
        for point in points
    ]
    compact: list[Point] = []
    for point in integer_points:
        if not compact or point != compact[-1]:
            compact.append(point)
    if len(compact) > 1 and compact[0] == compact[-1]:
        compact.pop()
    if len(compact) < 3:
        return None
    area = 0.0
    for first, second in zip(compact, compact[1:] + compact[:1], strict=True):
        area += first[0] * second[1] - second[0] * first[1]
    if abs(area) <= 1.0:
        return None
    return tuple(compact)


def polygon_bbox(polygon: Polygon) -> Box:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


@dataclass(frozen=True)
class DetectionGeometry:
    """One OCR result with a legacy box and optional polygon geometry."""

    bbox: Box
    polygon: Optional[Polygon] = None
    confidence: float = 1.0
    text: str = ""
    track_id: Optional[int] = None

    def __post_init__(self) -> None:
        box = tuple(int(value) for value in self.bbox[:4])
        if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("detection bbox must be non-degenerate")
        object.__setattr__(self, "bbox", box)
        if self.polygon is not None:
            object.__setattr__(
                self,
                "polygon",
                tuple((int(x), int(y)) for x, y in self.polygon),
            )
        try:
            confidence = float(self.confidence)
        except (TypeError, ValueError):
            confidence = 1.0
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "text", str(self.text or ""))
        track_id = self.track_id
        if track_id is not None:
            try:
                track_id = int(track_id)
            except (TypeError, ValueError):
                track_id = None
        object.__setattr__(self, "track_id", track_id)

    @classmethod
    def from_box(
        cls,
        values: Sequence[float],
        frame_shape: Optional[Sequence[int]] = None,
        *,
        confidence: float = 1.0,
        text: str = "",
        track_id: Optional[int] = None,
    ) -> Optional["DetectionGeometry"]:
        box = normalize_box(values, frame_shape)
        if box is None:
            return None
        return cls(box, None, confidence, text, track_id)

    @classmethod
    def from_polygon(
        cls,
        values: Any,
        frame_shape: Optional[Sequence[int]] = None,
        *,
        confidence: float = 1.0,
        text: str = "",
        track_id: Optional[int] = None,
    ) -> Optional["DetectionGeometry"]:
        polygon = normalize_polygon(values, frame_shape)
        if polygon is None:
            return None
        return cls(
            polygon_bbox(polygon), polygon, confidence, text, track_id
        )


def geometry_from_coords(
    values: Any,
    frame_shape: Optional[Sequence[int]] = None,
    *,
    polygon: bool = False,
    confidence: float = 1.0,
    text: str = "",
    track_id: Optional[int] = None,
) -> Optional[DetectionGeometry]:
    """Parse either a box or polygon while preserving the requested shape."""
    if polygon:
        result = DetectionGeometry.from_polygon(
            values,
            frame_shape,
            confidence=confidence,
            text=text,
            track_id=track_id,
        )
        if result is not None:
            return result
    try:
        return DetectionGeometry.from_box(
            values,
            frame_shape,
            confidence=confidence,
            text=text,
            track_id=track_id,
        )
    except (TypeError, ValueError):
        return None


def as_detection_geometry(value: Any) -> Optional[DetectionGeometry]:
    """Adapt a new record or a legacy tuple to the normalized record."""
    if isinstance(value, DetectionGeometry):
        return value
    if isinstance(value, dict):
        raw_polygon = value.get("polygon", value.get("poly"))
        confidence = value.get("confidence", value.get("score", 1.0))
        text = value.get("text", "")
        track_id = value.get("track_id", value.get("trackId"))
        if raw_polygon is not None:
            result = DetectionGeometry.from_polygon(
                raw_polygon,
                confidence=confidence,
                text=text,
                track_id=track_id,
            )
            if result is not None:
                return result
        return geometry_from_coords(
            value.get("bbox", value.get("box", ())),
            confidence=confidence,
            text=text,
            track_id=track_id,
        )
    try:
        result = DetectionGeometry.from_box(value[:4])
    except (TypeError, ValueError, IndexError):
        return None
    if result is None:
        return None
    text = ""
    confidence = 1.0
    try:
        if len(value) > 4 and isinstance(value[4], (float, int)):
            confidence = float(value[4])
        for extra in value[4:]:
            if isinstance(extra, str) and extra.strip():
                text = extra.strip()
                break
    except (TypeError, IndexError):
        pass
    return DetectionGeometry(result.bbox, None, confidence, text)


def geometry_mask(
    frame_shape: Sequence[int],
    detection: DetectionGeometry,
    expansion: int = 0,
) -> np.ndarray:
    """Rasterize one detection, expanding its own geometry only."""
    height, width = int(frame_shape[0]), int(frame_shape[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    if detection.polygon:
        points = np.asarray(detection.polygon, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
    else:
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    radius = max(0, int(expansion))
    if radius:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (radius * 2 + 1, radius * 2 + 1),
        )
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def expand_polygon_local(polygon: Polygon, padding: int) -> Polygon:
    """Expand a polygon in its own principal-axis coordinate system."""
    if padding <= 0 or len(polygon) < 3:
        return polygon
    points = np.asarray(polygon, dtype=np.float64)
    center = points.mean(axis=0)
    try:
        _u, _s, vh = np.linalg.svd(points - center, full_matrices=False)
        basis = vh[:2].T
    except np.linalg.LinAlgError:
        return polygon
    local = (points - center) @ basis
    widths = np.ptp(local, axis=0)
    scales = np.ones(2, dtype=np.float64)
    for index, width in enumerate(widths):
        if width > 1.0:
            scales[index] = (width + 2.0 * padding) / width
    expanded = center + (local * scales) @ basis.T
    return tuple(
        (int(round(point[0])), int(round(point[1])))
        for point in expanded
    )


def remap_polygon(
    polygon: Optional[Polygon],
    old_box: Box,
    new_box: Box,
) -> Optional[Polygon]:
    """Move polygon vertices with a tracked box's translation and scale."""
    if not polygon:
        return None
    ox1, oy1, ox2, oy2 = old_box
    nx1, ny1, nx2, ny2 = new_box
    old_width = max(1, ox2 - ox1)
    old_height = max(1, oy2 - oy1)
    new_width = max(1, nx2 - nx1)
    new_height = max(1, ny2 - ny1)
    mapped = []
    for x, y in polygon:
        mapped.append((
            int(round(nx1 + (x - ox1) * new_width / old_width)),
            int(round(ny1 + (y - oy1) * new_height / old_height)),
        ))
    return normalize_polygon(mapped)


__all__ = [
    "Box",
    "DetectionGeometry",
    "Point",
    "Polygon",
    "as_detection_geometry",
    "expand_polygon_local",
    "geometry_from_coords",
    "geometry_mask",
    "normalize_box",
    "normalize_polygon",
    "polygon_bbox",
    "remap_polygon",
]
