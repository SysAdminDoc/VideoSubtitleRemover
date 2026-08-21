"""Region shape and clean-reference methods for SubtitleRemover."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from backend.region_keyframes import region_shapes_at
from backend.safe_image import safe_imread
from backend.tracking import apply_clean_reference

logger = logging.getLogger(__name__)


def _shared_source_evidence(cache: dict, path: str) -> dict:
    """Hash a reference once per path.

    ``dict.setdefault`` evaluates its default eagerly, so using it here would
    still SHA-256 a multi-gigabyte donor once per span.
    """
    from backend.reference_fill import clean_reference_source_evidence

    if path not in cache:
        cache[path] = clean_reference_source_evidence(path)
    return cache[path]


class DonorReference:
    """A donor video standing in for a still clean plate.

    Holds the capture open for the job. A donor slower than the source
    repeats frames, so the last decode is cached; a donor at the same rate
    advances one frame per source frame, so that case reads sequentially
    instead of seeking -- an explicit seek flushes the decoder on every
    long-GOP frame, which is the common case and the expensive one.

    The frame is also scaled to the source size once per decode rather than
    once per lookup.
    """

    def __init__(self, path: str, capture, fps: float, frame_count: int,
                 offset_seconds: float, target_size=None):
        self.path = path
        self.capture = capture
        self.fps = float(fps)
        self.frame_count = int(frame_count)
        self.offset_seconds = float(offset_seconds)
        self.target_size = target_size
        self._cached_index = -1
        self._cached_frame = None

    def frame_at(self, seconds: float):
        """Return the donor frame for a source timestamp, or None."""
        from backend.reference_fill import donor_frame_index

        index = donor_frame_index(seconds, self.offset_seconds, self.fps)
        if index < 0:
            return None
        if self.frame_count > 0 and index >= self.frame_count:
            return None
        if index == self._cached_index and self._cached_frame is not None:
            return self._cached_frame
        try:
            if index == self._cached_index + 1:
                ok, frame = self.capture.read()
            else:
                from backend.processor import _seek_capture_to_frame

                _seek_capture_to_frame(self.capture, index)
                ok, frame = self.capture.read()
        except Exception as exc:
            logger.debug("Donor seek to frame %d failed: %s", index, exc)
            self._cached_index = -1
            self._cached_frame = None
            return None
        if not ok or frame is None:
            self._cached_index = -1
            self._cached_frame = None
            return None
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if self.target_size and frame.shape[1::-1] != tuple(self.target_size):
            frame = cv2.resize(
                frame, tuple(self.target_size), interpolation=cv2.INTER_AREA)
        self._cached_index = index
        self._cached_frame = frame
        return frame

    def release(self) -> None:
        self._cached_frame = None
        try:
            self.capture.release()
        except Exception:
            logger.debug("Donor capture release failed", exc_info=True)


class _CleanRefMixin:
    """Clean-reference image attachment, region shapes, and polygon masks."""

    def _fixed_region_shapes(
        self,
        time_seconds: Optional[float] = None,
    ) -> Optional[List[dict]]:
        """Return explicit manual-region shapes for the current time.

        Timed spans and moving keyframes intentionally override the legacy
        global fields: inactive ranges must stop masking rather than silently
        falling back to a broad global rectangle.
        """
        spans = getattr(self.config, "subtitle_region_spans", None)
        keyframe_tracks = getattr(
            self.config, "subtitle_region_keyframes", None)
        if spans or keyframe_tracks:
            try:
                seconds = float(time_seconds or 0.0)
            except (TypeError, ValueError):
                seconds = 0.0
            if not np.isfinite(seconds) or seconds < 0.0:
                seconds = 0.0
            active: List[dict] = []
            for span in spans or []:
                if not isinstance(span, dict):
                    continue
                rect = span.get("rect")
                if not rect:
                    continue
                try:
                    start = float(span.get("start", 0.0) or 0.0)
                    end = float(span.get("end", 0.0) or 0.0)
                except (TypeError, ValueError):
                    start, end = 0.0, 0.0
                if not np.isfinite(start) or start < 0.0:
                    start = 0.0
                if not np.isfinite(end) or end < 0.0:
                    end = 0.0
                if start <= seconds and (end <= 0.0 or seconds < end):
                    active.append({"rect": tuple(rect)})
            active.extend(region_shapes_at(keyframe_tracks, seconds))
            return active or None
        if self.config.subtitle_areas:
            return [{"rect": tuple(rect)} for rect in self.config.subtitle_areas]
        if self.config.subtitle_area:
            return [{"rect": tuple(self.config.subtitle_area)}]
        return None

    def _clean_reference_requested(self) -> bool:
        spans = getattr(self.config, "subtitle_region_spans", None) or []
        return any(
            isinstance(span, dict) and bool(span.get("clean_reference"))
            for span in spans
        )

    def _initialize_clean_references(self, width: int, height: int) -> None:
        """Load and fingerprint every timed-region clean plate once per job."""
        self._clean_reference_cache = {}
        self._clean_reference_warned = set()
        if not self._clean_reference_requested():
            self.last_clean_reference = {
                "requested": False,
                "status": "not-requested",
            }
            return
        from backend.reference_fill import CLEAN_REFERENCE_SCHEMA

        records = []
        donors_by_path: dict = {}
        evidence_by_path: dict = {}
        spans = getattr(self.config, "subtitle_region_spans", None) or []
        for span_index, span in enumerate(spans):
            spec = span.get("clean_reference") if isinstance(span, dict) else None
            if not spec:
                continue
            kind = str(spec.get("kind", "plate") or "plate")
            if kind == "video":
                # Several spans commonly share one donor. Opening and
                # SHA-256ing a multi-gigabyte file once per span would be
                # minutes of blocking I/O before the first frame decodes.
                donor = donors_by_path.get(spec["path"])
                if donor is None:
                    donor = self._open_donor_reference(spec, width, height)
                    donors_by_path[spec["path"]] = donor
                elif donor.offset_seconds != float(
                        spec.get("offset_seconds", 0.0) or 0.0):
                    donor = self._open_donor_reference(spec, width, height)
                self._clean_reference_cache[span_index] = donor
                donor_fields = {
                    "donorFps": round(donor.fps, 6),
                    "donorFrameCount": donor.frame_count,
                    "offsetSeconds": round(donor.offset_seconds, 6),
                }
            else:
                source = safe_imread(spec["path"])
                if source is None:
                    raise ValueError(
                        "Clean reference image could not be read: "
                        f"{spec['path']}")
                if source.ndim == 2:
                    source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
                if source.shape[:2] != (height, width):
                    raise ValueError(
                        "Clean reference dimensions must match the source "
                        f"video: expected {width}x{height}, got "
                        f"{source.shape[1]}x{source.shape[0]}")
                self._clean_reference_cache[span_index] = source
                donor_fields = {}
            records.append({
                "spanIndex": span_index,
                "startSeconds": float(span.get("start", 0.0)),
                "endSeconds": float(span.get("end", 0.0)),
                "rect": list(span["rect"]),
                "kind": kind,
                "alignment": spec["alignment"],
                "minimumConfidence": float(spec["min_confidence"]),
                "colorMatch": bool(spec["color_match"]),
                "source": _shared_source_evidence(
                    evidence_by_path, spec["path"]),
                **donor_fields,
                "attemptedFrames": 0,
                "unmappedFrames": 0,
                "acceptedFrames": 0,
                "fallbackFrames": 0,
                "methodCounts": {},
                "minimumObservedConfidence": None,
                "maximumObservedConfidence": None,
                "_confidenceTotal": 0.0,
                "_colorDeltaTotal": [0.0, 0.0, 0.0],
            })
        self.last_clean_reference = {
            "schema": CLEAN_REFERENCE_SCHEMA,
            "requested": True,
            "status": "ready",
            "acceptedFrames": 0,
            "fallbackFrames": 0,
            "references": records,
        }

    def _open_donor_reference(self, spec: dict, width: int, height: int):
        """Open a donor video and check it can stand in for this source."""
        from backend.io import _open_capture

        path = spec["path"]
        capture = _open_capture(path)
        if capture is None or not capture.isOpened():
            if capture is not None:
                capture.release()
            raise ValueError(
                f"Clean reference video could not be opened: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        donor_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        donor_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if not np.isfinite(fps) or fps <= 0.0:
            capture.release()
            raise ValueError(
                "Clean reference video reports no usable frame rate: "
                f"{path}")
        if donor_w <= 0 or donor_h <= 0:
            capture.release()
            raise ValueError(
                f"Clean reference video reports invalid dimensions: {path}")
        if (donor_w, donor_h) != (width, height):
            donor_aspect = donor_w / float(donor_h)
            source_aspect = width / float(height)
            if abs(donor_aspect - source_aspect) > 0.02:
                capture.release()
                raise ValueError(
                    "Clean reference video aspect ratio does not match the "
                    f"source: donor {donor_w}x{donor_h}, source "
                    f"{width}x{height}. Stretching it would fail alignment "
                    "on every frame."
                )
            logger.info(
                "Donor reference %sx%s differs from the source %sx%s; "
                "frames will be scaled before alignment",
                donor_w, donor_h, width, height,
            )
        return DonorReference(
            path, capture, fps, frame_count,
            float(spec.get("offset_seconds", 0.0) or 0.0),
            target_size=(width, height),
        )

    def _release_clean_references(self) -> None:
        """Close any donor captures. Safe to call more than once.

        Spans can share one DonorReference, so release each object once.
        """
        released = set()
        for entry in list(
                getattr(self, "_clean_reference_cache", {}).values()):
            if isinstance(entry, DonorReference) and id(entry) not in released:
                released.add(id(entry))
                entry.release()

    def _apply_clean_reference_overrides(
        self,
        frame: np.ndarray,
        final_mask: np.ndarray,
        seconds: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Use active clean plates, leaving rejected pixels for inpainting."""
        if not self._clean_reference_cache or not np.any(final_mask > 0):
            return frame.copy(), final_mask.copy()
        composite = frame.copy()
        remaining = final_mask.copy()
        spans = getattr(self.config, "subtitle_region_spans", None) or []
        records = {
            int(record["spanIndex"]): record
            for record in self.last_clean_reference.get("references", [])
        }
        for span_index, entry in self._clean_reference_cache.items():
            span = spans[span_index]
            start = float(span.get("start", 0.0))
            end = float(span.get("end", 0.0))
            if seconds < start or (end > 0.0 and seconds >= end):
                continue
            x1, y1, x2, y2 = span["rect"]
            x1, x2 = max(0, min(frame.shape[1], x1)), max(
                0, min(frame.shape[1], x2))
            y1, y2 = max(0, min(frame.shape[0], y1)), max(
                0, min(frame.shape[0], y2))
            scoped_mask = np.zeros_like(remaining)
            scoped_mask[y1:y2, x1:x2] = remaining[y1:y2, x1:x2]
            if not np.any(scoped_mask > 0):
                continue
            record = records[span_index]
            if isinstance(entry, DonorReference):
                reference = entry.frame_at(seconds)
                if reference is None:
                    # No donor frame maps here. Leaving the mask intact sends
                    # these pixels down the normal inpaint path, which is the
                    # documented fallback.
                    record["unmappedFrames"] += 1
                    record["fallbackFrames"] += 1
                    record["lastFallbackReason"] = (
                        "no donor frame maps to this timestamp")
                    self.last_clean_reference["fallbackFrames"] += 1
                    continue
            else:
                reference = entry
            result = apply_clean_reference(
                frame,
                reference,
                scoped_mask,
                span["clean_reference"],
                alignment_mask=final_mask,
            )
            record["attemptedFrames"] += 1
            record["_confidenceTotal"] += float(result.confidence)
            record["methodCounts"][result.method] = (
                int(record["methodCounts"].get(result.method, 0)) + 1)
            observed_min = record["minimumObservedConfidence"]
            observed_max = record["maximumObservedConfidence"]
            record["minimumObservedConfidence"] = (
                float(result.confidence) if observed_min is None
                else min(float(observed_min), float(result.confidence)))
            record["maximumObservedConfidence"] = (
                float(result.confidence) if observed_max is None
                else max(float(observed_max), float(result.confidence)))
            if result.accepted:
                record["acceptedFrames"] += 1
                self.last_clean_reference["acceptedFrames"] += 1
                for channel, value in enumerate(result.color_delta):
                    record["_colorDeltaTotal"][channel] += float(value)
                selected = scoped_mask > 0
                composite[selected] = result.composite[selected]
                remaining[selected] = 0
            else:
                record["fallbackFrames"] += 1
                record["lastFallbackReason"] = result.reason
                self.last_clean_reference["fallbackFrames"] += 1
                if span_index not in self._clean_reference_warned:
                    logger.warning(
                        "Clean reference %s fell back to inpainting: %s "
                        "(confidence %.3f)",
                        record["source"]["name"], result.reason,
                        result.confidence,
                    )
                    self._clean_reference_warned.add(span_index)
        return composite, remaining

    def _clean_reference_sidecar_evidence(self) -> Optional[dict]:
        if not self.last_clean_reference.get("requested"):
            return None
        payload = {
            key: value
            for key, value in self.last_clean_reference.items()
            if key != "references"
        }
        references = []
        for record in self.last_clean_reference.get("references", []):
            clean = {
                key: value
                for key, value in record.items()
                if not key.startswith("_")
            }
            attempted = int(record.get("attemptedFrames", 0))
            accepted = int(record.get("acceptedFrames", 0))
            if attempted:
                clean["meanConfidence"] = round(
                    float(record.get("_confidenceTotal", 0.0)) / attempted, 6)
            if accepted:
                totals = record.get("_colorDeltaTotal", [0.0, 0.0, 0.0])
                clean["meanColorDeltaBgr"] = [
                    round(float(value) / accepted, 3) for value in totals
                ]
            references.append(clean)
        payload["references"] = references
        if int(payload.get("acceptedFrames", 0)):
            payload["status"] = "applied"
        elif int(payload.get("fallbackFrames", 0)):
            payload["status"] = "fallback"
        else:
            payload["status"] = "unused"
        return payload

    def _fixed_region_boxes(
        self,
        time_seconds: Optional[float] = None,
    ) -> Optional[List[Tuple[int, int, int, int]]]:
        """Return active rectangle shapes for detection/mask creation."""
        shapes = self._fixed_region_shapes(time_seconds) or []
        boxes = [tuple(shape["rect"]) for shape in shapes if "rect" in shape]
        return boxes or None

    @staticmethod
    def _apply_polygon_region_shapes(
        mask: np.ndarray,
        shapes: Optional[List[dict]],
    ) -> np.ndarray:
        """Fill active polygon keyframes without widening them to bounds."""
        if not shapes:
            return mask
        h, w = mask.shape[:2]
        for shape in shapes:
            coords = shape.get("polygon") if isinstance(shape, dict) else None
            if not isinstance(coords, (list, tuple)) or len(coords) < 6:
                continue
            try:
                points = np.asarray(
                    [(int(coords[i]), int(coords[i + 1]))
                     for i in range(0, len(coords), 2)],
                    dtype=np.int32,
                )
                points[:, 0] = np.clip(points[:, 0], 0, w - 1)
                points[:, 1] = np.clip(points[:, 1], 0, h - 1)
                cv2.fillPoly(mask, [points], 255)
            except (TypeError, ValueError, IndexError):
                continue
        return mask

