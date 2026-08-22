"""RM-275: a reviewable pre-run plan of the temporal text tracks in a video.

A plan is what stands between "detect" and "destroy": one JSON document
listing every temporal text track found in a sampling pass, each with its
frame span, a sample of the recognized text, and a small thumbnail. The
user (or a script) marks tracks to keep, and consuming the plan turns every
kept track into a frame-bounded subtract mask correction, so exactly that
track's span and region are excluded from the inpaint mask while everything
else is handled as usual.

The plan is deterministic for a given file and settings: the same sampling
stride over the same frames produces the same tracks, and applying the same
edited plan reproduces the same corrections. A plan-driven run combined
with ``--export-mask`` yields a matte manifest that ``--frozen-matte`` can
reuse on later runs of the same source.
"""

from __future__ import annotations

import base64
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from backend.detection_geometry import (
    DetectionGeometry,
    as_detection_geometry,
    expand_polygon_local,
)

logger = logging.getLogger(__name__)

TRACK_PLAN_SCHEMA = "vsr.track_plan.v1"

# Sampling density: four looks per second keeps a scan of a feature-length
# file in the tens of seconds on CPU OCR while still catching any subtitle
# that lives on screen for half a second.
DEFAULT_SAMPLE_FPS = 4.0

# A track survives a miss this many consecutive samples long. Detection
# flicker on translucent or moving text is common; one missed sample must
# not split a caption into two tracks.
DEFAULT_GAP_SAMPLES = 2

THUMBNAIL_MAX_WIDTH = 160


def _iou(a: Sequence[int], b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = ((ax2 - ax1) * (ay2 - ay1)
             + (bx2 - bx1) * (by2 - by1) - inter)
    return inter / union if union > 0 else 0.0


class _OpenTrack:
    __slots__ = ("bbox", "start_frame", "last_frame", "texts",
                 "misses", "sample_count", "polygon", "polygon_history")

    def __init__(self, bbox, frame_idx, text, polygon=None):
        self.bbox = list(bbox)
        self.start_frame = frame_idx
        self.last_frame = frame_idx
        self.texts = Counter()
        if text:
            self.texts[text] += 1
        self.misses = 0
        self.sample_count = 1
        self.polygon = polygon
        self.polygon_history = []
        if polygon:
            self.polygon_history.append((int(frame_idx), polygon))

    def absorb(self, bbox, frame_idx, text, polygon=None):
        self.bbox = [
            min(self.bbox[0], bbox[0]), min(self.bbox[1], bbox[1]),
            max(self.bbox[2], bbox[2]), max(self.bbox[3], bbox[3]),
        ]
        self.last_frame = frame_idx
        if text:
            self.texts[text] += 1
        self.misses = 0
        self.sample_count += 1
        if polygon:
            self.polygon = polygon
            self.polygon_history.append((int(frame_idx), polygon))


def group_detections_into_tracks(
    samples: Sequence[Tuple[int, Sequence[Tuple]]],
    *,
    gap_samples: int = DEFAULT_GAP_SAMPLES,
    iou_threshold: float = 0.25,
) -> List[dict]:
    """Group per-sample detections into temporal tracks.

    ``samples`` is an ordered list of ``(frame_index, detections)`` where
    each detection is ``(x1, y1, x2, y2)`` optionally followed by a text
    string in any later position. Pure and deterministic: greedy highest-IoU
    matching, ties broken by track age, so the same input always yields the
    same tracks.
    """
    open_tracks: List[_OpenTrack] = []
    closed: List[_OpenTrack] = []
    for frame_idx, detections in samples:
        boxes = []
        for det in detections:
            detection = as_detection_geometry(det)
            if detection is None:
                continue
            x1, y1, x2, y2 = detection.bbox
            text = ""
            if detection.text.strip():
                text = detection.text.strip()
            else:
                try:
                    for extra in det[4:]:
                        if isinstance(extra, str) and extra.strip():
                            text = extra.strip()
                            break
                except (TypeError, IndexError):
                    pass
            boxes.append((detection, text))

        pairs = []
        for ti, track in enumerate(open_tracks):
            for bi, (detection, _text) in enumerate(boxes):
                score = _iou(track.bbox, detection.bbox)
                if score >= iou_threshold:
                    pairs.append((score, -track.sample_count, ti, bi))
        pairs.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
        used_tracks: set = set()
        used_boxes: set = set()
        for _score, _age, ti, bi in pairs:
            if ti in used_tracks or bi in used_boxes:
                continue
            used_tracks.add(ti)
            used_boxes.add(bi)
            detection, text = boxes[bi]
            open_tracks[ti].absorb(
                detection.bbox,
                frame_idx,
                text,
                detection.polygon,
            )

        for ti, track in enumerate(open_tracks):
            if ti not in used_tracks:
                track.misses += 1
        still_open = []
        for track in open_tracks:
            if track.misses > gap_samples:
                closed.append(track)
            else:
                still_open.append(track)
        open_tracks = still_open

        for bi, (detection, text) in enumerate(boxes):
            if bi not in used_boxes:
                open_tracks.append(_OpenTrack(
                    detection.bbox,
                    frame_idx,
                    text,
                    detection.polygon,
                ))

    closed.extend(open_tracks)
    closed.sort(key=lambda track: (track.start_frame, tuple(track.bbox)))
    tracks = []
    for index, track in enumerate(closed, 1):
        text, _count = (track.texts.most_common(1) or [("", 0)])[0]
        track_data = {
            "id": index,
            "start_frame": int(track.start_frame),
            "end_frame": int(track.last_frame),
            "bbox": [int(v) for v in track.bbox],
            "sample_text": text,
            "sample_count": int(track.sample_count),
            "keep": False,
        }
        if track.polygon:
            track_data["polygon"] = [
                [int(x), int(y)] for x, y in track.polygon
            ]
            track_data["polygon_history"] = [
                {
                    "frame": int(frame),
                    "points": [[int(x), int(y)] for x, y in polygon],
                }
                for frame, polygon in track.polygon_history
            ]
        tracks.append(track_data)
    return tracks


def _thumbnail_b64(frame: np.ndarray, bbox: Sequence[int]) -> str:
    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, int(bbox[0])))
    y1 = max(0, min(height - 1, int(bbox[1])))
    x2 = max(x1 + 1, min(width, int(bbox[2])))
    y2 = max(y1 + 1, min(height, int(bbox[3])))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return ""
    if crop.shape[1] > THUMBNAIL_MAX_WIDTH:
        scale = THUMBNAIL_MAX_WIDTH / crop.shape[1]
        crop = cv2.resize(
            crop, (THUMBNAIL_MAX_WIDTH, max(1, int(crop.shape[0] * scale))),
            interpolation=cv2.INTER_AREA)
    ok, payload = cv2.imencode(".png", crop)
    if not ok:
        return ""
    return base64.b64encode(payload.tobytes()).decode("ascii")


def scan_track_plan(
    video_path: str | Path,
    *,
    detector=None,
    config=None,
    device: str = "cpu",
    lang: str = "en",
    threshold: float = 0.5,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    gap_samples: int = DEFAULT_GAP_SAMPLES,
    thumbnails: bool = True,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Sample the video, detect text, and return a track plan document.

    ``detector`` may be supplied (any object with ``detect_with_text`` or
    ``detect``); otherwise a ``SubtitleDetector`` is built from ``config``
    or from the ``device``/``lang`` arguments.
    """
    source = Path(video_path)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {source}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        stride = max(1, int(round(fps / max(0.1, float(sample_fps)))))

        if detector is None:
            from backend.detection import SubtitleDetector
            if config is not None:
                detector = SubtitleDetector(
                    device=getattr(config, "device", device) or device,
                    lang=getattr(config, "detection_lang", lang) or lang,
                    engine=getattr(config, "detection_engine", "auto"),
                )
                threshold = float(
                    getattr(config, "detection_threshold", threshold)
                    or threshold)
            else:
                detector = SubtitleDetector(device=device, lang=lang)

        samples: List[Tuple[int, list]] = []
        frame_idx = 0
        while True:
            grabbed = cap.grab()
            if not grabbed:
                break
            if frame_idx % stride == 0:
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    break
                detections: list = []
                try:
                    with_geometry = getattr(
                        detector, "detect_with_geometry", None)
                    if callable(with_geometry):
                        detections = list(with_geometry(frame, threshold))
                    if not detections:
                        with_text = getattr(detector, "detect_with_text", None)
                        if callable(with_text):
                            detections = list(with_text(frame, threshold))
                        if not detections:
                            detections = [
                                tuple(box) for box in detector.detect(
                                    frame, threshold)
                            ]
                except Exception:
                    logger.warning(
                        "Track-plan detection failed at frame %d",
                        frame_idx, exc_info=True)
                    detections = []
                samples.append((frame_idx, detections))
                if on_progress is not None:
                    on_progress(frame_idx, total)
            frame_idx += 1
    finally:
        cap.release()

    tracks = group_detections_into_tracks(samples, gap_samples=gap_samples)
    # A sampled end frame undershoots the true end by up to one stride;
    # extend so "exactly its span" errs on covering the track's tail.
    for track in tracks:
        track["end_frame"] = min(
            max(track["end_frame"] + stride - 1, track["end_frame"]),
            max(0, (total - 1) if total else track["end_frame"] + stride - 1),
        )
    if thumbnails and tracks:
        # Second targeted pass: one approximate seek per track. Holding the
        # sampled frames through the scan instead would pin the whole video
        # in memory on a long file.
        _attach_thumbnails(source, tracks)

    return {
        "schema": TRACK_PLAN_SCHEMA,
        "source": str(source),
        "fps": fps,
        "frame_count": total,
        "sample_stride": stride,
        "detector": getattr(detector, "_engine_name", "") or "",
        "tracks": tracks,
    }


def _attach_thumbnails(source: Path, tracks: List[dict]) -> None:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        for track in tracks:
            track["thumbnail_png_base64"] = ""
        return
    try:
        for track in tracks:
            middle = (track["start_frame"] + track["end_frame"]) // 2
            thumbnail = ""
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(middle))
                ok, frame = cap.read()
                if ok and frame is not None:
                    thumbnail = _thumbnail_b64(frame, track["bbox"])
            except cv2.error:
                thumbnail = ""
            track["thumbnail_png_base64"] = thumbnail
    finally:
        cap.release()


def plan_to_mask_corrections(plan: dict, *, pad: int = 4) -> List[dict]:
    """Return subtract corrections for every track marked ``keep``.

    Each kept track becomes one frame-bounded subtract correction over its
    padded bounding box, so exactly that span and region are excluded from
    the inpaint mask. Tracks not marked keep contribute nothing: they stay
    on the normal detect-and-remove path.
    """
    corrections: List[dict] = []
    for track in plan.get("tracks", []):
        if not track.get("keep"):
            continue
        polygon = track.get("polygon")
        if isinstance(polygon, (list, tuple)) and len(polygon) >= 3:
            points = []
            for point in polygon:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    points = []
                    break
                points.append((int(point[0]), int(point[1])))
            if len(points) >= 3:
                expanded = expand_polygon_local(tuple(points), int(pad))
                coords = [value for point in expanded for value in point]
            else:
                coords = None
        else:
            coords = None
        if coords is None:
            x1, y1, x2, y2 = (int(v) for v in track["bbox"])
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = x2 + pad, y2 + pad
            coords = [x1, y1, x2, y1, x2, y2, x1, y2]
        corrections.append({
            "mode": "subtract",
            "polygons": [coords],
            "start_frame": int(track["start_frame"]),
            # correction_is_active treats end_frame as exclusive.
            "end_frame": int(track["end_frame"]) + 1,
            "source": "manual",
        })
    return corrections


def save_track_plan(plan: dict, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="")


def load_track_plan(path: str | Path) -> dict:
    """Load and validate a plan document, normalising track fields."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != TRACK_PLAN_SCHEMA:
        raise ValueError(
            f"not a track plan (expected schema {TRACK_PLAN_SCHEMA})")
    tracks = raw.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError("track plan has no tracks list")
    for index, track in enumerate(tracks):
        if not isinstance(track, dict):
            raise ValueError(f"track {index} is not an object")
        try:
            bbox = [int(v) for v in track["bbox"]]
            start = int(track["start_frame"])
            end = int(track["end_frame"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"track {index} is malformed: {exc}") from exc
        if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError(f"track {index} has an invalid bbox")
        if end < start or start < 0:
            raise ValueError(f"track {index} has an invalid frame span")
        track["bbox"] = bbox
        track["start_frame"] = start
        track["end_frame"] = end
        track["keep"] = bool(track.get("keep"))
        if "polygon" in track:
            detection = DetectionGeometry.from_polygon(track["polygon"])
            if detection is None:
                raise ValueError(f"track {index} has an invalid polygon")
            track["polygon"] = [
                [int(x), int(y)] for x, y in detection.polygon or ()
            ]
        if "polygon_history" in track:
            history = track["polygon_history"]
            if not isinstance(history, list):
                raise ValueError(f"track {index} has invalid polygon history")
            normalized_history = []
            for entry in history:
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"track {index} has invalid polygon history entry")
                polygon_detection = DetectionGeometry.from_polygon(
                    entry.get("points"))
                if polygon_detection is None:
                    raise ValueError(
                        f"track {index} has invalid polygon history points")
                normalized_history.append({
                    "frame": int(entry.get("frame", start)),
                    "points": [
                        [int(x), int(y)]
                        for x, y in polygon_detection.polygon or ()
                    ],
                })
            track["polygon_history"] = normalized_history
    return raw
