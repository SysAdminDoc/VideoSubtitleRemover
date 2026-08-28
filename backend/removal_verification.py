"""RM-325: ask the detector whether the text is actually gone.

The product measures residue with a contrast heuristic that says of itself
that it is not a replacement for OCR. Meanwhile a detector is loaded for
every job and never asked the one question that matters: run it over the
repaired region and see whether anything is still there. That is the
standard success check in the scene-text-removal literature, and it costs
one more detector pass over frames already sampled for quality.

Nothing here guesses. A frame that could not be checked is reported as
unchecked rather than as clean.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

REMOVAL_VERIFICATION_SCHEMA = "vsr.removal_verification.v1"

# A detection inside the repaired region at or above this confidence is
# treated as surviving text. The detector's own default operating point is
# 0.5; this sits above it so a marginal, low-confidence box on inpainted
# texture does not fail an otherwise clean job.
SURVIVING_DETECTION_CONFIDENCE = 0.6
# Fraction of the original detections that may still be matched in the
# output before the job is flagged. Zero would fail on a single ambiguous
# box; this allows one in ten to survive re-detection before it is a
# removal failure worth a human look.
SURVIVING_DETECTION_FRACTION = 0.10
# Two boxes are "the same text" at or above this intersection over union.
MATCH_IOU = 0.30


def _iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in first[:4])
    bx1, by1, bx2, by2 = (float(value) for value in second[:4])
    left = max(ax1, bx1)
    top = max(ay1, by1)
    right = min(ax2, bx2)
    bottom = min(ay2, by2)
    if right <= left or bottom <= top:
        return 0.0
    overlap = (right - left) * (bottom - top)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0


def _clamp_roi(roi: Sequence[int], width: int, height: int):
    x1, y1, x2, y2 = (int(value) for value in roi[:4])
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def _detect_in_roi(reader, frame: np.ndarray, roi, confidence: float) -> list:
    """Detect inside one region and return boxes in frame coordinates."""
    x1, y1, x2, y2 = roi
    crop = np.ascontiguousarray(np.asarray(frame)[y1:y2, x1:x2])
    found = reader(crop)
    boxes = []
    for box in found or []:
        try:
            bx1, by1, bx2, by2 = (int(value) for value in box[:4])
            score = float(box[4]) if len(box) > 4 else 1.0
        except (TypeError, ValueError):
            continue
        if score < confidence:
            continue
        boxes.append({
            "box": [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1],
            "confidence": round(score, 6),
        })
    return boxes


def verify_frame_removal(
    detector: Any,
    frame: np.ndarray,
    roi: Optional[Sequence[int]],
    *,
    source_frame: Optional[np.ndarray] = None,
    confidence: float = SURVIVING_DETECTION_CONFIDENCE,
    match_iou: float = MATCH_IOU,
) -> dict:
    """Re-detect inside one repaired region and report what survived.

    `roi` is in frame coordinates and is cropped before detection, so the
    detector only ever sees the region the product claims to have repaired.
    Returned boxes are translated back into frame coordinates.
    """
    result = {
        "checked": False,
        "reason": "",
        "roi": None,
        "detections": [],
        "survivingSourceBoxes": 0,
        "sourceBoxes": 0,
    }
    if detector is None:
        result["reason"] = "no detector was loaded for this job"
        return result
    reader = getattr(detector, "detect_with_confidence", None)
    if not callable(reader):
        result["reason"] = "the detector cannot report confidences"
        return result
    array = np.asarray(frame) if frame is not None else None
    if array is None or array.size == 0 or array.ndim < 2:
        result["reason"] = "the frame could not be read"
        return result

    height, width = array.shape[:2]
    x1, y1, x2, y2 = _clamp_roi(
        roi if roi is not None else (0, 0, width, height), width, height)
    crop = array[y1:y2, x1:x2]
    # The detector needs enough pixels to find a stroke at all; below this a
    # "no detections" answer means nothing.
    if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
        result["reason"] = "the repaired region is too small to re-detect"
        result["roi"] = [x1, y1, x2, y2]
        return result

    try:
        detections = _detect_in_roi(
            reader, array, (x1, y1, x2, y2), confidence)
        # The same region in the source frame gives the before picture, so
        # "what fraction survived" is measured against this clip rather than
        # against a mask the frame loop happened to record.
        source_boxes = (
            _detect_in_roi(reader, source_frame, (x1, y1, x2, y2), confidence)
            if source_frame is not None else []
        )
    except Exception as exc:
        result["reason"] = f"re-detection failed: {exc}"[:300]
        result["roi"] = [x1, y1, x2, y2]
        return result

    surviving = 0
    for source in source_boxes:
        if any(_iou(source["box"], item["box"]) >= match_iou
               for item in detections):
            surviving += 1
    result["sourceBoxes"] = len(source_boxes)

    result.update({
        "checked": True,
        "roi": [x1, y1, x2, y2],
        "detections": detections,
        "survivingSourceBoxes": surviving,
    })
    return result


class RemovalVerifier:
    """Accumulate per-frame re-detection results for one job."""

    def __init__(self, detector: Any, *,
                 confidence: float = SURVIVING_DETECTION_CONFIDENCE,
                 match_iou: float = MATCH_IOU):
        self.detector = detector
        self.confidence = confidence
        self.match_iou = match_iou
        self.frames: list[dict] = []
        self.seconds = 0.0

    def check(self, frame_index: int, frame: np.ndarray,
              roi: Optional[Sequence[int]],
              source_frame: Optional[np.ndarray] = None) -> dict:
        started = time.perf_counter()
        outcome = verify_frame_removal(
            self.detector, frame, roi,
            source_frame=source_frame,
            confidence=self.confidence,
            match_iou=self.match_iou,
        )
        self.seconds += time.perf_counter() - started
        outcome["frame"] = int(frame_index)
        self.frames.append(outcome)
        return outcome

    def evidence(self) -> dict:
        checked = [item for item in self.frames if item.get("checked")]
        with_text = [item for item in checked if item["detections"]]
        source_total = sum(int(item.get("sourceBoxes") or 0) for item in checked)
        surviving_total = sum(
            int(item.get("survivingSourceBoxes") or 0) for item in checked)
        return {
            "schema": REMOVAL_VERIFICATION_SCHEMA,
            "ran": bool(self.frames),
            "framesSampled": len(self.frames),
            "framesChecked": len(checked),
            "framesUnchecked": len(self.frames) - len(checked),
            "uncheckedReasons": sorted({
                str(item.get("reason") or "")
                for item in self.frames if not item.get("checked")
            } - {""}),
            "framesWithSurvivingText": len(with_text),
            "detectionCount": sum(len(item["detections"]) for item in checked),
            "maxConfidence": max(
                (detection["confidence"]
                 for item in checked for detection in item["detections"]),
                default=None,
            ),
            "sourceBoxes": source_total,
            "survivingSourceBoxes": surviving_total,
            "survivingFraction": (
                surviving_total / source_total if source_total else None),
            "confidenceThreshold": self.confidence,
            "matchIou": self.match_iou,
            "survivingFractionThreshold": SURVIVING_DETECTION_FRACTION,
            "seconds": round(self.seconds, 6),
            "frames": [
                {
                    "frame": item["frame"],
                    "roi": item.get("roi"),
                    "detections": item.get("detections", []),
                    "survivingSourceBoxes": item.get("survivingSourceBoxes", 0),
                }
                for item in checked if item["detections"]
            ],
        }


def verification_failed(evidence: Optional[dict]) -> bool:
    """Whether re-detection says text survived the repair."""
    if not isinstance(evidence, dict) or not evidence.get("ran"):
        return False
    if not evidence.get("framesChecked"):
        return False
    fraction = evidence.get("survivingFraction")
    if fraction is not None and fraction > SURVIVING_DETECTION_FRACTION:
        return True
    # No source boxes to match against still leaves the direct question: is
    # there text in the region the product says it repaired?
    return bool(
        evidence.get("sourceBoxes") == 0
        and evidence.get("framesWithSurvivingText")
    )
