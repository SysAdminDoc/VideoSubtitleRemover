"""RM-325: ask the detector whether the text is actually gone.

The product measures residue with a contrast heuristic that says of itself
that it is not a replacement for OCR. Meanwhile a detector is loaded for
every job and never asked the one question that matters: run it over the
repaired region and see whether anything is still there. That is the
standard success check in the scene-text-removal literature.

It costs two detector passes over frames that are decoded anyway: one over
the repaired region in the output, and one over the same region in the
source, because "what fraction of the text survived" cannot be answered
without knowing what was there. `evidence()["detectorPasses"]` reports the
count so the cost is on the record rather than buried in the quality stage's
wall clock.

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
# The ROI is the union bounding box of every mask in the clip, so it also
# contains scene text that was never inside a mask and was never meant to be
# removed: a shop sign, a scoreboard, a watermark outside the subtitle band.
# That text is detected in both the source and the output, matches itself,
# and used to count as "surviving". A box whose pixels are the same in both
# frames was not touched by the repair, so it is not a removal failure. Video
# re-encoding moves untouched pixels by a point or two; inpainted text strokes
# move by tens of levels, so the two cases do not overlap.
#
# The pixel test is only the fallback. When the per-frame mask is on hand it
# answers the question directly and without ambiguity: a box the mask never
# covered was never going to be removed. The pixel test cannot tell a frame
# with no subtitle on screen from a repair that did nothing at all, so when
# it has nothing to go on the frame is reported unchecked rather than clean.
UNTOUCHED_MEAN_ABS_DIFF = 3.0
# Fraction of a detected box that the mask has to cover before the box counts
# as text the product set out to remove.
MASK_OVERLAP_MIN = 0.10


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


def _mean_abs_difference(first: np.ndarray, second: np.ndarray,
                         box: Sequence[float]) -> Optional[float]:
    """Mean per-channel difference inside one box, or None if unmeasurable."""
    if first is None or second is None:
        return None
    a = np.asarray(first)
    b = np.asarray(second)
    if a.shape != b.shape or a.ndim < 2:
        return None
    x1, y1, x2, y2 = (int(value) for value in box[:4])
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(a.shape[1], x2)
    y2 = min(a.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop_a = a[y1:y2, x1:x2].astype(np.float32)
    crop_b = b[y1:y2, x1:x2].astype(np.float32)
    if crop_a.size == 0:
        return None
    return float(np.mean(np.abs(crop_a - crop_b)))


def _is_untouched(source_frame, frame, box,
                  tolerance: float = UNTOUCHED_MEAN_ABS_DIFF) -> bool:
    difference = _mean_abs_difference(source_frame, frame, box)
    return difference is not None and difference <= tolerance


def _mask_coverage(mask: np.ndarray, box: Sequence[float]) -> Optional[float]:
    """Fraction of one box that the frame's mask covers."""
    if mask is None:
        return None
    values = np.asarray(mask)
    if values.ndim == 3:
        values = values[:, :, 0]
    if values.ndim != 2 or values.size == 0:
        return None
    x1, y1, x2, y2 = (int(value) for value in box[:4])
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(values.shape[1], x2)
    y2 = min(values.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = values[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    return float(np.count_nonzero(crop) / crop.size)


def verify_frame_removal(
    detector: Any,
    frame: np.ndarray,
    roi: Optional[Sequence[int]],
    *,
    source_frame: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    mask_available: bool = False,
    confidence: float = SURVIVING_DETECTION_CONFIDENCE,
    match_iou: float = MATCH_IOU,
) -> dict:
    """Re-detect inside one repaired region and report what survived.

    `roi` is in frame coordinates and is cropped before detection, so the
    detector only ever sees the region the product claims to have repaired.
    Returned boxes are translated back into frame coordinates.

    `roi` is the union bounding box of every mask in the clip, so it also
    holds text no mask ever covered. Pass `mask` (this frame's mask, in
    frame coordinates) to separate the two exactly; pass
    `mask_available=True` with `mask=None` to say this frame genuinely had
    no mask. Without either, the pixel fallback applies.
    """
    result = {
        "checked": False,
        "reason": "",
        "roi": None,
        "detections": [],
        "survivingSourceBoxes": 0,
        "sourceBoxes": 0,
        "untouchedSourceBoxes": 0,
        "sourceScanned": False,
        "maskUsed": False,
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

    # Text the repair never touched is not text the repair failed to
    # remove. Splitting these first keeps it out of both the numerator and
    # the denominator, and out of the review spans built from `detections`.
    considered = []
    untouched = []
    if mask is not None or mask_available:
        # The mask says which text the product set out to remove, so the
        # answer does not depend on what the pixels did afterwards. A frame
        # with no mask has nothing that was meant to go.
        for source in source_boxes:
            coverage = _mask_coverage(mask, source["box"]) or 0.0
            if coverage < MASK_OVERLAP_MIN:
                untouched.append(source)
            else:
                considered.append(source)
        result["maskUsed"] = True
    elif source_boxes and source_frame is not None:
        roi_change = _mean_abs_difference(
            source_frame, array, (x1, y1, x2, y2))
        if roi_change is not None and roi_change <= UNTOUCHED_MEAN_ABS_DIFF:
            # Nothing in the region moved. That is either a frame with no
            # subtitle on it or a repair that did nothing, and without a
            # mask there is no way to tell which. Saying "clean" here would
            # turn a total failure into a pass.
            result["reason"] = (
                "the repaired region is identical to the source, so scene "
                "text cannot be told apart from text that was missed"
            )
            result["roi"] = [x1, y1, x2, y2]
            return result
        for source in source_boxes:
            if _is_untouched(source_frame, array, source["box"]):
                untouched.append(source)
            else:
                considered.append(source)
    else:
        considered = list(source_boxes)

    surviving = 0
    for source in considered:
        if any(_iou(source["box"], item["box"]) >= match_iou
               for item in detections):
            surviving += 1

    if untouched:
        detections = [
            item for item in detections
            if not any(_iou(item["box"], skip["box"]) >= match_iou
                       for skip in untouched)
        ]

    result.update({
        "checked": True,
        "roi": [x1, y1, x2, y2],
        "detections": detections,
        "survivingSourceBoxes": surviving,
        "sourceBoxes": len(considered),
        "untouchedSourceBoxes": len(untouched),
        "sourceScanned": source_frame is not None,
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
              source_frame: Optional[np.ndarray] = None,
              mask: Optional[np.ndarray] = None,
              mask_available: bool = False) -> dict:
        started = time.perf_counter()
        outcome = verify_frame_removal(
            self.detector, frame, roi,
            source_frame=source_frame,
            mask=mask,
            mask_available=mask_available,
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
        untouched_total = sum(
            int(item.get("untouchedSourceBoxes") or 0) for item in checked)
        source_scanned = [item for item in checked if item.get("sourceScanned")]
        mask_used = [item for item in checked if item.get("maskUsed")]
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
            # Detected in both frames with the pixels unchanged between
            # them: scene text inside the mask bounding box that no mask
            # ever covered.
            "untouchedSourceBoxes": untouched_total,
            "sourceScannedFrames": len(source_scanned),
            # Frames where the mask itself separated text that was meant to
            # go from scene text that was not, rather than the pixel
            # fallback.
            "maskCheckedFrames": len(mask_used),
            # One pass over the output region, and one over the same region
            # in the source when the source frame is available.
            "detectorPasses": len(checked) + len(source_scanned),
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
                    "untouchedSourceBoxes": item.get(
                        "untouchedSourceBoxes", 0),
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
    if evidence.get("sourceBoxes"):
        # What was there was measured and enough of it went. A stray box on
        # inpainted texture is not a second, stricter test run past the
        # tolerance that was just satisfied.
        return False
    # Nothing to match against: either no source frame was available, or
    # every box in the source was scene text the repair never touched. The
    # direct question is what is left: is there text in the region the
    # product says it repaired? The same one-in-ten tolerance applies, so a
    # single jittery detection does not fail a clip on its own.
    checked = int(evidence.get("framesChecked") or 0)
    flagged = int(evidence.get("framesWithSurvivingText") or 0)
    if not checked:
        return False
    return (flagged / checked) > SURVIVING_DETECTION_FRACTION
