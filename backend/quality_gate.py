"""Quality-gate classification for batch reports.

The processor already computes cheap PSNR/SSIM/ROI/VMAF metrics when the
quality report is enabled. This module turns that metric payload into a
small, stable decision object that batch reports can persist and future
fallback ladders can build on.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional


STATUS_PASSED = "passed"
STATUS_REVIEW = "review"
STATUS_UNKNOWN = "unknown"
STATUS_NOT_APPLICABLE = "not_applicable"

LADDER_NONE = "none"
LADDER_MANUAL_REVIEW = "manual-review"
LADDER_NOT_RUN = "not-run"
LADDER_NOT_APPLICABLE = "not-applicable"

SSIM_FLOOR = 0.95
VMAF_FLOOR = 90.0
TEMPORAL_FLICKER_CEILING = 0.08


def evaluate_quality_gate(metrics: Optional[dict]) -> Dict[str, Any]:
    """Return a stable gate result for a quality metric payload."""

    if not metrics:
        return quality_gate_unknown("quality report not available")
    samples = _number(metrics.get("samples"))
    if samples is not None and samples <= 0:
        return quality_gate_unknown("quality report has no sampled frames")

    reasons = []
    tag = str(metrics.get("tag") or "").strip().lower()
    if tag == "review":
        reasons.append("quality report tag is Review")

    roi_ssim = _number(metrics.get("roi_ssim"))
    frame_ssim = _number(metrics.get("ssim"))
    if roi_ssim is not None:
        if roi_ssim < SSIM_FLOOR:
            reasons.append(f"ROI SSIM {roi_ssim:.4f} below {SSIM_FLOOR:.4f}")
    elif frame_ssim is not None and frame_ssim < SSIM_FLOOR:
        reasons.append(f"frame SSIM {frame_ssim:.4f} below {SSIM_FLOOR:.4f}")

    roi_vmaf = _number(metrics.get("roi_vmaf"))
    frame_vmaf = _number(metrics.get("vmaf"))
    if roi_vmaf is not None:
        if roi_vmaf < VMAF_FLOOR:
            reasons.append(f"ROI VMAF {roi_vmaf:.2f} below {VMAF_FLOOR:.2f}")
    elif frame_vmaf is not None and frame_vmaf < VMAF_FLOOR:
        reasons.append(f"frame VMAF {frame_vmaf:.2f} below {VMAF_FLOOR:.2f}")

    flicker = _number(metrics.get("temporal_flicker_score"))
    if flicker is not None and flicker > TEMPORAL_FLICKER_CEILING:
        reasons.append(
            f"temporal flicker {flicker:.4f} above "
            f"{TEMPORAL_FLICKER_CEILING:.4f}"
        )

    previews = _preview_paths(metrics)
    if reasons:
        return {
            "status": STATUS_REVIEW,
            "ladderStep": LADDER_MANUAL_REVIEW,
            "reason": "; ".join(reasons),
            "previewFramePaths": previews,
        }
    return {
        "status": STATUS_PASSED,
        "ladderStep": LADDER_NONE,
        "reason": "quality metrics passed configured thresholds",
        "previewFramePaths": previews,
    }


def quality_gate_unknown(reason: str) -> Dict[str, Any]:
    return {
        "status": STATUS_UNKNOWN,
        "ladderStep": LADDER_NOT_RUN,
        "reason": reason,
        "previewFramePaths": [],
    }


def quality_gate_not_applicable(reason: str) -> Dict[str, Any]:
    return {
        "status": STATUS_NOT_APPLICABLE,
        "ladderStep": LADDER_NOT_APPLICABLE,
        "reason": reason,
        "previewFramePaths": [],
    }


def _preview_paths(metrics: dict) -> list[str]:
    paths = []
    sheet = metrics.get("sheet")
    if sheet:
        paths.append(str(sheet))
    for key in ("preview_frame_paths", "previewFramePaths"):
        value = metrics.get(key)
        if isinstance(value, (list, tuple)):
            for item in value:
                if item:
                    paths.append(str(item))
    return sorted(set(paths))


def _number(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result
