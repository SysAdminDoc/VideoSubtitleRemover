"""Quality-gate classification for batch reports.

The processor already computes cheap PSNR/SSIM/ROI/VMAF metrics when the
quality report is enabled. This module turns that metric payload into a
small, stable decision object that batch reports can persist.

The fallback ladder classifies each metric violation into a specific
remediation bucket. When multiple violations are present, the most
severe ladder step wins. The report includes the step taken, reasons
for every triggered check, and an actionable remediation suggestion.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence


STATUS_PASSED = "passed"
STATUS_REVIEW = "review"
STATUS_UNKNOWN = "unknown"
STATUS_NOT_APPLICABLE = "not_applicable"

LADDER_NONE = "none"
LADDER_INCREASE_DILATION = "increase-dilation"
LADDER_TEMPORAL_SMOOTH = "temporal-smooth"
LADDER_ALTERNATE_INPAINTER = "alternate-inpainter"
LADDER_MANUAL_REVIEW = "manual-review"
LADDER_NOT_RUN = "not-run"
LADDER_NOT_APPLICABLE = "not-applicable"

_LADDER_SEVERITY = {
    LADDER_NONE: 0,
    LADDER_INCREASE_DILATION: 1,
    LADDER_TEMPORAL_SMOOTH: 2,
    LADDER_ALTERNATE_INPAINTER: 3,
    LADDER_MANUAL_REVIEW: 4,
    LADDER_NOT_RUN: -1,
    LADDER_NOT_APPLICABLE: -1,
}

SSIM_FLOOR = 0.95
# RM-281: a single ruined frame in an otherwise clean run does not move the
# mean far enough to trip SSIM_FLOOR. This is a separate, much lower bar
# applied to the worst sampled frame on its own.
WORST_FRAME_SSIM_FLOOR = 0.80
VMAF_FLOOR = 90.0
TEMPORAL_FLICKER_CEILING = 0.08
RESIDUAL_TEXT_SCORE_CEILING = 0.025
SEAM_SCORE_CEILING = 0.35
# RM-304: these are normalized, mask-local fail bars. They are deliberately
# conservative synthetic-fixture bars, not a claim that a licensed corpus was
# available on every developer machine.
MASK_LOCAL_TEMPORAL_CEILING = 0.08
OUTSIDE_MASK_CIELAB_CEILING = 1.0
OUTSIDE_MASK_HDR_LINEAR_CEILING = 0.005

_REMEDIATION = {
    LADDER_NONE: "",
    LADDER_INCREASE_DILATION: (
        "Re-process with increased mask dilation (--mask-dilate) "
        "to cover residual text strokes at subtitle edges."
    ),
    LADDER_TEMPORAL_SMOOTH: (
        "Re-process with temporal smoothing enabled "
        "(--temporal-smooth 2) to reduce per-frame flicker "
        "in the inpainted region."
    ),
    LADDER_ALTERNATE_INPAINTER: (
        "Re-process with a different inpaint mode. If using STTN, "
        "try LAMA or ProPainter for better visual quality on this clip. "
        "If using LAMA, try ProPainter for motion-heavy footage."
    ),
    LADDER_MANUAL_REVIEW: (
        "Multiple quality checks failed or a single check failed "
        "severely. Inspect the quality sheet and decide whether to "
        "accept, re-process with different settings, or exclude "
        "this file from the batch."
    ),
    LADDER_NOT_RUN: "Enable --quality-report to run the quality gate.",
    LADDER_NOT_APPLICABLE: "",
}

_ALL_OPTIONAL_METRICS = ("vmaf", "roi_vmaf", "lpips", "dists",
                         "temporal_consistency", "mask_local_temporal_score",
                         "outside_mask_color_drift")
_ACTIONABLE_RETRY_STEPS = (
    LADDER_INCREASE_DILATION,
    LADDER_TEMPORAL_SMOOTH,
    LADDER_ALTERNATE_INPAINTER,
)


def evaluate_quality_gate(metrics: Optional[dict]) -> Dict[str, Any]:
    """Return a stable gate result for a quality metric payload.

    The result includes a graduated ``ladderStep`` that maps each
    failure mode to a specific remediation action, plus a human-readable
    ``remediation`` string and a ``degradedMetrics`` list of optional
    metrics that were unavailable.
    """
    if not metrics:
        return quality_gate_unknown("quality report not available")
    samples = _number(metrics.get("samples"))
    if samples is not None and samples <= 0:
        return quality_gate_unknown("quality report has no sampled frames")

    violations: List[Dict[str, Any]] = []

    tag = str(metrics.get("tag") or "").strip().lower()
    if tag == "review":
        violations.append({
            "metric": "tag",
            "detail": "quality report tag is Review",
            "ladder": LADDER_MANUAL_REVIEW,
        })

    _check_ssim(metrics, violations)
    _check_worst_frame(metrics, violations)
    _check_vmaf(metrics, violations)
    _check_flicker(metrics, violations)
    _check_mask_local_temporal(metrics, violations)
    _check_final_encode_quality(metrics, violations)
    _check_outside_mask_color_drift(metrics, violations)
    _check_residual_text(metrics, violations)
    _check_removal_verification(metrics, violations)
    _check_seam(metrics, violations)

    degraded = _find_degraded_metrics(metrics)
    previews = _preview_paths(metrics)

    if not violations:
        return {
            "status": STATUS_PASSED,
            "ladderStep": LADDER_NONE,
            "reason": "quality metrics passed configured thresholds",
            "reasons": [],
            "remediation": "",
            "degradedMetrics": degraded,
            "previewFramePaths": previews,
        }

    ladder = _select_ladder_step(violations)
    reasons_text = [v["detail"] for v in violations]
    return {
        "status": STATUS_REVIEW,
        "ladderStep": ladder,
        "reason": "; ".join(reasons_text),
        "reasons": violations,
        "remediation": _REMEDIATION.get(ladder, ""),
        "degradedMetrics": degraded,
        "previewFramePaths": previews,
    }


def quality_gate_unknown(reason: str) -> Dict[str, Any]:
    return {
        "status": STATUS_UNKNOWN,
        "ladderStep": LADDER_NOT_RUN,
        "reason": reason,
        "reasons": [],
        "remediation": _REMEDIATION[LADDER_NOT_RUN],
        "degradedMetrics": [],
        "previewFramePaths": [],
    }


def quality_gate_not_applicable(reason: str) -> Dict[str, Any]:
    return {
        "status": STATUS_NOT_APPLICABLE,
        "ladderStep": LADDER_NOT_APPLICABLE,
        "reason": reason,
        "reasons": [],
        "remediation": "",
        "degradedMetrics": [],
        "previewFramePaths": [],
    }


def retry_config_patch_for_gate(
    gate: Optional[Mapping[str, Any]],
    config: Any = None,
) -> Dict[str, Any]:
    """Return config fields to change for a quality-gate retry.

    The patch is deliberately small: it changes only settings tied to the
    selected ladder step, so a retry remains auditable and does not rewrite
    unrelated per-file overrides.
    """
    if not isinstance(gate, Mapping) or gate.get("status") != STATUS_REVIEW:
        return {}
    patch: Dict[str, Any] = {}
    steps = _retry_steps_for_gate(gate)
    for step in steps:
        _apply_retry_step(step, patch, config)
    if patch:
        patch.setdefault("quality_report", True)
    return patch


def _retry_steps_for_gate(gate: Mapping[str, Any]) -> list[str]:
    step = str(gate.get("ladderStep") or "")
    if step != LADDER_MANUAL_REVIEW:
        return [step] if step in _ACTIONABLE_RETRY_STEPS else []

    steps: list[str] = []
    reasons = gate.get("reasons") or []
    if isinstance(reasons, (list, tuple)):
        for reason in reasons:
            if not isinstance(reason, Mapping):
                continue
            ladder = str(reason.get("ladder") or "")
            metric = str(reason.get("metric") or "")
            if ladder in _ACTIONABLE_RETRY_STEPS:
                if ladder not in steps:
                    steps.append(ladder)
            elif metric in {"tag", "ssim", "vmaf"}:
                if LADDER_ALTERNATE_INPAINTER not in steps:
                    steps.append(LADDER_ALTERNATE_INPAINTER)
    if not steps:
        steps = [LADDER_ALTERNATE_INPAINTER]
    return steps


def _config_value(config: Any, name: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _coerce_int(value: Any, default: int) -> int:
    try:
        if isinstance(value, bool):
            return default
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def _mode_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "sttn").strip().lower()


def _alternate_mode(config: Any, patch: Mapping[str, Any]) -> str:
    current = _mode_value(patch.get("mode", _config_value(config, "mode", "sttn")))
    if current in {"sttn", "auto"}:
        return "lama"
    if current == "lama":
        return "propainter"
    return "lama"


def _apply_retry_step(step: str, patch: Dict[str, Any], config: Any) -> None:
    if step == LADDER_INCREASE_DILATION:
        current = _coerce_int(
            patch.get("mask_dilate_px", _config_value(config, "mask_dilate_px", 8)),
            8,
        )
        if current < 20:
            patch["mask_dilate_px"] = min(20, current + 4)
            return
        feather = _coerce_int(
            patch.get("mask_feather_px", _config_value(config, "mask_feather_px", 4)),
            4,
        )
        patch["mask_feather_px"] = min(15, feather + 2)
    elif step == LADDER_TEMPORAL_SMOOTH:
        current = _coerce_int(
            patch.get(
                "temporal_smooth_radius",
                _config_value(config, "temporal_smooth_radius", 0),
            ),
            0,
        )
        if current < 2:
            patch["temporal_smooth_radius"] = 2
        else:
            patch["tbe_flow_warp"] = True
    elif step == LADDER_ALTERNATE_INPAINTER:
        patch["mode"] = _alternate_mode(config, patch)


def _check_ssim(metrics: dict,
                violations: List[Dict[str, Any]]) -> None:
    roi_ssim = _number(metrics.get("roi_ssim"))
    frame_ssim = _number(metrics.get("ssim"))
    ssim = roi_ssim if roi_ssim is not None else frame_ssim
    label = "ROI SSIM" if roi_ssim is not None else "frame SSIM"
    if ssim is None:
        return
    if ssim < SSIM_FLOOR:
        violations.append({
            "metric": "ssim",
            "value": ssim,
            "threshold": SSIM_FLOOR,
            "detail": f"{label} {ssim:.4f} below {SSIM_FLOOR:.4f}",
            "ladder": LADDER_ALTERNATE_INPAINTER,
        })


def _check_worst_frame(metrics: dict,
                       violations: List[Dict[str, Any]]) -> None:
    """Flag a run whose worst sampled frame is bad even though the mean is fine."""
    worst = metrics.get("worst_frame")
    if not isinstance(worst, Mapping):
        return
    ssim = _number(worst.get("ssim"))
    frame = worst.get("frame")
    if ssim is None or frame is None or ssim >= WORST_FRAME_SSIM_FLOOR:
        return
    violations.append({
        "metric": "worst_frame",
        "value": ssim,
        "threshold": WORST_FRAME_SSIM_FLOOR,
        "frame": int(frame),
        "detail": (
            f"worst sampled frame {int(frame)} SSIM {ssim:.4f} "
            f"below {WORST_FRAME_SSIM_FLOOR:.4f}"
        ),
        "ladder": LADDER_TEMPORAL_SMOOTH,
    })


def _check_vmaf(metrics: dict,
                violations: List[Dict[str, Any]]) -> None:
    roi_vmaf = _number(metrics.get("roi_vmaf"))
    frame_vmaf = _number(metrics.get("vmaf"))
    vmaf = roi_vmaf if roi_vmaf is not None else frame_vmaf
    label = "ROI VMAF" if roi_vmaf is not None else "frame VMAF"
    if vmaf is None:
        return
    if vmaf < VMAF_FLOOR:
        violations.append({
            "metric": "vmaf",
            "value": vmaf,
            "threshold": VMAF_FLOOR,
            "detail": f"{label} {vmaf:.2f} below {VMAF_FLOOR:.2f}",
            "ladder": LADDER_ALTERNATE_INPAINTER,
        })


def _check_flicker(metrics: dict,
                   violations: List[Dict[str, Any]]) -> None:
    flicker = _number(metrics.get("temporal_flicker_score"))
    if flicker is None:
        return
    if flicker > TEMPORAL_FLICKER_CEILING:
        violations.append({
            "metric": "temporal_flicker_score",
            "value": flicker,
            "threshold": TEMPORAL_FLICKER_CEILING,
            "detail": (
                f"temporal flicker {flicker:.4f} above "
                f"{TEMPORAL_FLICKER_CEILING:.4f}"
            ),
            "ladder": LADDER_TEMPORAL_SMOOTH,
        })


def _check_mask_local_temporal(metrics: dict,
                               violations: List[Dict[str, Any]]) -> None:
    worst = metrics.get("mask_local_temporal_worst_pair")
    score = _number(metrics.get("mask_local_temporal_worst_score"))
    if score is None and isinstance(worst, Mapping):
        score = _number(worst.get("score"))
    if score is None:
        score = _number(metrics.get("mask_local_temporal_score"))
    if score is None:
        return
    if score > MASK_LOCAL_TEMPORAL_CEILING:
        suffix = ""
        if isinstance(worst, Mapping):
            start = worst.get("start_frame")
            timestamp = _number(worst.get("timestamp"))
            if start is not None and timestamp is not None:
                suffix = f" at frame {int(start)} ({timestamp:.3f}s)"
        violations.append({
            "metric": "mask_local_temporal_score",
            "value": score,
            "threshold": MASK_LOCAL_TEMPORAL_CEILING,
            "detail": (
                f"mask-local temporal score {score:.4f} above "
                f"{MASK_LOCAL_TEMPORAL_CEILING:.4f}{suffix}"
            ),
            "ladder": LADDER_TEMPORAL_SMOOTH,
        })


def _check_final_encode_quality(metrics: dict,
                                violations: List[Dict[str, Any]]) -> None:
    if metrics.get("quality_final_encode_verified") is not False:
        return
    violations.append({
        "metric": "quality_final_encode_verified",
        "value": False,
        "threshold": True,
        "detail": (
            "mask-local temporal and outside-mask color evidence could not "
            "be collected from the final encoded output"
        ),
        "ladder": LADDER_MANUAL_REVIEW,
    })


def _check_outside_mask_color_drift(metrics: dict,
                                    violations: List[Dict[str, Any]]) -> None:
    drift = _number(metrics.get("outside_mask_color_drift"))
    if drift is None:
        return
    metric = str(
        metrics.get("outside_mask_color_drift_metric") or "cielab_delta_e"
    ).strip().lower()
    if metric == "linear_rgb_mae":
        threshold = OUTSIDE_MASK_HDR_LINEAR_CEILING
        label = "HDR linear-light outside-mask drift"
    else:
        threshold = OUTSIDE_MASK_CIELAB_CEILING
        label = "outside-mask CIELAB drift"
    if drift > threshold:
        worst = metrics.get("outside_mask_color_drift_worst_frame")
        suffix = ""
        if isinstance(worst, Mapping) and worst.get("frame") is not None:
            suffix = f" at frame {int(worst['frame'])}"
        violations.append({
            "metric": "outside_mask_color_drift",
            "value": drift,
            "threshold": threshold,
            "detail": (
                f"{label} {drift:.4f} above {threshold:.4f}{suffix}; "
                "manual review required, no automatic recoloring applied"
            ),
            "ladder": LADDER_MANUAL_REVIEW,
        })


def _check_residual_text(metrics: dict,
                         violations: List[Dict[str, Any]]) -> None:
    residual = _number(metrics.get("residual_text_score"))
    if residual is None:
        return
    if residual > RESIDUAL_TEXT_SCORE_CEILING:
        violations.append({
            "metric": "residual_text_score",
            "value": residual,
            "threshold": RESIDUAL_TEXT_SCORE_CEILING,
            "detail": (
                f"residual text score {residual:.4f} above "
                f"{RESIDUAL_TEXT_SCORE_CEILING:.4f}"
            ),
            "ladder": LADDER_INCREASE_DILATION,
        })


def _check_removal_verification(metrics: dict,
                                violations: List[Dict[str, Any]]) -> None:
    """RM-325: the detector's own answer about the repaired region.

    The residual-text score beside this is a contrast heuristic that says of
    itself it is not a substitute for OCR. This one re-runs the detector the
    job already loaded and reports what it still finds.
    """
    from backend.removal_verification import verification_failed

    evidence = metrics.get("removal_verification")
    if not isinstance(evidence, Mapping) or not evidence.get("ran"):
        return
    if not verification_failed(dict(evidence)):
        return
    fraction = evidence.get("survivingFraction")
    frames = int(evidence.get("framesWithSurvivingText") or 0)
    if fraction is not None:
        detail = (
            f"re-detection still matched {fraction * 100:.0f}% of the "
            f"original text boxes in the repaired region, above "
            f"{float(evidence.get('survivingFractionThreshold', 0.0)) * 100:.0f}%"
        )
        value = float(fraction)
        threshold = float(evidence.get("survivingFractionThreshold") or 0.0)
    else:
        detail = (
            f"re-detection found text in {frames} repaired frame(s)")
        value = float(frames)
        threshold = 0.0
    violations.append({
        "metric": "removal_verification",
        "value": value,
        "threshold": threshold,
        "detail": detail,
        "ladder": LADDER_INCREASE_DILATION,
    })


def _check_seam(metrics: dict,
                violations: List[Dict[str, Any]]) -> None:
    seam = _number(metrics.get("seam_score"))
    if seam is None:
        return
    if seam > SEAM_SCORE_CEILING:
        violations.append({
            "metric": "seam_score",
            "value": seam,
            "threshold": SEAM_SCORE_CEILING,
            "detail": (
                f"mask-boundary seam score {seam:.4f} above "
                f"{SEAM_SCORE_CEILING:.4f}"
            ),
            "ladder": LADDER_INCREASE_DILATION,
        })


def _select_ladder_step(violations: Sequence[Dict[str, Any]]) -> str:
    """Pick the most severe ladder step from a set of violations.

    When multiple violations are present and at least two different
    ladder steps triggered, escalate to MANUAL_REVIEW since no single
    automated remediation covers all failures.
    """
    if not violations:
        return LADDER_NONE
    steps = set()
    max_severity = -1
    best = LADDER_MANUAL_REVIEW
    for v in violations:
        step = v.get("ladder", LADDER_MANUAL_REVIEW)
        steps.add(step)
        sev = _LADDER_SEVERITY.get(step, 4)
        if sev > max_severity:
            max_severity = sev
            best = step
    if len(steps) > 1:
        return LADDER_MANUAL_REVIEW
    return best


def _find_degraded_metrics(metrics: dict) -> List[str]:
    """Return names of optional metrics that were expected but absent."""
    degraded: List[str] = []
    for name in _ALL_OPTIONAL_METRICS:
        value = metrics.get(name)
        if value is None:
            degraded.append(name)
    return degraded


def _preview_paths(metrics: dict) -> list[str]:
    paths = []
    sheet = metrics.get("sheet")
    if sheet:
        paths.append(str(sheet))
    worst_pair = metrics.get("mask_local_temporal_worst_pair")
    if isinstance(worst_pair, Mapping) and worst_pair.get("overlay"):
        paths.append(str(worst_pair["overlay"]))
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
