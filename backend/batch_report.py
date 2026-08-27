"""Batch preflight and summary report helpers.

The GUI and CLI both need durable per-file evidence for long runs:
what was planned, which path was selected, what metadata was known
before processing, and how each item finished. The helpers here stay
free of GUI imports so they can be tested and reused by both surfaces.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, List, Optional

from backend.config_schema import CONFIG_SCHEMA_VERSION
from backend.execution_provenance import RequestedStageError
from backend.failure_reason import (
    FAILURE_REASON_LABELS,
    REASON_CANCELLED,
    REASON_NONE,
    REASON_PAUSED,
    classify_failure_reason,
    normalize_failure_reason,
)
from backend.io import (
    _choose_available_output_path,
    _path_key,
    _probe_codec_for_log,
    _probe_duration_seconds,
    _probe_subtitle_streams,
    _write_text_atomic,
)
from backend.output_quality_preflight import (
    evaluate_output_quality_preflight,
    output_quality_preflight_not_applicable,
    output_quality_preflight_messages,
)
from backend.quality_gate import (
    evaluate_quality_gate,
    quality_gate_not_applicable,
    quality_gate_unknown,
)
from backend.matte_interchange import mask_interchange_paths
from backend.resume_checkpoint import (
    config_identity_sha256,
    normalized_config_snapshot,
)


# Error classes/markers that are worth an automatic retry (transient hardware
# or subprocess hiccups) versus permanent failures that a retry cannot fix.
_RETRIABLE_MARKERS = (
    "out of memory",
    "cublas",
    "cudnn",
    "broken pipe",
    "timed out",
    "timeout",
    "temporarily",
    "device or resource busy",
    "connection reset",
)
_PERMANENT_MARKERS = (
    "no such file",
    "not found",
    "permission denied",
    "unsupported",
    "invalid",
    "insufficient disk space",
    "no decodable video",
    "codec",
)


def is_retriable_error(exc: BaseException) -> bool:
    """True when a failed batch item is worth re-attempting.

    Transient GPU/subprocess/timeout failures are retriable; missing files,
    permissions, unsupported formats, disk-full, and integrity failures are
    not (a retry would just fail the same way).
    """
    if isinstance(exc, (FileNotFoundError, PermissionError, IsADirectoryError,
                        NotADirectoryError, KeyboardInterrupt)):
        return False
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(marker in text for marker in _PERMANENT_MARKERS):
        return False
    if isinstance(exc, (subprocess.CalledProcessError,
                        subprocess.TimeoutExpired, BrokenPipeError,
                        TimeoutError, ConnectionError, MemoryError)):
        return True
    return any(marker in text for marker in _RETRIABLE_MARKERS)


STATUS_PENDING = "pending"
STATUS_SKIPPED_EXISTING = "skipped-existing"
STATUS_CHECKPOINT_DONE = "checkpoint-done"
STATUS_SOFT_REMUXED = "soft-subtitle-remuxed"
STATUS_HARDCODED_PROCESSED = "hardcoded-processed"
STATUS_REVIEW_NEEDED = "review-needed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
STATUS_PAUSED = "paused"
STAGE_TIMING_KEYS = (
    "decode",
    "ocr",
    "mask",
    "inpaint",
    "encode",
    "mux",
    "quality",
)


def choose_batch_output_path(source_path: str, out_dir: Path, suffix: str,
                             reserved_outputs: Optional[set] = None, *,
                             skip_existing: bool = False) -> Path:
    """Return the intended batch output path for a source file.

    When skip-existing is enabled, callers must see the canonical output
    path so an existing file can be skipped. Without this guard,
    collision-proof naming would select "(2)" and process work the user
    explicitly asked to skip.
    """
    source = Path(source_path)
    base = Path(out_dir) / f"{source.stem}{suffix}{source.suffix}"
    if skip_existing and _path_key(base) not in (reserved_outputs or set()):
        return base
    return _choose_available_output_path(base, reserved_outputs or set())


def make_batch_item_record(input_path: str, output_path: str, *, config: Any,
                           skip_existing: bool = False,
                           skip_existing_policy: str = "verified",
                           identity_config: Any = None,
                           checkpoint_done: bool = False,
                           soft_action: Optional[str] = None) -> dict:
    input_file = Path(input_path)
    output_file = Path(output_path)
    codec_line = _probe_codec_for_log(str(input_file)) if input_file.exists() else None
    codec_name, width, height, frame_rate = _parse_codec_line(codec_line)
    duration = _probe_duration_seconds(str(input_file)) if input_file.exists() else 0.0
    streams = _probe_subtitle_streams(str(input_file)) if input_file.exists() else []
    skip_evidence = (
        evaluate_skip_existing(
            input_path,
            output_path,
            identity_config if identity_config is not None else config,
            policy=skip_existing_policy,
        )
        if skip_existing
        else {
            "requested": False,
            "policy": "off",
            "action": "not-requested",
            "reason_code": "not-requested",
            "message": "Skip-existing was not requested.",
            "output_exists": output_file.exists(),
            "identity_verified": False,
        }
    )
    planned_result = planned_batch_status(
        output_exists=output_file.exists(),
        skip_existing=skip_evidence["action"] == "skip",
        checkpoint_done=checkpoint_done,
        soft_action=soft_action,
    )
    quality_preflight = _output_quality_preflight_for_record(
        input_file,
        config,
        planned_result,
    )
    return {
        "input": str(input_file),
        "input_name": input_file.name,
        "input_exists": input_file.exists(),
        "input_bytes": _file_size(input_file),
        "output": str(output_file),
        "output_name": output_file.name,
        "output_exists": output_file.exists(),
        "output_parent_free_bytes": _free_bytes(output_file.parent),
        "skip_existing": skip_evidence,
        "planned_result": planned_result,
        "status": STATUS_PENDING,
        "message": "",
        "failure_reason": REASON_NONE,
        "failed_stage": "",
        "failed_implementation": "",
        "stage_failure_class": "",
        "recovery_hint": "",
        "elapsed_seconds": None,
        "stage_timings": _empty_stage_timings(),
        "dominant_stage": None,
        "detection_stats": _empty_detection_stats(),
        "execution_provenance": {},
        "optimization_hint": "",
        "mode": str(_config_value(config, "mode", "")),
        "device": str(_config_value(config, "device", "")),
        "output_codec": str(_config_value(config, "output_codec", "")),
        "output_quality": _safe_int(_config_value(config, "output_quality", 23)),
        "output_quality_preflight": quality_preflight,
        "duration_seconds": round(float(duration), 3),
        "estimated_seconds": _estimate_seconds(
            duration,
            width,
            height,
            str(_config_value(config, "mode", "")),
            str(_config_value(config, "output_codec", "")),
            str(_config_value(config, "device", "")),
        ),
        "source_codec": codec_name,
        "source_width": width,
        "source_height": height,
        "source_frame_rate": frame_rate,
        "subtitle_stream_count": len(streams),
        "subtitle_streams": [_subtitle_stream_record(stream) for stream in streams],
        "soft_action": soft_action or "",
        "quality_report": None,
        "quality_gate": quality_gate_unknown("quality gate has not run yet"),
        "retry_attempts": 0,
        "retry_errors": [],
        "source_timing": {"mode": "unknown"},
        "output_contract": {"status": "unknown"},
        "color_preserved": None,
        "mask_export": {
            "requested": bool(_config_value(config, "export_mask_video", False)),
            "status": (
                "pending"
                if bool(_config_value(config, "export_mask_video", False))
                else "not-requested"
            ),
            "path": (
                str(mask_interchange_paths(
                    output_file,
                    str(_config_value(config, "mask_export_format", "ffv1")),
                )[0])
                if bool(_config_value(config, "export_mask_video", False))
                else ""
            ),
            "manifest": (
                str(mask_interchange_paths(
                    output_file,
                    str(_config_value(config, "mask_export_format", "ffv1")),
                )[1])
                if bool(_config_value(config, "export_mask_video", False))
                else ""
            ),
            "format": str(_config_value(config, "mask_export_format", "ffv1")),
        },
        "mask_import": {
            "requested": bool(_config_value(config, "mask_import_path", "")),
            "status": (
                "pending" if _config_value(config, "mask_import_path", "")
                else "not-requested"
            ),
            "manifest": str(_config_value(config, "mask_import_path", "")),
            "mode": str(_config_value(config, "mask_import_mode", "replace")),
        },
    }


def planned_batch_status(*, output_exists: bool, skip_existing: bool,
                         checkpoint_done: bool,
                         soft_action: Optional[str] = None) -> str:
    if skip_existing and output_exists:
        return STATUS_SKIPPED_EXISTING
    if checkpoint_done:
        return STATUS_CHECKPOINT_DONE
    if soft_action in {"strip", "keep_all"}:
        return STATUS_SOFT_REMUXED
    return STATUS_HARDCODED_PROCESSED


def _output_quality_preflight_for_record(
    input_file: Path,
    config: Any,
    planned_result: str,
) -> dict:
    if planned_result == STATUS_SOFT_REMUXED:
        return output_quality_preflight_not_applicable(
            config,
            "soft-subtitle remux copies the video stream",
        )
    if planned_result in {STATUS_SKIPPED_EXISTING, STATUS_CHECKPOINT_DONE}:
        return output_quality_preflight_not_applicable(
            config,
            "planned row does not process video",
        )
    return evaluate_output_quality_preflight(str(input_file), config)


def _failure_reason_for(status: str, message: str,
                        failure_reason: Optional[str],
                        error: Optional[BaseException]) -> str:
    """Return the closed-set reason for a finished item (blank if it worked)."""
    if status == STATUS_CANCELLED:
        return REASON_CANCELLED
    if status == STATUS_PAUSED:
        return REASON_PAUSED
    if status != STATUS_FAILED:
        return REASON_NONE
    explicit = normalize_failure_reason(failure_reason)
    if explicit:
        return explicit
    return classify_failure_reason(
        exc=error, reason=failure_reason, message=message)


def _stage_failure_fields(
    error: Optional[BaseException],
    execution_provenance: Optional[dict],
) -> dict:
    if isinstance(error, RequestedStageError):
        return {
            "failed_stage": error.stage,
            "failed_implementation": error.requested_implementation,
            "stage_failure_class": error.failure_class,
            "recovery_hint": error.recovery_hint,
        }
    stages = (
        execution_provenance.get("stages")
        if isinstance(execution_provenance, dict) else None
    )
    if isinstance(stages, dict):
        for name, stage in reversed(list(stages.items())):
            if not isinstance(stage, dict):
                continue
            if stage.get("status") != "failed" and not stage.get("failureClass"):
                continue
            return {
                "failed_stage": str(name),
                "failed_implementation": str(
                    stage.get("requestedImplementation") or ""
                ),
                "stage_failure_class": str(stage.get("failureClass") or ""),
                "recovery_hint": str(stage.get("recoveryHint") or ""),
            }
    return {
        "failed_stage": "",
        "failed_implementation": "",
        "stage_failure_class": "",
        "recovery_hint": "",
    }


def finish_batch_item(record: dict, status: str, *,
                      message: str = "",
                      elapsed_seconds: Optional[float] = None,
                      quality_report: Optional[dict] = None,
                      stage_timings: Optional[dict] = None,
                      detection_stats: Optional[dict] = None,
                      execution_provenance: Optional[dict] = None,
                      output_contract: Optional[dict] = None,
                      failure_reason: Optional[str] = None,
                      error: Optional[BaseException] = None) -> dict:
    record["status"] = status
    record["message"] = message
    # RM-279: the curated message stays as written; the reason is a closed
    # set so a batch can be counted and filtered by how its items died.
    record["failure_reason"] = _failure_reason_for(
        status, message, failure_reason, error)
    if elapsed_seconds is not None:
        record["elapsed_seconds"] = round(max(0.0, float(elapsed_seconds)), 3)
    if stage_timings is not None:
        record["stage_timings"] = _stage_timings_record(stage_timings)
    else:
        record["stage_timings"] = _stage_timings_record(record.get("stage_timings"))
    record["dominant_stage"] = _dominant_stage(record["stage_timings"])
    record["detection_stats"] = _detection_stats_record(
        detection_stats
        if detection_stats is not None
        else record.get("detection_stats")
    )
    record["optimization_hint"] = _optimization_hint(
        record["stage_timings"], record["detection_stats"])
    # RM-147: requested vs. effective device/engine/backend for this item.
    if isinstance(execution_provenance, dict):
        record["execution_provenance"] = dict(execution_provenance)
    if status == STATUS_FAILED:
        record.update(_stage_failure_fields(error, execution_provenance))
    else:
        record.update(_stage_failure_fields(None, None))
    if isinstance(output_contract, dict):
        record["output_contract"] = dict(output_contract)
    contract_record = record.get("output_contract")
    record["color_preserved"] = (
        contract_record.get("color_preserved")
        if isinstance(contract_record, dict)
        else None
    )
    if quality_report is not None:
        record["quality_report"] = _quality_report_record(quality_report)
        gate = _quality_gate_record(quality_report)
        record["quality_gate"] = gate
        if (
            status == STATUS_HARDCODED_PROCESSED
            and gate.get("status") == "review"
        ):
            record["status"] = STATUS_REVIEW_NEEDED
            step = gate.get("ladderStep", "")
            reason = gate.get("reason", "")
            parts = [message] if message else []
            parts.append(f"quality gate: {step}")
            if reason:
                parts.append(reason)
            record["message"] = "; ".join(parts)
    elif status == STATUS_HARDCODED_PROCESSED:
        record["quality_gate"] = quality_gate_unknown("quality report not enabled")
    elif status in {STATUS_SKIPPED_EXISTING, STATUS_CHECKPOINT_DONE, STATUS_SOFT_REMUXED}:
        record["quality_gate"] = quality_gate_not_applicable(
            "quality gate applies only to hardcoded cleanup outputs"
        )
    elif status in {STATUS_FAILED, STATUS_CANCELLED, STATUS_PAUSED}:
        record["quality_gate"] = quality_gate_not_applicable(
            "quality gate did not run because processing did not complete"
        )
    return record


def _redact_record(record: dict) -> dict:
    """Strip absolute paths from a batch record. Filenames stay; full
    paths are opt-in local-debug fields only."""
    redacted = dict(record)
    redacted.pop("input", None)
    redacted.pop("output", None)
    redacted.pop("output_parent_free_bytes", None)
    return redacted


def write_batch_reports(out_dir: Path, records: list[dict], *,
                        kind: str,
                        started_at: _dt.datetime,
                        completed_at: Optional[_dt.datetime] = None,
                        redact_paths: bool = True) -> tuple[Path, Path]:
    started = _as_utc(started_at)
    completed = _as_utc(completed_at or _dt.datetime.now(_dt.timezone.utc))
    files = [_redact_record(r) for r in records] if redact_paths else records
    stage_summary = summarize_stage_timings(records)
    detection_summary = summarize_detection_stats(records)
    payload = {
        "schema": "vsr.batch_summary.v2",
        "kind": kind,
        "started_at": _iso(started),
        "completed_at": _iso(completed),
        "elapsed_seconds": round(max(0.0, (completed - started).total_seconds()), 3),
        "count": len(records),
        "counts": _counts(records),
        "failure_reason_counts": _failure_reason_counts(records),
        "stage_summary": stage_summary,
        "detection_summary": detection_summary,
        "optimization_hint": _optimization_hint(
            stage_summary.get("stage_totals"), detection_summary),
        "files": files,
    }
    out = Path(out_dir)
    json_path = out / "vsr-batch-summary.json"
    md_path = out / "vsr-batch-summary.md"
    _write_text_atomic(
        json_path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(md_path, _markdown_summary(payload))
    return json_path, md_path


def _config_value(config: Any, name: str, default: Any) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    value = getattr(config, name, default)
    return getattr(value, "value", value)


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def _free_bytes(path: Path) -> Optional[int]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return int(shutil.disk_usage(path).free)
    except OSError:
        return None


def _parse_codec_line(codec_line: Optional[str]) -> tuple[str, int, int, str]:
    if not codec_line:
        return "", 0, 0, ""
    parts = [part.strip() for part in str(codec_line).split(",")]
    codec = parts[0] if parts else ""
    width = _safe_int(parts[1]) if len(parts) > 1 else 0
    height = _safe_int(parts[2]) if len(parts) > 2 else 0
    frame_rate = parts[3] if len(parts) > 3 else ""
    return codec, width, height, frame_rate


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return 0


def _estimate_seconds(duration: float, width: int, height: int, mode: str,
                      output_codec: str, device: str) -> Optional[float]:
    if duration <= 0:
        return None
    pixels = width * height if width > 0 and height > 0 else 1280 * 720
    pixel_factor = max(0.25, min(4.0, pixels / float(1280 * 720)))
    mode_factor = {
        "sttn": 1.0,
        "auto": 1.5,
        "lama": 2.0,
        "propainter": 2.25,
        "migan": 1.7,
    }.get(str(mode).lower(), 1.25)
    codec_factor = 1.2 if str(output_codec).lower() in {"h265", "hevc", "av1"} else 1.0
    device_factor = 0.75 if str(device).lower().startswith(("cuda", "directml")) else 1.5
    return round(max(1.0, duration * pixel_factor * mode_factor * codec_factor * device_factor), 3)


def _subtitle_stream_record(stream) -> dict:
    return {
        "index": int(getattr(stream, "index", -1)),
        "codec_name": str(getattr(stream, "codec_name", "") or ""),
        "language": str(getattr(stream, "language", "") or ""),
        "title": str(getattr(stream, "title", "") or ""),
        "default": bool(getattr(stream, "default", False)),
        "forced": bool(getattr(stream, "forced", False)),
    }


def _counts(records: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for record in records:
        status = str(record.get("status") or STATUS_PENDING)
        counts[status] = counts.get(status, 0) + 1
    return counts


def _failure_reason_counts(records: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for record in records:
        reason = str(record.get("failure_reason") or "")
        if not reason:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _empty_stage_timings() -> dict:
    return {stage: 0.0 for stage in STAGE_TIMING_KEYS}


def _empty_detection_stats() -> dict:
    return {
        "frames_total": 0,
        "frames_ocr": 0,
        "frames_skipped": 0,
        "unique_regions_detected": 0,
        "skip_reasons": {},
    }


def _detection_stats_record(value: Any) -> dict:
    stats = _empty_detection_stats()
    if not isinstance(value, dict):
        return stats
    for key in (
        "frames_total",
        "frames_ocr",
        "frames_skipped",
        "unique_regions_detected",
    ):
        stats[key] = _safe_int(value.get(key, 0))
    reasons = value.get("skip_reasons")
    if isinstance(reasons, dict):
        stats["skip_reasons"] = {
            str(reason): _safe_int(count)
            for reason, count in reasons.items()
            if _safe_int(count) > 0
        }
    return stats


def _optimization_hint(stage_timings: Any, detection_stats: Any) -> str:
    dominant = _dominant_stage(stage_timings)
    stats = _detection_stats_record(detection_stats)
    if (
        isinstance(dominant, dict)
        and dominant.get("name") == "ocr"
        and stats["frames_ocr"] >= 3
    ):
        return "OCR dominated; try --frame-skip 3 or the Fast preset."
    return ""


def _stage_timings_record(value: Any) -> dict:
    timings = _empty_stage_timings()
    if not isinstance(value, dict):
        return timings
    for key in STAGE_TIMING_KEYS:
        try:
            seconds = float(value.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            seconds = 0.0
        timings[key] = round(max(0.0, seconds), 3)
    return timings


def _dominant_stage(timings: Any) -> Optional[dict]:
    if not isinstance(timings, dict):
        return None
    normalized = _stage_timings_record(timings)
    stage, seconds = max(normalized.items(), key=lambda item: item[1])
    if seconds <= 0.0:
        return None
    return {"name": stage, "seconds": seconds}


def summarize_stage_timings(records: list[dict]) -> dict:
    totals = _empty_stage_timings()
    item_count = 0
    for record in records:
        timings = _stage_timings_record(record.get("stage_timings"))
        if any(seconds > 0 for seconds in timings.values()):
            item_count += 1
        for stage, seconds in timings.items():
            totals[stage] = round(totals[stage] + seconds, 3)
    return {
        "stage_totals": totals,
        "slowest_stage": _dominant_stage(totals),
        "items_with_timings": item_count,
    }


def summarize_detection_stats(records: list[dict]) -> dict:
    totals = _empty_detection_stats()
    for record in records:
        stats = _detection_stats_record(record.get("detection_stats"))
        for key in (
            "frames_total",
            "frames_ocr",
            "frames_skipped",
            "unique_regions_detected",
        ):
            totals[key] += stats[key]
        for reason, count in stats["skip_reasons"].items():
            totals["skip_reasons"][reason] = (
                totals["skip_reasons"].get(reason, 0) + count)
    return totals


def _iso(value: _dt.datetime) -> str:
    value = _as_utc(value)
    return value.isoformat(timespec="seconds")


def _as_utc(value: _dt.datetime) -> _dt.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=_dt.timezone.utc)
    return value.astimezone(_dt.timezone.utc)


def _failure_reason_label(reason: Any) -> str:
    text = str(reason or "")
    if not text:
        return ""
    return FAILURE_REASON_LABELS.get(text, text)


def _markdown_summary(payload: dict) -> str:
    lines = [
        "# VSR Batch Summary",
        "",
        f"- Kind: {_escape_md(payload.get('kind', ''))}",
        f"- Started: {_escape_md(payload.get('started_at', ''))}",
        f"- Completed: {_escape_md(payload.get('completed_at', ''))}",
        f"- Files: {payload.get('count', 0)}",
    ]
    stage_summary = payload.get("stage_summary")
    if isinstance(stage_summary, dict):
        slowest = stage_summary.get("slowest_stage")
        if isinstance(slowest, dict):
            lines.append(
                f"- Slowest stage: {_escape_md(_stage_label(slowest.get('name')))} "
                f"({_format_seconds(slowest.get('seconds'))})"
            )
    detection_summary = payload.get("detection_summary")
    if isinstance(detection_summary, dict):
        stats = _detection_stats_record(detection_summary)
        if stats["frames_total"]:
            lines.append(
                f"- Frames OCR'd: {stats['frames_ocr']}; "
                f"skipped: {stats['frames_skipped']}; "
                f"unique regions: {stats['unique_regions_detected']}"
            )
    optimization_hint = str(payload.get("optimization_hint") or "")
    if optimization_hint:
        lines.append(f"- Optimization: {_escape_md(optimization_hint)}")
    reason_counts = payload.get("failure_reason_counts")
    if isinstance(reason_counts, dict) and reason_counts:
        summary = "; ".join(
            f"{_failure_reason_label(reason)}: {count}"
            for reason, count in sorted(reason_counts.items())
        )
        lines.append(f"- Failure reasons: {_escape_md(summary)}")
    lines.extend([
        "",
        "| Status | Reason | Failed stage | Input | Output | Planned | Duration | Codec | Subtitles | Elapsed | Preflight | Quality | Color | Message | Recovery |",
        "|---|---|---|---|---|---|---:|---|---:|---:|---|---|---|---|---|",
    ])
    review_notes: List[str] = []
    preflight_notes: List[str] = []
    stage_notes: List[str] = []
    detection_notes: List[str] = []
    skip_notes: List[str] = []
    for record in payload.get("files", []):
        lines.append(
            "| "
            + " | ".join([
                _escape_md(record.get("status", "")),
                _escape_md(_failure_reason_label(record.get("failure_reason"))),
                _escape_md(record.get("failed_stage", "")),
                _escape_md(record.get("input_name", "")),
                _escape_md(record.get("output_name", "")),
                _escape_md(record.get("planned_result", "")),
                _format_seconds(record.get("duration_seconds")),
                _escape_md(record.get("source_codec", "")),
                str(record.get("subtitle_stream_count", 0)),
                _format_seconds(record.get("elapsed_seconds")),
                _format_quality_preflight(record.get("output_quality_preflight")),
                _format_quality_gate(record.get("quality_gate")),
                _format_color_preserved(record.get("color_preserved")),
                _escape_md(record.get("message", "")),
                _escape_md(record.get("recovery_hint", "")),
            ])
            + " |"
        )
        preflight = record.get("output_quality_preflight")
        if isinstance(preflight, dict) and preflight.get("status") == "warning":
            messages = output_quality_preflight_messages(preflight)
            if messages:
                preflight_notes.append(
                    f"- **{_escape_md(record.get('input_name', '?'))}**: "
                    + _escape_md(" ".join(messages))
                )
        gate = record.get("quality_gate")
        if isinstance(gate, dict) and gate.get("status") == "review":
            remediation = gate.get("remediation", "")
            if remediation:
                review_notes.append(
                    f"- **{_escape_md(record.get('input_name', '?'))}** "
                    f"({gate.get('ladderStep', '')}): {_escape_md(remediation)}"
                )
        stage_note = _format_stage_timings(record.get("stage_timings"))
        if stage_note:
            dominant = record.get("dominant_stage")
            suffix = ""
            if isinstance(dominant, dict):
                suffix = (
                    f"; slowest {_stage_label(dominant.get('name'))} "
                    f"{_format_seconds(dominant.get('seconds'))}"
                )
            stage_notes.append(
                f"- **{_escape_md(record.get('input_name', '?'))}**: "
                + _escape_md(stage_note + suffix)
            )
        stats = _detection_stats_record(record.get("detection_stats"))
        if stats["frames_total"]:
            note = (
                f"OCR'd {stats['frames_ocr']}/{stats['frames_total']} frames; "
                f"skipped {stats['frames_skipped']}; "
                f"unique regions {stats['unique_regions_detected']}"
            )
            hint = str(record.get("optimization_hint") or "")
            if hint:
                note += f". {hint}"
            detection_notes.append(
                f"- **{_escape_md(record.get('input_name', '?'))}**: "
                + _escape_md(note)
            )
        skip_evidence = record.get("skip_existing")
        if (
            isinstance(skip_evidence, dict)
            and skip_evidence.get("requested")
        ):
            skip_notes.append(
                f"- **{_escape_md(record.get('input_name', '?'))}**: "
                + _escape_md(
                    f"policy {skip_evidence.get('policy', '')}; "
                    f"decision {skip_evidence.get('action', '')}; "
                    f"{skip_evidence.get('message', '')}"
                )
            )
    if isinstance(stage_summary, dict):
        totals = _format_stage_timings(stage_summary.get("stage_totals"))
        if totals:
            lines.append("")
            lines.append("### Stage timing summary")
            lines.append("")
            lines.append(_escape_md(totals))
    if stage_notes:
        lines.append("")
        lines.append("### Per-item stage timings")
        lines.append("")
        lines.extend(stage_notes)
    if detection_notes:
        lines.append("")
        lines.append("### Detection efficiency")
        lines.append("")
        lines.extend(detection_notes)
    if skip_notes:
        lines.append("")
        lines.append("### Skip-existing evidence")
        lines.append("")
        lines.extend(skip_notes)
    if preflight_notes:
        lines.append("")
        lines.append("### Output quality preflight notes")
        lines.append("")
        lines.extend(preflight_notes)
    if review_notes:
        lines.append("")
        lines.append("### Quality review notes")
        lines.append("")
        lines.extend(review_notes)
    return "\n".join(lines) + "\n"


def _escape_md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def _format_seconds(value: Any) -> str:
    if value is None:
        return ""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return ""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rest = int(seconds % 60)
    return f"{minutes}m {rest}s"


def _stage_label(value: Any) -> str:
    labels = {
        "decode": "decode",
        "ocr": "OCR",
        "mask": "mask",
        "inpaint": "inpaint",
        "encode": "encode",
        "mux": "mux",
        "quality": "quality",
    }
    return labels.get(str(value or ""), str(value or ""))


def _format_stage_timings(value: Any) -> str:
    timings = _stage_timings_record(value)
    parts = []
    for stage in STAGE_TIMING_KEYS:
        seconds = timings.get(stage, 0.0)
        if seconds > 0:
            parts.append(f"{_stage_label(stage)} {_format_seconds(seconds)}")
    return "; ".join(parts)


def _quality_gate_record(metrics: dict) -> dict:
    gate = metrics.get("quality_gate")
    if isinstance(gate, dict):
        return gate
    return evaluate_quality_gate(metrics)


def _quality_report_record(metrics: dict) -> dict:
    keys = (
        "tag",
        "samples",
        "psnr",
        "ssim",
        "psnr_harmonic_mean",
        "ssim_harmonic_mean",
        "roi_psnr",
        "roi_ssim",
        "roi_psnr_harmonic_mean",
        "roi_ssim_harmonic_mean",
        "worst_frame",
        "vmaf",
        "roi_vmaf",
        "roi_bbox",
        "temporal_flicker_score",
        "temporal_consistency",
        "mask_local_temporal_score",
        "mask_local_temporal_threshold",
        "mask_local_temporal_pairs",
        "mask_local_temporal_scene_cuts_excluded",
        "mask_local_temporal_worst_pair",
        "outside_mask_color_drift",
        "outside_mask_color_drift_metric",
        "outside_mask_color_drift_frames",
        "outside_mask_color_drift_worst_frame",
        "outside_mask_color_drift_threshold",
        "residual_text_score",
        "seam_score",
        "lpips",
        "dists",
        "sheet",
        "mask_review_spans",
    )
    return {key: metrics.get(key) for key in keys if key in metrics}


def _format_quality_gate(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    status = str(value.get("status") or "")
    step = str(value.get("ladderStep") or "")
    if status and step and step not in {"none", "not-applicable", "not-run"}:
        return _escape_md(f"{status} ({step})")
    return _escape_md(status)


def _format_quality_preflight(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    status = str(value.get("status") or "")
    if status != "warning":
        return _escape_md(status)
    messages = output_quality_preflight_messages(value)
    if messages:
        return _escape_md("warning")
    return _escape_md(status)


def _format_color_preserved(value: Any) -> str:
    if value is True:
        return "preserved"
    if value is False:
        return "not preserved"
    return "n/a"


_sidecar_logger = logging.getLogger(__name__ + ".sidecar")

SIDECAR_SCHEMA = "vsr.output_sidecar.v3"
CONFIG_IDENTITY_SCHEMA = "vsr.processing_config_identity.v1"
SKIP_EXISTING_POLICIES = ("verified", "any")
_MAX_SIDECAR_BYTES = 4 * 1024 * 1024
_TRUSTED_OUTPUT_STATUSES = {"processed", "soft-subtitle-remuxed"}


def _sha256_file(path: Path) -> str:
    """Stream a complete source fingerprint with bounded memory.

    GUI processing invokes sidecar generation on its existing worker thread,
    so large inputs no longer lose provenance and Tk remains responsive.
    """
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def _config_snapshot(config: Any) -> dict:
    """Serialize processing config to a reproducibility-safe dict."""
    return normalized_config_snapshot(config)


def _path_identity(path: Path) -> Optional[dict]:
    """Hash a file or a deterministic manifest for a directory tree."""
    try:
        if path.is_file():
            before = path.stat()
            digest = _sha256_file(path)
            after = path.stat()
            if not digest or (
                before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
            ):
                return None
            return {
                "kind": "file",
                "bytes": int(after.st_size),
                "sha256": digest,
                "digestScheme": "sha256",
            }
        if not path.is_dir():
            return None
        entries = sorted(
            (item for item in path.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(path).as_posix().casefold(),
        )
        entry_names = tuple(
            item.relative_to(path).as_posix() for item in entries)
        digest = hashlib.sha256(b"vsr.directory-sha256.v1\0")
        total_bytes = 0
        for entry in entries:
            before = entry.stat()
            file_digest = _sha256_file(entry)
            after = entry.stat()
            if not file_digest or (
                before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
            ):
                return None
            relative = entry.relative_to(path).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(int(after.st_size).to_bytes(8, "big"))
            digest.update(bytes.fromhex(file_digest))
            total_bytes += int(after.st_size)
        final_names = tuple(
            item.relative_to(path).as_posix()
            for item in sorted(
                (candidate for candidate in path.rglob("*")
                 if candidate.is_file()),
                key=lambda candidate: (
                    candidate.relative_to(path).as_posix().casefold()),
            )
        )
        if final_names != entry_names:
            return None
        return {
            "kind": "directory",
            "bytes": total_bytes,
            "sha256": digest.hexdigest(),
            "digestScheme": "vsr.directory-sha256.v1",
        }
    except OSError:
        return None


def _skip_decision(
    policy: str,
    action: str,
    reason_code: str,
    message: str,
    *,
    output_exists: bool,
    verified: bool = False,
) -> dict:
    return {
        "requested": True,
        "policy": policy,
        "action": action,
        "reason_code": reason_code,
        "message": message,
        "output_exists": output_exists,
        "identity_verified": verified,
    }


def _reprocess_decision(policy: str, code: str, message: str) -> dict:
    return _skip_decision(
        policy,
        "reprocess",
        code,
        message,
        output_exists=True,
    )


def _valid_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def evaluate_skip_existing(
    input_path: str,
    output_path: str,
    config: Any,
    *,
    policy: str = "verified",
) -> dict:
    """Decide whether an existing output has enough identity evidence to skip."""
    selected_policy = str(policy or "verified").strip().lower()
    if selected_policy not in SKIP_EXISTING_POLICIES:
        raise ValueError(
            "skip-existing policy must be one of: "
            + ", ".join(SKIP_EXISTING_POLICIES)
        )
    source = Path(input_path)
    output = Path(output_path)
    if not output.exists():
        return _skip_decision(
            selected_policy,
            "process",
            "output-missing",
            "The output path does not exist.",
            output_exists=False,
        )
    if selected_policy == "any":
        return _skip_decision(
            selected_policy,
            "skip",
            "legacy-any",
            "Legacy any policy accepted the path without identity verification.",
            output_exists=True,
        )
    if not output.is_file() and not output.is_dir():
        return _reprocess_decision(
            selected_policy,
            "output-not-regular",
            "The output path is not a regular file or directory.",
        )

    sidecar_path = Path(str(output) + ".vsr.json")
    if not sidecar_path.is_file():
        return _reprocess_decision(
            selected_policy,
            "sidecar-missing",
            "The versioned output sidecar is missing.",
        )
    try:
        if sidecar_path.stat().st_size > _MAX_SIDECAR_BYTES:
            raise ValueError("sidecar exceeds the 4 MiB limit")
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return _reprocess_decision(
            selected_policy,
            "sidecar-unreadable",
            "The output sidecar cannot be read as bounded UTF-8 JSON.",
        )
    if not isinstance(payload, dict):
        return _reprocess_decision(
            selected_policy,
            "sidecar-unreadable",
            "The output sidecar is not a JSON object.",
        )
    if payload.get("schema") != SIDECAR_SCHEMA:
        return _reprocess_decision(
            selected_policy,
            "sidecar-schema-mismatch",
            f"Verified skipping requires sidecar schema {SIDECAR_SCHEMA}.",
        )
    if str(payload.get("status") or "") not in _TRUSTED_OUTPUT_STATUSES:
        return _reprocess_decision(
            selected_policy,
            "sidecar-status-untrusted",
            "The sidecar does not record a completed output.",
        )

    stored_output = payload.get("output")
    if not isinstance(stored_output, dict):
        return _reprocess_decision(
            selected_policy,
            "output-identity-missing",
            "The sidecar has no output identity.",
        )
    stored_path = str(stored_output.get("path") or "")
    if not stored_path or _path_key(stored_path) != _path_key(output):
        return _reprocess_decision(
            selected_policy,
            "output-path-mismatch",
            "The sidecar output path does not match this output.",
        )

    stored_config = payload.get("configIdentity")
    if (
        not isinstance(stored_config, dict)
        or stored_config.get("schema") != CONFIG_IDENTITY_SCHEMA
        or not isinstance(stored_config.get("normalized"), dict)
    ):
        return _reprocess_decision(
            selected_policy,
            "config-identity-missing",
            "The sidecar has no normalized processing configuration.",
        )
    stored_normalized = stored_config["normalized"]
    stored_config_digest = str(stored_config.get("sha256") or "")
    if (
        not _valid_sha256(stored_config_digest)
        or config_identity_sha256(stored_normalized) != stored_config_digest
    ):
        return _reprocess_decision(
            selected_policy,
            "config-digest-mismatch",
            "The sidecar processing configuration digest is invalid.",
        )
    current_config = _config_snapshot(config)
    if (
        stored_normalized != current_config
        or stored_config_digest != config_identity_sha256(current_config)
    ):
        return _reprocess_decision(
            selected_policy,
            "config-mismatch",
            "The normalized processing configuration changed.",
        )

    stored_source = payload.get("source")
    if not isinstance(stored_source, dict):
        return _reprocess_decision(
            selected_policy,
            "source-identity-missing",
            "The sidecar has no source identity.",
        )
    for label, stored in (("source", stored_source), ("output", stored_output)):
        if (
            stored.get("kind") not in {"file", "directory"}
            or not _valid_sha256(stored.get("sha256"))
        ):
            return _reprocess_decision(
                selected_policy,
                f"{label}-identity-missing",
                f"The sidecar has no complete {label} identity.",
            )

    source_identity = _path_identity(source)
    if source_identity is None:
        return _reprocess_decision(
            selected_policy,
            "source-unreadable",
            "The source identity could not be read safely.",
        )
    if source_identity["kind"] != stored_source.get("kind"):
        return _reprocess_decision(
            selected_policy,
            "source-kind-mismatch",
            "The source changed between a file and a directory.",
        )
    if source_identity["bytes"] != stored_source.get("bytes"):
        return _reprocess_decision(
            selected_policy,
            "source-size-mismatch",
            "The source byte size changed.",
        )
    if source_identity["sha256"] != stored_source.get("sha256"):
        return _reprocess_decision(
            selected_policy,
            "source-sha256-mismatch",
            "The source SHA-256 changed.",
        )

    output_identity = _path_identity(output)
    if output_identity is None:
        return _reprocess_decision(
            selected_policy,
            "output-unreadable",
            "The output identity could not be read safely.",
        )
    if output_identity["kind"] != stored_output.get("kind"):
        return _reprocess_decision(
            selected_policy,
            "output-kind-mismatch",
            "The output changed between a file and a directory.",
        )
    if output_identity["bytes"] != stored_output.get("bytes"):
        return _reprocess_decision(
            selected_policy,
            "output-size-mismatch",
            "The output byte size changed.",
        )
    if output_identity["sha256"] != stored_output.get("sha256"):
        return _reprocess_decision(
            selected_policy,
            "output-sha256-mismatch",
            "The output SHA-256 changed.",
        )
    return _skip_decision(
        selected_policy,
        "skip",
        "identity-match",
        "Source, configuration, output path, byte size, and output SHA-256 match.",
        output_exists=True,
        verified=True,
    )


def _ocr_engine_from_provenance(execution_provenance: Any) -> str:
    """Return the OCR engine the job actually ran, if provenance recorded it."""
    if not isinstance(execution_provenance, dict):
        return ""
    stages = execution_provenance.get("stages")
    if not isinstance(stages, dict):
        return ""
    ocr = stages.get("ocr")
    if not isinstance(ocr, dict):
        return ""
    engine = str(ocr.get("engine") or "").strip()
    return engine


def build_output_sidecar(
    *,
    input_path: str,
    output_path: str,
    config: Any,
    status: str,
    identity_config: Any = None,
    elapsed_seconds: Optional[float] = None,
    stage_timings: Optional[dict] = None,
    detection_stats: Optional[dict] = None,
    quality_report: Optional[dict] = None,
    quality_gate: Optional[dict] = None,
    output_contract: Optional[dict] = None,
    selective_rerun: Optional[dict] = None,
    mask_export: Optional[dict] = None,
    mask_import: Optional[dict] = None,
    frozen_matte: Optional[dict] = None,
    translation: Optional[dict] = None,
    clean_reference: Optional[dict] = None,
    execution_provenance: Optional[dict] = None,
    source_timing: Optional[dict] = None,
    checkpoint_resumed: bool = False,
    app_version: str = "",
) -> dict:
    """Build a per-output reproducibility sidecar payload."""
    input_file = Path(input_path)
    output_file = Path(output_path)
    now = _dt.datetime.now(_dt.timezone.utc)

    source_identity = _path_identity(input_file) or {
        "kind": "missing",
        "bytes": 0,
        "sha256": "",
        "digestScheme": "",
    }
    output_identity = _path_identity(output_file) or {
        "kind": "missing",
        "bytes": 0,
        "sha256": "",
        "digestScheme": "",
    }
    effective_config = _config_snapshot(config)
    normalized_identity_config = _config_snapshot(
        config if identity_config is None else identity_config)

    payload = {
        "schema": SIDECAR_SCHEMA,
        "configSchemaVersion": CONFIG_SCHEMA_VERSION,
        "generatedAt": now.isoformat(timespec="seconds"),
        "appVersion": app_version,
        "source": {
            "name": input_file.name,
            **source_identity,
        },
        "output": {
            "name": output_file.name,
            "path": _path_key(output_file),
            **output_identity,
        },
        "config": effective_config,
        "configIdentity": {
            "schema": CONFIG_IDENTITY_SCHEMA,
            "sha256": config_identity_sha256(normalized_identity_config),
            "normalized": normalized_identity_config,
        },
        "engine": (
            _ocr_engine_from_provenance(execution_provenance) or "unrecorded"
        ),
        "status": status,
        "checkpointResumed": checkpoint_resumed,
    }
    # RM-147: requested vs. effective device/engine/backend for this job.
    if execution_provenance is not None:
        payload["executionProvenance"] = execution_provenance
    if source_timing is not None:
        payload["sourceTiming"] = dict(source_timing)
    if elapsed_seconds is not None:
        payload["elapsedSeconds"] = round(max(0.0, float(elapsed_seconds)), 3)
    if stage_timings is not None:
        payload["stageTimings"] = _stage_timings_record(stage_timings)
    if detection_stats is not None:
        payload["detectionStats"] = _detection_stats_record(detection_stats)
    if quality_report is not None:
        payload["qualityReport"] = _quality_report_record(quality_report)
    if quality_gate is not None:
        payload["qualityGate"] = quality_gate
    if output_contract is not None:
        payload["outputContract"] = dict(output_contract)
    if selective_rerun is not None:
        payload["selectiveMaskRerun"] = dict(selective_rerun)
    if mask_export is not None:
        payload["maskExport"] = dict(mask_export)
    if mask_import is not None:
        payload["maskImport"] = dict(mask_import)
    if frozen_matte is not None:
        payload["frozenMatte"] = dict(frozen_matte)
    if translation is not None:
        payload["translation"] = dict(translation)
    if clean_reference is not None:
        payload["cleanReference"] = dict(clean_reference)
    return payload


def write_output_sidecar(
    *,
    input_path: str,
    output_path: str,
    config: Any,
    status: str,
    identity_config: Any = None,
    elapsed_seconds: Optional[float] = None,
    stage_timings: Optional[dict] = None,
    detection_stats: Optional[dict] = None,
    quality_report: Optional[dict] = None,
    quality_gate: Optional[dict] = None,
    output_contract: Optional[dict] = None,
    selective_rerun: Optional[dict] = None,
    mask_export: Optional[dict] = None,
    mask_import: Optional[dict] = None,
    frozen_matte: Optional[dict] = None,
    translation: Optional[dict] = None,
    clean_reference: Optional[dict] = None,
    execution_provenance: Optional[dict] = None,
    source_timing: Optional[dict] = None,
    checkpoint_resumed: bool = False,
    app_version: str = "",
) -> Optional[Path]:
    """Write a <output>.vsr.json sidecar next to the output file."""
    try:
        payload = build_output_sidecar(
            input_path=input_path,
            output_path=output_path,
            config=config,
            identity_config=identity_config,
            status=status,
            elapsed_seconds=elapsed_seconds,
            stage_timings=stage_timings,
            detection_stats=detection_stats,
            quality_report=quality_report,
            quality_gate=quality_gate,
            output_contract=output_contract,
            selective_rerun=selective_rerun,
            mask_export=mask_export,
            mask_import=mask_import,
            frozen_matte=frozen_matte,
            translation=translation,
            clean_reference=clean_reference,
            execution_provenance=execution_provenance,
            source_timing=source_timing,
            checkpoint_resumed=checkpoint_resumed,
            app_version=app_version,
        )
        for label in ("source", "output"):
            identity = payload.get(label)
            if (
                not isinstance(identity, dict)
                or identity.get("kind") not in {"file", "directory"}
                or not _valid_sha256(identity.get("sha256"))
            ):
                _sidecar_logger.warning(
                    "Sidecar write refused: %s identity is incomplete",
                    label,
                )
                return None
        sidecar_path = Path(str(output_path) + ".vsr.json")
        _write_text_atomic(
            sidecar_path,
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )
        _sidecar_logger.debug("Wrote sidecar: %s", sidecar_path)
        return sidecar_path
    except Exception as exc:
        _sidecar_logger.warning("Sidecar write failed: %s", exc, exc_info=True)
        return None
