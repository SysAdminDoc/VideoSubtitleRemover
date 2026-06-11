"""Batch preflight and summary report helpers.

The GUI and CLI both need durable per-file evidence for long runs:
what was planned, which path was selected, what metadata was known
before processing, and how each item finished. The helpers here stay
free of GUI imports so they can be tested and reused by both surfaces.
"""

from __future__ import annotations

import datetime as _dt
import json
import shutil
from pathlib import Path
from typing import Any, Optional

from backend.io import (
    _choose_available_output_path,
    _path_key,
    _probe_codec_for_log,
    _probe_duration_seconds,
    _probe_subtitle_streams,
    _write_text_atomic,
)
from backend.quality_gate import (
    evaluate_quality_gate,
    quality_gate_not_applicable,
    quality_gate_unknown,
)


STATUS_PENDING = "pending"
STATUS_SKIPPED_EXISTING = "skipped-existing"
STATUS_CHECKPOINT_DONE = "checkpoint-done"
STATUS_SOFT_REMUXED = "soft-subtitle-remuxed"
STATUS_HARDCODED_PROCESSED = "hardcoded-processed"
STATUS_REVIEW_NEEDED = "review-needed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"


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
    if skip_existing:
        return base
    return _choose_available_output_path(base, reserved_outputs or set())


def make_batch_item_record(input_path: str, output_path: str, *, config: Any,
                           skip_existing: bool = False,
                           checkpoint_done: bool = False,
                           soft_action: Optional[str] = None) -> dict:
    input_file = Path(input_path)
    output_file = Path(output_path)
    codec_line = _probe_codec_for_log(str(input_file)) if input_file.exists() else None
    codec_name, width, height, frame_rate = _parse_codec_line(codec_line)
    duration = _probe_duration_seconds(str(input_file)) if input_file.exists() else 0.0
    streams = _probe_subtitle_streams(str(input_file)) if input_file.exists() else []
    planned_result = planned_batch_status(
        output_exists=output_file.exists(),
        skip_existing=skip_existing,
        checkpoint_done=checkpoint_done,
        soft_action=soft_action,
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
        "planned_result": planned_result,
        "status": STATUS_PENDING,
        "message": "",
        "elapsed_seconds": None,
        "mode": str(_config_value(config, "mode", "")),
        "device": str(_config_value(config, "device", "")),
        "output_codec": str(_config_value(config, "output_codec", "")),
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


def finish_batch_item(record: dict, status: str, *,
                      message: str = "",
                      elapsed_seconds: Optional[float] = None,
                      quality_report: Optional[dict] = None) -> dict:
    record["status"] = status
    record["message"] = message
    if elapsed_seconds is not None:
        record["elapsed_seconds"] = round(max(0.0, float(elapsed_seconds)), 3)
    if quality_report is not None:
        record["quality_report"] = _quality_report_record(quality_report)
        gate = _quality_gate_record(quality_report)
        record["quality_gate"] = gate
        if (
            status == STATUS_HARDCODED_PROCESSED
            and gate.get("status") == "review"
        ):
            record["status"] = STATUS_REVIEW_NEEDED
            if message:
                record["message"] = f"{message}; quality gate review needed"
            else:
                record["message"] = "Quality gate review needed"
    elif status == STATUS_HARDCODED_PROCESSED:
        record["quality_gate"] = quality_gate_unknown("quality report not enabled")
    elif status in {STATUS_SKIPPED_EXISTING, STATUS_CHECKPOINT_DONE, STATUS_SOFT_REMUXED}:
        record["quality_gate"] = quality_gate_not_applicable(
            "quality gate applies only to hardcoded cleanup outputs"
        )
    elif status in {STATUS_FAILED, STATUS_CANCELLED}:
        record["quality_gate"] = quality_gate_not_applicable(
            "quality gate did not run because processing did not complete"
        )
    return record


def write_batch_reports(out_dir: Path, records: list[dict], *,
                        kind: str,
                        started_at: _dt.datetime,
                        completed_at: Optional[_dt.datetime] = None) -> tuple[Path, Path]:
    started = _as_utc(started_at)
    completed = _as_utc(completed_at or _dt.datetime.now(_dt.timezone.utc))
    payload = {
        "schema": "vsr.batch_summary.v1",
        "kind": kind,
        "started_at": _iso(started),
        "completed_at": _iso(completed),
        "elapsed_seconds": round(max(0.0, (completed - started).total_seconds()), 3),
        "count": len(records),
        "counts": _counts(records),
        "files": records,
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


def _iso(value: _dt.datetime) -> str:
    value = _as_utc(value)
    return value.isoformat(timespec="seconds")


def _as_utc(value: _dt.datetime) -> _dt.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=_dt.timezone.utc)
    return value.astimezone(_dt.timezone.utc)


def _markdown_summary(payload: dict) -> str:
    lines = [
        "# VSR Batch Summary",
        "",
        f"- Kind: {_escape_md(payload.get('kind', ''))}",
        f"- Started: {_escape_md(payload.get('started_at', ''))}",
        f"- Completed: {_escape_md(payload.get('completed_at', ''))}",
        f"- Files: {payload.get('count', 0)}",
        "",
        "| Status | Input | Output | Planned | Duration | Codec | Subtitles | Elapsed | Quality | Message |",
        "|---|---|---|---|---:|---|---:|---:|---|---|",
    ]
    for record in payload.get("files", []):
        lines.append(
            "| "
            + " | ".join([
                _escape_md(record.get("status", "")),
                _escape_md(record.get("input_name", "")),
                _escape_md(record.get("output_name", "")),
                _escape_md(record.get("planned_result", "")),
                _format_seconds(record.get("duration_seconds")),
                _escape_md(record.get("source_codec", "")),
                str(record.get("subtitle_stream_count", 0)),
                _format_seconds(record.get("elapsed_seconds")),
                _format_quality_gate(record.get("quality_gate")),
                _escape_md(record.get("message", "")),
            ])
            + " |"
        )
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
        "roi_psnr",
        "roi_ssim",
        "vmaf",
        "roi_vmaf",
        "roi_bbox",
        "temporal_flicker_score",
        "residual_text_score",
        "sheet",
    )
    return {key: metrics.get(key) for key in keys if key in metrics}


def _format_quality_gate(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    status = str(value.get("status") or "")
    step = str(value.get("ladderStep") or "")
    if status and step and step not in {"none", "not-applicable"}:
        return _escape_md(f"{status} ({step})")
    return _escape_md(status)
