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
from pathlib import Path
from typing import Any, List, Optional

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
        "planned_result": planned_result,
        "status": STATUS_PENDING,
        "message": "",
        "elapsed_seconds": None,
        "stage_timings": _empty_stage_timings(),
        "dominant_stage": None,
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


def finish_batch_item(record: dict, status: str, *,
                      message: str = "",
                      elapsed_seconds: Optional[float] = None,
                      quality_report: Optional[dict] = None,
                      stage_timings: Optional[dict] = None) -> dict:
    record["status"] = status
    record["message"] = message
    if elapsed_seconds is not None:
        record["elapsed_seconds"] = round(max(0.0, float(elapsed_seconds)), 3)
    if stage_timings is not None:
        record["stage_timings"] = _stage_timings_record(stage_timings)
    else:
        record["stage_timings"] = _stage_timings_record(record.get("stage_timings"))
    record["dominant_stage"] = _dominant_stage(record["stage_timings"])
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
    payload = {
        "schema": "vsr.batch_summary.v1",
        "kind": kind,
        "started_at": _iso(started),
        "completed_at": _iso(completed),
        "elapsed_seconds": round(max(0.0, (completed - started).total_seconds()), 3),
        "count": len(records),
        "counts": _counts(records),
        "stage_summary": summarize_stage_timings(records),
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


def _empty_stage_timings() -> dict:
    return {stage: 0.0 for stage in STAGE_TIMING_KEYS}


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
    ]
    stage_summary = payload.get("stage_summary")
    if isinstance(stage_summary, dict):
        slowest = stage_summary.get("slowest_stage")
        if isinstance(slowest, dict):
            lines.append(
                f"- Slowest stage: {_escape_md(_stage_label(slowest.get('name')))} "
                f"({_format_seconds(slowest.get('seconds'))})"
            )
    lines.extend([
        "",
        "| Status | Input | Output | Planned | Duration | Codec | Subtitles | Elapsed | Preflight | Quality | Message |",
        "|---|---|---|---|---:|---|---:|---:|---|---|---|",
    ])
    review_notes: List[str] = []
    preflight_notes: List[str] = []
    stage_notes: List[str] = []
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
                _format_quality_preflight(record.get("output_quality_preflight")),
                _format_quality_gate(record.get("quality_gate")),
                _escape_md(record.get("message", "")),
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
        "roi_psnr",
        "roi_ssim",
        "vmaf",
        "roi_vmaf",
        "roi_bbox",
        "temporal_flicker_score",
        "temporal_consistency",
        "residual_text_score",
        "lpips",
        "dists",
        "sheet",
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


_sidecar_logger = logging.getLogger(__name__ + ".sidecar")

SIDECAR_SCHEMA = "vsr.output_sidecar.v1"


_SIDECAR_HASH_SIZE_LIMIT = 512 * 1024 * 1024


def _sha256_file(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return ""
    if size > _SIDECAR_HASH_SIZE_LIMIT:
        return ""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _config_snapshot(config: Any) -> dict:
    """Serialize processing config to a reproducibility-safe dict."""
    fields = (
        "mode", "device", "sttn_skip_detection", "sttn_neighbor_stride",
        "sttn_reference_length", "sttn_max_load_num", "lama_super_fast",
        "subtitle_area", "subtitle_areas", "subtitle_region_spans",
        "manual_mask_corrections",
        "detection_threshold", "detection_lang", "detection_frame_skip",
        "detection_vertical", "whisper_fallback", "mask_dilate_px",
        "mask_feather_px", "tbe_enable", "tbe_min_coverage",
        "tbe_use_median", "tbe_flow_warp", "tbe_scene_cut_split",
        "kalman_tracking", "phash_skip_enable", "phash_skip_distance",
        "colour_tune_enable", "time_start", "time_end",
        "preserve_audio", "output_quality", "output_codec",
        "use_hw_encode", "decode_hw_accel", "output_frames",
        "adaptive_batch", "quality_report", "nle_sidecar",
        "keyframe_detection", "deinterlace", "deinterlace_auto",
        "preserve_color_metadata", "loudnorm_target",
        "multi_audio_passthrough", "prefetch_decode",
    )
    snapshot = {}
    for name in fields:
        value = _config_value(config, name, None)
        if value is None:
            continue
        if hasattr(value, "value"):
            value = value.value
        if isinstance(value, (list, tuple)):
            value = [list(v) if isinstance(v, tuple) else v for v in value]
        snapshot[name] = value
    return snapshot


def _detection_engine_status() -> str:
    """Return the name of the active OCR detection engine."""
    try:
        import importlib.util
        for name in ("rapidocr", "rapidocr_onnxruntime", "paddleocr",
                     "easyocr", "surya"):
            if importlib.util.find_spec(name) is not None:
                return name
    except Exception:
        pass
    return "opencv-fallback"


def build_output_sidecar(
    *,
    input_path: str,
    output_path: str,
    config: Any,
    status: str,
    elapsed_seconds: Optional[float] = None,
    stage_timings: Optional[dict] = None,
    quality_report: Optional[dict] = None,
    quality_gate: Optional[dict] = None,
    checkpoint_resumed: bool = False,
    app_version: str = "",
) -> dict:
    """Build a per-output reproducibility sidecar payload."""
    input_file = Path(input_path)
    output_file = Path(output_path)
    now = _dt.datetime.now(_dt.timezone.utc)

    source_fingerprint = ""
    source_bytes = 0
    if input_file.is_file():
        try:
            source_bytes = int(input_file.stat().st_size)
            source_fingerprint = _sha256_file(input_file)
        except OSError:
            pass

    output_bytes = 0
    if output_file.is_file():
        try:
            output_bytes = int(output_file.stat().st_size)
        except OSError:
            pass

    payload = {
        "schema": SIDECAR_SCHEMA,
        "generatedAt": now.isoformat(timespec="seconds"),
        "appVersion": app_version,
        "source": {
            "name": input_file.name,
            "bytes": source_bytes,
            "sha256": source_fingerprint,
        },
        "output": {
            "name": output_file.name,
            "bytes": output_bytes,
        },
        "config": _config_snapshot(config),
        "engine": _detection_engine_status(),
        "status": status,
        "checkpointResumed": checkpoint_resumed,
    }
    if elapsed_seconds is not None:
        payload["elapsedSeconds"] = round(max(0.0, float(elapsed_seconds)), 3)
    if stage_timings is not None:
        payload["stageTimings"] = _stage_timings_record(stage_timings)
    if quality_report is not None:
        payload["qualityReport"] = _quality_report_record(quality_report)
    if quality_gate is not None:
        payload["qualityGate"] = quality_gate
    return payload


def write_output_sidecar(
    *,
    input_path: str,
    output_path: str,
    config: Any,
    status: str,
    elapsed_seconds: Optional[float] = None,
    stage_timings: Optional[dict] = None,
    quality_report: Optional[dict] = None,
    quality_gate: Optional[dict] = None,
    checkpoint_resumed: bool = False,
    app_version: str = "",
) -> Optional[Path]:
    """Write a <output>.vsr.json sidecar next to the output file."""
    try:
        payload = build_output_sidecar(
            input_path=input_path,
            output_path=output_path,
            config=config,
            status=status,
            elapsed_seconds=elapsed_seconds,
            stage_timings=stage_timings,
            quality_report=quality_report,
            quality_gate=quality_gate,
            checkpoint_resumed=checkpoint_resumed,
            app_version=app_version,
        )
        sidecar_path = Path(output_path + ".vsr.json")
        _write_text_atomic(
            sidecar_path,
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )
        _sidecar_logger.debug("Wrote sidecar: %s", sidecar_path)
        return sidecar_path
    except Exception as exc:
        _sidecar_logger.warning("Sidecar write failed: %s", exc, exc_info=True)
        return None
