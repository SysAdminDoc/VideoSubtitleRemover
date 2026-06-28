"""Source-aware output quality preflight checks."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional


SCHEMA = "vsr.output_quality_preflight.v1"
STATUS_OK = "ok"
STATUS_WARNING = "warning"
STATUS_UNKNOWN = "unknown"
STATUS_NOT_APPLICABLE = "not_applicable"

_VIDEO_CODECS = {
    "h264": {"h264", "avc", "avc1"},
    "h265": {"h265", "hevc", "hvc1"},
    "av1": {"av1"},
    "vvc": {"vvc", "h266"},
}
_EFFICIENT_SOURCE_CODECS = {
    "hevc", "h265", "av1", "vvc", "h266", "vp9", "prores", "dnxhd", "dnxhr"
}


def evaluate_output_quality_preflight(
    input_path: str,
    config: Any,
    *,
    source: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Return a warning payload when output settings risk softer video."""
    output_codec = _normalise_codec(_config_value(config, "output_codec", "h264"))
    crf = _coerce_int(_config_value(config, "output_quality", 23), 23)
    if output_codec in {"copy", "remux"}:
        return _base_payload(
            source or {},
            output_codec,
            crf,
            STATUS_NOT_APPLICABLE,
            "output video is copied/remuxed",
        )
    src = dict(source) if source is not None else _probe_source_video_quality(input_path)
    if not src.get("ok"):
        return _base_payload(
            src,
            output_codec,
            crf,
            STATUS_UNKNOWN,
            str(src.get("error") or "source quality could not be probed"),
        )

    warnings = []
    width = _coerce_int(src.get("width"), 0)
    height = _coerce_int(src.get("height"), 0)
    bitrate = _coerce_int(src.get("bitrate_bps"), 0)
    source_codec = str(src.get("codec") or "").lower()
    recommended_crf = _recommended_crf(
        output_codec=output_codec,
        width=width,
        height=height,
        source_bitrate=bitrate,
    )
    if crf > recommended_crf:
        warnings.append({
            "id": "OUTPUT-CRF-SOURCE-RISK",
            "severity": "warning",
            "message": (
                f"CRF {crf} is above the source-aware recommendation "
                f"of {recommended_crf} or lower for this {width}x{height} "
                f"{_format_bitrate(bitrate)} source."
            ),
            "suggested": {"output_quality": recommended_crf},
        })
    if (
        output_codec == "h264"
        and (max(width, height) >= 2160 or bitrate >= 12_000_000
             or source_codec in _EFFICIENT_SOURCE_CODECS)
        and crf > 21
    ):
        warnings.append({
            "id": "OUTPUT-CODEC-EFFICIENCY-RISK",
            "severity": "warning",
            "message": (
                "H.264 at this quality may look softer or larger than the "
                "source; use H.265/AV1 when available, or lower H.264 CRF."
            ),
            "suggested": {
                "output_codec": "h265",
                "output_quality": min(crf, 21),
            },
        })

    payload = _base_payload(
        src,
        output_codec,
        crf,
        STATUS_WARNING if warnings else STATUS_OK,
        "",
    )
    payload["warnings"] = warnings
    payload["recommendation"] = _recommendation_text(payload)
    payload["overrideRequired"] = bool(warnings)
    payload["overridden"] = bool(warnings)
    return payload


def output_quality_preflight_not_applicable(config: Any, reason: str) -> dict:
    """Return a consistent payload for rows that do not re-encode video."""
    output_codec = _normalise_codec(_config_value(config, "output_codec", "h264"))
    crf = _coerce_int(_config_value(config, "output_quality", 23), 23)
    return _base_payload({}, output_codec, crf, STATUS_NOT_APPLICABLE, reason)


def output_quality_preflight_messages(preflight: Mapping[str, Any]) -> list[str]:
    """Return concise user-facing warning lines for CLI/GUI surfaces."""
    warnings = preflight.get("warnings") or []
    if not isinstance(warnings, list):
        return []
    lines = []
    for warning in warnings:
        if not isinstance(warning, Mapping):
            continue
        message = str(warning.get("message") or "").strip()
        if message:
            lines.append(message)
    recommendation = str(preflight.get("recommendation") or "").strip()
    if recommendation:
        lines.append(recommendation)
    return lines


def _probe_source_video_quality(path: str, timeout: float = 10.0) -> dict:
    if shutil.which("ffprobe") is None:
        return {
            "ok": False,
            "error": "ffprobe not found",
        }
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries",
                "stream=codec_name,width,height,bit_rate,avg_frame_rate:"
                "format=bit_rate",
                "-of", "json",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}
    if proc.returncode != 0:
        return {
            "ok": False,
            "error": (proc.stderr or proc.stdout or "").strip()[:300],
        }
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"ffprobe returned invalid JSON: {exc}"}
    streams = payload.get("streams") if isinstance(payload, dict) else None
    stream = streams[0] if isinstance(streams, list) and streams else {}
    fmt = payload.get("format") if isinstance(payload, dict) else {}
    stream_bitrate = _coerce_int(stream.get("bit_rate"), 0)
    format_bitrate = _coerce_int(
        fmt.get("bit_rate") if isinstance(fmt, dict) else 0,
        0,
    )
    return {
        "ok": bool(stream),
        "path": str(Path(path).name),
        "codec": str(stream.get("codec_name") or ""),
        "width": _coerce_int(stream.get("width"), 0),
        "height": _coerce_int(stream.get("height"), 0),
        "bitrate_bps": stream_bitrate or format_bitrate,
        "bitrate_source": "stream" if stream_bitrate else (
            "format" if format_bitrate else ""
        ),
        "frame_rate": str(stream.get("avg_frame_rate") or ""),
        "error": "" if stream else "no video stream found",
    }


def _base_payload(
    source: Mapping[str, Any],
    output_codec: str,
    crf: int,
    status: str,
    reason: str,
) -> dict:
    return {
        "schema": SCHEMA,
        "status": status,
        "source": {
            "codec": str(source.get("codec") or ""),
            "width": _coerce_int(source.get("width"), 0),
            "height": _coerce_int(source.get("height"), 0),
            "bitrate_bps": _coerce_int(source.get("bitrate_bps"), 0),
            "bitrate_source": str(source.get("bitrate_source") or ""),
            "frame_rate": str(source.get("frame_rate") or ""),
        },
        "output": {
            "codec": output_codec,
            "crf": crf,
        },
        "warnings": [],
        "recommendation": "",
        "overrideRequired": False,
        "overridden": False,
        "reason": reason,
    }


def _recommended_crf(
    *,
    output_codec: str,
    width: int,
    height: int,
    source_bitrate: int,
) -> int:
    baseline = {
        "h264": 23,
        "h265": 24,
        "av1": 30,
        "vvc": 30,
    }.get(output_codec, 23)
    max_dim = max(width, height)
    if max_dim >= 2160 or source_bitrate >= 12_000_000:
        baseline -= 2
    elif max_dim <= 720 and 0 < source_bitrate <= 2_500_000:
        baseline += 2
    return max(15, min(35, baseline))


def _recommendation_text(preflight: Mapping[str, Any]) -> str:
    warnings = preflight.get("warnings") or []
    if not warnings:
        return ""
    suggestions = [
        warning.get("suggested")
        for warning in warnings
        if isinstance(warning, Mapping) and isinstance(warning.get("suggested"), Mapping)
    ]
    suggested_codec = ""
    suggested_crf = None
    for suggestion in suggestions:
        suggested_codec = str(suggestion.get("output_codec") or suggested_codec)
        if suggestion.get("output_quality") is not None:
            suggested_crf = _coerce_int(suggestion.get("output_quality"), 0)
    parts = []
    if suggested_codec:
        parts.append(f"codec {suggested_codec}")
    if suggested_crf:
        parts.append(f"CRF {suggested_crf} or lower")
    if not parts:
        return ""
    return "Suggested safer output setting: " + ", ".join(parts) + "."


def _normalise_codec(value: Any) -> str:
    raw = str(value or "h264").strip().lower()
    for canonical, aliases in _VIDEO_CODECS.items():
        if raw in aliases:
            return canonical
    return raw


def _config_value(config: Any, name: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    value = getattr(config, name, default)
    return getattr(value, "value", value)


def _coerce_int(value: Any, default: int) -> int:
    try:
        if isinstance(value, bool):
            return default
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def _format_bitrate(value: int) -> str:
    if value <= 0:
        return "unknown-bitrate"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f} Mbps"
    return f"{value / 1_000:.0f} kbps"
