"""Hardware encoder probe.

Extracted from processor.py as part of RFP-L-1. The per-codec ffmpeg
argv assembly stays on ``SubtitleRemover._get_encode_args`` because it
reads HDR metadata + the user-configured codec; this module only
exposes the binary feature-detection.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import tempfile
from typing import Mapping, Optional

from backend.ffmpeg_profiles import collect_ffmpeg_capability_profiles
from backend.subprocess_policy import run_process

logger = logging.getLogger(__name__)

_D3D12_ENCODERS = {
    "h264": "h264_d3d12va",
    "h265": "hevc_d3d12va",
    "av1": "av1_d3d12va",
}
_D3D12_CODEC_NAMES = {
    "h264": "h264",
    "h265": "hevc",
    "av1": "av1",
}


def _error_text(result: object) -> str:
    value = getattr(result, "stderr", "") or getattr(result, "stdout", "") or ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    return str(value).strip()[-800:]


def probe_d3d12_encoder(codec: str = "h264", *, timeout: float = 30.0) -> dict:
    """Prove that one advertised D3D12 encoder produces a readable stream."""
    codec = str(codec or "h264").lower()
    encoder = _D3D12_ENCODERS.get(codec, "")
    report = {
        "schema": "vsr.d3d12_runtime.v1",
        "requested": True,
        "codec": codec,
        "encoder": encoder,
        "advertised": False,
        "smoke_attempted": False,
        "available": False,
        "reason": "",
        "frames": 0,
    }
    if sys.platform != "win32":
        report["reason"] = "D3D12 is available only on Windows"
        return report
    if not encoder:
        report["reason"] = f"D3D12 encoding is not supported for {codec}"
        return report

    capabilities = collect_ffmpeg_capability_profiles(timeout=min(timeout, 10.0))
    advertised = capabilities.get("windows_d3d12") or {}
    report["advertised_features"] = dict(advertised)
    report["advertised"] = bool(
        advertised.get("available")
        and encoder in (advertised.get("advertised_encoders") or [])
    )
    if not report["advertised"]:
        report["reason"] = str(
            advertised.get("reason") or f"FFmpeg does not advertise {encoder}"
        )
        return report

    expected_codec = _D3D12_CODEC_NAMES[codec]
    with tempfile.TemporaryDirectory(prefix="vsr-d3d12-") as temp_dir:
        output = Path(temp_dir) / "smoke.mp4"
        command = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats",
            "-init_hw_device", "d3d12va=vsr_d3d12",
            "-filter_hw_device", "vsr_d3d12",
            "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=1",
            "-vf", "format=nv12,hwupload,scale_d3d12=w=iw:h=ih",
            "-frames:v", "30", "-c:v", encoder,
            "-bf", "0", "-async_depth", "1",
        ]
        if codec in {"h264", "h265"}:
            command += ["-rc_mode", "CQP", "-qp", "23"]
        command += ["-an", str(output)]
        report["smoke_attempted"] = True
        try:
            encoded = run_process(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except Exception as exc:
            report["reason"] = f"D3D12 encode smoke failed: {exc}"
            return report
        if encoded.returncode != 0 or not output.is_file() or output.stat().st_size <= 0:
            report["reason"] = (
                f"{encoder} is advertised but its encode smoke failed"
                + (f": {_error_text(encoded)}" if _error_text(encoded) else "")
            )
            return report
        try:
            probed = run_process(
                [
                    "ffprobe", "-v", "error", "-count_frames",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=codec_name,nb_read_frames",
                    "-of", "json", str(output),
                ],
                capture_output=True,
                text=True,
                timeout=min(timeout, 15.0),
                check=False,
            )
            payload = json.loads(probed.stdout or "{}") if probed.returncode == 0 else {}
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            report["reason"] = f"D3D12 smoke output could not be validated: {exc}"
            return report
        streams = payload.get("streams") or []
        stream: Mapping[str, object] = streams[0] if streams else {}
        try:
            frames = int(stream.get("nb_read_frames") or 0)
        except (TypeError, ValueError):
            frames = 0
        report["frames"] = frames
        if stream.get("codec_name") != expected_codec or frames != 30:
            report["reason"] = (
                "D3D12 smoke output was incomplete or used the wrong codec"
            )
            return report
        report["available"] = True
        report["reason"] = "byte-valid 30-frame runtime smoke passed"
        return report


def _detect_hw_encoder(
    codec: str = "h264",
    *,
    prefer_d3d12: bool = False,
    d3d12_probe: Optional[Mapping[str, object]] = None,
) -> Optional[str]:
    """Probe FFmpeg for hardware encoder availability. Returns encoder
    name or None.

    `codec` scopes the probe to a codec family (`h264`, `h265`, `av1`).
    """
    if prefer_d3d12:
        probe = d3d12_probe or probe_d3d12_encoder(codec)
        encoder = str(probe.get("encoder") or "")
        if probe.get("available") and encoder:
            logger.info("D3D12 encoder runtime smoke passed: %s", encoder)
            return encoder

    family = {
        "h264": ("h264_nvenc", "h264_qsv", "h264_amf"),
        "h265": ("hevc_nvenc", "hevc_qsv", "hevc_amf"),
        "av1":  ("av1_nvenc",  "av1_qsv",  "av1_amf"),
        "vvc":  ("vvc_nvenc",  "vvc_qsv",  "vvc_amf"),
    }.get(codec, ("h264_nvenc", "h264_qsv", "h264_amf"))
    try:
        result = run_process(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True, text=True, timeout=10
        )
        for encoder in family:
            if encoder in result.stdout:
                logger.info(f"Hardware encoder available: {encoder}")
                return encoder
    except Exception:
        pass
    return None
