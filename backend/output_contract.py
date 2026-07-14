"""One authoritative media-output contract for every transform stage."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from backend.hdr import ColorMetadata, hdr_encode_args, probe_color_metadata

_CODEC_NAMES = {
    "h264": "h264",
    "h265": "hevc",
    "hevc": "hevc",
    "av1": "av1",
    "vvc": "vvc",
    "h266": "vvc",
}
_CONTAINER_NAMES = {
    ".mp4": {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"},
    ".mov": {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"},
    ".mkv": {"matroska", "webm"},
    ".webm": {"matroska", "webm"},
    ".avi": {"avi"},
}


def _probe_source_audio(path: str) -> bool:
    if Path(path).is_dir() or shutil.which("ffprobe") is None:
        return False
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "a",
                "-show_entries", "stream=index", "-of", "json", path,
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        payload = json.loads(result.stdout or "{}") if result.returncode == 0 else {}
        return bool(payload.get("streams"))
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return False


@dataclass(frozen=True)
class OutputContract:
    output_suffix: str
    codec: str
    preserve_audio: bool
    source_has_audio: bool
    preserve_color_metadata: bool
    color_metadata: Optional[ColorMetadata]
    hardware_requested: bool
    warnings: Tuple[str, ...] = ()

    @property
    def expected_codec_name(self) -> str:
        return _CODEC_NAMES.get(self.codec, self.codec)

    @property
    def is_hdr(self) -> bool:
        return bool(
            self.preserve_color_metadata
            and self.color_metadata is not None
            and self.color_metadata.is_hdr
        )

    def temp_path(self, temp_dir: str, stem: str) -> str:
        suffix = self.output_suffix or ".mkv"
        return str(Path(temp_dir) / f"{stem}{suffix}")

    def deinterlace_path(self, temp_dir: str) -> str:
        return str(Path(temp_dir) / "deinterlaced.mkv")

    def deinterlace_args(self) -> List[str]:
        args = ["-c:v", "ffv1", "-level", "3"]
        args += ["-pix_fmt", "yuv420p10le" if self.is_hdr else "yuv444p"]
        if self.preserve_color_metadata:
            args += hdr_encode_args(self.color_metadata)
        args += ["-c:a", "copy"] if self.preserve_audio else ["-an"]
        return args

    def report(self) -> dict:
        meta = self.color_metadata
        return {
            "container": self.output_suffix.lstrip(".") or "unknown",
            "codec": self.codec,
            "pixel_format": "10-bit" if self.is_hdr else "source-compatible",
            "preserve_audio": self.preserve_audio,
            "source_has_audio": self.source_has_audio,
            "hardware_requested": self.hardware_requested,
            "color": {
                "primaries": getattr(meta, "color_primaries", ""),
                "transfer": getattr(meta, "color_transfer", ""),
                "matrix": getattr(meta, "color_space", ""),
                "range": getattr(meta, "color_range", ""),
                "mastering_display": getattr(meta, "mastering_display", ""),
                "max_cll": getattr(meta, "max_cll", 0),
                "max_fall": getattr(meta, "max_fall", 0),
            },
            "warnings": list(self.warnings),
        }

    def validate(self, path: str) -> Tuple[bool, List[str]]:
        if shutil.which("ffprobe") is None:
            return True, []
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-show_entries",
                    "format=format_name:stream=codec_type,codec_name,pix_fmt",
                    "-of", "json", path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            payload = json.loads(result.stdout or "{}")
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            return False, [f"output contract probe failed: {exc}"]
        if result.returncode != 0:
            return False, ["output contract probe failed"]

        issues: List[str] = []
        format_names = set(
            str(payload.get("format", {}).get("format_name") or "").split(",")
        )
        expected_formats = _CONTAINER_NAMES.get(self.output_suffix)
        if expected_formats and not (format_names & expected_formats):
            issues.append(
                f"container is {','.join(sorted(format_names)) or 'unknown'}, "
                f"expected {self.output_suffix}"
            )
        streams = payload.get("streams") or []
        video = next(
            (stream for stream in streams if stream.get("codec_type") == "video"),
            {},
        )
        if video.get("codec_name") != self.expected_codec_name:
            issues.append(
                f"video codec is {video.get('codec_name') or 'missing'}, "
                f"expected {self.expected_codec_name}"
            )
        if self.is_hdr and video.get("pix_fmt") not in {
            "yuv420p10le", "yuv422p10le", "yuv444p10le", "p010le"
        }:
            issues.append(
                f"HDR pixel format is {video.get('pix_fmt') or 'missing'}, "
                "expected a 10-bit format"
            )
        has_audio = any(
            stream.get("codec_type") == "audio" for stream in streams
        )
        if self.preserve_audio and self.source_has_audio and not has_audio:
            issues.append("source audio was requested but is missing")
        if not self.preserve_audio and has_audio:
            issues.append("audio is present although audio preservation is disabled")

        if self.preserve_color_metadata and self.color_metadata is not None:
            actual = probe_color_metadata(path)
            expected = self.color_metadata
            for attr, label in (
                ("color_primaries", "color primaries"),
                ("color_transfer", "color transfer"),
                ("color_space", "color matrix"),
                ("color_range", "color range"),
            ):
                value = getattr(expected, attr, "")
                if attr == "color_space" and value in {"gbr", "rgb"}:
                    continue
                if value in {"unknown", "unspecified"}:
                    continue
                if value and getattr(actual, attr, "") != value:
                    issues.append(f"{label} is not preserved")
            if expected.mastering_display and (
                actual is None
                or actual.mastering_display != expected.mastering_display
            ):
                issues.append("mastering-display metadata is not preserved")
            if (expected.max_cll or expected.max_fall) and (
                actual is None
                or actual.max_cll != expected.max_cll
                or actual.max_fall != expected.max_fall
            ):
                issues.append("content-light metadata is not preserved")
            if actual is not None and actual.dynamic_metadata:
                issues.append("stale dynamic HDR metadata is still present")
        return not issues, issues


def build_output_contract(
    *,
    input_path: str,
    output_path: str,
    codec: str,
    preserve_audio: bool,
    preserve_color_metadata: bool,
    color_metadata: Optional[ColorMetadata],
    hardware_requested: bool,
) -> OutputContract:
    warnings = []
    if preserve_color_metadata and color_metadata is not None:
        for metadata_type in color_metadata.dynamic_metadata:
            warnings.append(
                f"Dropped stale {metadata_type}: pixel-changing processing "
                "invalidates frame-dependent HDR metadata."
            )
    return OutputContract(
        output_suffix=Path(output_path).suffix.lower() or ".mkv",
        codec=(codec or "h264").lower(),
        preserve_audio=bool(preserve_audio),
        source_has_audio=_probe_source_audio(input_path),
        preserve_color_metadata=bool(preserve_color_metadata),
        color_metadata=color_metadata if preserve_color_metadata else None,
        hardware_requested=bool(hardware_requested),
        warnings=tuple(warnings),
    )
