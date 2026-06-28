"""HDR / 10-bit pipeline helpers (RM-73).

The detector and inpainting models still consume 8-bit BGR working
frames, but HDR outputs must not be encoded as H.264 or as an 8-bit
stream. The processor keeps a high-bit source surface around those
working frames when FFmpeg can decode BGR48LE. These helpers centralise
the source color probe, final color tagging, and HDR-safe encoder
policy so all mux paths use the same rules.

- `probe_color_metadata(path)` reads HDR signalling via ffprobe so the
  user-facing log shows "Detected: BT.2020 / SMPTE2084 (PQ) HDR10" or
  "BT.709 SDR" up front.
- `hdr_encode_args(metadata)` returns ffmpeg flags that re-apply the
  source's colorspace tags to the final encode.
- `hdr_safe_codec(...)` and `hdr_pixel_format_args(...)` keep HDR
  sources on HEVC / AV1 / VVC and request 10-bit output pixel formats.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


HDR_COMPATIBLE_CODECS = {"h265", "av1", "vvc"}
HDR_DEFAULT_CODEC = "h265"
_UNSET_COLOR_VALUES = {"", "unknown", "unspecified"}
_RGB_MATRIX_VALUES = {"gbr", "rgb"}


@dataclass
class ColorMetadata:
    """Container for the four ffprobe stream-level color tags we care
    about. Any unset field carries an empty string; ffmpeg accepts an
    empty value as "do not tag" so passing this struct through is safe
    even on SDR sources."""

    color_primaries: str = ""
    color_transfer: str = ""
    color_space: str = ""
    color_range: str = ""

    @property
    def is_hdr(self) -> bool:
        return self.color_transfer in {"smpte2084", "arib-std-b67"}

    @property
    def label(self) -> str:
        if not (self.color_primaries or self.color_transfer or self.color_space):
            return "unknown"
        bits = []
        if self.color_primaries:
            bits.append(self.color_primaries)
        if self.color_transfer:
            bits.append(self.color_transfer)
        if self.color_space:
            bits.append(self.color_space)
        return " / ".join(bits)


def probe_color_metadata(path: str) -> Optional[ColorMetadata]:
    """Read the first video stream's color tags. Returns None when
    ffprobe is missing or the source has no video stream."""
    if shutil.which("ffprobe") is None:
        return None
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_streams", "-of", "json", path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if result.returncode != 0 or not result.stdout.strip():
            return None
        payload = json.loads(result.stdout)
        streams = payload.get("streams") or []
        if not streams:
            return None
        s = streams[0]
        return ColorMetadata(
            color_primaries=s.get("color_primaries", "") or "",
            color_transfer=s.get("color_transfer", "") or "",
            color_space=s.get("color_space", "") or "",
            color_range=s.get("color_range", "") or "",
        )
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
        logger.debug(f"Color-metadata probe failed: {exc}")
        return None


def hdr_encode_args(meta: Optional[ColorMetadata]) -> List[str]:
    """ffmpeg argv extension that re-tags the output with the source's
    color signalling. No-op for SDR sources and for missing metadata.
    """
    if meta is None:
        return []
    args: List[str] = []
    if meta.color_primaries and meta.color_primaries not in _UNSET_COLOR_VALUES:
        args += ["-color_primaries", meta.color_primaries]
    if meta.color_transfer and meta.color_transfer not in _UNSET_COLOR_VALUES:
        args += ["-color_trc", meta.color_transfer]
    if (
            meta.color_space
            and meta.color_space not in _UNSET_COLOR_VALUES
            and meta.color_space not in _RGB_MATRIX_VALUES):
        args += ["-colorspace", meta.color_space]
    if meta.color_range and meta.color_range not in _UNSET_COLOR_VALUES:
        args += ["-color_range", meta.color_range]
    return args


def hdr_safe_codec(requested_codec: str, meta: Optional[ColorMetadata]) -> str:
    """Return a codec that can legally carry HDR signalling.

    H.264 output is the application's historical default, but HDR10/HLG
    delivery should use HEVC, AV1, or VVC. When an HDR source is detected
    and the caller requested anything else, promote the encode to HEVC.
    """
    codec = (requested_codec or "").lower()
    if meta is None or not meta.is_hdr:
        return codec
    if codec in HDR_COMPATIBLE_CODECS:
        return codec
    return HDR_DEFAULT_CODEC


def hdr_pixel_format_args(
    meta: Optional[ColorMetadata],
    codec: str,
    *,
    hardware: bool = False,
) -> List[str]:
    """ffmpeg argv extension for 10-bit HDR output surfaces."""
    if meta is None or not meta.is_hdr:
        return []
    if codec not in HDR_COMPATIBLE_CODECS:
        return []
    return ["-pix_fmt", "p010le" if hardware else "yuv420p10le"]


def hdr_encoder_private_args(meta: Optional[ColorMetadata], codec: str) -> List[str]:
    """Codec-specific HDR parameters that container color tags do not cover."""
    if meta is None or not meta.is_hdr or codec != "h265":
        return []
    params = ["hdr-opt=1", "repeat-headers=1"]
    if meta.color_primaries:
        params.append(f"colorprim={meta.color_primaries}")
    if meta.color_transfer:
        params.append(f"transfer={meta.color_transfer}")
    if meta.color_space:
        params.append(f"colormatrix={meta.color_space}")
    return ["-x265-params", ":".join(params)]
