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
import re
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Tuple

from backend.subprocess_policy import run_process

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
    mastering_display: str = ""
    max_cll: int = 0
    max_fall: int = 0
    dynamic_metadata: Tuple[str, ...] = ()

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
            "-read_intervals", "%+#1", "-show_streams", "-show_frames",
            "-of", "json", path,
        ]
        result = run_process(cmd, capture_output=True, text=True, timeout=20)
        if result.returncode != 0 or not result.stdout.strip():
            return None
        payload = json.loads(result.stdout)
        streams = payload.get("streams") or []
        if not streams:
            return None
        s = streams[0]
        mastering_display = ""
        max_cll = 0
        max_fall = 0
        dynamic_metadata = []
        frames = payload.get("frames") or []
        first_frame = frames[0] if frames and isinstance(frames[0], dict) else {}
        side_data_entries = list(s.get("side_data_list") or [])
        side_data_entries.extend(first_frame.get("side_data_list") or [])
        for side_data in side_data_entries:
            if not isinstance(side_data, dict):
                continue
            side_type = str(side_data.get("side_data_type") or "").strip()
            lowered = side_type.lower()
            if "mastering display metadata" in lowered:
                mastering_display = _x265_mastering_display(side_data)
            elif "content light level metadata" in lowered:
                max_cll = _positive_int(side_data.get("max_content"))
                max_fall = _positive_int(side_data.get("max_average"))
            elif "dynamic metadata" in lowered or "dovi" in lowered:
                dynamic_metadata.append(side_type)
        codec_tag = str(s.get("codec_tag_string") or "").lower()
        if codec_tag in {"dvh1", "dvhe"}:
            dynamic_metadata.append("Dolby Vision Configuration Record")
        return ColorMetadata(
            color_primaries=(
                s.get("color_primaries")
                or first_frame.get("color_primaries")
                or ""
            ),
            color_transfer=(
                s.get("color_transfer")
                or first_frame.get("color_transfer")
                or ""
            ),
            color_space=(
                s.get("color_space") or first_frame.get("color_space") or ""
            ),
            color_range=(
                s.get("color_range") or first_frame.get("color_range") or ""
            ),
            mastering_display=mastering_display,
            max_cll=max_cll,
            max_fall=max_fall,
            dynamic_metadata=tuple(dict.fromkeys(dynamic_metadata)),
        )
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
        logger.debug(f"Color-metadata probe failed: {exc}")
        return None


def _rational_float(value) -> Optional[float]:
    try:
        result = float(Fraction(str(value)))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return result if result >= 0 else None


def _scaled_metadata_int(value, scale: float) -> Optional[int]:
    parsed = _rational_float(value)
    if parsed is None:
        return None
    return int(round(parsed * scale))


def _positive_int(value) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, result)


def _x265_mastering_display(side_data: dict) -> str:
    """Convert ffprobe mastering-display rationals to x265's integer form."""
    values = {
        "rx": _scaled_metadata_int(side_data.get("red_x"), 50000),
        "ry": _scaled_metadata_int(side_data.get("red_y"), 50000),
        "gx": _scaled_metadata_int(side_data.get("green_x"), 50000),
        "gy": _scaled_metadata_int(side_data.get("green_y"), 50000),
        "bx": _scaled_metadata_int(side_data.get("blue_x"), 50000),
        "by": _scaled_metadata_int(side_data.get("blue_y"), 50000),
        "wx": _scaled_metadata_int(side_data.get("white_point_x"), 50000),
        "wy": _scaled_metadata_int(side_data.get("white_point_y"), 50000),
        "max": _scaled_metadata_int(side_data.get("max_luminance"), 10000),
        "min": _scaled_metadata_int(side_data.get("min_luminance"), 10000),
    }
    if any(value is None for value in values.values()):
        return ""
    return (
        f"G({values['gx']},{values['gy']})"
        f"B({values['bx']},{values['by']})"
        f"R({values['rx']},{values['ry']})"
        f"WP({values['wx']},{values['wy']})"
        f"L({values['max']},{values['min']})"
    )


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
    if meta is None or not meta.is_hdr:
        return []
    if codec == "av1":
        params = []
        mastering = _svt_mastering_display(meta.mastering_display)
        if mastering:
            params.append(f"mastering-display={mastering}")
        if meta.max_cll or meta.max_fall:
            params.append(f"content-light={meta.max_cll},{meta.max_fall}")
        return ["-svtav1-params", ":".join(params)] if params else []
    if codec == "vvc":
        params = [
            "Hdr=" + (
                "pq_2020" if meta.color_transfer == "smpte2084"
                else "hlg_2020"
            )
        ]
        mastering = _vvenc_mastering_display(meta.mastering_display)
        if mastering:
            params.append(f"MasteringDisplayColourVolume={mastering}")
        if meta.max_cll or meta.max_fall:
            params.append(
                f"MaxContentLightLevel={meta.max_cll},{meta.max_fall}"
            )
        return ["-vvenc-params", ":".join(params)]
    if codec != "h265":
        return []
    params = ["hdr-opt=1", "repeat-headers=1"]
    if meta.color_primaries:
        params.append(
            f"colorprim={_COLOR_PRIMARIES.get(meta.color_primaries, meta.color_primaries)}"
        )
    if meta.color_transfer:
        params.append(
            f"transfer={_COLOR_TRANSFERS.get(meta.color_transfer, meta.color_transfer)}"
        )
    if meta.color_space:
        params.append(
            f"colormatrix={_COLOR_MATRICES.get(meta.color_space, meta.color_space)}"
        )
    if meta.mastering_display:
        params.append(f"master-display={meta.mastering_display}")
    if meta.max_cll or meta.max_fall:
        params.append(f"max-cll={meta.max_cll},{meta.max_fall}")
    return ["-x265-params", ":".join(params)]


_MASTERING_PATTERN = re.compile(
    r"G\((\d+),(\d+)\)B\((\d+),(\d+)\)R\((\d+),(\d+)\)"
    r"WP\((\d+),(\d+)\)L\((\d+),(\d+)\)"
)

_COLOR_PRIMARIES = {
    "bt709": 1, "bt470m": 4, "bt470bg": 5, "smpte170m": 6,
    "smpte240m": 7, "film": 8, "bt2020": 9, "smpte428": 10,
    "smpte431": 11, "smpte432": 12, "jedec-p22": 22,
}
_COLOR_TRANSFERS = {
    "bt709": 1, "gamma22": 4, "gamma28": 5, "smpte170m": 6,
    "smpte240m": 7, "linear": 8, "log": 9, "log_sqrt": 10,
    "iec61966-2-4": 11, "bt1361e": 12, "iec61966-2-1": 13,
    "bt2020-10": 14, "bt2020-12": 15, "smpte2084": 16,
    "smpte428": 17, "arib-std-b67": 18,
}
_COLOR_MATRICES = {
    "gbr": 0, "rgb": 0, "bt709": 1, "fcc": 4, "bt470bg": 5,
    "smpte170m": 6, "smpte240m": 7, "ycgco": 8, "bt2020nc": 9,
    "bt2020c": 10, "smpte2085": 11, "chroma-derived-nc": 12,
    "chroma-derived-c": 13, "ictcp": 14,
}


def _mastering_values(value: str) -> Optional[Tuple[int, ...]]:
    match = _MASTERING_PATTERN.fullmatch(value or "")
    if not match:
        return None
    return tuple(int(item) for item in match.groups())


def _svt_mastering_display(value: str) -> str:
    values = _mastering_values(value)
    if values is None:
        return ""
    chroma = [number / 50000.0 for number in values[:8]]
    luminance = [values[8] / 10000.0, values[9] / 10000.0]
    return (
        f"G({chroma[0]:.5f},{chroma[1]:.5f})"
        f"B({chroma[2]:.5f},{chroma[3]:.5f})"
        f"R({chroma[4]:.5f},{chroma[5]:.5f})"
        f"WP({chroma[6]:.5f},{chroma[7]:.5f})"
        f"L({luminance[0]:.4f},{luminance[1]:.4f})"
    )


def _vvenc_mastering_display(value: str) -> str:
    values = _mastering_values(value)
    return ",".join(str(item) for item in values) if values else ""
