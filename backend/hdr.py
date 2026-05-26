"""HDR / 10-bit pipeline helpers (RM-73 partial).

Full 10-bit HDR support requires re-plumbing every internal pixel
buffer from `uint8` to `uint16`. That is a large multi-file refactor
the v3.13 cycle deliberately deferred (the current pipeline reads
cv2 BGR8 frames; rewiring the full path costs touches in detector,
inpainter, intermediate writer, AND every metric routine).

What lands here is the *aware-passthrough* slice:

- `probe_color_metadata(path)` reads HDR signalling via ffprobe so the
  user-facing log shows "Detected: BT.2020 / SMPTE2084 (PQ) HDR10" or
  "BT.709 SDR" up front. This makes it explicit when a source is HDR
  and the current build will deliver SDR.
- `hdr_encode_args(metadata)` returns ffmpeg flags that re-apply the
  source's colorspace tags to the final encode. Even though we process
  in 8-bit BGR today, preserving the BT.2020 primaries / PQ transfer
  tags on the output keeps the file marked correctly so downstream
  players don't accidentally tone-map a tone-mapped result.

Calling code uses these helpers in `process_video`'s encode arg
construction; the heavyweight 16-bit refactor is tracked as a
follow-up to this commit.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


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
    if meta.color_primaries:
        args += ["-color_primaries", meta.color_primaries]
    if meta.color_transfer:
        args += ["-color_trc", meta.color_transfer]
    if meta.color_space:
        args += ["-colorspace", meta.color_space]
    if meta.color_range:
        args += ["-color_range", meta.color_range]
    return args
