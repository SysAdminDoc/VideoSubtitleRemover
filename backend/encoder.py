"""Hardware encoder probe.

Extracted from processor.py as part of RFP-L-1. The per-codec ffmpeg
argv assembly stays on ``SubtitleRemover._get_encode_args`` because it
reads HDR metadata + the user-configured codec; this module only
exposes the binary feature-detection.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


def _detect_hw_encoder(codec: str = "h264") -> Optional[str]:
    """Probe FFmpeg for hardware encoder availability. Returns encoder
    name or None.

    `codec` scopes the probe to a codec family (`h264`, `h265`, `av1`).
    """
    family = {
        "h264": ("h264_nvenc", "h264_qsv", "h264_amf"),
        "h265": ("hevc_nvenc", "hevc_qsv", "hevc_amf"),
        "av1":  ("av1_nvenc",  "av1_qsv",  "av1_amf"),
        "vvc":  ("vvc_nvenc",  "vvc_qsv",  "vvc_amf"),
    }.get(codec, ("h264_nvenc", "h264_qsv", "h264_amf"))
    try:
        result = subprocess.run(
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
