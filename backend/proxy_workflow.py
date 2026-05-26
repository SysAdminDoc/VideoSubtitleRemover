"""Low-res proxy workflow for fast preview / tuning passes.

RM-34: render a 480p (or configurable) proxy of the source via ffmpeg
so the GUI's mask-preview, detection-preview, and A/B compare flows
load instantly even on 4K source. The final batch run still uses the
full-res original; the proxy is purely a preview accelerant.

Proxy files live under a per-source cache directory keyed by an
md5 fingerprint of the (path, size, mtime) tuple, so a re-edit of the
source invalidates the proxy. The cache root is
`%APPDATA%/VideoSubtitleRemoverPro/proxy_cache/`.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _proxy_cache_dir() -> Path:
    base = Path(os.environ.get("APPDATA", Path.home() / ".config"))
    out = base / "VideoSubtitleRemoverPro" / "proxy_cache"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _source_fingerprint(path: str) -> str:
    try:
        stat = os.stat(path)
        payload = f"{path}|{stat.st_size}|{int(stat.st_mtime)}"
    except OSError:
        payload = path
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:16]


def ensure_proxy(source_path: str, target_height: int = 480,
                  crf: int = 26) -> Optional[str]:
    """Return a path to a re-encoded low-res proxy for `source_path`.
    Builds the proxy via ffmpeg on first use; cached for subsequent
    requests. Returns None when ffmpeg is missing or the encode fails.
    """
    if shutil.which("ffmpeg") is None:
        return None
    target_height = max(120, min(1080, int(target_height)))
    fingerprint = _source_fingerprint(source_path)
    cache = _proxy_cache_dir() / f"{fingerprint}-{target_height}p.mp4"
    if cache.is_file() and cache.stat().st_size > 0:
        return str(cache)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats",
        "-i", source_path,
        "-vf", f"scale=-2:{target_height}",
        "-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast",
        "-an", str(cache),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode == 0 and cache.is_file():
            logger.info(f"Proxy cached at {cache}")
            return str(cache)
        logger.warning(
            f"Proxy ffmpeg exit {result.returncode}: "
            f"{(result.stderr or '')[:400]}"
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning(f"Proxy build failed: {exc}")
    return None
