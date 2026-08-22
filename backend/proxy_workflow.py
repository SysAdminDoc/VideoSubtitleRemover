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
from typing import Any, Dict, Optional

from backend.subprocess_policy import run_process

logger = logging.getLogger(__name__)


def probe_video_metadata(video_path: str) -> Dict[str, Any]:
    """Return lightweight stream metadata for preview planning."""
    try:
        import cv2
    except ImportError:
        return {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return {}
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if fps <= 0.0:
        fps = 30.0
    duration = frame_count / fps if frame_count else 0.0
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration,
    }


def _scene_delta(previous, current) -> float:
    """Return a normalized grayscale difference for two proxy frames."""
    import cv2

    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    if previous_gray.shape != current_gray.shape:
        current_gray = cv2.resize(
            current_gray,
            (previous_gray.shape[1], previous_gray.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    return float(cv2.absdiff(previous_gray, current_gray).mean() / 255.0)


def probe_proxy_window(
    proxy_path: str,
    timestamp: float,
    *,
    radius_frames: int = 1,
    scene_cut_threshold: float = 0.30,
) -> Dict[str, Any]:
    """Plan a before/current/after window without returning proxy pixels.

    The returned frame indices belong to the source stream because the proxy
    is created without dropping frames. Proxy frames are used only to find a
    nearby scene boundary and to report the planning resolution.
    """
    import cv2

    metadata = probe_video_metadata(proxy_path)
    if not metadata or metadata["frame_count"] <= 0:
        return {
            "frame_indices": (),
            "frame_start": 0,
            "frame_end": 0,
            "target_frame": 0,
            "timestamp": max(0.0, float(timestamp or 0.0)),
            "proxy_resolution": "unknown",
            "scene_cut_before": False,
            "scene_cut_after": False,
            "planning_source": proxy_path,
        }

    fps = float(metadata["fps"])
    frame_count = int(metadata["frame_count"])
    target = max(0, min(frame_count - 1, int(round(float(timestamp or 0.0) * fps))))
    radius = max(1, min(8, int(radius_frames)))
    start = max(0, target - radius)
    end = min(frame_count - 1, target + radius)

    cap = cv2.VideoCapture(proxy_path)
    frames = {}
    if cap.isOpened():
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for index in range(start, end + 1):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frames[index] = frame
        finally:
            cap.release()

    available = sorted(frames)
    if not available:
        indices = (target,)
        scene_before = False
        scene_after = False
        bounded_start = target
        bounded_end = target
    else:
        bounded_start = available[0]
        bounded_end = available[-1]
        scene_before = False
        scene_after = False
        for left, right in zip(available, available[1:]):
            if _scene_delta(frames[left], frames[right]) < scene_cut_threshold:
                continue
            if right <= target:
                bounded_start = max(bounded_start, right)
                scene_before = True
            elif left >= target:
                bounded_end = min(bounded_end, left)
                scene_after = True
        bounded_start = min(max(bounded_start, target - radius), target)
        bounded_end = max(min(bounded_end, target + radius), target)
        indices = tuple(
            index
            for index in range(bounded_start, bounded_end + 1)
            if index in frames
        )
        if not indices:
            indices = (target,)

    width = int(metadata.get("width") or 0)
    height = int(metadata.get("height") or 0)
    resolution = f"{width}x{height}" if width and height else "unknown"
    return {
        "frame_indices": indices,
        "frame_start": int(indices[0]),
        "frame_end": int(indices[-1]),
        "target_frame": target,
        "timestamp": target / fps,
        "fps": fps,
        "frame_count": frame_count,
        "duration": float(metadata.get("duration") or 0.0),
        "proxy_resolution": resolution,
        "scene_cut_before": scene_before,
        "scene_cut_after": scene_after,
        "planning_source": proxy_path,
    }


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
        result = run_process(cmd, capture_output=True, text=True, timeout=1800)
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
