"""SSIM and quality-metric primitives.

Extracted from processor.py as part of RFP-L-1. ``_compute_quality_report``
and ``_write_quality_sheet`` are methods on ``SubtitleRemover`` (they read
``self.config`` + ``self._quality_mask_bbox``) so they stay there; only
the pure-numpy SSIM helper and optional ffmpeg-backed metrics live here.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural Similarity between two BGR frames. Mean over the three
    channels. Standard formulation (C1, C2 = (0.01*255)^2, (0.03*255)^2).
    Flat-colour regions where the variance and covariance are all zero
    can still drive (num/den) close to 0/0; we wrap in errstate +
    nan_to_num so the report never yields NaN or inf.
    """
    if a is None or b is None or a.shape != b.shape or a.ndim < 2:
        return 0.0
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    if a.ndim == 2:
        a32 = a32[..., None]
        b32 = b32[..., None]
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    channels = a32.shape[2]
    ssims: List[float] = []
    with np.errstate(invalid='ignore', divide='ignore'):
        for c in range(channels):
            x = a32[..., c]
            y = b32[..., c]
            mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
            mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
            mu_x2 = mu_x * mu_x
            mu_y2 = mu_y * mu_y
            mu_xy = mu_x * mu_y
            sig_x2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2
            sig_y2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2
            sig_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy
            num = (2 * mu_xy + C1) * (2 * sig_xy + C2)
            den = (mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2)
            ratio = np.where(den > 0, num / np.maximum(den, 1e-12), 1.0)
            ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=0.0)
            ssims.append(float(np.mean(ratio)))
    if not ssims:
        return 0.0
    return float(np.clip(np.mean(ssims), 0.0, 1.0))


def ffmpeg_libvmaf_available(ffmpeg: str = "ffmpeg") -> bool:
    """Return True when `ffmpeg -filters` reports libvmaf."""
    if shutil.which(ffmpeg) is None:
        return False
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    output = f"{result.stdout}\n{result.stderr}"
    return result.returncode == 0 and bool(
        re.search(r"(?m)^\s*[.A-Z| ]+\s+libvmaf\s+", output)
    )


def _escape_filter_value(value: str) -> str:
    escaped = str(value).replace("\\", "\\\\")
    for char in (":", "'", ",", "[", "]"):
        escaped = escaped.replace(char, f"\\{char}")
    return escaped


def _unescape_filter_value(value: str) -> str:
    out = []
    escaped = False
    for char in value:
        if escaped:
            out.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        else:
            out.append(char)
    if escaped:
        out.append("\\")
    return "".join(out)


def _vmaf_filter(log_path: str,
                 roi: Optional[Tuple[int, int, int, int]] = None) -> str:
    ref = "[0:v]setpts=PTS-STARTPTS"
    dist = "[1:v]setpts=PTS-STARTPTS"
    if roi is not None:
        x1, y1, x2, y2 = roi
        width = max(1, int(x2) - int(x1))
        height = max(1, int(y2) - int(y1))
        ref += f",crop={width}:{height}:{int(x1)}:{int(y1)}"
        dist += f",crop={width}:{height}:{int(x1)}:{int(y1)}"
    return (
        f"{ref}[ref];{dist}[dist];"
        f"[dist][ref]libvmaf=log_fmt=json:log_path={_escape_filter_value(log_path)}"
    )


def _read_vmaf_score(log_path: Path) -> Optional[float]:
    try:
        payload = json.loads(log_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    pooled = payload.get("pooled_metrics", {})
    score = pooled.get("vmaf", {}).get("mean")
    if isinstance(score, (int, float)):
        return float(score)
    frame_scores = []
    for frame in payload.get("frames", []) or []:
        value = (frame.get("metrics") or {}).get("vmaf")
        if isinstance(value, (int, float)):
            frame_scores.append(float(value))
    if frame_scores:
        return float(np.mean(frame_scores))
    return None


def compute_vmaf(
    reference_path: str,
    distorted_path: str,
    *,
    start_seconds: float = 0.0,
    duration_seconds: Optional[float] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
    ffmpeg: str = "ffmpeg",
) -> Optional[float]:
    """Compute VMAF via ffmpeg's libvmaf filter.

    Returns None when ffmpeg/libvmaf is unavailable or the invocation
    fails. The first input is the reference/original, the second is the
    distorted/cleaned output.
    """
    if not Path(reference_path).is_file() or not Path(distorted_path).is_file():
        return None
    if not ffmpeg_libvmaf_available(ffmpeg):
        logger.info("ffmpeg libvmaf filter unavailable; VMAF omitted.")
        return None
    duration = None if duration_seconds is None else max(0.1, float(duration_seconds))
    start = max(0.0, float(start_seconds))
    with tempfile.TemporaryDirectory(prefix="vsr_vmaf_") as tmpdir:
        log_path = Path(tmpdir) / "vmaf.json"
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats"]
        if start > 0:
            cmd += ["-ss", f"{start:.3f}"]
        if duration is not None:
            cmd += ["-t", f"{duration:.3f}"]
        cmd += ["-i", reference_path]
        if duration is not None:
            cmd += ["-t", f"{duration:.3f}"]
        cmd += [
            "-i", distorted_path,
            "-lavfi", _vmaf_filter(str(log_path), roi),
            "-f", "null", "-",
        ]
        timeout = 600.0 if duration is None else max(600.0, duration * 20.0)
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr or b""
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", "replace")
            logger.info(f"ffmpeg libvmaf failed: {stderr[:400]}")
            return None
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg libvmaf timed out")
            return None
        except FileNotFoundError:
            return None
        return _read_vmaf_score(log_path)
