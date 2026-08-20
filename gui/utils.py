"""Utility functions extracted from the GUI monolith."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from backend.i18n import N_, ntr, tr
from backend.dependency_caps import FROZEN_OPTIONAL_DEPENDENCIES
from backend.import_safety import module_can_import
from backend.inpainters.lama import _pytorch_lama_allowed
from backend.language_support import (
    CURATED_LANGUAGE_NAMES as _CURATED_LANG_NAMES,
    build_language_list as _build_language_list,
    engine_supported_languages as _engine_supported_languages,
    language_support_status,
)

__all__ = (
    "desktop_bounds",
    "dispatch_to_ui",
    "_CURATED_LANG_NAMES",
    "_build_language_list",
    "_engine_supported_languages",
    "language_support_status",
)

try:
    import tkinter as tk
except ImportError:  # pragma: no cover - headless imports have no Tk
    tk = None

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def desktop_bounds(primary_w: int, primary_h: int) -> tuple:
    """Return (x, y, w, h) covering every monitor, not just the primary.

    ``winfo_screenwidth``/``winfo_screenheight`` report the primary display
    only on Windows, so anything that clamps against them is pinned to the
    primary screen. Monitors left of or above the primary have negative
    origins, which is why this returns an origin as well as a size. Falls
    back to the primary bounds off Windows or when the metrics are
    unavailable.
    """
    if sys.platform == "win32":
        try:
            import ctypes

            user32 = ctypes.windll.user32
            vx = int(user32.GetSystemMetrics(76))   # SM_XVIRTUALSCREEN
            vy = int(user32.GetSystemMetrics(77))   # SM_YVIRTUALSCREEN
            vw = int(user32.GetSystemMetrics(78))   # SM_CXVIRTUALSCREEN
            vh = int(user32.GetSystemMetrics(79))   # SM_CYVIRTUALSCREEN
            if vw > 0 and vh > 0:
                return (vx, vy, vw, vh)
        except Exception:
            pass
    return (0, 0, int(primary_w), int(primary_h))


def dispatch_to_ui(root, callback, *args):
    """Marshal a worker-thread call onto the Tk main loop.

    ``after`` raises ``RuntimeError`` once the main loop has exited but
    ``tk.TclError`` once the interpreter itself is destroyed, and several
    call sites caught only the first. During close-while-processing the
    escaping TclError reached ``_process_item``'s blanket handler, which
    marked the item ERROR with the message ``can't invoke "after" command:
    application has been destroyed`` and persisted it, so the next session
    restored a "needs attention" item for a file that was only interrupted.
    A teardown race is not a processing failure.
    """
    errors = (RuntimeError,) if tk is None else (RuntimeError, tk.TclError)
    try:
        return root.after(0, callback, *args)
    except errors:
        return None


# RM-152: canonical queue-item messages. The model stores stable English
# so persisted queue state and the controllers' equality checks survive a
# locale change; `queue_message_text()` translates on the way to a label.
QUEUE_MESSAGE_READY = N_("Ready to process")
QUEUE_MESSAGE_PROBING = N_("Checking embedded subtitle tracks...")
QUEUE_MESSAGE_SOFT_SUBS_FOUND = N_(
    "Embedded subtitle tracks found. Right-click for fast strip/keep, "
    "or run burned-in cleanup."
)

CANONICAL_QUEUE_MESSAGES = frozenset({
    QUEUE_MESSAGE_READY,
    QUEUE_MESSAGE_PROBING,
    QUEUE_MESSAGE_SOFT_SUBS_FOUND,
})


def queue_message_text(message: Optional[str]) -> str:
    """Render a queue-item message, translating the canonical ones.

    Progress text produced at runtime ("Frame 412/9000") is already
    localized by its producer and passes through untouched."""
    text = str(message or "").strip() or QUEUE_MESSAGE_READY
    return tr(text) if text in CANONICAL_QUEUE_MESSAGES else text


def get_app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent.parent


def detect_gpu() -> List[dict]:
    gpus = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    parts = line.split(",")
                    if len(parts) >= 3:
                        try:
                            gpu_idx = int(parts[0].strip())
                            gpu_mem = f"{int(parts[2].strip())} MB"
                        except ValueError:
                            continue
                        gpus.append(
                            {
                                "index": gpu_idx,
                                "name": parts[1].strip(),
                                "memory": gpu_mem,
                                "type": "NVIDIA",
                            }
                        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if not gpus:
        try:
            import onnxruntime as ort

            providers = set(
                getattr(ort, "get_available_providers", lambda: [])()
            )
        except Exception:
            providers = set()
        if "DmlExecutionProvider" in providers:
            gpus.append(
                {
                    "index": 0,
                    "name": "DirectML Device",
                    "memory": "ONNX Runtime",
                    "type": "DirectML",
                }
            )

    return gpus


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h {int(m)}m"


def format_size(bytes_size: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} PB"


VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".avi", ".mkv", ".mov", ".wmv",
    ".flv", ".webm", ".m4v", ".mpeg", ".mpg",
})

IMAGE_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp",
})

SUPPORTED_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS


def is_video_file(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def is_image_file(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def filepicker_pattern(exts: frozenset) -> str:
    """Build a semicolon-joined *.ext pattern for tkinter file dialogs."""
    return ";".join(f"*{e}" for e in sorted(exts))


def detect_ai_engines() -> dict:
    engines = {"detection": [], "inpainting": []}
    if (
        module_can_import(
            "rapidocr",
            logger=logger,
            failure_context="RapidOCR engine probe skipped",
        )
        or module_can_import(
            "rapidocr_onnxruntime",
            logger=logger,
            failure_context="RapidOCR engine probe skipped",
        )
    ):
        engines["detection"].append("RapidOCR")
    if module_can_import(
        "paddleocr",
        logger=logger,
        failure_context="PaddleOCR engine probe skipped",
    ):
        engines["detection"].append("PaddleOCR")
    surya_opt_in = os.environ.get(
        "VSR_ALLOW_GPL", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    if module_can_import(
        "surya.detection",
        logger=logger,
        failure_context="Surya engine probe skipped",
    ):
        if surya_opt_in:
            engines["detection"].append("Surya")
        else:
            engines["detection"].append(
                "Surya (GPL -- set VSR_ALLOW_GPL=1)"
            )
    if module_can_import(
        "easyocr",
        logger=logger,
        failure_context="EasyOCR engine probe skipped",
    ):
        easyocr_release = FROZEN_OPTIONAL_DEPENDENCIES["easyocr"]["last_release"]
        engines["detection"].append(
            f"EasyOCR (frozen, last release {easyocr_release})")
    if not engines["detection"]:
        engines["detection"].append("OpenCV fallback")
    engines["inpainting"].append("Temporal BG (TBE)")
    if _pytorch_lama_allowed() and module_can_import(
        "simple_lama_inpainting",
        logger=logger,
        failure_context="LaMa engine probe skipped",
    ):
        lama_release = FROZEN_OPTIONAL_DEPENDENCIES[
            "simple-lama-inpainting"]["last_release"]
        engines["inpainting"].append(
            f"LaMa (PyTorch, frozen; last release {lama_release}, opt-in)")
    engines["inpainting"].append("OpenCV")
    return engines


def detect_ffmpeg() -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def get_file_info(path: str) -> str:
    p = Path(path)
    try:
        size = format_size(p.stat().st_size)
    except OSError:
        size = "?"
    ext = p.suffix.lower()
    if is_video_file(path):
        return tr("Video ({ext}) - {size}").format(ext=ext, size=size)
    elif is_image_file(path):
        return tr("Image ({ext}) - {size}").format(ext=ext, size=size)
    return tr("{ext} - {size}").format(ext=ext, size=size)


def truncate_middle(text: str, max_length: int = 56) -> str:
    if len(text) <= max_length:
        return text
    if max_length < 10:
        return text[:max_length]
    lead = max_length // 2 - 2
    tail = max_length - lead - 3
    return f"{text[:lead]}...{text[-tail:]}"


def _soft_subtitle_stream_record(stream) -> dict:
    return {
        "index": int(getattr(stream, "index", 0)),
        "codec_name": str(getattr(stream, "codec_name", "") or ""),
        "language": str(getattr(stream, "language", "") or ""),
        "title": str(getattr(stream, "title", "") or ""),
        "default": bool(getattr(stream, "default", False)),
        "forced": bool(getattr(stream, "forced", False)),
    }


def _format_soft_subtitle_summary(streams: List[dict]) -> str:
    if not streams:
        return ""
    labels = []
    for stream in streams[:3]:
        language = stream.get("language") or "und"
        codec = stream.get("codec_name") or "unknown"
        flags = []
        if stream.get("default"):
            flags.append("default")
        if stream.get("forced"):
            flags.append("forced")
        suffix = f" ({', '.join(flags)})" if flags else ""
        labels.append(f"{language}/{codec}{suffix}")
    if len(streams) > 3:
        labels.append(f"+{len(streams) - 3} more")
    return ntr("{count} embedded subtitle track: {names}",
               "{count} embedded subtitle tracks: {names}",
               len(streams)).format(
        count=len(streams), names=", ".join(labels))


def format_quality_report(
    metrics: Optional[dict], compact: bool = False
) -> str:
    if not metrics:
        return ""
    try:
        psnr = float(metrics.get("psnr"))
        ssim = float(metrics.get("ssim"))
    except (TypeError, ValueError):
        return ""

    roi_psnr_raw = metrics.get("roi_psnr")
    roi_ssim_raw = metrics.get("roi_ssim")
    roi_psnr = None
    roi_ssim = None
    try:
        if roi_psnr_raw is not None:
            roi_psnr = float(roi_psnr_raw)
        if roi_ssim_raw is not None:
            roi_ssim = float(roi_ssim_raw)
    except (TypeError, ValueError):
        roi_psnr = None
        roi_ssim = None

    if compact:
        if roi_psnr is not None and roi_ssim is not None:
            return (
                f"inpaint PSNR {roi_psnr:.1f} dB - SSIM {roi_ssim:.4f} "
                f"(frame SSIM {ssim:.4f})"
            )
        return f"PSNR {psnr:.1f} dB - SSIM {ssim:.4f}"

    samples = metrics.get("samples")
    try:
        sample_count = int(samples)
    except (TypeError, ValueError):
        sample_count = 0

    suffix = ""
    if sample_count > 0:
        suffix = ntr(" across {count} sampled frame",
                     " across {count} sampled frames",
                     sample_count).format(count=sample_count)
    if roi_psnr is not None and roi_ssim is not None:
        return (
            f"inpaint region PSNR {roi_psnr:.2f} dB and "
            f"SSIM {roi_ssim:.4f}, "
            f"whole frame PSNR {psnr:.2f} dB and "
            f"SSIM {ssim:.4f}{suffix}"
        )
    return f"PSNR {psnr:.2f} dB and SSIM {ssim:.4f}{suffix}"


def summarize_quality_reports(
    reports: List[Optional[dict]],
) -> Optional[dict]:
    valid = []
    total_samples = 0
    for report in reports:
        if not report:
            continue
        try:
            psnr = float(report.get("psnr"))
            ssim = float(report.get("ssim"))
            samples = int(report.get("samples", 0) or 0)
        except (TypeError, ValueError):
            continue
        valid.append((psnr, ssim, samples))
        total_samples += max(0, samples)

    if not valid:
        return None

    count = len(valid)
    return {
        "psnr": sum(item[0] for item in valid) / count,
        "ssim": sum(item[1] for item in valid) / count,
        "items": count,
        "samples": total_samples,
    }


def _queue_item_execution_text(item) -> str:
    """RM-147: one line describing how the job actually executed.

    A CUDA request that ran RapidOCR on CPU and LaMa on cv2 is labelled as
    such here, not hidden behind a generic engine name.
    """
    payload = getattr(item, "execution_provenance", None)
    if not isinstance(payload, dict) or not payload:
        return ""
    summary = str(payload.get("summary") or "")
    if not summary:
        return ""
    if payload.get("anyFallback"):
        requested = str(payload.get("requestedDevice") or "").upper()
        return tr("{summary}  [fallback from {device}]").format(
            summary=summary, device=requested or tr("request"))
    return summary


def _queue_item_info_text(item) -> str:
    parts = [get_file_info(item.file_path)]
    if getattr(item, "soft_subtitle_streams", None):
        parts.append(
            _format_soft_subtitle_summary(item.soft_subtitle_streams))
    elif (is_video_file(item.file_path)
          and not getattr(item, "soft_subtitle_probe_done", False)):
        parts.append(tr(QUEUE_MESSAGE_PROBING))
    execution = _queue_item_execution_text(item)
    if execution:
        parts.append(execution)
    return "   -   ".join(part for part in parts if part)
