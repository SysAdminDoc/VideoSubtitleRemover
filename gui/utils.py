"""Utility functions extracted from the GUI monolith."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from backend.import_safety import module_can_import
from backend.inpainters.lama import _pytorch_lama_allowed
from backend.language_support import (
    CURATED_LANGUAGE_NAMES as _CURATED_LANG_NAMES,
    build_language_list as _build_language_list,
    engine_supported_languages as _engine_supported_languages,
    language_support_status,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


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
        engines["detection"].append("EasyOCR")
    if not engines["detection"]:
        engines["detection"].append("OpenCV fallback")
    engines["inpainting"].append("Temporal BG (TBE)")
    if _pytorch_lama_allowed() and module_can_import(
        "simple_lama_inpainting",
        logger=logger,
        failure_context="LaMa engine probe skipped",
    ):
        engines["inpainting"].append("LaMa (PyTorch, opt-in)")
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
        return f"Video ({ext}) - {size}"
    elif is_image_file(path):
        return f"Image ({ext}) - {size}"
    return f"{ext} - {size}"


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
    noun = "track" if len(streams) == 1 else "tracks"
    return f"{len(streams)} embedded subtitle {noun}: {', '.join(labels)}"


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
        suffix = (
            f" across {sample_count} sampled "
            f"frame{'s' if sample_count != 1 else ''}"
        )
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


def _queue_item_info_text(item) -> str:
    parts = [get_file_info(item.file_path)]
    if getattr(item, "soft_subtitle_streams", None):
        parts.append(
            _format_soft_subtitle_summary(item.soft_subtitle_streams))
    elif (is_video_file(item.file_path)
          and not getattr(item, "soft_subtitle_probe_done", False)):
        parts.append("checking embedded subtitle tracks")
    parts.append(truncate_middle(item.file_path, 68))
    return "   -   ".join(part for part in parts if part)
