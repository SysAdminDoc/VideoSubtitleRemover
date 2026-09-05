"""Utility functions extracted from the GUI monolith."""

from __future__ import annotations

import logging
import math
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from backend.i18n import N_, ntr, tr
from gui.failure_copy import CANONICAL_FAILURE_MESSAGES
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
    "collect_supported_files",
    "desktop_bounds",
    "dispatch_to_ui",
    "install_ui_dispatcher",
    "stop_ui_dispatcher",
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


def collect_supported_files(
    folders,
    *,
    cap: int,
    on_progress=None,
    progress_every: int = 200,
) -> tuple[list[str], bool]:
    """Walk folders and return (supported file paths, hit_cap).

    Pure filesystem work so it can run on a worker thread: the old
    implementation ran ``sorted(folder.rglob("*"))`` inside the drop
    callback, which materialised and sorted the whole tree on the Tk main
    thread and froze the window for the entire walk of a large library or a
    slow network share. Only the matches are sorted here, and enumeration
    stops at ``cap`` because the queue cannot accept more than that anyway.

    ``on_progress(scanned_count)`` is invoked every ``progress_every``
    directory entries; exceptions from it are the caller's problem.
    """
    matches: list[str] = []
    scanned = 0
    hit_cap = False
    for folder in folders:
        root = Path(folder)
        if hit_cap:
            break
        try:
            walker = root.rglob("*")
        except OSError:
            continue
        while True:
            try:
                candidate = next(walker)
            except StopIteration:
                break
            except OSError:
                continue
            scanned += 1
            if on_progress is not None and scanned % max(1, progress_every) == 0:
                on_progress(scanned)
            try:
                path_text = str(candidate)
                if not candidate.is_file():
                    continue
            except OSError:
                continue
            if is_video_file(path_text) or is_image_file(path_text):
                matches.append(path_text)
                if len(matches) >= cap:
                    hit_cap = True
                    break
    matches.sort()
    return matches, hit_cap


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


UI_DISPATCH_INTERVAL_MS = 16
UI_DISPATCH_BATCH_SIZE = 128


def install_ui_dispatcher(root) -> None:
    """Install a UI-thread poller for worker results on one Tk root."""
    if getattr(root, "_vsr_ui_dispatch_queue", None) is not None:
        return
    root._vsr_ui_thread_id = threading.get_ident()
    root._vsr_ui_dispatch_queue = queue.SimpleQueue()
    root._vsr_ui_dispatch_running = True

    def _drain():
        if not getattr(root, "_vsr_ui_dispatch_running", False):
            return
        pending = root._vsr_ui_dispatch_queue
        for _index in range(UI_DISPATCH_BATCH_SIZE):
            try:
                callback, args = pending.get_nowait()
            except queue.Empty:
                break
            try:
                callback(*args)
            except Exception as exc:
                reporter = getattr(root, "report_callback_exception", None)
                if callable(reporter):
                    reporter(type(exc), exc, exc.__traceback__)
                else:
                    logger.exception("Unhandled UI callback exception")
        try:
            root._vsr_ui_dispatch_after_id = root.after(
                UI_DISPATCH_INTERVAL_MS, _drain)
        except (RuntimeError, tk.TclError if tk is not None else RuntimeError):
            root._vsr_ui_dispatch_running = False

    root._vsr_ui_dispatch_drain = _drain
    root._vsr_ui_dispatch_after_id = root.after(0, _drain)


def stop_ui_dispatcher(root) -> None:
    """Stop accepting worker results before the Tk root is destroyed."""
    try:
        root._vsr_ui_dispatch_running = False
    except Exception:
        return
    pending = getattr(root, "_vsr_ui_dispatch_queue", None)
    if pending is None:
        return
    while True:
        try:
            pending.get_nowait()
        except queue.Empty:
            break


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
    pending = getattr(root, "_vsr_ui_dispatch_queue", None)
    if pending is not None:
        if not getattr(root, "_vsr_ui_dispatch_running", False):
            return None
        pending.put((callback, args))
        return "queued"

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
}) | CANONICAL_FAILURE_MESSAGES


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
        from backend.subprocess_policy import run_process

        result = run_process(
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


def detect_ffmpeg_state(*, timeout: float = 8.0) -> dict:
    """Probe FFmpeg and classify it against the enforced security floor.

    RM-324: an exit code says only that something answered. The backend
    already classifies the banner, and a build below the floor fails later
    in the run, so the startup state has to carry that verdict rather than
    reporting a bare "ready".
    """
    from backend.ffmpeg_profiles import probe_ffmpeg_security

    try:
        state = dict(probe_ffmpeg_security(timeout=timeout))
    except Exception:
        logger.warning("FFmpeg security probe failed", exc_info=True)
        return {
            "available": False,
            "onPath": False,
            "parsed": False,
            "classification": "unknown",
            "version": "",
            "raw": "",
            "probeError": "the FFmpeg probe could not be run",
            "reason": "the FFmpeg probe could not be run",
        }
    # probe_ffmpeg_security reports `available` from shutil.which, so a
    # truncated or non-zero-exiting ffmpeg.exe on PATH still reads as
    # available with the failure only in `error`. The old boolean probe ran
    # the binary, so keep that meaning: available is "it answered".
    state["onPath"] = bool(state.get("available"))
    state["available"] = bool(
        state.get("available")
        and not state.get("probeError")
        and str(state.get("raw") or "").strip()
    )
    return state


def detect_ffmpeg() -> bool:
    """Whether an FFmpeg binary answered at all.

    This is the audio-merge capability question, not the security one: a
    build below the floor still merges audio. Use `detect_ffmpeg_state` for
    anything that reports readiness to the user.
    """
    return bool(detect_ffmpeg_state().get("available"))


def ffmpeg_status_summary(state) -> dict:
    """What the interface should say about one FFmpeg probe result.

    Returns the short header chip text, the startup status phrase, the
    warning-label copy (empty when there is nothing to warn about), and a
    tone name. A build below the enforced floor, or one whose version
    cannot be identified, is never described as ready.
    """
    from backend.ffmpeg_profiles import ffmpeg_security_floor_str

    payload = dict(state or {})
    available = bool(payload.get("available"))
    classification = str(payload.get("classification") or "unknown")
    version = str(payload.get("version") or "")
    floor = ffmpeg_security_floor_str()

    if not available:
        if payload.get("onPath"):
            # On PATH but it did not answer: a truncated download, a broken
            # install, or a binary that exits non-zero.
            return {
                "short": tr("FFmpeg broken"),
                "status": tr("FFmpeg did not run"),
                "warning": tr(
                    "An FFmpeg was found but it did not run, so outputs will "
                    "be saved without original audio. Reinstall it from a "
                    "stable {floor} or newer release build."
                ).format(floor=floor),
                "tone": "warning",
                "available": False,
                "safe": False,
            }
        return {
            "short": tr("No FFmpeg"),
            "status": tr("FFmpeg missing"),
            "warning": tr(
                "FFmpeg is not available, so outputs will be saved without "
                "original audio until it is installed."),
            "tone": "warning",
            "available": False,
            "safe": False,
        }
    if classification == "safe":
        return {
            "short": tr("FFmpeg {version}").format(version=version),
            "status": tr("FFmpeg {version}").format(version=version),
            "warning": "",
            "tone": "success",
            "available": True,
            "safe": True,
        }
    if not payload.get("parsed"):
        return {
            "short": tr("FFmpeg unclassified"),
            "status": tr("FFmpeg version not identified"),
            "warning": tr(
                "This FFmpeg build does not report a release version, so it "
                "cannot be checked against the required {floor} or newer. "
                "Install a stable release build.").format(floor=floor),
            "tone": "warning",
            "available": True,
            "safe": False,
        }
    return {
        "short": tr("FFmpeg {version}").format(version=version),
        "status": tr("FFmpeg {version} below the security floor").format(
            version=version),
        "warning": tr(
            "FFmpeg {version} is below the required {floor}. Processing will "
            "stop when it reaches the security check. Install {floor} or "
            "newer.").format(version=version, floor=floor),
        "tone": "warning",
        "available": True,
        "safe": False,
    }


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
    worst: Optional[dict] = None
    for position, report in enumerate(reports):
        if not report:
            continue
        try:
            psnr = float(report.get("psnr"))
            ssim = float(report.get("ssim"))
            samples = int(report.get("samples", 0) or 0)
        except (TypeError, ValueError):
            continue
        # RM-281: pool each report's OWN harmonic mean. Taking the harmonic
        # mean of the per-item averages would measure spread between clips,
        # which is not the defect this stat exists to expose.
        valid.append((
            psnr,
            ssim,
            samples,
            _optional_float(report.get("psnr_harmonic_mean")),
            _optional_float(report.get("ssim_harmonic_mean")),
        ))
        total_samples += max(0, samples)
        # RM-281: carry the single worst sampled frame across the batch, and
        # which report it came from, so the caller can open it.
        candidate = report.get("worst_frame")
        if isinstance(candidate, dict):
            try:
                candidate_ssim = float(candidate.get("ssim"))
                candidate_frame = int(candidate.get("frame"))
            except (TypeError, ValueError):
                continue
            if worst is None or candidate_ssim < worst["ssim"]:
                worst = {
                    "position": position,
                    "frame": candidate_frame,
                    "ssim": candidate_ssim,
                    "psnr": _optional_float(candidate.get("psnr")),
                }

    if not valid:
        return None

    count = len(valid)
    return {
        "psnr": sum(item[0] for item in valid) / count,
        "ssim": sum(item[1] for item in valid) / count,
        "harmonic_psnr": _mean_of([
            item[3] if item[3] is not None else item[0] for item in valid]),
        "harmonic_ssim": _mean_of([
            item[4] if item[4] is not None else item[1] for item in valid]),
        "items": count,
        "samples": total_samples,
        "worst_frame": worst,
    }


def _optional_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_of(values: List[float]) -> Optional[float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return None
    return sum(finite) / len(finite)


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
