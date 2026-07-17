"""
Backend Subtitle Removal Processor.

This module is the orchestration layer: ``ProcessingConfig``,
``InpaintMode``, ``normalize_processing_config``, the
``JsonLineLogHandler``, and the ``SubtitleRemover`` class that wires
everything together.

The implementation details for detection, inpainting, I/O, encoding,
quality metrics, and the CLI live in dedicated modules:

- ``backend.detection``         -- SubtitleDetector
- ``backend.tracking``          -- Kalman + karaoke + pHash
- ``backend.io``                -- captures, writer, ffprobe helpers, atomic file ops
- ``backend.encoder``           -- HW encoder probe
- ``backend.quality``           -- SSIM helper
- ``backend.inpainters``        -- BaseInpainter + 4 backends + TBE primitive
- ``backend.cli``               -- argparse + main()

For backward compatibility, every symbol that used to live here is
re-exported below so legacy callers (``from backend.processor import
_feather_blend``) keep working.
"""

import os
import sys
import json
import cv2
import datetime
import numpy as np
import logging
import shutil
import subprocess
import traceback
import time
from contextlib import contextmanager
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, List, Callable

logger = logging.getLogger(__name__)

# RFP-L-1 re-exports. Anything that used to be defined in this module
# but moved during the split is re-imported here so existing callers
# (`from backend.processor import _open_capture`) keep working.
from backend.io import (
    MediaInputError,
    SubtitleStreamInfo as SubtitleStreamInfo,
    _validate_video_input_file,
    _video_capture_open_error,
    _invalid_video_dimensions_error,
    _video_decode_error,
    _probe_codec_for_log,
    _probe_audio_stream_count as _probe_audio_stream_count,
    _probe_subtitle_streams as _probe_subtitle_streams,
    _probe_duration_seconds,
    _ffmpeg_subprocess_timeout,
    _probe_keyframe_indices,
    _probe_is_interlaced,
    _deinterlace_to_temp,
    _ensure_output_parent,
    _path_key,
    _choose_available_output_path as _choose_available_output_path,
    _write_text_atomic,
    _allocate_temp_output_path,
    _cleanup_temp_output,
    _promote_temp_output,
    _copy_file_atomic,
    validate_video_output,
    VideoFrameTiming,
    _probe_video_frame_timing,
    _FrameSequenceCapture as _FrameSequenceCapture,
    _open_capture,
    _open_bgr48_capture,
    _PrefetchReader,
    _LosslessIntermediateWriter,
    _FrameSequenceWriter,
    _run_subprocess_checked,
    _terminate_subprocess,
)
from backend.encoder import _detect_hw_encoder, probe_d3d12_encoder
from backend.device_provider import DeviceProvider, RuntimeDeviceProvider
from backend.container_payload import (
    build_container_mux_args,
    build_container_mux_plan,
    probe_container_manifest,
    validate_container_payload,
)
from backend.quality import (
    _ssim,
    compute_vmaf,
    compute_extended_metrics,
    temporal_consistency_score,
    residual_text_score,
    temporal_flicker_score,
    mask_boundary_seam_score,
)
from backend.quality_gate import (
    RESIDUAL_TEXT_SCORE_CEILING,
    TEMPORAL_FLICKER_CEILING,
    evaluate_quality_gate,
)
from backend.mask_corrections import (
    SELECTIVE_RERUN_SCHEMA,
    apply_mask_corrections,
    frame_is_in_ranges,
    make_review_span,
    merge_frame_ranges,
    merge_review_spans,
)
from backend.matte_interchange import (
    MaskInterchangeReader,
    MaskInterchangeWriter,
    compose_imported_matte,
    mask_interchange_paths,
)
from backend.resume_checkpoint import (
    ProcessingPaused,
    _checkpoint_key,
    _checkpoint_is_done as _checkpoint_is_done,
    _checkpoint_mark_done as _checkpoint_mark_done,
    _default_checkpoint_dir as _default_checkpoint_dir,
    cleanup_pause_checkpoint,
    config_fingerprint,
    load_pause_checkpoint,
    pause_frame_dir,
    write_pause_checkpoint,
)
from backend.safe_image import safe_imread
from backend.region_keyframes import region_shapes_at
from backend.work_directory import (
    StorageRequirement,
    assess_storage_volumes,
    make_work_temp_dir,
    resolve_work_directory,
)
from backend.tracking import (
    _KalmanBox as _KalmanBox,
    _box_from_state as _box_from_state,
    _iou as _iou,
    SubtitleTracker,
    _group_horizontal_line,
    _phash,
    _phash_distance,
    apply_clean_reference,
)
from backend.detection import SubtitleDetector, _surya_allowed as _surya_allowed
from backend.inpainters import (
    BaseInpainter,
    STTNInpainter,
    LAMAInpainter,
    ProPainterInpainter,
    AutoInpainter,
    is_oom_error,
    free_inference_memory,
    _cv2_inpaint,
    _feather_blend,
    _edge_ring_color_correct as _edge_ring_color_correct,
    _expand_mask_by_color,
    _detect_scene_cuts,
    _detect_scene_cuts_pyscenedetect as _detect_scene_cuts_pyscenedetect,
    stabilize_masks_rolling_union,
    _farneback_winsize as _farneback_winsize,
    _warp_to_reference as _warp_to_reference,
    _warp_mask_to_reference as _warp_mask_to_reference,
    _tbe_single_segment as _tbe_single_segment,
    _temporal_background_expose as _temporal_background_expose,
)
class JsonLineLogHandler(logging.Handler):
    """One JSON record per line, structured for jq / grep across long
    batch runs.

    Public so the GUI (which has its own logging.basicConfig) can opt in
    by calling `attach_json_log()` from `VideoSubtitleRemover.py`. The
    text log keeps writing in parallel; this handler is purely additive.
    """

    def __init__(self, stream):
        super().__init__()
        self._stream = stream

    def close(self):
        try:
            self._stream.close()
        except Exception:
            pass
        super().close()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = {
                "ts": datetime.datetime.fromtimestamp(
                    record.created, tz=datetime.timezone.utc
                ).isoformat(timespec="milliseconds"),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
            if record.exc_info:
                payload["exc"] = "".join(
                    traceback.format_exception(*record.exc_info)
                ).rstrip()
            line = json.dumps(payload, ensure_ascii=True) + "\n"
            self._stream.write(line)
            self._stream.flush()
        except Exception:  # pragma: no cover -- best-effort logging
            self.handleError(record)


def attach_json_log(path: str) -> Optional[JsonLineLogHandler]:
    """Append-mode JSON-lines log handler attached to the root logger.

    Safe to call multiple times -- detects an already-attached handler
    pointing at the same path and skips. Returns the handler so callers
    can detach on shutdown if they want; returns None on open failure.
    """
    target = str(Path(path))
    root = logging.getLogger()
    for existing in root.handlers:
        if (isinstance(existing, JsonLineLogHandler)
                and getattr(existing, "_json_log_path", None) == target):
            return existing
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Bound the JSON log: roll to a single `.1` backup once it passes 10 MB
        # so an always-on structured log cannot grow without limit.
        try:
            log_path = Path(path)
            if log_path.is_file() and log_path.stat().st_size > 10 * 1024 * 1024:
                backup = log_path.with_suffix(log_path.suffix + ".1")
                os.replace(log_path, backup)
        except OSError:
            pass
        stream = open(path, "a", encoding="utf-8")
    except OSError as exc:
        logger.warning(f"Could not open JSON log {path}: {exc}")
        return None
    handler = JsonLineLogHandler(stream)
    handler._json_log_path = target
    handler.setLevel(logging.INFO)
    root.addHandler(handler)
    logger.info(f"JSON log enabled at {path}")
    return handler


# Config types and coercion moved to backend.config (RM-114
# follow-up); re-exported here so legacy callers keep working.
from backend.config import (
    InpaintMode as InpaintMode,
    RegisteredMode as RegisteredMode,
    ProcessingConfig,
    _MODE_ALIASES as _MODE_ALIASES,
    _coerce_bool as _coerce_bool,
    _coerce_int as _coerce_int,
    _coerce_float as _coerce_float,
    _coerce_text as _coerce_text,
    _coerce_rect as _coerce_rect,
    _coerce_rect_list as _coerce_rect_list,
    _coerce_backend_mode as _coerce_backend_mode,
    _coerce_backend_device as _coerce_backend_device,
    _load_json_config as _load_json_config,
    _apply_auto_band_override as _apply_auto_band_override,
    is_known_backend_mode as is_known_backend_mode,
    normalize_processing_config,
)


def _available_host_ram_gb() -> Optional[float]:
    """Best-effort available physical memory in GB; None when no probe
    works. Used to keep the adaptive TBE batch within host RAM."""
    try:
        import psutil  # type: ignore
        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        logger.debug("psutil host-memory probe failed", exc_info=True)
    try:
        if os.name == "nt":
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemoryStatusEx()
            stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return stat.ullAvailPhys / (1024 ** 3)
        else:
            page = os.sysconf("SC_PAGE_SIZE")
            avail = os.sysconf("SC_AVPHYS_PAGES")
            return page * avail / (1024 ** 3)
    except Exception:
        logger.debug("platform host-memory probe failed", exc_info=True)
    return None


def _seek_capture_to_frame(cap, target: int) -> int:
    """Position ``cap`` so the next ``read()`` returns frame ``target``.

    ``cap.set(CAP_PROP_POS_FRAMES, N)`` snaps to the nearest keyframe on
    long-GOP CFR sources with some OpenCV backends (MSMF/DSHOW), so a plain
    set can start processing a few frames off the requested ``--start`` time.
    Seek near the target, then grab-and-discard forward to land exactly on it.

    On backends that already position accurately (the bundled FFmpeg backend
    reports the requested logical index), the reported position equals
    ``target`` and no forward grab happens -- so this is a no-op there and
    never over-advances. Returns the resulting position.
    """
    target = max(0, int(target))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    if target == 0:
        return 0
    try:
        pos = int(round(float(cap.get(cv2.CAP_PROP_POS_FRAMES))))
    except Exception:
        pos = target
    if pos > target:
        # Overshot; restart and scan forward from the beginning.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        pos = 0
    while pos < target:
        if not cap.grab():
            break
        pos += 1
    return pos


@dataclass(frozen=True)
class _FrameRange:
    """Resolved processing window plus the per-frame timing needed by the
    encode/matte stages. Produced by ``_resolve_frame_range`` so the
    time-range math lives in one testable place instead of inline in
    ``process_video``."""

    time_start_s: float
    time_end_s: float
    start_frame: int
    end_frame: int
    frames_to_process: int
    selected_frame_durations: Optional[List[float]]
    processed_time_start: float
    processed_time_end: float
    matte_timestamps: List[float]
    matte_durations: List[float]
    matte_time_base: float


@dataclass(frozen=True)
class _FrameLoopCheckpoint:
    active: bool
    root: Optional[Path]
    key: Optional[str]
    state_path: Optional[Path]
    config_hash: str
    frame_dir: Optional[Path]
    timing_manifest_path: Optional[Path]
    input_path: str
    output_path: str
    pause_check: Optional[Callable[[], bool]]


@dataclass(frozen=True)
class _FrameLoopContext:
    start_frame: int
    end_frame: int
    frames_to_process: int
    fps: float
    width: int
    height: int
    total_frames: int
    frame_timing: Optional[VideoFrameTiming]
    high_bit_depth_surface: Any
    batch_size: int
    frame_skip: int
    rife_stride: int
    keyframe_set: Optional[set]
    whisper_spans: List[Tuple[int, int]]
    timed_region_spans: bool
    static_fixed_shapes: Any
    selective_ranges: List[Tuple[int, int]]
    reader: Any
    selective_cap: Any
    matte_reader: Any
    writer: Any
    matte_writer: Any
    checkpoint: _FrameLoopCheckpoint


@dataclass
class _FrameLoopState:
    frame_idx: int
    last_mask: Optional[np.ndarray]
    last_hash: Any
    tracker: Optional[SubtitleTracker]
    fixed_mask_cache: dict


@dataclass
class _FrameBatch:
    frames: List[np.ndarray] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)
    source_frames: List[Optional[np.ndarray]] = field(default_factory=list)
    passthrough_flags: List[bool] = field(default_factory=list)
    active_segments: List[Tuple[int, int]] = field(default_factory=list)

    def add(self, frame: np.ndarray, mask: np.ndarray,
            source_frame: Optional[np.ndarray], *, passthrough: bool) -> None:
        self.frames.append(frame)
        self.masks.append(mask)
        self.source_frames.append(source_frame)
        self.passthrough_flags.append(bool(passthrough))


def _frame_seconds(index: int, fps: float,
                   frame_timing: Optional[VideoFrameTiming] = None) -> float:
    """Return one frame index on the shared VFR/CFR processing clock."""
    if frame_timing is not None:
        return frame_timing.frame_time(index, fps)
    return float(index) / max(float(fps), 1.0)


def _spans_from_segments(segments, *, fps: float, total_frames: int,
                         frame_timing: Optional[VideoFrameTiming] = None
                         ) -> List[Tuple[int, int]]:
    """Convert Whisper time segments to frame spans on the shared clock."""
    valid_segments = [segment for segment in (segments or [])
                      if len(segment) >= 2]
    if frame_timing is not None:
        return [
            frame_timing.frame_range(
                float(segment[0]),
                float(segment[1]),
                total_frames,
            )
            for segment in valid_segments
        ]
    from backend.whisper_fallback import segments_to_frame_spans
    return segments_to_frame_spans(valid_segments, fps)


def _resolve_frame_range(cap, total_frames: int, fps: float,
                         frame_timing, time_start: Any,
                         time_end: Any) -> _FrameRange:
    """Resolve the [start, end) frame window from the configured time range.

    Guards against NaN/inf/negative seconds, seeks ``cap`` to the start
    frame, and raises ``ValueError`` when the window is empty. VFR sources
    route through ``frame_timing`` for exact frame<->time mapping; CFR
    sources use ``fps``. Extracted verbatim from ``process_video``.
    """
    def _sane_seconds(value: Any) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(v) or v < 0.0:
            return 0.0
        return v

    time_start_s = _sane_seconds(time_start)
    time_end_s = _sane_seconds(time_end)
    start_frame = 0
    end_frame = total_frames
    if frame_timing is not None:
        start_frame, end_frame = frame_timing.frame_range(
            time_start_s, time_end_s, total_frames)
    else:
        if time_start_s > 0:
            start_frame = max(
                0, min(total_frames - 1, int(time_start_s * fps)))
        if time_end_s > 0:
            end_frame = max(
                0, min(total_frames, int(time_end_s * fps)))
    if start_frame > 0:
        _seek_capture_to_frame(cap, start_frame)
    if end_frame <= start_frame:
        raise ValueError(
            f"Invalid time range: end ({time_end_s}s) "
            f"must be after start ({time_start_s}s)")
    frames_to_process = end_frame - start_frame
    selected_frame_durations = (
        frame_timing.range_durations(start_frame, end_frame, fps)
        if frame_timing is not None else None
    )
    processed_time_start = _frame_seconds(
        start_frame, fps, frame_timing)
    processed_time_end = (
        processed_time_start + sum(selected_frame_durations or [])
        if selected_frame_durations is not None
        else _frame_seconds(end_frame, fps)
    )
    matte_timestamps = [
        _frame_seconds(index, fps, frame_timing)
        for index in range(start_frame, end_frame)
    ]
    matte_durations = list(selected_frame_durations or (
        [1.0 / fps] * frames_to_process
    ))
    matte_time_base = (
        frame_timing.time_base
        if frame_timing is not None else 1.0 / fps
    )
    return _FrameRange(
        time_start_s=time_start_s,
        time_end_s=time_end_s,
        start_frame=start_frame,
        end_frame=end_frame,
        frames_to_process=frames_to_process,
        selected_frame_durations=selected_frame_durations,
        processed_time_start=processed_time_start,
        processed_time_end=processed_time_end,
        matte_timestamps=matte_timestamps,
        matte_durations=matte_durations,
        matte_time_base=matte_time_base,
    )


# RFP-L-2: each built-in inpainter registers itself below so the
# dispatch in SubtitleRemover._create_inpainter no longer needs an
# if-elif chain. Opt-in third-party backends can `register()` from
# their own module to inject a new mode without modifying core code.
from backend import inpainter_registry as _inpainter_registry

_inpainter_registry.register("sttn", lambda device, config: STTNInpainter(device, config))
_inpainter_registry.register("lama", lambda device, config: LAMAInpainter(device, config))
_inpainter_registry.register("propainter", lambda device, config: ProPainterInpainter(device, config))
_inpainter_registry.register("auto", lambda device, config: AutoInpainter(device, config))

# RM-25 / RM-26: optional ONNX backends (LaMa-ONNX, MI-GAN). Import
# triggers `maybe_register()` which checks env vars and only patches
# the registry when the user has opted in.
try:
    from backend import inpainters_onnx as _inpainters_onnx  # noqa: F401
except Exception as _exc:
    logger.debug(f"ONNX inpainters module did not load: {_exc}")

# RM-59..RM-65: opt-in diffusion inpainter scaffolds. Each registers
# ONLY when the user has set its enable env var; otherwise the import
# is a no-op.
try:
    from backend import inpainters_diffusion as _inpainters_diffusion  # noqa: F401
except Exception as _exc:
    logger.debug(f"Diffusion inpainters module did not load: {_exc}")

try:
    from backend.inpainters.external import ExternalInpainter, is_available as _ext_available
    if _ext_available():
        _inpainter_registry.register(
            "external",
            lambda device, config: ExternalInpainter(device, config),
        )
        logger.info("External inpainter registered via VSR_EXTERNAL_INPAINTER")
except Exception as _exc:
    logger.debug(f"External inpainter did not load: {_exc}")


class OutputIntegrityError(Exception):
    """Raised when a finished video fails validation before promotion.

    Carries the human-readable ``reason`` and the probe ``details`` so callers
    can log evidence and preserve the existing destination.
    """

    def __init__(self, reason: str, details: Optional[dict] = None):
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}


class SubtitleRemover:
    """Coordinates detection and inpainting to remove subtitles from videos/images."""

    _STAGE_TIMING_KEYS = (
        "decode",
        "ocr",
        "mask",
        "inpaint",
        "encode",
        "mux",
        "quality",
    )

    def __init__(self, config: ProcessingConfig = None, *,
                 device_provider: Optional[DeviceProvider] = None):
        self.config = normalize_processing_config(config or ProcessingConfig())
        from backend.subtitle_translation import validate_translation_config
        validate_translation_config(self.config)
        self._work_directory_resolution = None
        self.last_work_directory_warning: Optional[str] = None
        self._resolve_work_directory()
        self.device_provider = device_provider or RuntimeDeviceProvider(
            self.config.device)
        requested_device = self.config.device
        self.config.device = self.device_provider.probe_available()
        if self.config.device != requested_device:
            logger.warning(
                "Inference device fallback: %s -> %s",
                requested_device,
                self.config.device,
            )
        self.detector = SubtitleDetector(
            self.config.device,
            lang=self.config.detection_lang,
            vertical=self.config.detection_vertical,
        )
        self.inpainter = self._create_inpainter()
        self.on_progress: Optional[Callable[[float, str], None]] = None
        # Live-preview callback: invoked with a BGR numpy frame roughly every
        # `live_preview_stride` frames while processing. The GUI marshals this
        # to the preview pane. Kept as a plain attribute so CLI users who
        # don't need it pay nothing.
        self.on_preview_frame: Optional[Callable[[np.ndarray, int, int], None]] = None
        self.live_preview_stride: int = 6   # emit every Nth processed frame
        self._hw_encoder: Optional[str] = None
        # SRT collection: (frame_idx, text) per detection used for export
        self._srt_entries: List[Tuple[int, str]] = []
        # v3.12 quality report -- populated at end of process_video when
        # config.quality_report is on. None until the first run completes.
        self.last_quality_report: Optional[dict] = None
        self.last_stage_timings: dict[str, float] = self._empty_stage_timings()
        self.last_detection_stats: dict = self._empty_detection_stats()
        self._unique_detected_regions: List[Tuple[int, int, int, int]] = []
        # Actual user-visible output path for the last run. This may differ
        # from the requested path when FFmpeg cannot encode the requested
        # container and the lossless intermediate is salvaged as .mkv.
        self.last_output_path: Optional[str] = None
        self.last_error_message: Optional[str] = None
        self.last_error_reason: Optional[str] = None
        self.last_mask_export: dict = {
            "requested": False,
            "status": "not-requested",
            "path": "",
        }
        self.last_mask_import: dict = {
            "requested": False,
            "status": "not-requested",
            "manifest": "",
            "mode": "replace",
        }
        self.last_translation: dict = {
            "requested": bool(self.config.translation_enabled),
            "status": (
                "pending" if self.config.translation_enabled else "not-requested"
            ),
        }
        self._translation_burn_path = ""
        self._whisper_segments: list[tuple[float, float, str]] = []
        clean_reference_requested = self._clean_reference_requested()
        self.last_clean_reference: dict = {
            "requested": clean_reference_requested,
            "status": (
                "pending" if clean_reference_requested else "not-requested"
            ),
        }
        self._clean_reference_cache: dict[int, np.ndarray] = {}
        self._clean_reference_warned: set[int] = set()
        self.last_timing_report: dict = {
            "mode": "unknown",
            "frame_count": 0,
            "duration_seconds": 0.0,
            "time_base_seconds": 0.0,
            "average_fps": 0.0,
        }
        self.last_output_contract: dict = {}
        self.last_container_payload: dict = {}
        self.last_resume_warning: Optional[str] = None
        self.last_pause_checkpoint: Optional[dict] = None
        self.last_pause_checkpoint_path: Optional[str] = None
        # B-3: union-mask bbox accumulated while processing. The quality
        # report metric (PSNR/SSIM) used to be measured over the whole
        # frame, so the unchanged 80-95% of pixels dominated the score and
        # an awful inpaint could still report 'Good'. We track the bbox of
        # the union mask and the metric runs against that ROI only.
        self._quality_mask_bbox: Optional[Tuple[int, int, int, int]] = None
        # Mask-boundary seam scores accumulated during inpainting (the report
        # pass no longer has per-frame masks). Sampled to keep cost flat.
        self._seam_scores: List[float] = []
        self._seam_score_failure_logged = False
        # RM-73 partial: source color signalling, populated lazily inside
        # process_video once we know the input path. Used by _get_encode_args
        # to preserve HDR / BT.2020 tagging on the output.
        self._color_metadata = None
        self._output_contract = None
        self._hdr_codec_warning_logged = False
        self._hdr_software_warning_logged = False
        self._active_writer = None
        self._active_subprocess: Optional[subprocess.Popen] = None
        self._teardown_requested = False
        self._d3d12_probe: dict = {
            "schema": "vsr.d3d12_runtime.v1",
            "requested": bool(self.config.d3d12_accel),
            "available": False,
            "reason": "not requested",
        }
        self._d3d12_fallback_encoder: Optional[str] = None
        self._d3d12_status: dict = {
            "requested": bool(self.config.d3d12_accel),
            "selected_encoder": "software",
            "fallback_encoder": "software",
            "runtime_fallback": False,
            "probe": dict(self._d3d12_probe),
        }

        self._select_hw_encoder(self.config.output_codec)

        # Adaptive batch sizing -- probe free VRAM, scale sttn_max_load_num.
        # Defaults to the user-configured value on probe failure.
        if self.config.adaptive_batch and 'cuda' in self.config.device:
            pynvml = None
            nvml_started = False
            try:
                import pynvml  # type: ignore
                pynvml.nvmlInit()
                nvml_started = True
                h = pynvml.nvmlDeviceGetHandleByIndex(
                    int(self.config.device.split(':')[-1] or 0))
                info = pynvml.nvmlDeviceGetMemoryInfo(h)
                free_gb = info.free / (1024 ** 3)
                # Rough heuristic: 1080p TBE costs ~50 MB per frame (RGB + mask +
                # scratch). Scale target batch by (free_vram / safety_factor).
                safety = 6.0  # GB reserved for model + OS
                budget_gb = max(1.0, free_gb - safety)
                estimated_frames = int(budget_gb * 1024 / 50.0)
                # The TBE path stacks the whole batch as float32 numpy in
                # HOST RAM (~100 MB per 1080p frame incl. nanmedian
                # scratch), so a large-VRAM GPU must not push the batch
                # past what system memory can actually hold.
                host_gb = _available_host_ram_gb()
                if host_gb is not None:
                    host_budget_gb = max(1.0, host_gb - 4.0)
                    estimated_frames = min(
                        estimated_frames, int(host_budget_gb * 1024 / 100.0))
                target = max(8, min(512, estimated_frames))
                if target != self.config.sttn_max_load_num:
                    logger.info(
                        f"Adaptive batch: {self.config.sttn_max_load_num} -> {target} "
                        f"(free VRAM {free_gb:.1f} GB)")
                    self.config.sttn_max_load_num = target
            except Exception:
                logger.warning("Adaptive batch VRAM probe failed", exc_info=True)
            finally:
                if pynvml is not None and nvml_started:
                    try:
                        pynvml.nvmlShutdown()
                    except Exception:
                        logger.warning("NVML shutdown failed", exc_info=True)

        logger.info(f"Detector: {self.detector._engine_name} | "
                    f"Inpainter: {self.config.mode.value} | "
                    f"Device: {self.config.device}"
                    f"{' | HW encode: ' + self._hw_encoder if self._hw_encoder else ''}")

    def _empty_stage_timings(self) -> dict[str, float]:
        return {stage: 0.0 for stage in self._STAGE_TIMING_KEYS}

    def _reset_stage_timings(self) -> None:
        self.last_stage_timings = self._empty_stage_timings()

    @staticmethod
    def _empty_detection_stats() -> dict:
        return {
            "frames_total": 0,
            "frames_ocr": 0,
            "frames_skipped": 0,
            "unique_regions_detected": 0,
            "skip_reasons": {},
        }

    def _reset_detection_stats(self) -> None:
        self.last_detection_stats = self._empty_detection_stats()
        self._unique_detected_regions = []

    def _record_detection_skip(self, reason: str) -> None:
        self.last_detection_stats["frames_skipped"] += 1
        reasons = self.last_detection_stats["skip_reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1

    @staticmethod
    def _detection_box_iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if intersection <= 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    def _record_ocr_detection(self, boxes) -> None:
        self.last_detection_stats["frames_ocr"] += 1
        for raw_box in boxes or []:
            box = tuple(int(value) for value in raw_box[:4])
            if any(
                self._detection_box_iou(box, known) >= 0.7
                for known in self._unique_detected_regions
            ):
                continue
            self._unique_detected_regions.append(box)
        self.last_detection_stats["unique_regions_detected"] = len(
            self._unique_detected_regions)

    @contextmanager
    def _time_stage(self, stage: str):
        started = time.monotonic()
        try:
            yield
        finally:
            if stage not in self.last_stage_timings:
                self.last_stage_timings[stage] = 0.0
            self.last_stage_timings[stage] = round(
                self.last_stage_timings.get(stage, 0.0)
                + max(0.0, time.monotonic() - started),
                6,
            )

    def _set_active_subprocess(self, proc: Optional[subprocess.Popen]) -> None:
        self._active_subprocess = proc

    def _is_teardown_requested(self) -> bool:
        return bool(self._teardown_requested)

    def _run_checked_ffmpeg(self, cmd: List[str], timeout: float) -> None:
        _run_subprocess_checked(
            cmd,
            timeout=timeout,
            on_process=self._set_active_subprocess,
            cancel_check=self._is_teardown_requested,
        )

    def terminate_active_work(self, timeout: float = 2.0) -> None:
        """Terminate the currently active writer or ffmpeg process."""
        self._teardown_requested = True
        writer = self._active_writer
        if writer is not None and hasattr(writer, "terminate"):
            try:
                writer.terminate(timeout=timeout)
            except Exception:
                logger.warning("Active writer termination failed", exc_info=True)
        proc = self._active_subprocess
        if proc is not None:
            _terminate_subprocess(proc, timeout=timeout)
            if self._active_subprocess is proc:
                self._active_subprocess = None

    # -----------------------------------------------------------------
    # Auto subtitle-band detection
    # -----------------------------------------------------------------
    def detect_subtitle_band(self, video_path: str, probe_frames: int = 30,
                               bands: int = 12) -> Optional[Tuple[int, int, int, int]]:
        """Scan the first `probe_frames` of a video, run OCR, cluster the
        detected boxes by vertical band, and return a single bounding rect
        covering the densest band. Returns None if nothing useful was found.
        Bands are horizontal slabs of the frame height.
        """
        cap = _open_capture(
            video_path, self.config.decode_hw_accel,
            input_fps=self.config.input_fps,
        )
        if not cap.isOpened():
            return None
        try:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if h <= 0 or w <= 0:
                return None
            band_height = max(1, h // bands)
            band_boxes: dict = {}
            read = 0
            while read < probe_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                boxes = self.detector.detect(frame, self.config.detection_threshold)
                for (x1, y1, x2, y2) in boxes:
                    cy = (y1 + y2) // 2
                    band_idx = min(bands - 1, cy // band_height)
                    band_boxes.setdefault(band_idx, []).append((x1, y1, x2, y2))
                read += 1
            if not band_boxes:
                return None
            # Pick the band with the most detections
            best_idx = max(band_boxes.keys(), key=lambda k: len(band_boxes[k]))
            boxes = band_boxes[best_idx]
            if len(boxes) < max(3, probe_frames // 5):
                return None
            xs1 = min(b[0] for b in boxes)
            ys1 = min(b[1] for b in boxes)
            xs2 = max(b[2] for b in boxes)
            ys2 = max(b[3] for b in boxes)
            # Expand horizontally to the full frame width -- subtitles are
            # centered but vary width; be generous so we catch every line.
            xs1 = 0
            xs2 = w
            return (int(xs1), int(ys1), int(xs2), int(ys2))
        finally:
            cap.release()

    def _create_inpainter(self) -> BaseInpainter:
        """Construct the configured backend through the device strategy."""
        return self.device_provider.create_inpainter(
            self.config.mode.value,
            self.config.device,
            self.config,
        )

    def _is_inference_oom(self, exc: BaseException) -> bool:
        provider = getattr(self, "device_provider", None)
        check = getattr(provider, "is_oom_error", None)
        return bool(check(exc)) if callable(check) else is_oom_error(exc)

    def _free_inference_memory(self) -> None:
        provider = getattr(self, "device_provider", None)
        release = getattr(provider, "free_inference_memory", None)
        if callable(release):
            release()
        else:
            free_inference_memory()

    def _report_progress(self, progress: float, message: str):
        if self.on_progress:
            self.on_progress(progress, message)

    def _create_mask(self, frame_shape: Tuple[int, int], boxes: List[Tuple[int, int, int, int]],
                     padding: int = 5, frame: Optional[np.ndarray] = None,
                     confidences: Optional[List[float]] = None) -> np.ndarray:
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        base_dilate = self.config.mask_dilate_px
        use_conf_dilate = (
            self.config.confidence_weighted_dilation
            and confidences is not None
            and base_dilate > 0
        )

        for idx, (x1, y1, x2, y2) in enumerate(boxes):
            bx1 = max(0, x1 - padding)
            by1 = max(0, y1 - padding)
            bx2 = min(w, x2 + padding)
            by2 = min(h, y2 + padding)
            mask[by1:by2, bx1:bx2] = 255

            if use_conf_dilate:
                conf = confidences[idx] if idx < len(confidences) else 1.0
                scale = self.config.confidence_dilation_scale
                effective = int(base_dilate * (1.0 + (1.0 - conf) * scale))
                if effective > 0:
                    k = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (effective * 2 + 1, effective * 2 + 1))
                    box_mask = np.zeros((h, w), dtype=np.uint8)
                    box_mask[by1:by2, bx1:bx2] = 255
                    dilated = cv2.dilate(box_mask, k, iterations=1)
                    mask = cv2.bitwise_or(mask, dilated)

        if not use_conf_dilate and base_dilate > 0 and mask.max() > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (base_dilate * 2 + 1, base_dilate * 2 + 1))
            mask = cv2.dilate(mask, kernel, iterations=1)

        # RM-66: optional SAM 2 mask refinement. When the user has set
        # VSR_SAM2_CHECKPOINT we replace each box's coarse rectangle
        # with the SAM 2 segmentation prompted by that box. Tighter
        # mask = less inpaint area = cleaner output. Skips when the
        # caller didn't pass `frame` (SAM 2 needs pixels).
        if frame is not None and boxes and self.config.sam2_refine:
            try:
                from backend.segmentation import refine_mask_with_sam2
                mask = refine_mask_with_sam2(frame, boxes, mask, self.config.device)
            except Exception as exc:
                logger.debug(f"SAM 2 refinement skipped: {exc}")

        return mask

    def _apply_manual_mask_corrections(
        self,
        mask: np.ndarray,
        frame_seconds: float,
        frame_index: Optional[int] = None,
    ) -> np.ndarray:
        return apply_mask_corrections(
            mask,
            getattr(self.config, "manual_mask_corrections", None),
            frame_seconds,
            frame_index,
        )

    def _select_hw_encoder(self, codec: str) -> Optional[str]:
        """Select the opt-in D3D12 path only after its runtime smoke."""
        if not self.config.use_hw_encode:
            self._hw_encoder = None
            self._d3d12_fallback_encoder = None
            self._d3d12_probe = {
                "schema": "vsr.d3d12_runtime.v1",
                "requested": bool(self.config.d3d12_accel),
                "codec": codec,
                "available": False,
                "reason": "hardware encoding is disabled",
            }
            self._d3d12_status = {
                "requested": bool(self.config.d3d12_accel),
                "selected_encoder": "software",
                "fallback_encoder": "software",
                "runtime_fallback": False,
                "probe": dict(self._d3d12_probe),
            }
            return None
        prefer_d3d12 = bool(
            self.config.use_hw_encode and self.config.d3d12_accel)
        if prefer_d3d12:
            self._d3d12_probe = probe_d3d12_encoder(codec)
        else:
            self._d3d12_probe = {
                "schema": "vsr.d3d12_runtime.v1",
                "requested": bool(self.config.d3d12_accel),
                "codec": codec,
                "available": False,
                "reason": (
                    "hardware encoding is disabled"
                    if self.config.d3d12_accel else "not requested"
                ),
            }
        self._hw_encoder = _detect_hw_encoder(
            codec,
            prefer_d3d12=prefer_d3d12,
            d3d12_probe=self._d3d12_probe,
        )
        self._d3d12_fallback_encoder = None
        if self._using_d3d12_encoder():
            self._d3d12_fallback_encoder = _detect_hw_encoder(codec)
        self._d3d12_status = {
            "requested": bool(self.config.d3d12_accel),
            "selected_encoder": self._hw_encoder or "software",
            "fallback_encoder": self._d3d12_fallback_encoder or "software",
            "runtime_fallback": False,
            "probe": dict(self._d3d12_probe),
        }
        return self._hw_encoder

    def _using_d3d12_encoder(self) -> bool:
        return bool(
            self._hw_encoder
            and self._hw_encoder.endswith("_d3d12va")
            and self.config.use_hw_encode
        )

    def _d3d12_device_args(self) -> List[str]:
        if not self._using_d3d12_encoder():
            return []
        return [
            "-init_hw_device", "d3d12va=vsr_d3d12",
            "-filter_hw_device", "vsr_d3d12",
        ]

    def _fallback_after_hw_failure(self, reason: object) -> bool:
        """Move D3D12 to the established HW chain, then to software."""
        failed = self._hw_encoder
        if not failed:
            return False
        if failed.endswith("_d3d12va"):
            self._hw_encoder = getattr(
                self, "_d3d12_fallback_encoder", None)
        else:
            self._hw_encoder = None
        status = dict(getattr(self, "_d3d12_status", {}) or {})
        status.update({
            "selected_encoder": self._hw_encoder or "software",
            "runtime_fallback": True,
            "failed_encoder": failed,
            "fallback_reason": str(reason),
        })
        self._d3d12_status = status
        return True

    def _attach_d3d12_evidence(self, report: dict) -> dict:
        selected = getattr(self, "_hw_encoder", None) or "software"
        report["windows_d3d12"] = dict(
            getattr(self, "_d3d12_status", {}) or {
                "requested": False,
                "selected_encoder": selected,
            }
        )
        return report

    def _refine_masks_with_matanyone(self,
                                     frames: List[np.ndarray],
                                     masks: List[np.ndarray]) -> List[np.ndarray]:
        if not getattr(self.config, "matanyone_refine", False):
            return masks
        if not frames or not masks:
            return masks
        try:
            from backend.segmentation import refine_masks_with_matanyone
            return refine_masks_with_matanyone(frames, masks, self.config.device)
        except Exception as exc:
            logger.debug(f"MatAnyone 2 refinement skipped: {exc}")
            return masks

    def _propagate_masks_with_cotracker(self,
                                        frames: List[np.ndarray],
                                        masks: List[np.ndarray]) -> List[np.ndarray]:
        if not getattr(self.config, "cotracker_propagate", False):
            return masks
        if not frames or not masks:
            return masks
        try:
            from backend.segmentation import propagate_masks_with_cotracker
            return propagate_masks_with_cotracker(
                frames,
                masks,
                device=self.config.device,
            )
        except Exception as exc:
            logger.debug(f"CoTracker3 propagation skipped: {exc}")
            return masks

    # -----------------------------------------------------------------
    # SRT export
    # -----------------------------------------------------------------
    def _collect_srt_entry(self, frame: np.ndarray, frame_idx: int,
                             boxes: List[Tuple[int, int, int, int]]):
        """Extract text strings for the detected boxes and append to the SRT
        buffer. We re-use the detector's already-loaded model where possible.
        """
        try:
            text = self._read_text_for_boxes(frame, boxes)
        except Exception:
            logger.warning("SRT text collection failed", exc_info=True)
            text = ""
        if text and getattr(self.config, "ocr_fix_enable", False):
            text = self._apply_ocr_fixes(text)
        if text:
            self._srt_entries.append((frame_idx, text))

    def _apply_ocr_fixes(self, text: str) -> str:
        """Apply the per-language OCR-fix replace list to detected SRT text.
        Loaded once per job and cached on the instance."""
        replacements = getattr(self, "_ocr_fix_replacements", None)
        if replacements is None:
            try:
                from backend.ocr_fix import load_ocr_fix_replacements
                replacements = load_ocr_fix_replacements(
                    getattr(self.config, "detection_lang", "en"))
            except Exception:
                logger.warning("OCR-fix list load failed", exc_info=True)
                replacements = {}
            self._ocr_fix_replacements = replacements
        if not replacements:
            return text
        try:
            from backend.ocr_fix import apply_ocr_fixes
            return apply_ocr_fixes(text, replacements)
        except Exception:
            logger.warning("OCR-fix application failed", exc_info=True)
            return text

    def _read_text_for_boxes(self, frame: np.ndarray,
                               boxes: List[Tuple[int, int, int, int]]) -> str:
        """Best-effort text extraction. Returns an empty string when the
        underlying engine doesn't expose a recognition path.
        """
        if not boxes:
            return ""
        # RapidOCR returns (poly, text, conf)
        if self.detector._rapid_model is not None:
            try:
                output = self.detector._rapid_model(frame)
                texts = []
                if isinstance(output, tuple) and output and output[0]:
                    for entry in output[0]:
                        if len(entry) >= 2 and entry[1]:
                            texts.append(entry[1])
                else:
                    txt_attr = getattr(output, 'txts', None)
                    if txt_attr:
                        texts.extend(t for t in txt_attr if t)
                return " ".join(texts).strip()
            except Exception:
                logger.warning("RapidOCR SRT extraction failed", exc_info=True)
        # PaddleOCR (line[1][0] is the recognised text)
        if self.detector._paddle_model is not None:
            try:
                results = self.detector._paddle_model.ocr(frame, cls=False)
                if results and results[0]:
                    return " ".join(line[1][0] for line in results[0] if line and line[1]).strip()
            except Exception:
                logger.warning("PaddleOCR SRT extraction failed", exc_info=True)
        # EasyOCR: readtext yields (bbox, text, conf)
        if self.detector._easyocr_reader is not None:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rows = self.detector._easyocr_reader.readtext(frame_rgb)
                return " ".join(r[1] for r in rows if len(r) >= 2 and r[1]).strip()
            except Exception:
                logger.warning("EasyOCR SRT extraction failed", exc_info=True)
        return ""

    def _accumulate_quality_bbox(self, mask: np.ndarray) -> None:
        """Update the union-mask bbox used by the quality report ROI.

        Tracking just the bbox keeps memory flat across long videos -- a
        full per-frame mask stack would be O(frames * H * W). The metric
        only needs a crop window large enough to contain every masked
        pixel ever seen, so the bbox is sufficient.
        """
        if mask is None or mask.size == 0 or mask.max() == 0:
            return
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            return
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        if self._quality_mask_bbox is None:
            self._quality_mask_bbox = (x1, y1, x2, y2)
        else:
            ox1, oy1, ox2, oy2 = self._quality_mask_bbox
            self._quality_mask_bbox = (
                min(ox1, x1), min(oy1, y1),
                max(ox2, x2), max(oy2, y2),
            )

    def _accumulate_seam_scores(self, frames, results, masks,
                                max_samples: int = 32) -> None:
        """Sample mask-boundary seam scores across a processed batch.

        Cheap and bounded: at most a few frames per batch and capped total
        so long videos keep flat cost. The mean feeds the quality report and
        the quality gate, which the report pass cannot compute because it no
        longer holds per-frame masks.
        """
        if len(self._seam_scores) >= max_samples:
            return
        n = min(len(frames), len(results), len(masks))
        if n == 0:
            return
        step = max(1, n // 3)
        for i in range(0, n, step):
            if len(self._seam_scores) >= max_samples:
                break
            try:
                score = mask_boundary_seam_score(frames[i], results[i], masks[i])
            except Exception:
                if not getattr(self, "_seam_score_failure_logged", False):
                    logger.warning(
                        "Seam-score sampling failed; the quality report may "
                        "omit boundary-seam evidence",
                        exc_info=True,
                    )
                    self._seam_score_failure_logged = True
                score = None
            if score is not None:
                self._seam_scores.append(score)

    def _compute_quality_report(self, input_path: str, output_path: str,
                                  start_frame: int, end_frame: int,
                                  fps: float, n_samples: int = 10) -> Optional[dict]:
        """Sample N random frames in [start_frame, end_frame), compute PSNR
        and SSIM between input and output on the frame as a whole. On the
        unmasked regions these should match almost exactly; divergence
        there indicates a pipeline bug (mis-configured feather, dilation,
        or encoder settings).

        When `quality_report_sheet` is set, also writes a side-by-side
        comparison PNG (original | cleaned per sampled frame) to
        `<output>.qualitysheet.png` so reviewers can scan visually
        instead of squinting at numeric metrics.

        Returns {'psnr', 'ssim', 'samples', 'tag', 'sheet'} or None.
        """
        # `_open_capture` handles the dir-vs-file split and honours the
        # HW-accel hint for video inputs.
        cap_in = _open_capture(
            input_path, self.config.decode_hw_accel,
            input_fps=self.config.input_fps,
        )
        # Force software decode on the output: the quality-report sample is
        # tiny (10 frames) and we want a deterministic decode path that
        # cannot fall back inconsistently. Honour the user's HW-accel hint
        # only for the source input above.
        cap_out = _open_capture(output_path, "off")
        if not cap_in.isOpened() or not cap_out.isOpened():
            try:
                cap_in.release()
            except Exception:
                logger.debug("Quality source capture release failed", exc_info=True)
            try:
                cap_out.release()
            except Exception:
                logger.debug("Quality output capture release failed", exc_info=True)
            return None
        try:
            span = max(1, end_frame - start_frame)
            out_total = int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT)) or span
            rng = np.random.default_rng(seed=42)
            metric_indices = sorted(set(rng.integers(0, span, size=n_samples).tolist()))
            metric_index_set = set(metric_indices)
            flicker_indices = sorted(set(
                metric_indices
                + [idx + 1 for idx in metric_indices if idx + 1 < span]
            ))

            psnrs: List[float] = []
            ssims: List[float] = []
            roi_psnrs: List[float] = []
            roi_ssims: List[float] = []
            temporal_samples: List[Tuple[int, np.ndarray]] = []
            residual_scores: List[float] = []
            review_spans = list(
                getattr(self, "_mask_review_signals", None) or [])
            # B-3: ROI-cropped metric uses the accumulated union-mask
            # bbox. Falls back to the whole-frame metric when the ROI is
            # too small (< 32px on either axis) for SSIM to be stable.
            roi = self._quality_mask_bbox
            roi_ready = (
                roi is not None
                and (roi[2] - roi[0]) >= 32
                and (roi[3] - roi[1]) >= 32
            )
            # Pairs kept for the optional sheet renderer. Each entry:
            # (frame_idx, original_bgr, cleaned_bgr, psnr, ssim)
            pairs: List[Tuple[int, np.ndarray, np.ndarray, float, float]] = []
            for idx in flicker_indices:
                cap_in.set(cv2.CAP_PROP_POS_FRAMES, start_frame + idx)
                ok_in, a = cap_in.read()
                cap_out.set(cv2.CAP_PROP_POS_FRAMES, min(out_total - 1, idx))
                ok_out, b = cap_out.read()
                if not (ok_in and ok_out):
                    continue
                if a.shape != b.shape:
                    b = cv2.resize(b, (a.shape[1], a.shape[0]),
                                    interpolation=cv2.INTER_AREA)
                a_roi = None
                b_roi = None
                if roi_ready:
                    x1, y1, x2, y2 = roi
                    x1 = max(0, min(a.shape[1] - 1, x1))
                    x2 = max(x1 + 1, min(a.shape[1], x2))
                    y1 = max(0, min(a.shape[0] - 1, y1))
                    y2 = max(y1 + 1, min(a.shape[0], y2))
                    a_roi = a[y1:y2, x1:x2]
                    b_roi = b[y1:y2, x1:x2]
                    if b_roi.size:
                        temporal_samples.append((idx, b_roi.copy()))
                        if idx in metric_index_set:
                            residual = residual_text_score(b_roi)
                            if residual is not None:
                                residual_scores.append(residual)
                                if residual > RESIDUAL_TEXT_SCORE_CEILING:
                                    review_spans.append(make_review_span(
                                        "residual",
                                        start_frame + idx,
                                        start_frame + idx + 1,
                                        fps=fps,
                                        score=residual,
                                        threshold=RESIDUAL_TEXT_SCORE_CEILING,
                                        reason=(
                                            "Residual text score exceeded "
                                            "the review threshold"
                                        ),
                                    ))
                if idx not in metric_index_set:
                    continue
                p = cv2.PSNR(a, b)
                s = _ssim(a, b)
                psnrs.append(p)
                ssims.append(s)
                # ROI metric: same frame, but cropped to the union-mask
                # bbox so the score reflects the inpaint quality instead
                # of the unchanged background.
                if a_roi is not None and b_roi is not None:
                    if a_roi.size and a_roi.shape == b_roi.shape:
                        try:
                            roi_psnrs.append(float(cv2.PSNR(a_roi, b_roi)))
                            roi_ssims.append(_ssim(a_roi, b_roi))
                        except Exception:
                            logger.warning(
                                "Quality ROI metric calculation failed",
                                exc_info=True,
                            )
                if self.config.quality_report_sheet:
                    pairs.append((idx, a, b, p, s))
            if not psnrs:
                return None
            mean_ssim = float(np.mean(ssims))
            mean_psnr = float(np.mean(psnrs))
            roi_mean_ssim = float(np.mean(roi_ssims)) if roi_ssims else None
            roi_mean_psnr = float(np.mean(roi_psnrs)) if roi_psnrs else None
            flicker_score = temporal_flicker_score(temporal_samples)
            for left, right in zip(temporal_samples, temporal_samples[1:]):
                if right[0] != left[0] + 1:
                    continue
                pair_score = temporal_flicker_score([left, right])
                if (
                    pair_score is not None
                    and pair_score > TEMPORAL_FLICKER_CEILING
                ):
                    review_spans.append(make_review_span(
                        "flicker",
                        start_frame + left[0],
                        start_frame + right[0] + 1,
                        fps=fps,
                        score=pair_score,
                        threshold=TEMPORAL_FLICKER_CEILING,
                        reason=(
                            "Adjacent cleaned frames exceeded "
                            "the flicker threshold"
                        ),
                    ))
            residual_mean_score = (
                float(np.mean(residual_scores)) if residual_scores else None
            )
            segment_duration = max(0.1, min(30.0, _frame_seconds(span, fps)))
            segment_start = _frame_seconds(start_frame, fps)
            vmaf = compute_vmaf(
                input_path,
                output_path,
                start_seconds=segment_start,
                duration_seconds=segment_duration,
            )
            roi_vmaf = None
            if roi_ready:
                roi_vmaf = compute_vmaf(
                    input_path,
                    output_path,
                    start_seconds=segment_start,
                    duration_seconds=segment_duration,
                    roi=roi,
                )
            # SSIM 0.95 is the common "visually indistinguishable" floor
            # for compressed video. Tag now uses the ROI score when
            # available -- that's the signal users actually care about.
            tag_ssim = roi_mean_ssim if roi_mean_ssim is not None else mean_ssim
            tag = "Good" if tag_ssim >= 0.95 else "Review"
            sheet_path = None
            if self.config.quality_report_sheet and pairs:
                try:
                    sheet_path = self._write_quality_sheet(
                        output_path, pairs, mean_psnr, mean_ssim, tag,
                    )
                except Exception as exc:
                    logger.warning(f"Quality sheet write failed: {exc}", exc_info=True)
            # RM-102: opt-in perceptual metrics (LPIPS, DISTS) via pyiqa.
            # Zero-cost when pyiqa is not installed -- the function
            # returns {} and these keys stay None in the report.
            extended = {}
            temporal_consistency = None
            if roi_ready and pairs:
                x1, y1, x2, y2 = roi
                x1 = max(0, min(pairs[0][1].shape[1] - 1, x1))
                x2 = max(x1 + 1, min(pairs[0][1].shape[1], x2))
                y1 = max(0, min(pairs[0][1].shape[0] - 1, y1))
                y2 = max(y1 + 1, min(pairs[0][1].shape[0], y2))
                roi_pairs = [
                    (a[y1:y2, x1:x2], b[y1:y2, x1:x2])
                    for (_, a, b, _, _) in pairs
                    if a[y1:y2, x1:x2].size > 0
                ]
                extended = compute_extended_metrics(roi_pairs)
                cleaned_roi_frames = [b for (_, b) in roi_pairs]
                temporal_consistency = temporal_consistency_score(
                    cleaned_roi_frames)
            elif pairs:
                extended = compute_extended_metrics(
                    [(a, b) for (_, a, b, _, _) in pairs])
                temporal_consistency = temporal_consistency_score(
                    [b for (_, _, b, _, _) in pairs])
            metrics = {
                'psnr': mean_psnr,
                'ssim': mean_ssim,
                'roi_psnr': roi_mean_psnr,
                'roi_ssim': roi_mean_ssim,
                'vmaf': vmaf,
                'roi_vmaf': roi_vmaf,
                'roi_bbox': list(roi) if roi else None,
                'temporal_flicker_score': flicker_score,
                'temporal_consistency': temporal_consistency,
                'residual_text_score': residual_mean_score,
                'seam_score': (
                    float(np.mean(getattr(self, '_seam_scores', None) or []))
                    if getattr(self, '_seam_scores', None) else None
                ),
                'lpips': extended.get('lpips'),
                'dists': extended.get('dists'),
                'samples': len(psnrs),
                'tag': tag,
                'sheet': sheet_path,
            }
            metrics["mask_review_spans"] = merge_review_spans(review_spans)
            if metrics["mask_review_spans"]:
                metrics["tag"] = "Review"
            metrics["quality_gate"] = evaluate_quality_gate(metrics)
            return metrics
        finally:
            cap_in.release()
            cap_out.release()

    def _write_quality_sheet(self,
                              output_path: str,
                              pairs: List[Tuple[int, np.ndarray, np.ndarray, float, float]],
                              mean_psnr: float,
                              mean_ssim: float,
                              tag: str,
                              max_row_h: int = 240) -> str:
        """Render the per-sample original | cleaned comparison sheet."""
        sheet_path = str(Path(output_path).with_suffix("")) + ".qualitysheet.png"
        gap = 6
        rows = []
        for idx, a, b, p, s in pairs:
            h = a.shape[0]
            scale = min(1.0, max_row_h / max(1, h))
            new_h = int(round(h * scale))
            new_w = int(round(a.shape[1] * scale))
            ar = cv2.resize(a, (new_w, new_h), interpolation=cv2.INTER_AREA)
            br = cv2.resize(b, (new_w, new_h), interpolation=cv2.INTER_AREA)
            sep = np.full((new_h, gap, 3), 32, dtype=np.uint8)
            row = np.concatenate([ar, sep, br], axis=1)
            # Caption strip below the row.
            caption_h = 26
            caption = np.full((caption_h, row.shape[1], 3), 16, dtype=np.uint8)
            text = f"Frame {idx}  PSNR={p:.2f} dB  SSIM={s:.4f}"
            cv2.putText(caption, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (220, 220, 220), 1, cv2.LINE_AA)
            rows.append(np.concatenate([row, caption], axis=0))
        # Stack rows top-to-bottom with a thin separator.
        body = []
        for i, r in enumerate(rows):
            if i:
                body.append(np.full((gap, r.shape[1], 3), 32, dtype=np.uint8))
            body.append(r)
        body_img = np.concatenate(body, axis=0)
        # Header strip.
        header_h = 56
        header = np.full((header_h, body_img.shape[1], 3), 10, dtype=np.uint8)
        title = f"VSR quality report  -  mean PSNR={mean_psnr:.2f} dB  mean SSIM={mean_ssim:.4f}  [{tag}]"
        cv2.putText(header, title, (10, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (245, 245, 245), 1, cv2.LINE_AA)
        sep = np.full((gap, body_img.shape[1], 3), 48, dtype=np.uint8)
        sheet = np.concatenate([header, sep, body_img], axis=0)
        cv2.imwrite(sheet_path, sheet, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        logger.info(f"Quality sheet written: {sheet_path}")
        return sheet_path

    def _write_srt(
        self,
        path: str,
        fps: float,
        offset_frames: int = 0,
        *,
        frame_timing: Optional[VideoFrameTiming] = None,
    ):
        """Collapse consecutive per-frame entries with the same text into SRT
        cues and write to disk. Gaps of up to 0.5s are bridged."""
        if not self._srt_entries:
            return
        fps = fps if fps and fps > 1.0 else 30.0
        gap_tol = max(1, int(fps * 0.5))

        def ts(t: float) -> str:
            ms = int(round(t * 1000))
            hh, rem = divmod(ms, 3600000)
            mm, rem = divmod(rem, 60000)
            ss, ms = divmod(rem, 1000)
            return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

        cues: List[Tuple[int, int, str]] = []
        cur_start, cur_end, cur_text = None, None, None
        for frame_idx, text in self._srt_entries:
            if cur_text is None:
                cur_start, cur_end, cur_text = frame_idx, frame_idx, text
                continue
            if frame_timing is not None:
                previous_end = (
                    frame_timing.frame_time(cur_end + offset_frames, fps)
                    + frame_timing.frame_duration(
                        cur_end + offset_frames, fps)
                )
                current_start = frame_timing.frame_time(
                    frame_idx + offset_frames, fps)
                bridge_gap = current_start - previous_end <= 0.5
            else:
                bridge_gap = frame_idx - cur_end <= gap_tol
            if text == cur_text and bridge_gap:
                cur_end = frame_idx
            else:
                cues.append((cur_start, cur_end, cur_text))
                cur_start, cur_end, cur_text = frame_idx, frame_idx, text
        if cur_text is not None:
            cues.append((cur_start, cur_end, cur_text))

        try:
            payload = []
            for i, (s, e, txt) in enumerate(cues, 1):
                if frame_timing is not None:
                    absolute_start = s + offset_frames
                    absolute_end = e + offset_frames
                    t_start = frame_timing.frame_time(absolute_start, fps)
                    t_end = (
                        frame_timing.frame_time(absolute_end, fps)
                        + frame_timing.frame_duration(absolute_end, fps)
                    )
                else:
                    t_start = (s + offset_frames) / fps
                    t_end = (e + offset_frames + 1) / fps
                payload.append(f"{i}\n{ts(t_start)} --> {ts(t_end)}\n{txt}\n\n")
            _write_text_atomic(Path(path), "".join(payload))
            logger.info(f"SRT written: {path} ({len(cues)} cues)")
        except Exception as exc:
            logger.warning(f"SRT write failed: {exc}", exc_info=True)

    def _prepare_translation_workflow(
        self,
        input_path: str,
        output_path: str,
        fps: float,
        offset_frames: int = 0,
        *,
        frame_timing: Optional[VideoFrameTiming] = None,
    ) -> None:
        """Resolve or generate the translated SRT before post-processing."""
        if not self.config.translation_enabled:
            return
        if self.config.restyle_subtitle:
            raise ValueError(
                "translation workflow cannot be combined with restyle_subtitle")
        if Path(output_path).is_dir():
            raise ValueError(
                "translation re-embedding requires encoded video output")

        from backend.subtitle_translation import (
            SubtitleTranslationError,
            provided_translation_evidence,
            render_segments_srt,
            translate_srt_file,
            translated_srt_path,
        )

        style_configured = bool(self.config.translation_style.strip())
        if self.config.translation_srt:
            translated_path = Path(self.config.translation_srt)
            report = provided_translation_evidence(
                translated_path,
                target_language=self.config.translation_target_lang,
            )
        else:
            source_kind = "provided-source-srt"
            if self.config.translation_source_srt:
                source_path = Path(self.config.translation_source_srt)
            else:
                source_path = (
                    Path(output_path).with_suffix(".srt")
                    if self.config.export_srt
                    else Path(output_path).with_name(
                        f"{Path(output_path).stem}.source.srt")
                )
                if self._srt_entries:
                    self._write_srt(
                        str(source_path),
                        fps,
                        offset_frames,
                        frame_timing=frame_timing,
                    )
                    source_kind = "ocr-srt"
                elif getattr(self, "_whisper_segments", None):
                    _write_text_atomic(
                        source_path,
                        render_segments_srt(self._whisper_segments),
                    )
                    source_kind = "whisper-srt"
                else:
                    raise SubtitleTranslationError(
                        "translation needs --translation-source-srt, OCR text, "
                        "or an enabled Whisper transcript")
            translated_path = translated_srt_path(
                output_path, self.config.translation_target_lang)
            report = translate_srt_file(
                source_path,
                translated_path,
                provider_name=self.config.translation_provider,
                source_language=self.config.translation_source_lang,
                target_language=self.config.translation_target_lang,
                provider_options={
                    "command": self.config.translation_command,
                    "timeout": self.config.translation_timeout_seconds,
                },
                source_kind=source_kind,
            )
        report["styleConfigured"] = style_configured
        report["mediaSource"] = Path(input_path).name
        self.last_translation = report
        self._translation_burn_path = str(translated_path)
        logger.info(
            "Translation captions ready: %s (%s, %d cues)",
            translated_path,
            report.get("provider", "unknown"),
            int(report.get("cueCount", 0) or 0),
        )

    def _fixed_region_shapes(
        self,
        time_seconds: Optional[float] = None,
    ) -> Optional[List[dict]]:
        """Return explicit manual-region shapes for the current time.

        Timed spans and moving keyframes intentionally override the legacy
        global fields: inactive ranges must stop masking rather than silently
        falling back to a broad global rectangle.
        """
        spans = getattr(self.config, "subtitle_region_spans", None)
        keyframe_tracks = getattr(
            self.config, "subtitle_region_keyframes", None)
        if spans or keyframe_tracks:
            try:
                seconds = float(time_seconds or 0.0)
            except (TypeError, ValueError):
                seconds = 0.0
            if not np.isfinite(seconds) or seconds < 0.0:
                seconds = 0.0
            active: List[dict] = []
            for span in spans or []:
                if not isinstance(span, dict):
                    continue
                rect = span.get("rect")
                if not rect:
                    continue
                try:
                    start = float(span.get("start", 0.0) or 0.0)
                    end = float(span.get("end", 0.0) or 0.0)
                except (TypeError, ValueError):
                    start, end = 0.0, 0.0
                if not np.isfinite(start) or start < 0.0:
                    start = 0.0
                if not np.isfinite(end) or end < 0.0:
                    end = 0.0
                if start <= seconds and (end <= 0.0 or seconds < end):
                    active.append({"rect": tuple(rect)})
            active.extend(region_shapes_at(keyframe_tracks, seconds))
            return active or None
        if self.config.subtitle_areas:
            return [{"rect": tuple(rect)} for rect in self.config.subtitle_areas]
        if self.config.subtitle_area:
            return [{"rect": tuple(self.config.subtitle_area)}]
        return None

    def _clean_reference_requested(self) -> bool:
        spans = getattr(self.config, "subtitle_region_spans", None) or []
        return any(
            isinstance(span, dict) and bool(span.get("clean_reference"))
            for span in spans
        )

    def _initialize_clean_references(self, width: int, height: int) -> None:
        """Load and fingerprint every timed-region clean plate once per job."""
        self._clean_reference_cache = {}
        self._clean_reference_warned = set()
        if not self._clean_reference_requested():
            self.last_clean_reference = {
                "requested": False,
                "status": "not-requested",
            }
            return
        from backend.reference_fill import (
            CLEAN_REFERENCE_SCHEMA,
            clean_reference_source_evidence,
        )

        records = []
        spans = getattr(self.config, "subtitle_region_spans", None) or []
        for span_index, span in enumerate(spans):
            spec = span.get("clean_reference") if isinstance(span, dict) else None
            if not spec:
                continue
            source = safe_imread(spec["path"])
            if source is None:
                raise ValueError(
                    f"Clean reference image could not be read: {spec['path']}")
            if source.ndim == 2:
                source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
            if source.shape[:2] != (height, width):
                raise ValueError(
                    "Clean reference dimensions must match the source video: "
                    f"expected {width}x{height}, got "
                    f"{source.shape[1]}x{source.shape[0]}")
            self._clean_reference_cache[span_index] = source
            records.append({
                "spanIndex": span_index,
                "startSeconds": float(span.get("start", 0.0)),
                "endSeconds": float(span.get("end", 0.0)),
                "rect": list(span["rect"]),
                "alignment": spec["alignment"],
                "minimumConfidence": float(spec["min_confidence"]),
                "colorMatch": bool(spec["color_match"]),
                "source": clean_reference_source_evidence(spec["path"]),
                "attemptedFrames": 0,
                "acceptedFrames": 0,
                "fallbackFrames": 0,
                "methodCounts": {},
                "minimumObservedConfidence": None,
                "maximumObservedConfidence": None,
                "_confidenceTotal": 0.0,
                "_colorDeltaTotal": [0.0, 0.0, 0.0],
            })
        self.last_clean_reference = {
            "schema": CLEAN_REFERENCE_SCHEMA,
            "requested": True,
            "status": "ready",
            "acceptedFrames": 0,
            "fallbackFrames": 0,
            "references": records,
        }

    def _apply_clean_reference_overrides(
        self,
        frame: np.ndarray,
        final_mask: np.ndarray,
        seconds: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Use active clean plates, leaving rejected pixels for inpainting."""
        if not self._clean_reference_cache or not np.any(final_mask > 0):
            return frame.copy(), final_mask.copy()
        composite = frame.copy()
        remaining = final_mask.copy()
        spans = getattr(self.config, "subtitle_region_spans", None) or []
        records = {
            int(record["spanIndex"]): record
            for record in self.last_clean_reference.get("references", [])
        }
        for span_index, reference in self._clean_reference_cache.items():
            span = spans[span_index]
            start = float(span.get("start", 0.0))
            end = float(span.get("end", 0.0))
            if seconds < start or (end > 0.0 and seconds >= end):
                continue
            x1, y1, x2, y2 = span["rect"]
            x1, x2 = max(0, min(frame.shape[1], x1)), max(
                0, min(frame.shape[1], x2))
            y1, y2 = max(0, min(frame.shape[0], y1)), max(
                0, min(frame.shape[0], y2))
            scoped_mask = np.zeros_like(remaining)
            scoped_mask[y1:y2, x1:x2] = remaining[y1:y2, x1:x2]
            if not np.any(scoped_mask > 0):
                continue
            result = apply_clean_reference(
                frame,
                reference,
                scoped_mask,
                span["clean_reference"],
                alignment_mask=final_mask,
            )
            record = records[span_index]
            record["attemptedFrames"] += 1
            record["_confidenceTotal"] += float(result.confidence)
            record["methodCounts"][result.method] = (
                int(record["methodCounts"].get(result.method, 0)) + 1)
            observed_min = record["minimumObservedConfidence"]
            observed_max = record["maximumObservedConfidence"]
            record["minimumObservedConfidence"] = (
                float(result.confidence) if observed_min is None
                else min(float(observed_min), float(result.confidence)))
            record["maximumObservedConfidence"] = (
                float(result.confidence) if observed_max is None
                else max(float(observed_max), float(result.confidence)))
            if result.accepted:
                record["acceptedFrames"] += 1
                self.last_clean_reference["acceptedFrames"] += 1
                for channel, value in enumerate(result.color_delta):
                    record["_colorDeltaTotal"][channel] += float(value)
                selected = scoped_mask > 0
                composite[selected] = result.composite[selected]
                remaining[selected] = 0
            else:
                record["fallbackFrames"] += 1
                record["lastFallbackReason"] = result.reason
                self.last_clean_reference["fallbackFrames"] += 1
                if span_index not in self._clean_reference_warned:
                    logger.warning(
                        "Clean reference %s fell back to inpainting: %s "
                        "(confidence %.3f)",
                        record["source"]["name"], result.reason,
                        result.confidence,
                    )
                    self._clean_reference_warned.add(span_index)
        return composite, remaining

    def _clean_reference_sidecar_evidence(self) -> Optional[dict]:
        if not self.last_clean_reference.get("requested"):
            return None
        payload = {
            key: value
            for key, value in self.last_clean_reference.items()
            if key != "references"
        }
        references = []
        for record in self.last_clean_reference.get("references", []):
            clean = {
                key: value
                for key, value in record.items()
                if not key.startswith("_")
            }
            attempted = int(record.get("attemptedFrames", 0))
            accepted = int(record.get("acceptedFrames", 0))
            if attempted:
                clean["meanConfidence"] = round(
                    float(record.get("_confidenceTotal", 0.0)) / attempted, 6)
            if accepted:
                totals = record.get("_colorDeltaTotal", [0.0, 0.0, 0.0])
                clean["meanColorDeltaBgr"] = [
                    round(float(value) / accepted, 3) for value in totals
                ]
            references.append(clean)
        payload["references"] = references
        if int(payload.get("acceptedFrames", 0)):
            payload["status"] = "applied"
        elif int(payload.get("fallbackFrames", 0)):
            payload["status"] = "fallback"
        else:
            payload["status"] = "unused"
        return payload

    def _fixed_region_boxes(
        self,
        time_seconds: Optional[float] = None,
    ) -> Optional[List[Tuple[int, int, int, int]]]:
        """Return active rectangle shapes for detection/mask creation."""
        shapes = self._fixed_region_shapes(time_seconds) or []
        boxes = [tuple(shape["rect"]) for shape in shapes if "rect" in shape]
        return boxes or None

    @staticmethod
    def _apply_polygon_region_shapes(
        mask: np.ndarray,
        shapes: Optional[List[dict]],
    ) -> np.ndarray:
        """Fill active polygon keyframes without widening them to bounds."""
        if not shapes:
            return mask
        h, w = mask.shape[:2]
        for shape in shapes:
            coords = shape.get("polygon") if isinstance(shape, dict) else None
            if not isinstance(coords, (list, tuple)) or len(coords) < 6:
                continue
            try:
                points = np.asarray(
                    [(int(coords[i]), int(coords[i + 1]))
                     for i in range(0, len(coords), 2)],
                    dtype=np.int32,
                )
                points[:, 0] = np.clip(points[:, 0], 0, w - 1)
                points[:, 1] = np.clip(points[:, 1], 0, h - 1)
                cv2.fillPoly(mask, [points], 255)
            except (TypeError, ValueError, IndexError):
                continue
        return mask

    def process_image(self, input_path: str, output_path: str) -> bool:
        self._teardown_requested = False
        self.last_output_path = None
        self._reset_stage_timings()
        self._reset_detection_stats()
        try:
            _ensure_output_parent(output_path)
            self._report_progress(0.1, "Loading image...")
            with self._time_stage("decode"):
                image = safe_imread(input_path)
            if image is None:
                raise ValueError(f"Could not load image: {input_path}")

            self._report_progress(0.3, "Detecting text regions...")
            fixed_shapes = self._fixed_region_shapes(0.0) or []
            fixed = self._fixed_region_boxes(0.0)
            confidences = None
            self.last_detection_stats["frames_total"] = 1
            with self._time_stage("ocr"):
                if fixed:
                    boxes = fixed
                    self._record_detection_skip("manual_region")
                elif fixed_shapes and self.config.sttn_skip_detection:
                    boxes = []
                    self._record_detection_skip("manual_region")
                elif self.config.confidence_weighted_dilation:
                    results = self.detector.detect_with_confidence(
                        image, self.config.detection_threshold)
                    boxes = [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in results]
                    confidences = [c for _, _, _, _, c in results]
                    self._record_ocr_detection(boxes)
                else:
                    boxes = self.detector.detect(image, self.config.detection_threshold)
                    self._record_ocr_detection(boxes)

            if not boxes and not fixed_shapes:
                logger.info("No text detected, copying original")
                with self._time_stage("encode"):
                    _copy_file_atomic(input_path, output_path)
                self.last_output_path = output_path
                self._write_reproducibility_sidecar(input_path, output_path)
                self.last_error_message = None
                self.last_error_reason = None
                self._report_progress(1.0, "Complete (no text found)")
                return True

            region_count = max(len(boxes), len(fixed_shapes))
            self._report_progress(0.5, f"Removing {region_count} text regions...")
            with self._time_stage("mask"):
                mask = self._create_mask(image.shape, boxes, frame=image,
                                         confidences=confidences)
                mask = self._apply_polygon_region_shapes(mask, fixed_shapes)
                mask = self._apply_manual_mask_corrections(mask, 0.0, 0)
                [mask] = self._refine_masks_with_matanyone([image], [mask])
            with self._time_stage("inpaint"):
                [result] = self.inpainter.inpaint([image], [mask])

            self._report_progress(0.9, "Saving result...")
            ext = Path(output_path).suffix.lower()
            temp_output = self._allocate_work_output(output_path)
            try:
                with self._time_stage("encode"):
                    if ext in ('.jpg', '.jpeg'):
                        ok = cv2.imwrite(str(temp_output), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    elif ext == '.png':
                        ok = cv2.imwrite(str(temp_output), result, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    elif ext == '.webp':
                        ok = cv2.imwrite(str(temp_output), result, [cv2.IMWRITE_WEBP_QUALITY, 95])
                    else:
                        ok = cv2.imwrite(str(temp_output), result)
                    if not ok:
                        raise IOError(f"Failed to write output image: {output_path}")
                    _promote_temp_output(temp_output, output_path)
            finally:
                _cleanup_temp_output(temp_output)
            self.last_output_path = output_path
            self._write_reproducibility_sidecar(input_path, output_path)
            self.last_error_message = None
            self.last_error_reason = None
            self._report_progress(1.0, "Complete!")
            return True

        except InterruptedError:
            logger.info("Image processing cancelled")
            raise
        except Exception as e:
            self.last_error_message = str(e)
            self.last_error_reason = "image_processing_error"
            logger.error(f"Image processing error: {e}", exc_info=True)
            return False

    def _resolve_work_directory(self):
        config = getattr(self, "config", None)
        requested = str(getattr(config, "work_directory", "") or "").strip()
        current = getattr(self, "_work_directory_resolution", None)
        if current is not None and current.requested == requested:
            return current
        resolution = resolve_work_directory(requested)
        self._work_directory_resolution = resolution
        self.last_work_directory_warning = resolution.warning or None
        if resolution.warning:
            logger.warning(resolution.warning)
        else:
            logger.info("Work directory: %s", resolution.path)
        return resolution

    def _make_temp_dir(self, *, prefix: str = "vsr_") -> str:
        return str(make_work_temp_dir(
            self._resolve_work_directory(), prefix=prefix))

    def _allocate_work_output(self, output_path: str) -> Path:
        return _allocate_temp_output_path(
            output_path,
            temp_dir=self._resolve_work_directory().path,
        )

    def _rife_fast_stride(self) -> int:
        try:
            stride = int(getattr(self.config, "rife_fast_stride", 0) or 0)
        except (TypeError, ValueError):
            return 0
        return stride if stride > 1 else 0

    @staticmethod
    def _valid_output_frame(candidate: Any,
                            fallback: np.ndarray) -> np.ndarray:
        if candidate is None:
            return fallback.copy()
        try:
            frame = np.asarray(candidate)
        except Exception:
            return fallback.copy()
        if frame.shape != fallback.shape:
            return fallback.copy()
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(frame)

    @staticmethod
    def _processing_frame(frame: np.ndarray) -> np.ndarray:
        """Return the uint8 BGR working copy expected by OCR/inpainters."""
        if frame.dtype == np.uint8:
            return np.ascontiguousarray(frame)
        if frame.dtype == np.uint16:
            return np.ascontiguousarray(
                np.clip(np.rint(frame.astype(np.float32) / 257.0), 0, 255)
                .astype(np.uint8)
            )
        return np.ascontiguousarray(np.clip(frame, 0, 255).astype(np.uint8))

    @staticmethod
    def _is_high_bit_frame(frame: Any) -> bool:
        return isinstance(frame, np.ndarray) and frame.dtype == np.uint16

    def _merge_high_bit_output(
        self,
        source_frame: Optional[np.ndarray],
        cleaned_frame: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """Blend the uint8 cleaned mask area back onto a uint16 source frame."""
        if not self._is_high_bit_frame(source_frame):
            return cleaned_frame
        if source_frame.shape != cleaned_frame.shape:
            return cleaned_frame
        if mask is None or mask.shape != source_frame.shape[:2] or not np.any(mask):
            return np.ascontiguousarray(source_frame)

        cleaned16 = np.clip(
            np.rint(cleaned_frame.astype(np.float32) * 257.0),
            0,
            65535,
        )
        feather = max(0, int(getattr(self.config, "mask_feather_px", 0) or 0))
        if feather > 0:
            k = feather * 2 + 1
            alpha = cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32) / 255.0
        else:
            alpha = (mask > 0).astype(np.float32)
        alpha = np.clip(alpha, 0.0, 1.0)[..., None]
        merged = (
            source_frame.astype(np.float32) * (1.0 - alpha)
            + cleaned16.astype(np.float32) * alpha
        )
        return np.ascontiguousarray(np.clip(np.rint(merged), 0, 65535).astype(np.uint16))

    def _rife_segment_has_scene_cut(self, frames: List[np.ndarray],
                                    start: int, end: int) -> bool:
        if end <= start + 1:
            return False
        segment = frames[start:end + 1]
        try:
            cuts = _detect_scene_cuts(
                segment,
                threshold=getattr(self.config, "tbe_scene_cut_threshold", 0.35),
                prefer_pyscenedetect=getattr(
                    self.config, "tbe_scene_cut_use_pyscenedetect", False),
                prefer_transnetv2=getattr(
                    self.config, "tbe_scene_cut_use_transnetv2", False),
            )
        except Exception as exc:
            logger.debug(f"RIFE scene-cut probe failed: {exc}")
            return False
        return any(cut > 0 for cut in cuts)

    def _inpaint_with_optional_rife_fast(self,
                                         frames: List[np.ndarray],
                                         masks: List[np.ndarray]) -> List[np.ndarray]:
        stride = self._rife_fast_stride()
        if stride <= 1 or len(frames) < 3:
            return self.inpainter.inpaint(frames, masks)

        key_indices = list(range(0, len(frames), stride))
        if key_indices[-1] != len(frames) - 1:
            key_indices.append(len(frames) - 1)
        if len(key_indices) >= len(frames):
            return self.inpainter.inpaint(frames, masks)

        key_frames = [frames[i] for i in key_indices]
        key_masks = [masks[i] for i in key_indices]
        key_results = self.inpainter.inpaint(key_frames, key_masks)
        if len(key_results) != len(key_indices):
            logger.warning(
                "RIFE fast mode disabled for batch: inpainter returned "
                f"{len(key_results)} keyframes for {len(key_indices)} inputs"
            )
            return self.inpainter.inpaint(frames, masks)

        results: List[Optional[np.ndarray]] = [None] * len(frames)
        for key_idx, cleaned in zip(key_indices, key_results):
            results[key_idx] = self._valid_output_frame(cleaned, frames[key_idx])

        try:
            from backend.decode_accel import maybe_interpolate_pair
        except Exception as exc:
            logger.debug(f"Could not import RIFE adapter: {exc}")
            maybe_interpolate_pair = None

        interpolation_missing_logged = False
        for left_pos, start_idx in enumerate(key_indices[:-1]):
            end_idx = key_indices[left_pos + 1]
            prev_clean = results[start_idx]
            next_clean = results[end_idx]
            if prev_clean is None or next_clean is None:
                continue

            scene_cut = self._rife_segment_has_scene_cut(
                frames, start_idx, end_idx)
            for out_idx in range(start_idx + 1, end_idx):
                t = (out_idx - start_idx) / max(1, end_idx - start_idx)
                fallback = prev_clean if t < 0.5 else next_clean
                if scene_cut or maybe_interpolate_pair is None:
                    results[out_idx] = fallback.copy()
                    continue
                interpolated = maybe_interpolate_pair(prev_clean, next_clean, t)
                if interpolated is None:
                    if not interpolation_missing_logged:
                        logger.info(
                            "RIFE fast mode is using nearest-keyframe fallback; "
                            "install practical-rife to synthesize intermediates."
                        )
                        interpolation_missing_logged = True
                    results[out_idx] = fallback.copy()
                    continue
                results[out_idx] = self._valid_output_frame(
                    interpolated, fallback)

        return [
            result if result is not None else frames[idx].copy()
            for idx, result in enumerate(results)
        ]

    def _inpaint_batch_resilient(self, frames: List[np.ndarray],
                                 masks: List[np.ndarray]) -> List[np.ndarray]:
        """Inpaint a batch, recovering from GPU OOM by shrinking and retrying.

        On an out-of-memory failure the CUDA cache is cleared and the batch is
        split in half and retried recursively down to a single frame; a frame
        that still cannot run on the GPU falls back to CPU (OpenCV) inpainting.
        The output list always has one frame per input, so a partial/corrupt
        write can never result from a recovered batch.
        """
        if not getattr(self.config, "gpu_oom_recovery", True):
            return self._inpaint_with_optional_rife_fast(frames, masks)
        try:
            return self._inpaint_with_optional_rife_fast(frames, masks)
        except Exception as exc:  # noqa: BLE001 - re-raised unless it is OOM
            if not self._is_inference_oom(exc):
                raise
            self._free_inference_memory()
            if len(frames) <= 1:
                logger.warning(
                    "GPU out of memory on a single frame; using CPU inpainting "
                    "fallback for this frame."
                )
                return self._inpaint_cpu_fallback(frames, masks)
            half = max(1, len(frames) // 2)
            logger.warning(
                "GPU out of memory on a batch of %d frames; clearing cache and "
                "retrying as %d + %d.", len(frames), half, len(frames) - half,
            )
            left = self._inpaint_batch_resilient(frames[:half], masks[:half])
            right = self._inpaint_batch_resilient(frames[half:], masks[half:])
            return left + right

    def _inpaint_cpu_fallback(self, frames: List[np.ndarray],
                              masks: List[np.ndarray]) -> List[np.ndarray]:
        """Guaranteed-CPU inpaint of a (usually single-frame) batch."""
        results: List[np.ndarray] = []
        for frame, mask in zip(frames, masks):
            filled = _cv2_inpaint(frame, mask, 5, cv2.INPAINT_TELEA)
            feather = getattr(self.config, "mask_feather_px", 0) or 0
            if feather > 0:
                filled = _feather_blend(frame, filled, mask, feather)
            results.append(self._valid_output_frame(filled, frame))
        return results

    def _decode_and_build_batch(self, ctx: _FrameLoopContext,
                                state: _FrameLoopState) -> _FrameBatch:
        """Decode frames and build masks until one processing batch is full."""
        batch = _FrameBatch()
        for _ in range(ctx.batch_size):
            if ctx.start_frame + state.frame_idx >= ctx.end_frame:
                break
            with self._time_stage("decode"):
                ret, raw_frame = ctx.reader.read()
                if not ret:
                    break
                source_frame = (
                    raw_frame
                    if ctx.high_bit_depth_surface
                    and self._is_high_bit_frame(raw_frame)
                    else None
                )
                frame = self._processing_frame(raw_frame)

            self.last_detection_stats["frames_total"] += 1
            absolute_idx = ctx.start_frame + state.frame_idx
            frame_seconds = _frame_seconds(
                absolute_idx, ctx.fps, ctx.frame_timing)
            if ctx.selective_cap is not None:
                prior_ok, prior_raw = ctx.selective_cap.read()
                if not prior_ok or prior_raw is None:
                    raise ValueError(
                        "Previous cleaned output ended during selective rerun"
                    )
                if not frame_is_in_ranges(
                    absolute_idx, ctx.selective_ranges
                ):
                    self._record_detection_skip("selective_rerun")
                    prior_frame = self._processing_frame(prior_raw)
                    batch.add(
                        prior_frame,
                        np.zeros(prior_frame.shape[:2], dtype=np.uint8),
                        None,
                        passthrough=True,
                    )
                    state.frame_idx += 1
                    continue
                if any(
                    absolute_idx == range_start
                    for range_start, _range_end in ctx.selective_ranges
                ):
                    state.last_mask = None
                    state.last_hash = None
                    if state.tracker is not None:
                        state.tracker = SubtitleTracker(
                            self.config.kalman_iou_threshold,
                            self.config.kalman_max_age,
                        )
            fixed_shapes = (
                self._fixed_region_shapes(frame_seconds)
                if ctx.timed_region_spans else ctx.static_fixed_shapes
            )
            fixed_boxes = (
                [tuple(shape["rect"]) for shape in fixed_shapes or []
                 if "rect" in shape]
                or None
            )

            if self.config.sttn_skip_detection and (
                    fixed_shapes or ctx.timed_region_spans):
                self._record_detection_skip("manual_region")
                if fixed_shapes:
                    has_polygon = any(
                        "polygon" in shape for shape in fixed_shapes)
                    dynamic_shape = ctx.timed_region_spans or has_polygon
                    mask_key = tuple(tuple(r) for r in (fixed_boxes or []))
                    fixed_mask = (
                        None if dynamic_shape
                        else state.fixed_mask_cache.get(mask_key)
                    )
                    if fixed_mask is None:
                        with self._time_stage("mask"):
                            fixed_mask = self._create_mask(
                                frame.shape, fixed_boxes or [])
                            fixed_mask = self._apply_polygon_region_shapes(
                                fixed_mask, fixed_shapes)
                        if not dynamic_shape:
                            state.fixed_mask_cache[mask_key] = fixed_mask
                else:
                    with self._time_stage("mask"):
                        fixed_mask = np.zeros(
                            frame.shape[:2], dtype=np.uint8)
                corrected = self._apply_manual_mask_corrections(
                    fixed_mask.copy(), frame_seconds, absolute_idx)
                batch.add(
                    frame, corrected, source_frame, passthrough=False)
                state.frame_idx += 1
                continue

            reuse_by_phash = False
            cur_hash = None
            if (not ctx.timed_region_spans
                    and self.config.phash_skip_enable
                    and state.last_mask is not None
                    and state.last_hash is not None):
                cur_hash = _phash(frame)
                if _phash_distance(
                    cur_hash, state.last_hash
                ) <= self.config.phash_skip_distance:
                    reuse_by_phash = True

            reuse_by_keyframe = False
            if (not ctx.timed_region_spans
                    and ctx.keyframe_set
                    and state.last_mask is not None):
                if absolute_idx not in ctx.keyframe_set:
                    reuse_by_keyframe = True

            if reuse_by_phash or reuse_by_keyframe:
                self._record_detection_skip(
                    "phash" if reuse_by_phash else "keyframe")
                batch.add(
                    frame, state.last_mask, source_frame, passthrough=False)
                state.frame_idx += 1
                continue
            if (not ctx.timed_region_spans
                    and ctx.frame_skip > 0
                    and state.last_mask is not None
                    and state.frame_idx % (ctx.frame_skip + 1) != 0):
                self._record_detection_skip("frame_skip")
                batch.add(
                    frame, state.last_mask, source_frame, passthrough=False)
                state.frame_idx += 1
                continue

            with self._time_stage("ocr"):
                if self.config.detection_denoise:
                    try:
                        from backend.preprocess import fastdvdnet_denoise_frame
                        det_frame = fastdvdnet_denoise_frame(frame)
                    except Exception as exc:
                        logger.warning(
                            f"Detection denoise fell back: {exc}",
                            exc_info=True,
                        )
                        det_frame = frame
                else:
                    det_frame = frame
                det_confs = None
                collect_confidence = bool(
                    self.config.confidence_weighted_dilation
                    or self.config.quality_report
                )
                if collect_confidence:
                    det_results = self.detector.detect_with_confidence(
                        det_frame, self.config.detection_threshold)
                    detected_boxes = [
                        (x1, y1, x2, y2)
                        for x1, y1, x2, y2, _ in det_results
                    ]
                    det_confs = [c for _, _, _, _, c in det_results]
                    if self.config.quality_report and det_confs:
                        review_floor = min(
                            0.9,
                            max(
                                0.6,
                                self.config.detection_threshold + 0.15,
                            ),
                        )
                        low_confidence = min(det_confs)
                        if low_confidence < review_floor:
                            self._mask_review_signals.append(
                                make_review_span(
                                    "low-confidence",
                                    absolute_idx,
                                    absolute_idx + 1,
                                    fps=ctx.fps,
                                    score=low_confidence,
                                    threshold=review_floor,
                                    reason=(
                                        "OCR confidence was below "
                                        "the review floor"
                                    ),
                                )
                            )
                    if not self.config.confidence_weighted_dilation:
                        det_confs = None
                else:
                    detected_boxes = self.detector.detect(
                        det_frame, self.config.detection_threshold)
                self._record_ocr_detection(detected_boxes)
                if self.config.karaoke_grouping and detected_boxes:
                    detected_boxes = _group_horizontal_line(
                        detected_boxes,
                        x_gap_px=self.config.karaoke_x_gap_px,
                        y_overlap_ratio=self.config.karaoke_y_overlap,
                    )
                    det_confs = None
                if state.tracker is not None:
                    smoothed = state.tracker.update(list(detected_boxes))
                    det_confs = None
                else:
                    smoothed = list(detected_boxes)
                if (state.tracker is not None
                        and (not self.config.remove_chyrons
                             or not self.config.remove_subtitles)):
                    cats = state.tracker.categorize(
                        self.config.chyron_min_hits)
                    smoothed = [
                        box for box, category in zip(smoothed, cats)
                        if (
                            category == "chyron"
                            and self.config.remove_chyrons
                        ) or (
                            category == "subtitle"
                            and self.config.remove_subtitles
                        )
                    ]
                    det_confs = None
                if fixed_boxes:
                    boxes = list(fixed_boxes) + smoothed
                    det_confs = None
                else:
                    boxes = smoothed
                if (
                    self.config.export_srt
                    or (
                        self.config.translation_enabled
                        and not self.config.translation_srt
                        and not self.config.translation_source_srt
                    )
                ):
                    self._collect_srt_entry(
                        frame, state.frame_idx, detected_boxes)

            if (not boxes and ctx.whisper_spans
                    and self.config.whisper_fallback):
                absolute = ctx.start_frame + state.frame_idx
                for span_start, span_end in ctx.whisper_spans:
                    if span_start <= absolute < span_end:
                        height, width = frame.shape[:2]
                        band_top = int(height * 0.80)
                        boxes = [(
                            int(width * 0.05),
                            band_top,
                            int(width * 0.95),
                            height - 4,
                        )]
                        break

            with self._time_stage("mask"):
                mask = self._create_mask(
                    frame.shape,
                    boxes,
                    frame=frame,
                    confidences=det_confs,
                )
                mask = self._apply_polygon_region_shapes(mask, fixed_shapes)
                if self.config.colour_tune_enable and boxes:
                    mask = _expand_mask_by_color(
                        frame,
                        mask,
                        boxes,
                        tolerance=self.config.colour_tune_tolerance,
                        padding=4,
                    )
                mask = self._apply_manual_mask_corrections(
                    mask, frame_seconds, absolute_idx)
            state.last_mask = mask
            if self.config.phash_skip_enable:
                state.last_hash = (
                    cur_hash if cur_hash is not None else _phash(frame)
                )
            batch.add(frame, mask, source_frame, passthrough=False)
            state.frame_idx += 1
        return batch

    def _refine_batch_masks(self, ctx: _FrameLoopContext,
                            state: _FrameLoopState,
                            batch: _FrameBatch) -> None:
        """Apply temporal/refiner/imported-matte passes to one batch."""
        with self._time_stage("mask"):
            segment_start = None
            for index, passthrough in enumerate(
                    batch.passthrough_flags + [True]):
                if not passthrough and segment_start is None:
                    segment_start = index
                elif passthrough and segment_start is not None:
                    batch.active_segments.append((segment_start, index))
                    segment_start = None
            for segment_start, segment_end in batch.active_segments:
                segment_frames = batch.frames[segment_start:segment_end]
                segment_masks = batch.masks[segment_start:segment_end]
                segment_masks = self._propagate_masks_with_cotracker(
                    segment_frames, segment_masks)
                segment_masks = self._refine_masks_with_matanyone(
                    segment_frames, segment_masks)
                if (self.config.temporal_mask_union
                        and not ctx.timed_region_spans
                        and not self.config.sttn_skip_detection
                        and len(segment_masks) > 1):
                    scene_starts = _detect_scene_cuts(segment_frames)
                    segment_masks = stabilize_masks_rolling_union(
                        segment_masks,
                        scene_starts,
                        self.config.temporal_mask_window,
                    )
                batch.masks[segment_start:segment_end] = segment_masks
            if ctx.matte_reader is not None:
                batch_start = state.frame_idx - len(batch.frames)
                for offset, passthrough in enumerate(
                        batch.passthrough_flags):
                    if passthrough:
                        continue
                    imported_matte = ctx.matte_reader.read(
                        batch_start + offset)
                    batch.masks[offset] = compose_imported_matte(
                        batch.masks[offset],
                        imported_matte,
                        ctx.matte_reader.mode,
                    )
            if self.config.quality_report:
                for index, mask in enumerate(batch.masks):
                    if not batch.passthrough_flags[index]:
                        self._accumulate_quality_bbox(mask)
        active_masks = [
            mask for index, mask in enumerate(batch.masks)
            if not batch.passthrough_flags[index]
        ]
        if active_masks:
            state.last_mask = active_masks[-1]
        progress = min(
            0.9,
            state.frame_idx / max(1, ctx.frames_to_process) * 0.8 + 0.1,
        )
        self._report_progress(
            progress,
            f"Processing frame {state.frame_idx}/{ctx.frames_to_process}...",
        )

    def _inpaint_batch(self, ctx: _FrameLoopContext,
                       state: _FrameLoopState,
                       batch: _FrameBatch) -> List[np.ndarray]:
        """Apply clean-reference overrides and inpaint active segments."""
        with self._time_stage("inpaint"):
            reference_frames = [frame.copy() for frame in batch.frames]
            fallback_masks = [mask.copy() for mask in batch.masks]
            if self._clean_reference_cache:
                batch_start = (
                    ctx.start_frame + state.frame_idx - len(batch.frames)
                )
                for offset, (frame, mask) in enumerate(zip(
                        batch.frames, batch.masks)):
                    if batch.passthrough_flags[offset]:
                        continue
                    absolute = batch_start + offset
                    seconds = _frame_seconds(
                        absolute, ctx.fps, ctx.frame_timing)
                    reference_frames[offset], fallback_masks[offset] = (
                        self._apply_clean_reference_overrides(
                            frame, mask, seconds)
                    )
            results = [frame.copy() for frame in reference_frames]
            for segment_start, segment_end in batch.active_segments:
                segment_masks = fallback_masks[segment_start:segment_end]
                if (
                    self._clean_reference_cache
                    and not any(np.any(mask > 0) for mask in segment_masks)
                ):
                    continue
                results[segment_start:segment_end] = (
                    self._inpaint_batch_resilient(
                        reference_frames[segment_start:segment_end],
                        segment_masks,
                    )
                )
            return results

    def _write_batch(self, ctx: _FrameLoopContext,
                     state: _FrameLoopState,
                     batch: _FrameBatch,
                     results: List[np.ndarray]) -> None:
        """Write cleaned frames, preview callbacks, and lossless mattes."""
        if self.config.quality_report and results:
            self._accumulate_seam_scores(
                batch.frames, results, batch.masks)
        stride = max(1, self.live_preview_stride)
        with self._time_stage("encode"):
            for offset, result in enumerate(results):
                write_frame = self._merge_high_bit_output(
                    batch.source_frames[offset]
                    if offset < len(batch.source_frames) else None,
                    result,
                    batch.masks[offset]
                    if offset < len(batch.masks) else None,
                )
                ctx.writer.write(write_frame)
        for offset, result in enumerate(results):
            frame_index = state.frame_idx - len(results) + offset
            if (self.on_preview_frame is not None
                    and frame_index % stride == 0):
                try:
                    self.on_preview_frame(
                        result,
                        frame_index + 1,
                        ctx.frames_to_process,
                    )
                except Exception as exc:
                    logger.warning(
                        f"on_preview_frame hook raised: {exc}",
                        exc_info=True,
                    )
        if ctx.matte_writer is not None:
            with self._time_stage("encode"):
                for mask in batch.masks:
                    ctx.matte_writer.write(mask)

    def _checkpoint_after_batch(self, ctx: _FrameLoopContext,
                                state: _FrameLoopState) -> None:
        """Persist running/paused state after one fully-written batch."""
        checkpoint = ctx.checkpoint
        if not checkpoint.active or checkpoint.root is None or not checkpoint.key:
            return
        should_pause = bool(
            checkpoint.pause_check and checkpoint.pause_check())
        payload = write_pause_checkpoint(
            checkpoint.root,
            checkpoint.key,
            input_path=checkpoint.input_path,
            output_path=checkpoint.output_path,
            config_hash=checkpoint.config_hash,
            frame_dir=checkpoint.frame_dir or pause_frame_dir(
                checkpoint.root, checkpoint.key),
            next_frame=state.frame_idx,
            total_frames=ctx.frames_to_process,
            width=ctx.width,
            height=ctx.height,
            fps=ctx.fps,
            status="paused" if should_pause else "running",
            timing_manifest_path=checkpoint.timing_manifest_path,
        )
        self.last_pause_checkpoint = payload
        if checkpoint.state_path is not None:
            self.last_pause_checkpoint_path = str(checkpoint.state_path)
        if should_pause:
            message = (
                f"Processing paused at frame "
                f"{state.frame_idx}/{ctx.frames_to_process}"
            )
            logger.info(message)
            raise ProcessingPaused(message, checkpoint.state_path)

    def process_video(self, input_path: str, output_path: str, *,
                      checkpoint_dir: Optional[str | Path] = None,
                      checkpoint_key: Optional[str] = None,
                      resume_checkpoint: bool = True,
                      pause_check: Optional[Callable[[], bool]] = None,
                      selective_rerun_from: Optional[str] = None,
                      selective_rerun_ranges: Optional[
                          List[Tuple[int, int]]
                      ] = None) -> bool:
        self._teardown_requested = False
        self.last_output_path = None
        self.last_resume_warning = None
        self.last_pause_checkpoint = None
        self.last_pause_checkpoint_path = None
        self._reset_stage_timings()
        self._reset_detection_stats()
        self._srt_entries = []
        self._ocr_fix_replacements = None
        self._quality_mask_bbox = None
        self._seam_scores = []
        self._seam_score_failure_logged = False
        self._mask_review_signals = []
        self.last_selective_rerun = None
        temp_dir = None
        cap = None
        selective_cap = None
        selective_ranges = merge_frame_ranges(selective_rerun_ranges or [])
        reader = None
        writer = None
        matte_writer = None
        matte_reader = None
        whisper_audio_dir = None
        self.last_error_message = None
        self.last_error_reason = None
        self.last_mask_export = {
            "requested": bool(self.config.export_mask_video),
            "status": (
                "pending" if self.config.export_mask_video else "not-requested"
            ),
            "path": "",
            "format": self.config.mask_export_format,
        }
        self.last_mask_import = {
            "requested": bool(self.config.mask_import_path),
            "status": "pending" if self.config.mask_import_path else "not-requested",
            "manifest": self.config.mask_import_path,
            "mode": self.config.mask_import_mode,
        }
        self.last_translation = {
            "requested": bool(self.config.translation_enabled),
            "status": (
                "pending" if self.config.translation_enabled else "not-requested"
            ),
        }
        self._translation_burn_path = ""
        self._whisper_segments = []
        clean_reference_requested = self._clean_reference_requested()
        self.last_clean_reference = {
            "requested": clean_reference_requested,
            "status": (
                "pending" if clean_reference_requested else "not-requested"
            ),
        }
        self._clean_reference_cache = {}
        self._clean_reference_warned = set()
        self.last_timing_report = {
            "mode": "unknown",
            "frame_count": 0,
            "duration_seconds": 0.0,
            "time_base_seconds": 0.0,
            "average_fps": 0.0,
        }
        self.last_output_contract = {}
        self.last_container_payload = {}
        self._color_metadata = None
        self._output_contract = None
        try:
            _ensure_output_parent(output_path)
            self._report_progress(0.0, "Opening video...")
            _validate_video_input_file(input_path)
            self._prepare_output_contract(input_path, output_path)

            # Optional deinterlace preprocessing. Produces a temp
            # progressive-scan mp4; the rest of the pipeline runs against
            # that file transparently.
            should_deinterlace = self.config.deinterlace
            if self.config.deinterlace_auto and not should_deinterlace:
                if _probe_is_interlaced(input_path):
                    logger.info("Interlaced source detected -- enabling yadif")
                    should_deinterlace = True
            if should_deinterlace:
                self._report_progress(0.02, "Deinterlacing source...")
                temp_dir = self._make_temp_dir()
                try:
                    processed_input = _deinterlace_to_temp(
                        input_path,
                        temp_dir,
                        output_contract=self._output_contract,
                        prefer_d3d12=(
                            self.config.d3d12_accel
                            and not self._source_is_hdr()
                        ),
                        on_process=self._set_active_subprocess,
                        cancel_check=self._is_teardown_requested,
                    )
                    logger.info(f"Using deinterlaced source: {processed_input}")
                    decode_path = processed_input
                except Exception as exc:
                    logger.warning(
                        f"Deinterlace failed, continuing with original: {exc}",
                        exc_info=True,
                    )
                    decode_path = input_path
            else:
                decode_path = input_path

            # Optional keyframe-driven detection: get the set of I-frame
            # indices once, OCR only those, propagate masks between.
            keyframe_set: Optional[set] = None
            if self.config.keyframe_detection:
                self._report_progress(0.04, "Probing keyframes...")
                keyframe_set = _probe_keyframe_indices(decode_path)
                if keyframe_set:
                    logger.info(f"Keyframe-driven detection: {len(keyframe_set)} I-frames")
                else:
                    logger.warning("Keyframe probe failed, falling back to pHash skip")

            cap = None
            if self._source_is_hdr():
                cap = _open_bgr48_capture(
                    decode_path,
                    input_fps=self.config.input_fps,
                )
                if cap is None:
                    logger.warning(
                        "HDR high-bit decode unavailable; falling back to "
                        "OpenCV BGR8 decode."
                    )
            if cap is None:
                cap = _open_capture(
                    decode_path,
                    self.config.decode_hw_accel,
                    input_fps=self.config.input_fps,
                )
            if not cap.isOpened():
                raise _video_capture_open_error(input_path, decode_path)
            # Stash the decode path so other routines (keyframe, audio merge)
            # can read it without re-resolving.
            self._decode_path = decode_path

            raw_fps = cap.get(cv2.CAP_PROP_FPS)
            # cv2 returns 0.0 on failure and can return NaN on exotic codecs;
            # both break downstream frame-to-time math, so coerce to a sane
            # default rather than let the pipeline divide by zero or NaN.
            try:
                raw_fps = float(raw_fps)
            except (TypeError, ValueError):
                raw_fps = 0.0
            if not np.isfinite(raw_fps) or raw_fps <= 0.0:
                logger.warning("Invalid / missing FPS metadata; falling back to 30.0")
                raw_fps = 30.0
            # Clamp absurdly high values (some malformed containers report
            # 1e6 FPS) so the writer doesn't stall on an impossible frame rate.
            fps = float(min(raw_fps, 1000.0))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            high_bit_depth_surface = (
                getattr(cap, "pixel_format", "") == "bgr48le"
            )
            frame_timing: Optional[VideoFrameTiming] = None
            if not Path(decode_path).is_dir():
                frame_timing = _probe_video_frame_timing(decode_path)
                if (
                    frame_timing is not None
                    and frame_timing.frame_count != total_frames
                ):
                    logger.warning(
                        "Using ffprobe's %d-frame PTS map instead of the "
                        "decoder's %d-frame header estimate",
                        frame_timing.frame_count,
                        total_frames,
                    )
                    total_frames = frame_timing.frame_count
            if frame_timing is not None:
                set_frame_timing = getattr(cap, "set_frame_timing", None)
                if callable(set_frame_timing):
                    set_frame_timing(frame_timing.timestamps)
                if frame_timing.average_fps > 0:
                    fps = float(min(frame_timing.average_fps, 1000.0))
                self.last_timing_report = frame_timing.report()
                timing_label = "variable" if frame_timing.is_vfr else "constant"
                logger.info(
                    "Source timing: %s frame rate, %d timestamps, "
                    "time base %.9fs",
                    timing_label,
                    frame_timing.frame_count,
                    frame_timing.time_base,
                )
            else:
                self.last_timing_report = {
                    "mode": "cfr-fallback",
                    "frame_count": total_frames,
                    "duration_seconds": round(total_frames / fps, 9),
                    "time_base_seconds": 0.0,
                    "average_fps": round(fps, 6),
                }

            if width == 0 or height == 0:
                raise _invalid_video_dimensions_error(input_path, width, height)
            self._initialize_clean_references(width, height)

            # Time range support. Resolved (with NaN/inf/negative guards and
            # the cap seek) in _resolve_frame_range so the frame<->time math is
            # unit-testable instead of buried inline here.
            _range = _resolve_frame_range(
                cap, total_frames, fps, frame_timing,
                self.config.time_start, self.config.time_end)
            start_frame = _range.start_frame
            end_frame = _range.end_frame
            frames_to_process = _range.frames_to_process
            selected_frame_durations = _range.selected_frame_durations
            processed_time_start = _range.processed_time_start
            processed_time_end = _range.processed_time_end
            matte_timestamps = _range.matte_timestamps
            matte_durations = _range.matte_durations
            matte_time_base = _range.matte_time_base
            if self.config.mask_import_path:
                matte_reader = MaskInterchangeReader(
                    self.config.mask_import_path,
                    width=width,
                    height=height,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    timestamps=matte_timestamps,
                    durations=matte_durations,
                    is_vfr=bool(
                        frame_timing is not None and frame_timing.is_vfr),
                    source_time_base=matte_time_base,
                    mode=self.config.mask_import_mode,
                )
                self.last_mask_import = {
                    **matte_reader.evidence,
                    "requested": True,
                    "status": "validated",
                }
                logger.info(
                    "Validated imported %s matte (%s mode, %d frames)",
                    matte_reader.export_format,
                    matte_reader.mode,
                    matte_reader.frame_count,
                )

            if selective_rerun_from and self.config.export_mask_video:
                logger.warning(
                    "A complete matte export was requested; running all frames "
                    "instead of reusing cleaned frames without their masks"
                )
                selective_rerun_from = None
                selective_ranges = []

            if selective_rerun_from:
                selective_path = Path(selective_rerun_from)
                if not selective_path.is_file():
                    raise ValueError(
                        "Selective mask rerun requires the previous cleaned output"
                    )
                selective_ranges = merge_frame_ranges(
                    (
                        max(start_frame, range_start),
                        min(end_frame, range_end),
                    )
                    for range_start, range_end in selective_ranges
                )
                if not selective_ranges:
                    raise ValueError(
                        "Selective mask rerun has no valid affected frame range"
                    )
                selective_cap = _open_capture(str(selective_path), "off")
                if not selective_cap.isOpened():
                    raise ValueError(
                        "Could not open the previous cleaned output for selective rerun"
                    )
                prior_width = int(selective_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                prior_height = int(selective_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                prior_frames = int(selective_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if (prior_width, prior_height) != (width, height):
                    raise ValueError(
                        "Previous cleaned output dimensions do not match the source"
                    )
                if prior_frames < end_frame:
                    raise ValueError(
                        "Previous cleaned output is missing frames required for selective rerun"
                    )
                prior_stat = selective_path.stat()
                rerun_frame_count = sum(end - start for start, end in selective_ranges)
                self.last_selective_rerun = {
                    "schema": SELECTIVE_RERUN_SCHEMA,
                    "source_output": selective_path.name,
                    "source_output_bytes": int(prior_stat.st_size),
                    "source_output_mtime_ns": int(prior_stat.st_mtime_ns),
                    "ranges": [list(frame_range) for frame_range in selective_ranges],
                    "rerun_frames": rerun_frame_count,
                    "reused_frames": max(0, frames_to_process - rerun_frame_count),
                }
                logger.info(
                    "Selective mask rerun: %d affected frame(s), %d reused frame(s)",
                    rerun_frame_count,
                    max(0, frames_to_process - rerun_frame_count),
                )

            if start_frame > 0 or end_frame < total_frames:
                logger.info(f"Video: {width}x{height} @ {fps:.1f}fps, "
                           f"frames {start_frame}-{end_frame} of {total_frames}")
            else:
                logger.info(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

            checkpoint_root: Optional[Path] = None
            checkpoint_frame_dir: Optional[Path] = None
            checkpoint_state_path: Optional[Path] = None
            checkpoint_config_hash = ""
            checkpoint_active = False
            checkpoint_remove_frames_on_success = True
            resume_frame_count = 0
            if checkpoint_dir is not None:
                checkpoint_root = Path(checkpoint_dir)
                checkpoint_root.mkdir(parents=True, exist_ok=True)
                checkpoint_key = checkpoint_key or _checkpoint_key(
                    input_path, output_path)
                checkpoint_state_path = (
                    checkpoint_root / f"{checkpoint_key}.pause.json"
                )
                checkpoint_config_hash = config_fingerprint(self.config)
                default_frame_dir = pause_frame_dir(
                    checkpoint_root, checkpoint_key)
                checkpoint_frame_dir = default_frame_dir
                checkpoint_active = True
                if resume_checkpoint:
                    state = load_pause_checkpoint(
                        checkpoint_root,
                        checkpoint_key,
                        input_path=input_path,
                        output_path=output_path,
                        config_hash=checkpoint_config_hash,
                        total_frames=frames_to_process,
                        width=width,
                        height=height,
                        fps=fps,
                    )
                    checkpoint_state_path = state.path
                    checkpoint_frame_dir = state.frame_dir
                    resume_frame_count = min(frames_to_process, state.next_frame)
                    if state.warning:
                        self.last_resume_warning = state.warning
                        logger.warning(state.warning)
                    if state.inpaint_complete and resume_frame_count >= frames_to_process:
                        logger.info(
                            f"Resuming {Path(input_path).name} at the encode "
                            "stage; all frames were already inpainted."
                        )
                    if resume_frame_count > 0:
                        _seek_capture_to_frame(
                            cap, start_frame + resume_frame_count)
                        logger.info(
                            f"Resuming {Path(input_path).name} from frame "
                            f"{resume_frame_count}/{frames_to_process}"
                        )

            if self.config.export_mask_video and resume_frame_count > 0:
                logger.warning(
                    "Restarting from frame zero so the lossless matte export "
                    "contains a complete, timestamp-aligned sequence"
                )
                resume_frame_count = 0
                _seek_capture_to_frame(cap, start_frame)

            if selective_cap is not None:
                _seek_capture_to_frame(
                    selective_cap, start_frame + resume_frame_count)

            # Fail fast on a drive that clearly cannot hold the encode, before
            # any temp file is created (only the frames still to write count).
            self._check_encode_disk_space(
                output_path,
                width=width,
                height=height,
                frames=max(0, frames_to_process - resume_frame_count),
                high_bit=bool(high_bit_depth_surface),
                checkpoint_dir=checkpoint_root,
            )

            # Re-use the deinterlace temp_dir if one was created, else fresh
            if temp_dir is None:
                temp_dir = self._make_temp_dir()
            # I-1: lossless FFV1 intermediate inside .mkv. The previous
            # mp4v intermediate cost a full generation of lossy
            # compression before the final ffmpeg encode pass. The
            # writer falls back to mp4v + .mp4 when ffmpeg is missing
            # so the pipeline still produces output, just at the old
            # quality.
            use_frame_output = getattr(self.config, "output_frames", False)
            vfr_frame_dir: Optional[Path] = None
            timing_manifest_path: Optional[Path] = None
            with self._time_stage("encode"):
                if use_frame_output:
                    frame_out_dir = output_path
                    if not frame_out_dir.endswith(os.sep):
                        frame_out_dir = str(Path(output_path).with_suffix(""))
                    if checkpoint_active:
                        checkpoint_frame_dir = Path(frame_out_dir)
                        checkpoint_remove_frames_on_success = False
                    writer = _FrameSequenceWriter(
                        frame_out_dir,
                        start_index=resume_frame_count,
                    )
                    temp_video = None
                elif checkpoint_active:
                    if checkpoint_frame_dir is None:
                        checkpoint_frame_dir = pause_frame_dir(
                            checkpoint_root, checkpoint_key)  # type: ignore[arg-type]
                    writer = _FrameSequenceWriter(
                        str(checkpoint_frame_dir),
                        start_index=resume_frame_count,
                    )
                    temp_video = None
                elif frame_timing is not None and frame_timing.is_vfr:
                    vfr_frame_dir = Path(temp_dir) / "vfr_frames"
                    writer = _FrameSequenceWriter(str(vfr_frame_dir))
                    temp_video = None
                else:
                    temp_video_target = os.path.join(temp_dir, "temp_video.mkv")
                    writer = _LosslessIntermediateWriter(
                        temp_video_target,
                        width,
                        height,
                        fps,
                        pixel_format="bgr48le" if high_bit_depth_surface else "bgr24",
                    )
                    self._active_writer = writer
                    temp_video = writer.path
                    if not writer.isOpened():
                        raise ValueError(
                            f"Could not create video writer for: {temp_video}"
                        )

            if (
                frame_timing is not None
                and frame_timing.is_vfr
            ):
                timing_dir = (
                    Path(frame_out_dir)
                    if use_frame_output else checkpoint_frame_dir or vfr_frame_dir
                )
                if timing_dir is not None:
                    timing_manifest_path = Path(timing_dir) / "frame_timing.json"
                    timing_payload = {
                        "schema": "vsr.frame_timing.v1",
                        "source": str(input_path),
                        "source_start_seconds": frame_timing.source_start,
                        "source_time_base_seconds": frame_timing.time_base,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "timestamps_seconds": [
                            frame_timing.frame_time(index, fps)
                            for index in range(start_frame, end_frame)
                        ],
                        "durations_seconds": selected_frame_durations,
                    }
                    _write_text_atomic(
                        timing_manifest_path,
                        json.dumps(
                            timing_payload,
                            indent=2,
                            ensure_ascii=True,
                        ) + "\n",
                    )

            if checkpoint_active and checkpoint_root is not None and checkpoint_key:
                payload = write_pause_checkpoint(
                    checkpoint_root,
                    checkpoint_key,
                    input_path=input_path,
                    output_path=output_path,
                    config_hash=checkpoint_config_hash,
                    frame_dir=checkpoint_frame_dir or pause_frame_dir(
                        checkpoint_root, checkpoint_key),
                    next_frame=resume_frame_count,
                    total_frames=frames_to_process,
                    width=width,
                    height=height,
                    fps=fps,
                    status="running",
                    timing_manifest_path=timing_manifest_path,
                )
                self.last_pause_checkpoint = payload
                if checkpoint_state_path is not None:
                    self.last_pause_checkpoint_path = str(checkpoint_state_path)

            frame_idx = resume_frame_count
            batch_size = self.config.sttn_max_load_num
            frame_skip = self.config.detection_frame_skip
            rife_stride = self._rife_fast_stride()
            if rife_stride:
                frame_skip = max(frame_skip, rife_stride - 1)
                logger.info(
                    f"RIFE fast mode on: stride={rife_stride}, "
                    f"effective detection frame-skip={frame_skip}"
                )

            # Decoupled prefetch: wrap the capture in a worker that fills
            # a bounded frame queue while the main thread runs detection +
            # inpainting. cv2.VideoCapture must NOT be touched directly
            # (.set / .get / .read) after this point -- the worker owns it.
            # Seek + metadata reads above happen *before* the wrap, so this
            # is safe; cleanup goes through `reader.release()`.
            with self._time_stage("decode"):
                if self.config.prefetch_decode:
                    qsize = self.config.prefetch_queue_size or max(8, batch_size * 2)
                    reader = _PrefetchReader(cap, max_frames=frames_to_process,
                                              queue_size=qsize)
                    logger.info(f"Prefetch decode on (queue={qsize})")
                else:
                    reader = cap
            last_mask = None  # cached mask for frame-skip optimization
            fixed_mask_cache = {}  # cached masks for skip_detection mode

            # RM-27 Whisper fallback: pre-compute frame spans where
            # Whisper detected speech. When OCR returns no boxes for a
            # frame inside one of these spans we apply a default
            # bottom-band mask so the subtitle band still gets
            # inpainted. Done once per file so we don't pay model load
            # for every batch.
            whisper_spans: List[Tuple[int, int]] = []
            whisper_audio_dir: Optional[str] = None
            segments = None
            if self.config.whisper_fallback and not Path(input_path).is_dir():
                try:
                    from backend import whisper_fallback as _wf
                    if self.config.whisper_backend == "ffmpeg":
                        segments = _wf.run_ffmpeg_whisper_segments(
                            input_path,
                            model_path=self.config.whisper_model_path,
                            language=(self.config.detection_lang or None),
                            queue_seconds=self.config.whisper_queue_seconds,
                            vad_model=self.config.whisper_vad_model,
                            vad_threshold=self.config.whisper_vad_threshold,
                            min_speech_duration=self.config.whisper_min_speech_duration,
                        )
                        if segments:
                            whisper_spans = _spans_from_segments(
                                segments,
                                fps=fps,
                                total_frames=total_frames,
                                frame_timing=frame_timing,
                            )
                            logger.info(
                                f"FFmpeg Whisper fallback active: "
                                f"{len(whisper_spans)} speech spans"
                            )
                    elif _wf.is_available():
                        whisper_audio_dir = self._make_temp_dir(
                            prefix="vsr_whisper_")
                        audio_path = _wf.extract_audio_to_temp(
                            input_path, whisper_audio_dir
                        )
                        if audio_path:
                            segments = _wf.run_whisper_segments(
                                audio_path,
                                model_size=self.config.whisper_model_size,
                                language=(self.config.detection_lang or None),
                            )
                            if segments:
                                whisper_spans = _spans_from_segments(
                                    segments,
                                    fps=fps,
                                    total_frames=total_frames,
                                    frame_timing=frame_timing,
                                )
                                logger.info(
                                    f"Whisper fallback active: "
                                    f"{len(whisper_spans)} speech spans"
                                )
                    if segments:
                        self._whisper_segments = [
                            (float(segment[0]), float(segment[1]), str(segment[2]))
                            for segment in segments
                            if len(segment) >= 3 and str(segment[2]).strip()
                        ]
                except Exception as exc:
                    logger.warning(
                        f"Whisper fallback setup failed: {exc}",
                        exc_info=True,
                    )

            # v3.10: Kalman tracker for detection smoothing
            tracker = (SubtitleTracker(self.config.kalman_iou_threshold,
                                         self.config.kalman_max_age)
                        if self.config.kalman_tracking else None)
            # v3.10: pHash for adaptive mask reuse
            last_hash = None

            # Lossless mask/alpha-matte interchange artifact.
            if self.config.export_mask_video:
                mask_path, mask_manifest_path = mask_interchange_paths(
                    output_path, self.config.mask_export_format)
                self.last_mask_export.update({
                    "path": str(mask_path),
                    "manifest": str(mask_manifest_path),
                })
                try:
                    matte_writer = MaskInterchangeWriter(
                        output_path,
                        self.config.mask_export_format,
                        width=width,
                        height=height,
                        fps=fps,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        timestamps=matte_timestamps,
                        durations=matte_durations,
                        is_vfr=bool(
                            frame_timing is not None and frame_timing.is_vfr),
                        source_time_base=matte_time_base,
                    )
                except Exception as exc:
                    self.last_mask_export.update({
                        "status": "failed",
                        "error": str(exc),
                    })
                    raise

            timed_region_spans = bool(
                getattr(self.config, "subtitle_region_spans", None)
                or getattr(self.config, "subtitle_region_keyframes", None)
            )
            static_fixed_shapes = (
                None if timed_region_spans else self._fixed_region_shapes())

            loop_ctx = _FrameLoopContext(
                start_frame=start_frame,
                end_frame=end_frame,
                frames_to_process=frames_to_process,
                fps=fps,
                width=width,
                height=height,
                total_frames=total_frames,
                frame_timing=frame_timing,
                high_bit_depth_surface=high_bit_depth_surface,
                batch_size=batch_size,
                frame_skip=frame_skip,
                rife_stride=rife_stride,
                keyframe_set=keyframe_set,
                whisper_spans=whisper_spans,
                timed_region_spans=timed_region_spans,
                static_fixed_shapes=static_fixed_shapes,
                selective_ranges=selective_ranges,
                reader=reader,
                selective_cap=selective_cap,
                matte_reader=matte_reader,
                writer=writer,
                matte_writer=matte_writer,
                checkpoint=_FrameLoopCheckpoint(
                    active=checkpoint_active,
                    root=checkpoint_root,
                    key=checkpoint_key,
                    state_path=checkpoint_state_path,
                    config_hash=checkpoint_config_hash,
                    frame_dir=checkpoint_frame_dir,
                    timing_manifest_path=timing_manifest_path,
                    input_path=input_path,
                    output_path=output_path,
                    pause_check=pause_check,
                ),
            )
            loop_state = _FrameLoopState(
                frame_idx=frame_idx,
                last_mask=last_mask,
                last_hash=last_hash,
                tracker=tracker,
                fixed_mask_cache=fixed_mask_cache,
            )
            while True:
                batch = self._decode_and_build_batch(loop_ctx, loop_state)
                if not batch.frames:
                    break
                self._refine_batch_masks(loop_ctx, loop_state, batch)
                results = self._inpaint_batch(loop_ctx, loop_state, batch)
                self._write_batch(loop_ctx, loop_state, batch, results)
                self._checkpoint_after_batch(loop_ctx, loop_state)
            frame_idx = loop_state.frame_idx

            if frame_idx < frames_to_process:
                raise _video_decode_error(
                    input_path,
                    decoded_frames=frame_idx,
                    expected_frames=frames_to_process,
                )

            # reader.release() (or cap.release() when prefetch is off)
            # also joins the worker thread and releases the underlying cap.
            reader.release()
            reader = None
            cap = None
            if selective_cap is not None:
                selective_cap.release()
                selective_cap = None
            with self._time_stage("encode"):
                writer.release()
            writer = None

            # Encode-stage marker: every frame is inpainted and on disk. If the
            # encode/mux below is interrupted, resume reloads this and jumps
            # straight to encoding instead of redoing detection/inpainting.
            if checkpoint_active and checkpoint_root is not None and checkpoint_key:
                write_pause_checkpoint(
                    checkpoint_root,
                    checkpoint_key,
                    input_path=input_path,
                    output_path=output_path,
                    config_hash=checkpoint_config_hash,
                    frame_dir=checkpoint_frame_dir or pause_frame_dir(
                        checkpoint_root, checkpoint_key),
                    next_frame=frames_to_process,
                    total_frames=frames_to_process,
                    width=width,
                    height=height,
                    fps=fps,
                    status="running",
                    timing_manifest_path=timing_manifest_path,
                    stage="encoding",
                    inpaint_complete=True,
                )
            if matte_reader is not None:
                matte_reader.close()
                self.last_mask_import["status"] = "composed"

            final_output_path, matte_writer = self._finalize_and_mux(
                input_path=input_path,
                output_path=output_path,
                temp_video=temp_video,
                temp_dir=temp_dir,
                fps=fps,
                start_frame=start_frame,
                end_frame=end_frame,
                width=width,
                height=height,
                use_frame_output=use_frame_output,
                frame_out_dir=frame_out_dir if use_frame_output else None,
                checkpoint_active=checkpoint_active,
                checkpoint_frame_dir=checkpoint_frame_dir,
                vfr_frame_dir=vfr_frame_dir,
                frame_timing=frame_timing,
                selected_frame_durations=selected_frame_durations,
                processed_time_start=processed_time_start,
                processed_time_end=processed_time_end,
                matte_writer=matte_writer,
            )

            self._emit_quality_report(
                input_path=input_path,
                final_output_path=final_output_path,
                start_frame=start_frame,
                end_frame=end_frame,
                fps=fps,
            )

            self.last_output_path = final_output_path
            self._write_reproducibility_sidecar(
                input_path, final_output_path,
                checkpoint_resumed=resume_frame_count > 0,
            )
            self.last_error_message = None
            self.last_error_reason = None
            if checkpoint_active and checkpoint_root is not None and checkpoint_key:
                remove_frames = (
                    checkpoint_remove_frames_on_success
                    and checkpoint_frame_dir is not None
                    and Path(final_output_path) != checkpoint_frame_dir
                )
                cleanup_pause_checkpoint(
                    checkpoint_root,
                    checkpoint_key,
                    remove_frames=remove_frames,
                )
            self._report_progress(1.0, "Complete!")
            return True

        except ProcessingPaused:
            logger.info("Video processing paused")
            raise
        except InterruptedError:
            logger.info("Video processing cancelled")
            raise
        except MediaInputError as e:
            self.last_error_message = e.user_message
            self.last_error_reason = e.reason
            logger.warning(
                "Video input rejected (%s): %s",
                e.reason,
                e.user_message,
            )
            if e.detail:
                logger.debug("Video input rejection detail: %s", e.detail)
            return False
        except Exception as e:
            self.last_error_message = str(e)
            self.last_error_reason = "video_processing_error"
            logger.error(f"Video processing error: {e}", exc_info=True)
            return False
        finally:
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    logger.warning("Video writer release failed", exc_info=True)
                finally:
                    if self._active_writer is writer:
                        self._active_writer = None
            if matte_writer is not None:
                try:
                    matte_writer.abort()
                except Exception:
                    logger.warning("Matte writer cleanup failed", exc_info=True)
            if matte_reader is not None:
                try:
                    matte_reader.close()
                except Exception:
                    logger.warning("Matte reader cleanup failed", exc_info=True)
            if self.last_mask_export.get("status") == "pending":
                self.last_mask_export.update({
                    "status": "failed",
                    "error": (
                        self.last_error_message
                        or "Processing ended before mask export completed"
                    ),
                })
            if self.last_mask_import.get("status") == "pending":
                self.last_mask_import.update({
                    "status": "failed",
                    "error": (
                        self.last_error_message
                        or "Processing ended before matte import was validated"
                    ),
                })
            # If a prefetch reader was set up, release it (which also stops
            # the worker thread and releases the underlying cap). Otherwise
            # release the raw cap. Tolerate either being unset on early
            # failures.
            if reader is not None:
                try:
                    reader.release()
                except Exception:
                    logger.warning("Prefetch reader release failed", exc_info=True)
            elif cap is not None:
                try:
                    cap.release()
                except Exception:
                    logger.warning("Input capture release failed", exc_info=True)
            if selective_cap is not None:
                try:
                    selective_cap.release()
                except Exception:
                    logger.warning(
                        "Selective-rerun capture release failed", exc_info=True)
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            # RM-27: Whisper audio temp dir is created lazily inside the
            # main try block; clean it up here only if it was set.
            try:
                if whisper_audio_dir and os.path.exists(whisper_audio_dir):
                    shutil.rmtree(whisper_audio_dir, ignore_errors=True)
            except Exception:
                logger.warning("Whisper temp cleanup failed", exc_info=True)

    def _finalize_and_mux(self, *, input_path: str, output_path: str,
                          temp_video, temp_dir, fps: float,
                          start_frame: int, end_frame: int,
                          width: int, height: int,
                          use_frame_output: bool, frame_out_dir,
                          checkpoint_active: bool, checkpoint_frame_dir,
                          vfr_frame_dir, frame_timing,
                          selected_frame_durations,
                          processed_time_start: float,
                          processed_time_end: float, matte_writer):
        """Encode/mux stage: assemble the final output file, finalize the
        matte, and write SRT/translation/NLE sidecars plus post-restore passes.

        Extracted verbatim from ``process_video``. Returns the resolved
        ``final_output_path`` together with the (now possibly ``None``)
        ``matte_writer`` so the caller's finally-block cleanup still observes
        the correct writer state: on success the writer is finalized and
        returned as ``None``; if any step raises, the live writer propagates
        back unchanged for the finally block to abort.
        """
        self._report_progress(0.9, "Preserving container streams...")
        with self._time_stage("mux"):
            if use_frame_output:
                logger.info(
                    f"Frame-sequence output written to {frame_out_dir}"
                )
                final_output_path = frame_out_dir
            elif checkpoint_active:
                assert checkpoint_frame_dir is not None
                processed_video = self._encode_frame_sequence(
                    checkpoint_frame_dir,
                    fps,
                    output_path,
                    frame_durations=(
                        selected_frame_durations
                        if frame_timing is not None and frame_timing.is_vfr
                        else None
                    ),
                    source_time_base=(
                        frame_timing.time_base
                        if frame_timing is not None else None
                    ),
                )
                final_output_path = processed_video
                is_frame_sequence_input = Path(input_path).is_dir()
                if not is_frame_sequence_input and not Path(processed_video).is_dir():
                    final_output_path = self._merge_audio(
                        input_path,
                        processed_video,
                        output_path,
                        start_seconds=processed_time_start,
                        end_seconds=processed_time_end,
                    )
            elif vfr_frame_dir is not None:
                processed_video = self._encode_frame_sequence(
                    vfr_frame_dir,
                    fps,
                    output_path,
                    frame_durations=selected_frame_durations,
                    source_time_base=frame_timing.time_base,
                )
                final_output_path = processed_video
                final_output_path = self._merge_audio(
                    input_path,
                    processed_video,
                    output_path,
                    start_seconds=processed_time_start,
                    end_seconds=processed_time_end,
                )
            else:
                final_output_path = output_path
                is_frame_sequence_input = Path(input_path).is_dir()
                if not is_frame_sequence_input:
                    final_output_path = self._merge_audio(
                        input_path,
                        temp_video,
                        output_path,
                        start_seconds=processed_time_start,
                        end_seconds=processed_time_end,
                    )
                else:
                    final_output_path = self._reencode_or_copy(
                        temp_video, output_path)
            if matte_writer is not None:
                with self._time_stage("encode"):
                    self.last_mask_export.update(matte_writer.finalize())
                matte_writer = None
                logger.info(
                    "Lossless matte written: %s",
                    self.last_mask_export.get("path"),
                )

            if self.config.export_srt and self._srt_entries:
                srt_path = str(Path(final_output_path).with_suffix('.srt'))
                self._write_srt(
                    srt_path,
                    fps,
                    start_frame,
                    frame_timing=frame_timing,
                )

            self._prepare_translation_workflow(
                input_path,
                final_output_path,
                fps,
                start_frame,
                frame_timing=frame_timing,
            )

            # RM-78 / RM-80: optional post-restore passes (Real-ESRGAN
            # upscale, film-grain re-synthesis). Run after the main mux
            # so the user-visible output is the post-processed file;
            # each adapter degrades gracefully when its dep is missing.
            self._run_post_restore_passes(final_output_path, temp_dir)
            if not use_frame_output:
                self._validate_output_contract(final_output_path)

            # RM-76: optional NLE round-trip sidecar (EDL / FCPXML).
            self._write_nle_sidecar(input_path, final_output_path,
                                     start_frame, end_frame, fps,
                                     width=width, height=height,
                                     start_seconds=processed_time_start,
                                     end_seconds=processed_time_end)
        return final_output_path, matte_writer

    def _emit_quality_report(self, *, input_path: str, final_output_path: str,
                             start_frame: int, end_frame: int,
                             fps: float) -> None:
        """Quality report: PSNR/SSIM across a sample of unmasked regions.

        Extracted verbatim from ``process_video``; records
        ``self.last_quality_report`` and logs the metrics. Failures are
        swallowed with a warning so reporting never aborts a finished encode.
        """
        if not self.config.quality_report:
            return
        try:
            with self._time_stage("quality"):
                metrics = self._compute_quality_report(
                    input_path, final_output_path, start_frame, end_frame, fps)
            if metrics:
                self.last_quality_report = metrics
                tag_suffix = f" [{metrics['tag']}]" if metrics.get('tag') else ""
                logger.info(
                    f"Quality report: PSNR={metrics['psnr']:.2f} dB, "
                    f"SSIM={metrics['ssim']:.4f} "
                    f"({metrics['samples']} samples){tag_suffix}")
                if metrics.get('vmaf') is not None:
                    logger.info(
                        f"Quality report VMAF={metrics['vmaf']:.2f}"
                        + (
                            f", ROI VMAF={metrics['roi_vmaf']:.2f}"
                            if metrics.get('roi_vmaf') is not None
                            else ""
                        )
                    )
                if metrics.get('sheet'):
                    logger.info(f"Quality sheet: {metrics['sheet']}")
        except Exception as exc:
            logger.warning(f"Quality report failed: {exc}", exc_info=True)

    def _write_nle_sidecar(self, input_path: str, output_path: str,
                             start_frame: int, end_frame: int,
                             fps: float, width: int = 0,
                             height: int = 0,
                             start_seconds: Optional[float] = None,
                             end_seconds: Optional[float] = None) -> None:
        """RM-76: emit an EDL or FCPXML sidecar next to the output so an
        NLE operator can hand-conform the cleaned clip into a Premiere
        / DaVinci timeline at the same timecode."""
        mode = self.config.nle_sidecar
        if mode not in ("edl", "fcpxml"):
            return
        try:
            from backend import nle_sidecar
        except Exception as exc:
            logger.debug(f"NLE sidecar module unavailable: {exc}")
            return
        try:
            if fps <= 0:
                fps = 30.0
            start_s = max(
                0.0,
                float(start_seconds)
                if start_seconds is not None else start_frame / fps,
            )
            end_s = max(
                start_s + 1.0 / fps,
                float(end_seconds)
                if end_seconds is not None else end_frame / fps,
            )
            spans = getattr(self.config, "subtitle_region_spans", None)
            keyframe_tracks = getattr(
                self.config, "subtitle_region_keyframes", None)
            segments = None
            if ((spans and isinstance(spans, list))
                    or (keyframe_tracks and isinstance(keyframe_tracks, list))):
                segments = []
                for span in (spans or []) + (keyframe_tracks or []):
                    if not isinstance(span, dict):
                        continue
                    s = max(0.0, float(span.get("start", 0.0)))
                    e = float(span.get("end", 0.0))
                    if e <= 0:
                        e = end_s
                    if e > s:
                        segments.append((s, e))
                if not segments:
                    segments = None
            base = str(Path(output_path).with_suffix(""))
            if mode == "edl":
                path = nle_sidecar.write_edl(
                    base + ".edl", input_path, output_path,
                    fps, start_s, end_s,
                    segments=segments, width=width, height=height,
                )
            else:
                path = nle_sidecar.write_fcpxml(
                    base + ".fcpxml", input_path, output_path,
                    fps, start_s, end_s,
                    segments=segments, width=width, height=height,
                )
            logger.info(f"NLE {mode.upper()} sidecar written: {path}")
        except Exception as exc:
            logger.warning(f"NLE sidecar write failed: {exc}", exc_info=True)

    def _write_reproducibility_sidecar(
        self,
        input_path: str,
        output_path: str,
        *,
        checkpoint_resumed: bool = False,
    ) -> None:
        try:
            from backend.batch_report import write_output_sidecar
            from gui.config import APP_VERSION
        except Exception:
            logger.debug(
                "Reproducibility sidecar dependencies unavailable",
                exc_info=True,
            )
            return
        try:
            quality_report = self.last_quality_report
            quality_gate = None
            if isinstance(quality_report, dict):
                quality_gate = quality_report.get("quality_gate")
            write_output_sidecar(
                input_path=input_path,
                output_path=output_path,
                config=self.config,
                status="processed",
                stage_timings=self.last_stage_timings,
                detection_stats=getattr(self, "last_detection_stats", None),
                quality_report=quality_report,
                quality_gate=quality_gate,
                output_contract=self.last_output_contract,
                selective_rerun=getattr(self, "last_selective_rerun", None),
                mask_export=(
                    self.last_mask_export
                    if self.last_mask_export.get("requested") else None
                ),
                mask_import=(
                    self.last_mask_import
                    if self.last_mask_import.get("requested") else None
                ),
                translation=(
                    self.last_translation
                    if self.last_translation.get("requested") else None
                ),
                clean_reference=self._clean_reference_sidecar_evidence(),
                checkpoint_resumed=checkpoint_resumed,
                app_version=APP_VERSION,
            )
        except Exception as exc:
            logger.warning(
                "Reproducibility sidecar write failed: %s", exc,
                exc_info=True,
            )

    def _prepare_output_contract(self, input_path: str, output_path: str) -> None:
        """Probe source media once and freeze the policy used by every pass."""
        meta = None
        if not Path(input_path).is_dir():
            try:
                from backend.hdr import probe_color_metadata

                meta = probe_color_metadata(input_path)
                codec_line = _probe_codec_for_log(input_path)
                if codec_line:
                    logger.info(f"Source codec: {codec_line}")
            except Exception:
                logger.warning("Source codec/color probe failed", exc_info=True)
        if self.config.preserve_color_metadata:
            self._color_metadata = meta
        requested = getattr(self.config, "output_codec", "h264")
        effective = requested
        if self.config.preserve_color_metadata:
            try:
                from backend.hdr import hdr_safe_codec

                effective = hdr_safe_codec(requested, meta)
            except Exception:
                logger.warning("HDR codec policy failed", exc_info=True)
        if effective != requested:
            logger.info(
                f"HDR output cannot use {requested}; promoting final encode "
                f"to {effective}."
            )
            self._hdr_codec_warning_logged = True
        if self.config.use_hw_encode and effective != requested:
            self._select_hw_encoder(effective)
        from backend.output_contract import build_output_contract

        self._output_contract = build_output_contract(
            input_path=input_path,
            output_path=output_path,
            codec=effective,
            preserve_audio=self.config.preserve_audio,
            preserve_color_metadata=self.config.preserve_color_metadata,
            color_metadata=meta,
            hardware_requested=self.config.use_hw_encode,
        )
        self.last_output_contract = self._attach_d3d12_evidence(
            self._output_contract.report())
        self.last_output_contract["container_payload"] = {
            "status": "pending",
        }
        if meta is not None:
            if meta.is_hdr:
                logger.info(
                    f"HDR source detected: {meta.label} -- output contract "
                    f"requires {effective}, 10-bit pixels, and source color tags."
                )
            else:
                logger.info(f"Color signalling: {meta.label}")
        for warning in self._output_contract.warnings:
            logger.warning(warning)

    def _validate_output_contract(self, output_path: str) -> None:
        contract = getattr(self, "_output_contract", None)
        if contract is None or Path(output_path).is_dir():
            return
        ok, issues = contract.validate(output_path)
        payload_issues = list((self.last_container_payload or {}).get("issues") or [])
        if (self.last_container_payload or {}).get("status") == "failed":
            issues.extend(f"container payload: {item}" for item in payload_issues)
            ok = False
        self.last_output_contract = self._attach_d3d12_evidence(
            contract.report())
        self.last_output_contract["container_payload"] = dict(
            self.last_container_payload or {"status": "not-probed"}
        )
        self.last_output_contract["status"] = "preserved" if ok else "failed"
        self.last_output_contract["issues"] = list(issues)
        color_preserved = contract.color_preserved(issues)
        self.last_output_contract["color_preserved"] = (
            color_preserved
            if isinstance(color_preserved, bool) or color_preserved is None
            else None
        )
        if color_preserved is False:
            logger.warning(
                "Output color metadata was not preserved: %s",
                "; ".join(issues),
            )
        if not ok:
            raise OutputIntegrityError(
                "output contract mismatch: " + "; ".join(issues),
                {"output_contract": self.last_output_contract},
            )

    def _remux_transformed_video(
        self,
        video_source: str,
        payload_source: str,
        output_path: str,
    ) -> dict:
        """Copy a transformed primary video while restoring container payload."""
        manifest = probe_container_manifest(payload_source)
        plan = build_container_mux_plan(
            manifest,
            output_path,
            preserve_audio=self.config.preserve_audio,
            multi_audio=True,
            loudnorm_target=0.0,
        )
        for warning in plan.get("warnings", []):
            logger.warning(warning)
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats",
            "-i", video_source,
            "-i", payload_source,
            "-map", "0:v:0",
            "-c:v", "copy",
        ]
        cmd += build_container_mux_args(plan, input_index=1)
        duration = _probe_duration_seconds(video_source)
        if duration > 0:
            cmd += ["-t", f"{duration:.9f}"]
        cmd += [output_path]
        self._run_checked_ffmpeg(
            cmd,
            _ffmpeg_subprocess_timeout(duration or _probe_duration_seconds(payload_source)),
        )
        report = validate_container_payload(plan, output_path)
        self.last_container_payload = report
        if report.get("issues"):
            raise OutputIntegrityError(
                "container payload mismatch: " + "; ".join(report["issues"]),
                {"container_payload": report},
            )
        return report

    def _mark_container_payload_failed(self, reason: str) -> None:
        report = dict(self.last_container_payload or {})
        issues = list(report.get("issues") or [])
        if reason not in issues:
            issues.append(reason)
        warnings = list(report.get("warnings") or [])
        if reason not in warnings:
            warnings.append(reason)
        report.update({
            "schema": report.get("schema", "vsr.container_payload.v1"),
            "status": "failed",
            "issues": issues,
            "warnings": warnings,
        })
        self.last_container_payload = report

    def _promote_post_restore_result(
        self,
        produced: str,
        output_path: str,
        temp_dir: str,
        label: str,
    ) -> bool:
        """Normalize an adapter result before it can replace the final output."""
        contract = getattr(self, "_output_contract", None)
        if contract is None:
            _promote_temp_output(produced, output_path)
            return True
        if _path_key(produced) == _path_key(output_path):
            return False
        previous_payload = dict(self.last_container_payload)
        normalized = contract.temp_path(temp_dir, f"{label}-contract")
        ok = False
        issues: list[str] = []
        try:
            self._remux_transformed_video(produced, output_path, normalized)
            ok, issues = contract.validate(normalized)
        except Exception as exc:
            logger.warning(
                "%s output-contract normalization failed: %s", label, exc
            )
            issues = [str(exc)]
        if not ok:
            self.last_container_payload = previous_payload
            logger.warning(
                "%s result was not promoted because it violates the output "
                "contract: %s",
                label,
                "; ".join(issues),
            )
            _cleanup_temp_output(normalized)
            _cleanup_temp_output(produced)
            return False
        _promote_temp_output(normalized, output_path)
        _cleanup_temp_output(produced)
        return True

    def _run_post_restore_passes(self, output_path: str, temp_dir: str) -> None:
        """RM-78 / RM-80: run optional post-restore passes against the
        finalised output in place. Each adapter is a no-op when its
        dep is missing; the original output is preserved on every
        failure path so users always have a result.
        """
        contract = getattr(self, "_output_contract", None)

        def post_path(stem: str) -> str:
            if contract is not None:
                return contract.temp_path(temp_dir, stem)
            return os.path.join(temp_dir, f"{stem}{Path(output_path).suffix or '.mp4'}")

        if self.config.upscale_factor in (2, 3, 4):
            try:
                from backend.post_restore import realesrgan_upscale
                upscaled = post_path("upscaled")
                produced = realesrgan_upscale(
                    output_path, upscaled,
                    scale=int(self.config.upscale_factor),
                )
                if produced and Path(produced).is_file():
                    if self._promote_post_restore_result(
                        produced, output_path, temp_dir, "realesrgan"
                    ):
                        logger.info(
                            f"Real-ESRGAN x{self.config.upscale_factor} pass complete"
                        )
            except Exception as exc:
                logger.warning(f"Real-ESRGAN pass failed: {exc}", exc_info=True)
        if self.config.swinir_restore:
            try:
                from backend.post_restore import swinir_restore
                restored = post_path("swinir")
                produced = swinir_restore(output_path, restored)
                if produced and Path(produced).is_file():
                    if self._promote_post_restore_result(
                        produced, output_path, temp_dir, "swinir"
                    ):
                        logger.info("SwinIR restoration pass complete")
            except Exception as exc:
                logger.warning(f"SwinIR pass failed: {exc}", exc_info=True)
        if self.config.seedvr2_restore:
            try:
                from backend.post_restore import seedvr2_restore
                restored = post_path("seedvr2")
                produced = seedvr2_restore(output_path, restored)
                if produced and Path(produced).is_file():
                    if self._promote_post_restore_result(
                        produced, output_path, temp_dir, "seedvr2"
                    ):
                        logger.info("SeedVR2 restoration pass complete")
            except Exception as exc:
                logger.warning(f"SeedVR2 pass failed: {exc}", exc_info=True)
        if self.config.film_grain_strength > 0.0:
            if self._uses_native_av1_film_grain():
                logger.info(
                    "SVT-AV1 native film grain was enabled during encode; "
                    "skipping additive post-encode grain pass."
                )
            else:
                try:
                    from backend.post_restore import add_film_grain
                    grain_out = post_path("grainy")
                    produced = add_film_grain(
                        output_path, grain_out,
                        strength=self.config.film_grain_strength,
                        video_encode_args=self._get_encode_args(allow_d3d12=False),
                        preserve_audio=self.config.preserve_audio,
                    )
                    if produced and Path(produced).is_file():
                        if self._promote_post_restore_result(
                            produced, output_path, temp_dir, "film-grain"
                        ):
                            logger.info(
                                f"Film-grain pass complete "
                                f"(strength={self.config.film_grain_strength:.3f})"
                            )
                except Exception as exc:
                    logger.warning(f"Film-grain pass failed: {exc}", exc_info=True)
        translation_path = str(getattr(self, "_translation_burn_path", "") or "")
        subtitle_path = translation_path or self.config.restyle_subtitle
        if subtitle_path:
            translation_requested = bool(translation_path)
            try:
                from backend.post_restore import burn_subtitles
                restyle_out = post_path("restyled")
                produced = burn_subtitles(
                    output_path, restyle_out,
                    subtitle_path=subtitle_path,
                    style_override=(
                        self.config.translation_style
                        if translation_requested else self.config.restyle_style
                    ),
                    video_encode_args=self._get_encode_args(allow_d3d12=False),
                    preserve_audio=self.config.preserve_audio,
                )
                promoted = False
                if produced and Path(produced).is_file():
                    promoted = self._promote_post_restore_result(
                        produced,
                        output_path,
                        temp_dir,
                        "translation" if translation_requested else "restyle",
                    )
                    if promoted:
                        logger.info(
                            "%s subtitle burn pass complete",
                            "Translated" if translation_requested else "Restyle",
                        )
                if translation_requested and not promoted:
                    raise RuntimeError(
                        "translated subtitle re-embedding produced no valid output")
                if translation_requested:
                    self.last_translation["status"] = "embedded"
            except Exception as exc:
                if translation_requested:
                    self.last_translation["status"] = "failed"
                    self.last_translation["error"] = str(exc)
                    raise
                logger.warning(f"Restyle pass failed: {exc}", exc_info=True)
        if self.config.watermark_image:
            try:
                from backend.post_restore import burn_watermark
                wm_out = post_path("watermarked")
                produced = burn_watermark(
                    output_path, wm_out,
                    watermark_path=self.config.watermark_image,
                    position=self.config.watermark_position,
                    opacity=self.config.watermark_opacity,
                    margin=self.config.watermark_margin,
                    video_encode_args=self._get_encode_args(allow_d3d12=False),
                    preserve_audio=self.config.preserve_audio,
                )
                if produced and Path(produced).is_file():
                    if self._promote_post_restore_result(
                        produced, output_path, temp_dir, "watermark"
                    ):
                        logger.info("Watermark burn pass complete")
            except Exception as exc:
                logger.warning(f"Watermark burn failed: {exc}", exc_info=True)

    def _uses_native_av1_film_grain(self) -> bool:
        return bool(
            self.config.film_grain_strength > 0.0
            and self._effective_output_codec() == "av1"
            and not (self._hw_encoder and self.config.use_hw_encode)
        )

    def _svt_av1_film_grain_args(self) -> List[str]:
        if not self._uses_native_av1_film_grain():
            return []
        grain = max(
            1,
            min(50, int(round(float(self.config.film_grain_strength) * 255))),
        )
        return ["-svtav1-params", f"film-grain={grain}"]

    def _get_encode_args(self, *, allow_d3d12: bool = True) -> List[str]:
        """Return FFmpeg video encoder arguments, preferring hardware encoding."""
        codec = self._effective_output_codec()
        hdr_mode = self._source_is_hdr()
        static_hdr = bool(
            hdr_mode
            and getattr(self._color_metadata, "mastering_display", "")
        )
        encoder_prefix = {
            "h264": "h264_",
            "h265": "hevc_",
            "av1": "av1_",
            "vvc": "vvc_",
        }.get(codec, "")
        hardware_matches = bool(
            self._hw_encoder
            and encoder_prefix
            and self._hw_encoder.startswith(encoder_prefix)
            and (allow_d3d12 or not self._hw_encoder.endswith("_d3d12va"))
        )
        if static_hdr and self._hw_encoder and self.config.use_hw_encode:
            if not getattr(self, "_hdr_software_warning_logged", False):
                logger.info(
                    "Static HDR mastering metadata requires the software "
                    "encoder so the final bitstream can reproduce its SEI."
                )
                self._hdr_software_warning_logged = True
        if hardware_matches and self.config.use_hw_encode and not static_hdr:
            if self._hw_encoder.endswith("_d3d12va"):
                base = [
                    "-vf", "format=nv12,hwupload,scale_d3d12=w=iw:h=ih",
                    "-c:v", self._hw_encoder,
                    "-bf", "0", "-async_depth", "1",
                    "-rc_mode", "CQP", "-qp", str(self.config.output_quality),
                ]
            elif 'nvenc' in self._hw_encoder:
                base = ['-c:v', self._hw_encoder, '-preset', 'p4',
                        '-cq', str(self.config.output_quality)]
            elif 'qsv' in self._hw_encoder:
                base = ['-c:v', self._hw_encoder,
                        '-global_quality', str(self.config.output_quality)]
            elif 'amf' in self._hw_encoder:
                base = ['-c:v', self._hw_encoder,
                        '-quality', 'balanced',
                        '-rc', 'cqp', '-qp', str(self.config.output_quality)]
            else:
                base = ['-c:v', 'libx264', '-crf', str(self.config.output_quality),
                        '-preset', 'medium']
            return (
                base
                + self._hdr_pixel_format_args(codec, hardware=True)
                + self._hdr_encode_args()
            )
        # F-8: software fallback honours the chosen output codec.
        if codec == "h265":
            base = ['-c:v', 'libx265', '-crf', str(self.config.output_quality),
                    '-preset', 'medium']
        elif codec == "av1":
            # SVT-AV1's CRF range tops out at 63; clamp our [0-51] scale
            # into the encoder's valid window. -preset 8 is the
            # speed/quality midpoint for libsvtav1.
            crf = min(63, self.config.output_quality)
            base = ['-c:v', 'libsvtav1', '-crf', str(crf), '-preset', '8']
        elif codec == "vvc":
            base = ['-c:v', 'libvvenc', '-qp', str(self.config.output_quality),
                    '-preset', 'medium']
        else:
            base = ['-c:v', 'libx264', '-crf', str(self.config.output_quality),
                    '-preset', 'medium']
        return (
            base
            + self._hdr_pixel_format_args(codec)
            + self._encoder_private_args(codec)
            + self._hdr_encode_args()
        )

    def _source_is_hdr(self) -> bool:
        meta = getattr(self, "_color_metadata", None)
        return bool(
            getattr(self.config, "preserve_color_metadata", True)
            and meta is not None
            and getattr(meta, "is_hdr", False)
        )

    def _effective_output_codec(self) -> str:
        contract = getattr(self, "_output_contract", None)
        if contract is not None:
            return contract.codec
        requested = getattr(self.config, "output_codec", "h264")
        if not getattr(self.config, "preserve_color_metadata", True):
            return requested
        try:
            from backend.hdr import hdr_safe_codec
            codec = hdr_safe_codec(requested, getattr(self, "_color_metadata", None))
        except Exception:
            logger.warning("HDR codec policy failed", exc_info=True)
            return requested
        if codec != requested and not getattr(self, "_hdr_codec_warning_logged", False):
            logger.info(
                f"HDR output cannot use {requested}; promoting final encode to {codec}."
            )
            self._hdr_codec_warning_logged = True
        return codec

    def _hdr_pixel_format_args(self, codec: str, *, hardware: bool = False) -> List[str]:
        if not self.config.preserve_color_metadata or self._color_metadata is None:
            return []
        try:
            from backend.hdr import hdr_pixel_format_args
            return hdr_pixel_format_args(
                self._color_metadata,
                codec,
                hardware=hardware,
            )
        except Exception:
            logger.warning("HDR pixel-format argument generation failed", exc_info=True)
            return []

    def _hdr_encoder_private_args(self, codec: str) -> List[str]:
        if not self.config.preserve_color_metadata or self._color_metadata is None:
            return []
        try:
            from backend.hdr import hdr_encoder_private_args
            return hdr_encoder_private_args(self._color_metadata, codec)
        except Exception:
            logger.warning("HDR encoder-private argument generation failed", exc_info=True)
            return []

    def _encoder_private_args(self, codec: str) -> List[str]:
        hdr_args = self._hdr_encoder_private_args(codec)
        grain_args = self._svt_av1_film_grain_args() if codec == "av1" else []
        if not hdr_args:
            return grain_args
        if not grain_args:
            return hdr_args
        if hdr_args[0] == grain_args[0] == "-svtav1-params":
            return ["-svtav1-params", f"{hdr_args[1]}:{grain_args[1]}"]
        return hdr_args + grain_args

    def _hdr_encode_args(self) -> List[str]:
        """RM-73 partial: re-tag the output with source color signalling."""
        if not self.config.preserve_color_metadata or self._color_metadata is None:
            return []
        try:
            from backend.hdr import hdr_encode_args
            return hdr_encode_args(self._color_metadata)
        except Exception:
            logger.warning("HDR encode argument generation failed", exc_info=True)
            return []

    def _check_encode_disk_space(self, output_path: str, *, width: int,
                                 height: int, frames: int,
                                 high_bit: bool,
                                 checkpoint_dir: Optional[Path] = None) -> None:
        """Estimate and probe every volume affected by this processing run."""
        if frames <= 0 or width <= 0 or height <= 0:
            return
        bytes_per_pixel = 6 if high_bit else 3
        raw = int(width) * int(height) * bytes_per_pixel * int(frames)
        work_resolution = self._resolve_work_directory()
        # Checkpoint frame sequences replace the FFV1 intermediate; without
        # checkpointing, FFV1 is the dominant work-volume consumer.
        work_bytes = int(raw * (0.10 if checkpoint_dir else 0.45))
        output_bytes = int(raw * 0.05)
        requirements = [
            StorageRequirement(
                work_resolution.path,
                work_bytes + 256 * 1024 * 1024,
                "temporary processing files",
            ),
            StorageRequirement(
                Path(output_path).parent,
                output_bytes + 64 * 1024 * 1024,
                "final output",
            ),
        ]
        if checkpoint_dir is not None:
            requirements.append(StorageRequirement(
                Path(checkpoint_dir),
                int(raw * 0.35) + 64 * 1024 * 1024,
                "checkpoint and resume frames",
            ))
        try:
            statuses = assess_storage_volumes(requirements)
        except OSError:
            return  # cannot probe; do not block processing
        for status in statuses:
            hard_floor = max(
                64 * 1024 * 1024,
                int(status.required_bytes * 0.25),
            )
            purposes = ", ".join(status.purposes)
            if status.free_bytes < hard_floor:
                raise ValueError(
                    f"Insufficient disk space at '{status.path}' for {purposes}: "
                    f"about {status.required_bytes // (1024*1024)} MB is "
                    f"estimated but only {status.free_bytes // (1024*1024)} MB "
                    "is free. Choose a work/output folder on a larger drive."
                )
            if status.free_bytes < status.required_bytes:
                logger.warning(
                    "Low disk space at '%s' for %s: ~%d MB estimated, %d MB "
                    "free. Processing will continue but may fail if the "
                    "estimate is high.",
                    status.path,
                    purposes,
                    status.required_bytes // (1024 * 1024),
                    status.free_bytes // (1024 * 1024),
                )

    def _promote_video_output(
        self,
        produced,
        output_path: str,
        *,
        reference=None,
        expected_frames: Optional[int] = None,
        expected_duration: Optional[float] = None,
    ) -> None:
        """Validate a finished video, then atomically replace the destination.

        Fails closed on truncation or a missing video stream: the existing
        destination is left untouched and ``OutputIntegrityError`` is raised so
        the caller can salvage a full-length fallback instead of shipping a
        short/broken file as success.
        """
        ok, reason, details = validate_video_output(
            produced,
            reference=reference,
            expected_frames=expected_frames,
            expected_duration=expected_duration,
        )
        if not ok:
            logger.error(
                "Output integrity check failed for '%s': %s", output_path, reason
            )
            raise OutputIntegrityError(reason, details)
        _promote_temp_output(produced, output_path)

    def _salvage_intermediate(self, source: str, output: str) -> str:
        """Copy the lossless intermediate to a container-correct path.

        The intermediate is an FFV1 (or cv2-fallback) MKV. Copying its
        bytes into the user's requested ``.mp4`` path produces a
        mislabeled, often unplayable file that is then reported as
        success. When the requested extension does not match, salvage
        next to the requested path with the real extension and warn.
        """
        src_ext = Path(source).suffix.lower()
        out_ext = Path(output).suffix.lower()
        if src_ext == out_ext:
            _copy_file_atomic(source, output)
            return output
        salvage = str(Path(output).with_suffix(src_ext))
        _copy_file_atomic(source, salvage)
        logger.warning(
            f"Encoding to '{output}' was not possible; saved the "
            f"unencoded intermediate as '{salvage}' instead, because "
            f"its stream lives in a '{src_ext}' container and renaming "
            f"it to '{out_ext}' would produce a broken file."
        )
        return salvage

    def _encode_frame_sequence(
        self,
        frame_dir: Path,
        fps: float,
        output: str,
        *,
        frame_durations: Optional[List[float]] = None,
        source_time_base: Optional[float] = None,
    ) -> str:
        """Encode checkpoint/output frames into the requested video path."""
        frame_dir = Path(frame_dir)
        pattern = frame_dir / "frame_%06d.png"
        if not (frame_dir / "frame_000000.png").is_file():
            raise ValueError(f"No checkpoint frames found in {frame_dir}")
        temp_output = self._allocate_work_output(output)
        concat_path: Optional[Path] = None
        try:
            _ensure_output_parent(output)
            frame_total = len(list(frame_dir.glob("frame_*.png")))
            use_vfr = bool(
                frame_durations
                and len(frame_durations) >= frame_total
                and frame_total > 0
            )
            if use_vfr:
                normalized_durations = []
                fallback = 1.0 / max(float(fps), 1.0)
                for value in frame_durations[:frame_total]:
                    try:
                        duration = float(value)
                    except (TypeError, ValueError):
                        duration = fallback
                    if not np.isfinite(duration) or duration <= 0:
                        duration = fallback
                    normalized_durations.append(duration)
                concat_path = frame_dir / ".vsr-vfr.ffconcat"
                concat_lines = ["ffconcat version 1.0"]
                try:
                    timing_tick = float(source_time_base or 0.0)
                except (TypeError, ValueError):
                    timing_tick = 0.0
                if not np.isfinite(timing_tick) or timing_tick <= 0:
                    timing_tick = min(normalized_durations) / 100.0
                concat_rate = max(
                    1000,
                    min(1_000_000, int(round(1.0 / max(timing_tick, 1e-6)))),
                )
                for index, duration in enumerate(normalized_durations):
                    concat_lines.append(f"file frame_{index:06d}.png")
                    concat_lines.append(f"option framerate {concat_rate}")
                    concat_lines.append(f"duration {duration:.9f}")
                _write_text_atomic(
                    concat_path, "\n".join(concat_lines) + "\n")
                cmd = [
                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-nostats',
                ] + self._d3d12_device_args() + [
                    '-f', 'concat', '-safe', '0',
                    '-i', str(concat_path),
                ]
            else:
                normalized_durations = []
                cmd = [
                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-nostats',
                ] + self._d3d12_device_args() + [
                    '-framerate', f"{float(fps):.6f}",
                    '-i', str(pattern),
                ]
            cmd += self._get_encode_args()
            if use_vfr:
                cmd += [
                    '-fps_mode:v', 'vfr',
                    '-enc_time_base:v', 'demux',
                ]
            cmd += ['-an', str(temp_output)]
            timeout = _ffmpeg_subprocess_timeout(
                max(
                    1.0,
                    sum(normalized_durations)
                    if normalized_durations
                    else _frame_seconds(frame_total, fps),
                )
            )
            self._run_checked_ffmpeg(cmd, timeout)
            self._promote_video_output(
                temp_output,
                output,
                expected_frames=frame_total,
                expected_duration=(
                    sum(normalized_durations)
                    if normalized_durations else None
                ),
            )
            return output
        except FileNotFoundError:
            logger.warning(
                "FFmpeg not found; leaving processed checkpoint frames as output"
            )
            return str(frame_dir)
        except (subprocess.CalledProcessError, OutputIntegrityError) as exc:
            if self._fallback_after_hw_failure(exc):
                logger.warning(
                    "Hardware encoder failed, retrying with %s: %s",
                    self._hw_encoder or "software",
                    exc,
                )
                return self._encode_frame_sequence(
                    frame_dir,
                    fps,
                    output,
                    frame_durations=frame_durations,
                    source_time_base=source_time_base,
                )
            raise
        finally:
            _cleanup_temp_output(temp_output)
            _cleanup_temp_output(concat_path)

    def _reencode_or_copy(self, source: str, output: str) -> str:
        """Re-encode with preferred encoder, or salvage the intermediate
        if FFmpeg is unavailable or keeps failing."""
        temp_output = self._allocate_work_output(output)
        try:
            _ensure_output_parent(output)
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-nostats',
            ] + self._d3d12_device_args() + ['-i', source]
            cmd += self._get_encode_args()
            cmd += ['-an', str(temp_output)]
            timeout = _ffmpeg_subprocess_timeout(_probe_duration_seconds(source))
            self._run_checked_ffmpeg(cmd, timeout)
            self._promote_video_output(temp_output, output, reference=source)
            return output
        except OutputIntegrityError as exc:
            if self._fallback_after_hw_failure(exc):
                logger.warning(
                    "Hardware encoder output failed validation, retrying with %s: %s",
                    self._hw_encoder or "software",
                    exc.reason,
                )
                return self._reencode_or_copy(source, output)
            logger.warning(
                "Re-encode failed integrity check (%s); salvaging intermediate.",
                exc.reason,
            )
            return self._salvage_intermediate(source, output)
        except subprocess.CalledProcessError as e:
            if self._fallback_after_hw_failure(e):
                logger.warning(
                    "Hardware encoder failed, retrying with %s: %s",
                    self._hw_encoder or "software",
                    e,
                )
                return self._reencode_or_copy(source, output)
            return self._salvage_intermediate(source, output)
        except Exception as exc:
            logger.warning(
                f"FFmpeg re-encode failed; salvaging intermediate: {exc}",
                exc_info=True,
            )
            return self._salvage_intermediate(source, output)
        finally:
            _cleanup_temp_output(temp_output)

    def _merge_audio(
        self,
        original: str,
        processed: str,
        output: str,
        *,
        start_seconds: Optional[float] = None,
        end_seconds: Optional[float] = None,
        _include_auxiliary: bool = True,
        _force_audio_transcode: bool = False,
    ) -> str:
        temp_output = self._allocate_work_output(output)
        plan: dict = {}
        try:
            _ensure_output_parent(output)
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-nostats',
            ] + self._d3d12_device_args() + [
                '-i', processed,
            ]
            # Seek audio to the first selected source PTS. For VFR input this
            # can differ from frame_index / average_fps.
            audio_start = (
                max(0.0, float(start_seconds))
                if start_seconds is not None else
                max(0.0, float(self.config.time_start or 0.0))
            )
            audio_end = (
                max(audio_start, float(end_seconds))
                if end_seconds is not None else
                max(0.0, float(self.config.time_end or 0.0))
            )
            if audio_start > 0:
                cmd += ['-ss', f'{audio_start:.9f}']
            cmd += ['-i', original]
            manifest = probe_container_manifest(original)
            target = self.config.loudnorm_target
            plan = build_container_mux_plan(
                manifest,
                output,
                preserve_audio=self.config.preserve_audio,
                multi_audio=self.config.multi_audio_passthrough,
                loudnorm_target=target,
                include_auxiliary=_include_auxiliary,
                force_audio_transcode=_force_audio_transcode,
                start_seconds=audio_start,
                end_seconds=audio_end,
            )
            self.last_container_payload = plan
            for warning in plan.get("warnings", []):
                logger.warning(warning)
            cmd += ['-map', '0:v:0']
            cmd += self._get_encode_args()
            cmd += build_container_mux_args(
                plan,
                input_index=1,
                loudnorm_target=target,
            )
            output_duration = (
                audio_end - audio_start
                if audio_end > audio_start
                else _probe_duration_seconds(processed)
            )
            if output_duration > 0:
                cmd += ['-t', f'{output_duration:.9f}']
            else:
                cmd += ['-shortest']
            cmd += [str(temp_output)]
            # Adaptive timeout: scales with the duration of the original
            # input so multi-hour videos do not silently lose audio when the
            # mux pass takes longer than the legacy 10-minute fixed budget.
            timeout = _ffmpeg_subprocess_timeout(_probe_duration_seconds(original))
            self._run_checked_ffmpeg(cmd, timeout)
            report = validate_container_payload(plan, temp_output)
            self.last_container_payload = report
            if report.get("issues"):
                raise OutputIntegrityError(
                    "container payload mismatch: " + "; ".join(report["issues"]),
                    {"container_payload": report},
                )
            # Guard against -shortest truncating the video to a shorter audio
            # stream (or any decode-integrity loss). The processed intermediate
            # is the length-of-record; on failure keep the full-length video
            # without audio rather than promote a truncated mux.
            try:
                self._promote_video_output(
                    temp_output, output, reference=processed
                )
            except OutputIntegrityError as exc:
                if self._fallback_after_hw_failure(exc):
                    logger.warning(
                        "Hardware encoder output failed validation, retrying "
                        "container merge with %s: %s",
                        self._hw_encoder or "software",
                        exc.reason,
                    )
                    return self._merge_audio(
                        original,
                        processed,
                        output,
                        start_seconds=start_seconds,
                        end_seconds=end_seconds,
                        _include_auxiliary=_include_auxiliary,
                        _force_audio_transcode=_force_audio_transcode,
                    )
                logger.warning(
                    "Audio merge produced a truncated/invalid output (%s); "
                    "saving the full-length video without audio instead.",
                    exc.reason,
                )
                self._mark_container_payload_failed(
                    "Container payload was not promoted because the muxed video "
                    f"failed integrity validation: {exc.reason}"
                )
                return self._salvage_intermediate(processed, output)
            encoder_name = (
                cmd[cmd.index('-c:v') + 1]
                if '-c:v' in cmd else 'unknown'
            )
            logger.info(
                f"Container payload merged successfully (encoder: {encoder_name})"
            )
            return output
        except OutputIntegrityError as exc:
            payload_failure = "container_payload" in exc.details
            mapped_auxiliary = any(
                item.get("type") in {"subtitle", "attachment", "data", "video"}
                and item.get("action") in {"copy", "transcode"}
                for item in plan.get("streams", [])
            )
            copied_audio = any(
                item.get("type") == "audio" and item.get("action") == "copy"
                for item in plan.get("streams", [])
            )
            if payload_failure and _include_auxiliary and mapped_auxiliary:
                logger.warning(
                    "Full container preservation failed (%s); retrying with "
                    "audio and metadata only.", exc.reason,
                )
                return self._merge_audio(
                    original,
                    processed,
                    output,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    _include_auxiliary=False,
                    _force_audio_transcode=_force_audio_transcode,
                )
            if payload_failure and copied_audio and not _force_audio_transcode:
                logger.warning(
                    "Audio stream copy failed validation (%s); retrying with "
                    "container-compatible audio encoding.", exc.reason,
                )
                return self._merge_audio(
                    original,
                    processed,
                    output,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    _include_auxiliary=False,
                    _force_audio_transcode=True,
                )
            logger.warning(
                "Container preservation failed integrity checks (%s); "
                "saving the processed video without source payload.",
                exc.reason,
            )
            self._mark_container_payload_failed(
                f"Source container payload could not be preserved: {exc.reason}"
            )
            return self._salvage_intermediate(processed, output)
        except subprocess.TimeoutExpired:
            # Do not re-run ffmpeg after a duration-adaptive timeout --
            # salvage the intermediate into a container-correct path.
            logger.warning("FFmpeg audio merge timed out, saving video without audio")
            self._mark_container_payload_failed(
                "Source container payload was omitted after the mux timed out."
            )
            return self._salvage_intermediate(processed, output)
        except subprocess.CalledProcessError as e:
            # If hardware encoder failed, retry with software
            if self._hw_encoder and self._hw_encoder in cmd:
                self._fallback_after_hw_failure(e)
                logger.warning(
                    "Hardware encoder failed, retrying with %s: %s",
                    self._hw_encoder or "software",
                    e,
                )
                return self._merge_audio(
                    original,
                    processed,
                    output,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    _include_auxiliary=_include_auxiliary,
                    _force_audio_transcode=_force_audio_transcode,
                )
            mapped_auxiliary = any(
                item.get("type") in {"subtitle", "attachment", "data", "video"}
                and item.get("action") in {"copy", "transcode"}
                for item in plan.get("streams", [])
            )
            copied_audio = any(
                item.get("type") == "audio" and item.get("action") == "copy"
                for item in plan.get("streams", [])
            )
            if _include_auxiliary and mapped_auxiliary:
                logger.warning(
                    "Auxiliary stream mux failed; retrying with audio and metadata only."
                )
                return self._merge_audio(
                    original,
                    processed,
                    output,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    _include_auxiliary=False,
                    _force_audio_transcode=_force_audio_transcode,
                )
            if copied_audio and not _force_audio_transcode:
                logger.warning(
                    "Audio stream copy failed; retrying with a compatible audio codec."
                )
                return self._merge_audio(
                    original,
                    processed,
                    output,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    _include_auxiliary=False,
                    _force_audio_transcode=True,
                )
            # The merge often fails on the AUDIO side (bad stream, odd
            # layout); a video-only encode to the requested container is
            # usually still possible and beats a mislabeled raw copy.
            logger.warning(f"Audio merge failed: {e}, encoding video without audio")
            self._mark_container_payload_failed(
                "Source container payload was omitted after the mux command failed."
            )
            return self._reencode_or_copy(processed, output)
        except FileNotFoundError:
            logger.warning("FFmpeg not found, saving video without audio")
            self._mark_container_payload_failed(
                "Source container payload was omitted because FFmpeg is unavailable."
            )
            return self._salvage_intermediate(processed, output)
        finally:
            _cleanup_temp_output(temp_output)


if __name__ == "__main__":
    # `python -m backend.processor` executes this file as `__main__`;
    # cli.main() then imports backend.processor, which would otherwise
    # re-execute the whole module and create a second, distinct set of
    # classes (two InpaintMode types, double registry registration).
    # Alias the module first so that import resolves to this instance.
    sys.modules.setdefault("backend.processor", sys.modules[__name__])
    from backend.cli import main as _cli_main
    _cli_main()
