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
from fractions import Fraction
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, List, Callable

logger = logging.getLogger(__name__)

# RFP-L-1 re-exports. Anything that used to be defined in this module
# but moved during the split is re-imported here so existing callers
# (`from backend.processor import _open_capture`) keep working.
from backend.io import (
    MediaInputError,
    MediaWriteError,
    SubtitleStreamInfo as SubtitleStreamInfo,
    _validate_video_input_file,
    _video_capture_open_error,
    _invalid_video_dimensions_error,
    _video_decode_error,
    _probe_codec_for_log as _probe_codec_for_log,
    _probe_audio_stream_count as _probe_audio_stream_count,
    _probe_subtitle_streams as _probe_subtitle_streams,
    _probe_duration_seconds as _probe_duration_seconds,
    _ffmpeg_subprocess_timeout as _ffmpeg_subprocess_timeout,
    _probe_keyframe_indices,
    _probe_is_interlaced,
    _deinterlace_to_temp,
    _ensure_output_parent,
    _path_key as _path_key,
    _choose_available_output_path as _choose_available_output_path,
    _write_text_atomic,
    _allocate_temp_output_path,
    _cleanup_temp_output,
    _promote_temp_output,
    _copy_file_atomic,
    validate_video_output as validate_video_output,
    VideoFrameTiming,
    _normalize_time_base,
    _seconds_to_ticks,
    _ticks_to_seconds,
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
from backend.execution_provenance import (
    FAILURE_OUTPUT_INVALID,
    FAILURE_RUNTIME,
    ExecutionProvenance,
    RequestedStageError,
    StageProvenance,
    device_from_provider,
    normalize_device,
)
from backend.device_provider import DeviceProvider, RuntimeDeviceProvider
from backend.container_payload import (
    build_container_mux_args as build_container_mux_args,
    build_container_mux_plan as build_container_mux_plan,
    probe_container_manifest as probe_container_manifest,
    validate_container_payload as validate_container_payload,
)
from backend.quality import (
    _ssim as _ssim,
    compute_vmaf as compute_vmaf,
    compute_extended_metrics as compute_extended_metrics,
    temporal_consistency_score as temporal_consistency_score,
    residual_text_score as residual_text_score,
    temporal_flicker_score as temporal_flicker_score,
    mask_boundary_seam_score as mask_boundary_seam_score,
)
from backend.quality_gate import (
    RESIDUAL_TEXT_SCORE_CEILING as RESIDUAL_TEXT_SCORE_CEILING,
    TEMPORAL_FLICKER_CEILING as TEMPORAL_FLICKER_CEILING,
    evaluate_quality_gate as evaluate_quality_gate,
)
from backend.mask_corrections import (
    SELECTIVE_RERUN_SCHEMA,
    apply_mask_corrections,
    frame_is_in_ranges,
    make_review_span as make_review_span,
    merge_frame_ranges,
    merge_review_spans as merge_review_spans,
    has_timed_corrections,
)
from backend.frozen_matte import (
    FrozenMatteError,
    normalize_frozen_matte,
    validate_frozen_matte,
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
from backend.detection_geometry import (
    DetectionGeometry,
    as_detection_geometry,
    geometry_mask,
)
from backend.region_keyframes import region_shapes_at as region_shapes_at
from backend.work_directory import (
    StorageRequirement as StorageRequirement,
    assess_storage_volumes as assess_storage_volumes,
    make_work_temp_dir,
    resolve_work_directory,
)
from backend.tracking import (
    _KalmanBox as _KalmanBox,
    _box_from_state as _box_from_state,
    _iou as _iou,
    SubtitleTracker,
    _group_horizontal_geometry,
    _group_horizontal_line,
    _phash,
    _phash_distance,
    apply_clean_reference as apply_clean_reference,
)
from backend.detection import SubtitleDetector, _surya_allowed as _surya_allowed
from backend.hdr import (
    hdr_proxy_from_high_bit,
    hdr_proxy_from_linear,
    hdr_proxy_to_linear,
    hdr_repair_block_reason,
    hdr_signal_to_linear,
    linear_to_hdr_signal,
)
from backend.inpainters import (
    BaseInpainter,
    STTNInpainter,
    LAMAInpainter,
    ProPainterInpainter,
    AutoInpainter,
    is_oom_error,
    free_inference_memory,
    _feather_blend as _feather_blend,
    _edge_ring_color_correct as _edge_ring_color_correct,
    _expand_mask_by_color,
    _detect_scene_cuts,
    _detect_scene_cuts_pyscenedetect as _detect_scene_cuts_pyscenedetect,
    extend_masks_across_fades,
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
    try:
        positioned = bool(cap.set(cv2.CAP_PROP_POS_FRAMES, target))
    except Exception as exc:
        positioned = False
        set_error = exc
    else:
        set_error = None
    if target > 0 and not positioned:
        raise MediaInputError(
            f"The decoder could not seek to frame {target}; retry from the "
            "beginning or use a different decoder.",
            reason="decoder_seek_failed",
            path=str(getattr(cap, "_path", "")),
            detail=(str(set_error) if set_error is not None
                    else "decoder rejected frame seek"),
        )
    if target == 0:
        return 0
    try:
        pos = int(round(float(cap.get(cv2.CAP_PROP_POS_FRAMES))))
    except Exception:
        pos = target
    if pos > target:
        # Overshot; restart and scan forward from the beginning.
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
            raise MediaInputError(
                "The decoder could not restart while seeking the requested frame.",
                reason="decoder_seek_failed",
                path=str(getattr(cap, "_path", "")),
                detail="decoder rejected frame-zero seek",
            )
        pos = 0
    while pos < target:
        if not cap.grab():
            raise MediaInputError(
                f"The decoder stopped before frame {target} while seeking.",
                reason="decoder_seek_failed",
                path=str(getattr(cap, "_path", "")),
                detail=f"reached frame {pos}",
            )
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
    selected_frame_duration_ticks: Optional[List[int]]
    processed_time_start_ticks: int
    processed_time_end_ticks: int
    matte_timestamp_ticks: List[int]
    matte_duration_ticks: List[int]
    matte_time_base_num: int
    matte_time_base_den: int


@dataclass(frozen=True)
class _FrameLoopCheckpoint:
    active: bool
    root: Optional[Path]
    key: Optional[str]
    state_path: Optional[Path]
    config_hash: str
    frame_dir: Optional[Path]
    timing_manifest_path: Optional[Path]
    timing_metadata: Optional[dict]
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
    timed_mask_corrections: bool
    static_fixed_shapes: Any
    selective_ranges: List[Tuple[int, int]]
    reader: Any
    selective_cap: Any
    matte_reader: Any
    frozen_matte: bool
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
    srt_tracker: Optional[SubtitleTracker] = None
    # RM-292: (mask, frames_remaining) so a fade-out hold survives a batch
    # boundary instead of ending wherever the decode happened to split.
    fade_carry: Any = None
    # RM-296: frames decoded but deliberately not yet processed, held back so
    # they can see the next batch's masks. frame_idx is the DECODE cursor;
    # written_idx is how many frames have actually reached the writer, which
    # is what the checkpoint's resume point and the preview index mean.
    fade_pending: Any = None
    written_idx: int = 0


@dataclass
class _FrameBatch:
    frames: List[np.ndarray] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)
    source_frames: List[Optional[np.ndarray]] = field(default_factory=list)
    passthrough_flags: List[bool] = field(default_factory=list)
    active_segments: List[Tuple[int, int]] = field(default_factory=list)

    _PARALLEL = ("frames", "masks", "source_frames", "passthrough_flags")

    def add(self, frame: np.ndarray, mask: np.ndarray,
            source_frame: Optional[np.ndarray], *, passthrough: bool) -> None:
        self.frames.append(frame)
        self.masks.append(mask)
        self.source_frames.append(source_frame)
        self.passthrough_flags.append(bool(passthrough))

    def split_tail(self, count: int) -> "_FrameBatch":
        """Remove the last `count` frames and return them as a new batch.

        RM-296: a held-back tail is how a fade-in hold reaches across a
        decode boundary. The frames leave this batch entirely, so nothing
        downstream can process them twice.
        """
        tail = _FrameBatch()
        if count <= 0:
            return tail
        for name in self._PARALLEL:
            values = getattr(self, name)
            setattr(tail, name, values[-count:])
            setattr(self, name, values[:-count])
        return tail

    def prepend(self, other: "_FrameBatch") -> None:
        """Put `other`'s frames in front of this batch's, in order."""
        if not other.frames:
            return
        for name in self._PARALLEL:
            setattr(self, name, getattr(other, name) + getattr(self, name))


def _frame_seconds(index: int, fps: float,
                   frame_timing: Optional[VideoFrameTiming] = None) -> float:
    """Return one frame index on the shared VFR/CFR processing clock."""
    if frame_timing is not None:
        return frame_timing.frame_time(index, fps)
    # A missing or bogus rate must not explode the clock. Dividing by an
    # epsilon turns frame 15 into 15 million seconds; a genuine sub-1 fps
    # timelapse still divides by its real rate.
    rate = float(fps)
    return float(index) / (rate if rate > 0.0 else 1.0)


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
                0, min(total_frames, int(time_start_s * fps)))
        if time_end_s > 0:
            end_frame = max(
                0, min(total_frames, int(time_end_s * fps)))
    if end_frame <= start_frame:
        raise ValueError(
            f"Invalid time range: end ({time_end_s}s) "
            f"must be after start ({time_start_s}s)")
    if start_frame > 0:
        _seek_capture_to_frame(cap, start_frame)
    frames_to_process = end_frame - start_frame
    selected_frame_durations = (
        frame_timing.range_durations(start_frame, end_frame, fps)
        if frame_timing is not None else None
    )
    if frame_timing is not None:
        timing_num, timing_den = _normalize_time_base(
            getattr(frame_timing, "time_base_num", 0),
            getattr(frame_timing, "time_base_den", 0),
            fallback_seconds=getattr(frame_timing, "time_base", 0.0),
        )
        range_tick_method = getattr(frame_timing, "range_duration_ticks", None)
        selected_frame_duration_ticks = (
            list(range_tick_method(start_frame, end_frame))
            if callable(range_tick_method) else [
                _seconds_to_ticks(value, timing_num, timing_den)
                for value in selected_frame_durations or []
            ]
        )
    else:
        timing_num, timing_den = 0, 1
        selected_frame_duration_ticks = None
    processed_time_start = _frame_seconds(
        start_frame, fps, frame_timing)
    processed_time_end = (
        processed_time_start + sum(selected_frame_durations or [])
        if selected_frame_durations is not None
        else _frame_seconds(end_frame, fps)
    )
    if frame_timing is not None:
        matte_time_base_num = timing_num
        matte_time_base_den = timing_den
        frame_tick_method = getattr(frame_timing, "frame_time_ticks", None)
        matte_timestamp_ticks = [
            frame_tick_method(index, fps)
            if callable(frame_tick_method) else _seconds_to_ticks(
                frame_timing.frame_time(index, fps),
                matte_time_base_num,
                matte_time_base_den,
            )
            for index in range(start_frame, end_frame)
        ]
        matte_duration_ticks = list(selected_frame_duration_ticks or [])
        processed_time_start_ticks = (
            frame_tick_method(start_frame, fps)
            if callable(frame_tick_method) else _seconds_to_ticks(
                frame_timing.frame_time(start_frame, fps),
                matte_time_base_num,
                matte_time_base_den,
            )
        )
        processed_time_end_ticks = (
            processed_time_start_ticks + sum(matte_duration_ticks)
        )
    else:
        try:
            rate = Fraction(str(float(fps)))
        except (TypeError, ValueError, ZeroDivisionError):
            rate = Fraction(1, 1)
        if rate <= 0:
            rate = Fraction(1, 1)
        matte_time_base_num, matte_time_base_den = _normalize_time_base(
            rate.denominator, rate.numerator,
            fallback_seconds=1.0,
        )
        matte_timestamp_ticks = list(range(start_frame, end_frame))
        matte_duration_ticks = [1] * frames_to_process
        processed_time_start_ticks = int(start_frame)
        processed_time_end_ticks = int(end_frame)
    matte_timestamps = [
        _ticks_to_seconds(
            value, matte_time_base_num, matte_time_base_den)
        for value in matte_timestamp_ticks
    ]
    matte_durations = [
        _ticks_to_seconds(
            value, matte_time_base_num, matte_time_base_den)
        for value in matte_duration_ticks
    ]
    matte_time_base = _ticks_to_seconds(
        1, matte_time_base_num, matte_time_base_den)
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
        selected_frame_duration_ticks=selected_frame_duration_ticks,
        processed_time_start_ticks=processed_time_start_ticks,
        processed_time_end_ticks=processed_time_end_ticks,
        matte_timestamp_ticks=matte_timestamp_ticks,
        matte_duration_ticks=matte_duration_ticks,
        matte_time_base_num=matte_time_base_num,
        matte_time_base_den=matte_time_base_den,
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


def _open_required_hdr_capture(path: str, *, input_fps: float):
    """Open the native high-bit HDR reader or fail before any 8-bit decode."""
    capture = _open_bgr48_capture(path, input_fps=input_fps)
    if capture is None:
        raise ValueError(
            "HDR high-bit decode unavailable; refusing an 8-bit fallback "
            "that would destroy the source surface."
        )
    return capture


from backend._encode_mixin import _EncodeMixin
from backend._quality_mixin import _QualityMixin
from backend._finalize_mixin import _FinalizeMixin
from backend._srt_mixin import _SrtMixin
from backend._clean_ref_mixin import _CleanRefMixin


class SubtitleRemover(
    _EncodeMixin, _QualityMixin, _FinalizeMixin, _SrtMixin, _CleanRefMixin,
):
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
        # RM-147: keep the raw request so reports can show requested vs.
        # effective instead of only the resolved value.
        self._requested_device = requested_device
        self._device_fallback_reason = ""
        if self.config.device != requested_device:
            self._device_fallback_reason = (
                f"{requested_device} is unavailable; using {self.config.device}"
            )
            logger.warning(
                "Inference device fallback: %s -> %s",
                requested_device,
                self.config.device,
            )
        self.detector = SubtitleDetector(
            self.config.device,
            lang=self.config.detection_lang,
            vertical=self.config.detection_vertical,
            engine=self.config.detection_engine,
            rapidocr_variant=getattr(self.config, "rapidocr_variant", "v6"),
            paddleocr_variant=getattr(
                self.config, "paddleocr_variant", "mobile"),
        )
        if self.config.language_mask_filter and not any(
            name in self.detector._engine_name
            for name in ("RapidOCR", "PaddleOCR", "EasyOCR")
        ):
            logger.warning(
                "Selected-language mask filtering needs recognized OCR text; "
                "%s cannot classify boxes, so unmatched regions will be kept.",
                self.detector._engine_name,
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
        # Rich tracked OCR observations; legacy (frame_idx, text) tuples remain
        # accepted by the writer for integrations that populate this directly.
        self._srt_entries: List[Any] = []
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
        self.last_frozen_matte: dict = {
            "requested": bool(
                normalize_frozen_matte(getattr(self.config, "frozen_matte", None))
            ),
            "status": "not-requested",
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
        self._clean_reference_cache: dict = {}
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
        # RM-304: final-encode quality evidence keeps only the previous frame
        # in memory and sparse masks on the existing work temp directory.
        self._quality_temporal_previous = None
        self._quality_temporal_scores: List[float] = []
        self._quality_temporal_scene_cuts_excluded = 0
        self._quality_temporal_worst_pair = None
        self._quality_temporal_failure_logged = False
        self._quality_color_drift_sum = 0.0
        self._quality_color_drift_count = 0
        self._quality_color_drift_metric = None
        self._quality_color_drift_worst_frame = None
        self._quality_color_failure_logged = False
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

        self._refresh_execution_provenance()
        logger.info(f"Detector: {self.detector._engine_name} | "
                    f"Inpainter: {self.config.mode.value} | "
                    f"Device: {self.config.device}"
                    f"{' | HW encode: ' + self._hw_encoder if self._hw_encoder else ''}")
        provenance = self.execution_provenance.summary()
        if provenance:
            logger.info("Execution: %s", provenance)
        if self.execution_provenance.any_fallback:
            logger.warning(
                "Requested %s but part of the pipeline ran elsewhere: %s",
                self.execution_provenance.requested_device or "auto",
                provenance,
            )

    @property
    def execution_provenance(self) -> ExecutionProvenance:
        """RM-147: how this job actually executed.

        Lazily created so harnesses that build a remover through ``__new__``
        (reference corpus, synthetic A/B tests) still get a valid record.
        """
        value = self.__dict__.get("_execution_provenance")
        if value is None:
            value = ExecutionProvenance()
            self.__dict__["_execution_provenance"] = value
        return value

    @execution_provenance.setter
    def execution_provenance(self, value: ExecutionProvenance) -> None:
        self.__dict__["_execution_provenance"] = value

    def _refresh_execution_provenance(self) -> None:
        """Refresh initialized stages without erasing runtime observations."""
        previous = self.__dict__.get("_execution_provenance")
        provenance = ExecutionProvenance(
            requested_device=getattr(self, "_requested_device", "")
            or self.config.device,
            effective_device=self.config.device,
            device_fallback_reason=getattr(self, "_device_fallback_reason", ""),
            inpaint_mode=self.config.mode.value,
        )
        if previous is not None:
            provenance.frames_processed = previous.frames_processed
            provenance.processing_seconds = previous.processing_seconds
            for name, stage in previous.stages.items():
                provenance.set_stage(stage)
        detector = getattr(self, "detector", None)
        collect = getattr(detector, "execution_provenance", None)
        old_ocr = provenance.stage("ocr")
        if callable(collect) and (
            old_ocr is None
            or (not old_ocr.actual_executions and not old_ocr.failed)
        ):
            try:
                stage = collect()
                # The detector is built after the device downgrade, so report
                # the user's original request rather than the resolved value.
                stage.requested_device = provenance.requested_device
                provenance.set_stage(stage)
            except Exception:
                logger.warning("OCR provenance probe failed", exc_info=True)
        inpainter = getattr(self, "inpainter", None)
        if inpainter is not None:
            old_inpaint = provenance.stage("inpaint")
            if old_inpaint is None or (
                not old_inpaint.actual_executions and not old_inpaint.failed
            ):
                provenance.set_stage(StageProvenance(
                    stage="inpaint",
                    requested_device=provenance.requested_device,
                    effective_device=self.config.device,
                    engine=self.config.mode.value,
                    backend=type(inpainter).__name__,
                    requested_implementation=self.config.mode.value,
                    selection_policy=(
                        "auto" if self.config.mode.value == "auto" else "explicit"
                    ),
                    outcome="initialized",
                ))
        # Any stage that fell back must carry a reason; inherit the job-level
        # device fallback when the stage itself did not record one.
        for stage in provenance.stages.values():
            if stage.fell_back and not stage.fallback_reason:
                stage.fallback_reason = (
                    provenance.device_fallback_reason
                    or f"{stage.engine or stage.backend} ran on "
                        f"{normalize_device(stage.effective_device)}"
                )
        self.execution_provenance = provenance

    @staticmethod
    def _provider_effective_device(provider: str, fallback: str) -> str:
        text = str(provider or "").lower()
        if any(token in text for token in ("cv2", "opencv", "tbe")):
            return "cpu"
        mapped = device_from_provider(provider)
        if mapped != "unknown":
            return mapped
        normalized = normalize_device(fallback)
        return normalized if normalized != "unknown" else str(fallback or "unknown")

    @staticmethod
    def _inpainter_provider_name(inpainter: object) -> str:
        try:
            return str(
                getattr(inpainter, "backend_name", "")
                or type(inpainter).__name__
            )
        except Exception:
            return type(inpainter).__name__

    def _sync_ocr_provenance(self) -> None:
        detector = getattr(self, "detector", None)
        collect = getattr(detector, "execution_provenance", None)
        if not callable(collect):
            return
        stage = collect()
        stage.requested_device = self.execution_provenance.requested_device
        self.execution_provenance.set_stage(stage)

    def _sync_inpaint_provenance(self, frame_count: int) -> None:
        """Record the implementation observed after a successful inpaint call."""
        inpainter = getattr(self, "inpainter", None)
        if inpainter is None:
            return
        sync_states = self.__dict__.setdefault(
            "_inpaint_provenance_sync_states", []
        )
        sync_state = next(
            (
                state for state in sync_states
                if state.get("inpainter") is inpainter
            ),
            None,
        )
        if sync_state is None:
            sync_state = {
                "inpainter": inpainter,
                "execution_counts": {},
                "route_count": 0,
            }
            sync_states.append(sync_state)
        collect = getattr(inpainter, "execution_identity", None)
        identity = collect() if callable(collect) else {}
        if not isinstance(identity, dict):
            identity = {}
        requested = self.config.mode.value
        implementation = str(
            identity.get("implementation")
            or getattr(inpainter, "_vsr_registered_implementation", "")
            or requested
        )
        provider = str(
            identity.get("provider")
            or self._inpainter_provider_name(inpainter)
        )
        effective = self._provider_effective_device(
            provider,
            str(identity.get("effectiveDevice") or self.config.device),
        )
        stage = self.execution_provenance.stage("inpaint")
        if stage is None:
            stage = StageProvenance(stage="inpaint")
            self.execution_provenance.set_stage(stage)
        stage.requested_device = self.execution_provenance.requested_device
        stage.effective_device = effective
        stage.engine = requested
        stage.backend = provider
        stage.provider = provider
        stage.requested_implementation = requested
        stage.selection_policy = "auto" if requested == "auto" else "explicit"
        stage.outcome = "executed"
        stage.failure_class = ""
        stage.recovery_hint = ""
        executions = identity.get("actualExecutions")
        if isinstance(executions, list) and executions:
            previous_counts = sync_state.get("execution_counts", {})
            if not isinstance(previous_counts, dict):
                previous_counts = {}
            current_counts = {}
            for item in executions:
                if not isinstance(item, dict):
                    continue
                execution_implementation = str(
                    item.get("implementation") or implementation
                )
                execution_provider = str(item.get("provider") or provider)
                execution_device = self._provider_effective_device(
                    execution_provider,
                    str(item.get("effectiveDevice") or effective),
                )
                try:
                    execution_count = max(
                        0, int(item.get("executionCount") or 0)
                    )
                except (TypeError, ValueError):
                    execution_count = 0
                key = (
                    execution_implementation,
                    execution_provider,
                    normalize_device(execution_device),
                )
                prior = max(0, int(previous_counts.get(key, 0) or 0))
                delta = (
                    execution_count - prior
                    if execution_count >= prior else execution_count
                )
                if delta:
                    stage.record_execution(
                        execution_implementation,
                        provider=execution_provider,
                        effective_device=execution_device,
                        count=delta,
                    )
                current_counts[key] = execution_count
            sync_state["execution_counts"] = current_counts
            stage.actual_implementation = stage.resolved_actual_implementation
        else:
            stage.record_execution(
                implementation,
                provider=provider,
                effective_device=effective,
                count=max(0, int(frame_count)),
            )
        chain = identity.get("fallbackChain")
        if isinstance(chain, list) and chain:
            try:
                synced_route_count = max(
                    0, int(sync_state.get("route_count", 0))
                )
            except (TypeError, ValueError):
                synced_route_count = 0
            if synced_route_count > len(chain):
                synced_route_count = 0
            for item in chain[synced_route_count:]:
                if not isinstance(item, dict):
                    continue
                route = dict(item)
                route["effectiveDevice"] = self._provider_effective_device(
                    str(route.get("provider") or ""),
                    str(route.get("effectiveDevice") or effective),
                )
                stage.fallback_chain.append(route)
            sync_state["route_count"] = len(chain)
        if stage.device_fell_back and not stage.fallback_reason:
            stage.fallback_reason = (
                getattr(self, "_device_fallback_reason", "")
                or f"{provider} executed on {effective}"
            )

    def _record_stage_success(
        self,
        stage_name: str,
        implementation: str,
        *,
        provider: str,
        count: int = 1,
    ) -> None:
        effective = self._provider_effective_device(provider, self.config.device)
        stage = self.execution_provenance.stage(stage_name)
        if stage is None:
            stage = self.execution_provenance.begin_stage(
                stage_name,
                requested_implementation=implementation,
                requested_device=self.execution_provenance.requested_device,
            )
        self.execution_provenance.record_success(
            stage_name,
            implementation=implementation,
            provider=provider,
            effective_device=effective,
            count=count,
        )

    def _record_requested_stage_failure(
        self, error: RequestedStageError
    ) -> None:
        self.execution_provenance.record_failure(
            error,
            requested_device=self.execution_provenance.requested_device,
            effective_device=self.config.device,
        )
        from backend.failure_reason import classify_failure_reason
        self.last_error_message = str(error)
        self.last_error_reason = classify_failure_reason(exc=error)

    def _reset_job_execution_provenance(self) -> None:
        detector = getattr(self, "detector", None)
        if detector is not None:
            detector._execution_counts = {}
        inpainter = getattr(self, "inpainter", None)
        if inpainter is not None:
            for name, empty in (
                ("_execution_counts", {}),
                ("_route_chain", []),
            ):
                if hasattr(inpainter, name):
                    setattr(inpainter, name, empty)
        self.__dict__["_inpaint_provenance_sync_states"] = []
        self.__dict__.pop("_execution_provenance", None)
        self._refresh_execution_provenance()

    def _validate_inpaint_results(
        self,
        frames: List[np.ndarray],
        results: Any,
        masks: Optional[List[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        requested, implementation, provider, chain, execution_contract = (
            self._inpaint_failure_identity()
        )

        def invalid_output(
            detail: str,
            recovery_hint: str,
        ) -> RequestedStageError:
            return RequestedStageError(
                stage="inpaint",
                requested_implementation=requested,
                actual_implementation=implementation,
                provider=provider,
                failure_class=FAILURE_OUTPUT_INVALID,
                detail=detail,
                recovery_hint=recovery_hint,
                fallback_chain=chain,
                selection_policy=("auto" if requested == "auto" else "explicit"),
            )

        if not isinstance(results, (list, tuple)):
            raise invalid_output(
                "the inpainter returned a non-sequence result",
                "Verify the selected inpainter output contract.",
            )
        if len(results) != len(frames):
            raise invalid_output(
                (
                    f"the inpainter returned {len(results)} frame(s) for "
                    f"{len(frames)} input frame(s)"
                ),
                "Verify the selected inpainter output contract.",
            )
        validated: List[np.ndarray] = []
        for index, (source, candidate) in enumerate(zip(frames, results)):
            if not isinstance(candidate, np.ndarray) or candidate.shape != source.shape:
                raise invalid_output(
                    f"inpainter frame {index} has an invalid shape or type",
                    "Verify the selected inpainter output contract.",
                )
            if candidate.dtype != np.uint8:
                raise invalid_output(
                    f"inpainter frame {index} is not uint8",
                    "Return uint8 BGR frames from the selected inpainter.",
                )
            validated.append(np.ascontiguousarray(candidate))
        if masks is not None:
            active_masks = 0
            changed_masked_pixels = False
            for source, candidate, mask in zip(frames, validated, masks):
                mask_array = np.asarray(mask)
                if mask_array.ndim == 3:
                    active = np.any(mask_array != 0, axis=2)
                elif mask_array.ndim == 2:
                    active = mask_array != 0
                else:
                    continue
                if active.shape != source.shape[:2] or not np.any(active):
                    continue
                active_masks += 1
                if np.any(candidate[active] != source[active]):
                    changed_masked_pixels = True
                    break
            if (
                active_masks
                and not changed_masked_pixels
                and execution_contract != "vsr-inpaint-v1"
            ):
                raise invalid_output(
                    "the inpainter left every active masked pixel unchanged",
                    (
                        "Verify the selected inpainter model and checkpoint, "
                        "then retry."
                    ),
                )
        return validated

    def _inpaint_failure_identity(
        self,
    ) -> tuple[str, str, str, list[dict], str]:
        """Return the last observed route without mistaking Auto for a model."""
        inpainter = self.inpainter
        collect = getattr(inpainter, "execution_identity", None)
        try:
            identity = collect() if callable(collect) else {}
        except Exception:
            identity = {}
        if not isinstance(identity, dict):
            identity = {}

        requested = self.config.mode.value
        try:
            registered = getattr(
                inpainter, "_vsr_registered_implementation", ""
            )
        except Exception:
            registered = ""
        implementation = str(
            identity.get("implementation") or registered or requested
        )
        provider = str(
            identity.get("provider")
            or self._inpainter_provider_name(inpainter)
        )
        executions = identity.get("actualExecutions")
        if isinstance(executions, list):
            executed = []
            for item in executions:
                if not isinstance(item, dict):
                    continue
                try:
                    count = int(item.get("executionCount") or 0)
                except (TypeError, ValueError):
                    count = 0
                if count > 0:
                    executed.append(item)
            implementations = {
                str(item.get("implementation") or "")
                for item in executed
                if str(item.get("implementation") or "")
            }
            providers = {
                str(item.get("provider") or "")
                for item in executed
                if str(item.get("provider") or "")
            }
            if implementations:
                implementation = (
                    next(iter(implementations))
                    if len(implementations) == 1 else "mixed"
                )
            if providers:
                provider = next(iter(providers)) if len(providers) == 1 else "mixed"
        chain = identity.get("fallbackChain")
        return (
            requested,
            implementation,
            provider,
            list(chain) if isinstance(chain, list) else [],
            str(identity.get("executionContract") or ""),
        )

    def _execute_inpainter(
        self,
        frames: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> Any:
        """Run the selected inpainter and classify every runtime failure."""
        try:
            result = self.inpainter.inpaint(frames, masks)
            result = self._validate_inpaint_results(frames, result, masks)
            self._sync_inpaint_provenance(len(frames))
            return result
        except RequestedStageError:
            raise
        except Exception as exc:
            requested, implementation, provider, chain, _execution_contract = (
                self._inpaint_failure_identity()
            )
            raise RequestedStageError(
                stage="inpaint",
                requested_implementation=requested,
                actual_implementation=implementation,
                provider=provider,
                failure_class=FAILURE_RUNTIME,
                detail=str(exc),
                recovery_hint=str(
                    getattr(self.inpainter, "_vsr_recovery_hint", "")
                    or "Verify the selected inpainter and provider, then retry."
                ),
                fallback_chain=chain,
                selection_policy=("auto" if requested == "auto" else "explicit"),
                cause=exc,
            ) from exc

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
        self._sync_ocr_provenance()
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

    @staticmethod
    def _legacy_geometry(boxes) -> List[DetectionGeometry]:
        output: List[DetectionGeometry] = []
        for value in boxes or []:
            detection = as_detection_geometry(value)
            if detection is not None:
                output.append(detection)
        return output

    def _detect_geometry(self, frame: np.ndarray, threshold: float):
        """Use polygon-aware detection when the detector exposes it."""
        method = getattr(self.detector, "detect_with_geometry", None)
        if callable(method):
            return list(method(frame, threshold))
        return self._legacy_geometry(
            self.detector.detect(frame, threshold))

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
        try:
            if not cap.isOpened():
                return None
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
                     confidences: Optional[List[float]] = None,
                     detections: Optional[List[DetectionGeometry]] = None) -> np.ndarray:
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        geometry = list(detections or [])
        mask_boxes = [detection.bbox for detection in geometry] or list(boxes)
        has_polygons = any(detection.polygon for detection in geometry)
        base_dilate = self.config.mask_dilate_px
        auto_dilate = bool(
            getattr(self.config, "auto_dilate_enable", False)
            and frame is not None
        )
        use_conf_dilate = (
            self.config.confidence_weighted_dilation
            and confidences is not None
            and base_dilate > 0
            and not auto_dilate
        )

        if has_polygons:
            from backend.segmentation import (
                estimate_auto_dilation_radius,
                soft_dilate_mask,
            )
            for index, detection in enumerate(geometry):
                if auto_dilate:
                    radius = estimate_auto_dilation_radius(
                        frame, detection.bbox)
                    local = geometry_mask(
                        frame_shape,
                        detection,
                        max(0, int(padding)),
                    )
                    mask = np.maximum(mask, soft_dilate_mask(local, radius))
                    continue
                effective = base_dilate
                if use_conf_dilate:
                    confidence = (
                        confidences[index]
                        if confidences is not None and index < len(confidences)
                        else detection.confidence
                    )
                    scale = self.config.confidence_dilation_scale
                    effective = int(
                        base_dilate
                        * (1.0 + (1.0 - confidence) * scale))
                local = geometry_mask(
                    frame_shape,
                    detection,
                    max(0, int(padding)) + max(0, int(effective)),
                )
                mask = cv2.bitwise_or(mask, local)
        elif auto_dilate:
            from backend.segmentation import (
                estimate_auto_dilation_radius,
                soft_dilate_mask,
            )
            for x1, y1, x2, y2 in boxes:
                bx1 = max(0, x1 - padding)
                by1 = max(0, y1 - padding)
                bx2 = min(w, x2 + padding)
                by2 = min(h, y2 + padding)
                box_mask = np.zeros((h, w), dtype=np.uint8)
                box_mask[by1:by2, bx1:bx2] = 255
                radius = estimate_auto_dilation_radius(frame, (x1, y1, x2, y2))
                mask = np.maximum(mask, soft_dilate_mask(box_mask, radius))
                logger.debug(
                    "Auto mask dilation box=(%d,%d,%d,%d) radius=%d",
                    x1, y1, x2, y2, radius,
                )
        else:
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
        if frame is not None and mask_boxes and self.config.sam2_refine:
            from backend.segmentation import (
                refine_mask_with_sam2,
                selected_segmentation_provider,
            )
            mask = refine_mask_with_sam2(
                frame, mask_boxes, mask, self.config.device)
            self._record_stage_success(
                "sam2",
                "sam2",
                provider=selected_segmentation_provider("sam2"),
            )

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
        from backend.segmentation import (
            refine_masks_with_matanyone,
            selected_segmentation_provider,
        )
        refined = refine_masks_with_matanyone(
            frames, masks, self.config.device)
        if any(np.any(mask > 0) for mask in masks):
            self._record_stage_success(
                "matanyone",
                "matanyone2",
                provider=selected_segmentation_provider("matanyone2"),
                count=len(frames),
            )
        return refined

    def _propagate_masks_with_cotracker(self,
                                        frames: List[np.ndarray],
                                        masks: List[np.ndarray]) -> List[np.ndarray]:
        if not getattr(self.config, "cotracker_propagate", False):
            return masks
        if not frames or not masks:
            return masks
        from backend.segmentation import (
            propagate_masks_with_cotracker,
            selected_segmentation_provider,
        )
        requires_tracking = (
            len(frames) >= 2
            and any(np.any(mask > 0) for mask in masks)
            and any(not np.any(mask > 0) for mask in masks)
        )
        propagated = propagate_masks_with_cotracker(
            frames,
            masks,
            device=self.config.device,
        )
        if requires_tracking:
            self._record_stage_success(
                "cotracker",
                "cotracker3",
                provider=selected_segmentation_provider("cotracker3"),
                count=len(frames),
            )
        return propagated

    # -----------------------------------------------------------------
    # SRT export
    # -----------------------------------------------------------------
    def process_image(self, input_path: str, output_path: str) -> bool:
        self._teardown_requested = False
        self.last_output_path = None
        self._reset_stage_timings()
        self._reset_detection_stats()
        self._reset_job_execution_provenance()
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
            manual_only = bool(self.config.sttn_skip_detection)
            if manual_only and not fixed_shapes:
                raise ValueError(
                    "Manual region mode needs a fixed region for image cleanup"
                )
            confidences = None
            detection_geometry: List[DetectionGeometry] = []
            self.last_detection_stats["frames_total"] = 1
            with self._time_stage("ocr"):
                if manual_only:
                    boxes = list(fixed or [])
                    detection_geometry = self._legacy_geometry(boxes)
                    self._record_detection_skip("manual_region")
                elif self.config.language_mask_filter:
                    from backend.detection import text_matches_detection_language
                    if callable(getattr(
                            self.detector, "detect_with_geometry", None)):
                        detection_geometry = self._detect_geometry(
                            image, self.config.detection_threshold)
                        matched_geometry = [
                            detection for detection in detection_geometry
                            if text_matches_detection_language(
                                detection.text,
                                self.config.detection_lang,
                            )
                        ]
                        if detection_geometry and not any(
                                detection.text for detection in detection_geometry):
                            matched_geometry = []
                        detection_geometry = matched_geometry
                        boxes = [
                            detection.bbox
                            for detection in detection_geometry
                        ]
                        if self.config.confidence_weighted_dilation:
                            confidences = [
                                detection.confidence
                                for detection in detection_geometry
                            ]
                    else:
                        results = self.detector.detect_with_text(
                            image, self.config.detection_threshold)
                        matched = [
                            result for result in results
                            if text_matches_detection_language(
                                result[5], self.config.detection_lang)
                        ]
                        boxes = [result[:4] for result in matched]
                        if self.config.confidence_weighted_dilation:
                            confidences = [result[4] for result in matched]
                    self._record_ocr_detection(boxes)
                elif self.config.confidence_weighted_dilation:
                    if callable(getattr(
                            self.detector, "detect_with_geometry", None)):
                        detection_geometry = self._detect_geometry(
                            image, self.config.detection_threshold)
                        boxes = [
                            detection.bbox
                            for detection in detection_geometry
                        ]
                        confidences = [
                            detection.confidence
                            for detection in detection_geometry
                        ]
                    else:
                        results = self.detector.detect_with_confidence(
                            image, self.config.detection_threshold)
                        boxes = [
                            (x1, y1, x2, y2)
                            for x1, y1, x2, y2, _ in results
                        ]
                        confidences = [c for _, _, _, _, c in results]
                    self._record_ocr_detection(boxes)
                else:
                    if callable(getattr(
                            self.detector, "detect_with_geometry", None)):
                        detection_geometry = self._detect_geometry(
                            image, self.config.detection_threshold)
                        boxes = [
                            detection.bbox
                            for detection in detection_geometry
                        ]
                    else:
                        boxes = self.detector.detect(
                            image, self.config.detection_threshold)
                    self._record_ocr_detection(boxes)

                if not manual_only and fixed:
                    boxes = list(fixed) + list(boxes)
                    detection_geometry = (
                        self._legacy_geometry(list(fixed))
                        + detection_geometry
                    )
                    confidences = None

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
                                         confidences=confidences,
                                         detections=detection_geometry)
                mask = self._apply_polygon_region_shapes(mask, fixed_shapes)
                mask = self._apply_manual_mask_corrections(mask, 0.0, 0)
                [mask] = self._refine_masks_with_matanyone([image], [mask])
            with self._time_stage("inpaint"):
                results = self._validate_inpaint_results(
                    [image], self._execute_inpainter([image], [mask])
                )
                [result] = results

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
        except RequestedStageError as e:
            self._record_requested_stage_failure(e)
            logger.error("Requested image stage failed: %s", e, exc_info=True)
            return False
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
    def _processing_frame(frame: np.ndarray, *, transfer: Optional[str] = None
                          ) -> np.ndarray:
        """Return the uint8 BGR working copy expected by OCR/inpainters."""
        if frame.dtype == np.uint8:
            return np.ascontiguousarray(frame)
        if frame.dtype == np.uint16:
            if transfer:
                try:
                    return hdr_proxy_from_high_bit(frame, transfer)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "HDR repair cannot create a tone-mapped proxy from "
                        f"transfer metadata {transfer!r}"
                    ) from exc
            return np.ascontiguousarray(
                np.clip(np.rint(frame.astype(np.float32) / 257.0), 0, 255)
                .astype(np.uint8)
            )
        return np.ascontiguousarray(np.clip(frame, 0, 255).astype(np.uint8))

    @staticmethod
    def _is_high_bit_frame(frame: Any) -> bool:
        return isinstance(frame, np.ndarray) and frame.dtype == np.uint16

    @staticmethod
    def _linear_precision_fill(source_linear: np.ndarray,
                               mask: np.ndarray) -> np.ndarray:
        """Recover sub-8-bit boundary detail inside the active mask ROI."""
        mask_u8 = np.asarray(mask, dtype=np.uint8)
        active = mask_u8 > 0
        if not np.any(active):
            return np.ascontiguousarray(source_linear)
        points = cv2.findNonZero(mask_u8)
        if points is None:
            return np.ascontiguousarray(source_linear)
        x, y, width, height = cv2.boundingRect(points)
        pad = max(4, min(64, max(width, height) // 8 or 4))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(source_linear.shape[1], x + width + pad)
        y1 = min(source_linear.shape[0], y + height + pad)
        region_mask = mask_u8[y0:y1, x0:x1]
        if np.all(region_mask > 0):
            return np.ascontiguousarray(source_linear)
        region = np.ascontiguousarray(source_linear[y0:y1, x0:x1])
        seed8 = np.clip(np.rint(region * 255.0), 0, 255).astype(np.uint8)
        seed8 = cv2.inpaint(seed8, region_mask, 5, cv2.INPAINT_TELEA)
        seed = seed8.astype(np.float32) / 255.0
        region[region_mask > 0] = seed[region_mask > 0]
        kernel = np.array(
            [[0.0, 0.25, 0.0],
             [0.25, 0.0, 0.25],
             [0.0, 0.25, 0.0]],
            dtype=np.float32,
        )
        iterations = max(12, min(64, max(region.shape[:2]) * 2))
        for _ in range(iterations):
            candidate = cv2.filter2D(
                region, -1, kernel, borderType=cv2.BORDER_REPLICATE)
            region[region_mask > 0] = candidate[region_mask > 0]
        result = np.ascontiguousarray(source_linear.copy())
        result[y0:y1, x0:x1][region_mask > 0] = region[region_mask > 0]
        return result

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

        meta = getattr(self, "_color_metadata", None)
        transfer = getattr(meta, "color_transfer", "")
        hdr_surface = bool(
            meta is not None
            and (getattr(meta, "is_hdr", False)
                 or getattr(meta, "is_high_bit", False))
        )
        if meta is None or hdr_surface:
            reason = hdr_repair_block_reason(meta)
            if reason or not getattr(self, "_hdr_repair_ready", False):
                raise ValueError(
                    "HDR repair is blocked because transfer metadata is not "
                    f"ready for linear-light processing: {reason or 'not ready'}"
                )
        if hdr_surface:
            mask_u8 = np.asarray(mask, dtype=np.uint8)
            points = cv2.findNonZero(mask_u8)
            if points is None:
                return np.ascontiguousarray(source_frame)
            x, y, width, height = cv2.boundingRect(points)
            feather = max(
                0, int(getattr(self.config, "mask_feather_px", 0) or 0))
            pad = max(4, feather * 2 + 4)
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(source_frame.shape[1], x + width + pad)
            y1 = min(source_frame.shape[0], y + height + pad)
            region_mask = mask_u8[y0:y1, x0:x1]
            source_region = source_frame[y0:y1, x0:x1]
            cleaned_region = cleaned_frame[y0:y1, x0:x1]
            source_linear = hdr_signal_to_linear(source_region, transfer)
            cleaned_linear = hdr_proxy_to_linear(cleaned_region, transfer)
            precision_fill = self._linear_precision_fill(
                source_linear, region_mask)
            precision_proxy = hdr_proxy_from_linear(
                precision_fill, transfer)
            quantized_linear = hdr_proxy_to_linear(
                precision_proxy, transfer)
            cleaned_linear = np.clip(
                cleaned_linear + (precision_fill - quantized_linear) * 0.75,
                0.0,
                1.0,
            )
            if feather > 0:
                k = feather * 2 + 1
                alpha = cv2.GaussianBlur(
                    region_mask, (k, k), 0).astype(np.float32) / 255.0
            else:
                alpha = (region_mask > 0).astype(np.float32)
            alpha = np.minimum(alpha, (region_mask > 0).astype(np.float32))
            alpha = np.clip(alpha, 0.0, 1.0)[..., None]
            merged_linear = (
                source_linear * (1.0 - alpha)
                + cleaned_linear * alpha
            )
            encoded = linear_to_hdr_signal(merged_linear, transfer)
            output = np.ascontiguousarray(source_frame.copy())
            changed = alpha[..., 0] > 0.0
            output[y0:y1, x0:x1][changed] = encoded[changed]
            return output
        if transfer:
            raise ValueError(
                "HDR repair is blocked because transfer metadata is not "
                "ready for linear-light processing"
            )
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
            return self._execute_inpainter(frames, masks)

        key_indices = list(range(0, len(frames), stride))
        if key_indices[-1] != len(frames) - 1:
            key_indices.append(len(frames) - 1)
        if len(key_indices) >= len(frames):
            return self._execute_inpainter(frames, masks)

        key_frames = [frames[i] for i in key_indices]
        key_masks = [masks[i] for i in key_indices]
        key_results = self._execute_inpainter(key_frames, key_masks)
        if len(key_results) != len(key_indices):
            logger.warning(
                "RIFE fast mode disabled for batch: inpainter returned "
                f"{len(key_results)} keyframes for {len(key_indices)} inputs"
            )
            return self._execute_inpainter(frames, masks)

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
        """Inpaint a batch, recovering from GPU OOM without changing models.

        On an out-of-memory failure the CUDA cache is cleared and the batch is
        split in half and retried recursively down to a single frame. A frame
        that still cannot run on the GPU retries the same registered
        implementation on CPU.
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
                    "GPU out of memory on a single frame; retrying the same "
                    "inpainting implementation on CPU."
                )
                return self._retry_inpaint_on_cpu(frames, masks, exc)
            half = max(1, len(frames) // 2)
            logger.warning(
                "GPU out of memory on a batch of %d frames; clearing cache and "
                "retrying as %d + %d.", len(frames), half, len(frames) - half,
            )
            left = self._inpaint_batch_resilient(frames[:half], masks[:half])
            right = self._inpaint_batch_resilient(frames[half:], masks[half:])
            return left + right

    def _retry_inpaint_on_cpu(
        self,
        frames: List[np.ndarray],
        masks: List[np.ndarray],
        original_error: BaseException,
    ) -> List[np.ndarray]:
        requested = self.config.mode.value
        previous = self.inpainter
        previous_identity = str(
            getattr(previous, "_vsr_registered_implementation", "")
            or requested
        )
        try:
            cpu_inpainter = self.device_provider.create_inpainter(
                requested, "cpu", self.config
            )
            cpu_identity = str(
                getattr(cpu_inpainter, "_vsr_registered_implementation", "")
                or requested
            )
            if cpu_identity != previous_identity:
                raise RequestedStageError(
                    stage="inpaint",
                    requested_implementation=requested,
                    actual_implementation=cpu_identity,
                    failure_class=FAILURE_RUNTIME,
                    detail=(
                        "CPU recovery resolved to a different implementation "
                        f"({previous_identity} to {cpu_identity})"
                    ),
                    recovery_hint=(
                        "Reduce the batch size or choose Auto before retrying."
                    ),
                )
            self.inpainter = cpu_inpainter
            self.config.device = "cpu"
            self._device_fallback_reason = (
                f"{previous_identity} exhausted GPU memory and retried on CPU"
            )
            self.execution_provenance.effective_device = "cpu"
            self.execution_provenance.device_fallback_reason = (
                self._device_fallback_reason
            )
            stage = self.execution_provenance.stage("inpaint")
            if stage is not None:
                stage.fallback_chain.extend([
                    {
                        "implementation": previous_identity,
                        "outcome": "runtime_failed",
                        "provider": str(
                            self._inpainter_provider_name(previous)
                        ),
                        "effectiveDevice": stage.effective_device,
                        "failureClass": FAILURE_RUNTIME,
                        "reason": str(original_error),
                    },
                    {
                        "implementation": cpu_identity,
                        "outcome": "selected",
                        "provider": str(
                            self._inpainter_provider_name(cpu_inpainter)
                        ),
                        "effectiveDevice": "cpu",
                        "reason": "same implementation CPU retry",
                    },
                ])
            return self._inpaint_with_optional_rife_fast(frames, masks)
        except RequestedStageError:
            raise
        except Exception as exc:
            raise RequestedStageError(
                stage="inpaint",
                requested_implementation=requested,
                actual_implementation=previous_identity,
                provider=self._inpainter_provider_name(previous),
                failure_class=FAILURE_RUNTIME,
                detail=f"GPU and same-implementation CPU retries failed: {exc}",
                recovery_hint=(
                    "Reduce the batch size, repair the selected provider, or "
                    "choose Auto before retrying."
                ),
                cause=exc,
            ) from exc

    def _decode_and_build_batch(self, ctx: _FrameLoopContext,
                                state: _FrameLoopState) -> _FrameBatch:
        """Decode frames and build masks until one processing batch is full."""
        batch = _FrameBatch()
        hdr_transfer = getattr(
            getattr(self, "_color_metadata", None), "color_transfer", "")
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
                frame = self._processing_frame(
                    raw_frame,
                    transfer=hdr_transfer if source_frame is not None else None,
                )

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
                    prior_source_frame = (
                        prior_raw
                        if ctx.high_bit_depth_surface
                        and self._is_high_bit_frame(prior_raw)
                        else None
                    )
                    prior_frame = self._processing_frame(
                        prior_raw,
                        transfer=(
                            hdr_transfer if prior_source_frame is not None
                            else None
                        ),
                    )
                    batch.add(
                        prior_frame,
                        np.zeros(prior_frame.shape[:2], dtype=np.uint8),
                        prior_source_frame,
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
                        state.tracker.reset()
                    if (
                        state.srt_tracker is not None
                        and state.srt_tracker is not state.tracker
                    ):
                        state.srt_tracker.reset()
            # RM-153: a frozen matte is this job's authoritative mask. It
            # was reviewed and approved frame by frame against exactly
            # these frames, so detection, tracking, and every heuristic
            # that would alter it are skipped outright -- re-deriving a
            # mask a human already signed off on is the cost the freeze
            # exists to avoid, and refining it would change the pixels
            # they approved.
            if ctx.frozen_matte:
                self._record_detection_skip("frozen_matte")
                with self._time_stage("mask"):
                    mask = ctx.matte_reader.read(state.frame_idx)
                state.last_mask = mask
                batch.add(frame, mask, source_frame, passthrough=False)
                state.frame_idx += 1
                continue

            fixed_shapes = (
                self._fixed_region_shapes(frame_seconds)
                if ctx.timed_region_spans else ctx.static_fixed_shapes
            )
            if (
                self.config.sttn_skip_detection
                and not fixed_shapes
                and not ctx.timed_region_spans
            ):
                raise ValueError(
                    "Manual region mode needs a fixed, timed, or moving region"
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
                    and not ctx.timed_mask_corrections
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
                    and not ctx.timed_mask_corrections
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
                    and not ctx.timed_mask_corrections
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
                detection_geometry: List[DetectionGeometry] = []
                geometry_detector = callable(getattr(
                    self.detector, "detect_with_geometry", None))
                collect_confidence = bool(
                    self.config.confidence_weighted_dilation
                    or self.config.quality_report
                )
                if self.config.language_mask_filter:
                    from backend.detection import text_matches_detection_language
                    if geometry_detector:
                        detection_geometry = self._detect_geometry(
                            det_frame, self.config.detection_threshold)
                        detection_geometry = [
                            detection for detection in detection_geometry
                            if text_matches_detection_language(
                                detection.text,
                                self.config.detection_lang,
                            )
                        ]
                        detected_boxes = [
                            detection.bbox
                            for detection in detection_geometry
                        ]
                        det_confs = [
                            detection.confidence
                            for detection in detection_geometry
                        ]
                    else:
                        text_results = self.detector.detect_with_text(
                            det_frame, self.config.detection_threshold)
                        matched = [
                            result for result in text_results
                            if text_matches_detection_language(
                                result[5], self.config.detection_lang)
                        ]
                        detection_geometry = [
                            detection
                            for result in matched
                            for detection in [DetectionGeometry.from_box(
                                result[:4],
                                frame.shape,
                                confidence=result[4],
                                text=result[5],
                            )]
                            if detection is not None
                        ]
                        detected_boxes = [
                            detection.bbox
                            for detection in detection_geometry
                        ]
                        det_confs = [
                            detection.confidence
                            for detection in detection_geometry
                        ]
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
                elif collect_confidence:
                    if geometry_detector:
                        detection_geometry = self._detect_geometry(
                            det_frame, self.config.detection_threshold)
                        detected_boxes = [
                            detection.bbox
                            for detection in detection_geometry
                        ]
                        det_confs = [
                            detection.confidence
                            for detection in detection_geometry
                        ]
                    else:
                        det_results = self.detector.detect_with_confidence(
                            det_frame, self.config.detection_threshold)
                        detection_geometry = [
                            detection
                            for x1, y1, x2, y2, confidence in det_results
                            for detection in [DetectionGeometry.from_box(
                                (x1, y1, x2, y2),
                                frame.shape,
                                confidence=confidence,
                            )]
                            if detection is not None
                        ]
                        detected_boxes = [
                            detection.bbox
                            for detection in detection_geometry
                        ]
                        det_confs = [
                            detection.confidence
                            for detection in detection_geometry
                        ]
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
                    if geometry_detector:
                        detection_geometry = self._detect_geometry(
                            det_frame, self.config.detection_threshold)
                        detected_boxes = [
                            detection.bbox
                            for detection in detection_geometry
                        ]
                    elif (
                        state.srt_tracker is not None
                        and callable(getattr(
                            self.detector, "detect_with_text", None))
                    ):
                        text_results = self.detector.detect_with_text(
                            det_frame, self.config.detection_threshold)
                        detection_geometry = [
                            detection
                            for result in text_results
                            for detection in [DetectionGeometry.from_box(
                                result[:4],
                                frame.shape,
                                confidence=result[4],
                                text=result[5],
                            )]
                            if detection is not None
                        ]
                        detected_boxes = [
                            detection.bbox
                            for detection in detection_geometry
                        ]
                    else:
                        detected_boxes = self.detector.detect(
                            det_frame, self.config.detection_threshold)
                self._record_ocr_detection(detected_boxes)
                if self.config.karaoke_grouping and detected_boxes:
                    if detection_geometry:
                        detection_geometry = _group_horizontal_geometry(
                            detection_geometry,
                            x_gap_px=self.config.karaoke_x_gap_px,
                            y_overlap_ratio=self.config.karaoke_y_overlap,
                        )
                        detected_boxes = [
                            detection.bbox
                            for detection in detection_geometry
                        ]
                    else:
                        detected_boxes = _group_horizontal_line(
                            detected_boxes,
                            x_gap_px=self.config.karaoke_x_gap_px,
                            y_overlap_ratio=self.config.karaoke_y_overlap,
                        )
                    det_confs = None
                if state.tracker is not None:
                    update_geometry = getattr(
                        state.tracker, "update_with_geometry", None)
                    if callable(update_geometry):
                        tracking_geometry = (
                            detection_geometry
                            if detection_geometry
                            else self._legacy_geometry(detected_boxes)
                        )
                        smoothed_geometry = list(
                            update_geometry(list(tracking_geometry)))
                    else:
                        smoothed_geometry = self._legacy_geometry(
                            state.tracker.update(list(detected_boxes)))
                    smoothed = [
                        detection.bbox
                        for detection in smoothed_geometry
                    ]
                    det_confs = None
                else:
                    smoothed = list(detected_boxes)
                    smoothed_geometry = (
                        list(detection_geometry)
                        if detection_geometry else self._legacy_geometry(smoothed)
                    )
                srt_geometry = list(smoothed_geometry)
                if (
                    state.srt_tracker is not None
                    and state.srt_tracker is not state.tracker
                ):
                    srt_geometry = list(
                        state.srt_tracker.update_with_geometry(
                            list(
                                detection_geometry
                                if detection_geometry
                                else self._legacy_geometry(detected_boxes)
                            )
                        )
                    )
                if (state.tracker is not None
                        and (not self.config.remove_chyrons
                             or not self.config.remove_subtitles)):
                    cats = state.tracker.categorize(
                        self.config.chyron_min_hits)
                    smoothed_geometry = [
                        detection for detection, category in zip(
                            smoothed_geometry, cats)
                        if (
                            category == "chyron"
                            and self.config.remove_chyrons
                        ) or (
                            category == "subtitle"
                            and self.config.remove_subtitles
                        )
                    ]
                    smoothed = [
                        detection.bbox for detection in smoothed_geometry
                    ]
                    det_confs = None
                if fixed_boxes:
                    boxes = list(fixed_boxes) + smoothed
                    detection_geometry = (
                        self._legacy_geometry(fixed_boxes)
                        + smoothed_geometry
                    )
                    det_confs = None
                else:
                    boxes = smoothed
                    detection_geometry = smoothed_geometry
                if (
                    self.config.export_srt
                    or (
                        self.config.translation_enabled
                        and not self.config.translation_srt
                        and not self.config.translation_source_srt
                    )
                ):
                    self._collect_srt_entry(
                        frame, state.frame_idx, srt_geometry)

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
                        detection_geometry = self._legacy_geometry(boxes)
                        break

            with self._time_stage("mask"):
                mask = self._create_mask(
                    frame.shape,
                    boxes,
                    frame=frame,
                    confidences=det_confs,
                    detections=detection_geometry,
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

    @staticmethod
    def _mark_active_segments(batch: _FrameBatch) -> None:
        """Record the contiguous non-passthrough runs to inpaint."""
        segment_start = None
        for index, passthrough in enumerate(
                batch.passthrough_flags + [True]):
            if not passthrough and segment_start is None:
                segment_start = index
            elif passthrough and segment_start is not None:
                batch.active_segments.append((segment_start, index))
                segment_start = None

    def _refine_batch_masks(self, ctx: _FrameLoopContext,
                            state: _FrameLoopState,
                            batch: _FrameBatch) -> None:
        """Apply temporal/refiner/imported-matte passes to one batch."""
        with self._time_stage("mask"):
            self._mark_active_segments(batch)
            # RM-153: the masks in a frozen batch came straight from the
            # approved matte. Propagation, refinement, and rolling-union
            # stabilization all exist to improve a *derived* mask, so
            # running them here would silently edit pixels a human signed
            # off on. Segments, quality accounting, and progress still
            # run -- only the mask-altering passes are skipped.
            if ctx.frozen_matte:
                if self.config.quality_report:
                    for index, mask in enumerate(batch.masks):
                        if not batch.passthrough_flags[index]:
                            self._accumulate_quality_bbox(mask)
                self._finish_batch_masks(ctx, state, batch)
                return
            for segment_start, segment_end in batch.active_segments:
                segment_frames = batch.frames[segment_start:segment_end]
                segment_masks = batch.masks[segment_start:segment_end]
                segment_masks = self._propagate_masks_with_cotracker(
                    segment_frames, segment_masks)
                segment_masks = self._refine_masks_with_matanyone(
                    segment_frames, segment_masks)
                if (self.config.temporal_mask_union
                        and not ctx.timed_region_spans
                        and not ctx.timed_mask_corrections
                        and not self.config.sttn_skip_detection
                        and len(segment_masks) > 1):
                    scene_starts = _detect_scene_cuts(segment_frames)
                    segment_masks = stabilize_masks_rolling_union(
                        segment_masks,
                        scene_starts,
                        self.config.temporal_mask_window,
                    )
                batch.masks[segment_start:segment_end] = segment_masks
            # RM-292: run the fade hold over the whole batch rather than per
            # active segment, so a fade that spans a passthrough gap is still
            # covered. A frozen matte never reaches here.
            fade_in = int(getattr(self.config, "mask_fade_in_frames", 0) or 0)
            fade_out = int(getattr(self.config, "mask_fade_out_frames", 0) or 0)
            if (fade_in > 0 or fade_out > 0) and batch.masks:
                batch.masks, state.fade_carry = extend_masks_across_fades(
                    batch.masks, fade_in, fade_out, state.fade_carry)
            if ctx.matte_reader is not None:
                batch_start = state.written_idx
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
        self._finish_batch_masks(ctx, state, batch)

    def _finish_batch_masks(self, ctx: _FrameLoopContext,
                            state: _FrameLoopState,
                            batch: _FrameBatch) -> None:
        """Carry the last mask forward and report batch progress."""
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
                batch_start = ctx.start_frame + state.written_idx
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
                segment_frames = reference_frames[segment_start:segment_end]
                segment_results = self._validate_inpaint_results(
                    segment_frames,
                    self._inpaint_batch_resilient(
                        segment_frames, segment_masks
                    ),
                )
                results[segment_start:segment_end] = segment_results
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
                if self.config.quality_report:
                    output_frame_index = state.written_idx + offset
                    self._persist_quality_mask(
                        output_frame_index,
                        batch.masks[offset]
                        if offset < len(batch.masks) else None,
                    )
                ctx.writer.write(write_frame)
        for offset, result in enumerate(results):
            frame_index = state.written_idx + offset
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
        state.written_idx += len(results)

    def _pause_requested(self, ctx: _FrameLoopContext) -> bool:
        """Ask once whether the caller wants to pause after this batch."""
        checkpoint = ctx.checkpoint
        if not checkpoint.active or checkpoint.root is None or not checkpoint.key:
            return False
        return bool(checkpoint.pause_check and checkpoint.pause_check())

    def _process_batch(self, ctx: _FrameLoopContext,
                       state: _FrameLoopState,
                       batch: _FrameBatch) -> None:
        """Refine, inpaint, and write one batch."""
        self._refine_batch_masks(ctx, state, batch)
        results = self._inpaint_batch(ctx, state, batch)
        self._write_batch(ctx, state, batch, results)

    def _checkpoint_after_batch(self, ctx: _FrameLoopContext,
                                state: _FrameLoopState,
                                should_pause: Optional[bool] = None) -> None:
        """Persist running/paused state after one fully-written batch."""
        checkpoint = ctx.checkpoint
        if not checkpoint.active or checkpoint.root is None or not checkpoint.key:
            return
        if should_pause is None:
            should_pause = bool(
                checkpoint.pause_check and checkpoint.pause_check())
        timing_metadata = checkpoint.timing_metadata
        if ctx.frame_timing is not None:
            timing_metadata = ctx.frame_timing.checkpoint_metadata(
                ctx.start_frame, ctx.end_frame, state.written_idx)
        payload = write_pause_checkpoint(
            checkpoint.root,
            checkpoint.key,
            input_path=checkpoint.input_path,
            output_path=checkpoint.output_path,
            config_hash=checkpoint.config_hash,
            frame_dir=checkpoint.frame_dir or pause_frame_dir(
                checkpoint.root, checkpoint.key),
            next_frame=state.written_idx,
            total_frames=ctx.frames_to_process,
            width=ctx.width,
            height=ctx.height,
            fps=ctx.fps,
            status="paused" if should_pause else "running",
            timing_manifest_path=checkpoint.timing_manifest_path,
            timing=timing_metadata,
        )
        self.last_pause_checkpoint = payload
        if checkpoint.state_path is not None:
            self.last_pause_checkpoint_path = str(checkpoint.state_path)
        if should_pause:
            message = (
                f"Processing paused at frame "
                f"{state.written_idx}/{ctx.frames_to_process}"
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
        self._reset_job_execution_provenance()
        self._srt_entries = []
        self._ocr_fix_replacements = None
        self._quality_mask_bbox = None
        self._seam_scores = []
        self._seam_score_failure_logged = False
        self._quality_temporal_previous = None
        self._quality_temporal_scores = []
        self._quality_temporal_scene_cuts_excluded = 0
        self._quality_temporal_worst_pair = None
        self._quality_temporal_failure_logged = False
        self._quality_color_drift_sum = 0.0
        self._quality_color_drift_count = 0
        self._quality_color_drift_metric = None
        self._quality_color_drift_worst_frame = None
        self._quality_color_failure_logged = False
        self._quality_frame_evidence_dir = None
        self._quality_frame_evidence_write_error = False
        self._quality_final_encode_verified = False
        self._mask_review_signals = []
        self.last_selective_rerun = None
        temp_dir = None
        cap = None
        selective_cap = None
        selective_prior_offset = 0
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
        frozen_record = normalize_frozen_matte(
            getattr(self.config, "frozen_matte", None))
        self.last_frozen_matte = {
            "requested": bool(frozen_record),
            "status": "pending" if frozen_record else "not-requested",
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
            if getattr(self, "_hdr_probe_failed", False):
                raise ValueError(
                    "HDR processing stopped because color metadata could not "
                    "be probed. Disable color preservation only as an explicit "
                    "override when the source is known to be SDR."
                )
            if getattr(self, "_hdr_override_blocked", False):
                raise ValueError(
                    "Color preservation cannot be disabled for an unknown, "
                    "invalid, high-bit, or HDR source; a verified SDR profile "
                    "is required before using that override."
                )
            if getattr(self, "_hdr_repair_blocked", False):
                reason = hdr_repair_block_reason(self._color_metadata)
                raise ValueError(
                    "HDR processing stopped because the source tags are not "
                    f"safe for linear-light repair: {reason}"
                )

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
                except InterruptedError:
                    # A cancel here is a cancel, not a deinterlace failure to
                    # shrug off; swallowing it kept the job decoding and
                    # inpainting until the next progress callback.
                    raise
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
            if self._source_is_hdr() and getattr(
                    self, "_hdr_repair_ready", False):
                cap = _open_required_hdr_capture(
                    decode_path,
                    input_fps=self.config.input_fps,
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
                    set_frame_timing(frame_timing)
                if frame_timing.average_fps > 0:
                    fps = float(min(frame_timing.average_fps, 1000.0))
                self.last_timing_report = frame_timing.report()
                timing_label = "variable" if frame_timing.is_vfr else "constant"
                logger.info(
                    "Source timing: %s frame rate, %d timestamps, "
                    "time base %d/%d s (%0.12fs)",
                    timing_label,
                    frame_timing.frame_count,
                    frame_timing.time_base_num,
                    frame_timing.time_base_den,
                    frame_timing.time_base,
                )
            else:
                self.last_timing_report = {
                    "mode": "cfr-fallback",
                    "frame_count": total_frames,
                    "duration_seconds": round(total_frames / fps, 9),
                    "time_base_seconds": 0.0,
                    "time_base_num": 0,
                    "time_base_den": 1,
                    "source_start_ticks": 0,
                    "timing_anomaly_count": 0,
                    "timing_anomalies": [],
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
            processed_time_start_ticks = _range.processed_time_start_ticks
            processed_time_end_ticks = _range.processed_time_end_ticks
            matte_timestamps = _range.matte_timestamps
            matte_durations = _range.matte_durations
            matte_time_base = _range.matte_time_base
            matte_timestamp_ticks = _range.matte_timestamp_ticks
            matte_duration_ticks = _range.matte_duration_ticks
            matte_time_base_num = _range.matte_time_base_num
            matte_time_base_den = _range.matte_time_base_den
            selected_frame_duration_ticks = (
                _range.selected_frame_duration_ticks
            )
            source_start_ticks = (
                int(frame_timing.source_start_ticks)
                if frame_timing is not None else None
            )
            stream_start_ticks = (
                int(frame_timing.stream_start_ticks)
                if frame_timing is not None else None
            )
            if frozen_record:
                # RM-153: revalidate before a single frame is decoded, so
                # a matte that no longer belongs to this job stops the run
                # with a specific reason instead of painting approved
                # pixels onto frames they were never approved for.
                if self.config.mask_import_path:
                    raise ValueError(
                        "A frozen matte and a manually imported matte cannot "
                        "both drive one job; clear one of them."
                    )
                frozen_evidence = validate_frozen_matte(
                    frozen_record,
                    source_path=input_path,
                    width=width,
                    height=height,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    timestamps=matte_timestamps,
                    durations=matte_durations,
                    is_vfr=bool(
                        frame_timing is not None and frame_timing.is_vfr),
                    source_time_base=matte_time_base,
                    timestamp_ticks=matte_timestamp_ticks,
                    duration_ticks=matte_duration_ticks,
                    source_time_base_num=matte_time_base_num,
                    source_time_base_den=matte_time_base_den,
                    source_start_ticks=source_start_ticks,
                    stream_start_ticks=stream_start_ticks,
                )
                matte_reader = MaskInterchangeReader(
                    frozen_record["manifest"],
                    width=width,
                    height=height,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    timestamps=matte_timestamps,
                    durations=matte_durations,
                    is_vfr=bool(
                        frame_timing is not None and frame_timing.is_vfr),
                    source_time_base=matte_time_base,
                    timestamp_ticks=matte_timestamp_ticks,
                    duration_ticks=matte_duration_ticks,
                    source_time_base_num=matte_time_base_num,
                    source_time_base_den=matte_time_base_den,
                    source_start_ticks=source_start_ticks,
                    stream_start_ticks=stream_start_ticks,
                    mode="replace",
                )
                self.last_frozen_matte = {
                    **frozen_evidence,
                    "requested": True,
                }
                logger.info(
                    "Reusing frozen %s matte (%d frames); skipping OCR, "
                    "tracking, and mask refiners",
                    frozen_record["format"],
                    frozen_record["frame_count"],
                )
            elif self.config.mask_import_path:
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
                    timestamp_ticks=matte_timestamp_ticks,
                    duration_ticks=matte_duration_ticks,
                    source_time_base_num=matte_time_base_num,
                    source_time_base_den=matte_time_base_den,
                    source_start_ticks=source_start_ticks,
                    stream_start_ticks=stream_start_ticks,
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
                if self._source_is_hdr() and getattr(
                        self, "_hdr_repair_ready", False):
                    selective_cap = _open_required_hdr_capture(
                        str(selective_path),
                        input_fps=self.config.input_fps,
                    )
                if selective_cap is None:
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
                if start_frame > 0 and prior_frames == frames_to_process:
                    # A range-relative output starts at source frame zero from
                    # the capture's point of view. Do not add start_frame a
                    # second time when reading it for selective reruns.
                    selective_prior_offset = 0
                elif prior_frames >= end_frame:
                    # A full-source output keeps source frame numbering.
                    selective_prior_offset = start_frame
                else:
                    raise ValueError(
                        "Previous cleaned output does not cover the requested "
                        "source range; selective rerun with a time range "
                        "requires a range-length output"
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
                    input_path, output_path, self.config)
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
                        timing=(
                            frame_timing.checkpoint_metadata(
                                start_frame, end_frame, 0)
                            if frame_timing is not None else None
                        ),
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

            restart_for_derived_outputs = any(
                bool(getattr(self.config, name, False))
                for name in (
                    "export_mask_video",
                    "export_srt",
                    "translation_enabled",
                    "quality_report",
                    "quality_report_sheet",
                )
            )
            if restart_for_derived_outputs and resume_frame_count > 0:
                logger.warning(
                    "Restarting from frame zero so requested sidecars, reports, "
                    "and exports contain a complete timestamp-aligned sequence"
                )
                resume_frame_count = 0
                _seek_capture_to_frame(cap, start_frame)

            if selective_cap is not None:
                _seek_capture_to_frame(
                    selective_cap, selective_prior_offset + resume_frame_count)

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
            if self.config.quality_report:
                self._quality_frame_evidence_dir = Path(temp_dir) / "quality_masks"
                self._quality_frame_evidence_dir.mkdir(
                    parents=True, exist_ok=True,
                )
                self._quality_frame_evidence_write_error = False
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
                        "schema": "vsr.frame_timing.v2",
                        "source": str(input_path),
                        "source_start_seconds": frame_timing.source_start,
                        "source_start_ticks": frame_timing.source_start_ticks,
                        "stream_start_ticks": frame_timing.stream_start_ticks,
                        "source_time_base_num": frame_timing.time_base_num,
                        "source_time_base_den": frame_timing.time_base_den,
                        "source_time_base": {
                            "num": frame_timing.time_base_num,
                            "den": frame_timing.time_base_den,
                        },
                        "source_time_base_seconds": frame_timing.time_base,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "timestamp_ticks": matte_timestamp_ticks,
                        "duration_ticks": matte_duration_ticks,
                        "timestamps_seconds": [
                            frame_timing.frame_time(index, fps)
                            for index in range(start_frame, end_frame)
                        ],
                        "durations_seconds": selected_frame_durations,
                        "timing_anomalies": frame_timing.anomalies,
                    }
                    _write_text_atomic(
                        timing_manifest_path,
                        json.dumps(
                            timing_payload,
                            indent=2,
                            ensure_ascii=True,
                        ) + "\n",
                    )

            timing_checkpoint_metadata = (
                frame_timing.checkpoint_metadata(
                    start_frame, end_frame, resume_frame_count)
                if frame_timing is not None else None
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
                    timing=timing_checkpoint_metadata,
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
            collect_srt_text = bool(
                self.config.export_srt
                or (
                    self.config.translation_enabled
                    and not self.config.translation_srt
                    and not self.config.translation_source_srt
                )
            )
            srt_tracker = (
                tracker
                if collect_srt_text and tracker is not None
                else SubtitleTracker(
                    self.config.kalman_iou_threshold,
                    self.config.kalman_max_age,
                ) if collect_srt_text else None
            )
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
                        timestamp_ticks=matte_timestamp_ticks,
                        duration_ticks=matte_duration_ticks,
                        source_time_base_num=matte_time_base_num,
                        source_time_base_den=matte_time_base_den,
                        source_start_ticks=source_start_ticks,
                        stream_start_ticks=stream_start_ticks,
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
            timed_mask_corrections = has_timed_corrections(
                getattr(self.config, "manual_mask_corrections", None)
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
                timed_mask_corrections=timed_mask_corrections,
                static_fixed_shapes=static_fixed_shapes,
                selective_ranges=selective_ranges,
                reader=reader,
                selective_cap=selective_cap,
                matte_reader=matte_reader,
                frozen_matte=bool(frozen_record),
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
                    timing_metadata=timing_checkpoint_metadata,
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
                srt_tracker=srt_tracker,
                fixed_mask_cache=fixed_mask_cache,
                written_idx=frame_idx,
            )
            # RM-296: a fade-in hold looks FORWARD, so the last `fade_in`
            # frames of a batch cannot be finalised until the next batch's
            # masks exist. Hold them back and re-attach them to the front of
            # the next batch, which always brings at least batch_size frames
            # of lookahead with it.
            fade_in_hold = (
                int(getattr(self.config, "mask_fade_in_frames", 0) or 0)
                if not loop_ctx.frozen_matte else 0
            )
            while True:
                batch = self._decode_and_build_batch(loop_ctx, loop_state)
                decoded = len(batch.frames)
                if loop_state.fade_pending is not None:
                    batch.prepend(loop_state.fade_pending)
                    loop_state.fade_pending = None
                if not batch.frames:
                    break
                at_end = (
                    decoded < loop_ctx.batch_size
                    or loop_ctx.start_frame + loop_state.frame_idx
                    >= loop_ctx.end_frame
                )
                if (
                    fade_in_hold > 0
                    and not at_end
                    and len(batch.frames) > fade_in_hold
                ):
                    loop_state.fade_pending = batch.split_tail(fade_in_hold)
                self._process_batch(loop_ctx, loop_state, batch)
                should_pause = self._pause_requested(loop_ctx)
                if should_pause and loop_state.fade_pending is not None:
                    # A pause has to leave the checkpoint on a decode-batch
                    # boundary. STTN pools a whole batch, so a resume that
                    # regrouped the frames would produce different pixels
                    # from an uninterrupted run. Flushing the held tail
                    # costs it the lookahead it was waiting for -- bounded
                    # by fade_in, and only when a human pauses.
                    tail = loop_state.fade_pending
                    loop_state.fade_pending = None
                    self._process_batch(loop_ctx, loop_state, tail)
                self._checkpoint_after_batch(
                    loop_ctx, loop_state, should_pause=should_pause)
            frame_idx = loop_state.written_idx

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
                    timing=(
                        frame_timing.checkpoint_metadata(
                            start_frame, end_frame, frames_to_process)
                        if frame_timing is not None else None
                    ),
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
                selected_frame_duration_ticks=selected_frame_duration_ticks,
                processed_time_start=processed_time_start,
                processed_time_end=processed_time_end,
                processed_time_start_ticks=processed_time_start_ticks,
                processed_time_end_ticks=processed_time_end_ticks,
                matte_time_base_num=matte_time_base_num,
                matte_time_base_den=matte_time_base_den,
                matte_writer=matte_writer,
            )

            if self.config.quality_report:
                self._quality_final_encode_verified = (
                    self._recompute_final_quality_evidence(
                        input_path,
                        final_output_path,
                        start_frame,
                        end_frame,
                        fps,
                    )
                )
            self._emit_quality_report(
                input_path=input_path,
                final_output_path=final_output_path,
                start_frame=start_frame,
                end_frame=end_frame,
                fps=fps,
            )

            self.last_output_path = final_output_path
            # RM-147: throughput is part of the provenance record -- a "CUDA"
            # run that quietly ran on CPU shows up here as well as in the
            # engine labels.
            self.execution_provenance.frames_processed = int(frames_to_process)
            self.execution_provenance.processing_seconds = round(sum(
                float(value)
                for key, value in self.last_stage_timings.items()
                if key in ("decode", "ocr", "mask", "inpaint", "encode")
            ), 3)
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
        except RequestedStageError as e:
            self._record_requested_stage_failure(e)
            logger.error("Requested video stage failed: %s", e, exc_info=True)
            return False
        except FrozenMatteError as e:
            # RM-153: the frozen matte no longer belongs to this job. Say
            # exactly what moved and ask for a re-freeze; painting approved
            # pixels onto frames they were not approved for, or silently
            # re-deriving a mask the user believes is pinned, are both worse
            # than stopping. Nothing has been decoded at this point.
            self.last_error_message = e.user_message
            self.last_error_reason = f"frozen_matte_{e.reason}"
            self.last_frozen_matte.update({
                "status": "invalid",
                "reason": e.reason,
                "error": e.user_message,
                "needs_revalidation": e.needs_revalidation,
            })
            logger.error(
                "Frozen matte rejected (%s): %s", e.reason, e.user_message)
            return False
        except MediaWriteError as e:
            # RM-139: a truncated intermediate or an unwritten frame must never
            # look like a completed job. The finally block below releases the
            # writer and the temp/partial output is never promoted.
            self.last_error_message = e.user_message
            self.last_error_reason = e.reason
            logger.error(
                "Output writer failed (%s): %s",
                e.reason,
                e.detail or e.user_message,
            )
            return False
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
            # RM-283: a donor-video reference holds its own capture open for
            # the whole job; leaving it open would keep a handle on the donor
            # file after the run ends.
            try:
                self._release_clean_references()
            except Exception:
                logger.warning(
                    "Clean reference release failed", exc_info=True)
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            # RM-27: Whisper audio temp dir is created lazily inside the
            # main try block; clean it up here only if it was set.
            try:
                if whisper_audio_dir and os.path.exists(whisper_audio_dir):
                    shutil.rmtree(whisper_audio_dir, ignore_errors=True)
            except Exception:
                logger.warning("Whisper temp cleanup failed", exc_info=True)

    # -- Encode, mux, and audio-merge methods live in _encode_mixin.py --


if __name__ == "__main__":
    # `python -m backend.processor` executes this file as `__main__`;
    # cli.main() then imports backend.processor, which would otherwise
    # re-execute the whole module and create a second, distinct set of
    # classes (two InpaintMode types, double registry registration).
    # Alias the module first so that import resolves to this instance.
    sys.modules.setdefault("backend.processor", sys.modules[__name__])
    from backend.cli import main as _cli_main
    _cli_main()
