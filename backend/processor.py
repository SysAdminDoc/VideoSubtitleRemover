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
import subprocess
import traceback
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Tuple, List, Callable

logger = logging.getLogger(__name__)

# RFP-L-1 re-exports. Anything that used to be defined in this module
# but moved during the split is re-imported here so existing callers
# (`from backend.processor import _open_capture`) keep working.
from backend.io import (
    MediaInputError as MediaInputError,
    MediaWriteError as MediaWriteError,
    SubtitleStreamInfo as SubtitleStreamInfo,
    _validate_video_input_file as _validate_video_input_file,
    _video_capture_open_error as _video_capture_open_error,
    _invalid_video_dimensions_error as _invalid_video_dimensions_error,
    _video_decode_error as _video_decode_error,
    _probe_codec_for_log as _probe_codec_for_log,
    _probe_audio_stream_count as _probe_audio_stream_count,
    _probe_subtitle_streams as _probe_subtitle_streams,
    _probe_duration_seconds as _probe_duration_seconds,
    _ffmpeg_subprocess_timeout as _ffmpeg_subprocess_timeout,
    _probe_keyframe_indices as _probe_keyframe_indices,
    _probe_is_interlaced as _probe_is_interlaced,
    _deinterlace_to_temp as _deinterlace_to_temp,
    _ensure_output_parent as _ensure_output_parent,
    _path_key as _path_key,
    _choose_available_output_path as _choose_available_output_path,
    _write_text_atomic as _write_text_atomic,
    _allocate_temp_output_path,
    _cleanup_temp_output as _cleanup_temp_output,
    _promote_temp_output as _promote_temp_output,
    _copy_file_atomic as _copy_file_atomic,
    validate_video_output as validate_video_output,
    VideoFrameTiming as VideoFrameTiming,
    _probe_video_frame_timing as _probe_video_frame_timing,
    _FrameSequenceCapture as _FrameSequenceCapture,
    _open_capture,
    _open_bgr48_capture as _open_bgr48_capture,
    _PrefetchReader as _PrefetchReader,
    _LosslessIntermediateWriter as _LosslessIntermediateWriter,
    _FrameSequenceWriter as _FrameSequenceWriter,
    _run_subprocess_checked,
    _terminate_subprocess,
)
from backend.encoder import _detect_hw_encoder, probe_d3d12_encoder
from backend.execution_provenance import (
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
    SELECTIVE_RERUN_SCHEMA as SELECTIVE_RERUN_SCHEMA,
    apply_mask_corrections,
    make_review_span as make_review_span,
    merge_frame_ranges as merge_frame_ranges,
    merge_review_spans as merge_review_spans,
    has_timed_corrections as has_timed_corrections,
)
from backend.frozen_matte import (
    FrozenMatteError as FrozenMatteError,
    normalize_frozen_matte,
    validate_frozen_matte as validate_frozen_matte,
)
from backend.matte_interchange import (
    MaskInterchangeReader as MaskInterchangeReader,
    MaskInterchangeWriter as MaskInterchangeWriter,
    mask_interchange_paths as mask_interchange_paths,
)
from backend.resume_checkpoint import (
    ProcessingPaused as ProcessingPaused,
    _checkpoint_key as _checkpoint_key,
    _checkpoint_is_done as _checkpoint_is_done,
    _checkpoint_mark_done as _checkpoint_mark_done,
    _default_checkpoint_dir as _default_checkpoint_dir,
    cleanup_pause_checkpoint as cleanup_pause_checkpoint,
    config_fingerprint as config_fingerprint,
    load_pause_checkpoint as load_pause_checkpoint,
    pause_frame_dir as pause_frame_dir,
    write_pause_checkpoint as write_pause_checkpoint,
)
from backend.safe_image import (
    safe_imread as safe_imread,
    safe_imwrite as safe_imwrite,
)
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
    SubtitleTracker as SubtitleTracker,
    _group_horizontal_line as _group_horizontal_line,
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
    STTNInpainter,
    LAMAInpainter,
    ProPainterInpainter,
    AutoInpainter,
    free_inference_memory as free_inference_memory,
    _feather_blend as _feather_blend,
    _edge_ring_color_correct as _edge_ring_color_correct,
    _detect_scene_cuts as _detect_scene_cuts,
    _detect_scene_cuts_pyscenedetect as _detect_scene_cuts_pyscenedetect,
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


# RM-349: the frame-loop window and the dataclasses its stages hand to each
# other now live in their own module, so the stages themselves can move out of
# this file without importing it back. Re-imported under the same names,
# because callers and tests reach for `backend.processor._FrameLoopContext`.
from backend._frame_loop_mixin import _FrameLoopMixin  # noqa: E402
from backend._inpaint_mixin import _InpaintMixin  # noqa: E402
from backend._pipeline_mixin import (  # noqa: E402
    _PipelineMixin,
    _open_required_hdr_capture as _open_required_hdr_capture,
)
from backend._frame_loop_types import (  # noqa: E402
    _FrameBatch as _FrameBatch,
    _FrameLoopCheckpoint as _FrameLoopCheckpoint,
    _FrameLoopContext as _FrameLoopContext,
    _FrameLoopState as _FrameLoopState,
    _FrameRange as _FrameRange,
    _frame_seconds as _frame_seconds,
    _resolve_frame_range as _resolve_frame_range,
    _seek_capture_to_frame as _seek_capture_to_frame,
    _spans_from_segments as _spans_from_segments,
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


from backend._encode_mixin import _EncodeMixin
from backend._quality_mixin import _QualityMixin
from backend._finalize_mixin import _FinalizeMixin
from backend._srt_mixin import _SrtMixin
from backend._clean_ref_mixin import _CleanRefMixin


class SubtitleRemover(
    _EncodeMixin, _QualityMixin, _FinalizeMixin, _SrtMixin, _CleanRefMixin,
    _FrameLoopMixin, _InpaintMixin, _PipelineMixin,
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
        model_provenance = identity.get("modelProvenance")
        if isinstance(model_provenance, dict):
            stage.model_provenance = dict(model_provenance)
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
