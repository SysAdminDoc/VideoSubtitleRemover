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
import tempfile
import traceback
from pathlib import Path
from typing import Any, Optional, Tuple, List, Callable

logger = logging.getLogger(__name__)

# RFP-L-1 re-exports. Anything that used to be defined in this module
# but moved during the split is re-imported here so existing callers
# (`from backend.processor import _open_capture`) keep working.
from backend.io import (
    MediaInputError,
    SubtitleStreamInfo,
    _validate_video_input_file,
    _video_capture_open_error,
    _invalid_video_dimensions_error,
    _video_decode_error,
    _probe_codec_for_log,
    _probe_audio_stream_count,
    _probe_subtitle_streams,
    _probe_duration_seconds,
    _ffmpeg_subprocess_timeout,
    _probe_keyframe_indices,
    _probe_is_interlaced,
    _deinterlace_to_temp,
    _ensure_output_parent,
    _path_key,
    _choose_available_output_path,
    _write_text_atomic,
    _allocate_temp_output_path,
    _cleanup_temp_output,
    _promote_temp_output,
    _copy_file_atomic,
    _FrameSequenceCapture,
    _open_capture,
    _PrefetchReader,
    _LosslessIntermediateWriter,
    _FrameSequenceWriter,
    _run_subprocess_checked,
    _terminate_subprocess,
)
from backend.encoder import _detect_hw_encoder
from backend.quality import (
    _ssim,
    compute_vmaf,
    compute_extended_metrics,
    temporal_consistency_score,
    residual_text_score,
    temporal_flicker_score,
)
from backend.quality_gate import evaluate_quality_gate
from backend.safe_image import safe_imread
from backend.tracking import (
    _KalmanBox,
    _box_from_state,
    _iou,
    SubtitleTracker,
    _group_horizontal_line,
    _phash,
    _phash_distance,
)
from backend.detection import SubtitleDetector, _surya_allowed
from backend.inpainters import (
    BaseInpainter,
    STTNInpainter,
    LAMAInpainter,
    ProPainterInpainter,
    AutoInpainter,
    _cv2_inpaint,
    _feather_blend,
    _edge_ring_color_correct,
    _expand_mask_by_color,
    _detect_scene_cuts,
    _detect_scene_cuts_pyscenedetect,
    _farneback_winsize,
    _warp_to_reference,
    _warp_mask_to_reference,
    _tbe_single_segment,
    _temporal_background_expose,
)
# CLI helpers moved to backend.cli during RFP-L-1; re-export so existing
# callers (e.g. `processor._load_json_config`,
# `processor._apply_auto_band_override`) keep resolving.
from backend.cli import (
    _default_checkpoint_dir,
    _checkpoint_key,
    _checkpoint_is_done,
    _checkpoint_mark_done,
    _load_json_config,
    _apply_auto_band_override,
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
    InpaintMode,
    RegisteredMode,
    ProcessingConfig,
    _MODE_ALIASES,
    _coerce_bool,
    _coerce_int,
    _coerce_float,
    _coerce_text,
    _coerce_rect,
    _coerce_rect_list,
    _coerce_backend_mode,
    _coerce_backend_device,
    is_known_backend_mode,
    normalize_processing_config,
)


def _available_host_ram_gb() -> Optional[float]:
    """Best-effort available physical memory in GB; None when no probe
    works. Used to keep the adaptive TBE batch within host RAM."""
    try:
        import psutil  # type: ignore
        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        pass
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
        pass
    return None


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


class SubtitleRemover:
    """Coordinates detection and inpainting to remove subtitles from videos/images."""

    def __init__(self, config: ProcessingConfig = None):
        self.config = normalize_processing_config(config or ProcessingConfig())
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
        # Actual user-visible output path for the last run. This may differ
        # from the requested path when FFmpeg cannot encode the requested
        # container and the lossless intermediate is salvaged as .mkv.
        self.last_output_path: Optional[str] = None
        self.last_error_message: Optional[str] = None
        self.last_error_reason: Optional[str] = None
        # B-3: union-mask bbox accumulated while processing. The quality
        # report metric (PSNR/SSIM) used to be measured over the whole
        # frame, so the unchanged 80-95% of pixels dominated the score and
        # an awful inpaint could still report 'Good'. We track the bbox of
        # the union mask and the metric runs against that ROI only.
        self._quality_mask_bbox: Optional[Tuple[int, int, int, int]] = None
        # RM-73 partial: source color signalling, populated lazily inside
        # process_video once we know the input path. Used by _get_encode_args
        # to preserve HDR / BT.2020 tagging on the output.
        self._color_metadata = None
        self._active_writer = None
        self._active_subprocess: Optional[subprocess.Popen] = None
        self._teardown_requested = False

        if self.config.use_hw_encode:
            self._hw_encoder = _detect_hw_encoder(self.config.output_codec)

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
        """RFP-L-2: dispatch through the plugin registry. A new backend
        becomes available as soon as it registers itself; we no longer
        need to edit this dispatch to add an InpaintMode value (though
        the enum still gates the GUI / CLI for safety)."""
        try:
            builder = _inpainter_registry.resolve(self.config.mode.value)
        except KeyError:
            logger.warning(
                f"No inpainter registered for {self.config.mode.value!r}; "
                f"falling back to STTN"
            )
            builder = _inpainter_registry.resolve("sttn")
        return builder(self.config.device, self.config)

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
        if text:
            self._srt_entries.append((frame_idx, text))

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
                pass
            try:
                cap_out.release()
            except Exception:
                pass
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
            residual_mean_score = (
                float(np.mean(residual_scores)) if residual_scores else None
            )
            segment_duration = max(0.1, min(30.0, span / max(fps, 1.0)))
            segment_start = start_frame / max(fps, 1.0)
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
                'lpips': extended.get('lpips'),
                'dists': extended.get('dists'),
                'samples': len(psnrs),
                'tag': tag,
                'sheet': sheet_path,
            }
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

    def _write_srt(self, path: str, fps: float, offset_frames: int = 0):
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
            if text == cur_text and frame_idx - cur_end <= gap_tol:
                cur_end = frame_idx
            else:
                cues.append((cur_start, cur_end, cur_text))
                cur_start, cur_end, cur_text = frame_idx, frame_idx, text
        if cur_text is not None:
            cues.append((cur_start, cur_end, cur_text))

        try:
            payload = []
            for i, (s, e, txt) in enumerate(cues, 1):
                t_start = (s + offset_frames) / fps
                t_end = (e + offset_frames + 1) / fps
                payload.append(f"{i}\n{ts(t_start)} --> {ts(t_end)}\n{txt}\n\n")
            _write_text_atomic(Path(path), "".join(payload))
            logger.info(f"SRT written: {path} ({len(cues)} cues)")
        except Exception as exc:
            logger.warning(f"SRT write failed: {exc}", exc_info=True)

    def _fixed_region_boxes(self) -> Optional[List[Tuple[int, int, int, int]]]:
        """Return explicit mask rects from config, preferring the multi-region
        list if set, falling back to the single-rect legacy field."""
        if self.config.subtitle_areas:
            return list(self.config.subtitle_areas)
        if self.config.subtitle_area:
            return [self.config.subtitle_area]
        return None

    def process_image(self, input_path: str, output_path: str) -> bool:
        self._teardown_requested = False
        self.last_output_path = None
        try:
            _ensure_output_parent(output_path)
            self._report_progress(0.1, "Loading image...")
            image = safe_imread(input_path)
            if image is None:
                raise ValueError(f"Could not load image: {input_path}")

            self._report_progress(0.3, "Detecting text regions...")
            fixed = self._fixed_region_boxes()
            confidences = None
            if fixed:
                boxes = fixed
            elif self.config.confidence_weighted_dilation:
                results = self.detector.detect_with_confidence(
                    image, self.config.detection_threshold)
                boxes = [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in results]
                confidences = [c for _, _, _, _, c in results]
            else:
                boxes = self.detector.detect(image, self.config.detection_threshold)

            if not boxes:
                logger.info("No text detected, copying original")
                _copy_file_atomic(input_path, output_path)
                self.last_output_path = output_path
                self.last_error_message = None
                self.last_error_reason = None
                self._report_progress(1.0, "Complete (no text found)")
                return True

            self._report_progress(0.5, f"Removing {len(boxes)} text regions...")
            mask = self._create_mask(image.shape, boxes, frame=image,
                                     confidences=confidences)
            [result] = self.inpainter.inpaint([image], [mask])

            self._report_progress(0.9, "Saving result...")
            ext = Path(output_path).suffix.lower()
            temp_output = _allocate_temp_output_path(output_path)
            try:
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

    def _make_temp_dir(self) -> str:
        work = getattr(self.config, "work_directory", "")
        if work and os.path.isdir(work):
            return tempfile.mkdtemp(prefix="vsr_", dir=work)
        return tempfile.mkdtemp(prefix="vsr_")

    def process_video(self, input_path: str, output_path: str) -> bool:
        self._teardown_requested = False
        self.last_output_path = None
        self._srt_entries = []
        self._quality_mask_bbox = None
        temp_dir = None
        cap = None
        reader = None
        writer = None
        mask_writer = None
        temp_mask_path = None
        whisper_audio_dir = None
        self.last_error_message = None
        self.last_error_reason = None
        try:
            _ensure_output_parent(output_path)
            self._report_progress(0.0, "Opening video...")
            _validate_video_input_file(input_path)

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

            # RM-73 partial: probe source color signalling once so HDR
            # / BT.2020 tags can be preserved on the output. The pixel
            # pipeline is still 8-bit BGR; the tag passthrough at least
            # keeps downstream players from accidentally tone-mapping a
            # tone-mapped output.
            # RM-74: explicit AV1 / VP9 ingest validation. cv2 4.12+
            # decodes both via the ffmpeg backend on every supported
            # platform, but the codec field varies enough in the wild
            # that we log it explicitly. Lets the user reproduce a
            # decode failure with the same ffmpeg invocation when
            # something goes wrong.
            if not Path(input_path).is_dir():
                try:
                    from backend.hdr import probe_color_metadata as _probe
                    _meta = _probe(input_path)  # only used for the codec probe
                    _codec_line = _probe_codec_for_log(input_path)
                    if _codec_line:
                        logger.info(f"Source codec: {_codec_line}")
                except Exception:
                    logger.warning("Source codec probe failed", exc_info=True)

            if (self.config.preserve_color_metadata
                    and not Path(input_path).is_dir()):
                try:
                    from backend.hdr import probe_color_metadata
                    meta = probe_color_metadata(input_path)
                    if meta is not None:
                        self._color_metadata = meta
                        if meta.is_hdr:
                            logger.info(
                                f"HDR source detected: {meta.label} -- "
                                f"current pipeline is 8-bit BGR; output will "
                                f"carry the source color tags but pixels are "
                                f"tone-mapped to SDR."
                            )
                        else:
                            logger.info(f"Color signalling: {meta.label}")
                except Exception as exc:
                    logger.warning(
                        f"Color-metadata probe failed: {exc}",
                        exc_info=True,
                    )

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

            if width == 0 or height == 0:
                raise _invalid_video_dimensions_error(input_path, width, height)

            # Time range support. Guard against NaN / inf / negative values
            # coming from a malformed preset or CLI overlay -- never let them
            # reach the ffmpeg command line or frame-count math.
            def _sane_seconds(value: Any) -> float:
                try:
                    v = float(value)
                except (TypeError, ValueError):
                    return 0.0
                if not np.isfinite(v) or v < 0.0:
                    return 0.0
                return v

            time_start_s = _sane_seconds(self.config.time_start)
            time_end_s = _sane_seconds(self.config.time_end)
            start_frame = 0
            end_frame = total_frames
            if time_start_s > 0:
                start_frame = max(0, min(total_frames - 1, int(time_start_s * fps)))
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            if time_end_s > 0:
                end_frame = max(0, min(total_frames, int(time_end_s * fps)))
            if end_frame <= start_frame:
                raise ValueError(
                    f"Invalid time range: end ({time_end_s}s) "
                    f"must be after start ({time_start_s}s)")
            frames_to_process = end_frame - start_frame

            if start_frame > 0 or end_frame < total_frames:
                logger.info(f"Video: {width}x{height} @ {fps:.1f}fps, "
                           f"frames {start_frame}-{end_frame} of {total_frames}")
            else:
                logger.info(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

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
            if use_frame_output:
                frame_out_dir = output_path
                if not frame_out_dir.endswith(os.sep):
                    frame_out_dir = str(Path(output_path).with_suffix(""))
                writer = _FrameSequenceWriter(frame_out_dir)
                temp_video = None
            else:
                temp_video_target = os.path.join(temp_dir, "temp_video.mkv")
                writer = _LosslessIntermediateWriter(
                    temp_video_target, width, height, fps
                )
                self._active_writer = writer
                temp_video = writer.path
                if not writer.isOpened():
                    raise ValueError(
                        f"Could not create video writer for: {temp_video}"
                    )

            frame_idx = 0
            batch_size = self.config.sttn_max_load_num
            frame_skip = self.config.detection_frame_skip

            # Decoupled prefetch: wrap the capture in a worker that fills
            # a bounded frame queue while the main thread runs detection +
            # inpainting. cv2.VideoCapture must NOT be touched directly
            # (.set / .get / .read) after this point -- the worker owns it.
            # Seek + metadata reads above happen *before* the wrap, so this
            # is safe; cleanup goes through `reader.release()`.
            if self.config.prefetch_decode:
                qsize = self.config.prefetch_queue_size or max(8, batch_size * 2)
                reader = _PrefetchReader(cap, max_frames=frames_to_process,
                                          queue_size=qsize)
                logger.info(f"Prefetch decode on (queue={qsize})")
            else:
                reader = cap
            last_mask = None  # cached mask for frame-skip optimization
            fixed_mask = None  # cached mask for skip_detection mode

            # RM-27 Whisper fallback: pre-compute frame spans where
            # Whisper detected speech. When OCR returns no boxes for a
            # frame inside one of these spans we apply a default
            # bottom-band mask so the subtitle band still gets
            # inpainted. Done once per file so we don't pay model load
            # for every batch.
            whisper_spans: List[Tuple[int, int]] = []
            whisper_audio_dir: Optional[str] = None
            if self.config.whisper_fallback and not Path(input_path).is_dir():
                try:
                    import tempfile as _tmp_mod
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
                            whisper_spans = _wf.segments_to_frame_spans(
                                segments, fps
                            )
                            logger.info(
                                f"FFmpeg Whisper fallback active: "
                                f"{len(whisper_spans)} speech spans"
                            )
                    elif _wf.is_available():
                        whisper_audio_dir = _tmp_mod.mkdtemp(prefix="vsr_whisper_")
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
                                whisper_spans = _wf.segments_to_frame_spans(
                                    segments, fps
                                )
                                logger.info(
                                    f"Whisper fallback active: "
                                    f"{len(whisper_spans)} speech spans"
                                )
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
            last_hash_frame_idx = -1

            # Mask video writer (optional debug artifact)
            mask_path = None
            if self.config.export_mask_video:
                mask_path = str(Path(output_path).with_suffix('')) + '.mask.mp4'
                temp_mask_path = _allocate_temp_output_path(mask_path)
                mask_writer = cv2.VideoWriter(
                    str(temp_mask_path), cv2.VideoWriter_fourcc(*'mp4v'),
                    fps, (width, height), isColor=False)
                if not mask_writer.isOpened():
                    logger.warning(f"Could not open mask video writer: {mask_path}")
                    mask_writer = None

            fixed_boxes = self._fixed_region_boxes()

            while True:
                frames = []
                masks = []

                for _ in range(batch_size):
                    if start_frame + frame_idx >= end_frame:
                        break
                    ret, frame = reader.read()
                    if not ret:
                        break

                    if self.config.sttn_skip_detection and fixed_boxes:
                        # Fixed region: create mask once and reuse for all frames
                        if fixed_mask is None:
                            fixed_mask = self._create_mask(frame.shape, fixed_boxes)
                        frames.append(frame)
                        masks.append(fixed_mask)
                        frame_idx += 1
                        continue

                    # Perceptual-hash adaptive mask reuse: skip detection when
                    # the frame is near-identical to the last detected one.
                    reuse_by_phash = False
                    cur_hash = None  # may be set below; reused to avoid double-compute
                    if (self.config.phash_skip_enable and last_mask is not None
                            and last_hash is not None):
                        cur_hash = _phash(frame)
                        if _phash_distance(cur_hash, last_hash) <= self.config.phash_skip_distance:
                            reuse_by_phash = True

                    # Keyframe-driven detection: if we have a keyframe index
                    # set, OCR only at I-frames, reuse last mask between.
                    reuse_by_keyframe = False
                    if keyframe_set and last_mask is not None:
                        absolute_idx = start_frame + frame_idx
                        if absolute_idx not in keyframe_set:
                            reuse_by_keyframe = True

                    if reuse_by_phash or reuse_by_keyframe:
                        frames.append(frame)
                        masks.append(last_mask)
                        frame_idx += 1
                        continue
                    elif frame_skip > 0 and last_mask is not None and frame_idx % (frame_skip + 1) != 0:
                        # Reuse cached mask for intermediate frames
                        frames.append(frame)
                        masks.append(last_mask)
                        frame_idx += 1
                        continue
                    else:
                        # RM-33: optionally denoise the detection-side
                        # frame copy before OCR. Output pixels stay
                        # unchanged because the inpainter still runs
                        # against `frame`, not `det_frame`.
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
                        if self.config.confidence_weighted_dilation:
                            det_results = self.detector.detect_with_confidence(
                                det_frame, self.config.detection_threshold)
                            detected_boxes = [
                                (x1, y1, x2, y2)
                                for x1, y1, x2, y2, _ in det_results
                            ]
                            det_confs = [c for _, _, _, _, c in det_results]
                        else:
                            detected_boxes = self.detector.detect(
                                det_frame, self.config.detection_threshold)
                        # Karaoke grouping: fuse per-syllable boxes on the
                        # same line before tracking so Kalman sees one
                        # composite per line, not one per syllable.
                        if self.config.karaoke_grouping and detected_boxes:
                            detected_boxes = _group_horizontal_line(
                                detected_boxes,
                                x_gap_px=self.config.karaoke_x_gap_px,
                                y_overlap_ratio=self.config.karaoke_y_overlap,
                            )
                            det_confs = None
                        # Smooth jitter + fill single-frame misses via Kalman
                        if tracker is not None:
                            smoothed = tracker.update(list(detected_boxes))
                            det_confs = None
                        else:
                            smoothed = list(detected_boxes)
                        # Chyron / subtitle filter: when either remove flag is
                        # off we drop the matching tracks before mask creation.
                        # No-op when both are True (default v3.12 behaviour).
                        if (tracker is not None
                                and (not self.config.remove_chyrons
                                     or not self.config.remove_subtitles)):
                            cats = tracker.categorize(self.config.chyron_min_hits)
                            smoothed = [
                                b for b, c in zip(smoothed, cats)
                                if (c == "chyron" and self.config.remove_chyrons)
                                or (c == "subtitle" and self.config.remove_subtitles)
                            ]
                            det_confs = None
                        # If fixed boxes are set without skip_detection, union them
                        # with per-frame detections so users can pin a region AND
                        # still clean incidental text elsewhere.
                        if fixed_boxes:
                            boxes = list(fixed_boxes) + smoothed
                            det_confs = None
                        else:
                            boxes = smoothed
                        if self.config.export_srt:
                            self._collect_srt_entry(frame, frame_idx, detected_boxes)

                    # RM-27: when OCR returned no boxes for this frame
                    # AND Whisper found speech in this timecode, mask a
                    # default bottom band so the inpaint pass still
                    # cleans the subtitle position. Catches frames the
                    # OCR cascade missed but the audio confirms have
                    # dialogue.
                    if (not boxes and whisper_spans
                            and self.config.whisper_fallback):
                        absolute = start_frame + frame_idx
                        for span_s, span_e in whisper_spans:
                            if span_s <= absolute < span_e:
                                h_full, w_full = frame.shape[:2]
                                band_top = int(h_full * 0.80)
                                boxes = [(int(w_full * 0.05), band_top,
                                          int(w_full * 0.95), h_full - 4)]
                                break

                    mask = self._create_mask(frame.shape, boxes, frame=frame,
                                             confidences=det_confs)
                    # B-3: accumulate the union-mask bbox for the quality
                    # report ROI. We track the bbox (not the per-frame mask
                    # stack) to keep memory flat; the bbox is enough to
                    # crop input and output for the metric pass.
                    if self.config.quality_report:
                        self._accumulate_quality_bbox(mask)
                    # Colour-tuned expansion -- grow the mask to match the
                    # dominant text colour inside each detected box.
                    if self.config.colour_tune_enable and boxes:
                        mask = _expand_mask_by_color(
                            frame, mask, boxes,
                            tolerance=self.config.colour_tune_tolerance,
                            padding=4,
                        )
                    last_mask = mask
                    if self.config.phash_skip_enable:
                        # Reuse the hash computed above for the skip-check if
                        # available; otherwise compute it now for the first time.
                        last_hash = cur_hash if cur_hash is not None else _phash(frame)
                        last_hash_frame_idx = frame_idx
                    frames.append(frame)
                    masks.append(mask)
                    frame_idx += 1

                if not frames:
                    break

                progress = min(0.9, frame_idx / max(1, frames_to_process) * 0.8 + 0.1)
                self._report_progress(progress, f"Processing frame {frame_idx}/{frames_to_process}...")

                results = self.inpainter.inpaint(frames, masks)
                stride = max(1, self.live_preview_stride)
                for offset, result in enumerate(results):
                    writer.write(result)
                    if (self.on_preview_frame is not None and
                            (frame_idx - len(results) + offset) % stride == 0):
                        try:
                            self.on_preview_frame(
                                result,
                                frame_idx - len(results) + offset + 1,
                                frames_to_process)
                        except Exception as exc:
                            # Never let a flaky preview hook break processing,
                            # but leave a breadcrumb so a broken UI is debuggable.
                            logger.warning(
                                f"on_preview_frame hook raised: {exc}",
                                exc_info=True,
                            )
                if mask_writer is not None:
                    for m in masks:
                        mask_writer.write(m)

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
            writer.release()
            writer = None
            if mask_writer is not None:
                mask_writer.release()
                mask_writer = None

            self._report_progress(0.9, "Merging audio...")
            if use_frame_output:
                logger.info(
                    f"Frame-sequence output written to {frame_out_dir}"
                )
                final_output_path = frame_out_dir
            else:
                final_output_path = output_path
                is_frame_sequence_input = Path(input_path).is_dir()
                if self.config.preserve_audio and not is_frame_sequence_input:
                    final_output_path = self._merge_audio(
                        input_path, temp_video, output_path)
                else:
                    final_output_path = self._reencode_or_copy(
                        temp_video, output_path)
            if mask_writer is not None and mask_path and temp_mask_path:
                _promote_temp_output(temp_mask_path, mask_path)
                temp_mask_path = None
                logger.info(f"Mask video written: {mask_path}")

            if self.config.export_srt and self._srt_entries:
                srt_path = str(Path(final_output_path).with_suffix('.srt'))
                self._write_srt(srt_path, fps, start_frame)

            # RM-78 / RM-80: optional post-restore passes (Real-ESRGAN
            # upscale, film-grain re-synthesis). Run after the main mux
            # so the user-visible output is the post-processed file;
            # each adapter degrades gracefully when its dep is missing.
            self._run_post_restore_passes(final_output_path, temp_dir)

            # RM-76: optional NLE round-trip sidecar (EDL / FCPXML).
            self._write_nle_sidecar(input_path, final_output_path,
                                     start_frame, end_frame, fps)

            # Quality report: PSNR/SSIM across a sample of unmasked regions
            if self.config.quality_report:
                try:
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

            self.last_output_path = final_output_path
            self.last_error_message = None
            self.last_error_reason = None
            self._report_progress(1.0, "Complete!")
            return True

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
            if mask_writer is not None:
                try:
                    mask_writer.release()
                except Exception:
                    logger.warning("Mask writer release failed", exc_info=True)
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
            _cleanup_temp_output(temp_mask_path)
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            # RM-27: Whisper audio temp dir is created lazily inside the
            # main try block; clean it up here only if it was set.
            try:
                if whisper_audio_dir and os.path.exists(whisper_audio_dir):
                    shutil.rmtree(whisper_audio_dir, ignore_errors=True)
            except Exception:
                logger.warning("Whisper temp cleanup failed", exc_info=True)

    def _write_nle_sidecar(self, input_path: str, output_path: str,
                             start_frame: int, end_frame: int,
                             fps: float) -> None:
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
            start_s = max(0.0, start_frame / fps)
            end_s = max(start_s + 1.0 / fps, end_frame / fps)
            base = str(Path(output_path).with_suffix(""))
            if mode == "edl":
                path = nle_sidecar.write_edl(
                    base + ".edl", input_path, output_path,
                    fps, start_s, end_s,
                )
            else:
                path = nle_sidecar.write_fcpxml(
                    base + ".fcpxml", input_path, output_path,
                    fps, start_s, end_s,
                )
            logger.info(f"NLE {mode.upper()} sidecar written: {path}")
        except Exception as exc:
            logger.warning(f"NLE sidecar write failed: {exc}", exc_info=True)

    def _run_post_restore_passes(self, output_path: str, temp_dir: str) -> None:
        """RM-78 / RM-80: run optional post-restore passes against the
        finalised output in place. Each adapter is a no-op when its
        dep is missing; the original output is preserved on every
        failure path so users always have a result.
        """
        if self.config.upscale_factor in (2, 3, 4):
            try:
                from backend.post_restore import realesrgan_upscale
                upscaled = os.path.join(temp_dir, "upscaled.mp4")
                produced = realesrgan_upscale(
                    output_path, upscaled,
                    scale=int(self.config.upscale_factor),
                )
                if produced and Path(produced).is_file():
                    _promote_temp_output(produced, output_path)
                    logger.info(
                        f"Real-ESRGAN x{self.config.upscale_factor} pass complete"
                    )
            except Exception as exc:
                logger.warning(f"Real-ESRGAN pass failed: {exc}", exc_info=True)
        if self.config.swinir_restore:
            try:
                from backend.post_restore import swinir_restore
                restored = os.path.join(temp_dir, "swinir.mp4")
                produced = swinir_restore(output_path, restored)
                if produced and Path(produced).is_file():
                    _promote_temp_output(produced, output_path)
                    logger.info("SwinIR restoration pass complete")
            except Exception as exc:
                logger.warning(f"SwinIR pass failed: {exc}", exc_info=True)
        if self.config.seedvr2_restore:
            try:
                from backend.post_restore import seedvr2_restore
                restored = os.path.join(temp_dir, "seedvr2.mp4")
                produced = seedvr2_restore(output_path, restored)
                if produced and Path(produced).is_file():
                    _promote_temp_output(produced, output_path)
                    logger.info("SeedVR2 restoration pass complete")
            except Exception as exc:
                logger.warning(f"SeedVR2 pass failed: {exc}", exc_info=True)
        if self.config.film_grain_strength > 0.0:
            try:
                from backend.post_restore import add_film_grain
                grain_out = os.path.join(temp_dir, "grainy.mp4")
                produced = add_film_grain(
                    output_path, grain_out,
                    strength=self.config.film_grain_strength,
                )
                if produced and Path(produced).is_file():
                    _promote_temp_output(produced, output_path)
                    logger.info(
                        f"Film-grain pass complete "
                        f"(strength={self.config.film_grain_strength:.3f})"
                    )
            except Exception as exc:
                logger.warning(f"Film-grain pass failed: {exc}", exc_info=True)
        if self.config.restyle_subtitle:
            try:
                from backend.post_restore import burn_subtitles
                restyle_out = os.path.join(temp_dir, "restyled.mp4")
                produced = burn_subtitles(
                    output_path, restyle_out,
                    subtitle_path=self.config.restyle_subtitle,
                    style_override=self.config.restyle_style,
                )
                if produced and Path(produced).is_file():
                    _promote_temp_output(produced, output_path)
                    logger.info("Restyle subtitle burn pass complete")
            except Exception as exc:
                logger.warning(f"Restyle pass failed: {exc}", exc_info=True)
        if self.config.watermark_image:
            try:
                from backend.post_restore import burn_watermark
                wm_out = os.path.join(temp_dir, "watermarked.mp4")
                produced = burn_watermark(
                    output_path, wm_out,
                    watermark_path=self.config.watermark_image,
                    position=self.config.watermark_position,
                    opacity=self.config.watermark_opacity,
                    margin=self.config.watermark_margin,
                )
                if produced and Path(produced).is_file():
                    _promote_temp_output(produced, output_path)
                    logger.info("Watermark burn pass complete")
            except Exception as exc:
                logger.warning(f"Watermark burn failed: {exc}", exc_info=True)

    def _get_encode_args(self) -> List[str]:
        """Return FFmpeg video encoder arguments, preferring hardware encoding."""
        if self._hw_encoder and self.config.use_hw_encode:
            if 'nvenc' in self._hw_encoder:
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
            return base + self._hdr_encode_args()
        # F-8: software fallback honours the chosen output codec.
        codec = self.config.output_codec
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
        return base + self._hdr_encode_args()

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

    def _reencode_or_copy(self, source: str, output: str) -> str:
        """Re-encode with preferred encoder, or salvage the intermediate
        if FFmpeg is unavailable or keeps failing."""
        temp_output = _allocate_temp_output_path(output)
        try:
            _ensure_output_parent(output)
            cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-nostats', '-i', source]
            cmd += self._get_encode_args()
            cmd += ['-an', str(temp_output)]
            timeout = _ffmpeg_subprocess_timeout(_probe_duration_seconds(source))
            self._run_checked_ffmpeg(cmd, timeout)
            _promote_temp_output(temp_output, output)
            return output
        except subprocess.CalledProcessError as e:
            if self._hw_encoder:
                logger.warning(f"HW encoder failed, retrying with libx264: {e}")
                self._hw_encoder = None
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

    def _merge_audio(self, original: str, processed: str, output: str) -> str:
        temp_output = _allocate_temp_output_path(output)
        try:
            _ensure_output_parent(output)
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-nostats',
                '-i', processed,
            ]
            # When a time range was used, seek audio to match the processed segment
            if self.config.time_start > 0:
                cmd += ['-ss', str(self.config.time_start)]
            cmd += ['-i', original]
            if self.config.time_end > 0 and self.config.time_start > 0:
                duration = self.config.time_end - self.config.time_start
                cmd += ['-t', str(duration)]
            elif self.config.time_end > 0:
                cmd += ['-t', str(self.config.time_end)]
            cmd += self._get_encode_args()
            cmd += [
                '-c:a', 'aac',
                '-map', '0:v:0',
            ]
            target = self.config.loudnorm_target
            loudnorm_active = target != 0.0
            multi_active = self.config.multi_audio_passthrough
            stream_count = (
                _probe_audio_stream_count(original) if multi_active else 1
            )
            if loudnorm_active and multi_active and stream_count > 1:
                # B-4: per-stream loudnorm via -filter_complex. Build one
                # loudnorm branch per input audio stream, label each
                # output (`[a0]`, `[a1]`, ...), and map every labelled
                # output. Each stream gets the same LUFS target so the
                # whole programme normalises uniformly.
                lf = ";".join(
                    f"[1:a:{i}]loudnorm=I={target}:TP=-1.5:LRA=11[a{i}]"
                    for i in range(stream_count)
                )
                cmd += ['-filter_complex', lf]
                for i in range(stream_count):
                    cmd += ['-map', f'[a{i}]']
                logger.info(
                    f"Applying per-stream EBU R128 loudnorm I={target} LUFS "
                    f"across {stream_count} audio tracks"
                )
            else:
                # Multi-track audio passthrough: '1:a?' selects every
                # audio stream from the original input (re-encoded to
                # AAC for mp4 container compatibility). When the user
                # disables it we keep the legacy single-track behaviour
                # ('1:a:0?').
                if multi_active:
                    cmd += ['-map', '1:a?']
                else:
                    cmd += ['-map', '1:a:0?']
                # Optional EBU R128 loudness normalisation. Single-pass;
                # for broadcast-grade accuracy a two-pass
                # measure-then-apply would be preferable, but the
                # single-pass filter is good enough for the platform-
                # target use case (YouTube -14, Apple -16, broadcast -23).
                if loudnorm_active:
                    cmd += ['-af', f'loudnorm=I={target}:TP=-1.5:LRA=11']
                    logger.info(f"Applying EBU R128 loudnorm I={target} LUFS")
            cmd += [
                '-shortest',
                str(temp_output),
            ]
            # Adaptive timeout: scales with the duration of the original
            # input so multi-hour videos do not silently lose audio when the
            # mux pass takes longer than the legacy 10-minute fixed budget.
            timeout = _ffmpeg_subprocess_timeout(_probe_duration_seconds(original))
            self._run_checked_ffmpeg(cmd, timeout)
            _promote_temp_output(temp_output, output)
            encoder_name = self._hw_encoder or 'libx264'
            logger.info(f"Audio merged successfully (encoder: {encoder_name})")
            return output
        except subprocess.TimeoutExpired:
            # Do not re-run ffmpeg after a duration-adaptive timeout --
            # salvage the intermediate into a container-correct path.
            logger.warning("FFmpeg audio merge timed out, saving video without audio")
            return self._salvage_intermediate(processed, output)
        except subprocess.CalledProcessError as e:
            # If hardware encoder failed, retry with software
            if self._hw_encoder:
                logger.warning(f"HW encoder failed, retrying with libx264: {e}")
                self._hw_encoder = None
                return self._merge_audio(original, processed, output)
            # The merge often fails on the AUDIO side (bad stream, odd
            # layout); a video-only encode to the requested container is
            # usually still possible and beats a mislabeled raw copy.
            logger.warning(f"Audio merge failed: {e}, encoding video without audio")
            return self._reencode_or_copy(processed, output)
        except FileNotFoundError:
            logger.warning("FFmpeg not found, saving video without audio")
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
