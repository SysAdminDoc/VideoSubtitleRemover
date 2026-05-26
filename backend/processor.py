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
from typing import Any, Optional, Tuple, List, Generator, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# RFP-L-1 re-exports. Anything that used to be defined in this module
# but moved during the split is re-imported here so existing callers
# (`from backend.processor import _open_capture`) keep working.
from backend.io import (
    _probe_codec_for_log,
    _probe_audio_stream_count,
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
)
from backend.encoder import _detect_hw_encoder
from backend.quality import _ssim
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
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        stream = open(path, "a", encoding="utf-8")
    except OSError as exc:
        logger.warning(f"Could not open JSON log {path}: {exc}")
        return None
    handler = JsonLineLogHandler(stream)
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    logger.info(f"JSON log enabled at {path}")
    return handler


class InpaintMode(Enum):
    """Supported inpainting algorithms."""
    STTN = "sttn"
    LAMA = "lama"
    PROPAINTER = "propainter"
    AUTO = "auto"   # per-batch routing between TBE (easy) and LaMa (hard)
    MIGAN = "migan"  # opt-in ONNX backend (RM-26)


@dataclass
class ProcessingConfig:
    """Configuration for subtitle removal."""
    mode: InpaintMode = InpaintMode.STTN
    device: str = "cuda:0"

    # STTN settings
    sttn_skip_detection: bool = False
    sttn_neighbor_stride: int = 10
    sttn_reference_length: int = 10
    sttn_max_load_num: int = 30

    # LAMA settings
    lama_super_fast: bool = False

    # Detection settings
    subtitle_area: Optional[Tuple[int, int, int, int]] = None
    detection_threshold: float = 0.5
    detection_lang: str = "en"
    detection_frame_skip: int = 0  # 0=detect every frame, N=reuse mask for N frames
    # RM-24 vertical-text mode: when on, detectors that support a
    # rotation flag are told to expect top-to-bottom text columns
    # (Japanese tategaki, classical Chinese). Bounding boxes still come
    # back axis-aligned so downstream masking is unchanged.
    detection_vertical: bool = False
    # RM-27 Whisper fallback: when on AND the optional faster-whisper
    # dep is installed, frames whose OCR yields no boxes get a default
    # bottom-band mask during Whisper-detected speech intervals. Catches
    # anti-aliased / motion-blurred subtitles the OCR cascade misses.
    whisper_fallback: bool = False
    whisper_model_size: str = "tiny"   # tiny / base / small / medium
    # RM-78 / RM-80 optional post-restore passes. Each runs after the
    # main encode + audio mux. Real-ESRGAN scale 0 = disabled; values
    # 2 or 4 invoke the upscale stage (requires the
    # realesrgan-ncnn-vulkan binary on PATH). Film-grain strength 0 =
    # disabled; positive values invoke the ffmpeg noise pass.
    upscale_factor: int = 0
    film_grain_strength: float = 0.0
    # RM-79: SwinIR restoration as an alternative to Real-ESRGAN for
    # sources where the cleanup left subtle local blur. Defaults off;
    # requires a SwinIR / RealSR-ncnn-vulkan binary on PATH.
    swinir_restore: bool = False
    # RM-77: SeedVR2 one-step video restoration. Heavy (16B params);
    # opt-in via flag and either a pip-published `seedvr2` wrapper or
    # VSR_SEEDVR2_CMD pointing at an external CLI.
    seedvr2_restore: bool = False
    # RM-73 (partial): preserve source color signalling on the output
    # encode. Default True so HDR sources at least stay tagged as HDR
    # even though the pixel pipeline is still 8-bit BGR. Disable when
    # the source has incorrect / misleading tags.
    preserve_color_metadata: bool = True
    # RM-76 NLE round-trip sidecars. None = off; "edl" or "fcpxml"
    # writes a sibling sidecar next to the output naming the source
    # and the processed range.
    nle_sidecar: str = "off"

    # Mask settings
    mask_dilate_px: int = 8  # morphological dilation on masks for cleaner removal
    mask_feather_px: int = 4  # gaussian feather for seamless alpha-blend at edges

    # Temporal Background Exposure (real STTN / ProPainter path)
    # When enabled, STTN/ProPainter sample masked pixels from neighbouring frames
    # in the same batch where the pixel is unmasked (subtitle text is sparse in
    # time -- adjacent frames reveal the true background).
    tbe_enable: bool = True       # enable temporal background exposure
    tbe_min_coverage: int = 3     # min frames where pixel must be unmasked to trust mean
    tbe_use_median: bool = True   # median is more robust than mean to motion
    tbe_flow_warp: bool = False   # Farneback flow-warp frames before aggregating (motion-heavy)
    tbe_scene_cut_split: bool = True   # split TBE batch at scene cuts
    tbe_scene_cut_threshold: float = 0.35   # histogram delta to call a cut
    # RM-32: prefer the PySceneDetect AdaptiveDetector when installed
    # (handles dissolves and flashes the histogram heuristic
    # mis-fires on). Defaults off so installations without the dep
    # keep the existing behaviour byte-identical.
    tbe_scene_cut_use_pyscenedetect: bool = False
    # RM-21: deep TransNetV2 scene-cut detector. When on AND the
    # `transnetv2` package + VSR_TRANSNETV2 weight path are set, we
    # try the deep detector before falling back to PySceneDetect /
    # histogram. Independent of `tbe_scene_cut_use_pyscenedetect`.
    tbe_scene_cut_use_transnetv2: bool = False
    # RM-33: optional denoise pass on the detection frame stream only;
    # output pixels are untouched. Helps OCR on VHS / phone clips.
    detection_denoise: bool = False
    # RM-66: opt-in SAM 2 mask refinement. Tighter mask = less inpaint
    # area = cleaner output. Requires VSR_SAM2_CHECKPOINT + sam2 pkg.
    sam2_refine: bool = False
    edge_ring_px: int = 2         # post-inpaint colour match ring width (0 disables)

    # Multi-region masks: list of (x1,y1,x2,y2) rects. When set, subtitle_area
    # is ignored and every rect is added to the composite mask.
    subtitle_areas: Optional[List[Tuple[int, int, int, int]]] = None

    # Optional debug artifacts
    export_mask_video: bool = False   # write a B/W mp4 of the per-frame masks
    export_srt: bool = False          # write an .srt sidecar of detected text

    # Adaptive batch sizing -- probe free VRAM on CUDA init, scale
    # sttn_max_load_num to match. Safe default: on.
    adaptive_batch: bool = True

    # v3.12 AUTO mode routing
    # Fraction of masked pixels that must be exposed in >=1 batch frame
    # to send the batch through TBE. Below threshold, route to LaMa.
    auto_exposure_threshold: float = 0.55

    # v3.12 preprocessing
    deinterlace: bool = False             # `ffmpeg -vf yadif` before the main pass
    deinterlace_auto: bool = True         # detect interlacing via ffprobe first

    # v3.12 keyframe-driven detection
    # OCR only at I-frames (parsed via ffprobe); between keyframes, propagate
    # Kalman-smoothed masks from the last anchor. Large speedup on streams.
    keyframe_detection: bool = False

    # v3.12 quality report
    quality_report: bool = False          # compute PSNR/SSIM on unmasked regions

    # v3.10 quality controls
    kalman_tracking: bool = True          # smooth per-frame detection jitter
    kalman_iou_threshold: float = 0.3
    kalman_max_age: int = 2               # frames a track survives w/o a hit

    # Perceptual-hash adaptive mask reuse: skip detection entirely when
    # the current frame's pHash is within N bits of the last detected frame.
    phash_skip_enable: bool = True
    phash_skip_distance: int = 4          # 0-64; higher = more aggressive skip

    # Colour-tuned mask expansion -- grow the mask inside each detected box
    # to cover pixels matching the dominant subtitle colour (catches serifs,
    # drop shadows, decorative strokes the OCR bbox clips).
    colour_tune_enable: bool = False
    colour_tune_tolerance: int = 25       # Lab-space distance threshold

    # Time range (video only, seconds from start)
    time_start: float = 0.0   # 0 = beginning
    time_end: float = 0.0     # 0 = entire video

    # Output settings
    preserve_audio: bool = True
    output_format: str = "mp4"
    output_quality: int = 23  # CRF value for x264
    use_hw_encode: bool = True  # try NVENC/QSV before falling back to libx264

    # F-8: output codec selector. "h264" (default) matches the legacy
    # behaviour; "h265" / "hevc" picks libx265 / hevc_nvenc; "av1"
    # picks libsvtav1 / av1_nvenc. Higher-efficiency codecs let users
    # keep manageable bitrates on 4K HDR sources where the previous
    # H.264-only pipeline ballooned the output.
    output_codec: str = "h264"

    # Optional EBU R128 loudness normalisation target (LUFS, e.g. -16.0).
    # 0.0 disables. Common platform targets: YouTube -14, Apple -16,
    # broadcast -23. Applied as an `ffmpeg -af loudnorm=I=...` pass during
    # audio mux; cost is one extra pass through libavfilter.
    loudnorm_target: float = 0.0

    # Opt-in hardware-accelerated video decode hint for cv2.VideoCapture.
    # "off" (default) preserves the existing software path; "auto"/"any"
    # lets cv2 pick; "d3d11" / "vaapi" / "mfx" target a specific backend.
    # _open_capture() falls back silently to software if HW returns empty
    # frames (cv2/FFmpeg known issue).
    decode_hw_accel: str = "off"

    # When >1 input audio stream exists, mux all of them through to the
    # output instead of dropping all but the first. Bluray/DVD rips
    # routinely ship 3-5 language tracks.
    multi_audio_passthrough: bool = True

    # Decouple cv2.VideoCapture.read() from the detect+inpaint critical
    # path by running it on a worker thread that feeds a bounded queue.
    # cv2 / numpy / onnxruntime release the GIL on heavy calls so simple
    # threading is enough. Default on; toggle off if you suspect a
    # decode-vs-process race or want strictly serial behaviour for debug.
    prefetch_decode: bool = True
    prefetch_queue_size: int = 0   # 0 = auto (max(8, batch_size * 2))

    # Frame-sequence input FPS. When `input_path` is a directory of
    # images, treat them as one frame each at this rate.
    input_fps: float = 24.0

    # Also render a side-by-side PNG comparison sheet alongside the
    # PSNR/SSIM numeric report so reviewers can scan visually instead of
    # squinting at metrics. Implies `quality_report = True`.
    quality_report_sheet: bool = False

    # Chyron classifier: persistent text boxes (station logos, lower-
    # thirds, breaking-news tickers) are categorised separately from
    # dialogue subtitles so users can keep one and remove the other.
    # `chyron_min_hits` is the threshold in matched frames a Kalman
    # track must accumulate before it's classified as a chyron; default
    # 90 catches ~3 s at 30 fps, which is longer than a typical dialogue
    # subtitle but well within the lifetime of a persistent graphic.
    # Both `remove_*` default True for backward compatibility (v3.12
    # removed every detected box unconditionally).
    remove_subtitles: bool = True
    remove_chyrons: bool = True
    chyron_min_hits: int = 90

    # Karaoke / per-syllable grouping: OCR engines emit per-syllable
    # boxes for animated karaoke captions; the gaps between them leak
    # the original highlighted text through the mask. When enabled,
    # boxes on the same horizontal line with at most `karaoke_x_gap_px`
    # of horizontal separation are fused into one wide box before
    # Kalman tracking runs.
    karaoke_grouping: bool = False
    karaoke_x_gap_px: int = 20
    karaoke_y_overlap: float = 0.5   # 0..1 vertical overlap to call "same line"


def _coerce_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off", ""}:
            return False
    return default


def _coerce_int(value, default: int, min_value: Optional[int] = None,
                max_value: Optional[int] = None) -> int:
    """Coerce to int, rejecting NaN / inf via the float round-trip."""
    import math as _math
    try:
        f = float(value)
        if not _math.isfinite(f):
            coerced = default
        else:
            coerced = int(f)
    except (TypeError, ValueError):
        coerced = default
    if min_value is not None:
        coerced = max(min_value, coerced)
    if max_value is not None:
        coerced = min(max_value, coerced)
    return coerced


def _coerce_float(value, default: float, min_value: Optional[float] = None,
                  max_value: Optional[float] = None) -> float:
    """Coerce to float, rejecting NaN / inf. These propagate into ffmpeg
    argv and cv2 frame-count math if allowed through; fall back to default."""
    import math as _math
    try:
        coerced = float(value)
        if not _math.isfinite(coerced):
            coerced = default
    except (TypeError, ValueError):
        coerced = default
    if min_value is not None:
        coerced = max(min_value, coerced)
    if max_value is not None:
        coerced = min(max_value, coerced)
    return coerced


def _coerce_text(value, default: str, max_length: int = 256) -> str:
    if isinstance(value, str):
        text = value.strip()
        if len(text) > max_length:
            text = text[:max_length]
        return text
    return default


def _coerce_rect(value) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(float(v)) for v in value]
    except (TypeError, ValueError):
        return None
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = max(0, x2), max(0, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _coerce_rect_list(value) -> Optional[List[Tuple[int, int, int, int]]]:
    if not isinstance(value, (list, tuple)):
        return None
    rects = []
    for item in value:
        rect = _coerce_rect(item)
        if rect:
            rects.append(rect)
    return rects or None


def _coerce_backend_mode(value) -> InpaintMode:
    if isinstance(value, InpaintMode):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        mode_map = {
            "sttn": InpaintMode.STTN,
            "lama": InpaintMode.LAMA,
            "propainter": InpaintMode.PROPAINTER,
            "pro painter": InpaintMode.PROPAINTER,
            "auto": InpaintMode.AUTO,
            "migan": InpaintMode.MIGAN,
            "mi-gan": InpaintMode.MIGAN,
        }
        if normalized in mode_map:
            return mode_map[normalized]
    return InpaintMode.STTN


def _coerce_backend_device(value) -> str:
    if isinstance(value, str):
        device = value.strip().lower()
        if device == "cpu" or device == "directml":
            return device
        if device.startswith("cuda:"):
            try:
                index = int(device.split(":", 1)[1])
            except (TypeError, ValueError):
                return "cpu"
            return f"cuda:{max(0, index)}"
    return "cpu"


def normalize_processing_config(config: ProcessingConfig) -> ProcessingConfig:
    """Coerce config values into a safe runtime shape."""
    config.mode = _coerce_backend_mode(config.mode)
    config.device = _coerce_backend_device(config.device)
    config.sttn_skip_detection = _coerce_bool(config.sttn_skip_detection, False)
    config.sttn_neighbor_stride = _coerce_int(config.sttn_neighbor_stride, 10, 1, 60)
    config.sttn_reference_length = _coerce_int(config.sttn_reference_length, 10, 1, 60)
    config.sttn_max_load_num = _coerce_int(config.sttn_max_load_num, 30, 1, 512)
    config.lama_super_fast = _coerce_bool(config.lama_super_fast, False)
    config.subtitle_area = _coerce_rect(config.subtitle_area)
    config.subtitle_areas = _coerce_rect_list(config.subtitle_areas)
    config.detection_threshold = _coerce_float(config.detection_threshold, 0.5, 0.1, 1.0)
    config.detection_lang = _coerce_text(config.detection_lang, "en", 24).lower()
    config.detection_frame_skip = _coerce_int(config.detection_frame_skip, 0, 0, 240)
    config.detection_vertical = _coerce_bool(config.detection_vertical, False)
    config.whisper_fallback = _coerce_bool(config.whisper_fallback, False)
    model_size = _coerce_text(config.whisper_model_size, "tiny", 16).lower()
    if model_size not in {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}:
        model_size = "tiny"
    config.whisper_model_size = model_size
    config.upscale_factor = _coerce_int(config.upscale_factor, 0, 0, 8)
    if config.upscale_factor not in (0, 2, 3, 4):
        config.upscale_factor = 0
    config.film_grain_strength = _coerce_float(
        config.film_grain_strength, 0.0, 0.0, 0.5)
    config.swinir_restore = _coerce_bool(config.swinir_restore, False)
    config.seedvr2_restore = _coerce_bool(config.seedvr2_restore, False)
    config.preserve_color_metadata = _coerce_bool(config.preserve_color_metadata, True)
    sidecar = _coerce_text(config.nle_sidecar, "off", 16).lower()
    if sidecar not in {"off", "edl", "fcpxml"}:
        sidecar = "off"
    config.nle_sidecar = sidecar
    config.mask_dilate_px = _coerce_int(config.mask_dilate_px, 8, 0, 100)
    config.mask_feather_px = _coerce_int(config.mask_feather_px, 4, 0, 100)
    config.tbe_enable = _coerce_bool(config.tbe_enable, True)
    config.tbe_min_coverage = _coerce_int(config.tbe_min_coverage, 3, 1, 32)
    config.tbe_use_median = _coerce_bool(config.tbe_use_median, True)
    config.tbe_flow_warp = _coerce_bool(config.tbe_flow_warp, False)
    config.tbe_scene_cut_split = _coerce_bool(config.tbe_scene_cut_split, True)
    config.tbe_scene_cut_threshold = _coerce_float(config.tbe_scene_cut_threshold, 0.35, 0.0, 1.0)
    config.tbe_scene_cut_use_pyscenedetect = _coerce_bool(
        config.tbe_scene_cut_use_pyscenedetect, False)
    config.tbe_scene_cut_use_transnetv2 = _coerce_bool(
        config.tbe_scene_cut_use_transnetv2, False)
    config.detection_denoise = _coerce_bool(config.detection_denoise, False)
    config.sam2_refine = _coerce_bool(config.sam2_refine, False)
    config.edge_ring_px = _coerce_int(config.edge_ring_px, 2, 0, 32)
    config.export_mask_video = _coerce_bool(config.export_mask_video, False)
    config.export_srt = _coerce_bool(config.export_srt, False)
    config.adaptive_batch = _coerce_bool(config.adaptive_batch, True)
    config.auto_exposure_threshold = _coerce_float(config.auto_exposure_threshold, 0.55, 0.0, 1.0)
    config.deinterlace = _coerce_bool(config.deinterlace, False)
    config.deinterlace_auto = _coerce_bool(config.deinterlace_auto, True)
    config.keyframe_detection = _coerce_bool(config.keyframe_detection, False)
    config.quality_report = _coerce_bool(config.quality_report, False)
    config.kalman_tracking = _coerce_bool(config.kalman_tracking, True)
    config.kalman_iou_threshold = _coerce_float(config.kalman_iou_threshold, 0.3, 0.0, 1.0)
    config.kalman_max_age = _coerce_int(config.kalman_max_age, 2, 0, 120)
    config.phash_skip_enable = _coerce_bool(config.phash_skip_enable, True)
    config.phash_skip_distance = _coerce_int(config.phash_skip_distance, 4, 0, 64)
    config.colour_tune_enable = _coerce_bool(config.colour_tune_enable, False)
    config.colour_tune_tolerance = _coerce_int(config.colour_tune_tolerance, 25, 0, 255)
    config.time_start = max(0.0, _coerce_float(config.time_start, 0.0))
    config.time_end = max(0.0, _coerce_float(config.time_end, 0.0))
    if config.time_end and config.time_end < config.time_start:
        config.time_end = 0.0
    config.preserve_audio = _coerce_bool(config.preserve_audio, True)
    config.output_format = _coerce_text(config.output_format, "mp4", 16).lower()
    config.output_quality = _coerce_int(config.output_quality, 23, 0, 51)
    config.use_hw_encode = _coerce_bool(config.use_hw_encode, True)
    codec = _coerce_text(config.output_codec, "h264", 16).lower()
    if codec in {"hevc", "h.265"}:
        codec = "h265"
    if codec not in {"h264", "h265", "av1"}:
        codec = "h264"
    config.output_codec = codec
    # loudnorm_target: 0.0 disables; otherwise clamp to the LUFS range
    # ffmpeg's loudnorm filter actually accepts (-70 to -5 inclusive).
    target = _coerce_float(config.loudnorm_target, 0.0)
    if target == 0.0 or -70.0 <= target <= -5.0:
        config.loudnorm_target = target
    else:
        config.loudnorm_target = 0.0
    accel = _coerce_text(config.decode_hw_accel, "off", 16).lower()
    if accel not in {"off", "auto", "any", "d3d11", "vaapi", "mfx"}:
        accel = "off"
    config.decode_hw_accel = accel
    config.multi_audio_passthrough = _coerce_bool(config.multi_audio_passthrough, True)
    config.prefetch_decode = _coerce_bool(config.prefetch_decode, True)
    config.prefetch_queue_size = _coerce_int(config.prefetch_queue_size, 0, 0, 512)
    config.input_fps = _coerce_float(config.input_fps, 24.0, 1.0, 240.0)
    config.quality_report_sheet = _coerce_bool(config.quality_report_sheet, False)
    if config.quality_report_sheet:
        # The sheet is rendered from the same sample _compute_quality_report
        # collects, so the numeric report must run for the sheet to exist.
        config.quality_report = True
    config.remove_subtitles = _coerce_bool(config.remove_subtitles, True)
    config.remove_chyrons = _coerce_bool(config.remove_chyrons, True)
    config.chyron_min_hits = _coerce_int(config.chyron_min_hits, 90, 1, 100000)
    config.karaoke_grouping = _coerce_bool(config.karaoke_grouping, False)
    config.karaoke_x_gap_px = _coerce_int(config.karaoke_x_gap_px, 20, 0, 1024)
    config.karaoke_y_overlap = _coerce_float(config.karaoke_y_overlap, 0.5, 0.0, 1.0)
    return config


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
                target = max(8, min(512, estimated_frames))
                if target != self.config.sttn_max_load_num:
                    logger.info(
                        f"Adaptive batch: {self.config.sttn_max_load_num} -> {target} "
                        f"(free VRAM {free_gb:.1f} GB)")
                    self.config.sttn_max_load_num = target
            except Exception:
                pass
            finally:
                if pynvml is not None and nvml_started:
                    try:
                        pynvml.nvmlShutdown()
                    except Exception:
                        pass

        logger.info(f"Detector: {self.detector._engine_name} | "
                    f"Inpainter: {self.config.mode.value} | "
                    f"Device: {self.config.device}"
                    f"{' | HW encode: ' + self._hw_encoder if self._hw_encoder else ''}")

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
                     padding: int = 5, frame: Optional[np.ndarray] = None) -> np.ndarray:
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for x1, y1, x2, y2 in boxes:
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            mask[y1:y2, x1:x2] = 255

        # Morphological dilation for cleaner inpainting boundaries
        dilate_px = self.config.mask_dilate_px
        if dilate_px > 0 and mask.max() > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
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
                pass
        # PaddleOCR (line[1][0] is the recognised text)
        if self.detector._paddle_model is not None:
            try:
                results = self.detector._paddle_model.ocr(frame, cls=False)
                if results and results[0]:
                    return " ".join(line[1][0] for line in results[0] if line and line[1]).strip()
            except Exception:
                pass
        # EasyOCR: readtext yields (bbox, text, conf)
        if self.detector._easyocr_reader is not None:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rows = self.detector._easyocr_reader.readtext(frame_rgb)
                return " ".join(r[1] for r in rows if len(r) >= 2 and r[1]).strip()
            except Exception:
                pass
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
            indices = sorted(set(rng.integers(0, span, size=n_samples).tolist()))

            psnrs: List[float] = []
            ssims: List[float] = []
            roi_psnrs: List[float] = []
            roi_ssims: List[float] = []
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
            for idx in indices:
                cap_in.set(cv2.CAP_PROP_POS_FRAMES, start_frame + idx)
                ok_in, a = cap_in.read()
                cap_out.set(cv2.CAP_PROP_POS_FRAMES, min(out_total - 1, idx))
                ok_out, b = cap_out.read()
                if not (ok_in and ok_out):
                    continue
                if a.shape != b.shape:
                    b = cv2.resize(b, (a.shape[1], a.shape[0]),
                                    interpolation=cv2.INTER_AREA)
                p = cv2.PSNR(a, b)
                s = _ssim(a, b)
                psnrs.append(p)
                ssims.append(s)
                # ROI metric: same frame, but cropped to the union-mask
                # bbox so the score reflects the inpaint quality instead
                # of the unchanged background.
                if roi_ready:
                    x1, y1, x2, y2 = roi
                    x1 = max(0, min(a.shape[1] - 1, x1))
                    x2 = max(x1 + 1, min(a.shape[1], x2))
                    y1 = max(0, min(a.shape[0] - 1, y1))
                    y2 = max(y1 + 1, min(a.shape[0], y2))
                    a_roi = a[y1:y2, x1:x2]
                    b_roi = b[y1:y2, x1:x2]
                    if a_roi.size and a_roi.shape == b_roi.shape:
                        try:
                            roi_psnrs.append(float(cv2.PSNR(a_roi, b_roi)))
                            roi_ssims.append(_ssim(a_roi, b_roi))
                        except Exception:
                            pass
                if self.config.quality_report_sheet:
                    pairs.append((idx, a, b, p, s))
            if not psnrs:
                return None
            mean_ssim = float(np.mean(ssims))
            mean_psnr = float(np.mean(psnrs))
            roi_mean_ssim = float(np.mean(roi_ssims)) if roi_ssims else None
            roi_mean_psnr = float(np.mean(roi_psnrs)) if roi_psnrs else None
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
                    logger.warning(f"Quality sheet write failed: {exc}")
            return {
                'psnr': mean_psnr,
                'ssim': mean_ssim,
                'roi_psnr': roi_mean_psnr,
                'roi_ssim': roi_mean_ssim,
                'roi_bbox': list(roi) if roi else None,
                'samples': len(psnrs),
                'tag': tag,
                'sheet': sheet_path,
            }
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
            logger.warning(f"SRT write failed: {exc}")

    def _fixed_region_boxes(self) -> Optional[List[Tuple[int, int, int, int]]]:
        """Return explicit mask rects from config, preferring the multi-region
        list if set, falling back to the single-rect legacy field."""
        if self.config.subtitle_areas:
            return list(self.config.subtitle_areas)
        if self.config.subtitle_area:
            return [self.config.subtitle_area]
        return None

    def process_image(self, input_path: str, output_path: str) -> bool:
        try:
            _ensure_output_parent(output_path)
            self._report_progress(0.1, "Loading image...")
            image = cv2.imread(input_path)
            if image is None:
                raise ValueError(f"Could not load image: {input_path}")

            self._report_progress(0.3, "Detecting text regions...")
            fixed = self._fixed_region_boxes()
            if fixed:
                boxes = fixed
            else:
                boxes = self.detector.detect(image, self.config.detection_threshold)

            if not boxes:
                logger.info("No text detected, copying original")
                _copy_file_atomic(input_path, output_path)
                self._report_progress(1.0, "Complete (no text found)")
                return True

            self._report_progress(0.5, f"Removing {len(boxes)} text regions...")
            mask = self._create_mask(image.shape, boxes, frame=image)
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
            self._report_progress(1.0, "Complete!")
            return True

        except InterruptedError:
            logger.info("Image processing cancelled")
            raise
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return False

    def process_video(self, input_path: str, output_path: str) -> bool:
        temp_dir = None
        cap = None
        reader = None
        writer = None
        mask_writer = None
        temp_mask_path = None
        try:
            _ensure_output_parent(output_path)
            self._report_progress(0.0, "Opening video...")

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
                temp_dir = tempfile.mkdtemp(prefix="vsr_")
                try:
                    processed_input = _deinterlace_to_temp(input_path, temp_dir)
                    logger.info(f"Using deinterlaced source: {processed_input}")
                    decode_path = processed_input
                except Exception as exc:
                    logger.warning(f"Deinterlace failed, continuing with original: {exc}")
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
                    pass

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
                    logger.debug(f"Color-metadata probe failed: {exc}")

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
                raise ValueError(f"Could not open video: {decode_path}")
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
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

            if width == 0 or height == 0:
                raise ValueError(f"Invalid video dimensions: {width}x{height}")

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
                temp_dir = tempfile.mkdtemp(prefix="vsr_")
            # I-1: lossless FFV1 intermediate inside .mkv. The previous
            # mp4v intermediate cost a full generation of lossy
            # compression before the final ffmpeg encode pass. The
            # writer falls back to mp4v + .mp4 when ffmpeg is missing
            # so the pipeline still produces output, just at the old
            # quality.
            temp_video_target = os.path.join(temp_dir, "temp_video.mkv")
            writer = _LosslessIntermediateWriter(
                temp_video_target, width, height, fps
            )
            temp_video = writer.path
            if not writer.isOpened():
                raise ValueError(f"Could not create video writer for: {temp_video}")

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
            self._srt_entries = []
            # B-3: reset the union-mask bbox so the report ROI reflects
            # only the current video's detections.
            self._quality_mask_bbox = None

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
                    if _wf.is_available():
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
                    logger.debug(f"Whisper fallback setup failed: {exc}")

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
                                logger.debug(
                                    f"Detection denoise fell back: {exc}"
                                )
                                det_frame = frame
                        else:
                            det_frame = frame
                        detected_boxes = self.detector.detect(det_frame, self.config.detection_threshold)
                        # Karaoke grouping: fuse per-syllable boxes on the
                        # same line before tracking so Kalman sees one
                        # composite per line, not one per syllable.
                        if self.config.karaoke_grouping and detected_boxes:
                            detected_boxes = _group_horizontal_line(
                                detected_boxes,
                                x_gap_px=self.config.karaoke_x_gap_px,
                                y_overlap_ratio=self.config.karaoke_y_overlap,
                            )
                        # Smooth jitter + fill single-frame misses via Kalman
                        if tracker is not None:
                            smoothed = tracker.update(list(detected_boxes))
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
                        # If fixed boxes are set without skip_detection, union them
                        # with per-frame detections so users can pin a region AND
                        # still clean incidental text elsewhere.
                        if fixed_boxes:
                            boxes = list(fixed_boxes) + smoothed
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

                    mask = self._create_mask(frame.shape, boxes, frame=frame)
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
                            logger.debug(f"on_preview_frame hook raised: {exc}")
                if mask_writer is not None:
                    for m in masks:
                        mask_writer.write(m)

            # reader.release() (or cap.release() when prefetch is off)
            # also joins the worker thread and releases the underlying cap.
            reader.release()
            reader = None
            cap = None
            writer.release()
            writer = None
            if mask_writer is not None:
                mask_writer.release()

            self._report_progress(0.9, "Merging audio...")
            # Frame-sequence inputs carry no audio stream; silently bypass
            # the merge step so ffmpeg doesn't error on `-i <directory>`.
            is_frame_sequence_input = Path(input_path).is_dir()
            if self.config.preserve_audio and not is_frame_sequence_input:
                self._merge_audio(input_path, temp_video, output_path)
            else:
                self._reencode_or_copy(temp_video, output_path)
            if mask_writer is not None and mask_path and temp_mask_path:
                _promote_temp_output(temp_mask_path, mask_path)
                temp_mask_path = None
                logger.info(f"Mask video written: {mask_path}")

            if self.config.export_srt and self._srt_entries:
                srt_path = str(Path(output_path).with_suffix('.srt'))
                self._write_srt(srt_path, fps, start_frame)

            # RM-78 / RM-80: optional post-restore passes (Real-ESRGAN
            # upscale, film-grain re-synthesis). Run after the main mux
            # so the user-visible output is the post-processed file;
            # each adapter degrades gracefully when its dep is missing.
            self._run_post_restore_passes(output_path, temp_dir)

            # RM-76: optional NLE round-trip sidecar (EDL / FCPXML).
            self._write_nle_sidecar(input_path, output_path,
                                     start_frame, end_frame, fps)

            # Quality report: PSNR/SSIM across a sample of unmasked regions
            if self.config.quality_report:
                try:
                    metrics = self._compute_quality_report(
                        input_path, output_path, start_frame, end_frame, fps)
                    if metrics:
                        self.last_quality_report = metrics
                        tag_suffix = f" [{metrics['tag']}]" if metrics.get('tag') else ""
                        logger.info(
                            f"Quality report: PSNR={metrics['psnr']:.2f} dB, "
                            f"SSIM={metrics['ssim']:.4f} "
                            f"({metrics['samples']} samples){tag_suffix}")
                        if metrics.get('sheet'):
                            logger.info(f"Quality sheet: {metrics['sheet']}")
                except Exception as exc:
                    logger.warning(f"Quality report failed: {exc}")

            self._report_progress(1.0, "Complete!")
            return True

        except InterruptedError:
            logger.info("Video processing cancelled")
            raise
        except Exception as e:
            logger.error(f"Video processing error: {e}")
            return False
        finally:
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass
            if mask_writer is not None:
                try:
                    mask_writer.release()
                except Exception:
                    pass
            # If a prefetch reader was set up, release it (which also stops
            # the worker thread and releases the underlying cap). Otherwise
            # release the raw cap. Tolerate either being unset on early
            # failures.
            if reader is not None:
                try:
                    reader.release()
                except Exception:
                    pass
            elif cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            _cleanup_temp_output(temp_mask_path)
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            # RM-27: Whisper audio temp dir is created lazily inside the
            # main try block; clean it up here only if it was set.
            try:
                _wda = locals().get("whisper_audio_dir", None)
                if _wda and os.path.exists(_wda):
                    shutil.rmtree(_wda, ignore_errors=True)
            except Exception:
                pass

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
            logger.warning(f"NLE sidecar write failed: {exc}")

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
                logger.warning(f"Real-ESRGAN pass failed: {exc}")
        if self.config.swinir_restore:
            try:
                from backend.post_restore import swinir_restore
                restored = os.path.join(temp_dir, "swinir.mp4")
                produced = swinir_restore(output_path, restored)
                if produced and Path(produced).is_file():
                    _promote_temp_output(produced, output_path)
                    logger.info("SwinIR restoration pass complete")
            except Exception as exc:
                logger.warning(f"SwinIR pass failed: {exc}")
        if self.config.seedvr2_restore:
            try:
                from backend.post_restore import seedvr2_restore
                restored = os.path.join(temp_dir, "seedvr2.mp4")
                produced = seedvr2_restore(output_path, restored)
                if produced and Path(produced).is_file():
                    _promote_temp_output(produced, output_path)
                    logger.info("SeedVR2 restoration pass complete")
            except Exception as exc:
                logger.warning(f"SeedVR2 pass failed: {exc}")
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
                logger.warning(f"Film-grain pass failed: {exc}")

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
            return []

    def _reencode_or_copy(self, source: str, output: str):
        """Re-encode with preferred encoder or just copy if FFmpeg unavailable."""
        temp_output = _allocate_temp_output_path(output)
        try:
            _ensure_output_parent(output)
            cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-nostats', '-i', source]
            cmd += self._get_encode_args()
            cmd += ['-an', str(temp_output)]
            timeout = _ffmpeg_subprocess_timeout(_probe_duration_seconds(source))
            subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
            _promote_temp_output(temp_output, output)
        except subprocess.CalledProcessError as e:
            if self._hw_encoder:
                logger.warning(f"HW encoder failed, retrying with libx264: {e}")
                self._hw_encoder = None
                self._reencode_or_copy(source, output)
                return
            _copy_file_atomic(source, output)
        except Exception:
            _copy_file_atomic(source, output)
        finally:
            _cleanup_temp_output(temp_output)

    def _merge_audio(self, original: str, processed: str, output: str):
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
            subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
            _promote_temp_output(temp_output, output)
            encoder_name = self._hw_encoder or 'libx264'
            logger.info(f"Audio merged successfully (encoder: {encoder_name})")
        except subprocess.TimeoutExpired:
            logger.warning("FFmpeg audio merge timed out, copying video without audio")
            _copy_file_atomic(processed, output)
        except subprocess.CalledProcessError as e:
            # If hardware encoder failed, retry with software
            if self._hw_encoder:
                logger.warning(f"HW encoder failed, retrying with libx264: {e}")
                self._hw_encoder = None
                self._merge_audio(original, processed, output)
                return
            logger.warning(f"Audio merge failed: {e}, copying video without audio")
            _copy_file_atomic(processed, output)
        except FileNotFoundError:
            logger.warning("FFmpeg not found, copying video without audio")
            _copy_file_atomic(processed, output)
        finally:
            _cleanup_temp_output(temp_output)


if __name__ == "__main__":
    from backend.cli import main as _cli_main
    _cli_main()
