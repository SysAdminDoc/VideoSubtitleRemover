"""
Backend Subtitle Removal Processor
Handles the actual subtitle detection and removal using AI models.

This module provides the core processing functionality that interfaces with
various inpainting models (STTN, LAMA, ProPainter) for subtitle removal.
"""

import os
import sys
import json
import cv2
import datetime
import numpy as np
import logging
import queue
import tempfile
import threading
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Any, Optional, Tuple, List, Generator, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


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


def _surya_allowed() -> bool:
    """Return True when the user has explicitly opted into the GPL Surya
    detector. Default off so an accidental `pip install surya-ocr` cannot
    license-pollute an MIT-clean distribution."""
    val = os.environ.get("VSR_ALLOW_GPL", "").strip().lower()
    return val in {"1", "true", "yes", "on"}


def _probe_audio_stream_count(path: str) -> int:
    """Return the number of audio streams in `path` via ffprobe.

    Used by B-4 to build a per-stream loudnorm filter chain when more
    than one audio track is present. Falls back to 1 on probe failure
    so the simple single-pass filter path stays usable.
    """
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'a',
            '-show_entries', 'stream=index',
            '-of', 'csv=p=0', path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if result.returncode == 0:
            lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
            return max(1, len(lines))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 1


def _probe_duration_seconds(path: str) -> float:
    """Return the container duration in seconds via ffprobe, or 0.0 if the
    probe fails. Used to budget ffmpeg subprocess timeouts so long videos
    don't get silently truncated to the legacy 10-minute cap."""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if result.returncode == 0:
            return max(0.0, float(result.stdout.strip()))
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return 0.0


def _ffmpeg_subprocess_timeout(duration_seconds: float,
                                base: float = 180.0,
                                factor: float = 4.0,
                                cap: float = 24 * 3600.0) -> float:
    """Budget an ffmpeg subprocess timeout that scales with content length.

    The legacy 600 s timeout silently strips audio on any video over ~1 hour
    because encode-pass time grows roughly linearly with duration. We scale
    a generous factor (4x realtime) on top of a fixed base so short clips
    still get a reasonable ceiling and long clips do not get truncated.
    `cap` is a sanity ceiling (24 h) so a corrupted probe cannot leave a
    runaway ffmpeg blocking the GUI forever.
    """
    if duration_seconds <= 0:
        return base + 600.0
    return max(base, min(cap, base + duration_seconds * factor))


def _ensure_output_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _path_key(path: Path | str) -> str:
    return str(Path(path).resolve(strict=False)).casefold()


def _choose_available_output_path(base_path: Path, reserved: Optional[set[str]] = None) -> Path:
    """Avoid overwriting an existing file or a path reserved earlier in the batch."""
    reserved = reserved or set()
    candidate = base_path
    counter = 2
    while candidate.exists() or _path_key(candidate) in reserved:
        candidate = base_path.with_name(f"{base_path.stem}({counter}){base_path.suffix}")
        counter += 1
    return candidate


def _write_text_atomic(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        temp_path = Path(temp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temp_path, path)
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def _allocate_temp_output_path(path: Path | str) -> Path:
    """Create a sibling temp file path that preserves the final suffix."""
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{final_path.stem}.",
        suffix=final_path.suffix or ".tmp",
        dir=str(final_path.parent),
    )
    os.close(fd)
    return Path(temp_name)


def _cleanup_temp_output(path: Optional[Path | str]):
    if not path:
        return
    try:
        Path(path).unlink()
    except OSError:
        pass


def _promote_temp_output(temp_path: Path | str, final_path: Path | str):
    _ensure_output_parent(str(final_path))
    os.replace(Path(temp_path), Path(final_path))


def _copy_file_atomic(source: str, output: str):
    temp_output = _allocate_temp_output_path(output)
    try:
        shutil.copy2(source, temp_output)
        _promote_temp_output(temp_output, output)
    finally:
        _cleanup_temp_output(temp_output)


# =============================================================================
# SUBTITLE DETECTION -- PaddleOCR > Surya > EasyOCR > OpenCV fallback
# =============================================================================

class SubtitleDetector:
    """Detects subtitle regions in video frames using text detection models."""

    def __init__(self, device: str = "cuda:0", lang: str = "en",
                 vertical: bool = False):
        self.device = device
        self.lang = lang
        # RM-24: when vertical=True, frames are rotated 90 CCW before
        # detection so horizontal-text-trained models still find
        # top-to-bottom CJK text columns. Boxes are rotated back to the
        # original frame's coordinate space before returning so callers
        # need no awareness of the flip.
        self.vertical = bool(vertical)
        self._engine_name = "none"
        self._rapid_model = None
        self._paddle_model = None
        self._surya_det = None
        self._surya_processor = None
        self._easyocr_reader = None
        self._load_model()

    def _is_gpu_device(self) -> bool:
        """Check if the device string indicates GPU acceleration."""
        return 'cuda' in self.device or self.device == 'directml'

    def _load_model(self):
        """Load detection model: RapidOCR > PaddleOCR > Surya > EasyOCR > OpenCV fallback."""
        # Try RapidOCR first -- PP-OCR via ONNX Runtime, 4-5x faster than PaddleOCR
        # and free of the memory-leak issues that plague the official paddlepaddle build.
        try:
            rapid_obj = None
            try:
                from rapidocr import RapidOCR as _RapidOCR
                rapid_obj = _RapidOCR()
            except ImportError:
                from rapidocr_onnxruntime import RapidOCR as _RapidOCR
                rapid_obj = _RapidOCR()
            if rapid_obj is not None:
                self._rapid_model = rapid_obj
                self._engine_name = "RapidOCR"
                logger.info(f"RapidOCR loaded via ONNX Runtime (lang={self.lang})")
                return
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"RapidOCR init failed: {e}")

        # PaddleOCR PP-OCRv5 (paddleocr>=3.0.0)
        try:
            from paddleocr import PaddleOCR
            self._paddle_model = PaddleOCR(
                use_angle_cls=False,
                lang=self.lang,
                use_gpu='cuda' in self.device,
                show_log=False
            )
            self._engine_name = "PaddleOCR"
            logger.info(f"PaddleOCR loaded (lang={self.lang})")
            return
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"PaddleOCR init failed: {e}")

        # Try Surya OCR (fast, layout-aware, 90+ languages). Surya is
        # GPL-licensed, so we never auto-select it unless the user has
        # explicitly opted in via the VSR_ALLOW_GPL env var. Distributing a
        # PyInstaller bundle that loaded a GPL detector at runtime would put
        # the MIT-clean release at risk.
        if _surya_allowed():
            try:
                from surya.detection import DetectionPredictor
                self._surya_det = DetectionPredictor()
                self._engine_name = "Surya"
                logger.info("Surya text detection loaded (GPL opt-in via VSR_ALLOW_GPL)")
                return
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"Surya init failed: {e}")
        else:
            try:
                import surya.detection  # noqa: F401
                logger.warning(
                    "Surya is installed but skipped: it is GPL-licensed and "
                    "VSR ships MIT-clean. Set VSR_ALLOW_GPL=1 to opt in."
                )
            except ImportError:
                pass
            except Exception:
                pass

        # Try EasyOCR
        try:
            import easyocr
            gpu = self._is_gpu_device()
            # Map PaddleOCR lang codes to EasyOCR equivalents
            easyocr_lang_map = {
                "ch": "ch_sim", "chinese_cht": "ch_tra",
                "ko": "ko", "ja": "ja", "en": "en",
                "fr": "fr", "de": "de", "es": "es", "pt": "pt",
                "ru": "ru", "ar": "ar", "hi": "hi", "it": "it",
            }
            mapped_lang = easyocr_lang_map.get(self.lang, self.lang)
            lang_list = [mapped_lang]
            if mapped_lang != "en":
                lang_list.append("en")
            self._easyocr_reader = easyocr.Reader(lang_list, gpu=gpu, verbose=False)
            self._engine_name = "EasyOCR"
            logger.info(f"EasyOCR loaded (lang={lang_list})")
            return
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"EasyOCR init failed: {e}")

        self._engine_name = "OpenCV fallback"
        logger.warning("No OCR engine available, using OpenCV fallback detection")

    def detect(self, frame: np.ndarray, threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """Detect text regions in a frame. Returns list of (x1, y1, x2, y2) boxes.

        When `self.vertical` is set we rotate the frame 90 CCW before
        running detection (so a vertically-stacked CJK column becomes a
        horizontal line the model is trained on) and rotate every
        returned box back into the original frame's coordinate space.
        """
        if self.vertical:
            h, w = frame.shape[:2]
            rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            rotated_boxes = self._detect_axis_aligned(rotated, threshold)
            # In the rotated frame, (rx1, ry1, rx2, ry2) maps back to
            # the original via: x = ry, y = w - rx2 .. w - rx1.
            out: List[Tuple[int, int, int, int]] = []
            for (rx1, ry1, rx2, ry2) in rotated_boxes:
                ox1 = max(0, ry1)
                oy1 = max(0, w - rx2)
                ox2 = min(h, ry2)
                oy2 = min(w, w - rx1)
                if ox2 > ox1 and oy2 > oy1:
                    out.append((ox1, oy1, ox2, oy2))
            return out
        return self._detect_axis_aligned(frame, threshold)

    def _detect_axis_aligned(self, frame: np.ndarray,
                              threshold: float) -> List[Tuple[int, int, int, int]]:
        """Dispatch to whichever engine loaded successfully."""
        if self._rapid_model is not None:
            return self._detect_rapid(frame, threshold)
        elif self._paddle_model is not None:
            return self._detect_paddle(frame, threshold)
        elif self._surya_det is not None:
            return self._detect_surya(frame, threshold)
        elif self._easyocr_reader is not None:
            return self._detect_easyocr(frame, threshold)
        else:
            return self._fallback_detection(frame)

    def _detect_rapid(self, frame: np.ndarray, threshold: float) -> List[Tuple[int, int, int, int]]:
        """Detect text using RapidOCR (ONNX Runtime PP-OCR)."""
        try:
            # RapidOCR accepts BGR numpy arrays directly and returns
            # (list_of_[box, text, conf], elapse) in the 1.x API and
            # a RapidOCROutput object in the 2.x API.
            output = self._rapid_model(frame)
            results = None
            if output is None:
                return []
            if isinstance(output, tuple) and len(output) >= 1:
                results = output[0]
            else:
                # 2.x API -- has .boxes and .scores attributes
                boxes_attr = getattr(output, 'boxes', None)
                scores_attr = getattr(output, 'scores', None)
                if boxes_attr is not None:
                    boxes = []
                    for i, poly in enumerate(boxes_attr):
                        conf = float(scores_attr[i]) if scores_attr is not None else 1.0
                        if conf >= threshold:
                            pts = np.array(poly, dtype=np.int32)
                            x1, y1 = pts.min(axis=0)
                            x2, y2 = pts.max(axis=0)
                            boxes.append((int(x1), int(y1), int(x2), int(y2)))
                    return boxes
                return []

            if not results:
                return []
            boxes = []
            for entry in results:
                # entry is typically [polygon, text, confidence]
                if len(entry) < 3:
                    continue
                poly, _text, conf = entry[0], entry[1], entry[2]
                if conf is None:
                    conf = 1.0
                if float(conf) >= threshold:
                    pts = np.array(poly, dtype=np.int32)
                    x1, y1 = pts.min(axis=0)
                    x2, y2 = pts.max(axis=0)
                    boxes.append((int(x1), int(y1), int(x2), int(y2)))
            return boxes
        except Exception as e:
            logger.error(f"RapidOCR detection error: {e}")
            return self._fallback_detection(frame)

    def _detect_paddle(self, frame: np.ndarray, threshold: float) -> List[Tuple[int, int, int, int]]:
        try:
            results = self._paddle_model.ocr(frame, cls=False)
            boxes = []
            if results and results[0]:
                for line in results[0]:
                    if line[1][1] >= threshold:
                        pts = np.array(line[0], dtype=np.int32)
                        x1, y1 = pts.min(axis=0)
                        x2, y2 = pts.max(axis=0)
                        boxes.append((int(x1), int(y1), int(x2), int(y2)))
            return boxes
        except Exception as e:
            logger.error(f"PaddleOCR detection error: {e}")
            return self._fallback_detection(frame)

    def _detect_surya(self, frame: np.ndarray, threshold: float) -> List[Tuple[int, int, int, int]]:
        """Detect text using Surya OCR text detection."""
        try:
            from PIL import Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            predictions = self._surya_det([pil_image])
            boxes = []
            if predictions and len(predictions) > 0:
                for bbox in predictions[0].bboxes:
                    if bbox.confidence >= threshold:
                        x1, y1, x2, y2 = [int(v) for v in bbox.bbox]
                        boxes.append((x1, y1, x2, y2))
            return boxes
        except Exception as e:
            logger.error(f"Surya detection error: {e}")
            return self._fallback_detection(frame)

    def _detect_easyocr(self, frame: np.ndarray, threshold: float) -> List[Tuple[int, int, int, int]]:
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._easyocr_reader.readtext(frame_rgb)
            boxes = []
            for (bbox, text, conf) in results:
                if conf >= threshold:
                    pts = np.array(bbox, dtype=np.int32)
                    x1, y1 = pts.min(axis=0)
                    x2, y2 = pts.max(axis=0)
                    boxes.append((int(x1), int(y1), int(x2), int(y2)))
            return boxes
        except Exception as e:
            logger.error(f"EasyOCR detection error: {e}")
            return self._fallback_detection(frame)

    def _fallback_detection(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Fallback detection using image processing when no OCR is available.

        EI-1: hardcoded 200 / 55 thresholds missed grey or mid-tone
        subtitles (semi-transparent banners, dimly-lit captions). We now
        pick thresholds from the frame's intensity percentiles so the
        operator scales with the actual luminance distribution. The
        bright threshold sits at the 95th percentile, the dark
        threshold at the 5th. Clamped to [220, 240] / [15, 35] so we do
        not flag the entire frame on a low-contrast source where the
        percentile band is itself near the middle.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        # Percentile-based thresholds with a sane min-contrast clamp so a
        # near-flat source cannot collapse both thresholds into the same
        # mid value and mark the whole frame.
        p_bright = float(np.percentile(gray, 95))
        p_dark = float(np.percentile(gray, 5))
        # Bright threshold: clamp BELOW the 95th percentile so we still
        # mark the brightest 5% even on a low-contrast source. Stay
        # comfortably above the median (>= median + 20) so the operator
        # is meaningful.
        median = float(np.median(gray))
        bright_thresh = int(max(median + 20, min(245, p_bright - 1)))
        dark_thresh = int(min(median - 20, max(10, p_dark + 1)))
        bright_thresh = max(0, min(255, bright_thresh))
        dark_thresh = max(0, min(255, dark_thresh))
        _, thresh_bright = cv2.threshold(gray, bright_thresh, 255, cv2.THRESH_BINARY)
        _, thresh_dark = cv2.threshold(gray, dark_thresh, 255, cv2.THRESH_BINARY_INV)
        combined = cv2.bitwise_or(thresh_bright, thresh_dark)

        # Morphological close to merge nearby character regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, w // 40), max(1, h // 80)))
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_boxes = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            # Accept subtitle-shaped regions in top 15% or bottom 40% of frame
            in_subtitle_zone = (y > h * 0.6) or (y + ch < h * 0.15)
            if in_subtitle_zone and cw > w * 0.08 and ch < h * 0.15 and ch > 4:
                raw_boxes.append((x, y, x + cw, y + ch))

        return self._merge_boxes(raw_boxes, margin=10)

    @staticmethod
    def _merge_boxes(boxes: List[Tuple[int, int, int, int]],
                     margin: int = 10) -> List[Tuple[int, int, int, int]]:
        """Merge overlapping or nearby bounding boxes."""
        if not boxes:
            return []
        expanded = [(x1 - margin, y1 - margin, x2 + margin, y2 + margin) for x1, y1, x2, y2 in boxes]
        merged = list(expanded)
        changed = True
        while changed:
            changed = False
            new_merged = []
            used = set()
            for i in range(len(merged)):
                if i in used:
                    continue
                ax1, ay1, ax2, ay2 = merged[i]
                for j in range(i + 1, len(merged)):
                    if j in used:
                        continue
                    bx1, by1, bx2, by2 = merged[j]
                    if ax1 <= bx2 and ax2 >= bx1 and ay1 <= by2 and ay2 >= by1:
                        ax1 = min(ax1, bx1)
                        ay1 = min(ay1, by1)
                        ax2 = max(ax2, bx2)
                        ay2 = max(ay2, by2)
                        used.add(j)
                        changed = True
                new_merged.append((ax1, ay1, ax2, ay2))
                used.add(i)
            merged = new_merged
        result = []
        for x1, y1, x2, y2 in merged:
            ux1 = max(0, x1 + margin)
            uy1 = max(0, y1 + margin)
            ux2 = x2 - margin
            uy2 = y2 - margin
            if ux2 > ux1 and uy2 > uy1:
                result.append((ux1, uy1, ux2, uy2))
        return result


# =============================================================================
# INPAINTING -- Real LaMa via simple-lama-inpainting, with cv2 fallback
# =============================================================================

class BaseInpainter(ABC):
    """Abstract base class for inpainting models."""

    @abstractmethod
    def inpaint(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        """Inpaint the masked regions in the frames."""
        pass


def _cv2_inpaint(frame: np.ndarray, mask: np.ndarray, radius: int = 5,
                 method: int = cv2.INPAINT_TELEA) -> np.ndarray:
    """OpenCV inpainting fallback."""
    if mask.max() > 0:
        return cv2.inpaint(frame, mask, radius, method)
    return frame.copy()


def _feather_blend(original: np.ndarray, filled: np.ndarray,
                   mask: np.ndarray, feather_px: int = 4) -> np.ndarray:
    """Alpha-blend the inpainted `filled` result back onto `original` using a
    Gaussian-softened mask so the boundary of the removed region is seamless."""
    if feather_px <= 0 or mask.max() == 0:
        return filled
    k = feather_px * 2 + 1
    soft = cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32) / 255.0
    if soft.ndim == 2:
        soft = soft[..., None]
    out = filled.astype(np.float32) * soft + original.astype(np.float32) * (1.0 - soft)
    return np.clip(out, 0, 255).astype(np.uint8)


class _KalmanBox:
    """Simple constant-velocity Kalman filter for a single subtitle box.
    State: [cx, cy, w, h, dx, dy, dw, dh]. Measurement: [cx, cy, w, h].
    Used to smooth per-frame OCR jitter and carry the box through a missed
    detection (single-frame occlusion)."""

    def __init__(self, box: Tuple[int, int, int, int]):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)
        self.kf.statePost = np.array(
            [cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32).reshape(8, 1)
        self.age = 0        # frames since last measurement
        self.hits = 1       # total measurements absorbed

    def predict(self) -> Tuple[int, int, int, int]:
        s = self.kf.predict().flatten()
        return _box_from_state(s)

    def update(self, box: Tuple[int, int, int, int]):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        m = np.array([cx, cy, w, h], dtype=np.float32).reshape(4, 1)
        self.kf.correct(m)
        self.age = 0
        self.hits += 1

    def is_chyron(self, min_hits: int) -> bool:
        """A track is chyron-like when it has matched a detection in at
        least `min_hits` frames -- station logos, lower-thirds, breaking-
        news tickers persist; dialogue subtitles turn over every few
        seconds. min_hits ~= fps * 3 catches typical persistent graphics
        without false-positiving on long dialogue lines."""
        return self.hits >= max(1, int(min_hits))

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return _box_from_state(self.kf.statePost.flatten())


def _box_from_state(state: np.ndarray) -> Tuple[int, int, int, int]:
    """Reconstruct (x1, y1, x2, y2) from a Kalman state vector. Width and
    height are clamped to >=1 so a noisy filter prediction never produces
    an inverted box (x2 < x1 or y2 < y1), which would corrupt IoU scoring
    and mask generation downstream."""
    cx = float(state[0])
    cy = float(state[1])
    w = max(1.0, float(state[2]))
    h = max(1.0, float(state[3]))
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return (0, 0, 1, 1)
    x1 = int(round(cx - w / 2.0))
    y1 = int(round(cy - h / 2.0))
    x2 = int(round(cx + w / 2.0))
    y2 = int(round(cy + h / 2.0))
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1
    return (x1, y1, x2, y2)


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
    area_b = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / float(union)


class SubtitleTracker:
    """Multi-box Kalman tracker that smooths per-frame detection jitter and
    carries boxes through single-frame misses. Pure numpy + cv2 -- no new
    dependency. Call `update()` with the raw OCR boxes per frame; it returns
    the smoothed, continuity-preserving box list for mask creation.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 2):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._tracks: List[_KalmanBox] = []

    def reset(self):
        self._tracks = []

    def update(self, detections: List[Tuple[int, int, int, int]]
                ) -> List[Tuple[int, int, int, int]]:
        if not self._tracks:
            self._tracks = [_KalmanBox(d) for d in detections]
            return [t.box for t in self._tracks]

        predictions = [t.predict() for t in self._tracks]
        used_det = set()
        used_trk = set()
        for ti, pred in enumerate(predictions):
            best_di, best_iou = -1, 0.0
            for di, det in enumerate(detections):
                if di in used_det:
                    continue
                score = _iou(pred, det)
                if score > best_iou:
                    best_iou, best_di = score, di
            if best_di >= 0 and best_iou >= self.iou_threshold:
                self._tracks[ti].update(detections[best_di])
                used_det.add(best_di)
                used_trk.add(ti)
            else:
                self._tracks[ti].age += 1

        for di, det in enumerate(detections):
            if di not in used_det:
                self._tracks.append(_KalmanBox(det))

        # Drop stale tracks
        self._tracks = [t for t in self._tracks if t.age <= self.max_age]
        return [t.box for t in self._tracks]

    def categorize(self, min_chyron_hits: int) -> List[str]:
        """Per-track category aligned with the last `update()` return value.
        'chyron' for tracks that have matched at least `min_chyron_hits`
        frames; 'subtitle' otherwise. Caller filters the box list against
        this to honour `remove_chyrons` / `remove_subtitles` flags."""
        return [
            "chyron" if t.is_chyron(min_chyron_hits) else "subtitle"
            for t in self._tracks
        ]


def _group_horizontal_line(
    boxes: List[Tuple[int, int, int, int]],
    x_gap_px: int = 20,
    y_overlap_ratio: float = 0.5,
) -> List[Tuple[int, int, int, int]]:
    """Merge boxes that sit on the same horizontal text line into a
    single composite.

    Karaoke / per-syllable rendering produces many small adjacent boxes
    that the rest of the pipeline treats as independent text lines.
    When we mask them individually the gaps between syllables leak the
    original highlighted text. Fusing them into one wide box closes the
    gap so the inpainter sees a single continuous region.

    Two boxes merge when their vertical extent overlaps by at least
    `y_overlap_ratio` (same line) AND the horizontal gap between them
    is at most `x_gap_px`. The merge loop iterates until no further
    merges happen, so five syllables fuse into one rectangle in one
    call. Pure transformation; safe to apply before Kalman so the
    smoothed track also covers the fused span.
    """
    if not boxes or len(boxes) == 1:
        return list(boxes)

    def _y_overlap(a, b) -> float:
        a_h = max(1, a[3] - a[1])
        b_h = max(1, b[3] - b[1])
        inter = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        return inter / float(min(a_h, b_h))

    def _x_gap(a, b) -> int:
        # Positive when there is a gap, negative when they overlap.
        return max(a[0], b[0]) - min(a[2], b[2])

    merged = [tuple(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        new_merged = []
        used = set()
        for i in range(len(merged)):
            if i in used:
                continue
            a = merged[i]
            ax1, ay1, ax2, ay2 = a
            for j in range(i + 1, len(merged)):
                if j in used:
                    continue
                b = merged[j]
                if (_y_overlap((ax1, ay1, ax2, ay2), b) >= y_overlap_ratio
                        and _x_gap((ax1, ay1, ax2, ay2), b) <= x_gap_px):
                    ax1 = min(ax1, b[0])
                    ay1 = min(ay1, b[1])
                    ax2 = max(ax2, b[2])
                    ay2 = max(ay2, b[3])
                    used.add(j)
                    changed = True
            new_merged.append((ax1, ay1, ax2, ay2))
            used.add(i)
        merged = new_merged
    return merged


def _phash(frame: np.ndarray, size: int = 8) -> np.ndarray:
    """Compact perceptual hash for adaptive frame-skip. Returns an 8x8 bit
    array based on DCT low-frequency coefficients vs their median. Hamming
    distance between two hashes is a cheap scene-similarity signal."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    low = dct[:size, :size]
    med = np.median(low)
    return (low > med).astype(np.uint8)


def _phash_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def _expand_mask_by_color(frame: np.ndarray, mask: np.ndarray,
                           boxes: List[Tuple[int, int, int, int]],
                           tolerance: int = 25,
                           padding: int = 4) -> np.ndarray:
    """Within each detected box, sample the dominant foreground colour
    (the cluster furthest from the mean background in Lab space) and
    extend the binary mask to every pixel within `tolerance` Lab
    distance. Catches serifs / drop shadows that the OCR bbox clips.
    """
    if not boxes or mask.max() == 0:
        return mask
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    out = mask.copy()
    h, w = mask.shape[:2]
    for (x1, y1, x2, y2) in boxes:
        x1 = max(0, x1 - padding); y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding); y2 = min(h, y2 + padding)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = lab[y1:y2, x1:x2].reshape(-1, 3).astype(np.int16)
        if roi.size == 0:
            continue
        # K-means-lite: split into two clusters by L channel median
        L = roi[:, 0]
        low = roi[L < np.median(L)]
        high = roi[L >= np.median(L)]
        if low.size == 0 or high.size == 0:
            continue
        # Foreground = cluster with smaller variance (subtitle text is
        # usually solid colour; background is more textured)
        low_var = float(low.var())
        high_var = float(high.var())
        fg = low.mean(axis=0) if low_var < high_var else high.mean(axis=0)
        # Compute per-pixel Lab distance to fg colour inside the box
        diff = roi - fg
        dist = np.sqrt((diff * diff).sum(axis=1))
        match = (dist < tolerance).reshape(y2 - y1, x2 - x1).astype(np.uint8) * 255
        out[y1:y2, x1:x2] = np.maximum(out[y1:y2, x1:x2], match)
    return out


def _edge_ring_color_correct(original: np.ndarray, filled: np.ndarray,
                              mask: np.ndarray, ring_px: int = 2) -> np.ndarray:
    """Sample a thin ring immediately outside the mask in both original and
    filled frames, compute the mean colour delta per channel across the ring,
    and apply the offset to pixels inside the mask. Nulls the faint colour
    seam that sometimes appears on gradient backgrounds after inpainting.

    Robust against degenerate masks (fully saturated, fully empty, no ring,
    non-finite sample means). Returns `filled` unchanged on any pathological
    input rather than propagating NaN to the output."""
    if filled is None or mask is None or ring_px <= 0:
        return filled
    if mask.size == 0 or mask.max() == 0:
        return filled
    mask_bool = mask > 0
    # Ring = dilate(mask) XOR mask -- the pixels immediately outside the mask
    k = ring_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    dilated = cv2.dilate(mask, kernel, iterations=1) > 0
    ring = dilated & ~mask_bool
    ring_count = int(ring.sum())
    # Require enough ring samples to make the mean meaningful; tiny rings
    # (e.g. mask touching the frame border) produce noisy deltas that
    # visibly shift the inpainted region.
    if ring_count < 16:
        return filled
    orig_mean = original[ring].astype(np.float32).mean(axis=0)
    fill_mean = filled[ring].astype(np.float32).mean(axis=0)
    delta = orig_mean - fill_mean                            # (3,)
    if not np.all(np.isfinite(delta)):
        return filled
    if np.abs(delta).max() < 0.5:
        return filled
    out = filled.astype(np.float32)
    out[mask_bool] = np.clip(out[mask_bool] + delta, 0, 255)
    return out.astype(np.uint8)


def _detect_scene_cuts_pyscenedetect(frames: List[np.ndarray]) -> Optional[List[int]]:
    """RM-32: optional PySceneDetect-backed scene cut detection.

    PySceneDetect's AdaptiveDetector handles dissolves and flashes the
    histogram heuristic mis-fires on. We feed it frames via its
    VideoStream wrapper but PySceneDetect normally reads from a file
    path, so we route the existing frame list through its in-memory
    SceneManager.process_frames interface. Returns the cut indices, or
    None when PySceneDetect is unavailable / errors out so the caller
    falls back to the histogram path.
    """
    try:
        from scenedetect import SceneManager  # type: ignore
        from scenedetect.detectors import AdaptiveDetector  # type: ignore
    except ImportError:
        return None
    try:
        sm = SceneManager()
        sm.add_detector(AdaptiveDetector())
        # PySceneDetect 0.7+ accepts frame-by-frame ingestion when we
        # call .process_frame directly. We don't have a VideoStream
        # but the detector only needs `(frame_num, frame)` so we
        # synthesise that.
        for i, f in enumerate(frames):
            sm._process_frame(i, f, callback=None)  # type: ignore[attr-defined]
        scene_list = sm.get_scene_list()
        if not scene_list:
            return [0]
        cuts = [0]
        for entry, _exit in scene_list:
            idx = int(entry.get_frames())
            if idx > 0 and idx < len(frames):
                cuts.append(idx)
        return sorted(set(cuts))
    except Exception as exc:
        logger.debug(f"PySceneDetect path failed: {exc}")
        return None


def _detect_scene_cuts(frames: List[np.ndarray],
                        threshold: float = 0.35,
                        prefer_pyscenedetect: bool = False) -> List[int]:
    """Return indices where a scene cut begins (inclusive). Uses histogram
    correlation on the luma channel -- cheap, robust for our TBE window of
    ~30 frames. Index 0 is always a segment start.

    When `prefer_pyscenedetect` is True and the optional dep is
    installed, we delegate to PySceneDetect's AdaptiveDetector which
    handles dissolves correctly. Falls through to the histogram path
    on any failure.
    """
    if len(frames) <= 1:
        return [0]
    if prefer_pyscenedetect:
        psd = _detect_scene_cuts_pyscenedetect(frames)
        if psd is not None:
            return psd
    cuts = [0]
    prev_hist = None
    for i, f in enumerate(frames):
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)
        if prev_hist is not None:
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if corr < (1.0 - threshold):
                cuts.append(i)
        prev_hist = hist
    return cuts


def _farneback_winsize(h: int, w: int) -> int:
    """Pick a Farneback window size proportional to frame dimensions. A
    fixed winsize=21 over-smooths flow on 4K and under-resolves it on
    sub-VGA clips. Scales to ~1/24 of the short edge, clamped to the
    usable range."""
    short_edge = max(1, min(h, w))
    return int(max(9, min(33, short_edge // 24)))


def _warp_to_reference(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Warp `src` so that its content aligns with `ref`, using Farneback dense
    optical flow on the luma channel. Used by flow-aware TBE to compensate
    for camera motion before aggregating temporal exposures."""
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    h, w = src.shape[:2]
    winsize = _farneback_winsize(h, w)
    flow = cv2.calcOpticalFlowFarneback(
        ref_gray, src_gray, None,
        pyr_scale=0.5, levels=3, winsize=winsize, iterations=3,
        poly_n=7, poly_sigma=1.5, flags=0,
    )
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                  np.arange(h, dtype=np.float32))
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    warped = cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
    return warped


def _warp_mask_to_reference(src_mask: np.ndarray, src_frame: np.ndarray,
                              ref_frame: np.ndarray) -> np.ndarray:
    """Same flow but for a binary mask -- we need the mask to follow the warp
    so that the unmasked/masked classification stays correct after warping.
    The border defaults to 255 (masked) so pixels shifted off-frame are
    treated conservatively as needing inpaint, never aggregated into the
    background estimate."""
    src_gray = cv2.cvtColor(src_frame, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    h, w = src_mask.shape[:2]
    winsize = _farneback_winsize(h, w)
    flow = cv2.calcOpticalFlowFarneback(
        ref_gray, src_gray, None,
        pyr_scale=0.5, levels=3, winsize=winsize, iterations=3,
        poly_n=7, poly_sigma=1.5, flags=0,
    )
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                  np.arange(h, dtype=np.float32))
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    warped = cv2.remap(src_mask, map_x, map_y, cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return warped


def _tbe_single_segment(frames: List[np.ndarray], masks: List[np.ndarray],
                         min_coverage: int, use_median: bool,
                         feather_px: int, edge_ring_px: int,
                         flow_warp: bool) -> List[np.ndarray]:
    """Aggregate a single scene-contiguous segment. Split out so TBE can be
    called per sub-segment when scene cuts are detected within a batch."""
    n = len(frames)
    if n == 0:
        return []
    if n == 1:
        filled = _cv2_inpaint(frames[0], masks[0], 7, cv2.INPAINT_NS)
        if edge_ring_px > 0:
            filled = _edge_ring_color_correct(frames[0], filled, masks[0], edge_ring_px)
        return [_feather_blend(frames[0], filled, masks[0], feather_px)]

    # Flow-warped TBE compensates for camera motion by aligning every frame
    # in the segment to a reference (middle) frame before aggregating.
    if flow_warp:
        ref_idx = n // 2
        ref_frame = frames[ref_idx]
        warped_frames: List[np.ndarray] = []
        warped_masks: List[np.ndarray] = []
        for i, (f, m) in enumerate(zip(frames, masks)):
            if i == ref_idx:
                warped_frames.append(f)
                warped_masks.append(m)
            else:
                try:
                    wf = _warp_to_reference(f, ref_frame)
                    wm = _warp_mask_to_reference(m, f, ref_frame)
                    warped_frames.append(wf)
                    warped_masks.append(wm)
                except Exception as exc:
                    logger.debug(f"Flow warp fell back for frame {i}: {exc}")
                    warped_frames.append(f)
                    warped_masks.append(m)
        agg_frames = warped_frames
        agg_masks = warped_masks
    else:
        agg_frames = list(frames)
        agg_masks = list(masks)

    frame_stack = np.stack(agg_frames, axis=0).astype(np.float32)   # (T,H,W,3)
    mask_stack = np.stack(agg_masks, axis=0)                         # (T,H,W)
    unmasked = (mask_stack == 0)                                     # (T,H,W) bool
    coverage = unmasked.sum(axis=0).astype(np.int32)                 # (H,W)

    if use_median and n <= 64:
        weighted = np.where(unmasked[..., None], frame_stack, np.nan)
        with np.errstate(all='ignore'):
            bg = np.nanmedian(weighted, axis=0)
        bg = np.nan_to_num(bg, nan=0.0)
    else:
        sum_vals = (frame_stack * unmasked[..., None]).sum(axis=0)
        count = np.maximum(coverage, 1).astype(np.float32)
        bg = sum_vals / count[..., None]
    bg = np.clip(bg, 0, 255).astype(np.uint8)                        # (H,W,3)

    results = []
    for t in range(n):
        frame = frames[t]
        mask = masks[t]
        if mask.max() == 0:
            results.append(frame.copy())
            continue

        # If we aggregated in warped space, warp the reference bg back into
        # frame `t`'s coordinate system so pixel lookups land correctly.
        if flow_warp and t != (n // 2):
            try:
                bg_for_t = _warp_to_reference(bg, frame)
            except Exception as exc:
                logger.debug(f"Flow back-warp fell back for frame {t}: {exc}")
                bg_for_t = bg
        else:
            bg_for_t = bg

        mask_bool = mask > 0
        has_exposure = mask_bool & (coverage >= min_coverage)
        no_exposure = mask_bool & (coverage < min_coverage)

        filled = frame.copy()
        if has_exposure.any():
            filled[has_exposure] = bg_for_t[has_exposure]

        if no_exposure.any():
            residual = np.zeros_like(mask)
            residual[no_exposure] = 255
            filled = _cv2_inpaint(filled, residual, 5, cv2.INPAINT_TELEA)

        if edge_ring_px > 0:
            filled = _edge_ring_color_correct(frame, filled, mask, edge_ring_px)
        results.append(_feather_blend(frame, filled, mask, feather_px))
    return results


def _temporal_background_expose(frames: List[np.ndarray], masks: List[np.ndarray],
                                 min_coverage: int = 3,
                                 use_median: bool = True,
                                 feather_px: int = 4,
                                 edge_ring_px: int = 2,
                                 flow_warp: bool = False,
                                 scene_cut_split: bool = True,
                                 scene_cut_threshold: float = 0.35,
                                 scene_cut_use_pyscenedetect: bool = False) -> List[np.ndarray]:
    """Video-inpainting primitive: for each pixel inside a frame's mask,
    look across the batch for frames where the same pixel is unmasked and
    reconstruct the true background from those exposures. Optionally splits
    the batch at scene cuts so we never aggregate background across a cut,
    and optionally uses Farneback optical flow to compensate for camera
    motion before aggregating.
    """
    if not scene_cut_split or len(frames) <= 1:
        segments = [(0, len(frames))]
    else:
        cuts = _detect_scene_cuts(
            frames, scene_cut_threshold,
            prefer_pyscenedetect=scene_cut_use_pyscenedetect,
        )
        segments = []
        for i, start in enumerate(cuts):
            end = cuts[i + 1] if i + 1 < len(cuts) else len(frames)
            segments.append((start, end))

    out: List[np.ndarray] = []
    for start, end in segments:
        sub_frames = frames[start:end]
        sub_masks = masks[start:end]
        out.extend(_tbe_single_segment(
            sub_frames, sub_masks,
            min_coverage=min_coverage,
            use_median=use_median,
            feather_px=feather_px,
            edge_ring_px=edge_ring_px,
            flow_warp=flow_warp,
        ))
    return out


class STTNInpainter(BaseInpainter):
    """Temporal-propagation video inpainting. Reconstructs the true background
    behind the mask by sampling adjacent frames where the pixel is unmasked
    (Temporal Background Exposure). Falls back to cv2 inpainting only for
    pixels that are masked in every frame of the batch.
    """

    def __init__(self, device: str = "cuda:0", config: ProcessingConfig = None):
        self.device = device
        self.config = config or ProcessingConfig()

    def inpaint(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        if self.config.tbe_enable and len(frames) > 1:
            return _temporal_background_expose(
                frames, masks,
                min_coverage=max(1, self.config.tbe_min_coverage),
                use_median=self.config.tbe_use_median,
                feather_px=self.config.mask_feather_px,
                edge_ring_px=self.config.edge_ring_px,
                flow_warp=self.config.tbe_flow_warp,
                scene_cut_split=self.config.tbe_scene_cut_split,
                scene_cut_threshold=self.config.tbe_scene_cut_threshold,
                scene_cut_use_pyscenedetect=self.config.tbe_scene_cut_use_pyscenedetect,
            )
        # Single-frame batch: fall back to cv2 with feathered blend
        out = []
        for f, m in zip(frames, masks):
            filled = _cv2_inpaint(f, m, 3, cv2.INPAINT_TELEA)
            if self.config.edge_ring_px > 0:
                filled = _edge_ring_color_correct(f, filled, m, self.config.edge_ring_px)
            out.append(_feather_blend(f, filled, m, self.config.mask_feather_px))
        return out


class LAMAInpainter(BaseInpainter):
    """LAMA-based image inpainting. Uses simple-lama-inpainting if available."""

    def __init__(self, device: str = "cuda:0", config: ProcessingConfig = None):
        self.device = device
        self.config = config or ProcessingConfig()
        self._lama = None
        self._load_model()

    def _load_model(self):
        try:
            from simple_lama_inpainting import SimpleLama
            self._lama = SimpleLama()
            logger.info("LaMa neural inpainting model loaded (simple-lama-inpainting)")
            # RM-49: best-effort SHA-256 check of the on-disk weights.
            # Scans the standard torch.hub cache directories for
            # `big-lama.pt` and compares against the vendored hash.
            # A mismatch logs a warning; we do not refuse to use the
            # already-loaded model because that would punish users on
            # a network where the upstream CDN has rotated weights
            # newer than our table.
            try:
                from backend.model_hashes import (
                    candidate_weight_dirs as _cands,
                    verify_weight_file as _verify,
                )
                for cache_dir in _cands():
                    for path in cache_dir.glob("**/big-lama*.pt"):
                        _verify(path)
                        break
            except Exception as exc:
                logger.debug(f"Weight verification skipped: {exc}")
        except ImportError:
            logger.warning("simple-lama-inpainting not installed, LAMA will use OpenCV fallback. "
                          "Install with: pip install simple-lama-inpainting")
        except Exception as e:
            logger.warning(f"LaMa model load failed: {e}")

    def inpaint(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        feather = self.config.mask_feather_px
        ring = self.config.edge_ring_px
        if self._lama is not None:
            raw = self._inpaint_lama(frames, masks)
        else:
            raw = [_cv2_inpaint(f, m, 7, cv2.INPAINT_NS) for f, m in zip(frames, masks)]
        out = []
        for f, r, m in zip(frames, raw, masks):
            if ring > 0 and m.max() > 0:
                r = _edge_ring_color_correct(f, r, m, ring)
            out.append(_feather_blend(f, r, m, feather))
        return out

    def _inpaint_lama(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        from PIL import Image
        results = []
        for frame, mask in zip(frames, masks):
            if mask.max() == 0:
                results.append(frame.copy())
                continue
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                pil_mask = Image.fromarray(mask)
                result_pil = self._lama(pil_image, pil_mask)
                result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
                results.append(result_bgr)
            except Exception as e:
                logger.warning(f"LaMa inpaint failed for frame, falling back to cv2: {e}")
                results.append(_cv2_inpaint(frame, mask, 7, cv2.INPAINT_NS))
        return results


class ProPainterInpainter(BaseInpainter):
    """Motion-robust video inpainting. Uses Temporal Background Exposure with
    a higher coverage bar and median aggregation (more tolerant of motion,
    matches ProPainter's niche of high-motion footage). Pixels with
    insufficient temporal exposure are refined with LaMa when available, else
    cv2. Designed to be faster than ProPainter while producing comparable
    quality for sparse occluders (subtitles, watermarks, logos).
    """

    def __init__(self, device: str = "cuda:0", config: ProcessingConfig = None):
        self.device = device
        self.config = config or ProcessingConfig()
        self._lama = None
        try:
            from simple_lama_inpainting import SimpleLama
            self._lama = SimpleLama()
            logger.info("ProPainter path will use LaMa for residual refinement")
        except Exception:
            pass

    def inpaint(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        feather = self.config.mask_feather_px
        if self.config.tbe_enable and len(frames) > 1:
            results = _temporal_background_expose(
                frames, masks,
                min_coverage=max(2, self.config.tbe_min_coverage + 1),
                use_median=True,
                feather_px=feather,
                edge_ring_px=self.config.edge_ring_px,
                flow_warp=self.config.tbe_flow_warp,
                scene_cut_split=self.config.tbe_scene_cut_split,
                scene_cut_threshold=self.config.tbe_scene_cut_threshold,
                scene_cut_use_pyscenedetect=self.config.tbe_scene_cut_use_pyscenedetect,
            )
            # Residual refinement with LaMa for pixels still visually rough
            if self._lama is not None:
                from PIL import Image
                refined = []
                for frame, inpainted, mask in zip(frames, results, masks):
                    if mask.max() == 0:
                        refined.append(inpainted)
                        continue
                    try:
                        frame_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)
                        pil_mask = Image.fromarray(mask)
                        lama_out = self._lama(pil_image, pil_mask)
                        bgr = cv2.cvtColor(np.array(lama_out), cv2.COLOR_RGB2BGR)
                        # Blend TBE (temporal) and LaMa (spatial) 65/35 -- TBE
                        # carries accurate background, LaMa kills ringing.
                        blend = cv2.addWeighted(inpainted, 0.65, bgr, 0.35, 0)
                        refined.append(_feather_blend(frame, blend, mask, feather))
                    except Exception:
                        refined.append(inpainted)
                return refined
            return results
        out = []
        for f, m in zip(frames, masks):
            filled = _cv2_inpaint(f, m, 5, cv2.INPAINT_TELEA)
            out.append(_feather_blend(f, filled, m, feather))
        return out


# RFP-L-2: each built-in inpainter registers itself below so the
# dispatch in SubtitleRemover._create_inpainter no longer needs an
# if-elif chain. Opt-in third-party backends can `register()` from
# their own module to inject a new mode without modifying core code.
from backend import inpainter_registry as _inpainter_registry


class AutoInpainter(BaseInpainter):
    """Per-batch routing between TBE (fast, temporal) and LaMa (robust,
    spatial). Computes a coverage score on the batch -- how many masked
    pixels are unmasked in at least one other frame -- and picks TBE for
    well-exposed batches, LaMa otherwise. Keeps both inpainters loaded
    lazily so single-use batches don't pay for the other path.

    B-5: when LaMa has been idle for `lama_unload_after_n_tbe` consecutive
    TBE batches, drop the reference so the weight tensor can be GC'd /
    moved off the GPU. A later hard batch re-loads on demand.
    """

    # Number of consecutive TBE-routed batches after which we drop a
    # previously-lazy-loaded LaMa. Long videos that hit one hard batch
    # early otherwise pinned ~1.5 GB VRAM for the rest of the run.
    LAMA_IDLE_UNLOAD_AFTER = 50

    def __init__(self, device: str = "cuda:0", config: ProcessingConfig = None):
        self.device = device
        self.config = config or ProcessingConfig()
        self._sttn = STTNInpainter(device, self.config)
        self._lama: Optional[LAMAInpainter] = None
        # Consecutive TBE batches since the last LaMa-routed batch. Used
        # to decide when to unload LaMa to reclaim VRAM.
        self._tbe_streak: int = 0

    def _ensure_lama(self) -> LAMAInpainter:
        if self._lama is None:
            self._lama = LAMAInpainter(self.device, self.config)
        return self._lama

    def _maybe_unload_lama(self) -> None:
        if self._lama is None:
            return
        if self._tbe_streak < self.LAMA_IDLE_UNLOAD_AFTER:
            return
        logger.info(
            f"AUTO: unloading idle LaMa after {self._tbe_streak} TBE batches"
        )
        self._lama = None
        try:
            import gc as _gc
            _gc.collect()
        except Exception:
            pass
        try:
            import torch as _torch
            if hasattr(_torch, "cuda") and _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _exposure_score(masks: List[np.ndarray]) -> float:
        """Fraction of masked pixels that are unmasked in >=1 other frame.
        Higher = easier for TBE. Range [0, 1]."""
        if len(masks) < 2:
            return 0.0
        stack = np.stack(masks, axis=0)                # (T,H,W)
        unmasked = (stack == 0)                         # (T,H,W)
        any_union = unmasked.any(axis=0)                # (H,W)
        ever_masked = (stack > 0).any(axis=0)           # (H,W)
        total = int(ever_masked.sum())
        if total == 0:
            return 1.0
        exposed = int((ever_masked & any_union).sum())
        return exposed / float(total)

    def inpaint(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        threshold = self.config.auto_exposure_threshold
        score = self._exposure_score(masks)
        if score >= threshold:
            logger.debug(f"AUTO: TBE path (exposure={score:.2f} >= {threshold:.2f})")
            self._tbe_streak += 1
            self._maybe_unload_lama()
            return self._sttn.inpaint(frames, masks)
        logger.debug(f"AUTO: LaMa path (exposure={score:.2f} < {threshold:.2f})")
        self._tbe_streak = 0
        return self._ensure_lama().inpaint(frames, masks)


# =============================================================================
# HARDWARE ENCODER DETECTION
# =============================================================================

def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural Similarity between two BGR frames. Mean over the three
    channels. Standard formulation (C1, C2 = (0.01*255)^2, (0.03*255)^2).
    Flat-colour regions where the variance and covariance are all zero can
    still drive (num/den) close to 0/0; we wrap in errstate + nan_to_num so
    the report never yields NaN or inf.
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


def _probe_keyframe_indices(video_path: str) -> Optional[set]:
    """Use ffprobe to list the decode-order indices of keyframes (I-frames)
    in a video. Returns None if ffprobe is unavailable or the probe fails.

    Each non-blank line of ffprobe's `-show_entries frame=...` output with
    `-select_streams v:0` corresponds to exactly one video frame in decode
    order, which matches how cv2.VideoCapture walks the stream. We build a
    set of line indices whose `key_frame` column is '1'. Blank lines and
    malformed rows are skipped rather than shifting the index."""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'frame=key_frame',
            '-of', 'csv=print_section=0', video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            if result.stderr:
                logger.debug(f"ffprobe keyframe scan stderr: {result.stderr.strip()[:400]}")
            return None
        keyframe_indices = set()
        frame_idx = 0
        for raw in result.stdout.splitlines():
            line = raw.strip()
            if not line:
                continue
            first = line.split(',', 1)[0].strip()
            if first in ('0', '1'):
                if first == '1':
                    keyframe_indices.add(frame_idx)
                frame_idx += 1
        return keyframe_indices if keyframe_indices else None
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        logger.warning("ffprobe keyframe scan timed out; falling back to pHash skip")
        return None
    except Exception as exc:
        logger.warning(f"ffprobe keyframe scan failed: {exc}")
        return None


def _probe_is_interlaced(video_path: str) -> bool:
    """Sample 200 frames via ffmpeg idet filter and return True if the
    majority report as interlaced. Cheap; skips on ffmpeg failure."""
    try:
        cmd = [
            'ffmpeg', '-hide_banner', '-nostats', '-i', video_path,
            '-vf', 'idet', '-frames:v', '200', '-an', '-f', 'null', '-',
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        stderr = result.stderr
        import re as _re
        m = _re.search(r'Multi frame detection:.*TFF:\s*(\d+).*BFF:\s*(\d+).*Progressive:\s*(\d+)',
                        stderr, _re.DOTALL)
        if m:
            tff, bff, prog = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return (tff + bff) > prog
    except Exception:
        pass
    return False


def _deinterlace_to_temp(src: str, temp_dir: str) -> str:
    """Run `ffmpeg -vf yadif` to produce a progressive copy of the input.
    Returns the path to the temp file. Caller is responsible for cleanup
    via the temp_dir lifecycle."""
    dst = os.path.join(temp_dir, "deinterlaced.mp4")
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-nostats',
        '-i', src,
        '-vf', 'yadif=1',
        '-c:v', 'libx264', '-crf', '16', '-preset', 'veryfast',
        '-c:a', 'copy', dst,
    ]
    timeout = _ffmpeg_subprocess_timeout(_probe_duration_seconds(src))
    subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
    return dst


class _FrameSequenceCapture:
    """cv2.VideoCapture-shaped adapter for a directory of images.

    Walks the directory in sorted filename order, treats each image as a
    single frame at a synthesised FPS. Mirrors enough of the
    VideoCapture surface that `process_video` can use it transparently:
    `.isOpened()`, `.read()`, `.set(CAP_PROP_POS_FRAMES, idx)`, and
    `.get(CAP_PROP_*)` for FPS / WIDTH / HEIGHT / FRAME_COUNT.

    Width / height come from the first image; mid-sequence size changes
    are letterboxed to the first-frame dims so the encoder pipeline
    never sees a dimension shift it can't handle. Useful for VFX
    workflows that never touch an mp4 (DPX/EXR/PNG sequences).
    """

    SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

    def __init__(self, dir_path: str, fps: float = 24.0):
        self._dir = Path(dir_path)
        if not self._dir.is_dir():
            raise ValueError(f"Not a directory: {dir_path}")
        self._files: List[Path] = sorted(
            p for p in self._dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTS
        )
        if not self._files:
            raise ValueError(
                f"No supported image files in {dir_path} "
                f"(expected one of {sorted(self.SUPPORTED_EXTS)})"
            )
        first = cv2.imread(str(self._files[0]))
        if first is None:
            raise ValueError(f"Could not read first frame: {self._files[0]}")
        self._h, self._w = first.shape[:2]
        self._fps = max(1.0, float(fps))
        self._pos = 0

    def isOpened(self) -> bool:
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._w)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._h)
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self._files))
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self._pos)
        return 0.0

    def set(self, prop, value) -> bool:
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._pos = max(0, min(len(self._files), int(value)))
            return True
        return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._pos >= len(self._files):
            return False, None
        frame = cv2.imread(str(self._files[self._pos]))
        self._pos += 1
        if frame is None:
            return False, None
        if frame.shape[:2] != (self._h, self._w):
            frame = cv2.resize(
                frame, (self._w, self._h), interpolation=cv2.INTER_AREA
            )
        return True, frame

    def release(self) -> None:
        # No external resources held; release is a no-op.
        return None


def _open_capture(path: str, hw_accel: str = "off", *,
                  input_fps: float = 24.0):
    """Open a frame source with an optional hardware-acceleration hint.

    If `path` is a directory, returns a `_FrameSequenceCapture` adapter
    that treats the contained images as frames at `input_fps`. Otherwise
    opens a real `cv2.VideoCapture` with the requested HW-accel hint.

    `hw_accel` is one of: "off" (default; status quo), "auto"/"any" (let
    cv2 pick the best available backend), "d3d11" (Windows DXVA2/D3D11VA),
    "vaapi" (Linux), or "mfx" (Intel Media SDK).

    OpenCV 4.7+ has a known issue where the HW path can return None frames
    against some FFmpeg builds (opencv/opencv#25185). To stay safe, we probe
    one frame after open; if it fails we re-open with the software backend
    and warn the user once. Caller treats the returned object as a plain
    VideoCapture regardless of the path taken.
    """
    # Directory input -> frame-sequence adapter; HW-accel hint does not apply.
    if Path(path).is_dir():
        logger.info(f"Frame-sequence input detected at {path} (fps={input_fps})")
        return _FrameSequenceCapture(path, fps=input_fps)
    if hw_accel in (None, "", "off"):
        return cv2.VideoCapture(path)
    accel_map = {
        "any": getattr(cv2, "VIDEO_ACCELERATION_ANY", 1),
        "auto": getattr(cv2, "VIDEO_ACCELERATION_ANY", 1),
        "d3d11": getattr(cv2, "VIDEO_ACCELERATION_D3D11", 2),
        "vaapi": getattr(cv2, "VIDEO_ACCELERATION_VAAPI", 3),
        "mfx": getattr(cv2, "VIDEO_ACCELERATION_MFX", 4),
    }
    accel_value = accel_map.get(hw_accel, accel_map["any"])
    try:
        cap = cv2.VideoCapture(
            path,
            cv2.CAP_FFMPEG,
            [cv2.CAP_PROP_HW_ACCELERATION, accel_value],
        )
        if cap.isOpened():
            ok, _frame = cap.read()
            if ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return cap
            cap.release()
            logger.warning(
                f"HW-accelerated decode '{hw_accel}' opened but returned no "
                f"frames (known cv2/FFmpeg issue); falling back to software."
            )
    except Exception as exc:
        logger.warning(
            f"HW-accelerated decode '{hw_accel}' raised {exc}; "
            f"falling back to software."
        )
    return cv2.VideoCapture(path)


class _PrefetchReader:
    """Background frame reader that wraps a cv2.VideoCapture.

    Exposes the same `.read()` / `.release()` shape as the underlying
    capture so the main loop can swap one for the other with no other
    changes. A daemon worker thread keeps a bounded queue full while the
    main thread runs detection + inpainting; reads release the GIL inside
    libavformat / FFmpeg, so plain threading is enough to overlap I/O
    with compute.

    Ownership rules: once a `_PrefetchReader` wraps a capture, the
    caller MUST NOT touch the underlying cv2 object until `.release()`
    has returned -- `.set()` / `.get()` / `.read()` on the wrapped cap
    from the main thread will race the worker.
    """

    _STOP = object()  # sentinel pushed by the worker on EOF or stop

    def __init__(self, cap, *, max_frames: int, queue_size: int = 16):
        self._cap = cap
        self._max = max(0, int(max_frames))
        self._q: "queue.Queue" = queue.Queue(maxsize=max(2, int(queue_size)))
        self._stop = threading.Event()
        self._exhausted = False
        self._thread = threading.Thread(
            target=self._loop, name="vsr-prefetch", daemon=True,
        )
        self._thread.start()

    def _loop(self) -> None:
        try:
            for _ in range(self._max):
                if self._stop.is_set():
                    break
                ret, frame = self._cap.read()
                if not ret:
                    break
                # Put with a poll loop so a hung consumer + stop_event
                # combination doesn't deadlock the worker forever.
                while not self._stop.is_set():
                    try:
                        self._q.put((True, frame), timeout=0.25)
                        break
                    except queue.Full:
                        continue
        except Exception as exc:  # pragma: no cover -- best-effort logging
            logger.warning(f"Prefetch reader crashed: {exc}")
        finally:
            try:
                self._q.put(self._STOP, timeout=1.0)
            except queue.Full:
                pass

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._exhausted:
            return False, None
        item = self._q.get()
        if item is self._STOP:
            self._exhausted = True
            return False, None
        return item

    def release(self) -> None:
        self._stop.set()
        # Drain so the worker can push its sentinel without blocking.
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=2.0)
        try:
            self._cap.release()
        except Exception:
            pass

    # Pass-through metadata accessors so the main loop can still query
    # FPS / dimensions before the worker starts consuming the cap.
    def isOpened(self) -> bool:
        return self._cap.isOpened()

    def get(self, prop):
        return self._cap.get(prop)


# Register the four built-in inpainters with the plugin registry so the
# dispatch in SubtitleRemover._create_inpainter is data-driven rather
# than a hardcoded if/elif chain. New backends can land as standalone
# modules that import `inpainter_registry.register` and a BaseInpainter
# subclass.
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


class _LosslessIntermediateWriter:
    """Streams BGR frames to ffmpeg via stdin and writes a lossless FFV1
    intermediate (`.mkv` container).

    Replaces the legacy `cv2.VideoWriter_fourcc(*'mp4v')` intermediate used
    in v3.12 and earlier. The previous pipeline double-encoded every
    output (mp4v intermediate -> libx264/NVENC final), so the inpainted
    regions users were inspecting carried two generations of lossy
    compression artefacts. FFV1 is mathematically lossless, so the final
    encode pass is the only lossy step.

    The writer exposes a `write(frame)` / `release()` shape identical to
    `cv2.VideoWriter` so callers do not have to special-case the
    intermediate; if `ffmpeg` is unavailable on PATH or the Popen fails,
    we fall back to the legacy mp4v writer with a logged warning so the
    pipeline keeps working (just at the old quality).
    """

    def __init__(self, path: str, width: int, height: int, fps: float):
        self._path = path
        self._width = int(width)
        self._height = int(height)
        # Avoid passing 0 / NaN to ffmpeg; downstream raw-stream rate must
        # be positive.
        try:
            fps_f = float(fps)
        except (TypeError, ValueError):
            fps_f = 30.0
        if not np.isfinite(fps_f) or fps_f <= 0.0:
            fps_f = 30.0
        self._fps = fps_f
        self._proc: Optional[subprocess.Popen] = None
        self._fallback: Optional[cv2.VideoWriter] = None
        self._opened = False
        self._lossless = False
        self._open()

    def _open(self):
        if shutil.which("ffmpeg") is None:
            self._open_fallback("ffmpeg not on PATH")
            return
        try:
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{self._width}x{self._height}",
                "-r", f"{self._fps:.6f}",
                "-i", "-",
                "-c:v", "ffv1", "-level", "3", "-coder", "1",
                "-context", "1", "-g", "1", "-slices", "16",
                "-slicecrc", "1",
                self._path,
            ]
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            self._opened = True
            self._lossless = True
            logger.info(
                f"Intermediate writer: FFV1 lossless via ffmpeg stdin "
                f"({self._width}x{self._height} @ {self._fps:.2f} fps)"
            )
        except Exception as exc:
            self._open_fallback(f"ffmpeg Popen failed: {exc}")

    def _open_fallback(self, reason: str) -> None:
        logger.warning(
            f"Lossless intermediate unavailable ({reason}); "
            f"falling back to mp4v writer."
        )
        # Legacy fallback path also lives in .mkv to avoid forcing callers
        # to know about the codec choice. cv2.VideoWriter accepts the
        # extension agnostic of the codec on most builds.
        # Use the original extension so the downstream reader knows what
        # to expect; mp4v inside mkv may not play back, so we pick mp4.
        fallback_path = self._path
        if fallback_path.lower().endswith(".mkv"):
            fallback_path = fallback_path[:-4] + ".mp4"
            self._path = fallback_path
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._fallback = cv2.VideoWriter(
            fallback_path, fourcc, self._fps, (self._width, self._height)
        )
        self._opened = self._fallback.isOpened()
        self._lossless = False

    @property
    def path(self) -> str:
        return self._path

    @property
    def lossless(self) -> bool:
        return self._lossless

    def isOpened(self) -> bool:
        return self._opened

    def write(self, frame: np.ndarray) -> None:
        if frame is None:
            return
        if self._proc is not None and self._proc.stdin is not None:
            try:
                self._proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError) as exc:
                # ffmpeg died mid-stream. Drain stderr for diagnostics and
                # promote the failure so process_video can decide whether
                # to abort the file.
                stderr_excerpt = ""
                if self._proc.stderr is not None:
                    try:
                        stderr_excerpt = self._proc.stderr.read().decode(
                            "utf-8", errors="replace")[-400:]
                    except Exception:
                        pass
                logger.error(
                    f"FFV1 ffmpeg stdin broke after writing frames: {exc}"
                    + (f"\nffmpeg stderr: {stderr_excerpt}" if stderr_excerpt else "")
                )
                raise
        elif self._fallback is not None:
            self._fallback.write(frame)

    def release(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin is not None:
                    self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                logger.warning("FFV1 ffmpeg flush timeout; killing")
                self._proc.kill()
                try:
                    self._proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            self._proc = None
        if self._fallback is not None:
            try:
                self._fallback.release()
            except Exception:
                pass
            self._fallback = None


def _detect_hw_encoder(codec: str = "h264") -> Optional[str]:
    """Probe FFmpeg for hardware encoder availability. Returns encoder name or None.

    F-8: `codec` lets callers ask for the HW encoder family that
    matches the user's chosen output codec. Defaults to h264 so legacy
    callers keep their behaviour.
    """
    family = {
        "h264": ("h264_nvenc", "h264_qsv", "h264_amf"),
        "h265": ("hevc_nvenc", "hevc_qsv", "hevc_amf"),
        "av1":  ("av1_nvenc",  "av1_qsv",  "av1_amf"),
    }.get(codec, ("h264_nvenc", "h264_qsv", "h264_amf"))
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True, text=True, timeout=10
        )
        for encoder in family:
            if encoder in result.stdout:
                logger.info(f"Hardware encoder available: {encoder}")
                return encoder
    except Exception:
        pass
    return None


# =============================================================================
# MAIN SUBTITLE REMOVER
# =============================================================================

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
                     padding: int = 5) -> np.ndarray:
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
            mask = self._create_mask(image.shape, boxes)
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
                        detected_boxes = self.detector.detect(frame, self.config.detection_threshold)
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

                    mask = self._create_mask(frame.shape, boxes)
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

    def _get_encode_args(self) -> List[str]:
        """Return FFmpeg video encoder arguments, preferring hardware encoding."""
        if self._hw_encoder and self.config.use_hw_encode:
            if 'nvenc' in self._hw_encoder:
                return ['-c:v', self._hw_encoder, '-preset', 'p4',
                        '-cq', str(self.config.output_quality)]
            elif 'qsv' in self._hw_encoder:
                return ['-c:v', self._hw_encoder,
                        '-global_quality', str(self.config.output_quality)]
            elif 'amf' in self._hw_encoder:
                return ['-c:v', self._hw_encoder,
                        '-quality', 'balanced',
                        '-rc', 'cqp', '-qp', str(self.config.output_quality)]
        # F-8: software fallback honours the chosen output codec.
        codec = self.config.output_codec
        if codec == "h265":
            return ['-c:v', 'libx265', '-crf', str(self.config.output_quality),
                    '-preset', 'medium']
        if codec == "av1":
            # SVT-AV1's CRF range tops out at 63; clamp our [0-51] scale
            # into the encoder's valid window. -preset 8 is the
            # speed/quality midpoint for libsvtav1.
            crf = min(63, self.config.output_quality)
            return ['-c:v', 'libsvtav1', '-crf', str(crf), '-preset', '8']
        return ['-c:v', 'libx264', '-crf', str(self.config.output_quality),
                '-preset', 'medium']

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


def _default_checkpoint_dir() -> Path:
    """Where to store per-file crash-resume markers."""
    base = Path(os.environ.get("APPDATA", Path.home() / ".config")) / "VideoSubtitleRemoverPro" / "checkpoints"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _checkpoint_key(input_path: str, output_path: str) -> str:
    """Stable identifier for a (input, output, size, mtime) pair. A size/mtime
    change on the input invalidates the checkpoint so users don't skip a
    freshly re-downloaded file by accident."""
    import hashlib
    try:
        stat = os.stat(input_path)
        fingerprint = f"{input_path}|{output_path}|{stat.st_size}|{int(stat.st_mtime)}"
    except OSError:
        fingerprint = f"{input_path}|{output_path}"
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:24]


def _checkpoint_is_done(ckpt_dir: Path, key: str, output_path: str) -> bool:
    marker = ckpt_dir / f"{key}.done"
    return marker.exists() and Path(output_path).exists()


def _checkpoint_mark_done(ckpt_dir: Path, key: str):
    marker = ckpt_dir / f"{key}.done"
    try:
        _write_text_atomic(marker, "ok")
    except Exception as exc:
        logger.warning(f"Could not write checkpoint {marker}: {exc}")


def _load_json_config(path: str) -> dict:
    """Load a JSON config file of {field: value} pairs for ProcessingConfig."""
    size = os.path.getsize(path)
    if size > 1 * 1024 * 1024:  # 1 MB sanity cap
        raise ValueError(f"config file is too large ({size:,} bytes); expected a small JSON object")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("config file must contain a top-level JSON object")
    return payload


def _apply_auto_band_override(remover, input_path: str, *, auto_band: bool,
                              base_subtitle_area, base_subtitle_areas):
    """Reset per-file region overrides before optionally probing a fresh band."""
    remover.config.subtitle_area = base_subtitle_area
    remover.config.subtitle_areas = list(base_subtitle_areas) if base_subtitle_areas else None
    if not auto_band or base_subtitle_area or base_subtitle_areas:
        return base_subtitle_area
    band = remover.detect_subtitle_band(input_path)
    remover.config.subtitle_area = band
    return band


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Video Subtitle Remover Pro CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m backend.processor -i input.mp4 -o output.mp4 -m sttn --lang en\n"
            "  python -m backend.processor --pattern \"inputs/*.mp4\" --out-dir cleaned --mode auto"
        ),
    )
    parser.add_argument("--input", "-i", help="Input file path")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--pattern", help="Glob pattern for batch mode (e.g. 'inputs/*.mp4')")
    parser.add_argument("--out-dir", help="Output directory for batch mode")
    parser.add_argument("--config", help="JSON config file (key=value pairs overriding CLI defaults)")
    parser.add_argument("--preset", metavar="NAME",
                       help="Apply a built-in or user preset by name. Listed by "
                            "--list-presets. Resolved before CLI flags so the CLI "
                            "always wins on a conflict.")
    parser.add_argument("--list-presets", action="store_true",
                       help="Print every known preset (built-in + user) and exit.")
    parser.add_argument("--checkpoint-dir", default=None,
                       help="Checkpoint dir for crash-resume (default: %%APPDATA%%/.../checkpoints)")
    parser.add_argument("--no-resume", action="store_true",
                       help="Ignore any existing checkpoint and reprocess every file")
    parser.add_argument("--mode", "-m", default="sttn",
                       choices=["sttn", "lama", "propainter", "auto", "migan"],
                       help="Inpainting algorithm (auto routes per batch; "
                            "migan requires VSR_MIGAN_ONNX + onnxruntime).")
    parser.add_argument("--gpu", "-g", type=int, default=0, help="GPU device ID (-1 for CPU)")
    parser.add_argument("--lang", "-l", default="en", help="Detection language (en, ch, ja, ko, etc.)")
    parser.add_argument("--skip-detection", action="store_true",
                       help="Skip automatic detection (STTN only)")
    parser.add_argument("--fast", action="store_true", help="Fast mode (LAMA only)")
    parser.add_argument("--no-audio", action="store_true", help="Don't preserve audio")
    parser.add_argument("--crf", type=int, default=23, help="Output CRF quality (15-35)")
    parser.add_argument("--start", type=float, default=0, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=0, help="End time in seconds (0=full)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold (0.1-1.0)")
    parser.add_argument("--vertical", action="store_true",
                       help="Vertical-text mode: rotate frames 90 CCW before "
                            "OCR so JP tategaki / classical CN columns parse. "
                            "Boxes rotate back to the original frame.")
    parser.add_argument("--frame-skip", type=int, default=0,
                       help="Reuse detection mask for N frames between detections (0=every frame)")
    parser.add_argument("--mask-dilate", type=int, default=8,
                       help="Mask dilation in pixels for cleaner removal (0=off)")
    parser.add_argument("--no-hw-encode", action="store_true",
                       help="Disable hardware encoding (force libx264)")
    parser.add_argument("--codec", default="h264",
                       choices=["h264", "h265", "av1"],
                       help="Output video codec. h264 (default) maximises "
                            "compatibility; h265 / av1 cut bitrate ~50%% on 4K. "
                            "HEVC + AV1 use NVENC/QSV/AMF when available with "
                            "libx265 / libsvtav1 software fallback.")
    parser.add_argument("--mask-feather", type=int, default=4,
                       help="Gaussian edge feathering in pixels (0=off)")
    parser.add_argument("--edge-ring", type=int, default=2,
                       help="Edge-ring colour match width in pixels (0=off)")
    parser.add_argument("--flow-warp", action="store_true",
                       help="Farneback flow-warp TBE frames before aggregation")
    parser.add_argument("--no-scene-split", action="store_true",
                       help="Disable scene-cut splitting inside TBE batches")
    parser.add_argument("--pyscenedetect", action="store_true",
                       help="Prefer PySceneDetect's AdaptiveDetector over the "
                            "built-in histogram cut detector (opt-in dep: "
                            "`pip install scenedetect`). Handles dissolves "
                            "and flashes the histogram heuristic mis-fires on.")
    parser.add_argument("--no-tbe", action="store_true",
                       help="Disable Temporal Background Exposure (STTN/ProPainter use cv2)")
    parser.add_argument("--no-adaptive-batch", action="store_true",
                       help="Disable VRAM-probe-driven batch sizing")
    parser.add_argument("--export-srt", action="store_true",
                       help="Write an .srt sidecar with detected text")
    parser.add_argument("--export-mask", action="store_true",
                       help="Write a B/W .mask.mp4 debug video")
    parser.add_argument("--auto-band", action="store_true",
                       help="Auto-detect the dominant subtitle band before processing")
    parser.add_argument("--no-kalman", action="store_true",
                       help="Disable Kalman detection smoothing")
    parser.add_argument("--no-phash", action="store_true",
                       help="Disable perceptual-hash adaptive mask reuse")
    parser.add_argument("--phash-distance", type=int, default=4,
                       help="pHash Hamming distance threshold for mask reuse (0-64)")
    parser.add_argument("--colour-tune", action="store_true",
                       help="Grow the mask by dominant-colour match inside each box")
    parser.add_argument("--colour-tolerance", type=int, default=25,
                       help="Lab-space colour distance tolerance for colour-tune")
    parser.add_argument("--auto-threshold", type=float, default=0.55,
                       help="AUTO-mode exposure threshold (0-1) for TBE-vs-LaMa routing")
    parser.add_argument("--deinterlace", action="store_true",
                       help="Force ffmpeg yadif deinterlace before processing")
    parser.add_argument("--no-deinterlace-detect", action="store_true",
                       help="Skip the automatic ffprobe interlacing detection")
    parser.add_argument("--keyframe-detect", action="store_true",
                       help="OCR only at video I-frames (ffprobe-probed)")
    parser.add_argument("--quality-report", action="store_true",
                       help="Compute PSNR/SSIM on a random frame sample after run")
    parser.add_argument("--quality-sheet", action="store_true",
                       help="Render a side-by-side comparison PNG alongside "
                            "the numeric quality report. Implies "
                            "--quality-report. Written to <output>.qualitysheet.png.")
    parser.add_argument("--input-fps", type=float, default=24.0, metavar="FPS",
                       help="FPS to use when the input is a directory of "
                            "images (frame sequence). Ignored for video inputs.")
    parser.add_argument("--keep-chyrons", action="store_true",
                       help="Leave persistent text (station logos, lower-thirds, "
                            "tickers) in the output. Chyron classification: a "
                            "Kalman track must match in >= --chyron-min-hits "
                            "frames to qualify.")
    parser.add_argument("--keep-subtitles", action="store_true",
                       help="Leave non-persistent text (dialogue captions) in the "
                            "output. Useful when paired with --keep-chyrons=off "
                            "for chyron-only cleaning.")
    parser.add_argument("--chyron-min-hits", type=int, default=90, metavar="N",
                       help="Frames a Kalman track must match to be classified "
                            "as a chyron. Default 90 (~3 s at 30 fps).")
    parser.add_argument("--karaoke-grouping", action="store_true",
                       help="Fuse per-syllable OCR boxes on the same line into "
                            "one composite before tracking. Closes the gap that "
                            "leaks original karaoke text through the mask.")
    parser.add_argument("--karaoke-x-gap", type=int, default=20, metavar="PX",
                       help="Max horizontal gap (px) between boxes to count as "
                            "the same karaoke line. Default 20.")
    parser.add_argument("--karaoke-y-overlap", type=float, default=0.5,
                       metavar="RATIO",
                       help="Min vertical overlap ratio (0..1) to count as the "
                            "same karaoke line. Default 0.5.")
    parser.add_argument("--loudnorm", type=float, default=0.0, metavar="LUFS",
                       help="EBU R128 loudness target in LUFS, e.g. -14 (YouTube), "
                            "-16 (Apple), -23 (broadcast). 0 disables.")
    parser.add_argument("--decode-accel", default="off",
                       choices=["off", "auto", "any", "d3d11", "vaapi", "mfx"],
                       help="Hardware-decode hint for cv2.VideoCapture. Falls "
                            "back silently to software if HW returns no frames "
                            "(known cv2/FFmpeg interop issue).")
    parser.add_argument("--single-audio", action="store_true",
                       help="Mux only the first audio stream (legacy v3.12 "
                            "behaviour). Default is now to pass through all "
                            "audio tracks (matters for Bluray/DVD rips).")
    parser.add_argument("--no-prefetch", action="store_true",
                       help="Disable the worker-thread frame prefetcher. "
                            "Strictly serial read+process; use for debugging "
                            "decode-vs-process races.")
    parser.add_argument("--prefetch-queue", type=int, default=0, metavar="N",
                       help="Bounded prefetch queue size in frames. "
                            "0 = auto (max(8, batch_size * 2)).")
    parser.add_argument("--skip-existing", action="store_true",
                       help="Skip any input whose output path already exists, "
                            "even without a checkpoint marker.")
    parser.add_argument("--validate-config", action="store_true",
                       help="Print the resolved ProcessingConfig as JSON and exit "
                            "without processing anything. Useful for shell scripts "
                            "that want to verify flags before launching a long batch.")
    parser.add_argument("--json-log", metavar="PATH",
                       help="Append a structured JSON-line log at PATH alongside "
                            "the rotating text log. Each record is one line, "
                            "jq-friendly. Useful for grepping across days of batch "
                            "jobs.")

    args = parser.parse_args()

    # Wire optional structured log before any work so the JSON sidecar
    # captures every record from arg validation onward.
    if args.json_log:
        attach_json_log(args.json_log)

    # --list-presets short-circuits everything else.
    if args.list_presets:
        from backend.presets import BUILTIN_PRESETS as _BUILTIN, load_user_presets as _load_user
        rows = []
        for name, payload in _BUILTIN.items():
            rows.append(("built-in", name, payload.get("description", "")))
        for name, payload in _load_user().items():
            if name in _BUILTIN:
                continue
            desc = payload.get("description", "") if isinstance(payload, dict) else ""
            rows.append(("user", name, desc))
        width = max((len(n) for _, n, _ in rows), default=4)
        for source, name, desc in rows:
            print(f"[{source:<8}] {name.ljust(width)}  {desc}")
        sys.exit(0)

    # F-10: resolve --preset BEFORE the CLI flags overlay so a flag
    # explicitly typed on the command line still wins on a conflict.
    # The preset's `fields` dict is applied through argparse's existing
    # defaults: we patch attributes on `args` whose default values
    # weren't supplanted by an explicit flag. This way the existing
    # `config = ProcessingConfig(... args.X ...)` block below picks up
    # preset values automatically without a second code path.
    if args.preset:
        from backend.presets import preset_fields as _preset_fields
        fields = _preset_fields(args.preset)
        if fields is None:
            parser.error(
                f"unknown preset {args.preset!r}; run --list-presets to see options"
            )
        # Map preset field names to argparse attr names. Most match
        # one-to-one; the legacy CLI uses different attr names for a
        # handful of fields. We honour the parser default sentinel so
        # only attrs the user did NOT pass on the command line are
        # overridden by the preset.
        attr_to_default = {a.dest: a.default for a in parser._actions}
        field_to_attr = {
            "mode": "mode",
            "detection_threshold": "threshold",
            "mask_dilate_px": "mask_dilate",
            "mask_feather_px": "mask_feather",
            "edge_ring_px": "edge_ring",
            "tbe_flow_warp": "flow_warp",
            "tbe_scene_cut_split": None,        # negative flag (--no-scene-split)
            "colour_tune_enable": "colour_tune",
            "colour_tune_tolerance": "colour_tolerance",
            "kalman_tracking": None,            # negative flag (--no-kalman)
            "phash_skip_enable": None,          # negative flag (--no-phash)
            "phash_skip_distance": "phash_distance",
            "auto_band": None,                  # GUI-only today
            "detection_frame_skip": "frame_skip",
        }
        for fname, value in fields.items():
            if fname == "mode":
                # Preset uses GUI labels (e.g. "STTN" / "Auto" / "ProPainter").
                # The CLI parser expects lowercase choices.
                if getattr(args, "mode", None) == attr_to_default.get("mode"):
                    args.mode = str(value).lower().replace(" ", "")
                continue
            attr = field_to_attr.get(fname, fname)
            if attr is None or not hasattr(args, attr):
                continue
            if getattr(args, attr) == attr_to_default.get(attr):
                setattr(args, attr, value)
        logger.info(f"Applied preset: {args.preset}")

    # Validate: either --input or --pattern must be given. `--validate-config`
    # bypasses this so the user can dry-run flag combinations without supplying
    # any file at all.
    if not args.validate_config:
        if not args.input and not args.pattern:
            parser.error("one of --input or --pattern is required")
        if args.input and args.pattern:
            parser.error("--input and --pattern are mutually exclusive")
        if args.pattern and not args.out_dir:
            parser.error("--pattern requires --out-dir")
        if args.input and not args.output:
            parser.error("--input requires --output")
    if not 0.1 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0.1 and 1.0")
    if not 15 <= args.crf <= 35:
        parser.error("--crf must be between 15 and 35")
    if args.start < 0 or args.end < 0:
        parser.error("--start and --end must be zero or positive")
    if args.end and args.end < args.start:
        parser.error("--end must be greater than or equal to --start")
    if args.frame_skip < 0:
        parser.error("--frame-skip must be zero or positive")
    if args.mask_dilate < 0:
        parser.error("--mask-dilate must be zero or positive")
    if args.mask_feather < 0:
        parser.error("--mask-feather must be zero or positive")
    if args.edge_ring < 0:
        parser.error("--edge-ring must be zero or positive")
    if not 0.0 <= args.auto_threshold <= 1.0:
        parser.error("--auto-threshold must be between 0 and 1")
    if not 0 <= args.phash_distance <= 64:
        parser.error("--phash-distance must be between 0 and 64")
    if args.colour_tolerance < 0:
        parser.error("--colour-tolerance must be zero or positive")
    if args.loudnorm != 0.0 and not -70.0 <= args.loudnorm <= -5.0:
        parser.error("--loudnorm must be 0 (off) or between -70 and -5 LUFS")

    config = ProcessingConfig(
        mode=InpaintMode(args.mode),
        device=f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu",
        sttn_skip_detection=args.skip_detection,
        lama_super_fast=args.fast,
        preserve_audio=not args.no_audio,
        detection_lang=args.lang,
        detection_threshold=args.threshold,
        detection_vertical=args.vertical,
        output_quality=args.crf,
        time_start=args.start,
        time_end=args.end,
        detection_frame_skip=args.frame_skip,
        mask_dilate_px=args.mask_dilate,
        mask_feather_px=args.mask_feather,
        edge_ring_px=args.edge_ring,
        tbe_enable=not args.no_tbe,
        tbe_flow_warp=args.flow_warp,
        tbe_scene_cut_split=not args.no_scene_split,
        tbe_scene_cut_use_pyscenedetect=args.pyscenedetect,
        adaptive_batch=not args.no_adaptive_batch,
        export_srt=args.export_srt,
        export_mask_video=args.export_mask,
        kalman_tracking=not args.no_kalman,
        phash_skip_enable=not args.no_phash,
        phash_skip_distance=args.phash_distance,
        colour_tune_enable=args.colour_tune,
        colour_tune_tolerance=args.colour_tolerance,
        auto_exposure_threshold=args.auto_threshold,
        deinterlace=args.deinterlace,
        deinterlace_auto=not args.no_deinterlace_detect,
        keyframe_detection=args.keyframe_detect,
        quality_report=args.quality_report,
        use_hw_encode=not args.no_hw_encode,
        output_codec=args.codec,
        loudnorm_target=args.loudnorm,
        decode_hw_accel=args.decode_accel,
        multi_audio_passthrough=not args.single_audio,
        prefetch_decode=not args.no_prefetch,
        prefetch_queue_size=args.prefetch_queue,
        input_fps=args.input_fps,
        quality_report_sheet=args.quality_sheet,
        remove_subtitles=not args.keep_subtitles,
        remove_chyrons=not args.keep_chyrons,
        chyron_min_hits=args.chyron_min_hits,
        karaoke_grouping=args.karaoke_grouping,
        karaoke_x_gap_px=args.karaoke_x_gap,
        karaoke_y_overlap=args.karaoke_y_overlap,
    )
    config = normalize_processing_config(config)

    ffmpeg_ready = shutil.which("ffmpeg") is not None

    # Optional JSON config overlay. Applied after CLI args so a config file
    # can set fields that aren't exposed as flags (kalman_max_age,
    # tbe_scene_cut_threshold, colour_tune_tolerance, etc.).
    if args.config:
        try:
            overlay = _load_json_config(args.config)
            for k, v in overlay.items():
                if k == "mode":
                    mode_value = _coerce_backend_mode(v)
                    if isinstance(v, str) and v.strip().casefold() in {
                        "sttn", "lama", "propainter", "pro painter", "auto"
                    }:
                        config.mode = mode_value
                    else:
                        logger.warning(f"Ignoring unknown mode in config: {v}")
                    continue
                if hasattr(config, k):
                    setattr(config, k, v)
                else:
                    logger.warning(f"Ignoring unknown config field: {k}")
            config = normalize_processing_config(config)
            logger.info(f"Loaded config overlay from {args.config}")
        except Exception as exc:
            parser.error(f"Could not load --config {args.config}: {exc}")

    # `--validate-config` dry-run: print the resolved config and exit before
    # we instantiate the (heavy) detector / inpainter stack. Exit code stays
    # 0 so this is useful in shell pre-checks.
    if args.validate_config:
        resolved = {
            "mode": config.mode.value,
            "device": config.device,
            "detection_lang": config.detection_lang,
            "detection_threshold": config.detection_threshold,
            "detection_frame_skip": config.detection_frame_skip,
            "subtitle_area": list(config.subtitle_area) if config.subtitle_area else None,
            "subtitle_areas": (
                [list(r) for r in config.subtitle_areas]
                if config.subtitle_areas else None
            ),
            "mask_dilate_px": config.mask_dilate_px,
            "mask_feather_px": config.mask_feather_px,
            "edge_ring_px": config.edge_ring_px,
            "tbe_enable": config.tbe_enable,
            "tbe_min_coverage": config.tbe_min_coverage,
            "tbe_flow_warp": config.tbe_flow_warp,
            "tbe_scene_cut_split": config.tbe_scene_cut_split,
            "tbe_scene_cut_threshold": config.tbe_scene_cut_threshold,
            "kalman_tracking": config.kalman_tracking,
            "kalman_iou_threshold": config.kalman_iou_threshold,
            "kalman_max_age": config.kalman_max_age,
            "phash_skip_enable": config.phash_skip_enable,
            "phash_skip_distance": config.phash_skip_distance,
            "colour_tune_enable": config.colour_tune_enable,
            "colour_tune_tolerance": config.colour_tune_tolerance,
            "auto_exposure_threshold": config.auto_exposure_threshold,
            "deinterlace": config.deinterlace,
            "deinterlace_auto": config.deinterlace_auto,
            "keyframe_detection": config.keyframe_detection,
            "quality_report": config.quality_report,
            "adaptive_batch": config.adaptive_batch,
            "sttn_skip_detection": config.sttn_skip_detection,
            "sttn_neighbor_stride": config.sttn_neighbor_stride,
            "sttn_reference_length": config.sttn_reference_length,
            "sttn_max_load_num": config.sttn_max_load_num,
            "lama_super_fast": config.lama_super_fast,
            "time_start": config.time_start,
            "time_end": config.time_end,
            "preserve_audio": config.preserve_audio,
            "output_format": config.output_format,
            "output_quality": config.output_quality,
            "use_hw_encode": config.use_hw_encode,
            "loudnorm_target": config.loudnorm_target,
            "decode_hw_accel": config.decode_hw_accel,
            "multi_audio_passthrough": config.multi_audio_passthrough,
            "prefetch_decode": config.prefetch_decode,
            "prefetch_queue_size": config.prefetch_queue_size,
            "input_fps": config.input_fps,
            "quality_report_sheet": config.quality_report_sheet,
            "remove_subtitles": config.remove_subtitles,
            "remove_chyrons": config.remove_chyrons,
            "chyron_min_hits": config.chyron_min_hits,
            "karaoke_grouping": config.karaoke_grouping,
            "karaoke_x_gap_px": config.karaoke_x_gap_px,
            "karaoke_y_overlap": config.karaoke_y_overlap,
            "export_srt": config.export_srt,
            "export_mask_video": config.export_mask_video,
        }
        print(json.dumps({"resolved_config": resolved}, indent=2, sort_keys=True))
        sys.exit(0)

    # Reusable remover -- loaded once and fed every input in the batch
    remover = SubtitleRemover(config)
    remover.on_progress = lambda p, m: print(f"[{int(p*100):3d}%] {m}")

    print(
        "[run] "
        f"mode={config.mode.value} | device={config.device} | lang={config.detection_lang} | "
        f"audio={'on' if config.preserve_audio else 'off'} | hw_encode={'on' if config.use_hw_encode else 'off'}"
    )
    if config.preserve_audio and not ffmpeg_ready:
        print("[note] FFmpeg is not available, so outputs will be saved without original audio.")

    video_exts = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpeg', '.mpg'}
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else _default_checkpoint_dir()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    base_subtitle_area = config.subtitle_area
    base_subtitle_areas = list(config.subtitle_areas) if config.subtitle_areas else None

    def _process_one(inp: str, outp: str) -> bool:
        # --skip-existing wins over the checkpoint machinery so the user can
        # re-run a glob against a partly-populated output dir without enabling
        # the full checkpoint workflow.
        if args.skip_existing and Path(outp).exists():
            print(f"[skip] {Path(inp).name} (output exists)")
            return True
        key = _checkpoint_key(inp, outp)
        if not args.no_resume and _checkpoint_is_done(ckpt_dir, key, outp):
            print(f"[skip] {Path(inp).name} (checkpoint)")
            return True
        _apply_auto_band_override(
            remover,
            inp,
            auto_band=False,
            base_subtitle_area=base_subtitle_area,
            base_subtitle_areas=base_subtitle_areas,
        )
        ext = Path(inp).suffix.lower()
        # Directory inputs are frame sequences -- route them through the
        # video pipeline (which now opens them via _FrameSequenceCapture).
        if Path(inp).is_dir() or ext in video_exts:
            if args.auto_band:
                band = _apply_auto_band_override(
                    remover,
                    inp,
                    auto_band=True,
                    base_subtitle_area=base_subtitle_area,
                    base_subtitle_areas=base_subtitle_areas,
                )
                if band:
                    print(f"[auto-band] {Path(inp).name}: {band}")
                elif not (base_subtitle_area or base_subtitle_areas):
                    print(f"[auto-band] {Path(inp).name}: no dominant band, full-frame")
            ok = remover.process_video(inp, outp)
        else:
            ok = remover.process_image(inp, outp)
        if ok:
            _checkpoint_mark_done(ckpt_dir, key)
        return ok

    # ---- Batch mode (--pattern + --out-dir) ----
    if args.pattern:
        from glob import glob
        inputs = sorted(glob(args.pattern, recursive=True))
        inputs = [p for p in inputs if Path(p).is_file()]
        if not inputs:
            parser.error(f"No files matched pattern: {args.pattern}")
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[batch] {len(inputs)} file(s) queued | out={out_dir} | resume={'on' if not args.no_resume else 'off'}")
        failures = 0
        reserved_outputs: set[str] = set()
        try:
            for i, inp in enumerate(inputs, 1):
                src = Path(inp)
                outp = str(_choose_available_output_path(
                    out_dir / f"{src.stem}_no_sub{src.suffix}",
                    reserved_outputs,
                ))
                reserved_outputs.add(_path_key(outp))
                print(f"\n[batch] ({i}/{len(inputs)}) {src.name}")
                try:
                    ok = _process_one(inp, outp)
                except Exception as exc:
                    logger.error(f"Failed on {src.name}: {exc}")
                    ok = False
                if not ok:
                    failures += 1
        except KeyboardInterrupt:
            print("\n[batch] Interrupted by user -- partial results kept on disk.")
            sys.exit(130)
        succeeded = len(inputs) - failures
        print(f"\n[batch] finished: {succeeded}/{len(inputs)} succeeded")
        if failures:
            print("[batch] Some items need attention. Review the errors above before retrying.")
        sys.exit(0 if failures == 0 else 1)

    # ---- Single-file mode ----
    print(f"[file] source={Path(args.input).name}")
    print(f"[file] output={args.output}")
    try:
        success = _process_one(args.input, args.output)
    except KeyboardInterrupt:
        print("\n[file] Interrupted by user.")
        sys.exit(130)
    print(f"[file] {'completed' if success else 'failed'}")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
