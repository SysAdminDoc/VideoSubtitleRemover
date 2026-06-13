"""Processing configuration types and coercion for the backend.

Extracted from processor.py (RM-114 follow-up to the RFP-L-1 split) so
inpainter modules can import ``ProcessingConfig`` / ``InpaintMode``
without importing the orchestrator. The previous arrangement made every
split module back-import ``backend.processor`` at runtime, inverting
the dependency graph and turning import order load-bearing.

``backend.processor`` re-exports everything defined here, so legacy
callers (``from backend.processor import ProcessingConfig``) keep
working unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class InpaintMode(Enum):
    """Supported inpainting algorithms."""
    STTN = "sttn"
    LAMA = "lama"
    PROPAINTER = "propainter"
    AUTO = "auto"   # per-batch routing between TBE (easy) and LaMa (hard)
    MIGAN = "migan"  # opt-in ONNX backend (RM-26)


class RegisteredMode:
    """Carrier for inpaint modes that exist only in the plugin registry
    (opt-in ONNX/diffusion backends such as `diffueraser` or
    `propainter-real`). Quacks like an InpaintMode member where the
    pipeline needs one (.value / .name) without widening the enum,
    which gates the GUI for safety."""

    __slots__ = ("value",)

    def __init__(self, value: str):
        self.value = value.strip().lower()

    @property
    def name(self) -> str:
        return self.value.upper()

    def __repr__(self) -> str:
        return f"RegisteredMode({self.value!r})"

    def __eq__(self, other):
        if isinstance(other, RegisteredMode):
            return self.value == other.value
        return NotImplemented

    def __hash__(self):
        return hash(("RegisteredMode", self.value))


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
    # RM-27 / #120 Whisper fallback: when on AND an optional Whisper
    # backend is available, frames whose OCR yields no boxes get a
    # default bottom-band mask during Whisper-detected speech intervals.
    # Catches anti-aliased / motion-blurred subtitles the OCR cascade
    # misses. The default backend is faster-whisper; FFmpeg's native
    # whisper filter is opt-in and needs a local whisper.cpp ggml model.
    whisper_fallback: bool = False
    whisper_backend: str = "faster-whisper"
    whisper_model_size: str = "tiny"   # tiny / base / small / medium
    whisper_model_path: str = ""
    whisper_queue_seconds: float = 3.0
    whisper_vad_model: str = ""
    whisper_vad_threshold: float = 0.5
    whisper_min_speech_duration: float = 0.0
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
    confidence_weighted_dilation: bool = False
    confidence_dilation_scale: float = 1.5
    lama_tile_size: int = 512
    lama_tile_overlap: int = 64

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
    # RM-35: write the processed result as a numbered frame sequence
    # (directory inputs only) instead of encoding a video. The CLI flag
    # --output-frames existed before this field did, so ProcessingConfig
    # raised TypeError the moment the flag was used.
    output_frames: bool = False

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


_MODE_ALIASES = {
    "sttn": InpaintMode.STTN,
    "lama": InpaintMode.LAMA,
    "propainter": InpaintMode.PROPAINTER,
    "pro painter": InpaintMode.PROPAINTER,
    "auto": InpaintMode.AUTO,
    "migan": InpaintMode.MIGAN,
    "mi-gan": InpaintMode.MIGAN,
}


def _coerce_backend_mode(value):
    if isinstance(value, (InpaintMode, RegisteredMode)):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in _MODE_ALIASES:
            return _MODE_ALIASES[normalized]
        # Opt-in backends (ONNX / diffusion scaffolds) register extra
        # mode names at import time; honour any registered name instead
        # of silently coercing it to STTN.
        from backend import inpainter_registry as _reg
        if _reg.is_registered(normalized):
            return RegisteredMode(normalized)
        logger.warning(f"Unknown inpaint mode {value!r}; falling back to STTN")
    return InpaintMode.STTN


def is_known_backend_mode(value) -> bool:
    """True when _coerce_backend_mode would honour `value` rather than
    falling back to STTN."""
    if isinstance(value, (InpaintMode, RegisteredMode)):
        return True
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in _MODE_ALIASES:
            return True
        from backend import inpainter_registry as _reg
        return _reg.is_registered(normalized)
    return False


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
    backend = _coerce_text(config.whisper_backend, "faster-whisper", 32).lower()
    if backend in {"faster", "faster_whisper"}:
        backend = "faster-whisper"
    if backend not in {"faster-whisper", "ffmpeg"}:
        backend = "faster-whisper"
    config.whisper_backend = backend
    model_size = _coerce_text(config.whisper_model_size, "tiny", 16).lower()
    if model_size not in {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}:
        model_size = "tiny"
    config.whisper_model_size = model_size
    config.whisper_model_path = _coerce_text(config.whisper_model_path, "", 512)
    config.whisper_queue_seconds = _coerce_float(
        config.whisper_queue_seconds, 3.0, 0.02, 3600.0)
    config.whisper_vad_model = _coerce_text(config.whisper_vad_model, "", 512)
    config.whisper_vad_threshold = _coerce_float(
        config.whisper_vad_threshold, 0.5, 0.0, 1.0)
    config.whisper_min_speech_duration = _coerce_float(
        config.whisper_min_speech_duration, 0.0, 0.0, 30.0)
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
    config.confidence_weighted_dilation = _coerce_bool(
        config.confidence_weighted_dilation, False)
    config.confidence_dilation_scale = _coerce_float(
        config.confidence_dilation_scale, 1.5, 0.0, 5.0)
    config.lama_tile_size = _coerce_int(config.lama_tile_size, 512, 256, 1024)
    config.lama_tile_overlap = _coerce_int(config.lama_tile_overlap, 64, 0, 256)
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
    config.output_frames = _coerce_bool(config.output_frames, False)
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

