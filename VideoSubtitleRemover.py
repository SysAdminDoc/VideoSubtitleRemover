#!/usr/bin/env python3
"""
Video Subtitle Remover Pro
A professional Windows application for AI-powered subtitle removal from videos
and images. Based on: https://github.com/YaoFANGUK/video-subtitle-remover

Author: SysAdminDoc
See APP_VERSION for the running version -- the docstring deliberately omits
a hardcoded number so there is a single source of truth.
"""

import os
import sys
import json
import math
import uuid
import threading
import subprocess
import time
import tempfile
import logging
import logging.handlers
import traceback
from pathlib import Path
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# =============================================================================
# LOGGING SETUP -- file + stream, crash handler
# =============================================================================

APP_NAME = "Video Subtitle Remover Pro"
# Single source of truth for the app's version string. Update here and it
# propagates to the banner, header, logs, About dialog, and CHANGELOG cue.
APP_VERSION = "3.15.0"
APP_AUTHOR = "SysAdminDoc"

LOG_DIR = Path(os.environ.get("APPDATA", Path.home())) / "VideoSubtitleRemoverPro"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "vsr_pro.log"
SETTINGS_FILE = LOG_DIR / "settings.json"

# Bump VSR_SETTINGS_FORMAT whenever settings.json keys are renamed or
# semantics change. _migrate_settings() must learn the upgrade path so
# users never silently lose state on an in-place upgrade.
# Format 1 -> 2 (B-1, v3.13 GUI wiring pass): added loudnorm_target,
# multi_audio_passthrough, decode_hw_accel, prefetch_decode,
# prefetch_queue_size, input_fps, quality_report_sheet,
# remove_subtitles, remove_chyrons, chyron_min_hits, karaoke_grouping,
# karaoke_x_gap_px, karaoke_y_overlap. All have backend-default values
# so a missing key in a format-1 file resolves to the same behaviour
# users saw before the bump -- no field-rename migration needed.
VSR_SETTINGS_FORMAT = 2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=2, encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)


def crash_handler(exc_type, exc_value, exc_tb):
    """Global crash handler -- log to file and show MessageBox."""
    msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
    logger.critical(f"UNHANDLED EXCEPTION:\n{msg}")
    try:
        import tkinter.messagebox as mb
        mb.showerror("Fatal Error",
                     f"{APP_NAME} crashed.\n\n{exc_value}\n\nLog: {LOG_FILE}")
    except Exception:
        pass
    sys.__excepthook__(exc_type, exc_value, exc_tb)


sys.excepthook = crash_handler

# RM-52: opt-in crash reporting. Strictly off unless the user sets BOTH
# VSR_GLITCHTIP_DSN AND VSR_CRASH_REPORTS=1. The install() call is a
# no-op when either is missing, so default installs never phone home.
try:
    from backend.crash_reporter import install as _install_crash_reporter
    _install_crash_reporter()
except Exception:
    pass

# GUI Imports
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from tkinter import font as tkfont
except ImportError:
    logger.error("Tkinter not found. Please install Python with Tkinter support.")
    sys.exit(1)

try:
    from PIL import Image, ImageTk, ImageDraw, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("Pillow not installed. Image preview will be limited.")

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

# =============================================================================
# DESIGN TOKENS -- cohesive, premium dark theme
# =============================================================================
class Theme:
    """Design system. Dark-first, refined tonal layering, calm accents."""

    # Surfaces -- deliberate tonal ladder (BG_DARK < BG_SECONDARY < BG_CARD < BG_TERTIARY < BG_RAISED)
    BG_DARK = "#06080f"            # App background (deepest)
    BG_SECONDARY = "#0c111c"       # Main panel surface
    BG_CARD = "#121927"            # Card / inner panel
    BG_CARD_HOVER = "#182132"      # Card hovered
    BG_CARD_SELECTED = "#1a2944"   # Card selected (subtle blue tint)
    BG_TERTIARY = "#1b2438"        # Elevated field (inputs, chips)
    BG_RAISED = "#222d44"          # Most-elevated surface (toast, popover)
    BG_LOG = "#070b13"             # Log panel
    BG_OVERLAY = "#0a0e17"         # Modal / overlay backdrop

    # Accents
    GREEN_PRIMARY = "#34d399"      # Emerald -- success and primary CTA
    GREEN_HOVER = "#10b981"        # Deeper emerald (hover)
    GREEN_PRESS = "#059669"        # Pressed
    GREEN_MUTED = "#0f3324"        # Success tint background

    BLUE_PRIMARY = "#60a5fa"       # Sky blue -- secondary CTA / info
    BLUE_HOVER = "#3b82f6"         # Deeper blue (hover)
    BLUE_PRESS = "#2563eb"         # Pressed
    BLUE_MUTED = "#13294a"         # Blue tint background

    # Text
    TEXT_PRIMARY = "#f4f7fd"       # Near-white -- primary text
    TEXT_SECONDARY = "#c5cfe2"     # High-contrast secondary
    TEXT_MUTED = "#8391ad"         # Support / helper text
    TEXT_DISABLED = "#4c5877"      # Disabled

    # Status
    SUCCESS = "#34d399"
    SUCCESS_BG = "#0e2e22"
    WARNING = "#fbbf24"
    WARNING_BG = "#352412"
    ERROR = "#f87171"
    ERROR_BG = "#351821"
    INFO = "#60a5fa"
    INFO_BG = "#0f2744"

    # Borders
    BORDER = "#27324a"             # Standard border
    BORDER_STRONG = "#364364"      # Emphasized border
    BORDER_SUBTLE = "#1a2234"      # Soft divider
    BORDER_FOCUS = "#60a5fa"       # Focus ring

    # Progress
    PROGRESS_BG = "#182236"
    PROGRESS_FILL = BLUE_PRIMARY

    # Typography (Segoe UI stack). Use these constants instead of inline fonts.
    FONT_FAMILY = "Segoe UI"
    FONT_MONO = "Consolas"

    # Size tokens
    F_DISPLAY = 22      # hero page title
    F_HEADING = 16      # section card heading
    F_TITLE = 12        # card title / subsection
    F_BODY = 10         # body text (default)
    F_BODY_SM = 9       # compact body
    F_LABEL = 9         # labels, helper
    F_META = 8          # meta / captions
    F_EYEBROW = 8       # small-caps eyebrow
    F_MICRO = 7         # ultra compact

    # Spacing rhythm (4pt baseline)
    S_XS = 4
    S_SM = 8
    S_MD = 12
    S_LG = 16
    S_XL = 20
    S_2XL = 24
    S_3XL = 32

    # Radii
    R_SM = 4
    R_MD = 6
    R_LG = 8
    R_XL = 12


def apply_high_contrast_theme():
    """RM-96: Swap the design tokens for a higher-contrast palette.

    Idempotent and reversible -- the original values are cached on
    Theme._defaults so apply_default_theme() can restore them. The
    custom Canvas widgets read Theme constants on every draw, so an
    in-place swap takes effect after the next redraw cycle.
    """
    if not hasattr(Theme, "_defaults"):
        Theme._defaults = {
            k: v for k, v in Theme.__dict__.items()
            if not k.startswith("_") and isinstance(v, str)
        }
    Theme.BG_DARK = "#000000"
    Theme.BG_SECONDARY = "#000000"
    Theme.BG_CARD = "#0c0c0c"
    Theme.BG_CARD_HOVER = "#1a1a1a"
    Theme.BG_CARD_SELECTED = "#1f1f1f"
    Theme.BG_TERTIARY = "#1a1a1a"
    Theme.BG_RAISED = "#262626"
    Theme.BG_LOG = "#000000"
    Theme.BG_OVERLAY = "#000000"
    Theme.GREEN_PRIMARY = "#00ff7f"
    Theme.GREEN_HOVER = "#00cc66"
    Theme.GREEN_PRESS = "#00994d"
    Theme.GREEN_MUTED = "#003319"
    Theme.BLUE_PRIMARY = "#00d4ff"
    Theme.BLUE_HOVER = "#00b3d9"
    Theme.BLUE_PRESS = "#0099b3"
    Theme.BLUE_MUTED = "#002633"
    Theme.TEXT_PRIMARY = "#ffffff"
    Theme.TEXT_SECONDARY = "#ffffff"
    Theme.TEXT_MUTED = "#dcdcdc"
    Theme.TEXT_DISABLED = "#888888"
    Theme.SUCCESS = "#00ff7f"
    Theme.SUCCESS_BG = "#003319"
    Theme.WARNING = "#ffff00"
    Theme.WARNING_BG = "#332f00"
    Theme.ERROR = "#ff5555"
    Theme.ERROR_BG = "#330000"
    Theme.INFO = "#00d4ff"
    Theme.INFO_BG = "#002633"
    Theme.BORDER = "#ffffff"
    Theme.BORDER_STRONG = "#ffffff"
    Theme.BORDER_SUBTLE = "#aaaaaa"
    Theme.BORDER_FOCUS = "#ffff00"
    Theme.PROGRESS_BG = "#1a1a1a"
    Theme.PROGRESS_FILL = "#00d4ff"


def apply_default_theme():
    """Restore the original Theme palette saved by apply_high_contrast_theme."""
    defaults = getattr(Theme, "_defaults", None)
    if not defaults:
        return
    for k, v in defaults.items():
        setattr(Theme, k, v)


def f(size: int, weight: str = "normal") -> tuple:
    """Shortcut to build a Segoe UI font tuple."""
    if weight == "bold":
        return (Theme.FONT_FAMILY, size, "bold")
    return (Theme.FONT_FAMILY, size)


def mono(size: int) -> tuple:
    return (Theme.FONT_MONO, size)

class InpaintMode(Enum):
    AUTO = "Auto"
    STTN = "STTN"
    LAMA = "LAMA"
    PROPAINTER = "ProPainter"


class ProcessingStatus(Enum):
    IDLE = "idle"
    LOADING = "loading"
    DETECTING = "detecting"
    PROCESSING = "processing"
    MERGING = "merging"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


STATUS_UI = {
    ProcessingStatus.IDLE: {
        "label": "Ready",
        "color": Theme.TEXT_SECONDARY,
        "bg": Theme.BG_TERTIARY,
    },
    ProcessingStatus.LOADING: {
        "label": "Loading",
        "color": Theme.INFO,
        "bg": Theme.INFO_BG,
    },
    ProcessingStatus.DETECTING: {
        "label": "Scanning",
        "color": Theme.INFO,
        "bg": Theme.INFO_BG,
    },
    ProcessingStatus.PROCESSING: {
        "label": "Removing",
        "color": Theme.SUCCESS,
        "bg": Theme.SUCCESS_BG,
    },
    ProcessingStatus.MERGING: {
        "label": "Finishing",
        "color": Theme.WARNING,
        "bg": Theme.WARNING_BG,
    },
    ProcessingStatus.COMPLETE: {
        "label": "Complete",
        "color": Theme.SUCCESS,
        "bg": Theme.SUCCESS_BG,
    },
    ProcessingStatus.ERROR: {
        "label": "Needs Attention",
        "color": Theme.ERROR,
        "bg": Theme.ERROR_BG,
    },
    ProcessingStatus.CANCELLED: {
        "label": "Stopped",
        "color": Theme.TEXT_MUTED,
        "bg": Theme.BG_TERTIARY,
    },
}


@dataclass
class ProcessingConfig:
    """Configuration for subtitle removal processing."""
    mode: InpaintMode = InpaintMode.STTN
    use_gpu: bool = True
    gpu_id: int = 0

    # STTN settings
    sttn_skip_detection: bool = False
    sttn_neighbor_stride: int = 10
    sttn_reference_length: int = 10
    sttn_max_load_num: int = 30

    # LAMA settings
    lama_super_fast: bool = False

    # Region settings
    subtitle_area: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2

    # Detection settings
    detection_lang: str = "en"
    detection_threshold: float = 0.5
    detection_vertical: bool = False    # RM-24 vertical-text mode
    whisper_fallback: bool = False       # RM-27 Whisper-driven mask fallback
    whisper_model_size: str = "tiny"
    upscale_factor: int = 0              # RM-78 post-cleanup upscale (0/2/3/4)
    film_grain_strength: float = 0.0     # RM-80 additive film grain (0..0.5)
    swinir_restore: bool = False         # RM-79 SwinIR restoration pass
    seedvr2_restore: bool = False        # RM-77 SeedVR2 restoration
    preserve_color_metadata: bool = True  # RM-73 partial
    nle_sidecar: str = "off"             # RM-76 (off / edl / fcpxml)

    # Time range (video only, seconds)
    time_start: float = 0.0
    time_end: float = 0.0

    # Detection frame skip (0=detect every frame, N=reuse mask for N frames)
    detection_frame_skip: int = 0

    # Mask dilation in pixels for cleaner removal
    mask_dilate_px: int = 8

    # Mask edge feathering (soft-blend width in pixels; 0 disables)
    mask_feather_px: int = 4

    # Temporal Background Exposure (real STTN / ProPainter backing)
    tbe_enable: bool = True
    tbe_min_coverage: int = 3
    tbe_use_median: bool = True

    # v3.9 quality controls
    tbe_flow_warp: bool = False         # Farneback flow-warp before TBE aggregation
    tbe_scene_cut_split: bool = True    # split TBE batch at scene cuts
    tbe_scene_cut_threshold: float = 0.35
    tbe_scene_cut_use_pyscenedetect: bool = False  # RM-32 opt-in dep
    tbe_scene_cut_use_transnetv2: bool = False     # RM-21 opt-in dep
    detection_denoise: bool = False                # RM-33 opt-in dep
    sam2_refine: bool = False                      # RM-66 opt-in dep
    edge_ring_px: int = 2               # post-inpaint colour-match ring width

    # v3.9 workflow features
    subtitle_areas: Optional[List[Tuple[int, int, int, int]]] = None  # multi-region
    auto_band: bool = False             # auto-detect dominant subtitle band on load
    export_srt: bool = False            # write detected text as SRT sidecar
    export_mask_video: bool = False     # write B/W mask debug mp4
    adaptive_batch: bool = True         # VRAM-probe-driven batch sizing

    # v3.12 AUTO mode + preprocessing
    auto_exposure_threshold: float = 0.55
    deinterlace: bool = False
    deinterlace_auto: bool = True
    keyframe_detection: bool = False
    quality_report: bool = False

    # v3.10 quality knobs
    kalman_tracking: bool = True        # smooth per-frame detection jitter
    kalman_iou_threshold: float = 0.3
    kalman_max_age: int = 2
    phash_skip_enable: bool = True      # adaptive mask reuse via perceptual hash
    phash_skip_distance: int = 4
    colour_tune_enable: bool = False    # grow mask by dominant-colour match
    colour_tune_tolerance: int = 25

    # Output settings
    output_format: str = "mp4"
    preserve_audio: bool = True
    output_quality: int = 23  # CRF value (15-35, lower = better quality)
    use_hw_encode: bool = True  # try hardware encoding (NVENC/QSV/AMF)
    output_codec: str = "h264"   # h264 (default) / h265 / av1

    # v3.13 -- exposed in GUI as of this build
    # Audio
    loudnorm_target: float = 0.0           # 0=off, otherwise LUFS in [-70,-5]
    multi_audio_passthrough: bool = True   # mux every audio stream
    # Performance / decode
    decode_hw_accel: str = "off"           # off/auto/any/d3d11/vaapi/mfx
    prefetch_decode: bool = True           # worker-thread frame prefetcher
    prefetch_queue_size: int = 0           # 0=auto (max(8, batch_size*2))
    input_fps: float = 24.0                # FPS for directory-of-images input
    # Quality
    quality_report_sheet: bool = False     # render side-by-side PNG
    # Editorial filters (chyron classifier)
    remove_subtitles: bool = True
    remove_chyrons: bool = True
    chyron_min_hits: int = 90              # Kalman hits to classify as chyron
    # Karaoke per-syllable grouping
    karaoke_grouping: bool = False
    karaoke_x_gap_px: int = 20
    karaoke_y_overlap: float = 0.5

    # UI state (persisted across sessions; not part of processing config)
    window_geometry: str = ""  # e.g. "1240x860+100+60"
    adv_panel_open: bool = False
    log_panel_open: bool = True
    onboarding_seen: bool = False
    high_contrast: bool = False     # RM-96 alt theme palette
    rtl_layout: bool = False        # RM-98 right-to-left UI mirror

    def to_dict(self) -> dict:
        """Persist every dataclass field automatically. Using
        `dataclasses.fields` means new fields land in settings.json without
        any further edits here -- the v3.13 GUI gap (13 backend fields
        unreachable from the GUI) was rooted in a manual enumeration that
        was easy to forget to update."""
        from dataclasses import fields as _dc_fields
        payload: dict = {}
        for field_def in _dc_fields(self):
            value = getattr(self, field_def.name)
            if isinstance(value, InpaintMode):
                payload[field_def.name] = value.value
            elif field_def.name == "subtitle_area":
                payload[field_def.name] = list(value) if value else None
            elif field_def.name == "subtitle_areas":
                payload[field_def.name] = (
                    [list(r) for r in value] if value else None
                )
            else:
                payload[field_def.name] = value
        payload["vsr_settings_format"] = VSR_SETTINGS_FORMAT
        return payload

    def normalized(self) -> 'ProcessingConfig':
        """Coerce persisted or imported values into a safe, UI-friendly shape."""
        self.mode = _coerce_gui_mode(self.mode)
        self.use_gpu = _coerce_bool(self.use_gpu, True)
        self.gpu_id = max(0, _coerce_int(self.gpu_id, 0))
        self.sttn_skip_detection = _coerce_bool(self.sttn_skip_detection, False)
        self.sttn_neighbor_stride = _coerce_int(self.sttn_neighbor_stride, 10, 5, 30)
        self.sttn_reference_length = _coerce_int(self.sttn_reference_length, 10, 5, 30)
        self.sttn_max_load_num = _coerce_int(self.sttn_max_load_num, 30, 10, 100)
        self.lama_super_fast = _coerce_bool(self.lama_super_fast, False)
        self.subtitle_area = _coerce_rect(self.subtitle_area)
        self.subtitle_areas = _coerce_rect_list(self.subtitle_areas)
        self.detection_lang = _coerce_text(self.detection_lang, "en", 24).lower()
        self.detection_threshold = _coerce_float(self.detection_threshold, 0.5, 0.1, 0.9)
        self.detection_vertical = _coerce_bool(self.detection_vertical, False)
        self.whisper_fallback = _coerce_bool(self.whisper_fallback, False)
        wm = _coerce_text(self.whisper_model_size, "tiny", 16).lower()
        if wm not in {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}:
            wm = "tiny"
        self.whisper_model_size = wm
        upscale = _coerce_int(self.upscale_factor, 0, 0, 4)
        if upscale not in (0, 2, 3, 4):
            upscale = 0
        self.upscale_factor = upscale
        self.film_grain_strength = _coerce_float(self.film_grain_strength, 0.0, 0.0, 0.5)
        self.swinir_restore = _coerce_bool(self.swinir_restore, False)
        self.seedvr2_restore = _coerce_bool(self.seedvr2_restore, False)
        self.preserve_color_metadata = _coerce_bool(self.preserve_color_metadata, True)
        sidecar = _coerce_text(self.nle_sidecar, "off", 16).lower()
        if sidecar not in {"off", "edl", "fcpxml"}:
            sidecar = "off"
        self.nle_sidecar = sidecar
        self.time_start = max(0.0, _coerce_float(self.time_start, 0.0))
        self.time_end = max(0.0, _coerce_float(self.time_end, 0.0))
        if self.time_end and self.time_end < self.time_start:
            self.time_end = 0.0
        self.detection_frame_skip = _coerce_int(self.detection_frame_skip, 0, 0, 10)
        self.mask_dilate_px = _coerce_int(self.mask_dilate_px, 8, 0, 20)
        self.mask_feather_px = _coerce_int(self.mask_feather_px, 4, 0, 15)
        self.tbe_enable = _coerce_bool(self.tbe_enable, True)
        self.tbe_min_coverage = _coerce_int(self.tbe_min_coverage, 3, 1, 10)
        self.tbe_use_median = _coerce_bool(self.tbe_use_median, True)
        self.tbe_flow_warp = _coerce_bool(self.tbe_flow_warp, False)
        self.tbe_scene_cut_split = _coerce_bool(self.tbe_scene_cut_split, True)
        self.tbe_scene_cut_threshold = _coerce_float(self.tbe_scene_cut_threshold, 0.35, 0.0, 1.0)
        self.tbe_scene_cut_use_pyscenedetect = _coerce_bool(
            self.tbe_scene_cut_use_pyscenedetect, False)
        self.tbe_scene_cut_use_transnetv2 = _coerce_bool(
            self.tbe_scene_cut_use_transnetv2, False)
        self.detection_denoise = _coerce_bool(self.detection_denoise, False)
        self.sam2_refine = _coerce_bool(self.sam2_refine, False)
        self.edge_ring_px = _coerce_int(self.edge_ring_px, 2, 0, 8)
        self.auto_band = _coerce_bool(self.auto_band, False)
        self.export_srt = _coerce_bool(self.export_srt, False)
        self.export_mask_video = _coerce_bool(self.export_mask_video, False)
        self.adaptive_batch = _coerce_bool(self.adaptive_batch, True)
        self.auto_exposure_threshold = _coerce_float(self.auto_exposure_threshold, 0.55, 0.0, 1.0)
        self.deinterlace = _coerce_bool(self.deinterlace, False)
        self.deinterlace_auto = _coerce_bool(self.deinterlace_auto, True)
        self.keyframe_detection = _coerce_bool(self.keyframe_detection, False)
        self.quality_report = _coerce_bool(self.quality_report, False)
        self.kalman_tracking = _coerce_bool(self.kalman_tracking, True)
        self.kalman_iou_threshold = _coerce_float(self.kalman_iou_threshold, 0.3, 0.0, 1.0)
        self.kalman_max_age = _coerce_int(self.kalman_max_age, 2, 0, 30)
        self.phash_skip_enable = _coerce_bool(self.phash_skip_enable, True)
        self.phash_skip_distance = _coerce_int(self.phash_skip_distance, 4, 0, 64)
        self.colour_tune_enable = _coerce_bool(self.colour_tune_enable, False)
        self.colour_tune_tolerance = _coerce_int(self.colour_tune_tolerance, 25, 0, 100)
        self.output_format = _coerce_text(self.output_format, "mp4", 16).lower()
        self.preserve_audio = _coerce_bool(self.preserve_audio, True)
        self.output_quality = _coerce_int(self.output_quality, 23, 15, 35)
        self.use_hw_encode = _coerce_bool(self.use_hw_encode, True)
        codec = _coerce_text(self.output_codec, "h264", 16).lower()
        if codec in {"hevc", "h.265"}:
            codec = "h265"
        if codec not in {"h264", "h265", "av1"}:
            codec = "h264"
        self.output_codec = codec
        # v3.13 GUI-exposed knobs -- mirror the backend coercion bounds so
        # values that round-trip through settings.json land in the same
        # safe shape that normalize_processing_config enforces.
        target = _coerce_float(self.loudnorm_target, 0.0)
        if target == 0.0 or -70.0 <= target <= -5.0:
            self.loudnorm_target = target
        else:
            self.loudnorm_target = 0.0
        accel = _coerce_text(self.decode_hw_accel, "off", 16).lower()
        if accel not in {"off", "auto", "any", "d3d11", "vaapi", "mfx"}:
            accel = "off"
        self.decode_hw_accel = accel
        self.multi_audio_passthrough = _coerce_bool(self.multi_audio_passthrough, True)
        self.prefetch_decode = _coerce_bool(self.prefetch_decode, True)
        self.prefetch_queue_size = _coerce_int(self.prefetch_queue_size, 0, 0, 512)
        self.input_fps = _coerce_float(self.input_fps, 24.0, 1.0, 240.0)
        self.quality_report_sheet = _coerce_bool(self.quality_report_sheet, False)
        if self.quality_report_sheet:
            self.quality_report = True
        self.remove_subtitles = _coerce_bool(self.remove_subtitles, True)
        self.remove_chyrons = _coerce_bool(self.remove_chyrons, True)
        self.chyron_min_hits = _coerce_int(self.chyron_min_hits, 90, 1, 100000)
        self.karaoke_grouping = _coerce_bool(self.karaoke_grouping, False)
        self.karaoke_x_gap_px = _coerce_int(self.karaoke_x_gap_px, 20, 0, 1024)
        self.karaoke_y_overlap = _coerce_float(self.karaoke_y_overlap, 0.5, 0.0, 1.0)
        self.window_geometry = _coerce_text(self.window_geometry, "", 64)
        self.adv_panel_open = _coerce_bool(self.adv_panel_open, False)
        self.log_panel_open = _coerce_bool(self.log_panel_open, True)
        self.onboarding_seen = _coerce_bool(self.onboarding_seen, False)
        self.high_contrast = _coerce_bool(self.high_contrast, False)
        self.rtl_layout = _coerce_bool(self.rtl_layout, False)
        return self

    @classmethod
    def from_dict(cls, data: dict) -> 'ProcessingConfig':
        """Reconstruct a config from a settings.json payload. Walks
        `dataclasses.fields` so every declared field is restored. Unknown
        keys are ignored; missing keys fall back to the dataclass default.
        Rect fields go through the dedicated coercers before normalisation
        so a non-iterable payload (`{"subtitle_area": 42}`) never crashes
        the loader -- the regression case covered by from_dict tests."""
        if not isinstance(data, dict):
            data = {}
        from dataclasses import fields as _dc_fields
        kwargs: dict = {}
        for field_def in _dc_fields(cls):
            name = field_def.name
            if name not in data:
                continue
            raw = data.get(name)
            if name == "mode":
                kwargs[name] = raw  # _coerce_gui_mode applied in normalize
            elif name == "subtitle_area":
                kwargs[name] = _coerce_rect(raw)
            elif name == "subtitle_areas":
                kwargs[name] = _coerce_rect_list(raw)
            else:
                kwargs[name] = raw
        return cls(**kwargs).normalized()


@dataclass
class QueueItem:
    """Represents an item in the processing queue."""
    id: str
    file_path: str
    output_path: str
    config: ProcessingConfig
    output_path_locked: bool = False
    status: ProcessingStatus = ProcessingStatus.IDLE
    progress: float = 0.0
    message: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    quality_report: Optional[dict] = None
    # F-7: per-item cancel flag. Set by the queue widget when the user
    # asks to cancel just this entry. The progress callback in
    # _process_item checks it alongside the global cancel_event so a
    # single-item cancellation does NOT abort the rest of the batch.
    cancel_requested: bool = False


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
    try:
        f = float(value)
        if not math.isfinite(f):
            raise ValueError("non-finite float")
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
    try:
        coerced = float(value)
        if not math.isfinite(coerced):
            raise ValueError("non-finite float")
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


def _coerce_gui_mode(value) -> InpaintMode:
    if isinstance(value, InpaintMode):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        mode_map = {
            "auto": InpaintMode.AUTO,
            "sttn": InpaintMode.STTN,
            "lama": InpaintMode.LAMA,
            "propainter": InpaintMode.PROPAINTER,
            "pro painter": InpaintMode.PROPAINTER,
        }
        if normalized in mode_map:
            return mode_map[normalized]
    return InpaintMode.STTN


def _read_json_object(path: Path, label: str) -> Optional[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Could not read {label} from {path}: {exc}")
        return None
    if not isinstance(payload, dict):
        logger.warning(f"Ignoring {label} at {path}: expected a JSON object")
        return None
    return payload


def _write_json_atomic(path: Path, payload: dict):
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
            json.dump(payload, handle, indent=2)
        os.replace(temp_path, path)
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


# =============================================================================
# SETTINGS PERSISTENCE
# =============================================================================

def _migrate_settings(data: dict) -> dict:
    """Upgrade an on-disk settings payload to the current schema.

    The settings file is a flat dict. A missing `vsr_settings_format` means
    "anything from v3.12.0 or earlier" -- those builds shipped no version
    tag and are treated as format 0. Each numbered case below documents the
    field rename / coercion needed to reach the next format level. Unknown
    future versions are accepted as-is on the assumption a newer build wrote
    them; the coercer / `from_dict` will drop fields it does not recognise.
    """
    if not isinstance(data, dict):
        return {}
    version = data.get("vsr_settings_format")
    try:
        version = int(version) if version is not None else 0
    except (TypeError, ValueError):
        version = 0

    if version > VSR_SETTINGS_FORMAT:
        logger.info(
            f"settings.json reports vsr_settings_format={version} "
            f"(this build understands up to {VSR_SETTINGS_FORMAT}); "
            f"unknown keys will be ignored."
        )
        return data

    # version == 0 -> 1: no field renames; stamp the version so future
    # migrations have a known floor. Add field-rename / coercion blocks
    # below as the schema evolves.
    if version < 1:
        data = dict(data)
        data["vsr_settings_format"] = 1
        version = 1

    # version 1 -> 2: B-1 wired the 13 v3.13 backend fields through the
    # GUI. The new keys default to the backend's existing defaults so a
    # legacy payload missing them resolves to the v3.12 behaviour; we
    # only need to bump the version tag here.
    if version < 2:
        data = dict(data)
        data["vsr_settings_format"] = 2
        version = 2

    return data


def load_settings() -> ProcessingConfig:
    """Load saved settings from disk."""
    try:
        if SETTINGS_FILE.exists():
            data = _read_json_object(SETTINGS_FILE, "settings")
            if not data:
                return ProcessingConfig()
            data = _migrate_settings(data)
            logger.info(f"Settings loaded from {SETTINGS_FILE}")
            return ProcessingConfig.from_dict(data)
    except Exception as e:
        logger.warning(f"Could not load settings: {e}")
    return ProcessingConfig()


def save_settings(config: ProcessingConfig):
    """Save settings to disk."""
    try:
        _write_json_atomic(SETTINGS_FILE, config.normalized().to_dict())
        logger.info(f"Settings saved to {SETTINGS_FILE}")
    except Exception as e:
        logger.warning(f"Could not save settings: {e}")


# =============================================================================
# PRESET LIBRARY
# =============================================================================

PRESETS_FILE = LOG_DIR / "presets.json"

# F-10: built-in presets live in backend/presets.py so the CLI's
# --preset flag and the GUI's preset picker resolve from the same source.
# The shared module also exposes resolve_preset/preset_fields helpers we
# could swap in here over time -- for now we just import the table.
from backend.presets import BUILTIN_PRESETS


def _load_user_presets() -> dict:
    if PRESETS_FILE.exists():
        payload = _read_json_object(PRESETS_FILE, "user presets")
        if payload is not None:
            return payload
    return {}


def _save_user_presets(presets: dict):
    try:
        _write_json_atomic(PRESETS_FILE, presets)
    except Exception as exc:
        logger.warning(f"Could not save user presets: {exc}")


def list_presets() -> List[Tuple[str, str]]:
    """Return [(name, description)] for every built-in + user preset."""
    items = [(n, p.get("description", "")) for n, p in BUILTIN_PRESETS.items()]
    for name, payload in _load_user_presets().items():
        if isinstance(payload, dict):
            items.append((name, _coerce_text(payload.get("description", "User preset"), "User preset", 120)))
    return items


def apply_preset(config: ProcessingConfig, name: str) -> bool:
    """Apply a named preset to `config` in-place. Returns True on success."""
    preset = BUILTIN_PRESETS.get(name)
    if preset is None:
        preset = _load_user_presets().get(name)
    if not isinstance(preset, dict):
        return False
    fields = preset.get("fields", {})
    if not isinstance(fields, dict):
        return False
    for k, v in fields.items():
        if k == "mode":
            config.mode = _coerce_gui_mode(v)
            continue
        if hasattr(config, k):
            setattr(config, k, v)
    config.normalized()
    return True


def save_user_preset(name: str, description: str, config: ProcessingConfig,
                      fields: Optional[List[str]] = None) -> bool:
    """Snapshot the selected fields from `config` into a user preset."""
    name = _coerce_text(name, "", 80)
    description = _coerce_text(description, "User preset", 160) or "User preset"
    if not name:
        return False
    if name in BUILTIN_PRESETS:
        return False  # don't let users overwrite built-ins
    default_fields = [
        "mode", "detection_threshold", "mask_dilate_px", "mask_feather_px",
        "edge_ring_px", "tbe_flow_warp", "tbe_scene_cut_split",
        "colour_tune_enable", "colour_tune_tolerance",
        "kalman_tracking", "phash_skip_enable", "phash_skip_distance",
        "auto_band",
    ]
    fields = fields or default_fields
    config = config.normalized()
    snap = {}
    for k in fields:
        v = getattr(config, k, None)
        if k == "mode" and hasattr(v, "value"):
            v = v.value
        if v is not None:
            snap[k] = v
    user = _load_user_presets()
    user[name] = {"description": description, "fields": snap}
    _save_user_presets(user)
    return True


def delete_user_preset(name: str) -> bool:
    if name in BUILTIN_PRESETS:
        return False
    user = _load_user_presets()
    if name not in user:
        return False
    del user[name]
    _save_user_presets(user)
    return True


def export_preset(name: str, path: str) -> bool:
    """Write a named preset (built-in or user) to a standalone JSON file
    so it can be shared or version-controlled alongside a project."""
    preset = BUILTIN_PRESETS.get(name) or _load_user_presets().get(name)
    if not preset:
        return False
    payload = {
        "name": name,
        "description": preset.get("description", ""),
        "fields": preset.get("fields", {}),
        "vsr_preset_format": 1,
    }
    try:
        _write_json_atomic(Path(path), payload)
        return True
    except Exception as exc:
        logger.warning(f"Could not export preset '{name}' to {path}: {exc}")
        return False


def import_preset(path: str) -> Optional[str]:
    """Load a shareable preset JSON and install it under the user's preset
    library. Returns the installed name on success, None on failure.
    Collisions with built-in names are rejected; collisions with existing
    user presets overwrite."""
    payload = _read_json_object(Path(path), "preset import")
    if payload is None:
        return None
    if payload.get("vsr_preset_format") != 1:
        logger.warning(f"Not a v1 VSR preset: {path}")
        return None
    name = _coerce_text(payload.get("name", ""), "", 80)
    fields = payload.get("fields", {})
    description = _coerce_text(payload.get("description", "Imported preset"), "Imported preset", 160)
    if not name or not isinstance(fields, dict):
        return None
    if name in BUILTIN_PRESETS:
        name = f"{name} (imported)"
    user = _load_user_presets()
    user[name] = {"description": description, "fields": fields}
    _save_user_presets(user)
    return name


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_app_dir() -> Path:
    """Get the application directory."""
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent
    return Path(__file__).parent


def detect_gpu() -> List[dict]:
    """Detect available GPUs."""
    gpus = []

    # Try NVIDIA GPU detection
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 3:
                        try:
                            gpu_idx = int(parts[0].strip())
                            gpu_mem = f"{int(parts[2].strip())} MB"
                        except ValueError:
                            continue
                        gpus.append({
                            "index": gpu_idx,
                            "name": parts[1].strip(),
                            "memory": gpu_mem,
                            "type": "NVIDIA"
                        })
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # If no NVIDIA GPU, check for DirectML support
    if not gpus:
        try:
            import torch_directml
            gpus.append({
                "index": 0,
                "name": "DirectML Device",
                "memory": "Unknown",
                "type": "DirectML"
            })
        except ImportError:
            pass

    return gpus


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
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
    """Format bytes into human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} PB"


def is_video_file(path: str) -> bool:
    """Check if file is a supported video format."""
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpeg', '.mpg'}
    return Path(path).suffix.lower() in video_extensions


def is_image_file(path: str) -> bool:
    """Check if file is a supported image format."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    return Path(path).suffix.lower() in image_extensions


# F-5: curated friendly names for languages the project always shipped
# with. Languages the active engine adds beyond this list inherit the
# raw code as their display name.
_CURATED_LANG_NAMES: Tuple[Tuple[str, str], ...] = (
    ("en", "English"),
    ("ch", "Chinese"),
    ("japan", "Japanese"),
    ("ja", "Japanese"),
    ("manga", "Manga / Anime (vertical JP via manga-ocr)"),
    ("ko", "Korean"),
    ("korean", "Korean"),
    ("fr", "French"),
    ("french", "French"),
    ("de", "German"),
    ("german", "German"),
    ("es", "Spanish"),
    ("spanish", "Spanish"),
    ("pt", "Portuguese"),
    ("portuguese", "Portuguese"),
    ("ru", "Russian"),
    ("ar", "Arabic"),
    ("arabic", "Arabic"),
    ("hi", "Hindi"),
    ("it", "Italian"),
    ("italian", "Italian"),
    ("nl", "Dutch"),
    ("pl", "Polish"),
    ("tr", "Turkish"),
    ("vi", "Vietnamese"),
    ("th", "Thai"),
    ("uk", "Ukrainian"),
    ("sv", "Swedish"),
    ("no", "Norwegian"),
    ("da", "Danish"),
    ("fi", "Finnish"),
    ("cs", "Czech"),
    ("hu", "Hungarian"),
    ("ro", "Romanian"),
    ("el", "Greek"),
    ("he", "Hebrew"),
    ("id", "Indonesian"),
    ("ms", "Malay"),
    ("fil", "Filipino"),
)


def _engine_supported_languages() -> List[str]:
    """Best-effort list of language codes the active OCR cascade can
    drive. We do not import the engines just to read their lang lists
    (heavy); we ask each module for a constant if it exposes one.
    """
    codes: List[str] = []
    # PaddleOCR exposes a published list of 80+ codes; we hard-code a
    # subset shared across PaddleOCR / RapidOCR / EasyOCR rather than
    # importing the engine to read its config.
    paddle_compatible = [
        "en", "ch", "chinese_cht", "japan", "korean", "ka",
        "fr", "german", "it", "es", "pt", "ru", "ar", "hi",
        "nl", "no", "pl", "tr", "th", "vi", "uk", "be",
        "bg", "hr", "cs", "da", "et", "fi", "hu", "is",
        "lv", "lt", "mt", "ro", "sk", "sl", "sv", "id", "ms",
        "fa", "he", "el",
    ]
    codes.extend(paddle_compatible)
    return codes


def _build_language_list() -> List[Tuple[str, str]]:
    """Return [(code, friendly_name)] pairs to populate the lang picker.

    Curated friendly names take precedence; engine-supported codes that
    aren't in the curated table fall through with the raw code as their
    label so the picker still surfaces them.
    """
    pretty: Dict[str, str] = {}
    for code, name in _CURATED_LANG_NAMES:
        pretty.setdefault(code, name)
    out: List[Tuple[str, str]] = []
    seen: set = set()
    # English first (always the default).
    out.append(("en", "English"))
    seen.add("en")
    # Curated order next so the picker leads with familiar options.
    for code, name in _CURATED_LANG_NAMES:
        if code in seen:
            continue
        seen.add(code)
        out.append((code, name))
    # Any extra engine-declared codes the curated list missed.
    for code in _engine_supported_languages():
        if code in seen:
            continue
        seen.add(code)
        out.append((code, pretty.get(code, code.upper())))
    return out


def detect_ai_engines() -> dict:
    """Probe which AI engines are available."""
    engines = {"detection": [], "inpainting": []}
    # RapidOCR first -- ONNX Runtime, 4-5x faster than PaddleOCR, leak-free
    try:
        try:
            import rapidocr  # noqa: F401
        except ImportError:
            import rapidocr_onnxruntime  # noqa: F401
        engines["detection"].append("RapidOCR")
    except ImportError:
        pass
    try:
        import paddleocr  # noqa: F401
        engines["detection"].append("PaddleOCR")
    except ImportError:
        pass
    surya_opt_in = os.environ.get("VSR_ALLOW_GPL", "").strip().lower() in {"1", "true", "yes", "on"}
    try:
        from surya.detection import DetectionPredictor  # noqa: F401
        if surya_opt_in:
            engines["detection"].append("Surya")
        else:
            engines["detection"].append("Surya (GPL -- set VSR_ALLOW_GPL=1)")
    except Exception:
        pass
    try:
        import easyocr  # noqa: F401
        engines["detection"].append("EasyOCR")
    except ImportError:
        pass
    if not engines["detection"]:
        engines["detection"].append("OpenCV fallback")
    # Temporal Background Exposure always available -- real video inpainting
    # from adjacent frames, no weights required.
    engines["inpainting"].append("Temporal BG (TBE)")
    try:
        from simple_lama_inpainting import SimpleLama  # noqa: F401
        engines["inpainting"].append("LaMa (neural)")
    except ImportError:
        pass
    engines["inpainting"].append("OpenCV")
    return engines


def detect_ffmpeg() -> bool:
    """Check whether FFmpeg is available on PATH."""
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
    """Get a short info string for a file (type + size)."""
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
    """Truncate long strings while preserving both ends."""
    if len(text) <= max_length:
        return text
    if max_length < 10:
        return text[:max_length]
    lead = max_length // 2 - 2
    tail = max_length - lead - 3
    return f"{text[:lead]}...{text[-tail:]}"


def format_quality_report(metrics: Optional[dict], compact: bool = False) -> str:
    """Format a PSNR / SSIM quality-report payload for the UI.

    When the backend returns a `roi_psnr`/`roi_ssim` pair (B-3), surface the
    inpaint-region score first -- that is the metric users actually care
    about. The whole-frame score is reported alongside as context."""
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
        suffix = f" across {sample_count} sampled frame{'s' if sample_count != 1 else ''}"
    if roi_psnr is not None and roi_ssim is not None:
        return (
            f"inpaint region PSNR {roi_psnr:.2f} dB and SSIM {roi_ssim:.4f}, "
            f"whole frame PSNR {psnr:.2f} dB and SSIM {ssim:.4f}{suffix}"
        )
    return f"PSNR {psnr:.2f} dB and SSIM {ssim:.4f}{suffix}"


def summarize_quality_reports(reports: List[Optional[dict]]) -> Optional[dict]:
    """Average PSNR / SSIM metrics across completed queue items."""
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


def status_ui(status: ProcessingStatus) -> dict:
    """Return display metadata for a processing status."""
    return STATUS_UI.get(
        status,
        {"label": status.value.title(), "color": Theme.TEXT_MUTED, "bg": Theme.BG_TERTIARY},
    )


# =============================================================================
# CUSTOM WIDGETS
# =============================================================================

def _get_dpi_scale(root) -> float:
    """Get the DPI scaling factor relative to 96 DPI baseline."""
    try:
        return root.winfo_fpixels('1i') / 96.0
    except Exception:
        return 1.0


def _scaled(root, px: int) -> int:
    """Scale a pixel value by the current DPI factor."""
    return int(px * _get_dpi_scale(root))


class Tooltip:
    """Refined hover tooltip. Appears after a short delay, styled as a raised
    surface with subtle border and proper text wrapping."""

    DELAY_MS = 380

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self._tip = None
        self._after_id = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")
        widget.bind("<Destroy>", self._hide, add="+")

    def _schedule(self, event):
        self._cancel()
        try:
            self._after_id = self.widget.after(self.DELAY_MS, self._show)
        except tk.TclError:
            self._after_id = None

    def _cancel(self):
        if self._after_id:
            try:
                self.widget.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    def _show(self):
        if self._tip:
            self._tip.destroy()
            self._tip = None
        try:
            self._tip = tk.Toplevel(self.widget)
            self._tip.wm_overrideredirect(True)
            self._tip.configure(bg=Theme.BORDER_STRONG)
            display_text = self.text if len(self.text) <= 160 else self.text[:157] + "..."
            inner = tk.Frame(self._tip, bg=Theme.BG_RAISED)
            inner.pack(padx=1, pady=1)
            tk.Label(
                inner,
                text=display_text,
                font=f(Theme.F_LABEL),
                bg=Theme.BG_RAISED,
                fg=Theme.TEXT_PRIMARY,
                padx=10, pady=6,
                wraplength=360,
                justify="left",
            ).pack()
            self._tip.update_idletasks()
            x = self.widget.winfo_rootx() + 14
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
            sw = self._tip.winfo_screenwidth()
            sh = self._tip.winfo_screenheight()
            tw = self._tip.winfo_reqwidth()
            th = self._tip.winfo_reqheight()
            if x + tw > sw:
                x = sw - tw - 6
            if y + th > sh:
                y = self.widget.winfo_rooty() - th - 6
            self._tip.wm_geometry(f"+{x}+{y}")
        except tk.TclError:
            self._tip = None

    def _hide(self, event=None):
        self._cancel()
        if self._tip:
            try:
                self._tip.destroy()
            except tk.TclError:
                pass
            self._tip = None


class ModernButton(tk.Canvas):
    """A refined button with hover/press/focus states, icon support,
    and consistent size tokens. Canvas-based so corner radius is rendered
    crisply regardless of ttk theme.

    Style variants: primary, accent, secondary, ghost, danger, success
    Size variants: sm (28), md (32), lg (36)
    """

    SIZES = {"sm": (28, Theme.F_META), "md": (32, Theme.F_LABEL), "lg": (36, Theme.F_BODY_SM)}

    def __init__(self, parent, text="Button", command=None, width=120, height=None,
                 bg=None, hover_bg=None, fg=Theme.TEXT_PRIMARY,
                 corner_radius=None, font_size=None, style="primary",
                 size="md", icon=None, **kwargs):
        if height is None:
            height = self.SIZES.get(size, self.SIZES["md"])[0]
        if font_size is None:
            font_size = self.SIZES.get(size, self.SIZES["md"])[1]
        if corner_radius is None:
            corner_radius = Theme.R_MD if height <= 30 else Theme.R_LG

        parent_bg = parent.cget('bg') if hasattr(parent, 'cget') else Theme.BG_DARK
        super().__init__(parent, width=width, height=height, highlightthickness=0,
                        bg=parent_bg, takefocus=1)

        self.text = text
        self.icon = icon  # optional single-char glyph (ASCII)
        self.command = command
        self.width = width
        self.height = height
        self.corner_radius = corner_radius
        self.font_size = font_size
        self.enabled = True
        self.focused = False
        self.pressed = False
        self.hovered = False
        self.style = style

        self._apply_style(style)
        self.current_bg = self.bg_color
        self._draw()

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<FocusIn>", self._on_focus_in)
        self.bind("<FocusOut>", self._on_focus_out)
        self.bind("<Return>", self._on_keyboard_activate)
        self.bind("<space>", self._on_keyboard_activate)

    def _apply_style(self, style):
        if style == "primary":
            self.bg_color = Theme.BLUE_PRIMARY
            self.hover_color = Theme.BLUE_HOVER
            self.press_color = Theme.BLUE_PRESS
            self.fg_color = "#071226"
            self.border_color = Theme.BLUE_HOVER
        elif style == "accent":
            self.bg_color = Theme.BLUE_MUTED
            self.hover_color = Theme.BG_RAISED
            self.press_color = Theme.BG_CARD_SELECTED
            self.fg_color = Theme.TEXT_PRIMARY
            self.border_color = Theme.BORDER
        elif style == "secondary":
            self.bg_color = Theme.BG_TERTIARY
            self.hover_color = Theme.BG_RAISED
            self.press_color = Theme.BG_CARD_HOVER
            self.fg_color = Theme.TEXT_PRIMARY
            self.border_color = Theme.BORDER
        elif style == "ghost":
            self.bg_color = Theme.BG_CARD
            self.hover_color = Theme.BG_CARD_HOVER
            self.press_color = Theme.BG_CARD_SELECTED
            self.fg_color = Theme.TEXT_SECONDARY
            self.border_color = Theme.BORDER_SUBTLE
        elif style == "danger":
            self.bg_color = Theme.ERROR
            self.hover_color = "#ef4444"
            self.press_color = "#dc2626"
            self.fg_color = "#ffffff"
            self.border_color = "#ef4444"
        elif style == "success":
            self.bg_color = Theme.GREEN_MUTED
            self.hover_color = Theme.SUCCESS_BG
            self.press_color = Theme.GREEN_MUTED
            self.fg_color = Theme.GREEN_PRIMARY
            self.border_color = Theme.GREEN_HOVER
        else:
            self.bg_color = Theme.BG_TERTIARY
            self.hover_color = Theme.BG_CARD_HOVER
            self.press_color = Theme.BG_CARD_HOVER
            self.fg_color = Theme.TEXT_PRIMARY
            self.border_color = Theme.BORDER

    def _draw(self):
        self.delete("all")

        # Focus ring -- crisp outer glow
        if self.focused and self.enabled:
            self._create_rounded_rect(
                0, 0, self.width, self.height,
                self.corner_radius + 2,
                fill=Theme.BG_DARK, outline=Theme.BORDER_FOCUS, width=2,
            )
            pad = 2
        else:
            pad = 0

        if not self.enabled:
            fill = Theme.BG_TERTIARY
            border = Theme.BORDER_SUBTLE
            text_color = Theme.TEXT_DISABLED
        else:
            fill = self.current_bg
            border = self.border_color if (self.hovered or self.focused) else self._subtle_border()
            text_color = self.fg_color

        self._create_rounded_rect(
            pad, pad, self.width - pad, self.height - pad,
            self.corner_radius,
            fill=fill, outline=border, width=1,
        )

        # Press offset
        text_y = self.height // 2 + (1 if self.pressed else 0)

        if self.icon:
            gap = 6
            icon_font = (Theme.FONT_FAMILY, self.font_size + 1, "bold")
            text_font = (Theme.FONT_FAMILY, self.font_size, "bold")
            icon_w = self._text_width(self.icon, icon_font)
            text_w = self._text_width(self.text, text_font)
            total = icon_w + gap + text_w
            start_x = (self.width - total) // 2
            self.create_text(start_x + icon_w // 2, text_y,
                             text=self.icon, fill=text_color, font=icon_font)
            self.create_text(start_x + icon_w + gap + text_w // 2, text_y,
                             text=self.text, fill=text_color, font=text_font)
        else:
            self.create_text(self.width // 2, text_y, text=self.text,
                             fill=text_color, font=(Theme.FONT_FAMILY, self.font_size, "bold"))

    def _subtle_border(self):
        # For filled CTAs, border should match the fill for a flat look
        if self.style in ("primary", "danger"):
            return self.bg_color
        return Theme.BORDER_SUBTLE

    def _text_width(self, text, font):
        try:
            return tkfont.Font(font=font).measure(text)
        except Exception:
            return len(text) * 7

    def _create_rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [
            x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
            x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
            x1, y2, x1, y2 - r, x1, y1 + r, x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

    def _on_enter(self, event):
        if self.enabled:
            self.hovered = True
            self.current_bg = self.hover_color
            self._draw()
            self.config(cursor="hand2")

    def _on_leave(self, event):
        if self.enabled:
            self.hovered = False
            self.pressed = False
            self.current_bg = self.bg_color
            self._draw()
            self.config(cursor="")

    def _on_click(self, event):
        if self.enabled:
            self.focus_set()
            self.pressed = True
            self.current_bg = self.press_color
            self._draw()

    def _on_release(self, event):
        if self.enabled:
            inside = 0 <= event.x <= self.width and 0 <= event.y <= self.height
            self.pressed = False
            self.current_bg = self.hover_color if inside else self.bg_color
            self._draw()
            if inside and self.command:
                self.command()

    def _on_focus_in(self, event):
        self.focused = True
        self._draw()

    def _on_focus_out(self, event):
        self.focused = False
        self.pressed = False
        self.current_bg = self.bg_color
        self._draw()

    def _on_keyboard_activate(self, event):
        if self.enabled and self.command:
            self.command()

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        self.current_bg = self.bg_color if enabled else Theme.BG_TERTIARY
        self.config(cursor="hand2" if enabled else "")
        self._draw()

    def set_text(self, text: str):
        self.text = text
        self._draw()

    def set_style(self, style: str):
        """Re-skin the button (e.g., primary -> danger during processing)."""
        self._apply_style(style)
        self.style = style
        self.current_bg = self.bg_color
        self._draw()


class ModernProgressBar(tk.Canvas):
    """A refined progress bar. Rounded track + fill. Smoothly tweens to
    target progress values so updates feel continuous rather than stepped."""

    TWEEN_STEP = 0.04
    TWEEN_DELAY_MS = 16  # ~60fps cap

    def __init__(self, parent, width=400, height=6, bg=Theme.PROGRESS_BG,
                 fill=Theme.PROGRESS_FILL, corner_radius=None, **kwargs):
        if corner_radius is None:
            corner_radius = max(2, height // 2)
        super().__init__(parent, width=width, height=height, highlightthickness=0,
                        bg=parent.cget('bg') if hasattr(parent, 'cget') else Theme.BG_DARK)

        self.bar_width = width
        self.bar_height = height
        self.corner_radius = corner_radius
        self.bg_color = bg
        self.fill_color = fill
        self.progress = 0.0
        self._target = 0.0
        self._tween_id = None

        self._draw()

    def _draw(self):
        self.delete("all")
        r = self.corner_radius

        self._create_rounded_rect(0, 0, self.bar_width, self.bar_height, r, fill=self.bg_color)

        if self.progress > 0:
            fill_width = max(r * 2, int(self.bar_width * self.progress))
            self._create_rounded_rect(0, 0, fill_width, self.bar_height, r, fill=self.fill_color)

    def _create_rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [
            x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
            x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
            x1, y2, x1, y2 - r, x1, y1 + r, x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

    def set_progress(self, value: float, animate: bool = True):
        """Set the displayed progress. With `animate=True`, eases from the
        current value to the target over several frames."""
        target = max(0.0, min(1.0, value))
        self._target = target
        if self._tween_id:
            try:
                self.after_cancel(self._tween_id)
            except tk.TclError:
                pass
            self._tween_id = None
        # For big backward jumps (e.g. reset to 0), snap directly
        if not animate or target == 0.0 or abs(target - self.progress) < 0.005:
            self.progress = target
            self._draw()
            return
        self._tween_step()

    def _tween_step(self):
        delta = self._target - self.progress
        if abs(delta) < 0.003:
            self.progress = self._target
            self._draw()
            self._tween_id = None
            return
        # Ease-out: move 18% of remaining distance per frame, min 0.4%
        step = delta * 0.18
        if abs(step) < 0.004:
            step = 0.004 if delta > 0 else -0.004
        self.progress = max(0.0, min(1.0, self.progress + step))
        self._draw()
        try:
            self._tween_id = self.after(self.TWEEN_DELAY_MS, self._tween_step)
        except tk.TclError:
            self._tween_id = None

    def set_color(self, color: str):
        self.fill_color = color
        self._draw()

    def resize(self, width: int, height: int = None):
        """Resize the progress bar (for DPI/layout changes)."""
        self.bar_width = width
        if height:
            self.bar_height = height
            self.corner_radius = max(2, height // 2)
        self.config(width=self.bar_width, height=self.bar_height)
        self._draw()


class ModernToggle(tk.Canvas):
    """Custom checkbox/toggle replacement for tk.Checkbutton.

    Renders as a rounded square indicator with a checkmark, followed by
    a text label. Full support for hover/focus/disabled states, keyboard
    activation, and tk.BooleanVar binding.
    """

    BOX = 18
    GAP = 10

    def __init__(self, parent, text="", variable=None, command=None,
                 bg=None, fg=None, **kwargs):
        self.variable = variable if variable is not None else tk.BooleanVar(value=False)
        self.text = text
        self.command = command
        self.enabled = True
        self.focused = False
        self.hovered = False
        self.parent_bg = bg or (parent.cget('bg') if hasattr(parent, 'cget') else Theme.BG_CARD)
        self.fg_color = fg or Theme.TEXT_PRIMARY

        # Measure text width for canvas sizing
        self._font = f(Theme.F_BODY_SM)
        text_w = tkfont.Font(font=self._font).measure(text)
        total_w = self.BOX + self.GAP + text_w + 4
        super().__init__(parent, width=total_w, height=max(self.BOX + 4, 24),
                         highlightthickness=0, bg=self.parent_bg, takefocus=1)

        self._draw()
        self.bind("<Button-1>", self._toggle)
        self.bind("<space>", self._toggle)
        self.bind("<Return>", self._toggle)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<FocusIn>", self._on_focus_in)
        self.bind("<FocusOut>", self._on_focus_out)
        if self.variable is not None:
            self.variable.trace_add("write", lambda *_: self._draw())

    def _draw(self):
        self.delete("all")
        y0 = (int(self["height"]) - self.BOX) // 2
        x0 = 2

        checked = bool(self.variable.get())

        # Focus ring
        if self.focused and self.enabled:
            self._rounded(x0 - 2, y0 - 2, x0 + self.BOX + 2, y0 + self.BOX + 2,
                          Theme.R_SM + 2, fill=Theme.BG_DARK, outline=Theme.BORDER_FOCUS, width=1)

        # Box
        if not self.enabled:
            box_fill = Theme.BG_TERTIARY
            box_border = Theme.BORDER_SUBTLE
        elif checked:
            box_fill = Theme.BLUE_PRIMARY
            box_border = Theme.BLUE_HOVER
        else:
            box_fill = Theme.BG_TERTIARY
            box_border = Theme.BORDER_STRONG if self.hovered else Theme.BORDER

        self._rounded(x0, y0, x0 + self.BOX, y0 + self.BOX, Theme.R_SM,
                      fill=box_fill, outline=box_border, width=1)

        # Checkmark
        if checked:
            stroke = "#04120b" if self.enabled else Theme.TEXT_DISABLED
            self.create_line(x0 + 4, y0 + 9, x0 + 8, y0 + 13,
                             fill=stroke, width=2, capstyle="round")
            self.create_line(x0 + 8, y0 + 13, x0 + 14, y0 + 5,
                             fill=stroke, width=2, capstyle="round")

        # Label
        text_color = self.fg_color if self.enabled else Theme.TEXT_DISABLED
        self.create_text(x0 + self.BOX + self.GAP, int(self["height"]) // 2,
                         text=self.text, anchor="w",
                         font=self._font, fill=text_color)

    def _rounded(self, x1, y1, x2, y2, r, **kw):
        points = [
            x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
            x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
            x1, y2, x1, y2 - r, x1, y1 + r, x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kw)

    def _toggle(self, event=None):
        if not self.enabled:
            return
        self.focus_set()
        self.variable.set(not self.variable.get())
        self._draw()
        if self.command:
            self.command()

    def _on_enter(self, event):
        if self.enabled:
            self.hovered = True
            self.config(cursor="hand2")
            self._draw()

    def _on_leave(self, event):
        self.hovered = False
        self.config(cursor="")
        self._draw()

    def _on_focus_in(self, event):
        self.focused = True
        self._draw()

    def _on_focus_out(self, event):
        self.focused = False
        self._draw()

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        self.config(cursor="hand2" if enabled else "")
        self._draw()


class ModernSlider(tk.Frame):
    """Premium slider: rounded track, filled portion in accent color,
    prominent thumb, value pill on the right. Canvas-based so styling is
    fully controlled."""

    TRACK_H = 4
    THUMB_R = 8
    HEIGHT = 28

    def __init__(self, parent, from_=0, to=100, value=0,
                 command=None, bg=None, width=220, **kwargs):
        self.parent_bg = bg or (parent.cget('bg') if hasattr(parent, 'cget') else Theme.BG_CARD)
        super().__init__(parent, bg=self.parent_bg)

        self.from_ = from_
        self.to = to
        self.value = max(from_, min(to, value))
        self.command = command
        self._width = width
        self._dragging = False

        self.canvas = tk.Canvas(self, width=width, height=self.HEIGHT,
                                highlightthickness=0, bg=self.parent_bg, takefocus=1)
        self.canvas.pack(side="left", fill="x", expand=True, padx=(0, 0))

        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Left>", lambda e: self._step(-1))
        self.canvas.bind("<Right>", lambda e: self._step(1))
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self._draw()

    def _on_resize(self, event):
        self._width = max(60, event.width)
        self._draw()

    def _value_to_x(self, v):
        if self.to == self.from_:
            return self.THUMB_R
        pct = (v - self.from_) / (self.to - self.from_)
        return int(self.THUMB_R + pct * (self._width - self.THUMB_R * 2))

    def _x_to_value(self, x):
        if self._width <= self.THUMB_R * 2:
            return self.from_
        pct = (x - self.THUMB_R) / (self._width - self.THUMB_R * 2)
        pct = max(0.0, min(1.0, pct))
        return self.from_ + pct * (self.to - self.from_)

    def _draw(self):
        self.canvas.delete("all")
        mid = self.HEIGHT // 2
        left = self.THUMB_R
        right = self._width - self.THUMB_R

        # Track background
        self.canvas.create_rectangle(
            left, mid - self.TRACK_H // 2, right, mid + self.TRACK_H // 2,
            fill=Theme.BG_TERTIARY, outline="",
        )

        thumb_x = self._value_to_x(self.value)
        # Filled portion
        if thumb_x > left:
            self.canvas.create_rectangle(
                left, mid - self.TRACK_H // 2, thumb_x, mid + self.TRACK_H // 2,
                fill=Theme.BLUE_PRIMARY, outline="",
            )

        # Thumb
        self.canvas.create_oval(
            thumb_x - self.THUMB_R - 1, mid - self.THUMB_R - 1,
            thumb_x + self.THUMB_R + 1, mid + self.THUMB_R + 1,
            fill=Theme.BG_DARK, outline="",
        )
        self.canvas.create_oval(
            thumb_x - self.THUMB_R, mid - self.THUMB_R,
            thumb_x + self.THUMB_R, mid + self.THUMB_R,
            fill=Theme.BLUE_PRIMARY, outline=Theme.BLUE_HOVER, width=1,
        )

    def _on_press(self, event):
        self.canvas.focus_set()
        self._dragging = True
        self._set_from_x(event.x)

    def _on_drag(self, event):
        if self._dragging:
            self._set_from_x(event.x)

    def _on_release(self, event):
        self._dragging = False

    def _on_wheel(self, event):
        self._step(1 if event.delta > 0 else -1)

    def _step(self, direction):
        step = max(1, int((self.to - self.from_) / 50))
        new_val = max(self.from_, min(self.to, int(self.value) + direction * step))
        self._set_value(new_val)

    def _set_from_x(self, x):
        new_val = int(round(self._x_to_value(x)))
        self._set_value(new_val)

    def _set_value(self, v):
        v = max(self.from_, min(self.to, v))
        if v == self.value:
            return
        self.value = v
        self._draw()
        if self.command:
            self.command(v)

    def set(self, v):
        self._set_value(int(v))

    def get(self):
        return int(self.value)


def show_confirm(parent, title: str, message: str, detail: str = "",
                 confirm_label: str = "Confirm",
                 cancel_label: str = "Cancel",
                 tone: str = "primary") -> bool:
    """Themed modal confirmation dialog that matches the app aesthetic.

    Returns True if confirmed, False if cancelled (or closed).
    `tone` selects the confirm button style: primary / danger / accent.
    """
    result = {"value": False}

    dialog = tk.Toplevel(parent)
    dialog.withdraw()
    dialog.title(title)
    dialog.configure(bg=Theme.BG_OVERLAY)
    dialog.resizable(False, False)
    dialog.transient(parent)

    outer = tk.Frame(dialog, bg=Theme.BORDER, padx=1, pady=1)
    outer.pack()
    body = tk.Frame(outer, bg=Theme.BG_SECONDARY)
    body.pack()

    # Content
    content = tk.Frame(body, bg=Theme.BG_SECONDARY)
    content.pack(padx=28, pady=(24, 14))

    tk.Label(content, text=title, font=f(Theme.F_HEADING, "bold"),
             bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY,
             anchor="w", justify="left").pack(anchor="w")
    tk.Label(content, text=message, font=f(Theme.F_BODY),
             bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY,
             anchor="w", justify="left", wraplength=420).pack(
                 anchor="w", pady=(6, 0))
    if detail:
        tk.Label(content, text=detail, font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED,
                 anchor="w", justify="left", wraplength=420).pack(
                     anchor="w", pady=(8, 0))

    # Action row
    actions = tk.Frame(body, bg=Theme.BG_CARD)
    actions.pack(fill="x")
    inner_actions = tk.Frame(actions, bg=Theme.BG_CARD)
    inner_actions.pack(side="right", padx=16, pady=14)

    def _cancel():
        dialog.grab_release()
        dialog.destroy()

    def _confirm():
        result["value"] = True
        dialog.grab_release()
        dialog.destroy()

    cancel_btn = ModernButton(inner_actions, text=cancel_label, width=96,
                              command=_cancel, style="ghost", size="md")
    cancel_btn.pack(side="left")

    confirm_btn = ModernButton(inner_actions, text=confirm_label, width=118,
                               command=_confirm, style=tone, size="md")
    confirm_btn.pack(side="left", padx=(Theme.S_SM, 0))

    dialog.bind("<Escape>", lambda e: _cancel())
    dialog.bind("<Return>", lambda e: _confirm())
    dialog.protocol("WM_DELETE_WINDOW", _cancel)

    # Center on parent
    dialog.update_idletasks()
    try:
        px = parent.winfo_rootx()
        py = parent.winfo_rooty()
        pw = parent.winfo_width()
        ph = parent.winfo_height()
        dw = dialog.winfo_reqwidth()
        dh = dialog.winfo_reqheight()
        x = px + (pw - dw) // 2
        y = py + (ph - dh) // 3
        dialog.geometry(f"+{x}+{y}")
    except Exception:
        pass

    dialog.deiconify()
    dialog.grab_set()
    confirm_btn.focus_set()
    dialog.wait_window()
    return result["value"]


class TaskbarProgress:
    """Thin wrapper over ITaskbarList3 for Windows 7+ taskbar progress.

    Falls back to no-op on non-Windows or when COM is unavailable.
    State values per MSDN:
        0 = NOPROGRESS, 1 = INDETERMINATE, 2 = NORMAL, 4 = ERROR, 8 = PAUSED
    """

    STATE_NONE = 0
    STATE_INDETERMINATE = 1
    STATE_NORMAL = 2
    STATE_ERROR = 4
    STATE_PAUSED = 8

    def __init__(self, hwnd):
        self._taskbar = None
        self._hwnd = hwnd
        if sys.platform != "win32":
            return
        try:
            import comtypes.client  # type: ignore
            # CLSID_TaskbarList
            self._taskbar = comtypes.client.CreateObject(
                "{56FDF344-FD6D-11D0-958A-006097C9A090}",
                interface=comtypes.GUID("{EA1AFB91-9E28-4B86-90E9-9E9F8A5EEFAF}"),
            )
            self._taskbar.HrInit()
        except Exception:
            self._taskbar = None

    def set_value(self, current: int, total: int):
        if not self._taskbar or not self._hwnd:
            return
        try:
            self._taskbar.SetProgressValue(self._hwnd, current, max(total, 1))
        except Exception:
            pass

    def set_state(self, state: int):
        if not self._taskbar or not self._hwnd:
            return
        try:
            self._taskbar.SetProgressState(self._hwnd, state)
        except Exception:
            pass

    def clear(self):
        self.set_state(self.STATE_NONE)


def make_themed_menu(parent) -> tk.Menu:
    """Create a `tk.Menu` styled for the dark theme."""
    menu = tk.Menu(
        parent,
        tearoff=0,
        bg=Theme.BG_RAISED,
        fg=Theme.TEXT_PRIMARY,
        activebackground=Theme.BLUE_MUTED,
        activeforeground=Theme.TEXT_PRIMARY,
        disabledforeground=Theme.TEXT_DISABLED,
        relief="flat",
        bd=0,
        font=f(Theme.F_BODY_SM),
        activeborderwidth=0,
    )
    return menu


class Toast:
    """Lightweight transient notification, anchored to the bottom-right of
    the root window. Fades after TIMEOUT_MS."""

    TIMEOUT_MS = 2600
    _active: List['Toast'] = []

    def __init__(self, root, message: str, tone: str = "success"):
        self.root = root
        self.message = message
        self.tone = tone
        self._win = None
        self._fade_id = None
        self._build()
        Toast._active.append(self)
        self._schedule_close()

    @classmethod
    def show(cls, root, message: str, tone: str = "success"):
        return cls(root, message, tone)

    def _tone_color(self):
        return {
            "success": Theme.SUCCESS,
            "warning": Theme.WARNING,
            "error": Theme.ERROR,
            "info": Theme.INFO,
        }.get(self.tone, Theme.TEXT_SECONDARY)

    def _build(self):
        try:
            self._win = tk.Toplevel(self.root)
            self._win.wm_overrideredirect(True)
            self._win.configure(bg=Theme.BORDER_STRONG)
            self._win.attributes("-topmost", True)
            try:
                self._win.attributes("-alpha", 0.97)
            except tk.TclError:
                pass

            card = tk.Frame(self._win, bg=Theme.BG_RAISED)
            card.pack(padx=1, pady=1)

            # Left color stripe
            stripe = tk.Frame(card, bg=self._tone_color(), width=3)
            stripe.pack(side="left", fill="y")

            content = tk.Frame(card, bg=Theme.BG_RAISED)
            content.pack(side="left", padx=(12, 18), pady=10)

            tk.Label(content, text=self.message, font=f(Theme.F_BODY_SM, "bold"),
                     bg=Theme.BG_RAISED, fg=Theme.TEXT_PRIMARY).pack(anchor="w")

            self._win.update_idletasks()
            self._position()
        except tk.TclError:
            self._win = None

    def _position(self):
        try:
            w = self._win.winfo_reqwidth()
            h = self._win.winfo_reqheight()
            rx = self.root.winfo_rootx()
            ry = self.root.winfo_rooty()
            rw = self.root.winfo_width()
            rh = self.root.winfo_height()
            # Stack toasts upward from the bottom-right
            offset = sum((t._win.winfo_reqheight() + 8)
                         for t in Toast._active[:-1] if t._win)
            x = rx + rw - w - 20
            y = ry + rh - h - 52 - offset
            self._win.wm_geometry(f"+{x}+{y}")
        except Exception:
            pass

    def _schedule_close(self):
        try:
            self._fade_id = self.root.after(self.TIMEOUT_MS, self._begin_fade)
        except tk.TclError:
            pass

    def _begin_fade(self):
        """Fade the toast out over ~300ms using the -alpha attribute, then
        destroy and restack any later toasts."""
        if not self._win:
            return
        steps = [0.85, 0.65, 0.45, 0.25, 0.08]

        def apply(i):
            if not self._win:
                return
            try:
                self._win.attributes("-alpha", steps[i])
            except tk.TclError:
                pass
            if i + 1 < len(steps):
                try:
                    self.root.after(45, lambda: apply(i + 1))
                except tk.TclError:
                    pass
            else:
                self._close()

        apply(0)

    def _close(self):
        try:
            if self._win:
                self._win.destroy()
        except tk.TclError:
            pass
        self._win = None
        if self in Toast._active:
            Toast._active.remove(self)
        # Reposition remaining toasts upward so gaps don't linger
        for t in Toast._active:
            try:
                t._position()
            except Exception:
                pass


class SegmentedPicker(tk.Frame):
    """A segmented radio-style selector. Renders a horizontal group of
    Canvas-based buttons. Used for the algorithm picker."""

    def __init__(self, parent, options: List[Tuple[str, str]],
                 value: str = None, command: Callable = None,
                 bg: str = None, **kwargs):
        """options: list of (value, label) tuples."""
        self.parent_bg = bg or (parent.cget('bg') if hasattr(parent, 'cget')
                                else Theme.BG_CARD)
        super().__init__(parent, bg=self.parent_bg)
        self.options = options
        self.value = value or (options[0][0] if options else None)
        self.command = command
        self._segments: dict = {}

        wrap = tk.Frame(self, bg=Theme.BG_TERTIARY, highlightthickness=1,
                        highlightbackground=Theme.BORDER)
        wrap.pack(fill="x")

        for val, label in options:
            seg = _Segment(wrap, label=label, value=val,
                            on_select=self._select,
                            selected=(val == self.value))
            seg.pack(side="left", fill="x", expand=True, padx=1, pady=1)
            self._segments[val] = seg

    def _select(self, val):
        if val == self.value:
            return
        self.value = val
        for v, seg in self._segments.items():
            seg.set_selected(v == val)
        if self.command:
            self.command(val)

    def set(self, val: str):
        if val in self._segments:
            self._select(val)

    def get(self) -> str:
        return self.value


class _Segment(tk.Canvas):
    """Single button inside a SegmentedPicker."""

    H = 30

    def __init__(self, parent, label: str, value: str, on_select: Callable,
                 selected: bool = False):
        super().__init__(parent, height=self.H, highlightthickness=0,
                         bg=Theme.BG_TERTIARY, takefocus=1)
        self.label = label
        self.value = value
        self.on_select = on_select
        self.selected = selected
        self.hovered = False
        self.focused = False

        self.bind("<Button-1>", self._click)
        self.bind("<Return>", self._click)
        self.bind("<space>", self._click)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<FocusIn>", lambda e: self._set_focused(True))
        self.bind("<FocusOut>", lambda e: self._set_focused(False))
        self.bind("<Configure>", lambda e: self._draw())
        self._draw()

    def _on_enter(self, event):
        self.hovered = True
        self.config(cursor="hand2")
        self._draw()

    def _on_leave(self, event):
        self.hovered = False
        self.config(cursor="")
        self._draw()

    def _set_focused(self, focused):
        self.focused = focused
        self._draw()

    def _click(self, event=None):
        self.focus_set()
        self.on_select(self.value)

    def set_selected(self, selected: bool):
        self.selected = selected
        self._draw()

    def _draw(self):
        self.delete("all")
        w = max(1, int(self["width"]) if int(self["width"]) > 1 else self.winfo_width())
        if w <= 1:
            w = self.winfo_width()
        h = self.H
        if self.selected:
            bg = Theme.GREEN_MUTED
            fg = Theme.GREEN_PRIMARY
            border = Theme.GREEN_HOVER
        elif self.hovered:
            bg = Theme.BG_CARD_HOVER
            fg = Theme.TEXT_PRIMARY
            border = Theme.BORDER_STRONG
        else:
            bg = Theme.BG_TERTIARY
            fg = Theme.TEXT_SECONDARY
            border = Theme.BG_TERTIARY
        self.create_rectangle(0, 0, w, h, fill=bg, outline=border, width=1)
        if self.focused:
            self.create_rectangle(1, 1, w - 1, h - 1, outline=Theme.BORDER_FOCUS,
                                  width=1)
        font_w = "bold" if self.selected else "normal"
        self.create_text(w // 2, h // 2, text=self.label,
                         fill=fg, font=f(Theme.F_BODY_SM, font_w))


class DragDropFrame(tk.Frame):
    """A calm drop target surface with a single clear import action."""

    def __init__(self, parent, on_drop: Callable[[List[str]], None],
                 width=400, height=200, **kwargs):
        super().__init__(parent, bg=Theme.BG_CARD, highlightthickness=1,
                        highlightbackground=Theme.BORDER, highlightcolor=Theme.BLUE_PRIMARY,
                        takefocus=1)

        self.on_drop = on_drop
        self.normal_bg = Theme.BG_CARD
        self.hover_bg = Theme.BG_CARD_HOVER
        self.hovered = False
        self.focused = False
        self.configure(height=height)
        self.pack_propagate(False)
        self.grid_propagate(False)
        self.config(cursor="hand2")

        # Inner content
        inner = tk.Frame(self, bg=self.normal_bg)
        inner.place(relx=0.5, rely=0.5, anchor="center")
        self._surface_widgets = [self, inner]

        # Import glyph
        glyph = tk.Label(inner, text="+", font=f(22, "bold"),
                         bg=self.normal_bg, fg=Theme.BLUE_PRIMARY)
        glyph.pack()

        # Main text
        main_text = tk.Label(inner, text="Add files to the queue",
                            font=f(Theme.F_TITLE, "bold"), bg=self.normal_bg,
                            fg=Theme.TEXT_PRIMARY)
        main_text.pack(pady=(2, 0))

        # Sub text
        sub_text = tk.Label(inner,
                           text="Drag files here, choose files, or choose a folder. Originals stay untouched.",
                           font=f(Theme.F_BODY_SM), bg=self.normal_bg,
                           fg=Theme.TEXT_SECONDARY, justify="center", wraplength=480)
        sub_text.pack(pady=(6, 12))

        actions = tk.Frame(inner, bg=self.normal_bg)
        actions.pack()

        self.add_files_btn = ModernButton(actions, text="Choose files", width=124,
                                          command=self._open_file_dialog,
                                          style="accent", size="md")
        self.add_files_btn.pack(side="left")

        self.add_folder_btn = ModernButton(actions, text="Choose folder", width=118,
                                           command=self._open_folder_dialog,
                                           style="secondary", size="md")
        self.add_folder_btn.pack(side="left", padx=(8, 0))

        support_text = tk.Label(inner,
                                text="Videos and images supported",
                                font=f(Theme.F_META, "bold"), bg=self.normal_bg,
                                fg=Theme.TEXT_DISABLED)
        support_text.pack(pady=(12, 0))
        self._surface_widgets.extend([glyph, main_text, sub_text, actions, support_text])

        # Bind click (left = files, right = folder)
        self.bind("<Button-1>", self._on_click)
        self.bind("<Button-3>", self._on_right_click)
        self.bind("<Enter>", self._on_enter, add="+")
        self.bind("<Leave>", self._on_leave, add="+")
        self.bind("<FocusIn>", self._on_focus_in, add="+")
        self.bind("<FocusOut>", self._on_focus_out, add="+")
        self.bind("<Return>", lambda e: self._open_file_dialog())
        self.bind("<space>", lambda e: self._open_file_dialog())
        for child in (inner, glyph, main_text, sub_text, support_text):
            child.bind("<Button-1>", self._on_click)
            child.bind("<Button-3>", self._on_right_click)
            child.bind("<Enter>", self._on_enter, add="+")
            child.bind("<Leave>", self._on_leave, add="+")

        # Try to enable native drag-drop (Windows)
        try:
            self._setup_dnd()
        except Exception:
            pass

    def _set_bg(self, bg: str, border: str):
        self.config(bg=bg, highlightbackground=border)
        for widget in self._surface_widgets:
            if isinstance(widget, tk.Widget):
                try:
                    widget.config(bg=bg)
                except tk.TclError:
                    pass
        for button in (self.add_files_btn, self.add_folder_btn):
            button.config(bg=bg)

    def _setup_dnd(self):
        """Setup native drag and drop if available."""
        try:
            import tkinterdnd2
            self.drop_target_register(tkinterdnd2.DND_FILES)
            self.dnd_bind('<<Drop>>', self._handle_drop)
        except ImportError:
            pass

    def _handle_drop(self, event):
        files = self.tk.splitlist(event.data)
        # Accept both files and folders
        valid = [f for f in files if is_video_file(f) or is_image_file(f) or Path(f).is_dir()]
        if valid:
            self.on_drop(valid)

    def _open_file_dialog(self):
        files = filedialog.askopenfilenames(
            title="Choose files to clean",
            filetypes=[
                ("All Supported", "*.mp4;*.avi;*.mkv;*.mov;*.wmv;*.flv;*.webm;*.m4v;*.mpeg;*.mpg;*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.webp"),
                ("Video Files", "*.mp4;*.avi;*.mkv;*.mov;*.wmv;*.flv;*.webm;*.m4v;*.mpeg;*.mpg"),
                ("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.webp"),
                ("All Files", "*.*"),
            ]
        )
        if files:
            self.on_drop(list(files))

    def _open_folder_dialog(self):
        folder = filedialog.askdirectory(title="Choose a folder to clean")
        if folder:
            self.on_drop([folder])

    def _on_click(self, event):
        self.focus_set()
        self._open_file_dialog()

    def _on_right_click(self, event):
        self.focus_set()
        self._open_folder_dialog()

    def _on_enter(self, event):
        self.hovered = True
        self._set_bg(self.hover_bg, Theme.BORDER_FOCUS if self.focused else Theme.BLUE_PRIMARY)

    def _on_leave(self, event):
        self.hovered = False
        self._set_bg(self.normal_bg, Theme.BORDER_FOCUS if self.focused else Theme.BORDER)

    def _on_focus_in(self, event):
        self.focused = True
        self._set_bg(self.hover_bg if self.hovered else self.normal_bg, Theme.BORDER_FOCUS)

    def _on_focus_out(self, event):
        self.focused = False
        self._set_bg(self.hover_bg if self.hovered else self.normal_bg,
                     Theme.BLUE_PRIMARY if self.hovered else Theme.BORDER)


class QueueItemWidget(tk.Frame):
    """A single queue item card. Clear hierarchy: filename + status pill,
    compact meta row, progress bar, and row of actions. Selected state
    shows a left-edge accent stripe."""

    def __init__(self, parent, item: QueueItem, on_remove: Callable,
                 on_select: Callable = None, on_rename: Callable = None,
                 on_repeat: Callable = None, on_cancel_item: Callable = None,
                 on_override: Callable = None,
                 **kwargs):
        super().__init__(parent, bg=Theme.BG_CARD, highlightthickness=1,
                        highlightbackground=Theme.BORDER)

        self.item = item
        self.on_remove = on_remove
        self.on_select = on_select
        self.on_rename = on_rename
        self.on_repeat = on_repeat
        self.on_cancel_item = on_cancel_item
        self.on_override = on_override
        self.is_selected = False
        self._surface_bg = Theme.BG_CARD
        self._pulse_id = None
        self._pulse_phase = 0

        # Left accent stripe (visible only when selected)
        self.accent_stripe = tk.Frame(self, bg=Theme.BG_CARD, width=3)
        self.accent_stripe.pack(side="left", fill="y")

        # Main container with padding
        self.container = tk.Frame(self, bg=self._surface_bg)
        self.container.pack(fill="x", padx=Theme.S_MD, pady=Theme.S_MD)

        # Top row: filename and status
        self.top_row = tk.Frame(self.container, bg=self._surface_bg)
        self.top_row.pack(fill="x")

        self.name_label = tk.Label(self.top_row,
                                   text=truncate_middle(Path(item.file_path).name, 46),
                                   font=f(Theme.F_BODY, "bold"),
                                   bg=self._surface_bg, fg=Theme.TEXT_PRIMARY,
                                   cursor="hand2")
        self.name_label.pack(side="left")
        Tooltip(self.name_label, item.file_path)

        # Status pill (rounded by adding generous padx)
        badge = status_ui(item.status)
        self.status_badge = tk.Label(self.top_row, text=badge["label"],
                                     font=f(Theme.F_META, "bold"),
                                     bg=badge["bg"], fg=badge["color"],
                                     padx=Theme.S_SM, pady=Theme.S_XS)
        self.status_badge.pack(side="right")

        # File info row (meta caption)
        file_info = get_file_info(item.file_path)
        self.info_label = tk.Label(self.container,
                                   text=f"{file_info}   -   {truncate_middle(item.file_path, 68)}",
                                   font=f(Theme.F_META),
                                   bg=self._surface_bg, fg=Theme.TEXT_MUTED, anchor="w")
        self.info_label.pack(fill="x", pady=(Theme.S_XS, 0))

        # Progress bar (resizes with container)
        self.progress_bar = ModernProgressBar(self.container, width=300, height=5,
                                              fill=self._get_status_color())
        self.progress_bar.pack(fill="x", pady=(Theme.S_MD, Theme.S_XS))
        self.progress_bar.set_progress(item.progress)
        def _resize_bar(event):
            bar_w = event.width - 4
            if bar_w > 20:
                self.progress_bar.resize(bar_w)
        self.container.bind("<Configure>", _resize_bar)

        # Bottom row: message + elapsed time
        self.bottom_row = tk.Frame(self.container, bg=self._surface_bg)
        self.bottom_row.pack(fill="x")

        self.message_label = tk.Label(self.bottom_row, text=item.message or "Ready to process",
                                      font=f(Theme.F_BODY_SM), bg=self._surface_bg,
                                      fg=Theme.TEXT_SECONDARY, anchor="w")
        self.message_label.pack(side="left", fill="x", expand=True)

        self.time_label = tk.Label(self.bottom_row, text="",
                                   font=f(Theme.F_META, "bold"),
                                   bg=self._surface_bg, fg=Theme.TEXT_MUTED, anchor="e")
        self.time_label.pack(side="right")

        self.actions_row = tk.Frame(self.container, bg=self._surface_bg)
        self.actions_row.pack(fill="x", pady=(Theme.S_MD, 0))

        self.remove_btn = ModernButton(self.actions_row, text="Remove", width=78,
                                       command=lambda: self.on_remove(self.item.id),
                                       style="ghost", size="sm")
        self.remove_btn.pack(side="left")

        self.open_btn = ModernButton(self.actions_row, text="Open result", width=104,
                                     command=self._open_output, style="accent",
                                     size="sm")
        self.open_btn.pack(side="right")

        self._interactive_widgets = [
            self, self.container, self.top_row, self.name_label, self.info_label,
            self.bottom_row, self.message_label, self.time_label, self.actions_row,
        ]
        for widget in self._interactive_widgets:
            widget.bind("<Enter>", self._on_enter, add="+")
            widget.bind("<Leave>", self._on_leave, add="+")
            widget.bind("<Button-1>", self._on_card_click, add="+")
            widget.bind("<Button-3>", self._on_context_menu, add="+")

        self.update_item(item)

    def _on_context_menu(self, event):
        """Show a themed right-click menu for this queue item."""
        menu = make_themed_menu(self)
        is_active = self.item.status in (
            ProcessingStatus.LOADING, ProcessingStatus.DETECTING,
            ProcessingStatus.PROCESSING, ProcessingStatus.MERGING,
        )
        is_complete = (self.item.status == ProcessingStatus.COMPLETE
                       and Path(self.item.output_path).exists())

        menu.add_command(label="Preview source frame",
                         command=self._request_preview)
        menu.add_command(label="Review subtitle mask",
                         command=self._request_mask_preview)
        menu.add_separator()
        menu.add_command(label="Open result",
                         command=self._open_output,
                         state="normal" if is_complete else "disabled")
        menu.add_command(label="Reveal output folder",
                         command=self._reveal_output,
                         state="normal" if is_complete else "disabled")
        menu.add_separator()
        # Only allow renaming output before processing has started.
        rename_allowed = self.item.status == ProcessingStatus.IDLE and self.on_rename is not None
        menu.add_command(label="Rename output...",
                         command=lambda: self.on_rename(self.item.id) if self.on_rename else None,
                         state="normal" if rename_allowed else "disabled")
        menu.add_command(label="Copy source path",
                         command=self._copy_source_path)
        # RM-28: re-queue the same source with the snapshot of settings
        # that was active on this item. Useful when re-running with a
        # tweaked global config but you still want exactly the same
        # per-file overrides as a previous run.
        if self.on_repeat is not None:
            menu.add_command(label="Repeat with these settings",
                             command=lambda: self.on_repeat(self.item.id))
        # F-7: per-item cancel. Only meaningful while the item is
        # actively running -- on IDLE entries we surface "Remove"
        # below as the equivalent action.
        if self.on_cancel_item is not None and is_active:
            menu.add_command(label="Cancel this item",
                             command=lambda: self.on_cancel_item(self.item.id))
        # RM-29: open the per-file override dialog so users can change
        # mode / language / sensitivity for a single queued item
        # without touching the global settings.
        if self.on_override is not None and self.item.status == ProcessingStatus.IDLE:
            menu.add_command(label="Override settings for this file...",
                             command=lambda: self.on_override(self.item.id))
        menu.add_separator()
        menu.add_command(label="Remove from queue",
                         command=lambda: self.on_remove(self.item.id),
                         state="disabled" if is_active else "normal")

        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _reveal_output(self):
        """Open the folder containing the output in Explorer."""
        if self.item.status == ProcessingStatus.COMPLETE and Path(self.item.output_path).exists():
            try:
                os.startfile(str(Path(self.item.output_path).parent))
            except Exception as exc:
                logger.warning(f"Could not open output folder: {exc}")

    def _copy_source_path(self):
        """Copy the source file path to the clipboard."""
        try:
            self.clipboard_clear()
            self.clipboard_append(self.item.file_path)
        except tk.TclError:
            pass

    def _request_preview(self):
        if self.on_select:
            self.on_select(self.item)

    def _request_mask_preview(self):
        if self.on_select:
            self.on_select(self.item, show_mask=True)

    def _on_card_click(self, event):
        if self.on_select:
            self.on_select(self.item)

    def _on_enter(self, event):
        if not self.is_selected:
            self._apply_surface_state(Theme.BG_CARD_HOVER, Theme.BORDER)

    def _on_leave(self, event):
        if not self.is_selected:
            self._apply_surface_state(Theme.BG_CARD, Theme.BORDER)

    def _apply_surface_state(self, bg: str, border: str, accent: str = None):
        self._surface_bg = bg
        self.config(bg=bg, highlightbackground=border)
        for widget in (self.container, self.name_label, self.info_label, self.message_label,
                       self.time_label):
            widget.config(bg=bg)
        for widget in (self.top_row, self.bottom_row, self.actions_row):
            widget.config(bg=bg)
        self.progress_bar.config(bg=bg)
        for button in (self.remove_btn, self.open_btn):
            button.config(bg=bg)
        # Accent stripe: painted when a value is passed, otherwise matches bg
        self.accent_stripe.config(bg=accent or bg)

    def set_selected(self, selected: bool):
        self.is_selected = selected
        if selected:
            self._apply_surface_state(
                Theme.BG_CARD_SELECTED, Theme.BLUE_PRIMARY, accent=Theme.BLUE_PRIMARY)
        else:
            self._apply_surface_state(Theme.BG_CARD, Theme.BORDER)

    def _open_output(self):
        """Open the output file if processing is complete."""
        if self.item.status == ProcessingStatus.COMPLETE and Path(self.item.output_path).exists():
            try:
                os.startfile(self.item.output_path)
            except Exception:
                pass

    def _get_status_color(self) -> str:
        return status_ui(self.item.status)["color"]

    def update_item(self, item: QueueItem):
        self.item = item
        badge = status_ui(item.status)
        self.status_badge.config(text=badge["label"], fg=badge["color"], bg=badge["bg"])
        self.progress_bar.set_progress(item.progress)
        self.progress_bar.set_color(self._get_status_color())
        status_message = truncate_middle(item.message or "Ready to process", 74)
        message_color = {
            ProcessingStatus.COMPLETE: Theme.SUCCESS,
            ProcessingStatus.ERROR: Theme.ERROR,
            ProcessingStatus.CANCELLED: Theme.WARNING,
            ProcessingStatus.LOADING: Theme.INFO,
            ProcessingStatus.DETECTING: Theme.INFO,
            ProcessingStatus.PROCESSING: Theme.INFO,
            ProcessingStatus.MERGING: Theme.WARNING,
        }.get(item.status, Theme.TEXT_SECONDARY)
        self.message_label.config(text=status_message, fg=message_color)
        can_open = item.status == ProcessingStatus.COMPLETE and Path(item.output_path).exists()
        self.open_btn.set_enabled(can_open)
        if can_open:
            if not self.open_btn.winfo_ismapped():
                self.open_btn.pack(side="right")
        elif self.open_btn.winfo_manager():
            self.open_btn.pack_forget()
        self.remove_btn.set_enabled(item.status not in (
            ProcessingStatus.LOADING,
            ProcessingStatus.DETECTING,
            ProcessingStatus.PROCESSING,
            ProcessingStatus.MERGING,
        ))

        # Elapsed time
        elapsed_text = ""
        if item.started_at:
            end = item.completed_at or datetime.now()
            elapsed = (end - item.started_at).total_seconds()
            elapsed_text = format_time(elapsed)
        pct_text = f"{int(item.progress * 100)}%" if item.progress > 0 else ""
        if pct_text and elapsed_text:
            meta_text = f"{pct_text} / {elapsed_text}"
        else:
            meta_text = pct_text or elapsed_text
        self.time_label.config(text=meta_text)

        # Active-state pulsing indicator (start/stop based on status)
        active = item.status in (ProcessingStatus.LOADING,
                                 ProcessingStatus.DETECTING,
                                 ProcessingStatus.PROCESSING,
                                 ProcessingStatus.MERGING)
        if active:
            self._start_pulse()
        else:
            self._stop_pulse()

    # Pulse-state helpers -----------------------------------------------
    _pulse_id = None
    _pulse_phase = 0

    def _start_pulse(self):
        if getattr(self, "_pulse_id", None) is not None:
            return
        self._pulse_phase = 0
        self._pulse_tick()

    def _stop_pulse(self):
        tid = getattr(self, "_pulse_id", None)
        if tid:
            try:
                self.after_cancel(tid)
            except tk.TclError:
                pass
        self._pulse_id = None
        # Restore the normal border for the current selection state
        border = Theme.BLUE_PRIMARY if self.is_selected else Theme.BORDER
        self.config(highlightbackground=border)
        if self.is_selected:
            self.accent_stripe.config(bg=Theme.BLUE_PRIMARY)
        else:
            self.accent_stripe.config(bg=self._surface_bg)

    def _pulse_tick(self):
        # Alternate between a bright and a calm border / accent stripe
        try:
            bright = (self._pulse_phase % 2 == 0)
            border = Theme.GREEN_PRIMARY if bright else Theme.GREEN_HOVER
            stripe = Theme.GREEN_PRIMARY if bright else Theme.GREEN_HOVER
            self.config(highlightbackground=border)
            self.accent_stripe.config(bg=stripe)
            self._pulse_phase += 1
            self._pulse_id = self.after(720, self._pulse_tick)
        except tk.TclError:
            self._pulse_id = None


# =============================================================================
# LOG PANEL HANDLER -- routes log messages into a tk.Text widget
# =============================================================================

class TextWidgetHandler(logging.Handler):
    """Logging handler that writes to a tk.Text widget and tracks
    WARN/ERROR counts so the UI can show live badges."""

    def __init__(self, text_widget: tk.Text, on_count_change: Callable = None):
        super().__init__()
        self.text_widget = text_widget
        self.on_count_change = on_count_change
        self.warn_count = 0
        self.error_count = 0

    def emit(self, record):
        msg = self.format(record) + '\n'
        # Skip cheaply if the widget has already been destroyed. tk.Text
        # raises TclError on both `winfo_exists` and `after` after destroy,
        # so we guard against both without re-entering a partially-torn-down
        # interpreter.
        try:
            if not int(self.text_widget.winfo_exists()):
                return
            self.text_widget.after(0, self._append, msg, record.levelno)
        except tk.TclError:
            # The widget went away between our check and the schedule; drop
            # silently because the root is shutting down.
            pass
        except Exception:
            pass

    def _append(self, msg, levelno):
        try:
            if not int(self.text_widget.winfo_exists()):
                return
        except tk.TclError:
            return
        self.text_widget.config(state="normal")
        tag = "info"
        if levelno >= logging.ERROR:
            tag = "error"
            self.error_count += 1
        elif levelno >= logging.WARNING:
            tag = "warning"
            self.warn_count += 1
        self.text_widget.insert("end", msg, tag)
        # Trim to 2000 lines to prevent unbounded memory growth
        line_count = int(self.text_widget.index("end-1c").split(".")[0])
        if line_count > 2000:
            self.text_widget.delete("1.0", f"{line_count - 2000}.0")
        self.text_widget.see("end")
        self.text_widget.config(state="disabled")
        if self.on_count_change:
            try:
                self.on_count_change(self.warn_count, self.error_count)
            except Exception:
                pass

    def reset_counts(self):
        self.warn_count = 0
        self.error_count = 0
        if self.on_count_change:
            try:
                self.on_count_change(0, 0)
            except Exception:
                pass


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class VideoSubtitleRemoverApp:
    """Main application class."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} v{APP_VERSION}")
        self.root.geometry("1240x860")
        self.root.minsize(980, 720)
        self.root.configure(bg=Theme.BG_DARK)

        # Set window icon
        try:
            icon_candidates = [
                get_app_dir() / "assets" / "icon.ico",
                get_app_dir() / "icon.ico",
                get_app_dir() / "favicon.ico",
            ]
            for icon_path in icon_candidates:
                if icon_path.exists():
                    self.root.iconbitmap(icon_path)
                    break
        except Exception:
            pass
        if PIL_AVAILABLE:
            try:
                for icon_path in (get_app_dir() / "icon.png", get_app_dir() / "banner.png"):
                    if icon_path.exists():
                        icon_img = Image.open(icon_path)
                        if icon_img.width > 128:
                            icon_img.thumbnail((128, 128), Image.LANCZOS)
                        self._app_icon_photo = ImageTk.PhotoImage(icon_img)
                        self.root.iconphoto(True, self._app_icon_photo)
                        break
            except Exception:
                pass

        # State
        self.config = load_settings()
        # RM-96: high-contrast theme applies BEFORE widget construction so
        # every Canvas / ttk style reads the swapped palette on first draw.
        if getattr(self.config, "high_contrast", False):
            apply_high_contrast_theme()
        # RM-98: RTL layout mirror -- set the Tk option DB before widgets
        # build so labels right-align and `pack(side="right")` becomes
        # the dominant orientation. Full pack-side flipping for every
        # widget is a much larger pass; this lands the framework.
        self._rtl_layout = bool(getattr(self.config, "rtl_layout", False))
        # RM-97: bind a gettext catalog if one matches the user's locale.
        # No-op when no `.mo` file ships -- every UI string falls back to
        # the literal english form.
        try:
            import locale as _locale
            _user_locale = (_locale.getlocale()[0] or "").split("_")[0]
            from backend.i18n import bind_locale as _bind_locale
            if _user_locale and _user_locale != "en":
                _bind_locale(_user_locale)
        except Exception:
            pass
        self.queue: List[QueueItem] = []
        self.queue_widgets: dict = {}
        self.is_processing = False
        self._stop_requested = False
        self._processing_thread: Optional[threading.Thread] = None
        self.cancel_event = threading.Event()
        self.queue_lock = threading.Lock()
        self.gpus = detect_gpu()
        self.ai_engines = detect_ai_engines()
        self.ffmpeg_ready = detect_ffmpeg()
        self._elapsed_timer_id = None
        self._output_dir: Optional[Path] = None  # None = use input_dir/output/
        self._preview_detector = None  # cached SubtitleDetector for mask preview
        self._preview_detector_lang = None  # lang the cached detector was created with
        self._cached_remover = None  # cached BackendRemover for batch reuse
        self._cached_remover_key = None  # (mode, device, lang) key for cache invalidation
        self._selected_queue_item_id: Optional[str] = None
        self._brand_photo = None
        self._status_tone = "neutral"
        self._shutdown_started = False
        self._taskbar = None  # created after the root is fully realized
        self._batch_times: List[float] = []  # seconds per item for ETA
        self._batch_started_at: Optional[datetime] = None
        self._preview_request_id = 0
        self._throbber_id = None
        self._throbber_phase = 0
        self._layout_mode = "wide"
        self._workflow_pills = []

        # Variables
        self.mode_var = tk.StringVar(value=self.config.mode.value)
        self.gpu_var = tk.StringVar()
        self.skip_detection_var = tk.BooleanVar(value=self.config.sttn_skip_detection)
        self.lama_fast_var = tk.BooleanVar(value=self.config.lama_super_fast)
        self.preserve_audio_var = tk.BooleanVar(value=self.config.preserve_audio)
        self.lang_var = tk.StringVar(value=self.config.detection_lang)

        # Build UI
        self._setup_styles()
        self._build_ui()
        self._bind_shortcuts()
        self.root.bind("<Configure>", self._on_root_configure, add="+")

        # GPU setup -- restore saved selection or default to first
        if self.gpus:
            matched = False
            for g in self.gpus:
                if g['index'] == self.config.gpu_id:
                    self.gpu_var.set(f"{g['name']} ({g['memory']})")
                    matched = True
                    break
            if not matched:
                self.gpu_var.set(f"{self.gpus[0]['name']} ({self.gpus[0]['memory']})")
        else:
            self.gpu_var.set("CPU Mode")
            self.config.use_gpu = False

        # Attach log panel handler (tracks warn/error counts for badges)
        self._log_handler = TextWidgetHandler(self.log_text,
                                              on_count_change=self._update_log_badges)
        self._log_handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
        logging.getLogger().addHandler(self._log_handler)

        self._update_output_label()
        self._update_region_label_display()
        self._refresh_action_states()
        self.root.after(0, lambda: self._apply_responsive_layout(self.root.winfo_width()))

        # Restore persisted panel visibility (defaults: advanced closed, log open)
        try:
            if self.config.adv_panel_open and not self.adv_visible:
                self._toggle_advanced()
            if not self.config.log_panel_open and self._log_visible:
                self._toggle_log_panel()
        except Exception:
            pass

        # Save settings on close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # First-run welcome overlay (only shown once, then persisted)
        self._maybe_show_onboarding()

    def _on_close(self):
        """Stop processing, save settings, and close."""
        if self._shutdown_started:
            return
        # Guard against re-entry while the confirmation dialog is open.
        # Without this, a second WM_DELETE_WINDOW could fire while the modal
        # is shown (e.g. from an external window manager).
        if getattr(self, '_close_dialog_open', False):
            return
        self._close_dialog_open = True
        try:
            active_thread = self._has_active_processing_thread()
            if self.is_processing or active_thread:
                n = sum(1 for it in self.queue
                        if it.status in (ProcessingStatus.LOADING,
                                         ProcessingStatus.DETECTING,
                                         ProcessingStatus.PROCESSING,
                                         ProcessingStatus.MERGING))
                label = f"{n} active item{'s' if n != 1 else ''} will be cancelled."
                if not show_confirm(
                    self.root,
                    title="Close while processing?",
                    message="A batch is still running.",
                    detail=label + " Completed outputs on disk are kept.",
                    confirm_label="Close anyway",
                    cancel_label="Keep working",
                    tone="danger",
                ):
                    return
                self.cancel_event.set()
                self._stop_elapsed_timer()
                self._stop_requested = True
                self._update_status(
                    "Closing after the current step stops safely...",
                    "warning",
                )
                if self._taskbar:
                    self._taskbar.set_state(TaskbarProgress.STATE_PAUSED)
        finally:
            self._close_dialog_open = False
        # Set the flag AFTER confirmation so that _on_processing_complete
        # callbacks scheduled before the dialog opened don't race-destroy root.
        self._shutdown_started = True
        self._sync_config_from_ui()
        # Persist window layout and panel states for next launch
        try:
            self.config.window_geometry = self.root.geometry()
            self.config.adv_panel_open = self.adv_visible
            self.config.log_panel_open = self._log_visible
        except Exception:
            pass
        save_settings(self.config)
        self._finish_close_when_safe(time.monotonic() + 2.0)

    def _finish_close_when_safe(self, deadline: float):
        """Wait briefly for active work to notice cancellation before exit."""
        if not self._has_active_processing_thread() or time.monotonic() >= deadline:
            try:
                self.root.destroy()
            except Exception:
                pass
            return
        try:
            self.root.after(100, lambda: self._finish_close_when_safe(deadline))
        except Exception:
            try:
                self.root.destroy()
            except Exception:
                pass

    def _sync_config_from_ui(self):
        """Sync config object from current UI state."""
        try:
            self.config.mode = InpaintMode(self.mode_var.get())
        except ValueError:
            pass
        self.config.sttn_skip_detection = self.skip_detection_var.get()
        self.config.lama_super_fast = self.lama_fast_var.get()
        self.config.preserve_audio = self.preserve_audio_var.get()
        self.config.detection_lang = self.lang_var.get()
        # Threshold slider stores as int percent, convert to float
        pct = getattr(self.config, '_detection_threshold_pct', 50)
        self.config.detection_threshold = pct / 100.0
        # Time range
        self.config.time_start = self._safe_float(self.time_start_entry.get())
        self.config.time_end = self._safe_float(self.time_end_entry.get())
        # HW encode
        self.config.use_hw_encode = self.hw_encode_var.get()
        # v3.9 quality + workflow toggles
        if hasattr(self, 'auto_band_var'):
            self.config.auto_band = self.auto_band_var.get()
        if hasattr(self, 'flow_warp_var'):
            self.config.tbe_flow_warp = self.flow_warp_var.get()
        if hasattr(self, 'scene_split_var'):
            self.config.tbe_scene_cut_split = self.scene_split_var.get()
        if hasattr(self, 'adaptive_batch_var'):
            self.config.adaptive_batch = self.adaptive_batch_var.get()
        if hasattr(self, 'export_srt_var'):
            self.config.export_srt = self.export_srt_var.get()
        if hasattr(self, 'export_mask_var'):
            self.config.export_mask_video = self.export_mask_var.get()
        if hasattr(self, 'kalman_var'):
            self.config.kalman_tracking = self.kalman_var.get()
        if hasattr(self, 'phash_var'):
            self.config.phash_skip_enable = self.phash_var.get()
        if hasattr(self, 'colour_tune_var'):
            self.config.colour_tune_enable = self.colour_tune_var.get()
        if hasattr(self, 'deinterlace_var'):
            self.config.deinterlace_auto = self.deinterlace_var.get()
        if hasattr(self, 'keyframe_var'):
            self.config.keyframe_detection = self.keyframe_var.get()
        if hasattr(self, 'quality_report_var'):
            self.config.quality_report = self.quality_report_var.get()
        # v3.13 GUI-exposed toggles
        if hasattr(self, 'quality_sheet_var'):
            self.config.quality_report_sheet = self.quality_sheet_var.get()
        if hasattr(self, 'multi_audio_var'):
            self.config.multi_audio_passthrough = self.multi_audio_var.get()
        if hasattr(self, 'loudnorm_var'):
            self.config.loudnorm_target = self._safe_float(self.loudnorm_var.get(), 0.0)
        if hasattr(self, 'decode_accel_var'):
            self.config.decode_hw_accel = self.decode_accel_var.get()
        if hasattr(self, 'prefetch_var'):
            self.config.prefetch_decode = self.prefetch_var.get()
        if hasattr(self, 'remove_subs_var'):
            self.config.remove_subtitles = self.remove_subs_var.get()
        if hasattr(self, 'remove_chyrons_var'):
            self.config.remove_chyrons = self.remove_chyrons_var.get()
        if hasattr(self, 'karaoke_grouping_var'):
            self.config.karaoke_grouping = self.karaoke_grouping_var.get()
        if hasattr(self, 'output_codec_var'):
            self.config.output_codec = self.output_codec_var.get()
        if hasattr(self, 'vertical_text_var'):
            self.config.detection_vertical = self.vertical_text_var.get()
        if hasattr(self, 'high_contrast_var'):
            self.config.high_contrast = self.high_contrast_var.get()
        # GPU sync
        selection = self.gpu_var.get()
        for gpu in self.gpus:
            if f"{gpu['name']} ({gpu['memory']})" == selection:
                self.config.gpu_id = gpu['index']
                break

    def _make_processing_snapshot(self) -> ProcessingConfig:
        """Build a fresh processing snapshot from the current UI state."""
        self._sync_config_from_ui()
        return ProcessingConfig.from_dict(self.config.to_dict())

    def _apply_current_settings_to_idle_items(self) -> int:
        """Refresh all not-yet-running queue items from the current UI state."""
        snapshot = self._make_processing_snapshot()
        updated = 0
        with self.queue_lock:
            for item in self.queue:
                if item.status == ProcessingStatus.IDLE:
                    item.config = ProcessingConfig.from_dict(snapshot.to_dict())
                    updated += 1
        output_updates = self._refresh_idle_output_paths()
        if output_updates:
            self._update_queue_display()
        return updated

    def _setup_styles(self):
        """Configure ttk styles for a cohesive dark theme."""
        style = ttk.Style()
        style.theme_use('clam')

        # ---- Combobox ---------------------------------------------------
        style.configure("Dark.TCombobox",
                       fieldbackground=Theme.BG_TERTIARY,
                       background=Theme.BG_TERTIARY,
                       foreground=Theme.TEXT_PRIMARY,
                       arrowcolor=Theme.TEXT_SECONDARY,
                       bordercolor=Theme.BORDER,
                       darkcolor=Theme.BG_TERTIARY,
                       lightcolor=Theme.BG_TERTIARY,
                       insertcolor=Theme.TEXT_PRIMARY,
                       padding=(10, 6))

        style.map("Dark.TCombobox",
                 fieldbackground=[('readonly', Theme.BG_TERTIARY),
                                  ('disabled', Theme.BG_CARD)],
                 background=[('active', Theme.BG_RAISED)],
                 foreground=[('disabled', Theme.TEXT_DISABLED)],
                 arrowcolor=[('active', Theme.TEXT_PRIMARY),
                             ('disabled', Theme.TEXT_DISABLED)],
                 bordercolor=[('focus', Theme.BORDER_FOCUS),
                              ('hover', Theme.BORDER_STRONG)],
                 selectbackground=[('readonly', Theme.BLUE_MUTED)],
                 selectforeground=[('readonly', Theme.TEXT_PRIMARY)])

        # Theme the combobox dropdown popup listbox
        self.root.option_add('*TCombobox*Listbox.background', Theme.BG_RAISED)
        self.root.option_add('*TCombobox*Listbox.foreground', Theme.TEXT_PRIMARY)
        self.root.option_add('*TCombobox*Listbox.selectBackground', Theme.BLUE_MUTED)
        self.root.option_add('*TCombobox*Listbox.selectForeground', Theme.TEXT_PRIMARY)
        self.root.option_add('*TCombobox*Listbox.borderWidth', 0)
        self.root.option_add('*TCombobox*Listbox.font', f(Theme.F_BODY_SM))

        # ---- Scrollbar (slimmer, quieter) -------------------------------
        style.configure("Dark.Vertical.TScrollbar",
                        background=Theme.BORDER,
                        troughcolor=Theme.BG_SECONDARY,
                        bordercolor=Theme.BG_SECONDARY,
                        arrowcolor=Theme.TEXT_MUTED,
                        gripcount=0,
                        width=10)
        style.map("Dark.Vertical.TScrollbar",
                 background=[('active', Theme.BORDER_STRONG),
                             ('pressed', Theme.BORDER_STRONG)],
                 arrowcolor=[('active', Theme.TEXT_SECONDARY)])

    def _create_surface(self, parent, bg: str = Theme.BG_SECONDARY) -> tk.Frame:
        """Create a bordered surface panel."""
        return tk.Frame(parent, bg=bg, highlightthickness=1,
                        highlightbackground=Theme.BORDER_SUBTLE)

    def _create_chip(self, parent, label: str, value: str, fg: str, bg: str) -> tk.Frame:
        """Minimal status chip with a single clear line of text."""
        chip = tk.Frame(parent, bg=bg, highlightthickness=1,
                        highlightbackground=Theme.BORDER_SUBTLE)
        tk.Label(
            chip,
            text=f"{label}: {value}",
            font=f(Theme.F_META, "bold"),
            bg=bg,
            fg=fg,
            padx=12,
            pady=7,
        ).pack(anchor="w")
        return chip

    def _section_title(self, parent, eyebrow: str, title: str, hint: str,
                       pad_x: int = 20, pad_top: int = 16):
        """Consistent section header: eyebrow label + title + hint line."""
        bg = parent.cget("bg")
        if eyebrow:
            tk.Label(parent, text=eyebrow.upper(), font=f(Theme.F_EYEBROW, "bold"),
                     bg=bg, fg=Theme.TEXT_MUTED).pack(
                         anchor="w", padx=pad_x, pady=(pad_top, 0))
        tk.Label(parent, text=title, font=f(Theme.F_HEADING, "bold"),
                 bg=bg, fg=Theme.TEXT_PRIMARY).pack(
                     anchor="w", padx=pad_x,
                     pady=(2 if eyebrow else pad_top, 0))
        if hint:
            tk.Label(parent, text=hint, font=f(Theme.F_BODY_SM),
                     bg=bg, fg=Theme.TEXT_MUTED, wraplength=560,
                     justify="left").pack(anchor="w", padx=pad_x, pady=(4, Theme.S_MD))

    def _create_card(self, parent, bg=Theme.BG_CARD) -> tk.Frame:
        """Bordered card container with consistent style."""
        return tk.Frame(parent, bg=bg, highlightthickness=1,
                        highlightbackground=Theme.BORDER_SUBTLE)

    def _card_header(self, parent, eyebrow: str, title: str, bg=Theme.BG_CARD,
                     pad_x: int = 16, pad_top: int = 14):
        """Card-internal section header with a single clear title."""
        tk.Label(parent, text=title, font=f(Theme.F_TITLE, "bold"),
                 bg=bg, fg=Theme.TEXT_PRIMARY).pack(anchor="w", padx=pad_x, pady=(pad_top, 10))

    def _divider(self, parent, pad: int = 0):
        tk.Frame(parent, bg=Theme.BORDER_SUBTLE, height=1).pack(
            fill="x", padx=pad, pady=0)

    def _update_output_label(self):
        """Refresh the output directory summary."""
        if self._output_dir:
            display = truncate_middle(str(self._output_dir), 54)
            self.output_dir_label.config(text=display, fg=Theme.TEXT_PRIMARY)
            self.output_dir_meta.config(text="Custom location")
        else:
            self.output_dir_label.config(text="Auto-create an output folder beside each source",
                                         fg=Theme.TEXT_PRIMARY)
            self.output_dir_meta.config(text="Default workflow")

    def _update_region_label_display(self):
        """Refresh the region summary line."""
        if self.config.subtitle_area:
            x1, y1, x2, y2 = self.config.subtitle_area
            self.region_label.config(
                text=f"Manual region: ({x1}, {y1}) to ({x2}, {y2})",
                fg=Theme.TEXT_PRIMARY,
            )
            self.region_meta.config(text="Fixed mask region", fg=Theme.SUCCESS)
        else:
            self.region_label.config(text="Automatic subtitle detection", fg=Theme.TEXT_PRIMARY)
            self.region_meta.config(text="Recommended default", fg=Theme.TEXT_MUTED)
        if hasattr(self, "region_reset_btn"):
            self.region_reset_btn.set_enabled(self.config.subtitle_area is not None and not self.is_processing)

    def _start_throbber(self):
        """Animate the preview area with a shimmer placeholder and moving dots
        to signal a background task in progress."""
        self._stop_throbber()
        self._throbber_phase = 0
        self._throbber_tick()

    def _stop_throbber(self):
        tid = getattr(self, "_throbber_id", None)
        if tid:
            try:
                self.root.after_cancel(tid)
            except Exception:
                pass
            self._throbber_id = None

    def _throbber_tick(self):
        if not PIL_AVAILABLE:
            self._preview_label.config(
                text="Detecting" + "." * (self._throbber_phase % 4))
            try:
                self._throbber_id = self.root.after(240, self._throbber_tick)
                self._throbber_phase += 1
            except tk.TclError:
                pass
            return
        try:
            w = max(220, self._preview_frame.winfo_width() - 36)
            h = 158
            base = Image.new("RGB", (w, h), self._hex_to_rgb(Theme.BG_TERTIARY))
            d = ImageDraw.Draw(base)
            d.rectangle([(0, 0), (w - 1, h - 1)],
                        outline=self._hex_to_rgb(Theme.BORDER), width=1)
            # Three animated dots pulsing left-to-right
            cx, cy = w // 2, h // 2
            phase = self._throbber_phase % 3
            for i in range(3):
                active = (i == phase)
                color = (Theme.BLUE_PRIMARY if active else Theme.BORDER)
                r = 6 if active else 4
                x = cx - 18 + i * 18
                d.ellipse([(x - r, cy - r), (x + r, cy + r)],
                          fill=self._hex_to_rgb(color))
            d.text((cx - 42, cy + 22), "DETECTING",
                   fill=self._hex_to_rgb(Theme.TEXT_MUTED))
            self._preview_photo = ImageTk.PhotoImage(base)
            self._preview_label.config(image=self._preview_photo, text="")
            self._throbber_phase += 1
            try:
                self._throbber_id = self.root.after(240, self._throbber_tick)
            except tk.TclError:
                pass
        except Exception:
            # Render failures shouldn't block detection
            pass

    def _push_live_preview(self, pil_img, cur_idx: int, total: int, file_name: str):
        """Render an inpainted frame into the preview pane during processing.
        Called on the Tk main thread via `root.after` from the worker thread."""
        try:
            self._stop_throbber()
            # Throttle: coalesce to at most ~15 FPS of UI updates
            now = time.monotonic()
            last = getattr(self, "_live_preview_last_ts", 0.0)
            if (now - last) < (1.0 / 15.0):
                return
            self._live_preview_last_ts = now
            if PIL_AVAILABLE:
                self._preview_photo = ImageTk.PhotoImage(pil_img)
                self._preview_label.config(image=self._preview_photo, text="")
            if total:
                pct = int(cur_idx / max(1, total) * 100)
                self.preview_title_label.config(text=f"Live preview: {file_name}")
                self.preview_meta_label.config(
                    text=f"Frame {cur_idx}/{total} ({pct}%)")
        except Exception:
            pass

    def _set_preview_placeholder(self, title: str, body: str):
        """Show the empty-state preview guidance with a subtle illustration."""
        self._stop_throbber()
        self.preview_title_label.config(text=title)
        self.preview_meta_label.config(text=body)
        # Render a minimalist placeholder card via PIL (if available) so the
        # preview never collapses to empty space.
        if PIL_AVAILABLE:
            try:
                w, h = 420, 128
                base = Image.new("RGB", (w, h), self._hex_to_rgb(Theme.BG_TERTIARY))
                draw = ImageDraw.Draw(base)
                # Outer border
                draw.rectangle([(0, 0), (w - 1, h - 1)],
                               outline=self._hex_to_rgb(Theme.BORDER_SUBTLE), width=1)
                # Faux film-strip glyph (three tall rects in the center)
                cx, cy = w // 2, h // 2
                for dx in (-44, 0, 44):
                    draw.rectangle(
                        [(cx + dx - 10, cy - 22), (cx + dx + 10, cy + 22)],
                        outline=self._hex_to_rgb(Theme.BORDER),
                        fill=self._hex_to_rgb(Theme.BG_CARD_HOVER),
                    )
                # Underline
                draw.line([(cx - 70, cy + 32), (cx + 70, cy + 32)],
                          fill=self._hex_to_rgb(Theme.BORDER_SUBTLE), width=1)
                self._preview_photo = ImageTk.PhotoImage(base)
                self._preview_label.config(image=self._preview_photo, text="")
            except Exception:
                self._preview_label.config(text="", image="")
                self._preview_photo = None
        else:
            self._preview_label.config(text="", image="")
            self._preview_photo = None

    @staticmethod
    def _hex_to_rgb(hex_str: str):
        hex_str = hex_str.lstrip('#')
        return tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))

    def _set_selected_queue_item(self, item_id: Optional[str]):
        """Update queue item selection state."""
        self._selected_queue_item_id = item_id
        for wid, widget in self.queue_widgets.items():
            widget.set_selected(wid == item_id)
        self._update_preview_actions()

    def _refresh_action_states(self):
        """Enable or disable primary queue actions based on current state."""
        has_queue = bool(self.queue)
        has_complete = any(item.status == ProcessingStatus.COMPLETE for item in self.queue)
        has_retry = any(item.status in (ProcessingStatus.ERROR, ProcessingStatus.CANCELLED)
                        for item in self.queue)
        active_thread = self._has_active_processing_thread()
        batch_busy = self.is_processing or active_thread

        if hasattr(self, "start_btn"):
            can_stop = active_thread and not self._stop_requested
            can_start = (not batch_busy) and has_queue
            self.start_btn.set_enabled(can_stop or can_start)
        if hasattr(self, "open_output_btn"):
            self.open_output_btn.set_enabled(has_complete)
        if hasattr(self, "retry_btn"):
            self.retry_btn.set_enabled((not batch_busy) and has_retry)
        if hasattr(self, "clear_btn"):
            self.clear_btn.set_enabled((not batch_busy) and has_queue)
        if hasattr(self, "batch_label") and not batch_busy:
            pending = sum(1 for item in self.queue if item.status == ProcessingStatus.IDLE)
            if pending:
                self.batch_label.config(
                    text=f"{pending} queued and ready to process",
                    fg=Theme.TEXT_SECONDARY,
                )
            elif has_complete:
                self.batch_label.config(
                    text="Outputs are ready for review",
                    fg=Theme.SUCCESS,
                )
            elif has_retry:
                self.batch_label.config(
                    text="Some items need attention",
                    fg=Theme.WARNING,
                )
            else:
                self.batch_label.config(text="Ready", fg=Theme.TEXT_MUTED)
        self._update_preview_actions()
        self._update_guidance_surface()

    def _bind_shortcuts(self):
        """Register global shortcuts for the most common actions."""
        self.root.bind("<Control-o>", lambda e: self._open_file_picker())
        self.root.bind("<Control-O>", lambda e: self._open_file_picker())
        self.root.bind("<Control-Return>", lambda e: self._start_processing())
        self.root.bind("<F5>", lambda e: self._start_processing())
        self.root.bind("<Control-l>", lambda e: self._toggle_log_panel())
        self.root.bind("<Control-L>", lambda e: self._toggle_log_panel())
        self.root.bind("<Control-f>", self._focus_queue_filter)
        self.root.bind("<Control-F>", self._focus_queue_filter)

    def _open_file_picker(self):
        if hasattr(self, "drop_area"):
            self.drop_area._open_file_dialog()

    def _focus_queue_filter(self, event=None):
        if len(self.queue) < 6 or not hasattr(self, "_queue_filter_entry"):
            return "break"
        try:
            self._queue_filter_frame.pack(
                fill="x", padx=Theme.S_XL, pady=(0, Theme.S_SM),
                before=self._queue_container)
            self._queue_filter_entry.focus_set()
            self._queue_filter_entry.selection_range(0, "end")
        except tk.TclError:
            pass
        return "break"

    def _on_root_configure(self, event):
        """Keep layout responsive as the window width changes."""
        if event.widget is not self.root:
            return
        self._apply_responsive_layout(event.width)

    def _apply_responsive_layout(self, width: int):
        """Stack columns and footer/help clusters on narrower windows."""
        if not hasattr(self, "_content"):
            return

        mode = "stacked" if width < 1180 else "wide"
        if mode == self._layout_mode:
            if hasattr(self, "preview_meta_label"):
                self.preview_meta_label.config(wraplength=520 if mode == "stacked" else 360)
            if hasattr(self, "header_guidance_body"):
                self.header_guidance_body.config(wraplength=520 if mode == "stacked" else 300)
            if hasattr(self, "status_hint"):
                self.status_hint.config(wraplength=520 if mode == "stacked" else 360)
            return

        self._layout_mode = mode
        stacked = (mode == "stacked")

        self._left_col.grid_forget()
        self._right_col.grid_forget()

        if stacked:
            self._content.columnconfigure(0, weight=1, minsize=0)
            self._content.columnconfigure(1, weight=0, minsize=0)
            self._content.rowconfigure(0, weight=0)
            self._content.rowconfigure(1, weight=1)
            self._left_col.grid(row=0, column=0, sticky="nsew", padx=0, pady=(0, Theme.S_MD))
            self._right_col.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)

            self._header_right.pack_forget()
            self._header_right.pack(fill="x", pady=(Theme.S_LG, 0))
            self._header_chips.pack_forget()
            self._header_chips.pack(anchor="w")
            self._header_help_btn.pack_forget()
            self._header_help_btn.pack(anchor="w", pady=(Theme.S_SM, 0))
            self._header_guidance_panel.pack_forget()
            self._header_guidance_panel.pack(fill="x", pady=(Theme.S_SM, 0))

            self._footer_left.pack_forget()
            self._footer_left.pack(anchor="w")
            self.status_hint.pack_forget()
            self.status_hint.pack(fill="x", pady=(Theme.S_XS, 0))
        else:
            self._content.columnconfigure(0, weight=57, minsize=440)
            self._content.columnconfigure(1, weight=43, minsize=360)
            self._content.rowconfigure(0, weight=1)
            self._content.rowconfigure(1, weight=0)
            self._left_col.grid(row=0, column=0, sticky="nsew", padx=(0, Theme.S_MD))
            self._right_col.grid(row=0, column=1, sticky="nsew", padx=(Theme.S_MD, 0))

            self._header_right.pack_forget()
            self._header_right.pack(side="right", anchor="n")
            self._header_chips.pack_forget()
            self._header_chips.pack(anchor="e")
            self._header_help_btn.pack_forget()
            self._header_help_btn.pack(anchor="e", pady=(Theme.S_SM, 0))
            self._header_guidance_panel.pack_forget()
            self._header_guidance_panel.pack(anchor="e", fill="x", pady=(Theme.S_SM, 0))

            self._footer_left.pack_forget()
            self._footer_left.pack(side="left")
            self.status_hint.pack_forget()
            self.status_hint.pack(side="right")

        self.preview_meta_label.config(wraplength=520 if stacked else 360)
        self.header_guidance_body.config(wraplength=520 if stacked else 300)
        self.status_hint.config(wraplength=520 if stacked else 360)

    def _get_selected_queue_item(self) -> Optional[QueueItem]:
        """Return the currently selected queue item, if any."""
        if not self._selected_queue_item_id:
            return None
        return next((item for item in self.queue if item.id == self._selected_queue_item_id), None)

    def _set_workflow_stage(self, stage: int):
        """Update the compact workflow pills in the header."""
        for idx, pill in enumerate(self._workflow_pills, start=1):
            if idx < stage:
                frame_bg = Theme.SUCCESS_BG
                frame_border = Theme.GREEN_HOVER
                badge_bg = Theme.GREEN_PRIMARY
                badge_fg = "#04120b"
                text_fg = Theme.SUCCESS
            elif idx == stage:
                frame_bg = Theme.BLUE_MUTED
                frame_border = Theme.BLUE_PRIMARY
                badge_bg = Theme.BLUE_PRIMARY
                badge_fg = "#071226"
                text_fg = Theme.TEXT_PRIMARY
            else:
                frame_bg = Theme.BG_CARD
                frame_border = Theme.BORDER
                badge_bg = Theme.BG_TERTIARY
                badge_fg = Theme.TEXT_MUTED
                text_fg = Theme.TEXT_SECONDARY
            pill["frame"].config(bg=frame_bg, highlightbackground=frame_border)
            pill["badge"].config(bg=badge_bg, fg=badge_fg)
            pill["text"].config(bg=frame_bg, fg=text_fg)

    def _update_guidance_surface(self):
        """Keep the header guidance card and footer hint aligned with state."""
        if not hasattr(self, "header_guidance_title"):
            return

        selected = self._get_selected_queue_item()
        has_queue = bool(self.queue)
        has_complete = any(item.status == ProcessingStatus.COMPLETE for item in self.queue)
        has_retry = any(item.status in (ProcessingStatus.ERROR, ProcessingStatus.CANCELLED)
                        for item in self.queue)

        if self._stop_requested:
            stage = 3
            title = "Stopping batch"
            body = ("The current item is wrapping up so the app can stop cleanly without risking overlapping work. "
                    "Finished outputs stay on disk and remaining items will be marked as stopped.")
            hint = "Stopping safely. Please wait for the current item to finish its active step."
        elif self.is_processing:
            stage = 3
            title = "Batch running"
            body = ("Live preview, ETA, and the activity log stay up to date while the batch works. "
                    "Stop is safe: completed outputs stay on disk.")
            hint = "Use Stop batch if you need to pause. Finished outputs are preserved."
        elif not has_queue:
            stage = 1
            title = "Build your batch"
            body = "Import files or choose a folder to start."
            hint = "Import files or choose a folder to start."
        elif has_retry:
            stage = 3 if has_complete else 2
            title = "Review the outliers"
            body = "Retry failed items or open the log for details."
            hint = "Retry failed items or open the log for details."
        elif not selected:
            stage = 2
            title = "Inspect a sample frame"
            body = "Select one item and review the mask before starting."
            hint = "Select one item and review the mask before starting."
        elif has_complete:
            stage = 3
            title = "Outputs are ready"
            body = "Preview a finished item or open the output folder."
            hint = "Preview a finished item or open the output folder."
        else:
            stage = 3
            title = "Ready to run"
            body = "Start the batch when the preview framing looks right."
            hint = "Start the batch when the preview framing looks right."

        self._set_workflow_stage(stage)
        self.header_guidance_title.config(text=title)
        self.header_guidance_body.config(text=body)
        if hasattr(self, "status_hint"):
            self.status_hint.config(text=hint)

    def _update_preview_actions(self):
        """Enable preview tools only when they make sense for the selection."""
        if not hasattr(self, "preview_mask_btn"):
            return
        selected = self._get_selected_queue_item()
        can_preview = bool(selected and PIL_AVAILABLE)
        self.preview_mask_btn.set_enabled(bool(selected) and not self.is_processing)
        self.preview_zoom_btn.set_enabled(can_preview)
        if hasattr(self, "preview_inpaint_btn"):
            self.preview_inpaint_btn.set_enabled(
                bool(selected) and not self.is_processing
            )
        if hasattr(self, "preview_ab_btn"):
            ab_ready = bool(
                selected
                and selected.status == ProcessingStatus.COMPLETE
                and Path(selected.output_path).exists()
            )
            self.preview_ab_btn.set_enabled(ab_ready)
        self._preview_label.config(cursor="hand2" if can_preview else "")

        if selected:
            badge = status_ui(selected.status)
            self.preview_status_chip.config(
                text=badge["label"],
                fg=badge["color"],
                bg=badge["bg"],
            )
        else:
            self.preview_status_chip.config(
                text="Waiting",
                fg=Theme.TEXT_MUTED,
                bg=Theme.BG_TERTIARY,
            )

    def _open_selected_mask_preview(self):
        item = self._get_selected_queue_item()
        if item:
            self._show_preview(item, show_mask=True)

    def _open_ab_scrubber(self):
        """RM-30: open a Toplevel that lets the user scrub frames AND
        wipe a vertical seam left/right to compare the original vs the
        cleaned output side-by-side at any frame.

        The window opens both video captures, holds them open for the
        duration of the modal, and composes a single image per scrub.
        """
        item = self._get_selected_queue_item()
        if item is None or item.status != ProcessingStatus.COMPLETE:
            self._update_status("Select a completed item first", "warning")
            return
        in_path = item.file_path
        out_path = item.output_path
        if not Path(out_path).exists():
            self._update_status("Output file is missing", "warning")
            return
        if not PIL_AVAILABLE:
            self._update_status("Pillow required for A/B compare", "warning")
            return

        import cv2 as _cv2
        cap_a = _cv2.VideoCapture(in_path)
        cap_b = _cv2.VideoCapture(out_path)
        if not cap_a.isOpened() or not cap_b.isOpened():
            cap_a.release(); cap_b.release()
            self._update_status("Could not open input/output for compare", "warning")
            return

        n_a = int(cap_a.get(_cv2.CAP_PROP_FRAME_COUNT)) or 1
        n_b = int(cap_b.get(_cv2.CAP_PROP_FRAME_COUNT)) or 1
        n_total = max(1, min(n_a, n_b))
        fps = cap_a.get(_cv2.CAP_PROP_FPS) or 30.0
        if fps <= 0:
            fps = 30.0
        max_w = min(1024, int(self.root.winfo_screenwidth() * 0.7))
        max_h = min(576, int(self.root.winfo_screenheight() * 0.6))

        win = tk.Toplevel(self.root)
        win.title(f"A/B compare: {Path(in_path).name}")
        win.configure(bg=Theme.BG_OVERLAY)
        win.resizable(False, False)

        canvas = tk.Canvas(win, width=max_w, height=max_h,
                            highlightthickness=0, bg=Theme.BG_DARK)
        canvas.pack()
        image_id = canvas.create_image(0, 0, anchor="nw")
        canvas._photo = None

        state = {"frame_idx": 0, "seam": max_w // 2}

        def _render():
            cap_a.set(_cv2.CAP_PROP_POS_FRAMES, state["frame_idx"])
            ok_a, fa = cap_a.read()
            cap_b.set(_cv2.CAP_PROP_POS_FRAMES, min(n_b - 1, state["frame_idx"]))
            ok_b, fb = cap_b.read()
            if not (ok_a and ok_b):
                return
            if fa.shape != fb.shape:
                fb = _cv2.resize(fb, (fa.shape[1], fa.shape[0]),
                                  interpolation=_cv2.INTER_AREA)
            h, w = fa.shape[:2]
            scale = min(max_w / w, max_h / h, 1.0)
            dw, dh = int(w * scale), int(h * scale)
            seam = max(0, min(dw, state["seam"]))
            fa_r = _cv2.resize(fa, (dw, dh), interpolation=_cv2.INTER_AREA)
            fb_r = _cv2.resize(fb, (dw, dh), interpolation=_cv2.INTER_AREA)
            composite = fa_r.copy()
            composite[:, seam:] = fb_r[:, seam:]
            # Draw a 2-pixel green seam line for the wipe boundary.
            if 0 < seam < dw:
                _cv2.line(composite, (seam, 0), (seam, dh - 1), (0, 255, 0), 2)
            rgb = _cv2.cvtColor(composite, _cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            canvas._photo = ImageTk.PhotoImage(pil)
            canvas.itemconfig(image_id, image=canvas._photo)

        # Frame slider (vertical -- below the image).
        slider_row = tk.Frame(win, bg=Theme.BG_OVERLAY)
        slider_row.pack(fill="x", padx=Theme.S_MD, pady=(Theme.S_SM, 0))
        tk.Label(slider_row, text="Frame", font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_OVERLAY, fg=Theme.TEXT_SECONDARY).pack(side="left")
        ts_label = tk.Label(slider_row, text="00:00:00",
                            font=f(Theme.F_META),
                            bg=Theme.BG_OVERLAY, fg=Theme.TEXT_MUTED)
        ts_label.pack(side="right")

        def _on_frame(value):
            try:
                state["frame_idx"] = max(0, min(n_total - 1, int(float(value))))
            except (TypeError, ValueError):
                return
            secs = state["frame_idx"] / fps
            hh = int(secs // 3600); mm = int((secs % 3600) // 60); ss = int(secs % 60)
            ts_label.config(text=f"{hh:02d}:{mm:02d}:{ss:02d}")
            _render()

        frame_slider = tk.Scale(
            win, from_=0, to=n_total - 1, orient="horizontal",
            command=_on_frame, showvalue=False, length=max_w - 24,
            bg=Theme.BG_OVERLAY, fg=Theme.TEXT_PRIMARY,
            troughcolor=Theme.BG_TERTIARY,
            activebackground=Theme.BLUE_PRIMARY, highlightthickness=0,
        )
        frame_slider.pack(fill="x", padx=Theme.S_MD, pady=(0, Theme.S_SM))

        # Seam slider (wipe boundary).
        seam_row = tk.Frame(win, bg=Theme.BG_OVERLAY)
        seam_row.pack(fill="x", padx=Theme.S_MD, pady=(0, 0))
        tk.Label(seam_row, text="Wipe", font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_OVERLAY, fg=Theme.TEXT_SECONDARY).pack(side="left")
        tk.Label(seam_row, text="source <-> cleaned",
                 font=f(Theme.F_META),
                 bg=Theme.BG_OVERLAY, fg=Theme.TEXT_MUTED).pack(side="right")

        def _on_seam(value):
            try:
                state["seam"] = max(0, min(max_w, int(float(value))))
            except (TypeError, ValueError):
                return
            _render()

        seam_slider = tk.Scale(
            win, from_=0, to=max_w, orient="horizontal",
            command=_on_seam, showvalue=False, length=max_w - 24,
            bg=Theme.BG_OVERLAY, fg=Theme.TEXT_PRIMARY,
            troughcolor=Theme.BG_TERTIARY,
            activebackground=Theme.GREEN_PRIMARY, highlightthickness=0,
        )
        seam_slider.set(state["seam"])
        seam_slider.pack(fill="x", padx=Theme.S_MD, pady=(0, Theme.S_MD))

        def _close():
            try:
                cap_a.release(); cap_b.release()
            except Exception:
                pass
            win.destroy()

        win.bind("<Escape>", lambda e: _close())
        win.protocol("WM_DELETE_WINDOW", _close)
        win.transient(self.root)
        win.grab_set()
        _render()

    def _open_selected_inpaint_preview(self):
        """F-3: run a single-frame detect + inpaint pass on the selected
        item and render the result in the preview pane. Lets users A/B
        settings without committing a full batch run."""
        item = self._get_selected_queue_item()
        if item is None:
            self._update_status("Select a queue item first", "warning")
            return
        if self.is_processing:
            self._update_status("Pause the batch before previewing", "warning")
            return
        if not PIL_AVAILABLE:
            self._update_status("Pillow required for inpaint preview", "warning")
            return

        self.preview_title_label.config(text=f"Inpainting {Path(item.file_path).name}")
        self.preview_meta_label.config(text="Running detect + inpaint on the first frame...")
        self._preview_label.config(image="", text="")
        self._preview_photo = None
        self._start_throbber()
        self._preview_label.update_idletasks()
        self._preview_request_id += 1
        request_id = self._preview_request_id
        snapshot_cfg = ProcessingConfig.from_dict(item.config.to_dict())
        source_path = item.file_path

        def _worker():
            import cv2 as _cv2
            try:
                from backend.processor import (
                    SubtitleRemover as _Remover,
                    ProcessingConfig as _BackendCfg,
                    InpaintMode as _BackendMode,
                )
                if is_image_file(source_path):
                    frame = _cv2.imread(source_path)
                elif is_video_file(source_path):
                    cap = _cv2.VideoCapture(source_path)
                    try:
                        ret, frame = cap.read()
                    finally:
                        cap.release()
                    if not ret:
                        frame = None
                else:
                    frame = None
                if frame is None:
                    self.root.after(0, lambda: (
                        self._stop_throbber(),
                        self.preview_title_label.config(text="Preview unavailable"),
                        self.preview_meta_label.config(text="The selected file could not be read."),
                    ))
                    return

                # Build a backend config snapshot from the item.
                mode_map = {
                    "Auto": _BackendMode.AUTO,
                    "STTN": _BackendMode.STTN,
                    "LAMA": _BackendMode.LAMA,
                    "ProPainter": _BackendMode.PROPAINTER,
                }
                backend_cfg = _BackendCfg(
                    mode=mode_map.get(snapshot_cfg.mode.value, _BackendMode.STTN),
                    device="cpu" if not snapshot_cfg.use_gpu else f"cuda:{snapshot_cfg.gpu_id}",
                    detection_lang=snapshot_cfg.detection_lang,
                    detection_threshold=snapshot_cfg.detection_threshold,
                    subtitle_area=snapshot_cfg.subtitle_area,
                    subtitle_areas=snapshot_cfg.subtitle_areas,
                    mask_dilate_px=snapshot_cfg.mask_dilate_px,
                    mask_feather_px=snapshot_cfg.mask_feather_px,
                    tbe_enable=snapshot_cfg.tbe_enable,
                )
                remover = _Remover(backend_cfg)
                # Single-frame inpaint -- detect, build mask, inpaint.
                fixed = (snapshot_cfg.subtitle_areas
                          or ([snapshot_cfg.subtitle_area] if snapshot_cfg.subtitle_area else None))
                if fixed:
                    boxes = list(fixed)
                else:
                    boxes = remover.detector.detect(
                        frame, snapshot_cfg.detection_threshold)
                if not boxes:
                    # No detection -- show the source with a hint.
                    pil = Image.fromarray(_cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB))
                    self.root.after(0, lambda: self._apply_inpaint_preview(
                        pil, "No text detected on the first frame", request_id, item.id))
                    return
                mask = remover._create_mask(frame.shape, boxes)
                [filled] = remover.inpainter.inpaint([frame], [mask])
                pil = Image.fromarray(_cv2.cvtColor(filled, _cv2.COLOR_BGR2RGB))
                meta = (f"Cleanup preview using {snapshot_cfg.mode.value}; "
                        f"{len(boxes)} region{'s' if len(boxes) != 1 else ''} masked.")
                self.root.after(0, lambda: self._apply_inpaint_preview(
                    pil, meta, request_id, item.id))
            except Exception as exc:
                logger.warning(f"Inpaint preview failed: {exc}")
                self.root.after(0, lambda: (
                    self._stop_throbber(),
                    self.preview_title_label.config(text="Inpaint preview failed"),
                    self.preview_meta_label.config(text=str(exc)),
                ))

        threading.Thread(target=_worker, daemon=True).start()

    def _apply_inpaint_preview(self, pil_img, meta_text, request_id, item_id):
        if (request_id != self._preview_request_id
                or self._selected_queue_item_id != item_id):
            return
        try:
            self._stop_throbber()
            max_w = max(220, self._preview_frame.winfo_width() - 36)
            max_h = 158
            pil_img.thumbnail((max_w, max_h), Image.LANCZOS)
            self._preview_photo = ImageTk.PhotoImage(pil_img)
            self._preview_label.config(image=self._preview_photo, text="")
            self.preview_title_label.config(text="Inpaint preview")
            self.preview_meta_label.config(text=meta_text)
        except Exception:
            pass

    def _build_ui(self):
        """Build the main user interface with balanced spacing rhythm."""
        main_container = tk.Frame(self.root, bg=Theme.BG_DARK)
        main_container.pack(fill="both", expand=True,
                            padx=Theme.S_XL, pady=(Theme.S_LG, Theme.S_MD))

        # Header
        self._build_header(main_container)

        # Content area (two columns via grid)
        content = tk.Frame(main_container, bg=Theme.BG_DARK)
        content.pack(fill="both", expand=True, pady=(Theme.S_MD, 0))
        content.columnconfigure(0, weight=57, minsize=440)
        content.columnconfigure(1, weight=43, minsize=360)
        content.rowconfigure(0, weight=1)

        # Left column - Input & Settings
        left_col = tk.Frame(content, bg=Theme.BG_DARK)
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, Theme.S_MD))
        self._content = content
        self._left_col = left_col

        self._build_input_section(left_col)
        self._build_settings_section(left_col)

        # Right column - Queue & Preview
        right_col = tk.Frame(content, bg=Theme.BG_DARK)
        right_col.grid(row=0, column=1, sticky="nsew", padx=(Theme.S_MD, 0))
        self._right_col = right_col

        self._build_queue_section(right_col)

        # Log panel
        self._build_log_panel(main_container)

        # Footer
        self._build_footer(main_container)

    def _build_header(self, parent):
        """Minimal app header with short guidance and a few live status signals."""
        header = self._create_surface(parent)
        header.pack(fill="x")

        inner = tk.Frame(header, bg=Theme.BG_SECONDARY)
        inner.pack(fill="x", padx=Theme.S_XL, pady=Theme.S_LG)

        left = tk.Frame(inner, bg=Theme.BG_SECONDARY)
        left.pack(side="left", fill="both", expand=True)
        self._header_left = left

        tk.Label(left, text="Video Subtitle Remover",
                 font=f(Theme.F_DISPLAY, "bold"), bg=Theme.BG_SECONDARY,
                 fg=Theme.TEXT_PRIMARY).pack(anchor="w")
        tk.Label(left, text=f"Version {APP_VERSION}",
                 font=f(Theme.F_META, "bold"), bg=Theme.BG_SECONDARY,
                 fg=Theme.TEXT_MUTED).pack(anchor="w", pady=(2, 0))
        tk.Label(
            left,
            text="Add files, review one sample, then run the batch.",
            font=f(Theme.F_BODY),
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_SECONDARY,
        ).pack(anchor="w", pady=(8, 0))

        right = tk.Frame(inner, bg=Theme.BG_SECONDARY)
        right.pack(side="right", anchor="n")
        self._header_right = right

        gpu_short = truncate_middle(self.gpus[0]["name"], 26) if self.gpus else "CPU mode"
        gpu_fg = Theme.SUCCESS if self.gpus else Theme.WARNING
        det_short = self.ai_engines["detection"][0] if self.ai_engines["detection"] else "OpenCV fallback"
        audio_short = "FFmpeg ready" if self.ffmpeg_ready else "No FFmpeg"
        audio_fg = Theme.SUCCESS if self.ffmpeg_ready else Theme.WARNING

        chips = tk.Frame(right, bg=Theme.BG_SECONDARY)
        chips.pack(anchor="e")
        self._header_chips = chips

        self._create_chip(chips, "Device", gpu_short, gpu_fg, Theme.BG_CARD).pack(side="left")
        self._create_chip(chips, "Detection", det_short, Theme.INFO, Theme.BG_CARD).pack(
            side="left", padx=(Theme.S_SM, 0))
        self._create_chip(chips, "Audio", audio_short, audio_fg, Theme.BG_CARD).pack(
            side="left", padx=(Theme.S_SM, 0))

        # About / help
        help_btn = ModernButton(right, text="Help", width=80,
                                command=self._show_about, style="ghost",
                                size="sm", icon="?")
        help_btn.pack(anchor="e", pady=(Theme.S_SM, 0))
        self._header_help_btn = help_btn

        self._header_guidance_panel = tk.Frame(right, bg=Theme.BG_SECONDARY)
        self._header_guidance_panel.pack(anchor="e", fill="x", pady=(Theme.S_SM, 0))

        # Workflow step pills (Import → Inspect → Run)
        pills_row = tk.Frame(self._header_guidance_panel, bg=Theme.BG_SECONDARY)
        pills_row.pack(anchor="w", pady=(0, Theme.S_SM))
        for idx, step_label in enumerate(("Import", "Inspect", "Run"), start=1):
            pill_frame = tk.Frame(pills_row, bg=Theme.BG_CARD,
                                  highlightthickness=1, highlightbackground=Theme.BORDER)
            badge_lbl = tk.Label(pill_frame, text=str(idx),
                                 font=f(Theme.F_META, "bold"),
                                 bg=Theme.BG_TERTIARY, fg=Theme.TEXT_MUTED,
                                 padx=5, pady=2)
            badge_lbl.pack(side="left", padx=(5, 0), pady=4)
            text_lbl = tk.Label(pill_frame, text=step_label,
                                font=f(Theme.F_BODY_SM),
                                bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY)
            text_lbl.pack(side="left", padx=(Theme.S_XS, 8), pady=4)
            pill_frame.pack(side="left",
                            padx=(0 if idx == 1 else Theme.S_XS, 0))
            self._workflow_pills.append({
                "frame": pill_frame, "badge": badge_lbl, "text": text_lbl,
            })

        self.header_guidance_title = tk.Label(
            self._header_guidance_panel,
            text="Build your batch",
            font=f(Theme.F_TITLE, "bold"),
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_PRIMARY,
        )
        self.header_guidance_title.pack(anchor="w")
        self.header_guidance_body = tk.Label(
            self._header_guidance_panel,
            text="Import files or choose a folder to start.",
            font=f(Theme.F_BODY_SM),
            wraplength=300,
            justify="left",
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_MUTED,
        )
        self.header_guidance_body.pack(anchor="w", fill="x", pady=(4, 0))

    def _build_input_section(self, parent):
        """Workspace section: drop zone + output location."""
        section = self._create_surface(parent)
        section.pack(fill="x")

        self._section_title(
            section,
            eyebrow="Workspace",
            title="Import media",
            hint="Add videos or images. Originals are never modified.",
        )

        self.drop_area = DragDropFrame(section, self._on_files_dropped, height=142)
        self.drop_area.pack(fill="x", padx=Theme.S_XL, pady=(0, Theme.S_MD))

        out_surface = self._create_card(section)
        out_surface.pack(fill="x", padx=Theme.S_XL, pady=(0, Theme.S_LG))

        out_row = tk.Frame(out_surface, bg=Theme.BG_CARD)
        out_row.pack(fill="x", padx=Theme.S_LG, pady=Theme.S_MD)

        label_col = tk.Frame(out_row, bg=Theme.BG_CARD)
        label_col.pack(side="left", fill="x", expand=True)

        tk.Label(label_col, text="OUTPUT LOCATION", font=f(Theme.F_EYEBROW, "bold"),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED).pack(anchor="w")

        self.output_dir_label = tk.Label(label_col, text="", font=f(Theme.F_BODY, "bold"),
                                         bg=Theme.BG_CARD, fg=Theme.TEXT_PRIMARY, anchor="w")
        self.output_dir_label.pack(anchor="w", pady=(4, 0))

        self.output_dir_meta = tk.Label(label_col, text="", font=f(Theme.F_META),
                                        bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED, anchor="w")
        self.output_dir_meta.pack(anchor="w", pady=(2, 0))

        actions = tk.Frame(out_row, bg=Theme.BG_CARD)
        actions.pack(side="right", padx=(Theme.S_MD, 0))

        choose_btn = ModernButton(actions, text="Choose folder", width=120,
                                  command=self._choose_output_dir, style="accent",
                                  size="sm")
        choose_btn.pack(side="left")

        reset_btn = ModernButton(actions, text="Reset", width=76,
                                 command=self._reset_output_dir, style="ghost",
                                 size="sm")
        reset_btn.pack(side="left", padx=(Theme.S_SM, 0))

        self._update_output_label()

    def _build_settings_section(self, parent):
        """Settings section: profile + workflow + collapsible advanced controls."""
        section = self._create_surface(parent)
        section.pack(fill="both", expand=True, pady=(Theme.S_MD, 0))

        self._section_title(
            section,
            eyebrow="Processing",
            title="Settings",
            hint="Pick a profile, confirm the region, then start the batch.",
        )

        settings = tk.Frame(section, bg=Theme.BG_SECONDARY)
        settings.pack(fill="both", expand=True, padx=Theme.S_XL, pady=(0, Theme.S_LG))

        # ---- Profile card -----------------------------------------------
        profile_panel = self._create_card(settings)
        profile_panel.pack(fill="x")

        self._card_header(profile_panel, "Profile", "Processing profile")

        # Preset picker -- one-click recipe application. Built-ins + user-saved.
        preset_row = tk.Frame(profile_panel, bg=Theme.BG_CARD)
        preset_row.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_XS, Theme.S_SM))

        tk.Label(preset_row, text="Preset", font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(side="left")

        self.preset_var = tk.StringVar(value="(custom)")
        preset_names = ["(custom)"] + [n for n, _ in list_presets()]
        self.preset_combo = ttk.Combobox(
            preset_row, textvariable=self.preset_var, values=preset_names,
            state="readonly", style="Dark.TCombobox", width=32,
            font=f(Theme.F_BODY_SM),
        )
        self.preset_combo.pack(side="left", padx=(Theme.S_SM, Theme.S_SM))
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset_applied)

        save_preset_btn = ModernButton(
            preset_row, text="Save as...", command=self._save_preset_dialog,
            size="sm", style="ghost",
        )
        save_preset_btn.pack(side="left")

        export_preset_btn = ModernButton(
            preset_row, text="Export", command=self._export_preset_dialog,
            size="sm", style="ghost",
        )
        export_preset_btn.pack(side="left", padx=(Theme.S_XS, 0))
        Tooltip(export_preset_btn, "Write the current preset to a shareable JSON file.")

        import_preset_btn = ModernButton(
            preset_row, text="Import", command=self._import_preset_dialog,
            size="sm", style="ghost",
        )
        import_preset_btn.pack(side="left", padx=(Theme.S_XS, 0))
        Tooltip(import_preset_btn, "Load a preset JSON file into the user library.")

        # Algorithm -- segmented picker replaces the Combobox for speed + clarity
        tk.Label(profile_panel, text="Algorithm", font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(
                     anchor="w", padx=Theme.S_LG)

        self.mode_picker = SegmentedPicker(
            profile_panel,
            options=[(m.value, m.value) for m in InpaintMode],
            value=self.mode_var.get(),
            command=self._on_mode_picker_changed,
            bg=Theme.BG_CARD,
        )
        self.mode_picker.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_XS, 0))

        self.algo_desc = tk.Label(profile_panel, text=self._get_algo_description(),
                                  font=f(Theme.F_BODY_SM), bg=Theme.BG_CARD,
                                  fg=Theme.TEXT_SECONDARY, justify="left", anchor="w",
                                  wraplength=520)
        self.algo_desc.pack(fill="x", padx=Theme.S_LG, pady=(2, Theme.S_MD))

        if self.gpus:
            row2 = tk.Frame(profile_panel, bg=Theme.BG_CARD)
            row2.pack(fill="x", padx=Theme.S_LG, pady=(0, Theme.S_SM))

            tk.Label(row2, text="Compute device", font=f(Theme.F_BODY_SM),
                     bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(side="left")

            gpu_options = [f"{g['name']} ({g['memory']})" for g in self.gpus]
            self.gpu_combo = ttk.Combobox(row2, textvariable=self.gpu_var, width=36,
                                          values=gpu_options, style="Dark.TCombobox",
                                          state="readonly", font=f(Theme.F_BODY_SM))
            self.gpu_combo.pack(side="right")
            self.gpu_combo.bind("<<ComboboxSelected>>", self._on_gpu_changed)

        lang_row = tk.Frame(profile_panel, bg=Theme.BG_CARD)
        lang_row.pack(fill="x", padx=Theme.S_LG, pady=(0, Theme.S_LG))

        tk.Label(lang_row, text="Subtitle language", font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(side="left")

        # F-5: language list = the union of curated friendly names and
        # any extra codes the active OCR engine declares it supports.
        # PaddleOCR / RapidOCR ship 100+ languages; we expose the union
        # so users can pick e.g. Thai or Polish without modifying code.
        self._lang_display = _build_language_list()
        self._lang_labels = [f"{name} ({code})" for code, name in self._lang_display]
        self._lang_by_label = {label: code for label, (code, _) in
                               zip(self._lang_labels, self._lang_display)}
        self._lang_display_var = tk.StringVar()
        self._set_lang_display(self.lang_var.get())

        self.lang_combo = ttk.Combobox(lang_row, textvariable=self._lang_display_var,
                                       width=20, values=self._lang_labels,
                                       style="Dark.TCombobox",
                                       state="readonly", font=f(Theme.F_BODY_SM))
        self.lang_combo.pack(side="right")
        self.lang_combo.bind("<<ComboboxSelected>>", self._on_lang_changed)

        # ---- Workflow card ----------------------------------------------
        workflow_panel = self._create_card(settings)
        workflow_panel.pack(fill="x", pady=(Theme.S_MD, 0))

        self._card_header(workflow_panel, "Workflow", "Detection and output")

        checks_frame = tk.Frame(workflow_panel, bg=Theme.BG_CARD)
        checks_frame.pack(fill="x", padx=Theme.S_LG, pady=(0, Theme.S_MD))

        self.skip_check = ModernToggle(
            checks_frame,
            text="Reuse a fixed subtitle region (skip per-frame scanning)",
            variable=self.skip_detection_var,
        )
        self.skip_check.pack(anchor="w")
        Tooltip(self.skip_check, "Skip repeated detection when you have already set a precise subtitle region.")

        self.lama_check = ModernToggle(
            checks_frame,
            text="LaMa fast mode - favor speed over fill detail",
            variable=self.lama_fast_var,
        )
        self.lama_check.pack(anchor="w", pady=(Theme.S_SM, 0))
        Tooltip(self.lama_check, "LaMa fast mode is useful for quick passes and lower-resolution drafts.")

        self.preserve_audio_check = ModernToggle(
            checks_frame,
            text="Preserve the source audio track",
            variable=self.preserve_audio_var,
        )
        self.preserve_audio_check.pack(anchor="w", pady=(Theme.S_SM, 0))
        if not self.ffmpeg_ready:
            tk.Label(
                checks_frame,
                text="FFmpeg is not available, so outputs will be saved without original audio until it is installed.",
                font=f(Theme.F_META),
                bg=Theme.BG_CARD,
                fg=Theme.WARNING,
                wraplength=520,
                justify="left",
            ).pack(anchor="w", pady=(Theme.S_XS, 0))

        # Region surface -- raised card-within-card
        region_surface = tk.Frame(workflow_panel, bg=Theme.BG_TERTIARY,
                                  highlightthickness=1,
                                  highlightbackground=Theme.BORDER_SUBTLE)
        region_surface.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_XS, Theme.S_LG))

        region_text = tk.Frame(region_surface, bg=Theme.BG_TERTIARY)
        region_text.pack(side="left", fill="x", expand=True, padx=Theme.S_MD, pady=Theme.S_MD)

        tk.Label(region_text, text="SUBTITLE REGION", font=f(Theme.F_EYEBROW, "bold"),
                 bg=Theme.BG_TERTIARY, fg=Theme.TEXT_MUTED).pack(anchor="w")

        self.region_label = tk.Label(region_text, text="", font=f(Theme.F_BODY, "bold"),
                                     bg=Theme.BG_TERTIARY, fg=Theme.TEXT_PRIMARY,
                                     anchor="w")
        self.region_label.pack(anchor="w", pady=(4, 0))

        self.region_meta = tk.Label(region_text, text="", font=f(Theme.F_META),
                                    bg=Theme.BG_TERTIARY, fg=Theme.TEXT_MUTED,
                                    anchor="w")
        self.region_meta.pack(anchor="w", pady=(2, 0))

        region_actions = tk.Frame(region_surface, bg=Theme.BG_TERTIARY)
        region_actions.pack(side="right", padx=Theme.S_MD, pady=Theme.S_MD)

        self.region_btn = ModernButton(region_actions, text="Set region", width=100,
                                       command=self._open_region_selector, style="accent",
                                       size="sm")
        self.region_btn.pack(side="left")

        self.region_reset_btn = ModernButton(region_actions, text="Reset", width=76,
                                             command=self._reset_region, style="ghost",
                                             size="sm")
        self.region_reset_btn.pack(side="left", padx=(Theme.S_SM, 0))

        # ---- Advanced toggle --------------------------------------------
        adv_frame = tk.Frame(settings, bg=Theme.BG_SECONDARY)
        adv_frame.pack(fill="x", pady=(Theme.S_MD, 0))

        self.adv_visible = False
        self.adv_toggle = ModernButton(adv_frame, text="Show detailed controls", width=188,
                                       command=self._toggle_advanced,
                                       style="ghost", size="sm", icon="+")
        self.adv_toggle.pack(anchor="w")

        self.adv_panel = tk.Frame(settings, bg=Theme.BG_SECONDARY)

        # STTN Motion card
        sttn_frame = self._create_card(self.adv_panel)
        sttn_frame.pack(fill="x", pady=(Theme.S_MD, Theme.S_SM))
        self._card_header(sttn_frame, "STTN motion", "Temporal coherence")

        self._create_slider(sttn_frame, "Neighbor stride", 5, 30,
                            self.config.sttn_neighbor_stride, "sttn_neighbor_stride")
        self._create_slider(sttn_frame, "Reference length", 5, 30,
                            self.config.sttn_reference_length, "sttn_reference_length")
        self._create_slider(sttn_frame, "Max load frames", 10, 100,
                            self.config.sttn_max_load_num, "sttn_max_load_num")
        tk.Frame(sttn_frame, bg=Theme.BG_CARD, height=Theme.S_SM).pack(fill="x")

        # Detection Precision card
        det_frame = self._create_card(self.adv_panel)
        det_frame.pack(fill="x", pady=(0, Theme.S_SM))
        self._card_header(det_frame, "Detection", "Precision tuning")

        self._create_slider(det_frame, "Sensitivity", 10, 90,
                            int(self.config.detection_threshold * 100),
                            "_detection_threshold_pct",
                            hint="Higher catches more text (lower confidence floor). Lower is stricter.")
        self._create_slider(det_frame, "Frame skip", 0, 10,
                            self.config.detection_frame_skip, "detection_frame_skip",
                            hint="Reuse the last mask for N frames to speed up long videos.")
        self._create_slider(det_frame, "Mask dilate", 0, 20,
                            self.config.mask_dilate_px, "mask_dilate_px",
                            hint="Expand detected regions for cleaner fill edges.")
        self._create_slider(det_frame, "Mask feather", 0, 15,
                            self.config.mask_feather_px, "mask_feather_px",
                            hint="Soft-blend the removal edge for seamless boundaries.")
        self._create_slider(det_frame, "Colour match ring", 0, 8,
                            self.config.edge_ring_px, "edge_ring_px",
                            hint="Post-inpaint edge-ring colour correction to kill faint seams.")

        self.auto_band_var = tk.BooleanVar(value=self.config.auto_band)
        auto_band_toggle = ModernToggle(
            det_frame,
            text="Auto-detect subtitle band on load",
            variable=self.auto_band_var,
        )
        auto_band_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(auto_band_toggle, "Scan the first 30 frames and pin the dominant subtitle band before processing.")

        self.flow_warp_var = tk.BooleanVar(value=self.config.tbe_flow_warp)
        flow_toggle = ModernToggle(
            det_frame,
            text="Flow-warped temporal exposure (motion-heavy)",
            variable=self.flow_warp_var,
        )
        flow_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(flow_toggle, "Farneback optical flow aligns frames before TBE aggregation. Slower but cleaner on pans and zooms.")

        self.scene_split_var = tk.BooleanVar(value=self.config.tbe_scene_cut_split)
        scene_toggle = ModernToggle(
            det_frame,
            text="Split TBE batches at scene cuts",
            variable=self.scene_split_var,
        )
        scene_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(scene_toggle, "Prevents background aggregation across hard cuts. Turn off if your footage is uncut.")

        self.kalman_var = tk.BooleanVar(value=self.config.kalman_tracking)
        kalman_toggle = ModernToggle(
            det_frame,
            text="Kalman box tracking (flicker reduction)",
            variable=self.kalman_var,
        )
        kalman_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(kalman_toggle, "Smooths per-frame OCR jitter and fills single-frame misses. Recommended.")

        self.phash_var = tk.BooleanVar(value=self.config.phash_skip_enable)
        phash_toggle = ModernToggle(
            det_frame,
            text="Adaptive mask reuse (perceptual hash)",
            variable=self.phash_var,
        )
        phash_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(phash_toggle, "Skip OCR on frames nearly identical to the last detected one. Speeds up long static shots.")

        self.colour_tune_var = tk.BooleanVar(value=self.config.colour_tune_enable)
        colour_toggle = ModernToggle(
            det_frame,
            text="Colour-tuned mask expansion",
            variable=self.colour_tune_var,
        )
        colour_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(colour_toggle, "Grow the mask to cover serifs / drop shadows that match the subtitle colour. Catches decorative lettering.")

        # RM-24: vertical-text toggle for Japanese tategaki / classical CN.
        self.vertical_text_var = tk.BooleanVar(value=getattr(self.config, "detection_vertical", False))
        vertical_toggle = ModernToggle(
            det_frame,
            text="Vertical text mode (Japanese tategaki / classical Chinese)",
            variable=self.vertical_text_var,
        )
        vertical_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, Theme.S_MD))
        Tooltip(vertical_toggle, "Rotates each frame 90 CCW before OCR so columnar CJK reads as a line. Boxes rotate back to the source frame.")

        tk.Frame(det_frame, bg=Theme.BG_CARD, height=Theme.S_SM).pack(fill="x")

        # Output Quality card
        quality_frame = self._create_card(self.adv_panel)
        quality_frame.pack(fill="x", pady=(0, Theme.S_SM))
        self._card_header(quality_frame, "Output", "Encoding quality")

        self._create_slider(quality_frame, "CRF target", 15, 35,
                            self.config.output_quality, "output_quality",
                            hint="Lower = higher quality. 23 is a balanced default.")

        self.hw_encode_var = tk.BooleanVar(value=self.config.use_hw_encode)
        self.hw_encode_check = ModernToggle(
            quality_frame,
            text="Hardware encoding (NVENC / QSV / AMF) with software fallback",
            variable=self.hw_encode_var,
        )
        self.hw_encode_check.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(self.hw_encode_check, "If hardware encoding fails the app retries automatically with libx264.")

        # F-8: output codec selector lives next to the HW-encode toggle.
        codec_row = tk.Frame(quality_frame, bg=Theme.BG_CARD)
        codec_row.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        tk.Label(codec_row, text="Output codec", font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(side="left")
        self.output_codec_var = tk.StringVar(value=getattr(self.config, "output_codec", "h264"))
        codec_combo = ttk.Combobox(
            codec_row, textvariable=self.output_codec_var, width=10,
            values=["h264", "h265", "av1"],
            state="readonly", style="Dark.TCombobox", font=f(Theme.F_BODY_SM),
        )
        codec_combo.pack(side="right")
        Tooltip(codec_combo,
                "h264 is universal; h265 and av1 cut bitrate ~50% on 4K. Uses NVENC/QSV/AMF when available.")

        self.adaptive_batch_var = tk.BooleanVar(value=self.config.adaptive_batch)
        adaptive_toggle = ModernToggle(
            quality_frame,
            text="Adaptive batch sizing (probe free VRAM on init)",
            variable=self.adaptive_batch_var,
        )
        adaptive_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(adaptive_toggle, "Scale the TBE window to fit free VRAM. Prevents OOM on 4K, unlocks headroom on 24 GB cards.")

        self.export_srt_var = tk.BooleanVar(value=self.config.export_srt)
        srt_toggle = ModernToggle(
            quality_frame,
            text="Export detected text as .srt sidecar",
            variable=self.export_srt_var,
        )
        srt_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(srt_toggle, "Writes an .srt file next to the output using OCR text and timings.")

        self.export_mask_var = tk.BooleanVar(value=self.config.export_mask_video)
        mask_toggle = ModernToggle(
            quality_frame,
            text="Export debug mask video (.mask.mp4)",
            variable=self.export_mask_var,
        )
        mask_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(mask_toggle, "Writes a black-and-white mp4 of the per-frame detection mask alongside the output.")

        self.deinterlace_var = tk.BooleanVar(value=self.config.deinterlace_auto)
        deinterlace_toggle = ModernToggle(
            quality_frame,
            text="Auto-deinterlace interlaced sources (yadif)",
            variable=self.deinterlace_var,
        )
        deinterlace_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(deinterlace_toggle, "ffprobe-checks the input for combing; runs ffmpeg yadif if detected.")

        self.keyframe_var = tk.BooleanVar(value=self.config.keyframe_detection)
        keyframe_toggle = ModernToggle(
            quality_frame,
            text="Keyframe-driven detection (OCR only at I-frames)",
            variable=self.keyframe_var,
        )
        keyframe_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(keyframe_toggle, "Large speedup for long videos. Falls back to pHash skip if ffprobe is missing.")

        self.quality_report_var = tk.BooleanVar(value=self.config.quality_report)
        quality_toggle = ModernToggle(
            quality_frame,
            text="Compute PSNR / SSIM quality report after run",
            variable=self.quality_report_var,
        )
        quality_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, Theme.S_MD))
        Tooltip(quality_toggle, "Samples 10 random frames, compares input vs output; logged and shown in the batch summary.")

        # Video Range card
        time_frame = self._create_card(self.adv_panel)
        time_frame.pack(fill="x")
        self._card_header(time_frame, "Video range", "Trim (videos only)")

        time_inner = tk.Frame(time_frame, bg=Theme.BG_CARD)
        time_inner.pack(fill="x", padx=Theme.S_LG, pady=(0, Theme.S_MD))

        tk.Label(time_inner, text="Start (s)", font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(side="left")
        self.time_start_entry = tk.Entry(
            time_inner, width=7, bg=Theme.BG_TERTIARY,
            fg=Theme.TEXT_PRIMARY, font=f(Theme.F_BODY_SM),
            insertbackground=Theme.TEXT_PRIMARY,
            highlightthickness=1,
            highlightbackground=Theme.BORDER,
            highlightcolor=Theme.BORDER_FOCUS,
            relief="flat", bd=6)
        self.time_start_entry.insert(0, str(self.config.time_start or 0))
        self.time_start_entry.pack(side="left", padx=(Theme.S_SM, Theme.S_MD))

        tk.Label(time_inner, text="End (s)", font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(side="left")
        self.time_end_entry = tk.Entry(
            time_inner, width=7, bg=Theme.BG_TERTIARY,
            fg=Theme.TEXT_PRIMARY, font=f(Theme.F_BODY_SM),
            insertbackground=Theme.TEXT_PRIMARY,
            highlightthickness=1,
            highlightbackground=Theme.BORDER,
            highlightcolor=Theme.BORDER_FOCUS,
            relief="flat", bd=6)
        self.time_end_entry.insert(0, str(self.config.time_end or 0))
        self.time_end_entry.pack(side="left", padx=(Theme.S_SM, 0))

        tk.Label(time_inner, text="0 uses the full clip", font=f(Theme.F_META),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED).pack(side="left", padx=(Theme.S_MD, 0))

        # ---- v3.13 GUI-exposed knobs ------------------------------------
        # Editorial: chyron vs subtitle filter + karaoke grouping
        editorial_frame = self._create_card(self.adv_panel)
        editorial_frame.pack(fill="x", pady=(Theme.S_MD, Theme.S_SM))
        self._card_header(editorial_frame, "Editorial", "Filter what gets removed")

        self.remove_subs_var = tk.BooleanVar(value=self.config.remove_subtitles)
        remove_subs_toggle = ModernToggle(
            editorial_frame,
            text="Remove dialogue subtitles (short-lived OCR tracks)",
            variable=self.remove_subs_var,
        )
        remove_subs_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_XS, 0))
        Tooltip(remove_subs_toggle, "Tracks the chyron classifier marks as dialogue subtitles.")

        self.remove_chyrons_var = tk.BooleanVar(value=self.config.remove_chyrons)
        remove_chyrons_toggle = ModernToggle(
            editorial_frame,
            text="Remove persistent text (logos, tickers, lower-thirds)",
            variable=self.remove_chyrons_var,
        )
        remove_chyrons_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(remove_chyrons_toggle, "Kalman tracks lasting longer than ~3s are treated as chyrons.")

        self.karaoke_grouping_var = tk.BooleanVar(value=self.config.karaoke_grouping)
        karaoke_toggle = ModernToggle(
            editorial_frame,
            text="Karaoke grouping: fuse per-syllable boxes on the same line",
            variable=self.karaoke_grouping_var,
        )
        karaoke_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, Theme.S_MD))
        Tooltip(karaoke_toggle, "Stops karaoke captions leaking original text through the gaps between syllables.")

        # Audio card: loudnorm target + multi-track passthrough
        audio_frame = self._create_card(self.adv_panel)
        audio_frame.pack(fill="x", pady=(0, Theme.S_SM))
        self._card_header(audio_frame, "Audio", "Loudness + tracks")

        self.multi_audio_var = tk.BooleanVar(value=self.config.multi_audio_passthrough)
        multi_audio_toggle = ModernToggle(
            audio_frame,
            text="Pass through every audio stream (Bluray/DVD multi-track)",
            variable=self.multi_audio_var,
        )
        multi_audio_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_XS, 0))
        Tooltip(multi_audio_toggle, "Mux every audio stream from the source. Off keeps only the first track.")

        loudnorm_row = tk.Frame(audio_frame, bg=Theme.BG_CARD)
        loudnorm_row.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_SM, Theme.S_MD))
        tk.Label(loudnorm_row, text="EBU R128 loudness target", font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(side="left")
        self.loudnorm_var = tk.StringVar(value=str(self.config.loudnorm_target or 0.0))
        loudnorm_entry = tk.Entry(
            loudnorm_row, width=7, bg=Theme.BG_TERTIARY,
            fg=Theme.TEXT_PRIMARY, font=f(Theme.F_BODY_SM),
            insertbackground=Theme.TEXT_PRIMARY,
            highlightthickness=1,
            highlightbackground=Theme.BORDER,
            highlightcolor=Theme.BORDER_FOCUS,
            relief="flat", bd=6, textvariable=self.loudnorm_var)
        loudnorm_entry.pack(side="left", padx=(Theme.S_SM, Theme.S_MD))
        tk.Label(loudnorm_row,
                 text="LUFS. 0 = off. YouTube -14, Apple -16, broadcast -23.",
                 font=f(Theme.F_META),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED).pack(side="left")

        # Performance card: decode HW accel + prefetch
        perf_frame = self._create_card(self.adv_panel)
        perf_frame.pack(fill="x", pady=(0, Theme.S_SM))
        self._card_header(perf_frame, "Performance", "Decode pipeline")

        accel_row = tk.Frame(perf_frame, bg=Theme.BG_CARD)
        accel_row.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_XS, 0))
        tk.Label(accel_row, text="Hardware-decode hint", font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(side="left")
        self.decode_accel_var = tk.StringVar(value=self.config.decode_hw_accel or "off")
        accel_combo = ttk.Combobox(
            accel_row, textvariable=self.decode_accel_var, width=10,
            values=["off", "auto", "any", "d3d11", "vaapi", "mfx"],
            state="readonly", style="Dark.TCombobox", font=f(Theme.F_BODY_SM),
        )
        accel_combo.pack(side="right")
        Tooltip(accel_combo, "Hint for cv2.VideoCapture. Falls back to software if the HW path returns no frames.")

        self.prefetch_var = tk.BooleanVar(value=self.config.prefetch_decode)
        prefetch_toggle = ModernToggle(
            perf_frame,
            text="Worker-thread frame prefetch (overlap decode and inpaint)",
            variable=self.prefetch_var,
        )
        prefetch_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, Theme.S_MD))
        Tooltip(prefetch_toggle, "Decouples cv2.VideoCapture.read() from the detect+inpaint critical path. On by default.")

        # Quality sheet toggle (lives under Output but kept separate so we
        # don't disturb the existing Output card layout)
        self.quality_sheet_var = tk.BooleanVar(value=self.config.quality_report_sheet)
        quality_sheet_toggle = ModernToggle(
            quality_frame,
            text="Quality report sheet (side-by-side PNG comparison)",
            variable=self.quality_sheet_var,
        )
        quality_sheet_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(0, Theme.S_SM))
        Tooltip(quality_sheet_toggle, "Renders <output>.qualitysheet.png with per-sample PSNR/SSIM. Implies the numeric report.")

        # RM-96: high-contrast theme toggle. Takes effect on next launch
        # because re-skinning every live widget mid-session would force
        # a tree-wide redraw the design tokens were not built for.
        self.high_contrast_var = tk.BooleanVar(value=getattr(self.config, "high_contrast", False))
        hc_toggle = ModernToggle(
            quality_frame,
            text="High-contrast theme (restart required)",
            variable=self.high_contrast_var,
        )
        hc_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(0, Theme.S_MD))
        Tooltip(hc_toggle, "Alternative palette tuned for low-vision users. Persists across sessions.")

        self._update_region_label_display()
        self._update_mode_options()

    def _create_slider(self, parent, label, min_val, max_val, default, attr_name,
                       hint: str = ""):
        """Create a labeled row with a ModernSlider and a value pill. Optional
        helper hint below."""
        parent_bg = parent.cget("bg") if hasattr(parent, "cget") else Theme.BG_CARD
        row = tk.Frame(parent, bg=parent_bg)
        row.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_XS, 2))

        tk.Label(row, text=label, font=f(Theme.F_BODY_SM),
                 bg=parent_bg, fg=Theme.TEXT_SECONDARY,
                 width=16, anchor="w").pack(side="left")

        # Value pill on the right
        pill = tk.Frame(row, bg=Theme.BG_TERTIARY, highlightthickness=1,
                        highlightbackground=Theme.BORDER_SUBTLE)
        pill.pack(side="right", padx=(Theme.S_MD, 0))
        value_label = tk.Label(pill, text=str(default), font=f(Theme.F_BODY_SM, "bold"),
                               bg=Theme.BG_TERTIARY, fg=Theme.GREEN_PRIMARY,
                               padx=8, pady=1, width=4)
        value_label.pack()

        slider = ModernSlider(row, from_=min_val, to=max_val, value=default,
                              bg=parent_bg)
        slider.pack(side="left", fill="x", expand=True, padx=(Theme.S_SM, 0))

        def on_change(val):
            value_label.config(text=str(int(val)))
            setattr(self.config, attr_name, int(val))

        slider.command = on_change

        if hint:
            tk.Label(parent, text=hint, font=f(Theme.F_META),
                     bg=parent_bg, fg=Theme.TEXT_MUTED,
                     anchor="w", justify="left").pack(
                         fill="x", padx=(Theme.S_LG, Theme.S_LG),
                         pady=(0, Theme.S_XS))

    def _toggle_advanced(self, event=None):
        """Toggle advanced settings visibility."""
        self.adv_visible = not self.adv_visible
        if self.adv_visible:
            self.adv_toggle.icon = "-"
            self.adv_toggle.set_text("Hide detailed controls")
            self.adv_panel.pack(fill="x")
        else:
            self.adv_toggle.icon = "+"
            self.adv_toggle.set_text("Show detailed controls")
            self.adv_panel.pack_forget()

    def _build_queue_section(self, parent):
        """Queue + preview + batch controls column."""
        section = self._create_surface(parent)
        section.pack(fill="both", expand=True)

        header = tk.Frame(section, bg=Theme.BG_SECONDARY)
        header.pack(fill="x", padx=Theme.S_XL, pady=(Theme.S_LG, Theme.S_XS))

        heading = tk.Frame(header, bg=Theme.BG_SECONDARY)
        heading.pack(side="left", fill="x", expand=True)

        tk.Label(heading, text="Queue",
                 font=f(Theme.F_HEADING, "bold"),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY).pack(anchor="w")
        tk.Label(heading, text="Review the list, then start the batch when ready.",
                 font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED).pack(anchor="w", pady=(2, 0))

        # Count + status chip cluster (right-aligned)
        count_cluster = tk.Frame(header, bg=Theme.BG_SECONDARY)
        count_cluster.pack(side="right", anchor="n")

        def _mk_stat_pill(fg=Theme.TEXT_SECONDARY, bg=Theme.BG_TERTIARY):
            pill = tk.Frame(count_cluster, bg=Theme.BG_SECONDARY)
            lbl = tk.Label(pill, text="", font=f(Theme.F_META, "bold"),
                           bg=bg, fg=fg, padx=8, pady=2)
            lbl.pack()
            return pill, lbl

        self.queue_total_pill, self.queue_count = _mk_stat_pill(
            fg=Theme.TEXT_PRIMARY, bg=Theme.BG_TERTIARY)
        self.queue_done_pill, self.queue_done_lbl = _mk_stat_pill(
            fg=Theme.SUCCESS, bg=Theme.SUCCESS_BG)
        self.queue_err_pill, self.queue_err_lbl = _mk_stat_pill(
            fg=Theme.ERROR, bg=Theme.ERROR_BG)

        self.queue_total_pill.pack(side="left")
        # done/err pills get shown conditionally in _update_queue_display
        self.queue_count.config(text="0 items")

        # Sort button -- hidden until queue has >= 3 items
        self._sort_btn = ModernButton(
            count_cluster, text="Sort", width=72,
            command=self._open_sort_menu, style="ghost", size="sm")
        # packed conditionally in _update_queue_display

        # Batch progress -- labels row above the bar
        batch_frame = tk.Frame(section, bg=Theme.BG_SECONDARY)
        batch_frame.pack(fill="x", padx=Theme.S_XL, pady=(Theme.S_MD, 0))

        meta_row = tk.Frame(batch_frame, bg=Theme.BG_SECONDARY)
        meta_row.pack(fill="x")

        self.batch_label = tk.Label(meta_row, text="Ready",
                                    font=f(Theme.F_META, "bold"),
                                    bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED)
        self.batch_label.pack(side="left")

        self.batch_percent_label = tk.Label(meta_row, text="",
                                            font=f(Theme.F_META, "bold"),
                                            bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY)
        self.batch_percent_label.pack(side="right")

        batch_bar_frame = tk.Frame(section, bg=Theme.BG_SECONDARY)
        batch_bar_frame.pack(fill="x", padx=Theme.S_XL, pady=(4, Theme.S_SM))

        self.batch_progress = ModernProgressBar(batch_bar_frame, width=300, height=5,
                                                 fill=Theme.BLUE_PRIMARY)
        self.batch_progress.pack(fill="x")
        def _resize_batch(event):
            if event.width > 40:
                self.batch_progress.resize(event.width - 4)
        batch_bar_frame.bind("<Configure>", _resize_batch)

        # Queue filter input -- appears when there are >5 items
        self._queue_filter_var = tk.StringVar()
        self._queue_filter_frame = tk.Frame(
            section, bg=Theme.BG_TERTIARY,
            highlightthickness=1, highlightbackground=Theme.BORDER)
        # Packed/unpacked dynamically in _update_queue_display
        filter_inner = tk.Frame(self._queue_filter_frame, bg=Theme.BG_TERTIARY)
        filter_inner.pack(fill="x", padx=Theme.S_SM, pady=2)

        tk.Label(filter_inner, text="Filter", font=f(Theme.F_META, "bold"),
                 bg=Theme.BG_TERTIARY, fg=Theme.TEXT_MUTED).pack(
                     side="left", padx=(Theme.S_SM, Theme.S_SM))
        self._queue_filter_entry = tk.Entry(
            filter_inner, textvariable=self._queue_filter_var,
            bg=Theme.BG_TERTIARY, fg=Theme.TEXT_PRIMARY,
            insertbackground=Theme.TEXT_PRIMARY,
            font=f(Theme.F_BODY_SM), relief="flat", bd=6,
            highlightthickness=0)
        self._queue_filter_entry.pack(side="left", fill="x", expand=True)
        self._queue_filter_entry.bind(
            "<FocusIn>",
            lambda e: self._queue_filter_frame.config(highlightbackground=Theme.BORDER_FOCUS),
        )
        self._queue_filter_entry.bind(
            "<FocusOut>",
            lambda e: self._queue_filter_frame.config(highlightbackground=Theme.BORDER),
        )
        self._queue_filter_clear = ModernButton(
            filter_inner, text="Clear", width=68,
            command=lambda: self._queue_filter_var.set(""),
            style="ghost", size="sm")
        self._queue_filter_clear.pack(side="right", padx=(Theme.S_SM, 0))
        self._queue_filter_var.trace_add(
            "write", lambda *_: self._apply_queue_filter())

        self._queue_container = tk.Frame(section, bg=Theme.BG_SECONDARY)
        self._queue_container.pack(fill="both", expand=True,
                                   padx=Theme.S_XL, pady=(0, Theme.S_SM))
        queue_container = self._queue_container

        self.queue_canvas = tk.Canvas(queue_container, bg=Theme.BG_SECONDARY,
                                     highlightthickness=0)
        scrollbar = ttk.Scrollbar(queue_container, orient="vertical",
                                 command=self.queue_canvas.yview,
                                 style="Dark.Vertical.TScrollbar")

        self.queue_frame = tk.Frame(self.queue_canvas, bg=Theme.BG_SECONDARY)

        self.queue_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.queue_canvas.pack(side="left", fill="both", expand=True)

        self.queue_window = self.queue_canvas.create_window((0, 0), window=self.queue_frame,
                                                            anchor="nw")

        self.queue_frame.bind("<Configure>", self._on_queue_configure)
        self.queue_canvas.bind("<Configure>", self._on_canvas_configure)

        # Mousewheel scrolling
        self.queue_canvas.bind("<Enter>", self._bind_mousewheel)
        self.queue_canvas.bind("<Leave>", self._unbind_mousewheel)

        self._build_queue_empty_state()

        # Preview card
        self._preview_frame = self._create_card(section)
        self._preview_frame.pack(fill="x", padx=Theme.S_XL, pady=(0, Theme.S_MD))

        preview_header = tk.Frame(self._preview_frame, bg=Theme.BG_CARD)
        preview_header.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_MD, 0))

        preview_text = tk.Frame(preview_header, bg=Theme.BG_CARD)
        preview_text.pack(side="left", fill="x", expand=True)

        self.preview_title_label = tk.Label(preview_text, text="Preview a sample frame",
                                            font=f(Theme.F_TITLE, "bold"),
                                            bg=Theme.BG_CARD, fg=Theme.TEXT_PRIMARY)
        self.preview_title_label.pack(anchor="w")
        self.preview_meta_label = tk.Label(
            preview_text,
            text="Select a queued item and review the mask before processing.",
            font=f(Theme.F_META), wraplength=360,
            justify="left", bg=Theme.BG_CARD,
            fg=Theme.TEXT_MUTED)
        self.preview_meta_label.pack(anchor="w", pady=(4, 0))

        preview_actions = tk.Frame(preview_header, bg=Theme.BG_CARD)
        preview_actions.pack(side="right", anchor="ne")
        self.preview_status_chip = tk.Label(
            preview_actions,
            text="Waiting",
            font=f(Theme.F_META, "bold"),
            bg=Theme.BG_TERTIARY,
            fg=Theme.TEXT_MUTED,
            padx=10,
            pady=4,
        )
        self.preview_status_chip.pack(side="left", padx=(0, Theme.S_SM))
        self.preview_mask_btn = ModernButton(
            preview_actions,
            text="Review mask",
            width=108,
            command=self._open_selected_mask_preview,
            style="ghost",
            size="sm",
        )
        self.preview_mask_btn.pack(side="left")
        Tooltip(self.preview_mask_btn,
                "Run detection on the selected item and show the first-frame mask.")
        self.preview_zoom_btn = ModernButton(
            preview_actions,
            text="Full size",
            width=92,
            command=self._open_preview_zoom,
            style="ghost",
            size="sm",
        )
        self.preview_zoom_btn.pack(side="left", padx=(Theme.S_SM, 0))
        Tooltip(self.preview_zoom_btn,
                "Open the selected source frame in a larger viewer.")
        # F-3: cheap inpaint preview for the first frame of the selected
        # item. Runs detect + inpaint once and renders the result inline
        # so users can A/B settings before committing the batch.
        self.preview_inpaint_btn = ModernButton(
            preview_actions,
            text="Preview cleanup",
            width=128,
            command=self._open_selected_inpaint_preview,
            style="ghost",
            size="sm",
        )
        self.preview_inpaint_btn.pack(side="left", padx=(Theme.S_SM, 0))
        Tooltip(self.preview_inpaint_btn,
                "Run detect + inpaint on the first frame of the selected item.")
        # RM-30: A/B flicker scrubber for completed items.
        self.preview_ab_btn = ModernButton(
            preview_actions,
            text="A/B compare",
            width=108,
            command=self._open_ab_scrubber,
            style="ghost",
            size="sm",
        )
        self.preview_ab_btn.pack(side="left", padx=(Theme.S_SM, 0))
        Tooltip(self.preview_ab_btn,
                "Open a frame slider that wipes between source and cleaned output.")

        self._preview_label = tk.Label(self._preview_frame, bg=Theme.BG_CARD,
                                       text="", font=f(Theme.F_META),
                                       fg=Theme.TEXT_MUTED, compound="bottom",
                                       justify="center", cursor="hand2")
        self._preview_label.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_MD, Theme.S_LG))
        self._preview_photo = None
        self._preview_label.bind("<Double-Button-1>", self._open_preview_zoom)
        Tooltip(self._preview_label,
                "Double-click to view at full size. Right-click a queue item for more actions.")

        # Action bar -- Start is primary, secondary actions right-aligned
        btn_frame = tk.Frame(section, bg=Theme.BG_SECONDARY)
        btn_frame.pack(fill="x", padx=Theme.S_XL, pady=(0, Theme.S_LG))

        self.start_btn = ModernButton(btn_frame, text="Start batch", width=156,
                                     command=self._start_processing,
                                     style="primary", size="lg", icon=">")
        self.start_btn.pack(side="left")

        self.open_output_btn = ModernButton(btn_frame, text="Open output", width=132,
                                            command=self._open_output_folder,
                                            style="ghost", size="lg", icon="^")
        self.open_output_btn.pack(side="left", padx=(Theme.S_SM, 0))

        self.retry_btn = ModernButton(btn_frame, text="Retry failed", width=124,
                                      command=self._retry_failed,
                                      style="ghost", size="lg")
        self.retry_btn.pack(side="right")

        self.clear_btn = ModernButton(btn_frame, text="Clear queue", width=120,
                                     command=self._clear_queue,
                                     style="ghost", size="lg")
        self.clear_btn.pack(side="right", padx=(0, Theme.S_SM))

        self._set_preview_placeholder(
            "Preview a sample frame",
            "Select a queued item to inspect it before processing.",
        )
        self._refresh_action_states()

    def _build_queue_empty_state(self):
        """Queue empty state with short, clear guidance."""
        self.empty_container = tk.Frame(self.queue_frame, bg=Theme.BG_SECONDARY)
        self.empty_container.pack(pady=(Theme.S_3XL, Theme.S_LG), fill="x")

        icon = tk.Canvas(self.empty_container, width=60, height=60,
                         bg=Theme.BG_SECONDARY, highlightthickness=0)
        icon.pack()
        # Simple minimalist film-strip icon
        icon.create_rectangle(6, 12, 54, 48, outline=Theme.BORDER_STRONG, width=2)
        for x in (14, 30, 46):
            icon.create_rectangle(x - 5, 20, x + 5, 40,
                                  fill=Theme.BG_TERTIARY, outline="")

        tk.Label(self.empty_container, text="Your queue is empty",
                 font=f(Theme.F_TITLE, "bold"),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY).pack(pady=(Theme.S_MD, 4))
        tk.Label(self.empty_container,
                 text="Add files on the left to start a batch.",
                 font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED,
                 wraplength=340, justify="center").pack()

    def _ensure_filter_empty_state(self):
        """Create the queue filter empty state on demand."""
        if hasattr(self, "_filter_empty_container") and self._filter_empty_container.winfo_exists():
            return
        self._filter_empty_container = tk.Frame(self.queue_frame, bg=Theme.BG_SECONDARY)

        icon = tk.Canvas(self._filter_empty_container, width=52, height=52,
                         bg=Theme.BG_SECONDARY, highlightthickness=0)
        icon.pack()
        icon.create_oval(10, 10, 34, 34, outline=Theme.BORDER_STRONG, width=2)
        icon.create_line(30, 30, 42, 42, fill=Theme.BORDER_STRONG, width=2)

        self._filter_empty_title = tk.Label(
            self._filter_empty_container,
            text="No queued items match this search",
            font=f(Theme.F_TITLE, "bold"),
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_SECONDARY,
        )
        self._filter_empty_title.pack(pady=(Theme.S_MD, 4))
        self._filter_empty_body = tk.Label(
            self._filter_empty_container,
            text="Clear the filter or search for part of a filename.",
            font=f(Theme.F_BODY_SM),
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_MUTED,
            wraplength=340,
            justify="center",
        )
        self._filter_empty_body.pack()
        ModernButton(
            self._filter_empty_container,
            text="Clear filter",
            width=110,
            command=lambda: self._queue_filter_var.set(""),
            style="ghost",
            size="sm",
        ).pack(pady=(Theme.S_MD, 0))

    def _hide_filter_empty_state(self):
        if hasattr(self, "_filter_empty_container") and self._filter_empty_container.winfo_exists():
            self._filter_empty_container.pack_forget()

    def _bind_mousewheel(self, event):
        self._mousewheel_bound = True
        self.queue_canvas.bind("<MouseWheel>", self._on_mousewheel)
        # Also bind on children so scroll works when hovering queue items
        for child in self.queue_frame.winfo_children():
            child.bind("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, event):
        self._mousewheel_bound = False
        self.queue_canvas.unbind("<MouseWheel>")

    def _on_mousewheel(self, event):
        self.queue_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _scroll_queue_to_item(self, item_id: str):
        """Scroll the queue canvas so the given item is fully visible."""
        widget = self.queue_widgets.get(item_id)
        if not widget:
            return
        try:
            self.queue_canvas.update_idletasks()
            bbox = self.queue_canvas.bbox("all")
            if not bbox:
                return
            total_h = max(1, bbox[3] - bbox[1])
            wy = widget.winfo_y()
            wh = widget.winfo_height()
            view_h = self.queue_canvas.winfo_height()
            top_frac, bot_frac = self.queue_canvas.yview()
            top_px = int(top_frac * total_h)
            bot_px = int(bot_frac * total_h)
            # Only scroll if not already in view
            if wy < top_px:
                self.queue_canvas.yview_moveto(max(0.0, wy / total_h))
            elif wy + wh > bot_px:
                target_top = wy + wh - view_h
                self.queue_canvas.yview_moveto(max(0.0, target_top / total_h))
        except Exception:
            pass

    def _on_queue_configure(self, event):
        self.queue_canvas.configure(scrollregion=self.queue_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.queue_canvas.itemconfig(self.queue_window, width=event.width)

    def _build_log_panel(self, parent):
        """Embedded, collapsible activity log."""
        log_section = self._create_surface(parent)
        log_section.pack(fill="x", pady=(Theme.S_MD, 0))

        log_header = tk.Frame(log_section, bg=Theme.BG_SECONDARY)
        log_header.pack(fill="x", padx=Theme.S_XL, pady=(Theme.S_MD, 0))

        # Title cluster (left)
        title_cluster = tk.Frame(log_header, bg=Theme.BG_SECONDARY)
        title_cluster.pack(side="left")
        tk.Label(title_cluster, text="ACTIVITY", font=f(Theme.F_EYEBROW, "bold"),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED).pack(anchor="w")
        tk.Label(title_cluster, text="Runtime log",
                 font=f(Theme.F_BODY, "bold"),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY).pack(anchor="w", pady=(2, 0))

        # Level badges: warn / error counts — packed in a row between title and toggle
        self._badge_row = tk.Frame(log_header, bg=Theme.BG_SECONDARY)
        self._badge_row.pack(side="left", padx=(Theme.S_MD, 0))
        self._log_warn_badge = tk.Label(
            self._badge_row, text="", font=f(Theme.F_META, "bold"),
            bg=Theme.WARNING_BG, fg=Theme.WARNING, padx=8, pady=3)
        self._log_error_badge = tk.Label(
            self._badge_row, text="", font=f(Theme.F_META, "bold"),
            bg=Theme.ERROR_BG, fg=Theme.ERROR, padx=8, pady=3)

        self._log_visible = True
        self._log_toggle_btn = ModernButton(log_header, text="Hide activity", width=120,
                                            command=self._toggle_log_panel,
                                            style="ghost", size="sm")
        self._log_toggle_btn.pack(side="left", padx=(Theme.S_MD, 0))

        open_log_btn = ModernButton(
            log_header, text="Open log file", width=118,
            command=self._open_log_file,
            style="ghost", size="sm")
        open_log_btn.pack(side="right")

        clear_log_btn = ModernButton(log_header, text="Clear", width=72,
                                     command=self._clear_log,
                                     style="ghost", size="sm")
        clear_log_btn.pack(side="right", padx=(0, Theme.S_SM))

        self._log_body = tk.Frame(log_section, bg=Theme.BG_LOG,
                                  highlightthickness=1,
                                  highlightbackground=Theme.BORDER_SUBTLE)
        self._log_body.pack(fill="x", padx=Theme.S_XL, pady=(Theme.S_SM, Theme.S_LG))

        self.log_text = tk.Text(self._log_body, height=6, bg=Theme.BG_LOG,
                                fg=Theme.TEXT_SECONDARY, font=mono(Theme.F_BODY_SM),
                                relief="flat", bd=8, state="disabled",
                                wrap="word", insertbackground=Theme.TEXT_PRIMARY,
                                selectbackground=Theme.BLUE_MUTED)
        log_scroll = ttk.Scrollbar(self._log_body, orient="vertical",
                                   command=self.log_text.yview,
                                   style="Dark.Vertical.TScrollbar")
        self.log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)

        # Tag colors
        self.log_text.tag_configure("info", foreground=Theme.TEXT_MUTED)
        self.log_text.tag_configure("warning", foreground=Theme.WARNING)
        self.log_text.tag_configure("error", foreground=Theme.ERROR)

        # Initialize closed-state toggle (no flip on first run)
        # We start visible, so text stays "Hide activity"

    def _toggle_log_panel(self):
        """Toggle log panel visibility."""
        self._log_visible = not self._log_visible
        if self._log_visible:
            self._log_body.pack(fill="x", padx=Theme.S_XL, pady=(Theme.S_SM, Theme.S_LG))
            self._log_toggle_btn.set_text("Hide activity")
        else:
            self._log_body.pack_forget()
            self._log_toggle_btn.set_text("Show activity")

    def _update_log_badges(self, warn_count: int, error_count: int):
        """Show/hide warn/error count pills in the log header (always before toggle)."""
        try:
            if warn_count > 0:
                self._log_warn_badge.config(
                    text=f"{warn_count} warning{'s' if warn_count != 1 else ''}")
                self._log_warn_badge.pack(side="left", padx=(0, Theme.S_XS))
            else:
                self._log_warn_badge.pack_forget()
            if error_count > 0:
                self._log_error_badge.config(
                    text=f"{error_count} error{'s' if error_count != 1 else ''}")
                self._log_error_badge.pack(side="left", padx=(0, Theme.S_XS))
            else:
                self._log_error_badge.pack_forget()
        except Exception:
            pass

    def _clear_log(self):
        """Clear the log panel."""
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")
        if hasattr(self, "_log_handler"):
            self._log_handler.reset_counts()
        self._update_status("Activity log cleared")

    def _open_log_file(self):
        """Reveal the current log file in the system shell."""
        if not LOG_FILE.exists():
            self._update_status("The log file is not available yet", "warning")
            return
        try:
            os.startfile(str(LOG_FILE))
            self._update_status("Opened the log file", "info")
        except Exception:
            self._update_status("The log file could not be opened", "warning")

    def _open_settings_folder(self):
        try:
            os.startfile(str(LOG_DIR))
            self._update_status("Opened the settings folder", "info")
        except Exception:
            self._update_status("The settings folder could not be opened", "warning")

    def _build_footer(self, parent):
        """Footer status bar with a colored dot + message and a right-side hint."""
        footer = tk.Frame(parent, bg=Theme.BG_DARK)
        footer.pack(fill="x", pady=(Theme.S_SM, 0))
        self._footer = footer

        left = tk.Frame(footer, bg=Theme.BG_DARK)
        left.pack(side="left")
        self._footer_left = left

        # Status dot
        self.status_dot = tk.Canvas(left, width=10, height=10, bg=Theme.BG_DARK,
                                    highlightthickness=0)
        self._status_dot_item = self.status_dot.create_oval(
            1, 1, 9, 9, fill=Theme.TEXT_SECONDARY, outline="")
        self.status_dot.pack(side="left", padx=(0, Theme.S_SM), pady=2)

        self.status_label = tk.Label(left, text="Ready to process",
                                     font=f(Theme.F_BODY_SM, "bold"),
                                     bg=Theme.BG_DARK, fg=Theme.TEXT_SECONDARY, anchor="w")
        self.status_label.pack(side="left")

        self.status_hint = tk.Label(
            footer,
            text="Add files, review a sample frame, then start.",
            font=f(Theme.F_META),
            bg=Theme.BG_DARK,
            fg=Theme.TEXT_MUTED,
        )
        self.status_hint.pack(side="right")

    def _get_algo_description(self) -> str:
        """Get description for current algorithm."""
        descriptions = {
            "Auto": "Routes each batch to TBE or LaMa based on temporal exposure. Fastest on easy footage, automatically falls back to neural fill on hard frames.",
            "STTN": "Temporal background exposure. Reconstructs the real background from neighbouring frames where the subtitle is absent. Fastest, usually the best choice for live action.",
            "LAMA": "Neural single-frame fill. Highest-quality spatial inpaint for stills, animation, and clean backgrounds. Slower per frame.",
            "ProPainter": "Hybrid temporal + LaMa refinement. Best for motion-heavy footage or thick text. Higher VRAM and slower than STTN.",
        }
        return descriptions.get(self.mode_var.get(), "")

    def _on_mode_changed(self, event=None):
        """Handle algorithm mode change."""
        self.config.mode = InpaintMode(self.mode_var.get())
        self.algo_desc.config(text=self._get_algo_description())
        self._update_mode_options()
        self._update_status(f"Switched to the {self.mode_var.get()} profile")

    def _on_mode_picker_changed(self, value: str):
        """Segmented picker callback -- keep `mode_var` and the combobox path
        compatible."""
        self.mode_var.set(value)
        self._on_mode_changed()

    def _on_preset_applied(self, event=None):
        """Apply the chosen preset to the live config and refresh the UI."""
        name = self.preset_var.get()
        if name == "(custom)":
            return
        if not apply_preset(self.config, name):
            self._update_status(f"Preset '{name}' not found", "warning")
            return
        # Reflect preset changes in the mode picker + toggle vars that back
        # the detection / quality / output cards. The dataclass carries the
        # authoritative state; just push it out to every widget we track.
        self.mode_var.set(self.config.mode.value)
        try:
            self.mode_picker.set(self.config.mode.value)
        except Exception:
            pass
        for attr, field in (
            ("auto_band_var", "auto_band"),
            ("flow_warp_var", "tbe_flow_warp"),
            ("scene_split_var", "tbe_scene_cut_split"),
            ("kalman_var", "kalman_tracking"),
            ("phash_var", "phash_skip_enable"),
            ("colour_tune_var", "colour_tune_enable"),
            ("adaptive_batch_var", "adaptive_batch"),
            ("export_srt_var", "export_srt"),
            ("export_mask_var", "export_mask_video"),
        ):
            if hasattr(self, attr):
                getattr(self, attr).set(getattr(self.config, field))
        self._on_mode_changed()
        save_settings(self.config)
        self._update_status(f"Applied preset '{name}'", "success")

    def _export_preset_dialog(self):
        """Export the currently-selected preset to a shareable JSON file."""
        try:
            from tkinter import filedialog
            name = self.preset_var.get()
            if name == "(custom)":
                self._update_status("Pick a preset first, then export", "warning")
                return
            path = filedialog.asksaveasfilename(
                parent=self.root,
                title=f"Export preset '{name}'",
                defaultextension=".json",
                filetypes=[("VSR preset", "*.json"), ("All files", "*.*")],
                initialfile=f"{name.replace('/', '-')}.vsr-preset.json",
            )
            if not path:
                return
            if export_preset(name, path):
                self._update_status(f"Exported '{name}' to {Path(path).name}", "success")
            else:
                self._update_status("Export failed", "error")
        except Exception as exc:
            self._update_status(f"Export failed: {exc}", "error")

    def _import_preset_dialog(self):
        """Import a preset JSON into the user library and select it."""
        try:
            from tkinter import filedialog
            path = filedialog.askopenfilename(
                parent=self.root,
                title="Import preset",
                filetypes=[("VSR preset", "*.json"), ("All files", "*.*")],
            )
            if not path:
                return
            new_name = import_preset(path)
            if new_name is None:
                self._update_status("Not a valid VSR preset file", "error")
                return
            self.preset_combo['values'] = ["(custom)"] + [n for n, _ in list_presets()]
            self.preset_var.set(new_name)
            self._on_preset_applied()
            self._update_status(f"Imported preset '{new_name}'", "success")
        except Exception as exc:
            self._update_status(f"Import failed: {exc}", "error")

    def _prompt_preset_details(self) -> Optional[Tuple[str, str]]:
        """Open a themed modal for naming and describing a user preset."""
        result = {"value": None}

        dialog = tk.Toplevel(self.root)
        dialog.withdraw()
        dialog.title("Save preset")
        dialog.configure(bg=Theme.BG_OVERLAY)
        dialog.resizable(False, False)
        dialog.transient(self.root)

        outer = tk.Frame(dialog, bg=Theme.BORDER, padx=1, pady=1)
        outer.pack()
        body = tk.Frame(outer, bg=Theme.BG_SECONDARY)
        body.pack()

        content = tk.Frame(body, bg=Theme.BG_SECONDARY)
        content.pack(padx=28, pady=(24, 14))

        tk.Label(content, text="Save the current setup as a preset",
                 font=f(Theme.F_HEADING, "bold"),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY).pack(anchor="w")
        tk.Label(content,
                 text="Use a short name you will recognize later. Saving to an existing user preset name will update it.",
                 font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED,
                 justify="left", wraplength=420).pack(anchor="w", pady=(6, Theme.S_LG))

        form = tk.Frame(content, bg=Theme.BG_SECONDARY)
        form.pack(fill="x")

        def entry_row(label_text: str, initial: str = ""):
            row = tk.Frame(form, bg=Theme.BG_SECONDARY)
            row.pack(fill="x", pady=(0, Theme.S_MD))
            tk.Label(row, text=label_text, font=f(Theme.F_BODY_SM),
                     bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY).pack(anchor="w")
            entry = tk.Entry(
                row, bg=Theme.BG_TERTIARY, fg=Theme.TEXT_PRIMARY,
                insertbackground=Theme.TEXT_PRIMARY,
                font=f(Theme.F_BODY_SM), relief="flat", bd=6,
                highlightthickness=1, highlightbackground=Theme.BORDER,
                highlightcolor=Theme.BORDER_FOCUS,
            )
            entry.pack(fill="x", pady=(Theme.S_XS, 0))
            entry.insert(0, initial)
            return entry

        name_entry = entry_row("Preset name")
        desc_entry = entry_row("Description", "User preset")

        helper = tk.Label(content, text="Built-in preset names are reserved.",
                          font=f(Theme.F_META), bg=Theme.BG_SECONDARY,
                          fg=Theme.TEXT_MUTED)
        helper.pack(anchor="w")

        error_label = tk.Label(content, text="", font=f(Theme.F_META, "bold"),
                               bg=Theme.BG_SECONDARY, fg=Theme.ERROR)
        error_label.pack(anchor="w", pady=(Theme.S_SM, 0))

        actions = tk.Frame(body, bg=Theme.BG_CARD)
        actions.pack(fill="x")
        actions_inner = tk.Frame(actions, bg=Theme.BG_CARD)
        actions_inner.pack(side="right", padx=16, pady=14)

        def _cancel():
            dialog.grab_release()
            dialog.destroy()

        def _submit():
            name = name_entry.get().strip()
            description = desc_entry.get().strip() or "User preset"
            if not name:
                error_label.config(text="Give this preset a short name.")
                name_entry.focus_set()
                return
            if name in BUILTIN_PRESETS:
                error_label.config(text="Built-in preset names are reserved.")
                name_entry.focus_set()
                return
            result["value"] = (name, description)
            dialog.grab_release()
            dialog.destroy()

        ModernButton(actions_inner, text="Cancel", width=96,
                     command=_cancel, style="ghost", size="md").pack(side="left")
        ModernButton(actions_inner, text="Save preset", width=120,
                     command=_submit, style="primary", size="md").pack(
                         side="left", padx=(Theme.S_SM, 0))

        dialog.bind("<Escape>", lambda e: _cancel())
        dialog.bind("<Return>", lambda e: _submit())
        dialog.protocol("WM_DELETE_WINDOW", _cancel)

        dialog.update_idletasks()
        try:
            px = self.root.winfo_rootx()
            py = self.root.winfo_rooty()
            pw = self.root.winfo_width()
            ph = self.root.winfo_height()
            dw = dialog.winfo_reqwidth()
            dh = dialog.winfo_reqheight()
            dialog.geometry(f"+{px + (pw - dw) // 2}+{py + (ph - dh) // 3}")
        except Exception:
            pass

        dialog.deiconify()
        dialog.grab_set()
        name_entry.focus_set()
        dialog.wait_window()
        return result["value"]

    def _save_preset_dialog(self):
        """Prompt for a name + description and save a user preset."""
        try:
            details = self._prompt_preset_details()
            if not details:
                return
            name, description = details
            existing_user = name in _load_user_presets()
            self._sync_config_from_ui()
            if save_user_preset(name, description, self.config):
                # Refresh combo
                self.preset_combo['values'] = ["(custom)"] + [n for n, _ in list_presets()]
                self.preset_var.set(name)
                verb = "Updated" if existing_user else "Saved"
                self._update_status(f"{verb} preset '{name}'", "success")
            else:
                self._update_status(f"Could not save preset '{name}'", "error")
        except Exception as exc:
            self._update_status(f"Save preset failed: {exc}", "error")

    def _update_mode_options(self):
        """Enable/disable mode-specific toggles based on selected algorithm."""
        mode = self.mode_var.get()

        # Skip detection only for STTN
        if mode == "STTN":
            self.skip_check.set_enabled(True)
        else:
            self.skip_detection_var.set(False)
            self.skip_check.set_enabled(False)

        # LAMA fast only for LAMA
        if mode == "LAMA":
            self.lama_check.set_enabled(True)
        else:
            self.lama_fast_var.set(False)
            self.lama_check.set_enabled(False)

    def _maybe_show_onboarding(self):
        """Show a short 3-card welcome overlay on first launch."""
        if self.config.onboarding_seen:
            return
        # Guard against showing twice in the same session
        self.config.onboarding_seen = True
        # Let the main window settle first
        try:
            self.root.after(420, self._show_onboarding)
        except tk.TclError:
            pass

    def _show_onboarding(self):
        dialog = tk.Toplevel(self.root)
        dialog.withdraw()
        dialog.title(f"Welcome to {APP_NAME}")
        dialog.configure(bg=Theme.BG_OVERLAY)
        dialog.resizable(False, False)
        dialog.transient(self.root)

        outer = tk.Frame(dialog, bg=Theme.BORDER, padx=1, pady=1)
        outer.pack()
        body = tk.Frame(outer, bg=Theme.BG_SECONDARY)
        body.pack()

        content = tk.Frame(body, bg=Theme.BG_SECONDARY)
        content.pack(padx=36, pady=(28, 16))

        # Headline
        hero = tk.Frame(content, bg=Theme.BG_SECONDARY)
        hero.pack(anchor="w")
        tk.Label(hero, text="Welcome", font=f(Theme.F_DISPLAY, "bold"),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY).pack(
                     side="left")
        tk.Label(hero, text=f"v{APP_VERSION}", font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED).pack(
                     side="left", padx=(Theme.S_SM, 0), pady=(14, 0))

        tk.Label(content,
                 text="Three things that make batch cleanup painless.",
                 font=f(Theme.F_BODY),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY).pack(
                     anchor="w", pady=(4, Theme.S_LG))

        # Cue cards
        cards = tk.Frame(content, bg=Theme.BG_SECONDARY)
        cards.pack(anchor="w")

        def card(num: str, heading: str, body_text: str, tone: str):
            c = tk.Frame(cards, bg=Theme.BG_CARD, highlightthickness=1,
                         highlightbackground=Theme.BORDER)
            inner = tk.Frame(c, bg=Theme.BG_CARD)
            inner.pack(fill="both", expand=True, padx=16, pady=14)
            top = tk.Frame(inner, bg=Theme.BG_CARD)
            top.pack(anchor="w")
            # Numbered step badge
            badge_bg = {"info": Theme.INFO_BG, "success": Theme.SUCCESS_BG,
                        "warning": Theme.WARNING_BG}.get(tone, Theme.BG_TERTIARY)
            badge_fg = {"info": Theme.INFO, "success": Theme.SUCCESS,
                        "warning": Theme.WARNING}.get(tone, Theme.TEXT_SECONDARY)
            tk.Label(top, text=num, font=f(Theme.F_BODY_SM, "bold"),
                     bg=badge_bg, fg=badge_fg, padx=8, pady=2).pack(side="left")
            tk.Label(top, text=heading, font=f(Theme.F_BODY, "bold"),
                     bg=Theme.BG_CARD, fg=Theme.TEXT_PRIMARY).pack(
                         side="left", padx=(Theme.S_SM, 0))
            tk.Label(inner, text=body_text, font=f(Theme.F_BODY_SM),
                     bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY,
                     wraplength=220, justify="left", anchor="w").pack(
                         anchor="w", pady=(Theme.S_SM, 0))
            return c

        card("1", "Import media",
             "Drop videos or images on the left, or pick an entire folder. "
             "Originals are never touched.",
             "info").pack(side="left", fill="both", expand=True,
                          padx=(0, Theme.S_SM))
        card("2", "Inspect the region",
             "Select a queued item and review the mask to confirm the subtitle "
             "mask before running the batch.",
             "warning").pack(side="left", fill="both", expand=True,
                             padx=(0, Theme.S_SM))
        card("3", "Run the batch",
             "Hit Start batch when the framing looks right. Progress, ETA, "
             "and completion summary are all live.",
             "success").pack(side="left", fill="both", expand=True)

        # Action row
        actions = tk.Frame(body, bg=Theme.BG_CARD)
        actions.pack(fill="x")
        actions_inner = tk.Frame(actions, bg=Theme.BG_CARD)
        actions_inner.pack(side="right", padx=16, pady=14)

        def _close():
            dialog.grab_release()
            dialog.destroy()

        ModernButton(actions_inner, text="Got it", width=118,
                     command=_close, style="primary", size="md").pack(
                         side="left")

        dialog.bind("<Escape>", lambda e: _close())
        dialog.bind("<Return>", lambda e: _close())
        dialog.protocol("WM_DELETE_WINDOW", _close)

        dialog.update_idletasks()
        try:
            px, py = self.root.winfo_rootx(), self.root.winfo_rooty()
            pw, ph = self.root.winfo_width(), self.root.winfo_height()
            dw, dh = dialog.winfo_reqwidth(), dialog.winfo_reqheight()
            dialog.geometry(f"+{px + (pw - dw) // 2}+{py + (ph - dh) // 3}")
        except Exception:
            pass
        dialog.deiconify()
        dialog.grab_set()

    def _open_preview_zoom(self, event=None):
        """Open the currently selected queue item's frame at a larger size."""
        if not PIL_AVAILABLE:
            return
        item_id = self._selected_queue_item_id
        if not item_id:
            return
        item = next((i for i in self.queue if i.id == item_id), None)
        if not item:
            return

        try:
            import cv2 as _cv2

            if is_video_file(item.file_path):
                cap = _cv2.VideoCapture(item.file_path)
                try:
                    ret, frame = cap.read()
                    if not ret:
                        return
                finally:
                    cap.release()
            else:
                frame = _cv2.imread(item.file_path)
                if frame is None:
                    return

            frame_rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
        except Exception:
            return

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        max_w = int(screen_w * 0.82)
        max_h = int(screen_h * 0.82)
        if img.width > max_w or img.height > max_h:
            img.thumbnail((max_w, max_h), Image.LANCZOS)

        win = tk.Toplevel(self.root)
        win.withdraw()
        win.title(f"Preview - {Path(item.file_path).name}")
        win.configure(bg=Theme.BG_DARK)
        win.transient(self.root)

        header = tk.Frame(win, bg=Theme.BG_SECONDARY,
                          highlightthickness=1,
                          highlightbackground=Theme.BORDER_SUBTLE)
        header.pack(fill="x")
        tk.Label(header, text=Path(item.file_path).name,
                 font=f(Theme.F_BODY, "bold"),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY).pack(
                     side="left", padx=Theme.S_LG, pady=Theme.S_MD)
        tk.Label(header, text=f"{img.width} x {img.height}",
                 font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED).pack(
                     side="left", padx=(0, Theme.S_LG), pady=Theme.S_MD)
        ModernButton(header, text="Close", width=86,
                     command=win.destroy, style="ghost", size="sm").pack(
                         side="right", padx=Theme.S_LG, pady=Theme.S_SM)

        canvas = tk.Frame(win, bg=Theme.BG_DARK)
        canvas.pack(fill="both", expand=True, padx=Theme.S_LG,
                    pady=(Theme.S_LG, Theme.S_LG))
        photo = ImageTk.PhotoImage(img)
        label = tk.Label(canvas, image=photo, bg=Theme.BG_DARK)
        label.image = photo  # prevent GC
        label.pack(anchor="center")

        win.bind("<Escape>", lambda e: win.destroy())
        win.update_idletasks()
        try:
            w = win.winfo_reqwidth()
            h = win.winfo_reqheight()
            x = (screen_w - w) // 2
            y = max(20, (screen_h - h) // 2)
            win.geometry(f"+{x}+{y}")
        except Exception:
            pass
        win.deiconify()

    def _show_batch_summary(self, complete: int, errors: int,
                            cancelled: int, elapsed: str,
                            quality_summary: Optional[dict] = None):
        """Themed summary modal shown when a batch finishes."""
        total = complete + errors + cancelled
        is_clean = errors == 0 and cancelled == 0

        dialog = tk.Toplevel(self.root)
        dialog.withdraw()
        dialog.title("Batch finished")
        dialog.configure(bg=Theme.BG_OVERLAY)
        dialog.resizable(False, False)
        dialog.transient(self.root)

        outer = tk.Frame(dialog, bg=Theme.BORDER, padx=1, pady=1)
        outer.pack()
        body = tk.Frame(outer, bg=Theme.BG_SECONDARY)
        body.pack()

        content = tk.Frame(body, bg=Theme.BG_SECONDARY)
        content.pack(padx=32, pady=(26, 16))

        title_text = "Batch finished" if is_clean else "Batch finished with issues"
        title_color = Theme.SUCCESS if is_clean else Theme.WARNING
        tk.Label(content, text=title_text, font=f(Theme.F_HEADING, "bold"),
                 bg=Theme.BG_SECONDARY, fg=title_color).pack(anchor="w")
        if elapsed:
            tk.Label(content, text=f"Total time {elapsed}  -  {total} item"
                                   f"{'s' if total != 1 else ''} processed",
                     font=f(Theme.F_BODY_SM),
                     bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED).pack(
                         anchor="w", pady=(2, 0))
        summary_note = ("Outputs are ready to review."
                        if is_clean else
                        "Completed outputs are ready. Review the outliers or open the log for details.")
        tk.Label(content, text=summary_note, font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY,
                 wraplength=420, justify="left").pack(anchor="w", pady=(Theme.S_SM, 0))

        # Stat row (compact pills)
        stats = tk.Frame(content, bg=Theme.BG_SECONDARY)
        stats.pack(anchor="w", pady=(Theme.S_LG, 0))

        def stat(parent, label, count, fg, bg):
            p = tk.Frame(parent, bg=bg, highlightthickness=1,
                         highlightbackground=Theme.BORDER_SUBTLE)
            tk.Label(
                p,
                text=str(count),
                font=f(Theme.F_HEADING, "bold"),
                bg=bg,
                fg=fg,
                padx=18,
                pady=0,
            ).pack(pady=(10, 0))
            tk.Label(
                p,
                text=label,
                font=f(Theme.F_META, "bold"),
                bg=bg,
                fg=Theme.TEXT_MUTED,
                padx=18,
                pady=0,
            ).pack(pady=(0, 10))
            return p

        stat(stats, "COMPLETED", complete, Theme.SUCCESS, Theme.SUCCESS_BG).pack(
            side="left")
        stat(stats, "FAILED", errors, Theme.ERROR, Theme.ERROR_BG).pack(
            side="left", padx=(Theme.S_SM, 0))
        stat(stats, "STOPPED", cancelled, Theme.WARNING, Theme.WARNING_BG).pack(
            side="left", padx=(Theme.S_SM, 0))

        if quality_summary:
            quality_card = tk.Frame(content, bg=Theme.BG_CARD, highlightthickness=1,
                                    highlightbackground=Theme.BORDER_SUBTLE)
            quality_card.pack(fill="x", pady=(Theme.S_LG, 0))

            tk.Label(
                quality_card,
                text="Sampled quality check",
                font=f(Theme.F_BODY_SM, "bold"),
                bg=Theme.BG_CARD,
                fg=Theme.TEXT_PRIMARY,
            ).pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_MD, 0))

            items_measured = int(quality_summary.get("items", 0) or 0)
            samples = int(quality_summary.get("samples", 0) or 0)
            tk.Label(
                quality_card,
                text=(
                    f"Measured {items_measured} completed item"
                    f"{'s' if items_measured != 1 else ''} across {samples} sampled frame"
                    f"{'s' if samples != 1 else ''}. Higher is generally better."
                ),
                font=f(Theme.F_META),
                bg=Theme.BG_CARD,
                fg=Theme.TEXT_MUTED,
                wraplength=420,
                justify="left",
            ).pack(anchor="w", padx=Theme.S_LG, pady=(4, Theme.S_MD))

            metrics = tk.Frame(quality_card, bg=Theme.BG_CARD)
            metrics.pack(anchor="w", padx=Theme.S_LG, pady=(0, Theme.S_MD))

            stat(metrics, "AVG PSNR", f"{quality_summary['psnr']:.2f} dB",
                 Theme.INFO, Theme.INFO_BG).pack(side="left")
            stat(metrics, "AVG SSIM", f"{quality_summary['ssim']:.4f}",
                 Theme.SUCCESS, Theme.SUCCESS_BG).pack(side="left", padx=(Theme.S_SM, 0))

        # Actions row
        actions = tk.Frame(body, bg=Theme.BG_CARD)
        actions.pack(fill="x")
        actions_inner = tk.Frame(actions, bg=Theme.BG_CARD)
        actions_inner.pack(side="right", padx=16, pady=14)

        def _close():
            dialog.grab_release()
            dialog.destroy()

        def _open_output_and_close():
            self._open_output_folder()
            _close()

        def _retry_failed_and_close():
            self._retry_failed()
            _close()

        if complete > 0:
            ModernButton(actions_inner, text="Open output", width=132,
                         command=_open_output_and_close,
                         style="accent", size="md", icon="^").pack(side="left")
        if errors > 0:
            ModernButton(actions_inner, text="Open log", width=104,
                         command=self._open_log_file,
                         style="ghost", size="md").pack(side="left", padx=(Theme.S_SM, 0))
        if errors > 0 or cancelled > 0:
            ModernButton(actions_inner, text="Retry failed", width=110,
                         command=_retry_failed_and_close,
                         style="ghost", size="md").pack(side="left", padx=(Theme.S_SM, 0))
        ModernButton(actions_inner, text="Close", width=92,
                     command=_close, style="primary", size="md").pack(
                         side="left", padx=(Theme.S_SM, 0))

        dialog.bind("<Escape>", lambda e: _close())
        dialog.bind("<Return>", lambda e: _close())
        dialog.protocol("WM_DELETE_WINDOW", _close)

        dialog.update_idletasks()
        try:
            px, py = self.root.winfo_rootx(), self.root.winfo_rooty()
            pw, ph = self.root.winfo_width(), self.root.winfo_height()
            dw, dh = dialog.winfo_reqwidth(), dialog.winfo_reqheight()
            dialog.geometry(f"+{px + (pw - dw) // 2}+{py + (ph - dh) // 3}")
        except Exception:
            pass
        dialog.deiconify()
        dialog.grab_set()

    def _show_about(self):
        """Open a themed About dialog with version, credits, and quick links."""
        dialog = tk.Toplevel(self.root)
        dialog.withdraw()
        dialog.title(f"About {APP_NAME}")
        dialog.configure(bg=Theme.BG_OVERLAY)
        dialog.resizable(False, False)
        dialog.transient(self.root)

        outer = tk.Frame(dialog, bg=Theme.BORDER, padx=1, pady=1)
        outer.pack()
        body = tk.Frame(outer, bg=Theme.BG_SECONDARY)
        body.pack()

        content = tk.Frame(body, bg=Theme.BG_SECONDARY)
        content.pack(padx=32, pady=(28, 14))

        # Brand row
        brand_row = tk.Frame(content, bg=Theme.BG_SECONDARY)
        brand_row.pack(anchor="w")
        if self._brand_photo:
            tk.Label(brand_row, image=self._brand_photo,
                     bg=Theme.BG_SECONDARY).pack(side="left", padx=(0, Theme.S_MD))
        title_stack = tk.Frame(brand_row, bg=Theme.BG_SECONDARY)
        title_stack.pack(side="left")
        tk.Label(title_stack, text=APP_NAME, font=f(Theme.F_HEADING, "bold"),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY).pack(anchor="w")
        tk.Label(title_stack, text=f"Version {APP_VERSION}",
                 font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED).pack(anchor="w", pady=(2, 0))

        # Fact rows
        fact_card = tk.Frame(content, bg=Theme.BG_CARD, highlightthickness=1,
                             highlightbackground=Theme.BORDER_SUBTLE)
        fact_card.pack(fill="x", pady=(Theme.S_LG, 0))

        def fact(label, value, tone=Theme.TEXT_PRIMARY):
            row = tk.Frame(fact_card, bg=Theme.BG_CARD)
            row.pack(fill="x", padx=14, pady=6)
            tk.Label(row, text=label, font=f(Theme.F_BODY_SM),
                     bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED).pack(side="left")
            tk.Label(row, text=value, font=f(Theme.F_BODY_SM, "bold"),
                     bg=Theme.BG_CARD, fg=tone).pack(side="right")

        det_label = ", ".join(self.ai_engines["detection"]) or "None"
        inp_label = ", ".join(self.ai_engines["inpainting"]) or "None"
        gpu_count = len(self.gpus)
        gpu_label = f"{gpu_count} GPU{'s' if gpu_count != 1 else ''}" if self.gpus else "CPU only"

        fact("Detection engines", det_label, Theme.INFO)
        fact("Inpainting engines", inp_label, Theme.SUCCESS)
        fact("Compute", gpu_label,
             Theme.SUCCESS if self.gpus else Theme.WARNING)
        fact("FFmpeg", "Ready" if self.ffmpeg_ready else "Missing",
             Theme.SUCCESS if self.ffmpeg_ready else Theme.WARNING)
        fact("Shortcuts", "Ctrl+O import   |   Ctrl+Enter start   |   Ctrl+L activity")
        fact("Settings", str(SETTINGS_FILE))
        fact("Log file", str(LOG_FILE))

        # Action row
        actions = tk.Frame(body, bg=Theme.BG_CARD)
        actions.pack(fill="x")
        actions_inner = tk.Frame(actions, bg=Theme.BG_CARD)
        actions_inner.pack(side="right", padx=16, pady=14)

        ModernButton(actions_inner, text="Open log", width=110,
                     command=self._open_log_file, style="ghost", size="md").pack(side="left")
        ModernButton(actions_inner, text="Settings folder", width=140,
                     command=self._open_settings_folder, style="ghost",
                     size="md").pack(side="left", padx=(Theme.S_SM, 0))
        ModernButton(actions_inner, text="Close", width=90,
                     command=dialog.destroy,
                     style="primary", size="md").pack(side="left", padx=(Theme.S_SM, 0))

        dialog.bind("<Escape>", lambda e: dialog.destroy())

        dialog.update_idletasks()
        try:
            px, py = self.root.winfo_rootx(), self.root.winfo_rooty()
            pw, ph = self.root.winfo_width(), self.root.winfo_height()
            dw, dh = dialog.winfo_reqwidth(), dialog.winfo_reqheight()
            dialog.geometry(f"+{px + (pw - dw) // 2}+{py + (ph - dh) // 3}")
        except Exception:
            pass
        dialog.deiconify()
        dialog.grab_set()

    def _set_lang_display(self, code: str):
        """Sync the friendly-name label to the underlying lang code."""
        for label, (c, _) in zip(self._lang_labels, self._lang_display):
            if c == code:
                self._lang_display_var.set(label)
                return
        # Unknown -- default to English
        self._lang_display_var.set(self._lang_labels[0])
        self.lang_var.set(self._lang_display[0][0])

    def _on_lang_changed(self, event=None):
        """Map selected friendly label back to the lang code."""
        label = self._lang_display_var.get()
        code = self._lang_by_label.get(label)
        if code:
            self.lang_var.set(code)
            self.config.detection_lang = code

    def _on_gpu_changed(self, event=None):
        """Handle GPU device selection change."""
        selection = self.gpu_var.get()
        for i, gpu in enumerate(self.gpus):
            label = f"{gpu['name']} ({gpu['memory']})"
            if label == selection:
                self.config.gpu_id = gpu['index']
                self.config.use_gpu = True
                self._update_status(f"Compute device set to {gpu['name']}", "info")
                logger.info(f"GPU set to: {gpu['name']} (index {gpu['index']})")
                break

    def _choose_output_dir(self):
        """Let user pick a custom output directory."""
        d = filedialog.askdirectory(title="Select Output Directory")
        if d:
            self._output_dir = Path(d)
            self._update_output_label()
            refreshed = self._refresh_idle_output_paths()
            if refreshed:
                self._update_queue_display()
            message = "Custom output location selected"
            if refreshed:
                message += f". Updated {refreshed} idle output path{'s' if refreshed != 1 else ''}"
            self._update_status(message, "success")
            logger.info(f"Output directory: {self._output_dir}")

    def _reset_output_dir(self):
        """Reset output directory to default (input_dir/output/)."""
        self._output_dir = None
        self._update_output_label()
        refreshed = self._refresh_idle_output_paths()
        if refreshed:
            self._update_queue_display()
        message = "Output location reset to the default per-folder workflow"
        if refreshed:
            message += f". Updated {refreshed} idle output path{'s' if refreshed != 1 else ''}"
        self._update_status(message)

    def _open_region_selector(self):
        """Open a region-selector window with frame scrubbing (F-1) and
        multi-rectangle drawing (F-2).

        Drag = primary rect (or new rect when "Add another" was clicked).
        The frame slider re-loads the chosen frame so users can target a
        non-zero timecode for clips that open on a black intro card. The
        backend already accepts a `subtitle_areas` list; once the user
        clicks "Save" we write a single rect to `subtitle_area` (back-
        compat) AND the full rect list to `subtitle_areas`.
        """
        source_path = None
        selected = self._get_selected_queue_item()
        if selected:
            source_path = selected.file_path
        else:
            for item in self.queue:
                source_path = item.file_path
                break

        if not source_path:
            source_path = filedialog.askopenfilename(
                title="Select a video/image to define subtitle region",
                filetypes=[("All Supported", "*.mp4;*.avi;*.mkv;*.mov;*.wmv;*.flv;*.webm;*.m4v;*.mpeg;*.mpg;*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.webp")]
            )
        if not source_path:
            return
        if not PIL_AVAILABLE:
            self._update_status("Pillow required for region selector")
            return

        import cv2 as _cv2

        is_video = is_video_file(source_path)
        cap = _cv2.VideoCapture(source_path) if is_video else None
        if is_video and not cap.isOpened():
            logger.error("Could not open video for region selection")
            return
        try:
            if is_video:
                frame_count = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT)) or 1
                fps = cap.get(_cv2.CAP_PROP_FPS) or 30.0
                if fps <= 0:
                    fps = 30.0
                ret, frame = cap.read()
                if not ret:
                    logger.error("Could not read video frame for region selection")
                    return
            else:
                frame = _cv2.imread(source_path)
                if frame is None:
                    logger.error("Could not read image for region selection")
                    return
                frame_count, fps = 1, 30.0

            orig_h, orig_w = frame.shape[:2]
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()
            max_w = min(900, int(screen_w * 0.8))
            max_h = min(540, int(screen_h * 0.7))
            scale = min(max_w / orig_w, max_h / orig_h, 1.0)
            disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)

            # Selector state: every saved rect lives in `rects` (image-
            # space coordinates). Drag creates the current rect; on
            # release it joins the list. Adding another rect re-arms the
            # canvas for the next drag.
            rects: List[Tuple[int, int, int, int]] = []
            preload = self.config.subtitle_areas or (
                [self.config.subtitle_area] if self.config.subtitle_area else []
            )
            rects.extend([tuple(r) for r in preload if r])

            win = tk.Toplevel(self.root)
            win.title("Choose subtitle region")
            win.configure(bg=Theme.BG_OVERLAY)
            win.resizable(False, False)

            canvas = tk.Canvas(win, width=disp_w, height=disp_h, highlightthickness=0,
                               bg=Theme.BG_DARK, cursor="cross")
            canvas.pack()
            canvas_image_id = canvas.create_image(0, 0, anchor="nw")
            canvas._photo = None

            rect_ids: List[int] = []
            drag_id = [None]
            drag_start = [0, 0]

            def _frame_to_pil(bgr):
                rgb = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb).resize((disp_w, disp_h), Image.LANCZOS)

            def _draw_image(bgr):
                pil = _frame_to_pil(bgr)
                canvas._photo = ImageTk.PhotoImage(pil)
                canvas.itemconfig(canvas_image_id, image=canvas._photo)

            def _draw_saved_rects():
                for rid in rect_ids:
                    canvas.delete(rid)
                rect_ids.clear()
                for (x1, y1, x2, y2) in rects:
                    rect_ids.append(canvas.create_rectangle(
                        x1 * scale, y1 * scale, x2 * scale, y2 * scale,
                        outline=Theme.GREEN_PRIMARY, width=2,
                        stipple="gray25", fill=Theme.GREEN_PRIMARY,
                    ))

            def _seek_video(frame_idx: int):
                if not is_video or cap is None:
                    return
                frame_idx = max(0, min(frame_count - 1, int(frame_idx)))
                cap.set(_cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, f = cap.read()
                if ok:
                    _draw_image(f)
                    _draw_saved_rects()

            _draw_image(frame)
            _draw_saved_rects()

            # Drawing handlers.
            def on_press(event):
                drag_start[0], drag_start[1] = event.x, event.y
                if drag_id[0]:
                    canvas.delete(drag_id[0])
                drag_id[0] = canvas.create_rectangle(
                    event.x, event.y, event.x, event.y,
                    outline=Theme.BLUE_PRIMARY, width=2,
                    stipple="gray12", fill=Theme.BLUE_PRIMARY,
                )

            def on_drag(event):
                if drag_id[0]:
                    canvas.coords(drag_id[0], drag_start[0], drag_start[1], event.x, event.y)

            def on_release(event):
                x1 = int(min(drag_start[0], event.x) / scale)
                y1 = int(min(drag_start[1], event.y) / scale)
                x2 = int(max(drag_start[0], event.x) / scale)
                y2 = int(max(drag_start[1], event.y) / scale)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(orig_w, x2), min(orig_h, y2)
                if (x2 - x1) > 10 and (y2 - y1) > 5:
                    rects.append((x1, y1, x2, y2))
                if drag_id[0] is not None:
                    canvas.delete(drag_id[0])
                    drag_id[0] = None
                _draw_saved_rects()

            canvas.bind("<ButtonPress-1>", on_press)
            canvas.bind("<B1-Motion>", on_drag)
            canvas.bind("<ButtonRelease-1>", on_release)

            # Frame slider for videos.
            if is_video and frame_count > 1:
                slider_row = tk.Frame(win, bg=Theme.BG_OVERLAY)
                slider_row.pack(fill="x", padx=Theme.S_MD, pady=(Theme.S_SM, 0))
                tk.Label(slider_row, text="Frame",
                         font=f(Theme.F_BODY_SM),
                         bg=Theme.BG_OVERLAY, fg=Theme.TEXT_SECONDARY).pack(side="left")
                ts_label = tk.Label(slider_row, text="00:00:00",
                                    font=f(Theme.F_META),
                                    bg=Theme.BG_OVERLAY, fg=Theme.TEXT_MUTED)
                ts_label.pack(side="right")

                def _on_slider(value):
                    try:
                        idx = int(float(value))
                    except (TypeError, ValueError):
                        return
                    _seek_video(idx)
                    secs = idx / fps
                    hh = int(secs // 3600); mm = int((secs % 3600) // 60); ss = int(secs % 60)
                    ts_label.config(text=f"{hh:02d}:{mm:02d}:{ss:02d}")

                slider = tk.Scale(
                    win, from_=0, to=frame_count - 1, orient="horizontal",
                    command=_on_slider, length=disp_w - 24,
                    bg=Theme.BG_OVERLAY, fg=Theme.TEXT_PRIMARY,
                    troughcolor=Theme.BG_TERTIARY,
                    activebackground=Theme.BLUE_PRIMARY,
                    highlightthickness=0, showvalue=False,
                )
                slider.pack(fill="x", padx=Theme.S_MD, pady=(0, Theme.S_SM))

            # Action row: Add another, Clear all, Save.
            actions = tk.Frame(win, bg=Theme.BG_OVERLAY)
            actions.pack(fill="x", padx=Theme.S_MD, pady=(Theme.S_SM, Theme.S_MD))

            def _clear_all():
                rects.clear()
                _draw_saved_rects()

            def _save_and_close():
                # Single-rect back-compat field stores the union or the
                # first rect; multi-rect field carries the full list.
                if rects:
                    self.config.subtitle_areas = [tuple(r) for r in rects]
                    self.config.subtitle_area = rects[0]
                    self._update_status(
                        f"Saved {len(rects)} subtitle region{'s' if len(rects) != 1 else ''}",
                        "success",
                    )
                else:
                    self.config.subtitle_areas = None
                    self.config.subtitle_area = None
                    self._update_status("Cleared manual subtitle regions", "info")
                self._update_region_label_display()
                win.destroy()

            ModernButton(actions, text="Clear all", command=_clear_all,
                         style="ghost", size="sm", width=92).pack(side="left")
            ModernButton(actions, text="Save", command=_save_and_close,
                         style="primary", size="sm", width=92).pack(
                             side="right")
            ModernButton(actions, text="Cancel", command=win.destroy,
                         style="ghost", size="sm", width=92).pack(
                             side="right", padx=(0, Theme.S_SM))

            hint_frame = tk.Frame(win, bg=Theme.BG_OVERLAY)
            hint_frame.pack(fill="x", pady=(0, Theme.S_MD))
            tk.Label(hint_frame,
                     text="Drag to add a region. Drag again for multi-region.",
                     font=f(Theme.F_BODY_SM, "bold"),
                     bg=Theme.BG_OVERLAY, fg=Theme.TEXT_PRIMARY).pack()
            tk.Label(hint_frame,
                     text="Scrub the slider to pick a frame where the subtitle is visible.",
                     font=f(Theme.F_META),
                     bg=Theme.BG_OVERLAY, fg=Theme.TEXT_MUTED).pack(pady=(2, 0))

            # The cap must outlive this function so slider scrubs work
            # while the (non-blocking) modal stays open. Release it when
            # the window is closed instead of at function return.
            def _release_cap():
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass

            win.bind("<Destroy>", lambda e: _release_cap() if e.widget is win else None)
            win.bind("<Escape>", lambda e: win.destroy())
            win.transient(self.root)
            win.grab_set()
        except Exception as e:
            logger.error(f"Region selector error: {e}")
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    def _reset_region(self):
        """Reset subtitle region to auto-detect."""
        self.config.subtitle_area = None
        self._update_region_label_display()
        self._update_status("Subtitle detection returned to automatic mode")

    @staticmethod
    def _safe_float(value: str, default: float = 0.0) -> float:
        """Parse a float from a string, returning default on failure."""
        try:
            return float(value or default)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _normalized_path_key(path: str | Path) -> str:
        """Return a case-folded absolute path for reliable Windows comparisons."""
        try:
            return str(Path(path).resolve(strict=False)).casefold()
        except TypeError:
            return str(Path(path).resolve()).casefold()
        except OSError:
            return str(Path(path).absolute()).casefold()

    @staticmethod
    def _new_import_stats() -> dict:
        return {
            "added": 0,
            "duplicate": 0,
            "missing": 0,
            "unsupported": 0,
            "queue_full": 0,
            "folders": 0,
            "supported_in_folders": 0,
        }

    def _merge_import_stats(self, base: dict, extra: dict):
        for key, value in extra.items():
            base[key] = base.get(key, 0) + value

    def _occupied_output_paths(self, exclude_item_id: Optional[str] = None) -> set[str]:
        with self.queue_lock:
            return {
                self._normalized_path_key(item.output_path)
                for item in self.queue
                if item.id != exclude_item_id
            }

    def _make_unique_output_path(self, desired: Path,
                                 exclude_item_id: Optional[str] = None) -> Path:
        """Avoid overwriting existing files or reserved queue outputs."""
        occupied = self._occupied_output_paths(exclude_item_id=exclude_item_id)
        candidate = desired
        counter = 2
        while candidate.exists() or self._normalized_path_key(candidate) in occupied:
            candidate = desired.with_name(f"{desired.stem}({counter}){desired.suffix}")
            counter += 1
        return candidate

    def _suggest_output_path(self, file_path: str, *,
                             output_dir: Optional[Path] = None,
                             exclude_item_id: Optional[str] = None) -> Path:
        input_path = Path(file_path)
        target_dir = output_dir if output_dir is not None else (
            self._output_dir or (input_path.parent / "output")
        )
        desired = target_dir / f"{input_path.stem}_no_sub{input_path.suffix}"
        return self._make_unique_output_path(desired, exclude_item_id=exclude_item_id)

    def _refresh_idle_output_paths(self) -> int:
        """Recompute output paths for idle items that still follow the live output rule."""
        refreshed = 0
        with self.queue_lock:
            idle_items = [
                item for item in self.queue
                if item.status == ProcessingStatus.IDLE and not item.output_path_locked
            ]
        for item in idle_items:
            new_path = self._suggest_output_path(item.file_path, exclude_item_id=item.id)
            if self._normalized_path_key(item.output_path) != self._normalized_path_key(new_path):
                item.output_path = str(new_path)
                refreshed += 1
        return refreshed

    def _announce_import_summary(self, stats: dict):
        """Surface one calm import summary instead of a burst of per-file notices."""
        added = stats.get("added", 0)
        duplicate = stats.get("duplicate", 0)
        missing = stats.get("missing", 0)
        unsupported = stats.get("unsupported", 0)
        queue_full = stats.get("queue_full", 0)
        folders = stats.get("folders", 0)
        supported_in_folders = stats.get("supported_in_folders", 0)

        if added > 0:
            parts = [f"Added {added} item{'s' if added != 1 else ''} to the queue"]
            if duplicate:
                parts.append(f"skipped {duplicate} duplicate{'s' if duplicate != 1 else ''}")
            if queue_full:
                parts.append("queue reached the 500-item limit")
            detail = ". ".join(parts)
            self._update_status(detail, "success")
            logger.info(detail)
            return

        if queue_full:
            self._update_status("The queue is already full (500 items max)", "warning")
            logger.warning("Queue full while importing items")
            return

        if folders and supported_in_folders == 0:
            self._update_status("No supported videos or images were found in the selected folder", "warning")
            logger.warning("No supported files found while importing folder selection")
            return

        if duplicate and not (missing or unsupported):
            self._update_status("Everything selected is already in the queue", "info")
            logger.info("Import skipped because every selected item was already queued")
            return

        if unsupported and not (duplicate or missing):
            self._update_status("Only supported video and image formats can be queued", "warning")
            logger.warning("Import skipped because the selection only contained unsupported files")
            return

        if missing and not (duplicate or unsupported):
            self._update_status("Some selected files could not be found", "warning")
            logger.warning("Import skipped because selected files were missing")
            return

        self._update_status("Nothing new was added to the queue", "warning")
        logger.warning("Import completed without adding new queue items")

    def _on_files_dropped(self, files: List[str]):
        """Handle dropped files."""
        stats = self._new_import_stats()
        for file_path in files:
            if Path(file_path).is_dir():
                self._merge_import_stats(stats, self._add_folder_to_queue(file_path))
            else:
                result = self._add_to_queue(file_path)
                stats[result] = stats.get(result, 0) + 1
        self._announce_import_summary(stats)

    def _add_folder_to_queue(self, folder_path: str):
        """Recursively add all supported files from a folder."""
        folder = Path(folder_path)
        stats = self._new_import_stats()
        stats["folders"] = 1
        for f in sorted(folder.rglob("*")):
            if f.is_file() and (is_video_file(str(f)) or is_image_file(str(f))):
                stats["supported_in_folders"] += 1
                result = self._add_to_queue(str(f))
                stats[result] = stats.get(result, 0) + 1
                if result == "queue_full":
                    break
        return stats

    def _add_to_queue(self, file_path: str):
        """Add a file to the processing queue."""
        # Check file exists and is valid
        if not Path(file_path).is_file():
            logger.warning(f"File not found: {file_path}")
            return "missing"
        if not (is_video_file(file_path) or is_image_file(file_path)):
            logger.warning(f"Unsupported file type: {file_path}")
            return "unsupported"

        # Queue size limit
        if len(self.queue) >= 500:
            logger.warning("Queue full (500 items max)")
            return "queue_full"

        # Prevent duplicate files in queue
        normalized = self._normalized_path_key(file_path)
        with self.queue_lock:
            for existing in self.queue:
                if self._normalized_path_key(existing.file_path) == normalized:
                    logger.info(f"Already in queue: {Path(file_path).name}")
                    return "duplicate"

        # Generate a collision-proof unique ID for this queue slot
        item_id = uuid.uuid4().hex

        # Generate an output path that stays unique against both disk and the
        # rest of the queued items.
        output_path = self._suggest_output_path(file_path)

        # Create config copy from the latest UI state.
        config = self._make_processing_snapshot()

        # Create queue item
        item = QueueItem(
            id=item_id,
            file_path=file_path,
            output_path=str(output_path),
            output_path_locked=False,
            config=config,
            message="Ready to process"
        )

        with self.queue_lock:
            self.queue.append(item)
        self._update_queue_display()
        if len(self.queue) == 1 and not self.is_processing:
            self._show_preview(item)
        logger.info(f"Queued: {Path(file_path).name} ({get_file_info(file_path)})")
        return "added"

    def _open_sort_menu(self):
        """Pop up a themed sort menu anchored to the sort button."""
        if self.is_processing:
            self._update_status(
                "Sorting is disabled while a batch is running", "warning")
            return
        menu = make_themed_menu(self.root)
        menu.add_command(label="Filename (A -> Z)",
                         command=lambda: self._sort_queue("name_asc"))
        menu.add_command(label="Filename (Z -> A)",
                         command=lambda: self._sort_queue("name_desc"))
        menu.add_separator()
        menu.add_command(label="File size (largest first)",
                         command=lambda: self._sort_queue("size_desc"))
        menu.add_command(label="File size (smallest first)",
                         command=lambda: self._sort_queue("size_asc"))
        menu.add_separator()
        menu.add_command(label="Status (pending first)",
                         command=lambda: self._sort_queue("status"))
        menu.add_command(label="Reverse current order",
                         command=lambda: self._sort_queue("reverse"))
        try:
            bx = self._sort_btn.winfo_rootx()
            by = self._sort_btn.winfo_rooty() + self._sort_btn.winfo_height() + 2
            menu.tk_popup(bx, by)
        finally:
            menu.grab_release()

    def _sort_queue(self, strategy: str):
        """Reorder queue items by the chosen strategy and re-render."""
        if self.is_processing:
            return
        key_map = {
            "name_asc": lambda it: Path(it.file_path).name.lower(),
            "name_desc": lambda it: Path(it.file_path).name.lower(),
            "size_asc": lambda it: self._safe_size(it.file_path),
            "size_desc": lambda it: self._safe_size(it.file_path),
            "status": lambda it: {
                ProcessingStatus.IDLE: 0,
                ProcessingStatus.LOADING: 1,
                ProcessingStatus.DETECTING: 2,
                ProcessingStatus.PROCESSING: 3,
                ProcessingStatus.MERGING: 4,
                ProcessingStatus.COMPLETE: 5,
                ProcessingStatus.CANCELLED: 6,
                ProcessingStatus.ERROR: 7,
            }.get(it.status, 99),
        }
        with self.queue_lock:
            if strategy == "reverse":
                self.queue.reverse()
            elif strategy in key_map:
                reverse = strategy.endswith("_desc")
                self.queue.sort(key=key_map[strategy], reverse=reverse)
        # Destroy all widgets so they get rebuilt in new order
        for wid, w in list(self.queue_widgets.items()):
            try:
                w.destroy()
            except Exception:
                pass
        self.queue_widgets.clear()
        self._update_queue_display()
        self._update_status("Queue sorted")

    @staticmethod
    def _safe_size(path: str) -> int:
        try:
            return Path(path).stat().st_size
        except OSError:
            return 0

    def _open_per_file_overrides(self, item_id: str):
        """RM-29: themed popover that edits a single queue item's
        ProcessingConfig without touching the global UI state.

        Only the most-asked fields are surfaced (mode, language,
        sensitivity, output codec). The rest of the config carries over
        from the snapshot taken when the item was queued.
        """
        item = next((it for it in self.queue if it.id == item_id), None)
        if item is None or item.status != ProcessingStatus.IDLE:
            return

        dialog = tk.Toplevel(self.root)
        dialog.title(f"Override settings: {Path(item.file_path).name}")
        dialog.configure(bg=Theme.BG_OVERLAY)
        dialog.resizable(False, False)
        dialog.transient(self.root)

        outer = tk.Frame(dialog, bg=Theme.BORDER, padx=1, pady=1)
        outer.pack()
        body = tk.Frame(outer, bg=Theme.BG_SECONDARY)
        body.pack()

        content = tk.Frame(body, bg=Theme.BG_SECONDARY)
        content.pack(padx=24, pady=(20, 12))

        tk.Label(content, text="Per-file overrides",
                 font=f(Theme.F_HEADING, "bold"),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY).pack(anchor="w")
        tk.Label(content,
                 text="These apply to this queued item only and survive a global settings change.",
                 font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY,
                 wraplength=380, justify="left").pack(anchor="w", pady=(2, Theme.S_LG))

        # Mode picker.
        mode_var = tk.StringVar(value=item.config.mode.value)
        tk.Label(content, text="Mode", font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY).pack(anchor="w")
        mode_picker = SegmentedPicker(
            content,
            options=[(m.value, m.value) for m in InpaintMode],
            value=mode_var.get(),
            command=lambda v: mode_var.set(v),
            bg=Theme.BG_SECONDARY,
        )
        mode_picker.pack(fill="x", pady=(2, Theme.S_MD))

        # Detection language.
        lang_row = tk.Frame(content, bg=Theme.BG_SECONDARY)
        lang_row.pack(fill="x", pady=(0, Theme.S_SM))
        tk.Label(lang_row, text="Subtitle language", font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY).pack(side="left")
        lang_codes = [code for code, _ in self._lang_display]
        lang_var = tk.StringVar(value=item.config.detection_lang)
        lang_combo = ttk.Combobox(
            lang_row, textvariable=lang_var, width=18,
            values=self._lang_labels,
            state="readonly", style="Dark.TCombobox", font=f(Theme.F_BODY_SM),
        )
        # Match the friendly label for the current code.
        for label, (code, _) in zip(self._lang_labels, self._lang_display):
            if code == lang_var.get():
                lang_combo.set(label)
                break
        lang_combo.pack(side="right")

        # Sensitivity slider (1-9 maps to 0.1-0.9).
        sens_row = tk.Frame(content, bg=Theme.BG_SECONDARY)
        sens_row.pack(fill="x", pady=(Theme.S_SM, Theme.S_SM))
        sens_var = tk.IntVar(value=int(round(item.config.detection_threshold * 100)))
        tk.Label(sens_row, text="Sensitivity", font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY).pack(side="left")
        sens_label = tk.Label(sens_row, text=f"{sens_var.get()}%",
                              font=f(Theme.F_BODY_SM, "bold"),
                              bg=Theme.BG_SECONDARY, fg=Theme.BLUE_PRIMARY)
        sens_label.pack(side="right")

        def _on_sens(value):
            try:
                sens_var.set(int(value))
                sens_label.config(text=f"{int(value)}%")
            except (TypeError, ValueError):
                pass

        sens_slider = tk.Scale(
            content, from_=10, to=90, orient="horizontal",
            command=_on_sens, showvalue=False, length=380,
            bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY,
            troughcolor=Theme.BG_TERTIARY,
            activebackground=Theme.BLUE_PRIMARY,
            highlightthickness=0,
        )
        sens_slider.set(sens_var.get())
        sens_slider.pack(fill="x", pady=(0, Theme.S_MD))

        # Output codec.
        codec_row = tk.Frame(content, bg=Theme.BG_SECONDARY)
        codec_row.pack(fill="x", pady=(0, Theme.S_SM))
        tk.Label(codec_row, text="Output codec", font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY).pack(side="left")
        codec_var = tk.StringVar(value=getattr(item.config, "output_codec", "h264"))
        ttk.Combobox(
            codec_row, textvariable=codec_var, width=8,
            values=["h264", "h265", "av1"],
            state="readonly", style="Dark.TCombobox", font=f(Theme.F_BODY_SM),
        ).pack(side="right")

        # Action buttons.
        actions = tk.Frame(body, bg=Theme.BG_CARD)
        actions.pack(fill="x")
        actions_inner = tk.Frame(actions, bg=Theme.BG_CARD)
        actions_inner.pack(side="right", padx=16, pady=14)

        def _save():
            try:
                item.config.mode = InpaintMode(mode_var.get())
            except ValueError:
                pass
            label = lang_combo.get()
            new_code = self._lang_by_label.get(label, item.config.detection_lang)
            item.config.detection_lang = new_code
            item.config.detection_threshold = sens_var.get() / 100.0
            item.config.output_codec = codec_var.get()
            item.config.normalized()
            if item.id in self.queue_widgets:
                self.queue_widgets[item.id].update_item(item)
            self._update_status(
                f"Overrides saved for {Path(item.file_path).name}",
                "success",
            )
            dialog.destroy()

        ModernButton(actions_inner, text="Cancel", command=dialog.destroy,
                     style="ghost", size="md", width=96).pack(side="left")
        ModernButton(actions_inner, text="Save", command=_save,
                     style="primary", size="md", width=96).pack(
                         side="left", padx=(Theme.S_SM, 0))

        dialog.bind("<Escape>", lambda e: dialog.destroy())
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.update_idletasks()
        try:
            px, py = self.root.winfo_rootx(), self.root.winfo_rooty()
            pw, ph = self.root.winfo_width(), self.root.winfo_height()
            dw, dh = dialog.winfo_reqwidth(), dialog.winfo_reqheight()
            dialog.geometry(f"+{px + (pw - dw) // 2}+{py + (ph - dh) // 3}")
        except Exception:
            pass

    def _cancel_queue_item(self, item_id: str):
        """F-7: per-item cancellation. Sets the QueueItem's cancel flag;
        _process_item's progress callback raises InterruptedError next
        time it fires so the worker drops this file and moves on to the
        next one. The global cancel_event stays untouched so the rest
        of the batch survives."""
        with self.queue_lock:
            item = next((it for it in self.queue if it.id == item_id), None)
        if item is None:
            return
        if item.status not in (ProcessingStatus.LOADING,
                                ProcessingStatus.DETECTING,
                                ProcessingStatus.PROCESSING,
                                ProcessingStatus.MERGING):
            self._update_status("That item is not running", "warning")
            return
        item.cancel_requested = True
        self._update_status(
            f"Cancelling {Path(item.file_path).name}", "warning", toast=True,
        )

    def _repeat_item_with_settings(self, item_id: str):
        """RM-28: snapshot the named item's config and re-queue the same
        source file with a fresh IDLE entry. Lets users re-run a clip
        with exactly the per-file settings that already worked once,
        even after the global UI state has changed."""
        with self.queue_lock:
            template = next((it for it in self.queue if it.id == item_id), None)
        if template is None:
            self._update_status("Item not found in queue", "warning")
            return
        # Snapshot via to_dict / from_dict so any future field churn does
        # not need an explicit copy update here.
        snapshot = ProcessingConfig.from_dict(template.config.to_dict())
        # Build a fresh output path (unique vs the original).
        desired = self._suggest_output_path(template.file_path)
        new_item = QueueItem(
            id=str(uuid.uuid4()),
            file_path=template.file_path,
            output_path=str(desired),
            config=snapshot,
            output_path_locked=False,
            status=ProcessingStatus.IDLE,
            progress=0.0,
            message="Ready to process",
        )
        with self.queue_lock:
            self.queue.append(new_item)
        self._update_queue_display()
        self._refresh_action_states()
        self._update_status(
            f"Re-queued {Path(template.file_path).name} with the same settings",
            "info", toast=True,
        )

    def _rename_output_for(self, item_id: str):
        """Open a file picker to customize the output path of a queued item.

        Disabled for items that have already started processing.
        """
        item = next((i for i in self.queue if i.id == item_id), None)
        if not item:
            return
        if item.status != ProcessingStatus.IDLE:
            self._update_status(
                "Only idle items can have their output renamed", "warning")
            return

        current = Path(item.output_path)
        suffix = current.suffix or Path(item.file_path).suffix
        ext_star = f"*{suffix}" if suffix else "*.*"
        new_path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Choose an output path",
            initialdir=str(current.parent),
            initialfile=current.name,
            defaultextension=suffix,
            filetypes=[("Keep extension", ext_star), ("All files", "*.*")],
        )
        if not new_path:
            return

        resolved_path = self._make_unique_output_path(
            Path(new_path),
            exclude_item_id=item.id,
        )
        item.output_path = str(resolved_path)
        item.output_path_locked = True
        if item.id in self.queue_widgets:
            self.queue_widgets[item.id].update_item(item)
        if self._normalized_path_key(new_path) != self._normalized_path_key(resolved_path):
            self._update_status(
                f"Output renamed to {resolved_path.name} to avoid an overwrite",
                "success",
            )
        else:
            self._update_status(
                f"Output renamed to {resolved_path.name}", "success")

    def _remove_from_queue(self, item_id: str):
        """Remove an item from the queue."""
        with self.queue_lock:
            # Don't remove items that are currently being processed
            item = next((i for i in self.queue if i.id == item_id), None)
            if item and item.status in (ProcessingStatus.LOADING, ProcessingStatus.DETECTING,
                                         ProcessingStatus.PROCESSING, ProcessingStatus.MERGING):
                self._update_status("Wait for the active item to finish before removing it", "warning")
                return
            self.queue = [i for i in self.queue if i.id != item_id]
        if self._selected_queue_item_id == item_id:
            self._selected_queue_item_id = None
        self._update_queue_display()
        if item:
            self._update_status(f"Removed {Path(item.file_path).name} from the queue")

    def _clear_queue(self):
        """Clear all items from the queue."""
        if self.is_processing:
            self._update_status("Stop the batch before clearing the queue", "warning")
            return
        if self.queue:
            n = len(self.queue)
            if not show_confirm(
                self.root,
                title="Clear the queue?",
                message=f"Remove {n} item{'s' if n != 1 else ''} from the batch.",
                detail="Completed outputs on disk are not deleted.",
                confirm_label="Clear queue",
                cancel_label="Keep",
                tone="danger",
            ):
                return

        with self.queue_lock:
            self.queue.clear()
        self._selected_queue_item_id = None
        self._update_queue_display()
        self._update_status("Queue cleared")

    def _update_queue_display(self):
        """Update the queue display. Only rebuilds widgets that changed."""
        with self.queue_lock:
            current_ids = {item.id for item in self.queue}

        # Remove widgets for items no longer in queue
        stale_ids = [wid for wid in self.queue_widgets if wid not in current_ids]
        for wid in stale_ids:
            self.queue_widgets[wid].destroy()
            del self.queue_widgets[wid]

        # Update count + stat chips
        total = len(self.queue)
        self.queue_count.config(text=f"{total} item{'s' if total != 1 else ''}")
        done = sum(1 for i in self.queue if i.status == ProcessingStatus.COMPLETE)
        err = sum(1 for i in self.queue
                  if i.status in (ProcessingStatus.ERROR, ProcessingStatus.CANCELLED))
        if done > 0:
            self.queue_done_lbl.config(text=f"{done} done")
            self.queue_done_pill.pack(side="left", padx=(Theme.S_XS, 0))
        else:
            self.queue_done_pill.pack_forget()
        if err > 0:
            self.queue_err_lbl.config(text=f"{err} failed")
            self.queue_err_pill.pack(side="left", padx=(Theme.S_XS, 0))
        else:
            self.queue_err_pill.pack_forget()
        # Sort button visibility
        try:
            if total >= 3:
                self._sort_btn.pack(side="left", padx=(Theme.S_SM, 0))
            else:
                self._sort_btn.pack_forget()
        except Exception:
            pass

        if not self.queue:
            # Clear any remaining children and show empty state
            for widget in self.queue_frame.winfo_children():
                widget.destroy()
            self.queue_widgets.clear()
            self._hide_filter_empty_state()
            self._build_queue_empty_state()
            self._set_preview_placeholder(
                "Preview a sample frame",
                "Select a queued item to inspect it before processing. Review mask is the fastest way to confirm the subtitle region.",
            )
        else:
            # Remove empty label if present
            for child in self.queue_frame.winfo_children():
                if child not in self.queue_widgets.values():
                    child.destroy()

            # Add widgets for new items only
            for item in self.queue:
                if item.id not in self.queue_widgets:
                    widget = QueueItemWidget(self.queue_frame, item, self._remove_from_queue,
                                             on_select=self._show_preview,
                                             on_rename=self._rename_output_for,
                                             on_repeat=self._repeat_item_with_settings,
                                             on_cancel_item=self._cancel_queue_item,
                                             on_override=self._open_per_file_overrides)
                    widget.pack(fill="x", pady=(0, 8))
                    self.queue_widgets[item.id] = widget
                    # Forward mousewheel to queue canvas
                    widget.bind("<MouseWheel>", self._on_mousewheel)
                    for child in widget.winfo_children():
                        child.bind("<MouseWheel>", self._on_mousewheel)
                        for subchild in child.winfo_children():
                            subchild.bind("<MouseWheel>", self._on_mousewheel)
                else:
                    self.queue_widgets[item.id].update_item(item)

        if self._selected_queue_item_id and self._selected_queue_item_id in self.queue_widgets:
            self._set_selected_queue_item(self._selected_queue_item_id)
        else:
            self._set_selected_queue_item(None)
        self._refresh_action_states()
        # Show filter only when the queue is long enough to justify it
        try:
            if len(self.queue) >= 6:
                self._queue_filter_frame.pack(
                    fill="x", padx=Theme.S_XL, pady=(0, Theme.S_SM),
                    before=self._queue_container)
            else:
                self._queue_filter_frame.pack_forget()
                if self._queue_filter_var.get():
                    self._queue_filter_var.set("")
        except Exception:
            pass
        # Re-apply any active filter so newly added items get filtered too
        if self._queue_filter_var.get():
            self._apply_queue_filter()

    def _apply_queue_filter(self):
        """Hide/show queue widgets whose filename doesn't match the filter."""
        query = (self._queue_filter_var.get() or "").strip().lower()
        visible = 0
        total = len(self.queue)
        for item in self.queue:
            widget = self.queue_widgets.get(item.id)
            if not widget:
                continue
            fname = Path(item.file_path).name.lower()
            match = (query in fname) or (query in item.file_path.lower())
            if not query or match:
                if not widget.winfo_ismapped():
                    widget.pack(fill="x", pady=(0, Theme.S_SM))
                visible += 1
            else:
                widget.pack_forget()
        if query:
            self.queue_count.config(text=f"{visible} of {total} shown")
        else:
            self.queue_count.config(text=f"{total} item{'s' if total != 1 else ''}")

        if query and total and visible == 0:
            self._ensure_filter_empty_state()
            self._filter_empty_title.config(
                text=f'No items match "{truncate_middle(query, 28)}"')
            self._filter_empty_body.config(
                text="Try a shorter filename search, or clear the filter to see the full batch again.")
            if not self._filter_empty_container.winfo_ismapped():
                self._filter_empty_container.pack(
                    pady=(Theme.S_3XL, Theme.S_LG), fill="x")
        else:
            self._hide_filter_empty_state()

    def _update_status(self, message: str, tone: str = "neutral", toast: bool = False):
        """Update the footer status dot + message.

        If `toast=True`, also surface as a transient toast in the bottom-right.
        """
        colors = {
            "neutral": Theme.TEXT_SECONDARY,
            "success": Theme.SUCCESS,
            "warning": Theme.WARNING,
            "error": Theme.ERROR,
            "info": Theme.INFO,
        }
        color = colors.get(tone, Theme.TEXT_SECONDARY)
        self.status_label.config(text=message, fg=color)
        try:
            self.status_dot.itemconfig(self._status_dot_item, fill=color)
        except Exception:
            pass
        self._status_tone = tone
        if toast:
            try:
                Toast.show(self.root, message, tone=tone)
            except Exception:
                pass

    def _open_output_folder(self):
        """Open the output folder for the most recently completed item."""
        selected = self._get_selected_queue_item()
        if selected and selected.status == ProcessingStatus.COMPLETE and Path(selected.output_path).exists():
            target = selected
        else:
            completed = [i for i in self.queue if i.status == ProcessingStatus.COMPLETE]
            target = completed[-1] if completed else None
        if target:
            output_dir = str(Path(target.output_path).parent)
            try:
                os.startfile(output_dir)
                self._update_status("Opened the output folder", "info")
            except Exception:
                logger.warning(f"Could not open folder: {output_dir}")
        else:
            self._update_status("No completed results are available yet", "warning")

    def _show_preview(self, item: QueueItem, show_mask: bool = False):
        """Show thumbnail preview. Side-by-side before/after for completed items.
        If show_mask=True, run detection and overlay red boxes on the frame."""
        self._preview_request_id += 1
        preview_request_id = self._preview_request_id
        # Any switch cancels a running throbber so it can't overwrite later UI
        if not show_mask:
            self._stop_throbber()
        self._set_selected_queue_item(item.id)
        if not PIL_AVAILABLE:
            self.preview_title_label.config(text="Preview unavailable")
            self.preview_meta_label.config(text="Install Pillow to enable image previews.")
            self._preview_label.config(text="Install Pillow for previews", image="")
            return

        try:
            import cv2 as _cv2

            def load_first_frame_raw(path):
                """Load first frame as BGR numpy array."""
                if is_image_file(path):
                    return _cv2.imread(path)
                elif is_video_file(path):
                    cap = _cv2.VideoCapture(path)
                    try:
                        ret, frame = cap.read()
                        return frame if ret else None
                    finally:
                        cap.release()
                return None

            def to_pil(bgr_frame):
                return Image.fromarray(_cv2.cvtColor(bgr_frame, _cv2.COLOR_BGR2RGB))

            raw_frame = load_first_frame_raw(item.file_path)
            if raw_frame is None:
                self.preview_title_label.config(text="Preview unavailable")
                self.preview_meta_label.config(text="The selected file could not be read for preview.")
                self._preview_label.config(text="Could not read file", image="")
                return

            badge = status_ui(item.status)
            self.preview_status_chip.config(text=badge["label"], fg=badge["color"], bg=badge["bg"])

            try:
                max_w = max(220, self._preview_frame.winfo_width() - 36)
            except Exception:
                max_w = 390
            max_h = 158

            # Mask preview mode -- run detection in background thread
            if show_mask:
                self.preview_title_label.config(text=f"Detecting {Path(item.file_path).name}")
                self.preview_meta_label.config(
                    text="Running detection on the first frame..."
                )
                # Clear any existing preview image, then start animated throbber
                self._preview_label.config(image="", text="")
                self._preview_photo = None
                self._start_throbber()
                self._preview_label.update_idletasks()
                frame_copy = raw_frame.copy()
                lang = self.lang_var.get()
                threshold = getattr(self.config, '_detection_threshold_pct', 50) / 100.0
                sub_area = self.config.subtitle_area

                def _detect_bg():
                    try:
                        from backend.processor import SubtitleDetector
                        # Reuse cached detector if lang hasn't changed
                        if self._preview_detector is None or self._preview_detector_lang != lang:
                            self._preview_detector = SubtitleDetector(lang=lang)
                            self._preview_detector_lang = lang
                        det = self._preview_detector
                        if sub_area:
                            boxes = [sub_area]
                        else:
                            boxes = det.detect(frame_copy, threshold)
                        vis = frame_copy.copy()
                        for (bx1, by1, bx2, by2) in boxes:
                            _cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                        img = to_pil(vis)
                        img.thumbnail((max_w, max_h), Image.LANCZOS)
                        engine = det._engine_name
                        n = len(boxes)
                        def _update_ui():
                            if (preview_request_id != self._preview_request_id
                                    or self._selected_queue_item_id != item.id):
                                return
                            self._stop_throbber()
                            self._preview_photo = ImageTk.PhotoImage(img)
                            self.preview_title_label.config(text=f"Detection mask for {Path(item.file_path).name}")
                            if sub_area:
                                meta = "Manual region applied. Detection used your saved subtitle band."
                            elif n:
                                meta = f"{engine} found {n} region{'s' if n != 1 else ''} on the first frame."
                            else:
                                meta = ("No regions were found on the first frame. Try Set region, or lower the "
                                        "Threshold in detailed controls.")
                            self.preview_meta_label.config(text=meta)
                            self._preview_label.config(
                                image=self._preview_photo,
                                text=f"{engine}: {n} detected" if n else "No text detected")
                        self.root.after(0, _update_ui)
                    except Exception as exc:
                        def _show_error():
                            if (preview_request_id != self._preview_request_id
                                    or self._selected_queue_item_id != item.id):
                                return
                            self._stop_throbber()
                            self.preview_title_label.config(text="Detection preview failed")
                            self.preview_meta_label.config(text="The detection preview could not be generated.")
                            self._preview_label.config(text=f"Detection error: {exc}", image="")
                        self.root.after(0, _show_error)

                threading.Thread(target=_detect_bg, daemon=True).start()
                return

            input_img = to_pil(raw_frame)

            # Check if completed and output exists -- show before/after
            output_img = None
            if item.status == ProcessingStatus.COMPLETE and Path(item.output_path).exists():
                out_frame = load_first_frame_raw(item.output_path)
                if out_frame is not None:
                    output_img = to_pil(out_frame)

            if output_img:
                half_w = max_w // 2 - 2
                input_img.thumbnail((half_w, max_h), Image.LANCZOS)
                output_img.thumbnail((half_w, max_h), Image.LANCZOS)
                total_w = input_img.width + output_img.width + 4
                total_h = max(input_img.height, output_img.height)
                composite = Image.new("RGB", (total_w, total_h), (15, 23, 42))
                composite.paste(input_img, (0, 0))
                composite.paste(output_img, (input_img.width + 4, 0))
                draw = ImageDraw.Draw(composite)
                draw.line([(input_img.width + 1, 0), (input_img.width + 1, total_h)],
                          fill="#22c55e", width=2)
                draw.rectangle((10, 10, 82, 28), fill=self._hex_to_rgb(Theme.BG_TERTIARY))
                draw.text((18, 14), "Source", fill=self._hex_to_rgb(Theme.TEXT_SECONDARY))
                draw.rectangle((input_img.width + 16, 10, input_img.width + 96, 28),
                               fill=self._hex_to_rgb(Theme.SUCCESS_BG))
                draw.text((input_img.width + 24, 14), "Cleaned",
                          fill=self._hex_to_rgb(Theme.SUCCESS))
                self._preview_photo = ImageTk.PhotoImage(composite)
                self.preview_title_label.config(text=f"Before / after for {Path(item.file_path).name}")
                meta = ("Completed items show the source frame beside the cleaned result so you can "
                        "spot-check the cleanup immediately.")
                quality_note = format_quality_report(item.quality_report)
                if quality_note:
                    meta += f" Quality check: {quality_note}."
                self.preview_meta_label.config(text=meta)
                self._preview_label.config(image=self._preview_photo, text="")
            else:
                input_img.thumbnail((max_w, max_h), Image.LANCZOS)
                self._preview_photo = ImageTk.PhotoImage(input_img)
                self.preview_title_label.config(text=f"Source frame for {Path(item.file_path).name}")
                self.preview_meta_label.config(
                    text="Review mask to confirm the subtitle band, then start the batch when the framing looks right."
                )
                self._preview_label.config(image=self._preview_photo, text="")
        except Exception as e:
            self.preview_title_label.config(text="Preview unavailable")
            self.preview_meta_label.config(text="An unexpected preview error occurred.")
            self._preview_label.config(text=f"Preview error: {e}", image="")

    def _retry_failed(self):
        """Reset failed/cancelled items so they can be reprocessed."""
        if self.is_processing:
            self._update_status("Stop the active batch before retrying failed items", "warning")
            return
        count = 0
        with self.queue_lock:
            for item in self.queue:
                if item.status in (ProcessingStatus.ERROR, ProcessingStatus.CANCELLED):
                    item.status = ProcessingStatus.IDLE
                    item.progress = 0.0
                    item.message = "Ready to retry"
                    item.error = None
                    item.quality_report = None
                    item.started_at = None
                    item.completed_at = None
                    count += 1
        if count:
            self._update_queue_display()
            # Force-refresh all widgets to show reset state
            for item in self.queue:
                if item.message == "Ready to retry" and item.id in self.queue_widgets:
                    self.queue_widgets[item.id].update_item(item)
            self._update_status(f"Reset {count} item{'s' if count != 1 else ''} for retry", "success")
        else:
            self._update_status("There are no failed items to retry", "warning")

    def _set_settings_locked(self, locked: bool):
        """Lock or unlock settings controls during processing."""
        entry_state = "disabled" if locked else "normal"
        combo_state = "disabled" if locked else "readonly"
        try:
            # Custom toggles
            self.skip_check.set_enabled(not locked)
            self.lama_check.set_enabled(not locked)
            self.preserve_audio_check.set_enabled(not locked)
            self.hw_encode_check.set_enabled(not locked)

            self.lang_combo.config(state=combo_state)
            if hasattr(self, 'gpu_combo'):
                self.gpu_combo.config(state=combo_state)
            self.time_start_entry.config(state=entry_state)
            self.time_end_entry.config(state=entry_state)

            self.region_btn.set_enabled(not locked)
            self.region_reset_btn.set_enabled(
                (not locked) and self.config.subtitle_area is not None)
            self.adv_toggle.set_enabled(not locked)
            # Segmented algo picker: dim/undim each segment
            try:
                for seg in self.mode_picker._segments.values():
                    seg.config(state="disabled" if locked else "normal")
            except Exception:
                pass
        except Exception:
            pass

        # Re-apply mode-specific toggle availability
        if not locked:
            try:
                self._update_mode_options()
            except Exception:
                pass

    def _start_processing(self):
        """Start processing the queue."""
        if not self.queue:
            self._update_status("Add media to the queue before starting a batch", "warning")
            return

        active_thread = self._has_active_processing_thread()
        batch_busy = self.is_processing or active_thread
        if batch_busy:
            if self._stop_requested or self.cancel_event.is_set():
                self._update_status(
                    "Batch is already stopping. Please wait for the current item to wrap up.",
                    "warning",
                )
                return
            if active_thread:
                self._stop_processing()
            else:
                self._update_status("Finalizing the previous batch...", "info")
            return

        self._apply_current_settings_to_idle_items()
        if self.preserve_audio_var.get() and not self.ffmpeg_ready:
            has_video = any(is_video_file(item.file_path) for item in self.queue)
            if has_video:
                self._update_status(
                    "FFmpeg is missing, so video outputs will be saved without original audio.",
                    "warning",
                    toast=True,
                )

        self.is_processing = True
        self._stop_requested = False
        self.cancel_event.clear()
        self._set_settings_locked(True)
        self.start_btn.set_style("danger")
        self.start_btn.icon = "x"
        self.start_btn.set_text("Stop batch")
        self._batch_times = []
        # F-9: probe the first queued video so the very first ETA tick
        # has a real number instead of "" until the first item finishes.
        try:
            self._probe_eta_seconds = self._probe_batch_eta()
        except Exception:
            self._probe_eta_seconds = 0.0
        self._batch_started_at = datetime.now()
        self._refresh_action_states()
        self._update_status("Batch processing started", "info")
        # Kick off Windows taskbar progress in indeterminate until first tick
        self._ensure_taskbar()
        if self._taskbar:
            self._taskbar.set_state(TaskbarProgress.STATE_INDETERMINATE)

        # Start elapsed timer
        self._start_elapsed_timer()

        # Start processing thread
        self._processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._processing_thread.start()

    def _stop_processing(self):
        """Stop the current processing."""
        if self._stop_requested:
            self._update_status("Batch is already stopping...", "warning")
            return
        self._stop_requested = True
        self.cancel_event.set()
        # Invalidate the cached remover so the next batch re-initialises with
        # fresh state. A cancelled run may have left detector / inpainter /
        # SRT buffers in an intermediate state.
        self._cached_remover = None
        self._cached_remover_key = None

        self.start_btn.set_style("primary")
        self.start_btn.icon = "x"
        self.start_btn.set_text("Stopping...")
        self._refresh_action_states()
        self._update_status(
            "Stopping after the current step. Finished outputs stay on disk.",
            "warning",
        )
        if self._taskbar:
            self._taskbar.set_state(TaskbarProgress.STATE_PAUSED)

    def _has_active_processing_thread(self) -> bool:
        return self._processing_thread is not None and self._processing_thread.is_alive()

    def _start_elapsed_timer(self):
        """Start a timer that updates elapsed times on in-progress queue items."""
        # Cancel any existing timer before starting a new one to avoid
        # stacking multiple concurrent tick loops.
        self._stop_elapsed_timer()
        def tick():
            if not self.is_processing:
                return
            try:
                for widget in list(self.queue_widgets.values()):
                    if widget.item.started_at and not widget.item.completed_at:
                        elapsed = (datetime.now() - widget.item.started_at).total_seconds()
                        widget.time_label.config(text=format_time(elapsed))
            except Exception:
                pass
            self._elapsed_timer_id = self.root.after(1000, tick)
        self._elapsed_timer_id = self.root.after(1000, tick)

    def _stop_elapsed_timer(self):
        if self._elapsed_timer_id:
            self.root.after_cancel(self._elapsed_timer_id)
            self._elapsed_timer_id = None

    def _process_queue(self):
        """Process all items in the queue."""
        with self.queue_lock:
            items_to_process = [i for i in self.queue
                                if i.status not in (ProcessingStatus.COMPLETE,
                                                     ProcessingStatus.ERROR,
                                                     ProcessingStatus.CANCELLED)]

        total = len(items_to_process)
        for idx, item in enumerate(items_to_process):
            if self.cancel_event.is_set():
                # Mark ALL remaining items as cancelled
                now = datetime.now()
                for remaining in items_to_process[idx:]:
                    remaining.status = ProcessingStatus.CANCELLED
                    remaining.message = "Cancelled"
                    remaining.completed_at = now
                    self._update_item_display(remaining)
                break

            # Update batch progress + window title
            try:
                self.root.after(0, self._update_batch_progress, idx, total)
            except RuntimeError:
                return  # root destroyed during shutdown
            self._process_item(item)

        # Final batch state
        try:
            self.root.after(0, self._update_batch_progress, total, total)
            self.root.after(0, self._on_processing_complete)
        except RuntimeError:
            pass  # root destroyed during shutdown

    def _process_item(self, item: QueueItem):
        """Process a single queue item using the backend processor."""
        try:
            item.status = ProcessingStatus.LOADING
            item.started_at = datetime.now()
            item.completed_at = None
            item.progress = 0.0
            item.message = "Initializing..."
            item.error = None
            item.quality_report = None
            item.cancel_requested = False  # F-7 reset on fresh attempt
            self._update_item_display(item)

            from backend.processor import (
                SubtitleRemover as BackendRemover,
                ProcessingConfig as BackendConfig,
                InpaintMode as BackendInpaintMode,
            )

            # Map GUI enum values to backend enum values
            mode_map = {
                "Auto": BackendInpaintMode.AUTO,
                "STTN": BackendInpaintMode.STTN,
                "LAMA": BackendInpaintMode.LAMA,
                "ProPainter": BackendInpaintMode.PROPAINTER,
            }

            # Determine device string based on GPU type
            if item.config.use_gpu:
                gpu_type = None
                for g in self.gpus:
                    if g['index'] == item.config.gpu_id:
                        gpu_type = g.get('type')
                        break
                if gpu_type == "DirectML":
                    device = "directml"
                else:
                    device = f"cuda:{item.config.gpu_id}"
            else:
                device = "cpu"

            backend_mode = mode_map.get(item.config.mode.value, BackendInpaintMode.STTN)
            lang = getattr(item.config, 'detection_lang', 'en')
            vertical = bool(getattr(item.config, 'detection_vertical', False))
            cache_key = (backend_mode, device, lang, vertical)

            backend_config = BackendConfig(
                mode=backend_mode,
                device=device,
                sttn_skip_detection=item.config.sttn_skip_detection,
                sttn_neighbor_stride=item.config.sttn_neighbor_stride,
                sttn_reference_length=item.config.sttn_reference_length,
                sttn_max_load_num=item.config.sttn_max_load_num,
                lama_super_fast=item.config.lama_super_fast,
                preserve_audio=item.config.preserve_audio,
                output_quality=item.config.output_quality,
                detection_lang=lang,
                detection_threshold=getattr(item.config, 'detection_threshold', 0.5),
                detection_vertical=getattr(item.config, 'detection_vertical', False),
                whisper_fallback=getattr(item.config, 'whisper_fallback', False),
                whisper_model_size=getattr(item.config, 'whisper_model_size', 'tiny'),
                upscale_factor=getattr(item.config, 'upscale_factor', 0),
                film_grain_strength=getattr(item.config, 'film_grain_strength', 0.0),
                swinir_restore=getattr(item.config, 'swinir_restore', False),
                seedvr2_restore=getattr(item.config, 'seedvr2_restore', False),
                preserve_color_metadata=getattr(item.config, 'preserve_color_metadata', True),
                nle_sidecar=getattr(item.config, 'nle_sidecar', 'off'),
                subtitle_area=item.config.subtitle_area,
                time_start=getattr(item.config, 'time_start', 0.0),
                time_end=getattr(item.config, 'time_end', 0.0),
                detection_frame_skip=getattr(item.config, 'detection_frame_skip', 0),
                mask_dilate_px=getattr(item.config, 'mask_dilate_px', 8),
                mask_feather_px=getattr(item.config, 'mask_feather_px', 4),
                tbe_enable=getattr(item.config, 'tbe_enable', True),
                tbe_min_coverage=getattr(item.config, 'tbe_min_coverage', 3),
                tbe_use_median=getattr(item.config, 'tbe_use_median', True),
                tbe_flow_warp=getattr(item.config, 'tbe_flow_warp', False),
                tbe_scene_cut_split=getattr(item.config, 'tbe_scene_cut_split', True),
                tbe_scene_cut_threshold=getattr(item.config, 'tbe_scene_cut_threshold', 0.35),
                tbe_scene_cut_use_pyscenedetect=getattr(item.config, 'tbe_scene_cut_use_pyscenedetect', False),
                tbe_scene_cut_use_transnetv2=getattr(item.config, 'tbe_scene_cut_use_transnetv2', False),
                detection_denoise=getattr(item.config, 'detection_denoise', False),
                sam2_refine=getattr(item.config, 'sam2_refine', False),
                edge_ring_px=getattr(item.config, 'edge_ring_px', 2),
                subtitle_areas=getattr(item.config, 'subtitle_areas', None),
                export_srt=getattr(item.config, 'export_srt', False),
                export_mask_video=getattr(item.config, 'export_mask_video', False),
                adaptive_batch=getattr(item.config, 'adaptive_batch', True),
                auto_exposure_threshold=getattr(item.config, 'auto_exposure_threshold', 0.55),
                deinterlace=getattr(item.config, 'deinterlace', False),
                deinterlace_auto=getattr(item.config, 'deinterlace_auto', True),
                keyframe_detection=getattr(item.config, 'keyframe_detection', False),
                quality_report=getattr(item.config, 'quality_report', False),
                kalman_tracking=getattr(item.config, 'kalman_tracking', True),
                kalman_iou_threshold=getattr(item.config, 'kalman_iou_threshold', 0.3),
                kalman_max_age=getattr(item.config, 'kalman_max_age', 2),
                phash_skip_enable=getattr(item.config, 'phash_skip_enable', True),
                phash_skip_distance=getattr(item.config, 'phash_skip_distance', 4),
                colour_tune_enable=getattr(item.config, 'colour_tune_enable', False),
                colour_tune_tolerance=getattr(item.config, 'colour_tune_tolerance', 25),
                use_hw_encode=getattr(item.config, 'use_hw_encode', True),
                output_codec=getattr(item.config, 'output_codec', 'h264'),
                # v3.13 GUI-exposed fields: previously CLI-only, now plumbed
                # through so a GUI user can drive every backend feature.
                loudnorm_target=getattr(item.config, 'loudnorm_target', 0.0),
                multi_audio_passthrough=getattr(item.config, 'multi_audio_passthrough', True),
                decode_hw_accel=getattr(item.config, 'decode_hw_accel', 'off'),
                prefetch_decode=getattr(item.config, 'prefetch_decode', True),
                prefetch_queue_size=getattr(item.config, 'prefetch_queue_size', 0),
                input_fps=getattr(item.config, 'input_fps', 24.0),
                quality_report_sheet=getattr(item.config, 'quality_report_sheet', False),
                remove_subtitles=getattr(item.config, 'remove_subtitles', True),
                remove_chyrons=getattr(item.config, 'remove_chyrons', True),
                chyron_min_hits=getattr(item.config, 'chyron_min_hits', 90),
                karaoke_grouping=getattr(item.config, 'karaoke_grouping', False),
                karaoke_x_gap_px=getattr(item.config, 'karaoke_x_gap_px', 20),
                karaoke_y_overlap=getattr(item.config, 'karaoke_y_overlap', 0.5),
            )

            # Auto subtitle-band detection -- run before the main pass so we
            # can pin the dominant band once per file. Cheap (30-frame probe).
            if getattr(item.config, 'auto_band', False) and not item.config.subtitle_area:
                try:
                    # Use a minimal config just for the band probe
                    probe_cfg = BackendConfig(
                        mode=backend_mode,
                        device=device,
                        detection_lang=lang,
                        detection_threshold=getattr(item.config, 'detection_threshold', 0.5),
                    )
                    probe = BackendRemover(probe_cfg)
                    band = probe.detect_subtitle_band(item.file_path, probe_frames=30)
                    if band:
                        backend_config.subtitle_area = band
                        logger.info(f"Auto-band: {band} for {Path(item.file_path).name}")
                except Exception as exc:
                    logger.warning(f"Auto-band detection failed: {exc}")

            # Reuse cached remover if mode/device/lang match (avoids reloading
            # OCR models and re-probing HW encoders for every queue item).
            # The constructor normalises the config; on hot-swap we re-run
            # normalisation explicitly so a NaN/inf/out-of-range value from a
            # bad per-item override cannot reach the pipeline.
            if self._cached_remover is not None and self._cached_remover_key == cache_key:
                remover = self._cached_remover
                from backend.processor import normalize_processing_config as _normalize_backend_config
                remover.config = _normalize_backend_config(backend_config)
            else:
                remover = BackendRemover(backend_config)
                self._cached_remover = remover
                self._cached_remover_key = cache_key
            if hasattr(remover, "last_quality_report"):
                remover.last_quality_report = None

            def on_progress(progress: float, message: str):
                if self.cancel_event.is_set():
                    raise InterruptedError("Processing cancelled")
                # F-7: per-item cancel raises the same exception so
                # process_video bails on THIS file; the outer
                # _process_queue loop then advances to the next item
                # because cancel_event was never set.
                if getattr(item, "cancel_requested", False):
                    raise InterruptedError("Item cancelled by user")
                # Map backend progress to GUI status
                if progress < 0.3:
                    item.status = ProcessingStatus.DETECTING
                elif progress < 0.9:
                    item.status = ProcessingStatus.PROCESSING
                elif progress < 1.0:
                    item.status = ProcessingStatus.MERGING
                else:
                    item.status = ProcessingStatus.COMPLETE
                item.progress = progress
                item.message = message
                self._update_item_display(item)

            remover.on_progress = on_progress

            # Live preview: pipe the latest inpainted frame into the preview
            # pane. The backend emits frames on its worker thread, so we
            # marshal to the Tk main loop via root.after.
            #
            # EI-4: also throttle on wall-clock so the worker does not
            # queue PIL conversions faster than the Tk thread can absorb
            # ImageTk.PhotoImage calls (~50 ms on 4K). The receiver still
            # throttles to ~15 FPS, but throttling in the worker too
            # avoids burning CPU on conversions that get dropped.
            preview_throttle_state = {"last_ts": 0.0}
            def on_preview_frame(frame, cur_idx, total):
                if self.cancel_event.is_set():
                    return
                now = time.monotonic()
                if (now - preview_throttle_state["last_ts"]) < (1.0 / 15.0):
                    return
                preview_throttle_state["last_ts"] = now
                try:
                    max_w, max_h = 520, 320
                    h, w = frame.shape[:2]
                    scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
                    if scale < 1.0:
                        new_w = max(1, int(w * scale))
                        new_h = max(1, int(h * scale))
                        import cv2 as _cv2_live
                        small = _cv2_live.resize(frame, (new_w, new_h),
                                                  interpolation=_cv2_live.INTER_AREA)
                    else:
                        small = frame
                    rgb = small[..., ::-1]  # BGR -> RGB
                    from PIL import Image as _Image
                    pil = _Image.fromarray(rgb)
                    self.root.after(0, self._push_live_preview, pil, cur_idx, total,
                                     Path(item.file_path).name)
                except Exception:
                    pass

            remover.on_preview_frame = on_preview_frame

            # Ensure output directory exists
            Path(item.output_path).parent.mkdir(parents=True, exist_ok=True)

            # Run the actual processing
            file_name = Path(item.file_path).name
            logger.info(f"Processing: {file_name} with {item.config.mode.value}")

            if is_video_file(item.file_path):
                success = remover.process_video(item.file_path, item.output_path)
            elif is_image_file(item.file_path):
                success = remover.process_image(item.file_path, item.output_path)
            else:
                raise ValueError(f"Unsupported file type: {Path(item.file_path).suffix}")

            if success:
                item.status = ProcessingStatus.COMPLETE
                item.progress = 1.0
                item.error = None
                item.quality_report = getattr(remover, "last_quality_report", None)
                item.message = "Complete!"
                quality_note = format_quality_report(item.quality_report, compact=True)
                if quality_note:
                    item.message = f"Complete - {quality_note}"
                item.completed_at = datetime.now()
                elapsed = (item.completed_at - item.started_at).total_seconds()
                # Track for ETA rolling average
                self._batch_times.append(elapsed)
                logger.info(f"Completed: {file_name} in {format_time(elapsed)}")
            else:
                item.status = ProcessingStatus.ERROR
                item.message = "Processing failed"
                item.quality_report = None
                item.completed_at = datetime.now()
                logger.error(f"Failed: {file_name}")
            self._update_item_display(item)

        except InterruptedError:
            item.status = ProcessingStatus.CANCELLED
            item.message = "Cancelled"
            item.error = None
            item.quality_report = None
            item.completed_at = datetime.now()
            self._update_item_display(item)
            logger.info(f"Cancelled: {Path(item.file_path).name}")
        except Exception as e:
            item.status = ProcessingStatus.ERROR
            item.error = str(e)
            item.message = f"Error: {str(e)}"
            item.quality_report = None
            item.completed_at = datetime.now()
            self._update_item_display(item)
            logger.error(f"Processing error for {item.file_path}: {e}")

    def _ensure_taskbar(self):
        """Lazily create the Windows taskbar progress client once the window
        is fully realized."""
        if self._taskbar is not None:
            return
        try:
            hwnd = self.root.winfo_id()
            # Walk up to the top-level window (important on some tk builds)
            import ctypes
            hwnd = ctypes.windll.user32.GetParent(hwnd) or hwnd
            self._taskbar = TaskbarProgress(hwnd)
        except Exception:
            self._taskbar = None

    def _compute_eta(self, current: int, total: int) -> str:
        """Estimate time-remaining based on rolling average per-item time.

        F-9: when no items have completed yet we fall back to the
        pre-batch probe estimate (`_probe_eta_seconds`) so users get a
        sensible "about X left" line from the very first frame instead
        of an empty string until the first item finishes.
        """
        remaining = total - current
        if remaining <= 0:
            return ""
        if self._batch_times:
            recent = self._batch_times[-5:]
            avg = sum(recent) / len(recent)
            eta_seconds = avg * remaining
            return format_time(eta_seconds)
        probe = getattr(self, "_probe_eta_seconds", 0.0) or 0.0
        if probe > 0:
            return format_time(probe * remaining) + " (estimated)"
        return ""

    def _probe_batch_eta(self) -> float:
        """F-9: cheap pre-batch ETA probe. Reads a 30-frame slice from
        the first queued video, runs detect + inpaint on that slice,
        scales the wall-time by the video's frame count divided by the
        probe size. Returns the estimated per-item seconds (or 0 if the
        probe can't run -- e.g. only images in the queue).

        Runs synchronously off the main thread so the GUI stays
        responsive; we always cap the probe at ~10 s so launch latency
        stays low even on slow CPUs.
        """
        first_video = None
        for item in self.queue:
            if is_video_file(item.file_path) and item.status == ProcessingStatus.IDLE:
                first_video = item
                break
        if first_video is None:
            return 0.0
        try:
            import cv2 as _cv2
            cap = _cv2.VideoCapture(first_video.file_path)
            try:
                if not cap.isOpened():
                    return 0.0
                total_frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT)) or 1
                fps = cap.get(_cv2.CAP_PROP_FPS) or 30.0
                if fps <= 0:
                    fps = 30.0
                duration = total_frames / fps
                probe_frames = min(30, total_frames)
                if probe_frames <= 0:
                    return 0.0
                from backend.processor import SubtitleDetector
                detector = self._preview_detector
                lang = first_video.config.detection_lang or "en"
                if detector is None or self._preview_detector_lang != lang:
                    detector = SubtitleDetector(lang=lang)
                    self._preview_detector = detector
                    self._preview_detector_lang = lang
                threshold = getattr(first_video.config, "detection_threshold", 0.5)
                t0 = time.monotonic()
                for _ in range(probe_frames):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    detector.detect(frame, threshold)
                elapsed = time.monotonic() - t0
            finally:
                cap.release()
        except Exception as exc:
            logger.debug(f"Pre-batch ETA probe failed: {exc}")
            return 0.0
        if elapsed <= 0 or probe_frames <= 0:
            return 0.0
        # Scale to the full video duration. Add a fudge factor for the
        # inpaint pass and ffmpeg mux which the detect-only probe does
        # not see. 1.8x leaves room for slower inpainters without
        # over-estimating to the point of being useless.
        per_frame_detect = elapsed / probe_frames
        est_per_video = per_frame_detect * total_frames * 1.8 + max(2.0, duration * 0.05)
        return est_per_video

    def _update_batch_progress(self, current: int, total: int):
        """Update the overall batch progress bar, percent label, and title."""
        if total > 0:
            progress = current / total
            pct = int(progress * 100)
            self.batch_progress.set_progress(progress)
            eta = self._compute_eta(current, total)
            label = f"{current} of {total} complete"
            if eta:
                label += f"   -   about {eta} left"
            self.batch_label.config(text=label, fg=Theme.TEXT_SECONDARY)
            self.batch_percent_label.config(text=f"{pct}%", fg=Theme.BLUE_PRIMARY)
            self.root.title(f"[{current}/{total}] {APP_NAME} v{APP_VERSION}")
            # Windows taskbar
            self._ensure_taskbar()
            if self._taskbar:
                self._taskbar.set_state(TaskbarProgress.STATE_NORMAL)
                self._taskbar.set_value(current, total)
        else:
            self.batch_progress.set_progress(0)
            self.batch_label.config(text="Ready", fg=Theme.TEXT_MUTED)
            self.batch_percent_label.config(text="")
            if self._taskbar:
                self._taskbar.clear()

    def _update_item_display(self, item: QueueItem):
        """Update the display for a queue item."""
        def update():
            if item.id in self.queue_widgets:
                self.queue_widgets[item.id].update_item(item)
                # Auto-scroll the queue to keep the active item visible
                if item.status in (ProcessingStatus.LOADING,
                                   ProcessingStatus.DETECTING,
                                   ProcessingStatus.PROCESSING,
                                   ProcessingStatus.MERGING):
                    self._scroll_queue_to_item(item.id)
            fname = Path(item.file_path).name
            if item.status == ProcessingStatus.COMPLETE:
                self._update_status(f"Completed {fname}", "success")
            elif item.status == ProcessingStatus.ERROR:
                self._update_status(f"{fname} needs attention: {item.message}", "error")
            elif item.status == ProcessingStatus.CANCELLED:
                self._update_status(f"Stopped {fname}", "warning")
            else:
                self._update_status(f"{fname}: {item.message}", "info")
            self._refresh_action_states()

        try:
            self.root.after(0, update)
        except RuntimeError:
            pass  # root already destroyed during shutdown

    def _on_processing_complete(self):
        """Handle processing completion."""
        self.is_processing = False
        self._stop_requested = False
        self._processing_thread = None
        self.cancel_event.clear()
        self._stop_elapsed_timer()
        self._set_settings_locked(False)
        # Clear cached remover so next batch picks up any setting changes
        self._cached_remover = None
        self._cached_remover_key = None
        if self._shutdown_started:
            if self._taskbar:
                self._taskbar.clear()
            try:
                self.root.destroy()
            except Exception:
                pass
            return
        self.start_btn.set_style("primary")
        self.start_btn.icon = ">"
        self.start_btn.set_text("Start batch")
        self.root.title(f"{APP_NAME} v{APP_VERSION}")
        self.batch_progress.set_progress(0)
        self.batch_label.config(text="Ready", fg=Theme.TEXT_MUTED)
        if hasattr(self, "batch_percent_label"):
            self.batch_percent_label.config(text="")
        if self._taskbar:
            self._taskbar.clear()
        self._refresh_action_states()

        complete = sum(1 for item in self.queue if item.status == ProcessingStatus.COMPLETE)
        errors = sum(1 for item in self.queue if item.status == ProcessingStatus.ERROR)
        cancelled = sum(1 for item in self.queue if item.status == ProcessingStatus.CANCELLED)

        summary = f"Batch finished: {complete} completed, {errors} failed"
        if cancelled:
            summary += f", {cancelled} stopped"
        is_clean = errors == 0 and cancelled == 0
        quality_summary = summarize_quality_reports(
            [item.quality_report for item in self.queue if item.status == ProcessingStatus.COMPLETE]
        )
        if quality_summary:
            summary += (
                f" | avg PSNR {quality_summary['psnr']:.2f} dB"
                f", avg SSIM {quality_summary['ssim']:.4f}"
            )
        self._update_status(summary, "success" if is_clean else "warning")
        logger.info(summary)
        self._notify_completion(complete, errors)
        # Surface a themed summary modal for meaningful batches
        total = complete + errors + cancelled
        if total >= 1:
            elapsed = ""
            if self._batch_started_at:
                secs = (datetime.now() - self._batch_started_at).total_seconds()
                elapsed = format_time(secs)
            self._show_batch_summary(
                complete,
                errors,
                cancelled,
                elapsed,
                quality_summary=quality_summary,
            )

    def _notify_completion(self, complete: int, errors: int):
        """Flash taskbar + play sound when batch processing finishes."""
        # RM-95: screen-reader announcement so NVDA / Narrator users
        # learn the batch finished without polling the activity log.
        try:
            from backend.a11y import announce
            if errors == 0:
                announce(f"Batch complete. {complete} items processed.")
            else:
                announce(
                    f"Batch finished with {errors} errors. "
                    f"{complete} items processed.",
                    importance="high",
                )
        except Exception:
            pass
        # Flash the taskbar icon to draw attention
        try:
            import ctypes
            import ctypes.wintypes
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())

            class FLASHWINFO(ctypes.Structure):
                _fields_ = [
                    ('cbSize', ctypes.wintypes.UINT),
                    ('hwnd', ctypes.wintypes.HWND),
                    ('dwFlags', ctypes.wintypes.DWORD),
                    ('uCount', ctypes.wintypes.UINT),
                    ('dwTimeout', ctypes.wintypes.DWORD),
                ]

            FLASHW_ALL = 0x03
            FLASHW_TIMERNOFG = 0x0C
            fwi = FLASHWINFO(
                ctypes.sizeof(FLASHWINFO), hwnd,
                FLASHW_ALL | FLASHW_TIMERNOFG, 5, 0)
            ctypes.windll.user32.FlashWindowEx(ctypes.byref(fwi))
        except Exception:
            pass
        # Completion sound
        try:
            import winsound
            if errors == 0:
                winsound.MessageBeep(winsound.MB_OK)
            else:
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except Exception:
            pass

    def run(self):
        """Run the application."""
        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        # Try to restore saved geometry if it still fits on-screen; otherwise
        # fall back to a sensibly centered default.
        restored = False
        saved = (self.config.window_geometry or "").strip()
        if saved:
            try:
                size_part, _, pos_part = saved.partition('+')
                w_s, _, h_s = size_part.partition('x')
                w = int(w_s); h = int(h_s)
                if pos_part:
                    x_s, _, y_s = pos_part.partition('+')
                    x = int(x_s); y = int(y_s)
                    # Reject off-screen saved positions
                    if (x < -80 or y < -40
                            or x + 120 > screen_w or y + 80 > screen_h):
                        raise ValueError("off-screen")
                    self.root.geometry(f"{w}x{h}+{x}+{y}")
                else:
                    self.root.geometry(f"{w}x{h}")
                restored = True
            except Exception:
                restored = False

        if not restored:
            width = min(self.root.winfo_width(), max(960, screen_w - 120))
            height = min(self.root.winfo_height(), max(720, screen_h - 120))
            x = max(24, (screen_w // 2) - (width // 2))
            y = max(24, (screen_h // 2) - (height // 2))
            self.root.geometry(f"{width}x{height}+{x}+{y}")

        logger.info(f"{APP_NAME} v{APP_VERSION} started")
        logger.info(f"Log file: {LOG_FILE}")
        self.root.mainloop()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    # High DPI support on Windows -- Per-Monitor V2 for best multi-monitor support
    try:
        from ctypes import windll
        # Try Per-Monitor V2 first (Windows 10 1703+), then fall back
        try:
            windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    app = VideoSubtitleRemoverApp()
    app.run()


if __name__ == "__main__":
    main()
