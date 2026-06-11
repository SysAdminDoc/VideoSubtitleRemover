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
APP_VERSION = "3.16.1"
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

try:
    from backend.security_checks import warn_if_vulnerable_opencv_libpng
    warn_if_vulnerable_opencv_libpng(logger)
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

from gui.theme import (  # noqa: E402  -- RM-114 extraction
    Theme, apply_high_contrast_theme, apply_default_theme, f, mono,
)

# =============================================================================
# CONFIG, ENUMS, SETTINGS, PRESETS -- RM-114: imported from gui.config
# =============================================================================

from gui.config import (  # noqa: E402
    InpaintMode, ProcessingStatus, STATUS_UI, VSR_SETTINGS_FORMAT,
    ProcessingConfig, QueueItem,
    _coerce_bool, _coerce_int, _coerce_float, _coerce_text,
    _coerce_rect, _coerce_rect_list, _coerce_gui_mode,
    _read_json_object, _write_json_atomic,
    _migrate_settings, load_settings, save_settings,
    PRESETS_FILE, list_presets, apply_preset,
    save_user_preset, delete_user_preset, export_preset, import_preset,
    status_ui,
)


from gui.utils import (  # noqa: E402
    get_app_dir, detect_gpu, format_time, format_size,
    is_video_file, is_image_file,
    _CURATED_LANG_NAMES, _engine_supported_languages, _build_language_list,
    detect_ai_engines, detect_ffmpeg, get_file_info,
    _soft_subtitle_stream_record, _format_soft_subtitle_summary,
    _queue_item_info_text, truncate_middle,
    format_quality_report, summarize_quality_reports,
)



# =============================================================================
# CUSTOM WIDGETS -- RM-114: imported from gui.widgets
# =============================================================================

from gui.widgets import (  # noqa: E402
    _get_dpi_scale, _scaled,
    Tooltip, ModernButton, ModernProgressBar, ModernToggle,
    ModernSlider, show_confirm, TaskbarProgress, make_themed_menu,
    Toast, SegmentedPicker, DragDropFrame, QueueItemWidget,
    TextWidgetHandler,
)



from gui.app import VideoSubtitleRemoverApp  # noqa: E402


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
