#!/usr/bin/env python3
"""
Video Subtitle Remover Pro
A professional Windows application for AI-powered subtitle removal from videos
and images. Based on: https://github.com/YaoFANGUK/video-subtitle-remover

Author: SysAdminDoc
See APP_VERSION for the running version -- the docstring deliberately omits
a hardcoded number so there is a single source of truth.
"""

import multiprocessing
multiprocessing.freeze_support()

import logging
import logging.handlers
import os
import sys
import traceback
# Kept for the back-compat surface: callers and tests reach
# `VideoSubtitleRemover.datetime` as a module attribute.
from datetime import datetime  # noqa: F401

# App identity and paths live in gui.config -- the single source of
# truth since the RM-114 extraction. gui.theme / gui.config import no
# tkinter, so this is safe before the GUI availability guard below.
from gui.config import (
    APP_NAME,
    APP_VERSION as APP_VERSION,
    APP_AUTHOR as APP_AUTHOR,
    LOG_DIR as LOG_DIR,
    LOG_FILE,
    SETTINGS_FILE as SETTINGS_FILE,
)

# =============================================================================
# LOGGING SETUP -- file + stream, crash handler
# =============================================================================

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

# GUI availability guard -- fail with a readable message, not a traceback.
try:
    import tkinter as tk  # noqa: F401
except ImportError:
    logger.error("Tkinter not found. Please install Python with Tkinter support.")
    sys.exit(1)

try:
    from PIL import Image, ImageTk  # noqa: F401
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("Pillow not installed. Image preview will be limited.")

# =============================================================================
# RM-114 back-compat surface
# =============================================================================
# Everything the monolith used to define is re-exported here so legacy
# callers (`import VideoSubtitleRemover; VideoSubtitleRemover.X`) keep
# resolving. The canonical re-export list lives in gui/__init__.py --
# add new names there first.

from gui import (  # noqa: E402, F401
    Theme, apply_high_contrast_theme, apply_default_theme, f, mono,
    InpaintMode, ProcessingStatus, STATUS_UI, VSR_SETTINGS_FORMAT,
    ProcessingConfig, QueueItem, BUILTIN_PRESETS,
    _coerce_bool, _coerce_int, _coerce_float, _coerce_text,
    _coerce_rect, _coerce_rect_list, _coerce_gui_mode,
    _read_json_object, _write_json_atomic,
    _migrate_settings, load_settings, save_settings,
    PRESETS_FILE, list_presets, apply_preset,
    save_user_preset, delete_user_preset, export_preset, import_preset,
    status_ui,
    get_app_dir, detect_gpu, format_time, format_size,
    is_video_file, is_image_file,
    _CURATED_LANG_NAMES, _engine_supported_languages, _build_language_list,
    detect_ai_engines, detect_ffmpeg, get_file_info,
    _soft_subtitle_stream_record, _format_soft_subtitle_summary,
    _queue_item_info_text, truncate_middle,
    format_quality_report, summarize_quality_reports,
    VideoSubtitleRemoverApp,
)

from gui.widgets import (  # noqa: E402, F401
    _get_dpi_scale, _scaled,
    Tooltip, ModernButton, ModernProgressBar, ModernToggle,
    ModernSlider, show_confirm, TaskbarProgress, make_themed_menu,
    Toast, SegmentedPicker, DragDropFrame, QueueItemWidget,
    TextWidgetHandler,
)


def _run_smoke_test() -> int:
    """RM-106: bundled GUI smoke path for strict release verification.

    Constructs the full application off-screen, pumps one idle cycle, and
    tears it down without entering the Tk mainloop. Settings are pinned to
    a throwaway temp dir so a release-runner smoke does not clobber a real
    user's %APPDATA% config. Returns 0 on success, 1 on any failure so the
    release workflow can gate on the exit code.
    """
    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory(prefix="vsr_smoke_") as tmp:
        # Redirect settings persistence to the throwaway dir. gui.config is
        # the single source of truth; VideoSubtitleRemover re-exports the
        # name for back-compat, so update both views.
        import gui.config as _gc
        smoke_settings = _Path(tmp) / "settings.json"
        _gc.SETTINGS_FILE = smoke_settings
        global SETTINGS_FILE
        SETTINGS_FILE = smoke_settings

        app = None
        try:
            app = VideoSubtitleRemoverApp()
            app.root.withdraw()
            app.root.update_idletasks()
            title = app.root.title()
            if not title.startswith(APP_NAME):
                logger.error("Smoke test: unexpected window title %r", title)
                return 1
            smoke_locale = os.environ.get("VSR_SMOKE_LOCALE", "").strip()
            if smoke_locale:
                from backend.i18n import bind_locale as _bind_locale, tr as _tr

                bound = _bind_locale(smoke_locale)
                translated = _tr("Start batch")
                if bound == "en" or translated == "Start batch":
                    logger.error(
                        "Smoke test: locale %r was not loaded from the frozen payload",
                        smoke_locale,
                    )
                    return 1
            logger.info("Smoke test passed: GUI constructed and torn down.")
            return 0
        except Exception:
            logger.critical("Smoke test failed:\n%s", traceback.format_exc())
            return 1
        finally:
            if app is not None:
                try:
                    app.root.destroy()
                except Exception:
                    pass


def main():
    """Main entry point."""
    # RM-106: headless self-test for release verification. Must run before
    # the DPI/mainloop path so it can exit cleanly on a CI runner.
    if "--smoke-test" in sys.argv[1:]:
        sys.exit(_run_smoke_test())

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
