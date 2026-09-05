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

import sys as _sys_early

# RM-155: a frozen build re-executes this same exe as its job worker. The
# worker must stay lean and isolated: short-circuit BEFORE the module-level
# logging setup (which would open the parent GUI's rotating log file from a
# second process and break rotation on Windows) and before the `from gui
# import ...` block (which drags tkinter, PIL, and the whole widget tree
# into a process that only runs backend code). `main()` keeps its own
# `--job-worker` branch as a belt-and-braces fallback.
if __name__ == "__main__" and "--job-worker" in _sys_early.argv[1:]:
    from backend.job_worker import main as _job_worker_main

    raise SystemExit(_job_worker_main(
        [arg for arg in _sys_early.argv[1:] if arg != "--job-worker"]))

import json
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
        # RM-314: delay=True so importing this module does not open or
        # create the shared log. A launch that is about to be refused for a
        # second instance must leave the running instance's files alone.
        logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=2,
            encoding='utf-8', delay=True),
    ]
)
logger = logging.getLogger(__name__)


def crash_handler(exc_type, exc_value, exc_tb):
    """Global crash handler -- log to file and show MessageBox."""
    msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
    logger.critical(f"UNHANDLED EXCEPTION:\n{msg}")
    try:
        from backend.i18n import tr
        import tkinter.messagebox as mb
        mb.showerror(
            tr("Something went wrong"),
            tr(
                "{app} had to close.\n\n{error}\n\nA full report is in the log:\n{log}"
            ).format(app=APP_NAME, error=exc_value, log=LOG_FILE),
        )
    except Exception:
        try:
            import tkinter.messagebox as mb
            mb.showerror(
                "Fatal Error",
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
            import cv2 as _cv2
            import numpy as _np

            logger.info(
                "Smoke imports passed: numpy %s, cv2 %s",
                _np.__version__,
                _cv2.__version__,
            )
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


def _run_frozen_import_smoke(result_path: str) -> int:
    """Prove frozen core imports without constructing any visible UI."""
    path = os.path.abspath(result_path)
    payload = {
        "schema": "vsr.frozen_import_smoke.v1",
        "appName": APP_NAME,
        "appVersion": APP_VERSION,
        "startedMessage": f"{APP_NAME} v{APP_VERSION} started",
        "imports": {},
        "passed": False,
        "error": "",
    }
    try:
        import cv2 as _cv2
        import numpy as _np

        payload["imports"] = {
            "cv2": str(_cv2.__version__),
            "numpy": str(_np.__version__),
        }
        payload["passed"] = True
        logger.info(payload["startedMessage"])
    except Exception:
        payload["error"] = traceback.format_exc()
        logger.critical("Frozen import smoke failed:\n%s", payload["error"])
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except Exception:
        logger.critical(
            "Frozen import smoke could not write %s:\n%s",
            path,
            traceback.format_exc(),
        )
        return 1
    return 0 if payload["passed"] else 1


def _run_frozen_provider_smoke(result_path: str) -> int:
    """RM-350: which provider does THIS artifact actually get?

    The dependency profile smoke answers that for the environment the build
    ran in. This answers it for the frozen payload, from inside the frozen
    payload, so a CPU-only bundle cannot be published under a CUDA name.
    """
    path = os.path.abspath(result_path)
    payload = {
        "schema": "vsr.frozen_provider_smoke.v1",
        "appVersion": APP_VERSION,
        "frozen": bool(getattr(sys, "frozen", False)),
        "profile": "",
        "profileSource": "",
        "declaredProvider": "",
        "availableProviders": [],
        "activeProviders": [],
        "ran": False,
        "passed": False,
        "fellBack": None,
        "error": "",
    }
    try:
        from backend.build_profile import resolve_build_profile
        from backend.dependency_profiles import run_profile_provider_smoke

        build = resolve_build_profile()
        payload["profile"] = build["profile"]
        payload["profileSource"] = build["source"]
        payload["declaredProvider"] = build["provider"]

        smoke = run_profile_provider_smoke(build["profile"])
        payload.update({
            "availableProviders": list(smoke.get("availableProviders") or []),
            "activeProviders": list(smoke.get("activeProviders") or []),
            "ran": bool(smoke.get("ran")),
            "passed": smoke.get("passed") is True,
            "fellBack": smoke.get("fellBack"),
            "error": str(smoke.get("error") or ""),
        })
    except Exception:  # noqa: BLE001 - this is the probe, not the product
        # An artifact that cannot answer is exactly what this smoke exists to
        # catch, and the answer has to reach the caller as a written record
        # rather than as a traceback on a stream nobody reads. The failure is
        # reported in the payload and in the exit code; nothing is swallowed.
        payload["error"] = traceback.format_exc()
        logger.critical("Frozen provider smoke failed:\n%s", payload["error"])
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except OSError:
        logger.critical(
            "Frozen provider smoke could not write %s:\n%s",
            path,
            traceback.format_exc(),
        )
        return 1
    return 0 if payload["passed"] else 1


def _run_ui_release_probe(
    result_path: str,
    *,
    scale: int,
    theme: str,
    locale: str,
) -> int:
    """Run one real Tk layout case from the frozen executable."""
    from gui.release_probe import run_probe

    try:
        payload = run_probe(
            int(scale), theme == "high-contrast", str(locale)
        )
    except Exception:
        payload = {
            "ok": False,
            "scale": int(scale),
            "theme": str(theme),
            "locale": str(locale),
            "failures": [traceback.format_exc()],
        }
    payload["appVersion"] = APP_VERSION
    payload["frozen"] = bool(getattr(sys, "frozen", False))
    try:
        os.makedirs(os.path.dirname(os.path.abspath(result_path)), exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except Exception:
        logger.critical(
            "UI release probe could not write %s:\n%s",
            result_path,
            traceback.format_exc(),
        )
        return 1
    return 0 if payload.get("ok") else 1


def _required_option(args: list[str], option: str) -> str:
    try:
        return args[args.index(option) + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"{option} requires a value") from exc


APP_USER_MODEL_ID = "SysAdminDoc.VideoSubtitleRemoverPro"


def _set_app_user_model_id() -> bool:
    """Give the app its own taskbar identity. RM-346.

    Without this a frozen Python application inherits the interpreter's
    identity, so Windows groups it under whatever else is running and pins the
    wrong thing. It has to happen before the first window is presented, which
    is why it sits beside the DPI call rather than in the GUI layer.

    Returns whether the call was made, for the test to assert on rather than
    inspect the taskbar.
    """
    if sys.platform != "win32":
        return False
    try:
        from ctypes import windll

        windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            APP_USER_MODEL_ID)
        return True
    except Exception:  # noqa: BLE001 - see below
        # A missing shell32 export or a locked-down container is not worth
        # refusing to start over: the only cost is taskbar grouping.
        logger.debug("Could not set the AppUserModelID", exc_info=True)
        return False


def main():
    """Main entry point."""
    # RM-155: a frozen build has no importable `-m backend.job_worker`
    # target, so the supervisor re-executes this same exe with a marker.
    # Must be the first branch: a job worker must never touch DPI, Tk, or
    # the settings file.
    if "--job-worker" in sys.argv[1:]:
        from backend.job_worker import main as job_worker_main

        argv = [arg for arg in sys.argv[1:] if arg != "--job-worker"]
        sys.exit(job_worker_main(argv))

    if "--frozen-import-smoke" in sys.argv[1:]:
        try:
            result_index = sys.argv.index("--frozen-import-smoke") + 1
            result_path = sys.argv[result_index]
        except (IndexError, ValueError):
            logger.error("--frozen-import-smoke requires a result path")
            sys.exit(2)
        sys.exit(_run_frozen_import_smoke(result_path))

    if "--frozen-provider-smoke" in sys.argv[1:]:
        try:
            result_index = sys.argv.index("--frozen-provider-smoke") + 1
            result_path = sys.argv[result_index]
        except (IndexError, ValueError):
            logger.error("--frozen-provider-smoke requires a result path")
            sys.exit(2)
        sys.exit(_run_frozen_provider_smoke(result_path))

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

    _set_app_user_model_id()

    if "--ui-release-probe" in sys.argv[1:]:
        try:
            result_path = _required_option(sys.argv, "--ui-release-probe")
            scale = int(_required_option(sys.argv, "--scale"))
            theme = _required_option(sys.argv, "--theme")
            locale = _required_option(sys.argv, "--locale")
            if scale not in {100, 125, 150, 175, 200}:
                raise ValueError("--scale must be 100, 125, 150, 175, or 200")
            if theme not in {"default", "high-contrast"}:
                raise ValueError("--theme must be default or high-contrast")
            if locale not in {"en", "pseudo", "rtl"}:
                raise ValueError("--locale must be en, pseudo, or rtl")
        except ValueError as exc:
            logger.error("Invalid UI release probe arguments: %s", exc)
            sys.exit(2)
        sys.exit(_run_ui_release_probe(
            result_path,
            scale=scale,
            theme=theme,
            locale=locale,
        ))

    # RM-314: a second GUI shares settings.json and queue_state.json with
    # the first, so refuse before anything is read or written. Report and
    # exit without raising or focusing the running window.
    from gui import single_instance

    guard = single_instance.acquire()
    if guard.already_running:
        guard.release()
        # Deliberately stderr only. The log file lives in the same per-user
        # directory the running instance has open, so a refused launch must
        # not append to it or trip its rotation.
        print(single_instance.ALREADY_RUNNING_MESSAGE, file=sys.stderr)
        sys.exit(3)

    # Hold the slot for the whole session. Releasing it here and letting the
    # app re-acquire leaves a window, measured in tens of milliseconds, where
    # a second launch during a cold start sees the slot free.
    try:
        app = VideoSubtitleRemoverApp(instance_guard=guard)
        app.run()
    finally:
        guard.release()


if __name__ == "__main__":
    main()
