"""Command-line wrapper for the packaged GUI release probe."""

from __future__ import annotations

from pathlib import Path
import sys


def _enable_windows_dpi_awareness() -> None:
    """Exercise the same DPI path as the real Windows application."""
    if sys.platform != "win32":
        return
    try:
        from ctypes import windll

        try:
            windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_enable_windows_dpi_awareness()

from gui.release_probe import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
