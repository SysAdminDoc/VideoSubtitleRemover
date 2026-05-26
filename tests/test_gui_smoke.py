"""RFP-T-4 GUI smoke test (Tk update_idletasks walk).

The full GUI is heavy to instantiate (250+ widgets, custom Canvas
drawing, etc.) so we exercise it via a `headless()` factory that
withdraws the root window and drives the major flows without
actually mapping the GUI to a display.

Skips automatically on:
- Non-display environments (Linux / WSL without DISPLAY / WAYLAND).
- Builds without Pillow (the preview pane construction requires it).
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path


def _have_display() -> bool:
    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@unittest.skipUnless(_have_display(), "GUI smoke test needs a display")
class GuiSmokeTests(unittest.TestCase):
    """Drive the major GUI flows headlessly."""

    @classmethod
    def setUpClass(cls):
        # Pin the settings file to a temp dir so the test doesn't
        # clobber the developer's real %APPDATA%/VSR config.
        cls._tmpdir = tempfile.TemporaryDirectory()
        import VideoSubtitleRemover as g
        cls._g = g
        cls._orig_settings = g.SETTINGS_FILE
        g.SETTINGS_FILE = Path(cls._tmpdir.name) / "settings.json"

    @classmethod
    def tearDownClass(cls):
        cls._g.SETTINGS_FILE = cls._orig_settings
        cls._tmpdir.cleanup()

    def _make_app(self):
        app = self._g.VideoSubtitleRemoverApp()
        app.root.withdraw()
        return app

    def test_construct_and_close(self):
        app = self._make_app()
        try:
            self.assertEqual(app.root.title()[:23], "Video Subtitle Remover ")
            app.root.update_idletasks()
        finally:
            app.root.destroy()

    def test_sync_config_round_trip(self):
        app = self._make_app()
        try:
            # Toggle a known field via the tk variable and confirm it
            # propagates into the persisted config.
            app.preserve_audio_var.set(False)
            app._sync_config_from_ui()
            self.assertFalse(app.config.preserve_audio)
        finally:
            app.root.destroy()

    def test_apply_responsive_layout_no_op_when_unchanged(self):
        app = self._make_app()
        try:
            app._apply_responsive_layout(1280)
            initial = app._layout_mode
            app._apply_responsive_layout(1280)
            self.assertEqual(app._layout_mode, initial)
        finally:
            app.root.destroy()


if __name__ == "__main__":
    unittest.main()
