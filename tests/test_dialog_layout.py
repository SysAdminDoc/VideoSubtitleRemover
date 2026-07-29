"""RM-148: major dialogs reflow and scroll inside the screen work area."""

import os
from pathlib import Path
import sys
import unittest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tkinter as tk

from gui import dialog_layout


class _TkTestCase(unittest.TestCase):
    """One Tk root per class, torn down like the other GUI test files."""

    @classmethod
    def setUpClass(cls):
        try:
            cls._shared_root = tk.Tk()
        except Exception as exc:  # pragma: no cover - headless CI
            raise unittest.SkipTest(f"Tk display unavailable: {exc}")
        cls._shared_root.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls._shared_root.destroy()
        # Clear the stale default-root pointer so a later test file creating
        # its own Tk root does not inherit the finalized interpreter.
        try:
            tk._default_root = None
        except Exception:
            pass

    def setUp(self):
        self.root = self._shared_root
        self.addCleanup(
            lambda: self.root.__dict__.pop("_vsr_work_area_override", None))


class WorkAreaTests(_TkTestCase):

    def test_work_area_leaves_room_for_the_taskbar(self):
        width, height = dialog_layout.work_area(self.root)
        self.assertLessEqual(width, self.root.winfo_screenwidth())
        self.assertLess(height, self.root.winfo_screenheight())

    def test_override_is_honoured_for_deterministic_probes(self):
        self.root._vsr_work_area_override = (980, 720)
        self.assertEqual(dialog_layout.work_area(self.root), (980, 720))

    def test_override_is_clamped_to_a_usable_minimum(self):
        self.root._vsr_work_area_override = (10, 10)
        width, height = dialog_layout.work_area(self.root)
        self.assertGreaterEqual(width, dialog_layout.MIN_DIALOG_WIDTH)
        self.assertGreaterEqual(height, dialog_layout.MIN_DIALOG_HEIGHT)


class ScrollableDialogTests(_TkTestCase):
    def _dialog(self, rows: int):
        dialog = tk.Toplevel(self.root)
        dialog.withdraw()
        body = dialog_layout.scrollable_dialog_body(dialog)
        for index in range(rows):
            tk.Label(body, text=f"row {index} " * 8).pack(anchor="w")
        return dialog

    def test_tall_content_is_clamped_and_scrollable(self):
        self.root._vsr_work_area_override = (980, 720)
        dialog = self._dialog(120)
        self.addCleanup(dialog.destroy)
        width, height = dialog_layout.fit_dialog_to_work_area(
            dialog, self.root)
        self.assertLessEqual(width, 980)
        self.assertLessEqual(height, 720)
        canvas = dialog._vsr_scroll_canvas
        dialog.update_idletasks()
        bbox = canvas.bbox("all") or (0, 0, 0, 0)
        self.assertGreater(bbox[3] - bbox[1], canvas.winfo_height())
        canvas.yview_moveto(1.0)
        canvas.update_idletasks()
        self.assertGreater(canvas.yview()[0], 0.0)

    def test_scroll_surface_is_keyboard_focusable(self):
        dialog = self._dialog(60)
        self.addCleanup(dialog.destroy)
        self.assertEqual(int(dialog._vsr_scroll_canvas.cget("takefocus")), 1)

    def test_short_content_fits_without_clamping(self):
        self.root._vsr_work_area_override = (980, 720)
        dialog = self._dialog(2)
        self.addCleanup(dialog.destroy)
        width, height = dialog_layout.fit_dialog_to_work_area(
            dialog, self.root)
        self.assertLess(height, 720)
        self.assertLessEqual(width, 980)

    def test_dialog_becomes_resizable_with_a_work_area_ceiling(self):
        self.root._vsr_work_area_override = (980, 720)
        dialog = self._dialog(120)
        self.addCleanup(dialog.destroy)
        dialog_layout.fit_dialog_to_work_area(dialog, self.root)
        self.assertEqual(
            [bool(value) for value in dialog.resizable()], [True, True])
        self.assertEqual(tuple(dialog.maxsize()), (980, 720))


class DialogSourceContractTests(unittest.TestCase):
    """The three previously fixed dialogs must use the shared helper."""

    MODULES = (
        "gui/onboarding.py",
        "gui/region_controller.py",
        "gui/mask_correction_controller.py",
    )

    def test_no_major_dialog_is_fixed_size(self):
        for name in self.MODULES:
            source = (_ROOT / name).read_text(encoding="utf-8")
            with self.subTest(module=name):
                self.assertIn("scrollable_dialog_body", source)
                self.assertIn("fit_dialog_to_work_area", source)
                self.assertNotIn("resizable(False, False)", source)

    def test_scaling_probe_covers_dialog_fit(self):
        source = (_ROOT / "tools" / "ui_scaling_probe.py").read_text(
            encoding="utf-8")
        self.assertIn("_probe_dialog_fit", source)
        self.assertIn("(980, 720)", source)
        self.assertIn("(2752, 1152)", source)


if __name__ == "__main__":
    os.environ.setdefault("VSR_UI_BACKGROUND", "1")
    unittest.main()
