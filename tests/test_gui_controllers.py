"""Controller boundaries that run without constructing a Tk application."""

from __future__ import annotations

import ast
import importlib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from gui.mask_correction_controller import (
    MaskCorrectionControllerMixin,
    MaskCorrectionWindow,
)
from gui.preview_controller import PreviewControllerMixin
from gui.region_controller import (
    RegionEditorControllerMixin,
    RegionSelectorWindow,
)
from gui.settings_controller import AdvancedSettingsControllerMixin


ROOT = Path(__file__).resolve().parents[1]
CONTROLLERS = (
    "mask_correction_controller",
    "preview_controller",
    "processing_controller",
    "quality_controller",
    "region_controller",
    "settings_controller",
    "support_controller",
)


class ControllerBoundaryTests(unittest.TestCase):
    def test_controllers_declare_protocols_and_never_import_app(self):
        for name in CONTROLLERS:
            module = importlib.import_module(f"gui.{name}")
            protocols = [
                value
                for value in vars(module).values()
                if isinstance(value, type)
                and value.__name__.endswith("ControllerHost")
            ]
            self.assertEqual(len(protocols), 1, name)
            self.assertTrue(getattr(protocols[0], "_is_protocol", False), name)

            tree = ast.parse((ROOT / "gui" / f"{name}.py").read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotEqual(node.module, "gui.app", name)
                    self.assertNotIn("*", [alias.name for alias in node.names], name)

    def test_app_composes_extracted_controllers_without_redefining_methods(self):
        from gui.app import VideoSubtitleRemoverApp

        self.assertTrue(issubclass(VideoSubtitleRemoverApp, RegionEditorControllerMixin))
        self.assertTrue(issubclass(VideoSubtitleRemoverApp, AdvancedSettingsControllerMixin))
        for method in (
            "_open_region_selector_modal",
            "_reset_region",
            "_toggle_advanced",
            "_on_preset_applied",
        ):
            self.assertNotIn(method, VideoSubtitleRemoverApp.__dict__)

    def test_region_reset_uses_only_its_declared_host_surface(self):
        controller = RegionEditorControllerMixin()
        controller.config = SimpleNamespace(
            subtitle_area=(1, 2, 3, 4),
            subtitle_areas=[(1, 2, 3, 4)],
            subtitle_region_spans=[{"rect": (1, 2, 3, 4)}],
            subtitle_region_keyframes=[{"id": "track"}],
        )
        controller._apply_region_settings_to_idle_items = mock.Mock()
        controller._update_region_label_display = mock.Mock()
        controller._update_status = mock.Mock()

        controller._reset_region()

        self.assertIsNone(controller.config.subtitle_area)
        self.assertIsNone(controller.config.subtitle_areas)
        self.assertIsNone(controller.config.subtitle_region_spans)
        self.assertIsNone(controller.config.subtitle_region_keyframes)
        controller._apply_region_settings_to_idle_items.assert_called_once_with()
        controller._update_region_label_display.assert_called_once_with()

    def test_region_selector_callbacks_are_explicit_window_methods(self):
        tree = ast.parse(
            (ROOT / "gui" / "region_controller.py").read_text(encoding="utf-8")
        )
        window = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == RegionSelectorWindow.__name__
        )
        methods = {
            node.name: node for node in window.body
            if isinstance(node, ast.FunctionDef)
        }
        for name in (
            "on_press",
            "on_drag",
            "on_release",
            "_draw_saved_rects",
            "_apply_numeric_region_edit",
            "_add_region_keyframe",
            "_save_and_close",
            "_release_cap",
        ):
            self.assertIn(name, methods)
        for method in methods.values():
            nested = [
                node for node in ast.walk(method)
                if isinstance(node, ast.FunctionDef) and node is not method
            ]
            self.assertEqual(nested, [], method.name)

        mixin = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == RegionEditorControllerMixin.__name__
        )
        entry = next(
            node for node in mixin.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_open_region_selector_modal"
        )
        self.assertLessEqual(entry.end_lineno - entry.lineno + 1, 5)

    def test_mask_correction_callbacks_are_explicit_window_methods(self):
        tree = ast.parse(
            (ROOT / "gui" / "mask_correction_controller.py").read_text(
                encoding="utf-8"
            )
        )
        window = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == MaskCorrectionWindow.__name__
        )
        methods = {
            node.name: node for node in window.body
            if isinstance(node, ast.FunctionDef)
        }
        for name in (
            "render",
            "paint_press",
            "paint_drag",
            "paint_release",
            "detect_mask",
            "load_frame",
            "_load_frame_worker",
            "_apply_frame_mask",
            "prepare_rerun",
            "release_capture",
        ):
            self.assertIn(name, methods)
        for method in methods.values():
            nested = [
                node for node in ast.walk(method)
                if isinstance(node, ast.FunctionDef) and node is not method
            ]
            self.assertEqual(nested, [], method.name)

        mixin = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == MaskCorrectionControllerMixin.__name__
        )
        entry = next(
            node for node in mixin.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_open_mask_correction_editor"
        )
        self.assertLessEqual(entry.end_lineno - entry.lineno + 1, 8)

    def test_settings_visibility_toggles_without_app_construction(self):
        controller = AdvancedSettingsControllerMixin()
        controller.adv_visible = False
        controller.adv_toggle = SimpleNamespace(icon="", set_text=mock.Mock())
        controller.adv_panel = SimpleNamespace(
            pack=mock.Mock(),
            pack_forget=mock.Mock(),
        )

        controller._toggle_advanced()
        self.assertTrue(controller.adv_visible)
        self.assertEqual(controller.adv_toggle.icon, "-")
        controller.adv_panel.pack.assert_called_once_with(fill="x")

        controller._toggle_advanced()
        self.assertFalse(controller.adv_visible)
        self.assertEqual(controller.adv_toggle.icon, "+")
        controller.adv_panel.pack_forget.assert_called_once_with()

    def test_preview_dispatch_honors_shutdown_without_tk(self):
        controller = PreviewControllerMixin()
        controller.root = SimpleNamespace(after=mock.Mock(return_value="after-1"))
        callback = mock.Mock()

        controller._shutdown_started = False
        self.assertEqual(controller._dispatch_preview_ui(callback, 3), "after-1")
        controller.root.after.assert_called_once_with(0, callback, 3)

        controller._shutdown_started = True
        self.assertIsNone(controller._dispatch_preview_ui(callback, 4))
        self.assertEqual(controller.root.after.call_count, 1)


if __name__ == "__main__":
    unittest.main()
