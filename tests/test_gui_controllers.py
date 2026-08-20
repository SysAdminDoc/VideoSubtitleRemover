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


class WindowGeometryRestoreTests(unittest.TestCase):
    """A saved position on ANY monitor must survive a restart."""

    def _app_class(self):
        from gui.app import VideoSubtitleRemoverApp

        return VideoSubtitleRemoverApp

    def test_secondary_monitor_positions_are_accepted(self):
        app = self._app_class()
        # Primary is 1920x1080; a second monitor sits to its right and a
        # third to its left (negative origin). Both are legitimate homes.
        bounds = (-1920, 0, 5760, 1080)
        self.assertTrue(app._saved_position_visible(2500, 200, bounds))
        self.assertTrue(app._saved_position_visible(-1800, 100, bounds))
        self.assertTrue(app._saved_position_visible(100, 100, bounds))

    def test_truly_offscreen_positions_are_still_rejected(self):
        app = self._app_class()
        bounds = (0, 0, 1920, 1080)
        self.assertFalse(app._saved_position_visible(5360, 0, bounds))
        self.assertFalse(app._saved_position_visible(-500, 100, bounds))
        self.assertFalse(app._saved_position_visible(100, 1080, bounds))

    def test_desktop_bounds_fall_back_to_the_primary_display(self):
        app = self._app_class()
        with mock.patch("gui.app.sys") as fake_sys:
            fake_sys.platform = "linux"
            self.assertEqual(
                app._desktop_bounds(1920, 1080), (0, 0, 1920, 1080))


if __name__ == "__main__":
    unittest.main()


class DirectMlDeviceMappingTests(unittest.TestCase):
    """RM-159: a DirectML selection must reach the backend as "directml".

    gui_to_backend_payload derived the device from use_gpu/gpu_id alone, so a
    DirectML adapter became "cuda:N"; device_provider then probed CUDA, found
    none, and fell back to CPU -- while the batch report still recorded
    "directml". Only the preview path used the DirectML-aware mapper.
    """

    def _gui_config(self, **overrides):
        from gui.config import ProcessingConfig

        config = ProcessingConfig()
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    def test_directml_selection_maps_to_the_directml_device(self):
        from backend.config_schema import gui_to_backend_payload

        payload = gui_to_backend_payload(
            self._gui_config(use_gpu=True, gpu_id=0, gpu_backend="directml")
        )

        self.assertEqual(payload["device"], "directml")

    def test_cuda_selection_still_maps_to_an_indexed_cuda_device(self):
        from backend.config_schema import gui_to_backend_payload

        payload = gui_to_backend_payload(
            self._gui_config(use_gpu=True, gpu_id=2, gpu_backend="cuda")
        )

        self.assertEqual(payload["device"], "cuda:2")

    def test_gpu_disabled_still_maps_to_cpu(self):
        from backend.config_schema import gui_to_backend_payload

        payload = gui_to_backend_payload(
            self._gui_config(use_gpu=False, gpu_backend="directml")
        )

        self.assertEqual(payload["device"], "cpu")

    def test_unknown_family_falls_back_to_cuda_indexing(self):
        from backend.config_schema import gui_to_backend_payload

        payload = gui_to_backend_payload(
            self._gui_config(use_gpu=True, gpu_id=1, gpu_backend="")
        )

        self.assertEqual(payload["device"], "cuda:1")


class PreviewMatchesBatchConfigTests(unittest.TestCase):
    """The previews exist to A/B settings, so they must be built from the
    same configuration the run would use."""

    def _preview_source(self):
        return (ROOT / "gui" / "preview_controller.py").read_text(
            encoding="utf-8")

    def test_test_cleanup_converts_the_whole_config(self):
        source = self._preview_source()
        self.assertIn("backend_cfg = gui_to_backend_config(snapshot_cfg)", source)
        # A hand-built partial copy silently dropped detector fields.
        self.assertNotIn("_BackendCfg(", source)

    def test_test_cleanup_overrides_only_the_device(self):
        source = self._preview_source()
        converted = source.index("backend_cfg = gui_to_backend_config")
        self.assertLess(converted, source.index("backend_cfg.device", converted))

    def test_converter_carries_the_fields_the_partial_copy_dropped(self):
        from backend.config_schema import gui_to_backend_config
        from gui.config import ProcessingConfig as GuiConfig

        gui_cfg = GuiConfig()
        gui_cfg.detection_engine = "easyocr"
        gui_cfg.detection_lang = "ja"
        backend_cfg = gui_to_backend_config(gui_cfg)
        self.assertEqual(backend_cfg.detection_engine, "easyocr")
        self.assertEqual(backend_cfg.detection_lang, "ja")
        for field in (
            "detection_vertical",
            "language_mask_filter",
            "lama_super_fast",
        ):
            self.assertTrue(hasattr(backend_cfg, field), field)

    def test_mask_preview_reads_detector_settings_from_the_item(self):
        source = self._preview_source()
        start = source.index("ocr_engine = getattr(")
        block = source[start - 120:start + 320]
        # Engine, variant and threshold must come from the item, which is
        # where a per-file override lives.
        self.assertIn('getattr(item_config, "detection_engine"', block)
        self.assertIn('getattr(item_config, "rapidocr_variant"', block)
        self.assertIn("getattr(item_config, 'detection_threshold'", block)
        self.assertNotIn('getattr(self.config, "detection_engine"', block)


class SoftSubtitleCancelTests(unittest.TestCase):
    """Per-item Stop must reach the soft-subtitle remux path."""

    def test_cancel_check_honours_item_cancel_requested(self):
        import threading
        from datetime import datetime

        from gui.processing_controller import ProcessingControllerMixin

        captured = {}

        def fake_remux(src, dst, *, action, on_process, cancel_check):
            captured["cancel_check"] = cancel_check

        controller = ProcessingControllerMixin.__new__(ProcessingControllerMixin)
        controller.cancel_event = threading.Event()
        controller._set_active_subprocess = lambda proc: None
        controller._update_item_display = lambda _item: None
        controller._batch_times = []

        item = SimpleNamespace(
            soft_subtitle_action="strip",
            file_path="in.mkv",
            output_path="out.mkv",
            cancel_requested=False,
            status=None,
            progress=0.0,
            error="x",
            quality_report={},
            started_at=datetime.now(),
            completed_at=None,
            stage_timings=None,
            message="",
            soft_subtitle_summary=None,
        )

        with mock.patch("backend.remux.remux_soft_subtitles", fake_remux), \
                mock.patch("pathlib.Path.mkdir", lambda *a, **k: None):
            controller._process_soft_subtitle_item(item)

        cancel_check = captured["cancel_check"]
        self.assertFalse(cancel_check())
        # The global Stop still works...
        controller.cancel_event.set()
        self.assertTrue(cancel_check())
        controller.cancel_event.clear()
        # ...and so does one item's Stop, which previously did nothing here.
        item.cancel_requested = True
        self.assertTrue(cancel_check())
