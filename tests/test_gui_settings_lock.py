import unittest
import unittest.mock
from types import SimpleNamespace

import numpy as np
from PIL import Image

from gui.app import VideoSubtitleRemoverApp
from gui.preview_controller import PreviewControllerMixin
from gui.widgets import ModernSlider


class _SetEnabledStub:
    def __init__(self):
        self.calls = []

    def set_enabled(self, value):
        self.calls.append(value)


class _ConfigStub:
    def __init__(self):
        self.calls = []

    def config(self, **kwargs):
        self.calls.append(kwargs)


class GuiSettingsLockTests(unittest.TestCase):
    def _app_stub(self):
        app = VideoSubtitleRemoverApp.__new__(VideoSubtitleRemoverApp)
        app.skip_check = _SetEnabledStub()
        app.lama_check = _SetEnabledStub()
        app.preserve_audio_check = _SetEnabledStub()
        app.hw_encode_check = _SetEnabledStub()
        app.region_btn = _SetEnabledStub()
        app.region_reset_btn = _SetEnabledStub()
        app.adv_toggle = _SetEnabledStub()
        app.lang_combo = _ConfigStub()
        app.ocr_engine_combo = _ConfigStub()
        app.language_filter_toggle = _SetEnabledStub()
        app.gpu_combo = _ConfigStub()
        app.time_start_entry = _ConfigStub()
        app.time_end_entry = _ConfigStub()
        app._settings_sliders = [_SetEnabledStub(), _SetEnabledStub()]
        app.mode_picker = SimpleNamespace(
            _segments={"sttn": _ConfigStub(), "lama": _ConfigStub()}
        )
        app.config = SimpleNamespace(subtitle_area=None)
        app._mode_updates = 0
        app._update_mode_options = lambda: setattr(
            app, "_mode_updates", app._mode_updates + 1
        )
        return app

    def test_processing_lock_disables_every_settings_slider(self):
        app = self._app_stub()

        app._set_settings_locked(True)

        self.assertEqual([slider.calls for slider in app._settings_sliders], [[False], [False]])
        self.assertEqual(app.lang_combo.calls[-1]["state"], "disabled")
        self.assertEqual(app.ocr_engine_combo.calls[-1]["state"], "disabled")
        self.assertEqual(app.language_filter_toggle.calls[-1], False)
        self.assertEqual(app.gpu_combo.calls[-1]["state"], "disabled")
        self.assertEqual(app.time_start_entry.calls[-1]["state"], "disabled")
        self.assertEqual(app.time_end_entry.calls[-1]["state"], "disabled")
        self.assertEqual(app.mode_picker._segments["sttn"].calls[-1]["state"], "disabled")
        self.assertEqual(app._mode_updates, 0)

    def test_processing_unlock_restores_settings_sliders(self):
        app = self._app_stub()
        app.config.subtitle_area = (0, 0, 100, 40)

        app._set_settings_locked(False)

        self.assertEqual([slider.calls for slider in app._settings_sliders], [[True], [True]])
        self.assertEqual(app.lang_combo.calls[-1]["state"], "readonly")
        self.assertEqual(app.ocr_engine_combo.calls[-1]["state"], "readonly")
        self.assertEqual(app.language_filter_toggle.calls[-1], True)
        self.assertEqual(app.gpu_combo.calls[-1]["state"], "readonly")
        self.assertEqual(app.region_reset_btn.calls[-1], True)
        self.assertEqual(app.mode_picker._segments["lama"].calls[-1]["state"], "normal")
        self.assertEqual(app._mode_updates, 1)


class ModernSliderStateTests(unittest.TestCase):
    def test_programmatic_value_update_clamps_without_notifying_by_default(self):
        slider = object.__new__(ModernSlider)
        slider.from_ = 0
        slider.to = 10
        slider.value = 3
        slider.command = unittest.mock.Mock()
        slider._sync_a11y = unittest.mock.Mock()
        slider._draw = unittest.mock.Mock()

        ModernSlider.set_value(slider, 99)

        self.assertEqual(slider.value, 10)
        slider.command.assert_not_called()
        slider._sync_a11y.assert_called_once_with()
        slider._draw.assert_called_once_with()

    def test_disabled_slider_ignores_keyboard_steps(self):
        slider = object.__new__(ModernSlider)
        slider.from_ = 0
        slider.to = 100
        slider.value = 50
        slider.enabled = False
        slider.command = lambda _value: self.fail("disabled slider changed")

        ModernSlider._step(slider, 1)

        self.assertEqual(slider.value, 50)

    def test_set_enabled_clears_drag_focus_and_tab_stop(self):
        slider = object.__new__(ModernSlider)
        slider._dragging = True
        slider._focused = True
        slider.value = 50
        slider.enabled = True
        slider.draws = 0

        class _Canvas:
            def __init__(self):
                self.config_calls = []

            def config(self, **kwargs):
                self.config_calls.append(kwargs)

        slider.canvas = _Canvas()
        slider._draw = lambda: setattr(slider, "draws", slider.draws + 1)

        ModernSlider.set_enabled(slider, False)

        self.assertFalse(slider.enabled)
        self.assertFalse(slider._dragging)
        self.assertFalse(slider._focused)
        self.assertEqual(slider.canvas.config_calls[-1]["takefocus"], 0)
        self.assertEqual(slider.accessibility_snapshot()["role"], "slider")
        self.assertIn("disabled", slider.accessibility_snapshot()["state"])
        self.assertIn("50", slider.accessibility_snapshot()["value"])
        self.assertEqual(slider.draws, 1)


class OnboardingActionTests(unittest.TestCase):
    def test_preset_choice_uses_regular_application_path(self):
        app = VideoSubtitleRemoverApp.__new__(VideoSubtitleRemoverApp)
        app.preset_var = unittest.mock.Mock()
        app._on_preset_applied = unittest.mock.Mock()

        app._apply_onboarding_preset("Fast")

        app.preset_var.set.assert_called_once_with("Fast")
        app._on_preset_applied.assert_called_once_with()

    @unittest.mock.patch("gui.app.save_settings")
    def test_auto_band_action_updates_ui_and_persists(self, save_mock):
        app = VideoSubtitleRemoverApp.__new__(VideoSubtitleRemoverApp)
        app.config = SimpleNamespace(auto_band=False)
        app.auto_band_var = unittest.mock.Mock()
        app._update_status = unittest.mock.Mock()

        app._enable_onboarding_auto_band()

        self.assertTrue(app.config.auto_band)
        app.auto_band_var.set.assert_called_once_with(True)
        save_mock.assert_called_once_with(app.config)
        app._update_status.assert_called_once_with(
            "Automatic subtitle-band detection enabled", "success"
        )


class CachedMaskPreviewTests(unittest.TestCase):
    def test_dilation_recomposition_expands_cached_mask(self):
        base_mask = np.zeros((21, 21), dtype=np.uint8)
        base_mask[10, 10] = 255
        cache = {
            "base_mask": base_mask,
            "base_vis": np.zeros((21, 21, 3), dtype=np.uint8),
            "mask_corrections": None,
            "imported": None,
            "color": np.asarray([0, 0, 255], dtype=np.float32),
            "to_pil": lambda bgr: Image.fromarray(bgr[:, :, ::-1]),
            "max_w": 100,
            "max_h": 100,
        }

        narrow = PreviewControllerMixin._render_cached_preview_mask(cache, 0)
        wide = PreviewControllerMixin._render_cached_preview_mask(cache, 4)

        self.assertGreater(
            np.count_nonzero(np.asarray(wide)),
            np.count_nonzero(np.asarray(narrow)),
        )


if __name__ == "__main__":
    unittest.main()
