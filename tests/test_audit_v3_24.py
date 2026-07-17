"""Regression tests for the v3.24.0 deep-audit fixes.

Covers:
- CLI --preset now applies auto_band (direct store_true) and the inverted
  no-* tri-state flags (kalman/phash/scene-split), respecting explicit
  user overrides.
- crash_reporter UNC path redaction actually matches \\\\server\\share paths.
- whisper_fallback filter-value escaping covers the ';' chain separator.
- dependency_caps version comparison keeps the 4th build component.
- LaMa neural inpainters binarize a soft alpha matte before model input.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


_MODE_CHOICES = ["sttn", "lama", "propainter", "auto", "migan"]


def _prepare(argv):
    from backend import cli as _cli

    parser = _cli._build_parser(list(_MODE_CHOICES))
    args = parser.parse_args(argv)
    _cli._prepare_cli_args(args, parser, argv)
    return args


class CliPresetMergeTests(unittest.TestCase):
    def test_preset_applies_auto_band(self):
        # TikTok / Vertical short sets auto_band=True; the direct store_true
        # mapping must now propagate it (previously mapped to None -> dropped).
        args = _prepare(["--validate-config",
                         "--preset", "TikTok / Vertical short"])
        self.assertTrue(args.auto_band)

    def test_explicit_auto_band_flag_still_wins(self):
        args = _prepare(["--validate-config", "--auto-band",
                         "--preset", "YouTube (default)"])
        self.assertTrue(args.auto_band)

    def test_inverted_flag_preset_disable(self):
        # A user preset that disables kalman tracking must flip --no-kalman on.
        from backend import presets

        original = dict(presets.BUILTIN_PRESETS)
        presets.BUILTIN_PRESETS["__audit_test__"] = {
            "description": "test",
            "fields": {"kalman_tracking": False, "phash_skip_enable": False,
                       "tbe_scene_cut_split": False},
        }
        try:
            args = _prepare(["--validate-config", "--preset", "__audit_test__"])
        finally:
            presets.BUILTIN_PRESETS.clear()
            presets.BUILTIN_PRESETS.update(original)
        self.assertTrue(args.no_kalman)
        self.assertTrue(args.no_phash)
        self.assertTrue(args.no_scene_split)

    def test_inverted_flag_preset_enable_is_noop_default(self):
        # YouTube sets kalman_tracking=True (the parser default), so the
        # negative flags must stay off.
        args = _prepare(["--validate-config", "--preset", "YouTube (default)"])
        self.assertFalse(args.no_kalman)
        self.assertFalse(args.no_phash)
        self.assertFalse(args.no_scene_split)

    def test_explicit_threshold_equal_to_default_beats_preset(self):
        # TikTok preset sets detection_threshold 0.4; an explicit --threshold
        # 0.5 (which equals the parser default) must survive the preset merge.
        args = _prepare(["--validate-config", "--threshold", "0.5",
                         "--preset", "TikTok / Vertical short"])
        self.assertEqual(args.threshold, 0.5)

    def test_omitted_threshold_takes_preset_value(self):
        args = _prepare(["--validate-config",
                         "--preset", "TikTok / Vertical short"])
        self.assertEqual(args.threshold, 0.4)

    def test_explicit_threshold_equals_form_beats_preset(self):
        args = _prepare(["--validate-config", "--threshold=0.5",
                         "--preset", "TikTok / Vertical short"])
        self.assertEqual(args.threshold, 0.5)

    def test_explicit_no_kalman_overrides_preset(self):
        from backend import presets

        original = dict(presets.BUILTIN_PRESETS)
        presets.BUILTIN_PRESETS["__audit_test__"] = {
            "description": "test",
            "fields": {"kalman_tracking": True},
        }
        try:
            args = _prepare(["--validate-config", "--no-kalman",
                             "--preset", "__audit_test__"])
        finally:
            presets.BUILTIN_PRESETS.clear()
            presets.BUILTIN_PRESETS.update(original)
        self.assertTrue(args.no_kalman)


class CrashReporterRedactionTests(unittest.TestCase):
    def test_unc_path_is_redacted(self):
        from backend.crash_reporter import _path_scrub

        text = r"Traceback: opened \\NAS01\media\clientX\confidential.mkv here"
        scrubbed = _path_scrub(text)
        # The directory tree (server, share, sub-dirs) is stripped; like the
        # drive-letter branch, only the leaf filename is retained.
        self.assertNotIn("NAS01", scrubbed)
        self.assertNotIn("clientX", scrubbed)
        self.assertNotIn("media", scrubbed)
        self.assertIn("<path>", scrubbed)

    def test_drive_path_still_redacted(self):
        from backend.crash_reporter import _path_scrub

        scrubbed = _path_scrub(r"C:\Users\bob\secret\file.py")
        self.assertNotIn("bob", scrubbed)
        self.assertIn("<path>", scrubbed)


class WhisperFilterEscapeTests(unittest.TestCase):
    def test_semicolon_is_escaped_and_round_trips(self):
        from backend.whisper_fallback import (
            _escape_filter_value,
            _unescape_filter_value,
        )

        raw = r"C:\models\weird;name.bin"
        escaped = _escape_filter_value(raw)
        self.assertIn(r"\;", escaped)
        self.assertEqual(_unescape_filter_value(escaped), raw)


class DependencyVersionKeyTests(unittest.TestCase):
    def test_fourth_component_is_compared(self):
        from backend.dependency_caps import _version_gte

        # 5.0.0.80 is below the 5.0.0.93 floor and must NOT report satisfied.
        self.assertFalse(_version_gte("5.0.0.80", "5.0.0.93"))
        self.assertTrue(_version_gte("5.0.0.93", "5.0.0.93"))
        self.assertTrue(_version_gte("5.0.1.0", "5.0.0.93"))

    def test_three_component_comparison_unaffected(self):
        from backend.dependency_caps import _version_gte

        self.assertTrue(_version_gte("1.6.54", "1.6.54"))
        self.assertFalse(_version_gte("1.6.53", "1.6.54"))


class LamaBinarizeTests(unittest.TestCase):
    def test_soft_mask_is_binarized_before_model(self):
        from backend.inpainters._common import _binarize_mask

        soft = np.array([[0, 60, 127, 128, 200, 255]], dtype=np.uint8)
        out = _binarize_mask(soft)
        self.assertEqual(out.tolist(), [[0, 0, 0, 255, 255, 255]])

    def test_binarize_handles_three_channel(self):
        from backend.inpainters._common import _binarize_mask

        soft = np.zeros((2, 2, 3), dtype=np.uint8)
        soft[0, 0] = 200
        out = _binarize_mask(soft)
        self.assertEqual(out.shape, (2, 2))
        self.assertEqual(out[0, 0], 255)
        self.assertEqual(out[1, 1], 0)


class PresetApplySliderSyncTests(unittest.TestCase):
    """The Sensitivity slider is registered under the runtime-only
    ``_detection_threshold_pct`` attribute, which does not exist on a fresh
    config until the slider is dragged. Applying a preset must not crash and
    must sync the slider to the preset's detection_threshold."""

    def _stub_app(self, config):
        import unittest.mock
        from types import SimpleNamespace
        from gui.app import VideoSubtitleRemoverApp

        app = VideoSubtitleRemoverApp.__new__(VideoSubtitleRemoverApp)
        app.config = config
        app.preset_var = SimpleNamespace(get=lambda: "YouTube (default)")
        app.mode_var = SimpleNamespace(set=lambda *_: None)
        app.mode_picker = SimpleNamespace(set=lambda *_: None)
        app._on_mode_changed = lambda *_: None
        app._update_status = lambda *a, **k: None
        slider_calls = []
        label_calls = []
        slider = SimpleNamespace(set_value=lambda v: slider_calls.append(v))
        label = SimpleNamespace(config=lambda **k: label_calls.append(k))
        app._settings_slider_by_attr = {"_detection_threshold_pct": (slider, label)}
        return app, slider_calls, label_calls

    def test_apply_preset_on_fresh_config_syncs_sensitivity(self):
        import unittest.mock
        from types import SimpleNamespace
        from gui.config import InpaintMode

        config = SimpleNamespace(
            mode=InpaintMode.STTN,
            detection_threshold=0.55,
        )  # deliberately no _detection_threshold_pct
        app, slider_calls, _ = self._stub_app(config)
        with unittest.mock.patch("gui.settings_controller.apply_preset",
                                 return_value=True), \
                unittest.mock.patch("gui.settings_controller.save_settings"):
            app._on_preset_applied()
        self.assertEqual(getattr(config, "_detection_threshold_pct"), 55)
        self.assertEqual(slider_calls, [55])


class SafeImreadPrecomputedFlagTests(unittest.TestCase):
    """A caller reading many PNGs can resolve the libpng status once and pass
    it in, skipping the per-frame status evaluation, without changing the
    default per-call behavior."""

    def test_precomputed_true_diverts_to_pillow_without_status_call(self):
        import unittest.mock
        from PIL import Image
        from backend import safe_image
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.png"
            Image.new("RGB", (2, 1), (10, 20, 30)).save(path)
            with unittest.mock.patch(
                    "backend.safe_image.opencv_libpng_status") as status:
                frame = safe_image.safe_imread(path, png_vulnerable=True)
                status.assert_not_called()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (1, 2, 3))

    def test_precomputed_false_uses_cv2(self):
        import unittest.mock
        import numpy as _np
        from backend import safe_image

        expected = _np.zeros((1, 1, 3), dtype=_np.uint8)
        with unittest.mock.patch(
                "backend.safe_image.opencv_libpng_status") as status:
            with unittest.mock.patch("cv2.imread", return_value=expected) as imread:
                frame = safe_image.safe_imread("s.png", png_vulnerable=False)
            status.assert_not_called()
        self.assertIs(frame, expected)
        imread.assert_called_once_with("s.png")

    def test_default_still_resolves_status_per_call(self):
        import unittest.mock
        import numpy as _np
        from backend import safe_image

        expected = _np.zeros((1, 1, 3), dtype=_np.uint8)
        with unittest.mock.patch(
                "backend.safe_image.opencv_libpng_status",
                return_value={"vulnerable": False}) as status:
            with unittest.mock.patch("cv2.imread", return_value=expected):
                safe_image.safe_imread("s.png")
            status.assert_called()


class BackendOnlyModeNoticeTests(unittest.TestCase):
    """A settings.json carrying a backend-only inpaint mode (e.g. migan) must
    surface a load notice instead of silently downgrading to STTN."""

    def _clear(self):
        from gui import config as gcfg
        gcfg.consume_settings_load_notice()

    def test_backend_only_mode_emits_notice(self):
        from gui import config as gcfg
        self._clear()
        gcfg._notice_if_backend_only_mode("migan")
        notice = gcfg.consume_settings_load_notice()
        self.assertIsNotNone(notice)
        self.assertIn("migan", notice)

    def test_gui_mode_emits_no_notice(self):
        from gui import config as gcfg
        self._clear()
        gcfg._notice_if_backend_only_mode("lama")
        self.assertIsNone(gcfg.consume_settings_load_notice())

    def test_unknown_mode_emits_no_notice(self):
        from gui import config as gcfg
        self._clear()
        gcfg._notice_if_backend_only_mode("banana")
        self.assertIsNone(gcfg.consume_settings_load_notice())

    def test_blank_and_nonstring_are_ignored(self):
        from gui import config as gcfg
        self._clear()
        gcfg._notice_if_backend_only_mode("")
        gcfg._notice_if_backend_only_mode(None)
        self.assertIsNone(gcfg.consume_settings_load_notice())


if __name__ == "__main__":
    unittest.main()
