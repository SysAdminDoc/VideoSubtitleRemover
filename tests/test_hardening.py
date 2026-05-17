import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import VideoSubtitleRemover as gui
from backend import processor


def _has_display() -> bool:
    """Return True if a GUI display is available."""
    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class GuiConfigHardeningTests(unittest.TestCase):
    def test_processing_config_from_dict_normalizes_invalid_values(self):
        cfg = gui.ProcessingConfig.from_dict(
            {
                "mode": "pro painter",
                "use_gpu": "false",
                "gpu_id": "-4",
                "sttn_neighbor_stride": "999",
                "sttn_reference_length": "1",
                "sttn_max_load_num": "-20",
                "subtitle_area": [10, 20, 5, 30],
                "subtitle_areas": [[1, 2, 9, 12], ["bad"], [8, 8, 8, 20]],
                "detection_lang": " JA ",
                "detection_threshold": "2.5",
                "time_start": "15",
                "time_end": "4",
                "output_quality": "999",
                "phash_skip_distance": "-2",
                "colour_tune_tolerance": "250",
                "window_geometry": 123,
            }
        )

        self.assertEqual(cfg.mode, gui.InpaintMode.PROPAINTER)
        self.assertFalse(cfg.use_gpu)
        self.assertEqual(cfg.gpu_id, 0)
        self.assertEqual(cfg.sttn_neighbor_stride, 30)
        self.assertEqual(cfg.sttn_reference_length, 5)
        self.assertEqual(cfg.sttn_max_load_num, 10)
        self.assertIsNone(cfg.subtitle_area)
        self.assertEqual(cfg.subtitle_areas, [(1, 2, 9, 12)])
        self.assertEqual(cfg.detection_lang, "ja")
        self.assertEqual(cfg.detection_threshold, 0.9)
        self.assertEqual(cfg.time_start, 15.0)
        self.assertEqual(cfg.time_end, 0.0)
        self.assertEqual(cfg.output_quality, 35)
        self.assertEqual(cfg.phash_skip_distance, 0)
        self.assertEqual(cfg.colour_tune_tolerance, 100)
        self.assertEqual(cfg.window_geometry, "")

    def test_load_settings_falls_back_cleanly_from_non_object_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            original = gui.SETTINGS_FILE
            try:
                gui.SETTINGS_FILE = Path(tmpdir) / "settings.json"
                gui.SETTINGS_FILE.write_text(json.dumps(["bad"]), encoding="utf-8")
                cfg = gui.load_settings()
                self.assertIsInstance(cfg, gui.ProcessingConfig)
                self.assertEqual(cfg.mode, gui.InpaintMode.STTN)
            finally:
                gui.SETTINGS_FILE = original

    @unittest.skipUnless(_has_display(), "No display available -- skipping GUI test")
    def test_on_processing_complete_during_shutdown_skips_summary_ui(self):
        app = gui.VideoSubtitleRemoverApp()
        try:
            app.is_processing = True
            app._shutdown_started = True
            app._show_batch_summary = lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("summary should not open during shutdown")
            )
            app._notify_completion = lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("notifications should not fire during shutdown")
            )

            app._on_processing_complete()

            self.assertFalse(app.is_processing)
            self.assertIsNone(app._processing_thread)
            self.assertFalse(app.cancel_event.is_set())
        finally:
            try:
                if app.root.winfo_exists():
                    app.root.destroy()
            except Exception:
                pass


class BackendHardeningTests(unittest.TestCase):
    def test_normalize_processing_config_clamps_unsafe_values(self):
        cfg = processor.normalize_processing_config(
            processor.ProcessingConfig(
                mode="AUTO",
                device="cuda:-3",
                sttn_max_load_num="-2",
                subtitle_area=[4, 4, 2, 10],
                subtitle_areas=[[2, 3, 10, 12], ["bad"]],
                detection_threshold="9",
                detection_lang=" EN ",
                detection_frame_skip="-5",
                output_quality="99",
                time_start="12",
                time_end="4",
                kalman_iou_threshold="-1",
                phash_skip_distance="80",
                colour_tune_tolerance="-7",
            )
        )

        self.assertEqual(cfg.mode, processor.InpaintMode.AUTO)
        self.assertEqual(cfg.device, "cuda:0")
        self.assertEqual(cfg.sttn_max_load_num, 1)
        self.assertIsNone(cfg.subtitle_area)
        self.assertEqual(cfg.subtitle_areas, [(2, 3, 10, 12)])
        self.assertEqual(cfg.detection_threshold, 1.0)
        self.assertEqual(cfg.detection_lang, "en")
        self.assertEqual(cfg.detection_frame_skip, 0)
        self.assertEqual(cfg.output_quality, 51)
        self.assertEqual(cfg.time_end, 0.0)
        self.assertEqual(cfg.kalman_iou_threshold, 0.0)
        self.assertEqual(cfg.phash_skip_distance, 64)
        self.assertEqual(cfg.colour_tune_tolerance, 0)

    def test_load_json_config_rejects_non_object_payloads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps(["bad"]), encoding="utf-8")
            with self.assertRaises(ValueError):
                processor._load_json_config(str(config_path))

    def test_choose_available_output_path_avoids_existing_and_reserved_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base = root / "clip_no_sub.mp4"
            base.write_text("taken", encoding="utf-8")
            reserved = {processor._path_key(root / "clip_no_sub(2).mp4")}
            chosen = processor._choose_available_output_path(base, reserved)
            self.assertEqual(chosen.name, "clip_no_sub(3).mp4")

    def test_copy_file_atomic_replaces_existing_output_without_leaking_temp_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.txt"
            output = root / "result.txt"
            source.write_text("fresh payload", encoding="utf-8")
            output.write_text("stale payload", encoding="utf-8")

            processor._copy_file_atomic(str(source), str(output))

            self.assertEqual(output.read_text(encoding="utf-8"), "fresh payload")
            leaked = [p.name for p in root.iterdir() if p.name.startswith(".result.")]
            self.assertEqual(leaked, [])

    def test_apply_auto_band_override_resets_stale_region_when_probe_finds_none(self):
        config = SimpleNamespace(subtitle_area=(10, 20, 30, 40), subtitle_areas=None)
        calls = []

        class FakeRemover:
            def __init__(self):
                self.config = config

            def detect_subtitle_band(self, input_path):
                calls.append(input_path)
                return None

        remover = FakeRemover()
        band = processor._apply_auto_band_override(
            remover,
            "clip-two.mp4",
            auto_band=True,
            base_subtitle_area=None,
            base_subtitle_areas=None,
        )

        self.assertIsNone(band)
        self.assertIsNone(remover.config.subtitle_area)
        self.assertEqual(calls, ["clip-two.mp4"])


class CoerceHardeningTests(unittest.TestCase):
    """Tests for NaN/inf guard in _coerce_int/_coerce_float and
    pre-sanitisation fixes in ProcessingConfig.from_dict."""

    def test_coerce_int_rejects_nan(self):
        result = gui._coerce_int(float("nan"), default=99)
        self.assertEqual(result, 99)

    def test_coerce_int_rejects_inf(self):
        result = gui._coerce_int(float("inf"), default=7, min_value=0, max_value=100)
        self.assertEqual(result, 7)

    def test_coerce_float_rejects_nan(self):
        result = gui._coerce_float(float("nan"), default=0.5, min_value=0.0, max_value=1.0)
        self.assertEqual(result, 0.5)

    def test_coerce_float_rejects_negative_inf(self):
        result = gui._coerce_float(float("-inf"), default=0.5)
        self.assertEqual(result, 0.5)

    def test_from_dict_subtitle_areas_non_iterable_value_falls_back_to_none(self):
        """subtitle_areas with a non-iterable root (e.g. integer) must not crash."""
        cfg = gui.ProcessingConfig.from_dict({"subtitle_areas": 42})
        self.assertIsNone(cfg.subtitle_areas)

    def test_from_dict_subtitle_area_non_iterable_falls_back_to_none(self):
        """subtitle_area with a scalar value must not crash."""
        cfg = gui.ProcessingConfig.from_dict({"subtitle_area": "bad"})
        self.assertIsNone(cfg.subtitle_area)


class BackendWriteSrtTests(unittest.TestCase):
    """Tests for _write_srt fps guard."""

    def _make_remover_with_entries(self, entries):
        """Construct a minimal SubtitleRemover-like object with SRT entries."""
        cfg = processor.ProcessingConfig()
        remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        remover.config = cfg
        remover._srt_entries = entries
        return remover

    def test_write_srt_uses_fallback_fps_for_zero(self):
        """fps=0.0 should fall back to 30.0 and not divide-by-zero."""
        import tempfile, os
        remover = self._make_remover_with_entries([(0, "Hello world")])
        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as f:
            path = f.name
        try:
            remover._write_srt(path, fps=0.0)
            content = open(path, encoding="utf-8").read()
            self.assertIn("00:00:00,033", content)  # frame 0 / 30 fps
        finally:
            os.unlink(path)

    def test_write_srt_uses_fallback_fps_for_tiny_value(self):
        """fps=0.001 (absurd) should also fall back to 30.0."""
        import tempfile, os
        remover = self._make_remover_with_entries([(0, "Test")])
        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as f:
            path = f.name
        try:
            remover._write_srt(path, fps=0.001)
            content = open(path, encoding="utf-8").read()
            # Should have sane timestamp, not a 1000-second-long cue
            self.assertIn("00:00:00", content)
            # The end timestamp at 30fps for frame 0 is 0.033s, nowhere near 1000s
            self.assertNotIn("00:16:", content)
        finally:
            os.unlink(path)


class LoudnormCoerceTests(unittest.TestCase):
    """normalize_processing_config must clamp loudnorm_target to valid
    LUFS, with 0.0 reserved as 'disabled'."""

    def test_zero_passes_through_as_disabled(self):
        cfg = processor.normalize_processing_config(
            processor.ProcessingConfig(loudnorm_target=0.0)
        )
        self.assertEqual(cfg.loudnorm_target, 0.0)

    def test_in_range_youtube_target_kept(self):
        cfg = processor.normalize_processing_config(
            processor.ProcessingConfig(loudnorm_target=-14.0)
        )
        self.assertEqual(cfg.loudnorm_target, -14.0)

    def test_in_range_broadcast_target_kept(self):
        cfg = processor.normalize_processing_config(
            processor.ProcessingConfig(loudnorm_target=-23.0)
        )
        self.assertEqual(cfg.loudnorm_target, -23.0)

    def test_out_of_range_silently_disables(self):
        """A value outside ffmpeg's loudnorm range (-70 to -5) is rejected
        as 0.0 (off) rather than crashing the encode."""
        for bad in (5.0, -100.0, -2.0):
            cfg = processor.normalize_processing_config(
                processor.ProcessingConfig(loudnorm_target=bad)
            )
            self.assertEqual(cfg.loudnorm_target, 0.0, f"bad={bad}")

    def test_nan_and_inf_become_zero(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            cfg = processor.normalize_processing_config(
                processor.ProcessingConfig(loudnorm_target=bad)
            )
            self.assertEqual(cfg.loudnorm_target, 0.0, f"bad={bad}")


class SettingsMigrationTests(unittest.TestCase):
    """Tests for _migrate_settings(): legacy payloads must round-trip into
    a current-format ProcessingConfig without losing user state, and the
    output of save_settings()/to_dict() must carry the version stamp."""

    def test_migrate_settings_stamps_missing_version(self):
        legacy = {"mode": "STTN", "detection_lang": "ja"}
        migrated = gui._migrate_settings(legacy)
        self.assertEqual(migrated.get("vsr_settings_format"), gui.VSR_SETTINGS_FORMAT)
        self.assertEqual(migrated["mode"], "STTN")
        self.assertEqual(migrated["detection_lang"], "ja")

    def test_migrate_settings_passes_through_current_version(self):
        current = {"vsr_settings_format": gui.VSR_SETTINGS_FORMAT, "mode": "LAMA"}
        migrated = gui._migrate_settings(current)
        self.assertEqual(migrated["vsr_settings_format"], gui.VSR_SETTINGS_FORMAT)

    def test_migrate_settings_accepts_future_version_without_loss(self):
        """A settings file written by a newer build is honoured as-is so we
        don't downgrade it on load."""
        future = {"vsr_settings_format": gui.VSR_SETTINGS_FORMAT + 5, "mode": "AUTO"}
        migrated = gui._migrate_settings(future)
        self.assertEqual(migrated["vsr_settings_format"], gui.VSR_SETTINGS_FORMAT + 5)
        self.assertEqual(migrated["mode"], "AUTO")

    def test_migrate_settings_handles_non_dict(self):
        self.assertEqual(gui._migrate_settings("oops"), {})
        self.assertEqual(gui._migrate_settings(None), {})
        self.assertEqual(gui._migrate_settings(["bad"]), {})

    def test_migrate_settings_handles_garbage_version_field(self):
        """A non-int version field should be treated as 0 and stamped."""
        garbage = {"vsr_settings_format": "lolwat", "mode": "STTN"}
        migrated = gui._migrate_settings(garbage)
        self.assertEqual(migrated["vsr_settings_format"], gui.VSR_SETTINGS_FORMAT)

    def test_to_dict_emits_current_version(self):
        cfg = gui.ProcessingConfig()
        payload = cfg.to_dict()
        self.assertEqual(payload.get("vsr_settings_format"), gui.VSR_SETTINGS_FORMAT)

    def test_load_settings_round_trips_legacy_file(self):
        """A v3.12-era settings.json (no version field) loads, gets stamped,
        and round-trips through from_dict without data loss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original = gui.SETTINGS_FILE
            try:
                gui.SETTINGS_FILE = Path(tmpdir) / "settings.json"
                legacy = {
                    "mode": "LAMA",
                    "detection_lang": "ja",
                    "subtitle_area": [10, 20, 100, 50],
                    "tbe_flow_warp": True,
                }
                gui.SETTINGS_FILE.write_text(json.dumps(legacy), encoding="utf-8")
                cfg = gui.load_settings()
                self.assertEqual(cfg.mode, gui.InpaintMode.LAMA)
                self.assertEqual(cfg.detection_lang, "ja")
                self.assertTrue(cfg.tbe_flow_warp)
            finally:
                gui.SETTINGS_FILE = original


class LoadJsonConfigTests(unittest.TestCase):
    def test_load_json_config_rejects_oversized_file(self):
        """Files larger than 1 MB should raise ValueError without being parsed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            big = Path(tmpdir) / "big.json"
            # Write >1 MB of valid JSON — use enough entries to exceed the cap
            big.write_text("{" + ", ".join(f'"{i}": {i}' for i in range(150_000)) + "}",
                           encoding="utf-8")
            self.assertGreater(big.stat().st_size, 1 * 1024 * 1024,
                               "test fixture must be >1 MB")
            with self.assertRaises(ValueError):
                processor._load_json_config(str(big))


if __name__ == "__main__":
    unittest.main()
