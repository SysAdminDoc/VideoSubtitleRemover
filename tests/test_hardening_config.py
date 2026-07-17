import json
import os
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path


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
                "subtitle_region_spans": [
                    {"rect": [2, 4, 20, 24], "start": "1.5", "end": "4"},
                    {"rect": [8, 8, 8, 12], "start": "2", "end": "3"},
                    [[4, 5, 30, 40], "bad", "1"],
                ],
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
        self.assertEqual(
            cfg.subtitle_region_spans,
            [
                {"rect": (2, 4, 20, 24), "start": 1.5, "end": 4.0},
                {"rect": (4, 5, 30, 40), "start": 0.0, "end": 1.0},
            ],
        )
        self.assertEqual(cfg.detection_lang, "ja")
        self.assertEqual(cfg.detection_threshold, 0.9)
        self.assertEqual(cfg.time_start, 15.0)
        self.assertEqual(cfg.time_end, 0.0)
        self.assertEqual(cfg.output_quality, 35)
        self.assertEqual(cfg.phash_skip_distance, 0)
        self.assertEqual(cfg.colour_tune_tolerance, 100)
        self.assertEqual(cfg.window_geometry, "")

    def test_load_settings_falls_back_cleanly_from_non_object_json(self):
        from gui import config as _gui_config
        with tempfile.TemporaryDirectory() as tmpdir:
            original = _gui_config.SETTINGS_FILE
            try:
                _gui_config.SETTINGS_FILE = Path(tmpdir) / "settings.json"
                _gui_config.SETTINGS_FILE.write_text(json.dumps(["bad"]), encoding="utf-8")
                cfg = gui.load_settings()
                self.assertIsInstance(cfg, gui.ProcessingConfig)
                self.assertEqual(cfg.mode, gui.InpaintMode.STTN)
            finally:
                _gui_config.SETTINGS_FILE = original

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
                    app._shutdown_ui_resources()
                    app.root.destroy()
            except Exception:
                pass


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
        from backend.config_schema import CONFIG_SCHEMA_VERSION
        self.assertEqual(payload.get("config_schema_version"), CONFIG_SCHEMA_VERSION)

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


class ConfigFuzzTests(unittest.TestCase):
    """Deterministic fuzz pass over ProcessingConfig.from_dict() (GUI) and
    normalize_processing_config() (backend).

    Formalises the invariant proved one-off by CoerceHardeningTests:
    *no input shape crashes the loader*. We don't pull in Hypothesis to
    keep the dependency closure minimal; a seeded random.Random walks the
    cross product of (known field name) x (small pool of pathological
    values) and asserts post-conditions on the normalised result.
    """

    BAD_VALUES = [
        None, "", "garbage", "true", "false", "1.5e3", "-1", "NaN", "Inf",
        0, 1, -1, 9999999, -9999999,
        float("nan"), float("inf"), float("-inf"),
        -1e30, 1e30,
        [], {}, [None], [1, 2, 3, 4], (5, 6, 7, 8),
        {"x": 1}, True, False,
        "STTN", "lama", "AUTO", "pro painter", "cuda:0", "cpu", "directml",
    ]

    GUI_FIELDS = [
        "mode", "use_gpu", "gpu_id",
        "sttn_skip_detection", "sttn_neighbor_stride", "sttn_reference_length",
        "sttn_max_load_num", "lama_super_fast",
        "subtitle_area", "subtitle_areas", "subtitle_region_spans",
        "detection_lang",
        "detection_threshold", "detection_frame_skip",
        "mask_dilate_px", "mask_feather_px", "edge_ring_px",
        "tbe_enable", "tbe_min_coverage", "tbe_use_median", "tbe_flow_warp",
        "tbe_scene_cut_split", "tbe_scene_cut_threshold",
        "auto_band", "export_srt", "export_mask_video",
        "mask_export_format", "mask_import_path", "mask_import_mode",
        "adaptive_batch", "auto_exposure_threshold",
        "deinterlace", "deinterlace_auto", "keyframe_detection",
        "quality_report", "kalman_tracking", "kalman_iou_threshold",
        "kalman_max_age", "phash_skip_enable", "phash_skip_distance",
        "colour_tune_enable", "colour_tune_tolerance",
        "time_start", "time_end",
        "preserve_audio", "output_format", "output_quality", "use_hw_encode",
        "d3d12_accel",
        "translation_enabled", "translation_srt", "translation_source_srt",
        "translation_provider", "translation_source_lang",
        "translation_target_lang", "translation_command", "translation_style",
        "translation_timeout_seconds",
        # v3.13 GUI-exposed fields (B-1 + F-8).
        "loudnorm_target", "multi_audio_passthrough",
        "decode_hw_accel", "prefetch_decode", "prefetch_queue_size",
        "input_fps", "quality_report_sheet", "rife_fast_stride",
        "remove_subtitles", "remove_chyrons", "chyron_min_hits",
        "karaoke_grouping", "karaoke_x_gap_px", "karaoke_y_overlap",
        "output_codec",
        "window_geometry", "adv_panel_open", "log_panel_open",
        "text_scale_percent",
        "onboarding_seen", "vsr_settings_format",
    ]

    BACKEND_FIELDS = GUI_FIELDS + [
        "device",
    ]

    def _random_payload(self, rng, fields, max_keys=8):
        n = rng.randint(0, max_keys)
        return {rng.choice(fields): rng.choice(self.BAD_VALUES) for _ in range(n)}

    def test_gui_from_dict_normalize_never_crashes(self):
        import random as _random
        rng = _random.Random(0xC0FFEE)
        for i in range(1500):
            payload = self._random_payload(rng, self.GUI_FIELDS)
            try:
                cfg = gui.ProcessingConfig.from_dict(payload).normalized()
            except Exception as exc:
                self.fail(f"iter={i} payload={payload!r} raised {exc!r}")
            # Numeric invariants -- finite + within declared bounds.
            self.assertTrue(0.1 <= cfg.detection_threshold <= 0.9)
            self.assertTrue(0 <= cfg.output_quality <= 51)
            self.assertTrue(0 <= cfg.phash_skip_distance <= 64)
            self.assertGreaterEqual(cfg.time_start, 0.0)
            self.assertGreaterEqual(cfg.time_end, 0.0)
            if cfg.time_end:
                self.assertGreaterEqual(cfg.time_end, cfg.time_start)
            # B-1 + F-8 invariants on the newly exposed fields.
            self.assertTrue(cfg.loudnorm_target == 0.0
                            or -70.0 <= cfg.loudnorm_target <= -5.0)
            self.assertIn(cfg.decode_hw_accel,
                          {"off", "auto", "any", "d3d11", "vaapi", "mfx",
                           "pynv", "nvdec"})
            self.assertGreaterEqual(cfg.rife_fast_stride, 0)
            self.assertLessEqual(cfg.rife_fast_stride, 60)
            self.assertIn(cfg.output_codec, {"h264", "h265", "av1", "vvc"})
            self.assertIsInstance(cfg.multi_audio_passthrough, bool)
            self.assertGreaterEqual(cfg.input_fps, 1.0)
            self.assertLessEqual(cfg.input_fps, 240.0)
            self.assertIn(cfg.text_scale_percent, {100, 125, 150, 175, 200})

    def test_backend_normalize_never_crashes(self):
        import random as _random
        rng = _random.Random(0xBADF00D)
        for i in range(1500):
            payload = self._random_payload(rng, self.BACKEND_FIELDS)
            try:
                cfg = processor.ProcessingConfig()
                # Apply the random payload field-by-field (mimics how the
                # JSON overlay loader does it in main()).
                for k, v in payload.items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
                cfg = processor.normalize_processing_config(cfg)
            except Exception as exc:
                self.fail(f"iter={i} payload={payload!r} raised {exc!r}")
            # Numeric invariants.
            self.assertTrue(0.1 <= cfg.detection_threshold <= 1.0)
            self.assertTrue(0 <= cfg.output_quality <= 51)
            self.assertTrue(0 <= cfg.phash_skip_distance <= 64)
            self.assertTrue(cfg.loudnorm_target == 0.0 or
                            -70.0 <= cfg.loudnorm_target <= -5.0)
            self.assertIn(cfg.decode_hw_accel,
                          {"off", "auto", "any", "d3d11", "vaapi", "mfx",
                           "pynv", "nvdec"})
            self.assertGreaterEqual(cfg.rife_fast_stride, 0)
            self.assertLessEqual(cfg.rife_fast_stride, 60)
            self.assertIsInstance(cfg.multi_audio_passthrough, bool)


class GuiToBackendFieldWiringTests(unittest.TestCase):
    """Backend fields must round-trip through the GUI
    dataclass without being silently dropped, and reach the backend
    config when _process_item builds the BackendConfig."""

    EXPECTED_GUI_FIELDS = (
        "loudnorm_target", "multi_audio_passthrough", "decode_hw_accel",
        "prefetch_decode", "prefetch_queue_size", "input_fps",
        "quality_report_sheet", "rife_fast_stride",
        "remove_subtitles", "remove_chyrons",
        "chyron_min_hits", "karaoke_grouping", "karaoke_x_gap_px",
        "karaoke_y_overlap",
    )

    def test_backend_fields_declared_on_gui_dataclass(self):
        cfg = gui.ProcessingConfig()
        for name in self.EXPECTED_GUI_FIELDS:
            self.assertTrue(
                hasattr(cfg, name),
                f"GUI ProcessingConfig is missing field {name!r}; was "
                "removed or never wired through B-1.",
            )

    def test_backend_fields_persist_through_to_dict(self):
        cfg = gui.ProcessingConfig(
            loudnorm_target=-14.0,
            multi_audio_passthrough=False,
            decode_hw_accel="d3d11",
            prefetch_decode=False,
            prefetch_queue_size=24,
            input_fps=30.0,
            quality_report_sheet=True,
            rife_fast_stride=3,
            remove_subtitles=False,
            remove_chyrons=False,
            chyron_min_hits=120,
            karaoke_grouping=True,
            karaoke_x_gap_px=15,
            karaoke_y_overlap=0.4,
        )
        payload = cfg.to_dict()
        for name in self.EXPECTED_GUI_FIELDS:
            self.assertIn(name, payload, f"{name} dropped from to_dict")
        # Round trip back through from_dict
        restored = gui.ProcessingConfig.from_dict(payload)
        self.assertEqual(restored.loudnorm_target, -14.0)
        self.assertEqual(restored.decode_hw_accel, "d3d11")
        self.assertEqual(restored.rife_fast_stride, 3)
        self.assertTrue(restored.karaoke_grouping)
        self.assertEqual(restored.chyron_min_hits, 120)

    def test_quality_sheet_implies_quality_report(self):
        cfg = gui.ProcessingConfig(quality_report=False, quality_report_sheet=True)
        cfg.normalized()
        self.assertTrue(cfg.quality_report,
                        "enabling the sheet must enable the numeric report")

    def test_loudnorm_out_of_range_resets_to_zero(self):
        cfg = gui.ProcessingConfig(loudnorm_target=99.0).normalized()
        self.assertEqual(cfg.loudnorm_target, 0.0)
        cfg = gui.ProcessingConfig(loudnorm_target=-200.0).normalized()
        self.assertEqual(cfg.loudnorm_target, 0.0)

    def test_decode_hw_accel_garbage_resets_to_off(self):
        cfg = gui.ProcessingConfig(decode_hw_accel="cuda-experimental").normalized()
        self.assertEqual(cfg.decode_hw_accel, "off")

    def test_from_dict_unknown_keys_are_ignored(self):
        cfg = gui.ProcessingConfig.from_dict(
            {"mode": "STTN", "totally_unknown_field": 7}
        )
        self.assertEqual(cfg.mode, gui.InpaintMode.STTN)

    def test_soft_subtitle_summary_formats_tracks(self):
        summary = gui._format_soft_subtitle_summary([
            {
                "index": 2,
                "codec_name": "subrip",
                "language": "eng",
                "title": "SDH",
                "default": True,
                "forced": False,
            },
            {
                "index": 3,
                "codec_name": "ass",
                "language": "",
                "title": "",
                "default": False,
                "forced": True,
            },
        ])
        self.assertIn("2 embedded subtitle tracks", summary)
        self.assertIn("eng/subrip (default)", summary)
        self.assertIn("und/ass (forced)", summary)


class PresetLibraryTests(unittest.TestCase):
    """F-10: built-in presets must be shared between the GUI and the CLI
    so `python -m backend.processor --preset NAME` resolves to the same
    payload the GUI's picker would apply."""

    def test_builtin_presets_exposed(self):
        from backend import presets
        self.assertIn("YouTube (default)", presets.BUILTIN_PRESETS)
        self.assertIn("Anime / Animation", presets.BUILTIN_PRESETS)
        self.assertIn("Film / Live action", presets.BUILTIN_PRESETS)
        self.assertIn("Fast", presets.BUILTIN_PRESETS)
        self.assertIn("Logo / Watermark removal", presets.BUILTIN_PRESETS)

    def test_fast_preset_bounds_expensive_work(self):
        from backend import presets
        fields = presets.preset_fields("Fast")
        self.assertEqual(fields["mode"], "STTN")
        self.assertEqual(fields["detection_frame_skip"], 5)
        self.assertTrue(fields["phash_skip_enable"])
        self.assertEqual(fields["phash_skip_distance"], 4)
        self.assertFalse(fields["tbe_flow_warp"])

    def test_film_preset_uses_quality_first_auto_routing(self):
        from backend import presets
        fields = presets.preset_fields("Film / Live action")
        self.assertEqual(fields["mode"], "Auto")
        self.assertTrue(fields["tbe_flow_warp"])
        self.assertTrue(fields["tbe_scene_cut_split"])

    def test_preset_fields_returns_dict_or_none(self):
        from backend import presets
        fields = presets.preset_fields("YouTube (default)")
        self.assertIsInstance(fields, dict)
        self.assertEqual(fields["mode"], "STTN")
        self.assertIsNone(presets.preset_fields("not-a-real-preset"))

    def test_list_preset_names_returns_builtins(self):
        from backend import presets
        names = presets.list_preset_names()
        self.assertIn("YouTube (default)", names)
        self.assertIn("News / Chyron (bottom-third)", names)

    def test_gui_uses_shared_builtin_table(self):
        # The GUI module must import the same dict so a future change to
        # the table cannot drift between the GUI's picker and the CLI's
        # --preset resolver.
        from backend import presets
        self.assertIs(gui.BUILTIN_PRESETS, presets.BUILTIN_PRESETS)


class DependencyFloorTests(unittest.TestCase):
    """Verify minimum dependency versions across all install surfaces."""

    def _read_requirements(self):
        root = Path(__file__).resolve().parents[1]
        return (root / "requirements.txt").read_text(encoding="utf-8")

    def test_pillow_floor_is_12_3_0_in_requirements(self):
        text = self._read_requirements()
        self.assertIn("Pillow>=12.3.0", text)

    def test_pillow_floor_is_12_3_0_in_setup_py(self):
        root = Path(__file__).resolve().parents[1]
        setup_text = (root / "setup.py").read_text(encoding="utf-8")
        self.assertIn("Pillow>=12.3.0", setup_text)

    def test_pillow_floor_is_12_3_0_in_build_workflow(self):
        root = Path(__file__).resolve().parents[1]
        workflow_path = root / ".github" / "workflows" / "build.yml"
        if not workflow_path.exists():
            self.skipTest("GitHub Actions workflow is absent in local-build mode")
        workflow = workflow_path.read_text(encoding="utf-8")
        self.assertIn("Pillow>=12.3.0", workflow)


class CanonicalConfigSchemaTests(unittest.TestCase):
    def test_registry_round_trips_every_backend_field_losslessly(self):
        from backend.config import (
            InpaintMode,
            ProcessingConfig,
            normalize_processing_config,
        )
        from backend.config_schema import (
            apply_backend_payload,
            processing_field_names,
            serialize_backend_config,
        )

        source = normalize_processing_config(ProcessingConfig(
            mode=InpaintMode.AUTO,
            device="cpu",
            work_directory="D:/vsr-work",
            detection_lang="ja",
            subtitle_area=(4, 8, 640, 700),
            subtitle_areas=[(4, 8, 640, 700), (20, 30, 500, 600)],
            subtitle_region_spans=[
                {"rect": (4, 8, 640, 700), "start": 1.25, "end": 5.5},
            ],
            manual_mask_corrections=[
                {"polygons": [[1, 2, 10, 2, 10, 12]], "start": 2.0, "end": 3.0},
            ],
            confidence_weighted_dilation=True,
            confidence_dilation_scale=2.25,
            whisper_vad_model="models/vad.onnx",
            whisper_vad_threshold=0.7,
            whisper_min_speech_duration=0.4,
            watermark_image="branding/logo.png",
            watermark_position="top-left",
            watermark_opacity=0.65,
            restyle_subtitle="captions/clean.ass",
            gpu_oom_recovery=False,
            output_codec="h265",
            batch_max_retries=3,
        ))
        payload = serialize_backend_config(source)
        restored = apply_backend_payload(ProcessingConfig(), payload)

        self.assertEqual(tuple(payload), processing_field_names())
        self.assertEqual(serialize_backend_config(restored), payload)

    def test_gui_projection_covers_backend_schema_and_per_item_overrides(self):
        from backend.config_schema import (
            gui_to_backend_config,
            validate_gui_schema,
        )
        from gui.config import ProcessingConfig as GuiConfig

        validate_gui_schema(GuiConfig)
        gui_config = GuiConfig(
            use_gpu=False,
            work_directory="D:/vsr-work",
            manual_mask_corrections=[
                {"polygons": [[1, 2, 10, 2, 10, 12]], "start": 0.0, "end": 2.0},
            ],
            confidence_dilation_scale=2.0,
            whisper_vad_threshold=0.8,
            watermark_image="logo.png",
            gpu_oom_recovery=False,
        )
        backend_config = gui_to_backend_config(gui_config)

        self.assertEqual(backend_config.device, "cpu")
        self.assertEqual(backend_config.work_directory, "D:/vsr-work")
        self.assertEqual(backend_config.manual_mask_corrections,
                         gui_config.manual_mask_corrections)
        self.assertEqual(backend_config.confidence_dilation_scale, 2.0)
        self.assertEqual(backend_config.whisper_vad_threshold, 0.8)
        self.assertEqual(backend_config.watermark_image, "logo.png")
        self.assertFalse(backend_config.gpu_oom_recovery)

    def test_generated_cli_overrides_round_trip_non_default_fields(self):
        from backend.config import ProcessingConfig, normalize_processing_config
        from backend.config_schema import (
            apply_backend_payload,
            backend_config_cli_args,
            parse_cli_assignments,
            serialize_backend_config,
        )

        source = normalize_processing_config(ProcessingConfig(
            device="cpu",
            work_directory="D:/vsr-work",
            manual_mask_corrections=[
                {"polygons": [[0, 0, 8, 0, 8, 8]], "start": 1.0, "end": 4.0},
            ],
            confidence_dilation_scale=2.5,
            gpu_oom_recovery=False,
            restyle_subtitle="caption.ass",
        ))
        args = backend_config_cli_args(source)
        assignments = [args[index + 1] for index, value in enumerate(args)
                       if value == "--set"]
        restored = apply_backend_payload(
            ProcessingConfig(), parse_cli_assignments(assignments))

        self.assertEqual(serialize_backend_config(restored),
                         serialize_backend_config(source))

    def test_validate_config_cli_exposes_complete_registry(self):
        from backend.config_schema import (
            CONFIG_SCHEMA_VERSION,
            processing_field_names,
        )

        completed = subprocess.run(
            [
                sys.executable, "-m", "backend.processor", "--validate-config",
                "--config-schema-version", str(CONFIG_SCHEMA_VERSION),
                "--work-dir", "D:/vsr-work",
                "--set", 'manual_mask_corrections=[{"polygons":[[0,0,8,0,8,8]]}]',
                "--set", "gpu_oom_recovery=false",
            ],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)["resolved_config"]
        self.assertEqual(set(payload), set(processing_field_names()))
        self.assertFalse(payload["gpu_oom_recovery"])
        self.assertTrue(payload["manual_mask_corrections"])
        self.assertEqual(payload["work_directory"], "D:/vsr-work")



if __name__ == "__main__":
    unittest.main()
