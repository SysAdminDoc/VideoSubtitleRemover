import json
import logging
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


class DecodeHwAccelCoerceTests(unittest.TestCase):
    """decode_hw_accel must clamp to the allowed token set; anything else
    silently disables the hint so we never pass garbage to cv2."""

    def test_default_is_off(self):
        cfg = processor.normalize_processing_config(processor.ProcessingConfig())
        self.assertEqual(cfg.decode_hw_accel, "off")

    def test_known_tokens_kept(self):
        for token in ("off", "auto", "any", "d3d11", "vaapi", "mfx"):
            cfg = processor.normalize_processing_config(
                processor.ProcessingConfig(decode_hw_accel=token)
            )
            self.assertEqual(cfg.decode_hw_accel, token)

    def test_unknown_token_becomes_off(self):
        cfg = processor.normalize_processing_config(
            processor.ProcessingConfig(decode_hw_accel="cuda-experimental")
        )
        self.assertEqual(cfg.decode_hw_accel, "off")

    def test_mixed_case_token_normalised(self):
        cfg = processor.normalize_processing_config(
            processor.ProcessingConfig(decode_hw_accel="D3D11")
        )
        self.assertEqual(cfg.decode_hw_accel, "d3d11")


class MultiAudioPassthroughTests(unittest.TestCase):
    def test_default_is_on(self):
        cfg = processor.normalize_processing_config(processor.ProcessingConfig())
        self.assertTrue(cfg.multi_audio_passthrough)

    def test_explicit_off(self):
        cfg = processor.normalize_processing_config(
            processor.ProcessingConfig(multi_audio_passthrough=False)
        )
        self.assertFalse(cfg.multi_audio_passthrough)


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


class PrefetchReaderTests(unittest.TestCase):
    """_PrefetchReader contract:
    - returns the same frames in the same order as the underlying cap
    - read() returns (False, None) after exhaustion or release
    - release() stops the worker even when the queue is full
    """

    class _FakeCap:
        """Minimal cv2.VideoCapture stand-in for unit tests. Returns
        deterministic 'frames' (small numpy arrays) one per read until
        n_frames is reached."""

        def __init__(self, n_frames: int):
            self._n = n_frames
            self._i = 0
            self._released = False
            self._lock = __import__("threading").Lock()

        def isOpened(self):
            return not self._released

        def read(self):
            with self._lock:
                if self._released or self._i >= self._n:
                    return False, None
                import numpy as _np
                frame = _np.full((4, 4, 3), self._i, dtype=_np.uint8)
                self._i += 1
                return True, frame

        def get(self, _prop):
            return 0

        def release(self):
            with self._lock:
                self._released = True

    def test_read_yields_every_frame_in_order(self):
        cap = self._FakeCap(n_frames=20)
        reader = processor._PrefetchReader(cap, max_frames=20, queue_size=4)
        try:
            seen = []
            while True:
                ret, frame = reader.read()
                if not ret:
                    break
                seen.append(int(frame.flat[0]))
            self.assertEqual(seen, list(range(20)))
        finally:
            reader.release()

    def test_release_stops_worker_with_full_queue(self):
        # A worker that has filled the queue must still exit on release().
        cap = self._FakeCap(n_frames=1000)
        reader = processor._PrefetchReader(cap, max_frames=1000, queue_size=4)
        # Don't consume; let the queue fill, then release.
        import time as _time
        _time.sleep(0.05)
        reader.release()
        # Thread must have stopped within the release() join window.
        self.assertFalse(reader._thread.is_alive())

    def test_read_after_exhaustion_is_idempotent(self):
        cap = self._FakeCap(n_frames=3)
        reader = processor._PrefetchReader(cap, max_frames=3, queue_size=2)
        try:
            for _ in range(3):
                ret, _ = reader.read()
                self.assertTrue(ret)
            # After exhaustion, repeated reads keep returning (False, None).
            for _ in range(5):
                ret, frame = reader.read()
                self.assertFalse(ret)
                self.assertIsNone(frame)
        finally:
            reader.release()


class JsonLineLogHandlerTests(unittest.TestCase):
    """JsonLineLogHandler must write exactly one JSON record per emit,
    include the level / logger / msg / ts fields, and capture exception
    text when the record carries exc_info."""

    def test_emit_writes_one_json_record_per_line(self):
        import io
        sink = io.StringIO()
        handler = processor.JsonLineLogHandler(sink)
        record = logging.LogRecord(
            "vsr_test", logging.WARNING, __file__, 42,
            "hello %s", ("world",), None,
        )
        handler.emit(record)
        lines = sink.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        payload = json.loads(lines[0])
        self.assertEqual(payload["level"], "WARNING")
        self.assertEqual(payload["msg"], "hello world")
        self.assertEqual(payload["logger"], "vsr_test")
        self.assertIn("ts", payload)
        self.assertNotIn("exc", payload)

    def test_emit_includes_exception_text_when_present(self):
        import io
        sink = io.StringIO()
        handler = processor.JsonLineLogHandler(sink)
        try:
            raise RuntimeError("kaboom")
        except RuntimeError:
            record = logging.LogRecord(
                "vsr_test", logging.ERROR, __file__, 42,
                "fell over", (), sys.exc_info(),
            )
        handler.emit(record)
        payload = json.loads(sink.getvalue())
        self.assertIn("RuntimeError", payload["exc"])
        self.assertIn("kaboom", payload["exc"])


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
        "subtitle_area", "subtitle_areas", "detection_lang",
        "detection_threshold", "detection_frame_skip",
        "mask_dilate_px", "mask_feather_px", "edge_ring_px",
        "tbe_enable", "tbe_min_coverage", "tbe_use_median", "tbe_flow_warp",
        "tbe_scene_cut_split", "tbe_scene_cut_threshold",
        "auto_band", "export_srt", "export_mask_video",
        "adaptive_batch", "auto_exposure_threshold",
        "deinterlace", "deinterlace_auto", "keyframe_detection",
        "quality_report", "kalman_tracking", "kalman_iou_threshold",
        "kalman_max_age", "phash_skip_enable", "phash_skip_distance",
        "colour_tune_enable", "colour_tune_tolerance",
        "time_start", "time_end",
        "preserve_audio", "output_format", "output_quality", "use_hw_encode",
        "window_geometry", "adv_panel_open", "log_panel_open",
        "onboarding_seen", "vsr_settings_format",
    ]

    BACKEND_FIELDS = GUI_FIELDS + [
        "device", "loudnorm_target", "decode_hw_accel", "multi_audio_passthrough",
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
                          {"off", "auto", "any", "d3d11", "vaapi", "mfx"})
            self.assertIsInstance(cfg.multi_audio_passthrough, bool)


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
