import json
import logging
import os
import shutil
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


class KaraokeGroupingTests(unittest.TestCase):
    """_group_horizontal_line must fuse same-line boxes within the gap
    threshold and leave separate-line boxes alone."""

    def test_empty_input_returns_empty(self):
        self.assertEqual(processor._group_horizontal_line([]), [])

    def test_single_box_passes_through(self):
        self.assertEqual(
            processor._group_horizontal_line([(10, 10, 50, 30)]),
            [(10, 10, 50, 30)],
        )

    def test_five_close_syllables_fuse_into_one(self):
        # Five 30-px-wide syllables on the same line, 10-px gap each.
        syllables = [(i * 40, 100, i * 40 + 30, 140) for i in range(5)]
        merged = processor._group_horizontal_line(
            syllables, x_gap_px=20, y_overlap_ratio=0.5,
        )
        self.assertEqual(len(merged), 1)
        x1, y1, x2, y2 = merged[0]
        self.assertEqual(x1, 0)
        self.assertEqual(x2, 4 * 40 + 30)
        self.assertEqual(y1, 100)
        self.assertEqual(y2, 140)

    def test_boxes_on_separate_lines_are_not_fused(self):
        # Same x range, totally non-overlapping y.
        a = (10, 10, 100, 40)
        b = (10, 200, 100, 240)
        merged = processor._group_horizontal_line(
            [a, b], x_gap_px=20, y_overlap_ratio=0.5,
        )
        self.assertEqual(set(merged), {a, b})

    def test_boxes_with_gap_larger_than_threshold_stay_separate(self):
        a = (0, 100, 30, 140)
        b = (100, 100, 130, 140)   # gap = 70 px
        merged = processor._group_horizontal_line(
            [a, b], x_gap_px=20, y_overlap_ratio=0.5,
        )
        self.assertEqual(set(merged), {a, b})


class ChyronClassifierTests(unittest.TestCase):
    """_KalmanBox.is_chyron + SubtitleTracker.categorize must classify a
    long-lived track as 'chyron' and a short-lived one as 'subtitle'."""

    def test_is_chyron_below_threshold(self):
        box = processor._KalmanBox((10, 10, 50, 30))
        # Fresh box: 1 hit
        self.assertFalse(box.is_chyron(min_hits=90))

    def test_is_chyron_above_threshold(self):
        box = processor._KalmanBox((10, 10, 50, 30))
        for _ in range(120):
            box.update((10, 10, 50, 30))
        self.assertTrue(box.is_chyron(min_hits=90))

    def test_tracker_categorizes_persistent_track_as_chyron(self):
        tr = processor.SubtitleTracker(iou_threshold=0.3, max_age=2)
        for _ in range(120):
            tr.update([(100, 100, 200, 130)])
        cats = tr.categorize(min_chyron_hits=90)
        # Should have exactly one persistent track classified as chyron.
        self.assertEqual(len(cats), 1)
        self.assertEqual(cats[0], "chyron")

    def test_tracker_categorizes_brief_track_as_subtitle(self):
        tr = processor.SubtitleTracker(iou_threshold=0.3, max_age=2)
        for _ in range(10):
            tr.update([(100, 100, 200, 130)])
        cats = tr.categorize(min_chyron_hits=90)
        self.assertEqual(cats, ["subtitle"])


class FrameSequenceCaptureTests(unittest.TestCase):
    """_FrameSequenceCapture must mirror cv2.VideoCapture closely enough
    that process_video does not notice the swap."""

    def _make_seq_dir(self, n: int, size=(32, 48)):
        """Returns a TemporaryDirectory holding `n` PNG frames numbered
        00.png ... (n-1).png, each filled with the frame index value."""
        import numpy as _np
        import cv2 as _cv2
        tmp = tempfile.mkdtemp(prefix="vsr-seq-")
        h, w = size
        for i in range(n):
            arr = _np.full((h, w, 3), i + 10, dtype=_np.uint8)
            ok = _cv2.imwrite(str(Path(tmp) / f"{i:03d}.png"), arr)
            assert ok, f"could not write {i:03d}.png in {tmp}"
        return tmp

    def test_open_capture_routes_dir_to_frame_sequence_adapter(self):
        tmp = self._make_seq_dir(5)
        try:
            cap = processor._open_capture(tmp, "off", input_fps=12.0)
            self.assertIsInstance(cap, processor._FrameSequenceCapture)
            self.assertTrue(cap.isOpened())
            self.assertEqual(int(cap.get(processor.cv2.CAP_PROP_FRAME_COUNT)), 5)
            self.assertEqual(cap.get(processor.cv2.CAP_PROP_FPS), 12.0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_read_walks_files_in_sorted_order(self):
        tmp = self._make_seq_dir(4)
        try:
            cap = processor._FrameSequenceCapture(tmp, fps=24.0)
            seen = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                seen.append(int(frame.flat[0]))
            self.assertEqual(seen, [10, 11, 12, 13])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_set_pos_frames_supports_seek(self):
        tmp = self._make_seq_dir(6)
        try:
            cap = processor._FrameSequenceCapture(tmp, fps=24.0)
            cap.set(processor.cv2.CAP_PROP_POS_FRAMES, 4)
            ok, frame = cap.read()
            self.assertTrue(ok)
            self.assertEqual(int(frame.flat[0]), 14)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_empty_dir_raises(self):
        tmp = tempfile.mkdtemp(prefix="vsr-empty-")
        try:
            with self.assertRaises(ValueError):
                processor._FrameSequenceCapture(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class QualitySheetTests(unittest.TestCase):
    """_write_quality_sheet must produce a single PNG with one row per
    sampled pair and a header carrying the mean metrics + Good/Review tag."""

    def test_sheet_written_with_expected_dimensions(self):
        import numpy as _np
        import cv2 as _cv2
        with tempfile.TemporaryDirectory() as tmp:
            out_path = str(Path(tmp) / "result.mp4")
            # Three synthetic pairs.
            pairs = []
            for i in range(3):
                a = _np.full((120, 160, 3), 100 + i, dtype=_np.uint8)
                b = _np.full((120, 160, 3), 110 + i, dtype=_np.uint8)
                pairs.append((i * 5, a, b, 35.0 + i, 0.96 - 0.01 * i))
            remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
            remover.config = processor.ProcessingConfig()
            sheet_path = remover._write_quality_sheet(
                out_path, pairs, mean_psnr=36.0, mean_ssim=0.95, tag="Good",
            )
            self.assertTrue(Path(sheet_path).exists())
            sheet = _cv2.imread(sheet_path)
            self.assertIsNotNone(sheet)
            # Width should match a single pair-row (two scaled frames + gap).
            # Height must include the header + N rows + N caption strips.
            self.assertGreater(sheet.shape[0], 200)
            self.assertGreater(sheet.shape[1], 200)
            # Filename convention.
            self.assertTrue(sheet_path.endswith(".qualitysheet.png"))


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


class SuryaOptInTests(unittest.TestCase):
    """B-2: Surya is GPL; the detector cascade must skip it unless the user
    explicitly opts in via VSR_ALLOW_GPL."""

    def setUp(self):
        self._saved = os.environ.pop("VSR_ALLOW_GPL", None)

    def tearDown(self):
        os.environ.pop("VSR_ALLOW_GPL", None)
        if self._saved is not None:
            os.environ["VSR_ALLOW_GPL"] = self._saved

    def test_surya_disallowed_by_default(self):
        self.assertFalse(processor._surya_allowed())

    def test_surya_allowed_when_env_set(self):
        for token in ("1", "true", "yes", "on", "TRUE"):
            os.environ["VSR_ALLOW_GPL"] = token
            self.assertTrue(processor._surya_allowed(), f"token={token}")

    def test_surya_disallowed_for_unknown_tokens(self):
        for token in ("0", "false", "no", "off", "", "maybe"):
            os.environ["VSR_ALLOW_GPL"] = token
            self.assertFalse(processor._surya_allowed(), f"token={token}")


class FfmpegTimeoutBudgetTests(unittest.TestCase):
    """F-6: the ffmpeg subprocess timeout must scale with content length so
    multi-hour videos do not silently fall back to copy-without-audio."""

    def test_zero_duration_falls_back_to_safe_base(self):
        # When ffprobe is unavailable the helper returns 0; the timeout
        # should still leave a generous floor (base + 600s).
        t = processor._ffmpeg_subprocess_timeout(0.0)
        self.assertGreaterEqual(t, 600.0)
        self.assertLess(t, 24 * 3600.0)

    def test_one_hour_video_gets_factor_4_budget(self):
        t = processor._ffmpeg_subprocess_timeout(3600.0)
        # Factor 4 -> 14400s plus the 180s base.
        self.assertGreaterEqual(t, 4 * 3600.0)

    def test_eight_hour_video_gets_proportional_budget(self):
        t = processor._ffmpeg_subprocess_timeout(8 * 3600.0)
        # Must exceed the legacy 600s cap by a wide margin.
        self.assertGreater(t, 8 * 3600.0)

    def test_timeout_caps_at_24_hours(self):
        # Even an absurd duration must not produce a runaway timeout that
        # blocks the GUI forever.
        t = processor._ffmpeg_subprocess_timeout(10 * 24 * 3600.0)
        self.assertLessEqual(t, 24 * 3600.0)


class GuiToBackendFieldWiringTests(unittest.TestCase):
    """B-1: the 13 v3.13 backend fields must round-trip through the GUI
    dataclass without being silently dropped, and reach the backend
    config when _process_item builds the BackendConfig."""

    EXPECTED_GUI_FIELDS = (
        "loudnorm_target", "multi_audio_passthrough", "decode_hw_accel",
        "prefetch_decode", "prefetch_queue_size", "input_fps",
        "quality_report_sheet", "remove_subtitles", "remove_chyrons",
        "chyron_min_hits", "karaoke_grouping", "karaoke_x_gap_px",
        "karaoke_y_overlap",
    )

    def test_all_thirteen_fields_declared_on_gui_dataclass(self):
        cfg = gui.ProcessingConfig()
        for name in self.EXPECTED_GUI_FIELDS:
            self.assertTrue(
                hasattr(cfg, name),
                f"GUI ProcessingConfig is missing field {name!r}; was "
                "removed or never wired through B-1.",
            )

    def test_all_thirteen_fields_persist_through_to_dict(self):
        cfg = gui.ProcessingConfig(
            loudnorm_target=-14.0,
            multi_audio_passthrough=False,
            decode_hw_accel="d3d11",
            prefetch_decode=False,
            prefetch_queue_size=24,
            input_fps=30.0,
            quality_report_sheet=True,
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


class CachedRemoverHotSwapNormalizationTests(unittest.TestCase):
    """I-2: hot-swap of `remover.config` must run through
    normalize_processing_config so a bad per-item override cannot reach the
    pipeline. The GUI's _process_item now applies this; verify the contract
    by exercising the normaliser on a payload that mimics a hot-swap."""

    def test_hot_swap_payload_clamps_nan(self):
        raw = processor.ProcessingConfig(
            mode=processor.InpaintMode.STTN,
            device="cuda:0",
            loudnorm_target=float("nan"),
            decode_hw_accel="not-a-token",
            detection_threshold=float("inf"),
        )
        cfg = processor.normalize_processing_config(raw)
        self.assertEqual(cfg.loudnorm_target, 0.0)
        self.assertEqual(cfg.decode_hw_accel, "off")
        self.assertTrue(0.1 <= cfg.detection_threshold <= 1.0)


class QualityReportMaskedRoiTests(unittest.TestCase):
    """B-3: union-mask bbox accumulator + ROI-cropped PSNR/SSIM metric so
    a bad inpaint is no longer masked by 80-95% of unchanged pixels."""

    def _bare_remover(self):
        remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        remover.config = processor.ProcessingConfig()
        remover._quality_mask_bbox = None
        return remover

    def test_accumulator_ignores_empty_mask(self):
        import numpy as _np
        r = self._bare_remover()
        r._accumulate_quality_bbox(_np.zeros((10, 10), dtype=_np.uint8))
        self.assertIsNone(r._quality_mask_bbox)

    def test_accumulator_tracks_single_box(self):
        import numpy as _np
        r = self._bare_remover()
        mask = _np.zeros((100, 200), dtype=_np.uint8)
        mask[20:40, 50:120] = 255
        r._accumulate_quality_bbox(mask)
        self.assertEqual(r._quality_mask_bbox, (50, 20, 120, 40))

    def test_accumulator_unions_across_frames(self):
        import numpy as _np
        r = self._bare_remover()
        m1 = _np.zeros((100, 200), dtype=_np.uint8)
        m1[20:40, 50:120] = 255
        m2 = _np.zeros((100, 200), dtype=_np.uint8)
        m2[60:90, 30:80] = 255
        r._accumulate_quality_bbox(m1)
        r._accumulate_quality_bbox(m2)
        self.assertEqual(r._quality_mask_bbox, (30, 20, 120, 90))


class LosslessIntermediateWriterTests(unittest.TestCase):
    """I-1: the intermediate writer must roundtrip frames losslessly when
    ffmpeg is available (FFV1 in .mkv) and degrade gracefully to the
    legacy mp4v writer when it is not."""

    def _have_ffmpeg(self):
        return shutil.which("ffmpeg") is not None

    def test_writer_round_trips_frames_losslessly(self):
        if not self._have_ffmpeg():
            self.skipTest("ffmpeg not on PATH")
        import numpy as _np
        import cv2 as _cv2
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "intermediate.mkv")
            w, h, fps = 32, 24, 12.0
            writer = processor._LosslessIntermediateWriter(path, w, h, fps)
            self.assertTrue(writer.isOpened())
            self.assertTrue(writer.lossless,
                            "FFV1 path should engage when ffmpeg is present")
            frames = []
            for i in range(10):
                # Each frame is uniformly coloured with (i, i*2, i*3) so a
                # lossless round-trip yields bit-identical values back.
                arr = _np.empty((h, w, 3), dtype=_np.uint8)
                arr[:] = (i, (i * 2) % 256, (i * 3) % 256)
                frames.append(arr)
                writer.write(arr)
            writer.release()
            self.assertTrue(Path(path).exists())
            cap = _cv2.VideoCapture(path)
            seen = []
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    seen.append(frame)
            finally:
                cap.release()
            self.assertEqual(len(seen), len(frames))
            # Lossless: per-frame max channel delta is 0 for FFV1 + bgr24.
            for i, (src, decoded) in enumerate(zip(frames, seen)):
                delta = int(_np.abs(src.astype(_np.int16) - decoded.astype(_np.int16)).max())
                self.assertEqual(delta, 0,
                                 f"frame {i} expected lossless roundtrip, got delta={delta}")

    def test_writer_fallback_when_ffmpeg_path_is_blank(self):
        # Simulate a missing ffmpeg by patching shutil.which inside the
        # processor module. The writer must open the cv2 fallback and stay
        # functional rather than raising.
        import shutil as _shutil
        original_which = _shutil.which
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "intermediate.mkv")
            try:
                _shutil.which = lambda name: None
                writer = processor._LosslessIntermediateWriter(path, 16, 12, 24.0)
                # Fallback path renames .mkv to .mp4 because mp4v in .mkv
                # is rarely playable on consumer builds.
                self.assertFalse(writer.lossless)
                self.assertTrue(writer.path.endswith(".mp4"))
                writer.release()
            finally:
                _shutil.which = original_which


class AutoInpainterUnloadTests(unittest.TestCase):
    """B-5: AutoInpainter must drop the lazily-loaded LaMa after enough
    consecutive TBE batches to reclaim VRAM on long, mostly-easy videos."""

    def _auto_inpainter(self):
        cfg = processor.ProcessingConfig(mode=processor.InpaintMode.AUTO)
        cfg = processor.normalize_processing_config(cfg)
        return processor.AutoInpainter(device="cpu", config=cfg)

    def test_streak_resets_on_lama_route(self):
        auto = self._auto_inpainter()
        auto._tbe_streak = 5
        # Stub _lama and STTN inpaint to avoid heavy model loads.
        auto._lama = object()
        auto._sttn.inpaint = lambda f, m: f  # type: ignore[assignment]
        # Force LaMa route by feeding a fully-covered mask (zero exposure).
        import numpy as _np
        frame = _np.zeros((4, 4, 3), dtype=_np.uint8)
        mask = _np.full((4, 4), 255, dtype=_np.uint8)
        # Patch _ensure_lama to return a stub that returns frames as-is
        class _StubLama:
            def inpaint(self, frames, masks):
                return frames
        auto._lama = _StubLama()
        _ = auto.inpaint([frame, frame], [mask, mask])
        self.assertEqual(auto._tbe_streak, 0)

    def test_lama_unloaded_after_streak_threshold(self):
        auto = self._auto_inpainter()
        # Shorten the threshold for the test so we don't synthesise 50
        # batches; we mutate the class constant directly.
        auto.LAMA_IDLE_UNLOAD_AFTER = 3
        class _StubLama:
            def inpaint(self, frames, masks):
                return frames
        auto._lama = _StubLama()
        # Force TBE path: fully exposed (no overlap between masked frames).
        import numpy as _np
        frame = _np.zeros((4, 4, 3), dtype=_np.uint8)
        m1 = _np.zeros((4, 4), dtype=_np.uint8); m1[0, 0] = 255
        m2 = _np.zeros((4, 4), dtype=_np.uint8); m2[3, 3] = 255
        # Stub STTN inpaint to skip the heavy TBE path.
        auto._sttn.inpaint = lambda f, m: f  # type: ignore[assignment]
        for _ in range(3):
            _ = auto.inpaint([frame, frame], [m1, m2])
        self.assertIsNone(auto._lama, "LaMa must be released after streak hits the threshold")


class MultiTrackLoudnormFilterTests(unittest.TestCase):
    """B-4: when both loudnorm and multi-track passthrough are active and
    the source has multiple audio streams, _merge_audio must build a
    -filter_complex pipeline instead of relying on the single-pass
    `-af loudnorm`. We exercise the audio-stream probe helper here
    (the full _merge_audio orchestration needs real ffmpeg + a video)."""

    def test_audio_stream_count_falls_back_to_one_when_ffprobe_missing(self):
        # The helper must not crash when ffprobe is absent. Returning 1
        # means _merge_audio takes the legacy single-stream path.
        import shutil as _shutil
        original = _shutil.which
        try:
            _shutil.which = lambda name: None
            count = processor._probe_audio_stream_count("/non-existent.mkv")
            # ffprobe absent -> falls back to 1.
            self.assertEqual(count, 1)
        finally:
            _shutil.which = original


class LanguagePickerTests(unittest.TestCase):
    """F-5: lang picker must expose more than the legacy 12 languages
    while keeping the curated English-first ordering."""

    def test_language_list_starts_with_english(self):
        langs = gui._build_language_list()
        self.assertEqual(langs[0][0], "en")
        self.assertEqual(langs[0][1], "English")

    def test_language_list_includes_extra_codes(self):
        codes = {code for code, _ in gui._build_language_list()}
        # Languages outside the legacy 12-language set.
        for new_code in ("th", "vi", "pl", "tr", "uk", "el"):
            self.assertIn(new_code, codes, f"expected {new_code} in expanded list")

    def test_language_list_deduplicates(self):
        codes = [code for code, _ in gui._build_language_list()]
        self.assertEqual(len(codes), len(set(codes)),
                         "language picker must not contain duplicate codes")


class OutputCodecTests(unittest.TestCase):
    """F-8: output_codec must coerce to one of h264 / h265 / av1 and
    drive the right software encoder when no HW encoder is available."""

    def test_default_is_h264(self):
        cfg = processor.normalize_processing_config(processor.ProcessingConfig())
        self.assertEqual(cfg.output_codec, "h264")

    def test_hevc_alias_normalises_to_h265(self):
        cfg = processor.normalize_processing_config(
            processor.ProcessingConfig(output_codec="hevc")
        )
        self.assertEqual(cfg.output_codec, "h265")

    def test_unknown_codec_resets_to_h264(self):
        cfg = processor.normalize_processing_config(
            processor.ProcessingConfig(output_codec="vp9")
        )
        self.assertEqual(cfg.output_codec, "h264")

    def test_software_encoder_args_match_codec(self):
        remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        remover._hw_encoder = None
        remover.config = processor.ProcessingConfig(output_codec="h265",
                                                     output_quality=22,
                                                     use_hw_encode=False)
        args = remover._get_encode_args()
        self.assertIn("libx265", args)
        remover.config.output_codec = "av1"
        args = remover._get_encode_args()
        self.assertIn("libsvtav1", args)


class ModelHashVerificationTests(unittest.TestCase):
    """RM-49: verify_weight_file should return True for a match,
    False for a mismatch, and True (with a debug log) when no vendored
    hash exists for the filename."""

    def test_verify_match(self):
        from backend.model_hashes import verify_weight_file, hash_file
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "weights.pt"
            p.write_bytes(b"hello world")
            expected = hash_file(p)
            self.assertTrue(verify_weight_file(p, expected_hash=expected))

    def test_verify_mismatch(self):
        from backend.model_hashes import verify_weight_file
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "weights.pt"
            p.write_bytes(b"hello world")
            self.assertFalse(verify_weight_file(p, expected_hash="0" * 64))

    def test_verify_unknown_filename_returns_true(self):
        from backend.model_hashes import verify_weight_file
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "not-tracked.bin"
            p.write_bytes(b"some bytes")
            # No vendored hash entry; verifier returns True (no-op).
            self.assertTrue(verify_weight_file(p))

    def test_verify_missing_file_returns_false(self):
        from backend.model_hashes import verify_weight_file
        result = verify_weight_file(Path("/nonexistent/weights.pt"),
                                      expected_hash="0" * 64)
        self.assertFalse(result)


class PresetLibraryTests(unittest.TestCase):
    """F-10: built-in presets must be shared between the GUI and the CLI
    so `python -m backend.processor --preset NAME` resolves to the same
    payload the GUI's picker would apply."""

    def test_builtin_presets_exposed(self):
        from backend import presets
        self.assertIn("YouTube (default)", presets.BUILTIN_PRESETS)
        self.assertIn("Anime / Animation", presets.BUILTIN_PRESETS)

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


class OtsuFallbackDetectionTests(unittest.TestCase):
    """EI-1: the OpenCV fallback detector must catch mid-tone subtitle
    luminance the fixed 200 / 55 thresholds missed."""

    def test_fallback_finds_grey_text_on_grey(self):
        import numpy as _np
        import cv2 as _cv2
        # Mid-tone grey frame with slightly darker grey text-shaped strip
        # in the bottom band -- both within the [55, 200] dead zone of
        # the old fixed thresholds.
        frame = _np.full((180, 320, 3), 130, dtype=_np.uint8)
        frame[150:170, 40:280] = 100  # darker grey "subtitle"
        detector = processor.SubtitleDetector.__new__(processor.SubtitleDetector)
        detector._engine_name = "OpenCV fallback"
        detector._rapid_model = None
        detector._paddle_model = None
        detector._surya_det = None
        detector._easyocr_reader = None
        boxes = detector._fallback_detection(frame)
        self.assertTrue(boxes, "Otsu fallback must detect the mid-tone band")


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


class EndToEndPipelineTests(unittest.TestCase):
    """T-2: end-to-end test that synthesises a tiny BGR clip, runs the
    full SubtitleRemover.process_video pipeline against it (using
    skip_detection + a fixed subtitle_area so we do not depend on an
    OCR engine being installed), and asserts the output exists and
    decodes back the expected frame count."""

    def _write_clip(self, dir_path: Path, n_frames: int = 30,
                    size=(64, 48)) -> Path:
        """Write a tiny synthesised clip via the lossless intermediate
        path so OpenCV's container support does not bias the test."""
        import cv2 as _cv2
        import numpy as _np
        out = dir_path / "synth.mkv"
        writer = processor._LosslessIntermediateWriter(
            str(out), size[0], size[1], 24.0
        )
        try:
            self.assertTrue(writer.isOpened())
            for i in range(n_frames):
                frame = _np.full((size[1], size[0], 3), 30, dtype=_np.uint8)
                # Burn a horizontal "subtitle" band that the fixed
                # subtitle_area covers; the inpainter will turn it back
                # into the surrounding background tone.
                frame[size[1] - 12:size[1] - 4, 8:size[0] - 8] = 240
                writer.write(frame)
        finally:
            writer.release()
        return Path(writer.path)

    def test_pipeline_runs_with_skip_detection(self):
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not on PATH")
        import cv2 as _cv2
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = self._write_clip(tmp, n_frames=24, size=(64, 48))
            output = tmp / "cleaned.mp4"
            cfg = processor.ProcessingConfig(
                mode=processor.InpaintMode.STTN,
                device="cpu",
                sttn_skip_detection=True,
                subtitle_area=(8, 36, 56, 44),
                tbe_enable=True,
                preserve_audio=False,
                output_quality=23,
                adaptive_batch=False,
                use_hw_encode=False,
            )
            cfg = processor.normalize_processing_config(cfg)
            # Build a remover that bypasses the detector load.
            remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
            remover.config = cfg
            remover.detector = processor.SubtitleDetector.__new__(
                processor.SubtitleDetector
            )
            remover.detector.device = "cpu"
            remover.detector.lang = "en"
            remover.detector._engine_name = "skip"
            remover.detector._rapid_model = None
            remover.detector._paddle_model = None
            remover.detector._surya_det = None
            remover.detector._easyocr_reader = None
            remover.inpainter = processor.STTNInpainter("cpu", cfg)
            remover.on_progress = None
            remover.on_preview_frame = None
            remover.live_preview_stride = 6
            remover._hw_encoder = None
            remover._srt_entries = []
            remover.last_quality_report = None
            remover._quality_mask_bbox = None
            ok = remover.process_video(str(src), str(output))
            self.assertTrue(ok, "process_video must succeed end-to-end")
            self.assertTrue(output.exists(), "output file must be written")
            cap = _cv2.VideoCapture(str(output))
            try:
                self.assertTrue(cap.isOpened())
                frames_read = 0
                while True:
                    ret, _ = cap.read()
                    if not ret:
                        break
                    frames_read += 1
            finally:
                cap.release()
            self.assertGreaterEqual(frames_read, 20)


class OcrCascadeOrderTests(unittest.TestCase):
    """T-3: the OCR loader must follow the documented priority order
    (RapidOCR > PaddleOCR > Surya > EasyOCR > OpenCV fallback). We
    patch importlib so the test does not require any optional OCR
    engine to be installed."""

    def _make_detector(self):
        det = processor.SubtitleDetector.__new__(processor.SubtitleDetector)
        det.device = "cpu"
        det.lang = "en"
        det._engine_name = "none"
        det._rapid_model = None
        det._paddle_model = None
        det._surya_det = None
        det._easyocr_reader = None
        return det

    def test_falls_back_to_opencv_when_no_engine_installed(self):
        det = self._make_detector()
        det._load_model()
        # All optional engines absent on this CI -> OpenCV fallback.
        self.assertEqual(det._engine_name, "OpenCV fallback")

    def test_surya_skipped_unless_env_set(self):
        # The cascade should not pick Surya even when its module is
        # importable; gating is via VSR_ALLOW_GPL.
        os.environ.pop("VSR_ALLOW_GPL", None)
        det = self._make_detector()
        det._load_model()
        self.assertNotEqual(det._engine_name, "Surya")


if __name__ == "__main__":
    unittest.main()
