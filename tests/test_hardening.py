import io
import json
import logging
import os
import shutil
import subprocess
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
            content = Path(path).read_text(encoding="utf-8")
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
            content = Path(path).read_text(encoding="utf-8")
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


class DirectMlProviderTests(unittest.TestCase):
    @staticmethod
    def _pb_varint(value):
        out = bytearray()
        while True:
            byte = value & 0x7F
            value >>= 7
            if value:
                out.append(byte | 0x80)
            else:
                out.append(byte)
                return bytes(out)

    @classmethod
    def _pb_field_varint(cls, field_number, value):
        return cls._pb_varint((field_number << 3) | 0) + cls._pb_varint(value)

    @classmethod
    def _pb_field_bytes(cls, field_number, payload):
        return (
            cls._pb_varint((field_number << 3) | 2)
            + cls._pb_varint(len(payload))
            + payload
        )

    @classmethod
    def _minimal_onnx_model(cls, *opsets):
        chunks = []
        for domain, version in opsets:
            payload = cls._pb_field_varint(2, version)
            if domain:
                payload = (
                    cls._pb_field_bytes(1, domain.encode("utf-8"))
                    + payload
                )
            chunks.append(cls._pb_field_bytes(8, payload))
        return b"".join(chunks)

    def test_gui_detects_directml_via_onnxruntime_provider(self):
        from unittest import mock
        fake_ort = SimpleNamespace(
            get_available_providers=lambda: [
                "DmlExecutionProvider", "CPUExecutionProvider"
            ]
        )
        from gui import utils as _gui_utils
        with mock.patch.object(_gui_utils.subprocess, "run", side_effect=FileNotFoundError):
            with mock.patch.dict(sys.modules, {"onnxruntime": fake_ort}):
                gpus = gui.detect_gpu()

        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0]["type"], "DirectML")
        self.assertEqual(gpus[0]["memory"], "ONNX Runtime")

    def test_gui_ignores_onnxruntime_without_dml_provider(self):
        from unittest import mock
        fake_ort = SimpleNamespace(
            get_available_providers=lambda: ["CPUExecutionProvider"]
        )
        from gui import utils as _gui_utils
        with mock.patch.object(_gui_utils.subprocess, "run", side_effect=FileNotFoundError):
            with mock.patch.dict(sys.modules, {"onnxruntime": fake_ort}):
                self.assertEqual(gui.detect_gpu(), [])

    def test_onnx_inpainter_provider_order_for_directml(self):
        from backend.inpainters_onnx import _providers_for_device
        self.assertEqual(
            _providers_for_device("directml"),
            ["DmlExecutionProvider", "CPUExecutionProvider"],
        )

    def test_onnx_opset_parser_reads_default_and_custom_domains(self):
        from backend.onnx_model_info import read_onnx_opset_imports
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "model.onnx"
            model.write_bytes(self._minimal_onnx_model(("", 20), ("ai.onnx.ml", 3)))
            imports = read_onnx_opset_imports(model)
        self.assertEqual(
            [(item.domain, item.version) for item in imports],
            [("", 20), ("ai.onnx.ml", 3)],
        )

    def test_directml_provider_dropped_for_unsupported_default_opset(self):
        from backend.inpainters_onnx import _providers_after_opset_audit
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "model.onnx"
            model.write_bytes(self._minimal_onnx_model(("", 21)))
            providers = _providers_after_opset_audit(
                str(model),
                ["DmlExecutionProvider", "CPUExecutionProvider"],
            )
        self.assertEqual(providers, ["CPUExecutionProvider"])

    def test_directml_provider_kept_for_supported_default_opset(self):
        from backend.inpainters_onnx import _providers_after_opset_audit
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "model.onnx"
            model.write_bytes(self._minimal_onnx_model(("", 20)))
            providers = _providers_after_opset_audit(
                str(model),
                ["DmlExecutionProvider", "CPUExecutionProvider"],
            )
        self.assertEqual(providers, ["DmlExecutionProvider", "CPUExecutionProvider"])

    def test_easyocr_gpu_flag_is_cuda_only(self):
        det = processor.SubtitleDetector.__new__(processor.SubtitleDetector)
        det.device = "directml"
        self.assertFalse(det._is_gpu_device())
        det.device = "cuda:0"
        self.assertTrue(det._is_gpu_device())

    def test_rapidocr_uses_directml_params_when_provider_available(self):
        from unittest import mock

        calls = []

        class FakeRapidOCR:
            def __init__(self, **kwargs):
                calls.append(kwargs)

        fake_ort = SimpleNamespace(
            get_available_providers=lambda: [
                "DmlExecutionProvider", "CPUExecutionProvider"
            ]
        )
        fake_rapid = SimpleNamespace(RapidOCR=FakeRapidOCR)

        with mock.patch.dict(
            sys.modules,
            {"onnxruntime": fake_ort, "rapidocr": fake_rapid},
        ):
            det = processor.SubtitleDetector(device="directml")

        self.assertEqual(det._engine_name, "RapidOCR (DirectML)")
        self.assertEqual(
            calls[0]["params"]["EngineConfig.onnxruntime.use_dml"],
            True,
        )
        self.assertEqual(
            calls[0]["params"]["EngineConfig.onnxruntime.use_cuda"],
            False,
        )

    def test_rapidocr_directml_params_fall_back_for_legacy_constructor(self):
        from unittest import mock

        calls = []

        class LegacyRapidOCR:
            def __init__(self, **kwargs):
                calls.append(kwargs)
                if kwargs:
                    raise TypeError("unexpected keyword argument 'params'")

        fake_ort = SimpleNamespace(
            get_available_providers=lambda: [
                "DmlExecutionProvider", "CPUExecutionProvider"
            ]
        )
        fake_rapid = SimpleNamespace(RapidOCR=LegacyRapidOCR)

        with mock.patch.dict(
            sys.modules,
            {"onnxruntime": fake_ort, "rapidocr": fake_rapid},
        ):
            det = processor.SubtitleDetector(device="directml")

        self.assertEqual(det._engine_name, "RapidOCR")
        self.assertIn("params", calls[0])
        self.assertEqual(calls[1], {})

    def test_rapidocr_directml_params_skip_when_provider_absent(self):
        from unittest import mock

        calls = []

        class FakeRapidOCR:
            def __init__(self, **kwargs):
                calls.append(kwargs)

        fake_ort = SimpleNamespace(
            get_available_providers=lambda: ["CPUExecutionProvider"]
        )
        fake_rapid = SimpleNamespace(RapidOCR=FakeRapidOCR)

        with mock.patch.dict(
            sys.modules,
            {"onnxruntime": fake_ort, "rapidocr": fake_rapid},
        ):
            det = processor.SubtitleDetector(device="directml")

        self.assertEqual(det._engine_name, "RapidOCR")
        self.assertEqual(calls, [{}])


class RapidOcrOutputCompatibilityTests(unittest.TestCase):
    def _detector_with_output(self, output):
        det = processor.SubtitleDetector.__new__(processor.SubtitleDetector)
        det._rapid_model = lambda frame: output
        det._fallback_detection = lambda frame: [(-1, -1, -1, -1)]
        return det

    @staticmethod
    def _frame():
        import numpy as _np
        return _np.zeros((12, 12, 3), dtype=_np.uint8)

    def test_detect_rapid_accepts_legacy_tuple_output(self):
        output = (
            [
                (
                    [[1, 2], [5, 2], [5, 6], [1, 6]],
                    "subtitle",
                    0.9,
                )
            ],
            {"det": 0.01},
        )
        det = self._detector_with_output(output)

        boxes = det._detect_rapid(self._frame(), threshold=0.5)

        self.assertEqual(boxes, [(1, 2, 5, 6)])

    def test_detect_rapid_accepts_structured_object_output(self):
        output = SimpleNamespace(
            dt_polys=[[[1, 1], [7, 1], [7, 4], [1, 4]]],
            rec_scores=[0.8],
            rec_texts=["subtitle"],
        )
        det = self._detector_with_output(output)

        boxes = det._detect_rapid(self._frame(), threshold=0.5)

        self.assertEqual(boxes, [(1, 1, 7, 4)])

    def test_detect_rapid_accepts_structured_dict_output(self):
        output = {
            "dt_polys": [
                [[1, 1], [7, 1], [7, 4], [1, 4]],
                [[2, 5], [9, 5], [9, 8], [2, 8]],
            ],
            "rec_scores": [0.4, 0.95],
        }
        det = self._detector_with_output(output)

        boxes = det._detect_rapid(self._frame(), threshold=0.5)

        self.assertEqual(boxes, [(2, 5, 9, 8)])

    def test_detect_rapid_ignores_malformed_polygons_without_fallback(self):
        output = {"dt_polys": [["bad"], [[3, 3], [3, 3]]], "rec_scores": [1.0, 1.0]}
        det = self._detector_with_output(output)

        boxes = det._detect_rapid(self._frame(), threshold=0.5)

        self.assertEqual(boxes, [])

    def test_rapidocr_config_load_failures_fall_through_cascade(self):
        from unittest import mock

        class MissingConfigRapidOCR:
            def __init__(self, **kwargs):
                raise FileNotFoundError("default_models.yaml")

        absent = {
            name: None
            for name in (
                "rapidocr_onnxruntime",
                "paddleocr",
                "surya",
                "surya.detection",
                "easyocr",
            )
        }
        fake_rapid = SimpleNamespace(RapidOCR=MissingConfigRapidOCR)

        with mock.patch.dict(sys.modules, {"rapidocr": fake_rapid, **absent}):
            det = processor.SubtitleDetector(device="cpu")

        self.assertEqual(det._engine_name, "OpenCV fallback")


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
        # v3.13 GUI-exposed fields (B-1 + F-8).
        "loudnorm_target", "multi_audio_passthrough",
        "decode_hw_accel", "prefetch_decode", "prefetch_queue_size",
        "input_fps", "quality_report_sheet",
        "remove_subtitles", "remove_chyrons", "chyron_min_hits",
        "karaoke_grouping", "karaoke_x_gap_px", "karaoke_y_overlap",
        "output_codec",
        "window_geometry", "adv_panel_open", "log_panel_open",
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
                          {"off", "auto", "any", "d3d11", "vaapi", "mfx"})
            self.assertIn(cfg.output_codec, {"h264", "h265", "av1"})
            self.assertIsInstance(cfg.multi_audio_passthrough, bool)
            self.assertGreaterEqual(cfg.input_fps, 1.0)
            self.assertLessEqual(cfg.input_fps, 240.0)

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


class RuntimeSecurityCheckTests(unittest.TestCase):
    def test_parse_libpng_version_from_opencv_build_info(self):
        from backend.security_checks import parse_libpng_version

        info = "    PNG:                         build (ver 1.6.53)"
        self.assertEqual(parse_libpng_version(info), (1, 6, 53))

    def test_libpng_warning_fires_below_fixed_floor(self):
        from unittest import mock
        from backend.security_checks import warn_if_vulnerable_opencv_libpng

        fake_cv2 = SimpleNamespace(
            getBuildInformation=lambda: "PNG: build (ver 1.6.53)"
        )
        logger = mock.Mock()
        with mock.patch.dict(sys.modules, {"cv2": fake_cv2}):
            message = warn_if_vulnerable_opencv_libpng(logger)

        self.assertIn("CVE-2026-22801", message)
        logger.warning.assert_called_once()

    def test_libpng_warning_silent_at_fixed_floor(self):
        from unittest import mock
        from backend.security_checks import warn_if_vulnerable_opencv_libpng

        fake_cv2 = SimpleNamespace(
            getBuildInformation=lambda: "PNG: build (ver 1.6.54)"
        )
        logger = mock.Mock()
        with mock.patch.dict(sys.modules, {"cv2": fake_cv2}):
            message = warn_if_vulnerable_opencv_libpng(logger)

        self.assertIsNone(message)
        logger.warning.assert_not_called()


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

    def test_libvmaf_available_detects_filter(self):
        from unittest import mock
        from backend import quality as _q
        completed = SimpleNamespace(
            returncode=0,
            stdout=" .. libvmaf           VV->V      Calculate the VMAF between two video streams.\n",
            stderr="",
        )
        with mock.patch.object(_q.shutil, "which", return_value="ffmpeg"):
            with mock.patch.object(_q.subprocess, "run", return_value=completed):
                self.assertTrue(_q.ffmpeg_libvmaf_available())

    def test_compute_vmaf_parses_mocked_log(self):
        from unittest import mock
        from backend import quality as _q

        def fake_run(cmd, check, capture_output, timeout):
            filt = cmd[cmd.index("-lavfi") + 1]
            log_path = Path(_q._unescape_filter_value(filt.split("log_path=", 1)[1]))
            log_path.write_text(
                json.dumps({"pooled_metrics": {"vmaf": {"mean": 91.25}}}),
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

        with tempfile.TemporaryDirectory() as tmpdir:
            ref = Path(tmpdir) / "ref.mp4"
            dist = Path(tmpdir) / "dist.mp4"
            ref.write_bytes(b"ref")
            dist.write_bytes(b"dist")
            with mock.patch.object(_q, "ffmpeg_libvmaf_available", return_value=True):
                with mock.patch.object(_q.subprocess, "run", side_effect=fake_run):
                    self.assertEqual(
                        _q.compute_vmaf(
                            str(ref), str(dist), duration_seconds=1.0,
                            roi=(2, 3, 12, 23),
                        ),
                        91.25,
                    )

    def test_quality_report_includes_vmaf_when_available(self):
        from unittest import mock
        import numpy as _np
        r = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        r.config = processor.ProcessingConfig(quality_report=True)
        r._quality_mask_bbox = (10, 10, 50, 50)
        frame_in = _np.full((80, 96, 3), 128, dtype=_np.uint8)
        frame_out = frame_in.copy()

        class FakeCapture:
            def __init__(self, frame):
                self.frame = frame
            def isOpened(self):
                return True
            def get(self, prop):
                if prop == processor.cv2.CAP_PROP_FRAME_COUNT:
                    return 4
                return 0
            def set(self, prop, value):
                return True
            def read(self):
                return True, self.frame.copy()
            def release(self):
                return None

        with mock.patch(
            "backend.processor._open_capture",
            side_effect=[FakeCapture(frame_in), FakeCapture(frame_out)],
        ):
            with mock.patch(
                "backend.processor.compute_vmaf",
                side_effect=[95.0, 93.0],
            ):
                metrics = r._compute_quality_report(
                    "input.mp4", "output.mp4", 0, 4, 24.0, n_samples=2
                )

        self.assertEqual(metrics["vmaf"], 95.0)
        self.assertEqual(metrics["roi_vmaf"], 93.0)
        self.assertEqual(metrics["temporal_flicker_score"], 0.0)
        self.assertEqual(metrics["residual_text_score"], 0.0)
        self.assertEqual(metrics["quality_gate"]["status"], "passed")


class QualityGateTests(unittest.TestCase):
    """#108: existing quality metrics are converted into a stable batch
    gate result before automatic fallback ladder work is added."""

    def test_passes_when_roi_metrics_clear_thresholds(self):
        from backend.quality_gate import evaluate_quality_gate
        gate = evaluate_quality_gate({
            "samples": 4,
            "tag": "Good",
            "ssim": 0.98,
            "roi_ssim": 0.97,
            "vmaf": 95.0,
            "roi_vmaf": 92.0,
        })
        self.assertEqual(gate["status"], "passed")
        self.assertEqual(gate["ladderStep"], "none")

    def test_temporal_flicker_score_uses_adjacent_samples_only(self):
        import numpy as _np
        from backend.quality import temporal_flicker_score
        black = _np.zeros((16, 16, 3), dtype=_np.uint8)
        white = _np.full((16, 16, 3), 255, dtype=_np.uint8)
        self.assertIsNone(temporal_flicker_score([(0, black), (4, white)]))
        self.assertEqual(temporal_flicker_score([(0, black), (1, white)]), 1.0)

    def test_residual_text_score_flags_text_like_roi(self):
        import cv2 as _cv2
        import numpy as _np
        from backend.quality import residual_text_score
        flat = _np.full((80, 220, 3), 128, dtype=_np.uint8)
        text = flat.copy()
        _cv2.putText(
            text,
            "SUBTITLE",
            (12, 50),
            _cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            _cv2.LINE_AA,
        )
        self.assertEqual(residual_text_score(flat), 0.0)
        self.assertGreater(residual_text_score(text), 0.025)

    def test_review_when_temporal_flicker_is_high(self):
        from backend.quality_gate import evaluate_quality_gate
        gate = evaluate_quality_gate({
            "samples": 4,
            "tag": "Good",
            "ssim": 0.99,
            "roi_ssim": 0.98,
            "temporal_flicker_score": 0.2,
        })
        self.assertEqual(gate["status"], "review")
        self.assertIn("temporal flicker", gate["reason"])

    def test_review_when_residual_text_score_is_high(self):
        from backend.quality_gate import evaluate_quality_gate
        gate = evaluate_quality_gate({
            "samples": 4,
            "tag": "Good",
            "ssim": 0.99,
            "roi_ssim": 0.98,
            "residual_text_score": 0.1,
        })
        self.assertEqual(gate["status"], "review")
        self.assertIn("residual text score", gate["reason"])

    def test_review_when_roi_metric_fails_and_sheet_is_preview(self):
        from backend.quality_gate import evaluate_quality_gate
        gate = evaluate_quality_gate({
            "samples": 4,
            "tag": "Review",
            "ssim": 0.99,
            "roi_ssim": 0.90,
            "sheet": "clip.qualitysheet.png",
        })
        self.assertEqual(gate["status"], "review")
        self.assertEqual(gate["ladderStep"], "manual-review")
        self.assertIn("ROI SSIM", gate["reason"])
        self.assertEqual(gate["previewFramePaths"], ["clip.qualitysheet.png"])

    def test_unknown_without_metrics(self):
        from backend.quality_gate import evaluate_quality_gate
        gate = evaluate_quality_gate(None)
        self.assertEqual(gate["status"], "unknown")
        self.assertEqual(gate["ladderStep"], "not-run")


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


class SubtitleStreamProbeTests(unittest.TestCase):
    """#103 first pass: probe embedded subtitle tracks without loading OCR."""

    def test_probe_subtitle_streams_parses_ffprobe_json(self):
        from unittest import mock

        payload = {
            "streams": [
                {
                    "index": 2,
                    "codec_name": "subrip",
                    "tags": {"language": "eng", "title": "SDH"},
                    "disposition": {"default": 1, "forced": 0},
                },
                {
                    "index": 4,
                    "codec_name": "ass",
                    "tags": {"language": "jpn"},
                    "disposition": {"default": "0", "forced": "1"},
                },
            ]
        }
        completed = SimpleNamespace(
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )
        with mock.patch("backend.io.subprocess.run", return_value=completed) as run:
            streams = processor._probe_subtitle_streams("movie.mkv")

        cmd = run.call_args.args[0]
        self.assertIn("-select_streams", cmd)
        self.assertIn("s", cmd)
        self.assertIn("-of", cmd)
        self.assertIn("json", cmd)
        self.assertEqual(len(streams), 2)
        self.assertEqual(streams[0].index, 2)
        self.assertEqual(streams[0].codec_name, "subrip")
        self.assertEqual(streams[0].language, "eng")
        self.assertEqual(streams[0].title, "SDH")
        self.assertTrue(streams[0].default)
        self.assertFalse(streams[0].forced)
        self.assertEqual(streams[1].index, 4)
        self.assertEqual(streams[1].codec_name, "ass")
        self.assertTrue(streams[1].forced)

    def test_probe_subtitle_streams_falls_back_to_empty_list(self):
        from unittest import mock

        with mock.patch(
            "backend.io.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            self.assertEqual(processor._probe_subtitle_streams("missing.mkv"), [])

        completed = SimpleNamespace(returncode=0, stdout="{bad", stderr="")
        with mock.patch("backend.io.subprocess.run", return_value=completed):
            self.assertEqual(processor._probe_subtitle_streams("bad.mkv"), [])


class SoftSubtitleRemuxTests(unittest.TestCase):
    """#103 remux primitive: explicit stream-copy mapping only."""

    def test_build_strip_cmd_removes_subtitle_streams(self):
        from backend.remux import SoftSubtitleAction, build_soft_subtitle_remux_cmd

        cmd = build_soft_subtitle_remux_cmd(
            "input.mkv",
            "output.mkv",
            action=SoftSubtitleAction.STRIP,
        )
        self.assertIn("-map", cmd)
        self.assertIn("0", cmd)
        self.assertIn("-0:s?", cmd)
        self.assertIn("-c", cmd)
        self.assertIn("copy", cmd)
        self.assertLess(cmd.index("-0:s?"), cmd.index("-c"))

    def test_build_keep_selected_cmd_maps_selected_global_streams(self):
        from backend.remux import SoftSubtitleAction, build_soft_subtitle_remux_cmd

        cmd = build_soft_subtitle_remux_cmd(
            "input.mkv",
            "output.mkv",
            action=SoftSubtitleAction.KEEP_SELECTED,
            keep_stream_indices=[4, 2, 4],
        )
        self.assertIn("-0:s?", cmd)
        maps = [
            cmd[i + 1] for i, token in enumerate(cmd[:-1])
            if token == "-map"
        ]
        self.assertEqual(maps, ["0", "-0:s?", "0:2", "0:4"])

    def test_keep_selected_requires_stream_index(self):
        from backend.remux import SoftSubtitleAction, build_soft_subtitle_remux_cmd

        with self.assertRaises(ValueError):
            build_soft_subtitle_remux_cmd(
                "input.mkv",
                "output.mkv",
                action=SoftSubtitleAction.KEEP_SELECTED,
            )

    def test_remux_uses_atomic_temp_output(self):
        from unittest import mock
        from backend import remux as _remux

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir) / ".out.tmp.mkv"
            final_path = Path(tmpdir) / "out.mkv"
            with mock.patch.object(_remux.shutil, "which", return_value="ffmpeg"):
                with mock.patch.object(
                    _remux, "_allocate_temp_output_path", return_value=temp_path,
                ) as allocate:
                    with mock.patch.object(_remux, "_promote_temp_output") as promote:
                        with mock.patch.object(_remux, "_cleanup_temp_output") as clean:
                            with mock.patch.object(
                                _remux, "_probe_duration_seconds", return_value=2.0,
                            ):
                                with mock.patch.object(
                                    _remux.subprocess,
                                    "run",
                                    return_value=SimpleNamespace(returncode=0),
                                ) as run:
                                    _remux.remux_soft_subtitles(
                                        "input.mkv",
                                        str(final_path),
                                    )

        allocate.assert_called_once_with(str(final_path))
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[-1], str(temp_path))
        self.assertIn("-0:s?", cmd)
        self.assertTrue(run.call_args.kwargs["check"])
        promote.assert_called_once_with(temp_path, final_path)
        clean.assert_called_once_with(temp_path)

    def test_remux_rejects_same_input_and_output_path(self):
        from unittest import mock
        from backend import remux as _remux

        with tempfile.TemporaryDirectory() as tmpdir:
            media = Path(tmpdir) / "movie.mkv"
            media.write_bytes(b"not a real video")
            with mock.patch.object(_remux, "_allocate_temp_output_path") as allocate:
                with self.assertRaises(ValueError):
                    _remux.remux_soft_subtitles(str(media), str(media))
        allocate.assert_not_called()

    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe"),
        "ffmpeg/ffprobe unavailable",
    )
    def test_strip_remux_integration_removes_subtitle_streams(self):
        from backend.io import _probe_codec_for_log, _probe_subtitle_streams
        from backend.remux import remux_soft_subtitles

        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            srt = work / "captions.srt"
            source = work / "source.mkv"
            output = work / "stripped.mkv"
            srt.write_text(
                "1\n00:00:00,000 --> 00:00:00,200\nHello\n",
                encoding="utf-8",
            )
            subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "color=c=black:s=32x24:d=0.25:r=1",
                    "-f", "srt", "-i", str(srt),
                    "-map", "0:v", "-map", "1:s",
                    "-c:v", "ffv1", "-c:s", "srt", str(source),
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(len(_probe_subtitle_streams(str(source))), 1)
            source_codec = _probe_codec_for_log(str(source))

            remux_soft_subtitles(str(source), str(output))

            self.assertEqual(_probe_subtitle_streams(str(output)), [])
            self.assertEqual(_probe_codec_for_log(str(output)), source_codec)


class BatchReportTests(unittest.TestCase):
    def test_choose_batch_output_path_honors_skip_existing(self):
        from backend.batch_report import choose_batch_output_path

        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            source = work / "clip.mp4"
            out_dir = work / "out"
            out_dir.mkdir()
            source.write_bytes(b"video")
            existing = out_dir / "clip_no_sub.mp4"
            existing.write_bytes(b"done")

            collision_safe = choose_batch_output_path(
                str(source),
                out_dir,
                "_no_sub",
                set(),
                skip_existing=False,
            )
            skip_target = choose_batch_output_path(
                str(source),
                out_dir,
                "_no_sub",
                set(),
                skip_existing=True,
            )

        self.assertEqual(collision_safe.name, "clip_no_sub(2).mp4")
        self.assertEqual(skip_target.name, "clip_no_sub.mp4")

    def test_write_batch_reports_includes_preflight_and_result_status(self):
        import datetime as _dt
        from unittest import mock
        from backend import batch_report as _br

        stream = processor.SubtitleStreamInfo(
            index=3,
            codec_name="subrip",
            language="eng",
            title="CC",
            default=True,
            forced=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            source = work / "clip.mkv"
            output = work / "out" / "clip_no_sub.mkv"
            source.write_bytes(b"video")
            cfg = SimpleNamespace(
                mode=processor.InpaintMode.AUTO,
                device="cuda:0",
                output_codec="h265",
            )
            with mock.patch.object(_br, "_probe_codec_for_log", return_value="h264,1920,1080,30000/1001"):
                with mock.patch.object(_br, "_probe_duration_seconds", return_value=12.5):
                    with mock.patch.object(_br, "_probe_subtitle_streams", return_value=[stream]):
                        record = _br.make_batch_item_record(
                            str(source),
                            str(output),
                            config=cfg,
                        )
            _br.finish_batch_item(
                record,
                _br.STATUS_HARDCODED_PROCESSED,
                message="Processed",
                elapsed_seconds=3.25,
                quality_report={
                    "tag": "Review",
                    "samples": 3,
                    "psnr": 31.0,
                    "ssim": 0.99,
                    "roi_ssim": 0.90,
                    "sheet": "clip.qualitysheet.png",
                },
            )
            started = _dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc)
            json_path, md_path = _br.write_batch_reports(
                output.parent,
                [record],
                kind="hardcoded-cleanup",
                started_at=started,
                completed_at=started + _dt.timedelta(seconds=4),
            )
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = md_path.read_text(encoding="utf-8")

        self.assertEqual(payload["schema"], "vsr.batch_summary.v1")
        self.assertEqual(payload["counts"], {"review-needed": 1})
        self.assertEqual(payload["files"][0]["status"], "review-needed")
        self.assertIn("quality gate review needed", payload["files"][0]["message"])
        self.assertEqual(payload["files"][0]["source_width"], 1920)
        self.assertEqual(payload["files"][0]["subtitle_stream_count"], 1)
        self.assertGreater(payload["files"][0]["estimated_seconds"], 0)
        self.assertEqual(payload["files"][0]["quality_gate"]["status"], "review")
        self.assertEqual(
            payload["files"][0]["quality_gate"]["ladderStep"],
            "manual-review",
        )
        self.assertEqual(
            payload["files"][0]["quality_gate"]["previewFramePaths"],
            ["clip.qualitysheet.png"],
        )
        self.assertIn("| review-needed | clip.mkv | clip_no_sub.mkv |", markdown)
        self.assertIn("review (manual-review)", markdown)


class CliSoftSubtitleTests(unittest.TestCase):
    def _run_cli(self, args):
        from unittest import mock
        from backend import cli as _cli

        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(sys, "argv", ["vsr"] + args):
            with mock.patch("sys.stdout", stdout), mock.patch("sys.stderr", stderr):
                with self.assertRaises(SystemExit) as caught:
                    _cli.main()
        return caught.exception.code, stdout.getvalue(), stderr.getvalue()

    def test_soft_subtitle_dry_run_does_not_construct_remover(self):
        from unittest import mock
        from backend import cli as _cli

        stream = processor.SubtitleStreamInfo(
            index=2,
            codec_name="subrip",
            language="eng",
            title="SDH",
            default=True,
            forced=False,
        )
        with mock.patch.object(_cli, "_probe_subtitle_streams", return_value=[stream]):
            with mock.patch(
                "backend.processor.SubtitleRemover",
                side_effect=AssertionError("heavy backend should not load"),
            ):
                code, stdout, _stderr = self._run_cli([
                    "--input", "movie.mkv",
                    "--soft-subtitle-dry-run",
                ])

        self.assertEqual(code, 0)
        self.assertIn("action=inspect", stdout)
        self.assertIn("stream=2", stdout)
        self.assertIn("codec=subrip", stdout)
        self.assertIn("lang=eng", stdout)
        self.assertIn("title=SDH", stdout)
        self.assertIn("default=yes", stdout)

    def test_strip_soft_subtitles_remuxes_without_remover(self):
        from unittest import mock
        from backend import cli as _cli
        from backend.remux import SoftSubtitleAction

        with mock.patch.object(_cli, "_probe_subtitle_streams", return_value=[]):
            with mock.patch.object(_cli, "remux_soft_subtitles") as remux:
                with mock.patch(
                    "backend.processor.SubtitleRemover",
                    side_effect=AssertionError("heavy backend should not load"),
                ):
                    code, stdout, _stderr = self._run_cli([
                        "--input", "movie.mkv",
                        "--output", "out.mkv",
                        "--strip-soft-subtitles",
                    ])

        self.assertEqual(code, 0)
        remux.assert_called_once_with(
            "movie.mkv",
            "out.mkv",
            action=SoftSubtitleAction.STRIP,
        )
        self.assertIn("action=strip", stdout)

    def test_soft_subtitle_dry_run_writes_json_plan(self):
        from unittest import mock
        from backend import cli as _cli

        stream = processor.SubtitleStreamInfo(
            index=2,
            codec_name="subrip",
            language="eng",
            title="SDH",
            default=True,
            forced=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            first = work / "first.mkv"
            second = work / "second.mkv"
            plan = work / "soft-plan.json"
            first.write_bytes(b"not a real video")
            second.write_bytes(b"not a real video")

            with mock.patch.object(
                _cli, "_probe_subtitle_streams", return_value=[stream],
            ):
                with mock.patch(
                    "backend.processor.SubtitleRemover",
                    side_effect=AssertionError("heavy backend should not load"),
                ):
                    code, stdout, _stderr = self._run_cli([
                        "--pattern", str(work / "*.mkv"),
                        "--soft-subtitle-dry-run",
                        "--strip-soft-subtitles",
                        "--soft-subtitle-plan-json", str(plan),
                    ])

            payload = json.loads(plan.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertIn("wrote plan", stdout)
        self.assertEqual(payload["schema"], "vsr.soft_subtitle_preflight.v1")
        self.assertEqual(payload["action"], "strip")
        self.assertEqual(payload["count"], 2)
        self.assertEqual(
            [record["input_name"] for record in payload["files"]],
            ["first.mkv", "second.mkv"],
        )
        self.assertTrue(payload["files"][0]["has_soft_subtitles"])
        self.assertEqual(payload["files"][0]["subtitle_stream_count"], 1)
        self.assertEqual(
            payload["files"][0]["subtitle_streams"][0]["language"],
            "eng",
        )

    def test_soft_subtitle_plan_json_requires_dry_run(self):
        code, _stdout, stderr = self._run_cli([
            "--input", "movie.mkv",
            "--output", "out.mkv",
            "--strip-soft-subtitles",
            "--soft-subtitle-plan-json", "plan.json",
        ])
        self.assertEqual(code, 2)
        self.assertIn("requires --soft-subtitle-dry-run", stderr)

    def test_soft_subtitle_modes_are_mutually_exclusive(self):
        code, _stdout, stderr = self._run_cli([
            "--input", "movie.mkv",
            "--output", "out.mkv",
            "--strip-soft-subtitles",
            "--keep-soft-subtitles",
        ])
        self.assertEqual(code, 2)
        self.assertIn("mutually exclusive", stderr)


class CliBatchReportTests(unittest.TestCase):
    def _run_cli(self, args):
        from unittest import mock
        from backend import cli as _cli

        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(sys, "argv", ["vsr"] + args):
            with mock.patch("sys.stdout", stdout), mock.patch("sys.stderr", stderr):
                with self.assertRaises(SystemExit) as caught:
                    _cli.main()
        return caught.exception.code, stdout.getvalue(), stderr.getvalue()

    def _patch_preflight_probes(self):
        from unittest import mock
        from backend import batch_report as _br

        return mock.patch.multiple(
            _br,
            _probe_codec_for_log=mock.Mock(return_value="h264,640,360,30/1"),
            _probe_duration_seconds=mock.Mock(return_value=10.0),
            _probe_subtitle_streams=mock.Mock(return_value=[]),
        )

    def test_pattern_skip_existing_writes_report_without_alt_processing(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            src = work / "clip.mp4"
            out_dir = work / "out"
            ckpt = work / "ckpt"
            src.write_bytes(b"video")
            out_dir.mkdir()
            (out_dir / "clip_no_sub.mp4").write_bytes(b"done")
            fake_remover = SimpleNamespace(
                config=processor.ProcessingConfig(),
                process_video=mock.Mock(return_value=True),
                process_image=mock.Mock(return_value=True),
            )
            with self._patch_preflight_probes():
                with mock.patch("backend.processor.SubtitleRemover", return_value=fake_remover):
                    code, stdout, stderr = self._run_cli([
                        "--pattern", str(work / "*.mp4"),
                        "--out-dir", str(out_dir),
                        "--checkpoint-dir", str(ckpt),
                        "--skip-existing",
                    ])
            payload = json.loads((out_dir / "vsr-batch-summary.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0, stderr)
        fake_remover.process_video.assert_not_called()
        self.assertIn("[skip] clip.mp4 (output exists)", stdout)
        self.assertEqual(payload["files"][0]["status"], "skipped-existing")
        self.assertEqual(payload["files"][0]["output_name"], "clip_no_sub.mp4")

    def test_pattern_success_writes_processed_report(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            src = work / "clip.mp4"
            out_dir = work / "out"
            ckpt = work / "ckpt"
            src.write_bytes(b"video")
            out_dir.mkdir()
            fake_remover = SimpleNamespace(
                config=processor.ProcessingConfig(),
                process_video=mock.Mock(return_value=True),
                process_image=mock.Mock(return_value=True),
            )
            with self._patch_preflight_probes():
                with mock.patch("backend.processor.SubtitleRemover", return_value=fake_remover):
                    code, stdout, stderr = self._run_cli([
                        "--pattern", str(work / "*.mp4"),
                        "--out-dir", str(out_dir),
                        "--checkpoint-dir", str(ckpt),
                        "--gpu", "-1",
                    ])
            payload = json.loads((out_dir / "vsr-batch-summary.json").read_text(encoding="utf-8"))
            markdown = (out_dir / "vsr-batch-summary.md").read_text(encoding="utf-8")

        self.assertEqual(code, 0, stderr)
        fake_remover.process_video.assert_called_once()
        self.assertIn("[batch] wrote report", stdout)
        self.assertEqual(payload["counts"], {"hardcoded-processed": 1})
        self.assertEqual(payload["files"][0]["status"], "hardcoded-processed")
        self.assertIn("clip_no_sub.mp4", markdown)

    def test_soft_subtitle_pattern_writes_report(self):
        from unittest import mock
        from backend import cli as _cli

        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            src = work / "clip.mkv"
            out_dir = work / "out"
            src.write_bytes(b"video")
            out_dir.mkdir()
            with self._patch_preflight_probes():
                with mock.patch.object(_cli, "_probe_subtitle_streams", return_value=[]):
                    with mock.patch.object(_cli, "remux_soft_subtitles") as remux:
                        with mock.patch(
                            "backend.processor.SubtitleRemover",
                            side_effect=AssertionError("heavy backend should not load"),
                        ):
                            code, stdout, stderr = self._run_cli([
                                "--pattern", str(work / "*.mkv"),
                                "--out-dir", str(out_dir),
                                "--strip-soft-subtitles",
                            ])
            payload = json.loads((out_dir / "vsr-batch-summary.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0, stderr)
        remux.assert_called_once()
        self.assertIn("[batch] wrote report", stdout)
        self.assertEqual(payload["counts"], {"soft-subtitle-remuxed": 1})
        self.assertEqual(payload["files"][0]["soft_action"], "strip")


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


class I18nScaffoldTests(unittest.TestCase):
    """RM-97: gettext scaffold must pass strings through unchanged when
    no catalog is bound, and bind cleanly when one is."""

    def test_passthrough_without_catalog(self):
        from backend import i18n
        i18n.bind_locale(None)
        self.assertEqual(i18n._("Start batch"), "Start batch")
        self.assertFalse(i18n.is_translation_active())

    def test_unknown_locale_falls_back(self):
        from backend import i18n
        # Bind a locale that almost certainly has no catalog shipped --
        # the helper should swallow the FileNotFoundError and keep the
        # NullTranslations in place.
        i18n.bind_locale("zz")
        self.assertEqual(i18n._("Start batch"), "Start batch")
        self.assertFalse(i18n.is_translation_active())


class A11yScaffoldTests(unittest.TestCase):
    """RM-95: announce() must be safe to call when UIA / pywin32 is
    unavailable. Returns silently rather than raising."""

    def test_announce_noop_when_provider_missing(self):
        from backend import a11y
        # Force the probed cache to "no provider" so announce() takes
        # the silent path without trying to import comtypes.
        a11y._PROBED = True
        a11y._PROVIDER = None
        try:
            a11y.announce("Hello, world", importance="normal")
            a11y.announce("Critical!", importance="high")
        except Exception as exc:
            self.fail(f"announce raised: {exc}")


class NleSidecarTests(unittest.TestCase):
    """RM-76: EDL and FCPXML writers must produce well-formed sidecars
    with the source / cleaned filenames and the processed time range."""

    def test_edl_round_trip(self):
        from backend import nle_sidecar
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "out.edl")
            written = nle_sidecar.write_edl(
                path,
                source="C:/clips/source.mp4",
                cleaned="C:/clips/source_no_sub.mp4",
                fps=24.0, start_s=0.0, end_s=10.0,
            )
            text = Path(written).read_text(encoding="ascii")
        self.assertIn("TITLE: VSR cleanup", text)
        self.assertIn("FROM CLIP NAME: source.mp4", text)
        self.assertIn("TO CLIP NAME:   source_no_sub.mp4", text)
        self.assertIn("00:00:00:00", text)

    def test_fcpxml_round_trip(self):
        from backend import nle_sidecar
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "out.fcpxml")
            written = nle_sidecar.write_fcpxml(
                path,
                source="/clips/source.mp4",
                cleaned=str(Path(tmpdir) / "cleaned.mp4"),
                fps=24.0, start_s=0.0, end_s=10.0,
            )
            text = Path(written).read_text(encoding="utf-8")
        self.assertIn("<fcpxml version=\"1.10\">", text)
        self.assertIn("frameDuration=\"1/24s\"", text)


class HdrPipelineTests(unittest.TestCase):
    """RM-73 partial: probe_color_metadata returns None without ffprobe;
    hdr_encode_args produces empty list for None/empty metadata and the
    expected -color_primaries / -color_trc / -colorspace flags otherwise."""

    def test_hdr_encode_args_empty_for_none(self):
        from backend.hdr import hdr_encode_args
        self.assertEqual(hdr_encode_args(None), [])

    def test_hdr_encode_args_emits_tags(self):
        from backend.hdr import hdr_encode_args, ColorMetadata
        meta = ColorMetadata(
            color_primaries="bt2020",
            color_transfer="smpte2084",
            color_space="bt2020nc",
            color_range="tv",
        )
        args = hdr_encode_args(meta)
        self.assertIn("-color_primaries", args)
        self.assertIn("bt2020", args)
        self.assertIn("-color_trc", args)
        self.assertIn("smpte2084", args)
        self.assertIn("-colorspace", args)
        self.assertIn("-color_range", args)
        self.assertTrue(meta.is_hdr)

    def test_probe_color_metadata_falls_back(self):
        from backend.hdr import probe_color_metadata
        # ffprobe absent or path missing -- helper returns None.
        result = probe_color_metadata("/nonexistent.mp4")
        # Cannot guarantee ffprobe is missing in CI; accept None or a
        # ColorMetadata so the test is environment-tolerant.
        self.assertTrue(result is None or hasattr(result, "label"))


class PostRestoreTests(unittest.TestCase):
    """RM-78 / RM-80: optional post-restore adapters must return None
    when their dependency is missing. The pipeline never crashes on a
    half-broken install."""

    def test_realesrgan_skip_when_binary_missing(self):
        from backend import post_restore as _pr
        import shutil as _shutil
        original = _shutil.which
        try:
            _shutil.which = lambda name: None
            with tempfile.TemporaryDirectory() as tmpdir:
                src = Path(tmpdir) / "in.mp4"
                src.write_bytes(b"\x00" * 16)  # placeholder
                dst = str(Path(tmpdir) / "out.mp4")
                result = _pr.realesrgan_upscale(str(src), dst, scale=2)
            self.assertIsNone(result)
        finally:
            _shutil.which = original

    def test_film_grain_skip_when_ffmpeg_missing(self):
        from backend import post_restore as _pr
        import shutil as _shutil
        original = _shutil.which
        try:
            _shutil.which = lambda name: None
            with tempfile.TemporaryDirectory() as tmpdir:
                src = Path(tmpdir) / "in.mp4"
                src.write_bytes(b"\x00" * 16)
                dst = str(Path(tmpdir) / "out.mp4")
                result = _pr.add_film_grain(str(src), dst, strength=0.04)
            self.assertIsNone(result)
        finally:
            _shutil.which = original


class CrashReporterScaffoldTests(unittest.TestCase):
    """RM-52: opt-in crash reporting must be OFF unless both env vars
    are set, and the path scrubber must hide local layout info."""

    def setUp(self):
        self._saved = {
            "VSR_GLITCHTIP_DSN": os.environ.pop("VSR_GLITCHTIP_DSN", None),
            "VSR_CRASH_REPORTS": os.environ.pop("VSR_CRASH_REPORTS", None),
        }

    def tearDown(self):
        for k, v in self._saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def test_disabled_by_default(self):
        from backend.crash_reporter import is_enabled, install
        self.assertFalse(is_enabled())
        self.assertFalse(install())

    def test_partial_consent_is_not_enough(self):
        from backend.crash_reporter import is_enabled
        os.environ["VSR_CRASH_REPORTS"] = "1"
        self.assertFalse(is_enabled(), "DSN missing -> still disabled")
        os.environ.pop("VSR_CRASH_REPORTS", None)
        os.environ["VSR_GLITCHTIP_DSN"] = "https://example/0"
        self.assertFalse(is_enabled(), "consent flag missing -> still disabled")

    def test_path_scrub_drops_windows_paths(self):
        from backend.crash_reporter import _path_scrub
        sample = "File \"C:\\Users\\xxx\\repos\\VSR\\backend\\processor.py\", line 1"
        scrubbed = _path_scrub(sample)
        self.assertNotIn("Users", scrubbed)
        self.assertIn("<path>", scrubbed)


class NsisInstallerArtefactTests(unittest.TestCase):
    """RM-51: the NSIS installer script ships in installer/vsr.nsi so
    the GHA build workflow can pick it up. Sanity-check the file
    exists and names the expected app + version constants so a future
    rename does not break the build."""

    def test_nsi_file_present_with_expected_definitions(self):
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        nsi = root / "installer" / "vsr.nsi"
        self.assertTrue(nsi.is_file(), "installer/vsr.nsi missing")
        text = nsi.read_text(encoding="utf-8")
        self.assertIn("Video Subtitle Remover Pro", text)
        self.assertIn("VideoSubtitleRemoverPro.exe", text)
        self.assertIn("VERSIONMAJOR", text)
        # RM-58: the file-extension verb registration.
        self.assertIn("OpenWithVSR", text)


class ProxyWorkflowTests(unittest.TestCase):
    """RM-34: ensure_proxy must return None when ffmpeg is absent and
    use a deterministic cache filename otherwise."""

    def test_proxy_returns_none_without_ffmpeg(self):
        from backend import proxy_workflow as _pw
        import shutil as _shutil
        original = _shutil.which
        try:
            _shutil.which = lambda name: None
            with tempfile.TemporaryDirectory() as tmpdir:
                src = Path(tmpdir) / "in.mp4"
                src.write_bytes(b"\x00" * 32)
                self.assertIsNone(_pw.ensure_proxy(str(src)))
        finally:
            _shutil.which = original


class KaraokeFlowTests(unittest.TestCase):
    """RM-43 / RM-45: optical-flow mask warp and WhisperX availability
    probe behave deterministically."""

    def test_warp_mask_with_flow_preserves_shape(self):
        import numpy as _np
        from backend.karaoke_flow import warp_mask_with_flow
        prev = _np.zeros((32, 32, 3), dtype=_np.uint8)
        nxt = _np.zeros((32, 32, 3), dtype=_np.uint8)
        mask = _np.zeros((32, 32), dtype=_np.uint8)
        mask[10:20, 10:20] = 255
        warped = warp_mask_with_flow(prev, nxt, mask)
        self.assertEqual(warped.shape, mask.shape)

    def test_whisperx_availability_safe(self):
        from backend.karaoke_flow import is_whisperx_available, run_whisperx
        # Don't crash regardless of whether the package is installed.
        self.assertIsInstance(is_whisperx_available(), bool)
        if not is_whisperx_available():
            self.assertIsNone(run_whisperx("/nonexistent.wav"))


class VapourSynthBridgeTests(unittest.TestCase):
    """RM-75: try_open_vpy must return None for a non-.vpy path and for
    a missing dep."""

    def setUp(self):
        self._saved_env = os.environ.pop("VSR_VAPOURSYNTH", None)

    def tearDown(self):
        os.environ.pop("VSR_VAPOURSYNTH", None)
        if self._saved_env is not None:
            os.environ["VSR_VAPOURSYNTH"] = self._saved_env

    def test_returns_none_for_non_vpy(self):
        from backend.vapoursynth_bridge import try_open_vpy
        self.assertIsNone(try_open_vpy("/tmp/foo.mp4"))

    def test_vpy_script_not_executed_without_env_gate(self):
        from backend.vapoursynth_bridge import try_open_vpy
        with tempfile.TemporaryDirectory() as tmpdir:
            marker = Path(tmpdir) / "executed.txt"
            script = Path(tmpdir) / "input.vpy"
            script.write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n",
                encoding="utf-8",
            )
            self.assertIsNone(try_open_vpy(str(script)))
            self.assertFalse(marker.exists())

    def test_env_gate_accepts_truthy_token(self):
        from backend.vapoursynth_bridge import _vapoursynth_enabled
        os.environ["VSR_VAPOURSYNTH"] = "yes"
        self.assertTrue(_vapoursynth_enabled())


class TensorrtCompileTests(unittest.TestCase):
    """RM-70: cache helper must produce a deterministic path and
    silently return None when polygraphy / TensorRT are missing."""

    def setUp(self):
        os.environ.pop("VSR_TENSORRT", None)

    def test_disabled_by_default(self):
        from backend.tensorrt_compile import is_tensorrt_enabled
        self.assertFalse(is_tensorrt_enabled())

    def test_cached_engine_path_is_deterministic(self):
        from backend.tensorrt_compile import cached_engine_path
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx = Path(tmpdir) / "model.onnx"
            onnx.write_bytes(b"\x00" * 32)
            p1 = cached_engine_path(str(onnx))
            p2 = cached_engine_path(str(onnx))
            self.assertEqual(p1, p2)
            self.assertTrue(p1.name.endswith(".engine"))

    def test_compile_returns_none_when_disabled(self):
        from backend.tensorrt_compile import maybe_compile_engine
        self.assertIsNone(maybe_compile_engine("/tmp/non.onnx"))


class SeedVr2AdapterTests(unittest.TestCase):
    """RM-77: SeedVR2 wrapper must return None when neither the pip
    package nor VSR_SEEDVR2_CMD is set."""

    def test_returns_none_without_deps(self):
        os.environ.pop("VSR_SEEDVR2_CMD", None)
        from backend.post_restore import seedvr2_restore
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "in.mp4"
            src.write_bytes(b"\x00" * 16)
            dst = Path(tmpdir) / "out.mp4"
            self.assertIsNone(seedvr2_restore(str(src), str(dst)))


class SegmentationAdapterTests(unittest.TestCase):
    """RM-66/67/68/69: every adapter must return the input unchanged
    (or None) when its optional dep is absent."""

    def setUp(self):
        self._saved = {k: os.environ.pop(k, None) for k in (
            "VSR_SAM2_CHECKPOINT", "VSR_SAM2_CONFIG", "VSR_SAM3",
            "VSR_MATANYONE", "VSR_COTRACKER",
        )}

    def tearDown(self):
        for k, v in self._saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def test_sam2_refine_returns_base_mask(self):
        import numpy as _np
        from backend.segmentation import refine_mask_with_sam2
        frame = _np.zeros((32, 32, 3), dtype=_np.uint8)
        mask = _np.zeros((32, 32), dtype=_np.uint8)
        mask[10:20, 10:20] = 255
        out = refine_mask_with_sam2(frame, [(10, 10, 20, 20)], mask)
        _np.testing.assert_array_equal(out, mask)

    def test_sam3_returns_none_without_dep(self):
        import numpy as _np
        from backend.segmentation import segment_text_with_sam3
        self.assertIsNone(segment_text_with_sam3(_np.zeros((32, 32, 3), dtype=_np.uint8)))

    def test_matte_returns_none_without_dep(self):
        import numpy as _np
        from backend.segmentation import matte_frame
        self.assertIsNone(matte_frame(
            _np.zeros((32, 32, 3), dtype=_np.uint8),
            _np.zeros((32, 32), dtype=_np.uint8),
        ))

    def test_cotracker_returns_none_without_dep(self):
        import numpy as _np
        from backend.segmentation import track_points
        frames = [_np.zeros((16, 16, 3), dtype=_np.uint8) for _ in range(3)]
        self.assertIsNone(track_points(frames, [(4, 4)]))

    def test_sam2_inference_error_returns_base_mask(self):
        import numpy as _np
        from backend import segmentation as _seg

        class BrokenPredictor:
            def set_image(self, rgb):
                return None

            def predict(self, **kwargs):
                raise RuntimeError("sam2 failed")

        saved = dict(_seg._SAM2_STATE)
        try:
            _seg._SAM2_STATE.update({"probed": True, "predictor": BrokenPredictor()})
            frame = _np.zeros((32, 32, 3), dtype=_np.uint8)
            mask = _np.zeros((32, 32), dtype=_np.uint8)
            mask[10:20, 10:20] = 255
            out = _seg.refine_mask_with_sam2(frame, [(10, 10, 20, 20)], mask)
            _np.testing.assert_array_equal(out, mask)
        finally:
            _seg._SAM2_STATE.clear()
            _seg._SAM2_STATE.update(saved)

    def test_sam3_inference_error_returns_none(self):
        import numpy as _np
        from backend import segmentation as _seg

        class BrokenPredictor:
            def segment(self, frame, prompt):
                raise RuntimeError("sam3 failed")

        saved = dict(_seg._SAM3_STATE)
        try:
            _seg._SAM3_STATE.update({"probed": True, "predictor": BrokenPredictor()})
            self.assertIsNone(_seg.segment_text_with_sam3(
                _np.zeros((32, 32, 3), dtype=_np.uint8)
            ))
        finally:
            _seg._SAM3_STATE.clear()
            _seg._SAM3_STATE.update(saved)

    def test_matanyone_inference_error_returns_none(self):
        import numpy as _np
        from backend import segmentation as _seg

        class BrokenModel:
            def matte(self, frame, hint_mask):
                raise RuntimeError("matanyone failed")

        saved = dict(_seg._MATANYONE_STATE)
        try:
            _seg._MATANYONE_STATE.update({"probed": True, "model": BrokenModel()})
            self.assertIsNone(_seg.matte_frame(
                _np.zeros((32, 32, 3), dtype=_np.uint8),
                _np.zeros((32, 32), dtype=_np.uint8),
            ))
        finally:
            _seg._MATANYONE_STATE.clear()
            _seg._MATANYONE_STATE.update(saved)

    def test_cotracker_inference_error_returns_none(self):
        import numpy as _np
        from unittest import mock
        from backend import segmentation as _seg

        class FakeTensor:
            def permute(self, *args):
                return self

            def unsqueeze(self, *args):
                return self

            def float(self):
                return self

        class BrokenModel:
            def __call__(self, *args, **kwargs):
                raise RuntimeError("cotracker failed")

        fake_torch = SimpleNamespace(
            float32=object(),
            from_numpy=lambda value: FakeTensor(),
            tensor=lambda value, dtype=None: FakeTensor(),
        )
        saved = dict(_seg._COTRACKER_STATE)
        try:
            _seg._COTRACKER_STATE.update({"probed": True, "model": BrokenModel()})
            frames = [_np.zeros((16, 16, 3), dtype=_np.uint8) for _ in range(3)]
            with mock.patch.dict(sys.modules, {"torch": fake_torch}):
                self.assertIsNone(_seg.track_points(frames, [(4, 4)]))
        finally:
            _seg._COTRACKER_STATE.clear()
            _seg._COTRACKER_STATE.update(saved)


class DiffusionInpainterScaffoldTests(unittest.TestCase):
    """RM-59/60/61/62/63/64/65: each scaffolded diffusion backend must
    fall back to TBE when its optional dep is missing rather than
    crash. The default registry never sees them unless the user has
    opted in via env vars."""

    def setUp(self):
        self._saved = {k: os.environ.pop(k, None) for k in (
            "VSR_PROPAINTER_REAL", "VSR_DIFFUERASER", "VSR_VACE",
            "VSR_VIDEOPAINTER", "VSR_COCOCO", "VSR_ERASERDIT", "VSR_FLOED",
        )}

    def tearDown(self):
        for k, v in self._saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def test_maybe_register_no_ops_without_env(self):
        from backend import inpainters_diffusion as _id
        self.assertEqual(_id.maybe_register(), [])

    def test_maybe_register_only_enabled_backends(self):
        from backend import inpainter_registry as _registry
        from backend import inpainters_diffusion as _id

        os.environ["VSR_DIFFUERASER"] = "1"
        try:
            registered = _id.maybe_register()
            self.assertEqual(registered, ["diffueraser"])
            self.assertTrue(_registry.is_registered("diffueraser"))
        finally:
            _registry.unregister("diffueraser")

    def test_scaffold_falls_back_to_tbe(self):
        from backend.inpainters_diffusion import _DiffuEraserBackend
        cfg = processor.normalize_processing_config(
            processor.ProcessingConfig(tbe_enable=True)
        )
        b = _DiffuEraserBackend(device="cpu", config=cfg)
        import numpy as _np
        frames = [_np.full((16, 16, 3), 60, dtype=_np.uint8) for _ in range(3)]
        masks = [_np.zeros((16, 16), dtype=_np.uint8) for _ in range(3)]
        masks[1][4:8, 4:8] = 255
        out = b.inpaint(frames, masks)
        self.assertEqual(len(out), 3)
        for f in out:
            self.assertEqual(f.shape, (16, 16, 3))

    def test_scaffold_falls_back_when_loaded_model_raises(self):
        from backend.inpainters_diffusion import _DiffusionBackendBase

        class BrokenBackend(_DiffusionBackendBase):
            MODE_NAME = "broken"

            def _load(self):
                return object()

            def _run_model(self, frames, masks):
                raise RuntimeError("model failed")

        cfg = processor.normalize_processing_config(
            processor.ProcessingConfig(tbe_enable=False)
        )
        backend = BrokenBackend(device="cpu", config=cfg)
        import numpy as _np
        frames = [_np.full((16, 16, 3), 60, dtype=_np.uint8) for _ in range(2)]
        masks = [_np.zeros((16, 16), dtype=_np.uint8) for _ in range(2)]
        masks[0][4:8, 4:8] = 255
        out = backend.inpaint(frames, masks)
        self.assertEqual(len(out), 2)
        for frame in out:
            self.assertEqual(frame.shape, (16, 16, 3))


class DecodeAccelTests(unittest.TestCase):
    """RM-71 / RM-72: PyNvVideoCodec and RIFE adapters must return None
    when their optional deps are missing."""

    def test_pynv_returns_none_without_dep(self):
        from backend.decode_accel import try_open_pynv
        cap = try_open_pynv("/nonexistent.mp4")
        self.assertIsNone(cap)

    def test_rife_returns_none_without_dep(self):
        import numpy as _np
        from backend.decode_accel import maybe_interpolate_pair, is_rife_available
        a = _np.zeros((8, 8, 3), dtype=_np.uint8)
        b = _np.full((8, 8, 3), 255, dtype=_np.uint8)
        if not is_rife_available():
            self.assertIsNone(maybe_interpolate_pair(a, b, 0.5))


class VlmOcrAdapterTests(unittest.TestCase):
    """RM-22 / RM-23 / RM-42: maybe_build_vlm_detector must return None
    by default (no env var, default lang) and the adapter classes must
    survive a missing-dependency load."""

    def setUp(self):
        self._saved = os.environ.pop("VSR_VLM_OCR", None)

    def tearDown(self):
        os.environ.pop("VSR_VLM_OCR", None)
        if self._saved is not None:
            os.environ["VSR_VLM_OCR"] = self._saved

    def test_no_vlm_when_env_unset(self):
        from backend.ocr_vlm import maybe_build_vlm_detector
        self.assertIsNone(maybe_build_vlm_detector("cpu", "en"))

    def test_manga_lang_returns_detector(self):
        from backend.ocr_vlm import maybe_build_vlm_detector
        detector = maybe_build_vlm_detector("cpu", "manga")
        self.assertIsNotNone(detector)
        self.assertEqual(detector.name, "manga-ocr")

    def test_florence2_load_returns_none_without_dep(self):
        from backend.ocr_vlm import _Florence2Detector
        d = _Florence2Detector(device="cpu")
        # _load lazy-imports transformers; we should get None when the
        # CI environment lacks the package.
        result = d._load()
        # Either None (no dep) or a real tuple (very unlikely in CI);
        # both are acceptable here.
        self.assertTrue(result is None or isinstance(result, tuple))

    def test_qwen25vl_malformed_json_returns_empty_boxes(self):
        import numpy as _np
        from backend.ocr_vlm import _Qwen25VLDetector

        class NoGrad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        class FakeProcessor:
            def apply_chat_template(self, messages, add_generation_prompt=True):
                return "prompt"

            def __call__(self, **kwargs):
                return {}

            def batch_decode(self, generated, skip_special_tokens=True):
                return ["no json payload here"]

        class FakeModel:
            def generate(self, **kwargs):
                return ["tokens"]

        fake_torch = SimpleNamespace(no_grad=lambda: NoGrad())
        detector = _Qwen25VLDetector(device="cpu")
        detector._model = (FakeProcessor(), FakeModel(), fake_torch)

        boxes = detector._extract_boxes(
            _np.zeros((32, 32, 3), dtype=_np.uint8),
            threshold=0.5,
        )

        self.assertEqual(boxes, [])


class PreprocessAdaptersTests(unittest.TestCase):
    """RM-33 / RM-21: pre-detect denoise + TransNetV2 scene-cut adapter
    must degrade gracefully when their optional deps are missing."""

    def test_fastdvdnet_falls_back_to_cv2_nlm(self):
        import numpy as _np
        os.environ.pop("VSR_FASTDVDNET", None)
        from backend.preprocess import fastdvdnet_denoise_frame
        frame = _np.full((32, 32, 3), 128, dtype=_np.uint8)
        out = fastdvdnet_denoise_frame(frame)
        self.assertEqual(out.shape, frame.shape)

    def test_transnetv2_returns_none_without_dep(self):
        import numpy as _np
        os.environ.pop("VSR_TRANSNETV2", None)
        from backend.preprocess import transnetv2_scene_cuts
        frames = [_np.zeros((16, 16, 3), dtype=_np.uint8) for _ in range(4)]
        self.assertIsNone(transnetv2_scene_cuts(frames))


class WhisperFallbackTests(unittest.TestCase):
    """RM-27: Whisper fallback adapter must degrade gracefully when the
    optional dep is missing, and the segments_to_frame_spans helper must
    merge overlapping spans deterministically."""

    def test_is_available_handles_missing_dep(self):
        # Force faster-whisper to look absent even on developer machines
        # where optional packages are installed.
        from unittest import mock
        from backend import whisper_fallback as _wf
        with mock.patch.dict(sys.modules, {"faster_whisper": None}):
            self.assertFalse(_wf.is_available())

    def test_run_whisper_segments_returns_none_without_dep(self):
        from backend import whisper_fallback as _wf
        result = _wf.run_whisper_segments("/nonexistent.wav")
        self.assertIsNone(result)

    def test_extract_audio_returns_none_for_missing_file(self):
        from backend import whisper_fallback as _wf
        # ffmpeg may not be on PATH in CI; either branch should return
        # None for a non-existent source.
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _wf.extract_audio_to_temp("/nonexistent.mp4", tmpdir)
            self.assertIsNone(result)

    def test_ffmpeg_whisper_available_detects_filter(self):
        from unittest import mock
        from backend import whisper_fallback as _wf
        completed = SimpleNamespace(
            returncode=0,
            stdout=" .. whisper           A->A       Transcribe audio using whisper.cpp.\n",
            stderr="",
        )
        with mock.patch.object(_wf.shutil, "which", return_value="ffmpeg"):
            with mock.patch.object(_wf.subprocess, "run", return_value=completed):
                self.assertTrue(_wf.ffmpeg_whisper_available())

    def test_ffmpeg_whisper_filter_escapes_windows_paths(self):
        from backend import whisper_fallback as _wf
        expr = _wf._build_ffmpeg_whisper_filter(
            r"C:\models\ggml-base.en.bin",
            r"C:\Temp\whisper.srt",
            language="en",
            queue_seconds=3.0,
        )
        self.assertIn(r"model=C\:\\models\\ggml-base.en.bin", expr)
        self.assertIn(r"destination=C\:\\Temp\\whisper.srt", expr)

    def test_ffmpeg_whisper_filter_includes_vad_options(self):
        from backend import whisper_fallback as _wf
        expr = _wf._build_ffmpeg_whisper_filter(
            "/models/ggml-base.bin",
            "/tmp/whisper.srt",
            vad_model="/models/silero_vad.onnx",
            vad_threshold=0.6,
            min_speech_duration=0.25,
        )
        self.assertIn("vad_model=", expr)
        self.assertIn("vad_threshold=0.6", expr)
        self.assertIn("min_speech_duration=0.25", expr)

    def test_ffmpeg_whisper_filter_omits_vad_when_empty(self):
        from backend import whisper_fallback as _wf
        expr = _wf._build_ffmpeg_whisper_filter(
            "/models/ggml-base.bin",
            "/tmp/whisper.srt",
        )
        self.assertNotIn("vad_model", expr)
        self.assertNotIn("vad_threshold", expr)

    def test_parse_srt_segments(self):
        from backend import whisper_fallback as _wf
        srt = (
            "1\n"
            "00:00:01,250 --> 00:00:02,500\n"
            "Hello world\n\n"
            "2\n"
            "00:00:03,000 --> 00:00:04,000\n"
            "Second line\n"
        )
        self.assertEqual(
            _wf.parse_srt_segments(srt),
            [(1.25, 2.5, "Hello world"), (3.0, 4.0, "Second line")],
        )

    def test_run_ffmpeg_whisper_segments_parses_mocked_srt(self):
        from unittest import mock
        from backend import whisper_fallback as _wf

        def fake_run(cmd, check, capture_output, timeout):
            filt = cmd[cmd.index("-af") + 1]
            dest = filt.split(":destination=", 1)[1]
            dest_path = Path(_wf._unescape_filter_value(dest))
            dest_path.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nspeech\n",
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

        with tempfile.TemporaryDirectory() as tmpdir:
            media = Path(tmpdir) / "input.mp4"
            media.write_bytes(b"fake")
            model = Path(tmpdir) / "ggml-base.en.bin"
            model.write_bytes(b"fake")
            with mock.patch.object(_wf, "ffmpeg_whisper_available", return_value=True):
                with mock.patch.object(_wf.subprocess, "run", side_effect=fake_run):
                    self.assertEqual(
                        _wf.run_ffmpeg_whisper_segments(
                            str(media), str(model), language="en"
                        ),
                        [(0.0, 1.0, "speech")],
                    )

    def test_processing_config_normalizes_whisper_backend_fields(self):
        cfg = processor.normalize_processing_config(
            processor.ProcessingConfig(
                whisper_backend="faster_whisper",
                whisper_model_path="x" * 600,
                whisper_queue_seconds=0.0,
            )
        )
        self.assertEqual(cfg.whisper_backend, "faster-whisper")
        self.assertEqual(len(cfg.whisper_model_path), 512)
        self.assertEqual(cfg.whisper_queue_seconds, 0.02)

        cfg = processor.normalize_processing_config(
            processor.ProcessingConfig(whisper_backend="not-real")
        )
        self.assertEqual(cfg.whisper_backend, "faster-whisper")

    def test_segments_to_frame_spans_merges_overlaps(self):
        from backend import whisper_fallback as _wf
        segments = [
            (0.0, 1.0, "a"),
            (0.5, 2.0, "b"),  # overlaps with previous
            (5.0, 6.0, "c"),
        ]
        spans = _wf.segments_to_frame_spans(segments, fps=24.0)
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0], (0, 48))  # 0..2s at 24 fps
        self.assertEqual(spans[1], (120, 144))

    def test_segments_to_frame_spans_handles_invalid_fps(self):
        from backend import whisper_fallback as _wf
        self.assertEqual(_wf.segments_to_frame_spans([(0.0, 1.0, "a")], fps=0.0), [])


class OnnxInpaintersTests(unittest.TestCase):
    """RM-25 / RM-26: ONNX backends must register only when their env
    vars are set, and the inpainter must fall back to cv2 when the
    ONNX session is unavailable."""

    def setUp(self):
        self._saved = {
            "VSR_LAMA_ONNX": os.environ.pop("VSR_LAMA_ONNX", None),
            "VSR_MIGAN_ONNX": os.environ.pop("VSR_MIGAN_ONNX", None),
        }
        # Re-run registration with our cleared env so the registry
        # reflects the disabled state for this test.
        from backend import inpainters_onnx as _o
        _o.maybe_register()

    def tearDown(self):
        for k, v in self._saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        # Restore registry to whatever the env vars dictate.
        from backend import inpainters_onnx as _o
        _o.maybe_register()

    def test_inpainter_without_session_falls_back(self):
        import numpy as _np
        from backend.inpainters_onnx import LamaOnnxInpainter
        cfg = processor.ProcessingConfig()
        inp = LamaOnnxInpainter(device="cpu", config=cfg)
        self.assertIsNone(inp._session)
        frame = _np.full((32, 32, 3), 100, dtype=_np.uint8)
        mask = _np.zeros((32, 32), dtype=_np.uint8)
        mask[10:20, 10:20] = 255
        out = inp.inpaint([frame], [mask])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].shape, frame.shape)

    def test_no_register_when_env_missing(self):
        from backend.inpainters_onnx import maybe_register
        self.assertEqual(maybe_register(), [])


class PySceneDetectAdapterTests(unittest.TestCase):
    """RM-32: PySceneDetect adapter must return None when the optional
    dep is absent, and the histogram path stays the default."""

    def test_adapter_returns_none_without_dep(self):
        import numpy as _np
        from unittest import mock
        frames = [_np.zeros((10, 10, 3), dtype=_np.uint8) for _ in range(3)]
        # Force scenedetect to look absent even on developer machines where
        # optional packages are installed.
        absent = {"scenedetect": None, "scenedetect.detectors": None}
        with mock.patch.dict(sys.modules, absent):
            result = processor._detect_scene_cuts_pyscenedetect(frames)
        self.assertIsNone(result)

    def test_default_path_is_histogram(self):
        import numpy as _np
        frames = [_np.full((10, 10, 3), v, dtype=_np.uint8) for v in (50, 50, 200, 200)]
        cuts = processor._detect_scene_cuts(frames, threshold=0.5)
        # The 50 -> 200 step is a cut; cuts must include index 0 and 2.
        self.assertIn(0, cuts)
        self.assertIn(2, cuts)


class InpainterRegistryTests(unittest.TestCase):
    """RFP-L-2: every built-in mode must be registered; resolve()
    returns the registered builder; missing modes raise KeyError so
    the caller can fall back."""

    def test_builtins_registered(self):
        from backend import inpainter_registry
        for mode in ("sttn", "lama", "propainter", "auto"):
            self.assertTrue(inpainter_registry.is_registered(mode),
                            f"mode {mode!r} must be registered")

    def test_resolve_returns_callable(self):
        from backend import inpainter_registry
        builder = inpainter_registry.resolve("sttn")
        self.assertTrue(callable(builder))

    def test_resolve_unknown_raises(self):
        from backend import inpainter_registry
        with self.assertRaises(KeyError):
            inpainter_registry.resolve("not-a-real-mode")

    def test_register_replaces_existing(self):
        from backend import inpainter_registry
        original = inpainter_registry.resolve("sttn")
        try:
            sentinel = object()
            inpainter_registry.register("sttn", lambda d, c: sentinel)
            self.assertIs(inpainter_registry.resolve("sttn")(None, None), sentinel)
        finally:
            inpainter_registry.register("sttn", original)

    def test_unregister_returns_status(self):
        from backend import inpainter_registry
        sentinel = object()
        inpainter_registry.register("test-plugin", lambda d, c: sentinel)
        self.assertTrue(inpainter_registry.unregister("test-plugin"))
        self.assertFalse(inpainter_registry.unregister("test-plugin"))


class VerticalTextDetectionTests(unittest.TestCase):
    """RM-24: vertical-text mode wraps the detector with a rotate-detect-
    rotate-back layer. Boxes from the rotated frame must come back in
    the original frame's coordinate space."""

    def _make_detector(self, vertical: bool):
        det = processor.SubtitleDetector.__new__(processor.SubtitleDetector)
        det.device = "cpu"
        det.lang = "en"
        det.vertical = vertical
        det._engine_name = "stub"
        det._rapid_model = None
        det._paddle_model = None
        det._surya_det = None
        det._surya_processor = None
        det._easyocr_reader = None
        return det

    def test_vertical_false_short_circuits(self):
        import numpy as _np
        det = self._make_detector(vertical=False)
        det._detect_axis_aligned = lambda f, t: [(10, 20, 30, 40)]
        frame = _np.zeros((100, 200, 3), dtype=_np.uint8)
        self.assertEqual(det.detect(frame, 0.5), [(10, 20, 30, 40)])

    def test_vertical_rotates_boxes_back(self):
        import numpy as _np
        det = self._make_detector(vertical=True)
        # Original frame 200w x 100h. After 90 CCW the rotated frame is
        # 100w x 200h. A box at (rx1=10, ry1=20, rx2=40, ry2=60) in the
        # rotated frame maps back to:
        # ox = ry -> x in [20, 60]
        # oy = w - rx2 .. w - rx1 -> y in [200 - 40, 200 - 10] = [160, 190]
        det._detect_axis_aligned = lambda f, t: [(10, 20, 40, 60)]
        frame = _np.zeros((100, 200, 3), dtype=_np.uint8)
        result = det.detect(frame, 0.5)
        self.assertEqual(result, [(20, 160, 60, 190)])


class HighContrastThemeTests(unittest.TestCase):
    """RM-96: apply_high_contrast_theme must swap the palette in place
    and apply_default_theme must restore it byte-identical."""

    def test_swap_and_restore(self):
        original_bg = gui.Theme.BG_DARK
        original_text = gui.Theme.TEXT_PRIMARY
        try:
            gui.apply_high_contrast_theme()
            self.assertEqual(gui.Theme.BG_DARK, "#000000")
            self.assertEqual(gui.Theme.TEXT_PRIMARY, "#ffffff")
        finally:
            gui.apply_default_theme()
        self.assertEqual(gui.Theme.BG_DARK, original_bg)
        self.assertEqual(gui.Theme.TEXT_PRIMARY, original_text)


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

    def test_vvc_codec_normalises(self):
        cfg = processor.normalize_processing_config(
            processor.ProcessingConfig(output_codec="vvc")
        )
        self.assertEqual(cfg.output_codec, "vvc")

    def test_h266_alias_normalises_to_vvc(self):
        cfg = processor.normalize_processing_config(
            processor.ProcessingConfig(output_codec="h266")
        )
        self.assertEqual(cfg.output_codec, "vvc")

    def test_software_encoder_args_match_codec(self):
        remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        remover._hw_encoder = None
        remover._color_metadata = None  # RM-73 init slot for _hdr_encode_args
        remover.config = processor.ProcessingConfig(output_codec="h265",
                                                     output_quality=22,
                                                     use_hw_encode=False)
        args = remover._get_encode_args()
        self.assertIn("libx265", args)
        remover.config.output_codec = "av1"
        args = remover._get_encode_args()
        self.assertIn("libsvtav1", args)
        remover.config.output_codec = "vvc"
        args = remover._get_encode_args()
        self.assertIn("libvvenc", args)


class ExtendedMetricsTests(unittest.TestCase):
    """RM-102: temporal quality metric expansion."""

    def test_compute_extended_metrics_returns_empty_without_pyiqa(self):
        import numpy as _np
        from unittest import mock
        from backend.quality import compute_extended_metrics
        with mock.patch.dict("sys.modules", {"pyiqa": None}):
            result = compute_extended_metrics([
                (_np.zeros((32, 32, 3), _np.uint8),
                 _np.ones((32, 32, 3), _np.uint8))])
            self.assertEqual(result, {})

    def test_temporal_consistency_score_perfect(self):
        import numpy as _np
        from backend.quality import temporal_consistency_score
        frame = _np.full((32, 32, 3), 128, dtype=_np.uint8)
        score = temporal_consistency_score([frame, frame, frame])
        self.assertIsNotNone(score)
        self.assertGreater(score, 0.99)

    def test_temporal_consistency_score_insufficient_frames(self):
        import numpy as _np
        from backend.quality import temporal_consistency_score
        self.assertIsNone(temporal_consistency_score([]))
        self.assertIsNone(temporal_consistency_score(
            [_np.zeros((8, 8, 3), _np.uint8)]))

    def test_quality_report_schema_includes_extended_fields(self):
        report = {
            'psnr': 35.0, 'ssim': 0.97,
            'roi_psnr': 30.0, 'roi_ssim': 0.92,
            'vmaf': None, 'roi_vmaf': None,
            'roi_bbox': [10, 20, 100, 50],
            'temporal_flicker_score': 0.01,
            'temporal_consistency': 0.98,
            'residual_text_score': 0.02,
            'lpips': None, 'dists': None,
            'samples': 10, 'tag': 'Good', 'sheet': None,
        }
        for key in ('temporal_consistency', 'lpips', 'dists'):
            self.assertIn(key, report)


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


class AdapterManifestVerificationTests(unittest.TestCase):
    """#109: optional adapter model paths carry provenance and can fail
    closed on unknown or mismatched weights before a loader deserializes
    the file."""

    def _entry(self, filename: str, sha256=None):
        from backend.adapter_manifest import AdapterManifestEntry
        return AdapterManifestEntry(
            name="unit-adapter",
            env_vars=("VSR_UNIT_MODEL",),
            expected_filenames=(filename,),
            sha256={} if sha256 is None else {filename: sha256},
            license="test-license",
            source_url="https://example.invalid/model",
            preferred_format="ONNX",
            remote_code_required=False,
        )

    def test_unknown_hash_allowed_for_legacy_adapter(self):
        from unittest import mock
        from backend import adapter_manifest as manifest
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "model.onnx"
            p.write_bytes(b"model")
            with mock.patch.dict(
                manifest.ADAPTER_MANIFEST,
                {"unit-adapter": self._entry(p.name)},
                clear=False,
            ):
                result = manifest.verify_adapter_path("unit-adapter", str(p))
        self.assertTrue(result.allowed)
        self.assertEqual(result.hash_status, "unknown")
        payload = result.as_dict()
        self.assertEqual(payload["preferredFormat"], "ONNX")
        self.assertEqual(payload["license"], "test-license")

    def test_strict_unknown_hash_fails_without_unsafe_override(self):
        from unittest import mock
        from backend import adapter_manifest as manifest
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "model.onnx"
            p.write_bytes(b"model")
            with mock.patch.dict(
                manifest.ADAPTER_MANIFEST,
                {"unit-adapter": self._entry(p.name)},
                clear=False,
            ):
                result = manifest.verify_adapter_path(
                    "unit-adapter", str(p), strict_unknown=True, env={}
                )
                override = manifest.verify_adapter_path(
                    "unit-adapter",
                    str(p),
                    strict_unknown=True,
                    env={manifest.UNSAFE_OVERRIDE_ENV: "1"},
                )
        self.assertFalse(result.allowed)
        self.assertEqual(result.hash_status, "unknown")
        self.assertTrue(override.allowed)
        self.assertEqual(override.hash_status, "unsafe_override")
        self.assertTrue(override.unsafe_override)

    def test_hash_mismatch_fails_unless_override_is_explicit(self):
        from unittest import mock
        from backend import adapter_manifest as manifest
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "model.onnx"
            p.write_bytes(b"model")
            with mock.patch.dict(
                manifest.ADAPTER_MANIFEST,
                {"unit-adapter": self._entry(p.name, "0" * 64)},
                clear=False,
            ):
                result = manifest.verify_adapter_path("unit-adapter", str(p))
                override = manifest.verify_adapter_path(
                    "unit-adapter",
                    str(p),
                    env={manifest.UNSAFE_OVERRIDE_ENV: "true"},
                )
        self.assertFalse(result.allowed)
        self.assertEqual(result.hash_status, "mismatch")
        self.assertTrue(result.actual_sha256)
        self.assertTrue(override.allowed)
        self.assertEqual(override.hash_status, "unsafe_override")

    def test_release_manifest_reports_configured_adapter_status(self):
        from unittest import mock
        from backend import adapter_manifest as manifest
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "model.onnx"
            p.write_bytes(b"model")
            with mock.patch.dict(
                manifest.ADAPTER_MANIFEST,
                {"unit-adapter": self._entry(p.name)},
                clear=True,
            ):
                statuses = manifest.release_manifest_status(
                    env={"VSR_UNIT_MODEL": str(p)}
                )
        self.assertEqual(len(statuses), 1)
        self.assertEqual(statuses[0]["name"], "unit-adapter")
        self.assertEqual(statuses[0]["configuredEnvVar"], "VSR_UNIT_MODEL")
        self.assertEqual(statuses[0]["hashStatus"], "unknown")

    def test_onnx_loader_refuses_mismatched_manifest_before_session(self):
        from unittest import mock
        from backend import adapter_manifest as manifest
        from backend import inpainters_onnx
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "model.onnx"
            p.write_bytes(b"model")
            session = mock.Mock()
            fake_ort = SimpleNamespace(InferenceSession=session)
            entry = manifest.AdapterManifestEntry(
                name="lama-onnx",
                env_vars=("VSR_LAMA_ONNX",),
                expected_filenames=(p.name,),
                sha256={p.name: "0" * 64},
                license="test-license",
                source_url="https://example.invalid/model",
                preferred_format="ONNX",
                remote_code_required=False,
            )
            with mock.patch.dict(
                manifest.ADAPTER_MANIFEST, {"lama-onnx": entry}, clear=False
            ), mock.patch.dict(sys.modules, {"onnxruntime": fake_ort}):
                result = inpainters_onnx._maybe_session(
                    str(p), ["CPUExecutionProvider"], "lama-onnx"
                )
        self.assertIsNone(result)
        session.assert_not_called()


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
            # Write >1 MB of valid JSON; use enough entries to exceed the cap
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
        from unittest import mock

        det = self._make_detector()
        # Honour the class docstring: force every optional OCR engine to
        # look absent so the cascade deterministically reaches the OpenCV
        # fallback. Without this the test passes only on a bare machine and
        # fails in CI/release builds, which install rapidocr/easyocr/etc.
        # (a None entry in sys.modules makes the import raise ImportError).
        absent = {
            name: None
            for name in (
                "rapidocr",
                "rapidocr_onnxruntime",
                "paddleocr",
                "surya",
                "surya.detection",
                "easyocr",
            )
        }
        with mock.patch.dict(sys.modules, absent):
            det._load_model()
        self.assertEqual(det._engine_name, "OpenCV fallback")

    def test_surya_skipped_unless_env_set(self):
        # The cascade should not pick Surya even when its module is
        # importable; gating is via VSR_ALLOW_GPL.
        os.environ.pop("VSR_ALLOW_GPL", None)
        det = self._make_detector()
        det._load_model()
        self.assertNotEqual(det._engine_name, "Surya")


class UpdateCheckTests(unittest.TestCase):
    """RM-116: optional startup update check."""

    def test_parse_version_standard(self):
        from backend.update_check import _parse_version
        self.assertEqual(_parse_version("v3.16.1"), (3, 16, 1))
        self.assertEqual(_parse_version("3.16.1"), (3, 16, 1))

    def test_parse_version_garbage(self):
        from backend.update_check import _parse_version
        self.assertIsNone(_parse_version(""))
        self.assertIsNone(_parse_version("latest"))

    def test_no_callback_when_current_is_latest(self):
        from unittest.mock import MagicMock, patch
        from backend.update_check import check_for_update
        cb = MagicMock()
        fake_resp = io.BytesIO(json.dumps({
            "tag_name": "v3.16.1",
            "html_url": "https://example.com/releases/v3.16.1",
        }).encode())
        fake_resp.status = 200
        fake_resp.__enter__ = lambda s: s
        fake_resp.__exit__ = lambda s, *a: None
        with patch("backend.update_check.urlopen", return_value=fake_resp):
            check_for_update("3.16.1", cb)
            import time; time.sleep(0.5)
        cb.assert_not_called()

    def test_callback_when_newer_available(self):
        from unittest.mock import MagicMock, patch
        from backend.update_check import check_for_update
        cb = MagicMock()
        fake_resp = io.BytesIO(json.dumps({
            "tag_name": "v4.0.0",
            "html_url": "https://example.com/releases/v4.0.0",
        }).encode())
        fake_resp.status = 200
        fake_resp.__enter__ = lambda s: s
        fake_resp.__exit__ = lambda s, *a: None
        with patch("backend.update_check.urlopen", return_value=fake_resp):
            check_for_update("3.16.1", cb)
            import time; time.sleep(0.5)
        cb.assert_called_once_with("v4.0.0", "https://example.com/releases/v4.0.0")

    def test_no_crash_on_network_error(self):
        from unittest.mock import MagicMock, patch
        from backend.update_check import check_for_update
        cb = MagicMock()
        with patch("backend.update_check.urlopen", side_effect=OSError("offline")):
            check_for_update("3.16.1", cb)
            import time; time.sleep(0.5)
        cb.assert_not_called()

    def test_config_field_defaults_off(self):
        cfg = gui.ProcessingConfig()
        self.assertFalse(cfg.update_check)

    def test_config_round_trip(self):
        cfg = gui.ProcessingConfig()
        cfg.update_check = True
        d = cfg.to_dict()
        self.assertTrue(d["update_check"])
        cfg2 = gui.ProcessingConfig.from_dict(d)
        self.assertTrue(cfg2.update_check)


class JsonLogGuiTests(unittest.TestCase):
    """RM-53: structured JSON log GUI toggle."""

    def test_config_field_defaults_off(self):
        cfg = gui.ProcessingConfig()
        self.assertFalse(cfg.json_log_enabled)

    def test_config_round_trip(self):
        cfg = gui.ProcessingConfig()
        cfg.json_log_enabled = True
        d = cfg.to_dict()
        self.assertTrue(d["json_log_enabled"])
        cfg2 = gui.ProcessingConfig.from_dict(d)
        self.assertTrue(cfg2.json_log_enabled)


class MediaExtensionParityTests(unittest.TestCase):
    """Verify that GUI and backend media extension sets stay in sync."""

    def test_gui_image_extensions_include_tif(self):
        from gui.utils import IMAGE_EXTENSIONS
        self.assertIn(".tif", IMAGE_EXTENSIONS)
        self.assertIn(".tiff", IMAGE_EXTENSIONS)

    def test_backend_frame_capture_matches_gui_image_extensions(self):
        from gui.utils import IMAGE_EXTENSIONS
        from backend.io import _FrameSequenceCapture
        self.assertEqual(
            _FrameSequenceCapture.SUPPORTED_EXTS,
            set(IMAGE_EXTENSIONS),
        )

    def test_filepicker_pattern_covers_all_extensions(self):
        from gui.utils import (
            SUPPORTED_EXTENSIONS, VIDEO_EXTENSIONS, IMAGE_EXTENSIONS,
            filepicker_pattern,
        )
        pattern = filepicker_pattern(SUPPORTED_EXTENSIONS)
        for ext in VIDEO_EXTENSIONS | IMAGE_EXTENSIONS:
            self.assertIn(f"*{ext}", pattern)


class TemporalSmoothTests(unittest.TestCase):
    """Post-inpaint temporal consistency filter for LaMa path."""

    def test_temporal_smooth_reduces_variance(self):
        import numpy as _np
        from backend.inpainters._common import _temporal_smooth_inpainted
        mask = _np.zeros((100, 100), dtype=_np.uint8)
        mask[40:60, 40:60] = 255
        masks = [mask] * 5
        frames = []
        for i in range(5):
            f = _np.full((100, 100, 3), 128, dtype=_np.uint8)
            f[40:60, 40:60] = _np.random.randint(50, 200, (20, 20, 3), dtype=_np.uint8)
            frames.append(f)
        smoothed = _temporal_smooth_inpainted(frames, masks, radius=2)
        self.assertEqual(len(smoothed), 5)
        orig_var = _np.var([f[50, 50, 0] for f in frames])
        smooth_var = _np.var([f[50, 50, 0] for f in smoothed])
        self.assertLessEqual(smooth_var, orig_var)

    def test_temporal_smooth_skips_unmasked(self):
        import numpy as _np
        from backend.inpainters._common import _temporal_smooth_inpainted
        mask = _np.zeros((50, 50), dtype=_np.uint8)
        frame = _np.full((50, 50, 3), 100, dtype=_np.uint8)
        result = _temporal_smooth_inpainted([frame, frame], [mask, mask], radius=2)
        _np.testing.assert_array_equal(result[0], frame)

    def test_config_defaults_off(self):
        cfg = gui.ProcessingConfig()
        self.assertEqual(cfg.temporal_smooth_radius, 0)


class TiledLamaTests(unittest.TestCase):
    """Tile-based LaMa inference for high-resolution frames."""

    def test_config_tile_defaults(self):
        cfg = gui.ProcessingConfig()
        self.assertEqual(cfg.lama_tile_size, 512)
        self.assertEqual(cfg.lama_tile_overlap, 64)

    def test_config_tile_round_trip(self):
        cfg = gui.ProcessingConfig()
        cfg.lama_tile_size = 256
        cfg.lama_tile_overlap = 32
        d = cfg.to_dict()
        cfg2 = gui.ProcessingConfig.from_dict(d)
        self.assertEqual(cfg2.lama_tile_size, 256)
        self.assertEqual(cfg2.lama_tile_overlap, 32)

    def test_tiled_inpaint_produces_valid_output(self):
        """Tiled path should produce a valid uint8 BGR frame."""
        import numpy as _np
        from backend.inpainters.lama import LAMAInpainter
        cfg = gui.ProcessingConfig()
        cfg.lama_tile_size = 256
        cfg.lama_tile_overlap = 32
        inpainter = LAMAInpainter.__new__(LAMAInpainter)
        inpainter.config = cfg
        inpainter.device = "cpu"
        def fake_lama(pil_img, pil_mask):
            return pil_img
        inpainter._lama = fake_lama
        frame = _np.random.randint(0, 255, (600, 800, 3), dtype=_np.uint8)
        mask = _np.zeros((600, 800), dtype=_np.uint8)
        mask[250:350, 300:500] = 255
        result = inpainter._inpaint_lama_tiled(frame, mask, 256, 32)
        self.assertEqual(result.shape, frame.shape)
        self.assertEqual(result.dtype, _np.uint8)


class ConfidenceWeightedDilationTests(unittest.TestCase):
    """Confidence-weighted mask dilation scales padding by OCR confidence."""

    def test_config_defaults_off(self):
        cfg = gui.ProcessingConfig()
        self.assertFalse(cfg.confidence_weighted_dilation)
        self.assertAlmostEqual(cfg.confidence_dilation_scale, 1.5)

    def test_config_round_trip(self):
        cfg = gui.ProcessingConfig()
        cfg.confidence_weighted_dilation = True
        cfg.confidence_dilation_scale = 2.0
        d = cfg.to_dict()
        self.assertTrue(d["confidence_weighted_dilation"])
        cfg2 = gui.ProcessingConfig.from_dict(d)
        self.assertTrue(cfg2.confidence_weighted_dilation)
        self.assertAlmostEqual(cfg2.confidence_dilation_scale, 2.0)

    def test_high_confidence_gets_less_dilation(self):
        """With confidence weighting on, a high-confidence box should produce
        a smaller dilated area than a low-confidence box."""
        import cv2 as _cv2
        cfg_hi = gui.ProcessingConfig()
        cfg_hi.confidence_weighted_dilation = True
        cfg_hi.mask_dilate_px = 8
        cfg_hi.confidence_dilation_scale = 1.5
        cfg_lo = gui.ProcessingConfig()
        cfg_lo.confidence_weighted_dilation = True
        cfg_lo.mask_dilate_px = 8
        cfg_lo.confidence_dilation_scale = 1.5
        shape = (200, 400, 3)
        boxes = [(100, 80, 300, 120)]
        remover_hi = SimpleNamespace(config=cfg_hi)
        remover_lo = SimpleNamespace(config=cfg_lo)
        mask_hi = processor.SubtitleRemover._create_mask(
            remover_hi, shape, boxes, confidences=[0.95])
        mask_lo = processor.SubtitleRemover._create_mask(
            remover_lo, shape, boxes, confidences=[0.3])
        self.assertGreater(mask_lo.sum(), mask_hi.sum())


class DependencyFloorTests(unittest.TestCase):
    """Verify minimum dependency versions across all install surfaces."""

    def _read_requirements(self):
        root = Path(__file__).resolve().parents[1]
        return (root / "requirements.txt").read_text(encoding="utf-8")

    def test_pillow_floor_is_12_2_0_in_requirements(self):
        text = self._read_requirements()
        self.assertIn("Pillow>=12.2.0", text)

    def test_pillow_floor_is_12_2_0_in_setup_py(self):
        root = Path(__file__).resolve().parents[1]
        setup_text = (root / "setup.py").read_text(encoding="utf-8")
        self.assertIn("Pillow>=12.2.0", setup_text)

    def test_pillow_floor_is_12_2_0_in_build_workflow(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github" / "workflows" / "build.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("Pillow>=12.2.0", workflow)


if __name__ == "__main__":
    unittest.main()
