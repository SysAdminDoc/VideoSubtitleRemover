import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import VideoSubtitleRemover as gui
from backend import processor


def _has_display() -> bool:
    """Return True if a GUI display is available."""
    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class OptionalImportProbeTests(unittest.TestCase):
    def tearDown(self):
        from backend.import_safety import clear_import_probe_cache
        clear_import_probe_cache()

    def test_probe_respects_none_sentinel(self):
        from backend.import_safety import module_can_import
        with unittest.mock.patch.dict(sys.modules, {"native_broken_probe": None}):
            self.assertFalse(module_can_import("native_broken_probe"))

    def test_probe_respects_injected_module(self):
        from backend.import_safety import module_can_import
        injected = types.ModuleType("native_injected_probe")
        with unittest.mock.patch.dict(sys.modules, {"native_injected_probe": injected}):
            self.assertTrue(module_can_import("native_injected_probe"))

    def test_probe_rejects_windows_native_crash_signature(self):
        from backend import import_safety
        import_safety.clear_import_probe_cache()
        completed = SimpleNamespace(
            returncode=3221225477,
            stdout="",
            stderr="Fatal Python error: access violation",
        )
        with unittest.mock.patch.object(
            import_safety.importlib.util,
            "find_spec",
            return_value=object(),
        ):
            with unittest.mock.patch.object(
                import_safety,
                "run_process",
                return_value=completed,
            ):
                self.assertFalse(
                    import_safety.module_can_import("native_crash_probe")
                )


class BackendHardeningTests(unittest.TestCase):
    def test_normalize_processing_config_clamps_unsafe_values(self):
        cfg = processor.normalize_processing_config(
            processor.ProcessingConfig(
                mode="AUTO",
                device="cuda:-3",
                sttn_max_load_num="-2",
                subtitle_area=[4, 4, 2, 10],
                subtitle_areas=[[2, 3, 10, 12], ["bad"]],
                subtitle_region_spans=[
                    {"rect": [4, 5, 14, 20], "start": 2, "end": 8},
                    {"region": [1, 1, 8, 8], "start": 5, "end": 4},
                    {"rect": [0, 0, 0, 5], "start": 1, "end": 2},
                ],
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
        self.assertEqual(
            cfg.subtitle_region_spans,
            [
                {"rect": (4, 5, 14, 20), "start": 2.0, "end": 8.0},
                {"rect": (1, 1, 8, 8), "start": 5.0, "end": 0.0},
            ],
        )
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
            base_subtitle_region_spans=None,
        )

        self.assertIsNone(band)
        self.assertIsNone(remover.config.subtitle_area)
        self.assertEqual(calls, ["clip-two.mp4"])

    def test_timed_region_boxes_are_selected_by_frame_time(self):
        remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        remover.config = processor.normalize_processing_config(
            processor.ProcessingConfig(
                subtitle_area=(1, 1, 99, 99),
                subtitle_region_spans=[
                    {"rect": [2, 2, 10, 12], "start": 0.5, "end": 1.0},
                    {"rect": [12, 14, 40, 30], "start": 1.0, "end": 0},
                ],
            )
        )

        self.assertIsNone(remover._fixed_region_boxes(0.25))
        self.assertEqual(remover._fixed_region_boxes(0.5), [(2, 2, 10, 12)])
        self.assertEqual(remover._fixed_region_boxes(1.0), [(12, 14, 40, 30)])


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
        import tempfile
        import os
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
        import tempfile
        import os
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

    def test_onnx_session_preloads_cuda_dlls_before_session(self):
        from backend import adapter_manifest as manifest
        from backend import inpainters_onnx
        from backend.onnxruntime_cuda import (
            collect_onnxruntime_cuda_preload_status,
            reset_onnxruntime_cuda_preload_status,
        )
        from unittest import mock

        events = []

        class FakeOrt:
            __version__ = "1.21.0"

            @staticmethod
            def preload_dlls():
                events.append("preload")

            @staticmethod
            def InferenceSession(model_path, providers):
                events.append("session")
                return object()

        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "model.onnx"
            model.write_bytes(b"model")
            reset_onnxruntime_cuda_preload_status()
            try:
                with mock.patch.object(
                    manifest,
                    "verify_adapter_path",
                    return_value=SimpleNamespace(allowed=True),
                ), mock.patch.object(
                    manifest, "log_adapter_verification"
                ), mock.patch.dict(sys.modules, {"onnxruntime": FakeOrt}):
                    session = inpainters_onnx._maybe_session(
                        str(model),
                        ["CUDAExecutionProvider", "CPUExecutionProvider"],
                        "lama-onnx",
                    )
                status = collect_onnxruntime_cuda_preload_status()
            finally:
                reset_onnxruntime_cuda_preload_status()

        self.assertIsNotNone(session)
        self.assertEqual(events, ["preload", "session"])
        self.assertTrue(status["attempted"])
        self.assertTrue(status["succeeded"])
        self.assertEqual(status["callCount"], 1)

    def test_onnx_session_records_missing_cuda_preload_without_blocking(self):
        from backend import adapter_manifest as manifest
        from backend import inpainters_onnx
        from backend.onnxruntime_cuda import (
            collect_onnxruntime_cuda_preload_status,
            reset_onnxruntime_cuda_preload_status,
        )
        from unittest import mock

        class FakeOrt:
            __version__ = "1.20.0"

            @staticmethod
            def InferenceSession(model_path, providers):
                return object()

        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "model.onnx"
            model.write_bytes(b"model")
            reset_onnxruntime_cuda_preload_status()
            try:
                with mock.patch.object(
                    manifest,
                    "verify_adapter_path",
                    return_value=SimpleNamespace(allowed=True),
                ), mock.patch.object(
                    manifest, "log_adapter_verification"
                ), mock.patch.dict(sys.modules, {"onnxruntime": FakeOrt}):
                    session = inpainters_onnx._maybe_session(
                        str(model),
                        ["CUDAExecutionProvider", "CPUExecutionProvider"],
                        "lama-onnx",
                    )
                status = collect_onnxruntime_cuda_preload_status()
            finally:
                reset_onnxruntime_cuda_preload_status()

        self.assertIsNotNone(session)
        self.assertTrue(status["attempted"])
        self.assertFalse(status["available"])
        self.assertFalse(status["succeeded"])
        self.assertIn("unavailable", status["error"])

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
            with mock.patch.dict(
                os.environ,
                {"VSR_RAPIDOCR_ENGINE": "onnxruntime"},
                clear=False,
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
            with mock.patch.dict(
                os.environ,
                {"VSR_RAPIDOCR_ENGINE": "onnxruntime"},
                clear=False,
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
            with mock.patch.dict(
                os.environ,
                {"VSR_RAPIDOCR_ENGINE": "onnxruntime"},
                clear=False,
            ):
                det = processor.SubtitleDetector(device="directml")

        self.assertEqual(det._engine_name, "RapidOCR")
        self.assertEqual(calls, [{}])


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

    def test_opencv_libpng_status_reports_runtime_evidence(self):
        from unittest import mock
        from backend.security_checks import opencv_libpng_status

        fake_cv2 = SimpleNamespace(
            __version__="4.13.0",
            getBuildInformation=lambda: "PNG: build (ver 1.6.53)",
        )
        with mock.patch.dict(sys.modules, {"cv2": fake_cv2}):
            status = opencv_libpng_status()

        self.assertTrue(status["available"])
        self.assertEqual(status["opencv_version"], "4.13.0")
        self.assertEqual(status["libpng_version"], "1.6.53")
        self.assertEqual(status["fixed_version"], "1.6.54")
        self.assertTrue(status["vulnerable"])
        self.assertIn("CVE-2026-22801", status["warning"])
        self.assertIsNone(status["error"])

    def test_libpng_warning_fires_below_fixed_floor(self):
        from unittest import mock
        from backend.security_checks import warn_if_vulnerable_opencv_libpng

        fake_cv2 = SimpleNamespace(
            __version__="4.13.0",
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
            __version__="4.13.0",
            getBuildInformation=lambda: "PNG: build (ver 1.6.54)"
        )
        logger = mock.Mock()
        with mock.patch.dict(sys.modules, {"cv2": fake_cv2}):
            message = warn_if_vulnerable_opencv_libpng(logger)

        self.assertIsNone(message)
        logger.warning.assert_not_called()

    def test_vulnerable_png_decode_uses_pillow_not_opencv_imread(self):
        from unittest import mock
        from PIL import Image
        from backend.safe_image import safe_imread

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (2, 1), (10, 20, 30)).save(path)
            with mock.patch(
                "backend.safe_image.opencv_libpng_status",
                return_value={"vulnerable": True},
            ):
                with mock.patch("cv2.imread",
                                side_effect=AssertionError("unsafe read")):
                    frame = safe_imread(path)

        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (1, 2, 3))
        self.assertEqual(frame[0, 0].tolist(), [30, 20, 10])

    def test_fixed_png_decode_uses_opencv_imread(self):
        from unittest import mock
        import numpy as _np
        from backend.safe_image import safe_imread

        expected = _np.zeros((1, 1, 3), dtype=_np.uint8)
        with mock.patch(
            "backend.safe_image.opencv_libpng_status",
            return_value={"vulnerable": False},
        ):
            with mock.patch("cv2.imread", return_value=expected) as imread:
                frame = safe_imread("sample.png")

        self.assertIs(frame, expected)
        imread.assert_called_once_with("sample.png")

    def test_non_png_decode_uses_opencv_even_when_libpng_vulnerable(self):
        from unittest import mock
        import numpy as _np
        from backend.safe_image import safe_imread

        expected = _np.zeros((1, 1, 3), dtype=_np.uint8)
        with mock.patch(
            "backend.safe_image.opencv_libpng_status",
            return_value={"vulnerable": True},
        ):
            with mock.patch("cv2.imread", return_value=expected) as imread:
                frame = safe_imread("sample.jpg")

        self.assertIs(frame, expected)
        imread.assert_called_once_with("sample.jpg")

    def test_production_image_reads_go_through_safe_helper(self):
        roots = [Path("backend"), Path("gui")]
        offenders = []
        for root in roots:
            for path in root.rglob("*.py"):
                if path.as_posix() == "backend/safe_image.py":
                    continue
                text = path.read_text(encoding="utf-8")
                if "cv2.imread" in text or "_cv2.imread" in text:
                    offenders.append(path.as_posix())
        self.assertEqual(offenders, [])


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

    def test_nsi_enforces_current_security_floor(self):
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        text = (root / "installer" / "vsr.nsi").read_text(encoding="utf-8")
        self.assertIn("NSIS >= 3.12 required", text)
        self.assertIn("0x030C0000", text)
        self.assertNotIn("0x030B0000", text)


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


class VapourSynthBridgeTests(unittest.TestCase):
    """RM-75: try_open_vpy must return None for a non-.vpy path and for
    a missing dep."""

    def setUp(self):
        self._saved_env = {
            name: os.environ.pop(name, None)
            for name in (
                "VSR_VAPOURSYNTH",
                "VSR_VAPOURSYNTH_SCRIPT_DIR",
            )
        }

    def tearDown(self):
        for name, value in self._saved_env.items():
            os.environ.pop(name, None)
            if value is not None:
                os.environ[name] = value

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

    def test_enabled_gate_still_requires_explicit_trusted_root(self):
        from backend.vapoursynth_bridge import try_open_vpy
        with tempfile.TemporaryDirectory() as tmpdir:
            script = Path(tmpdir) / "input.vpy"
            script.write_text("raise RuntimeError('must not run')\n", encoding="utf-8")
            os.environ["VSR_VAPOURSYNTH"] = "1"

            self.assertIsNone(try_open_vpy(str(script)))

    def test_script_outside_trusted_root_is_rejected(self):
        from backend.vapoursynth_bridge import try_open_vpy
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            trusted = base / "trusted"
            trusted.mkdir()
            script = base / "outside.vpy"
            script.write_text("raise RuntimeError('must not run')\n", encoding="utf-8")
            os.environ["VSR_VAPOURSYNTH"] = "1"
            os.environ["VSR_VAPOURSYNTH_SCRIPT_DIR"] = str(trusted)

            self.assertIsNone(try_open_vpy(str(script)))

    def test_script_inside_trusted_root_reaches_capture(self):
        from backend import vapoursynth_bridge as bridge
        with tempfile.TemporaryDirectory() as tmpdir:
            trusted = Path(tmpdir) / "trusted"
            trusted.mkdir()
            script = trusted / "approved.vpy"
            script.write_text("clip = None\n", encoding="utf-8")
            os.environ["VSR_VAPOURSYNTH"] = "1"
            os.environ["VSR_VAPOURSYNTH_SCRIPT_DIR"] = str(trusted)
            fake_capture = unittest.mock.Mock()
            fake_capture.isOpened.return_value = True

            with unittest.mock.patch.object(
                bridge, "_VapourSynthCapture", return_value=fake_capture,
            ) as capture_type:
                self.assertIs(bridge.try_open_vpy(str(script)), fake_capture)

            capture_type.assert_called_once_with(str(script.resolve()))


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

    def _ffmpeg_has_encoder(self, *names: str) -> bool:
        if shutil.which("ffmpeg") is None:
            return False
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        text = result.stdout + result.stderr
        return result.returncode == 0 and any(name in text for name in names)

    def _write_modern_codec_clip(self, dir_path: Path, codec: str) -> Path:
        if codec == "av1":
            if self._ffmpeg_has_encoder("libsvtav1"):
                encode_args = ["-c:v", "libsvtav1", "-preset", "13", "-crf", "40"]
            elif self._ffmpeg_has_encoder("libaom-av1"):
                encode_args = [
                    "-c:v", "libaom-av1", "-cpu-used", "8", "-row-mt", "1",
                    "-crf", "40", "-b:v", "0",
                ]
            else:
                self.skipTest("no AV1 encoder available in ffmpeg")
        elif codec == "vp9":
            if not self._ffmpeg_has_encoder("libvpx-vp9"):
                self.skipTest("no VP9 encoder available in ffmpeg")
            encode_args = [
                "-c:v", "libvpx-vp9", "-deadline", "realtime",
                "-cpu-used", "8", "-crf", "42", "-b:v", "0",
            ]
        else:
            raise AssertionError(f"unexpected codec fixture: {codec}")
        out = dir_path / f"{codec}.mkv"
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=12",
            "-frames:v", "12", "-pix_fmt", "yuv420p",
            *encode_args,
            str(out),
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        return out

    def _stub_remover(self, cfg: processor.ProcessingConfig, inpainter):
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
        remover.inpainter = inpainter
        remover.on_progress = None
        remover.on_preview_frame = None
        remover.live_preview_stride = 6
        remover._hw_encoder = None
        remover._srt_entries = []
        remover.last_quality_report = None
        remover.last_output_path = None
        remover.last_error_message = None
        remover.last_error_reason = None
        remover._quality_mask_bbox = None
        remover._color_metadata = None
        remover._hdr_codec_warning_logged = False
        remover._hdr_software_warning_logged = False
        remover._active_writer = None
        remover._active_subprocess = None
        remover._teardown_requested = False
        remover.last_resume_warning = None
        remover.last_pause_checkpoint = None
        remover.last_pause_checkpoint_path = None
        return remover

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
            actual_output = Path(remover.last_output_path or output)
            self.assertTrue(actual_output.exists(), "output file must be written")
            cap = _cv2.VideoCapture(str(actual_output))
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

    def test_lossless_mask_video_and_manifest_are_promoted_and_reported(self):
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not on PATH")

        class PassthroughInpainter:
            def inpaint(self, frames, masks):
                return frames

        import cv2 as _cv2
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = self._write_clip(tmp, n_frames=8, size=(64, 48))
            output = tmp / "cleaned.mp4"
            mask_output = tmp / "cleaned.mask.mkv"
            mask_manifest = tmp / "cleaned.mask.json"
            cfg = processor.normalize_processing_config(
                processor.ProcessingConfig(
                    mode=processor.InpaintMode.STTN,
                    device="cpu",
                    sttn_skip_detection=True,
                    subtitle_area=(8, 36, 56, 44),
                    preserve_audio=False,
                    adaptive_batch=False,
                    use_hw_encode=False,
                    export_mask_video=True,
                    sttn_max_load_num=4,
                    prefetch_decode=False,
                )
            )
            remover = self._stub_remover(cfg, PassthroughInpainter())
            ok = remover.process_video(str(src), str(output))

            self.assertTrue(ok)
            self.assertTrue(mask_output.is_file())
            self.assertTrue(mask_manifest.is_file())
            self.assertGreater(mask_output.stat().st_size, 0)
            self.assertEqual(remover.last_mask_export["status"], "created")
            self.assertEqual(
                Path(remover.last_mask_export["path"]), mask_output)
            self.assertEqual(
                Path(remover.last_mask_export["manifest"]), mask_manifest)
            self.assertEqual(remover.last_mask_export["format"], "ffv1")
            manifest = json.loads(mask_manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], "vsr.mask_interchange.v1")
            self.assertEqual(manifest["frame_count"], 8)
            cap = _cv2.VideoCapture(str(mask_output))
            try:
                self.assertTrue(cap.isOpened())
                frames_read = 0
                while True:
                    ret, _frame = cap.read()
                    if not ret:
                        break
                    frames_read += 1
            finally:
                cap.release()
            self.assertEqual(frames_read, 8)

    def test_vfr_pipeline_preserves_pts_tail_and_audio_sync(self):
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg/ffprobe not on PATH")

        class PassthroughInpainter:
            def inpaint(self, frames, masks):
                return frames

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            frame_dir = tmp / "source-frames"
            frame_dir.mkdir()
            durations = [0.033, 0.067, 0.041, 0.109, 0.052, 0.074, 0.038, 0.096]
            concat_lines = ["ffconcat version 1.0"]
            for index, duration in enumerate(durations):
                frame = np.full(
                    (48, 64, 3),
                    (index * 27) % 255,
                    dtype=np.uint8,
                )
                frame_path = frame_dir / f"frame_{index:06d}.png"
                self.assertTrue(processor.cv2.imwrite(str(frame_path), frame))
                concat_lines.extend([
                    f"file frame_{index:06d}.png",
                    "option framerate 1000",
                    f"duration {duration:.9f}",
                ])
            concat_lines.append(f"file frame_{len(durations) - 1:06d}.png")
            concat_lines.append("option framerate 1000")
            concat_path = frame_dir / "source.ffconcat"
            concat_path.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")
            silent = tmp / "silent-vfr.mkv"
            subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", str(concat_path),
                "-fps_mode:v", "vfr", "-frames:v", str(len(durations)),
                "-c:v", "ffv1", str(silent),
            ], check=True, timeout=30)
            source = tmp / "source-vfr.mkv"
            subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(silent),
                "-f", "lavfi", "-i",
                f"sine=frequency=880:sample_rate=48000:duration={sum(durations):.9f}",
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                "-c:a", "pcm_s16le", "-shortest", str(source),
            ], check=True, timeout=30)

            source_timing = processor._probe_video_frame_timing(str(source))
            self.assertIsNotNone(source_timing)
            self.assertTrue(source_timing.is_vfr)
            output = tmp / "cleaned-vfr.mkv"
            cfg = processor.normalize_processing_config(
                processor.ProcessingConfig(
                    mode=processor.InpaintMode.STTN,
                    device="cpu",
                    sttn_skip_detection=True,
                    subtitle_area=(8, 36, 56, 44),
                    preserve_audio=True,
                    adaptive_batch=False,
                    use_hw_encode=False,
                    output_codec="h264",
                    sttn_max_load_num=4,
                    prefetch_decode=False,
                )
            )
            remover = self._stub_remover(cfg, PassthroughInpainter())
            ok = remover.process_video(
                str(source),
                str(output),
                checkpoint_dir=tmp / "checkpoints",
                checkpoint_key="vfr-test",
            )

            self.assertTrue(ok, remover.last_error_message)
            output_timing = processor._probe_video_frame_timing(str(output))
            self.assertIsNotNone(output_timing)
            self.assertTrue(output_timing.is_vfr)
            self.assertEqual(output_timing.frame_count, source_timing.frame_count)
            tick = max(
                source_timing.time_base,
                output_timing.time_base,
                0.001,
            )
            for expected, actual in zip(
                source_timing.timestamps, output_timing.timestamps
            ):
                self.assertLessEqual(abs(expected - actual), tick + 1e-6)
            self.assertLessEqual(
                abs(source_timing.duration - output_timing.duration),
                tick + 1e-6,
            )
            self.assertEqual(remover.last_timing_report["mode"], "vfr")
            self.assertEqual(processor._probe_audio_stream_count(str(output)), 1)

            direct_cfg = processor.normalize_processing_config(
                processor.ProcessingConfig(
                    mode=processor.InpaintMode.STTN,
                    device="cpu",
                    sttn_skip_detection=True,
                    subtitle_area=(8, 36, 56, 44),
                    preserve_audio=False,
                    adaptive_batch=False,
                    use_hw_encode=False,
                    output_codec="h264",
                    sttn_max_load_num=4,
                    prefetch_decode=False,
                )
            )
            direct_output = tmp / "cleaned-vfr-direct.mkv"
            direct_remover = self._stub_remover(
                direct_cfg, PassthroughInpainter())
            self.assertTrue(direct_remover.process_video(
                str(source), str(direct_output)))
            direct_timing = processor._probe_video_frame_timing(
                str(direct_output))
            self.assertEqual(direct_timing.frame_count, source_timing.frame_count)
            for expected, actual in zip(
                source_timing.timestamps, direct_timing.timestamps
            ):
                self.assertLessEqual(abs(expected - actual), tick + 1e-6)

            def packet_end(selector: str) -> float:
                result = subprocess.run([
                    "ffprobe", "-v", "error", "-select_streams", selector,
                    "-show_packets", "-show_entries",
                    "packet=pts_time,duration_time", "-of", "csv=p=0",
                    str(output),
                ], capture_output=True, text=True, check=True, timeout=30)
                ends = []
                for line in result.stdout.splitlines():
                    fields = line.split(",")
                    try:
                        ends.append(float(fields[0]) + float(fields[1]))
                    except (IndexError, TypeError, ValueError):
                        continue
                return max(ends)

            self.assertLessEqual(abs(packet_end("v:0") - packet_end("a:0")), 0.05)

    def test_vfr_pts_drive_ranges_srt_and_checkpoint_manifest(self):
        from backend import resume_checkpoint as rc
        from backend.io import _FfmpegBgr48Capture

        timing = processor.VideoFrameTiming(
            timestamps=[0.0, 0.04, 0.12, 0.16, 0.28],
            durations=[0.04, 0.08, 0.04, 0.12, 0.06],
            time_base=0.001,
            average_fps=14.705882,
            source_start=0.0,
            is_vfr=True,
        )
        self.assertEqual(timing.frame_range(0.05, 0.20, 5), (2, 4))
        remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        remover._srt_entries = [(0, "first"), (1, "second")]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            srt = tmp / "timed.srt"
            remover._write_srt(
                str(srt), 25.0, 1, frame_timing=timing)
            payload = srt.read_text(encoding="utf-8")
            self.assertIn("00:00:00,040 --> 00:00:00,120", payload)
            self.assertIn("00:00:00,120 --> 00:00:00,160", payload)

            source = tmp / "source.mp4"
            source.write_bytes(b"source")
            manifest = tmp / "frames" / "frame_timing.json"
            checkpoint = rc.write_pause_checkpoint(
                tmp,
                "timed",
                input_path=str(source),
                output_path=str(tmp / "out.mp4"),
                config_hash="abc",
                frame_dir=tmp / "frames",
                next_frame=0,
                total_frames=5,
                width=64,
                height=48,
                fps=25.0,
                status="running",
                timing_manifest_path=manifest,
            )
            self.assertEqual(checkpoint["timing_manifest"], str(manifest))

        capture = _FfmpegBgr48Capture(
            "hdr-vfr.mkv",
            width=64,
            height=48,
            fps=25.0,
            frame_count=5,
        )
        capture.set_frame_timing(timing.timestamps)
        capture.set(processor.cv2.CAP_PROP_POS_FRAMES, 2)
        fake_process = SimpleNamespace(poll=lambda: None)
        with unittest.mock.patch(
            "backend.io.popen_process", return_value=fake_process
        ) as popen:
            self.assertTrue(capture._ensure_proc())
        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("-ss") + 1], "0.120000000")

    def test_hdr_output_contract_survives_audio_and_post_restore(self):
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg/ffprobe not on PATH")
        encoders = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout
        if "libx265" not in encoders:
            self.skipTest("libx265 not available")

        class PassthroughInpainter:
            def inpaint(self, frames, masks):
                return frames

        mastering = (
            "G(13250,34500)B(7500,3000)R(34000,16000)"
            "WP(15635,16450)L(10000000,1)"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "hdr-source.mkv"
            subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i",
                "color=c=red:size=64x48:rate=4:duration=1",
                "-f", "lavfi", "-i",
                "sine=frequency=440:sample_rate=48000:duration=1",
                "-vf", "format=yuv420p10le",
                "-c:v", "libx265", "-preset", "ultrafast",
                "-x265-params",
                "hdr-opt=1:repeat-headers=1:colorprim=9:"
                "transfer=16:colormatrix=9:"
                f"master-display={mastering}:max-cll=1000,400",
                "-pix_fmt", "yuv420p10le",
                "-color_primaries", "bt2020",
                "-color_trc", "smpte2084",
                "-colorspace", "bt2020nc",
                "-color_range", "tv",
                "-c:a", "pcm_s16le",
                str(source),
            ], check=True, timeout=60)
            output = tmp / "hdr-cleaned.mkv"
            cfg = processor.normalize_processing_config(
                processor.ProcessingConfig(
                    mode=processor.InpaintMode.STTN,
                    device="cpu",
                    sttn_skip_detection=True,
                    subtitle_area=(8, 36, 56, 44),
                    preserve_audio=True,
                    preserve_color_metadata=True,
                    output_codec="h265",
                    output_quality=28,
                    film_grain_strength=0.02,
                    adaptive_batch=False,
                    use_hw_encode=False,
                    sttn_max_load_num=4,
                    prefetch_decode=False,
                )
            )
            remover = self._stub_remover(cfg, PassthroughInpainter())

            self.assertTrue(
                remover.process_video(str(source), str(output)),
                remover.last_error_message,
            )
            contract = remover.last_output_contract
            self.assertEqual(contract["status"], "preserved")
            self.assertEqual(contract["container"], "mkv")
            self.assertEqual(contract["codec"], "h265")
            self.assertEqual(contract["pixel_format"], "10-bit")
            self.assertTrue(contract["preserve_audio"])
            self.assertTrue(contract["color_preserved"])
            self.assertEqual(contract["color"]["mastering_display"], mastering)
            self.assertEqual(contract["color"]["max_cll"], 1000)
            self.assertEqual(processor._probe_audio_stream_count(str(output)), 1)
            sidecar = json.loads(
                Path(str(output) + ".vsr.json").read_text(encoding="utf-8")
            )
            self.assertEqual(sidecar["outputContract"]["status"], "preserved")
            self.assertTrue(sidecar["outputContract"]["color_preserved"])

    def test_av1_and_vp9_decode_serial_and_prefetch_paths(self):
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not on PATH")

        class PassthroughInpainter:
            def inpaint(self, frames, masks):
                return frames

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            for codec in ("av1", "vp9"):
                src = self._write_modern_codec_clip(tmp, codec)
                for label, prefetch, accel in (
                    ("serial", False, "off"),
                    ("prefetch_auto", True, "auto"),
                ):
                    output = tmp / f"{codec}_{label}_frames"
                    cfg = processor.normalize_processing_config(
                        processor.ProcessingConfig(
                            mode=processor.InpaintMode.STTN,
                            device="cpu",
                            sttn_skip_detection=True,
                            subtitle_area=(8, 36, 56, 44),
                            preserve_audio=False,
                            adaptive_batch=False,
                            use_hw_encode=False,
                            output_frames=True,
                            input_fps=12.0,
                            sttn_max_load_num=4,
                            prefetch_decode=prefetch,
                            decode_hw_accel=accel,
                        )
                    )
                    remover = self._stub_remover(cfg, PassthroughInpainter())
                    ok = remover.process_video(str(src), str(output))
                    self.assertTrue(ok, f"{codec} {label} decode should succeed")
                    actual = Path(remover.last_output_path or output)
                    frames = sorted(actual.glob("frame_*.png"))
                    self.assertGreaterEqual(
                        len(frames),
                        10,
                        f"{codec} {label} should decode the generated clip",
                    )

    def test_pause_checkpoint_resumes_current_video(self):
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not on PATH")

        class PassthroughInpainter:
            def inpaint(self, frames, masks):
                return frames

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = self._write_clip(tmp, n_frames=10, size=(64, 48))
            output = tmp / "resumed.mp4"
            ckpt = tmp / "checkpoints"
            cfg = processor.normalize_processing_config(
                processor.ProcessingConfig(
                    mode=processor.InpaintMode.STTN,
                    device="cpu",
                    sttn_skip_detection=True,
                    subtitle_area=(8, 36, 56, 44),
                    preserve_audio=False,
                    adaptive_batch=False,
                    use_hw_encode=False,
                    sttn_max_load_num=4,
                    prefetch_decode=False,
                )
            )
            first = self._stub_remover(cfg, PassthroughInpainter())
            with self.assertRaises(processor.ProcessingPaused):
                first.process_video(
                    str(src),
                    str(output),
                    checkpoint_dir=ckpt,
                    checkpoint_key="demo",
                    pause_check=lambda: True,
                )
            pause_path = ckpt / "demo.pause.json"
            frames_dir = ckpt / "demo.frames"
            pause_payload = json.loads(pause_path.read_text(encoding="utf-8"))
            self.assertEqual(pause_payload["status"], "paused")
            self.assertGreater(pause_payload["next_frame"], 0)
            self.assertTrue((frames_dir / "frame_000000.png").is_file())

            second = self._stub_remover(cfg, PassthroughInpainter())
            ok = second.process_video(
                str(src),
                str(output),
                checkpoint_dir=ckpt,
                checkpoint_key="demo",
                resume_checkpoint=True,
                pause_check=lambda: False,
            )
            self.assertTrue(ok)
            self.assertTrue(output.exists())
            self.assertFalse(pause_path.exists())
            self.assertFalse(frames_dir.exists())

    def test_stale_pause_checkpoint_returns_warning_and_zero_resume(self):
        from backend.resume_checkpoint import (
            config_fingerprint,
            load_pause_checkpoint,
            write_pause_checkpoint,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "clip.mp4"
            out = tmp / "out.mp4"
            src.write_bytes(b"video")
            ckpt = tmp / "checkpoints"
            frame_dir = ckpt / "demo.frames"
            frame_dir.mkdir(parents=True)
            cfg = processor.ProcessingConfig(device="cpu")
            good_hash = config_fingerprint(cfg)
            write_pause_checkpoint(
                ckpt,
                "demo",
                input_path=str(src),
                output_path=str(out),
                config_hash="old-settings",
                frame_dir=frame_dir,
                next_frame=1,
                total_frames=10,
                width=64,
                height=48,
                fps=24.0,
                status="paused",
            )
            state = load_pause_checkpoint(
                ckpt,
                "demo",
                input_path=str(src),
                output_path=str(out),
                config_hash=good_hash,
                total_frames=10,
                width=64,
                height=48,
                fps=24.0,
            )
            self.assertEqual(state.next_frame, 0)
            self.assertIn("settings changed", state.warning)

    def test_pipeline_timed_regions_do_not_reuse_masks_outside_span(self):
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not on PATH")

        class RecordingInpainter:
            def __init__(self):
                self.mask_sums = []

            def inpaint(self, frames, masks):
                self.mask_sums.extend(int(mask.sum()) for mask in masks)
                return frames

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = self._write_clip(tmp, n_frames=24, size=(64, 48))
            output = tmp / "timed-cleaned.mp4"
            cfg = processor.normalize_processing_config(
                processor.ProcessingConfig(
                    mode=processor.InpaintMode.STTN,
                    device="cpu",
                    sttn_skip_detection=True,
                    sttn_max_load_num=6,
                    subtitle_region_spans=[
                        {"rect": (8, 36, 56, 44), "start": 0.0, "end": 0.5},
                    ],
                    preserve_audio=False,
                    adaptive_batch=False,
                    use_hw_encode=False,
                )
            )
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
            recorder = RecordingInpainter()
            remover.inpainter = recorder
            remover.on_progress = None
            remover.on_preview_frame = None
            remover.live_preview_stride = 6
            remover._hw_encoder = None
            remover._srt_entries = []
            remover.last_quality_report = None
            remover._quality_mask_bbox = None

            ok = remover.process_video(str(src), str(output))

            self.assertTrue(ok, "process_video must succeed end-to-end")
            self.assertEqual(len(recorder.mask_sums), 24)
            self.assertGreater(recorder.mask_sums[0], 0)
            self.assertGreater(recorder.mask_sums[11], 0)
            self.assertEqual(recorder.mask_sums[12], 0)
            self.assertEqual(recorder.mask_sums[-1], 0)



if __name__ == "__main__":
    unittest.main()
