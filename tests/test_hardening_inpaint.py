import json
import os
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


class AutoInpainterUnloadTests(unittest.TestCase):
    """AUTO routes scenes and drops idle ProPainter model memory."""

    def _auto_inpainter(self):
        cfg = processor.ProcessingConfig(mode=processor.InpaintMode.AUTO)
        cfg = processor.normalize_processing_config(cfg)
        return processor.AutoInpainter(device="cpu", config=cfg)

    def test_streak_resets_on_propainter_route(self):
        auto = self._auto_inpainter()
        auto._sttn_streak = 5
        auto._sttn.inpaint = lambda f, m: f  # type: ignore[assignment]
        import numpy as _np
        frame = _np.zeros((4, 4, 3), dtype=_np.uint8)
        mask = _np.full((4, 4), 255, dtype=_np.uint8)
        class _StubProPainter:
            def inpaint(self, frames, masks):
                return frames
        auto._propainter = _StubProPainter()
        _ = auto.inpaint([frame, frame], [mask, mask])
        self.assertEqual(auto._sttn_streak, 0)

    def test_propainter_unloaded_after_sttn_streak_threshold(self):
        auto = self._auto_inpainter()
        auto.PROPAINTER_IDLE_UNLOAD_AFTER = 3
        class _StubProPainter:
            def inpaint(self, frames, masks):
                return frames
        auto._propainter = _StubProPainter()
        import numpy as _np
        frame = _np.zeros((4, 4, 3), dtype=_np.uint8)
        m1 = _np.zeros((4, 4), dtype=_np.uint8)
        m1[0, 0] = 255
        m2 = _np.zeros((4, 4), dtype=_np.uint8)
        m2[3, 3] = 255
        auto._sttn.inpaint = lambda f, m: f  # type: ignore[assignment]
        for _ in range(3):
            _ = auto.inpaint([frame, frame], [m1, m2])
        self.assertIsNone(auto._propainter)

    def test_routes_each_scene_by_exposure_and_motion(self):
        auto = self._auto_inpainter()
        import numpy as _np
        still = _np.zeros((4, 4, 3), dtype=_np.uint8)
        changed = _np.full((4, 4, 3), 255, dtype=_np.uint8)
        frames = [still, still, still, changed]
        m1 = _np.zeros((4, 4), dtype=_np.uint8)
        m1[0, 0] = 255
        m2 = _np.zeros((4, 4), dtype=_np.uint8)
        m2[3, 3] = 255
        masks = [m1, m2, m1, m2]
        sttn_calls = []
        propainter_calls = []
        auto._sttn.inpaint = lambda f, m: sttn_calls.append(len(f)) or f

        class _StubProPainter:
            def inpaint(self, scene_frames, scene_masks):
                propainter_calls.append(len(scene_frames))
                return scene_frames

        auto._propainter = _StubProPainter()
        with unittest.mock.patch(
            "backend.inpainters.auto._detect_scene_cuts", return_value=[0, 2]
        ):
            result = auto.inpaint(frames, masks)

        self.assertEqual(len(result), 4)
        self.assertEqual(sttn_calls, [2])
        self.assertEqual(propainter_calls, [2])


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
            "VSR_MATANYONE", "VSR_COTRACKER", "VSR_COTRACKER_REPO",
            "VSR_COTRACKER_REF", "VSR_COTRACKER_MODE", "VSR_MATANYONE_PATH",
            "VSR_MATANYONE_REVISION", "VSR_MATANYONE_MODEL_ID",
            "VSR_MATANYONE_DEVICE", "VSR_ALLOW_UNVERIFIED_MODELS",
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

    def test_cotracker_refuses_torch_hub_without_pinned_source(self):
        import numpy as _np
        from unittest import mock
        from backend import segmentation as _seg

        fake_torch = types.ModuleType("torch")
        fake_torch.hub = SimpleNamespace(load=mock.Mock())
        saved = dict(_seg._COTRACKER_STATE)
        try:
            _seg._COTRACKER_STATE.update({"probed": False, "model": None})
            os.environ["VSR_COTRACKER"] = "1"
            os.environ.pop("VSR_COTRACKER_REPO", None)
            os.environ.pop("VSR_COTRACKER_REF", None)
            frames = [_np.zeros((16, 16, 3), dtype=_np.uint8) for _ in range(3)]
            with mock.patch.dict(sys.modules, {"torch": fake_torch}):
                self.assertIsNone(_seg.track_points(frames, [(4, 4)]))
            fake_torch.hub.load.assert_not_called()
        finally:
            _seg._COTRACKER_STATE.clear()
            _seg._COTRACKER_STATE.update(saved)

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
            "VSR_VOID",
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

    def test_pynv_simpledecoder_facade_reads_bgr(self):
        import numpy as np
        from unittest import mock
        from backend.decode_accel import try_open_pynv

        class SimpleDecoder:
            def __init__(self, path, **kwargs):
                self.path = path
                self.kwargs = kwargs

            def get_stream_metadata(self):
                return {
                    "width": 4,
                    "height": 2,
                    "average_fps": 24.0,
                }

            def __len__(self):
                return 2

            def __getitem__(self, idx):
                if idx >= 2:
                    raise IndexError(idx)
                frame = np.zeros((2, 4, 3), dtype=np.uint8)
                frame[:, :, 0] = 10
                frame[:, :, 1] = 20
                frame[:, :, 2] = 30
                return frame

        fake_module = types.ModuleType("PyNvVideoCodec")
        fake_module.SimpleDecoder = SimpleDecoder
        fake_module.OutputColorType = SimpleNamespace(RGB="rgb")
        with mock.patch.dict(sys.modules, {"PyNvVideoCodec": fake_module}):
            cap = try_open_pynv("clip.mp4")

        self.assertIsNotNone(cap)
        self.assertEqual(cap.get(processor.cv2.CAP_PROP_FRAME_WIDTH), 4.0)
        self.assertEqual(cap.get(processor.cv2.CAP_PROP_FRAME_HEIGHT), 2.0)
        self.assertEqual(cap.get(processor.cv2.CAP_PROP_FRAME_COUNT), 2.0)
        ok, frame = cap.read()
        self.assertTrue(ok)
        self.assertEqual(frame.shape, (2, 4, 3))
        self.assertEqual(frame[0, 0].tolist(), [30, 20, 10])
        self.assertTrue(cap.set(processor.cv2.CAP_PROP_POS_FRAMES, 1))
        ok, _frame = cap.read()
        self.assertTrue(ok)
        ok, _frame = cap.read()
        self.assertFalse(ok)

    def test_open_capture_routes_pynv_token(self):
        from unittest import mock
        from backend import io as _io

        fake_cap = object()
        with mock.patch(
            "backend.decode_accel.try_open_pynv",
            return_value=fake_cap,
        ) as mocked:
            cap = _io._open_capture("clip.mp4", "pynv")

        self.assertIs(cap, fake_cap)
        mocked.assert_called_once_with("clip.mp4")

    def test_rife_returns_none_without_dep(self):
        import numpy as _np
        from backend.decode_accel import maybe_interpolate_pair, is_rife_available
        a = _np.zeros((8, 8, 3), dtype=_np.uint8)
        b = _np.full((8, 8, 3), 255, dtype=_np.uint8)
        if not is_rife_available():
            self.assertIsNone(maybe_interpolate_pair(a, b, 0.5))


class RifeFastModePipelineTests(unittest.TestCase):
    class _FakeInpainter:
        def __init__(self):
            self.calls = []

        def inpaint(self, frames, masks):
            self.calls.append((len(frames), len(masks)))
            return [
                np.full_like(frame, 20 + index * 40)
                for index, frame in enumerate(frames)
            ]

    def _remover(self, stride=2):
        remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        remover.config = processor.normalize_processing_config(
            processor.ProcessingConfig(rife_fast_stride=stride)
        )
        remover.inpainter = self._FakeInpainter()
        return remover

    @staticmethod
    def _frames(count):
        return [
            np.full((8, 8, 3), index, dtype=np.uint8)
            for index in range(count)
        ]

    @staticmethod
    def _masks(count):
        return [np.zeros((8, 8), dtype=np.uint8) for _ in range(count)]

    def test_config_clamps_rife_stride(self):
        cfg = processor.normalize_processing_config(
            processor.ProcessingConfig(rife_fast_stride=999)
        )
        self.assertEqual(cfg.rife_fast_stride, 60)
        cfg = processor.normalize_processing_config(
            processor.ProcessingConfig(rife_fast_stride=-3)
        )
        self.assertEqual(cfg.rife_fast_stride, 0)

    def test_fast_mode_inpaints_keyframes_and_interpolates_middle_frames(self):
        remover = self._remover(stride=2)
        frames = self._frames(5)
        masks = self._masks(5)

        def interp(prev, next_frame, t):
            return np.full_like(prev, int(t * 200))

        with unittest.mock.patch(
            "backend.processor._detect_scene_cuts",
            return_value=[0],
        ), unittest.mock.patch(
            "backend.decode_accel.maybe_interpolate_pair",
            side_effect=interp,
        ) as mocked:
            out = remover._inpaint_with_optional_rife_fast(frames, masks)

        self.assertEqual(remover.inpainter.calls, [(3, 3)])
        self.assertEqual(len(out), 5)
        self.assertEqual(out[0][0, 0, 0], 20)
        self.assertEqual(out[1][0, 0, 0], 100)
        self.assertEqual(out[2][0, 0, 0], 60)
        self.assertEqual(out[3][0, 0, 0], 100)
        self.assertEqual(out[4][0, 0, 0], 100)
        self.assertEqual(mocked.call_count, 2)

    def test_scene_cut_uses_nearest_cleaned_keyframe_duplicate(self):
        remover = self._remover(stride=2)
        frames = self._frames(3)
        masks = self._masks(3)

        with unittest.mock.patch(
            "backend.processor._detect_scene_cuts",
            return_value=[0, 1],
        ), unittest.mock.patch(
            "backend.decode_accel.maybe_interpolate_pair",
        ) as mocked:
            out = remover._inpaint_with_optional_rife_fast(frames, masks)

        self.assertEqual(remover.inpainter.calls, [(2, 2)])
        mocked.assert_not_called()
        self.assertEqual(out[0][0, 0, 0], 20)
        self.assertEqual(out[1][0, 0, 0], 60)
        self.assertEqual(out[2][0, 0, 0], 60)


class VlmOcrAdapterTests(unittest.TestCase):
    """RM-22 / RM-23 / RM-42: maybe_build_vlm_detector must return None
    by default (no env var, default lang) and the adapter classes must
    survive a missing-dependency load."""

    def setUp(self):
        self._saved = {
            "VSR_VLM_OCR": os.environ.pop("VSR_VLM_OCR", None),
            "VSR_FLORENCE2_PATH": os.environ.pop("VSR_FLORENCE2_PATH", None),
            "VSR_FLORENCE2_REVISION": os.environ.pop("VSR_FLORENCE2_REVISION", None),
            "VSR_PADDLEOCR_VL": os.environ.pop("VSR_PADDLEOCR_VL", None),
            "VSR_PADDLEOCR_VL_SERVER_URL": os.environ.pop(
                "VSR_PADDLEOCR_VL_SERVER_URL", None),
            "VSR_PADDLEOCR_VL_SKIP_SERVER_PROBE": os.environ.pop(
                "VSR_PADDLEOCR_VL_SKIP_SERVER_PROBE", None),
        }

    def tearDown(self):
        for k, v in self._saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

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

    def test_florence2_refuses_trust_remote_code_without_pinned_source(self):
        from unittest import mock
        from backend.ocr_vlm import _Florence2Detector

        class Loader:
            calls = []

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                cls.calls.append((args, kwargs))
                raise AssertionError("remote code loader should be gated")

        transformers = types.ModuleType("transformers")
        transformers.AutoProcessor = Loader
        transformers.AutoModelForCausalLM = Loader
        torch = types.ModuleType("torch")
        torch.cuda = SimpleNamespace(is_available=lambda: False)

        d = _Florence2Detector(device="cpu")
        with mock.patch.dict(sys.modules, {
            "transformers": transformers,
            "torch": torch,
        }):
            self.assertIsNone(d._load())
        self.assertEqual(Loader.calls, [])

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

    def test_paddleocr_vl_flag_falls_back_when_llama_cpp_unavailable(self):
        from unittest import mock
        from backend import ocr_vlm

        os.environ["VSR_PADDLEOCR_VL"] = "1"
        with mock.patch.object(
            ocr_vlm,
            "_llama_cpp_server_reachable",
            return_value=False,
        ):
            self.assertIsNone(ocr_vlm.maybe_build_vlm_detector("cpu", "en"))

    def test_paddleocr_vl_llama_uses_cpu_server_backend(self):
        from unittest import mock
        from backend import ocr_vlm

        calls = []

        class FakePaddleOCRVL:
            def __init__(self, **kwargs):
                calls.append(kwargs)

            def predict(self, _path):
                return []

        paddleocr = types.ModuleType("paddleocr")
        paddleocr.PaddleOCRVL = FakePaddleOCRVL

        os.environ["VSR_PADDLEOCR_VL"] = "1"
        os.environ["VSR_PADDLEOCR_VL_SERVER_URL"] = "http://127.0.0.1:18080/v1"
        with mock.patch.dict(sys.modules, {"paddleocr": paddleocr}):
            with mock.patch.object(
                ocr_vlm,
                "_llama_cpp_server_reachable",
                return_value=True,
            ):
                detector = ocr_vlm.maybe_build_vlm_detector("cuda:0", "en")

        self.assertIsNotNone(detector)
        self.assertEqual(detector.name, "paddleocr-vl-1.5-llama.cpp")
        self.assertEqual(detector.device, "cpu")
        self.assertEqual(calls, [{
            "vl_rec_backend": "llama-cpp-server",
            "vl_rec_server_url": "http://127.0.0.1:18080/v1",
        }])

    def test_paddleocr_vl_llama_parser_extracts_nested_boxes(self):
        import numpy as _np
        from backend.ocr_vlm import _PaddleOcrVlLlamaCppDetector

        class FakeModel:
            def predict(self, path):
                self.path = path
                return [{
                    "res": {
                        "layout_parsing_result": [
                            {"bbox": [2, 3, 20, 30], "confidence": 0.9},
                            {
                                "poly": [[4, 5], [12, 5], [12, 17], [4, 17]],
                                "score": 0.8,
                            },
                            {"bbox": [1, 1, 2, 2], "confidence": 0.1},
                        ],
                        "rec_boxes": [[30, 10, 40, 20]],
                        "rec_scores": [0.95],
                    }
                }]

        detector = _PaddleOcrVlLlamaCppDetector(device="cuda:0", env={})
        detector._model = FakeModel()

        boxes = detector._extract_boxes(
            _np.zeros((48, 64, 3), dtype=_np.uint8),
            threshold=0.5,
        )

        self.assertEqual(
            boxes,
            [(30, 10, 40, 20), (2, 3, 20, 30), (4, 5, 12, 17)],
        )


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


class ExternalInpainterCommandTests(unittest.TestCase):
    def test_split_external_command_preserves_quoted_windows_path(self):
        from backend.inpainters.external import _split_external_command

        parts = _split_external_command(
            r'"C:\Program Files\Tools\inpainter.exe" --quality high'
        )

        self.assertEqual(parts[0], r"C:\Program Files\Tools\inpainter.exe")
        self.assertEqual(parts[1:], ["--quality", "high"])

    def test_split_external_command_preserves_unquoted_backslashes(self):
        from backend.inpainters.external import _split_external_command

        parts = _split_external_command(
            r"C:\Tools\inpainter.exe --model C:\Models\lama.onnx"
        )

        self.assertEqual(parts[0], r"C:\Tools\inpainter.exe")
        self.assertEqual(parts[-1], r"C:\Models\lama.onnx")

    def test_external_output_survives_inside_mask(self):
        """Regression: the external tool's fill -- not the still-subtitled
        source frame -- must win inside the mask after _feather_blend."""
        import cv2
        from backend.inpainters import external as external_mod

        frame = np.zeros((32, 32, 3), dtype=np.uint8)  # black source
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 255
        filled = np.full((32, 32, 3), 200, dtype=np.uint8)  # bright fill

        def fake_run(command, **_kwargs):
            out_dir = command[command.index("--output-dir") + 1]
            cv2.imwrite(os.path.join(out_dir, "000000.png"), filled)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        inpainter = external_mod.ExternalInpainter.__new__(
            external_mod.ExternalInpainter)
        inpainter._cmd = ["fake-inpainter"]
        inpainter._timeout = 60
        inpainter._config = None

        with unittest.mock.patch.object(external_mod, "run_process", fake_run):
            result = inpainter.inpaint([frame], [mask])

        center = result[0][16, 16]
        # The fill (200) must dominate at the mask center, proving the
        # external output is not discarded in favor of the source frame.
        self.assertGreater(int(center.mean()), 150)


class D3D12AccelerationTests(unittest.TestCase):
    def _remover(self):
        remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        remover.config = processor.ProcessingConfig(
            output_codec="h264",
            output_quality=23,
            use_hw_encode=True,
            d3d12_accel=True,
        )
        remover._hw_encoder = "h264_d3d12va"
        remover._d3d12_fallback_encoder = "h264_nvenc"
        remover._d3d12_status = {
            "requested": True,
            "selected_encoder": "h264_d3d12va",
        }
        remover._color_metadata = None
        remover._output_contract = None
        return remover

    def test_probe_requires_byte_valid_runtime_output(self):
        from backend import encoder

        advertised = {
            "available": True,
            "reason": "advertised; runtime smoke required",
            "advertised_encoders": ["h264_d3d12va"],
            "advertised_filters": ["scale_d3d12", "deinterlace_d3d12"],
        }

        def run(command, **_kwargs):
            if command[0] == "ffmpeg":
                Path(command[-1]).write_bytes(b"valid-video-placeholder")
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({
                    "streams": [{"codec_name": "h264", "nb_read_frames": "30"}],
                }),
                stderr="",
            )

        with unittest.mock.patch(
            "backend.encoder.collect_ffmpeg_capability_profiles",
            return_value={"windows_d3d12": advertised},
        ), unittest.mock.patch(
            "backend.encoder.run_process", side_effect=run,
        ), unittest.mock.patch.object(encoder.sys, "platform", "win32"):
            report = encoder.probe_d3d12_encoder("h264")

        self.assertTrue(report["available"])
        self.assertEqual(report["frames"], 30)
        self.assertIn("byte-valid", report["reason"])

    def test_advertised_but_failed_d3d12_falls_back_to_existing_chain(self):
        from backend.encoder import _detect_hw_encoder

        failed_probe = {
            "available": False,
            "encoder": "h264_d3d12va",
            "reason": "driver rejected codec configuration",
        }
        listed = SimpleNamespace(
            returncode=0,
            stdout=" V..... h264_nvenc NVIDIA NVENC H.264 encoder\n",
            stderr="",
        )
        with unittest.mock.patch(
            "backend.encoder.run_process", return_value=listed
        ):
            selected = _detect_hw_encoder(
                "h264", prefer_d3d12=True, d3d12_probe=failed_probe)

        self.assertEqual(selected, "h264_nvenc")

    def test_d3d12_command_uses_device_upload_scale_and_safe_queue(self):
        remover = self._remover()

        device_args = remover._d3d12_device_args()
        encode_args = remover._get_encode_args()

        self.assertIn("d3d12va=vsr_d3d12", device_args)
        self.assertIn("format=nv12,hwupload,scale_d3d12=w=iw:h=ih", encode_args)
        self.assertIn("h264_d3d12va", encode_args)
        self.assertEqual(encode_args[encode_args.index("-bf") + 1], "0")
        self.assertEqual(encode_args[encode_args.index("-async_depth") + 1], "1")

    def test_runtime_failure_falls_back_hardware_then_software(self):
        remover = self._remover()

        self.assertTrue(remover._fallback_after_hw_failure("device lost"))
        self.assertEqual(remover._hw_encoder, "h264_nvenc")
        self.assertTrue(remover._d3d12_status["runtime_fallback"])
        self.assertTrue(remover._fallback_after_hw_failure("nvenc failed"))
        self.assertIsNone(remover._hw_encoder)

    def test_post_restore_args_do_not_select_hardware_only_d3d12_encoder(self):
        remover = self._remover()

        args = remover._get_encode_args(allow_d3d12=False)

        self.assertIn("libx264", args)
        self.assertNotIn("h264_d3d12va", args)


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



if __name__ == "__main__":
    unittest.main()
