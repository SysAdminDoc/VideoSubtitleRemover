"""RM-369: RapidOCR runs on CUDA when the NVIDIA build is asked to.

A cuda device used to fall through to RapidOCR's default CPU provider, so OCR
never touched the card and the 720p benchmark measured the NVIDIA build at
the CPU build's speed. These tests pin the CUDA request, the verification of
the sessions actually built, and the recorded fallback when CUDA cannot run.
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from backend import processor
from backend.detection import _build_rapidocr

CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"


class _Session:
    def __init__(self, provider: str):
        self._provider = provider

    def get_providers(self):
        return [self._provider] if self._provider == CPU else [
            self._provider, CPU]


def _fake_rapidocr(providers):
    """A RapidOCR stand-in whose sessions report `providers(params)`."""
    calls = []

    class FakeRapidOCR:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            chosen = providers(kwargs.get("params") or {})
            for name, provider in zip(
                    ("text_det", "text_cls", "text_rec"), chosen, strict=True):
                setattr(self, name, SimpleNamespace(
                    session=SimpleNamespace(session=_Session(provider))))

    return FakeRapidOCR, calls


def _wants_cuda(params):
    return bool(params.get("EngineConfig.onnxruntime.use_cuda"))


class _Env:
    """Fake onnxruntime and rapidocr modules for one detector build."""

    def __init__(self, rapid_cls, available=(CUDA, CPU)):
        self.preloads = []
        self.modules = {
            "onnxruntime": SimpleNamespace(
                get_available_providers=lambda: list(available)),
            "rapidocr": SimpleNamespace(RapidOCR=rapid_cls),
        }

    def __enter__(self):
        self._stack = [
            mock.patch.dict(sys.modules, self.modules),
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch(
                "backend.onnxruntime_cuda.preload_onnxruntime_cuda_dlls_if_needed",
                side_effect=lambda ort, providers, **_: self.preloads.append(
                    list(providers))),
        ]
        for patcher in self._stack:
            patcher.__enter__()
        os.environ.pop("VSR_RAPIDOCR_ENGINE", None)
        return self

    def __exit__(self, *exc):
        for patcher in reversed(self._stack):
            patcher.__exit__(*exc)


class BuildTests(unittest.TestCase):
    def test_a_cuda_device_asks_for_cuda_on_that_card(self):
        rapid_cls, calls = _fake_rapidocr(
            lambda p: [CUDA] * 3 if _wants_cuda(p) else [CPU] * 3)
        with _Env(rapid_cls) as env:
            _instance, provider, reason = _build_rapidocr(rapid_cls, "cuda:1")
        params = calls[0]["params"]
        self.assertIs(params["EngineConfig.onnxruntime.use_cuda"], True)
        self.assertEqual(
            params["EngineConfig.onnxruntime.cuda_ep_cfg.device_id"], 1)
        self.assertEqual(
            params["EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search"],
            "HEURISTIC")
        self.assertEqual((provider, reason), ("CUDA", ""))
        # The CUDA runtime is preloaded before the sessions exist, as for
        # every other CUDA session the product builds.
        self.assertEqual(env.preloads, [[CUDA]])

    def test_sessions_that_landed_on_cpu_are_reported_not_trusted(self):
        rapid_cls, calls = _fake_rapidocr(lambda p: [CUDA, CPU, CPU])
        with _Env(rapid_cls):
            _instance, provider, reason = _build_rapidocr(rapid_cls, "cuda:0")
        self.assertEqual(provider, "CPU")
        self.assertIn("text_rec=CPUExecutionProvider", reason)
        # The working instance is kept rather than built twice.
        self.assertEqual(len(calls), 1)

    def test_a_cuda_start_failure_falls_back_with_the_reason(self):
        attempts = []

        class Flaky:
            def __init__(self, **kwargs):
                attempts.append(kwargs)
                if _wants_cuda(kwargs.get("params") or {}):
                    raise RuntimeError("cublasLt64_13.dll missing")

        with _Env(Flaky):
            _instance, provider, reason = _build_rapidocr(Flaky, "cuda:0")
        self.assertEqual(provider, "CPU")
        self.assertIn("cublasLt64_13.dll missing", reason)
        self.assertEqual(len(attempts), 2)
        self.assertFalse(_wants_cuda(attempts[1].get("params") or {}))

    def test_no_cuda_request_without_the_provider_or_the_device(self):
        rapid_cls, calls = _fake_rapidocr(lambda p: [CPU] * 3)
        with _Env(rapid_cls, available=(CPU,)):
            _build_rapidocr(rapid_cls, "cuda:0")
        with _Env(rapid_cls):
            _build_rapidocr(rapid_cls, "cpu")
            os.environ["VSR_RAPIDOCR_ENGINE"] = "cpu"
            _build_rapidocr(rapid_cls, "cuda:0")
        for call in calls:
            self.assertFalse(_wants_cuda(call.get("params") or {}), call)


class DetectorProvenanceTests(unittest.TestCase):
    def _detector(self, providers):
        rapid_cls, _calls = _fake_rapidocr(providers)
        with _Env(rapid_cls):
            return processor.SubtitleDetector(device="cuda:0")

    def test_cuda_sessions_are_not_reported_as_a_fallback(self):
        detector = self._detector(
            lambda p: [CUDA] * 3 if _wants_cuda(p) else [CPU] * 3)
        self.assertEqual(detector._engine_name, "RapidOCR (CUDA)")
        stage = detector.execution_provenance()
        self.assertEqual(stage.provider, CUDA)
        self.assertFalse(stage.fell_back, stage.to_dict())

    def test_a_cpu_landing_is_a_fallback_with_its_reason(self):
        detector = self._detector(lambda p: [CPU] * 3)
        stage = detector.execution_provenance()
        self.assertTrue(stage.fell_back)
        self.assertIn("RapidOCR asked for CUDA", stage.to_dict()["fallbackReason"])


@unittest.skipUnless(
    os.environ.get("VSR_GPU_TESTS", "").strip().lower() in {"1", "true", "yes", "on"},
    "set VSR_GPU_TESTS=1 to run the GPU integration lane")
class RealCudaTests(unittest.TestCase):
    def test_the_real_detector_runs_ocr_on_cuda(self):
        try:
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnxruntime is not installed")
        if CUDA not in ort.get_available_providers():
            self.skipTest("this onnxruntime build offers no CUDA provider")
        detector = processor.SubtitleDetector(device="cuda:0", engine="rapidocr")
        if detector._engine_name != "RapidOCR (CUDA)":
            self.skipTest(
                "CUDA runtime unavailable on this host: "
                + detector.execution_provenance().to_dict()["fallbackReason"])
        frame = np.full((96, 480, 3), 32, np.uint8)
        import cv2

        cv2.putText(frame, "CUDA OCR", (20, 64), cv2.FONT_HERSHEY_SIMPLEX,
                    1.6, (240, 240, 240), 3, cv2.LINE_AA)
        boxes = detector.detect(frame, 0.3)
        self.assertTrue(boxes)
        from backend.detection import _rapidocr_session_providers

        self.assertEqual(
            set(_rapidocr_session_providers(detector._rapid_model).values()),
            {CUDA})


if __name__ == "__main__":
    unittest.main()
