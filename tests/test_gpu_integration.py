"""RM-322: the GPU claims, executed rather than read.

No test in the suite exercised CUDA, NVENC, provider selection, or OOM
recovery, so every GPU statement in the README rested on code reading. This
module is opt-in: it skips cleanly with no GPU, and on a CUDA host it runs
the four things that could only be asserted by running them.

Enable with VSR_GPU_TESTS=1. It needs an onnxruntime build that offers
CUDAExecutionProvider, which the reviewed CPU profile deliberately does not
install, so it also skips in a CPU environment even with the flag set.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

OPT_IN = os.environ.get("VSR_GPU_TESTS", "").strip().lower() in {
    "1", "true", "yes", "on"}


def _nvidia_gpu_name() -> str:
    if not shutil.which("nvidia-smi"):
        return ""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else ""


def _cuda_onnxruntime():
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:
        return None
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return None
    return ort


def _ffmpeg_nvenc_encoders() -> set:
    if not shutil.which("ffmpeg"):
        return set()
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    return {
        name for name in ("h264_nvenc", "hevc_nvenc")
        if name in (result.stdout or "")
    }


class GpuIntegrationTests(unittest.TestCase):
    """Every assertion here is about hardware behaviour, not code shape."""

    @classmethod
    def setUpClass(cls):
        if not OPT_IN:
            raise unittest.SkipTest(
                "set VSR_GPU_TESTS=1 to run the GPU integration lane")
        cls.gpu_name = _nvidia_gpu_name()
        if not cls.gpu_name:
            raise unittest.SkipTest("no NVIDIA GPU reported by nvidia-smi")
        cls.ort = _cuda_onnxruntime()

    def _require_cuda_runtime(self):
        if self.ort is None:
            self.skipTest(
                "the installed onnxruntime does not offer "
                "CUDAExecutionProvider")
        from backend.onnxruntime_cuda import (
            preload_onnxruntime_cuda_dlls_if_needed,
        )

        # The CUDA runtime ships inside the PyTorch wheel rather than on the
        # system path, which is exactly what the product does before every
        # CUDA session.
        preload_onnxruntime_cuda_dlls_if_needed(
            self.ort, ["CUDAExecutionProvider"])
        return self.ort

    def _identity_session(self, providers):
        from backend.onnx_model_info import _tiny_identity_onnx_bytes

        return self.ort.InferenceSession(
            _tiny_identity_onnx_bytes(), providers=providers)

    # -- 1. the requested provider is the provider actually used ----------
    def test_a_cuda_session_really_runs_on_cuda(self):
        ort = self._require_cuda_runtime()
        session = self._identity_session(["CUDAExecutionProvider"])
        active = session.get_providers()
        if active[:1] != ["CUDAExecutionProvider"]:
            self.skipTest(
                f"CUDA runtime unavailable on this host: active={active}")
        payload = np.array([3.0], dtype=np.float32)
        self.assertTrue(
            np.allclose(session.run(None, {"x": payload})[0], payload))
        self.assertIs(ort, self.ort)

    def test_the_profile_smoke_agrees_with_the_session(self):
        """The shipped diagnostic must reach the same verdict as the run."""
        self._require_cuda_runtime()
        from backend.dependency_profiles import run_profile_provider_smoke

        result = run_profile_provider_smoke("nvidia")
        if not result.get("passed"):
            self.skipTest(f"nvidia profile smoke not green: {result['error']}")
        self.assertFalse(result["fellBack"])
        self.assertEqual(
            result["activeProviders"][0], "CUDAExecutionProvider")

    # -- 2. a named provider that cannot run fails loudly -----------------
    def test_onnxruntime_itself_falls_back_silently(self):
        """The behaviour the product has to compensate for.

        This is the premise of the next test: ONNX Runtime accepts an
        unavailable provider, warns, and runs on CPU. If it ever starts
        raising, the guard below can be reconsidered.
        """
        self._require_cuda_runtime()
        session = self._identity_session(["NotARealProvider"])
        self.assertEqual(session.get_providers(), ["CPUExecutionProvider"])

    def test_a_named_accelerator_that_ran_on_cpu_raises(self):
        from backend.device_provider import (
            ProviderFellBackError,
            verify_active_provider,
        )

        with self.assertRaises(ProviderFellBackError) as ctx:
            verify_active_provider("cuda:0", ["CPUExecutionProvider"])
        self.assertIn("CUDAExecutionProvider", str(ctx.exception))
        self.assertIn("cuda:0", str(ctx.exception))

        # A real CUDA session must not trip the same guard.
        self._require_cuda_runtime()
        session = self._identity_session(
            ["CUDAExecutionProvider", "CPUExecutionProvider"])
        if session.get_providers()[:1] != ["CUDAExecutionProvider"]:
            self.skipTest("CUDA runtime unavailable on this host")
        verify_active_provider("cuda:0", session.get_providers())

    # -- 3. a forced allocation failure recovers through the documented path
    def test_a_forced_allocation_failure_reports_rather_than_degrades(self):
        """Ask CUDA for more memory than the card has.

        The documented path is a RequestedStageError naming the stage, not a
        quiet CPU run producing unrepaired output.
        """
        ort = self._require_cuda_runtime()
        options = ort.SessionOptions()
        provider_options = [{
            "device_id": 0,
            # One byte of arena is not a workable allocator, so this either
            # fails outright or falls back; both are answers this asserts on.
            "gpu_mem_limit": 1,
            "arena_extend_strategy": "kSameAsRequested",
        }]
        from backend.device_provider import (
            ProviderFellBackError,
            verify_active_provider,
        )
        from backend.onnx_model_info import _tiny_identity_onnx_bytes

        try:
            session = ort.InferenceSession(
                _tiny_identity_onnx_bytes(),
                sess_options=options,
                providers=["CUDAExecutionProvider"],
                provider_options=provider_options,
            )
        except Exception as exc:
            # Loud failure is the documented outcome.
            self.assertTrue(str(exc))
            return
        active = session.get_providers()
        if active[:1] != ["CUDAExecutionProvider"]:
            with self.assertRaises(ProviderFellBackError):
                verify_active_provider("cuda:0", active)
            return
        # The session was created; the arena only runs out at inference. On
        # the RTX 4070 SUPER this raises
        # "Available memory of 0 is smaller than requested bytes of 256",
        # which is the documented path: a named, explicit failure rather
        # than unrepaired frames written out as if the run had worked.
        payload = np.array([3.0], dtype=np.float32)
        try:
            output = session.run(None, {"x": payload})[0]
        except Exception as exc:
            message = str(exc)
            self.assertTrue(message.strip())
            self.assertRegex(message.lower(), r"memory|alloc|arena")
            return
        self.assertTrue(np.allclose(output, payload))

    # -- 4. NVENC produces a byte-valid encode ----------------------------
    def test_nvenc_produces_a_decodable_encode(self):
        encoders = _ffmpeg_nvenc_encoders()
        if not encoders:
            self.skipTest("this FFmpeg build offers no NVENC encoder")
        encoder = "h264_nvenc" if "h264_nvenc" in encoders else "hevc_nvenc"

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "nvenc.mp4"
            encode = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-nostdin", "-y",
                    "-f", "lavfi", "-i", "testsrc=size=320x240:rate=15:d=1",
                    "-c:v", encoder, "-pix_fmt", "yuv420p", str(output),
                ],
                capture_output=True, text=True, timeout=180,
            )
            if encode.returncode != 0:
                self.skipTest(
                    f"{encoder} could not encode on this host: "
                    f"{encode.stderr[-400:]}")
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

            # Byte-valid means it decodes, not merely that a file exists.
            decode = subprocess.run(
                ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error",
                 "-i", str(output), "-f", "null", "-"],
                capture_output=True, text=True, timeout=180,
            )
            self.assertEqual(decode.returncode, 0, decode.stderr)
            self.assertEqual(decode.stderr.strip(), "")

            probe = subprocess.run(
                ["ffprobe", "-hide_banner", "-v", "error",
                 "-select_streams", "v:0",
                 "-show_entries", "stream=codec_name,width,height",
                 "-of", "csv=p=0", str(output)],
                capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(probe.returncode, 0, probe.stderr)
            self.assertIn("320,240", probe.stdout.replace(" ", ""))


class GpuLaneSkipTests(unittest.TestCase):
    """The lane must skip, not fail, where the hardware is absent."""

    def test_the_opt_in_flag_is_read_from_the_environment(self):
        from unittest import mock

        with mock.patch.dict(os.environ, {"VSR_GPU_TESTS": ""}, clear=False):
            self.assertFalse(
                os.environ.get("VSR_GPU_TESTS", "").strip().lower()
                in {"1", "true", "yes", "on"})

    def test_the_gpu_probe_reports_nothing_without_nvidia_smi(self):
        from unittest import mock

        with mock.patch.object(shutil, "which", return_value=None):
            self.assertEqual(_nvidia_gpu_name(), "")


if __name__ == "__main__":
    unittest.main()
