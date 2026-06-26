import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock


def _cfg(mode="sttn", **overrides):
    data = {
        "mode": SimpleNamespace(value=mode),
        "whisper_fallback": False,
        "whisper_backend": "faster-whisper",
        "whisper_model_size": "tiny",
        "whisper_model_path": "",
    }
    data.update(overrides)
    return SimpleNamespace(**data)


class ModelDownloadHintTests(unittest.TestCase):
    def test_rapidocr_default_needs_no_download_hint(self):
        from backend import model_downloads as md

        def find_spec(name):
            return object() if name == "rapidocr" else None

        with tempfile.TemporaryDirectory() as tmpdir:
            env = {"HOME": tmpdir, "USERPROFILE": tmpdir, "APPDATA": tmpdir}
            with mock.patch.object(md.importlib.util, "find_spec", side_effect=find_spec):
                self.assertEqual(md.pending_model_download_hints(_cfg(), env), ())

    def test_lama_fallback_reports_weight_download_when_opted_in(self):
        from backend import model_downloads as md

        def find_spec(name):
            return object() if name == "simple_lama_inpainting" else None

        with tempfile.TemporaryDirectory() as tmpdir:
            env = {
                "HOME": tmpdir,
                "USERPROFILE": tmpdir,
                "APPDATA": tmpdir,
                "VSR_ENABLE_PYTORCH_LAMA": "1",
            }
            with mock.patch.object(md.importlib.util, "find_spec", side_effect=find_spec):
                hints = md.pending_model_download_hints(_cfg(mode="lama"), env)

        self.assertEqual(len(hints), 1)
        self.assertEqual(hints[0].label, "LaMa inpainting weights")
        self.assertIn("200 MB", hints[0].size_estimate)

    def test_lama_fallback_does_not_report_download_when_disabled(self):
        from backend import model_downloads as md

        def find_spec(name):
            return object() if name == "simple_lama_inpainting" else None

        with tempfile.TemporaryDirectory() as tmpdir:
            env = {"HOME": tmpdir, "USERPROFILE": tmpdir, "APPDATA": tmpdir}
            with mock.patch.object(md.importlib.util, "find_spec", side_effect=find_spec):
                hints = md.pending_model_download_hints(_cfg(mode="lama"), env)

        self.assertEqual(hints, ())

    def test_florence_pinned_revision_reports_huggingface_download(self):
        from backend import model_downloads as md

        with tempfile.TemporaryDirectory() as tmpdir:
            env = {
                "HOME": tmpdir,
                "USERPROFILE": tmpdir,
                "APPDATA": tmpdir,
                "VSR_VLM_OCR": "florence2",
                "VSR_FLORENCE2_REVISION": "v1.0.0",
            }
            with mock.patch.object(md.importlib.util, "find_spec", return_value=None):
                hints = md.pending_model_download_hints(_cfg(), env)

        self.assertEqual(len(hints), 1)
        self.assertEqual(hints[0].label, "Florence-2 VLM OCR")

    def test_paddleocr_vl_llama_reports_local_server_setup(self):
        from backend import model_downloads as md

        with tempfile.TemporaryDirectory() as tmpdir:
            env = {
                "HOME": tmpdir,
                "USERPROFILE": tmpdir,
                "APPDATA": tmpdir,
                "VSR_PADDLEOCR_VL": "1",
            }
            with mock.patch.object(md.importlib.util, "find_spec", return_value=None):
                hints = md.pending_model_download_hints(_cfg(), env)

        self.assertEqual(len(hints), 1)
        self.assertEqual(hints[0].label, "PaddleOCR-VL-1.5 GGUF")
        self.assertIn("llama-server", hints[0].detail)

    def test_whisper_fallback_reports_model_size(self):
        from backend import model_downloads as md

        with tempfile.TemporaryDirectory() as tmpdir:
            env = {"HOME": tmpdir, "USERPROFILE": tmpdir, "APPDATA": tmpdir}
            with mock.patch.object(md.importlib.util, "find_spec", return_value=None):
                hints = md.pending_model_download_hints(
                    _cfg(whisper_fallback=True, whisper_model_size="small"),
                    env,
                )

        self.assertEqual(len(hints), 1)
        self.assertEqual(hints[0].label, "Whisper small model")
        self.assertIn("460 MB", hints[0].size_estimate)

    def test_installed_backend_status_is_privacy_safe_and_actionable(self):
        from backend import model_downloads as md

        rapid = {
            "package": {"name": "rapidocr", "version": "3.9.0"},
            "model_count": 3,
            "model_families": ["pp-ocrv6"],
            "required_assets": [
                {"name": "RapidOCR config.yaml", "required": True, "present": True},
                {"name": "PP-OCRv6 detection ONNX", "required": True, "present": True},
            ],
            "packaging_compatible": True,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            env = {"HOME": tmpdir, "USERPROFILE": tmpdir, "APPDATA": tmpdir}
            with mock.patch.object(md, "_onnxruntime_provider_status",
                                   return_value={
                                       "available": True,
                                       "version": "1.23.0",
                                       "providers": ["CPUExecutionProvider"],
                                       "next_action": "",
                                   }):
                with mock.patch.object(md, "_opencv_runtime_status",
                                       return_value={
                                           "available": True,
                                           "version": "4.13.0",
                                           "dnn_available": True,
                                           "opencv5": False,
                                           "next_action": "",
                                       }):
                    with mock.patch.object(md, "_module_status",
                                           return_value={
                                               "name": "PyTorch",
                                               "available": False,
                                               "version": None,
                                               "status": "not_installed",
                                               "next_action": "",
                                           }):
                        with mock.patch(
                            "backend.onnx_model_info.rapidocr_release_provenance",
                            return_value=rapid,
                        ):
                            with mock.patch.object(md.importlib.util, "find_spec",
                                                   return_value=None):
                                status = md.installed_backend_status(
                                    _cfg(mode="lama"),
                                    env,
                                )

        self.assertEqual(status["schema"], "vsr.backend_status.v1")
        self.assertEqual(status["detection"][0]["status"], "ready")
        self.assertEqual(status["detection"][0]["model_count"], 3)
        self.assertIn("CPUExecutionProvider", status["summary"]["providers"])
        self.assertIn("LaMa neural weights not ready",
                      status["summary"]["model_files"])
        self.assertIn("VSR_LAMA_ONNX", status["summary"]["next_action"])
        self.assertNotIn(tmpdir, repr(status))


if __name__ == "__main__":
    unittest.main()
