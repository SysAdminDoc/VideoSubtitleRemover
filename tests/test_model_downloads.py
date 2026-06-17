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

    def test_lama_fallback_reports_weight_download(self):
        from backend import model_downloads as md

        def find_spec(name):
            return object() if name == "simple_lama_inpainting" else None

        with tempfile.TemporaryDirectory() as tmpdir:
            env = {"HOME": tmpdir, "USERPROFILE": tmpdir, "APPDATA": tmpdir}
            with mock.patch.object(md.importlib.util, "find_spec", side_effect=find_spec):
                hints = md.pending_model_download_hints(_cfg(mode="lama"), env)

        self.assertEqual(len(hints), 1)
        self.assertEqual(hints[0].label, "LaMa inpainting weights")
        self.assertIn("200 MB", hints[0].size_estimate)

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


if __name__ == "__main__":
    unittest.main()
