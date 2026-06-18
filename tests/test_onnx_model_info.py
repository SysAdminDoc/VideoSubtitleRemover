import hashlib
from pathlib import Path
import tempfile
import unittest

from backend.onnx_model_info import rapidocr_release_provenance


def _minimal_onnx_model_with_opset(version: int) -> bytes:
    if not 0 <= version < 128:
        raise ValueError("test helper only supports one-byte varints")
    return bytes([0x42, 0x02, 0x10, version])


class RapidOcrProvenanceTests(unittest.TestCase):
    def test_rapidocr_provenance_hashes_and_audits_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "models"
            model_dir.mkdir()
            model = model_dir / "det.onnx"
            payload = _minimal_onnx_model_with_opset(11)
            model.write_bytes(payload)

            result = rapidocr_release_provenance(
                model_dir,
                package_name="rapidocr",
                package_version="2.1.0",
            )

            self.assertFalse(result["missing"])
            self.assertEqual(result["model_count"], 1)
            self.assertEqual(result["package"]["name"], "rapidocr")
            self.assertEqual(result["package"]["version"], "2.1.0")
            record = result["models"][0]
            self.assertEqual(record["filename"], "det.onnx")
            self.assertEqual(record["relative_path"], "det.onnx")
            self.assertEqual(record["bytes"], len(payload))
            self.assertEqual(record["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(record["max_default_opset"], 11)
            self.assertTrue(record["directml_compatible"])

    def test_rapidocr_provenance_marks_missing_model_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = rapidocr_release_provenance(
                Path(tmp) / "missing",
                package_name="rapidocr",
                package_version="2.1.0",
            )

            self.assertTrue(result["missing"])
            self.assertEqual(result["model_count"], 0)
            self.assertEqual(result["models"], [])


if __name__ == "__main__":
    unittest.main()
