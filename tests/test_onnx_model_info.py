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
            root = Path(tmp)
            (root / "config.yaml").write_text("Global: {}\n", encoding="utf-8")
            (root / "default_models.yaml").write_text(
                "onnxruntime: {}\n",
                encoding="utf-8",
            )
            model_dir = root / "models"
            model_dir.mkdir()
            model = model_dir / "PP-OCRv6_det_small.onnx"
            rec_model = model_dir / "PP-OCRv6_rec_small.onnx"
            payload = _minimal_onnx_model_with_opset(11)
            model.write_bytes(payload)
            rec_model.write_bytes(payload)

            result = rapidocr_release_provenance(
                model_dir,
                package_name="rapidocr",
                package_version="3.9.0",
            )

            self.assertFalse(result["missing"])
            self.assertEqual(result["model_count"], 2)
            self.assertEqual(result["package"]["name"], "rapidocr")
            self.assertEqual(result["package"]["version"], "3.9.0")
            self.assertIn("pp-ocrv6", result["model_families"])
            self.assertTrue(result["packaging_compatible"])
            self.assertTrue(all(item["exists"] for item in result["config_files"]))
            self.assertTrue(all(item["present"] for item in result["required_assets"]))
            record = result["models"][0]
            self.assertEqual(record["filename"], "PP-OCRv6_det_small.onnx")
            self.assertEqual(record["relative_path"], "PP-OCRv6_det_small.onnx")
            self.assertEqual(record["bytes"], len(payload))
            self.assertEqual(record["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(record["max_default_opset"], 11)
            self.assertTrue(record["directml_compatible"])

    def test_rapidocr_v3_provenance_flags_missing_required_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.yaml").write_text("Global: {}\n", encoding="utf-8")
            model_dir = root / "models"
            model_dir.mkdir()
            model = model_dir / "PP-OCRv6_det_small.onnx"
            model.write_bytes(_minimal_onnx_model_with_opset(11))

            result = rapidocr_release_provenance(
                model_dir,
                package_name="rapidocr",
                package_version="3.9.0",
            )

            self.assertFalse(result["packaging_compatible"])
            missing = [
                item["name"] for item in result["required_assets"]
                if item["required"] and not item["present"]
            ]
            self.assertIn("RapidOCR default_models.yaml", missing)
            self.assertIn("PP-OCRv6 recognition ONNX", missing)

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
