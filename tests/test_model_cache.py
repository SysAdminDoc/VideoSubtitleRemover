import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest import mock
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def _env(root: Path) -> dict:
    env = os.environ.copy()
    env["APPDATA"] = str(root / "appdata")
    env["USERPROFILE"] = str(root / "home")
    env["HOME"] = str(root / "home")
    return env


class ModelCacheBundleTests(unittest.TestCase):
    def test_export_imports_verified_known_file_into_app_cache(self):
        from backend import cache_inventory as ci
        from backend import model_hashes

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src_env = _env(root / "src")
            src_models = (
                Path(src_env["APPDATA"])
                / "VideoSubtitleRemoverPro"
                / "models"
            )
            src_models.mkdir(parents=True)
            weight = src_models / "unit.pt"
            weight.write_bytes(b"unit test weight")
            expected = model_hashes.hash_file(weight)
            bundle = root / "models.zip"
            dst_env = _env(root / "dst")

            with mock.patch.dict(
                model_hashes.KNOWN_WEIGHT_HASHES,
                {"unit.pt": expected},
                clear=False,
            ):
                exported = ci.export_model_cache_bundle(bundle, env=src_env)
                imported = ci.import_model_cache_bundle(bundle, env=dst_env)

            self.assertEqual(exported["schema"], ci.PORTABLE_MODEL_CACHE_SCHEMA)
            self.assertEqual(len(exported["files"]), 1)
            self.assertTrue(exported["files"][0]["known_hash"])
            self.assertEqual(imported["rejected"], [])
            self.assertEqual(len(imported["imported"]), 1)
            restored = (
                Path(dst_env["APPDATA"])
                / "VideoSubtitleRemoverPro"
                / "models"
                / "unit.pt"
            )
            self.assertEqual(restored.read_bytes(), b"unit test weight")

            with zipfile.ZipFile(bundle) as zf:
                manifest = json.loads(zf.read(ci.MODEL_CACHE_MANIFEST))
            self.assertEqual(manifest["files"][0]["sha256"], expected)

    def test_import_rejects_traversal_executables_and_hash_mismatch(self):
        from backend import cache_inventory as ci
        from backend.model_hashes import hash_file

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = root / "bad.zip"
            payload = root / "payload.onnx"
            payload.write_bytes(b"payload")
            good_hash = hash_file(payload)
            manifest = {
                "schema": ci.PORTABLE_MODEL_CACHE_SCHEMA,
                "files": [
                    {
                        "cache": "app-model-cache",
                        "relative_path": "../evil.onnx",
                        "archive_path": "files/app-model-cache/evil.onnx",
                        "filename": "evil.onnx",
                        "sha256": good_hash,
                    },
                    {
                        "cache": "app-model-cache",
                        "relative_path": "bad.py",
                        "archive_path": "files/app-model-cache/bad.py",
                        "filename": "bad.py",
                        "sha256": good_hash,
                    },
                    {
                        "cache": "app-model-cache",
                        "relative_path": "wrong.onnx",
                        "archive_path": "files/app-model-cache/wrong.onnx",
                        "filename": "wrong.onnx",
                        "sha256": "0" * 64,
                    },
                ],
            }
            with zipfile.ZipFile(bundle, "w") as zf:
                zf.writestr(ci.MODEL_CACHE_MANIFEST, json.dumps(manifest))
                for entry in manifest["files"]:
                    zf.write(payload, entry["archive_path"])

            result = ci.import_model_cache_bundle(bundle, env=_env(root / "dst"))

            self.assertEqual(result["imported"], [])
            self.assertEqual(len(result["rejected"]), 3)
            reasons = " ".join(item["reason"] for item in result["rejected"])
            self.assertIn("unsafe manifest path", reasons)
            self.assertIn("executable code", reasons)
            self.assertIn("SHA-256 mismatch", reasons)
            self.assertFalse((root / "evil.onnx").exists())

    def test_cli_export_import_entrypoints_do_not_require_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = root / "empty-cache.zip"
            export_env = _env(root / "export")
            import_env = _env(root / "import")

            export_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "backend.cli",
                    "--model-cache-export",
                    str(bundle),
                ],
                cwd=ROOT,
                env=export_env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(export_result.returncode, 0, export_result.stderr)
            self.assertIn("[model-cache] exported", export_result.stdout)
            self.assertTrue(bundle.is_file())

            import_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "backend.cli",
                    "--model-cache-import",
                    str(bundle),
                ],
                cwd=ROOT,
                env=import_env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(import_result.returncode, 0, import_result.stderr)
            self.assertIn("[model-cache] imported", import_result.stdout)


if __name__ == "__main__":
    unittest.main()
