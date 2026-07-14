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

    def _write_bundle(self, bundle: Path, entries, members):
        from backend import cache_inventory as ci

        manifest = {"schema": ci.PORTABLE_MODEL_CACHE_SCHEMA, "files": entries}
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(ci.MODEL_CACHE_MANIFEST, json.dumps(manifest))
            for archive_path, data in members.items():
                zf.writestr(archive_path, data)

    def test_import_rejects_zip_bomb_compression_ratio(self):
        from backend import cache_inventory as ci
        from backend.model_hashes import hash_file

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bomb = b"\x00" * (2 * 1024 * 1024)  # 2 MiB of zeros -> tiny deflate
            probe = root / "probe.onnx"
            probe.write_bytes(bomb)
            entries = [{
                "cache": "app-model-cache",
                "relative_path": "bomb.onnx",
                "archive_path": "files/app-model-cache/bomb.onnx",
                "filename": "bomb.onnx",
                "sha256": hash_file(probe),
            }]
            bundle = root / "bomb.zip"
            self._write_bundle(bundle, entries,
                               {"files/app-model-cache/bomb.onnx": bomb})

            result = ci.import_model_cache_bundle(bundle, env=_env(root / "dst"))
            self.assertEqual(result["imported"], [])
            reasons = " ".join(i["reason"] for i in result["rejected"])
            self.assertIn("zip bomb", reasons)

    def test_import_rejects_member_count_ceiling(self):
        from backend import cache_inventory as ci

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            entries = [{
                "cache": "app-model-cache",
                "relative_path": f"m{i}.onnx",
                "archive_path": f"files/app-model-cache/m{i}.onnx",
                "filename": f"m{i}.onnx",
                "sha256": "0" * 64,
            } for i in range(ci._IMPORT_MAX_MEMBERS + 1)]
            bundle = root / "many.zip"
            self._write_bundle(bundle, entries, {})
            with self.assertRaises(ValueError) as ctx:
                ci.import_model_cache_bundle(bundle, env=_env(root / "dst"))
            self.assertIn("member ceiling", str(ctx.exception))

    def test_import_rejects_insufficient_free_space(self):
        from backend import cache_inventory as ci
        from backend.model_hashes import hash_file

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload = root / "p.onnx"
            payload.write_bytes(b"weight-bytes")
            entries = [{
                "cache": "app-model-cache",
                "relative_path": "p.onnx",
                "archive_path": "files/app-model-cache/p.onnx",
                "filename": "p.onnx",
                "sha256": hash_file(payload),
            }]
            bundle = root / "b.zip"
            self._write_bundle(bundle, entries,
                               {"files/app-model-cache/p.onnx": b"weight-bytes"})

            import collections
            fake = collections.namedtuple("du", "total used free")(0, 0, 0)
            with mock.patch.object(ci.shutil, "disk_usage", return_value=fake):
                with self.assertRaises(ValueError) as ctx:
                    ci.import_model_cache_bundle(bundle, env=_env(root / "dst"))
            self.assertIn("free space", str(ctx.exception))

    def test_import_rejects_duplicate_targets(self):
        from backend import cache_inventory as ci
        from backend.model_hashes import hash_file

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload = root / "p.onnx"
            payload.write_bytes(b"dup-weight")
            h = hash_file(payload)
            entry = {
                "cache": "app-model-cache",
                "relative_path": "dup.onnx",
                "archive_path": "files/app-model-cache/dup.onnx",
                "filename": "dup.onnx",
                "sha256": h,
            }
            bundle = root / "dup.zip"
            self._write_bundle(bundle, [dict(entry), dict(entry)],
                               {"files/app-model-cache/dup.onnx": b"dup-weight"})

            result = ci.import_model_cache_bundle(bundle, env=_env(root / "dst"))
            self.assertEqual(len(result["imported"]), 1)
            reasons = " ".join(i["reason"] for i in result["rejected"])
            self.assertIn("duplicate import target", reasons)

    def test_import_rolls_back_on_commit_failure(self):
        from backend import cache_inventory as ci
        from backend.model_hashes import hash_file

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            a = root / "a.onnx"
            b = root / "b.onnx"
            a.write_bytes(b"alpha")
            b.write_bytes(b"bravo")
            entries = [
                {"cache": "app-model-cache", "relative_path": "a.onnx",
                 "archive_path": "files/app-model-cache/a.onnx",
                 "filename": "a.onnx", "sha256": hash_file(a)},
                {"cache": "app-model-cache", "relative_path": "b.onnx",
                 "archive_path": "files/app-model-cache/b.onnx",
                 "filename": "b.onnx", "sha256": hash_file(b)},
            ]
            bundle = root / "two.zip"
            self._write_bundle(bundle, entries, {
                "files/app-model-cache/a.onnx": b"alpha",
                "files/app-model-cache/b.onnx": b"bravo",
            })
            dst_env = _env(root / "dst")
            models = (Path(dst_env["APPDATA"]) / "VideoSubtitleRemoverPro" / "models")

            real_replace = os.replace
            calls = {"n": 0}

            def flaky_replace(src, dst):
                # fail the commit of the second staged file (targets end in .onnx)
                if str(dst).endswith("b.onnx"):
                    calls["n"] += 1
                    raise OSError("simulated commit failure")
                return real_replace(src, dst)

            with mock.patch.object(ci.os, "replace", side_effect=flaky_replace):
                with self.assertRaises(ValueError) as ctx:
                    ci.import_model_cache_bundle(bundle, env=dst_env)
            self.assertIn("rolled back", str(ctx.exception))
            # first file must not survive a rolled-back import
            self.assertFalse((models / "a.onnx").exists())
            self.assertFalse((models / "b.onnx").exists())

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
