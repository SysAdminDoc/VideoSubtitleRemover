"""RM-134: module-boundary unit tests for backend.adapter_manifest.

adapter_manifest is the license/hash trust gate for opt-in model adapters:
it decides whether a user-supplied weight file is allowed to load based on
pinned SHA-256, remote-code requirements, and the unsafe override. It was
only referenced indirectly by the hardening suite; these lock the gate.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend import adapter_manifest as am
from backend.model_hashes import hash_file


class ManifestParseTests(unittest.TestCase):
    def test_known_entry_is_returned(self):
        entry = am.get_manifest_entry("lama-onnx")
        self.assertEqual(entry.name, "lama-onnx")
        self.assertIn("lama.onnx", entry.expected_filenames)

    def test_unknown_entry_raises_keyerror(self):
        with self.assertRaises(KeyError):
            am.get_manifest_entry("does-not-exist")


class VerifyAdapterPathTests(unittest.TestCase):
    def _write(self, tmp: Path, name: str, data: bytes) -> str:
        p = tmp / name
        p.write_bytes(data)
        return str(p)

    def test_not_configured_when_path_empty(self):
        result = am.verify_adapter_path("lama-onnx", "")
        self.assertFalse(result.configured)
        self.assertFalse(result.allowed)
        self.assertEqual(result.hash_status, "not_configured")

    def test_missing_file_is_blocked(self):
        result = am.verify_adapter_path("lama-onnx", "/no/such/model.onnx")
        self.assertTrue(result.configured)
        self.assertFalse(result.exists)
        self.assertFalse(result.allowed)
        self.assertEqual(result.hash_status, "missing")

    def test_pinned_hash_mismatch_is_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._write(
                Path(d), "inpainting_lama_2025jan.onnx", b"not the real weights")
            result = am.verify_adapter_path("opencv-lama", path)
        self.assertTrue(result.exists)
        self.assertFalse(result.allowed)
        self.assertEqual(result.hash_status, "mismatch")

    def test_unsafe_override_allows_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._write(
                Path(d), "inpainting_lama_2025jan.onnx", b"not the real weights")
            result = am.verify_adapter_path(
                "opencv-lama", path,
                env={am.UNSAFE_OVERRIDE_ENV: "1"})
        self.assertTrue(result.allowed)
        self.assertTrue(result.unsafe_override)
        self.assertEqual(result.hash_status, "unsafe_override")

    def test_matching_pinned_hash_is_verified(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._write(Path(d), "model.onnx", b"deterministic content")
            digest = hash_file(Path(path)).lower()
            entry = am.AdapterManifestEntry(
                name="unit-test-adapter",
                env_vars=("VSR_UNIT_TEST",),
                expected_filenames=("model.onnx",),
                sha256={"model.onnx": digest},
                license="MIT",
            )
            with mock.patch.dict(am.ADAPTER_MANIFEST,
                                 {"unit-test-adapter": entry}):
                result = am.verify_adapter_path("unit-test-adapter", path)
        self.assertTrue(result.allowed)
        self.assertEqual(result.hash_status, "verified")
        self.assertEqual(result.actual_sha256, digest)

    def test_hashless_legacy_adapter_is_allowed_unknown(self):
        # migan-onnx has no pinned hash and does not require remote code.
        with tempfile.TemporaryDirectory() as d:
            path = self._write(Path(d), "migan.onnx", b"legacy weights")
            result = am.verify_adapter_path("migan-onnx", path)
        self.assertTrue(result.allowed)
        self.assertEqual(result.hash_status, "unknown")

    def test_remote_code_adapter_without_hash_is_blocked(self):
        # clear-maskfree requires remote code and has no pinned hash.
        with tempfile.TemporaryDirectory() as d:
            path = self._write(Path(d), "clear.safetensors", b"unpinned weights")
            result = am.verify_adapter_path("clear-maskfree", path)
        self.assertFalse(result.allowed)
        self.assertEqual(result.hash_status, "unknown")
        self.assertTrue(result.strict_unknown)

    def test_strict_unknown_blocks_hashless_legacy_adapter(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._write(Path(d), "migan.onnx", b"legacy weights")
            result = am.verify_adapter_path(
                "migan-onnx", path, strict_unknown=True)
        self.assertFalse(result.allowed)
        self.assertEqual(result.hash_status, "unknown")


if __name__ == "__main__":
    unittest.main()
