import os
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace




def _has_display() -> bool:
    """Return True if a GUI display is available."""
    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class RemoteModelPolicyTests(unittest.TestCase):
    def test_code_executing_remote_adapter_requires_full_commit_sha(self):
        from backend.remote_model_policy import resolve_remote_model_source

        tag = resolve_remote_model_source(
            "cotracker3", {"VSR_COTRACKER_REF": "v1.2.3"})
        short_sha = resolve_remote_model_source(
            "cotracker3", {"VSR_COTRACKER_REF": "deadbeef"})
        full_sha = resolve_remote_model_source(
            "cotracker3", {"VSR_COTRACKER_REF": "a" * 40})

        self.assertFalse(tag.allowed)
        self.assertFalse(short_sha.allowed)
        self.assertIn("tags and branches are mutable", tag.reason)
        self.assertTrue(full_sha.allowed)
        self.assertEqual(full_sha.reason, "approved immutable remote commit")

    def test_non_executing_adapter_can_use_version_tag(self):
        from backend.remote_model_policy import resolve_remote_model_source

        source = resolve_remote_model_source(
            "qwen25vl", {"VSR_QWEN25VL_REVISION": "v1.2.3"})

        self.assertTrue(source.allowed)


class ModelHashVerificationTests(unittest.TestCase):
    """RM-49: verify_weight_file should return True for a match,
    False for a mismatch, and True (with a debug log) when no vendored
    hash exists for the filename."""

    def test_verify_match(self):
        from backend.model_hashes import verify_weight_file, hash_file
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "weights.pt"
            p.write_bytes(b"hello world")
            expected = hash_file(p)
            self.assertTrue(verify_weight_file(p, expected_hash=expected))

    def test_verify_mismatch(self):
        from backend.model_hashes import verify_weight_file
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "weights.pt"
            p.write_bytes(b"hello world")
            self.assertFalse(verify_weight_file(p, expected_hash="0" * 64))

    def test_verify_unknown_filename_returns_true(self):
        from backend.model_hashes import verify_weight_file
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "not-tracked.bin"
            p.write_bytes(b"some bytes")
            # No vendored hash entry; verifier returns True (no-op).
            self.assertTrue(verify_weight_file(p))

    def test_verify_missing_file_returns_false(self):
        from backend.model_hashes import verify_weight_file
        result = verify_weight_file(Path("/nonexistent/weights.pt"),
                                      expected_hash="0" * 64)
        self.assertFalse(result)


class AdapterManifestVerificationTests(unittest.TestCase):
    """#109: optional adapter model paths carry provenance and can fail
    closed on unknown or mismatched weights before a loader deserializes
    the file."""

    def _entry(self, filename: str, sha256=None):
        from backend.adapter_manifest import AdapterManifestEntry
        return AdapterManifestEntry(
            name="unit-adapter",
            env_vars=("VSR_UNIT_MODEL",),
            expected_filenames=(filename,),
            sha256={} if sha256 is None else {filename: sha256},
            license="test-license",
            source_url="https://example.invalid/model",
            preferred_format="ONNX",
            remote_code_required=False,
        )

    def test_unknown_hash_allowed_for_legacy_adapter(self):
        from unittest import mock
        from backend import adapter_manifest as manifest
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "model.onnx"
            p.write_bytes(b"model")
            with mock.patch.dict(
                manifest.ADAPTER_MANIFEST,
                {"unit-adapter": self._entry(p.name)},
                clear=False,
            ):
                result = manifest.verify_adapter_path("unit-adapter", str(p))
        self.assertTrue(result.allowed)
        self.assertEqual(result.hash_status, "unknown")
        payload = result.as_dict()
        self.assertEqual(payload["preferredFormat"], "ONNX")
        self.assertEqual(payload["license"], "test-license")

    def test_strict_unknown_hash_fails_without_unsafe_override(self):
        from unittest import mock
        from backend import adapter_manifest as manifest
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "model.onnx"
            p.write_bytes(b"model")
            with mock.patch.dict(
                manifest.ADAPTER_MANIFEST,
                {"unit-adapter": self._entry(p.name)},
                clear=False,
            ):
                result = manifest.verify_adapter_path(
                    "unit-adapter", str(p), strict_unknown=True, env={}
                )
                override = manifest.verify_adapter_path(
                    "unit-adapter",
                    str(p),
                    strict_unknown=True,
                    env={manifest.UNSAFE_OVERRIDE_ENV: "1"},
                )
        self.assertFalse(result.allowed)
        self.assertEqual(result.hash_status, "unknown")
        self.assertTrue(override.allowed)
        self.assertEqual(override.hash_status, "unsafe_override")
        self.assertTrue(override.unsafe_override)

    def test_hash_mismatch_fails_unless_override_is_explicit(self):
        from unittest import mock
        from backend import adapter_manifest as manifest
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "model.onnx"
            p.write_bytes(b"model")
            with mock.patch.dict(
                manifest.ADAPTER_MANIFEST,
                {"unit-adapter": self._entry(p.name, "0" * 64)},
                clear=False,
            ):
                result = manifest.verify_adapter_path("unit-adapter", str(p))
                override = manifest.verify_adapter_path(
                    "unit-adapter",
                    str(p),
                    env={manifest.UNSAFE_OVERRIDE_ENV: "true"},
                )
        self.assertFalse(result.allowed)
        self.assertEqual(result.hash_status, "mismatch")
        self.assertTrue(result.actual_sha256)
        self.assertTrue(override.allowed)
        self.assertEqual(override.hash_status, "unsafe_override")

    def test_release_manifest_reports_configured_adapter_status(self):
        from unittest import mock
        from backend import adapter_manifest as manifest
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "model.onnx"
            p.write_bytes(b"model")
            with mock.patch.dict(
                manifest.ADAPTER_MANIFEST,
                {"unit-adapter": self._entry(p.name)},
                clear=True,
            ):
                statuses = manifest.release_manifest_status(
                    env={"VSR_UNIT_MODEL": str(p)}
                )
        self.assertEqual(len(statuses), 1)
        self.assertEqual(statuses[0]["name"], "unit-adapter")
        self.assertEqual(statuses[0]["configuredEnvVar"], "VSR_UNIT_MODEL")
        self.assertEqual(statuses[0]["hashStatus"], "unknown")

    def test_onnx_loader_refuses_mismatched_manifest_before_session(self):
        from unittest import mock
        from backend import adapter_manifest as manifest
        from backend import inpainters_onnx
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "model.onnx"
            p.write_bytes(b"model")
            session = mock.Mock()
            fake_ort = SimpleNamespace(InferenceSession=session)
            entry = manifest.AdapterManifestEntry(
                name="lama-onnx",
                env_vars=("VSR_LAMA_ONNX",),
                expected_filenames=(p.name,),
                sha256={p.name: "0" * 64},
                license="test-license",
                source_url="https://example.invalid/model",
                preferred_format="ONNX",
                remote_code_required=False,
            )
            with mock.patch.dict(
                manifest.ADAPTER_MANIFEST, {"lama-onnx": entry}, clear=False
            ), mock.patch.dict(sys.modules, {"onnxruntime": fake_ort}):
                result = inpainters_onnx._maybe_session(
                    str(p), ["CPUExecutionProvider"], "lama-onnx"
                )
        self.assertIsNone(result)
        session.assert_not_called()


class AdapterConformanceMatrixTests(unittest.TestCase):
    def test_conformance_matrix_schema_and_structure(self):
        from backend.adapter_manifest import (
            collect_adapter_conformance_matrix,
            CONFORMANCE_MATRIX_SCHEMA,
        )
        matrix = collect_adapter_conformance_matrix(env={})
        self.assertEqual(matrix["schema"], CONFORMANCE_MATRIX_SCHEMA)
        self.assertGreater(matrix["adapterCount"], 0)
        self.assertFalse(matrix["unsafeOverride"])
        self.assertEqual(
            matrix["summary"]["notConfigured"],
            matrix["adapterCount"],
        )
        adapter = matrix["adapters"][0]
        self.assertIn("name", adapter)
        self.assertIn("envVars", adapter)
        self.assertIn("license", adapter)
        self.assertIn("availability", adapter)
        self.assertIn("hasPinnedHash", adapter)
        self.assertEqual(adapter["availability"], "not-configured")

    def test_conformance_matrix_detects_configured_adapter(self):
        from backend.adapter_manifest import collect_adapter_conformance_matrix
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_model = os.path.join(tmpdir, "lama_fp32.onnx")
            Path(fake_model).write_bytes(b"fake")
            matrix = collect_adapter_conformance_matrix(
                env={"VSR_LAMA_ONNX": fake_model},
            )
        lama = next(a for a in matrix["adapters"] if a["name"] == "lama-onnx")
        self.assertTrue(lama["configured"])
        self.assertTrue(lama["pathExists"])
        self.assertEqual(lama["availability"], "ready")

    def test_conformance_format_is_human_readable(self):
        from backend.adapter_manifest import (
            collect_adapter_conformance_matrix,
            format_adapter_conformance_matrix,
        )
        matrix = collect_adapter_conformance_matrix(env={})
        text = format_adapter_conformance_matrix(matrix)
        self.assertIn("Adapter Conformance Matrix", text)
        self.assertIn("lama-onnx", text)
        self.assertIn("Total:", text)



if __name__ == "__main__":
    unittest.main()
