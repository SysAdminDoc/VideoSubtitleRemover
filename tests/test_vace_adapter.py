from __future__ import annotations

import os
import hashlib
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend import inpainter_registry
from backend.config import ProcessingConfig, RegisteredMode, _coerce_backend_mode
from backend.inpainters_diffusion import (
    VACE_DEFAULT_REVISION,
    VACE_DEFAULT_REPO_ID,
    _VaceBackend,
    _resolve_vace_checkpoint_dir,
    maybe_register,
)


class VaceAdapterTests(unittest.TestCase):
    def setUp(self):
        self._saved = {
            key: os.environ.pop(key, None)
            for key in (
                "VSR_VACE",
                "VSR_VACE_CKPT_DIR",
                "VSR_VACE_MODEL_DIR",
                "VSR_VACE_WEIGHTS",
                "VSR_VACE_AUTO_FETCH",
                "VSR_VACE_REPO_ID",
                "VSR_VACE_REVISION",
                "VSR_ALLOW_UNVERIFIED_MODELS",
            )
        }
        inpainter_registry.unregister("vace")

    def tearDown(self):
        inpainter_registry.unregister("vace")
        for key, value in self._saved.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value

    def test_vace_registers_only_when_enabled(self):
        self.assertEqual(maybe_register(), [])
        self.assertFalse(inpainter_registry.is_registered("vace"))

        os.environ["VSR_VACE"] = "1"
        registered = maybe_register()

        self.assertEqual(registered, ["vace"])
        self.assertTrue(inpainter_registry.is_registered("vace"))
        mode = _coerce_backend_mode("vace")
        self.assertIsInstance(mode, RegisteredMode)
        self.assertEqual(mode.value, "vace")

    def test_vace_uses_configured_checkpoint_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "Wan2.1-VACE-1.3B"
            ckpt.mkdir()
            (ckpt / "model.bin").write_bytes(b"verified model")
            env = {
                "APPDATA": tmpdir,
                "VSR_VACE_CKPT_DIR": str(ckpt),
            }
            with self._fake_manifest({"model.bin": b"verified model"}):
                resolved = _resolve_vace_checkpoint_dir(env, auto_fetch=False)

        self.assertEqual(resolved, ckpt)

    def _fake_manifest(self, files):
        from backend import adapter_manifest as manifest

        entry = replace(
            manifest.get_manifest_entry("vace-wan13b"),
            expected_filenames=tuple(files),
            sha256={
                name: hashlib.sha256(content).hexdigest()
                for name, content in files.items()
            },
            repository=VACE_DEFAULT_REPO_ID,
            revision=VACE_DEFAULT_REVISION,
        )
        return mock.patch.dict(
            manifest.ADAPTER_MANIFEST,
            {"vace-wan13b": entry},
            clear=False,
        )

    def _fake_huggingface(self, files, calls):
        fake_hf = types.ModuleType("huggingface_hub")

        def snapshot_download(**kwargs):
            calls.append(kwargs)
            target = Path(kwargs["local_dir"])
            target.mkdir(parents=True, exist_ok=True)
            for relative, content in files.items():
                path = target / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            return str(target)

        fake_hf.snapshot_download = snapshot_download
        return fake_hf

    def test_vace_auto_fetch_uses_allowlisted_commit_and_files(self):
        files = {"model.bin": b"verified model"}
        calls = []
        fake_hf = self._fake_huggingface(files, calls)
        with tempfile.TemporaryDirectory() as tmpdir:
            env = {
                "APPDATA": tmpdir,
                "VSR_VACE_AUTO_FETCH": "1",
            }
            sys.modules["huggingface_hub"] = fake_hf
            try:
                with self._fake_manifest(files):
                    resolved = _resolve_vace_checkpoint_dir(env)
                    from backend.adapter_manifest import load_adapter_provenance
                    provenance = load_adapter_provenance("vace-wan13b", env)
            finally:
                sys.modules.pop("huggingface_hub", None)

        self.assertIsNotNone(resolved)
        self.assertEqual(calls[0]["repo_id"], VACE_DEFAULT_REPO_ID)
        self.assertEqual(calls[0]["revision"], VACE_DEFAULT_REVISION)
        self.assertEqual(calls[0]["allow_patterns"], ["model.bin"])
        self.assertEqual(provenance["repository"], VACE_DEFAULT_REPO_ID)
        self.assertEqual(provenance["commit"], VACE_DEFAULT_REVISION)
        self.assertEqual(provenance["cachePath"], str(resolved))
        self.assertTrue(provenance["verified"])
        self.assertFalse(provenance["unsafeOverride"])

    def test_vace_auto_fetch_blocks_mutable_or_unallowlisted_identity(self):
        calls = []
        fake_hf = self._fake_huggingface({"model.bin": b"model"}, calls)
        with tempfile.TemporaryDirectory() as tmpdir:
            env = {
                "APPDATA": tmpdir,
                "VSR_VACE_AUTO_FETCH": "1",
                "VSR_VACE_REPO_ID": "attacker/model",
                "VSR_VACE_REVISION": "main",
            }
            sys.modules["huggingface_hub"] = fake_hf
            try:
                with self._fake_manifest({"model.bin": b"model"}):
                    resolved = _resolve_vace_checkpoint_dir(env)
            finally:
                sys.modules.pop("huggingface_hub", None)

        self.assertIsNone(resolved)
        self.assertEqual(calls, [])

    def test_vace_auto_fetch_rejects_mutable_revision_even_with_override(self):
        calls = []
        files = {"model.bin": b"model"}
        fake_hf = self._fake_huggingface(files, calls)
        with tempfile.TemporaryDirectory() as tmpdir:
            env = {
                "APPDATA": tmpdir,
                "VSR_VACE_AUTO_FETCH": "1",
                "VSR_VACE_REVISION": "main",
                "VSR_ALLOW_UNVERIFIED_MODELS": "1",
            }
            sys.modules["huggingface_hub"] = fake_hf
            try:
                with self._fake_manifest(files):
                    resolved = _resolve_vace_checkpoint_dir(env)
            finally:
                sys.modules.pop("huggingface_hub", None)

        self.assertIsNone(resolved)
        self.assertEqual(calls, [])

    def test_vace_auto_fetch_missing_artifact_fails_before_use(self):
        approved = {"one.bin": b"one", "two.bin": b"two"}
        calls = []
        fake_hf = self._fake_huggingface({"one.bin": b"one"}, calls)
        with tempfile.TemporaryDirectory() as tmpdir:
            env = {"APPDATA": tmpdir, "VSR_VACE_AUTO_FETCH": "1"}
            sys.modules["huggingface_hub"] = fake_hf
            try:
                with self._fake_manifest(approved):
                    resolved = _resolve_vace_checkpoint_dir(env)
            finally:
                sys.modules.pop("huggingface_hub", None)

        self.assertIsNone(resolved)
        self.assertEqual(len(calls), 1)

    def test_vace_auto_fetch_hash_mismatch_fails_without_override(self):
        calls = []
        fake_hf = self._fake_huggingface({"model.bin": b"tampered"}, calls)
        with tempfile.TemporaryDirectory() as tmpdir:
            env = {"APPDATA": tmpdir, "VSR_VACE_AUTO_FETCH": "1"}
            sys.modules["huggingface_hub"] = fake_hf
            try:
                with self._fake_manifest({"model.bin": b"approved"}):
                    resolved = _resolve_vace_checkpoint_dir(env)
            finally:
                sys.modules.pop("huggingface_hub", None)

        self.assertIsNone(resolved)
        self.assertEqual(len(calls), 1)

    def test_vace_reuses_verified_app_cache_without_network(self):
        files = {"model.bin": b"verified model"}
        calls = []
        fake_hf = self._fake_huggingface(files, calls)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = (
                Path(tmpdir)
                / "VideoSubtitleRemoverPro"
                / "models"
                / "vace"
                / "Wan2.1-VACE-1.3B"
            )
            root.mkdir(parents=True)
            (root / "model.bin").write_bytes(files["model.bin"])
            env = {"APPDATA": tmpdir, "VSR_VACE_AUTO_FETCH": "1"}
            sys.modules["huggingface_hub"] = fake_hf
            try:
                with self._fake_manifest(files):
                    resolved = _resolve_vace_checkpoint_dir(env)
            finally:
                sys.modules.pop("huggingface_hub", None)

        self.assertEqual(resolved, root)
        self.assertEqual(calls, [])

    def test_vace_production_manifest_is_complete_and_immutable(self):
        from backend.adapter_manifest import get_manifest_entry

        entry = get_manifest_entry("vace-wan13b")

        self.assertEqual(entry.repository, VACE_DEFAULT_REPO_ID)
        self.assertEqual(entry.revision, VACE_DEFAULT_REVISION)
        self.assertEqual(len(entry.revision), 40)
        self.assertEqual(set(entry.sha256), set(entry.expected_filenames))
        self.assertTrue(all(
            len(value) == 64
            and all(char in "0123456789abcdef" for char in value)
            for value in entry.sha256.values()
        ))

    def test_vace_unsafe_identity_override_is_persisted_in_provenance(self):
        from backend.adapter_manifest import load_adapter_provenance

        files = {"model.bin": b"different model"}
        calls = []
        fake_hf = self._fake_huggingface(files, calls)
        with tempfile.TemporaryDirectory() as tmpdir:
            env = {
                "APPDATA": tmpdir,
                "VSR_VACE_AUTO_FETCH": "1",
                "VSR_VACE_REPO_ID": "reviewed-fork/model",
                "VSR_VACE_REVISION": "d" * 40,
                "VSR_ALLOW_UNVERIFIED_MODELS": "1",
            }
            sys.modules["huggingface_hub"] = fake_hf
            try:
                with self._fake_manifest({"model.bin": b"approved model"}):
                    resolved = _resolve_vace_checkpoint_dir(env)
                    provenance = load_adapter_provenance("vace-wan13b", env)
            finally:
                sys.modules.pop("huggingface_hub", None)

        self.assertIsNotNone(resolved)
        self.assertEqual(calls[0]["repo_id"], "reviewed-fork/model")
        self.assertEqual(calls[0]["revision"], "d" * 40)
        self.assertTrue(provenance["unsafeOverride"])
        self.assertEqual(provenance["repository"], "reviewed-fork/model")
        self.assertEqual(provenance["commit"], "d" * 40)
        self.assertEqual(provenance["files"][0]["hashStatus"], "mismatch")

    def test_vace_loads_fake_local_package_and_blends_output(self):
        fake_module = types.ModuleType("vace")
        captured = {}

        class FakeVACE:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def mv2v(self, frames=None, masks=None, prompt=None):
                return [np.full_like(frame, 90) for frame in frames]

        fake_module.VACE = FakeVACE
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "Wan2.1-VACE-1.3B"
            ckpt.mkdir()
            (ckpt / "model_index.json").write_text("{}", encoding="utf-8")
            sys.modules["vace"] = fake_module
            try:
                with self._fake_manifest({"model_index.json": b"{}"}), mock.patch.dict(
                    os.environ,
                    {"APPDATA": tmpdir, "VSR_VACE_CKPT_DIR": str(ckpt)},
                    clear=False,
                ):
                    backend = _VaceBackend(device="cpu", config=ProcessingConfig())
                    frames = [
                        np.full((16, 16, 3), 30, dtype=np.uint8)
                        for _ in range(2)
                    ]
                    masks = [
                        np.zeros((16, 16), dtype=np.uint8)
                        for _ in range(2)
                    ]
                    masks[0][4:12, 4:12] = 255
                    out = backend.inpaint(frames, masks)
            finally:
                sys.modules.pop("vace", None)

        self.assertEqual(captured["ckpt_dir"], str(ckpt))
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, (16, 16, 3))
        self.assertGreater(int(out[0][8, 8, 0]), 30)

    def test_vace_never_constructs_without_verified_checkpoint_argument(self):
        fake_module = types.ModuleType("vace")
        constructor_calls = []

        class PathlessVACE:
            def __init__(self, device=None):
                constructor_calls.append(device)

        fake_module.VACE = PathlessVACE
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "Wan2.1-VACE-1.3B"
            ckpt.mkdir()
            (ckpt / "model.bin").write_bytes(b"verified model")
            sys.modules["vace"] = fake_module
            try:
                with self._fake_manifest({"model.bin": b"verified model"}), mock.patch.dict(
                    os.environ,
                    {"APPDATA": tmpdir, "VSR_VACE_CKPT_DIR": str(ckpt)},
                    clear=False,
                ):
                    backend = _VaceBackend(device="cpu", config=ProcessingConfig())
                    loaded = backend._load()
            finally:
                sys.modules.pop("vace", None)

        self.assertIsNone(loaded)
        self.assertEqual(constructor_calls, [])

    def test_vace_uses_upstream_wan_inference_entrypoint(self):
        fake_pkg = types.ModuleType("vace")
        fake_pkg.__path__ = []
        fake_script = types.ModuleType("vace.vace_wan_inference")
        captured = {}

        def main(args):
            captured.update(args)
            self.assertTrue(Path(args["src_video"]).is_file())
            self.assertTrue(Path(args["src_mask"]).is_file())
            self.assertEqual(args["frame_num"], 5)
            cap = cv2.VideoCapture(args["src_video"])
            try:
                self.assertTrue(cap.isOpened())
                self.assertGreaterEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
            finally:
                cap.release()
            out_path = Path(args["save_file"])
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                16.0,
                (16, 16),
            )
            self.assertTrue(writer.isOpened())
            try:
                for _idx in range(args["frame_num"]):
                    writer.write(np.full((16, 16, 3), 120, dtype=np.uint8))
            finally:
                writer.release()
            return {"out_video": str(out_path)}

        fake_script.main = main
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "Wan2.1-VACE-1.3B"
            ckpt.mkdir()
            (ckpt / "model_index.json").write_text("{}", encoding="utf-8")
            sys.modules["vace"] = fake_pkg
            sys.modules["vace.vace_wan_inference"] = fake_script
            try:
                with self._fake_manifest({"model_index.json": b"{}"}), mock.patch.dict(
                    os.environ,
                    {"APPDATA": tmpdir, "VSR_VACE_CKPT_DIR": str(ckpt)},
                    clear=False,
                ):
                    backend = _VaceBackend(device="cpu", config=ProcessingConfig())
                    frames = [
                        np.full((16, 16, 3), 30, dtype=np.uint8)
                        for _ in range(2)
                    ]
                    masks = [
                        np.full((16, 16), 255, dtype=np.uint8)
                        for _ in range(2)
                    ]
                    out = backend.inpaint(frames, masks)
            finally:
                sys.modules.pop("vace", None)
                sys.modules.pop("vace.vace_wan_inference", None)

        self.assertEqual(captured["ckpt_dir"], str(ckpt))
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, (16, 16, 3))
        self.assertGreater(int(out[0][8, 8, 0]), 60)


if __name__ == "__main__":
    unittest.main()
