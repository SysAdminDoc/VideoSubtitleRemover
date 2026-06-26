from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend import inpainter_registry
from backend.config import ProcessingConfig, RegisteredMode, _coerce_backend_mode
from backend.inpainters_diffusion import (
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
            (ckpt / "model_index.json").write_text("{}", encoding="utf-8")
            env = {"VSR_VACE_CKPT_DIR": str(ckpt)}

            resolved = _resolve_vace_checkpoint_dir(env, auto_fetch=False)

        self.assertEqual(resolved, ckpt)

    def test_vace_auto_fetch_uses_huggingface_snapshot_download(self):
        fake_hf = types.ModuleType("huggingface_hub")
        calls = []

        def snapshot_download(**kwargs):
            calls.append(kwargs)
            target = Path(kwargs["local_dir"])
            target.mkdir(parents=True, exist_ok=True)
            (target / "model_index.json").write_text("{}", encoding="utf-8")
            return str(target)

        fake_hf.snapshot_download = snapshot_download
        with tempfile.TemporaryDirectory() as tmpdir:
            env = {
                "APPDATA": tmpdir,
                "VSR_VACE_AUTO_FETCH": "1",
                "VSR_VACE_REVISION": "unit-test-revision",
            }
            sys.modules["huggingface_hub"] = fake_hf
            try:
                resolved = _resolve_vace_checkpoint_dir(env)
            finally:
                sys.modules.pop("huggingface_hub", None)

        self.assertIsNotNone(resolved)
        self.assertEqual(calls[0]["repo_id"], VACE_DEFAULT_REPO_ID)
        self.assertEqual(calls[0]["revision"], "unit-test-revision")

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
            os.environ["VSR_VACE_CKPT_DIR"] = str(ckpt)
            sys.modules["vace"] = fake_module
            try:
                backend = _VaceBackend(device="cpu", config=ProcessingConfig())
                frames = [np.full((16, 16, 3), 30, dtype=np.uint8) for _ in range(2)]
                masks = [np.zeros((16, 16), dtype=np.uint8) for _ in range(2)]
                masks[0][4:12, 4:12] = 255
                out = backend.inpaint(frames, masks)
            finally:
                sys.modules.pop("vace", None)

        self.assertEqual(captured["ckpt_dir"], str(ckpt))
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, (16, 16, 3))
        self.assertGreater(int(out[0][8, 8, 0]), 30)

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
            os.environ["VSR_VACE_CKPT_DIR"] = str(ckpt)
            sys.modules["vace"] = fake_pkg
            sys.modules["vace.vace_wan_inference"] = fake_script
            try:
                backend = _VaceBackend(device="cpu", config=ProcessingConfig())
                frames = [np.full((16, 16, 3), 30, dtype=np.uint8) for _ in range(2)]
                masks = [np.full((16, 16), 255, dtype=np.uint8) for _ in range(2)]
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
