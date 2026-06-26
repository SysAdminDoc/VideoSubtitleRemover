from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend import inpainter_registry
from backend.config import ProcessingConfig, RegisteredMode, _coerce_backend_mode
from backend.inpainters_diffusion import _VoidBackend, maybe_register


class VoidAdapterTests(unittest.TestCase):
    def setUp(self):
        self._saved = {
            key: os.environ.pop(key, None)
            for key in (
                "VSR_VOID",
                "VSR_VOID_WEIGHTS",
                "VSR_VOID_PASS1",
                "VSR_VOID_PASS2",
                "VSR_ALLOW_UNVERIFIED_MODELS",
            )
        }
        inpainter_registry.unregister("void")

    def tearDown(self):
        inpainter_registry.unregister("void")
        for key, value in self._saved.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value

    def test_void_registers_only_when_env_enabled(self):
        self.assertEqual(maybe_register(), [])
        self.assertFalse(inpainter_registry.is_registered("void"))

        os.environ["VSR_VOID"] = "1"
        registered = maybe_register()

        self.assertEqual(registered, ["void"])
        self.assertTrue(inpainter_registry.is_registered("void"))
        mode = _coerce_backend_mode("void")
        self.assertIsInstance(mode, RegisteredMode)
        self.assertEqual(mode.value, "void")

    def test_void_falls_back_when_weights_are_absent(self):
        cfg = ProcessingConfig(tbe_enable=True)
        backend = _VoidBackend(device="cpu", config=cfg)
        frames = [np.full((16, 16, 3), 60, dtype=np.uint8) for _ in range(3)]
        masks = [np.zeros((16, 16), dtype=np.uint8) for _ in range(3)]
        masks[1][4:8, 4:8] = 255

        out = backend.inpaint(frames, masks)

        self.assertEqual(len(out), 3)
        for frame in out:
            self.assertEqual(frame.shape, (16, 16, 3))

    def test_void_manifest_rejects_unverified_weights_before_import(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "void_pass1.safetensors"
            path.write_bytes(b"unverified void weights")
            os.environ["VSR_VOID_WEIGHTS"] = str(path)
            backend = _VoidBackend(device="cpu", config=ProcessingConfig())

            loaded = backend._load()

        self.assertIsNone(loaded)

    def test_void_can_load_reviewed_local_package_with_explicit_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "void_pass1.safetensors"
            path.write_bytes(b"reviewed local weights for test")
            os.environ["VSR_VOID_WEIGHTS"] = str(path)
            os.environ["VSR_ALLOW_UNVERIFIED_MODELS"] = "1"
            fake = types.ModuleType("void_model")
            fake.inpaint = lambda frames, masks, **kwargs: [
                frame.copy() for frame in frames
            ]
            sys.modules["void_model"] = fake
            try:
                backend = _VoidBackend(device="cpu", config=ProcessingConfig())
                loaded = backend._load()
            finally:
                sys.modules.pop("void_model", None)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["weights"], [str(path)])


if __name__ == "__main__":
    unittest.main()
