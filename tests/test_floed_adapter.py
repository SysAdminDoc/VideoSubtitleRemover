from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend import inpainter_registry
from backend.config import ProcessingConfig, RegisteredMode, _coerce_backend_mode
from backend.inpainters_diffusion import _FloedBackend, maybe_register


class FloedAdapterTests(unittest.TestCase):
    def setUp(self):
        self._saved = {
            key: os.environ.pop(key, None)
            for key in (
                "VSR_FLOED",
                "VSR_FLOED_WEIGHTS",
                "VSR_FLOED_CKPT",
                "VSR_FLOED_CKPT_DIR",
                "VSR_FLOED_COMMAND",
                "VSR_FLOED_TIMEOUT",
                "VSR_ALLOW_UNVERIFIED_MODELS",
            )
        }
        inpainter_registry.unregister("floed")

    def tearDown(self):
        inpainter_registry.unregister("floed")
        for key, value in self._saved.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value

    def test_floed_registers_only_when_enabled(self):
        self.assertEqual(maybe_register(), [])
        self.assertFalse(inpainter_registry.is_registered("floed"))

        os.environ["VSR_FLOED"] = "1"
        registered = maybe_register()

        self.assertEqual(registered, ["floed"])
        self.assertTrue(inpainter_registry.is_registered("floed"))
        mode = _coerce_backend_mode("floed")
        self.assertIsInstance(mode, RegisteredMode)
        self.assertEqual(mode.value, "floed")

    def test_floed_command_adapter_blends_output_frames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            weights = root / "floed.ckpt"
            weights.write_bytes(b"stub")
            wrapper = root / "floed_wrapper.py"
            wrapper.write_text(textwrap.dedent(
                """
                import argparse
                import json
                from pathlib import Path

                import cv2
                import numpy as np

                parser = argparse.ArgumentParser()
                parser.add_argument("--input-dir", required=True)
                parser.add_argument("--mask-dir", required=True)
                parser.add_argument("--output-dir", required=True)
                parser.add_argument("--weights", required=True)
                parser.add_argument("--prompt", required=True)
                parser.add_argument("--config", required=True)
                args = parser.parse_args()

                if not Path(args.weights).is_file():
                    raise SystemExit(2)
                Path(args.output_dir).mkdir(parents=True, exist_ok=True)
                count = 0
                for frame_path in sorted(Path(args.input_dir).glob("*.png")):
                    mask_path = Path(args.mask_dir) / frame_path.name
                    if not mask_path.is_file():
                        raise SystemExit(3)
                    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                    if frame is None:
                        raise SystemExit(4)
                    cv2.imwrite(
                        str(Path(args.output_dir) / frame_path.name),
                        np.full_like(frame, 120),
                    )
                    count += 1
                if count == 0:
                    raise SystemExit(5)
                print(json.dumps({"outputDir": args.output_dir}))
                """
            ), encoding="utf-8")
            os.environ["VSR_FLOED_WEIGHTS"] = str(weights)
            os.environ["VSR_FLOED_COMMAND"] = f'"{sys.executable}" "{wrapper}"'
            os.environ["VSR_ALLOW_UNVERIFIED_MODELS"] = "1"

            backend = _FloedBackend(device="cpu", config=ProcessingConfig())
            frames = [np.full((16, 16, 3), 30, dtype=np.uint8) for _ in range(2)]
            masks = [np.full((16, 16), 255, dtype=np.uint8) for _ in range(2)]
            out = backend.inpaint(frames, masks)

        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, (16, 16, 3))
        self.assertGreater(int(out[0][8, 8, 0]), 60)


if __name__ == "__main__":
    unittest.main()
