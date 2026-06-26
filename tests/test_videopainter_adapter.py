from __future__ import annotations

import os
import sys
import tempfile
import textwrap
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
from backend.inpainters_diffusion import _VideoPainterBackend, maybe_register


class VideoPainterAdapterTests(unittest.TestCase):
    def setUp(self):
        self._saved = {
            key: os.environ.pop(key, None)
            for key in (
                "VSR_VIDEOPAINTER",
                "VSR_VIDEOPAINTER_CKPT_DIR",
                "VSR_VIDEOPAINTER_MODEL_DIR",
                "VSR_VIDEOPAINTER_WEIGHTS",
                "VSR_VIDEOPAINTER_BRANCH_DIR",
                "VSR_VIDEOPAINTER_COMMAND",
                "VSR_VIDEOPAINTER_TIMEOUT",
                "VSR_ALLOW_UNVERIFIED_MODELS",
            )
        }
        inpainter_registry.unregister("videopainter")

    def tearDown(self):
        inpainter_registry.unregister("videopainter")
        for key, value in self._saved.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value

    def test_videopainter_registers_only_when_enabled(self):
        self.assertEqual(maybe_register(), [])
        self.assertFalse(inpainter_registry.is_registered("videopainter"))

        os.environ["VSR_VIDEOPAINTER"] = "1"
        registered = maybe_register()

        self.assertEqual(registered, ["videopainter"])
        self.assertTrue(inpainter_registry.is_registered("videopainter"))
        mode = _coerce_backend_mode("videopainter")
        self.assertIsInstance(mode, RegisteredMode)
        self.assertEqual(mode.value, "videopainter")

    def test_videopainter_command_adapter_blends_output_video(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ckpt = root / "ckpt"
            branch = ckpt / "VideoPainter" / "checkpoints" / "branch"
            branch.mkdir(parents=True)
            (branch / "config.json").write_text("{}", encoding="utf-8")
            (branch / "diffusion_pytorch_model.safetensors").write_bytes(b"stub")
            base = ckpt / "CogVideoX-5b-I2V"
            base.mkdir()
            (base / "model_index.json").write_text("{}", encoding="utf-8")
            wrapper = root / "vp_wrapper.py"
            wrapper.write_text(textwrap.dedent(
                """
                import argparse
                import json
                from pathlib import Path

                import cv2
                import numpy as np

                parser = argparse.ArgumentParser()
                parser.add_argument("--input-video", required=True)
                parser.add_argument("--mask-video", required=True)
                parser.add_argument("--output-video", required=True)
                parser.add_argument("--model-dir", required=True)
                parser.add_argument("--branch-dir", required=True)
                parser.add_argument("--base-model-dir", required=True)
                parser.add_argument("--prompt", required=True)
                parser.add_argument("--config", required=True)
                args = parser.parse_args()

                for path in (args.input_video, args.mask_video, args.config):
                    if not Path(path).is_file():
                        raise SystemExit(3)
                cap = cv2.VideoCapture(args.input_video)
                if not cap.isOpened():
                    raise SystemExit(4)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 16
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 16
                fps = cap.get(cv2.CAP_PROP_FPS) or 8.0
                writer = cv2.VideoWriter(
                    args.output_video,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise SystemExit(5)
                count = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    writer.write(np.full_like(frame, 120))
                    count += 1
                cap.release()
                writer.release()
                if count == 0:
                    raise SystemExit(6)
                print(json.dumps({"outputVideo": args.output_video}))
                """
            ), encoding="utf-8")
            os.environ["VSR_VIDEOPAINTER_CKPT_DIR"] = str(ckpt)
            os.environ["VSR_VIDEOPAINTER_COMMAND"] = (
                f'"{sys.executable}" "{wrapper}"'
            )
            os.environ["VSR_ALLOW_UNVERIFIED_MODELS"] = "1"

            backend = _VideoPainterBackend(device="cpu", config=ProcessingConfig())
            frames = [np.full((16, 16, 3), 30, dtype=np.uint8) for _ in range(2)]
            masks = [np.full((16, 16), 255, dtype=np.uint8) for _ in range(2)]
            out = backend.inpaint(frames, masks)

        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, (16, 16, 3))
        self.assertGreater(int(out[0][8, 8, 0]), 60)


if __name__ == "__main__":
    unittest.main()
