from pathlib import Path
from types import SimpleNamespace
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from backend import opencv_ocr


class OpenCVDnnOcrTests(unittest.TestCase):
    def _model_root(self, root: Path) -> Path:
        model_root = root / "models"
        model_root.mkdir()
        for filename in opencv_ocr.MODEL_FILENAMES.values():
            (model_root / filename).write_bytes(b"model")
        return root

    def test_status_requires_opencv5_assets_and_fixed_libpng(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._model_root(Path(tmp))
            status = opencv_ocr.collect_opencv_dnn_ocr_status(
                rapidocr_root=root,
                rapidocr_version="3.9.1",
                libpng={
                    "vulnerable": False,
                    "libpng_version": "1.6.57",
                },
                cv_module=SimpleNamespace(
                    __version__="5.0.0",
                    dnn=SimpleNamespace(readNetFromONNX=lambda _path: None),
                ),
            )
        self.assertTrue(status["eligible"])
        self.assertEqual(status["schema"], opencv_ocr.OPENCV_DNN_OCR_SCHEMA)
        self.assertTrue(all(
            item["present"] for item in status["models"].values()
        ))

    def test_status_rejects_unproven_libpng(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._model_root(Path(tmp))
            status = opencv_ocr.collect_opencv_dnn_ocr_status(
                rapidocr_root=root,
                libpng={
                    "vulnerable": True,
                    "libpng_version": "1.6.53",
                },
                cv_module=SimpleNamespace(
                    __version__="5.0.0",
                    dnn=SimpleNamespace(readNetFromONNX=lambda _path: None),
                ),
            )
        self.assertFalse(status["eligible"])
        self.assertTrue(any("libpng" in error for error in status["errors"]))

    def test_session_runs_contiguous_float_input_and_exposes_metadata(self):
        class FakeNet:
            def __init__(self):
                self.input = None

            def setInput(self, value):
                self.input = value

            def forward(self):
                return np.array([[[0.25, 0.75]]], dtype=np.float32)

        fake_net = FakeNet()
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "rec.onnx"
            model.write_bytes(b"model")
            with mock.patch.object(
                opencv_ocr,
                "read_onnx_metadata_props",
                return_value={"character": "blank\nA"},
            ):
                session = opencv_ocr.OpenCVDnnSession(
                    {"model_path": model},
                    cv_module=SimpleNamespace(
                        dnn=SimpleNamespace(
                            readNetFromONNX=lambda _path: fake_net,
                        ),
                    ),
                )
                output = session(np.zeros((1, 3, 2, 2), dtype=np.float64))

        self.assertEqual(fake_net.input.dtype, np.float32)
        self.assertTrue(fake_net.input.flags.c_contiguous)
        self.assertEqual(output.shape, (1, 1, 2))
        self.assertTrue(session.have_key())
        self.assertEqual(session.get_character_list(), ["blank", "A"])

    def test_detector_uses_forced_opencv_provider(self):
        from backend import detection

        engine = SimpleNamespace()
        with mock.patch.dict(
            os.environ,
            {"VSR_RAPIDOCR_ENGINE": "opencv"},
            clear=False,
        ), mock.patch(
            "backend.opencv_ocr.build_opencv_dnn_rapidocr",
            return_value=engine,
        ):
            detector = detection.SubtitleDetector(device="cpu")

        self.assertIs(detector._rapid_model, engine)
        self.assertEqual(detector._engine_name, "RapidOCR (OpenCV 5 DNN)")


if __name__ == "__main__":
    unittest.main()
