"""Tests for the OpenCV 5 DNN LaMa inpainting backend and fallback chain.

Mocks cv2.__version__ and model discovery to verify that the four-tier
priority chain (ONNX Runtime > OpenCV 5 DNN > PyTorch > cv2) activates
the correct backend and degrades gracefully.
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np


def _make_frame(h=64, w=64):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _make_mask(h=64, w=64):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[20:40, 20:40] = 255
    return mask


class TestOpenCVVersionDetection(unittest.TestCase):
    def test_opencv5_detected(self):
        with patch.object(cv2, "__version__", "5.0.0"):
            from backend.inpainters.lama import _opencv5_available
            self.assertTrue(_opencv5_available())

    def test_opencv5_prerelease_detected(self):
        with patch.object(cv2, "__version__", "5.1.0-dev"):
            from backend.inpainters.lama import _opencv5_available
            self.assertTrue(_opencv5_available())

    def test_opencv4_not_detected(self):
        with patch.object(cv2, "__version__", "4.12.0"):
            from backend.inpainters.lama import _opencv5_available
            self.assertFalse(_opencv5_available())

    def test_version_tuple_parses_cleanly(self):
        from backend.inpainters.lama import _opencv_version_tuple
        with patch.object(cv2, "__version__", "5.2.1"):
            self.assertEqual(_opencv_version_tuple(), (5, 2, 1))
        with patch.object(cv2, "__version__", "4.12.0"):
            self.assertEqual(_opencv_version_tuple(), (4, 12, 0))
        with patch.object(cv2, "__version__", "5.0.0-pre"):
            self.assertEqual(_opencv_version_tuple(), (5, 0, 0))


class TestDnnPathActivation(unittest.TestCase):
    """Verify the cv2.dnn path activates when OpenCV >= 5.0 and a model
    file is found, and that it falls back correctly otherwise."""

    @patch("backend.inpainters.lama._find_lama_onnx_weight", return_value=None)
    @patch("backend.inpainters.lama._opencv5_available", return_value=True)
    @patch("backend.inpainters.lama._find_opencv_lama_weight",
           return_value="/fake/inpainting_lama_2025jan.onnx")
    @patch("backend.inpainters.lama._try_cv2dnn_net")
    def test_dnn_backend_selected_when_available(
        self, mock_try_net, mock_find, mock_cv5, mock_onnx
    ):
        mock_net = MagicMock()
        mock_try_net.return_value = mock_net

        from backend.inpainters.lama import LAMAInpainter
        inpainter = LAMAInpainter(device="cpu")

        self.assertIsNotNone(inpainter._dnn_net)
        self.assertIsNone(inpainter._onnx_session)
        self.assertIsNone(inpainter._lama)
        self.assertEqual(inpainter.backend_name, "OpenCV DNN")

    @patch("backend.inpainters.lama._find_lama_onnx_weight", return_value=None)
    @patch("backend.inpainters.lama._opencv5_available", return_value=False)
    def test_dnn_skipped_on_opencv4(self, mock_cv5, mock_onnx):
        from backend.inpainters.lama import LAMAInpainter
        inpainter = LAMAInpainter(device="cpu")

        self.assertIsNone(inpainter._dnn_net)
        # Falls through to PyTorch or cv2 depending on install
        self.assertIn(
            inpainter.backend_name,
            ("cv2", "PyTorch (simple-lama-inpainting)")
        )

    @patch("backend.inpainters.lama._find_lama_onnx_weight", return_value=None)
    @patch("backend.inpainters.lama._opencv5_available", return_value=True)
    @patch("backend.inpainters.lama._find_opencv_lama_weight",
           return_value=None)
    def test_dnn_skipped_when_no_model(self, mock_find, mock_cv5, mock_onnx):
        from backend.inpainters.lama import LAMAInpainter
        inpainter = LAMAInpainter(device="cpu")

        self.assertIsNone(inpainter._dnn_net)

    @patch("backend.inpainters.lama._find_lama_onnx_weight", return_value=None)
    @patch("backend.inpainters.lama._opencv5_available", return_value=True)
    @patch("backend.inpainters.lama._find_opencv_lama_weight",
           return_value="/fake/model.onnx")
    @patch("backend.inpainters.lama._try_cv2dnn_net", return_value=None)
    def test_dnn_skipped_when_net_load_fails(
        self, mock_try, mock_find, mock_cv5, mock_onnx
    ):
        from backend.inpainters.lama import LAMAInpainter
        inpainter = LAMAInpainter(device="cpu")

        self.assertIsNone(inpainter._dnn_net)


class TestDnnInference(unittest.TestCase):
    """Test that the cv2.dnn inference path produces valid output shapes."""

    def _make_inpainter_with_mock_net(self):
        mock_net = MagicMock()

        def mock_forward():
            return np.random.rand(1, 3, 64, 64).astype(np.float32)

        mock_net.forward = mock_forward
        mock_net.setInput = MagicMock()

        from backend.inpainters.lama import LAMAInpainter
        with patch("backend.inpainters.lama._find_lama_onnx_weight",
                   return_value=None), \
             patch("backend.inpainters.lama._opencv5_available",
                   return_value=True), \
             patch("backend.inpainters.lama._find_opencv_lama_weight",
                   return_value="/fake/model.onnx"), \
             patch("backend.inpainters.lama._try_cv2dnn_net",
                   return_value=mock_net):
            inpainter = LAMAInpainter(device="cpu")
        return inpainter

    def test_single_frame_inpaint_returns_correct_shape(self):
        inpainter = self._make_inpainter_with_mock_net()
        frame = _make_frame(64, 64)
        mask = _make_mask(64, 64)
        results = inpainter.inpaint([frame], [mask])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].shape, frame.shape)

    def test_empty_mask_returns_copy(self):
        inpainter = self._make_inpainter_with_mock_net()
        frame = _make_frame(64, 64)
        mask = np.zeros((64, 64), dtype=np.uint8)
        results = inpainter.inpaint([frame], [mask])
        self.assertEqual(len(results), 1)
        np.testing.assert_array_equal(results[0], frame)

    def test_multi_frame_batch(self):
        inpainter = self._make_inpainter_with_mock_net()
        frames = [_make_frame(64, 64) for _ in range(3)]
        masks = [_make_mask(64, 64) for _ in range(3)]
        results = inpainter.inpaint(frames, masks)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertEqual(r.shape, (64, 64, 3))


class TestTryCv2DnnNet(unittest.TestCase):
    """Test _try_cv2dnn_net guard logic."""

    @patch("backend.inpainters.lama._opencv5_available", return_value=False)
    def test_returns_none_on_opencv4(self, mock_cv5):
        from backend.inpainters.lama import _try_cv2dnn_net
        result = _try_cv2dnn_net("/fake/model.onnx", "cpu")
        self.assertIsNone(result)

    @patch("backend.inpainters.lama._opencv5_available", return_value=True)
    @patch.object(cv2.dnn, "readNetFromONNX", side_effect=Exception("fail"))
    def test_returns_none_on_load_failure(self, mock_read, mock_cv5):
        from backend.inpainters.lama import _try_cv2dnn_net
        with patch("backend.inpainters.lama.Path") as mock_path:
            mock_path.return_value.name = "model.onnx"
            result = _try_cv2dnn_net("/fake/model.onnx", "cpu")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
