from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class Sam2RefinementTests(unittest.TestCase):
    def test_sam2_replaces_coarse_box_with_prompted_mask(self):
        from backend import segmentation as seg

        calls = []

        class FakePredictor:
            def set_image(self, rgb):
                calls.append(("set_image", rgb.shape))

            def predict(self, **kwargs):
                calls.append(("predict", kwargs))
                mask = np.zeros((32, 32), dtype=np.uint8)
                mask[13:17, 13:17] = 1
                return np.asarray([mask]), np.asarray([0.99]), None

        saved = dict(seg._SAM2_STATE)
        try:
            seg._SAM2_STATE.update({"probed": True, "predictor": FakePredictor()})
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            base = np.zeros((32, 32), dtype=np.uint8)
            base[10:20, 10:20] = 255
            base[1:3, 1:3] = 255

            out = seg.refine_mask_with_sam2(frame, [(10, 10, 20, 20)], base)
        finally:
            seg._SAM2_STATE.clear()
            seg._SAM2_STATE.update(saved)

        self.assertEqual(int(out[1:3, 1:3].min()), 255)
        self.assertEqual(int(out[10:13, 10:20].max()), 0)
        self.assertEqual(int(out[13:17, 13:17].min()), 255)
        predict_kwargs = calls[1][1]
        self.assertIn("point_coords", predict_kwargs)
        self.assertIn("point_labels", predict_kwargs)


class AutoDilationTests(unittest.TestCase):
    def test_outline_falloff_gets_more_radius_than_plain_glyph(self):
        import cv2
        from backend.segmentation import estimate_auto_dilation_radius

        # OpenCV 5 saturates putText thickness at about two pixels, so a
        # wider dark stroke under the glyph renders no halo at all. Build the
        # outline by dilating the glyph mask instead, which is both faithful
        # to a real outlined subtitle and independent of the font renderer.
        glyph = np.zeros((100, 200), dtype=np.uint8)
        cv2.putText(
            glyph, "TEST", (40, 60), cv2.FONT_HERSHEY_SIMPLEX,
            1.5, 255, 3, cv2.LINE_AA,
        )
        halo = cv2.dilate(glyph, np.ones((7, 7), np.uint8))

        plain = np.full((100, 200, 3), 80, dtype=np.uint8)
        plain[glyph > 40] = (240, 240, 240)
        outlined = np.full((100, 200, 3), 80, dtype=np.uint8)
        outlined[halo > 40] = (20, 20, 20)
        outlined[glyph > 40] = (240, 240, 240)
        box = (35, 25, 170, 70)

        plain_radius = estimate_auto_dilation_radius(plain, box)
        outline_radius = estimate_auto_dilation_radius(outlined, box)

        self.assertGreaterEqual(plain_radius, 0)
        self.assertGreater(outline_radius, plain_radius)
        self.assertLessEqual(outline_radius, 20)

    def test_soft_dilation_has_a_continuous_distance_edge(self):
        from backend.segmentation import soft_dilate_mask

        base = np.zeros((20, 20), dtype=np.uint8)
        base[8:12, 8:12] = 255
        result = soft_dilate_mask(base, 4)

        self.assertEqual(int(result[8, 8]), 255)
        self.assertGreater(int(result[7, 8]), int(result[6, 8]))
        self.assertGreater(int(result[6, 8]), int(result[5, 8]))
        self.assertEqual(int(result[3, 8]), 0)

    def test_processor_auto_mask_is_soft_but_manual_mask_stays_binary(self):
        import cv2
        from backend import processor

        frame = np.full((100, 200, 3), 80, dtype=np.uint8)
        cv2.putText(
            frame, "TEST", (40, 60), cv2.FONT_HERSHEY_SIMPLEX,
            1.5, (20, 20, 20), 8, cv2.LINE_AA,
        )
        cv2.putText(
            frame, "TEST", (40, 60), cv2.FONT_HERSHEY_SIMPLEX,
            1.5, (240, 240, 240), 3, cv2.LINE_AA,
        )
        box = (35, 25, 170, 70)
        remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)

        remover.config = processor.ProcessingConfig(auto_dilate_enable=True)
        adaptive = remover._create_mask(frame.shape, [box], frame=frame)
        self.assertTrue(np.any((adaptive > 0) & (adaptive < 255)))

        remover.config = processor.ProcessingConfig(auto_dilate_enable=False)
        manual = remover._create_mask(frame.shape, [box], frame=frame)
        expected = np.zeros(frame.shape[:2], dtype=np.uint8)
        expected[20:75, 30:175] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        expected = cv2.dilate(expected, kernel, iterations=1)
        np.testing.assert_array_equal(manual, expected)


class MatAnyoneRefinementTests(unittest.TestCase):
    def test_matte_frame_normalizes_float_alpha(self):
        from backend import segmentation as seg

        class FakeModel:
            def matte(self, frame, hint_mask):
                alpha = np.zeros((16, 16), dtype=np.float32)
                alpha[4:8, 5:9] = 0.75
                return alpha

        saved = dict(seg._MATANYONE_STATE)
        try:
            seg._MATANYONE_STATE.update({"probed": True, "model": FakeModel()})
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            hint = np.zeros((32, 32), dtype=np.uint8)
            hint[8:16, 10:18] = 255

            out = seg.matte_frame(frame, hint)
        finally:
            seg._MATANYONE_STATE.clear()
            seg._MATANYONE_STATE.update(saved)

        self.assertIsNotNone(out)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, (32, 32))
        self.assertGreaterEqual(int(out.max()), 190)

    def test_refine_masks_preserves_empty_hints(self):
        from backend import segmentation as seg

        class FakeModel:
            def matte_frames(self, frames, masks):
                return [np.full(frame.shape[:2], 255, dtype=np.uint8) for frame in frames]

        saved = dict(seg._MATANYONE_STATE)
        try:
            seg._MATANYONE_STATE.update({"probed": True, "model": FakeModel()})
            frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(2)]
            empty = np.zeros((16, 16), dtype=np.uint8)
            hint = np.zeros((16, 16), dtype=np.uint8)
            hint[4:8, 4:8] = 255

            out = seg.refine_masks_with_matanyone(frames, [empty, hint])
        finally:
            seg._MATANYONE_STATE.clear()
            seg._MATANYONE_STATE.update(saved)

        self.assertEqual(int(out[0].max()), 0)
        self.assertEqual(int(out[1].min()), 255)

    def test_processor_batch_refinement_uses_matanyone_flag(self):
        from backend import processor
        from backend import segmentation as seg

        class FakeModel:
            def matte_frames(self, frames, masks):
                alpha = np.zeros(frames[0].shape[:2], dtype=np.uint8)
                alpha[6:10, 6:10] = 255
                return [alpha]

        remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        remover.config = processor.ProcessingConfig(
            matanyone_refine=True,
            device="cpu",
        )
        saved = dict(seg._MATANYONE_STATE)
        try:
            seg._MATANYONE_STATE.update({"probed": True, "model": FakeModel()})
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            mask = np.zeros((16, 16), dtype=np.uint8)
            mask[2:14, 2:14] = 255

            [out] = remover._refine_masks_with_matanyone([frame], [mask])
        finally:
            seg._MATANYONE_STATE.clear()
            seg._MATANYONE_STATE.update(saved)

        self.assertEqual(int(out[2:6, 2:14].max()), 0)
        self.assertEqual(int(out[6:10, 6:10].min()), 255)


class CoTrackerPropagationTests(unittest.TestCase):
    def test_propagates_empty_masks_from_anchor_translation(self):
        from unittest import mock
        from backend import segmentation as seg

        def fake_tracks(frames, points, **_kwargs):
            tracks = [
                points,
                [(x + 3, y + 2) for x, y in points],
                [(x + 6, y + 4) for x, y in points],
            ]
            visibility = [[1.0] * len(points) for _ in tracks]
            return tracks, visibility

        frames = [np.zeros((24, 24, 3), dtype=np.uint8) for _ in range(3)]
        anchor = np.zeros((24, 24), dtype=np.uint8)
        anchor[4:8, 4:8] = 255
        empty = np.zeros((24, 24), dtype=np.uint8)
        existing = np.zeros((24, 24), dtype=np.uint8)
        existing[12:16, 12:16] = 255

        with mock.patch.object(seg, "track_points_with_visibility", side_effect=fake_tracks):
            out = seg.propagate_masks_with_cotracker(
                frames,
                [anchor, empty, existing],
            )

        self.assertEqual(int(out[0][4:8, 4:8].min()), 255)
        self.assertEqual(int(out[1][6:10, 7:11].min()), 255)
        self.assertEqual(int(out[2][12:16, 12:16].min()), 255)
        self.assertEqual(int(out[2][16:20, 18:22].max()), 0)

    def test_processor_uses_cotracker_flag(self):
        from unittest import mock
        from backend import processor

        remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        remover.config = processor.ProcessingConfig(
            cotracker_propagate=True,
            device="cpu",
        )
        frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(2)]
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[2:6, 2:6] = 255
        empty = np.zeros((16, 16), dtype=np.uint8)
        propagated = np.zeros((16, 16), dtype=np.uint8)
        propagated[4:8, 4:8] = 255

        with mock.patch(
            "backend.segmentation.propagate_masks_with_cotracker",
            return_value=[mask, propagated],
        ) as mocked:
            out = remover._propagate_masks_with_cotracker(frames, [mask, empty])

        mocked.assert_called_once()
        self.assertEqual(int(out[1][4:8, 4:8].min()), 255)


if __name__ == "__main__":
    unittest.main()


class MatAnyoneAlphaGapTests(unittest.TestCase):
    """RM-136: a subtitle-gap (fully transparent) alpha frame must not
    discard the whole MatAnyone refinement."""

    @staticmethod
    def _write_alpha_video(path, values, size=(32, 48)):
        import cv2

        height, width = size
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (width, height))
        if not writer.isOpened():
            return False
        try:
            for value in values:
                frame = np.full((height, width, 3), value, dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()
        return True

    def test_empty_frame_falls_back_to_the_hint_instead_of_none(self):
        import tempfile

        from backend import segmentation as seg

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "alpha.mp4"
            if not self._write_alpha_video(path, [255, 0, 255]):
                self.skipTest("mp4v VideoWriter unavailable")
            frames = [np.zeros((32, 48, 3), dtype=np.uint8) for _ in range(3)]
            hints = [np.full((32, 48), 128, dtype=np.uint8) for _ in range(3)]
            out = seg._read_alpha_video(path, 3, frames, hints)

        self.assertIsNotNone(out, "gap frame discarded the whole result")
        self.assertEqual(len(out), 3)
        self.assertGreater(int(out[0].max()), 0)
        # The empty frame becomes the hint, not a dropped result.
        self.assertEqual(int(out[1].max()), 128)
        self.assertGreater(int(out[2].max()), 0)

    def test_empty_frame_without_a_hint_becomes_an_empty_matte(self):
        import tempfile

        from backend import segmentation as seg

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "alpha.mp4"
            if not self._write_alpha_video(path, [0, 0]):
                self.skipTest("mp4v VideoWriter unavailable")
            frames = [np.zeros((32, 48, 3), dtype=np.uint8) for _ in range(2)]
            out = seg._read_alpha_video(path, 2, frames)

        self.assertIsNotNone(out)
        self.assertEqual(len(out), 2)
        for item in out:
            self.assertEqual(item.shape, (32, 48))
            self.assertEqual(int(item.max()), 0)

    def test_truncated_output_is_still_rejected(self):
        import tempfile

        from backend import segmentation as seg

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "alpha.mp4"
            if not self._write_alpha_video(path, [255]):
                self.skipTest("mp4v VideoWriter unavailable")
            frames = [np.zeros((32, 48, 3), dtype=np.uint8) for _ in range(4)]
            self.assertIsNone(seg._read_alpha_video(path, 4, frames))

    def test_image_directory_reader_tolerates_a_gap_frame(self):
        import tempfile

        import cv2

        from backend import segmentation as seg

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cv2.imwrite(
                str(root / "a_000.png"), np.full((32, 48), 255, np.uint8))
            cv2.imwrite(
                str(root / "a_001.png"), np.zeros((32, 48), np.uint8))
            frames = [np.zeros((32, 48, 3), dtype=np.uint8) for _ in range(2)]
            hints = [np.full((32, 48), 64, dtype=np.uint8) for _ in range(2)]
            out = seg._read_alpha_image_dir(root, 2, frames, hints)

        self.assertIsNotNone(out)
        self.assertEqual(len(out), 2)
        self.assertEqual(int(out[1].max()), 64)


class LegacyDecoderSeekTests(unittest.TestCase):
    """RM-138: legacy-mode seek must not advertise success it cannot deliver."""

    def _capture(self, decoder, mode="legacy"):
        import cv2

        from backend.decode_accel import _PyNvVideoCapture

        cap = _PyNvVideoCapture.__new__(_PyNvVideoCapture)
        cap._decoder = decoder
        cap._mode = mode
        cap._pos = 0
        cap._frame_count = 100
        cap._width = 16
        cap._height = 16
        cap._opened = True
        return cap, cv2

    def test_seek_fails_when_the_legacy_decoder_cannot_reposition(self):
        class _NoSeek:
            def GetNextFrame(self):
                return None

        cap, cv2 = self._capture(_NoSeek())
        self.assertFalse(cap.set(cv2.CAP_PROP_POS_FRAMES, 42))
        self.assertEqual(cap._pos, 0)

    def test_seek_to_the_current_position_is_a_no_op_success(self):
        class _NoSeek:
            def GetNextFrame(self):
                return None

        cap, cv2 = self._capture(_NoSeek())
        self.assertTrue(cap.set(cv2.CAP_PROP_POS_FRAMES, 0))
        self.assertEqual(cap._pos, 0)

    def test_seek_succeeds_when_the_decoder_supports_it(self):
        seen = []

        class _Seekable:
            def SeekFrame(self, index):
                seen.append(index)

            def GetNextFrame(self):
                return None

        cap, cv2 = self._capture(_Seekable())
        self.assertTrue(cap.set(cv2.CAP_PROP_POS_FRAMES, 42))
        self.assertEqual(seen, [42])
        self.assertEqual(cap._pos, 42)

    def test_a_raising_seek_reports_failure_and_keeps_the_position(self):
        class _Broken:
            def SeekFrame(self, index):
                raise RuntimeError("nope")

            def GetNextFrame(self):
                return None

        cap, cv2 = self._capture(_Broken())
        self.assertFalse(cap.set(cv2.CAP_PROP_POS_FRAMES, 42))
        self.assertEqual(cap._pos, 0)

    def test_simple_mode_indexes_directly_and_still_seeks(self):
        class _Indexable:
            def __getitem__(self, index):
                return None

        cap, cv2 = self._capture(_Indexable(), mode="simple")
        self.assertTrue(cap.set(cv2.CAP_PROP_POS_FRAMES, 42))
        self.assertEqual(cap._pos, 42)

    def test_unsupported_property_is_rejected(self):
        class _NoSeek:
            def GetNextFrame(self):
                return None

        cap, cv2 = self._capture(_NoSeek())
        self.assertFalse(cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920))
