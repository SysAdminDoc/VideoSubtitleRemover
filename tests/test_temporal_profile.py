"""RM-150: mask-aware temporal regression profile."""

from pathlib import Path
import sys
import time
import unittest

import numpy as np
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend import temporal_profile as tp  # noqa: E402
from backend.inpainters import _common  # noqa: E402


class SyntheticFixtureTests(unittest.TestCase):
    def test_axes_vary_independently(self):
        static_clean, _s, static_masks = tp.synthetic_clip()
        camera_clean, _c, camera_masks = tp.synthetic_clip(camera_motion=3.0)
        background_clean, _b, _bm = tp.synthetic_clip(background_motion=9.0)
        _mc, _m, mask_masks = tp.synthetic_clip(mask_motion=5.0)

        # Camera motion changes the plate but not the mask geometry.
        self.assertFalse(np.array_equal(static_clean[-1], camera_clean[-1]))
        self.assertTrue(np.array_equal(static_masks[-1], camera_masks[-1]))
        # Background motion changes the plate without moving the mask.
        self.assertFalse(np.array_equal(static_clean[-1], background_clean[-1]))
        # Mask motion moves the box.
        self.assertFalse(np.array_equal(static_masks[-1], mask_masks[-1]))
        self.assertTrue(np.array_equal(static_masks[0], mask_masks[0]))

    def test_fixtures_are_deterministic(self):
        first = tp.synthetic_clip(camera_motion=2.0, seed=3)[0]
        second = tp.synthetic_clip(camera_motion=2.0, seed=3)[0]
        for a, b in zip(first, second):
            self.assertTrue(np.array_equal(a, b))

    def test_subtitled_frames_differ_only_inside_the_mask(self):
        clean, subtitled, masks = tp.synthetic_clip()
        delta = np.abs(
            subtitled[0].astype(int) - clean[0].astype(int)).sum(axis=2)
        outside = masks[0] == 0
        self.assertEqual(int(delta[outside].sum()), 0)
        self.assertGreater(int(delta[~outside].sum()), 0)


class DenseFlowEstimatorTests(unittest.TestCase):
    @staticmethod
    def _large_motion_fixture():
        height, width = 160, 240
        rng = np.random.default_rng(7)
        reference = rng.integers(
            0, 256, (height, width, 3), dtype=np.uint8)
        reference = _common.cv2.GaussianBlur(reference, (3, 3), 0)
        dx, dy = 16.0, 8.0
        source = _common.cv2.warpAffine(
            reference,
            np.asarray([[1, 0, dx], [0, 1, dy]], dtype=np.float32),
            (width, height),
            borderMode=_common.cv2.BORDER_REFLECT,
        )
        expected = np.zeros((height, width, 2), dtype=np.float32)
        expected[..., 0] = dx
        expected[..., 1] = dy
        valid = np.zeros((height, width), dtype=bool)
        valid[32:-32, 32:-32] = True
        return reference, source, expected, valid

    def test_dis_beats_farneback_on_large_blurred_motion(self):
        reference, source, expected, valid = self._large_motion_fixture()
        ref_gray = _common.cv2.cvtColor(
            reference, _common.cv2.COLOR_BGR2GRAY)
        src_gray = _common.cv2.cvtColor(source, _common.cv2.COLOR_BGR2GRAY)
        measurements = {}
        for estimator in ("farneback", "dis"):
            _common._calc_dense_flow(ref_gray, src_gray, estimator)
            timings = []
            errors = []
            for _ in range(5):
                started = time.perf_counter()
                flow = _common._calc_dense_flow(
                    ref_gray, src_gray, estimator)
                timings.append(time.perf_counter() - started)
                errors.append(float(np.linalg.norm(
                    flow[valid] - expected[valid], axis=1).mean()))
            measurements[estimator] = (
                float(np.median(errors)), float(np.median(timings)))

        dis_error, dis_time = measurements["dis"]
        farneback_error, farneback_time = measurements["farneback"]
        self.assertLessEqual(dis_error, farneback_error)
        self.assertLessEqual(dis_time, farneback_time)

    def test_frame_and_mask_warps_honour_the_selected_estimator(self):
        reference, source, _expected, _valid = self._large_motion_fixture()
        mask = np.zeros(reference.shape[:2], dtype=np.uint8)
        mask[48:92, 84:138] = 255
        with mock.patch.object(
            _common, "_calc_dense_flow",
            wraps=_common._calc_dense_flow,
        ) as estimator:
            _common._warp_to_reference(source, reference, "farneback")
            _common._warp_mask_to_reference(
                mask, source, reference, "dis")
        self.assertEqual(
            [call.args[2] for call in estimator.call_args_list],
            ["farneback", "dis"],
        )

    def test_dis_unavailable_falls_back_to_farneback(self):
        reference, source, _expected, _valid = self._large_motion_fixture()
        ref_gray = _common.cv2.cvtColor(
            reference, _common.cv2.COLOR_BGR2GRAY)
        src_gray = _common.cv2.cvtColor(source, _common.cv2.COLOR_BGR2GRAY)
        with mock.patch.object(_common.cv2, "DISOpticalFlow_create", None), \
                self.assertLogs(_common.logger.name, level="DEBUG") as captured:
            flow = _common._calc_dense_flow(ref_gray, src_gray, "dis")
        self.assertEqual(flow.shape[:2], ref_gray.shape)
        self.assertTrue(any("unavailable" in line for line in captured.output))


class MetricTests(unittest.TestCase):
    def test_a_perfect_fill_scores_near_zero_on_every_axis(self):
        for kwargs in (
            {},
            {"camera_motion": 3.0},
            {"background_motion": 9.0},
            {"mask_motion": 5.0},
        ):
            with self.subTest(**kwargs):
                clean, subtitled, masks = tp.synthetic_clip(**kwargs)
                report = tp.evaluate_temporal_profile(subtitled, clean, masks)
            self.assertTrue(report["passed"], report["failures"])
            self.assertTrue(report["measured"])


class TemporalBackgroundExposureTests(unittest.TestCase):
    @staticmethod
    def _recover(*, global_motion_align: bool, camera_motion: float = 4.0):
        clean, subtitled, masks = tp.synthetic_clip(
            frames=8,
            width=192,
            height=128,
            camera_motion=camera_motion,
            seed=7,
        )
        filled = _common._temporal_background_expose(
            subtitled,
            masks,
            min_coverage=1,
            use_median=True,
            feather_px=0,
            edge_ring_px=0,
            flow_warp=False,
            global_motion_align=global_motion_align,
            scene_cut_split=False,
        )
        return clean, masks, filled

    def test_global_alignment_improves_pan_recovery_metrics(self):
        clean, masks, baseline = self._recover(global_motion_align=False)
        _clean, _masks, aligned = self._recover(global_motion_align=True)

        baseline_mae = np.mean([
            np.mean(np.abs(
                actual[mask > 0].astype(np.int16)
                - expected[mask > 0].astype(np.int16)
            ))
            for actual, expected, mask in zip(baseline, clean, masks)
        ])
        aligned_mae = np.mean([
            np.mean(np.abs(
                actual[mask > 0].astype(np.int16)
                - expected[mask > 0].astype(np.int16)
            ))
            for actual, expected, mask in zip(aligned, clean, masks)
        ])
        self.assertLess(aligned_mae, baseline_mae * 0.8)
        self.assertLess(
            tp.masked_flicker(aligned, masks),
            tp.masked_flicker(baseline, masks),
        )
        self.assertLess(
            tp.masked_warp_residual(aligned, masks),
            tp.masked_warp_residual(baseline, masks),
        )

    def test_static_camera_output_is_unchanged_with_alignment(self):
        clean, _masks, baseline = self._recover(
            global_motion_align=False, camera_motion=0.0)
        _clean, _masks, aligned = self._recover(
            global_motion_align=True, camera_motion=0.0)
        self.assertLessEqual(
            max(
                int(np.abs(actual.astype(np.int16) - expected.astype(np.int16)).max())
                for actual, expected in zip(aligned, baseline)
            ),
            1,
        )
        self.assertLessEqual(
            float(np.mean([
                np.mean(np.abs(actual.astype(np.int16) - expected.astype(np.int16)))
                for actual, expected in zip(aligned, clean)
            ])),
            float(np.mean([
                np.mean(np.abs(actual.astype(np.int16) - expected.astype(np.int16)))
                for actual, expected in zip(baseline, clean)
            ])) + 1.0,
        )

    def test_robust_aggregation_rejects_corrupted_exposure(self):
        shape = (12, 16, 3)
        frames = [np.full(shape, 100, dtype=np.uint8) for _ in range(65)]
        masks = [np.zeros(shape[:2], dtype=np.uint8) for _ in frames]
        region = (slice(3, 9), slice(4, 12))
        for mask in masks[:61]:
            mask[region] = 255
        frames[-1][region] = 255

        recovered = _common._tbe_single_segment(
            frames,
            masks,
            min_coverage=3,
            use_median=False,
            feather_px=0,
            edge_ring_px=0,
            flow_warp=False,
            global_motion_align=False,
        )

        current_mean = int((100 * 3 + 255) / 4)
        self.assertGreater(current_mean, 100)
        np.testing.assert_array_equal(recovered[0][region], 100)
        np.testing.assert_array_equal(recovered[0][masks[0] == 0], 100)

    def test_robust_aggregation_falls_back_for_sparse_pixels(self):
        shape = (12, 16, 3)
        frames = [
            np.full(shape, value, dtype=np.uint8)
            for value in (0, 100, 255)
        ]
        masks = [np.zeros(shape[:2], dtype=np.uint8) for _ in frames]
        region = (slice(3, 9), slice(4, 12))
        masks[0][region] = 255

        recovered = _common._tbe_single_segment(
            frames,
            masks,
            min_coverage=1,
            use_median=False,
            feather_px=0,
            edge_ring_px=0,
            flow_warp=False,
            global_motion_align=False,
        )

        np.testing.assert_array_equal(recovered[0][region], 177)

    def test_low_ransac_inlier_ratio_falls_back_to_identity(self):
        frames = [np.full((32, 48, 3), value, dtype=np.uint8)
                  for value in (20, 40, 60)]
        masks = [np.zeros((32, 48), dtype=np.uint8) for _ in frames]
        masks[0][10:20, 10:30] = 255
        fake_matrix = np.asarray([[1.0, 0.0, 4.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        with mock.patch.object(
            _common,
            "estimate_global_motion_quality",
            return_value=(fake_matrix, 0.1),
        ), self.assertLogs(_common.logger.name, level="DEBUG") as captured:
            aligned = _common._tbe_single_segment(
                frames,
                masks,
                min_coverage=1,
                use_median=True,
                feather_px=0,
                edge_ring_px=0,
                flow_warp=False,
                global_motion_align=True,
            )
        self.assertTrue(any("inlier ratio" in line for line in captured.output))
        baseline = _common._tbe_single_segment(
            frames,
            masks,
            min_coverage=1,
            use_median=True,
            feather_px=0,
            edge_ring_px=0,
            flow_warp=False,
            global_motion_align=False,
        )
        for actual, expected in zip(aligned, baseline):
            np.testing.assert_array_equal(actual, expected)

    def test_a_frozen_fill_fails_the_warp_residual(self):
        clean, subtitled, masks = tp.synthetic_clip(camera_motion=3.0)
        filled = tp.inject_regression(clean, masks, "frozen")
        report = tp.evaluate_temporal_profile(subtitled, filled, masks)
        self.assertFalse(report["passed"])
        self.assertTrue(any("warp residual" in f for f in report["failures"]))

    def test_a_flickering_fill_fails(self):
        clean, subtitled, masks = tp.synthetic_clip(camera_motion=3.0)
        filled = tp.inject_regression(clean, masks, "flicker")
        report = tp.evaluate_temporal_profile(subtitled, filled, masks)
        self.assertFalse(report["passed"])

    def test_a_leaking_fill_fails_the_edge_bar(self):
        clean, subtitled, masks = tp.synthetic_clip(camera_motion=3.0)
        filled = tp.inject_regression(clean, masks, "leak")
        report = tp.evaluate_temporal_profile(subtitled, filled, masks)
        self.assertFalse(report["passed"])
        self.assertTrue(any("edge leakage" in f for f in report["failures"]))

    def test_camera_motion_alone_is_not_reported_as_a_defect(self):
        # The whole point of motion compensation: a heavily panning clip with
        # a perfect fill must not look like a temporal regression.
        clean, subtitled, masks = tp.synthetic_clip(camera_motion=6.0)
        report = tp.evaluate_temporal_profile(subtitled, clean, masks)
        self.assertTrue(report["passed"], report["failures"])

    def test_metrics_return_none_without_enough_frames(self):
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=np.uint8)
        self.assertIsNone(tp.masked_warp_residual([frame], [mask]))
        self.assertIsNone(tp.masked_flicker([frame, frame], [mask, mask]))

    def test_unknown_regression_kind_is_rejected(self):
        clean, _s, masks = tp.synthetic_clip()
        with self.assertRaises(ValueError):
            tp.inject_regression(clean, masks, "banana")


class ProfileRunTests(unittest.TestCase):
    def test_full_profile_passes_on_clean_fills(self):
        result = tp.run_temporal_regression_profile()
        self.assertTrue(result["ran"])
        self.assertTrue(result["passed"], result["failures"])
        labels = [case["label"] for case in result["cases"]]
        self.assertEqual(
            labels,
            ["static", "camera-motion", "background-motion",
             "mask-motion", "all-motion"],
        )

    def test_profile_needs_no_download_or_licensed_media(self):
        import inspect

        source = inspect.getsource(tp)
        for banned in ("http://", "https://", "urlopen", "requests.", "hf_hub"):
            self.assertNotIn(banned, source)

    def test_tightened_thresholds_make_the_profile_fail(self):
        # Proves the bars are actually load-bearing, not decorative.
        strict = tp.TemporalThresholds(
            warp_residual=0.01, edge_leakage=0.0, masked_flicker=0.01)
        result = tp.run_temporal_regression_profile(thresholds=strict)
        self.assertFalse(result["passed"])
        self.assertTrue(result["failures"])


class ReleaseEvidenceTests(unittest.TestCase):
    def test_strict_verification_fails_on_a_temporal_regression(self):
        from backend.release_verification import _validation_errors

        evidence = {
            "releaseTools": {
                "temporalProfile": {
                    "ran": True,
                    "passed": False,
                    "failures": ["camera-motion: masked warp residual 9 exceeds 2.5"],
                },
            },
        }
        messages = list(_validation_errors(evidence))
        self.assertTrue(
            any("Mask-aware temporal profile failed" in item for item in messages),
            messages,
        )

    def test_a_skipped_profile_is_not_a_failure(self):
        from backend.release_verification import _validation_errors

        evidence = {
            "releaseTools": {
                "temporalProfile": {"ran": False, "passed": None, "failures": []},
            },
        }
        self.assertFalse(any(
            "Mask-aware temporal profile" in item
            for item in _validation_errors(evidence)
        ))

    def test_release_runner_returns_a_structured_result(self):
        from backend.release_verification import _run_temporal_profile

        result = _run_temporal_profile()
        self.assertEqual(result["schema"], tp.TEMPORAL_PROFILE_SCHEMA)
        self.assertTrue(result["ran"])
        self.assertTrue(result["passed"], result["failures"])


if __name__ == "__main__":
    unittest.main()
