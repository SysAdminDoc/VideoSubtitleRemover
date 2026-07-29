"""RM-150: mask-aware temporal regression profile."""

from pathlib import Path
import sys
import unittest

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend import temporal_profile as tp


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
