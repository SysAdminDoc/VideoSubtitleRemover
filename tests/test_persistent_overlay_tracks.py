"""RM-361: a logo present all runtime is not a subtitle.

Every track was created with ``keep`` False, so a station logo that sits in a
corner for the whole video was queued for removal on exactly the same footing
as a caption that appears for two seconds. The plan format already recorded
the span needed to tell them apart and never used it.
"""

from __future__ import annotations

import unittest

from backend.track_plan import (
    CAPTION_BAND_TOP_FRACTION,
    PERSISTENT_COVERAGE_THRESHOLD,
    classify_persistent_overlays,
    plan_to_mask_corrections,
)

FRAME_H = 1080
FRAME_W = 1920
TOTAL = 3000


def _track(track_id, start, end, bbox):
    return {
        "id": track_id,
        "start_frame": start,
        "end_frame": end,
        "bbox": list(bbox),
        "sample_text": "",
        "sample_count": 1,
        "keep": False,
    }


def _corner_logo():
    # Top-right, present for the whole runtime.
    return _track(1, 0, TOTAL - 1, (1650, 40, 1880, 130))


def _caption():
    # Lower third, two seconds at 30 fps.
    return _track(2, 900, 960, (400, 950, 1500, 1030))


class PersistentOverlayClassificationTests(unittest.TestCase):
    def test_a_full_runtime_corner_mark_is_flagged(self):
        tracks = classify_persistent_overlays(
            [_corner_logo()], frame_count=TOTAL, frame_height=FRAME_H)
        track = tracks[0]
        self.assertTrue(track["persistent_overlay"])
        self.assertEqual(track["position"], "upper")
        self.assertAlmostEqual(track["coverage"], 1.0, places=3)

    def test_a_short_caption_is_not_flagged(self):
        tracks = classify_persistent_overlays(
            [_caption()], frame_count=TOTAL, frame_height=FRAME_H)
        track = tracks[0]
        self.assertFalse(track["persistent_overlay"])
        self.assertEqual(track["position"], "lower")
        self.assertFalse(track["keep"])

    def test_a_caption_running_the_whole_clip_is_still_not_flagged(self):
        """Coverage alone must not condemn a subtitle.

        A talking-head clip can carry burned-in captions from the first frame
        to the last. Position is what separates that from a logo.
        """
        long_caption = _track(3, 0, TOTAL - 1, (400, 950, 1500, 1030))
        tracks = classify_persistent_overlays(
            [long_caption], frame_count=TOTAL, frame_height=FRAME_H)
        self.assertFalse(tracks[0]["persistent_overlay"])
        self.assertFalse(tracks[0]["keep"])

    def test_a_brief_corner_mark_is_not_flagged_either(self):
        """Position alone must not condemn a chyron or an upper caption."""
        brief = _track(4, 100, 260, (1650, 40, 1880, 130))
        tracks = classify_persistent_overlays(
            [brief], frame_count=TOTAL, frame_height=FRAME_H)
        self.assertFalse(tracks[0]["persistent_overlay"])

    def test_both_conditions_are_recorded_on_the_track(self):
        tracks = classify_persistent_overlays(
            [_corner_logo(), _caption()],
            frame_count=TOTAL, frame_height=FRAME_H)
        for track in tracks:
            with self.subTest(track=track["id"]):
                self.assertIn("coverage", track)
                self.assertIn("position", track)
                self.assertIn("persistent_overlay", track)


class IntentTests(unittest.TestCase):
    def test_subtitle_intent_defaults_a_flagged_track_to_keep(self):
        tracks = classify_persistent_overlays(
            [_corner_logo()], frame_count=TOTAL, frame_height=FRAME_H,
            remove_subtitles=True)
        self.assertTrue(tracks[0]["keep"])
        self.assertIn("logo", tracks[0]["keep_reason"])

    def test_logo_intent_leaves_the_default_alone(self):
        tracks = classify_persistent_overlays(
            [_corner_logo()], frame_count=TOTAL, frame_height=FRAME_H,
            remove_subtitles=False)
        self.assertTrue(
            tracks[0]["persistent_overlay"],
            "the classification is still reported, only the default changes",
        )
        self.assertFalse(
            tracks[0]["keep"],
            "under logo intent the overlay is the target, not the exception",
        )

    def test_a_kept_flagged_track_is_excluded_from_the_inpaint_mask(self):
        """The flag has to reach the mask, not just the report."""
        tracks = classify_persistent_overlays(
            [_corner_logo(), _caption()],
            frame_count=TOTAL, frame_height=FRAME_H, remove_subtitles=True)
        corrections = plan_to_mask_corrections({"tracks": tracks})
        self.assertEqual(
            len(corrections), 1,
            "exactly the logo should be subtracted from the mask",
        )
        self.assertEqual(corrections[0]["start_frame"], 0)

    def test_logo_intent_subtracts_nothing(self):
        tracks = classify_persistent_overlays(
            [_corner_logo(), _caption()],
            frame_count=TOTAL, frame_height=FRAME_H, remove_subtitles=False)
        self.assertEqual(plan_to_mask_corrections({"tracks": tracks}), [])


class ScanIntegrationTests(unittest.TestCase):
    """The classifier has to actually run inside scan_track_plan.

    Testing classify_persistent_overlays alone passes even if the scan never
    calls it, or calls it with a released capture whose height reads zero.
    """

    def _scan(self, remove_subtitles):
        import types
        from unittest import mock

        import numpy as np

        from backend import track_plan

        detections = []

        class _Detector:
            _engine_name = "stub"

            def detect(self, frame, threshold=0.5):
                # A corner mark on every sampled frame.
                return [(1650, 40, 1880, 130)]

        class _Capture:
            def __init__(self, *_a, **_k):
                self._i = 0

            def isOpened(self):
                return True

            def get(self, prop):
                import cv2
                return {
                    cv2.CAP_PROP_FPS: 30.0,
                    cv2.CAP_PROP_FRAME_COUNT: 300.0,
                    cv2.CAP_PROP_FRAME_HEIGHT: float(FRAME_H),
                }.get(prop, 0.0)

            def grab(self):
                self._i += 1
                return self._i <= 300

            def retrieve(self):
                return True, np.zeros((FRAME_H, FRAME_W, 3), np.uint8)

            def release(self):
                return None

        config = types.SimpleNamespace(remove_subtitles=remove_subtitles)
        with mock.patch.object(track_plan.cv2, "VideoCapture", _Capture),              mock.patch.object(track_plan, "_probe_video_frame_timing",
                               return_value=None),              mock.patch.object(track_plan, "_attach_thumbnails",
                               lambda *a, **k: None):
            return track_plan.scan_track_plan(
                __file__, detector=_Detector(), config=config,
                thumbnails=False)

    def test_the_scan_classifies_and_keeps_under_subtitle_intent(self):
        plan = self._scan(remove_subtitles=True)
        self.assertTrue(plan["tracks"], "the stub detector produced no track")
        track = plan["tracks"][0]
        self.assertTrue(
            track.get("persistent_overlay"),
            "scan_track_plan did not classify; a released capture reports "
            "height 0 and the classifier returns early",
        )
        self.assertTrue(track["keep"])

    def test_the_scan_leaves_logo_intent_alone(self):
        plan = self._scan(remove_subtitles=False)
        track = plan["tracks"][0]
        self.assertTrue(track.get("persistent_overlay"))
        self.assertFalse(track["keep"])


class BoundaryTests(unittest.TestCase):
    def test_a_zero_frame_count_changes_nothing(self):
        original = _corner_logo()
        tracks = classify_persistent_overlays(
            [dict(original)], frame_count=0, frame_height=FRAME_H)
        self.assertNotIn("persistent_overlay", tracks[0])
        self.assertFalse(tracks[0]["keep"])

    def test_an_unknown_frame_height_changes_nothing(self):
        tracks = classify_persistent_overlays(
            [_corner_logo()], frame_count=TOTAL, frame_height=0)
        self.assertNotIn("persistent_overlay", tracks[0])

    def test_coverage_just_under_the_threshold_is_not_flagged(self):
        span = int(TOTAL * PERSISTENT_COVERAGE_THRESHOLD) - 2
        track = _track(5, 0, span, (1650, 40, 1880, 130))
        tracks = classify_persistent_overlays(
            [track], frame_count=TOTAL, frame_height=FRAME_H)
        self.assertLess(tracks[0]["coverage"], PERSISTENT_COVERAGE_THRESHOLD)
        self.assertFalse(tracks[0]["persistent_overlay"])

    def test_a_box_straddling_the_band_edge_is_judged_by_its_centre(self):
        edge = int(FRAME_H * CAPTION_BAND_TOP_FRACTION)
        # Centre one pixel above the band: outside it.
        above = _track(6, 0, TOTAL - 1, (100, edge - 20, 300, edge - 2))
        # Centre below: inside the caption band.
        below = _track(7, 0, TOTAL - 1, (100, edge + 2, 300, edge + 20))
        tracks = classify_persistent_overlays(
            [above, below], frame_count=TOTAL, frame_height=FRAME_H)
        self.assertTrue(tracks[0]["persistent_overlay"])
        self.assertFalse(tracks[1]["persistent_overlay"])


if __name__ == "__main__":
    unittest.main()
