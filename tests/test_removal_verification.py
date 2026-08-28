"""RM-325: prove removal worked by re-running the detector on the output.

The product measured residue with a contrast heuristic whose own docstring
says it is not a replacement for OCR, while a detector sat loaded for every
job and was never asked. Re-detecting inside the repaired region is the
standard success check in the scene-text-removal literature.
"""

from __future__ import annotations

import unittest

import numpy as np

from backend.quality_gate import evaluate_quality_gate
from backend.removal_verification import (
    MATCH_IOU,
    REMOVAL_VERIFICATION_SCHEMA,
    SURVIVING_DETECTION_CONFIDENCE,
    UNTOUCHED_MEAN_ABS_DIFF,
    RemovalVerifier,
    verification_failed,
    verify_frame_removal,
)


class _Detector:
    """A detector that returns whatever the test says is in each frame.

    Keyed on the crop's mean, so an "output" frame that still contains text
    can be distinguished from a repaired one without running real OCR.
    """

    def __init__(self, by_marker):
        self.by_marker = by_marker
        self.calls = 0

    def detect_with_confidence(self, frame, threshold: float = 0.5):
        self.calls += 1
        marker = int(round(float(np.asarray(frame).mean())))
        return list(self.by_marker.get(marker, []))


def _frame(value: int, size=(64, 96)) -> np.ndarray:
    return np.full((size[0], size[1], 3), value, dtype=np.uint8)


class FrameVerificationTests(unittest.TestCase):
    ROI = (10, 10, 80, 50)

    def test_a_clean_output_reports_no_surviving_text(self):
        detector = _Detector({
            200: [(4, 4, 40, 20, 0.95)],   # the source still has the text
            120: [],                        # the output does not
        })
        result = verify_frame_removal(
            detector, _frame(120), self.ROI, source_frame=_frame(200))
        self.assertTrue(result["checked"])
        self.assertEqual(result["detections"], [])
        self.assertEqual(result["sourceBoxes"], 1)
        self.assertEqual(result["survivingSourceBoxes"], 0)

    def test_text_that_survived_is_reported_with_its_box_and_confidence(self):
        box = (4, 4, 40, 20, 0.91)
        detector = _Detector({200: [box], 120: [box]})
        result = verify_frame_removal(
            detector, _frame(120), self.ROI, source_frame=_frame(200))
        self.assertTrue(result["checked"])
        self.assertEqual(len(result["detections"]), 1)
        self.assertEqual(result["survivingSourceBoxes"], 1)
        # Boxes come back in frame coordinates, not crop coordinates.
        self.assertEqual(result["detections"][0]["box"], [14, 14, 50, 30])
        self.assertAlmostEqual(
            result["detections"][0]["confidence"], 0.91, places=5)

    def test_a_low_confidence_box_is_not_called_surviving_text(self):
        detector = _Detector({
            200: [(4, 4, 40, 20, 0.95)],
            120: [(4, 4, 40, 20, SURVIVING_DETECTION_CONFIDENCE - 0.05)],
        })
        result = verify_frame_removal(
            detector, _frame(120), self.ROI, source_frame=_frame(200))
        self.assertEqual(result["detections"], [])
        self.assertEqual(result["survivingSourceBoxes"], 0)

    def test_a_box_somewhere_else_does_not_count_as_the_same_text(self):
        detector = _Detector({
            200: [(2, 2, 20, 12, 0.95)],
            120: [(45, 25, 68, 38, 0.95)],
        })
        result = verify_frame_removal(
            detector, _frame(120), self.ROI, source_frame=_frame(200))
        self.assertEqual(len(result["detections"]), 1)
        # Found something, but not the text that was there before.
        self.assertEqual(result["survivingSourceBoxes"], 0)

    def test_only_the_repaired_region_is_shown_to_the_detector(self):
        seen = []

        class _Recording:
            def detect_with_confidence(self, frame, threshold=0.5):
                seen.append(np.asarray(frame).shape)
                return []

        verify_frame_removal(
            _Recording(), _frame(120), (10, 10, 80, 50),
            source_frame=_frame(200))
        self.assertTrue(seen)
        for shape in seen:
            self.assertEqual(shape[:2], (40, 70))

    def test_an_unusable_frame_is_unchecked_rather_than_clean(self):
        detector = _Detector({})
        for frame, roi, reason in (
            (None, self.ROI, "could not be read"),
            (_frame(120, size=(8, 8)), (0, 0, 8, 8), "too small"),
        ):
            with self.subTest(reason=reason):
                result = verify_frame_removal(detector, frame, roi)
                self.assertFalse(result["checked"])
                self.assertIn(reason, result["reason"])

    def test_no_detector_is_unchecked_rather_than_clean(self):
        result = verify_frame_removal(None, _frame(120), self.ROI)
        self.assertFalse(result["checked"])
        self.assertIn("no detector", result["reason"])

    def test_a_detector_that_raises_is_reported_not_swallowed(self):
        class _Broken:
            def detect_with_confidence(self, frame, threshold=0.5):
                raise RuntimeError("model unloaded")

        result = verify_frame_removal(_Broken(), _frame(120), self.ROI)
        self.assertFalse(result["checked"])
        self.assertIn("model unloaded", result["reason"])


class UntouchedSceneTextTests(unittest.TestCase):
    """The ROI is the union bounding box of every mask in the clip.

    Text that sits inside that box but was never inside a mask - a shop
    sign above the subtitle band, a scoreboard, a logo - is detected in the
    source and detected again in the output, because nothing was ever meant
    to happen to it. It used to count as text that survived the repair, and
    with a short clip that alone could push the surviving fraction past the
    tolerance and fail a job that did exactly what it was asked to do.
    """

    ROI = (0, 0, 96, 64)
    SIGN = (60, 4, 90, 20)
    SUBTITLE = (10, 40, 70, 58)

    def _pair(self, *, repaired: bool):
        """A source frame and an output frame that share the sign."""
        source = np.full((64, 96, 3), 40, dtype=np.uint8)
        source[self.SIGN[1]:self.SIGN[3], self.SIGN[0]:self.SIGN[2]] = 230
        source[self.SUBTITLE[1]:self.SUBTITLE[3],
               self.SUBTITLE[0]:self.SUBTITLE[2]] = 230
        output = source.copy()
        if repaired:
            # The subtitle band is inpainted back to the plate; the sign is
            # bit-for-bit what it was.
            output[self.SUBTITLE[1]:self.SUBTITLE[3],
                   self.SUBTITLE[0]:self.SUBTITLE[2]] = 40
        return source, output

    def _detector(self, source_boxes, output_boxes):
        """Answer by which frame is being looked at, not by its mean."""
        state = {"calls": 0}

        class _ByCall:
            def detect_with_confidence(self, frame, threshold: float = 0.5):
                state["calls"] += 1
                # verify_frame_removal detects the output first, source
                # second.
                return list(output_boxes if state["calls"] == 1
                            else source_boxes)

        return _ByCall()

    def test_scene_text_the_repair_never_touched_is_not_a_survivor(self):
        source, output = self._pair(repaired=True)
        sign = (*self.SIGN, 0.95)
        subtitle = (*self.SUBTITLE, 0.95)
        result = verify_frame_removal(
            self._detector([sign, subtitle], [sign]),
            output, self.ROI, source_frame=source,
        )
        self.assertTrue(result["checked"])
        self.assertEqual(result["untouchedSourceBoxes"], 1)
        # Only the subtitle counted as text that was supposed to go.
        self.assertEqual(result["sourceBoxes"], 1)
        self.assertEqual(result["survivingSourceBoxes"], 0)
        # And the sign does not become a review span either.
        self.assertEqual(result["detections"], [])

    def test_the_same_clip_does_not_fail_the_gate(self):
        source, output = self._pair(repaired=True)
        sign = (*self.SIGN, 0.95)
        subtitle = (*self.SUBTITLE, 0.95)
        verifier = RemovalVerifier(None)
        for index in range(4):
            outcome = verify_frame_removal(
                self._detector([sign, subtitle], [sign]),
                output, self.ROI, source_frame=source,
            )
            outcome["frame"] = index
            verifier.frames.append(outcome)
        evidence = verifier.evidence()
        self.assertEqual(evidence["untouchedSourceBoxes"], 4)
        self.assertEqual(evidence["framesWithSurvivingText"], 0)
        self.assertFalse(verification_failed(evidence))

    def _mask(self):
        """This frame's mask: the subtitle band, and nothing else."""
        mask = np.zeros((64, 96), dtype=np.uint8)
        mask[self.SUBTITLE[1]:self.SUBTITLE[3],
             self.SUBTITLE[0]:self.SUBTITLE[2]] = 255
        return mask

    def test_the_mask_separates_them_without_looking_at_the_pixels(self):
        source, output = self._pair(repaired=True)
        sign = (*self.SIGN, 0.95)
        subtitle = (*self.SUBTITLE, 0.95)
        result = verify_frame_removal(
            self._detector([sign, subtitle], [sign]),
            output, self.ROI, source_frame=source, mask=self._mask(),
        )
        self.assertTrue(result["maskUsed"])
        self.assertEqual(result["untouchedSourceBoxes"], 1)
        self.assertEqual(result["sourceBoxes"], 1)
        self.assertEqual(result["survivingSourceBoxes"], 0)

    def test_text_inside_the_mask_that_survived_still_fails(self):
        """The exclusion must not swallow a real failure.

        This is the case the pixel test alone cannot see: the repair did
        nothing, so every box looks untouched. The mask says the subtitle
        band was supposed to be repaired, so the survivor counts.
        """
        source, output = self._pair(repaired=False)
        sign = (*self.SIGN, 0.95)
        subtitle = (*self.SUBTITLE, 0.95)
        verifier = RemovalVerifier(None)
        for index in range(4):
            outcome = verify_frame_removal(
                self._detector([sign, subtitle], [sign, subtitle]),
                output, self.ROI, source_frame=source, mask=self._mask(),
            )
            outcome["frame"] = index
            verifier.frames.append(outcome)
        evidence = verifier.evidence()
        self.assertEqual(evidence["survivingSourceBoxes"], 4)
        self.assertEqual(evidence["survivingFraction"], 1.0)
        self.assertTrue(verification_failed(evidence))

    def test_a_frame_with_no_mask_has_nothing_that_was_meant_to_go(self):
        """Between subtitles. Detecting the sign there proves nothing."""
        source, output = self._pair(repaired=True)
        sign = (*self.SIGN, 0.95)
        result = verify_frame_removal(
            self._detector([sign], [sign]),
            output, self.ROI, source_frame=source,
            mask=None, mask_available=True,
        )
        self.assertTrue(result["checked"])
        self.assertTrue(result["maskUsed"])
        self.assertEqual(result["sourceBoxes"], 0)
        self.assertEqual(result["detections"], [])

    def test_an_unrepaired_frame_without_a_mask_is_unchecked_not_clean(self):
        """The pixel fallback cannot tell a gap from a total failure.

        Reporting either one as clean would hide a job that did nothing, so
        the frame says why it could not answer instead.
        """
        source, output = self._pair(repaired=False)
        sign = (*self.SIGN, 0.95)
        subtitle = (*self.SUBTITLE, 0.95)
        result = verify_frame_removal(
            self._detector([sign, subtitle], [sign, subtitle]),
            output, self.ROI, source_frame=source,
        )
        self.assertFalse(result["checked"])
        self.assertIn("identical to the source", result["reason"])

    def test_the_tolerance_is_wide_enough_for_codec_noise(self):
        """An untouched region still moves a little through a re-encode."""
        source, output = self._pair(repaired=True)
        rng = np.random.default_rng(7)
        noise = rng.integers(-2, 3, size=output.shape, dtype=np.int16)
        noisy = np.clip(output.astype(np.int16) + noise, 0, 255).astype(
            np.uint8)
        sign = (*self.SIGN, 0.95)
        subtitle = (*self.SUBTITLE, 0.95)
        result = verify_frame_removal(
            self._detector([sign, subtitle], [sign]),
            noisy, self.ROI, source_frame=source,
        )
        self.assertEqual(result["untouchedSourceBoxes"], 1)
        self.assertLess(UNTOUCHED_MEAN_ABS_DIFF, 10.0)


class VerifierEvidenceTests(unittest.TestCase):
    ROI = (10, 10, 80, 50)

    def _run(self, per_frame):
        detector = _Detector({})
        verifier = RemovalVerifier(detector)
        for index, (source, output) in enumerate(per_frame):
            detector.by_marker = {200: source, 120: output}
            verifier.check(index, _frame(120), self.ROI,
                           source_frame=_frame(200))
        return verifier

    def test_a_fully_clean_job_does_not_fail(self):
        box = (4, 4, 40, 20, 0.95)
        verifier = self._run([([box], []) for _ in range(5)])
        evidence = verifier.evidence()
        self.assertEqual(evidence["schema"], REMOVAL_VERIFICATION_SCHEMA)
        self.assertTrue(evidence["ran"])
        self.assertEqual(evidence["framesChecked"], 5)
        self.assertEqual(evidence["survivingSourceBoxes"], 0)
        self.assertEqual(evidence["survivingFraction"], 0.0)
        self.assertFalse(verification_failed(evidence))
        self.assertGreaterEqual(evidence["seconds"], 0.0)

    def test_a_job_that_left_the_text_behind_fails(self):
        box = (4, 4, 40, 20, 0.95)
        verifier = self._run([([box], [box]) for _ in range(5)])
        evidence = verifier.evidence()
        self.assertEqual(evidence["survivingFraction"], 1.0)
        self.assertEqual(evidence["framesWithSurvivingText"], 5)
        self.assertTrue(verification_failed(evidence))
        self.assertEqual(len(evidence["frames"]), 5)

    def test_one_survivor_in_ten_does_not_fail_the_job(self):
        box = (4, 4, 40, 20, 0.95)
        frames = [([box], []) for _ in range(9)] + [([box], [box])]
        evidence = self._run(frames).evidence()
        self.assertAlmostEqual(evidence["survivingFraction"], 0.1)
        self.assertFalse(verification_failed(evidence))

    def test_two_survivors_in_ten_do_fail_the_job(self):
        box = (4, 4, 40, 20, 0.95)
        frames = [([box], []) for _ in range(8)] + [([box], [box])] * 2
        evidence = self._run(frames).evidence()
        self.assertGreater(evidence["survivingFraction"], 0.1)
        self.assertTrue(verification_failed(evidence))

    def test_text_appearing_where_the_source_had_none_still_fails(self):
        """No source boxes leaves the direct question, and it is answered."""
        box = (4, 4, 40, 20, 0.95)
        evidence = self._run([([], [box]) for _ in range(3)]).evidence()
        self.assertEqual(evidence["sourceBoxes"], 0)
        self.assertIsNone(evidence["survivingFraction"])
        self.assertTrue(verification_failed(evidence))

    def test_the_evidence_reports_what_the_check_cost(self):
        """Two detector passes per sampled frame, said out loud."""
        box = (4, 4, 40, 20, 0.95)
        evidence = self._run([([box], []) for _ in range(5)]).evidence()
        self.assertEqual(evidence["framesChecked"], 5)
        self.assertEqual(evidence["sourceScannedFrames"], 5)
        self.assertEqual(evidence["detectorPasses"], 10)

    def test_an_unchecked_job_is_not_reported_as_passing(self):
        verifier = RemovalVerifier(None)
        verifier.check(0, _frame(120), self.ROI)
        evidence = verifier.evidence()
        self.assertTrue(evidence["ran"])
        self.assertEqual(evidence["framesChecked"], 0)
        self.assertEqual(evidence["framesUnchecked"], 1)
        self.assertTrue(evidence["uncheckedReasons"])
        # Nothing was measured, so nothing is claimed either way.
        self.assertFalse(verification_failed(evidence))

    def test_evidence_before_any_check_says_it_did_not_run(self):
        evidence = RemovalVerifier(_Detector({})).evidence()
        self.assertFalse(evidence["ran"])
        self.assertFalse(verification_failed(evidence))


class QualityGateIntegrationTests(unittest.TestCase):
    BASE = {
        "samples": 8, "tag": "Good", "psnr": 44.0,
        "ssim": 0.99, "roi_ssim": 0.99, "residual_text_score": 0.001,
    }

    def _gate(self, verification):
        metrics = dict(self.BASE)
        metrics["removal_verification"] = verification
        return evaluate_quality_gate(metrics)

    def test_a_clean_verification_leaves_the_gate_passing(self):
        gate = self._gate({
            "schema": REMOVAL_VERIFICATION_SCHEMA, "ran": True,
            "framesChecked": 8, "framesWithSurvivingText": 0,
            "sourceBoxes": 12, "survivingSourceBoxes": 0,
            "survivingFraction": 0.0, "survivingFractionThreshold": 0.10,
            "confidenceThreshold": SURVIVING_DETECTION_CONFIDENCE,
        })
        self.assertEqual(gate["status"], "passed")

    def test_surviving_text_fails_the_gate_and_names_the_metric(self):
        gate = self._gate({
            "schema": REMOVAL_VERIFICATION_SCHEMA, "ran": True,
            "framesChecked": 8, "framesWithSurvivingText": 6,
            "sourceBoxes": 12, "survivingSourceBoxes": 9,
            "survivingFraction": 0.75, "survivingFractionThreshold": 0.10,
            "confidenceThreshold": SURVIVING_DETECTION_CONFIDENCE,
        })
        self.assertEqual(gate["status"], "review")
        metrics = [item["metric"] for item in gate["reasons"]]
        self.assertIn("removal_verification", metrics)
        detail = next(
            item["detail"] for item in gate["reasons"]
            if item["metric"] == "removal_verification"
        )
        self.assertIn("75%", detail)

    def test_a_pass_that_did_not_run_adds_no_violation(self):
        for value in (None, {}, {"ran": False}, {"ran": True, "framesChecked": 0}):
            with self.subTest(value=value):
                gate = self._gate(value)
                self.assertEqual(gate["status"], "passed")


class ConfigurationTests(unittest.TestCase):
    def test_the_pass_is_on_by_default_and_can_be_switched_off(self):
        from backend.config import ProcessingConfig, normalize_processing_config

        self.assertTrue(ProcessingConfig().verify_removal)
        config = normalize_processing_config(
            ProcessingConfig(verify_removal=False))
        self.assertFalse(config.verify_removal)

    def test_the_cli_exposes_the_skip_flag(self):
        from backend.cli import _build_parser

        parser = _build_parser(["sttn"])
        text = parser.format_help()
        self.assertIn("--no-verify-removal", text)

    def test_the_matching_constants_are_stated_not_hidden(self):
        self.assertGreater(SURVIVING_DETECTION_CONFIDENCE, 0.5)
        self.assertGreater(MATCH_IOU, 0.0)
        self.assertLess(MATCH_IOU, 1.0)


if __name__ == "__main__":
    unittest.main()
