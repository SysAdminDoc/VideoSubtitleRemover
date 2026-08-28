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
