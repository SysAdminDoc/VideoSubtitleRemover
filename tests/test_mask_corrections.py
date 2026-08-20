from __future__ import annotations

import unittest

import numpy as np

from backend.mask_corrections import (
    MASK_CORRECTION_SCHEMA,
    apply_mask_corrections,
    brush_polygon,
    make_review_span,
    merge_frame_ranges,
    merge_review_spans,
    normalize_mask_correction,
)


class MaskCorrectionTests(unittest.TestCase):
    def test_ordered_add_then_subtract_uses_exact_frame_bounds(self):
        corrections = [
            {
                "schema": MASK_CORRECTION_SCHEMA,
                "mode": "add",
                "polygons": [[2, 2, 20, 2, 20, 20, 2, 20]],
                "start": 0.0,
                "end": 10.0,
                "start_frame": 2,
                "end_frame": 4,
            },
            {
                "schema": MASK_CORRECTION_SCHEMA,
                "mode": "subtract",
                "polygons": [[8, 8, 14, 8, 14, 14, 8, 14]],
                "start": 0.0,
                "end": 10.0,
                "start_frame": 2,
                "end_frame": 4,
            },
        ]
        before = apply_mask_corrections(
            np.zeros((24, 24), np.uint8), corrections, 1.0, 1)
        during = apply_mask_corrections(
            np.zeros((24, 24), np.uint8), corrections, 1.0, 2)
        self.assertEqual(int(before.sum()), 0)
        self.assertEqual(int(during[4, 4]), 255)
        self.assertEqual(int(during[10, 10]), 0)

    def test_legacy_correction_stays_additive_without_payload_churn(self):
        legacy = {
            "polygons": [[1, 2, 8, 2, 8, 9]],
            "start": 0.0,
            "end": 2.0,
        }
        self.assertEqual(normalize_mask_correction(legacy), legacy)

    def test_brush_and_span_helpers_are_bounded_and_deterministic(self):
        polygon = brush_polygon(0, 0, 12, 32, 24)
        self.assertEqual(len(polygon), 24)
        self.assertTrue(all(0 <= value <= 31 for value in polygon[::2]))
        self.assertTrue(all(0 <= value <= 23 for value in polygon[1::2]))
        spans = merge_review_spans([
            make_review_span("residual", 4, 5, fps=10.0, score=0.03),
            make_review_span("residual", 5, 6, fps=10.0, score=0.05),
            make_review_span("flicker", 8, 10, fps=10.0, score=0.1),
        ])
        self.assertEqual([(span["kind"], span["start_frame"], span["end_frame"])
                          for span in spans], [
            ("residual", 4, 6),
            ("flicker", 8, 10),
        ])
        self.assertEqual(merge_frame_ranges([(8, 10), (2, 4), (4, 7)]),
                         [(2, 7), (8, 10)])

    def test_batch_quality_record_preserves_review_queue(self):
        from backend.batch_report import _quality_report_record

        span = make_review_span("low-confidence", 2, 3, fps=10.0, score=0.55)
        record = _quality_report_record({
            "tag": "Review",
            "samples": 1,
            "mask_review_spans": [span],
        })
        self.assertEqual(record["mask_review_spans"], [span])


class TimedCorrectionGateTests(unittest.TestCase):
    """RM-191: mask reuse has to know a correction is time-gated."""

    def test_untimed_corrections_do_not_trip_the_gate(self):
        from backend.mask_corrections import has_timed_corrections

        self.assertFalse(has_timed_corrections(None))
        self.assertFalse(has_timed_corrections([]))
        self.assertFalse(has_timed_corrections(
            [{"mode": "add", "polygons": [[0, 0, 8, 0, 8, 8, 0, 8]]}]))

    def test_each_kind_of_bound_trips_the_gate(self):
        from backend.mask_corrections import has_timed_corrections

        base = {"mode": "add", "polygons": [[0, 0, 8, 0, 8, 8, 0, 8]]}
        for bound in ({"start_frame": 100}, {"end_frame": 200},
                      {"start": 1.5}, {"end": 9.0}):
            with self.subTest(bound=bound):
                self.assertTrue(has_timed_corrections([dict(base, **bound)]))


class FrameOnlyCorrectionBoundTests(unittest.TestCase):
    """RM-243b: an end_frame with no start_frame must still bound by frame."""

    def _correction(self, **bounds):
        return {"mode": "add", "polygons": [[0, 0, 8, 0, 8, 8, 0, 8]],
                **bounds}

    def test_end_frame_alone_is_honoured(self):
        from backend.mask_corrections import correction_is_active

        correction = self._correction(end_frame=200)
        self.assertTrue(correction_is_active(correction, 0.0, frame_index=199))
        self.assertFalse(correction_is_active(correction, 0.0, frame_index=200))

    def test_start_frame_alone_is_honoured(self):
        from backend.mask_corrections import correction_is_active

        correction = self._correction(start_frame=100)
        self.assertFalse(correction_is_active(correction, 0.0, frame_index=99))
        self.assertTrue(correction_is_active(correction, 0.0, frame_index=100))


class ReviewSpanMergeTests(unittest.TestCase):
    """RM-243b: merging only against the previous span left duplicates."""

    def test_an_interleaved_other_kind_does_not_block_the_merge(self):
        from backend.mask_corrections import merge_review_spans

        merged = merge_review_spans([
            {"kind": "low_confidence", "start_frame": 0, "end_frame": 10},
            {"kind": "residual", "start_frame": 4, "end_frame": 6},
            {"kind": "low_confidence", "start_frame": 10, "end_frame": 20},
        ])

        by_kind = {}
        for span in merged:
            by_kind.setdefault(span["kind"], []).append(span)
        self.assertEqual(len(by_kind["low_confidence"]), 1)
        self.assertEqual(by_kind["low_confidence"][0]["end_frame"], 20)

    def test_output_is_ordered_by_start_frame(self):
        from backend.mask_corrections import merge_review_spans

        merged = merge_review_spans([
            {"kind": "residual", "start_frame": 50, "end_frame": 60},
            {"kind": "low_confidence", "start_frame": 0, "end_frame": 10},
        ])

        self.assertEqual(
            [span["start_frame"] for span in merged], [0, 50])


if __name__ == "__main__":
    unittest.main()
