import unittest

from backend import whisper_fallback
from backend.processor import _spans_from_segments


class WhisperFallbackSpanTests(unittest.TestCase):
    def test_segments_to_frame_spans_skips_malformed_times(self):
        segments = [
            (float("nan"), 1.0, "nan"),
            (0.0, float("inf"), "inf"),
            (4.0, 3.0, "reverse"),
            ("bad", 3.0, "bad"),
            (1.0, 1.5, "valid"),
        ]

        self.assertEqual(
            whisper_fallback.segments_to_frame_spans(segments, fps=10.0),
            [(10, 15)],
        )

    def test_processor_span_helper_uses_cfr_fallback(self):
        self.assertEqual(
            _spans_from_segments(
                [(1.0, 1.5, "line")],
                fps=10.0,
                total_frames=100,
            ),
            [(10, 15)],
        )

    def test_processor_span_helper_uses_vfr_frame_range(self):
        class Timing:
            def __init__(self):
                self.calls = []

            def frame_range(self, start, end, total):
                self.calls.append((start, end, total))
                return 7, 11

        timing = Timing()
        self.assertEqual(
            _spans_from_segments(
                [(1.0, 1.5, "line")],
                fps=30.0,
                total_frames=200,
                frame_timing=timing,
            ),
            [(7, 11)],
        )
        self.assertEqual(timing.calls, [(1.0, 1.5, 200)])


if __name__ == "__main__":
    unittest.main()
