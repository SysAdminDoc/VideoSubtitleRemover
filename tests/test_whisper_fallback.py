import unittest

from backend import whisper_fallback


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


if __name__ == "__main__":
    unittest.main()
