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


class AudioExtractionTimeoutTests(unittest.TestCase):
    """RM-137: the demux+resample budget scales with the source duration."""

    def _run_with(self, duration, probe_error=False):
        import tempfile
        from unittest import mock

        from backend import io as backend_io

        seen = {}

        def _fake_run_process(cmd, **kwargs):
            seen["timeout"] = kwargs.get("timeout")
            raise TimeoutError("stop before ffmpeg runs")

        def _probe(_path):
            if probe_error:
                raise OSError("probe failed")
            return duration

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(backend_io, "_probe_duration_seconds", _probe):
                with mock.patch.object(
                    whisper_fallback.shutil, "which", lambda _name: "ffmpeg"
                ):
                    with mock.patch.object(
                        whisper_fallback, "run_process", _fake_run_process
                    ):
                        try:
                            whisper_fallback.extract_audio_to_temp(
                                "source.mp4", tmpdir)
                        except TimeoutError:
                            pass
        return seen.get("timeout")

    def test_long_source_gets_a_timeout_above_the_600s_floor(self):
        # A four-hour source must not be capped at the old flat 600 s.
        timeout = self._run_with(4 * 3600.0)
        self.assertIsNotNone(timeout)
        self.assertGreater(timeout, 600.0)

    def test_short_source_keeps_the_floor(self):
        self.assertGreaterEqual(self._run_with(5.0), 600.0)

    def test_probe_failure_falls_back_to_a_safe_default(self):
        self.assertEqual(self._run_with(0.0, probe_error=True), 600.0)


if __name__ == "__main__":
    unittest.main()
