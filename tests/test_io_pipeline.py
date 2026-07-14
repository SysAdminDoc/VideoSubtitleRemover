import importlib
import sys
import types
import unittest
from contextlib import contextmanager


class _FakeStream:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


@contextmanager
def _fresh_io_module():
    old_io = sys.modules.pop("backend.io", None)
    old_cv2 = sys.modules.get("cv2")
    had_cv2 = "cv2" in sys.modules
    fake_cv2 = types.SimpleNamespace(
        VideoWriter=lambda *args, **kwargs: None,
        VideoWriter_fourcc=lambda *args: 0,
    )
    sys.modules["cv2"] = fake_cv2
    try:
        yield importlib.import_module("backend.io")
    finally:
        sys.modules.pop("backend.io", None)
        if old_io is not None:
            sys.modules["backend.io"] = old_io
        if had_cv2:
            sys.modules["cv2"] = old_cv2
        else:
            sys.modules.pop("cv2", None)


class LosslessIntermediateWriterTests(unittest.TestCase):
    def test_release_kills_ffmpeg_when_flush_times_out(self):
        with _fresh_io_module() as io:
            class FakeProcess:
                def __init__(self):
                    self.stdin = _FakeStream()
                    self.stderr = _FakeStream()
                    self.wait_timeouts = []
                    self.killed = False

                def wait(self, timeout):
                    self.wait_timeouts.append(timeout)
                    if len(self.wait_timeouts) == 1:
                        raise io.subprocess.TimeoutExpired(
                            cmd="ffmpeg",
                            timeout=timeout,
                        )
                    return 0

                def kill(self):
                    self.killed = True

            process = FakeProcess()
            writer = object.__new__(io._LosslessIntermediateWriter)
            writer._proc = process
            writer._fallback = None

            writer.release()

            self.assertTrue(process.stdin.closed)
            self.assertTrue(process.killed)
            self.assertEqual(process.wait_timeouts, [300, 10])
            self.assertTrue(process.stderr.closed)
            self.assertIsNone(writer._proc)

    def test_terminate_aborts_active_ffmpeg_writer(self):
        with _fresh_io_module() as io:
            class FakeProcess:
                def __init__(self):
                    self.stdin = _FakeStream()
                    self.stderr = _FakeStream()
                    self.terminated = False
                    self.killed = False
                    self.wait_timeouts = []

                def poll(self):
                    return None

                def terminate(self):
                    self.terminated = True

                def wait(self, timeout):
                    self.wait_timeouts.append(timeout)
                    return 0

                def kill(self):
                    self.killed = True

            process = FakeProcess()
            writer = object.__new__(io._LosslessIntermediateWriter)
            writer._proc = process
            writer._fallback = None

            writer.terminate(timeout=0.25)

            self.assertTrue(process.stdin.closed)
            self.assertTrue(process.terminated)
            self.assertFalse(process.killed)
            self.assertEqual(process.wait_timeouts, [0.25])
            self.assertTrue(process.stderr.closed)
            self.assertIsNone(writer._proc)


class LosslessWriterStderrDrainTests(unittest.TestCase):
    def test_stderr_drain_prevents_deadlock_on_noisy_ffmpeg(self):
        # A real ffmpeg run that emits many stderr warnings must not deadlock
        # the stdin frame writer even when the stderr pipe would otherwise fill.
        import shutil
        import tempfile
        from pathlib import Path
        import numpy as np
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not installed")
        from backend import io as real_io
        tmp = Path(tempfile.mkdtemp())
        out = tmp / "noisy.mkv"
        writer = real_io._LosslessIntermediateWriter(str(out), 48, 48, 24.0)
        self.assertTrue(writer.isOpened())
        for i in range(120):
            frame = (np.random.RandomState(i).randint(0, 255, (48, 48, 3))
                     ).astype(np.uint8)
            writer.write(frame)
        writer.release()
        self.assertIsNone(writer._proc)
        self.assertGreater(out.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
