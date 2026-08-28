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
        if had_cv2:
            sys.modules["cv2"] = old_cv2
        else:
            sys.modules.pop("cv2", None)
        if old_io is None:
            # Import against the real cv2 so nothing downstream inherits the
            # stub.
            old_io = importlib.import_module("backend.io")
        sys.modules["backend.io"] = old_io
        # ``from backend import io`` reads the package attribute, which the
        # fake-cv2 import above rebound; restore it too or the stub leaks into
        # every later test in the session.
        setattr(importlib.import_module("backend"), "io", old_io)


def _bare_writer(io, process):
    """A ``_LosslessIntermediateWriter`` with only the fields release() reads."""
    import threading

    writer = object.__new__(io._LosslessIntermediateWriter)
    writer._proc = process
    writer._fallback = None
    writer._path = "temp_video.mkv"
    writer._opened = True
    writer._stderr_thread = None
    writer._stderr_tail_buf = bytearray(b"ffmpeg said no")
    writer._stderr_lock = threading.Lock()
    return writer


class LosslessIntermediateWriterTests(unittest.TestCase):
    def test_release_kills_ffmpeg_and_fails_closed_when_flush_times_out(self):
        with _fresh_io_module() as io:
            class FakeProcess:
                returncode = None

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
            writer = _bare_writer(io, process)

            with self.assertRaises(io.MediaWriteError) as ctx:
                writer.release()

            self.assertEqual(ctx.exception.reason, "intermediate_writer_timeout")
            self.assertTrue(process.stdin.closed)
            self.assertTrue(process.killed)
            self.assertEqual(process.wait_timeouts, [300, 10])
            self.assertTrue(process.stderr.closed)
            self.assertIsNone(writer._proc)
            self.assertFalse(writer.isOpened())
            # Cleanup release() in the caller's finally must be a no-op.
            writer.release()

    def test_release_fails_closed_on_nonzero_ffmpeg_exit(self):
        with _fresh_io_module() as io:
            class FakeProcess:
                def __init__(self):
                    self.stdin = _FakeStream()
                    self.stderr = _FakeStream()
                    self.returncode = 1

                def wait(self, timeout):
                    return self.returncode

                def kill(self):  # pragma: no cover - not reached
                    raise AssertionError("kill should not be needed")

            writer = _bare_writer(io, FakeProcess())

            with self.assertRaises(io.MediaWriteError) as ctx:
                writer.release()

            self.assertEqual(ctx.exception.reason, "intermediate_writer_failed")
            self.assertIn("exit code 1", ctx.exception.detail)
            self.assertIsNone(writer._proc)

    def test_release_succeeds_on_clean_ffmpeg_exit(self):
        with _fresh_io_module() as io:
            class FakeProcess:
                def __init__(self):
                    self.stdin = _FakeStream()
                    self.stderr = _FakeStream()
                    self.returncode = 0

                def wait(self, timeout):
                    return 0

            writer = _bare_writer(io, FakeProcess())
            writer.release()
            self.assertIsNone(writer._proc)


class FrameSequenceWriterFailureTests(unittest.TestCase):
    def test_write_fails_closed_when_imwrite_returns_false(self):
        import tempfile
        from pathlib import Path
        import numpy as np
        from backend import io as real_io

        tmp = Path(tempfile.mkdtemp())
        writer = real_io._FrameSequenceWriter(str(tmp))
        frame = np.zeros((4, 4, 3), dtype=np.uint8)

        original = real_io.safe_imwrite
        real_io.safe_imwrite = lambda *args, **kwargs: False
        try:
            with self.assertRaises(real_io.MediaWriteError) as ctx:
                writer.write(frame)
        finally:
            real_io.safe_imwrite = original

        self.assertEqual(ctx.exception.reason, "frame_write_failed")
        # The frame index must not advance past a frame that never landed.
        self.assertEqual(writer._idx, 0)
        writer.write(frame)
        self.assertEqual(writer._idx, 1)
        self.assertTrue((tmp / "frame_000000.png").is_file())

    def test_write_wraps_encoder_error(self):
        import tempfile
        from pathlib import Path
        import numpy as np
        from backend import io as real_io

        tmp = Path(tempfile.mkdtemp())
        writer = real_io._FrameSequenceWriter(str(tmp))

        # RM-317: safe_imwrite turns an OpenCV encode error into False, so the
        # writer sees a failed write rather than a cv2.error. Either way the
        # frame index must not advance.
        original = real_io.safe_imwrite

        def _boom(*args, **kwargs):
            raise real_io.cv2.error("disk on fire")

        real_io.safe_imwrite = _boom
        try:
            with self.assertRaises(real_io.cv2.error):
                writer.write(np.zeros((4, 4, 3), dtype=np.uint8))
        finally:
            real_io.safe_imwrite = original
        self.assertEqual(writer._idx, 0)

        real_io.safe_imwrite = lambda *args, **kwargs: False
        try:
            with self.assertRaises(real_io.MediaWriteError):
                writer.write(np.zeros((4, 4, 3), dtype=np.uint8))
        finally:
            real_io.safe_imwrite = original
        self.assertEqual(writer._idx, 0)

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
