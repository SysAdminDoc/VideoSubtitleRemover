"""Active cancellation and clean-shutdown release checks."""

import threading
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest import mock

from gui.app import VideoSubtitleRemoverApp
from gui.processing_controller import ProcessingControllerMixin


class _FakeRemover:
    def __init__(self):
        self.calls = []

    def terminate_active_work(self, timeout):
        self.calls.append(timeout)


class _FakeProcess:
    def __init__(self):
        self.terminated = False
        self.killed = False
        self.waits = []

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout):
        self.waits.append(timeout)
        return 0

    def kill(self):
        self.killed = True


class _FakeThread:
    def __init__(self):
        self.joins = []

    def is_alive(self):
        return True

    def join(self, timeout):
        self.joins.append(timeout)


class ShutdownTeardownTests(unittest.TestCase):
    def test_terminate_active_backend_work_stops_remover_and_process(self):
        app = object.__new__(VideoSubtitleRemoverApp)
        remover = _FakeRemover()
        process = _FakeProcess()
        app._active_remover = remover
        app._cached_remover = None
        app._active_subprocess = process

        app._terminate_active_backend_work()

        self.assertEqual(remover.calls, [2.0])
        self.assertTrue(process.terminated)
        self.assertEqual(process.waits, [2.0])
        self.assertIsNone(app._active_subprocess)

    def test_join_processing_thread_uses_timeout(self):
        app = object.__new__(VideoSubtitleRemoverApp)
        thread = _FakeThread()
        app._processing_thread = thread

        app._join_processing_thread(0.05)

        self.assertEqual(thread.joins, [0.05])

    def test_soft_subtitle_cancel_checks_global_and_per_item_stop(self):
        captured = {}

        def fake_remux(src, dst, *, action, on_process, cancel_check):
            captured["cancel_check"] = cancel_check

        controller = ProcessingControllerMixin.__new__(
            ProcessingControllerMixin
        )
        controller.cancel_event = threading.Event()
        controller._set_active_subprocess = lambda proc: None
        controller._update_item_display = lambda _item: None
        controller._batch_times = []
        item = SimpleNamespace(
            soft_subtitle_action="strip",
            file_path="in.mkv",
            output_path="out.mkv",
            cancel_requested=False,
            status=None,
            progress=0.0,
            error="x",
            quality_report={},
            started_at=datetime.now(),
            completed_at=None,
            stage_timings=None,
            message="",
            soft_subtitle_summary=None,
        )

        with mock.patch(
            "backend.remux.remux_soft_subtitles", fake_remux
        ), mock.patch("pathlib.Path.mkdir", lambda *args, **kwargs: None):
            controller._process_soft_subtitle_item(item)

        cancel_check = captured["cancel_check"]
        self.assertFalse(cancel_check())
        controller.cancel_event.set()
        self.assertTrue(cancel_check())
        controller.cancel_event.clear()
        item.cancel_requested = True
        self.assertTrue(cancel_check())


if __name__ == "__main__":
    unittest.main()
