import unittest

from gui.app import VideoSubtitleRemoverApp


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


if __name__ == "__main__":
    unittest.main()
