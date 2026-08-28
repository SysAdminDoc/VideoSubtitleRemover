"""RM-316: hold the system awake for a job, and only for a job.

A long run could be cut short by idle sleep. Windows offers a scoped power
request for this, and the risk in using it is the opposite failure: leaving
the hold behind so the machine never sleeps again, or taking a wider hold
than the job needs and pinning the display on.
"""

from __future__ import annotations

import ast
import threading
import unittest
from pathlib import Path
from unittest import mock

from backend import keep_awake

ROOT = Path(__file__).resolve().parent.parent


class _Recorder:
    """Stand in for SetThreadExecutionState and remember every request."""

    def __init__(self, accept: bool = True):
        self.calls: list[int] = []
        self.accept = accept

    def __call__(self, flags: int) -> bool:
        self.calls.append(flags)
        return self.accept


class KeepAwakeLifecycleTests(unittest.TestCase):
    def setUp(self):
        keep_awake.release_all()
        self.addCleanup(keep_awake.release_all)

    def _patched(self, recorder, windows=True):
        return mock.patch.multiple(
            keep_awake,
            _set_execution_state=recorder,
            _is_windows=lambda: windows,
        )

    def test_one_job_takes_and_releases_exactly_one_hold(self):
        recorder = _Recorder()
        with self._patched(recorder):
            with keep_awake.keep_system_awake():
                self.assertTrue(keep_awake.status()["held"])
                self.assertEqual(keep_awake.status()["depth"], 1)
            self.assertFalse(keep_awake.status()["held"])

        self.assertEqual(recorder.calls, [
            keep_awake.ES_CONTINUOUS | keep_awake.ES_SYSTEM_REQUIRED,
            keep_awake.ES_CONTINUOUS,
        ])

    def test_overlapping_jobs_do_not_release_each_others_hold(self):
        recorder = _Recorder()
        with self._patched(recorder):
            keep_awake.acquire()
            keep_awake.acquire()
            self.assertEqual(keep_awake.status()["depth"], 2)
            keep_awake.release()
            # The first job finished; the second is still running.
            self.assertTrue(keep_awake.status()["held"])
            keep_awake.release()
            self.assertFalse(keep_awake.status()["held"])
        self.assertEqual(len(recorder.calls), 2)

    def test_an_exception_inside_the_block_still_releases(self):
        recorder = _Recorder()
        with self._patched(recorder):
            with self.assertRaises(RuntimeError):
                with keep_awake.keep_system_awake():
                    raise RuntimeError("cancelled")
            self.assertFalse(keep_awake.status()["held"])
        self.assertEqual(recorder.calls[-1], keep_awake.ES_CONTINUOUS)

    def test_release_all_drops_every_reference(self):
        recorder = _Recorder()
        with self._patched(recorder):
            for _ in range(4):
                keep_awake.acquire()
            keep_awake.release_all()
            status = keep_awake.status()
        self.assertEqual(status["depth"], 0)
        self.assertFalse(status["held"])

    def test_an_unbalanced_release_cannot_drive_the_count_negative(self):
        recorder = _Recorder()
        with self._patched(recorder):
            keep_awake.release()
            keep_awake.release()
            self.assertEqual(keep_awake.status()["depth"], 0)
            keep_awake.acquire()
            self.assertTrue(keep_awake.status()["held"])
            keep_awake.release()
            self.assertFalse(keep_awake.status()["held"])

    def test_concurrent_jobs_leave_the_count_balanced(self):
        recorder = _Recorder()
        errors: list = []

        def _job():
            try:
                with keep_awake.keep_system_awake():
                    pass
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(exc)

        with self._patched(recorder):
            threads = [threading.Thread(target=_job) for _ in range(12)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(30)
            status = keep_awake.status()

        self.assertEqual(errors, [])
        self.assertEqual(status["depth"], 0)
        self.assertFalse(status["held"])

    def test_a_refused_request_is_reported_rather_than_assumed(self):
        recorder = _Recorder(accept=False)
        with self._patched(recorder):
            keep_awake.acquire()
            status = keep_awake.status()
            keep_awake.release()
        self.assertFalse(status["held"])
        self.assertTrue(status["error"])

    def test_it_is_a_no_op_off_windows(self):
        recorder = _Recorder()
        with self._patched(recorder, windows=False):
            with keep_awake.keep_system_awake():
                status = keep_awake.status()
        self.assertFalse(status["supported"])
        self.assertFalse(status["held"])
        self.assertEqual(recorder.calls, [])


class KeepAwakeScopeTests(unittest.TestCase):
    """The request must stay narrow: system only, never the display."""

    def test_the_display_and_away_mode_flags_are_never_requested(self):
        recorder = _Recorder()
        with mock.patch.multiple(
            keep_awake,
            _set_execution_state=recorder,
            _is_windows=lambda: True,
        ):
            keep_awake.acquire()
            keep_awake.release()
        for flags in recorder.calls:
            self.assertFalse(flags & keep_awake.ES_DISPLAY_REQUIRED)
            self.assertFalse(flags & keep_awake.ES_AWAYMODE_REQUIRED)

    def test_no_caller_passes_the_wider_flags(self):
        """Grepping the constant is not enough; parse the call sites."""
        banned = {"ES_DISPLAY_REQUIRED", "ES_AWAYMODE_REQUIRED"}
        offenders = []
        for root in (ROOT / "backend", ROOT / "gui"):
            for path in sorted(root.rglob("*.py")):
                if path.name == "keep_awake.py":
                    continue
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name) and node.id in banned:
                        offenders.append(f"{path.name}:{node.lineno}")
                    elif (isinstance(node, ast.Attribute)
                            and node.attr in banned):
                        offenders.append(f"{path.name}:{node.lineno}")
        self.assertEqual(offenders, [])


class KeepAwakeWiringTests(unittest.TestCase):
    """The hold has to be taken where the work happens."""

    def test_the_batch_worker_holds_and_releases_around_the_whole_run(self):
        source = (ROOT / "gui" / "processing_controller.py").read_text(
            encoding="utf-8")
        tree = ast.parse(source)
        worker = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_process_queue"
        )
        # acquire before the try, release in the finally: a pause that breaks
        # out of the batch loop, a cancellation, and a destroyed root all
        # leave through the same path.
        body = worker.body
        self.assertTrue(
            any(isinstance(node, ast.Try) for node in body),
            "the batch worker must guard its hold with try/finally",
        )
        guard = next(node for node in body if isinstance(node, ast.Try))
        finally_calls = [
            node.func.attr
            for node in ast.walk(ast.Module(body=guard.finalbody,
                                            type_ignores=[]))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        ]
        self.assertIn("release", finally_calls)

    def test_the_cli_run_takes_the_hold(self):
        source = (ROOT / "backend" / "cli.py").read_text(encoding="utf-8")
        self.assertIn("keep_awake.acquire()", source)
        self.assertIn("keep_awake.release_all", source)

    def test_the_gui_drops_the_hold_on_shutdown(self):
        source = (ROOT / "gui" / "app.py").read_text(encoding="utf-8")
        self.assertIn("keep_awake.release_all()", source)


if __name__ == "__main__":
    unittest.main()
