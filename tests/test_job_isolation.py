"""RM-155: supervise each queue job in an isolated child process."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import unittest.mock

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.job_worker import (
    CONTROL_POLL_SECONDS,
    JOB_PROTOCOL_SCHEMA,
    JOB_PROTOCOL_VERSION,
    NATIVE_FAULT_CODES,
    _ControlFile,
    describe_exit_code,
    write_control_file,
)
from gui import job_supervisor
from gui.job_supervisor import (
    SCRATCH_REAP_TIMEOUT,
    JobOutcome,
    JobSupervisor,
    build_request,
    worker_command,
)


class ExitCodeDecodingTests(unittest.TestCase):
    """A native fault must read as a sentence, not a 10-digit integer."""

    def test_windows_fault_codes_are_named(self):
        self.assertIn(
            "access violation", describe_exit_code(0xC0000005).lower())
        self.assertIn("0xC0000005", describe_exit_code(0xC0000005))
        self.assertIn("stack overflow", describe_exit_code(0xC00000FD))
        self.assertIn("illegal instruction", describe_exit_code(0xC000001D))

    def test_a_signed_windows_status_decodes_the_same_way(self):
        # Popen.wait returns the DWORD as a signed int on Windows.
        signed = 0xC0000005 - (1 << 32)
        self.assertEqual(
            describe_exit_code(signed), describe_exit_code(0xC0000005))

    def test_posix_signals_are_named(self):
        self.assertIn("signal 11", describe_exit_code(-11))
        self.assertIn("signal 9", describe_exit_code(-9))

    def test_an_ordinary_status_stays_plain(self):
        self.assertEqual(describe_exit_code(0), "exit code 0")
        self.assertEqual(describe_exit_code(2), "exit code 2")

    def test_every_documented_code_produces_a_description(self):
        for code in NATIVE_FAULT_CODES:
            with self.subTest(code=hex(code)):
                text = describe_exit_code(code)
                self.assertNotIn("exit code", text)
                self.assertIn(f"0x{code:08X}", text)


class ControlFileTests(unittest.TestCase):
    """Cancel and pause travel through a polled file, never stdin."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "control.json"

    def tearDown(self):
        self._tmp.cleanup()

    def test_state_round_trips(self):
        write_control_file(self.path, cancel=True, pause=False)
        control = _ControlFile(self.path)
        self.assertTrue(control.cancelled.is_set())
        self.assertFalse(control.paused.is_set())

    def test_an_update_is_observed_after_the_poll_interval(self):
        write_control_file(self.path)
        control = _ControlFile(self.path)
        self.assertFalse(control.cancelled.is_set())
        write_control_file(self.path, cancel=True)
        time.sleep(CONTROL_POLL_SECONDS * 1.5)
        self.assertTrue(control.cancelled.is_set())

    def test_reads_are_cached_so_a_per_frame_callback_is_cheap(self):
        write_control_file(self.path)
        control = _ControlFile(self.path)
        control.cancelled.is_set()
        # Change the file, then poll again immediately: the cached answer
        # must still be returned rather than stat()ing on every frame.
        write_control_file(self.path, cancel=True)
        self.assertFalse(control.cancelled.is_set())

    def test_a_missing_or_corrupt_file_never_means_cancel(self):
        # Losing the control file must not abort a job that is running
        # fine; "no request pending" is the safe reading.
        control = _ControlFile(Path(self._tmp.name) / "absent.json")
        self.assertFalse(control.cancelled.is_set())
        self.assertFalse(control.paused.is_set())

        self.path.write_text("{ not json", encoding="utf-8")
        corrupt = _ControlFile(self.path)
        self.assertFalse(corrupt.cancelled.is_set())

    def test_writing_is_atomic_and_leaves_no_partial_file(self):
        write_control_file(self.path, cancel=True, pause=True)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(payload, {"cancel": True, "pause": True})
        self.assertFalse(self.path.with_name(self.path.name + ".tmp").exists())

    def test_no_control_path_means_nothing_is_ever_requested(self):
        control = _ControlFile("")
        self.assertFalse(control.cancelled.is_set())
        self.assertFalse(control.paused.is_set())


class RequestTests(unittest.TestCase):
    def test_a_request_carries_the_protocol_version(self):
        request = build_request(
            input_path="a.mp4", output_path="b.mp4",
            config_payload={"device": "cpu"}, is_image=False,
        )
        self.assertEqual(request["schema"], JOB_PROTOCOL_SCHEMA)
        self.assertEqual(request["version"], JOB_PROTOCOL_VERSION)
        self.assertEqual(request["config"]["device"], "cpu")
        self.assertFalse(request["is_image"])

    def test_a_request_carries_auto_band(self):
        # Auto-band has to cross the process boundary explicitly, or an
        # isolated job silently runs unpinned full-frame detection.
        self.assertFalse(build_request(
            input_path="a.mp4", output_path="b.mp4",
            config_payload={}, is_image=False)["auto_band"])
        self.assertTrue(build_request(
            input_path="a.mp4", output_path="b.mp4",
            config_payload={}, is_image=False, auto_band=True)["auto_band"])

    def test_wiring_a_preview_callback_provisions_a_preview_dir(self):
        # A caller that asks for previews must get them without having to
        # invent a scratch directory of its own.
        request = build_request(
            input_path="a.mp4", output_path="b.mp4",
            config_payload={}, is_image=False,
        )
        silent = JobSupervisor(request)
        try:
            self.assertFalse(silent.request.get("preview_dir"))
        finally:
            silent._cleanup_scratch()
        wired = JobSupervisor(request, on_preview=lambda p, i, t: None)
        try:
            preview_dir = wired.request.get("preview_dir")
            self.assertTrue(preview_dir)
            self.assertIn(str(wired._scratch), preview_dir)
        finally:
            wired._cleanup_scratch()

    def test_the_worker_command_targets_the_module_from_source(self):
        command = worker_command("request.json")
        self.assertIn("-m", command)
        self.assertIn("backend.job_worker", command)
        self.assertIn("request.json", command)

    def test_the_worker_reports_its_protocol_version_without_a_job(self):
        # The version probe is how a parent confirms agreement before it
        # has any work to hand over, so it must not require a request.
        result = subprocess.run(
            [sys.executable, "-m", "backend.job_worker", "--protocol-version"],
            cwd=_ROOT, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(JOB_PROTOCOL_VERSION))

    def test_running_a_job_without_a_request_is_a_usage_error(self):
        result = subprocess.run(
            [sys.executable, "-m", "backend.job_worker"],
            cwd=_ROOT, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--request", result.stderr)


class SupervisorFailureTests(unittest.TestCase):
    """The whole point: a child that dies is one item, not the batch."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.request = build_request(
            input_path=str(self.root / "in.mp4"),
            output_path=str(self.root / "out.mp4"),
            config_payload={}, is_image=False,
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _stub(self, body: str) -> list:
        stub = self.root / "stub_worker.py"
        stub.write_text(body, encoding="utf-8")
        return [sys.executable, str(stub)]

    def test_a_child_that_dies_without_a_result_is_reported_as_crashed(self):
        outcome = JobSupervisor(
            self.request,
            command=self._stub(
                "import sys\n"
                "sys.stderr.write('native fault imminent\\n')\n"
                "sys.stderr.flush()\n"
                "sys.exit(3)\n"
            ),
        ).run(timeout=120)
        self.assertEqual(outcome.status, "crashed")
        self.assertTrue(outcome.crashed)
        self.assertFalse(outcome.success)
        self.assertEqual(outcome.reason, "worker_crashed")
        self.assertEqual(outcome.exit_code, 3)
        # The message has to say the rest of the queue is fine, because
        # that is the user's first question after a crash.
        self.assertIn("queue", outcome.error)
        # The child's dying words are the only diagnostic that survives.
        self.assertIn("native fault imminent", outcome.stderr_tail)

    def test_a_child_killed_mid_run_is_still_only_one_item(self):
        outcome = JobSupervisor(
            self.request,
            command=self._stub(
                "import os, sys\n"
                "sys.stderr.write('going down hard\\n'); sys.stderr.flush()\n"
                "os._exit(9)\n"
            ),
        ).run(timeout=120)
        self.assertEqual(outcome.status, "crashed")
        self.assertIn("going down hard", outcome.stderr_tail)

    def test_timeout_is_wall_clock_bounded_while_stdout_stays_open(self):
        supervisor = JobSupervisor(
            self.request,
            command=self._stub(
                "import time\n"
                "time.sleep(30)\n"
            ),
        )
        started = time.monotonic()
        outcome = supervisor.run(timeout=0.2)
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 3.0)
        self.assertEqual(outcome.status, "error")
        self.assertEqual(outcome.reason, "worker_timeout")
        self.assertIn("time budget", outcome.error)

    def test_a_worker_that_cannot_be_spawned_is_an_error_not_a_crash(self):
        outcome = JobSupervisor(
            self.request,
            command=[str(self.root / "does-not-exist.exe")],
        ).run(timeout=30)
        self.assertEqual(outcome.status, "error")
        self.assertEqual(outcome.reason, "worker_spawn_failed")
        self.assertFalse(outcome.crashed)

    def test_non_protocol_output_is_ignored_not_misread(self):
        # Libraries print progress bars and warnings to stdout; only the
        # protocol lines are events.
        result = json.dumps({
            "schema": JOB_PROTOCOL_SCHEMA,
            "event": "result",
            "status": "complete",
            "success": True,
            "evidence": {"last_output_path": "x.mp4"},
        })
        outcome = JobSupervisor(
            self.request,
            command=self._stub(
                "import sys\n"
                "print('Loading model...')\n"
                "print('[####    ] 50 percent')\n"
                "sys.stdout.write(" + repr(result) + " + '\\n')\n"
            ),
        ).run(timeout=120)
        self.assertEqual(outcome.status, "complete")
        self.assertTrue(outcome.success)
        self.assertEqual(outcome.evidence["last_output_path"], "x.mp4")

    def test_a_protocol_version_mismatch_is_refused_by_the_real_worker(self):
        stale = dict(self.request)
        stale["version"] = JOB_PROTOCOL_VERSION + 99
        outcome = JobSupervisor(stale).run(timeout=180)
        self.assertEqual(outcome.reason, "protocol_version_mismatch")
        self.assertFalse(outcome.success)
        self.assertNotEqual(outcome.status, "crashed")

    def test_an_unreadable_request_is_refused_by_the_real_worker(self):
        outcome = JobSupervisor(
            build_request(input_path="", output_path="",
                          config_payload={}, is_image=False),
        ).run(timeout=180)
        self.assertEqual(outcome.reason, "invalid_request")
        self.assertFalse(outcome.success)

    def test_an_unknown_config_field_is_refused_before_any_work(self):
        bad = build_request(
            input_path=str(self.root / "in.mp4"),
            output_path=str(self.root / "out.mp4"),
            config_payload={"not_a_real_field": 1}, is_image=False,
        )
        outcome = JobSupervisor(bad).run(timeout=180)
        self.assertEqual(outcome.reason, "invalid_config")
        self.assertFalse(outcome.success)

    def test_cancel_and_pause_publish_control_state_for_the_child(self):
        supervisor = JobSupervisor(self.request)
        supervisor._write_request()
        control = Path(supervisor.request["control_path"])
        self.assertTrue(control.is_file())

        supervisor.pause()
        self.assertTrue(json.loads(control.read_text(encoding="utf-8"))["pause"])
        supervisor.resume()
        self.assertFalse(json.loads(control.read_text(encoding="utf-8"))["pause"])
        supervisor.cancel()
        payload = json.loads(control.read_text(encoding="utf-8"))
        self.assertTrue(payload["cancel"])
        supervisor._cleanup_scratch()

    def test_a_cancelled_child_reports_cancelled_not_crashed(self):
        # A user-requested stop must not look like a native fault.
        supervisor = JobSupervisor(
            self.request,
            command=self._stub(
                "import sys, time\n"
                "sys.stderr.write('working\\n'); sys.stderr.flush()\n"
                "time.sleep(30)\n"
            ),
        )

        def stop_soon():
            time.sleep(1.0)
            supervisor.cancel()
            supervisor.terminate()

        import threading
        threading.Thread(target=stop_soon, daemon=True).start()
        outcome = supervisor.run(timeout=60)
        self.assertEqual(outcome.status, "cancelled")
        self.assertFalse(outcome.crashed)

    def test_the_scratch_directory_is_removed_after_a_run(self):
        supervisor = JobSupervisor(
            self.request, command=self._stub("import sys\nsys.exit(0)\n"))
        supervisor.run(timeout=60)
        self.assertFalse(Path(supervisor._scratch).exists())


class SupervisedRunTests(unittest.TestCase):
    """A real supervised job, end to end, through the real worker."""

    @classmethod
    def setUpClass(cls):
        if shutil.which("ffmpeg") is None:
            raise unittest.SkipTest("ffmpeg not on PATH")
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.source = cls.root / "clip.avi"
        writer = cv2.VideoWriter(
            str(cls.source), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
        if not writer.isOpened():
            cls._tmp.cleanup()
            raise unittest.SkipTest("OpenCV MJPG writer unavailable")
        try:
            for value in (30, 60, 90, 120):
                writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
        finally:
            writer.release()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _config_payload(self):
        from backend.config import ProcessingConfig
        from backend.config_schema import serialize_dataclass_config

        return serialize_dataclass_config(ProcessingConfig(
            mode="sttn", device="cpu", sttn_skip_detection=True,
            subtitle_area=(8, 32, 56, 44), preserve_audio=False,
            adaptive_batch=False, use_hw_encode=False, prefetch_decode=False,
            sttn_max_load_num=4,
        ))

    def test_a_supervised_job_completes_and_returns_its_evidence(self):
        output = self.root / "supervised.mp4"
        preview_dir = self.root / "preview"
        progress = []
        previews = []
        request = build_request(
            input_path=str(self.source),
            output_path=str(output),
            config_payload=self._config_payload(),
            is_image=False,
            preview_dir=str(preview_dir),
        )
        outcome = JobSupervisor(
            request,
            on_progress=lambda p, m: progress.append((p, m)),
            on_preview=lambda path, i, t: previews.append(path),
        ).run(timeout=900)

        self.assertEqual(outcome.status, "complete", outcome.error)
        self.assertTrue(outcome.success)
        self.assertTrue(output.is_file())
        self.assertFalse(outcome.crashed)

        # Progress crossed the process boundary and reached completion.
        self.assertTrue(progress)
        self.assertAlmostEqual(progress[-1][0], 1.0, places=3)

        # Preview frames crossed as file paths, since JSON cannot carry
        # a numpy array cheaply.
        self.assertTrue(previews, "no preview frame crossed the boundary")

        # The evidence a completed job must publish back.
        for key in (
            "last_stage_timings", "last_detection_stats",
            "last_output_path", "execution_provenance",
        ):
            with self.subTest(key=key):
                self.assertIn(key, outcome.evidence)
        self.assertEqual(
            Path(outcome.evidence["last_output_path"]).name, output.name)

    def test_the_child_runs_with_no_stdin_so_grandchildren_cannot_inherit_one(self):
        # A live stdin pipe was the original deadlock: a reader thread
        # parked on it blocked against C-extension module init, and any
        # ffmpeg or import probe the job spawned inherited it.
        probe = self.root / "stdin_probe.py"
        probe.write_text(
            "import json, sys\n"
            "closed = sys.stdin is None or sys.stdin.read() == ''\n"
            "print(json.dumps({'schema': " + repr(JOB_PROTOCOL_SCHEMA) + ",\n"
            "                  'event': 'result', 'status': 'complete',\n"
            "                  'success': closed, 'evidence': {}}))\n",
            encoding="utf-8",
        )
        outcome = JobSupervisor(
            build_request(
                input_path="x", output_path="y",
                config_payload={}, is_image=False),
            command=[sys.executable, str(probe)],
        ).run(timeout=120)
        self.assertEqual(outcome.status, "complete")
        self.assertTrue(outcome.success, "the child saw an open stdin")


class ControllerIntegrationTests(unittest.TestCase):
    """The queue-model mapping, without spawning a real job."""

    def _host(self):
        from gui.config import ProcessingConfig, ProcessingStatus, QueueItem
        from gui.processing_controller import ProcessingControllerMixin

        item = QueueItem(
            id="job-1",
            file_path="clip.mp4",
            output_path="clip.out.mp4",
            config=ProcessingConfig(job_isolation=True),
            status=ProcessingStatus.PROCESSING,
        )
        from datetime import datetime

        item.started_at = datetime.now()
        updates = []

        class Host(ProcessingControllerMixin):
            def __init__(self):
                self._batch_times = []

            def _update_item_display(self, entry):
                updates.append(entry.status)

            @staticmethod
            def _normalized_path_key(value):
                return str(value).lower()

        return Host(), item, updates

    def test_a_crashed_outcome_marks_only_that_item_and_keeps_its_logs(self):
        """Drive the real outcome branch, not just the status table: the
        stderr tail is the only surviving evidence of a native fault, so
        losing it is the failure this test exists to catch."""
        import threading
        from unittest import mock

        from gui.config import ProcessingStatus

        host, item, updates = self._host()
        tail = "cudnn: CUDNN_STATUS_NOT_INITIALIZED"
        outcome = JobOutcome(
            status="crashed",
            error="The job worker stopped before finishing.",
            reason="worker_crashed",
            exit_code=0xC0000005,
            stderr_tail=tail,
        )

        class FakeSupervisor:
            def __init__(self, *args, **kwargs):
                pass

            def run(self):
                return outcome

        host.cancel_event = threading.Event()
        host.pause_event = threading.Event()
        host.queue = [item]
        host._batch_report_records = {}
        host._watch_isolated_controls = lambda *a, **k: None
        host._announce_model_download_guidance = lambda *a, **k: None
        host._push_live_preview = lambda *a, **k: None
        host._update_status = lambda *a, **k: None
        host._dispatch_preview_ui = lambda *a, **k: None
        host.root = None

        with mock.patch("gui.job_supervisor.JobSupervisor", FakeSupervisor):
            host._process_item_isolated(item)

        self.assertEqual(item.status, ProcessingStatus.ERROR)
        # Only this item is touched.
        self.assertEqual(len(host.queue), 1)
        # The half of this test's own name that was never checked: the
        # child's diagnostics survive onto the item.
        self.assertIn(tail, item.retry_errors or [])
        self.assertEqual(item.error, "The job worker stopped before finishing.")
        # The fixture records display updates; a crashed item must refresh.
        self.assertIn(ProcessingStatus.ERROR, updates)

    def test_every_child_status_maps_to_a_queue_status(self):
        from gui.config import ProcessingStatus

        host, _item, _updates = self._host()
        self.assertEqual(
            host._ISOLATED_STATUS["complete"], ProcessingStatus.COMPLETE)
        self.assertEqual(
            host._ISOLATED_STATUS["paused"], ProcessingStatus.PAUSED)
        self.assertEqual(
            host._ISOLATED_STATUS["cancelled"], ProcessingStatus.CANCELLED)
        self.assertEqual(
            host._ISOLATED_STATUS["error"], ProcessingStatus.ERROR)
        # A crash is an error to the queue, but a distinct outcome to the
        # supervisor, which is what lets the log say what really happened.
        self.assertEqual(
            host._ISOLATED_STATUS["crashed"], ProcessingStatus.ERROR)

    def test_evidence_is_copied_onto_the_item(self):
        host, item, _updates = self._host()
        host._apply_isolated_evidence(item, {
            "last_stage_timings": {"ocr": 1.5},
            "last_detection_stats": {"frames_total": 4},
            "execution_provenance": {"stages": []},
            "last_quality_report": {"psnr": 42.0},
            "last_output_path": "clip.out.mp4",
            "last_pause_checkpoint_path": "ckpt.json",
        })
        self.assertEqual(item.stage_timings, {"ocr": 1.5})
        self.assertEqual(item.detection_stats, {"frames_total": 4})
        self.assertEqual(item.execution_provenance, {"stages": []})
        self.assertEqual(item.quality_report, {"psnr": 42.0})
        self.assertEqual(item.pause_checkpoint_path, "ckpt.json")

    def test_a_changed_output_path_is_adopted_and_locked(self):
        host, item, _updates = self._host()
        host._apply_isolated_evidence(
            item, {"last_output_path": "clip.fallback.mkv"})
        self.assertEqual(item.output_path, "clip.fallback.mkv")
        self.assertTrue(item.output_path_locked)

    def test_missing_evidence_degrades_to_empty_not_none(self):
        host, item, _updates = self._host()
        host._apply_isolated_evidence(item, {})
        self.assertEqual(item.stage_timings, {})
        self.assertEqual(item.detection_stats, {})
        self.assertIsNone(item.quality_report)

    def test_the_setting_round_trips(self):
        from gui.config import ProcessingConfig

        config = ProcessingConfig(job_isolation=True).normalized()
        self.assertTrue(config.job_isolation)
        restored = ProcessingConfig.from_dict(config.to_dict()).normalized()
        self.assertTrue(restored.job_isolation)
        self.assertFalse(ProcessingConfig().normalized().job_isolation)


class WatchdogTests(unittest.TestCase):
    """The control watchdog must neither spin forever nor wait forever."""

    def _host(self):
        import threading

        from gui.config import ProcessingConfig, ProcessingStatus, QueueItem
        from gui.processing_controller import ProcessingControllerMixin

        item = QueueItem(
            id="job-w", file_path="clip.mp4", output_path="clip.out.mp4",
            config=ProcessingConfig(job_isolation=True),
            status=ProcessingStatus.PROCESSING,
        )

        class Host(ProcessingControllerMixin):
            def __init__(self):
                self.cancel_event = threading.Event()
                self.pause_event = threading.Event()

        return Host(), item

    def test_a_failed_spawn_does_not_leave_the_watchdog_spinning(self):
        # When the worker cannot be spawned, supervisor.pid stays None and
        # the item goes terminal; the watchdog must notice and stand down
        # instead of sleeping in a loop for the life of the process.
        import threading

        from gui.config import ProcessingStatus

        host, item = self._host()
        item.status = ProcessingStatus.ERROR

        class NeverSpawned:
            pid = None

        thread = threading.Thread(
            target=host._watch_isolated_controls,
            args=(NeverSpawned(), item), daemon=True)
        thread.start()
        thread.join(timeout=5.0)
        self.assertFalse(thread.is_alive(), "watchdog kept spinning")

    def test_a_child_that_ignores_cancel_is_terminated(self):
        # A child wedged in native code never reads the control file; the
        # watchdog must escalate to terminate() after the grace period.
        import threading

        host, item = self._host()
        host._ISOLATED_CANCEL_GRACE_SECONDS = 0.2
        item.cancel_requested = True
        calls = []

        class Wedged:
            pid = 4242

            def cancel(self):
                calls.append("cancel")
                return True

            def pause(self):
                calls.append("pause")
                return True

            def resume(self):
                calls.append("resume")
                return True

            def terminate(self):
                calls.append("terminate")

        thread = threading.Thread(
            target=host._watch_isolated_controls,
            args=(Wedged(), item), daemon=True)
        thread.start()
        thread.join(timeout=10.0)
        self.assertFalse(thread.is_alive(), "watchdog never escalated")
        self.assertIn("cancel", calls)
        self.assertIn("terminate", calls)
        self.assertEqual(calls.index("cancel"), 0)


class FrozenEntryPointTests(unittest.TestCase):
    def test_the_frozen_entry_point_routes_the_job_worker_marker(self):
        # A frozen build cannot `-m backend.job_worker`, so the exe has to
        # recognise the marker itself.
        source = (_ROOT / "VideoSubtitleRemover.py").read_text(
            encoding="utf-8")
        self.assertIn("--job-worker", source)
        marker = source.index("--job-worker")
        smoke = source.index("--frozen-import-smoke")
        # It must be handled before any DPI/Tk/settings work.
        self.assertLess(marker, smoke)

    def test_the_marker_short_circuits_before_logging_and_gui_imports(self):
        # The worker child must not open the parent's rotating log file
        # (two processes rotating one file breaks on Windows) or import
        # the widget tree. The short-circuit has to sit above both.
        source = (_ROOT / "VideoSubtitleRemover.py").read_text(
            encoding="utf-8")
        marker = source.index("--job-worker")
        self.assertLess(marker, source.index("logging.basicConfig"))
        self.assertLess(marker, source.index("from gui import"))

    def test_the_entry_script_answers_the_worker_marker_without_tk(self):
        # End to end: the exe path a frozen supervisor uses must reach the
        # worker's argument parser without touching logging or the GUI.
        result = subprocess.run(
            [sys.executable, str(_ROOT / "VideoSubtitleRemover.py"),
             "--job-worker", "--protocol-version"],
            cwd=_ROOT, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(JOB_PROTOCOL_VERSION))

    def test_the_supervisor_targets_the_marker_when_frozen(self):
        import gui.job_supervisor as supervisor_module

        original = getattr(sys, "frozen", None)
        try:
            sys.frozen = True
            command = supervisor_module.worker_command("r.json")
            self.assertIn("--job-worker", command)
            self.assertIn("--request", command)
            self.assertNotIn("-m", command)
        finally:
            if original is None:
                del sys.frozen
            else:
                sys.frozen = original


class SupervisorReliabilityTests(unittest.TestCase):
    """Termination, cleanup and control-publish edges around a child job."""

    def _supervisor(self):
        request = build_request(
            input_path="a.mp4", output_path="b.mp4",
            config_payload={}, is_image=False,
        )
        supervisor = JobSupervisor(request)
        self.addCleanup(supervisor._cleanup_scratch)
        return supervisor

    def test_a_published_result_beats_the_wall_clock_timeout(self):
        """The child wrote its output and cleaned its checkpoint, then hung
        before stdout reached EOF. Reporting worker_timeout made the caller
        retry or fail an item that had already succeeded."""
        supervisor = self._supervisor()
        supervisor._timed_out = True
        supervisor._result = {
            "status": "complete",
            "success": True,
            "evidence": {"last_stage_timings": {"total": 1.0}},
        }

        outcome = supervisor._build_outcome(0)

        self.assertEqual(outcome.status, "complete")
        self.assertTrue(outcome.success)
        self.assertNotEqual(outcome.reason, "worker_timeout")
        self.assertEqual(outcome.evidence["last_stage_timings"], {"total": 1.0})

    def test_a_timeout_without_a_result_still_reports_the_timeout(self):
        supervisor = self._supervisor()
        supervisor._timed_out = True
        supervisor._result = None

        outcome = supervisor._build_outcome(None)

        self.assertEqual(outcome.reason, "worker_timeout")
        self.assertFalse(outcome.success)

    def test_scratch_cleanup_keeps_its_handle_when_removal_fails(self):
        """Dropping the handle after a sharing violation leaked the
        directory permanently, because nothing retried it."""
        supervisor = self._supervisor()
        owned = supervisor._owned_scratch
        self.assertIsNotNone(owned)
        calls = []

        def failing_cleanup():
            calls.append(1)
            if len(calls) == 1:
                raise OSError("being used by another process")

        owned.cleanup = failing_cleanup

        supervisor._cleanup_scratch()
        # Still held, so a later pass can try again.
        self.assertIsNotNone(supervisor._owned_scratch)

        supervisor._cleanup_scratch()
        self.assertIsNone(supervisor._owned_scratch)
        self.assertEqual(len(calls), 2)

    def test_scratch_cleanup_waits_for_a_live_child_before_removing(self):
        supervisor = self._supervisor()
        waited = []

        class LiveProcess:
            def poll(self):
                return None

            def wait(self, timeout=None):
                waited.append(timeout)
                return 0

        supervisor._process = LiveProcess()
        supervisor._cleanup_scratch()

        self.assertEqual(waited, [SCRATCH_REAP_TIMEOUT])

    def test_control_publish_does_not_recreate_a_cleaned_scratch_dir(self):
        """A stale watchdog publishing a control file used to resurrect the
        directory the supervisor had already removed."""
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "gone" / "control.json"
            with self.assertRaises(OSError):
                write_control_file(missing, cancel=True)
            self.assertFalse(missing.parent.exists())


class _RecordingJob:
    """Stand-in for the Windows job object, so the wiring is testable."""

    instances = []

    def __init__(self, assign_ok=True):
        self.assign_ok = assign_ok
        self.assigned = []
        self.terminated = 0
        self.closed = 0
        _RecordingJob.instances.append(self)

    @classmethod
    def create(cls):
        return cls()

    def assign_pid(self, pid):
        self.assigned.append(pid)
        return self.assign_ok

    def terminate(self, exit_code=1):
        self.terminated += 1
        return True

    def close(self):
        self.closed += 1


class JobContainmentTests(unittest.TestCase):
    """RM-209: the worker and everything it spawns die together."""

    def setUp(self):
        _RecordingJob.instances = []

    def _supervisor(self, command):
        request = build_request(
            input_path="a.mp4", output_path="b.mp4",
            config_payload={}, is_image=False,
        )
        supervisor = JobSupervisor(request, command=command)
        self.addCleanup(supervisor._cleanup_scratch)
        return supervisor

    def test_the_worker_is_put_in_a_job_and_the_job_is_released(self):
        supervisor = self._supervisor([sys.executable, "-c", "pass"])
        with unittest.mock.patch.object(
                job_supervisor, "ProcessJob", _RecordingJob):
            supervisor.run(timeout=30)

        self.assertEqual(len(_RecordingJob.instances), 1)
        job = _RecordingJob.instances[0]
        self.assertEqual(len(job.assigned), 1)
        self.assertIsInstance(job.assigned[0], int)
        # Released once the outcome is known, before the scratch removal.
        self.assertEqual(job.closed, 1)

    def test_a_job_that_cannot_hold_the_worker_is_closed_immediately(self):
        """A kill-on-close handle we failed to assign must not be kept: it
        would take the worker down when the supervisor is collected."""
        supervisor = self._supervisor([sys.executable, "-c", "pass"])

        class RefusingJob(_RecordingJob):
            @classmethod
            def create(cls):
                return cls(assign_ok=False)

        with unittest.mock.patch.object(
                job_supervisor, "ProcessJob", RefusingJob):
            supervisor.run(timeout=30)

        job = _RecordingJob.instances[0]
        self.assertEqual(job.closed, 1)
        self.assertIsNone(supervisor._job)

    def test_killing_a_stuck_worker_also_terminates_its_job(self):
        supervisor = self._supervisor(
            [sys.executable, "-c", "import time; time.sleep(30)"])
        with unittest.mock.patch.object(
                job_supervisor, "ProcessJob", _RecordingJob):
            outcome = supervisor.run(timeout=1.5)

        job = _RecordingJob.instances[0]
        self.assertEqual(outcome.reason, "worker_timeout")
        self.assertGreaterEqual(job.terminated, 1)
        self.assertEqual(job.closed, 1)


@unittest.skipUnless(sys.platform == "win32", "job objects are Windows only")
class ProcessJobTests(unittest.TestCase):
    """The real Windows API path, exercised against real processes."""

    @staticmethod
    def _alive(pid):
        import ctypes
        SYNCHRONIZE = 0x00100000
        handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if not handle:
            return False
        try:
            # WAIT_OBJECT_0 means the process is signaled, so it has exited.
            return ctypes.windll.kernel32.WaitForSingleObject(handle, 0) != 0
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)

    def test_terminating_the_job_reaps_a_grandchild(self):
        from gui.process_job import ProcessJob

        job = ProcessJob.create()
        self.assertIsNotNone(job, "expected a job object on Windows")
        child = subprocess.Popen(
            [sys.executable, "-c",
             "import subprocess, sys, time;"
             "p = subprocess.Popen([sys.executable, '-c',"
             " 'import time; time.sleep(60)']);"
             "print(p.pid, flush=True); time.sleep(60)"],
            stdout=subprocess.PIPE, text=True,
        )
        self.addCleanup(child.kill)
        self.assertTrue(job.assign_pid(child.pid))
        grandchild = int(child.stdout.readline().strip())
        self.addCleanup(job.close)
        self.assertTrue(self._alive(grandchild))

        self.assertTrue(job.terminate())

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and self._alive(grandchild):
            time.sleep(0.05)
        self.assertFalse(
            self._alive(grandchild),
            "the ffmpeg-equivalent grandchild outlived the job")
        self.assertIsNotNone(child.poll() or child.wait(timeout=5))

    def test_closing_the_job_kills_what_is_left_in_it(self):
        """The case no cleanup code covers: the GUI process itself dying."""
        from gui.process_job import ProcessJob

        job = ProcessJob.create()
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"])
        self.addCleanup(child.kill)
        self.assertTrue(job.assign_pid(child.pid))

        job.close()

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and self._alive(child.pid):
            time.sleep(0.05)
        self.assertFalse(self._alive(child.pid))
        self.assertFalse(job.active)

    def test_a_released_job_reports_nothing_to_do(self):
        from gui.process_job import ProcessJob

        job = ProcessJob.create()
        job.close()
        self.assertFalse(job.terminate())
        self.assertFalse(job.assign_pid(os.getpid()))
        job.close()


if __name__ == "__main__":
    unittest.main()
