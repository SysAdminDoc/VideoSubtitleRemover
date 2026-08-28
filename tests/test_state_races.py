"""RM-314: two processes must not race over the shared per-user state.

`settings.json`, `queue_state.json`, and `presets.json` are one set of files
per user, and every launch reads and writes them. The writes are atomic, so
nobody sees a torn file, but interleaved read-modify-write still drops the
newer state, and a process-local `threading.Lock` cannot see the other
process. These tests run real child processes.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _bump_counter(lock_path: str, counter_path: str, use_lock: bool) -> None:
    """Read a counter, wait, then write it back one higher."""
    sys.path.insert(0, str(REPO_ROOT))
    from gui.state_lock import state_file_lock

    def _work():
        with open(counter_path, encoding="utf-8") as handle:
            value = json.load(handle)["count"]
        # Widen the window the lock has to close.
        time.sleep(0.05)
        tmp = counter_path + f".{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump({"count": value + 1}, handle)
        os.replace(tmp, counter_path)

    if use_lock:
        with state_file_lock(Path(lock_path)):
            _work()
        return
    try:
        _work()
    except OSError:
        # Unlocked writers also collide inside os.replace on Windows. That
        # is another way to lose the update, not a test harness failure.
        pass


def _save_queue_records(state_dir: str, marker: str, rounds: int) -> None:
    """Round-trip the real queue-state writer from a child process."""
    sys.path.insert(0, str(REPO_ROOT))
    from gui import config as gcfg

    state = Path(state_dir)
    gcfg.QUEUE_STATE_FILE = state / "queue_state.json"
    gcfg.STATE_LOCK_FILE = state / "state.lock"

    config = gcfg.ProcessingConfig()
    for index in range(rounds):
        items = [
            gcfg.QueueItem(
                id=f"{marker}-{index}-{n}",
                file_path=f"{marker}-{index}-{n}.mp4",
                output_path=f"{marker}-{index}-{n}_clean.mp4",
                config=config,
            )
            for n in range(4)
        ]
        result = gcfg.save_queue_state(items)
        if not result.ok:
            raise AssertionError(f"queue state save failed: {result.error}")
        loaded = gcfg.load_queue_state()
        if loaded is None:
            raise AssertionError("queue state read back as unusable")


def _save_preset(state_dir: str, name: str, ready_path: str,
                 wait_for: str) -> None:
    """Save one user preset through the real public entry point."""
    sys.path.insert(0, str(REPO_ROOT))
    from gui import config as gcfg

    state = Path(state_dir)
    gcfg.PRESETS_FILE = state / "presets.json"
    gcfg.STATE_LOCK_FILE = state / "state.lock"

    Path(ready_path).write_text("ready", encoding="utf-8")
    deadline = time.monotonic() + 60
    while wait_for and not Path(wait_for).exists():
        if time.monotonic() > deadline:
            raise AssertionError("peer never signalled")
        time.sleep(0.02)

    config = gcfg.ProcessingConfig()
    if not gcfg.save_user_preset(name, "cross-process", config):
        raise AssertionError(f"preset {name} was not saved")


def _claim_instance(lock_path: str, result_path: str, hold_seconds: float) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from gui import single_instance

    if os.name == "nt":
        token = Path(lock_path).parent.name
        guard = single_instance.acquire(name=f"Local\\VSRTest.{token}")
    else:
        guard = single_instance.acquire(lock_path=Path(lock_path))
    Path(result_path).write_text(
        json.dumps({"already_running": guard.already_running}),
        encoding="utf-8",
    )
    time.sleep(hold_seconds)
    guard.release()


class CrossProcessStateLockTests(unittest.TestCase):
    WORKERS = 6

    def _run_counter(self, use_lock: bool) -> int:
        ctx = mp.get_context("spawn")
        with tempfile.TemporaryDirectory() as tmpdir:
            counter = Path(tmpdir) / "counter.json"
            counter.write_text(json.dumps({"count": 0}), encoding="utf-8")
            lock = Path(tmpdir) / "state.lock"
            procs = [
                ctx.Process(
                    target=_bump_counter,
                    args=(str(lock), str(counter), use_lock),
                )
                for _ in range(self.WORKERS)
            ]
            for proc in procs:
                proc.start()
            for proc in procs:
                proc.join(120)
                self.assertEqual(proc.exitcode, 0)
            return json.loads(counter.read_text(encoding="utf-8"))["count"]

    def test_lock_keeps_every_concurrent_update(self):
        self.assertEqual(self._run_counter(use_lock=True), self.WORKERS)

    def test_unlocked_control_loses_updates(self):
        # Proves the locked case above is measuring the lock and not an
        # accidentally serial test.
        self.assertLess(self._run_counter(use_lock=False), self.WORKERS)

    def test_reentrant_within_one_thread(self):
        from gui.state_lock import close_lock_handles, state_file_lock

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                lock = Path(tmpdir) / "state.lock"
                with state_file_lock(lock) as outer:
                    self.assertTrue(outer)
                    with state_file_lock(lock, timeout=0.1) as inner:
                        self.assertTrue(inner)
                # The handle is cached for the life of the process, so the
                # temp directory cannot be removed until it is closed.
                close_lock_handles()
        finally:
            close_lock_handles()


class ConcurrentQueueStateTests(unittest.TestCase):
    WORKERS = 4
    ROUNDS = 5

    def test_queue_state_stays_valid_under_concurrent_writers(self):
        ctx = mp.get_context("spawn")
        with tempfile.TemporaryDirectory() as tmpdir:
            procs = [
                ctx.Process(
                    target=_save_queue_records,
                    args=(tmpdir, f"w{index}", self.ROUNDS),
                )
                for index in range(self.WORKERS)
            ]
            for proc in procs:
                proc.start()
            for proc in procs:
                proc.join(180)
                self.assertEqual(proc.exitcode, 0)

            state = Path(tmpdir) / "queue_state.json"
            self.assertTrue(state.is_file())
            payload = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], 5)
            self.assertEqual(len(payload["items"]), 4)
            # A partially applied write would leave records from two writers
            # mixed into one file.
            markers = {
                item["file_path"].split("-")[0]
                for item in payload["items"]
            }
            self.assertEqual(len(markers), 1)
            # No quarantine file: nobody ever read a state they could not use.
            self.assertEqual(
                sorted(p.name for p in Path(tmpdir).glob("*.corrupt-*")), [])


class ConcurrentPresetTests(unittest.TestCase):
    """RM-314: the preset read-modify-write must not straddle the lock.

    Holding the lock only for the write keeps presets.json valid but still
    loses a peer's preset that landed between this process's read and its
    write, which is the exact failure the lock exists to remove.
    """

    WORKERS = 4

    def test_every_concurrent_preset_survives(self):
        ctx = mp.get_context("spawn")
        with tempfile.TemporaryDirectory() as tmpdir:
            state = Path(tmpdir)
            (state / "presets.json").write_text("{}", encoding="utf-8")
            ready = [state / f"ready_{i}" for i in range(self.WORKERS)]
            go = state / "go"
            procs = [
                ctx.Process(
                    target=_save_preset,
                    args=(tmpdir, f"peer_{index}", str(ready[index]), str(go)),
                )
                for index in range(self.WORKERS)
            ]
            for proc in procs:
                proc.start()
            # Release them together so their read-modify-writes really overlap.
            deadline = time.monotonic() + 60
            while not all(path.exists() for path in ready):
                if time.monotonic() > deadline:
                    self.fail("a preset worker never started")
                time.sleep(0.02)
            go.write_text("go", encoding="utf-8")
            for proc in procs:
                proc.join(120)
                self.assertEqual(proc.exitcode, 0)

            saved = json.loads(
                (state / "presets.json").read_text(encoding="utf-8"))
            self.assertEqual(
                sorted(saved), [f"peer_{i}" for i in range(self.WORKERS)])


class EntryPointSlotOwnershipTests(unittest.TestCase):
    """RM-314: the slot must be held across app construction.

    Acquiring, releasing, and letting the app re-acquire leaves a window of
    tens of milliseconds where a second launch during a cold start sees the
    slot free, and both processes then read and write the same state files.
    """

    def test_the_slot_is_not_released_before_the_app_is_built(self):
        from unittest import mock

        import VideoSubtitleRemover as entry
        from gui import single_instance

        events = []

        class _Guard:
            already_running = False

            def release(self):
                events.append("released")

        class _App:
            def __init__(self, *, instance_guard=None):
                events.append(("constructed", instance_guard is not None))

            def run(self):
                events.append("ran")

        with mock.patch.object(sys, "argv", ["VideoSubtitleRemover.py"]):
            with mock.patch.object(
                    single_instance, "acquire", return_value=_Guard()):
                with mock.patch.object(entry, "VideoSubtitleRemoverApp", _App):
                    entry.main()

        self.assertEqual(
            events, [("constructed", True), "ran", "released"])

    def test_a_second_launch_exits_before_the_app_is_built(self):
        from unittest import mock

        import VideoSubtitleRemover as entry
        from gui import single_instance

        built = []

        class _Guard:
            already_running = True

            def release(self):
                pass

        class _App:
            def __init__(self, *, instance_guard=None):
                built.append(instance_guard)

            def run(self):
                built.append("ran")

        with mock.patch.object(sys, "argv", ["VideoSubtitleRemover.py"]):
            with mock.patch.object(
                    single_instance, "acquire", return_value=_Guard()):
                with mock.patch.object(entry, "VideoSubtitleRemoverApp", _App):
                    with self.assertRaises(SystemExit) as ctx:
                        entry.main()

        self.assertEqual(ctx.exception.code, 3)
        self.assertEqual(built, [])


class SecondInstanceTests(unittest.TestCase):
    def test_second_process_sees_the_first_and_writes_nothing(self):
        ctx = mp.get_context("spawn")
        with tempfile.TemporaryDirectory() as tmpdir:
            lock = Path(tmpdir) / "instance.lock"
            first_result = Path(tmpdir) / "first.json"
            second_result = Path(tmpdir) / "second.json"

            first = ctx.Process(
                target=_claim_instance,
                args=(str(lock), str(first_result), 6.0),
            )
            first.start()
            try:
                deadline = time.monotonic() + 30
                while not first_result.exists():
                    if time.monotonic() > deadline:
                        self.fail("the first instance never reported")
                    time.sleep(0.05)
                self.assertFalse(
                    json.loads(first_result.read_text(encoding="utf-8"))
                    ["already_running"])

                second = ctx.Process(
                    target=_claim_instance,
                    args=(str(lock), str(second_result), 0.0),
                )
                second.start()
                second.join(60)
                self.assertEqual(second.exitcode, 0)
                self.assertTrue(
                    json.loads(second_result.read_text(encoding="utf-8"))
                    ["already_running"])
            finally:
                first.join(60)

    @unittest.skipUnless(sys.platform == "win32", "named mutex is Windows only")
    def test_entry_point_refuses_a_second_launch(self):
        script = textwrap.dedent(
            """
            import json, os, sys, time
            sys.path.insert(0, sys.argv[1])
            from gui import single_instance
            guard = single_instance.acquire()
            print(json.dumps({"already": guard.already_running}), flush=True)
            time.sleep(float(sys.argv[2]))
            guard.release()
            """
        )
        holder = subprocess.Popen(
            [sys.executable, "-c", script, str(REPO_ROOT), "8"],
            stdout=subprocess.PIPE, text=True,
        )
        try:
            first_line = holder.stdout.readline()
            if json.loads(first_line)["already"]:
                self.skipTest("a real instance already holds the mutex")

            with tempfile.TemporaryDirectory() as appdata:
                env = dict(os.environ)
                # Point every per-user state file at a throwaway directory,
                # and keep the window hidden, so a regression that lets the
                # second launch through cannot touch real state or the
                # user's desktop.
                env["APPDATA"] = appdata
                env["VSR_UI_BACKGROUND"] = "1"
                second = subprocess.run(
                    [sys.executable, str(REPO_ROOT / "VideoSubtitleRemover.py")],
                    capture_output=True, text=True, timeout=90, env=env,
                    cwd=str(REPO_ROOT),
                )
                self.assertEqual(second.returncode, 3)
                self.assertIn("already running", second.stderr.lower())
                # Nothing at all may land in the per-user directory: the
                # log file there belongs to the running instance, and a second
                # process appending to it can trip its rotation.
                self.assertEqual(
                    sorted(p.name for p in Path(appdata).rglob("*")
                           if p.is_file()),
                    [])
        finally:
            holder.wait(timeout=60)


if __name__ == "__main__":
    unittest.main()
