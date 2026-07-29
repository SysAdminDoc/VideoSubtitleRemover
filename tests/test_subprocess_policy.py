import subprocess
import sys
import threading
import time
import unittest
from unittest import mock

from backend import subprocess_policy as policy


class SubprocessPolicyTests(unittest.TestCase):
    def test_windows_launch_is_hidden_non_shell_and_has_closed_stdin(self):
        fake = mock.Mock()
        with mock.patch.object(policy.os, "name", "nt"):
            with mock.patch.object(policy.subprocess, "Popen", return_value=fake) as popen:
                self.assertIs(policy.popen_process(["tool"]), fake)

        kwargs = popen.call_args.kwargs
        self.assertFalse(kwargs["shell"])
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        self.assertTrue(kwargs["close_fds"])
        self.assertTrue(
            kwargs["creationflags"] & policy.WINDOWS_CREATE_NO_WINDOW
        )

    def test_existing_creation_flags_are_preserved(self):
        fake = mock.Mock()
        with mock.patch.object(policy.os, "name", "nt"):
            with mock.patch.object(policy.subprocess, "Popen", return_value=fake) as popen:
                policy.popen_process(["tool"], creationflags=0x20)
        self.assertEqual(
            popen.call_args.kwargs["creationflags"],
            0x20 | policy.WINDOWS_CREATE_NO_WINDOW,
        )

    def test_shell_launches_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "shell=True"):
            policy.popen_process("echo unsafe", shell=True)

    def test_capture_is_drained_and_bounded(self):
        result = policy.run_process(
            [
                sys.executable,
                "-c",
                "import sys; sys.stdout.write('A' * 4096 + 'END')",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            max_output_bytes=128,
            check=True,
        )
        self.assertLessEqual(len(result.stdout), 128)
        self.assertTrue(result.stdout.endswith("END"))
        self.assertEqual(result.stderr, "")

    def test_timeout_terminates_without_waiting_for_child_budget(self):
        started = time.monotonic()
        with self.assertRaises(subprocess.TimeoutExpired):
            policy.run_process(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                capture_output=True,
                timeout=0.1,
            )
        self.assertLess(time.monotonic() - started, 4.0)

    def test_cancel_notifies_owner_and_cleans_up(self):
        active = []
        with self.assertRaisesRegex(InterruptedError, "cancelled"):
            policy.run_process(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                capture_output=True,
                timeout=10,
                cancel_check=lambda: True,
                on_process=active.append,
            )
        self.assertEqual(len(active), 2)
        self.assertIsNotNone(active[0])
        self.assertIsNone(active[1])
        self.assertIsNotNone(active[0].poll())

    def test_stdin_write_cannot_outlive_the_timeout(self):
        """RM-142: a child that never reads stdin must still time out."""
        # 16 MiB overflows every OS pipe buffer, so the write would block
        # forever if it ran inline before the deadline loop.
        payload = "x" * (16 * 1024 * 1024)
        started = time.monotonic()
        with self.assertRaises(subprocess.TimeoutExpired):
            policy.run_process(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                input=payload,
                capture_output=True,
                text=True,
                timeout=0.5,
            )
        self.assertLess(time.monotonic() - started, 15.0)
        self.assertFalse([
            thread for thread in threading.enumerate()
            if thread.name == "vsr-subprocess-stdin" and thread.is_alive()
        ])

    def test_stdin_write_cancels_with_the_child(self):
        payload = "y" * (16 * 1024 * 1024)
        active = []
        with self.assertRaisesRegex(InterruptedError, "cancelled"):
            policy.run_process(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                input=payload,
                capture_output=True,
                text=True,
                timeout=30,
                cancel_check=lambda: True,
                on_process=active.append,
            )
        self.assertIsNotNone(active[0].poll())
        self.assertFalse([
            thread for thread in threading.enumerate()
            if thread.name == "vsr-subprocess-stdin" and thread.is_alive()
        ])

    def test_large_stdin_payload_still_round_trips(self):
        payload = "z" * (4 * 1024 * 1024)
        result = policy.run_process(
            [
                sys.executable,
                "-c",
                "import sys; data = sys.stdin.read(); "
                "sys.stdout.write(str(len(data)))",
            ],
            input=payload,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        self.assertEqual(result.stdout.strip(), str(len(payload)))

    def test_child_that_exits_before_reading_stdin_does_not_raise(self):
        # A child closing stdin early used to surface BrokenPipeError from the
        # inline write; the captured output is what matters.
        result = policy.run_process(
            [sys.executable, "-c", "import sys; sys.stdout.write('done')"],
            input="q" * (8 * 1024 * 1024),
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.stdout, "done")
        self.assertEqual(result.returncode, 0)

    def test_terminate_escalates_to_kill(self):
        proc = mock.Mock()
        proc.poll.return_value = None
        proc.wait.side_effect = [subprocess.TimeoutExpired(["tool"], 0.1), 1]
        policy.terminate_process(proc, timeout=0.1)
        proc.terminate.assert_called_once_with()
        proc.kill.assert_called_once_with()
        self.assertEqual(proc.wait.call_count, 2)


if __name__ == "__main__":
    unittest.main()
