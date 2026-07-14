import subprocess
import sys
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
