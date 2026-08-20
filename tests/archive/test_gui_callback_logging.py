"""Tests for GUI Tk callback exception logging.

Unhandled exceptions raised inside Tk ``.after`` / event callbacks must be
logged with a traceback (not silently dumped to stderr), while teardown races
(TclError / RuntimeError once the app is being destroyed) stay at debug.
"""

from __future__ import annotations

import sys
import tkinter as tk
import unittest

from gui.app import VideoSubtitleRemoverApp


class _Dummy:
    """Stand-in for the app instance; the handler only uses the module logger."""


def _traceback_for(error: Exception):
    try:
        raise error
    except Exception:
        return sys.exc_info()[2]


class CallbackLoggingTests(unittest.TestCase):
    def _invoke(self, exc_type, error):
        tb = _traceback_for(error)
        with self.assertLogs("gui.app", level="DEBUG") as captured:
            VideoSubtitleRemoverApp._log_callback_exception(
                _Dummy(), exc_type, error, tb)
        return captured.output

    def test_generic_exception_logged_as_error_with_traceback(self):
        output = self._invoke(ValueError, ValueError("boom"))
        joined = "\n".join(output)
        self.assertIn("Unhandled Tk callback exception", joined)
        self.assertTrue(any(line.startswith("ERROR") for line in output))
        # exc_info=True renders the traceback into the captured record.
        self.assertIn("ValueError: boom", joined)

    def test_teardown_race_logged_as_debug(self):
        output = self._invoke(tk.TclError, tk.TclError("application destroyed"))
        joined = "\n".join(output)
        self.assertIn("teardown", joined)
        self.assertTrue(any(line.startswith("DEBUG") for line in output))
        self.assertFalse(any(line.startswith("ERROR") for line in output))

    def test_runtime_error_treated_as_teardown(self):
        output = self._invoke(RuntimeError, RuntimeError("main thread is not in main loop"))
        self.assertTrue(any(line.startswith("DEBUG") for line in output))


if __name__ == "__main__":
    unittest.main()
