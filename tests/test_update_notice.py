"""RM-338: an available update has to reach the release page.

The notice was a 2.6 second toast plus a line in the log file telling the
user the link was "in the log panel", and the whole handler swallowed every
exception. The release URL was already in hand.
"""

from __future__ import annotations

import unittest
from unittest import mock

from backend.update_check import (
    RELEASE_NOTES_MAX_CHARS,
    check_for_update,
    summarize_release_notes,
)


class ReleaseNotesSummaryTests(unittest.TestCase):
    def test_markdown_decoration_is_stripped(self):
        summary = summarize_release_notes(
            "## What's new\n\n"
            "- Fixed **image I/O** on non-ASCII paths\n"
            "- Moved the [NVIDIA lane](https://example.com/x) to CUDA 13\n"
        )
        self.assertIn("Fixed image I/O on non-ASCII paths", summary)
        self.assertIn("Moved the NVIDIA lane to CUDA 13", summary)
        for marker in ("#", "**", "[", "](", "- "):
            self.assertNotIn(marker, summary)

    def test_an_empty_or_missing_body_is_an_empty_summary(self):
        for value in ("", None, "   \n\n", "## Heading only\n"):
            with self.subTest(value=value):
                self.assertEqual(summarize_release_notes(value), "")

    def test_a_long_body_is_truncated_rather_than_pasted_whole(self):
        summary = summarize_release_notes("word " * 500)
        self.assertLessEqual(len(summary), RELEASE_NOTES_MAX_CHARS)
        self.assertTrue(summary.endswith("\u2026"))

    def test_tables_and_quotes_are_left_out(self):
        summary = summarize_release_notes(
            "> quoted line\n"
            "| a | b |\n"
            "--- \n"
            "Real prose here.\n"
        )
        self.assertEqual(summary, "Real prose here.")


class UpdateCallbackTests(unittest.TestCase):
    def _response(self, payload):
        import io
        import json

        body = json.dumps(payload).encode("utf-8")

        class _Resp(io.BytesIO):
            headers: dict = {}

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return _Resp(body)

    def test_the_callback_receives_the_notes_with_the_url(self):
        seen = []
        payload = {
            "tag_name": "v4.0.0",
            "html_url": "https://example.invalid/releases/v4.0.0",
            "body": "- Something worth reading\n",
        }
        with mock.patch("backend.update_check.urlopen",
                        return_value=self._response(payload)):
            with mock.patch("backend.update_check._save_state"):
                thread = check_for_update(
                    "3.16.1", lambda *args: seen.append(args))
                if thread is not None:
                    thread.join(30)

        self.assertEqual(len(seen), 1)
        tag, url, notes = seen[0]
        self.assertEqual(tag, "v4.0.0")
        self.assertEqual(url, "https://example.invalid/releases/v4.0.0")
        self.assertEqual(notes, "Something worth reading")


class _FakeLabel:
    def __init__(self):
        self.kwargs = {}
        self.mapped = False

    def config(self, **kwargs):
        self.kwargs.update(kwargs)

    def pack(self, **_kwargs):
        self.mapped = True

    def pack_forget(self):
        self.mapped = False

    def winfo_ismapped(self):
        return self.mapped


class _FakeButton(_FakeLabel):
    def __init__(self):
        super().__init__()
        self.enabled = None

    def set_enabled(self, value):
        self.enabled = bool(value)


class UpdateNoticeTests(unittest.TestCase):
    """The handler's behaviour, without driving tkinter."""

    def _app(self):
        from gui.app import VideoSubtitleRemoverApp

        app = VideoSubtitleRemoverApp.__new__(VideoSubtitleRemoverApp)
        app.root = object()
        app._update_banner = _FakeLabel()
        app._update_banner_title = _FakeLabel()
        app._update_banner_notes = _FakeLabel()
        app._update_banner_open_btn = _FakeButton()
        app._update_banner_dismiss_btn = _FakeButton()
        app.statuses = []
        app._update_status = lambda text, tone="info": app.statuses.append(
            (text, tone))
        return app

    def _show(self, app, *args, **kwargs):
        with mock.patch("gui.app.Toast") as toast:
            toast.show.return_value = None
            app._show_update_toast(*args, **kwargs)

    def test_the_notice_is_shown_and_names_the_version(self):
        app = self._app()
        self._show(app, "v4.0.0", "https://example.invalid/r/4",
                   "Two things changed.")

        self.assertTrue(app._update_banner.mapped)
        self.assertIn("v4.0.0", app._update_banner_title.kwargs["text"])
        self.assertEqual(
            app._update_banner_notes.kwargs["text"], "Two things changed.")
        self.assertTrue(app._update_banner_notes.mapped)
        self.assertTrue(app._update_banner_open_btn.enabled)
        # It must not tell the user to go and read the log file.
        self.assertNotIn("log", app._update_banner_title.kwargs["text"].lower())

    def test_a_release_with_no_notes_hides_the_summary_line(self):
        app = self._app()
        self._show(app, "v4.0.0", "https://example.invalid/r/4", "")
        self.assertFalse(app._update_banner_notes.mapped)
        self.assertEqual(app._update_banner_notes.kwargs["text"], "")

    def test_the_notice_is_dismissible_and_stays_dismissed(self):
        app = self._app()
        self._show(app, "v4.0.0", "https://example.invalid/r/4", "notes")
        self.assertTrue(app._update_banner.mapped)
        app._dismiss_update_banner()
        self.assertFalse(app._update_banner.mapped)

    def test_the_control_opens_the_release_page_in_a_browser(self):
        app = self._app()
        self._show(app, "v4.0.0", "https://example.invalid/r/4", "")
        with mock.patch("webbrowser.open") as opener:
            app._open_update_release_page()
        opener.assert_called_once_with("https://example.invalid/r/4", new=2)

    def test_nothing_is_downloaded_or_installed(self):
        """The only outward action is handing a URL to the browser."""
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        source = (root / "gui" / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        target = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_open_update_release_page"
        )
        body = ast.dump(target)
        for forbidden in ("urlretrieve", "urlopen", "Popen", "startfile"):
            self.assertNotIn(forbidden, body)
        self.assertIn("webbrowser", body)

    def test_a_non_https_url_is_refused(self):
        app = self._app()
        self._show(app, "v4.0.0", "file:///C:/Windows/System32/cmd.exe", "")
        with mock.patch("webbrowser.open") as opener:
            app._open_update_release_page()
        opener.assert_not_called()
        self.assertTrue(app.statuses)

    def test_a_failure_is_logged_rather_than_swallowed(self):
        app = self._app()

        import tkinter as tk

        def _boom(**_kwargs):
            # The clause is narrowed to what a destroyed widget actually
            # raises, so the fixture has to raise that.
            raise tk.TclError("widget gone")

        app._update_banner_title.config = _boom
        with self.assertLogs("gui.app", level="WARNING") as logs:
            self._show(app, "v4.0.0", "https://example.invalid/r/4", "")
        self.assertTrue(
            any("Update notice" in line for line in logs.output), logs.output)
        # And it still tells the user, on the surface that survived.
        self.assertTrue(app.statuses)

    def test_a_browser_failure_is_reported(self):
        app = self._app()
        self._show(app, "v4.0.0", "https://example.invalid/r/4", "")
        with mock.patch("webbrowser.open", side_effect=OSError("no browser")):
            with self.assertLogs("gui.app", level="WARNING"):
                app._open_update_release_page()
        self.assertTrue(app.statuses)


if __name__ == "__main__":
    unittest.main()
