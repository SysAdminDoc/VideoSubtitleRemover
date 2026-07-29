"""RM-145: opt-in crash reporting must fail closed on privacy."""

import json
from pathlib import Path
import sys
import unittest
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend import crash_reporter


HOME = "C:\\Users\\bob"
SECRET_TOKENS = (
    "bob", "NAS01", "clientX", "family holiday", "confidential",
    "huge-array", "Hello subtitle text", "session-cookie",
)


class PathScrubTests(unittest.TestCase):
    def _assert_clean(self, text):
        scrubbed = crash_reporter._path_scrub(text)
        for token in SECRET_TOKENS:
            self.assertNotIn(token, scrubbed, f"{token!r} survived in {scrubbed!r}")
        return scrubbed

    def test_windows_drive_path_including_basename(self):
        scrubbed = self._assert_clean(HOME + "\\videos\\family holiday.mp4")
        self.assertIn("<path>", scrubbed)

    def test_forward_slash_windows_path(self):
        self._assert_clean("C:/Users/bob/videos/family holiday.mkv")

    def test_unc_share_path(self):
        scrubbed = self._assert_clean(
            "opened \\\\NAS01\\media\\clientX\\confidential.mkv here")
        self.assertIn("<path>", scrubbed)
        self.assertIn("opened", scrubbed)
        self.assertIn("here", scrubbed)

    def test_file_url(self):
        self._assert_clean(
            "file:///C:/Users/bob/family holiday.mp4 could not be read")

    def test_posix_home_and_temp_paths(self):
        self._assert_clean("/home/bob/Videos/confidential.mkv missing")
        self._assert_clean("/tmp/vsr-abc/clientX.png missing")

    def test_bare_basename_is_redacted(self):
        scrubbed = self._assert_clean(
            "Could not open family holiday.mp4 for writing")
        self.assertIn("<file>", scrubbed)

    def test_traceback_line_keeps_structure(self):
        scrubbed = crash_reporter._path_scrub(
            'File "' + HOME + '\\repos\\VSR\\backend\\processor.py", line 12')
        self.assertNotIn("bob", scrubbed)
        self.assertIn("line 12", scrubbed)

    def test_support_bundle_mode_keeps_the_leaf_name(self):
        # The local support bundle is reviewed by the user before sharing, so
        # it strips the directory tree but keeps the filename. Telemetry does
        # not get that allowance.
        text = HOME + "\\videos\\source.mp4"
        lenient = crash_reporter._path_scrub(text, keep_basename=True)
        self.assertIn("source.mp4", lenient)
        self.assertNotIn("bob", lenient)
        self.assertNotIn(
            "source.mp4", crash_reporter._path_scrub(text))

    def test_ordinary_text_is_untouched(self):
        text = "Decoder returned 3 frames but 10 were expected"
        self.assertEqual(crash_reporter._path_scrub(text), text)


def _rich_event():
    return {
        "event_id": "abc123",
        "timestamp": 1234.5,
        "platform": "python",
        "level": "error",
        "release": "3.29.0",
        "server_name": "BOBS-DESKTOP",
        "message": HOME + "\\family holiday.mp4",
        "logger": "backend.processor",
        "exception": {"values": [{
            "type": "ValueError",
            "value": "Cannot open " + HOME + "\\confidential.mkv",
            "module": "backend.processor",
            "mechanism": {"type": "excepthook", "handled": False},
            "stacktrace": {"frames": [{
                "abs_path": HOME + "\\app.py",
                "filename": "app.py",
                "module": "gui.app",
                "function": "run",
                "lineno": 42,
                "in_app": True,
                "vars": {"frame": "huge-array", "ocr": "Hello subtitle text"},
                "context_line": "open(r'" + HOME + "\\confidential.mkv')",
            }]},
        }]},
        "breadcrumbs": {"values": [
            {"message": "read " + HOME + "\\confidential.srt"},
            {"message": "OCR: Hello subtitle text"},
        ]},
        "extra": {"path": HOME + "\\out.mp4", "ocr": "Hello subtitle text"},
        "request": {
            "url": "https://example/upload",
            "cookies": "session-cookie",
            "headers": {"Authorization": "Bearer session-cookie"},
        },
        "user": {"id": "bob", "username": "bob", "ip_address": "10.0.0.5"},
        "tags": {"machine": "NAS01"},
        "modules": {"numpy": "2.2.6"},
        "contexts": {
            "runtime": {"name": "CPython", "version": "3.12.10"},
            "os": {"name": "Windows", "version": "10.0.26100"},
            "device": {"name": "BOBS-DESKTOP"},
        },
    }


class MinimalEventTests(unittest.TestCase):
    def test_only_allowlisted_fields_survive(self):
        out = crash_reporter._before_send(_rich_event(), {})
        self.assertIsNotNone(out)
        self.assertEqual(
            set(out),
            {
                "event_id", "timestamp", "platform", "level", "logger",
                "release", "environment", "contexts", "exception",
            },
        )
        blob = json.dumps(out)
        for token in SECRET_TOKENS:
            self.assertNotIn(token, blob, f"{token!r} leaked: {blob}")
        for banned in (
            "breadcrumbs", "extra", "request", "user", "tags", "modules",
            "server_name", "message", "vars", "abs_path", "filename",
            "context_line", "device",
        ):
            self.assertNotIn(banned, blob, f"{banned} leaked")

    def test_useful_diagnostics_are_retained(self):
        out = crash_reporter._before_send(_rich_event(), {})
        value = out["exception"]["values"][0]
        self.assertEqual(value["type"], "ValueError")
        self.assertEqual(value["module"], "backend.processor")
        self.assertIn("Cannot open", value["value"])
        frame = value["stacktrace"]["frames"][0]
        self.assertEqual(frame["module"], "gui.app")
        self.assertEqual(frame["function"], "run")
        self.assertEqual(frame["lineno"], 42)
        self.assertEqual(out["release"], "3.29.0")
        self.assertEqual(out["contexts"]["runtime"]["version"], "3.12.10")

    def test_scrubber_failure_drops_the_report(self):
        with mock.patch.object(
            crash_reporter, "build_minimal_event",
            side_effect=RuntimeError("boom"),
        ):
            self.assertIsNone(crash_reporter._before_send(_rich_event(), {}))

    def test_non_dict_event_is_dropped(self):
        self.assertIsNone(crash_reporter._before_send(["not", "a", "dict"], {}))

    def test_hostile_field_types_do_not_leak(self):
        event = {
            "event_id": HOME,
            "logger": HOME + "\\confidential.mkv",
            "platform": HOME,
            "exception": {"values": [{
                "type": "Cannot open " + HOME,
                "module": HOME,
                "stacktrace": {"frames": [{
                    "module": HOME,
                    "function": "open(" + HOME + ")",
                    "lineno": "not-an-int",
                }]},
            }]},
        }
        out = crash_reporter._before_send(event, {})
        blob = json.dumps(out)
        self.assertNotIn("bob", blob)
        self.assertNotIn("Users", blob)

    def test_frame_count_is_capped(self):
        frames = [
            {"module": "m", "function": "f", "lineno": i, "in_app": True}
            for i in range(crash_reporter.MAX_FRAMES + 25)
        ]
        event = {"exception": {"values": [{
            "type": "ValueError", "value": "x",
            "stacktrace": {"frames": frames},
        }]}}
        out = crash_reporter._before_send(event, {})
        self.assertEqual(
            len(out["exception"]["values"][0]["stacktrace"]["frames"]),
            crash_reporter.MAX_FRAMES,
        )

    def test_long_exception_value_is_truncated(self):
        event = {"exception": {"values": [{
            "type": "ValueError",
            "value": "x" * 5000,
        }]}}
        out = crash_reporter._before_send(event, {})
        self.assertLessEqual(
            len(out["exception"]["values"][0]["value"]),
            crash_reporter.MAX_VALUE_CHARS,
        )


class InitContractTests(unittest.TestCase):
    def test_sdk_is_initialised_without_pii_collection(self):
        import types

        captured = {}
        fake_sdk = types.SimpleNamespace(
            init=lambda **kwargs: captured.update(kwargs))
        with mock.patch.dict(sys.modules, {"sentry_sdk": fake_sdk}):
            with mock.patch.dict("os.environ", {
                "VSR_CRASH_REPORTS": "1",
                "VSR_GLITCHTIP_DSN": "https://key@example/0",
            }):
                crash_reporter._INSTALLED = False
                try:
                    self.assertTrue(crash_reporter.install())
                finally:
                    crash_reporter._INSTALLED = False
        self.assertIs(captured["send_default_pii"], False)
        self.assertEqual(captured["max_breadcrumbs"], 0)
        self.assertIs(captured["include_local_variables"], False)
        self.assertEqual(captured["traces_sample_rate"], 0.0)
        self.assertIs(captured["before_send"], crash_reporter._before_send)


if __name__ == "__main__":
    unittest.main()
