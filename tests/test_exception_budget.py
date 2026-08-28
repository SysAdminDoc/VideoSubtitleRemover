"""RM-326: the blind-exception count is frozen and can only go down.

Shipping code carried hundreds of broad `except Exception` blocks with
nothing holding the number, and at least one of them silently shortened the
diagnostics the error path later reported from. The lint that finds them
cannot be added to `select` while the count is this high, because the
release gate runs that command and would fail, so the count is frozen here
instead: a new blind except fails this test, and removing one requires
lowering the number on purpose.

Lower these when you remove sites. Never raise them.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Measured on 2026-08-28. Every entry is a place where a failure is caught
# and not re-raised; each one is a place the product could be hiding
# something from the user.
BUDGET = {
    "BLE001": 537,
    # RM-338 narrowed one; lowered on purpose, never raised.
    "S110": 126,
}

# These have no budget: the count is zero and must stay there.
FORBIDDEN = ("B904", "S324")

SCOPE = ("backend", "gui", "scripts", "tools", "VideoSubtitleRemover.py")


def _counts(rules) -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", *SCOPE,
         "--select", ",".join(rules), "--no-cache", "--output-format", "json"],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    if result.returncode not in (0, 1):
        raise AssertionError(f"ruff failed: {result.stderr[-800:]}")
    counts = {rule: 0 for rule in rules}
    for item in json.loads(result.stdout or "[]"):
        code = str(item.get("code") or "")
        if code in counts:
            counts[code] += 1
    return counts


class ExceptionBudgetTests(unittest.TestCase):
    def test_the_blind_exception_count_has_not_grown(self):
        counts = _counts(tuple(BUDGET))
        for rule, budget in sorted(BUDGET.items()):
            with self.subTest(rule=rule):
                self.assertLessEqual(
                    counts[rule], budget,
                    f"{rule} rose from {budget} to {counts[rule]}. A new "
                    "blind except hides a failure from the user; handle the "
                    "error, or narrow the clause.",
                )

    def test_the_budget_is_not_stale(self):
        """A budget well above the real count stops catching anything."""
        counts = _counts(tuple(BUDGET))
        for rule, budget in sorted(BUDGET.items()):
            with self.subTest(rule=rule):
                self.assertGreaterEqual(
                    counts[rule], budget - 25,
                    f"{rule} is now {counts[rule]} against a budget of "
                    f"{budget}. Lower the budget in this file so the next "
                    "regression is caught.",
                )

    def test_the_rules_with_no_budget_stay_at_zero(self):
        counts = _counts(FORBIDDEN)
        self.assertEqual(
            {rule: 0 for rule in FORBIDDEN}, counts,
            "raise ... from and a non-security hash use are one-line fixes; "
            "they do not get a budget.",
        )


class NamedSiteTests(unittest.TestCase):
    """The specific swallows this item named, and what they now do."""

    def test_the_ffmpeg_stderr_tail_says_when_it_is_incomplete(self):
        from backend import io as real_io

        writer = real_io._LosslessIntermediateWriter.__new__(
            real_io._LosslessIntermediateWriter)
        import threading

        writer._stderr_lock = threading.Lock()
        writer._stderr_tail_buf = bytearray(b"some ffmpeg output")
        writer._stderr_tail_truncated = False
        self.assertEqual(writer._stderr_tail(), "some ffmpeg output")

        writer._stderr_tail_truncated = True
        text = writer._stderr_tail()
        self.assertIn("incomplete", text)
        self.assertIn("some ffmpeg output", text)

    def test_the_drain_records_a_failure_instead_of_swallowing_it(self):
        import threading

        from backend import io as real_io

        class _Exploding:
            def read(self, _size):
                raise OSError("pipe died")

        class _Proc:
            stderr = _Exploding()

        writer = real_io._LosslessIntermediateWriter.__new__(
            real_io._LosslessIntermediateWriter)
        writer._proc = _Proc()
        writer._stderr_lock = threading.Lock()
        writer._stderr_tail_buf = bytearray(b"partial")
        writer._stderr_tail_truncated = False

        # It still must not raise into the writer thread.
        writer._drain_stderr()

        self.assertTrue(writer._stderr_tail_truncated)
        self.assertIn("incomplete", writer._stderr_tail())

    def test_the_proxy_cache_key_works_where_md5_is_restricted(self):
        """A FIPS-mode host refuses MD5 unless told it is not for security."""
        import ast

        source = (ROOT / "backend" / "proxy_workflow.py").read_text(
            encoding="utf-8")
        tree = ast.parse(source)
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "md5"
        ]
        self.assertTrue(calls)
        for call in calls:
            keywords = {kw.arg for kw in call.keywords}
            self.assertIn("usedforsecurity", keywords)

        from backend.proxy_workflow import _source_fingerprint

        self.assertEqual(len(_source_fingerprint(str(ROOT))), 16)

    def test_a_failed_preview_render_reports_instead_of_blanking(self):
        import ast

        source = (ROOT / "gui" / "preview_controller.py").read_text(
            encoding="utf-8")
        tree = ast.parse(source)
        target = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_apply_inpaint_preview"
        )
        handlers = [
            node for node in ast.walk(target)
            if isinstance(node, ast.ExceptHandler)
        ]
        self.assertTrue(handlers)
        for handler in handlers:
            body = ast.dump(ast.Module(body=handler.body, type_ignores=[]))
            self.assertNotEqual(
                [type(node).__name__ for node in handler.body], ["Pass"],
                "a failed render must say so, not leave a blank pane",
            )
            self.assertIn("_set_preview_empty_state_visible", body)


if __name__ == "__main__":
    unittest.main()
