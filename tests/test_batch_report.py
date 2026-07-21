"""RM-133: module-boundary unit tests for backend.batch_report.

batch_report builds the per-item records and the batch summary that the
CLI and GUI write to disk and that release tooling consumes. It was only
exercised indirectly through the release-workflow test; these cover its
public surface directly so schema/field/redaction drift is caught.
"""

import datetime as _dt
import json
import tempfile
import subprocess
import unittest
from pathlib import Path

from backend import batch_report as br


def _record(status: str, *, input_path: str, output_path: str) -> dict:
    return {
        "input": input_path,
        "output": output_path,
        "output_parent_free_bytes": 123456,
        "status": status,
        "message": "",
        "elapsed_seconds": 1.0,
    }


class RetriableErrorTests(unittest.TestCase):
    def test_permanent_errors_are_not_retriable(self):
        self.assertFalse(br.is_retriable_error(FileNotFoundError("no such file")))
        self.assertFalse(br.is_retriable_error(PermissionError("permission denied")))
        self.assertFalse(br.is_retriable_error(ValueError("unsupported codec")))

    def test_transient_errors_are_retriable(self):
        self.assertTrue(br.is_retriable_error(MemoryError("CUDA out of memory")))
        self.assertTrue(br.is_retriable_error(TimeoutError("op timed out")))
        self.assertTrue(
            br.is_retriable_error(subprocess.TimeoutExpired(cmd="ffmpeg", timeout=1))
        )

    def test_keyboard_interrupt_is_not_retriable(self):
        self.assertFalse(br.is_retriable_error(KeyboardInterrupt()))


class PlannedStatusTests(unittest.TestCase):
    def test_skip_existing_beats_everything(self):
        self.assertEqual(
            br.planned_batch_status(
                output_exists=True, skip_existing=True,
                checkpoint_done=True, soft_action="strip"),
            br.STATUS_SKIPPED_EXISTING,
        )

    def test_checkpoint_done_when_not_skipping(self):
        self.assertEqual(
            br.planned_batch_status(
                output_exists=False, skip_existing=False,
                checkpoint_done=True),
            br.STATUS_CHECKPOINT_DONE,
        )

    def test_soft_action_remuxes(self):
        for action in ("strip", "keep_all"):
            self.assertEqual(
                br.planned_batch_status(
                    output_exists=False, skip_existing=False,
                    checkpoint_done=False, soft_action=action),
                br.STATUS_SOFT_REMUXED,
            )

    def test_default_is_hardcoded_processing(self):
        self.assertEqual(
            br.planned_batch_status(
                output_exists=False, skip_existing=False,
                checkpoint_done=False),
            br.STATUS_HARDCODED_PROCESSED,
        )


class FinishBatchItemTests(unittest.TestCase):
    def test_sets_status_message_and_rounds_elapsed(self):
        rec: dict = {"status": br.STATUS_PENDING}
        out = br.finish_batch_item(
            rec, br.STATUS_HARDCODED_PROCESSED,
            message="done", elapsed_seconds=2.34567)
        self.assertEqual(out["status"], br.STATUS_HARDCODED_PROCESSED)
        self.assertEqual(out["message"], "done")
        self.assertEqual(out["elapsed_seconds"], 2.346)

    def test_review_gate_promotes_processed_to_review_needed(self):
        rec: dict = {"status": br.STATUS_PENDING}
        out = br.finish_batch_item(
            rec, br.STATUS_HARDCODED_PROCESSED,
            quality_report={"quality_gate": {"status": "review"}})
        self.assertEqual(out["status"], br.STATUS_REVIEW_NEEDED)


class WriteBatchReportsTests(unittest.TestCase):
    def _records(self):
        return [
            _record(br.STATUS_HARDCODED_PROCESSED, input_path="/abs/in1.mp4",
                    output_path="/abs/out1.mp4"),
            _record(br.STATUS_FAILED, input_path="/abs/in2.mp4",
                    output_path="/abs/out2.mp4"),
            _record(br.STATUS_CANCELLED, input_path="/abs/in3.mp4",
                    output_path="/abs/out3.mp4"),
        ]

    def _write(self, redact_paths: bool):
        started = _dt.datetime(2026, 7, 20, tzinfo=_dt.timezone.utc)
        completed = started + _dt.timedelta(seconds=5)
        with tempfile.TemporaryDirectory() as d:
            json_path, md_path = br.write_batch_reports(
                Path(d), self._records(), kind="batch",
                started_at=started, completed_at=completed,
                redact_paths=redact_paths)
            payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
            md = Path(md_path).read_text(encoding="utf-8")
        return payload, md

    def test_schema_counts_and_files(self):
        payload, md = self._write(redact_paths=True)
        self.assertEqual(payload["schema"], "vsr.batch_summary.v1")
        self.assertEqual(payload["count"], 3)
        self.assertEqual(payload["counts"][br.STATUS_HARDCODED_PROCESSED], 1)
        self.assertEqual(payload["counts"][br.STATUS_FAILED], 1)
        self.assertEqual(payload["counts"][br.STATUS_CANCELLED], 1)
        self.assertEqual(payload["elapsed_seconds"], 5.0)
        self.assertEqual(len(payload["files"]), 3)
        self.assertTrue(md.strip())

    def test_redaction_is_default_on(self):
        payload, _ = self._write(redact_paths=True)
        for row in payload["files"]:
            self.assertNotIn("input", row)
            self.assertNotIn("output", row)
            self.assertNotIn("output_parent_free_bytes", row)

    def test_redaction_can_be_disabled(self):
        payload, _ = self._write(redact_paths=False)
        self.assertTrue(any("input" in row for row in payload["files"]))


if __name__ == "__main__":
    unittest.main()
