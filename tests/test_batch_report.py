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
from backend import failure_reason as fr


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


class FailureReasonTests(unittest.TestCase):
    """RM-279: every terminal outcome carries a closed-set reason."""

    def test_reason_vocabulary_covers_the_required_classes(self):
        for required in (
            fr.REASON_NO_SPACE,
            fr.REASON_WRITER_FAILED,
            fr.REASON_OUTPUT_EMPTY,
            fr.REASON_FFMPEG_FAILED,
            fr.REASON_MODEL_MISSING,
            fr.REASON_DECODE_FAILED,
            fr.REASON_CANCELLED,
            fr.REASON_UNKNOWN,
        ):
            self.assertIn(required, fr.FAILURE_REASONS)
            self.assertIn(required, fr.FAILURE_REASON_LABELS)

    def test_processor_reason_codes_map_onto_the_closed_set(self):
        cases = {
            "truncated_decode": fr.REASON_DECODE_FAILED,
            "unsupported_codec": fr.REASON_DECODE_FAILED,
            "intermediate_writer_timeout": fr.REASON_WRITER_FAILED,
            "frame_write_failed": fr.REASON_WRITER_FAILED,
            "frozen_matte_artifact_changed": fr.REASON_FROZEN_MATTE,
        }
        for code, expected in cases.items():
            with self.subTest(code=code):
                self.assertEqual(
                    fr.classify_failure_reason(reason=code), expected)

    def test_exceptions_and_text_are_classified(self):
        self.assertEqual(
            fr.classify_failure_reason(exc=FileNotFoundError("gone")),
            fr.REASON_INPUT_MISSING,
        )
        self.assertEqual(
            fr.classify_failure_reason(exc=OSError(28, "No space left on device")),
            fr.REASON_NO_SPACE,
        )
        self.assertEqual(
            fr.classify_failure_reason(
                message="Insufficient disk space at 'D:/out' for output"),
            fr.REASON_NO_SPACE,
        )
        self.assertEqual(
            fr.classify_failure_reason(message="ffmpeg exited with code 1"),
            fr.REASON_FFMPEG_FAILED,
        )

    def test_an_unrecognised_failure_is_unknown_not_blank(self):
        self.assertEqual(
            fr.classify_failure_reason(message="Processing failed"),
            fr.REASON_UNKNOWN,
        )

    def test_an_output_integrity_failure_is_not_unknown(self):
        """RM-279 names output_empty; nothing could reach it before."""
        for message in (
            "output has no decodable video stream",
            "output duration 3.200s is shorter than the expected 60.000s",
            "output frame count 12 is below the expected 1500 (tolerance 15)",
        ):
            with self.subTest(message=message):
                self.assertEqual(
                    fr.classify_failure_reason(message=message),
                    fr.REASON_OUTPUT_EMPTY,
                )

    def test_the_output_integrity_exception_maps_by_type(self):
        from backend.processor import OutputIntegrityError

        self.assertEqual(
            fr.classify_failure_reason(
                exc=OutputIntegrityError("something opaque", {})),
            fr.REASON_OUTPUT_EMPTY,
        )

    def test_late_added_reason_codes_are_classified(self):
        for code, expected in (
            ("decoder_seek_failed", fr.REASON_DECODE_FAILED),
            ("worker_timeout", fr.REASON_TIMED_OUT),
            ("output_integrity_failed", fr.REASON_OUTPUT_EMPTY),
        ):
            with self.subTest(code=code):
                self.assertEqual(
                    fr.classify_failure_reason(reason=code), expected)

    def test_finish_batch_item_records_a_reason_and_keeps_the_message(self):
        record = _record(
            br.STATUS_PENDING, input_path="a.mp4", output_path="a_out.mp4")
        curated = "The selected video appears corrupt or incomplete."
        br.finish_batch_item(
            record,
            br.STATUS_FAILED,
            message=curated,
            failure_reason="corrupt_or_truncated",
        )
        self.assertEqual(record["message"], curated)
        self.assertEqual(record["failure_reason"], fr.REASON_DECODE_FAILED)

    def test_non_failures_carry_no_reason(self):
        record = _record(
            br.STATUS_PENDING, input_path="a.mp4", output_path="a_out.mp4")
        br.finish_batch_item(
            record, br.STATUS_HARDCODED_PROCESSED, message="Processed")
        self.assertEqual(record["failure_reason"], fr.REASON_NONE)

    def test_cancel_and_pause_are_their_own_reasons(self):
        for status, expected in (
            (br.STATUS_CANCELLED, fr.REASON_CANCELLED),
            (br.STATUS_PAUSED, fr.REASON_PAUSED),
        ):
            record = _record(
                br.STATUS_PENDING, input_path="a.mp4", output_path="a_out.mp4")
            br.finish_batch_item(record, status, message="Interrupted")
            self.assertEqual(record["failure_reason"], expected)

    def test_summary_aggregates_and_prints_failure_reasons(self):
        records = []
        for name, reason in (
            ("a", "intermediate_writer_failed"),
            ("b", "frame_write_failed"),
            ("c", "unsupported_codec"),
        ):
            record = _record(
                br.STATUS_PENDING,
                input_path=f"{name}.mp4",
                output_path=f"{name}_out.mp4",
            )
            record["input_name"] = f"{name}.mp4"
            record["output_name"] = f"{name}_out.mp4"
            br.finish_batch_item(
                record, br.STATUS_FAILED, message="boom",
                failure_reason=reason,
            )
            records.append(record)
        with tempfile.TemporaryDirectory() as tmp:
            started = _dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc)
            json_path, md_path = br.write_batch_reports(
                Path(tmp), records, kind="batch", started_at=started)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            md = md_path.read_text(encoding="utf-8")
        self.assertEqual(
            payload["failure_reason_counts"],
            {fr.REASON_WRITER_FAILED: 2, fr.REASON_DECODE_FAILED: 1},
        )
        self.assertIn("| Status | Reason |", md)
        self.assertIn(
            fr.FAILURE_REASON_LABELS[fr.REASON_WRITER_FAILED], md)


if __name__ == "__main__":
    unittest.main()
