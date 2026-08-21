"""Queue-row failure copy must not leak paths or raw exceptions."""

from __future__ import annotations

from gui.failure_copy import (
    MSG_CRASHED,
    MSG_FAILED,
    MSG_MISSING,
    MSG_TIMEOUT,
    user_facing_isolated_error,
    user_facing_processing_error,
)
from gui.utils import CANONICAL_QUEUE_MESSAGES, queue_message_text


def test_file_not_found_does_not_echo_the_path():
    message = user_facing_processing_error(
        FileNotFoundError("C:\\\\Users\\\\secret\\\\clip.mp4"))
    assert message == MSG_MISSING
    assert "secret" not in message
    assert message in CANONICAL_QUEUE_MESSAGES
    assert "secret" not in queue_message_text(message)


def test_unknown_exception_is_generic():
    class WeirdInternal(RuntimeError):
        pass

    message = user_facing_processing_error(
        WeirdInternal("absolute path C:\\\\tmp\\\\x"))
    assert message == MSG_FAILED
    assert "C:" not in message
    assert message in CANONICAL_QUEUE_MESSAGES


def test_isolated_crash_and_timeout_are_classified():
    assert user_facing_isolated_error(
        "Worker timed out after 30s") == MSG_TIMEOUT
    crash = user_facing_isolated_error(
        "Worker crashed (exit -1073741819): C:\\\\models\\\\foo.onnx")
    assert crash == MSG_CRASHED
    assert "C:" not in crash
    assert "onnx" not in crash.lower()
    assert crash in CANONICAL_QUEUE_MESSAGES


def test_queue_state_round_trips_the_failure_reason():
    """RM-279: the classified reason survives a save/load of queue_state."""
    from unittest import mock

    from backend import failure_reason as fr
    from gui import config as gcfg

    item = gcfg.QueueItem(
        id="1", file_path="a.mp4", output_path="b.mp4",
        config=gcfg.ProcessingConfig(),
        status=gcfg.ProcessingStatus.ERROR,
        message=MSG_FAILED,
        failure_reason=fr.REASON_WRITER_FAILED,
    )
    captured: dict = {}
    with mock.patch.object(
        gcfg, "_write_json_atomic",
        side_effect=lambda path, payload: captured.update(payload),
    ):
        gcfg.save_queue_state([item])
    record = captured["items"][0]
    assert record["failure_reason"] == fr.REASON_WRITER_FAILED
    assert record["message"] == MSG_FAILED
    assert captured["schema"] == gcfg.QUEUE_STATE_SCHEMA


def test_an_off_vocabulary_reason_is_dropped_on_save():
    from unittest import mock

    from gui import config as gcfg

    item = gcfg.QueueItem(
        id="1", file_path="a.mp4", output_path="b.mp4",
        config=gcfg.ProcessingConfig(),
        status=gcfg.ProcessingStatus.ERROR,
        failure_reason="something_invented",
    )
    captured: dict = {}
    with mock.patch.object(
        gcfg, "_write_json_atomic",
        side_effect=lambda path, payload: captured.update(payload),
    ):
        gcfg.save_queue_state([item])
    assert captured["items"][0]["failure_reason"] == ""
def test_the_batch_harmonic_stat_uses_each_report_own_harmonic_mean():
    """RM-281: pooling the per-item averages measured the wrong spread."""
    from gui.utils import summarize_quality_reports

    summary = summarize_quality_reports([{
        "psnr": 40.0,
        "ssim": 0.98,
        "samples": 10,
        "psnr_harmonic_mean": 21.0,
        "ssim_harmonic_mean": 0.62,
    }])
    assert summary["ssim"] == 0.98
    assert summary["harmonic_ssim"] == 0.62
    assert summary["harmonic_psnr"] == 21.0


def test_a_report_without_harmonic_keys_falls_back_to_its_mean():
    from gui.utils import summarize_quality_reports

    summary = summarize_quality_reports([{"psnr": 40.0, "ssim": 0.98, "samples": 10}])
    assert summary["harmonic_ssim"] == 0.98
    assert summary["harmonic_psnr"] == 40.0
