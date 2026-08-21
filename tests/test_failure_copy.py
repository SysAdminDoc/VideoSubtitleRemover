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
