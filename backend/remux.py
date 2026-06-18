"""Soft-subtitle remux helpers.

These helpers intentionally stay below the processing pipeline: they never
open video frames, instantiate OCR, or touch inpainters. They only build and
run explicit ffmpeg stream-copy commands for embedded subtitle tracks.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
import shutil
import subprocess
from typing import Callable, Iterable, List, Optional

from backend.io import (
    _allocate_temp_output_path,
    _cleanup_temp_output,
    _ffmpeg_subprocess_timeout,
    _path_key,
    _probe_duration_seconds,
    _promote_temp_output,
    _run_subprocess_checked,
)


class SoftSubtitleAction(str, Enum):
    STRIP = "strip"
    KEEP_ALL = "keep_all"
    KEEP_SELECTED = "keep_selected"


def build_soft_subtitle_remux_cmd(
    input_path: str,
    output_path: str,
    *,
    action: SoftSubtitleAction | str = SoftSubtitleAction.STRIP,
    keep_stream_indices: Optional[Iterable[int]] = None,
) -> List[str]:
    """Build an ffmpeg stream-copy command for embedded subtitle tracks."""
    resolved_action = SoftSubtitleAction(action)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-nostats",
        "-i", str(input_path),
    ]
    if resolved_action == SoftSubtitleAction.KEEP_ALL:
        cmd += ["-map", "0"]
    elif resolved_action == SoftSubtitleAction.STRIP:
        cmd += ["-map", "0", "-map", "-0:s?"]
    elif resolved_action == SoftSubtitleAction.KEEP_SELECTED:
        selected = _normalize_stream_indices(keep_stream_indices)
        if not selected:
            raise ValueError("keep_selected requires at least one stream index")
        cmd += ["-map", "0", "-map", "-0:s?"]
        for stream_index in selected:
            cmd += ["-map", f"0:{stream_index}"]
    cmd += ["-c", "copy", str(output_path)]
    return cmd


def remux_soft_subtitles(
    input_path: str,
    output_path: str,
    *,
    action: SoftSubtitleAction | str = SoftSubtitleAction.STRIP,
    keep_stream_indices: Optional[Iterable[int]] = None,
    on_process: Optional[Callable[[Optional[subprocess.Popen]], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> None:
    """Run a soft-subtitle remux through a sibling temp output path."""
    if _path_key(input_path) == _path_key(output_path):
        raise ValueError("soft-subtitle remux output must differ from input")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for soft-subtitle remux")
    temp_output = _allocate_temp_output_path(output_path)
    try:
        cmd = build_soft_subtitle_remux_cmd(
            input_path,
            str(temp_output),
            action=action,
            keep_stream_indices=keep_stream_indices,
        )
        timeout = _ffmpeg_subprocess_timeout(_probe_duration_seconds(input_path))
        _run_subprocess_checked(
            cmd,
            timeout=timeout,
            on_process=on_process,
            cancel_check=cancel_check,
        )
        _promote_temp_output(temp_output, Path(output_path))
    finally:
        _cleanup_temp_output(temp_output)


def _normalize_stream_indices(indices: Optional[Iterable[int]]) -> List[int]:
    if indices is None:
        return []
    normalized = {int(value) for value in indices}
    if any(value < 0 for value in normalized):
        raise ValueError("stream indices must be non-negative")
    return sorted(normalized)
