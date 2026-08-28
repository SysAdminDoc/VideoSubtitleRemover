"""RM-349: the frame loop's window, context, and per-batch carrier.

`processor.py` held the decode window resolver and the dataclasses the frame
loop passes between its stages alongside the 4,500 lines that use them. They
live here so the loop's stages can move out of that file without importing it
back, and so the window arithmetic can be read on its own.

Nothing here knows about `SubtitleRemover`. These are the values the loop
carries, not the work it does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, List, Optional, Tuple

import cv2
import numpy as np

from backend.io import (
    MediaInputError,
    VideoFrameTiming,
    _normalize_time_base,
    _seconds_to_ticks,
    _ticks_to_seconds,
)
from backend.tracking import SubtitleTracker

from pathlib import Path


def _seek_capture_to_frame(cap, target: int) -> int:
    """Position ``cap`` so the next ``read()`` returns frame ``target``.

    ``cap.set(CAP_PROP_POS_FRAMES, N)`` snaps to the nearest keyframe on
    long-GOP CFR sources with some OpenCV backends (MSMF/DSHOW), so a plain
    set can start processing a few frames off the requested ``--start`` time.
    Seek near the target, then grab-and-discard forward to land exactly on it.

    On backends that already position accurately (the bundled FFmpeg backend
    reports the requested logical index), the reported position equals
    ``target`` and no forward grab happens -- so this is a no-op there and
    never over-advances. Returns the resulting position.
    """
    target = max(0, int(target))
    try:
        positioned = bool(cap.set(cv2.CAP_PROP_POS_FRAMES, target))
    except Exception as exc:
        positioned = False
        set_error = exc
    else:
        set_error = None
    if target > 0 and not positioned:
        raise MediaInputError(
            f"The decoder could not seek to frame {target}; retry from the "
            "beginning or use a different decoder.",
            reason="decoder_seek_failed",
            path=str(getattr(cap, "_path", "")),
            detail=(str(set_error) if set_error is not None
                    else "decoder rejected frame seek"),
        )
    if target == 0:
        return 0
    try:
        pos = int(round(float(cap.get(cv2.CAP_PROP_POS_FRAMES))))
    except Exception:
        pos = target
    if pos > target:
        # Overshot; restart and scan forward from the beginning.
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
            raise MediaInputError(
                "The decoder could not restart while seeking the requested frame.",
                reason="decoder_seek_failed",
                path=str(getattr(cap, "_path", "")),
                detail="decoder rejected frame-zero seek",
            )
        pos = 0
    while pos < target:
        if not cap.grab():
            raise MediaInputError(
                f"The decoder stopped before frame {target} while seeking.",
                reason="decoder_seek_failed",
                path=str(getattr(cap, "_path", "")),
                detail=f"reached frame {pos}",
            )
        pos += 1
    return pos


@dataclass(frozen=True)
class _FrameRange:
    """Resolved processing window plus the per-frame timing needed by the
    encode/matte stages. Produced by ``_resolve_frame_range`` so the
    time-range math lives in one testable place instead of inline in
    ``process_video``."""

    time_start_s: float
    time_end_s: float
    start_frame: int
    end_frame: int
    frames_to_process: int
    selected_frame_durations: Optional[List[float]]
    processed_time_start: float
    processed_time_end: float
    matte_timestamps: List[float]
    matte_durations: List[float]
    matte_time_base: float
    selected_frame_duration_ticks: Optional[List[int]]
    processed_time_start_ticks: int
    processed_time_end_ticks: int
    matte_timestamp_ticks: List[int]
    matte_duration_ticks: List[int]
    matte_time_base_num: int
    matte_time_base_den: int


@dataclass(frozen=True)
class _FrameLoopCheckpoint:
    active: bool
    root: Optional[Path]
    key: Optional[str]
    state_path: Optional[Path]
    config_hash: str
    frame_dir: Optional[Path]
    timing_manifest_path: Optional[Path]
    timing_metadata: Optional[dict]
    input_path: str
    output_path: str
    pause_check: Optional[Callable[[], bool]]


@dataclass(frozen=True)
class _FrameLoopContext:
    start_frame: int
    end_frame: int
    frames_to_process: int
    fps: float
    width: int
    height: int
    total_frames: int
    frame_timing: Optional[VideoFrameTiming]
    high_bit_depth_surface: Any
    batch_size: int
    frame_skip: int
    rife_stride: int
    keyframe_set: Optional[set]
    whisper_spans: List[Tuple[int, int]]
    timed_region_spans: bool
    timed_mask_corrections: bool
    static_fixed_shapes: Any
    selective_ranges: List[Tuple[int, int]]
    reader: Any
    selective_cap: Any
    matte_reader: Any
    frozen_matte: bool
    writer: Any
    matte_writer: Any
    checkpoint: _FrameLoopCheckpoint


@dataclass
class _FrameLoopState:
    frame_idx: int
    last_mask: Optional[np.ndarray]
    last_hash: Any
    tracker: Optional[SubtitleTracker]
    fixed_mask_cache: dict
    srt_tracker: Optional[SubtitleTracker] = None
    # RM-292: (mask, frames_remaining) so a fade-out hold survives a batch
    # boundary instead of ending wherever the decode happened to split.
    fade_carry: Any = None
    # RM-296: frames decoded but deliberately not yet processed, held back so
    # they can see the next batch's masks. frame_idx is the DECODE cursor;
    # written_idx is how many frames have actually reached the writer, which
    # is what the checkpoint's resume point and the preview index mean.
    fade_pending: Any = None
    written_idx: int = 0


@dataclass
class _FrameBatch:
    frames: List[np.ndarray] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)
    source_frames: List[Optional[np.ndarray]] = field(default_factory=list)
    passthrough_flags: List[bool] = field(default_factory=list)
    active_segments: List[Tuple[int, int]] = field(default_factory=list)

    _PARALLEL = ("frames", "masks", "source_frames", "passthrough_flags")

    def add(self, frame: np.ndarray, mask: np.ndarray,
            source_frame: Optional[np.ndarray], *, passthrough: bool) -> None:
        self.frames.append(frame)
        self.masks.append(mask)
        self.source_frames.append(source_frame)
        self.passthrough_flags.append(bool(passthrough))

    def split_tail(self, count: int) -> "_FrameBatch":
        """Remove the last `count` frames and return them as a new batch.

        RM-296: a held-back tail is how a fade-in hold reaches across a
        decode boundary. The frames leave this batch entirely, so nothing
        downstream can process them twice.
        """
        tail = _FrameBatch()
        if count <= 0:
            return tail
        for name in self._PARALLEL:
            values = getattr(self, name)
            setattr(tail, name, values[-count:])
            setattr(self, name, values[:-count])
        return tail

    def prepend(self, other: "_FrameBatch") -> None:
        """Put `other`'s frames in front of this batch's, in order."""
        if not other.frames:
            return
        for name in self._PARALLEL:
            setattr(self, name, getattr(other, name) + getattr(self, name))


def _frame_seconds(index: int, fps: float,
                   frame_timing: Optional[VideoFrameTiming] = None) -> float:
    """Return one frame index on the shared VFR/CFR processing clock."""
    if frame_timing is not None:
        return frame_timing.frame_time(index, fps)
    # A missing or bogus rate must not explode the clock. Dividing by an
    # epsilon turns frame 15 into 15 million seconds; a genuine sub-1 fps
    # timelapse still divides by its real rate.
    rate = float(fps)
    return float(index) / (rate if rate > 0.0 else 1.0)


def _spans_from_segments(segments, *, fps: float, total_frames: int,
                         frame_timing: Optional[VideoFrameTiming] = None
                         ) -> List[Tuple[int, int]]:
    """Convert Whisper time segments to frame spans on the shared clock."""
    valid_segments = [segment for segment in (segments or [])
                      if len(segment) >= 2]
    if frame_timing is not None:
        return [
            frame_timing.frame_range(
                float(segment[0]),
                float(segment[1]),
                total_frames,
            )
            for segment in valid_segments
        ]
    from backend.whisper_fallback import segments_to_frame_spans
    return segments_to_frame_spans(valid_segments, fps)


def _resolve_frame_range(cap, total_frames: int, fps: float,
                         frame_timing, time_start: Any,
                         time_end: Any) -> _FrameRange:
    """Resolve the [start, end) frame window from the configured time range.

    Guards against NaN/inf/negative seconds, seeks ``cap`` to the start
    frame, and raises ``ValueError`` when the window is empty. VFR sources
    route through ``frame_timing`` for exact frame<->time mapping; CFR
    sources use ``fps``. Extracted verbatim from ``process_video``.
    """
    def _sane_seconds(value: Any) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(v) or v < 0.0:
            return 0.0
        return v

    time_start_s = _sane_seconds(time_start)
    time_end_s = _sane_seconds(time_end)
    start_frame = 0
    end_frame = total_frames
    if frame_timing is not None:
        start_frame, end_frame = frame_timing.frame_range(
            time_start_s, time_end_s, total_frames)
    else:
        if time_start_s > 0:
            start_frame = max(
                0, min(total_frames, int(time_start_s * fps)))
        if time_end_s > 0:
            end_frame = max(
                0, min(total_frames, int(time_end_s * fps)))
    if end_frame <= start_frame:
        raise ValueError(
            f"Invalid time range: end ({time_end_s}s) "
            f"must be after start ({time_start_s}s)")
    if start_frame > 0:
        _seek_capture_to_frame(cap, start_frame)
    frames_to_process = end_frame - start_frame
    selected_frame_durations = (
        frame_timing.range_durations(start_frame, end_frame, fps)
        if frame_timing is not None else None
    )
    if frame_timing is not None:
        timing_num, timing_den = _normalize_time_base(
            getattr(frame_timing, "time_base_num", 0),
            getattr(frame_timing, "time_base_den", 0),
            fallback_seconds=getattr(frame_timing, "time_base", 0.0),
        )
        range_tick_method = getattr(frame_timing, "range_duration_ticks", None)
        selected_frame_duration_ticks = (
            list(range_tick_method(start_frame, end_frame))
            if callable(range_tick_method) else [
                _seconds_to_ticks(value, timing_num, timing_den)
                for value in selected_frame_durations or []
            ]
        )
    else:
        timing_num, timing_den = 0, 1
        selected_frame_duration_ticks = None
    processed_time_start = _frame_seconds(
        start_frame, fps, frame_timing)
    processed_time_end = (
        processed_time_start + sum(selected_frame_durations or [])
        if selected_frame_durations is not None
        else _frame_seconds(end_frame, fps)
    )
    if frame_timing is not None:
        matte_time_base_num = timing_num
        matte_time_base_den = timing_den
        frame_tick_method = getattr(frame_timing, "frame_time_ticks", None)
        matte_timestamp_ticks = [
            frame_tick_method(index, fps)
            if callable(frame_tick_method) else _seconds_to_ticks(
                frame_timing.frame_time(index, fps),
                matte_time_base_num,
                matte_time_base_den,
            )
            for index in range(start_frame, end_frame)
        ]
        matte_duration_ticks = list(selected_frame_duration_ticks or [])
        processed_time_start_ticks = (
            frame_tick_method(start_frame, fps)
            if callable(frame_tick_method) else _seconds_to_ticks(
                frame_timing.frame_time(start_frame, fps),
                matte_time_base_num,
                matte_time_base_den,
            )
        )
        processed_time_end_ticks = (
            processed_time_start_ticks + sum(matte_duration_ticks)
        )
    else:
        try:
            rate = Fraction(str(float(fps)))
        except (TypeError, ValueError, ZeroDivisionError):
            rate = Fraction(1, 1)
        if rate <= 0:
            rate = Fraction(1, 1)
        matte_time_base_num, matte_time_base_den = _normalize_time_base(
            rate.denominator, rate.numerator,
            fallback_seconds=1.0,
        )
        matte_timestamp_ticks = list(range(start_frame, end_frame))
        matte_duration_ticks = [1] * frames_to_process
        processed_time_start_ticks = int(start_frame)
        processed_time_end_ticks = int(end_frame)
    matte_timestamps = [
        _ticks_to_seconds(
            value, matte_time_base_num, matte_time_base_den)
        for value in matte_timestamp_ticks
    ]
    matte_durations = [
        _ticks_to_seconds(
            value, matte_time_base_num, matte_time_base_den)
        for value in matte_duration_ticks
    ]
    matte_time_base = _ticks_to_seconds(
        1, matte_time_base_num, matte_time_base_den)
    return _FrameRange(
        time_start_s=time_start_s,
        time_end_s=time_end_s,
        start_frame=start_frame,
        end_frame=end_frame,
        frames_to_process=frames_to_process,
        selected_frame_durations=selected_frame_durations,
        processed_time_start=processed_time_start,
        processed_time_end=processed_time_end,
        matte_timestamps=matte_timestamps,
        matte_durations=matte_durations,
        matte_time_base=matte_time_base,
        selected_frame_duration_ticks=selected_frame_duration_ticks,
        processed_time_start_ticks=processed_time_start_ticks,
        processed_time_end_ticks=processed_time_end_ticks,
        matte_timestamp_ticks=matte_timestamp_ticks,
        matte_duration_ticks=matte_duration_ticks,
        matte_time_base_num=matte_time_base_num,
        matte_time_base_den=matte_time_base_den,
    )
