"""Tests for _resolve_frame_range, the time-range resolver extracted from
process_video."""

from __future__ import annotations

import math

import cv2
import pytest

from backend.processor import _frame_seconds, _resolve_frame_range


class _FakeCap:
    """Accurate-seek capture stub: set() positions exactly, get() reports it."""

    def __init__(self):
        self.pos = 0

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.pos = int(value)
        return True

    def get(self, prop):
        return float(self.pos) if prop == cv2.CAP_PROP_POS_FRAMES else 0.0

    def grab(self):
        self.pos += 1
        return True


def test_cfr_full_range_when_no_time_window():
    r = _resolve_frame_range(_FakeCap(), 300, 30.0, None, 0, 0)
    assert (r.start_frame, r.end_frame) == (0, 300)
    assert r.frames_to_process == 300
    assert r.selected_frame_durations is None
    assert r.processed_time_start == 0.0
    assert r.processed_time_end == 10.0
    assert math.isclose(r.matte_time_base, 1.0 / 30.0)
    assert len(r.matte_timestamps) == 300
    assert len(r.matte_durations) == 300


def test_cfr_time_window_maps_to_frames_and_seeks():
    cap = _FakeCap()
    r = _resolve_frame_range(cap, 300, 30.0, None, 1.0, 5.0)
    assert (r.start_frame, r.end_frame) == (30, 150)
    assert r.frames_to_process == 120
    assert math.isclose(r.processed_time_start, 1.0)
    assert math.isclose(r.processed_time_end, 5.0)
    assert cap.pos == 30  # cap was sought to the start frame
    assert r.matte_timestamps[0] == 30 / 30.0


def test_non_finite_and_negative_seconds_are_clamped():
    r = _resolve_frame_range(_FakeCap(), 100, 25.0, None,
                             float("nan"), float("inf"))
    assert (r.start_frame, r.end_frame) == (0, 100)
    assert r.time_start_s == 0.0 and r.time_end_s == 0.0


def test_empty_window_raises():
    with pytest.raises(ValueError):
        _resolve_frame_range(_FakeCap(), 300, 30.0, None, 5.0, 1.0)


class _FakeTiming:
    """Minimal VFR timing: constant 0.04s frames but exercises the VFR path."""

    is_vfr = True
    time_base = 0.001

    def frame_range(self, start_s, end_s, total):
        start = int(round(start_s / 0.04))
        end = total if end_s <= 0 else int(round(end_s / 0.04))
        return start, end

    def frame_time(self, index, fps):
        return index * 0.04

    def frame_duration(self, index, fps):
        return 0.04

    def range_durations(self, start, end, fps):
        return [0.04] * (end - start)


def test_vfr_path_uses_frame_timing():
    r = _resolve_frame_range(_FakeCap(), 250, 25.0, _FakeTiming(), 0.0, 4.0)
    assert r.start_frame == 0
    assert r.end_frame == 100
    assert r.selected_frame_durations == [0.04] * 100
    assert math.isclose(r.processed_time_end, 4.0)
    assert math.isclose(r.matte_time_base, 0.001)


def test_frame_seconds_uses_one_safe_cfr_vfr_clock():
    assert math.isclose(_frame_seconds(15, 30.0), 0.5)
    assert math.isclose(_frame_seconds(15, 0.0), 15.0)
    assert math.isclose(_frame_seconds(15, 30.0, _FakeTiming()), 0.6)
