"""Tests for accurate frame seeking on CFR sources.

``cv2.VideoCapture.set(CAP_PROP_POS_FRAMES, N)`` snaps to the nearest keyframe
on long-GOP sources with some OpenCV backends, so time-range processing could
start a few frames off the requested ``--start``. ``_seek_capture_to_frame``
grabs forward to land exactly on the requested frame.
"""

from __future__ import annotations

import cv2

from backend.processor import _seek_capture_to_frame


class _SnappingCapture:
    """Simulates a backend whose POS_FRAMES set snaps to the previous
    keyframe (every `gop` frames) and reports that snapped index."""

    def __init__(self, gop: int = 30):
        self.gop = gop
        self.pos = 0          # index of the frame the next read() returns
        self.grabs = 0

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            snapped = (int(value) // self.gop) * self.gop
            self.pos = snapped
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self.pos)
        return 0.0

    def grab(self):
        self.pos += 1
        self.grabs += 1
        return True


class _AccurateCapture:
    """Simulates the bundled FFmpeg backend: set positions exactly and
    reports the requested logical index."""

    def __init__(self):
        self.pos = 0
        self.grabs = 0

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.pos = int(value)
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self.pos)
        return 0.0

    def grab(self):
        self.pos += 1
        self.grabs += 1
        return True


def test_snapping_backend_lands_on_exact_frame():
    cap = _SnappingCapture(gop=30)
    result = _seek_capture_to_frame(cap, 47)
    assert result == 47
    assert cap.pos == 47          # next read() returns frame 47
    assert cap.grabs == 17        # 47 - 30 keyframe


def test_accurate_backend_no_over_advance():
    cap = _AccurateCapture()
    result = _seek_capture_to_frame(cap, 47)
    assert result == 47
    assert cap.pos == 47
    assert cap.grabs == 0         # already accurate -> no forward grab


def test_seek_to_zero_is_noop_position():
    cap = _SnappingCapture(gop=30)
    result = _seek_capture_to_frame(cap, 0)
    assert result == 0
    assert cap.pos == 0
    assert cap.grabs == 0


def test_no_raw_seek_outside_wrapper():
    """Every cap.set(CAP_PROP_POS_FRAMES) in processor.py must live inside
    _seek_capture_to_frame so long-GOP sources always land on the right
    frame.  This is a source-level guard against regressions like the
    quality-report seek bug fixed in v3.26."""
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent
           / "backend" / "processor.py").read_text(encoding="utf-8")
    pattern = re.compile(
        r"\.set\(\s*cv2\.CAP_PROP_POS_FRAMES\b",
    )
    inside_wrapper = False
    violations = []
    for lineno, line in enumerate(src.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("def _seek_capture_to_frame("):
            inside_wrapper = True
        elif inside_wrapper and stripped.startswith("def "):
            inside_wrapper = False
        if pattern.search(line) and not inside_wrapper:
            violations.append(f"line {lineno}: {stripped}")
    assert violations == [], (
        "raw cap.set(CAP_PROP_POS_FRAMES) found outside "
        "_seek_capture_to_frame:\n" + "\n".join(violations)
    )


def test_overshoot_backend_rescans_from_start():
    class _Overshoot(_AccurateCapture):
        def get(self, prop):
            # Report a position past the request to exercise the reset path.
            if prop == cv2.CAP_PROP_POS_FRAMES:
                return float(self.pos + 5)
            return 0.0

    cap = _Overshoot()
    result = _seek_capture_to_frame(cap, 10)
    assert result == 10
    assert cap.pos == 10
