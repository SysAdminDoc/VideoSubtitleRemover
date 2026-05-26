"""Kalman-based subtitle tracking + helpers.

Extracted from processor.py as part of RFP-L-1. Provides:

- ``_KalmanBox``: constant-velocity filter per subtitle box; absorbs
  per-frame OCR jitter and survives a single-frame miss.
- ``SubtitleTracker``: greedy IoU-matched multi-box wrapper around
  ``_KalmanBox``. Output preserves identity across frames so the
  chyron classifier (``categorize``) can distinguish persistent
  graphics from dialogue subtitles.
- ``_group_horizontal_line``: pre-track fusion for karaoke captions
  that arrive as many small per-syllable boxes.
- ``_phash`` / ``_phash_distance``: 64-bit perceptual hash used by
  the adaptive frame-skip path.

Pure numpy + cv2 dependencies. No optional packages required.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


class _KalmanBox:
    """Simple constant-velocity Kalman filter for a single subtitle box.
    State: [cx, cy, w, h, dx, dy, dw, dh]. Measurement: [cx, cy, w, h].
    Used to smooth per-frame OCR jitter and carry the box through a missed
    detection (single-frame occlusion)."""

    def __init__(self, box: Tuple[int, int, int, int]):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)
        self.kf.statePost = np.array(
            [cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32).reshape(8, 1)
        self.age = 0
        self.hits = 1

    def predict(self) -> Tuple[int, int, int, int]:
        s = self.kf.predict().flatten()
        return _box_from_state(s)

    def update(self, box: Tuple[int, int, int, int]):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        m = np.array([cx, cy, w, h], dtype=np.float32).reshape(4, 1)
        self.kf.correct(m)
        self.age = 0
        self.hits += 1

    def is_chyron(self, min_hits: int) -> bool:
        return self.hits >= max(1, int(min_hits))

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return _box_from_state(self.kf.statePost.flatten())


def _box_from_state(state: np.ndarray) -> Tuple[int, int, int, int]:
    """Reconstruct (x1, y1, x2, y2) from a Kalman state vector. Width and
    height are clamped to >=1 so a noisy filter prediction never produces
    an inverted box."""
    cx = float(state[0])
    cy = float(state[1])
    w = max(1.0, float(state[2]))
    h = max(1.0, float(state[3]))
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return (0, 0, 1, 1)
    x1 = int(round(cx - w / 2.0))
    y1 = int(round(cy - h / 2.0))
    x2 = int(round(cx + w / 2.0))
    y2 = int(round(cy + h / 2.0))
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1
    return (x1, y1, x2, y2)


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
    area_b = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / float(union)


class SubtitleTracker:
    """Multi-box Kalman tracker that smooths per-frame detection jitter
    and carries boxes through single-frame misses."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 2):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._tracks: List[_KalmanBox] = []

    def reset(self):
        self._tracks = []

    def update(self, detections: List[Tuple[int, int, int, int]]
                ) -> List[Tuple[int, int, int, int]]:
        if not self._tracks:
            self._tracks = [_KalmanBox(d) for d in detections]
            return [t.box for t in self._tracks]

        predictions = [t.predict() for t in self._tracks]
        used_det = set()
        used_trk = set()
        for ti, pred in enumerate(predictions):
            best_di, best_iou = -1, 0.0
            for di, det in enumerate(detections):
                if di in used_det:
                    continue
                score = _iou(pred, det)
                if score > best_iou:
                    best_iou, best_di = score, di
            if best_di >= 0 and best_iou >= self.iou_threshold:
                self._tracks[ti].update(detections[best_di])
                used_det.add(best_di)
                used_trk.add(ti)
            else:
                self._tracks[ti].age += 1

        for di, det in enumerate(detections):
            if di not in used_det:
                self._tracks.append(_KalmanBox(det))

        self._tracks = [t for t in self._tracks if t.age <= self.max_age]
        return [t.box for t in self._tracks]

    def categorize(self, min_chyron_hits: int) -> List[str]:
        return [
            "chyron" if t.is_chyron(min_chyron_hits) else "subtitle"
            for t in self._tracks
        ]


def _group_horizontal_line(
    boxes: List[Tuple[int, int, int, int]],
    x_gap_px: int = 20,
    y_overlap_ratio: float = 0.5,
) -> List[Tuple[int, int, int, int]]:
    """Merge boxes that sit on the same horizontal text line."""
    if not boxes or len(boxes) == 1:
        return list(boxes)

    def _y_overlap(a, b) -> float:
        a_h = max(1, a[3] - a[1])
        b_h = max(1, b[3] - b[1])
        inter = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        return inter / float(min(a_h, b_h))

    def _x_gap(a, b) -> int:
        return max(a[0], b[0]) - min(a[2], b[2])

    merged = [tuple(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        new_merged = []
        used = set()
        for i in range(len(merged)):
            if i in used:
                continue
            a = merged[i]
            ax1, ay1, ax2, ay2 = a
            for j in range(i + 1, len(merged)):
                if j in used:
                    continue
                b = merged[j]
                if (_y_overlap((ax1, ay1, ax2, ay2), b) >= y_overlap_ratio
                        and _x_gap((ax1, ay1, ax2, ay2), b) <= x_gap_px):
                    ax1 = min(ax1, b[0])
                    ay1 = min(ay1, b[1])
                    ax2 = max(ax2, b[2])
                    ay2 = max(ay2, b[3])
                    used.add(j)
                    changed = True
            new_merged.append((ax1, ay1, ax2, ay2))
            used.add(i)
        merged = new_merged
    return merged


def _phash(frame: np.ndarray, size: int = 8) -> np.ndarray:
    """Compact perceptual hash for adaptive frame-skip."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    low = dct[:size, :size]
    med = np.median(low)
    return (low > med).astype(np.uint8)


def _phash_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))
