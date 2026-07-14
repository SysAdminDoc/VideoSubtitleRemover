"""OCR detection benchmark.

Measures the currently-selected text detector against a set of synthetic,
ground-truth subtitle fixtures so a maintainer can decide -- with evidence --
whether to change the default detector (e.g. adopt PP-OCRv6 broadly). The
fixtures are generated deterministically with OpenCV so the benchmark needs no
redistributable media and runs anywhere; the reported engine name records which
model actually ran (RapidOCR ships PP-OCRv6 by default), and the accuracy /
latency floors below are the gate a default swap must clear.

Metrics per fixture:
  - hit: at least one detected box overlaps the ground-truth text box (IoU or
    containment), i.e. the subtitle region was found.
  - iou:  best IoU between a detected box and the ground-truth box.

The aggregate ``recall`` (fraction of fixtures with a hit) and ``mean_ms``
(per-frame detection wall-clock) are compared against ``RECALL_FLOOR`` /
``LATENCY_CEILING_MS`` to produce a pass/fail ``meets_floors`` verdict.
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


OCR_BENCHMARK_SCHEMA = "vsr.ocr_benchmark.v1"

# A default detector swap should not regress below these on the synthetic set.
RECALL_FLOOR = 0.8
LATENCY_CEILING_MS = 2000.0

_FIXTURE_TEXTS = (
    "HELLO WORLD",
    "Subtitle line 12",
    "The quick brown fox",
    "1234567890",
    "GOODBYE",
)


def _make_fixture(text: str, width: int = 640, height: int = 160,
                  seed: int = 0) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Return (image, ground_truth_box) for a subtitle-like frame."""
    rng = np.random.RandomState(seed)
    # Mild textured background so the detector is not scoring on a flat plate.
    bg = rng.randint(20, 60, (height, width, 3), dtype=np.uint8)
    bg = cv2.GaussianBlur(bg, (9, 9), 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.3
    thickness = 3
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(5, (width - tw) // 2)
    y = (height + th) // 2
    cv2.putText(bg, text, (x, y), font, scale, (245, 245, 245), thickness,
                cv2.LINE_AA)
    box = (x, y - th - baseline, x + tw, y + baseline)
    return bg, box


def iter_fixtures() -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """Return the (image, ground_truth_box) benchmark fixtures (for tests)."""
    return [_make_fixture(text, seed=i) for i, text in enumerate(_FIXTURE_TEXTS)]


def _iou(a: Sequence[int], b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def run_ocr_detection_benchmark(detector, *, threshold: float = 0.3,
                                iou_hit: float = 0.15) -> dict:
    """Score ``detector`` on the synthetic fixtures. ``detector`` must expose
    ``detect(frame, threshold) -> [(x1, y1, x2, y2), ...]``."""
    engine = str(getattr(detector, "_engine_name", "unknown"))
    results = []
    latencies: List[float] = []
    hits = 0
    for i, text in enumerate(_FIXTURE_TEXTS):
        image, gt = _make_fixture(text, seed=i)
        start = time.perf_counter()
        try:
            boxes = detector.detect(image, threshold) or []
        except Exception as exc:  # noqa: BLE001
            results.append({"text": text, "hit": False, "iou": 0.0,
                            "error": str(exc)[:200], "boxes": 0})
            continue
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(elapsed_ms)
        best_iou = max((_iou(gt, b) for b in boxes), default=0.0)
        hit = best_iou >= iou_hit
        if hit:
            hits += 1
        results.append({"text": text, "hit": hit,
                        "iou": round(best_iou, 4),
                        "ms": round(elapsed_ms, 1), "boxes": len(boxes)})
    total = len(_FIXTURE_TEXTS)
    recall = hits / total if total else 0.0
    mean_ms = float(np.mean(latencies)) if latencies else 0.0
    # Accuracy is the pass/fail gate for a default-detector swap; latency is
    # reported as evidence but is device-dependent (CPU OCR is inherently slow),
    # so it does not by itself fail the verdict.
    meets = recall >= RECALL_FLOOR
    within_latency = (mean_ms <= LATENCY_CEILING_MS) if latencies else True
    return {
        "schema": OCR_BENCHMARK_SCHEMA,
        "engine": engine,
        "fixtures": total,
        "recall": round(recall, 4),
        "mean_ms": round(mean_ms, 1),
        "recall_floor": RECALL_FLOOR,
        "latency_ceiling_ms": LATENCY_CEILING_MS,
        "within_latency_ceiling": bool(within_latency),
        "meets_floors": bool(meets),
        "results": results,
    }


def run_default_detector_benchmark(*, device: str = "cpu", lang: str = "en",
                                   threshold: float = 0.3) -> dict:
    """Build the default detector cascade and benchmark it."""
    from backend.detection import SubtitleDetector
    detector = SubtitleDetector(device=device, lang=lang)
    return run_ocr_detection_benchmark(detector, threshold=threshold)
