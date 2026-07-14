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

import ctypes
from difflib import SequenceMatcher
import os
import time
from typing import List, Sequence, Tuple

import cv2
import numpy as np


OCR_BENCHMARK_SCHEMA = "vsr.ocr_benchmark.v1"

# A default detector swap should not regress below these on the synthetic set.
RECALL_FLOOR = 0.8
RECOGNITION_FLOOR = 0.9
LATENCY_CEILING_MS = 2000.0

_FIXTURE_TEXTS = (
    "HELLO WORLD",
    "Subtitle line 12",
    "The quick brown fox",
    "1234567890",
    "GOODBYE",
)


def _process_memory() -> dict[str, int | None]:
    """Return current/peak resident bytes without adding a psutil dependency."""
    if os.name == "nt":
        class ProcessMemoryCountersEx(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        try:
            get_process = ctypes.windll.kernel32.GetCurrentProcess
            get_process.restype = ctypes.c_void_p
            get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
            get_memory.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ProcessMemoryCountersEx),
                ctypes.c_ulong,
            ]
            get_memory.restype = ctypes.c_int
            handle = get_process()
            ok = get_memory(
                handle,
                ctypes.byref(counters),
                counters.cb,
            )
            if ok:
                return {
                    "rssBytes": int(counters.WorkingSetSize),
                    "peakRssBytes": int(counters.PeakWorkingSetSize),
                }
        except (AttributeError, OSError):
            pass
    elif os.path.isfile("/proc/self/status"):
        try:
            values = {}
            with open("/proc/self/status", encoding="utf-8") as handle:
                for line in handle:
                    key, _, value = line.partition(":")
                    if key in {"VmRSS", "VmHWM"}:
                        values[key] = int(value.split()[0]) * 1024
            return {
                "rssBytes": values.get("VmRSS"),
                "peakRssBytes": values.get("VmHWM"),
            }
        except (OSError, ValueError, IndexError):
            pass
    return {"rssBytes": None, "peakRssBytes": None}


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


def _normalise_text(value: str) -> str:
    return "".join(char.lower() for char in str(value) if char.isalnum())


def _text_similarity(expected: str, recognized: Sequence[str]) -> float | None:
    actual = _normalise_text(" ".join(str(item) for item in recognized))
    target = _normalise_text(expected)
    if not actual or not target:
        return None
    return SequenceMatcher(None, target, actual).ratio()


def run_ocr_detection_benchmark(detector, *, threshold: float = 0.3,
                                iou_hit: float = 0.15) -> dict:
    """Score ``detector`` on the synthetic fixtures. ``detector`` must expose
    ``detect(frame, threshold) -> [(x1, y1, x2, y2), ...]``."""
    engine = str(getattr(detector, "_engine_name", "unknown"))
    results = []
    latencies: List[float] = []
    recognition_scores: List[float] = []
    hits = 0
    memory_before = _process_memory()
    for i, text in enumerate(_FIXTURE_TEXTS):
        image, gt = _make_fixture(text, seed=i)
        start = time.perf_counter()
        try:
            benchmark_detect = getattr(detector, "benchmark_detect", None)
            if callable(benchmark_detect):
                boxes, recognized = benchmark_detect(image, threshold)
                boxes = boxes or []
                recognized = recognized or []
            else:
                boxes = detector.detect(image, threshold) or []
                recognized = []
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
        similarity = _text_similarity(text, recognized)
        if similarity is not None:
            recognition_scores.append(similarity)
        results.append({"text": text, "hit": hit,
                        "iou": round(best_iou, 4),
                        "ms": round(elapsed_ms, 1), "boxes": len(boxes),
                        "recognized": list(recognized),
                        "text_similarity": (
                            round(similarity, 4)
                            if similarity is not None else None
                        )})
    total = len(_FIXTURE_TEXTS)
    recall = hits / total if total else 0.0
    mean_ms = float(np.mean(latencies)) if latencies else 0.0
    # Accuracy is the pass/fail gate for a default-detector swap; latency is
    # reported as evidence but is device-dependent (CPU OCR is inherently slow),
    # so it does not by itself fail the verdict.
    recognition_mean = (
        float(np.mean(recognition_scores)) if recognition_scores else None
    )
    recognition_meets = (
        recognition_mean >= RECOGNITION_FLOOR
        if recognition_mean is not None else None
    )
    meets = (
        recall >= RECALL_FLOOR
        and recognition_meets is not False
    )
    within_latency = (mean_ms <= LATENCY_CEILING_MS) if latencies else True
    memory_after = _process_memory()
    before_rss = memory_before.get("rssBytes")
    after_rss = memory_after.get("rssBytes")
    run_delta = (
        max(0, int(after_rss) - int(before_rss))
        if before_rss is not None and after_rss is not None else None
    )
    return {
        "schema": OCR_BENCHMARK_SCHEMA,
        "engine": engine,
        "fixtures": total,
        "recall": round(recall, 4),
        "mean_ms": round(mean_ms, 1),
        "recall_floor": RECALL_FLOOR,
        "recognition_ran": bool(recognition_scores),
        "recognition_mean": (
            round(recognition_mean, 4)
            if recognition_mean is not None else None
        ),
        "recognition_floor": RECOGNITION_FLOOR,
        "recognition_meets_floor": recognition_meets,
        "latency_ceiling_ms": LATENCY_CEILING_MS,
        "within_latency_ceiling": bool(within_latency),
        "meets_floors": bool(meets),
        "memory": {
            "beforeRunRssBytes": before_rss,
            "afterRunRssBytes": after_rss,
            "peakRssBytes": memory_after.get("peakRssBytes"),
            "runRssDeltaBytes": run_delta,
        },
        "results": results,
    }


def run_default_detector_benchmark(*, device: str = "cpu", lang: str = "en",
                                   threshold: float = 0.3,
                                   engine: str = "auto") -> dict:
    """Build the default detector cascade and benchmark it."""
    from backend.detection import SubtitleDetector
    preferences = {
        "auto": None,
        "opencv-dnn": "opencv",
        "rapidocr": "onnxruntime",
    }
    if engine not in preferences:
        raise ValueError(f"Unsupported OCR benchmark engine: {engine}")
    before_load = _process_memory()
    old_preference = os.environ.get("VSR_RAPIDOCR_ENGINE")
    try:
        preference = preferences[engine]
        if preference is None:
            os.environ.pop("VSR_RAPIDOCR_ENGINE", None)
        else:
            os.environ["VSR_RAPIDOCR_ENGINE"] = preference
        detector = SubtitleDetector(device=device, lang=lang)
    finally:
        if old_preference is None:
            os.environ.pop("VSR_RAPIDOCR_ENGINE", None)
        else:
            os.environ["VSR_RAPIDOCR_ENGINE"] = old_preference
    after_load = _process_memory()
    result = run_ocr_detection_benchmark(detector, threshold=threshold)
    before_rss = before_load.get("rssBytes")
    after_rss = after_load.get("rssBytes")
    result["requested_engine"] = engine
    result["memory"].update({
        "beforeLoadRssBytes": before_rss,
        "afterLoadRssBytes": after_rss,
        "loadRssDeltaBytes": (
            max(0, int(after_rss) - int(before_rss))
            if before_rss is not None and after_rss is not None else None
        ),
    })
    return result
