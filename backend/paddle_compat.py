"""PaddleOCR 2.x/3.x compatibility helpers.

PaddleOCR 3.0 removed ``use_gpu``/``show_log``/``use_angle_cls`` from the
constructor (device selection moved to ``device=``, angle classification
to ``use_textline_orientation=``) and replaced the 2.x
``.ocr(img, cls=...)`` nested-list result with ``.predict(img)``
returning dict-like result objects keyed by ``rec_polys``/``rec_scores``.
requirements.txt pins ``paddleocr>=3.0.0``, but a pre-existing 2.x
install must keep working, so every PaddleOCR call site goes through
these two helpers instead of touching the API directly.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]


def build_paddleocr(lang: str, device: str, **extra):
    """Construct a PaddleOCR instance on either major version.

    Tries the 3.x constructor first (matching the requirements pin),
    falls back to the 2.x keyword set on TypeError. Unknown ``extra``
    kwargs raise TypeError from both attempts so callers can detect
    unsupported variants (e.g. the VL model selector).
    """
    from paddleocr import PaddleOCR

    use_cuda = "cuda" in device
    try:
        return PaddleOCR(
            lang=lang,
            device="gpu" if use_cuda else "cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            **extra,
        )
    except TypeError:
        return PaddleOCR(
            lang=lang,
            use_angle_cls=False,
            use_gpu=use_cuda,
            show_log=False,
            **extra,
        )


def extract_paddle_boxes(model, frame: np.ndarray,
                         threshold: float) -> List[Box]:
    """Run det+rec on ``frame`` and return axis-aligned boxes for lines
    scoring at or above ``threshold``, on either major version."""
    if hasattr(model, "predict"):
        return _extract_v3(model, frame, threshold)
    return _extract_v2(model, frame, threshold)


def _extract_v3(model, frame: np.ndarray, threshold: float) -> List[Box]:
    boxes: List[Box] = []
    results = model.predict(frame)
    for res in results or []:
        if isinstance(res, dict):
            data = res
        else:
            data = getattr(res, "json", None)
        if not isinstance(data, dict):
            continue
        # Some PaddleX releases nest the payload under a "res" key.
        data = data.get("res", data)
        polys = data.get("rec_polys")
        if polys is None:
            polys = data.get("dt_polys")
        if polys is None:
            continue
        scores = data.get("rec_scores") or []
        for idx, poly in enumerate(polys):
            try:
                score = float(scores[idx]) if idx < len(scores) else 1.0
            except (TypeError, ValueError):
                score = 1.0
            if score < threshold:
                continue
            try:
                pts = np.array(poly, dtype=np.float32)
            except (TypeError, ValueError):
                continue
            if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2:
                continue
            x1, y1 = pts[:, 0].min(), pts[:, 1].min()
            x2, y2 = pts[:, 0].max(), pts[:, 1].max()
            if x2 > x1 and y2 > y1:
                boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes


def _extract_v2(model, frame: np.ndarray, threshold: float) -> List[Box]:
    boxes: List[Box] = []
    try:
        results = model.ocr(frame, cls=False)
    except TypeError:
        results = model.ocr(frame)
    if results and results[0]:
        for line in results[0]:
            if line[1][1] >= threshold:
                pts = np.array(line[0], dtype=np.int32)
                x1, y1 = pts.min(axis=0)
                x2, y2 = pts.max(axis=0)
                boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes
