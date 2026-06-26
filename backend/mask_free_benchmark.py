"""Benchmark helpers for mask-free subtitle-erasure research.

CLEAR and SEDiT-style systems do not consume OCR masks at inference time, but
VSR still needs a rights-cleared evaluation target before any adapter becomes
user-facing. This module only evaluates supplied candidate outputs; it does not
run model code, shell commands, or downloads.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

import cv2
import numpy as np

from backend.quality import (
    _ssim,
    residual_text_score,
    temporal_consistency_score,
    temporal_flicker_score,
)

MASK_FREE_BENCHMARK_SCHEMA = "vsr.mask_free_subtitle_benchmark.v1"
MASK_FREE_CATEGORY = "mask_free_subtitle"
REFERENCE_LICENSE_ALLOWLIST = {
    "cc0",
    "cc0-1.0",
    "public-domain",
    "public domain",
    "mit",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc-by-4.0",
}


class MaskFreeBenchmarkError(ValueError):
    """Raised when mask-free benchmark inputs violate the clip policy."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_license(value: object) -> str:
    return str(value or "").strip().lower()


def _load_manifest(path: Path | str) -> dict:
    manifest_path = Path(path)
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MaskFreeBenchmarkError(
            f"reference manifest could not be read: {manifest_path}"
        ) from exc
    if not isinstance(data, dict):
        raise MaskFreeBenchmarkError("reference manifest must be a JSON object")
    if not isinstance(data.get("clips"), list):
        raise MaskFreeBenchmarkError("reference manifest clips must be a list")
    return data


def iter_mask_free_manifest_entries(
    manifest_path: Path | str,
    clips_dir: Path | str,
) -> list[dict]:
    """Return verified mask-free subtitle benchmark entries."""
    data = _load_manifest(manifest_path)
    root = Path(clips_dir)
    entries: list[dict] = []
    required = {
        "filename", "license", "contributor", "sha256", "failure_category",
        "config", "metric_floors", "evaluation",
    }
    for index, entry in enumerate(data.get("clips", [])):
        if not isinstance(entry, dict):
            raise MaskFreeBenchmarkError(f"clip entry {index} must be an object")
        if entry.get("failure_category") != MASK_FREE_CATEGORY:
            continue
        missing = required - set(entry.keys())
        if missing:
            raise MaskFreeBenchmarkError(
                f"mask-free clip {index} missing fields: {sorted(missing)}"
            )
        evaluation = entry.get("evaluation")
        if not isinstance(evaluation, dict) or "subtitle_regions" not in evaluation:
            raise MaskFreeBenchmarkError(
                f"mask-free clip {entry.get('filename')} needs evaluation.subtitle_regions"
            )
        license_name = _normalize_license(entry.get("license"))
        if license_name not in REFERENCE_LICENSE_ALLOWLIST:
            raise MaskFreeBenchmarkError(
                f"mask-free clip {entry.get('filename')} has unsupported "
                f"license {entry.get('license')!r}"
            )
        filename = str(entry.get("filename") or "").replace("\\", "/")
        if filename.startswith("/") or ".." in Path(filename).parts:
            raise MaskFreeBenchmarkError(
                f"mask-free clip has unsafe filename: {filename!r}"
            )
        path = root / filename
        if not path.is_file():
            raise MaskFreeBenchmarkError(f"mask-free clip file is missing: {filename}")
        expected = str(entry.get("sha256") or "").strip().lower()
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise MaskFreeBenchmarkError(
                f"mask-free clip {filename} has invalid sha256"
            )
        actual = _sha256_file(path)
        if actual != expected:
            raise MaskFreeBenchmarkError(
                f"mask-free clip {filename} sha256 mismatch"
            )
        enriched = dict(entry)
        enriched["path"] = str(path)
        entries.append(enriched)
    return entries


def _is_rect(value: object) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    return all(isinstance(v, (int, float)) for v in value)


def _coerce_rect(rect: Sequence[float],
                 shape: tuple[int, int]) -> Optional[tuple[int, int, int, int]]:
    height, width = shape
    x1, y1, x2, y2 = [int(round(float(v))) for v in rect]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _rects_for_frame(
    subtitle_regions: object,
    frame_index: int,
    frame_count: int,
) -> list[Sequence[float]]:
    if _is_rect(subtitle_regions):
        return [subtitle_regions]  # type: ignore[list-item]
    if not isinstance(subtitle_regions, (list, tuple)):
        raise MaskFreeBenchmarkError("subtitle_regions must be a rect or rect list")
    if all(_is_rect(item) for item in subtitle_regions):
        return list(subtitle_regions)  # type: ignore[arg-type]
    if len(subtitle_regions) != frame_count:
        raise MaskFreeBenchmarkError(
            "per-frame subtitle_regions must match frame count"
        )
    frame_regions = subtitle_regions[frame_index]
    if _is_rect(frame_regions):
        return [frame_regions]  # type: ignore[list-item]
    if isinstance(frame_regions, (list, tuple)) and all(
        _is_rect(item) for item in frame_regions
    ):
        return list(frame_regions)  # type: ignore[arg-type]
    raise MaskFreeBenchmarkError(
        f"subtitle_regions[{frame_index}] must be a rect or rect list"
    )


def masks_from_subtitle_regions(
    frame_shape: tuple[int, int] | tuple[int, int, int],
    frame_count: int,
    subtitle_regions: object,
) -> list[np.ndarray]:
    """Build evaluation masks from static or per-frame subtitle rectangles."""
    height, width = int(frame_shape[0]), int(frame_shape[1])
    masks: list[np.ndarray] = []
    for idx in range(frame_count):
        mask = np.zeros((height, width), dtype=np.uint8)
        for rect in _rects_for_frame(subtitle_regions, idx, frame_count):
            coerced = _coerce_rect(rect, (height, width))
            if coerced is None:
                continue
            x1, y1, x2, y2 = coerced
            mask[y1:y2, x1:x2] = 255
        masks.append(mask)
    return masks


def _validate_masks(
    masks: Sequence[np.ndarray],
    frame_shape: tuple[int, int, int],
    frame_count: int,
) -> list[np.ndarray]:
    if len(masks) != frame_count:
        raise MaskFreeBenchmarkError("mask count must match frame count")
    out: list[np.ndarray] = []
    for idx, mask in enumerate(masks):
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        if arr.shape != frame_shape[:2]:
            raise MaskFreeBenchmarkError(f"mask {idx} shape does not match frames")
        out.append((arr > 0).astype(np.uint8) * 255)
    return out


def _bbox(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    y, x = np.where(mask > 0)
    if not x.size:
        return None
    return int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1


def _crop(frame: np.ndarray,
          bbox: Optional[tuple[int, int, int, int]]) -> np.ndarray:
    if bbox is None:
        return frame
    x1, y1, x2, y2 = bbox
    return frame[y1:y2, x1:x2]


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(clean)) if clean else None


def _outside_mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> Optional[float]:
    outside = mask == 0
    if not bool(np.any(outside)):
        return None
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return float(np.mean(diff[outside]) / 255.0)


def _resolve_masks(
    frames: Sequence[np.ndarray],
    *,
    subtitle_regions: object = None,
    masks: Optional[Sequence[np.ndarray]] = None,
) -> list[np.ndarray]:
    if masks is not None:
        return _validate_masks(masks, frames[0].shape, len(frames))
    if subtitle_regions is None:
        raise MaskFreeBenchmarkError("subtitle_regions or masks are required")
    return masks_from_subtitle_regions(frames[0].shape, len(frames), subtitle_regions)


def compare_mask_free_subtitle_outputs(
    original_frames: Sequence[np.ndarray],
    method_outputs: Mapping[str, Sequence[np.ndarray]],
    *,
    subtitle_regions: object = None,
    masks: Optional[Sequence[np.ndarray]] = None,
    reference_frames: Optional[Sequence[np.ndarray]] = None,
    runtime_seconds: Optional[Mapping[str, float]] = None,
) -> dict:
    """Score mask-free subtitle-erasure candidate outputs.

    Evaluation masks are ground-truth regions used only for scoring. Candidate
    methods are assumed to have run without consuming those masks.
    """
    originals = list(original_frames)
    if not originals:
        raise MaskFreeBenchmarkError("benchmark needs at least one frame")
    first_shape = originals[0].shape
    for idx, frame in enumerate(originals):
        if frame.shape != first_shape:
            raise MaskFreeBenchmarkError(f"frame {idx} shape does not match frame 0")
    eval_masks = _resolve_masks(
        originals, subtitle_regions=subtitle_regions, masks=masks)
    refs = list(reference_frames) if reference_frames is not None else None
    if refs is not None and len(refs) != len(originals):
        raise MaskFreeBenchmarkError("reference frames must match input frame count")
    runtimes = dict(runtime_seconds or {})

    metrics_by_method: dict[str, dict] = {}
    for name, output_sequence in sorted(method_outputs.items()):
        outputs = list(output_sequence)
        if len(outputs) != len(originals):
            raise MaskFreeBenchmarkError(
                f"method {name!r} frame count does not match input"
            )
        roi_frames: list[np.ndarray] = []
        residuals: list[Optional[float]] = []
        reference_scores: list[Optional[float]] = []
        artifact_scores: list[Optional[float]] = []
        for idx, (original, output, mask) in enumerate(zip(originals, outputs, eval_masks)):
            if output.shape != first_shape:
                raise MaskFreeBenchmarkError(
                    f"method {name!r} frame {idx} shape does not match input"
                )
            box = _bbox(mask)
            roi = _crop(output, box)
            roi_frames.append(roi)
            residuals.append(residual_text_score(roi))
            outside_reference = refs[idx] if refs is not None else original
            artifact_scores.append(_outside_mae(output, outside_reference, mask))
            if refs is not None:
                reference_scores.append(_ssim(_crop(refs[idx], box), roi))
        residual_mean = _mean(residuals)
        if reference_scores:
            subtitle_quality = _mean(reference_scores)
        elif residual_mean is not None:
            subtitle_quality = float(max(0.0, 1.0 - residual_mean))
        else:
            subtitle_quality = None
        runtime = runtimes.get(name)
        method_metrics = {
            "frameCount": len(outputs),
            "runtimeSeconds": (
                float(runtime) if runtime is not None and runtime >= 0 else None
            ),
            "secondsPerFrame": (
                float(runtime) / len(outputs)
                if runtime is not None and runtime >= 0 and outputs else None
            ),
            "artifactScore": _mean(artifact_scores),
            "subtitleRemovalQuality": subtitle_quality,
            "subtitleResidualTextScore": residual_mean,
            "temporalFlickerScore": temporal_flicker_score(list(enumerate(roi_frames))),
            "temporalConsistency": temporal_consistency_score(roi_frames),
        }
        if reference_scores:
            method_metrics["subtitleReferenceSsimMean"] = _mean(reference_scores)
        metrics_by_method[str(name)] = method_metrics

    return {
        "evaluationMaskCoverage": float(
            np.mean([np.count_nonzero(m) / m.size for m in eval_masks])
        ),
        "methods": metrics_by_method,
        "winners": {
            "highestSubtitleRemovalQuality": _best_method(
                metrics_by_method, "subtitleRemovalQuality",
                higher_is_better=True),
            "lowestArtifactScore": _best_method(
                metrics_by_method, "artifactScore"),
            "fastestRuntime": _best_method(
                metrics_by_method, "runtimeSeconds"),
        },
    }


def _best_method(method_metrics: Mapping[str, Mapping[str, object]],
                 key: str,
                 *,
                 higher_is_better: bool = False) -> Optional[str]:
    candidates: list[tuple[float, str]] = []
    for name, metrics in method_metrics.items():
        value = metrics.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            candidates.append((float(value), name))
    if not candidates:
        return None
    candidates.sort(reverse=higher_is_better)
    return candidates[0][1]


def run_mask_free_subtitle_benchmark(
    original_frames: Sequence[np.ndarray],
    method_outputs: Mapping[str, Sequence[np.ndarray]],
    *,
    subtitle_regions: object = None,
    masks: Optional[Sequence[np.ndarray]] = None,
    reference_frames: Optional[Sequence[np.ndarray]] = None,
    runtime_seconds: Optional[Mapping[str, float]] = None,
) -> dict:
    metrics = compare_mask_free_subtitle_outputs(
        original_frames,
        method_outputs,
        subtitle_regions=subtitle_regions,
        masks=masks,
        reference_frames=reference_frames,
        runtime_seconds=runtime_seconds,
    )
    return {
        "schema": MASK_FREE_BENCHMARK_SCHEMA,
        "category": MASK_FREE_CATEGORY,
        "frameCount": len(original_frames),
        "methods": sorted(method_outputs.keys()),
        "metrics": metrics,
    }
