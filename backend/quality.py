"""SSIM and quality-metric primitives.

Extracted from processor.py as part of RFP-L-1. ``_compute_quality_report``
and ``_write_quality_sheet`` are methods on ``SubtitleRemover`` (they read
``self.config`` + ``self._quality_mask_bbox``) so they stay there; only
the pure-numpy SSIM helper and optional ffmpeg-backed metrics live here.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural Similarity between two BGR frames. Mean over the three
    channels. Standard formulation (C1, C2 = (0.01*255)^2, (0.03*255)^2).
    Flat-colour regions where the variance and covariance are all zero
    can still drive (num/den) close to 0/0; we wrap in errstate +
    nan_to_num so the report never yields NaN or inf.
    """
    if a is None or b is None or a.shape != b.shape or a.ndim < 2:
        return 0.0
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    if a.ndim == 2:
        a32 = a32[..., None]
        b32 = b32[..., None]
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    channels = a32.shape[2]
    ssims: List[float] = []
    with np.errstate(invalid='ignore', divide='ignore'):
        for c in range(channels):
            x = a32[..., c]
            y = b32[..., c]
            mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
            mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
            mu_x2 = mu_x * mu_x
            mu_y2 = mu_y * mu_y
            mu_xy = mu_x * mu_y
            sig_x2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2
            sig_y2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2
            sig_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy
            num = (2 * mu_xy + C1) * (2 * sig_xy + C2)
            den = (mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2)
            ratio = np.where(den > 0, num / np.maximum(den, 1e-12), 1.0)
            ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=0.0)
            ssims.append(float(np.mean(ratio)))
    if not ssims:
        return 0.0
    return float(np.clip(np.mean(ssims), 0.0, 1.0))


def temporal_flicker_score(
    samples: Sequence[Tuple[int, np.ndarray]],
    *,
    max_frame_gap: int = 1,
) -> Optional[float]:
    """Mean adjacent-frame absolute delta for sampled ROI frames.

    The score is normalized to 0..1. Only adjacent sample indices are compared
    by default so sparse random quality samples do not confuse ordinary motion
    with flicker.
    """
    if len(samples) < 2:
        return None
    prepared: list[Tuple[int, np.ndarray]] = []
    for idx, frame in samples:
        arr = _prepare_flicker_frame(frame)
        if arr is not None:
            prepared.append((int(idx), arr))
    if len(prepared) < 2:
        return None
    diffs: List[float] = []
    last_idx, last = prepared[0]
    for idx, cur in prepared[1:]:
        if idx - last_idx <= max_frame_gap:
            diffs.append(float(np.mean(np.abs(cur - last)) / 255.0))
        last_idx, last = idx, cur
    if not diffs:
        return None
    return float(np.mean(diffs))


def residual_text_score(frame: np.ndarray) -> Optional[float]:
    """Return a cheap 0..1 text-residue score for a cleaned ROI.

    This is intentionally dependency-free. It is not a replacement for OCR; it
    flags sharp high-contrast strokes that commonly remain when subtitle text
    was under-masked or when an inpaint fallback left outlines behind.
    """
    if frame is None or frame.size == 0:
        return None
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 2:
        gray = frame
    else:
        return None
    h, w = gray.shape[:2]
    if h < 8 or w < 8:
        return None
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    median = float(np.median(blurred))
    contrast = cv2.absdiff(blurred, np.full_like(blurred, int(round(median))))
    _, mask = cv2.threshold(
        contrast, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(1, w // 96), max(1, h // 96)),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(blurred, 50, 150)
    contour_area = 0.0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if cw < 3 or ch < 3:
            continue
        if ch > h * 0.65:
            continue
        aspect = cw / max(1.0, float(ch))
        if aspect < 0.15 or aspect > 30.0:
            continue
        contour_area += float(cw * ch)
    area = float(max(1, h * w))
    edge_density = float(np.count_nonzero(edges)) / area
    contour_density = contour_area / area
    return float(min(1.0, max(edge_density, contour_density)))


def _pyiqa_available() -> bool:
    """True when pyiqa is importable (opt-in advanced metrics)."""
    try:
        import pyiqa  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def compute_extended_metrics(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    metric_names: Sequence[str] = ("lpips", "dists"),
    device: str = "cpu",
) -> dict:
    """Compute perceptual quality metrics via pyiqa on (original, cleaned)
    frame pairs. Returns {metric_name: mean_score} for each available
    metric, or an empty dict when pyiqa is not installed.

    Metrics are computed on the ROI crop when provided; pairs should
    already be cropped by the caller. All scores are lower-is-better
    (distance metrics); pyiqa handles the convention internally.
    """
    try:
        import pyiqa  # type: ignore
    except ImportError:
        return {}
    if not pairs:
        return {}
    try:
        import torch
    except ImportError:
        return {}
    results: dict = {}
    for name in metric_names:
        try:
            metric_fn = pyiqa.create_metric(name, device=device)
        except Exception:
            logger.debug(f"pyiqa metric {name!r} unavailable")
            continue
        scores: List[float] = []
        for orig, cleaned in pairs:
            try:
                a = _frame_to_tensor(orig, device)
                b = _frame_to_tensor(cleaned, device)
                with torch.no_grad():
                    score = metric_fn(a, b)
                scores.append(float(score.item()))
            except Exception:
                continue
        if scores:
            results[name] = float(np.mean(scores))
    return results


def temporal_consistency_score(
    frames: Sequence[np.ndarray],
) -> Optional[float]:
    """Mean pairwise SSIM between consecutive cleaned ROI frames.

    High values (close to 1.0) indicate temporally stable inpainting;
    low values indicate flicker or per-frame inconsistency. Complements
    the simpler absolute-delta flicker score with a structural measure.
    """
    if len(frames) < 2:
        return None
    scores: List[float] = []
    for i in range(len(frames) - 1):
        s = _ssim(frames[i], frames[i + 1])
        if s > 0:
            scores.append(s)
    return float(np.mean(scores)) if scores else None


def _mask_bbox_from_masks(
    masks: Sequence[np.ndarray],
    *,
    padding: int = 4,
) -> Optional[Tuple[int, int, int, int]]:
    if not masks:
        return None
    height = width = None
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for mask in masks:
        if mask is None:
            continue
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        if arr.ndim != 2:
            continue
        if height is None or width is None:
            height, width = arr.shape[:2]
        y, x = np.where(arr > 0)
        if x.size:
            xs.append(x)
            ys.append(y)
    if not xs or height is None or width is None:
        return None
    all_x = np.concatenate(xs)
    all_y = np.concatenate(ys)
    x1 = max(0, int(all_x.min()) - padding)
    y1 = max(0, int(all_y.min()) - padding)
    x2 = min(width, int(all_x.max()) + padding + 1)
    y2 = min(height, int(all_y.max()) + padding + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _crop_bbox(frame: np.ndarray,
               bbox: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    if bbox is None:
        return frame
    x1, y1, x2, y2 = bbox
    return frame[y1:y2, x1:x2]


def _mean_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(clean)) if clean else None


def _static_logo_mask_coverage(masks: Sequence[np.ndarray]) -> float:
    total_pixels = 0
    masked_pixels = 0
    for mask in masks:
        if mask is None:
            continue
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        if arr.ndim != 2:
            continue
        total_pixels += int(arr.shape[0] * arr.shape[1])
        masked_pixels += int(np.count_nonzero(arr))
    if total_pixels <= 0:
        return 0.0
    return float(masked_pixels / total_pixels)


def _best_method(
    method_metrics: dict,
    key: str,
    *,
    higher_is_better: bool = False,
) -> Optional[str]:
    candidates = []
    for name, metrics in method_metrics.items():
        value = metrics.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            candidates.append((float(value), name))
    if not candidates:
        return None
    candidates.sort(reverse=higher_is_better)
    return candidates[0][1]


def compare_static_logo_cleanup(
    original_frames: Sequence[np.ndarray],
    method_outputs: dict,
    masks: Sequence[np.ndarray],
    *,
    reference_frames: Optional[Sequence[np.ndarray]] = None,
) -> dict:
    """Compare static-logo cleanup outputs using dependency-free ROI metrics.

    Lower residual/flicker is better; higher temporal consistency and reference
    SSIM are better. This is a benchmark primitive, not a production quality
    gate, so it returns all available metrics and omits unavailable optional
    values instead of trying to judge pass/fail.
    """
    originals = list(original_frames)
    masks = list(masks)
    if len(originals) != len(masks):
        raise ValueError("original frames and masks must have the same length")
    if not originals:
        raise ValueError("static-logo comparison needs at least one frame")
    bbox = _mask_bbox_from_masks(masks)
    refs = list(reference_frames) if reference_frames is not None else None
    if refs is not None and len(refs) != len(originals):
        raise ValueError("reference frames must match original frame count")

    metrics_by_method = {}
    for name, frames in sorted(method_outputs.items()):
        output_frames = list(frames)
        if len(output_frames) != len(originals):
            raise ValueError(f"method {name!r} frame count does not match input")
        roi_frames = [_crop_bbox(frame, bbox) for frame in output_frames]
        residuals = [residual_text_score(frame) for frame in roi_frames]
        samples = list(enumerate(roi_frames))
        method_metrics = {
            "roiFrameCount": len(roi_frames),
            "residualTextScoreMean": _mean_optional(residuals),
            "temporalFlickerScore": temporal_flicker_score(samples),
            "temporalConsistency": temporal_consistency_score(roi_frames),
        }
        if refs is not None:
            ref_scores: List[Optional[float]] = []
            for ref, out in zip(refs, output_frames):
                ref_roi = _crop_bbox(ref, bbox)
                out_roi = _crop_bbox(out, bbox)
                ref_scores.append(_ssim(ref_roi, out_roi))
            method_metrics["ssimVsReferenceMean"] = _mean_optional(ref_scores)
        metrics_by_method[str(name)] = method_metrics

    return {
        "maskCoverage": _static_logo_mask_coverage(masks),
        "roiBbox": list(bbox) if bbox is not None else None,
        "methods": metrics_by_method,
        "winners": {
            "lowestResidualText": _best_method(
                metrics_by_method, "residualTextScoreMean"),
            "lowestFlicker": _best_method(
                metrics_by_method, "temporalFlickerScore"),
            "highestTemporalConsistency": _best_method(
                metrics_by_method, "temporalConsistency",
                higher_is_better=True),
            "highestReferenceSsim": _best_method(
                metrics_by_method, "ssimVsReferenceMean",
                higher_is_better=True),
        },
    }


def _frame_to_tensor(frame: np.ndarray, device: str = "cpu"):
    """BGR uint8 frame -> torch float32 NCHW tensor in [0, 1]."""
    import torch
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return t.to(device)


def _prepare_flicker_frame(frame: np.ndarray) -> Optional[np.ndarray]:
    if frame is None or frame.size == 0:
        return None
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 2:
        gray = frame
    else:
        return None
    h, w = gray.shape[:2]
    if h <= 0 or w <= 0:
        return None
    scale = min(1.0, 96.0 / max(h, w))
    if scale < 1.0:
        gray = cv2.resize(
            gray,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return gray.astype(np.float32)


def ffmpeg_libvmaf_available(ffmpeg: str = "ffmpeg") -> bool:
    """Return True when `ffmpeg -filters` reports libvmaf."""
    if shutil.which(ffmpeg) is None:
        return False
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    output = f"{result.stdout}\n{result.stderr}"
    return result.returncode == 0 and bool(
        re.search(r"(?m)^\s*[.A-Z| ]+\s+libvmaf\s+", output)
    )


def _escape_filter_value(value: str) -> str:
    escaped = str(value).replace("\\", "\\\\")
    for char in (":", "'", ",", "[", "]"):
        escaped = escaped.replace(char, f"\\{char}")
    return escaped


def _unescape_filter_value(value: str) -> str:
    out = []
    escaped = False
    for char in value:
        if escaped:
            out.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        else:
            out.append(char)
    if escaped:
        out.append("\\")
    return "".join(out)


def _vmaf_filter(log_path: str,
                 roi: Optional[Tuple[int, int, int, int]] = None) -> str:
    ref = "[0:v]setpts=PTS-STARTPTS"
    dist = "[1:v]setpts=PTS-STARTPTS"
    if roi is not None:
        x1, y1, x2, y2 = roi
        width = max(1, int(x2) - int(x1))
        height = max(1, int(y2) - int(y1))
        ref += f",crop={width}:{height}:{int(x1)}:{int(y1)}"
        dist += f",crop={width}:{height}:{int(x1)}:{int(y1)}"
    return (
        f"{ref}[ref];{dist}[dist];"
        f"[dist][ref]libvmaf=log_fmt=json:log_path={_escape_filter_value(log_path)}"
    )


def _read_vmaf_score(log_path: Path) -> Optional[float]:
    try:
        payload = json.loads(log_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    pooled = payload.get("pooled_metrics", {})
    score = pooled.get("vmaf", {}).get("mean")
    if isinstance(score, (int, float)):
        return float(score)
    frame_scores = []
    for frame in payload.get("frames", []) or []:
        value = (frame.get("metrics") or {}).get("vmaf")
        if isinstance(value, (int, float)):
            frame_scores.append(float(value))
    if frame_scores:
        return float(np.mean(frame_scores))
    return None


def compute_vmaf(
    reference_path: str,
    distorted_path: str,
    *,
    start_seconds: float = 0.0,
    duration_seconds: Optional[float] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
    ffmpeg: str = "ffmpeg",
) -> Optional[float]:
    """Compute VMAF via ffmpeg's libvmaf filter.

    Returns None when ffmpeg/libvmaf is unavailable or the invocation
    fails. The first input is the reference/original, the second is the
    distorted/cleaned output.
    """
    if not Path(reference_path).is_file() or not Path(distorted_path).is_file():
        return None
    if not ffmpeg_libvmaf_available(ffmpeg):
        logger.info("ffmpeg libvmaf filter unavailable; VMAF omitted.")
        return None
    duration = None if duration_seconds is None else max(0.1, float(duration_seconds))
    start = max(0.0, float(start_seconds))
    with tempfile.TemporaryDirectory(prefix="vsr_vmaf_") as tmpdir:
        log_path = Path(tmpdir) / "vmaf.json"
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats"]
        if start > 0:
            cmd += ["-ss", f"{start:.3f}"]
        if duration is not None:
            cmd += ["-t", f"{duration:.3f}"]
        cmd += ["-i", reference_path]
        if duration is not None:
            cmd += ["-t", f"{duration:.3f}"]
        cmd += [
            "-i", distorted_path,
            "-lavfi", _vmaf_filter(str(log_path), roi),
            "-f", "null", "-",
        ]
        timeout = 600.0 if duration is None else max(600.0, duration * 20.0)
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr or b""
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", "replace")
            logger.info(f"ffmpeg libvmaf failed: {stderr[:400]}")
            return None
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg libvmaf timed out")
            return None
        except FileNotFoundError:
            return None
        return _read_vmaf_score(log_path)
