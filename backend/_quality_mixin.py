"""Quality report and seam-score methods for SubtitleRemover.

This mixin is mixed into ``SubtitleRemover`` so the methods retain full
``self`` access while living in a dedicated file. It covers the PSNR/SSIM
quality report, side-by-side sheet rendering, and per-batch seam-score
accumulation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from backend.io import _open_capture
from backend.mask_corrections import make_review_span, merge_review_spans
from backend.quality import (
    _ssim,
    compute_vmaf,
    compute_extended_metrics,
    temporal_consistency_score,
    residual_text_score,
    temporal_flicker_score,
    mask_boundary_seam_score,
)
from backend.quality_gate import (
    RESIDUAL_TEXT_SCORE_CEILING,
    TEMPORAL_FLICKER_CEILING,
    evaluate_quality_gate,
)

logger = logging.getLogger(__name__)


def _seek_capture_to_frame_deferred(cap, target):
    from backend.processor import _seek_capture_to_frame
    return _seek_capture_to_frame(cap, target)


def _frame_seconds_deferred(index, fps, timing=None):
    from backend.processor import _frame_seconds
    return _frame_seconds(index, fps, timing)


class _QualityMixin:
    """Quality report, quality sheet, and seam-score methods."""

    def _accumulate_quality_bbox(self, mask: np.ndarray) -> None:
        """Update the union-mask bbox used by the quality report ROI."""
        if mask is None or mask.size == 0 or mask.max() == 0:
            return
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            return
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        if self._quality_mask_bbox is None:
            self._quality_mask_bbox = (x1, y1, x2, y2)
        else:
            ox1, oy1, ox2, oy2 = self._quality_mask_bbox
            self._quality_mask_bbox = (
                min(ox1, x1), min(oy1, y1),
                max(ox2, x2), max(oy2, y2),
            )

    def _accumulate_seam_scores(self, frames, results, masks,
                                max_samples: int = 32) -> None:
        """Sample mask-boundary seam scores across a processed batch."""
        if len(self._seam_scores) >= max_samples:
            return
        n = min(len(frames), len(results), len(masks))
        if n == 0:
            return
        step = max(1, n // 3)
        for i in range(0, n, step):
            if len(self._seam_scores) >= max_samples:
                break
            try:
                score = mask_boundary_seam_score(frames[i], results[i], masks[i])
            except Exception:
                if not getattr(self, "_seam_score_failure_logged", False):
                    logger.warning(
                        "Seam-score sampling failed; the quality report may "
                        "omit boundary-seam evidence",
                        exc_info=True,
                    )
                    self._seam_score_failure_logged = True
                score = None
            if score is not None:
                self._seam_scores.append(score)

    def _compute_quality_report(self, input_path: str, output_path: str,
                                  start_frame: int, end_frame: int,
                                  fps: float, n_samples: int = 10) -> Optional[dict]:
        """Sample N random frames, compute PSNR/SSIM between input and output."""
        cap_in = _open_capture(
            input_path, self.config.decode_hw_accel,
            input_fps=self.config.input_fps,
        )
        cap_out = _open_capture(output_path, "off")
        if not cap_in.isOpened() or not cap_out.isOpened():
            try:
                cap_in.release()
            except Exception:
                logger.debug("Quality source capture release failed", exc_info=True)
            try:
                cap_out.release()
            except Exception:
                logger.debug("Quality output capture release failed", exc_info=True)
            return None
        try:
            span = max(1, end_frame - start_frame)
            out_total = int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT)) or span
            rng = np.random.default_rng(seed=42)
            metric_indices = sorted(set(rng.integers(0, span, size=n_samples).tolist()))
            metric_index_set = set(metric_indices)
            flicker_indices = sorted(set(
                metric_indices
                + [idx + 1 for idx in metric_indices if idx + 1 < span]
            ))

            psnrs: List[float] = []
            ssims: List[float] = []
            roi_psnrs: List[float] = []
            roi_ssims: List[float] = []
            temporal_samples: List[Tuple[int, np.ndarray]] = []
            residual_scores: List[float] = []
            review_spans = list(
                getattr(self, "_mask_review_signals", None) or [])
            roi = self._quality_mask_bbox
            roi_ready = (
                roi is not None
                and (roi[2] - roi[0]) >= 32
                and (roi[3] - roi[1]) >= 32
            )
            pairs: List[Tuple[int, np.ndarray, np.ndarray, float, float]] = []
            for idx in flicker_indices:
                _seek_capture_to_frame_deferred(cap_in, start_frame + idx)
                ok_in, a = cap_in.read()
                _seek_capture_to_frame_deferred(cap_out, min(out_total - 1, idx))
                ok_out, b = cap_out.read()
                if not (ok_in and ok_out):
                    continue
                if a.shape != b.shape:
                    b = cv2.resize(b, (a.shape[1], a.shape[0]),
                                    interpolation=cv2.INTER_AREA)
                a_roi = None
                b_roi = None
                if roi_ready:
                    x1, y1, x2, y2 = roi
                    x1 = max(0, min(a.shape[1] - 1, x1))
                    x2 = max(x1 + 1, min(a.shape[1], x2))
                    y1 = max(0, min(a.shape[0] - 1, y1))
                    y2 = max(y1 + 1, min(a.shape[0], y2))
                    a_roi = a[y1:y2, x1:x2]
                    b_roi = b[y1:y2, x1:x2]
                    if b_roi.size:
                        temporal_samples.append((idx, b_roi.copy()))
                        if idx in metric_index_set:
                            residual = residual_text_score(b_roi)
                            if residual is not None:
                                residual_scores.append(residual)
                                if residual > RESIDUAL_TEXT_SCORE_CEILING:
                                    review_spans.append(make_review_span(
                                        "residual",
                                        start_frame + idx,
                                        start_frame + idx + 1,
                                        fps=fps,
                                        score=residual,
                                        threshold=RESIDUAL_TEXT_SCORE_CEILING,
                                        reason=(
                                            "Residual text score exceeded "
                                            "the review threshold"
                                        ),
                                    ))
                if idx not in metric_index_set:
                    continue
                p = cv2.PSNR(a, b)
                s = _ssim(a, b)
                psnrs.append(p)
                ssims.append(s)
                if a_roi is not None and b_roi is not None:
                    if a_roi.size and a_roi.shape == b_roi.shape:
                        try:
                            roi_psnrs.append(float(cv2.PSNR(a_roi, b_roi)))
                            roi_ssims.append(_ssim(a_roi, b_roi))
                        except Exception:
                            logger.warning(
                                "Quality ROI metric calculation failed",
                                exc_info=True,
                            )
                if self.config.quality_report_sheet:
                    pairs.append((idx, a, b, p, s))
            if not psnrs:
                return None
            mean_ssim = float(np.mean(ssims))
            mean_psnr = float(np.mean(psnrs))
            roi_mean_ssim = float(np.mean(roi_ssims)) if roi_ssims else None
            roi_mean_psnr = float(np.mean(roi_psnrs)) if roi_psnrs else None
            flicker_score = temporal_flicker_score(temporal_samples)
            for left, right in zip(temporal_samples, temporal_samples[1:]):
                if right[0] != left[0] + 1:
                    continue
                pair_score = temporal_flicker_score([left, right])
                if (
                    pair_score is not None
                    and pair_score > TEMPORAL_FLICKER_CEILING
                ):
                    review_spans.append(make_review_span(
                        "flicker",
                        start_frame + left[0],
                        start_frame + right[0] + 1,
                        fps=fps,
                        score=pair_score,
                        threshold=TEMPORAL_FLICKER_CEILING,
                        reason=(
                            "Adjacent cleaned frames exceeded "
                            "the flicker threshold"
                        ),
                    ))
            residual_mean_score = (
                float(np.mean(residual_scores)) if residual_scores else None
            )
            segment_duration = max(0.1, min(30.0, _frame_seconds_deferred(span, fps)))
            segment_start = _frame_seconds_deferred(start_frame, fps)
            vmaf = compute_vmaf(
                input_path,
                output_path,
                start_seconds=segment_start,
                duration_seconds=segment_duration,
            )
            roi_vmaf = None
            if roi_ready:
                roi_vmaf = compute_vmaf(
                    input_path,
                    output_path,
                    start_seconds=segment_start,
                    duration_seconds=segment_duration,
                    roi=roi,
                )
            tag_ssim = roi_mean_ssim if roi_mean_ssim is not None else mean_ssim
            tag = "Good" if tag_ssim >= 0.95 else "Review"
            sheet_path = None
            if self.config.quality_report_sheet and pairs:
                try:
                    sheet_path = self._write_quality_sheet(
                        output_path, pairs, mean_psnr, mean_ssim, tag,
                    )
                except Exception as exc:
                    logger.warning(f"Quality sheet write failed: {exc}", exc_info=True)
            extended = {}
            temporal_consistency = None
            if roi_ready and pairs:
                x1, y1, x2, y2 = roi
                x1 = max(0, min(pairs[0][1].shape[1] - 1, x1))
                x2 = max(x1 + 1, min(pairs[0][1].shape[1], x2))
                y1 = max(0, min(pairs[0][1].shape[0] - 1, y1))
                y2 = max(y1 + 1, min(pairs[0][1].shape[0], y2))
                roi_pairs = [
                    (a[y1:y2, x1:x2], b[y1:y2, x1:x2])
                    for (_, a, b, _, _) in pairs
                    if a[y1:y2, x1:x2].size > 0
                ]
                extended = compute_extended_metrics(roi_pairs)
                cleaned_roi_frames = [b for (_, b) in roi_pairs]
                temporal_consistency = temporal_consistency_score(
                    cleaned_roi_frames)
            elif pairs:
                extended = compute_extended_metrics(
                    [(a, b) for (_, a, b, _, _) in pairs])
                temporal_consistency = temporal_consistency_score(
                    [b for (_, _, b, _, _) in pairs])
            metrics = {
                'psnr': mean_psnr,
                'ssim': mean_ssim,
                'roi_psnr': roi_mean_psnr,
                'roi_ssim': roi_mean_ssim,
                'vmaf': vmaf,
                'roi_vmaf': roi_vmaf,
                'roi_bbox': list(roi) if roi else None,
                'temporal_flicker_score': flicker_score,
                'temporal_consistency': temporal_consistency,
                'residual_text_score': residual_mean_score,
                'seam_score': (
                    float(np.mean(getattr(self, '_seam_scores', None) or []))
                    if getattr(self, '_seam_scores', None) else None
                ),
                'lpips': extended.get('lpips'),
                'dists': extended.get('dists'),
                'samples': len(psnrs),
                'tag': tag,
                'sheet': sheet_path,
            }
            metrics["mask_review_spans"] = merge_review_spans(review_spans)
            if metrics["mask_review_spans"]:
                metrics["tag"] = "Review"
            metrics["quality_gate"] = evaluate_quality_gate(metrics)
            return metrics
        finally:
            cap_in.release()
            cap_out.release()

    def _write_quality_sheet(self,
                              output_path: str,
                              pairs: List[Tuple[int, np.ndarray, np.ndarray, float, float]],
                              mean_psnr: float,
                              mean_ssim: float,
                              tag: str,
                              max_row_h: int = 240) -> str:
        """Render the per-sample original | cleaned comparison sheet."""
        sheet_path = str(Path(output_path).with_suffix("")) + ".qualitysheet.png"
        gap = 6
        rows = []
        for idx, a, b, p, s in pairs:
            h = a.shape[0]
            scale = min(1.0, max_row_h / max(1, h))
            new_h = int(round(h * scale))
            new_w = int(round(a.shape[1] * scale))
            ar = cv2.resize(a, (new_w, new_h), interpolation=cv2.INTER_AREA)
            br = cv2.resize(b, (new_w, new_h), interpolation=cv2.INTER_AREA)
            sep = np.full((new_h, gap, 3), 32, dtype=np.uint8)
            row = np.concatenate([ar, sep, br], axis=1)
            caption_h = 26
            caption = np.full((caption_h, row.shape[1], 3), 16, dtype=np.uint8)
            text = f"Frame {idx}  PSNR={p:.2f} dB  SSIM={s:.4f}"
            cv2.putText(caption, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (220, 220, 220), 1, cv2.LINE_AA)
            rows.append(np.concatenate([row, caption], axis=0))
        body = []
        for i, r in enumerate(rows):
            if i:
                body.append(np.full((gap, r.shape[1], 3), 32, dtype=np.uint8))
            body.append(r)
        body_img = np.concatenate(body, axis=0)
        header_h = 56
        header = np.full((header_h, body_img.shape[1], 3), 10, dtype=np.uint8)
        title = f"VSR quality report  -  mean PSNR={mean_psnr:.2f} dB  mean SSIM={mean_ssim:.4f}  [{tag}]"
        cv2.putText(header, title, (10, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (245, 245, 245), 1, cv2.LINE_AA)
        sep = np.full((gap, body_img.shape[1], 3), 48, dtype=np.uint8)
        sheet = np.concatenate([header, sep, body_img], axis=0)
        cv2.imwrite(sheet_path, sheet, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        logger.info(f"Quality sheet written: {sheet_path}")
        return sheet_path
