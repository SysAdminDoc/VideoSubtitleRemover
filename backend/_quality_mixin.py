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

from backend.io import _open_bgr48_capture, _open_capture
from backend.mask_corrections import make_review_span, merge_review_spans
from backend.quality import (
    _ssim,
    compute_vmaf,
    compute_extended_metrics,
    harmonic_mean,
    temporal_consistency_score,
    residual_text_score,
    temporal_flicker_score,
    mask_boundary_seam_score,
    mask_local_temporal_pair,
    outside_mask_color_drift,
    worst_sample,
)
from backend.quality_gate import (
    MASK_LOCAL_TEMPORAL_CEILING,
    OUTSIDE_MASK_CIELAB_CEILING,
    OUTSIDE_MASK_HDR_LINEAR_CEILING,
    RESIDUAL_TEXT_SCORE_CEILING,
    TEMPORAL_FLICKER_CEILING,
    evaluate_quality_gate,
)
from backend.safe_image import safe_imread

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

    def _persist_quality_mask(self, frame_index: int,
                              mask: Optional[np.ndarray]) -> None:
        """Persist sparse masks so temporal checks can read final output."""
        directory = getattr(self, "_quality_frame_evidence_dir", None)
        if directory is None or mask is None:
            return
        values = np.asarray(mask)
        if values.ndim == 3:
            values = cv2.cvtColor(values, cv2.COLOR_BGR2GRAY)
        if values.ndim != 2 or not np.any(values > 0):
            return
        normalized = np.where(values > 0, 255, 0).astype(np.uint8)
        path = Path(directory) / f"{int(frame_index):08d}.png"
        try:
            if not cv2.imwrite(str(path), normalized):
                raise OSError(f"could not write {path}")
        except Exception:
            self._quality_frame_evidence_write_error = True
            logger.warning(
                "Final-encode quality mask write failed for frame %d",
                int(frame_index),
                exc_info=True,
            )

    def _quality_mask_from_disk(self, frame_index: int,
                                shape: Tuple[int, int]) -> np.ndarray:
        directory = getattr(self, "_quality_frame_evidence_dir", None)
        if directory is None:
            return np.zeros(shape, dtype=np.uint8)
        path = Path(directory) / f"{int(frame_index):08d}.png"
        if not path.is_file():
            return np.zeros(shape, dtype=np.uint8)
        mask = safe_imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise OSError(f"could not read {path}")
        if mask.shape != shape:
            mask = cv2.resize(
                mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST,
            )
        return mask

    def _open_final_quality_capture(self, path: str, *, source: bool):
        """Open a final-quality capture in the source's native HDR depth."""
        meta = getattr(self, "_color_metadata", None)
        if bool(getattr(meta, "is_hdr", False)) and not Path(path).is_dir():
            capture = _open_bgr48_capture(
                path, input_fps=float(getattr(self.config, "input_fps", 24.0)),
            )
            if capture is not None and capture.isOpened():
                return capture
            return None
        return _open_capture(
            path,
            getattr(self.config, "decode_hw_accel", "off") if source else "off",
            input_fps=float(getattr(self.config, "input_fps", 24.0)),
        )

    def _recompute_final_quality_evidence(
        self,
        input_path: str,
        output_path: str,
        start_frame: int,
        end_frame: int,
        fps: float,
    ) -> bool:
        """Score the final encoded frames, not the lossless intermediate."""
        if not getattr(self.config, "quality_report", False):
            return True
        if (
            getattr(self, "_quality_frame_evidence_dir", None) is None
            or getattr(self, "_quality_frame_evidence_write_error", False)
        ):
            return False
        cap_in = self._open_final_quality_capture(input_path, source=True)
        cap_out = self._open_final_quality_capture(output_path, source=False)
        if cap_in is None or cap_out is None:
            for capture in (cap_in, cap_out):
                if capture is not None:
                    capture.release()
            return False
        try:
            if _seek_capture_to_frame_deferred(cap_in, start_frame) is False:
                return False
            if _seek_capture_to_frame_deferred(cap_out, 0) is False:
                return False
            self._quality_temporal_previous = None
            self._quality_temporal_scores = []
            self._quality_temporal_scene_cuts_excluded = 0
            self._quality_temporal_worst_pair = None
            self._quality_color_drift_sum = 0.0
            self._quality_color_drift_count = 0
            self._quality_color_drift_metric = None
            self._quality_color_drift_worst_frame = None
            expected = max(0, int(end_frame) - int(start_frame))
            for frame_index in range(expected):
                ok_in, reference = cap_in.read()
                ok_out, output = cap_out.read()
                if not ok_in or not ok_out or reference is None or output is None:
                    return False
                mask = self._quality_mask_from_disk(
                    frame_index, reference.shape[:2],
                )
                self._accumulate_frame_quality(
                    reference,
                    output,
                    mask,
                    frame_index=frame_index,
                    timestamp=_frame_seconds_deferred(
                        int(start_frame) + frame_index, fps,
                    ),
                )
            return True
        except Exception:
            logger.warning(
                "Final-encode quality evidence collection failed",
                exc_info=True,
            )
            return False
        finally:
            cap_in.release()
            cap_out.release()

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

    @staticmethod
    def _quality_display_frame(frame: np.ndarray) -> np.ndarray:
        values = np.asarray(frame)
        if values.ndim == 2:
            values = cv2.cvtColor(values, cv2.COLOR_GRAY2BGR)
        if values.dtype == np.uint8:
            return values.copy()
        if np.issubdtype(values.dtype, np.integer):
            scale = float(np.iinfo(values.dtype).max)
            return np.clip(
                np.rint(values.astype(np.float32) * 255.0 / scale), 0, 255
            ).astype(np.uint8)
        return np.clip(np.rint(values.astype(np.float32) * 255.0), 0, 255).astype(
            np.uint8
        )

    @classmethod
    def _quality_overlay_frame(
        cls,
        frame: np.ndarray,
        mask: Optional[np.ndarray],
        label: str,
    ) -> np.ndarray:
        image = cls._quality_display_frame(frame)
        if mask is not None and mask.shape[:2] == image.shape[:2]:
            active = np.asarray(mask) > 0
            if np.any(active):
                tint = image.copy()
                tint[active] = (0, 48, 255)
                image = cv2.addWeighted(image, 0.72, tint, 0.28, 0.0)
                contours, _ = cv2.findContours(
                    active.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(image, contours, -1, (0, 96, 255), 1)
        cv2.putText(
            image, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
            0.62, (245, 245, 245), 1, cv2.LINE_AA,
        )
        return image

    def _write_temporal_quality_overlay(
        self,
        output_path: str,
        record: dict,
    ) -> Optional[str]:
        """Write the worst mask-local pair as a reviewable A/B overlay."""
        previous = record.get("previous_frame")
        current = record.get("current_frame")
        if previous is None or current is None:
            return None
        left = self._quality_overlay_frame(
            previous, record.get("previous_mask"),
            f"Frame {int(record['start_frame'])}",
        )
        right = self._quality_overlay_frame(
            current, record.get("current_mask"),
            f"Frame {int(record['end_frame'])}",
        )
        if left.shape[0] != right.shape[0]:
            height = max(left.shape[0], right.shape[0])
            left = cv2.resize(left, (left.shape[1], height), interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, (right.shape[1], height), interpolation=cv2.INTER_AREA)
        gap = np.full((left.shape[0], 8, 3), 32, dtype=np.uint8)
        body = np.concatenate([left, gap, right], axis=1)
        header = np.full((44, body.shape[1], 3), 12, dtype=np.uint8)
        cv2.putText(
            header,
            (
                f"Worst mask-local pair  t={float(record['timestamp']):.3f}s  "
                f"score={float(record['score']):.4f}"
            ),
            (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
            (245, 245, 245), 1, cv2.LINE_AA,
        )
        overlay = np.concatenate([header, body], axis=0)
        overlay_path = str(Path(output_path).with_suffix("")) + ".temporalworst.png"
        if not cv2.imwrite(overlay_path, overlay, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            return None
        return overlay_path

    def _accumulate_frame_quality(
        self,
        reference_frame: np.ndarray,
        output_frame: np.ndarray,
        mask: Optional[np.ndarray],
        *,
        frame_index: int,
        timestamp: float,
    ) -> None:
        """Accumulate streaming temporal and outside-mask color evidence."""
        previous = getattr(self, "_quality_temporal_previous", None)
        if (
            isinstance(previous, dict)
            and int(previous.get("frame_index", -2)) + 1 == int(frame_index)
        ):
            try:
                pair = mask_local_temporal_pair(
                    previous["frame"],
                    output_frame,
                    previous.get("mask"),
                    mask,
                    reference_previous=previous.get("reference"),
                    reference_current=reference_frame,
                    scene_cut_threshold=float(
                        getattr(self.config, "tbe_scene_cut_threshold", 0.35)
                    ),
                )
            except Exception:
                pair = None
                if not getattr(self, "_quality_temporal_failure_logged", False):
                    logger.warning(
                        "Mask-local temporal quality sampling failed",
                        exc_info=True,
                    )
                    self._quality_temporal_failure_logged = True
            if pair is not None:
                if pair.get("scene_cut"):
                    self._quality_temporal_scene_cuts_excluded = (
                        int(getattr(self, "_quality_temporal_scene_cuts_excluded", 0))
                        + 1
                    )
                else:
                    score = float(pair["score"])
                    scores = getattr(self, "_quality_temporal_scores", None)
                    if scores is None:
                        scores = []
                        self._quality_temporal_scores = scores
                    scores.append(score)
                    worst = getattr(self, "_quality_temporal_worst_pair", None)
                    if worst is None or score > float(worst["score"]):
                        self._quality_temporal_worst_pair = {
                            "start_frame": int(previous["frame_index"]),
                            "end_frame": int(frame_index),
                            "timestamp": float(timestamp),
                            "score": score,
                            "inside_error": float(pair["inside_error"]),
                            "outside_error": float(pair["outside_error"]),
                            "reference_inside_error": (
                                None
                                if pair.get("reference_inside_error") is None
                                else float(pair["reference_inside_error"])
                            ),
                            "motion_inlier_ratio": float(
                                pair["motion_inlier_ratio"]
                            ),
                            "pixels": int(pair["pixels"]),
                            "previous_frame": previous["frame"].copy(),
                            "current_frame": np.asarray(output_frame).copy(),
                            "previous_mask": (
                                None if previous.get("mask") is None
                                else np.asarray(previous["mask"]).copy()
                            ),
                            "current_mask": (
                                None if mask is None
                                else np.asarray(mask).copy()
                            ),
                        }
        self._quality_temporal_previous = {
            "frame_index": int(frame_index),
            "frame": np.asarray(output_frame).copy(),
            "reference": np.asarray(reference_frame).copy(),
            "mask": None if mask is None else np.asarray(mask).copy(),
        }

        meta = getattr(self, "_color_metadata", None)
        hdr_transfer = ""
        if bool(getattr(meta, "is_hdr", False)):
            hdr_transfer = str(getattr(meta, "color_transfer", "") or "")
        try:
            drift = outside_mask_color_drift(
                reference_frame,
                output_frame,
                mask,
                hdr_transfer=hdr_transfer,
            )
        except Exception:
            drift = None
            if not getattr(self, "_quality_color_failure_logged", False):
                logger.warning(
                    "Outside-mask color quality sampling failed",
                    exc_info=True,
                )
                self._quality_color_failure_logged = True
        if drift is None:
            return
        self._quality_color_drift_sum = (
            float(getattr(self, "_quality_color_drift_sum", 0.0))
            + float(drift["score"])
        )
        self._quality_color_drift_count = (
            int(getattr(self, "_quality_color_drift_count", 0)) + 1
        )
        self._quality_color_drift_metric = str(drift["metric"])
        worst_color = getattr(self, "_quality_color_drift_worst_frame", None)
        if worst_color is None or float(drift["score"]) > float(worst_color["score"]):
            self._quality_color_drift_worst_frame = {
                "frame": int(frame_index),
                "timestamp": float(timestamp),
                "score": float(drift["score"]),
                "p95": float(drift["p95"]),
                "metric": str(drift["metric"]),
            }

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
            sample_frames: List[int] = []
            roi_psnrs: List[float] = []
            roi_ssims: List[float] = []
            roi_sample_frames: List[int] = []
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
                # The frame the sample actually measures in the OUTPUT file.
                # Everything downstream (the report, the gate text, the quality
                # sheet, the A/B compare) indexes the output, so this -- not
                # start_frame + idx -- is the number to carry around.
                out_idx = min(out_total - 1, idx)
                _seek_capture_to_frame_deferred(cap_out, out_idx)
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
                sample_frames.append(out_idx)
                if a_roi is not None and b_roi is not None:
                    if a_roi.size and a_roi.shape == b_roi.shape:
                        try:
                            # Compute both before appending: a raise
                            # between the two appends would offset the lists
                            # permanently and misattribute every later sample.
                            roi_psnr_sample = float(cv2.PSNR(a_roi, b_roi))
                            roi_ssim_sample = _ssim(a_roi, b_roi)
                            roi_psnrs.append(roi_psnr_sample)
                            roi_ssims.append(roi_ssim_sample)
                            roi_sample_frames.append(out_idx)
                        except Exception:
                            logger.warning(
                                "Quality ROI metric calculation failed",
                                exc_info=True,
                            )
                if self.config.quality_report_sheet:
                    pairs.append((out_idx, a, b, p, s))
            if not psnrs:
                return None
            mean_ssim = float(np.mean(ssims))
            mean_psnr = float(np.mean(psnrs))
            roi_mean_ssim = float(np.mean(roi_ssims)) if roi_ssims else None
            roi_mean_psnr = float(np.mean(roi_psnrs)) if roi_psnrs else None
            # RM-281: the arithmetic mean hides a few ruined frames on an
            # otherwise sharp clip. Pool the harmonic mean beside it and
            # name the single worst frame so the user can go look at it.
            #
            # This is deliberately the WHOLE-FRAME worst, not the ROI worst.
            # Inside the subtitle box the input has text and the output does
            # not, so a low ROI SSIM is the job working; gating on it would
            # send every successful removal to review. A whole-frame SSIM is
            # dominated by the pixels that should not have changed, so it
            # only drops when something actually went wrong.
            worst_frame = worst_sample(sample_frames, psnrs, ssims)
            roi_worst_frame = worst_sample(
                roi_sample_frames, roi_psnrs, roi_ssims)
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
            temporal_scores = list(
                getattr(self, "_quality_temporal_scores", None) or []
            )
            temporal_local_mean_score = (
                float(np.mean(temporal_scores))
                if temporal_scores else None
            )
            temporal_local_worst_score = (
                float(max(temporal_scores)) if temporal_scores else None
            )
            # The public score is the worst valid pair. A mean remains in the
            # report for trend analysis, but must not dilute one severe local
            # repair failure.
            temporal_local_score = temporal_local_worst_score
            temporal_worst_record = getattr(
                self, "_quality_temporal_worst_pair", None
            )
            temporal_overlay = None
            temporal_worst_pair = None
            if isinstance(temporal_worst_record, dict):
                try:
                    temporal_overlay = self._write_temporal_quality_overlay(
                        output_path, temporal_worst_record,
                    )
                except Exception:
                    logger.warning(
                        "Temporal quality overlay write failed",
                        exc_info=True,
                    )
                temporal_worst_pair = {
                    key: temporal_worst_record[key]
                    for key in (
                        "start_frame", "end_frame", "timestamp", "score",
                        "inside_error", "outside_error",
                        "reference_inside_error",
                        "motion_inlier_ratio", "pixels",
                    )
                    if key in temporal_worst_record
                }
                if temporal_overlay:
                    temporal_worst_pair["overlay"] = temporal_overlay
            color_drift_count = int(
                getattr(self, "_quality_color_drift_count", 0) or 0
            )
            color_drift = (
                float(getattr(self, "_quality_color_drift_sum", 0.0))
                / color_drift_count
                if color_drift_count else None
            )
            color_drift_metric = getattr(
                self, "_quality_color_drift_metric", None
            )
            color_drift_worst = getattr(
                self, "_quality_color_drift_worst_frame", None
            )
            color_drift_threshold = (
                OUTSIDE_MASK_HDR_LINEAR_CEILING
                if color_drift_metric == "linear_rgb_mae"
                else OUTSIDE_MASK_CIELAB_CEILING
            )
            metrics = {
                'psnr': mean_psnr,
                'ssim': mean_ssim,
                'psnr_harmonic_mean': harmonic_mean(psnrs),
                'ssim_harmonic_mean': harmonic_mean(ssims),
                'roi_psnr': roi_mean_psnr,
                'roi_ssim': roi_mean_ssim,
                'roi_psnr_harmonic_mean': harmonic_mean(roi_psnrs),
                'roi_ssim_harmonic_mean': harmonic_mean(roi_ssims),
                'worst_frame': worst_frame,
                'roi_worst_frame': roi_worst_frame,
                'vmaf': vmaf,
                'roi_vmaf': roi_vmaf,
                'roi_bbox': list(roi) if roi else None,
                'temporal_flicker_score': flicker_score,
                'temporal_consistency': temporal_consistency,
                'mask_local_temporal_score': temporal_local_score,
                'mask_local_temporal_mean_score': temporal_local_mean_score,
                'mask_local_temporal_worst_score': temporal_local_worst_score,
                'mask_local_temporal_threshold': MASK_LOCAL_TEMPORAL_CEILING,
                'mask_local_temporal_pairs': len(temporal_scores),
                'mask_local_temporal_scene_cuts_excluded': int(
                    getattr(self, "_quality_temporal_scene_cuts_excluded", 0)
                    or 0
                ),
                'mask_local_temporal_worst_pair': temporal_worst_pair,
                'outside_mask_color_drift': color_drift,
                'outside_mask_color_drift_metric': color_drift_metric,
                'outside_mask_color_drift_frames': color_drift_count,
                'outside_mask_color_drift_worst_frame': color_drift_worst,
                'outside_mask_color_drift_threshold': color_drift_threshold,
                'quality_final_encode_verified': getattr(
                    self, "_quality_final_encode_verified", None
                ),
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
            if (
                temporal_local_score is not None
                and temporal_local_score > MASK_LOCAL_TEMPORAL_CEILING
                and isinstance(temporal_worst_pair, dict)
            ):
                review_spans.append(make_review_span(
                    "flicker",
                    int(temporal_worst_pair["start_frame"]),
                    int(temporal_worst_pair["end_frame"]) + 1,
                    fps=fps,
                    score=temporal_local_score,
                    threshold=MASK_LOCAL_TEMPORAL_CEILING,
                    reason=(
                        "Motion-compensated mask-local temporal score exceeded "
                        "the review threshold"
                    ),
                ))
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
