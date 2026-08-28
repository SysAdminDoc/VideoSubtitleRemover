"""RM-349: the per-batch stages of the frame loop.

`process_video` decodes a batch of frames, refines and finishes their masks,
inpaints them, writes them, and checkpoints. Those five stages ran to some
740 lines inside the same 4,500-line file as everything else the processor
does, which is why the loop could not be read without reading the rest.

The methods are mixed into `SubtitleRemover`, so they keep full `self`
access and behave exactly as they did in place. They take the loop's context
and mutable state as explicit arguments rather than reading them off the
instance, which is what makes each stage testable on its own.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from backend._frame_loop_types import (
    _FrameBatch,
    _FrameLoopContext,
    _FrameLoopState,
    _frame_seconds,
)
from backend.detection_geometry import DetectionGeometry
from backend.inpainters import (
    _detect_scene_cuts,
    _expand_mask_by_color,
    extend_masks_across_fades,
    stabilize_masks_rolling_union,
)
from backend.mask_corrections import frame_is_in_ranges, make_review_span
from backend.matte_interchange import compose_imported_matte
from backend.resume_checkpoint import (
    ProcessingPaused,
    pause_frame_dir,
    write_pause_checkpoint,
)
from backend.tracking import (
    _group_horizontal_geometry,
    _group_horizontal_line,
    _phash,
    _phash_distance,
)

logger = logging.getLogger(__name__)


class _FrameLoopMixin:
    """Per-batch stages of `process_video`'s frame loop."""

    def _decode_and_build_batch(self, ctx: _FrameLoopContext,
                                state: _FrameLoopState) -> _FrameBatch:
        """Decode frames and build masks until one processing batch is full."""
        batch = _FrameBatch()
        hdr_transfer = getattr(
            getattr(self, "_color_metadata", None), "color_transfer", "")
        for _ in range(ctx.batch_size):
            if ctx.start_frame + state.frame_idx >= ctx.end_frame:
                break
            with self._time_stage("decode"):
                ret, raw_frame = ctx.reader.read()
                if not ret:
                    break
                source_frame = (
                    raw_frame
                    if ctx.high_bit_depth_surface
                    and self._is_high_bit_frame(raw_frame)
                    else None
                )
                frame = self._processing_frame(
                    raw_frame,
                    transfer=hdr_transfer if source_frame is not None else None,
                )

            self.last_detection_stats["frames_total"] += 1
            absolute_idx = ctx.start_frame + state.frame_idx
            frame_seconds = _frame_seconds(
                absolute_idx, ctx.fps, ctx.frame_timing)
            if ctx.selective_cap is not None:
                prior_ok, prior_raw = ctx.selective_cap.read()
                if not prior_ok or prior_raw is None:
                    raise ValueError(
                        "Previous cleaned output ended during selective rerun"
                    )
                if not frame_is_in_ranges(
                    absolute_idx, ctx.selective_ranges
                ):
                    self._record_detection_skip("selective_rerun")
                    prior_source_frame = (
                        prior_raw
                        if ctx.high_bit_depth_surface
                        and self._is_high_bit_frame(prior_raw)
                        else None
                    )
                    prior_frame = self._processing_frame(
                        prior_raw,
                        transfer=(
                            hdr_transfer if prior_source_frame is not None
                            else None
                        ),
                    )
                    batch.add(
                        prior_frame,
                        np.zeros(prior_frame.shape[:2], dtype=np.uint8),
                        prior_source_frame,
                        passthrough=True,
                    )
                    state.frame_idx += 1
                    continue
                if any(
                    absolute_idx == range_start
                    for range_start, _range_end in ctx.selective_ranges
                ):
                    state.last_mask = None
                    state.last_hash = None
                    if state.tracker is not None:
                        state.tracker.reset()
                    if (
                        state.srt_tracker is not None
                        and state.srt_tracker is not state.tracker
                    ):
                        state.srt_tracker.reset()
            # RM-153: a frozen matte is this job's authoritative mask. It
            # was reviewed and approved frame by frame against exactly
            # these frames, so detection, tracking, and every heuristic
            # that would alter it are skipped outright -- re-deriving a
            # mask a human already signed off on is the cost the freeze
            # exists to avoid, and refining it would change the pixels
            # they approved.
            if ctx.frozen_matte:
                self._record_detection_skip("frozen_matte")
                with self._time_stage("mask"):
                    mask = ctx.matte_reader.read(state.frame_idx)
                state.last_mask = mask
                batch.add(frame, mask, source_frame, passthrough=False)
                state.frame_idx += 1
                continue

            fixed_shapes = (
                self._fixed_region_shapes(frame_seconds)
                if ctx.timed_region_spans else ctx.static_fixed_shapes
            )
            if (
                self.config.sttn_skip_detection
                and not fixed_shapes
                and not ctx.timed_region_spans
            ):
                raise ValueError(
                    "Manual region mode needs a fixed, timed, or moving region"
                )
            fixed_boxes = (
                [tuple(shape["rect"]) for shape in fixed_shapes or []
                 if "rect" in shape]
                or None
            )

            if self.config.sttn_skip_detection and (
                    fixed_shapes or ctx.timed_region_spans):
                self._record_detection_skip("manual_region")
                if fixed_shapes:
                    has_polygon = any(
                        "polygon" in shape for shape in fixed_shapes)
                    dynamic_shape = ctx.timed_region_spans or has_polygon
                    mask_key = tuple(tuple(r) for r in (fixed_boxes or []))
                    fixed_mask = (
                        None if dynamic_shape
                        else state.fixed_mask_cache.get(mask_key)
                    )
                    if fixed_mask is None:
                        with self._time_stage("mask"):
                            fixed_mask = self._create_mask(
                                frame.shape, fixed_boxes or [])
                            fixed_mask = self._apply_polygon_region_shapes(
                                fixed_mask, fixed_shapes)
                        if not dynamic_shape:
                            state.fixed_mask_cache[mask_key] = fixed_mask
                else:
                    with self._time_stage("mask"):
                        fixed_mask = np.zeros(
                            frame.shape[:2], dtype=np.uint8)
                corrected = self._apply_manual_mask_corrections(
                    fixed_mask.copy(), frame_seconds, absolute_idx)
                batch.add(
                    frame, corrected, source_frame, passthrough=False)
                state.frame_idx += 1
                continue

            reuse_by_phash = False
            cur_hash = None
            if (not ctx.timed_region_spans
                    and not ctx.timed_mask_corrections
                    and self.config.phash_skip_enable
                    and state.last_mask is not None
                    and state.last_hash is not None):
                cur_hash = _phash(frame)
                if _phash_distance(
                    cur_hash, state.last_hash
                ) <= self.config.phash_skip_distance:
                    reuse_by_phash = True

            reuse_by_keyframe = False
            if (not ctx.timed_region_spans
                    and not ctx.timed_mask_corrections
                    and ctx.keyframe_set
                    and state.last_mask is not None):
                if absolute_idx not in ctx.keyframe_set:
                    reuse_by_keyframe = True

            if reuse_by_phash or reuse_by_keyframe:
                self._record_detection_skip(
                    "phash" if reuse_by_phash else "keyframe")
                batch.add(
                    frame, state.last_mask, source_frame, passthrough=False)
                state.frame_idx += 1
                continue
            if (not ctx.timed_region_spans
                    and not ctx.timed_mask_corrections
                    and ctx.frame_skip > 0
                    and state.last_mask is not None
                    and state.frame_idx % (ctx.frame_skip + 1) != 0):
                self._record_detection_skip("frame_skip")
                batch.add(
                    frame, state.last_mask, source_frame, passthrough=False)
                state.frame_idx += 1
                continue

            with self._time_stage("ocr"):
                if self.config.detection_denoise:
                    try:
                        from backend.preprocess import fastdvdnet_denoise_frame
                        det_frame = fastdvdnet_denoise_frame(frame)
                    except Exception as exc:
                        logger.warning(
                            f"Detection denoise fell back: {exc}",
                            exc_info=True,
                        )
                        det_frame = frame
                else:
                    det_frame = frame
                det_confs = None
                detection_geometry: List[DetectionGeometry] = []
                geometry_detector = callable(getattr(
                    self.detector, "detect_with_geometry", None))
                collect_confidence = bool(
                    self.config.confidence_weighted_dilation
                    or self.config.quality_report
                )
                if self.config.language_mask_filter:
                    from backend.detection import text_matches_detection_language
                    if geometry_detector:
                        detection_geometry = self._detect_geometry(
                            det_frame, self.config.detection_threshold)
                        detection_geometry = [
                            detection for detection in detection_geometry
                            if text_matches_detection_language(
                                detection.text,
                                self.config.detection_lang,
                            )
                        ]
                        detected_boxes = [
                            detection.bbox
                            for detection in detection_geometry
                        ]
                        det_confs = [
                            detection.confidence
                            for detection in detection_geometry
                        ]
                    else:
                        text_results = self.detector.detect_with_text(
                            det_frame, self.config.detection_threshold)
                        matched = [
                            result for result in text_results
                            if text_matches_detection_language(
                                result[5], self.config.detection_lang)
                        ]
                        detection_geometry = [
                            detection
                            for result in matched
                            for detection in [DetectionGeometry.from_box(
                                result[:4],
                                frame.shape,
                                confidence=result[4],
                                text=result[5],
                            )]
                            if detection is not None
                        ]
                        detected_boxes = [
                            detection.bbox
                            for detection in detection_geometry
                        ]
                        det_confs = [
                            detection.confidence
                            for detection in detection_geometry
                        ]
                    if self.config.quality_report and det_confs:
                        review_floor = min(
                            0.9,
                            max(
                                0.6,
                                self.config.detection_threshold + 0.15,
                            ),
                        )
                        low_confidence = min(det_confs)
                        if low_confidence < review_floor:
                            self._mask_review_signals.append(
                                make_review_span(
                                    "low-confidence",
                                    absolute_idx,
                                    absolute_idx + 1,
                                    fps=ctx.fps,
                                    score=low_confidence,
                                    threshold=review_floor,
                                    reason=(
                                        "OCR confidence was below "
                                        "the review floor"
                                    ),
                                )
                            )
                    if not self.config.confidence_weighted_dilation:
                        det_confs = None
                elif collect_confidence:
                    if geometry_detector:
                        detection_geometry = self._detect_geometry(
                            det_frame, self.config.detection_threshold)
                        detected_boxes = [
                            detection.bbox
                            for detection in detection_geometry
                        ]
                        det_confs = [
                            detection.confidence
                            for detection in detection_geometry
                        ]
                    else:
                        det_results = self.detector.detect_with_confidence(
                            det_frame, self.config.detection_threshold)
                        detection_geometry = [
                            detection
                            for x1, y1, x2, y2, confidence in det_results
                            for detection in [DetectionGeometry.from_box(
                                (x1, y1, x2, y2),
                                frame.shape,
                                confidence=confidence,
                            )]
                            if detection is not None
                        ]
                        detected_boxes = [
                            detection.bbox
                            for detection in detection_geometry
                        ]
                        det_confs = [
                            detection.confidence
                            for detection in detection_geometry
                        ]
                    if self.config.quality_report and det_confs:
                        review_floor = min(
                            0.9,
                            max(
                                0.6,
                                self.config.detection_threshold + 0.15,
                            ),
                        )
                        low_confidence = min(det_confs)
                        if low_confidence < review_floor:
                            self._mask_review_signals.append(
                                make_review_span(
                                    "low-confidence",
                                    absolute_idx,
                                    absolute_idx + 1,
                                    fps=ctx.fps,
                                    score=low_confidence,
                                    threshold=review_floor,
                                    reason=(
                                        "OCR confidence was below "
                                        "the review floor"
                                    ),
                                )
                            )
                    if not self.config.confidence_weighted_dilation:
                        det_confs = None
                else:
                    if geometry_detector:
                        detection_geometry = self._detect_geometry(
                            det_frame, self.config.detection_threshold)
                        detected_boxes = [
                            detection.bbox
                            for detection in detection_geometry
                        ]
                    elif (
                        state.srt_tracker is not None
                        and callable(getattr(
                            self.detector, "detect_with_text", None))
                    ):
                        text_results = self.detector.detect_with_text(
                            det_frame, self.config.detection_threshold)
                        detection_geometry = [
                            detection
                            for result in text_results
                            for detection in [DetectionGeometry.from_box(
                                result[:4],
                                frame.shape,
                                confidence=result[4],
                                text=result[5],
                            )]
                            if detection is not None
                        ]
                        detected_boxes = [
                            detection.bbox
                            for detection in detection_geometry
                        ]
                    else:
                        detected_boxes = self.detector.detect(
                            det_frame, self.config.detection_threshold)
                self._record_ocr_detection(detected_boxes)
                if self.config.karaoke_grouping and detected_boxes:
                    if detection_geometry:
                        detection_geometry = _group_horizontal_geometry(
                            detection_geometry,
                            x_gap_px=self.config.karaoke_x_gap_px,
                            y_overlap_ratio=self.config.karaoke_y_overlap,
                        )
                        detected_boxes = [
                            detection.bbox
                            for detection in detection_geometry
                        ]
                    else:
                        detected_boxes = _group_horizontal_line(
                            detected_boxes,
                            x_gap_px=self.config.karaoke_x_gap_px,
                            y_overlap_ratio=self.config.karaoke_y_overlap,
                        )
                    det_confs = None
                if state.tracker is not None:
                    update_geometry = getattr(
                        state.tracker, "update_with_geometry", None)
                    if callable(update_geometry):
                        tracking_geometry = (
                            detection_geometry
                            if detection_geometry
                            else self._legacy_geometry(detected_boxes)
                        )
                        smoothed_geometry = list(
                            update_geometry(list(tracking_geometry)))
                    else:
                        smoothed_geometry = self._legacy_geometry(
                            state.tracker.update(list(detected_boxes)))
                    smoothed = [
                        detection.bbox
                        for detection in smoothed_geometry
                    ]
                    det_confs = None
                else:
                    smoothed = list(detected_boxes)
                    smoothed_geometry = (
                        list(detection_geometry)
                        if detection_geometry else self._legacy_geometry(smoothed)
                    )
                srt_geometry = list(smoothed_geometry)
                if (
                    state.srt_tracker is not None
                    and state.srt_tracker is not state.tracker
                ):
                    srt_geometry = list(
                        state.srt_tracker.update_with_geometry(
                            list(
                                detection_geometry
                                if detection_geometry
                                else self._legacy_geometry(detected_boxes)
                            )
                        )
                    )
                if (state.tracker is not None
                        and (not self.config.remove_chyrons
                             or not self.config.remove_subtitles)):
                    cats = state.tracker.categorize(
                        self.config.chyron_min_hits)
                    smoothed_geometry = [
                        detection for detection, category in zip(
                            smoothed_geometry, cats, strict=True)
                        if (
                            category == "chyron"
                            and self.config.remove_chyrons
                        ) or (
                            category == "subtitle"
                            and self.config.remove_subtitles
                        )
                    ]
                    smoothed = [
                        detection.bbox for detection in smoothed_geometry
                    ]
                    det_confs = None
                if fixed_boxes:
                    boxes = list(fixed_boxes) + smoothed
                    detection_geometry = (
                        self._legacy_geometry(fixed_boxes)
                        + smoothed_geometry
                    )
                    det_confs = None
                else:
                    boxes = smoothed
                    detection_geometry = smoothed_geometry
                if (
                    self.config.export_srt
                    or (
                        self.config.translation_enabled
                        and not self.config.translation_srt
                        and not self.config.translation_source_srt
                    )
                ):
                    self._collect_srt_entry(
                        frame, state.frame_idx, srt_geometry)

            if (not boxes and ctx.whisper_spans
                    and self.config.whisper_fallback):
                absolute = ctx.start_frame + state.frame_idx
                for span_start, span_end in ctx.whisper_spans:
                    if span_start <= absolute < span_end:
                        height, width = frame.shape[:2]
                        band_top = int(height * 0.80)
                        boxes = [(
                            int(width * 0.05),
                            band_top,
                            int(width * 0.95),
                            height - 4,
                        )]
                        detection_geometry = self._legacy_geometry(boxes)
                        break

            with self._time_stage("mask"):
                mask = self._create_mask(
                    frame.shape,
                    boxes,
                    frame=frame,
                    confidences=det_confs,
                    detections=detection_geometry,
                )
                mask = self._apply_polygon_region_shapes(mask, fixed_shapes)
                if self.config.colour_tune_enable and boxes:
                    mask = _expand_mask_by_color(
                        frame,
                        mask,
                        boxes,
                        tolerance=self.config.colour_tune_tolerance,
                        padding=4,
                    )
                mask = self._apply_manual_mask_corrections(
                    mask, frame_seconds, absolute_idx)
            state.last_mask = mask
            if self.config.phash_skip_enable:
                state.last_hash = (
                    cur_hash if cur_hash is not None else _phash(frame)
                )
            batch.add(frame, mask, source_frame, passthrough=False)
            state.frame_idx += 1
        return batch

    @staticmethod
    def _mark_active_segments(batch: _FrameBatch) -> None:
        """Record the contiguous non-passthrough runs to inpaint."""
        segment_start = None
        for index, passthrough in enumerate(
                batch.passthrough_flags + [True]):
            if not passthrough and segment_start is None:
                segment_start = index
            elif passthrough and segment_start is not None:
                batch.active_segments.append((segment_start, index))
                segment_start = None

    def _refine_batch_masks(self, ctx: _FrameLoopContext,
                            state: _FrameLoopState,
                            batch: _FrameBatch) -> None:
        """Apply temporal/refiner/imported-matte passes to one batch."""
        with self._time_stage("mask"):
            self._mark_active_segments(batch)
            # RM-153: the masks in a frozen batch came straight from the
            # approved matte. Propagation, refinement, and rolling-union
            # stabilization all exist to improve a *derived* mask, so
            # running them here would silently edit pixels a human signed
            # off on. Segments, quality accounting, and progress still
            # run -- only the mask-altering passes are skipped.
            if ctx.frozen_matte:
                if self.config.quality_report:
                    for index, mask in enumerate(batch.masks):
                        if not batch.passthrough_flags[index]:
                            self._accumulate_quality_bbox(mask)
                self._finish_batch_masks(ctx, state, batch)
                return
            for segment_start, segment_end in batch.active_segments:
                segment_frames = batch.frames[segment_start:segment_end]
                segment_masks = batch.masks[segment_start:segment_end]
                segment_masks = self._propagate_masks_with_cotracker(
                    segment_frames, segment_masks)
                segment_masks = self._refine_masks_with_matanyone(
                    segment_frames, segment_masks)
                if (self.config.temporal_mask_union
                        and not ctx.timed_region_spans
                        and not ctx.timed_mask_corrections
                        and not self.config.sttn_skip_detection
                        and len(segment_masks) > 1):
                    scene_starts = _detect_scene_cuts(segment_frames)
                    segment_masks = stabilize_masks_rolling_union(
                        segment_masks,
                        scene_starts,
                        self.config.temporal_mask_window,
                    )
                batch.masks[segment_start:segment_end] = segment_masks
            # RM-292: run the fade hold over the whole batch rather than per
            # active segment, so a fade that spans a passthrough gap is still
            # covered. A frozen matte never reaches here.
            fade_in = int(getattr(self.config, "mask_fade_in_frames", 0) or 0)
            fade_out = int(getattr(self.config, "mask_fade_out_frames", 0) or 0)
            if (fade_in > 0 or fade_out > 0) and batch.masks:
                batch.masks, state.fade_carry = extend_masks_across_fades(
                    batch.masks, fade_in, fade_out, state.fade_carry)
            if ctx.matte_reader is not None:
                batch_start = state.written_idx
                for offset, passthrough in enumerate(
                        batch.passthrough_flags):
                    if passthrough:
                        continue
                    imported_matte = ctx.matte_reader.read(
                        batch_start + offset)
                    batch.masks[offset] = compose_imported_matte(
                        batch.masks[offset],
                        imported_matte,
                        ctx.matte_reader.mode,
                    )
            if self.config.quality_report:
                for index, mask in enumerate(batch.masks):
                    if not batch.passthrough_flags[index]:
                        self._accumulate_quality_bbox(mask)
        self._finish_batch_masks(ctx, state, batch)

    def _finish_batch_masks(self, ctx: _FrameLoopContext,
                            state: _FrameLoopState,
                            batch: _FrameBatch) -> None:
        """Carry the last mask forward and report batch progress."""
        active_masks = [
            mask for index, mask in enumerate(batch.masks)
            if not batch.passthrough_flags[index]
        ]
        if active_masks:
            state.last_mask = active_masks[-1]
        progress = min(
            0.9,
            state.frame_idx / max(1, ctx.frames_to_process) * 0.8 + 0.1,
        )
        self._report_progress(
            progress,
            f"Processing frame {state.frame_idx}/{ctx.frames_to_process}...",
        )

    def _inpaint_batch(self, ctx: _FrameLoopContext,
                       state: _FrameLoopState,
                       batch: _FrameBatch) -> List[np.ndarray]:
        """Apply clean-reference overrides and inpaint active segments."""
        with self._time_stage("inpaint"):
            reference_frames = [frame.copy() for frame in batch.frames]
            fallback_masks = [mask.copy() for mask in batch.masks]
            if self._clean_reference_cache:
                batch_start = ctx.start_frame + state.written_idx
                for offset, (frame, mask) in enumerate(zip(
                        batch.frames, batch.masks, strict=True)):
                    if batch.passthrough_flags[offset]:
                        continue
                    absolute = batch_start + offset
                    seconds = _frame_seconds(
                        absolute, ctx.fps, ctx.frame_timing)
                    reference_frames[offset], fallback_masks[offset] = (
                        self._apply_clean_reference_overrides(
                            frame, mask, seconds)
                    )
            results = [frame.copy() for frame in reference_frames]
            for segment_start, segment_end in batch.active_segments:
                segment_masks = fallback_masks[segment_start:segment_end]
                if (
                    self._clean_reference_cache
                    and not any(np.any(mask > 0) for mask in segment_masks)
                ):
                    continue
                segment_frames = reference_frames[segment_start:segment_end]
                segment_results = self._validate_inpaint_results(
                    segment_frames,
                    self._inpaint_batch_resilient(
                        segment_frames, segment_masks
                    ),
                )
                results[segment_start:segment_end] = segment_results
            return results

    def _write_batch(self, ctx: _FrameLoopContext,
                     state: _FrameLoopState,
                     batch: _FrameBatch,
                     results: List[np.ndarray]) -> None:
        """Write cleaned frames, preview callbacks, and lossless mattes."""
        if self.config.quality_report and results:
            self._accumulate_seam_scores(
                batch.frames, results, batch.masks)
        stride = max(1, self.live_preview_stride)
        with self._time_stage("encode"):
            for offset, result in enumerate(results):
                write_frame = self._merge_high_bit_output(
                    batch.source_frames[offset]
                    if offset < len(batch.source_frames) else None,
                    result,
                    batch.masks[offset]
                    if offset < len(batch.masks) else None,
                )
                if self.config.quality_report:
                    output_frame_index = state.written_idx + offset
                    self._persist_quality_mask(
                        output_frame_index,
                        batch.masks[offset]
                        if offset < len(batch.masks) else None,
                    )
                ctx.writer.write(write_frame)
        for offset, result in enumerate(results):
            frame_index = state.written_idx + offset
            if (self.on_preview_frame is not None
                    and frame_index % stride == 0):
                try:
                    self.on_preview_frame(
                        result,
                        frame_index + 1,
                        ctx.frames_to_process,
                    )
                except Exception as exc:
                    logger.warning(
                        f"on_preview_frame hook raised: {exc}",
                        exc_info=True,
                    )
        if ctx.matte_writer is not None:
            with self._time_stage("encode"):
                for mask in batch.masks:
                    ctx.matte_writer.write(mask)
        state.written_idx += len(results)

    def _pause_requested(self, ctx: _FrameLoopContext) -> bool:
        """Ask once whether the caller wants to pause after this batch."""
        checkpoint = ctx.checkpoint
        if not checkpoint.active or checkpoint.root is None or not checkpoint.key:
            return False
        return bool(checkpoint.pause_check and checkpoint.pause_check())

    def _process_batch(self, ctx: _FrameLoopContext,
                       state: _FrameLoopState,
                       batch: _FrameBatch) -> None:
        """Refine, inpaint, and write one batch."""
        self._refine_batch_masks(ctx, state, batch)
        results = self._inpaint_batch(ctx, state, batch)
        self._write_batch(ctx, state, batch, results)

    def _checkpoint_after_batch(self, ctx: _FrameLoopContext,
                                state: _FrameLoopState,
                                should_pause: Optional[bool] = None) -> None:
        """Persist running/paused state after one fully-written batch."""
        checkpoint = ctx.checkpoint
        if not checkpoint.active or checkpoint.root is None or not checkpoint.key:
            return
        if should_pause is None:
            should_pause = bool(
                checkpoint.pause_check and checkpoint.pause_check())
        timing_metadata = checkpoint.timing_metadata
        if ctx.frame_timing is not None:
            timing_metadata = ctx.frame_timing.checkpoint_metadata(
                ctx.start_frame, ctx.end_frame, state.written_idx)
        payload = write_pause_checkpoint(
            checkpoint.root,
            checkpoint.key,
            input_path=checkpoint.input_path,
            output_path=checkpoint.output_path,
            config_hash=checkpoint.config_hash,
            frame_dir=checkpoint.frame_dir or pause_frame_dir(
                checkpoint.root, checkpoint.key),
            next_frame=state.written_idx,
            total_frames=ctx.frames_to_process,
            width=ctx.width,
            height=ctx.height,
            fps=ctx.fps,
            status="paused" if should_pause else "running",
            timing_manifest_path=checkpoint.timing_manifest_path,
            timing=timing_metadata,
        )
        self.last_pause_checkpoint = payload
        if checkpoint.state_path is not None:
            self.last_pause_checkpoint_path = str(checkpoint.state_path)
        if should_pause:
            message = (
                f"Processing paused at frame "
                f"{state.written_idx}/{ctx.frames_to_process}"
            )
            logger.info(message)
            raise ProcessingPaused(message, checkpoint.state_path)

