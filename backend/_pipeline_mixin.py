"""RM-349: the two entry points that run a job end to end.

`process_video` opens the input, resolves the window, sets up the readers,
writers, checkpoint and matte, drives the frame loop, then encodes and muxes.
`process_image` is the still-frame equivalent. Between them they were more
than a third of `processor.py`, which is why the file could not be read.

Nothing about them changed in the move. They are mixed into
`SubtitleRemover`, so `self` is the same object it always was, and the stages
they call still live on it.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from backend._frame_loop_types import (
    _FrameLoopCheckpoint,
    _FrameLoopContext,
    _FrameLoopState,
    _resolve_frame_range,
    _seek_capture_to_frame,
    _spans_from_segments,
)
from backend.detection_geometry import DetectionGeometry
from backend.execution_provenance import RequestedStageError
from backend.frozen_matte import (
    FrozenMatteError,
    normalize_frozen_matte,
    validate_frozen_matte,
)
from backend.hdr import hdr_repair_block_reason
from backend.io import (
    MediaInputError,
    MediaWriteError,
    VideoFrameTiming,
    _FrameSequenceWriter,
    _LosslessIntermediateWriter,
    _PrefetchReader,
    _cleanup_temp_output,
    _copy_file_atomic,
    _deinterlace_to_temp,
    _ensure_output_parent,
    _invalid_video_dimensions_error,
    _open_bgr48_capture,
    _open_capture,
    _probe_is_interlaced,
    _probe_keyframe_indices,
    _probe_video_frame_timing,
    _promote_temp_output,
    _validate_video_input_file,
    _video_capture_open_error,
    _video_decode_error,
    _write_text_atomic,
)
from backend.mask_corrections import (
    SELECTIVE_RERUN_SCHEMA,
    has_timed_corrections,
    merge_frame_ranges,
)
from backend.matte_interchange import (
    MaskInterchangeReader,
    MaskInterchangeWriter,
    mask_interchange_paths,
)
from backend.resume_checkpoint import (
    ProcessingPaused,
    _checkpoint_key,
    cleanup_pause_checkpoint,
    config_fingerprint,
    load_pause_checkpoint,
    pause_frame_dir,
    write_pause_checkpoint,
)
from backend.safe_image import safe_imread, safe_imwrite
from backend.tracking import SubtitleTracker

logger = logging.getLogger(__name__)


def _open_required_hdr_capture(path: str, *, input_fps: float):
    """Open the native high-bit HDR reader or fail before any 8-bit decode."""
    capture = _open_bgr48_capture(path, input_fps=input_fps)
    if capture is None:
        raise ValueError(
            "HDR high-bit decode unavailable; refusing an 8-bit fallback "
            "that would destroy the source surface."
        )
    return capture


class _PipelineMixin:
    """`process_video` and `process_image`."""

    def process_image(self, input_path: str, output_path: str) -> bool:
        self._teardown_requested = False
        self.last_output_path = None
        self._reset_stage_timings()
        self._reset_detection_stats()
        self._reset_job_execution_provenance()
        try:
            _ensure_output_parent(output_path)
            self._report_progress(0.1, "Loading image...")
            with self._time_stage("decode"):
                image = safe_imread(input_path)
            if image is None:
                raise ValueError(f"Could not load image: {input_path}")

            self._report_progress(0.3, "Detecting text regions...")
            fixed_shapes = self._fixed_region_shapes(0.0) or []
            fixed = self._fixed_region_boxes(0.0)
            manual_only = bool(self.config.sttn_skip_detection)
            if manual_only and not fixed_shapes:
                raise ValueError(
                    "Manual region mode needs a fixed region for image cleanup"
                )
            confidences = None
            detection_geometry: List[DetectionGeometry] = []
            self.last_detection_stats["frames_total"] = 1
            with self._time_stage("ocr"):
                if manual_only:
                    boxes = list(fixed or [])
                    detection_geometry = self._legacy_geometry(boxes)
                    self._record_detection_skip("manual_region")
                elif self.config.language_mask_filter:
                    from backend.detection import text_matches_detection_language
                    if callable(getattr(
                            self.detector, "detect_with_geometry", None)):
                        detection_geometry = self._detect_geometry(
                            image, self.config.detection_threshold)
                        matched_geometry = [
                            detection for detection in detection_geometry
                            if text_matches_detection_language(
                                detection.text,
                                self.config.detection_lang,
                            )
                        ]
                        if detection_geometry and not any(
                                detection.text for detection in detection_geometry):
                            matched_geometry = []
                        detection_geometry = matched_geometry
                        boxes = [
                            detection.bbox
                            for detection in detection_geometry
                        ]
                        if self.config.confidence_weighted_dilation:
                            confidences = [
                                detection.confidence
                                for detection in detection_geometry
                            ]
                    else:
                        results = self.detector.detect_with_text(
                            image, self.config.detection_threshold)
                        matched = [
                            result for result in results
                            if text_matches_detection_language(
                                result[5], self.config.detection_lang)
                        ]
                        boxes = [result[:4] for result in matched]
                        if self.config.confidence_weighted_dilation:
                            confidences = [result[4] for result in matched]
                    self._record_ocr_detection(boxes)
                elif self.config.confidence_weighted_dilation:
                    if callable(getattr(
                            self.detector, "detect_with_geometry", None)):
                        detection_geometry = self._detect_geometry(
                            image, self.config.detection_threshold)
                        boxes = [
                            detection.bbox
                            for detection in detection_geometry
                        ]
                        confidences = [
                            detection.confidence
                            for detection in detection_geometry
                        ]
                    else:
                        results = self.detector.detect_with_confidence(
                            image, self.config.detection_threshold)
                        boxes = [
                            (x1, y1, x2, y2)
                            for x1, y1, x2, y2, _ in results
                        ]
                        confidences = [c for _, _, _, _, c in results]
                    self._record_ocr_detection(boxes)
                else:
                    if callable(getattr(
                            self.detector, "detect_with_geometry", None)):
                        detection_geometry = self._detect_geometry(
                            image, self.config.detection_threshold)
                        boxes = [
                            detection.bbox
                            for detection in detection_geometry
                        ]
                    else:
                        boxes = self.detector.detect(
                            image, self.config.detection_threshold)
                    self._record_ocr_detection(boxes)

                if not manual_only and fixed:
                    boxes = list(fixed) + list(boxes)
                    detection_geometry = (
                        self._legacy_geometry(list(fixed))
                        + detection_geometry
                    )
                    confidences = None

            if not boxes and not fixed_shapes:
                logger.info("No text detected, copying original")
                with self._time_stage("encode"):
                    _copy_file_atomic(input_path, output_path)
                self.last_output_path = output_path
                self._write_reproducibility_sidecar(input_path, output_path)
                self.last_error_message = None
                self.last_error_reason = None
                self._report_progress(1.0, "Complete (no text found)")
                return True

            region_count = max(len(boxes), len(fixed_shapes))
            self._report_progress(0.5, f"Removing {region_count} text regions...")
            with self._time_stage("mask"):
                mask = self._create_mask(image.shape, boxes, frame=image,
                                         confidences=confidences,
                                         detections=detection_geometry)
                mask = self._apply_polygon_region_shapes(mask, fixed_shapes)
                mask = self._apply_manual_mask_corrections(mask, 0.0, 0)
                [mask] = self._refine_masks_with_matanyone([image], [mask])
            with self._time_stage("inpaint"):
                results = self._validate_inpaint_results(
                    [image], self._execute_inpainter([image], [mask])
                )
                [result] = results

            self._report_progress(0.9, "Saving result...")
            ext = Path(output_path).suffix.lower()
            temp_output = self._allocate_work_output(output_path)
            try:
                with self._time_stage("encode"):
                    if ext in ('.jpg', '.jpeg'):
                        ok = safe_imwrite(temp_output, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    elif ext == '.png':
                        ok = safe_imwrite(temp_output, result, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    elif ext == '.webp':
                        ok = safe_imwrite(temp_output, result, [cv2.IMWRITE_WEBP_QUALITY, 95])
                    else:
                        ok = safe_imwrite(temp_output, result)
                    if not ok:
                        raise IOError(f"Failed to write output image: {output_path}")
                    _promote_temp_output(temp_output, output_path)
            finally:
                _cleanup_temp_output(temp_output)
            self.last_output_path = output_path
            self._write_reproducibility_sidecar(input_path, output_path)
            self.last_error_message = None
            self.last_error_reason = None
            self._report_progress(1.0, "Complete!")
            return True

        except InterruptedError:
            logger.info("Image processing cancelled")
            raise
        except RequestedStageError as e:
            self._record_requested_stage_failure(e)
            logger.error("Requested image stage failed: %s", e, exc_info=True)
            return False
        except Exception as e:
            self.last_error_message = str(e)
            self.last_error_reason = "image_processing_error"
            logger.error(f"Image processing error: {e}", exc_info=True)
            return False

    def process_video(self, input_path: str, output_path: str, *,
                      checkpoint_dir: Optional[str | Path] = None,
                      checkpoint_key: Optional[str] = None,
                      resume_checkpoint: bool = True,
                      pause_check: Optional[Callable[[], bool]] = None,
                      selective_rerun_from: Optional[str] = None,
                      selective_rerun_ranges: Optional[
                          List[Tuple[int, int]]
                      ] = None) -> bool:
        self._teardown_requested = False
        self.last_output_path = None
        self.last_resume_warning = None
        self.last_pause_checkpoint = None
        self.last_pause_checkpoint_path = None
        self._reset_stage_timings()
        self._reset_detection_stats()
        self._reset_job_execution_provenance()
        self._srt_entries = []
        self._ocr_fix_replacements = None
        self._quality_mask_bbox = None
        self._seam_scores = []
        self._seam_score_failure_logged = False
        self._quality_temporal_previous = None
        self._quality_temporal_scores = []
        self._quality_temporal_scene_cuts_excluded = 0
        self._quality_temporal_worst_pair = None
        self._quality_temporal_failure_logged = False
        self._quality_color_drift_sum = 0.0
        self._quality_color_drift_count = 0
        self._quality_color_drift_metric = None
        self._quality_color_drift_worst_frame = None
        self._quality_color_failure_logged = False
        self._quality_frame_evidence_dir = None
        self._quality_frame_evidence_write_error = False
        self._quality_final_encode_verified = False
        self._mask_review_signals = []
        self.last_selective_rerun = None
        temp_dir = None
        cap = None
        selective_cap = None
        selective_prior_offset = 0
        selective_ranges = merge_frame_ranges(selective_rerun_ranges or [])
        reader = None
        writer = None
        matte_writer = None
        matte_reader = None
        whisper_audio_dir = None
        self.last_error_message = None
        self.last_error_reason = None
        self.last_mask_export = {
            "requested": bool(self.config.export_mask_video),
            "status": (
                "pending" if self.config.export_mask_video else "not-requested"
            ),
            "path": "",
            "format": self.config.mask_export_format,
        }
        self.last_mask_import = {
            "requested": bool(self.config.mask_import_path),
            "status": "pending" if self.config.mask_import_path else "not-requested",
            "manifest": self.config.mask_import_path,
            "mode": self.config.mask_import_mode,
        }
        frozen_record = normalize_frozen_matte(
            getattr(self.config, "frozen_matte", None))
        self.last_frozen_matte = {
            "requested": bool(frozen_record),
            "status": "pending" if frozen_record else "not-requested",
        }
        self.last_translation = {
            "requested": bool(self.config.translation_enabled),
            "status": (
                "pending" if self.config.translation_enabled else "not-requested"
            ),
        }
        self._translation_burn_path = ""
        self._whisper_segments = []
        clean_reference_requested = self._clean_reference_requested()
        self.last_clean_reference = {
            "requested": clean_reference_requested,
            "status": (
                "pending" if clean_reference_requested else "not-requested"
            ),
        }
        self._clean_reference_cache = {}
        self._clean_reference_warned = set()
        self.last_timing_report = {
            "mode": "unknown",
            "frame_count": 0,
            "duration_seconds": 0.0,
            "time_base_seconds": 0.0,
            "average_fps": 0.0,
        }
        self.last_output_contract = {}
        self.last_container_payload = {}
        self._color_metadata = None
        self._output_contract = None
        try:
            _ensure_output_parent(output_path)
            self._report_progress(0.0, "Opening video...")
            _validate_video_input_file(input_path)
            self._prepare_output_contract(input_path, output_path)
            if getattr(self, "_hdr_probe_failed", False):
                raise ValueError(
                    "HDR processing stopped because color metadata could not "
                    "be probed. Disable color preservation only as an explicit "
                    "override when the source is known to be SDR."
                )
            if getattr(self, "_hdr_override_blocked", False):
                raise ValueError(
                    "Color preservation cannot be disabled for an unknown, "
                    "invalid, high-bit, or HDR source; a verified SDR profile "
                    "is required before using that override."
                )
            if getattr(self, "_hdr_repair_blocked", False):
                reason = hdr_repair_block_reason(self._color_metadata)
                raise ValueError(
                    "HDR processing stopped because the source tags are not "
                    f"safe for linear-light repair: {reason}"
                )

            # Optional deinterlace preprocessing. Produces a temp
            # progressive-scan mp4; the rest of the pipeline runs against
            # that file transparently.
            should_deinterlace = self.config.deinterlace
            if self.config.deinterlace_auto and not should_deinterlace:
                if _probe_is_interlaced(input_path):
                    logger.info("Interlaced source detected -- enabling yadif")
                    should_deinterlace = True
            if should_deinterlace:
                self._report_progress(0.02, "Deinterlacing source...")
                temp_dir = self._make_temp_dir()
                try:
                    processed_input = _deinterlace_to_temp(
                        input_path,
                        temp_dir,
                        output_contract=self._output_contract,
                        prefer_d3d12=(
                            self.config.d3d12_accel
                            and not self._source_is_hdr()
                        ),
                        on_process=self._set_active_subprocess,
                        cancel_check=self._is_teardown_requested,
                    )
                    logger.info(f"Using deinterlaced source: {processed_input}")
                    decode_path = processed_input
                except InterruptedError:
                    # A cancel here is a cancel, not a deinterlace failure to
                    # shrug off; swallowing it kept the job decoding and
                    # inpainting until the next progress callback.
                    raise
                except Exception as exc:
                    logger.warning(
                        f"Deinterlace failed, continuing with original: {exc}",
                        exc_info=True,
                    )
                    decode_path = input_path
            else:
                decode_path = input_path

            # Optional keyframe-driven detection: get the set of I-frame
            # indices once, OCR only those, propagate masks between.
            keyframe_set: Optional[set] = None
            if self.config.keyframe_detection:
                self._report_progress(0.04, "Probing keyframes...")
                keyframe_set = _probe_keyframe_indices(decode_path)
                if keyframe_set:
                    logger.info(f"Keyframe-driven detection: {len(keyframe_set)} I-frames")
                else:
                    logger.warning("Keyframe probe failed, falling back to pHash skip")

            cap = None
            if self._source_is_hdr() and getattr(
                    self, "_hdr_repair_ready", False):
                cap = _open_required_hdr_capture(
                    decode_path,
                    input_fps=self.config.input_fps,
                )
            if cap is None:
                cap = _open_capture(
                    decode_path,
                    self.config.decode_hw_accel,
                    input_fps=self.config.input_fps,
                )
            if not cap.isOpened():
                raise _video_capture_open_error(input_path, decode_path)
            # Stash the decode path so other routines (keyframe, audio merge)
            # can read it without re-resolving.
            self._decode_path = decode_path

            raw_fps = cap.get(cv2.CAP_PROP_FPS)
            # cv2 returns 0.0 on failure and can return NaN on exotic codecs;
            # both break downstream frame-to-time math, so coerce to a sane
            # default rather than let the pipeline divide by zero or NaN.
            try:
                raw_fps = float(raw_fps)
            except (TypeError, ValueError):
                raw_fps = 0.0
            if not np.isfinite(raw_fps) or raw_fps <= 0.0:
                logger.warning("Invalid / missing FPS metadata; falling back to 30.0")
                raw_fps = 30.0
            # Clamp absurdly high values (some malformed containers report
            # 1e6 FPS) so the writer doesn't stall on an impossible frame rate.
            fps = float(min(raw_fps, 1000.0))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            high_bit_depth_surface = (
                getattr(cap, "pixel_format", "") == "bgr48le"
            )
            frame_timing: Optional[VideoFrameTiming] = None
            if not Path(decode_path).is_dir():
                frame_timing = _probe_video_frame_timing(decode_path)
                if (
                    frame_timing is not None
                    and frame_timing.frame_count != total_frames
                ):
                    logger.warning(
                        "Using ffprobe's %d-frame PTS map instead of the "
                        "decoder's %d-frame header estimate",
                        frame_timing.frame_count,
                        total_frames,
                    )
                    total_frames = frame_timing.frame_count
            if frame_timing is not None:
                set_frame_timing = getattr(cap, "set_frame_timing", None)
                if callable(set_frame_timing):
                    set_frame_timing(frame_timing)
                if frame_timing.average_fps > 0:
                    fps = float(min(frame_timing.average_fps, 1000.0))
                self.last_timing_report = frame_timing.report()
                timing_label = "variable" if frame_timing.is_vfr else "constant"
                logger.info(
                    "Source timing: %s frame rate, %d timestamps, "
                    "time base %d/%d s (%0.12fs)",
                    timing_label,
                    frame_timing.frame_count,
                    frame_timing.time_base_num,
                    frame_timing.time_base_den,
                    frame_timing.time_base,
                )
            else:
                self.last_timing_report = {
                    "mode": "cfr-fallback",
                    "frame_count": total_frames,
                    "duration_seconds": round(total_frames / fps, 9),
                    "time_base_seconds": 0.0,
                    "time_base_num": 0,
                    "time_base_den": 1,
                    "source_start_ticks": 0,
                    "timing_anomaly_count": 0,
                    "timing_anomalies": [],
                    "average_fps": round(fps, 6),
                }

            if width == 0 or height == 0:
                raise _invalid_video_dimensions_error(input_path, width, height)
            self._initialize_clean_references(width, height)

            # Time range support. Resolved (with NaN/inf/negative guards and
            # the cap seek) in _resolve_frame_range so the frame<->time math is
            # unit-testable instead of buried inline here.
            _range = _resolve_frame_range(
                cap, total_frames, fps, frame_timing,
                self.config.time_start, self.config.time_end)
            start_frame = _range.start_frame
            end_frame = _range.end_frame
            frames_to_process = _range.frames_to_process
            selected_frame_durations = _range.selected_frame_durations
            processed_time_start = _range.processed_time_start
            processed_time_end = _range.processed_time_end
            processed_time_start_ticks = _range.processed_time_start_ticks
            processed_time_end_ticks = _range.processed_time_end_ticks
            matte_timestamps = _range.matte_timestamps
            matte_durations = _range.matte_durations
            matte_time_base = _range.matte_time_base
            matte_timestamp_ticks = _range.matte_timestamp_ticks
            matte_duration_ticks = _range.matte_duration_ticks
            matte_time_base_num = _range.matte_time_base_num
            matte_time_base_den = _range.matte_time_base_den
            selected_frame_duration_ticks = (
                _range.selected_frame_duration_ticks
            )
            source_start_ticks = (
                int(frame_timing.source_start_ticks)
                if frame_timing is not None else None
            )
            stream_start_ticks = (
                int(frame_timing.stream_start_ticks)
                if frame_timing is not None else None
            )
            if frozen_record:
                # RM-153: revalidate before a single frame is decoded, so
                # a matte that no longer belongs to this job stops the run
                # with a specific reason instead of painting approved
                # pixels onto frames they were never approved for.
                if self.config.mask_import_path:
                    raise ValueError(
                        "A frozen matte and a manually imported matte cannot "
                        "both drive one job; clear one of them."
                    )
                frozen_evidence = validate_frozen_matte(
                    frozen_record,
                    source_path=input_path,
                    width=width,
                    height=height,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    timestamps=matte_timestamps,
                    durations=matte_durations,
                    is_vfr=bool(
                        frame_timing is not None and frame_timing.is_vfr),
                    source_time_base=matte_time_base,
                    timestamp_ticks=matte_timestamp_ticks,
                    duration_ticks=matte_duration_ticks,
                    source_time_base_num=matte_time_base_num,
                    source_time_base_den=matte_time_base_den,
                    source_start_ticks=source_start_ticks,
                    stream_start_ticks=stream_start_ticks,
                )
                matte_reader = MaskInterchangeReader(
                    frozen_record["manifest"],
                    width=width,
                    height=height,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    timestamps=matte_timestamps,
                    durations=matte_durations,
                    is_vfr=bool(
                        frame_timing is not None and frame_timing.is_vfr),
                    source_time_base=matte_time_base,
                    timestamp_ticks=matte_timestamp_ticks,
                    duration_ticks=matte_duration_ticks,
                    source_time_base_num=matte_time_base_num,
                    source_time_base_den=matte_time_base_den,
                    source_start_ticks=source_start_ticks,
                    stream_start_ticks=stream_start_ticks,
                    mode="replace",
                )
                self.last_frozen_matte = {
                    **frozen_evidence,
                    "requested": True,
                }
                logger.info(
                    "Reusing frozen %s matte (%d frames); skipping OCR, "
                    "tracking, and mask refiners",
                    frozen_record["format"],
                    frozen_record["frame_count"],
                )
            elif self.config.mask_import_path:
                matte_reader = MaskInterchangeReader(
                    self.config.mask_import_path,
                    width=width,
                    height=height,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    timestamps=matte_timestamps,
                    durations=matte_durations,
                    is_vfr=bool(
                        frame_timing is not None and frame_timing.is_vfr),
                    source_time_base=matte_time_base,
                    timestamp_ticks=matte_timestamp_ticks,
                    duration_ticks=matte_duration_ticks,
                    source_time_base_num=matte_time_base_num,
                    source_time_base_den=matte_time_base_den,
                    source_start_ticks=source_start_ticks,
                    stream_start_ticks=stream_start_ticks,
                    mode=self.config.mask_import_mode,
                )
                self.last_mask_import = {
                    **matte_reader.evidence,
                    "requested": True,
                    "status": "validated",
                }
                logger.info(
                    "Validated imported %s matte (%s mode, %d frames)",
                    matte_reader.export_format,
                    matte_reader.mode,
                    matte_reader.frame_count,
                )

            if selective_rerun_from and self.config.export_mask_video:
                logger.warning(
                    "A complete matte export was requested; running all frames "
                    "instead of reusing cleaned frames without their masks"
                )
                selective_rerun_from = None
                selective_ranges = []

            if selective_rerun_from:
                selective_path = Path(selective_rerun_from)
                if not selective_path.is_file():
                    raise ValueError(
                        "Selective mask rerun requires the previous cleaned output"
                    )
                selective_ranges = merge_frame_ranges(
                    (
                        max(start_frame, range_start),
                        min(end_frame, range_end),
                    )
                    for range_start, range_end in selective_ranges
                )
                if not selective_ranges:
                    raise ValueError(
                        "Selective mask rerun has no valid affected frame range"
                )
                if self._source_is_hdr() and getattr(
                        self, "_hdr_repair_ready", False):
                    selective_cap = _open_required_hdr_capture(
                        str(selective_path),
                        input_fps=self.config.input_fps,
                    )
                if selective_cap is None:
                    selective_cap = _open_capture(str(selective_path), "off")
                if not selective_cap.isOpened():
                    raise ValueError(
                        "Could not open the previous cleaned output for selective rerun"
                    )
                prior_width = int(selective_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                prior_height = int(selective_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                prior_frames = int(selective_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if (prior_width, prior_height) != (width, height):
                    raise ValueError(
                        "Previous cleaned output dimensions do not match the source"
                    )
                if start_frame > 0 and prior_frames == frames_to_process:
                    # A range-relative output starts at source frame zero from
                    # the capture's point of view. Do not add start_frame a
                    # second time when reading it for selective reruns.
                    selective_prior_offset = 0
                elif prior_frames >= end_frame:
                    # A full-source output keeps source frame numbering.
                    selective_prior_offset = start_frame
                else:
                    raise ValueError(
                        "Previous cleaned output does not cover the requested "
                        "source range; selective rerun with a time range "
                        "requires a range-length output"
                    )
                prior_stat = selective_path.stat()
                rerun_frame_count = sum(end - start for start, end in selective_ranges)
                self.last_selective_rerun = {
                    "schema": SELECTIVE_RERUN_SCHEMA,
                    "source_output": selective_path.name,
                    "source_output_bytes": int(prior_stat.st_size),
                    "source_output_mtime_ns": int(prior_stat.st_mtime_ns),
                    "ranges": [list(frame_range) for frame_range in selective_ranges],
                    "rerun_frames": rerun_frame_count,
                    "reused_frames": max(0, frames_to_process - rerun_frame_count),
                }
                logger.info(
                    "Selective mask rerun: %d affected frame(s), %d reused frame(s)",
                    rerun_frame_count,
                    max(0, frames_to_process - rerun_frame_count),
                )

            if start_frame > 0 or end_frame < total_frames:
                logger.info(f"Video: {width}x{height} @ {fps:.1f}fps, "
                           f"frames {start_frame}-{end_frame} of {total_frames}")
            else:
                logger.info(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

            checkpoint_root: Optional[Path] = None
            checkpoint_frame_dir: Optional[Path] = None
            checkpoint_state_path: Optional[Path] = None
            checkpoint_config_hash = ""
            checkpoint_active = False
            checkpoint_remove_frames_on_success = True
            resume_frame_count = 0
            if checkpoint_dir is not None:
                checkpoint_root = Path(checkpoint_dir)
                checkpoint_root.mkdir(parents=True, exist_ok=True)
                checkpoint_key = checkpoint_key or _checkpoint_key(
                    input_path, output_path, self.config)
                checkpoint_state_path = (
                    checkpoint_root / f"{checkpoint_key}.pause.json"
                )
                checkpoint_config_hash = config_fingerprint(self.config)
                default_frame_dir = pause_frame_dir(
                    checkpoint_root, checkpoint_key)
                checkpoint_frame_dir = default_frame_dir
                checkpoint_active = True
                if resume_checkpoint:
                    state = load_pause_checkpoint(
                        checkpoint_root,
                        checkpoint_key,
                        input_path=input_path,
                        output_path=output_path,
                        config_hash=checkpoint_config_hash,
                        total_frames=frames_to_process,
                        width=width,
                        height=height,
                        fps=fps,
                        timing=(
                            frame_timing.checkpoint_metadata(
                                start_frame, end_frame, 0)
                            if frame_timing is not None else None
                        ),
                    )
                    checkpoint_state_path = state.path
                    checkpoint_frame_dir = state.frame_dir
                    resume_frame_count = min(frames_to_process, state.next_frame)
                    if state.warning:
                        self.last_resume_warning = state.warning
                        logger.warning(state.warning)
                    if state.inpaint_complete and resume_frame_count >= frames_to_process:
                        logger.info(
                            f"Resuming {Path(input_path).name} at the encode "
                            "stage; all frames were already inpainted."
                        )
                    if resume_frame_count > 0:
                        _seek_capture_to_frame(
                            cap, start_frame + resume_frame_count)
                        logger.info(
                            f"Resuming {Path(input_path).name} from frame "
                            f"{resume_frame_count}/{frames_to_process}"
                        )

            restart_for_derived_outputs = any(
                bool(getattr(self.config, name, False))
                for name in (
                    "export_mask_video",
                    "export_srt",
                    "translation_enabled",
                    "quality_report",
                    "quality_report_sheet",
                )
            )
            if restart_for_derived_outputs and resume_frame_count > 0:
                logger.warning(
                    "Restarting from frame zero so requested sidecars, reports, "
                    "and exports contain a complete timestamp-aligned sequence"
                )
                resume_frame_count = 0
                _seek_capture_to_frame(cap, start_frame)

            if selective_cap is not None:
                _seek_capture_to_frame(
                    selective_cap, selective_prior_offset + resume_frame_count)

            # Fail fast on a drive that clearly cannot hold the encode, before
            # any temp file is created (only the frames still to write count).
            self._check_encode_disk_space(
                output_path,
                width=width,
                height=height,
                frames=max(0, frames_to_process - resume_frame_count),
                high_bit=bool(high_bit_depth_surface),
                checkpoint_dir=checkpoint_root,
            )

            # Re-use the deinterlace temp_dir if one was created, else fresh
            if temp_dir is None:
                temp_dir = self._make_temp_dir()
            if self.config.quality_report:
                self._quality_frame_evidence_dir = Path(temp_dir) / "quality_masks"
                self._quality_frame_evidence_dir.mkdir(
                    parents=True, exist_ok=True,
                )
                self._quality_frame_evidence_write_error = False
            # I-1: lossless FFV1 intermediate inside .mkv. The previous
            # mp4v intermediate cost a full generation of lossy
            # compression before the final ffmpeg encode pass. The
            # writer falls back to mp4v + .mp4 when ffmpeg is missing
            # so the pipeline still produces output, just at the old
            # quality.
            use_frame_output = getattr(self.config, "output_frames", False)
            vfr_frame_dir: Optional[Path] = None
            timing_manifest_path: Optional[Path] = None
            with self._time_stage("encode"):
                if use_frame_output:
                    frame_out_dir = output_path
                    if not frame_out_dir.endswith(os.sep):
                        frame_out_dir = str(Path(output_path).with_suffix(""))
                    if checkpoint_active:
                        checkpoint_frame_dir = Path(frame_out_dir)
                        checkpoint_remove_frames_on_success = False
                    writer = _FrameSequenceWriter(
                        frame_out_dir,
                        start_index=resume_frame_count,
                    )
                    temp_video = None
                elif checkpoint_active:
                    if checkpoint_frame_dir is None:
                        checkpoint_frame_dir = pause_frame_dir(
                            checkpoint_root, checkpoint_key)  # type: ignore[arg-type]
                    writer = _FrameSequenceWriter(
                        str(checkpoint_frame_dir),
                        start_index=resume_frame_count,
                    )
                    temp_video = None
                elif frame_timing is not None and frame_timing.is_vfr:
                    vfr_frame_dir = Path(temp_dir) / "vfr_frames"
                    writer = _FrameSequenceWriter(str(vfr_frame_dir))
                    temp_video = None
                else:
                    temp_video_target = os.path.join(temp_dir, "temp_video.mkv")
                    writer = _LosslessIntermediateWriter(
                        temp_video_target,
                        width,
                        height,
                        fps,
                        pixel_format="bgr48le" if high_bit_depth_surface else "bgr24",
                    )
                    self._active_writer = writer
                    temp_video = writer.path
                    if not writer.isOpened():
                        raise ValueError(
                            f"Could not create video writer for: {temp_video}"
                        )

            if (
                frame_timing is not None
                and frame_timing.is_vfr
            ):
                timing_dir = (
                    Path(frame_out_dir)
                    if use_frame_output else checkpoint_frame_dir or vfr_frame_dir
                )
                if timing_dir is not None:
                    timing_manifest_path = Path(timing_dir) / "frame_timing.json"
                    timing_payload = {
                        "schema": "vsr.frame_timing.v2",
                        "source": str(input_path),
                        "source_start_seconds": frame_timing.source_start,
                        "source_start_ticks": frame_timing.source_start_ticks,
                        "stream_start_ticks": frame_timing.stream_start_ticks,
                        "source_time_base_num": frame_timing.time_base_num,
                        "source_time_base_den": frame_timing.time_base_den,
                        "source_time_base": {
                            "num": frame_timing.time_base_num,
                            "den": frame_timing.time_base_den,
                        },
                        "source_time_base_seconds": frame_timing.time_base,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "timestamp_ticks": matte_timestamp_ticks,
                        "duration_ticks": matte_duration_ticks,
                        "timestamps_seconds": [
                            frame_timing.frame_time(index, fps)
                            for index in range(start_frame, end_frame)
                        ],
                        "durations_seconds": selected_frame_durations,
                        "timing_anomalies": frame_timing.anomalies,
                    }
                    _write_text_atomic(
                        timing_manifest_path,
                        json.dumps(
                            timing_payload,
                            indent=2,
                            ensure_ascii=True,
                        ) + "\n",
                    )

            timing_checkpoint_metadata = (
                frame_timing.checkpoint_metadata(
                    start_frame, end_frame, resume_frame_count)
                if frame_timing is not None else None
            )

            if checkpoint_active and checkpoint_root is not None and checkpoint_key:
                payload = write_pause_checkpoint(
                    checkpoint_root,
                    checkpoint_key,
                    input_path=input_path,
                    output_path=output_path,
                    config_hash=checkpoint_config_hash,
                    frame_dir=checkpoint_frame_dir or pause_frame_dir(
                        checkpoint_root, checkpoint_key),
                    next_frame=resume_frame_count,
                    total_frames=frames_to_process,
                    width=width,
                    height=height,
                    fps=fps,
                    status="running",
                    timing_manifest_path=timing_manifest_path,
                    timing=timing_checkpoint_metadata,
                )
                self.last_pause_checkpoint = payload
                if checkpoint_state_path is not None:
                    self.last_pause_checkpoint_path = str(checkpoint_state_path)

            frame_idx = resume_frame_count
            batch_size = self.config.sttn_max_load_num
            frame_skip = self.config.detection_frame_skip
            rife_stride = self._rife_fast_stride()
            if rife_stride:
                frame_skip = max(frame_skip, rife_stride - 1)
                logger.info(
                    f"RIFE fast mode on: stride={rife_stride}, "
                    f"effective detection frame-skip={frame_skip}"
                )

            # Decoupled prefetch: wrap the capture in a worker that fills
            # a bounded frame queue while the main thread runs detection +
            # inpainting. cv2.VideoCapture must NOT be touched directly
            # (.set / .get / .read) after this point -- the worker owns it.
            # Seek + metadata reads above happen *before* the wrap, so this
            # is safe; cleanup goes through `reader.release()`.
            with self._time_stage("decode"):
                if self.config.prefetch_decode:
                    qsize = self.config.prefetch_queue_size or max(8, batch_size * 2)
                    reader = _PrefetchReader(cap, max_frames=frames_to_process,
                                              queue_size=qsize)
                    logger.info(f"Prefetch decode on (queue={qsize})")
                else:
                    reader = cap
            last_mask = None  # cached mask for frame-skip optimization
            fixed_mask_cache = {}  # cached masks for skip_detection mode

            # RM-27 Whisper fallback: pre-compute frame spans where
            # Whisper detected speech. When OCR returns no boxes for a
            # frame inside one of these spans we apply a default
            # bottom-band mask so the subtitle band still gets
            # inpainted. Done once per file so we don't pay model load
            # for every batch.
            whisper_spans: List[Tuple[int, int]] = []
            whisper_audio_dir: Optional[str] = None
            segments = None
            if self.config.whisper_fallback and not Path(input_path).is_dir():
                try:
                    from backend import whisper_fallback as _wf
                    if self.config.whisper_backend == "ffmpeg":
                        segments = _wf.run_ffmpeg_whisper_segments(
                            input_path,
                            model_path=self.config.whisper_model_path,
                            language=(self.config.detection_lang or None),
                            queue_seconds=self.config.whisper_queue_seconds,
                            vad_model=self.config.whisper_vad_model,
                            vad_threshold=self.config.whisper_vad_threshold,
                            min_speech_duration=self.config.whisper_min_speech_duration,
                        )
                        if segments:
                            whisper_spans = _spans_from_segments(
                                segments,
                                fps=fps,
                                total_frames=total_frames,
                                frame_timing=frame_timing,
                            )
                            logger.info(
                                f"FFmpeg Whisper fallback active: "
                                f"{len(whisper_spans)} speech spans"
                            )
                    elif _wf.is_available():
                        whisper_audio_dir = self._make_temp_dir(
                            prefix="vsr_whisper_")
                        audio_path = _wf.extract_audio_to_temp(
                            input_path, whisper_audio_dir
                        )
                        if audio_path:
                            segments = _wf.run_whisper_segments(
                                audio_path,
                                model_size=self.config.whisper_model_size,
                                language=(self.config.detection_lang or None),
                            )
                            if segments:
                                whisper_spans = _spans_from_segments(
                                    segments,
                                    fps=fps,
                                    total_frames=total_frames,
                                    frame_timing=frame_timing,
                                )
                                logger.info(
                                    f"Whisper fallback active: "
                                    f"{len(whisper_spans)} speech spans"
                                )
                    if segments:
                        self._whisper_segments = [
                            (float(segment[0]), float(segment[1]), str(segment[2]))
                            for segment in segments
                            if len(segment) >= 3 and str(segment[2]).strip()
                        ]
                except Exception as exc:
                    logger.warning(
                        f"Whisper fallback setup failed: {exc}",
                        exc_info=True,
                    )

            # v3.10: Kalman tracker for detection smoothing
            tracker = (SubtitleTracker(self.config.kalman_iou_threshold,
                                         self.config.kalman_max_age)
                        if self.config.kalman_tracking else None)
            collect_srt_text = bool(
                self.config.export_srt
                or (
                    self.config.translation_enabled
                    and not self.config.translation_srt
                    and not self.config.translation_source_srt
                )
            )
            srt_tracker = (
                tracker
                if collect_srt_text and tracker is not None
                else SubtitleTracker(
                    self.config.kalman_iou_threshold,
                    self.config.kalman_max_age,
                ) if collect_srt_text else None
            )
            # v3.10: pHash for adaptive mask reuse
            last_hash = None

            # Lossless mask/alpha-matte interchange artifact.
            if self.config.export_mask_video:
                mask_path, mask_manifest_path = mask_interchange_paths(
                    output_path, self.config.mask_export_format)
                self.last_mask_export.update({
                    "path": str(mask_path),
                    "manifest": str(mask_manifest_path),
                })
                try:
                    matte_writer = MaskInterchangeWriter(
                        output_path,
                        self.config.mask_export_format,
                        width=width,
                        height=height,
                        fps=fps,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        timestamps=matte_timestamps,
                        durations=matte_durations,
                        is_vfr=bool(
                            frame_timing is not None and frame_timing.is_vfr),
                        source_time_base=matte_time_base,
                        timestamp_ticks=matte_timestamp_ticks,
                        duration_ticks=matte_duration_ticks,
                        source_time_base_num=matte_time_base_num,
                        source_time_base_den=matte_time_base_den,
                        source_start_ticks=source_start_ticks,
                        stream_start_ticks=stream_start_ticks,
                    )
                except Exception as exc:
                    self.last_mask_export.update({
                        "status": "failed",
                        "error": str(exc),
                    })
                    raise

            timed_region_spans = bool(
                getattr(self.config, "subtitle_region_spans", None)
                or getattr(self.config, "subtitle_region_keyframes", None)
            )
            timed_mask_corrections = has_timed_corrections(
                getattr(self.config, "manual_mask_corrections", None)
            )
            static_fixed_shapes = (
                None if timed_region_spans else self._fixed_region_shapes())

            loop_ctx = _FrameLoopContext(
                start_frame=start_frame,
                end_frame=end_frame,
                frames_to_process=frames_to_process,
                fps=fps,
                width=width,
                height=height,
                total_frames=total_frames,
                frame_timing=frame_timing,
                high_bit_depth_surface=high_bit_depth_surface,
                batch_size=batch_size,
                frame_skip=frame_skip,
                rife_stride=rife_stride,
                keyframe_set=keyframe_set,
                whisper_spans=whisper_spans,
                timed_region_spans=timed_region_spans,
                timed_mask_corrections=timed_mask_corrections,
                static_fixed_shapes=static_fixed_shapes,
                selective_ranges=selective_ranges,
                reader=reader,
                selective_cap=selective_cap,
                matte_reader=matte_reader,
                frozen_matte=bool(frozen_record),
                writer=writer,
                matte_writer=matte_writer,
                checkpoint=_FrameLoopCheckpoint(
                    active=checkpoint_active,
                    root=checkpoint_root,
                    key=checkpoint_key,
                    state_path=checkpoint_state_path,
                    config_hash=checkpoint_config_hash,
                    frame_dir=checkpoint_frame_dir,
                    timing_manifest_path=timing_manifest_path,
                    timing_metadata=timing_checkpoint_metadata,
                    input_path=input_path,
                    output_path=output_path,
                    pause_check=pause_check,
                ),
            )
            loop_state = _FrameLoopState(
                frame_idx=frame_idx,
                last_mask=last_mask,
                last_hash=last_hash,
                tracker=tracker,
                srt_tracker=srt_tracker,
                fixed_mask_cache=fixed_mask_cache,
                written_idx=frame_idx,
            )
            # RM-296: a fade-in hold looks FORWARD, so the last `fade_in`
            # frames of a batch cannot be finalised until the next batch's
            # masks exist. Hold them back and re-attach them to the front of
            # the next batch, which always brings at least batch_size frames
            # of lookahead with it.
            fade_in_hold = (
                int(getattr(self.config, "mask_fade_in_frames", 0) or 0)
                if not loop_ctx.frozen_matte else 0
            )
            while True:
                batch = self._decode_and_build_batch(loop_ctx, loop_state)
                decoded = len(batch.frames)
                if loop_state.fade_pending is not None:
                    batch.prepend(loop_state.fade_pending)
                    loop_state.fade_pending = None
                if not batch.frames:
                    break
                at_end = (
                    decoded < loop_ctx.batch_size
                    or loop_ctx.start_frame + loop_state.frame_idx
                    >= loop_ctx.end_frame
                )
                if (
                    fade_in_hold > 0
                    and not at_end
                    and len(batch.frames) > fade_in_hold
                ):
                    loop_state.fade_pending = batch.split_tail(fade_in_hold)
                self._process_batch(loop_ctx, loop_state, batch)
                should_pause = self._pause_requested(loop_ctx)
                if should_pause and loop_state.fade_pending is not None:
                    # A pause has to leave the checkpoint on a decode-batch
                    # boundary. STTN pools a whole batch, so a resume that
                    # regrouped the frames would produce different pixels
                    # from an uninterrupted run. Flushing the held tail
                    # costs it the lookahead it was waiting for -- bounded
                    # by fade_in, and only when a human pauses.
                    tail = loop_state.fade_pending
                    loop_state.fade_pending = None
                    self._process_batch(loop_ctx, loop_state, tail)
                self._checkpoint_after_batch(
                    loop_ctx, loop_state, should_pause=should_pause)
            frame_idx = loop_state.written_idx

            if frame_idx < frames_to_process:
                raise _video_decode_error(
                    input_path,
                    decoded_frames=frame_idx,
                    expected_frames=frames_to_process,
                )

            # reader.release() (or cap.release() when prefetch is off)
            # also joins the worker thread and releases the underlying cap.
            reader.release()
            reader = None
            cap = None
            if selective_cap is not None:
                selective_cap.release()
                selective_cap = None
            with self._time_stage("encode"):
                writer.release()
            writer = None

            # Encode-stage marker: every frame is inpainted and on disk. If the
            # encode/mux below is interrupted, resume reloads this and jumps
            # straight to encoding instead of redoing detection/inpainting.
            if checkpoint_active and checkpoint_root is not None and checkpoint_key:
                write_pause_checkpoint(
                    checkpoint_root,
                    checkpoint_key,
                    input_path=input_path,
                    output_path=output_path,
                    config_hash=checkpoint_config_hash,
                    frame_dir=checkpoint_frame_dir or pause_frame_dir(
                        checkpoint_root, checkpoint_key),
                    next_frame=frames_to_process,
                    total_frames=frames_to_process,
                    width=width,
                    height=height,
                    fps=fps,
                    status="running",
                    timing_manifest_path=timing_manifest_path,
                    timing=(
                        frame_timing.checkpoint_metadata(
                            start_frame, end_frame, frames_to_process)
                        if frame_timing is not None else None
                    ),
                    stage="encoding",
                    inpaint_complete=True,
                )
            if matte_reader is not None:
                matte_reader.close()
                self.last_mask_import["status"] = "composed"

            final_output_path, matte_writer = self._finalize_and_mux(
                input_path=input_path,
                output_path=output_path,
                temp_video=temp_video,
                temp_dir=temp_dir,
                fps=fps,
                start_frame=start_frame,
                end_frame=end_frame,
                width=width,
                height=height,
                use_frame_output=use_frame_output,
                frame_out_dir=frame_out_dir if use_frame_output else None,
                checkpoint_active=checkpoint_active,
                checkpoint_frame_dir=checkpoint_frame_dir,
                vfr_frame_dir=vfr_frame_dir,
                frame_timing=frame_timing,
                selected_frame_durations=selected_frame_durations,
                selected_frame_duration_ticks=selected_frame_duration_ticks,
                processed_time_start=processed_time_start,
                processed_time_end=processed_time_end,
                processed_time_start_ticks=processed_time_start_ticks,
                processed_time_end_ticks=processed_time_end_ticks,
                matte_time_base_num=matte_time_base_num,
                matte_time_base_den=matte_time_base_den,
                matte_writer=matte_writer,
            )

            if self.config.quality_report:
                self._quality_final_encode_verified = (
                    self._recompute_final_quality_evidence(
                        input_path,
                        final_output_path,
                        start_frame,
                        end_frame,
                        fps,
                    )
                )
            self._emit_quality_report(
                input_path=input_path,
                final_output_path=final_output_path,
                start_frame=start_frame,
                end_frame=end_frame,
                fps=fps,
            )

            self.last_output_path = final_output_path
            # RM-147: throughput is part of the provenance record -- a "CUDA"
            # run that quietly ran on CPU shows up here as well as in the
            # engine labels.
            self.execution_provenance.frames_processed = int(frames_to_process)
            self.execution_provenance.processing_seconds = round(sum(
                float(value)
                for key, value in self.last_stage_timings.items()
                if key in ("decode", "ocr", "mask", "inpaint", "encode")
            ), 3)
            self._write_reproducibility_sidecar(
                input_path, final_output_path,
                checkpoint_resumed=resume_frame_count > 0,
            )
            self.last_error_message = None
            self.last_error_reason = None
            if checkpoint_active and checkpoint_root is not None and checkpoint_key:
                remove_frames = (
                    checkpoint_remove_frames_on_success
                    and checkpoint_frame_dir is not None
                    and Path(final_output_path) != checkpoint_frame_dir
                )
                cleanup_pause_checkpoint(
                    checkpoint_root,
                    checkpoint_key,
                    remove_frames=remove_frames,
                )
            self._report_progress(1.0, "Complete!")
            return True

        except ProcessingPaused:
            logger.info("Video processing paused")
            raise
        except InterruptedError:
            logger.info("Video processing cancelled")
            raise
        except RequestedStageError as e:
            self._record_requested_stage_failure(e)
            logger.error("Requested video stage failed: %s", e, exc_info=True)
            return False
        except FrozenMatteError as e:
            # RM-153: the frozen matte no longer belongs to this job. Say
            # exactly what moved and ask for a re-freeze; painting approved
            # pixels onto frames they were not approved for, or silently
            # re-deriving a mask the user believes is pinned, are both worse
            # than stopping. Nothing has been decoded at this point.
            self.last_error_message = e.user_message
            self.last_error_reason = f"frozen_matte_{e.reason}"
            self.last_frozen_matte.update({
                "status": "invalid",
                "reason": e.reason,
                "error": e.user_message,
                "needs_revalidation": e.needs_revalidation,
            })
            logger.error(
                "Frozen matte rejected (%s): %s", e.reason, e.user_message)
            return False
        except MediaWriteError as e:
            # RM-139: a truncated intermediate or an unwritten frame must never
            # look like a completed job. The finally block below releases the
            # writer and the temp/partial output is never promoted.
            self.last_error_message = e.user_message
            self.last_error_reason = e.reason
            logger.error(
                "Output writer failed (%s): %s",
                e.reason,
                e.detail or e.user_message,
            )
            return False
        except MediaInputError as e:
            self.last_error_message = e.user_message
            self.last_error_reason = e.reason
            logger.warning(
                "Video input rejected (%s): %s",
                e.reason,
                e.user_message,
            )
            if e.detail:
                logger.debug("Video input rejection detail: %s", e.detail)
            return False
        except Exception as e:
            self.last_error_message = str(e)
            self.last_error_reason = "video_processing_error"
            logger.error(f"Video processing error: {e}", exc_info=True)
            return False
        finally:
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    logger.warning("Video writer release failed", exc_info=True)
                finally:
                    if self._active_writer is writer:
                        self._active_writer = None
            if matte_writer is not None:
                try:
                    matte_writer.abort()
                except Exception:
                    logger.warning("Matte writer cleanup failed", exc_info=True)
            if matte_reader is not None:
                try:
                    matte_reader.close()
                except Exception:
                    logger.warning("Matte reader cleanup failed", exc_info=True)
            if self.last_mask_export.get("status") == "pending":
                self.last_mask_export.update({
                    "status": "failed",
                    "error": (
                        self.last_error_message
                        or "Processing ended before mask export completed"
                    ),
                })
            if self.last_mask_import.get("status") == "pending":
                self.last_mask_import.update({
                    "status": "failed",
                    "error": (
                        self.last_error_message
                        or "Processing ended before matte import was validated"
                    ),
                })
            # If a prefetch reader was set up, release it (which also stops
            # the worker thread and releases the underlying cap). Otherwise
            # release the raw cap. Tolerate either being unset on early
            # failures.
            if reader is not None:
                try:
                    reader.release()
                except Exception:
                    logger.warning("Prefetch reader release failed", exc_info=True)
            elif cap is not None:
                try:
                    cap.release()
                except Exception:
                    logger.warning("Input capture release failed", exc_info=True)
            if selective_cap is not None:
                try:
                    selective_cap.release()
                except Exception:
                    logger.warning(
                        "Selective-rerun capture release failed", exc_info=True)
            # RM-283: a donor-video reference holds its own capture open for
            # the whole job; leaving it open would keep a handle on the donor
            # file after the run ends.
            try:
                self._release_clean_references()
            except Exception:
                logger.warning(
                    "Clean reference release failed", exc_info=True)
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            # RM-27: Whisper audio temp dir is created lazily inside the
            # main try block; clean it up here only if it was set.
            try:
                if whisper_audio_dir and os.path.exists(whisper_audio_dir):
                    shutil.rmtree(whisper_audio_dir, ignore_errors=True)
            except Exception:
                logger.warning("Whisper temp cleanup failed", exc_info=True)
