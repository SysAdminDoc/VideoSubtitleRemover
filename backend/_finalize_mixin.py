"""Finalize, output contract, NLE sidecar, and post-restore methods.

This mixin is mixed into SubtitleRemover so the methods retain full
self access while living in a dedicated file.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from backend.container_payload import (
    build_container_mux_args,
    build_container_mux_plan,
    probe_container_manifest,
    validate_container_payload,
)
from backend.io import (
    _cleanup_temp_output,
    _ffmpeg_subprocess_timeout,
    _path_key,
    _probe_codec_for_log,
    _probe_duration_seconds,
    _promote_temp_output,
)

logger = logging.getLogger(__name__)


def _get_output_integrity_error():
    from backend.processor import OutputIntegrityError
    return OutputIntegrityError


class _FinalizeMixin:
    """Finalize, output contract, NLE sidecar, and post-restore methods."""

    def _finalize_and_mux(self, *, input_path: str, output_path: str,
                          temp_video, temp_dir, fps: float,
                          start_frame: int, end_frame: int,
                          width: int, height: int,
                          use_frame_output: bool, frame_out_dir,
                          checkpoint_active: bool, checkpoint_frame_dir,
                          vfr_frame_dir, frame_timing,
                          selected_frame_durations,
                          processed_time_start: float,
                          processed_time_end: float, matte_writer):
        """Encode/mux stage: assemble the final output file, finalize the
        matte, and write SRT/translation/NLE sidecars plus post-restore passes.

        Extracted verbatim from ``process_video``. Returns the resolved
        ``final_output_path`` together with the (now possibly ``None``)
        ``matte_writer`` so the caller's finally-block cleanup still observes
        the correct writer state: on success the writer is finalized and
        returned as ``None``; if any step raises, the live writer propagates
        back unchanged for the finally block to abort.
        """
        self._report_progress(0.9, "Preserving container streams...")
        with self._time_stage("mux"):
            if use_frame_output:
                logger.info(
                    f"Frame-sequence output written to {frame_out_dir}"
                )
                final_output_path = frame_out_dir
            elif checkpoint_active:
                assert checkpoint_frame_dir is not None
                processed_video = self._encode_frame_sequence(
                    checkpoint_frame_dir,
                    fps,
                    output_path,
                    frame_durations=(
                        selected_frame_durations
                        if frame_timing is not None and frame_timing.is_vfr
                        else None
                    ),
                    source_time_base=(
                        frame_timing.time_base
                        if frame_timing is not None else None
                    ),
                )
                final_output_path = processed_video
                is_frame_sequence_input = Path(input_path).is_dir()
                if not is_frame_sequence_input and not Path(processed_video).is_dir():
                    final_output_path = self._merge_audio(
                        input_path,
                        processed_video,
                        output_path,
                        start_seconds=processed_time_start,
                        end_seconds=processed_time_end,
                    )
            elif vfr_frame_dir is not None:
                processed_video = self._encode_frame_sequence(
                    vfr_frame_dir,
                    fps,
                    output_path,
                    frame_durations=selected_frame_durations,
                    source_time_base=frame_timing.time_base,
                )
                final_output_path = processed_video
                final_output_path = self._merge_audio(
                    input_path,
                    processed_video,
                    output_path,
                    start_seconds=processed_time_start,
                    end_seconds=processed_time_end,
                )
            else:
                final_output_path = output_path
                is_frame_sequence_input = Path(input_path).is_dir()
                if not is_frame_sequence_input:
                    final_output_path = self._merge_audio(
                        input_path,
                        temp_video,
                        output_path,
                        start_seconds=processed_time_start,
                        end_seconds=processed_time_end,
                    )
                else:
                    final_output_path = self._reencode_or_copy(
                        temp_video, output_path)
            if matte_writer is not None:
                with self._time_stage("encode"):
                    self.last_mask_export.update(matte_writer.finalize())
                matte_writer = None
                logger.info(
                    "Lossless matte written: %s",
                    self.last_mask_export.get("path"),
                )

            if self.config.export_srt and self._srt_entries:
                srt_path = str(Path(final_output_path).with_suffix('.srt'))
                self._write_srt(
                    srt_path,
                    fps,
                    start_frame,
                    frame_timing=frame_timing,
                )

            self._prepare_translation_workflow(
                input_path,
                final_output_path,
                fps,
                start_frame,
                frame_timing=frame_timing,
            )

            # RM-78 / RM-80: optional post-restore passes (Real-ESRGAN
            # upscale, film-grain re-synthesis). Run after the main mux
            # so the user-visible output is the post-processed file;
            # each adapter degrades gracefully when its dep is missing.
            self._run_post_restore_passes(final_output_path, temp_dir)
            if not use_frame_output:
                self._validate_output_contract(final_output_path)

            # RM-76: optional NLE round-trip sidecar (EDL / FCPXML).
            self._write_nle_sidecar(input_path, final_output_path,
                                     start_frame, end_frame, fps,
                                     width=width, height=height,
                                     start_seconds=processed_time_start,
                                     end_seconds=processed_time_end)
        return final_output_path, matte_writer

    def _emit_quality_report(self, *, input_path: str, final_output_path: str,
                             start_frame: int, end_frame: int,
                             fps: float) -> None:
        """Quality report: PSNR/SSIM across a sample of unmasked regions.

        Extracted verbatim from ``process_video``; records
        ``self.last_quality_report`` and logs the metrics. Failures are
        swallowed with a warning so reporting never aborts a finished encode.
        """
        if not self.config.quality_report:
            return
        try:
            with self._time_stage("quality"):
                metrics = self._compute_quality_report(
                    input_path, final_output_path, start_frame, end_frame, fps)
            if metrics:
                self.last_quality_report = metrics
                tag_suffix = f" [{metrics['tag']}]" if metrics.get('tag') else ""
                logger.info(
                    f"Quality report: PSNR={metrics['psnr']:.2f} dB, "
                    f"SSIM={metrics['ssim']:.4f} "
                    f"({metrics['samples']} samples){tag_suffix}")
                if metrics.get('vmaf') is not None:
                    logger.info(
                        f"Quality report VMAF={metrics['vmaf']:.2f}"
                        + (
                            f", ROI VMAF={metrics['roi_vmaf']:.2f}"
                            if metrics.get('roi_vmaf') is not None
                            else ""
                        )
                    )
                if metrics.get('sheet'):
                    logger.info(f"Quality sheet: {metrics['sheet']}")
        except Exception as exc:
            logger.warning(f"Quality report failed: {exc}", exc_info=True)

    def _write_nle_sidecar(self, input_path: str, output_path: str,
                             start_frame: int, end_frame: int,
                             fps: float, width: int = 0,
                             height: int = 0,
                             start_seconds: Optional[float] = None,
                             end_seconds: Optional[float] = None) -> None:
        """RM-76: emit an EDL or FCPXML sidecar next to the output so an
        NLE operator can hand-conform the cleaned clip into a Premiere
        / DaVinci timeline at the same timecode."""
        mode = self.config.nle_sidecar
        if mode not in ("edl", "fcpxml"):
            return
        try:
            from backend import nle_sidecar
        except Exception as exc:
            logger.debug(f"NLE sidecar module unavailable: {exc}")
            return
        try:
            if fps <= 0:
                fps = 30.0
            start_s = max(
                0.0,
                float(start_seconds)
                if start_seconds is not None else start_frame / fps,
            )
            end_s = max(
                start_s + 1.0 / fps,
                float(end_seconds)
                if end_seconds is not None else end_frame / fps,
            )
            spans = getattr(self.config, "subtitle_region_spans", None)
            keyframe_tracks = getattr(
                self.config, "subtitle_region_keyframes", None)
            segments = None
            if ((spans and isinstance(spans, list))
                    or (keyframe_tracks and isinstance(keyframe_tracks, list))):
                segments = []
                for span in (spans or []) + (keyframe_tracks or []):
                    if not isinstance(span, dict):
                        continue
                    s = max(0.0, float(span.get("start", 0.0)))
                    e = float(span.get("end", 0.0))
                    if e <= 0:
                        e = end_s
                    if e > s:
                        segments.append((s, e))
                if not segments:
                    segments = None
            base = str(Path(output_path).with_suffix(""))
            if mode == "edl":
                path = nle_sidecar.write_edl(
                    base + ".edl", input_path, output_path,
                    fps, start_s, end_s,
                    segments=segments, width=width, height=height,
                )
            else:
                path = nle_sidecar.write_fcpxml(
                    base + ".fcpxml", input_path, output_path,
                    fps, start_s, end_s,
                    segments=segments, width=width, height=height,
                )
            logger.info(f"NLE {mode.upper()} sidecar written: {path}")
        except Exception as exc:
            logger.warning(f"NLE sidecar write failed: {exc}", exc_info=True)

    def _write_reproducibility_sidecar(
        self,
        input_path: str,
        output_path: str,
        *,
        checkpoint_resumed: bool = False,
    ) -> None:
        try:
            from backend.batch_report import write_output_sidecar
            from gui.config import APP_VERSION
        except Exception:
            logger.debug(
                "Reproducibility sidecar dependencies unavailable",
                exc_info=True,
            )
            return
        try:
            quality_report = self.last_quality_report
            quality_gate = None
            if isinstance(quality_report, dict):
                quality_gate = quality_report.get("quality_gate")
            write_output_sidecar(
                input_path=input_path,
                output_path=output_path,
                config=self.config,
                status="processed",
                stage_timings=self.last_stage_timings,
                detection_stats=getattr(self, "last_detection_stats", None),
                quality_report=quality_report,
                quality_gate=quality_gate,
                output_contract=self.last_output_contract,
                selective_rerun=getattr(self, "last_selective_rerun", None),
                mask_export=(
                    self.last_mask_export
                    if self.last_mask_export.get("requested") else None
                ),
                mask_import=(
                    self.last_mask_import
                    if self.last_mask_import.get("requested") else None
                ),
                translation=(
                    self.last_translation
                    if self.last_translation.get("requested") else None
                ),
                clean_reference=self._clean_reference_sidecar_evidence(),
                checkpoint_resumed=checkpoint_resumed,
                app_version=APP_VERSION,
            )
        except Exception as exc:
            logger.warning(
                "Reproducibility sidecar write failed: %s", exc,
                exc_info=True,
            )

    def _prepare_output_contract(self, input_path: str, output_path: str) -> None:
        """Probe source media once and freeze the policy used by every pass."""
        meta = None
        if not Path(input_path).is_dir():
            try:
                from backend.hdr import probe_color_metadata

                meta = probe_color_metadata(input_path)
                codec_line = _probe_codec_for_log(input_path)
                if codec_line:
                    logger.info(f"Source codec: {codec_line}")
            except Exception:
                logger.warning("Source codec/color probe failed", exc_info=True)
        if self.config.preserve_color_metadata:
            self._color_metadata = meta
        requested = getattr(self.config, "output_codec", "h264")
        effective = requested
        if self.config.preserve_color_metadata:
            try:
                from backend.hdr import hdr_safe_codec

                effective = hdr_safe_codec(requested, meta)
            except Exception:
                logger.warning("HDR codec policy failed", exc_info=True)
        if effective != requested:
            logger.info(
                f"HDR output cannot use {requested}; promoting final encode "
                f"to {effective}."
            )
            self._hdr_codec_warning_logged = True
        if self.config.use_hw_encode and effective != requested:
            self._select_hw_encoder(effective)
        from backend.output_contract import build_output_contract

        self._output_contract = build_output_contract(
            input_path=input_path,
            output_path=output_path,
            codec=effective,
            preserve_audio=self.config.preserve_audio,
            preserve_color_metadata=self.config.preserve_color_metadata,
            color_metadata=meta,
            hardware_requested=self.config.use_hw_encode,
        )
        self.last_output_contract = self._attach_d3d12_evidence(
            self._output_contract.report())
        self.last_output_contract["container_payload"] = {
            "status": "pending",
        }
        if meta is not None:
            if meta.is_hdr:
                logger.info(
                    f"HDR source detected: {meta.label} -- output contract "
                    f"requires {effective}, 10-bit pixels, and source color tags."
                )
            else:
                logger.info(f"Color signalling: {meta.label}")
        for warning in self._output_contract.warnings:
            logger.warning(warning)

    def _validate_output_contract(self, output_path: str) -> None:
        contract = getattr(self, "_output_contract", None)
        if contract is None or Path(output_path).is_dir():
            return
        ok, issues = contract.validate(output_path)
        payload_issues = list((self.last_container_payload or {}).get("issues") or [])
        if (self.last_container_payload or {}).get("status") == "failed":
            issues.extend(f"container payload: {item}" for item in payload_issues)
            ok = False
        self.last_output_contract = self._attach_d3d12_evidence(
            contract.report())
        self.last_output_contract["container_payload"] = dict(
            self.last_container_payload or {"status": "not-probed"}
        )
        self.last_output_contract["status"] = "preserved" if ok else "failed"
        self.last_output_contract["issues"] = list(issues)
        color_preserved = contract.color_preserved(issues)
        self.last_output_contract["color_preserved"] = (
            color_preserved
            if isinstance(color_preserved, bool) or color_preserved is None
            else None
        )
        if color_preserved is False:
            logger.warning(
                "Output color metadata was not preserved: %s",
                "; ".join(issues),
            )
        if not ok:
            raise _get_output_integrity_error()(
                "output contract mismatch: " + "; ".join(issues),
                {"output_contract": self.last_output_contract},
            )

    def _remux_transformed_video(
        self,
        video_source: str,
        payload_source: str,
        output_path: str,
    ) -> dict:
        """Copy a transformed primary video while restoring container payload."""
        manifest = probe_container_manifest(payload_source)
        plan = build_container_mux_plan(
            manifest,
            output_path,
            preserve_audio=self.config.preserve_audio,
            multi_audio=True,
            loudnorm_target=0.0,
        )
        for warning in plan.get("warnings", []):
            logger.warning(warning)
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats",
            "-i", video_source,
            "-i", payload_source,
            "-map", "0:v:0",
            "-c:v", "copy",
        ]
        cmd += build_container_mux_args(plan, input_index=1)
        duration = _probe_duration_seconds(video_source)
        if duration > 0:
            cmd += ["-t", f"{duration:.9f}"]
        cmd += [output_path]
        self._run_checked_ffmpeg(
            cmd,
            _ffmpeg_subprocess_timeout(duration or _probe_duration_seconds(payload_source)),
        )
        report = validate_container_payload(plan, output_path)
        self.last_container_payload = report
        if report.get("issues"):
            raise _get_output_integrity_error()(
                "container payload mismatch: " + "; ".join(report["issues"]),
                {"container_payload": report},
            )
        return report

    def _mark_container_payload_failed(self, reason: str) -> None:
        report = dict(self.last_container_payload or {})
        issues = list(report.get("issues") or [])
        if reason not in issues:
            issues.append(reason)
        warnings = list(report.get("warnings") or [])
        if reason not in warnings:
            warnings.append(reason)
        report.update({
            "schema": report.get("schema", "vsr.container_payload.v1"),
            "status": "failed",
            "issues": issues,
            "warnings": warnings,
        })
        self.last_container_payload = report

    def _promote_post_restore_result(
        self,
        produced: str,
        output_path: str,
        temp_dir: str,
        label: str,
    ) -> bool:
        """Normalize an adapter result before it can replace the final output."""
        contract = getattr(self, "_output_contract", None)
        if contract is None:
            _promote_temp_output(produced, output_path)
            return True
        if _path_key(produced) == _path_key(output_path):
            return False
        previous_payload = dict(self.last_container_payload)
        normalized = contract.temp_path(temp_dir, f"{label}-contract")
        ok = False
        issues: list[str] = []
        try:
            self._remux_transformed_video(produced, output_path, normalized)
            ok, issues = contract.validate(normalized)
        except Exception as exc:
            logger.warning(
                "%s output-contract normalization failed: %s", label, exc
            )
            issues = [str(exc)]
        if not ok:
            self.last_container_payload = previous_payload
            logger.warning(
                "%s result was not promoted because it violates the output "
                "contract: %s",
                label,
                "; ".join(issues),
            )
            _cleanup_temp_output(normalized)
            _cleanup_temp_output(produced)
            return False
        _promote_temp_output(normalized, output_path)
        _cleanup_temp_output(produced)
        return True

    def _run_post_restore_passes(self, output_path: str, temp_dir: str) -> None:
        """RM-78 / RM-80: run optional post-restore passes against the
        finalised output in place. Each adapter is a no-op when its
        dep is missing; the original output is preserved on every
        failure path so users always have a result.
        """
        contract = getattr(self, "_output_contract", None)

        def post_path(stem: str) -> str:
            if contract is not None:
                return contract.temp_path(temp_dir, stem)
            return os.path.join(temp_dir, f"{stem}{Path(output_path).suffix or '.mp4'}")

        if self.config.upscale_factor in (2, 3, 4):
            try:
                from backend.post_restore import realesrgan_upscale
                upscaled = post_path("upscaled")
                produced = realesrgan_upscale(
                    output_path, upscaled,
                    scale=int(self.config.upscale_factor),
                )
                if produced and Path(produced).is_file():
                    if self._promote_post_restore_result(
                        produced, output_path, temp_dir, "realesrgan"
                    ):
                        logger.info(
                            f"Real-ESRGAN x{self.config.upscale_factor} pass complete"
                        )
            except Exception as exc:
                logger.warning(f"Real-ESRGAN pass failed: {exc}", exc_info=True)
        if self.config.swinir_restore:
            try:
                from backend.post_restore import swinir_restore
                restored = post_path("swinir")
                produced = swinir_restore(output_path, restored)
                if produced and Path(produced).is_file():
                    if self._promote_post_restore_result(
                        produced, output_path, temp_dir, "swinir"
                    ):
                        logger.info("SwinIR restoration pass complete")
            except Exception as exc:
                logger.warning(f"SwinIR pass failed: {exc}", exc_info=True)
        if self.config.seedvr2_restore:
            try:
                from backend.post_restore import seedvr2_restore
                restored = post_path("seedvr2")
                produced = seedvr2_restore(output_path, restored)
                if produced and Path(produced).is_file():
                    if self._promote_post_restore_result(
                        produced, output_path, temp_dir, "seedvr2"
                    ):
                        logger.info("SeedVR2 restoration pass complete")
            except Exception as exc:
                logger.warning(f"SeedVR2 pass failed: {exc}", exc_info=True)
        if self.config.film_grain_strength > 0.0:
            if self._uses_native_av1_film_grain():
                logger.info(
                    "SVT-AV1 native film grain was enabled during encode; "
                    "skipping additive post-encode grain pass."
                )
            else:
                try:
                    from backend.post_restore import add_film_grain
                    grain_out = post_path("grainy")
                    produced = add_film_grain(
                        output_path, grain_out,
                        strength=self.config.film_grain_strength,
                        video_encode_args=self._get_encode_args(allow_d3d12=False),
                        preserve_audio=self.config.preserve_audio,
                    )
                    if produced and Path(produced).is_file():
                        if self._promote_post_restore_result(
                            produced, output_path, temp_dir, "film-grain"
                        ):
                            logger.info(
                                f"Film-grain pass complete "
                                f"(strength={self.config.film_grain_strength:.3f})"
                            )
                except Exception as exc:
                    logger.warning(f"Film-grain pass failed: {exc}", exc_info=True)
        translation_path = str(getattr(self, "_translation_burn_path", "") or "")
        subtitle_path = translation_path or self.config.restyle_subtitle
        if subtitle_path:
            translation_requested = bool(translation_path)
            try:
                from backend.post_restore import burn_subtitles
                restyle_out = post_path("restyled")
                produced = burn_subtitles(
                    output_path, restyle_out,
                    subtitle_path=subtitle_path,
                    style_override=(
                        self.config.translation_style
                        if translation_requested else self.config.restyle_style
                    ),
                    video_encode_args=self._get_encode_args(allow_d3d12=False),
                    preserve_audio=self.config.preserve_audio,
                )
                promoted = False
                if produced and Path(produced).is_file():
                    promoted = self._promote_post_restore_result(
                        produced,
                        output_path,
                        temp_dir,
                        "translation" if translation_requested else "restyle",
                    )
                    if promoted:
                        logger.info(
                            "%s subtitle burn pass complete",
                            "Translated" if translation_requested else "Restyle",
                        )
                if translation_requested and not promoted:
                    raise RuntimeError(
                        "translated subtitle re-embedding produced no valid output")
                if translation_requested:
                    self.last_translation["status"] = "embedded"
            except Exception as exc:
                if translation_requested:
                    self.last_translation["status"] = "failed"
                    self.last_translation["error"] = str(exc)
                    raise
                logger.warning(f"Restyle pass failed: {exc}", exc_info=True)
        if self.config.watermark_image:
            try:
                from backend.post_restore import burn_watermark
                wm_out = post_path("watermarked")
                produced = burn_watermark(
                    output_path, wm_out,
                    watermark_path=self.config.watermark_image,
                    position=self.config.watermark_position,
                    opacity=self.config.watermark_opacity,
                    margin=self.config.watermark_margin,
                    video_encode_args=self._get_encode_args(allow_d3d12=False),
                    preserve_audio=self.config.preserve_audio,
                )
                if produced and Path(produced).is_file():
                    if self._promote_post_restore_result(
                        produced, output_path, temp_dir, "watermark"
                    ):
                        logger.info("Watermark burn pass complete")
            except Exception as exc:
                logger.warning(f"Watermark burn failed: {exc}", exc_info=True)

