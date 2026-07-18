"""Encode, mux, and audio-merge methods for SubtitleRemover.

This mixin is mixed into ``SubtitleRemover`` so the methods retain full
``self`` access while living in a dedicated file. It covers everything
from FFmpeg argument construction through container-payload merge and
the multi-tier hardware-encoder fallback chain.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import List, Optional

import numpy as np

from backend.container_payload import (
    build_container_mux_args,
    build_container_mux_plan,
    probe_container_manifest,
    validate_container_payload,
)
from backend.io import (
    _cleanup_temp_output,
    _copy_file_atomic,
    _ensure_output_parent,
    _ffmpeg_subprocess_timeout,
    _probe_duration_seconds,
    _promote_temp_output,
    _write_text_atomic,
    validate_video_output,
)
from backend.work_directory import StorageRequirement, assess_storage_volumes

logger = logging.getLogger(__name__)


def _frame_seconds(index: int, fps: float,
                   timing=None) -> float:
    if timing is not None:
        pts = getattr(timing, "frame_pts", None)
        if pts and 0 <= index < len(pts):
            return float(pts[index])
    return float(index) / max(float(fps), 0.001)


class _EncodeMixin:
    """Encode, mux, and audio-merge methods extracted from SubtitleRemover."""

    def _uses_native_av1_film_grain(self) -> bool:
        return bool(
            self.config.film_grain_strength > 0.0
            and self._effective_output_codec() == "av1"
            and not (self._hw_encoder and self.config.use_hw_encode)
        )

    def _svt_av1_film_grain_args(self) -> List[str]:
        if not self._uses_native_av1_film_grain():
            return []
        grain = max(
            1,
            min(50, int(round(float(self.config.film_grain_strength) * 255))),
        )
        return ["-svtav1-params", f"film-grain={grain}"]

    def _get_encode_args(self, *, allow_d3d12: bool = True) -> List[str]:
        """Return FFmpeg video encoder arguments, preferring hardware encoding."""
        codec = self._effective_output_codec()
        hdr_mode = self._source_is_hdr()
        static_hdr = bool(
            hdr_mode
            and getattr(self._color_metadata, "mastering_display", "")
        )
        encoder_prefix = {
            "h264": "h264_",
            "h265": "hevc_",
            "av1": "av1_",
            "vvc": "vvc_",
        }.get(codec, "")
        hardware_matches = bool(
            self._hw_encoder
            and encoder_prefix
            and self._hw_encoder.startswith(encoder_prefix)
            and (allow_d3d12 or not self._hw_encoder.endswith("_d3d12va"))
        )
        if static_hdr and self._hw_encoder and self.config.use_hw_encode:
            if not getattr(self, "_hdr_software_warning_logged", False):
                logger.info(
                    "Static HDR mastering metadata requires the software "
                    "encoder so the final bitstream can reproduce its SEI."
                )
                self._hdr_software_warning_logged = True
        if hardware_matches and self.config.use_hw_encode and not static_hdr:
            if self._hw_encoder.endswith("_d3d12va"):
                base = [
                    "-vf", "format=nv12,hwupload,scale_d3d12=w=iw:h=ih",
                    "-c:v", self._hw_encoder,
                    "-bf", "0", "-async_depth", "1",
                    "-rc_mode", "CQP", "-qp", str(self.config.output_quality),
                ]
            elif 'nvenc' in self._hw_encoder:
                base = ['-c:v', self._hw_encoder, '-preset', 'p4',
                        '-cq', str(self.config.output_quality)]
            elif 'qsv' in self._hw_encoder:
                base = ['-c:v', self._hw_encoder,
                        '-global_quality', str(self.config.output_quality)]
            elif 'amf' in self._hw_encoder:
                base = ['-c:v', self._hw_encoder,
                        '-quality', 'balanced',
                        '-rc', 'cqp', '-qp', str(self.config.output_quality)]
            else:
                base = ['-c:v', 'libx264', '-crf', str(self.config.output_quality),
                        '-preset', 'medium']
            return (
                base
                + self._hdr_pixel_format_args(codec, hardware=True)
                + self._hdr_encode_args()
            )
        if codec == "h265":
            base = ['-c:v', 'libx265', '-crf', str(self.config.output_quality),
                    '-preset', 'medium']
        elif codec == "av1":
            crf = min(63, self.config.output_quality)
            base = ['-c:v', 'libsvtav1', '-crf', str(crf), '-preset', '8']
        elif codec == "vvc":
            base = ['-c:v', 'libvvenc', '-qp', str(self.config.output_quality),
                    '-preset', 'medium']
        else:
            base = ['-c:v', 'libx264', '-crf', str(self.config.output_quality),
                    '-preset', 'medium']
        return (
            base
            + self._hdr_pixel_format_args(codec)
            + self._encoder_private_args(codec)
            + self._hdr_encode_args()
        )

    def _source_is_hdr(self) -> bool:
        meta = getattr(self, "_color_metadata", None)
        return bool(
            getattr(self.config, "preserve_color_metadata", True)
            and meta is not None
            and getattr(meta, "is_hdr", False)
        )

    def _effective_output_codec(self) -> str:
        contract = getattr(self, "_output_contract", None)
        if contract is not None:
            return contract.codec
        requested = getattr(self.config, "output_codec", "h264")
        if not getattr(self.config, "preserve_color_metadata", True):
            return requested
        try:
            from backend.hdr import hdr_safe_codec
            codec = hdr_safe_codec(requested, getattr(self, "_color_metadata", None))
        except Exception:
            logger.warning("HDR codec policy failed", exc_info=True)
            return requested
        if codec != requested and not getattr(self, "_hdr_codec_warning_logged", False):
            logger.info(
                f"HDR output cannot use {requested}; promoting final encode to {codec}."
            )
            self._hdr_codec_warning_logged = True
        return codec

    def _hdr_pixel_format_args(self, codec: str, *, hardware: bool = False) -> List[str]:
        if not self.config.preserve_color_metadata or self._color_metadata is None:
            return []
        try:
            from backend.hdr import hdr_pixel_format_args
            return hdr_pixel_format_args(
                self._color_metadata,
                codec,
                hardware=hardware,
            )
        except Exception:
            logger.warning("HDR pixel-format argument generation failed", exc_info=True)
            return []

    def _hdr_encoder_private_args(self, codec: str) -> List[str]:
        if not self.config.preserve_color_metadata or self._color_metadata is None:
            return []
        try:
            from backend.hdr import hdr_encoder_private_args
            return hdr_encoder_private_args(self._color_metadata, codec)
        except Exception:
            logger.warning("HDR encoder-private argument generation failed", exc_info=True)
            return []

    def _encoder_private_args(self, codec: str) -> List[str]:
        hdr_args = self._hdr_encoder_private_args(codec)
        grain_args = self._svt_av1_film_grain_args() if codec == "av1" else []
        if not hdr_args:
            return grain_args
        if not grain_args:
            return hdr_args
        if hdr_args[0] == grain_args[0] == "-svtav1-params":
            return ["-svtav1-params", f"{hdr_args[1]}:{grain_args[1]}"]
        return hdr_args + grain_args

    def _hdr_encode_args(self) -> List[str]:
        """RM-73 partial: re-tag the output with source color signalling."""
        if not self.config.preserve_color_metadata or self._color_metadata is None:
            return []
        try:
            from backend.hdr import hdr_encode_args
            return hdr_encode_args(self._color_metadata)
        except Exception:
            logger.warning("HDR encode argument generation failed", exc_info=True)
            return []

    def _check_encode_disk_space(self, output_path: str, *, width: int,
                                 height: int, frames: int,
                                 high_bit: bool,
                                 checkpoint_dir: Optional[Path] = None) -> None:
        """Estimate and probe every volume affected by this processing run."""
        if frames <= 0 or width <= 0 or height <= 0:
            return
        bytes_per_pixel = 6 if high_bit else 3
        raw = int(width) * int(height) * bytes_per_pixel * int(frames)
        work_resolution = self._resolve_work_directory()
        work_bytes = int(raw * (0.10 if checkpoint_dir else 0.45))
        output_bytes = int(raw * 0.05)
        requirements = [
            StorageRequirement(
                work_resolution.path,
                work_bytes + 256 * 1024 * 1024,
                "temporary processing files",
            ),
            StorageRequirement(
                Path(output_path).parent,
                output_bytes + 64 * 1024 * 1024,
                "final output",
            ),
        ]
        if checkpoint_dir is not None:
            requirements.append(StorageRequirement(
                Path(checkpoint_dir),
                int(raw * 0.35) + 64 * 1024 * 1024,
                "checkpoint and resume frames",
            ))
        try:
            statuses = assess_storage_volumes(requirements)
        except OSError:
            return
        for status in statuses:
            hard_floor = max(
                64 * 1024 * 1024,
                int(status.required_bytes * 0.25),
            )
            purposes = ", ".join(status.purposes)
            if status.free_bytes < hard_floor:
                raise ValueError(
                    f"Insufficient disk space at '{status.path}' for {purposes}: "
                    f"about {status.required_bytes // (1024*1024)} MB is "
                    f"estimated but only {status.free_bytes // (1024*1024)} MB "
                    "is free. Choose a work/output folder on a larger drive."
                )
            if status.free_bytes < status.required_bytes:
                logger.warning(
                    "Low disk space at '%s' for %s: ~%d MB estimated, %d MB "
                    "free. Processing will continue but may fail if the "
                    "estimate is high.",
                    status.path,
                    purposes,
                    status.required_bytes // (1024 * 1024),
                    status.free_bytes // (1024 * 1024),
                )

    def _promote_video_output(
        self,
        produced,
        output_path: str,
        *,
        reference=None,
        expected_frames: Optional[int] = None,
        expected_duration: Optional[float] = None,
    ) -> None:
        """Validate a finished video, then atomically replace the destination."""
        ok, reason, details = validate_video_output(
            produced,
            reference=reference,
            expected_frames=expected_frames,
            expected_duration=expected_duration,
        )
        if not ok:
            logger.error(
                "Output integrity check failed for '%s': %s", output_path, reason
            )
            from backend.processor import OutputIntegrityError
            raise OutputIntegrityError(reason, details)
        _promote_temp_output(produced, output_path)

    def _salvage_intermediate(self, source: str, output: str) -> str:
        """Copy the lossless intermediate to a container-correct path."""
        src_ext = Path(source).suffix.lower()
        out_ext = Path(output).suffix.lower()
        if src_ext == out_ext:
            _copy_file_atomic(source, output)
            return output
        salvage = str(Path(output).with_suffix(src_ext))
        _copy_file_atomic(source, salvage)
        logger.warning(
            f"Encoding to '{output}' was not possible; saved the "
            f"unencoded intermediate as '{salvage}' instead, because "
            f"its stream lives in a '{src_ext}' container and renaming "
            f"it to '{out_ext}' would produce a broken file."
        )
        return salvage

    def _encode_frame_sequence(
        self,
        frame_dir: Path,
        fps: float,
        output: str,
        *,
        frame_durations: Optional[List[float]] = None,
        source_time_base: Optional[float] = None,
    ) -> str:
        """Encode checkpoint/output frames into the requested video path."""
        frame_dir = Path(frame_dir)
        pattern = frame_dir / "frame_%06d.png"
        if not (frame_dir / "frame_000000.png").is_file():
            raise ValueError(f"No checkpoint frames found in {frame_dir}")
        temp_output = self._allocate_work_output(output)
        concat_path: Optional[Path] = None
        try:
            _ensure_output_parent(output)
            frame_total = len(list(frame_dir.glob("frame_*.png")))
            use_vfr = bool(
                frame_durations
                and len(frame_durations) >= frame_total
                and frame_total > 0
            )
            if use_vfr:
                normalized_durations = []
                fallback = 1.0 / max(float(fps), 1.0)
                for value in frame_durations[:frame_total]:
                    try:
                        duration = float(value)
                    except (TypeError, ValueError):
                        duration = fallback
                    if not np.isfinite(duration) or duration <= 0:
                        duration = fallback
                    normalized_durations.append(duration)
                concat_path = frame_dir / ".vsr-vfr.ffconcat"
                concat_lines = ["ffconcat version 1.0"]
                try:
                    timing_tick = float(source_time_base or 0.0)
                except (TypeError, ValueError):
                    timing_tick = 0.0
                if not np.isfinite(timing_tick) or timing_tick <= 0:
                    timing_tick = min(normalized_durations) / 100.0
                concat_rate = max(
                    1000,
                    min(1_000_000, int(round(1.0 / max(timing_tick, 1e-6)))),
                )
                for index, duration in enumerate(normalized_durations):
                    concat_lines.append(f"file frame_{index:06d}.png")
                    concat_lines.append(f"option framerate {concat_rate}")
                    concat_lines.append(f"duration {duration:.9f}")
                _write_text_atomic(
                    concat_path, "\n".join(concat_lines) + "\n")
                cmd = [
                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-nostats',
                ] + self._d3d12_device_args() + [
                    '-f', 'concat', '-safe', '0',
                    '-i', str(concat_path),
                ]
            else:
                normalized_durations = []
                cmd = [
                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-nostats',
                ] + self._d3d12_device_args() + [
                    '-framerate', f"{float(fps):.6f}",
                    '-i', str(pattern),
                ]
            cmd += self._get_encode_args()
            if use_vfr:
                cmd += [
                    '-fps_mode:v', 'vfr',
                    '-enc_time_base:v', 'demux',
                ]
            cmd += ['-an', str(temp_output)]
            timeout = _ffmpeg_subprocess_timeout(
                max(
                    1.0,
                    sum(normalized_durations)
                    if normalized_durations
                    else _frame_seconds(frame_total, fps),
                )
            )
            self._run_checked_ffmpeg(cmd, timeout)
            self._promote_video_output(
                temp_output,
                output,
                expected_frames=frame_total,
                expected_duration=(
                    sum(normalized_durations)
                    if normalized_durations else None
                ),
            )
            return output
        except FileNotFoundError:
            logger.warning(
                "FFmpeg not found; leaving processed checkpoint frames as output"
            )
            return str(frame_dir)
        except (subprocess.CalledProcessError,) as exc:
            if self._fallback_after_hw_failure(exc):
                logger.warning(
                    "Hardware encoder failed, retrying with %s: %s",
                    self._hw_encoder or "software",
                    exc,
                )
                return self._encode_frame_sequence(
                    frame_dir,
                    fps,
                    output,
                    frame_durations=frame_durations,
                    source_time_base=source_time_base,
                )
            raise
        except Exception as exc:
            from backend.processor import OutputIntegrityError
            if isinstance(exc, OutputIntegrityError):
                if self._fallback_after_hw_failure(exc):
                    logger.warning(
                        "Hardware encoder failed, retrying with %s: %s",
                        self._hw_encoder or "software",
                        exc,
                    )
                    return self._encode_frame_sequence(
                        frame_dir,
                        fps,
                        output,
                        frame_durations=frame_durations,
                        source_time_base=source_time_base,
                    )
            raise
        finally:
            _cleanup_temp_output(temp_output)
            _cleanup_temp_output(concat_path)

    def _reencode_or_copy(self, source: str, output: str) -> str:
        """Re-encode with preferred encoder, or salvage the intermediate
        if FFmpeg is unavailable or keeps failing."""
        temp_output = self._allocate_work_output(output)
        try:
            _ensure_output_parent(output)
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-nostats',
            ] + self._d3d12_device_args() + ['-i', source]
            cmd += self._get_encode_args()
            cmd += ['-an', str(temp_output)]
            timeout = _ffmpeg_subprocess_timeout(_probe_duration_seconds(source))
            self._run_checked_ffmpeg(cmd, timeout)
            self._promote_video_output(temp_output, output, reference=source)
            return output
        except Exception as exc:
            from backend.processor import OutputIntegrityError
            if isinstance(exc, OutputIntegrityError):
                if self._fallback_after_hw_failure(exc):
                    logger.warning(
                        "Hardware encoder output failed validation, retrying with %s: %s",
                        self._hw_encoder or "software",
                        exc.reason,
                    )
                    return self._reencode_or_copy(source, output)
                logger.warning(
                    "Re-encode failed integrity check (%s); salvaging intermediate.",
                    exc.reason,
                )
                return self._salvage_intermediate(source, output)
            if isinstance(exc, subprocess.CalledProcessError):
                if self._fallback_after_hw_failure(exc):
                    logger.warning(
                        "Hardware encoder failed, retrying with %s: %s",
                        self._hw_encoder or "software",
                        exc,
                    )
                    return self._reencode_or_copy(source, output)
                return self._salvage_intermediate(source, output)
            logger.warning(
                f"FFmpeg re-encode failed; salvaging intermediate: {exc}",
                exc_info=True,
            )
            return self._salvage_intermediate(source, output)
        finally:
            _cleanup_temp_output(temp_output)

    def _merge_audio(
        self,
        original: str,
        processed: str,
        output: str,
        *,
        start_seconds: Optional[float] = None,
        end_seconds: Optional[float] = None,
        _include_auxiliary: bool = True,
        _force_audio_transcode: bool = False,
    ) -> str:
        from backend.processor import OutputIntegrityError
        temp_output = self._allocate_work_output(output)
        plan: dict = {}
        try:
            _ensure_output_parent(output)
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-nostats',
            ] + self._d3d12_device_args() + [
                '-i', processed,
            ]
            audio_start = (
                max(0.0, float(start_seconds))
                if start_seconds is not None else
                max(0.0, float(self.config.time_start or 0.0))
            )
            audio_end = (
                max(audio_start, float(end_seconds))
                if end_seconds is not None else
                max(0.0, float(self.config.time_end or 0.0))
            )
            if audio_start > 0:
                cmd += ['-ss', f'{audio_start:.9f}']
            cmd += ['-i', original]
            manifest = probe_container_manifest(original)
            target = self.config.loudnorm_target
            plan = build_container_mux_plan(
                manifest,
                output,
                preserve_audio=self.config.preserve_audio,
                multi_audio=self.config.multi_audio_passthrough,
                loudnorm_target=target,
                include_auxiliary=_include_auxiliary,
                force_audio_transcode=_force_audio_transcode,
                start_seconds=audio_start,
                end_seconds=audio_end,
            )
            self.last_container_payload = plan
            for warning in plan.get("warnings", []):
                logger.warning(warning)
            cmd += ['-map', '0:v:0']
            cmd += self._get_encode_args()
            cmd += build_container_mux_args(
                plan,
                input_index=1,
                loudnorm_target=target,
            )
            output_duration = (
                audio_end - audio_start
                if audio_end > audio_start
                else _probe_duration_seconds(processed)
            )
            if output_duration > 0:
                cmd += ['-t', f'{output_duration:.9f}']
            else:
                cmd += ['-shortest']
            cmd += [str(temp_output)]
            timeout = _ffmpeg_subprocess_timeout(_probe_duration_seconds(original))
            self._run_checked_ffmpeg(cmd, timeout)
            report = validate_container_payload(plan, temp_output)
            self.last_container_payload = report
            if report.get("issues"):
                raise OutputIntegrityError(
                    "container payload mismatch: " + "; ".join(report["issues"]),
                    {"container_payload": report},
                )
            try:
                self._promote_video_output(
                    temp_output, output, reference=processed
                )
            except OutputIntegrityError as exc:
                if self._fallback_after_hw_failure(exc):
                    logger.warning(
                        "Hardware encoder output failed validation, retrying "
                        "container merge with %s: %s",
                        self._hw_encoder or "software",
                        exc.reason,
                    )
                    return self._merge_audio(
                        original,
                        processed,
                        output,
                        start_seconds=start_seconds,
                        end_seconds=end_seconds,
                        _include_auxiliary=_include_auxiliary,
                        _force_audio_transcode=_force_audio_transcode,
                    )
                logger.warning(
                    "Audio merge produced a truncated/invalid output (%s); "
                    "saving the full-length video without audio instead.",
                    exc.reason,
                )
                self._mark_container_payload_failed(
                    "Container payload was not promoted because the muxed video "
                    f"failed integrity validation: {exc.reason}"
                )
                return self._salvage_intermediate(processed, output)
            encoder_name = (
                cmd[cmd.index('-c:v') + 1]
                if '-c:v' in cmd else 'unknown'
            )
            logger.info(
                f"Container payload merged successfully (encoder: {encoder_name})"
            )
            return output
        except OutputIntegrityError as exc:
            payload_failure = "container_payload" in exc.details
            mapped_auxiliary = any(
                item.get("type") in {"subtitle", "attachment", "data", "video"}
                and item.get("action") in {"copy", "transcode"}
                for item in plan.get("streams", [])
            )
            copied_audio = any(
                item.get("type") == "audio" and item.get("action") == "copy"
                for item in plan.get("streams", [])
            )
            if payload_failure and _include_auxiliary and mapped_auxiliary:
                logger.warning(
                    "Full container preservation failed (%s); retrying with "
                    "audio and metadata only.", exc.reason,
                )
                return self._merge_audio(
                    original,
                    processed,
                    output,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    _include_auxiliary=False,
                    _force_audio_transcode=_force_audio_transcode,
                )
            if payload_failure and copied_audio and not _force_audio_transcode:
                logger.warning(
                    "Audio stream copy failed validation (%s); retrying with "
                    "container-compatible audio encoding.", exc.reason,
                )
                return self._merge_audio(
                    original,
                    processed,
                    output,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    _include_auxiliary=False,
                    _force_audio_transcode=True,
                )
            logger.warning(
                "Container preservation failed integrity checks (%s); "
                "saving the processed video without source payload.",
                exc.reason,
            )
            self._mark_container_payload_failed(
                f"Source container payload could not be preserved: {exc.reason}"
            )
            return self._salvage_intermediate(processed, output)
        except subprocess.TimeoutExpired:
            logger.warning("FFmpeg audio merge timed out, saving video without audio")
            self._mark_container_payload_failed(
                "Source container payload was omitted after the mux timed out."
            )
            return self._salvage_intermediate(processed, output)
        except subprocess.CalledProcessError as e:
            if self._hw_encoder and self._hw_encoder in cmd:
                self._fallback_after_hw_failure(e)
                logger.warning(
                    "Hardware encoder failed, retrying with %s: %s",
                    self._hw_encoder or "software",
                    e,
                )
                return self._merge_audio(
                    original,
                    processed,
                    output,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    _include_auxiliary=_include_auxiliary,
                    _force_audio_transcode=_force_audio_transcode,
                )
            mapped_auxiliary = any(
                item.get("type") in {"subtitle", "attachment", "data", "video"}
                and item.get("action") in {"copy", "transcode"}
                for item in plan.get("streams", [])
            )
            copied_audio = any(
                item.get("type") == "audio" and item.get("action") == "copy"
                for item in plan.get("streams", [])
            )
            if _include_auxiliary and mapped_auxiliary:
                logger.warning(
                    "Auxiliary stream mux failed; retrying with audio and metadata only."
                )
                return self._merge_audio(
                    original,
                    processed,
                    output,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    _include_auxiliary=False,
                    _force_audio_transcode=_force_audio_transcode,
                )
            if copied_audio and not _force_audio_transcode:
                logger.warning(
                    "Audio stream copy failed; retrying with a compatible audio codec."
                )
                return self._merge_audio(
                    original,
                    processed,
                    output,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    _include_auxiliary=False,
                    _force_audio_transcode=True,
                )
            logger.warning(f"Audio merge failed: {e}, encoding video without audio")
            self._mark_container_payload_failed(
                "Source container payload was omitted after the mux command failed."
            )
            return self._reencode_or_copy(processed, output)
        except FileNotFoundError:
            logger.warning("FFmpeg not found, saving video without audio")
            self._mark_container_payload_failed(
                "Source container payload was omitted because FFmpeg is unavailable."
            )
            return self._salvage_intermediate(processed, output)
        finally:
            _cleanup_temp_output(temp_output)
