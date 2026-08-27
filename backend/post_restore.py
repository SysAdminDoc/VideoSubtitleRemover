"""Optional post-restore passes that run after the inpaint stage.

RM-78 Real-ESRGAN super-resolution and RM-80 film-grain re-synthesis
shape the output video after inpainting. Shared finishing also restores
local grain inside masked fills before this full-frame pass. Each
adapter here:

- Imports lazily so the rest of the codebase keeps working when the
  optional dependency / weight file is missing.
- Operates on a finished video file (the FFV1 intermediate or final
  output, depending on where the caller wires it in).
- Returns the path to a validated new file on success. A requested
  adapter that cannot run raises a classified error with recovery
  guidance instead of silently retaining the old output.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence


from backend.import_safety import module_can_import
from backend.execution_provenance import (
    FAILURE_DEPENDENCY_MISSING,
    FAILURE_OUTPUT_MISSING,
    FAILURE_POLICY_BLOCKED,
    FAILURE_RUNTIME,
    RequestedStageError,
)
from backend.subprocess_policy import run_process

logger = logging.getLogger(__name__)

# ASS force_style is a comma-separated list of Name=Value pairs. Legitimate
# values use letters, digits, '&H..' colour literals, '.', '-', '+', '#',
# spaces, '=' and ','. Anything else (notably single quotes, backslashes,
# ':', ';', '[', ']') could break out of the quoted filtergraph value, so a
# style string containing such characters is rejected rather than escaped.
_FORCE_STYLE_ALLOWED = re.compile(r"^[A-Za-z0-9 ,=&#.+\-]*$")


def _sanitize_force_style(style: str) -> str:
    """Return the style string if it is a safe ASS force_style value, else ''."""
    if not style:
        return ""
    return style if _FORCE_STYLE_ALLOWED.match(style) else ""


def realesrgan_upscale(input_path: str, output_path: str,
                       scale: int = 2,
                       model_name: str = "RealESRGAN_x4plus") -> str:
    """RM-78: 2x or 4x upscale via Real-ESRGAN.

    Tries the `realesrgan-ncnn-vulkan` standalone binary. If it is not
    available, the error reports whether the `realesrgan` Python package
    is present. Returns the output path on success and raises
    RequestedStageError when the requested pass cannot run.
    """
    if shutil.which("realesrgan-ncnn-vulkan"):
        try:
            cmd = [
                "realesrgan-ncnn-vulkan",
                "-i", input_path,
                "-o", output_path,
                "-s", str(scale),
                "-n", model_name,
            ]
            result = run_process(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode == 0 and Path(output_path).is_file():
                logger.info(f"Real-ESRGAN upscaled to {output_path}")
                return output_path
            raise RequestedStageError(
                stage="restoration",
                requested_implementation="realesrgan",
                actual_implementation="realesrgan",
                provider="realesrgan-ncnn-vulkan",
                failure_class=FAILURE_OUTPUT_MISSING,
                detail=(
                    f"realesrgan-ncnn-vulkan exited {result.returncode} "
                    "without a valid output"
                ),
                recovery_hint=(
                    "Verify the Real-ESRGAN binary and selected model, then "
                    "retry or disable upscaling."
                ),
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            raise RequestedStageError(
                stage="restoration",
                requested_implementation="realesrgan",
                actual_implementation="realesrgan",
                provider="realesrgan-ncnn-vulkan",
                failure_class=FAILURE_RUNTIME,
                detail=type(exc).__name__,
                recovery_hint=(
                    "Verify the Real-ESRGAN binary can run, then retry or "
                    "disable upscaling."
                ),
                cause=exc,
                retriable=isinstance(exc, subprocess.TimeoutExpired),
            ) from exc
    python_present = module_can_import(
        "realesrgan",
        logger=logger,
        failure_context="Real-ESRGAN Python package probe failed",
    )
    raise RequestedStageError(
        stage="restoration",
        requested_implementation="realesrgan",
        failure_class=FAILURE_DEPENDENCY_MISSING,
        detail=(
            "the Python package is installed but this video adapter requires "
            "realesrgan-ncnn-vulkan"
            if python_present else "realesrgan-ncnn-vulkan was not found on PATH"
        ),
        recovery_hint=(
            "Install realesrgan-ncnn-vulkan from the upstream release and add "
            "it to PATH, or disable upscaling."
        )
    )


def seedvr2_restore(input_path: str, output_path: str,
                     adapter: str = "seedvr2") -> str:
    """RM-77 SeedVR2 one-step video restoration.

    SeedVR2 ships as a 16B-param diffusion transformer with adversarial
    post-training -- single sampling step. Best-in-class quality on
    heavy-degradation footage but the install footprint is large (the
    user is expected to clone IceClear/SeedVR2 separately and either
    set `VSR_SEEDVR2_CMD` to the CLI entrypoint or install a
    pip-published wrapper named `seedvr2`).

    Returns the path on success or raises a classified stage error.
    """
    cmd_env = os.environ.get("VSR_SEEDVR2_CMD", "")
    if cmd_env:
        try:
            import shlex
            cmd = shlex.split(cmd_env) + ["-i", input_path, "-o", output_path]
            result = run_process(cmd, capture_output=True, text=True, timeout=14400)
            if result.returncode == 0 and Path(output_path).is_file():
                logger.info(f"SeedVR2 restoration complete via {cmd_env}")
                return output_path
            raise RequestedStageError(
                stage="restoration",
                requested_implementation="seedvr2",
                actual_implementation="seedvr2",
                provider="VSR_SEEDVR2_CMD",
                failure_class=FAILURE_OUTPUT_MISSING,
                detail=(
                    f"the configured SeedVR2 command exited {result.returncode} "
                    "without a valid output"
                ),
                recovery_hint=(
                    "Repair VSR_SEEDVR2_CMD, then retry or disable SeedVR2."
                ),
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            raise RequestedStageError(
                stage="restoration",
                requested_implementation="seedvr2",
                actual_implementation="seedvr2",
                provider="VSR_SEEDVR2_CMD",
                failure_class=FAILURE_RUNTIME,
                detail=type(exc).__name__,
                recovery_hint=(
                    "Repair VSR_SEEDVR2_CMD, then retry or disable SeedVR2."
                ),
                cause=exc,
                retriable=isinstance(exc, subprocess.TimeoutExpired),
            ) from exc
    try:
        from seedvr2 import SeedVR2  # type: ignore
    except ImportError:
        raise RequestedStageError(
            stage="restoration",
            requested_implementation="seedvr2",
            failure_class=FAILURE_DEPENDENCY_MISSING,
            detail="no configured SeedVR2 command or importable wrapper was found",
            recovery_hint=(
                "Install the reviewed SeedVR2 wrapper or set "
                "VSR_SEEDVR2_CMD, then retry or disable SeedVR2."
            ),
        )
    try:
        model = SeedVR2(adapter=adapter)
        produced = model.restore(input_path, output_path)
        if produced and Path(produced).is_file():
            return produced
    except Exception as exc:
        raise RequestedStageError(
            stage="restoration",
            requested_implementation="seedvr2",
            actual_implementation="seedvr2",
            provider="seedvr2.SeedVR2",
            failure_class=FAILURE_RUNTIME,
            detail=str(exc),
            recovery_hint=(
                "Verify the SeedVR2 wrapper and model files, then retry or "
                "disable SeedVR2."
            ),
            cause=exc,
        ) from exc
    raise RequestedStageError(
        stage="restoration",
        requested_implementation="seedvr2",
        actual_implementation="seedvr2",
        provider="seedvr2.SeedVR2",
        failure_class=FAILURE_OUTPUT_MISSING,
        detail="the SeedVR2 wrapper returned no valid output",
        recovery_hint=(
            "Verify the SeedVR2 wrapper output contract, then retry or disable "
            "SeedVR2."
        ),
    )


def swinir_restore(input_path: str, output_path: str,
                    task: str = "classical_sr",
                    scale: int = 2) -> str:
    """RM-79: SwinIR restoration. Pairs with Real-ESRGAN as an
    alternative single-image-restoration backend. Prefers the
    `realsr-ncnn-vulkan` family of binaries (which ship a SwinIR
    variant) when present on PATH; otherwise raises a classified error.

    SwinIR weights are large enough that we do NOT auto-download; the
    user is expected to install the binary distribution separately.
    """
    binaries = ("swinir-ncnn-vulkan", "realsr-ncnn-vulkan", "swinir")
    binary = next((b for b in binaries if shutil.which(b)), None)
    if binary is None:
        raise RequestedStageError(
            stage="restoration",
            requested_implementation="swinir",
            failure_class=FAILURE_DEPENDENCY_MISSING,
            detail="no supported SwinIR binary was found on PATH",
            recovery_hint=(
                "Install a reviewed SwinIR or RealSR NCNN binary and add it "
                "to PATH, or disable SwinIR."
            ),
        )
    try:
        cmd = [binary, "-i", input_path, "-o", output_path, "-s", str(scale)]
        if "swinir" in binary and task:
            cmd += ["-t", task]
        result = run_process(cmd, capture_output=True, text=True, timeout=7200)
        if result.returncode == 0 and Path(output_path).is_file():
            logger.info(f"SwinIR restoration complete ({binary})")
            return output_path
        raise RequestedStageError(
            stage="restoration",
            requested_implementation="swinir",
            actual_implementation="swinir",
            provider=binary,
            failure_class=FAILURE_OUTPUT_MISSING,
            detail=f"{binary} exited {result.returncode} without a valid output",
            recovery_hint=(
                "Verify the SwinIR binary and task settings, then retry or "
                "disable SwinIR."
            ),
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise RequestedStageError(
            stage="restoration",
            requested_implementation="swinir",
            actual_implementation="swinir",
            provider=binary,
            failure_class=FAILURE_RUNTIME,
            detail=type(exc).__name__,
            recovery_hint=(
                "Verify the SwinIR binary can run, then retry or disable "
                "SwinIR."
            ),
            cause=exc,
            retriable=isinstance(exc, subprocess.TimeoutExpired),
        ) from exc


def selected_restoration_provider(implementation: str) -> str:
    """Return the provider selected by the same policy as the adapter."""
    name = str(implementation or "").strip().lower()
    if name == "realesrgan":
        if shutil.which("realesrgan-ncnn-vulkan"):
            return "realesrgan-ncnn-vulkan"
        return ""
    if name == "seedvr2":
        return "VSR_SEEDVR2_CMD" if os.environ.get(
            "VSR_SEEDVR2_CMD", "").strip() else "seedvr2.SeedVR2"
    if name == "swinir":
        return next(
            (candidate for candidate in (
                "swinir-ncnn-vulkan", "realsr-ncnn-vulkan", "swinir"
            ) if shutil.which(candidate)),
            "",
        )
    return ""


_WM_POSITION_MAP = {
    "top-left": ("overlay={margin}:{margin}",),
    "top-right": ("overlay=W-w-{margin}:{margin}",),
    "bottom-left": ("overlay={margin}:H-h-{margin}",),
    "bottom-right": ("overlay=W-w-{margin}:H-h-{margin}",),
    "center": ("overlay=(W-w)/2:(H-h)/2",),
}


def burn_watermark(
    input_path: str,
    output_path: str,
    watermark_path: str,
    position: str = "bottom-right",
    opacity: float = 1.0,
    margin: int = 16,
    video_encode_args: Optional[Sequence[str]] = None,
    preserve_audio: bool = True,
) -> str:
    """Burn a PNG watermark onto the output at a named corner position.

    Uses the FFmpeg overlay filter. Returns the produced path on success
    or raises a classified stage error.
    """
    if shutil.which("ffmpeg") is None:
        raise RequestedStageError(
            stage="watermark",
            requested_implementation="ffmpeg-overlay",
            failure_class=FAILURE_DEPENDENCY_MISSING,
            detail="FFmpeg was not found on PATH",
            recovery_hint="Install the supported FFmpeg runtime or disable watermarking.",
        )
    if not Path(watermark_path).is_file():
        raise RequestedStageError(
            stage="watermark",
            requested_implementation="ffmpeg-overlay",
            failure_class=FAILURE_DEPENDENCY_MISSING,
            detail="the configured watermark image was not found",
            recovery_hint="Select an existing watermark image or disable watermarking.",
        )
    position = position.lower().strip()
    if position not in _WM_POSITION_MAP:
        raise RequestedStageError(
            stage="watermark",
            requested_implementation="ffmpeg-overlay",
            failure_class=FAILURE_POLICY_BLOCKED,
            detail=f"unsupported watermark position {position!r}",
            recovery_hint=(
                "Choose top-left, top-right, bottom-left, bottom-right, or center."
            ),
        )
    overlay_tpl = _WM_POSITION_MAP[position]
    overlay_expr = overlay_tpl[0].format(margin=margin)
    filter_parts = []
    if 0.0 < opacity < 1.0:
        filter_parts.append(f"[1:v]format=rgba,colorchannelmixer=aa={opacity:.2f}[wm]")
        filter_parts.append(f"[0:v][wm]{overlay_expr}")
    else:
        filter_parts.append(f"[0:v][1:v]{overlay_expr}")
    filter_str = ";".join(filter_parts)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats",
        "-i", input_path,
        "-i", watermark_path,
        "-filter_complex", filter_str,
    ]
    cmd += list(video_encode_args or (
        "-c:v", "libx264", "-crf", "20", "-preset", "veryfast"
    ))
    cmd += ["-c:a", "copy"] if preserve_audio else ["-an"]
    cmd += [output_path]
    try:
        run_process(cmd, check=True, capture_output=True, timeout=7200)
        if Path(output_path).is_file():
            logger.info(f"Watermark burned at {position}: {output_path}")
            return output_path
    except subprocess.CalledProcessError as exc:
        raise RequestedStageError(
            stage="watermark",
            requested_implementation="ffmpeg-overlay",
            actual_implementation="ffmpeg-overlay",
            provider="ffmpeg",
            failure_class=FAILURE_RUNTIME,
            detail=f"FFmpeg exited {exc.returncode}",
            recovery_hint="Verify the watermark and output codec, then retry.",
            cause=exc,
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RequestedStageError(
            stage="watermark",
            requested_implementation="ffmpeg-overlay",
            actual_implementation="ffmpeg-overlay",
            provider="ffmpeg",
            failure_class=FAILURE_RUNTIME,
            detail="FFmpeg timed out",
            recovery_hint="Retry the watermark pass or disable watermarking.",
            cause=exc,
            retriable=True,
        ) from exc
    raise RequestedStageError(
        stage="watermark",
        requested_implementation="ffmpeg-overlay",
        actual_implementation="ffmpeg-overlay",
        provider="ffmpeg",
        failure_class=FAILURE_OUTPUT_MISSING,
        detail="FFmpeg returned without a watermark output",
        recovery_hint="Verify the watermark and output codec, then retry.",
    )


def burn_subtitles(
    input_path: str,
    output_path: str,
    subtitle_path: str,
    style_override: str = "",
    video_encode_args: Optional[Sequence[str]] = None,
    preserve_audio: bool = True,
) -> str:
    """Re-burn a subtitle file (.srt, .vtt, or .ass) into the cleaned video.

    Uses ffmpeg's subtitles filter. An optional ASS style override string
    lets callers restyle the burned text (font, size, colour, position).
    Returns the produced path on success or raises a classified stage error.
    """
    if shutil.which("ffmpeg") is None:
        raise RequestedStageError(
            stage="subtitle_burn",
            requested_implementation="ffmpeg-subtitles",
            failure_class=FAILURE_DEPENDENCY_MISSING,
            detail="FFmpeg was not found on PATH",
            recovery_hint="Install the supported FFmpeg runtime or disable subtitle burn.",
        )
    if not Path(subtitle_path).is_file():
        raise RequestedStageError(
            stage="subtitle_burn",
            requested_implementation="ffmpeg-subtitles",
            failure_class=FAILURE_DEPENDENCY_MISSING,
            detail="the configured subtitle file was not found",
            recovery_hint="Select an existing subtitle file or disable subtitle burn.",
        )
    # Escape the path for ffmpeg's filtergraph single-quoted value context.
    # Backslashes become forward slashes (valid on Windows for the subtitles
    # filter), ':' is escaped, and a literal single quote is emitted via the
    # close-escape-reopen sequence so a quote in the filename cannot break out
    # of the filter and inject additional filtergraph clauses.
    sub_escaped = (
        str(subtitle_path)
        .replace("\\", "/")
        .replace(":", "\\:")
        .replace("'", "'\\''")
    )
    vf = f"subtitles='{sub_escaped}'"
    safe_style = _sanitize_force_style(style_override)
    if style_override and not safe_style:
        raise RequestedStageError(
            stage="subtitle_burn",
            requested_implementation="ffmpeg-subtitles",
            failure_class=FAILURE_POLICY_BLOCKED,
            detail="the subtitle style contains unsafe force_style characters",
            recovery_hint="Remove unsupported characters from the style override.",
        )
    if safe_style:
        vf += f":force_style='{safe_style}'"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats",
        "-i", input_path,
        "-vf", vf,
    ]
    cmd += list(video_encode_args or (
        "-c:v", "libx264", "-crf", "20", "-preset", "veryfast"
    ))
    cmd += ["-c:a", "copy"] if preserve_audio else ["-an"]
    cmd += [output_path]
    try:
        run_process(cmd, check=True, capture_output=True, timeout=7200)
        if Path(output_path).is_file():
            logger.info(f"Subtitles re-burned: {output_path}")
            return output_path
    except subprocess.CalledProcessError as exc:
        raise RequestedStageError(
            stage="subtitle_burn",
            requested_implementation="ffmpeg-subtitles",
            actual_implementation="ffmpeg-subtitles",
            provider="ffmpeg",
            failure_class=FAILURE_RUNTIME,
            detail=f"FFmpeg exited {exc.returncode}",
            recovery_hint="Verify the subtitle file and output codec, then retry.",
            cause=exc,
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RequestedStageError(
            stage="subtitle_burn",
            requested_implementation="ffmpeg-subtitles",
            actual_implementation="ffmpeg-subtitles",
            provider="ffmpeg",
            failure_class=FAILURE_RUNTIME,
            detail="FFmpeg timed out",
            recovery_hint="Retry the subtitle burn or disable it.",
            cause=exc,
            retriable=True,
        ) from exc
    raise RequestedStageError(
        stage="subtitle_burn",
        requested_implementation="ffmpeg-subtitles",
        actual_implementation="ffmpeg-subtitles",
        provider="ffmpeg",
        failure_class=FAILURE_OUTPUT_MISSING,
        detail="FFmpeg returned without a subtitle-burn output",
        recovery_hint="Verify the subtitle file and output codec, then retry.",
    )


def add_film_grain(
    input_path: str,
    output_path: str,
    strength: float = 0.04,
    *,
    video_encode_args: Optional[Sequence[str]] = None,
    preserve_audio: bool = True,
) -> str:
    """RM-80: cheap additive film grain.

    Two paths:
    - For software AV1 output, `SubtitleRemover` enables SVT-AV1's
      native film-grain table during encode. The additive path here is
      a fallback for H.264 / H.265 / hardware-encoder outputs.
    - For other codecs we use ffmpeg's `noise` filter to add per-channel
      uniform noise to every frame. `strength` is roughly the noise
      amplitude as a fraction of full-scale (0.04 ~= 10/255). Returns
      the validated output path or raises a classified stage error.
    """
    if shutil.which("ffmpeg") is None:
        raise RequestedStageError(
            stage="film_grain",
            requested_implementation="ffmpeg-noise",
            failure_class=FAILURE_DEPENDENCY_MISSING,
            detail="FFmpeg was not found on PATH",
            recovery_hint="Install the supported FFmpeg runtime or disable film grain.",
        )
    strength = float(strength)
    if not (0.0 < strength <= 0.5):
        raise RequestedStageError(
            stage="film_grain",
            requested_implementation="ffmpeg-noise",
            failure_class=FAILURE_POLICY_BLOCKED,
            detail="film-grain strength must be greater than 0 and at most 0.5",
            recovery_hint="Choose a film-grain strength between 0 and 0.5.",
        )
    noise_level = max(1, int(round(strength * 255)))
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats",
        "-i", input_path,
        "-vf", f"noise=alls={noise_level}:allf=t",
    ]
    cmd += list(video_encode_args or (
        "-c:v", "libx264", "-crf", "20", "-preset", "veryfast"
    ))
    cmd += ["-c:a", "copy"] if preserve_audio else ["-an"]
    cmd += [output_path]
    try:
        run_process(cmd, check=True, capture_output=True, timeout=7200)
        if Path(output_path).is_file():
            return output_path
    except subprocess.CalledProcessError as exc:
        raise RequestedStageError(
            stage="film_grain",
            requested_implementation="ffmpeg-noise",
            actual_implementation="ffmpeg-noise",
            provider="ffmpeg",
            failure_class=FAILURE_RUNTIME,
            detail=f"FFmpeg exited {exc.returncode}",
            recovery_hint="Verify the output codec, then retry or disable film grain.",
            cause=exc,
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RequestedStageError(
            stage="film_grain",
            requested_implementation="ffmpeg-noise",
            actual_implementation="ffmpeg-noise",
            provider="ffmpeg",
            failure_class=FAILURE_RUNTIME,
            detail="FFmpeg timed out",
            recovery_hint="Retry the film-grain pass or disable it.",
            cause=exc,
            retriable=True,
        ) from exc
    raise RequestedStageError(
        stage="film_grain",
        requested_implementation="ffmpeg-noise",
        actual_implementation="ffmpeg-noise",
        provider="ffmpeg",
        failure_class=FAILURE_OUTPUT_MISSING,
        detail="FFmpeg returned without a film-grain output",
        recovery_hint="Verify the output codec, then retry or disable film grain.",
    )
