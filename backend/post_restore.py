"""Optional post-restore passes that run after the inpaint stage.

RM-78 Real-ESRGAN super-resolution and RM-80 film-grain re-synthesis
both shape the *output* video rather than the inpainted region. Each
adapter here:

- Imports lazily so the rest of the codebase keeps working when the
  optional dependency / weight file is missing.
- Operates on a finished video file (the FFV1 intermediate or final
  output, depending on where the caller wires it in).
- Returns the path to a new file on success, or the input path
  unchanged when the optional dep is unavailable so the pipeline
  always produces *some* output.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def realesrgan_upscale(input_path: str, output_path: str,
                       scale: int = 2,
                       model_name: str = "RealESRGAN_x4plus") -> Optional[str]:
    """RM-78: 2x or 4x upscale via Real-ESRGAN.

    Tries the `realesrgan-ncnn-vulkan` standalone binary first (the
    most portable distribution -- one .exe, no Python deps). Falls
    back to the `realesrgan` Python package when available. Returns
    the output path on success, None on failure so the caller can
    keep the original output.
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
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode == 0 and Path(output_path).is_file():
                logger.info(f"Real-ESRGAN upscaled to {output_path}")
                return output_path
            logger.warning(
                f"realesrgan-ncnn-vulkan exit {result.returncode}: "
                f"{(result.stderr or '')[:400]}"
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.warning(f"realesrgan-ncnn-vulkan failed: {exc}")
    try:
        from realesrgan import RealESRGANer  # type: ignore
    except ImportError:
        logger.info(
            "Neither realesrgan-ncnn-vulkan binary nor the realesrgan "
            "python package was found; skipping upscale stage."
        )
        return None
    # The Python package operates on PIL images; we shell out to ffmpeg
    # to dump frames, upscale each, then re-mux. Avoiding that path here
    # because the binary distribution is both faster and cheaper to
    # ship. When the user installs the python package without the
    # binary, log and skip rather than pull in another temp-dir dance.
    logger.warning(
        "realesrgan python package detected but the binary is preferred. "
        "Install realesrgan-ncnn-vulkan from "
        "https://github.com/xinntao/Real-ESRGAN/releases to enable upscale."
    )
    return None


def seedvr2_restore(input_path: str, output_path: str,
                     adapter: str = "seedvr2") -> Optional[str]:
    """RM-77 SeedVR2 one-step video restoration.

    SeedVR2 ships as a 16B-param diffusion transformer with adversarial
    post-training -- single sampling step. Best-in-class quality on
    heavy-degradation footage but the install footprint is large (the
    user is expected to clone IceClear/SeedVR2 separately and either
    set `VSR_SEEDVR2_CMD` to the CLI entrypoint or install a
    pip-published wrapper named `seedvr2`).

    Returns the path on success, None on missing-dep / failure.
    """
    cmd_env = os.environ.get("VSR_SEEDVR2_CMD", "")
    if cmd_env:
        try:
            import shlex
            cmd = shlex.split(cmd_env) + ["-i", input_path, "-o", output_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
            if result.returncode == 0 and Path(output_path).is_file():
                logger.info(f"SeedVR2 restoration complete via {cmd_env}")
                return output_path
            logger.warning(
                f"VSR_SEEDVR2_CMD exit {result.returncode}: {(result.stderr or '')[:400]}"
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.warning(f"VSR_SEEDVR2_CMD failed: {exc}")
    try:
        from seedvr2 import SeedVR2  # type: ignore
    except ImportError:
        logger.info(
            "SeedVR2 wrapper not importable; install via the upstream "
            "IceClear/SeedVR2 project or set VSR_SEEDVR2_CMD to a CLI "
            "entrypoint that accepts `-i INPUT -o OUTPUT`."
        )
        return None
    try:
        model = SeedVR2(adapter=adapter)
        produced = model.restore(input_path, output_path)
        if produced and Path(produced).is_file():
            return produced
    except Exception as exc:
        logger.warning(f"SeedVR2 wrapper failed: {exc}")
    return None


def swinir_restore(input_path: str, output_path: str,
                    task: str = "classical_sr",
                    scale: int = 2) -> Optional[str]:
    """RM-79: SwinIR restoration. Pairs with Real-ESRGAN as an
    alternative single-image-restoration backend. Prefers the
    `realsr-ncnn-vulkan` family of binaries (which ship a SwinIR
    variant) when present on PATH; otherwise logs and returns None.

    SwinIR weights are large enough that we do NOT auto-download; the
    user is expected to install the binary distribution separately.
    """
    binaries = ("swinir-ncnn-vulkan", "realsr-ncnn-vulkan", "swinir")
    binary = next((b for b in binaries if shutil.which(b)), None)
    if binary is None:
        logger.info(
            "No SwinIR binary found on PATH (looked for "
            f"{', '.join(binaries)}). Skipping restoration pass."
        )
        return None
    try:
        cmd = [binary, "-i", input_path, "-o", output_path, "-s", str(scale)]
        if "swinir" in binary and task:
            cmd += ["-t", task]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if result.returncode == 0 and Path(output_path).is_file():
            logger.info(f"SwinIR restoration complete ({binary})")
            return output_path
        logger.warning(
            f"{binary} exit {result.returncode}: {(result.stderr or '')[:400]}"
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning(f"SwinIR pass failed: {exc}")
    return None


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
) -> Optional[str]:
    """Burn a PNG watermark onto the output at a named corner position.

    Uses ffmpeg overlay filter. Returns the produced path on success,
    None when ffmpeg or the watermark image is missing.
    """
    if shutil.which("ffmpeg") is None:
        logger.info("ffmpeg not on PATH; cannot burn watermark")
        return None
    if not Path(watermark_path).is_file():
        logger.warning(f"Watermark image not found: {watermark_path}")
        return None
    position = position.lower().strip()
    overlay_tpl = _WM_POSITION_MAP.get(position, _WM_POSITION_MAP["bottom-right"])
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
        "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
        "-c:a", "copy", output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=7200)
        if Path(output_path).is_file():
            logger.info(f"Watermark burned at {position}: {output_path}")
            return output_path
    except subprocess.CalledProcessError as exc:
        logger.warning(
            f"Watermark burn failed: "
            f"{exc.stderr.decode('utf-8', 'replace')[:400]}"
        )
    except subprocess.TimeoutExpired:
        logger.warning("Watermark burn timed out")
    return None


def burn_subtitles(
    input_path: str,
    output_path: str,
    subtitle_path: str,
    style_override: str = "",
) -> Optional[str]:
    """Re-burn a subtitle file (.srt or .ass) into the cleaned video.

    Uses ffmpeg's subtitles filter. An optional ASS style override string
    lets callers restyle the burned text (font, size, colour, position).
    Returns the produced path on success, None when ffmpeg or the
    subtitle file is missing.
    """
    if shutil.which("ffmpeg") is None:
        logger.info("ffmpeg not on PATH; cannot burn subtitles")
        return None
    if not Path(subtitle_path).is_file():
        logger.warning(f"Subtitle file not found: {subtitle_path}")
        return None
    sub_escaped = str(subtitle_path).replace("\\", "/").replace(":", "\\\\:")
    vf = f"subtitles='{sub_escaped}'"
    if style_override:
        vf += f":force_style='{style_override}'"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats",
        "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
        "-c:a", "copy", output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=7200)
        if Path(output_path).is_file():
            logger.info(f"Subtitles re-burned: {output_path}")
            return output_path
    except subprocess.CalledProcessError as exc:
        logger.warning(
            f"Subtitle burn failed: "
            f"{exc.stderr.decode('utf-8', 'replace')[:400]}"
        )
    except subprocess.TimeoutExpired:
        logger.warning("Subtitle burn timed out")
    return None


def add_film_grain(input_path: str, output_path: str,
                    strength: float = 0.04) -> Optional[str]:
    """RM-80: cheap additive film grain.

    Two paths:
    - For software AV1 output, `SubtitleRemover` enables SVT-AV1's
      native film-grain table during encode. The additive path here is
      a fallback for H.264 / H.265 / hardware-encoder outputs.
    - For other codecs we use ffmpeg's `noise` filter to add per-channel
      uniform noise to every frame. `strength` is roughly the noise
      amplitude as a fraction of full-scale (0.04 ~= 10/255). Returns
      the produced output path or None when ffmpeg is missing.
    """
    if shutil.which("ffmpeg") is None:
        logger.info("ffmpeg not on PATH; cannot add film grain")
        return None
    strength = float(strength)
    if not (0.0 < strength <= 0.5):
        strength = 0.04
    noise_level = max(1, int(round(strength * 255)))
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats",
        "-i", input_path,
        "-vf", f"noise=alls={noise_level}:allf=t",
        "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
        "-c:a", "copy", output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=7200)
        return output_path
    except subprocess.CalledProcessError as exc:
        logger.warning(
            f"Film-grain pass failed: {exc.stderr.decode('utf-8', 'replace')[:400]}"
        )
        return None
    except subprocess.TimeoutExpired:
        logger.warning("Film-grain pass timed out")
        return None
