"""Whisper fallback for low-OCR-confidence frames.

RM-27: when OCR confidence drops below a threshold (anti-aliased
subtitle, motion blur, decorative font the cascade misreads), call
faster-whisper on the audio track to derive an SRT-shaped timing
list. Frames whose timestamp falls inside a Whisper segment get the
bottom subtitle band masked, even when OCR returns no boxes.

This module is opt-in: faster-whisper and FFmpeg's whisper filter are
not hard dependencies. The adapters import/probe lazily and degrade
gracefully when:
- `faster-whisper` is not installed,
- the installed ffmpeg lacks the `whisper` audio filter,
- the FFmpeg backend has no local whisper.cpp ggml model path,
- ffprobe / ffmpeg can't extract the audio,
- the source has no audio stream at all.

`run_whisper_segments(path, model_size, language)` returns a sorted
list of (start_seconds, end_seconds, text) tuples or None on failure.
Caller turns each segment into a mask span over its frame range.

The model_size default is "tiny" because we're matching dialogue
timing, not transcribing for accuracy; tiny runs comfortably on CPU.
The first call downloads weights to the HuggingFace cache.
"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from backend.import_safety import module_can_import as _safe_module_can_import

logger = logging.getLogger(__name__)

_SRT_TIME_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*"
    r"(?P<end>\d{2}:\d{2}:\d{2},\d{3})"
)


def is_available() -> bool:
    """Best-effort check for faster-whisper availability."""
    return _module_can_import("faster_whisper")


def _module_can_import(module_name: str, *, timeout: float = 20.0) -> bool:
    return _safe_module_can_import(
        module_name,
        timeout=timeout,
        logger=logger,
        failure_context="optional Whisper fallback disabled",
    )


def ffmpeg_whisper_available(ffmpeg: str = "ffmpeg") -> bool:
    """Return True when `ffmpeg -filters` reports the whisper filter."""
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
        re.search(r"(?m)^\s*[.A-Z| ]+\s+whisper\s+", output)
    )


def _escape_filter_value(value: str) -> str:
    """Escape a string for use as an FFmpeg filter option value."""
    escaped = str(value).replace("\\", "\\\\")
    for char in (":", "'", ",", "[", "]"):
        escaped = escaped.replace(char, f"\\{char}")
    return escaped


def _unescape_filter_value(value: str) -> str:
    """Inverse of `_escape_filter_value`, used by tests and diagnostics."""
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


def _build_ffmpeg_whisper_filter(
    model_path: str,
    destination_path: str,
    language: Optional[str] = None,
    queue_seconds: float = 3.0,
    vad_model: str = "",
    vad_threshold: float = 0.5,
    min_speech_duration: float = 0.0,
) -> str:
    lang = (language or "auto").strip() or "auto"
    queue = max(0.02, float(queue_seconds))
    options = [
        f"model={_escape_filter_value(model_path)}",
        f"language={_escape_filter_value(lang)}",
        f"queue={queue:g}",
        "format=srt",
        f"destination={_escape_filter_value(destination_path)}",
    ]
    if vad_model:
        options.append(f"vad_model={_escape_filter_value(vad_model)}")
    if vad_model and 0.0 < vad_threshold < 1.0:
        options.append(f"vad_threshold={vad_threshold:g}")
    if min_speech_duration > 0:
        options.append(f"min_speech_duration={min_speech_duration:g}")
    return (
        "aformat=sample_rates=16000:channel_layouts=mono,"
        f"whisper={':'.join(options)}"
    )


def _parse_srt_timestamp(value: str) -> float:
    hh, mm, rest = value.split(":", 2)
    ss, ms = rest.split(",", 1)
    return (
        int(hh) * 3600.0
        + int(mm) * 60.0
        + int(ss)
        + int(ms) / 1000.0
    )


def parse_srt_segments(srt_text: str) -> List[Tuple[float, float, str]]:
    """Parse SRT text into Whisper segment tuples."""
    segments: List[Tuple[float, float, str]] = []
    blocks = re.split(r"\r?\n\s*\r?\n", srt_text.strip())
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        time_index = next((i for i, line in enumerate(lines) if "-->" in line), -1)
        if time_index < 0:
            continue
        match = _SRT_TIME_RE.search(lines[time_index])
        if not match:
            continue
        text = " ".join(lines[time_index + 1:]).strip()
        if not text:
            continue
        start = _parse_srt_timestamp(match.group("start"))
        end = _parse_srt_timestamp(match.group("end"))
        if end > start:
            segments.append((start, end, text))
    return segments


def run_ffmpeg_whisper_segments(
    media_path: str,
    model_path: str,
    language: Optional[str] = None,
    queue_seconds: float = 3.0,
    ffmpeg: str = "ffmpeg",
    vad_model: str = "",
    vad_threshold: float = 0.5,
    min_speech_duration: float = 0.0,
) -> Optional[List[Tuple[float, float, str]]]:
    """Transcribe `media_path` through FFmpeg's whisper filter.

    The caller must provide a local whisper.cpp ggml model path. Returns
    None when ffmpeg lacks the filter, the model is absent, or the
    command fails.
    """
    if not Path(media_path).is_file():
        logger.warning(f"FFmpeg Whisper input media missing: {media_path}")
        return None
    if not model_path or not Path(model_path).is_file():
        logger.info(
            "FFmpeg Whisper backend requires a local whisper.cpp ggml "
            "model path."
        )
        return None
    if not ffmpeg_whisper_available(ffmpeg):
        logger.info("ffmpeg whisper filter unavailable; Whisper fallback disabled.")
        return None

    try:
        from backend.io import _ffmpeg_subprocess_timeout, _probe_duration_seconds
        duration = _probe_duration_seconds(media_path)
        timeout = _ffmpeg_subprocess_timeout(duration, base=600.0, factor=12.0)
    except Exception:
        timeout = 3600.0

    with tempfile.TemporaryDirectory(prefix="vsr_ffmpeg_whisper_") as tmpdir:
        srt_path = str(Path(tmpdir) / "whisper.srt")
        filter_expr = _build_ffmpeg_whisper_filter(
            model_path,
            srt_path,
            language=language,
            queue_seconds=queue_seconds,
            vad_model=vad_model,
            vad_threshold=vad_threshold,
            min_speech_duration=min_speech_duration,
        )
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats",
            "-i", media_path,
            "-vn",
            "-af", filter_expr,
            "-f", "null", "-",
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or b"")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", "replace")
            logger.info(f"FFmpeg Whisper transcription failed: {stderr[:400]}")
            return None
        except subprocess.TimeoutExpired:
            logger.warning("FFmpeg Whisper transcription timed out")
            return None
        except FileNotFoundError:
            logger.info("ffmpeg not on PATH; cannot run FFmpeg Whisper")
            return None
        out_path = Path(srt_path)
        if not out_path.is_file() or out_path.stat().st_size == 0:
            return None
        return parse_srt_segments(out_path.read_text(encoding="utf-8", errors="replace"))


def run_whisper_segments(audio_path: str, model_size: str = "tiny",
                          language: Optional[str] = None,
                          compute_type: str = "int8") -> Optional[List[Tuple[float, float, str]]]:
    """Transcribe `audio_path` with faster-whisper and return segment
    timings. Returns None when faster-whisper is missing or the run
    fails so the caller can fall back to the OCR-only path.

    `compute_type="int8"` is the CPU-friendly default; pass "float16"
    when running on a CUDA box.
    """
    if not Path(audio_path).is_file():
        logger.warning(f"Whisper input audio missing: {audio_path}")
        return None
    if not _module_can_import("faster_whisper"):
        logger.info(
            "faster-whisper is not installed or cannot be imported; "
            "Whisper fallback disabled."
        )
        return None
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except (ImportError, OSError, RuntimeError) as exc:
        logger.info(
            f"faster-whisper import failed; Whisper fallback disabled: {exc}"
        )
        return None
    try:
        model = WhisperModel(model_size, device="auto", compute_type=compute_type)
        segments, _info = model.transcribe(audio_path, language=language)
        out: List[Tuple[float, float, str]] = []
        for seg in segments:
            text = (seg.text or "").strip()
            if text:
                out.append((float(seg.start), float(seg.end), text))
        return out
    except Exception as exc:
        logger.warning(f"Whisper transcription failed: {exc}")
        return None


def extract_audio_to_temp(video_path: str, temp_dir: str) -> Optional[str]:
    """Strip audio from `video_path` via ffmpeg to a wav file in
    `temp_dir`. Returns the wav path or None on any failure (no audio
    stream, ffmpeg missing, etc.).
    """
    if shutil.which("ffmpeg") is None:
        logger.info("ffmpeg not on PATH; cannot extract audio for Whisper")
        return None
    dst = os.path.join(temp_dir, "whisper_audio.wav")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats",
        "-i", video_path, "-vn", "-ac", "1", "-ar", "16000",
        "-acodec", "pcm_s16le", dst,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
    except subprocess.CalledProcessError as exc:
        logger.info(f"Audio extraction failed (no stream?): {exc}")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("Audio extraction timed out")
        return None
    if not Path(dst).is_file() or Path(dst).stat().st_size == 0:
        return None
    return dst


def segments_to_frame_spans(
    segments: List[Tuple[float, float, str]], fps: float
) -> List[Tuple[int, int]]:
    """Convert a Whisper segment list into [(start_frame, end_frame), ...]
    intervals using the source's FPS. Empty segments are skipped and
    overlapping intervals are merged so the caller can do a single
    membership check per frame index.
    """
    if not segments or fps <= 0:
        return []
    spans: List[Tuple[int, int]] = []
    for start_s, end_s, _text in segments:
        try:
            start = float(start_s)
            end = float(end_s)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(start) and math.isfinite(end)) or end <= start:
            continue
        s = max(0, int(start * fps))
        e = max(s + 1, int(end * fps))
        spans.append((s, e))
    spans.sort()
    merged: List[Tuple[int, int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged
