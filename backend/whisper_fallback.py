"""Whisper fallback for low-OCR-confidence frames.

RM-27: when OCR confidence drops below a threshold (anti-aliased
subtitle, motion blur, decorative font the cascade misreads), call
faster-whisper on the audio track to derive an SRT-shaped timing
list. Frames whose timestamp falls inside a Whisper segment get the
bottom subtitle band masked, even when OCR returns no boxes.

This module is opt-in: faster-whisper is not a hard dependency. The
adapter imports lazily and degrades gracefully when:
- `faster-whisper` is not installed,
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
import os
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def is_available() -> bool:
    """Best-effort check for faster-whisper availability."""
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def run_whisper_segments(audio_path: str, model_size: str = "tiny",
                          language: Optional[str] = None,
                          compute_type: str = "int8") -> Optional[List[Tuple[float, float, str]]]:
    """Transcribe `audio_path` with faster-whisper and return segment
    timings. Returns None when faster-whisper is missing or the run
    fails so the caller can fall back to the OCR-only path.

    `compute_type="int8"` is the CPU-friendly default; pass "float16"
    when running on a CUDA box.
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError:
        logger.info(
            "faster-whisper is not installed; Whisper fallback disabled. "
            "Install with `pip install faster-whisper` to enable."
        )
        return None
    if not Path(audio_path).is_file():
        logger.warning(f"Whisper input audio missing: {audio_path}")
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
    import shutil
    import subprocess
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
        s = max(0, int(start_s * fps))
        e = max(s + 1, int(end_s * fps))
        spans.append((s, e))
    spans.sort()
    merged: List[Tuple[int, int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged
