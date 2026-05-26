"""Optical-flow-based karaoke-line tracking + WhisperX chyron helper.

RM-43: even with the karaoke_grouping fusion shipped in v3.13, music-
video captions whose syllables morph per-frame can still produce
short-lived Kalman tracks that get missed. The helper here uses dense
Farneback flow between consecutive detection frames to extend the
mask across the warp so the inpaint pass covers the moving karaoke
line as a single contiguous region.

RM-45: WhisperX exposes word-level timestamps + speaker labels. We
extend the Whisper fallback (RM-27) with a richer classifier hint:
text regions detected during a Whisper word span are confirmed as
dialogue subtitles; boxes that persist OUTSIDE every span are more
likely chyrons. The chyron classifier in the existing SubtitleTracker
already does this through hit-count; WhisperX adds a complementary
signal for borderline cases.

Both helpers are opt-in; missing deps degrade gracefully.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def warp_mask_with_flow(prev_frame: np.ndarray,
                         next_frame: np.ndarray,
                         mask: np.ndarray) -> np.ndarray:
    """RM-43: warp `mask` from `prev_frame`'s coords into the next
    frame's coords using Farneback dense optical flow. Lets callers
    union the warped mask with the next detection to catch karaoke
    text that moved between frames."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None,
        pyr_scale=0.5, levels=3, winsize=21, iterations=3,
        poly_n=7, poly_sigma=1.5, flags=0,
    )
    h, w = mask.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = gx + flow[..., 0]
    map_y = gy + flow[..., 1]
    return cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def is_whisperx_available() -> bool:
    try:
        import whisperx  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def run_whisperx(audio_path: str, language: Optional[str] = None,
                  model_size: str = "tiny") -> Optional[List[Tuple[float, float, str]]]:
    """RM-45: transcribe + align via WhisperX. Returns word-level
    (start, end, text) tuples or None when WhisperX is missing /
    fails. Caller can merge these into the existing Whisper-span
    fallback for finer-grained mask gating.
    """
    try:
        import whisperx  # type: ignore
    except ImportError:
        return None
    try:
        device = "cuda" if _torch_cuda_available() else "cpu"
        model = whisperx.load_model(model_size, device, compute_type="int8")
        result = model.transcribe(audio_path, language=language)
        try:
            align_model, metadata = whisperx.load_align_model(
                language_code=result.get("language") or "en", device=device
            )
            aligned = whisperx.align(result["segments"], align_model, metadata,
                                       audio_path, device, return_char_alignments=False)
            words: List[Tuple[float, float, str]] = []
            for seg in aligned.get("segments", []):
                for w in seg.get("words", []):
                    if "start" in w and "end" in w and w.get("word"):
                        words.append((float(w["start"]), float(w["end"]), w["word"]))
            return words
        except Exception:
            # Fall back to segment-level timestamps; still useful.
            words = [
                (float(s["start"]), float(s["end"]), (s.get("text") or "").strip())
                for s in result.get("segments", [])
                if s.get("start") is not None and s.get("end") is not None
            ]
            return words
    except Exception as exc:
        logger.warning(f"WhisperX inference failed: {exc}")
        return None


def _torch_cuda_available() -> bool:
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False
