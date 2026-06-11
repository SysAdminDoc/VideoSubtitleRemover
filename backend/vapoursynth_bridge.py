"""VapourSynth bridge.

RM-75: a thin adapter that lets advanced users slot VSR into a larger
post pipeline. The user passes a trusted `.vpy` script as input; when
`VSR_VAPOURSYNTH=1` is set we evaluate it via the VapourSynth Python
API and expose a `cv2.VideoCapture`-shaped reader so the rest of
`process_video` does not need to know about the bridge.

This is opt-in at two levels: users must enable the env gate because
`.vpy` files are Python scripts, and users without VapourSynth installed
see a graceful fallback when `_open_capture` is asked to handle a `.vpy`
file.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _vapoursynth_enabled() -> bool:
    return os.environ.get("VSR_VAPOURSYNTH", "").strip().lower() in _TRUE_VALUES


class _VapourSynthCapture:
    """`cv2.VideoCapture`-shaped wrapper around a VapourSynth clip."""

    def __init__(self, script_path: str):
        self._script_path = script_path
        self._clip = None
        self._frame_count = 0
        self._fps = 30.0
        self._width = 0
        self._height = 0
        self._pos = 0
        self._opened = False
        self._open()

    def _open(self) -> None:
        try:
            import vapoursynth as vs  # type: ignore
        except ImportError:
            logger.info(
                "vapoursynth not installed; cannot evaluate .vpy "
                "input. `pip install vapoursynth` to enable."
            )
            return
        try:
            ns: dict = {}
            with open(self._script_path, "r", encoding="utf-8") as f:
                exec(f.read(), ns)
            clip = ns.get("clip") or ns.get("video") or ns.get("output")
            if clip is None and hasattr(vs, "get_output"):
                try:
                    clip = vs.get_output(0)
                except Exception:
                    clip = None
            if clip is None:
                logger.warning(
                    f".vpy script {self._script_path!r} did not assign a "
                    "`clip`/`video`/`output` variable and no vs.get_output(0)."
                )
                return
            # Convert to RGB24 so downstream cv2 / numpy frames are
            # uint8 BGR after the channel-swap.
            try:
                clip = clip.resize.Point(format=vs.RGB24, matrix_in_s="709")
            except Exception:
                pass
            self._clip = clip
            self._frame_count = int(clip.num_frames)
            self._width = int(clip.width)
            self._height = int(clip.height)
            if clip.fps and clip.fps.denominator:
                self._fps = float(clip.fps.numerator) / float(clip.fps.denominator)
            self._opened = True
        except Exception as exc:
            logger.warning(f"VapourSynth evaluation failed: {exc}")

    def isOpened(self) -> bool:
        return self._opened

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._frame_count)
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self._pos)
        return 0.0

    def set(self, prop, value) -> bool:
        if prop == cv2.CAP_PROP_POS_FRAMES:
            try:
                self._pos = max(0, min(self._frame_count, int(value)))
                return True
            except Exception:
                return False
        return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._opened or self._pos >= self._frame_count:
            return False, None
        try:
            vsf = self._clip.get_frame(self._pos)
            self._pos += 1
            r = np.asarray(vsf[0])
            g = np.asarray(vsf[1])
            b = np.asarray(vsf[2])
            bgr = np.stack([b, g, r], axis=2).astype(np.uint8)
            return True, bgr
        except Exception as exc:
            logger.warning(f"VapourSynth frame read failed: {exc}")
            return False, None

    def release(self) -> None:
        self._clip = None
        self._opened = False


def try_open_vpy(path: str):
    """Return a _VapourSynthCapture when the open succeeded, else None."""
    if not path.lower().endswith(".vpy"):
        return None
    if not _vapoursynth_enabled():
        logger.info(
            "VapourSynth .vpy input disabled; set VSR_VAPOURSYNTH=1 "
            "to execute trusted scripts."
        )
        return None
    cap = _VapourSynthCapture(path)
    return cap if cap.isOpened() else None
