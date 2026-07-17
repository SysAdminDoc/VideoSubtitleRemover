"""AUTO inpainter: per-scene STTN/ProPainter routing with lazy loading."""

from __future__ import annotations

import logging
import sys
from typing import List, Optional

import numpy as np

import cv2

from backend.inpainters._common import BaseInpainter, _detect_scene_cuts
from backend.inpainters.sttn import STTNInpainter
from backend.inpainters.propainter import ProPainterInpainter

logger = logging.getLogger(__name__)


class AutoInpainter(BaseInpainter):
    """Route each scene to STTN or motion-robust ProPainter mode.

    STTN handles low-motion scenes whose masked pixels become exposed over
    time. ProPainter mode handles fast motion or persistently covered pixels.
    Its optional LaMa refinement remains lazy and is unloaded after a long
    streak of STTN scenes so easy videos do not pin model memory.
    """

    PROPAINTER_IDLE_UNLOAD_AFTER = 50
    MOTION_THRESHOLD = 0.04

    def __init__(self, device: str = "cuda:0", config=None):
        self.device = device
        from backend.config import ProcessingConfig
        self.config = config or ProcessingConfig()
        self._sttn = STTNInpainter(device, self.config)
        self._propainter: Optional[ProPainterInpainter] = None
        self._sttn_streak: int = 0

    def _ensure_propainter(self) -> ProPainterInpainter:
        if self._propainter is None:
            self._propainter = ProPainterInpainter(self.device, self.config)
        return self._propainter

    def _maybe_unload_propainter(self) -> None:
        if self._propainter is None:
            return
        if self._sttn_streak < self.PROPAINTER_IDLE_UNLOAD_AFTER:
            return
        logger.info(
            "AUTO: unloading idle ProPainter mode after %d STTN scenes",
            self._sttn_streak,
        )
        self._propainter = None
        try:
            import gc as _gc
            _gc.collect()
        except Exception:
            logger.warning(
                "AUTO ProPainter idle GC cleanup failed", exc_info=True)
        torch_mod = sys.modules.get("torch")
        if torch_mod is None:
            return
        try:
            if hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
                torch_mod.cuda.empty_cache()
        except Exception:
            logger.warning(
                "AUTO ProPainter idle CUDA cleanup failed", exc_info=True)

    @staticmethod
    def _exposure_score(masks: List[np.ndarray]) -> float:
        if len(masks) < 2:
            return 0.0
        stack = np.stack(masks, axis=0)
        unmasked = (stack == 0)
        any_union = unmasked.any(axis=0)
        ever_masked = (stack > 0).any(axis=0)
        total = int(ever_masked.sum())
        if total == 0:
            return 1.0
        exposed = int((ever_masked & any_union).sum())
        return exposed / float(total)

    @staticmethod
    def _motion_score(frames: List[np.ndarray]) -> float:
        """Return mean normalized luminance change between adjacent frames."""
        if len(frames) < 2:
            return 0.0
        scores = []
        previous = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        for frame in frames[1:]:
            current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scores.append(float(cv2.absdiff(previous, current).mean()) / 255.0)
            previous = current
        return sum(scores) / len(scores)

    def _scene_starts(self, frames: List[np.ndarray]) -> List[int]:
        if not self.config.tbe_scene_cut_split or len(frames) < 2:
            return [0]
        return _detect_scene_cuts(
            frames,
            self.config.tbe_scene_cut_threshold,
            prefer_pyscenedetect=self.config.tbe_scene_cut_use_pyscenedetect,
            prefer_transnetv2=self.config.tbe_scene_cut_use_transnetv2,
        )

    def _inpaint_scene(
        self, frames: List[np.ndarray], masks: List[np.ndarray], scene_index: int
    ) -> List[np.ndarray]:
        threshold = self.config.auto_exposure_threshold
        exposure = self._exposure_score(masks)
        motion = self._motion_score(frames)
        if exposure >= threshold and motion < self.MOTION_THRESHOLD:
            logger.debug(
                "AUTO scene %d: STTN path (exposure=%.2f, motion=%.3f)",
                scene_index,
                exposure,
                motion,
            )
            self._sttn_streak += 1
            self._maybe_unload_propainter()
            return self._sttn.inpaint(frames, masks)
        logger.debug(
            "AUTO scene %d: ProPainter path (exposure=%.2f, motion=%.3f)",
            scene_index,
            exposure,
            motion,
        )
        self._sttn_streak = 0
        return self._ensure_propainter().inpaint(frames, masks)

    def inpaint(
        self, frames: List[np.ndarray], masks: List[np.ndarray]
    ) -> List[np.ndarray]:
        if not frames:
            return []
        starts = self._scene_starts(frames)
        results: List[np.ndarray] = []
        for scene_index, start in enumerate(starts):
            end = starts[scene_index + 1] if scene_index + 1 < len(starts) else len(frames)
            results.extend(self._inpaint_scene(
                frames[start:end], masks[start:end], scene_index))
        return results
