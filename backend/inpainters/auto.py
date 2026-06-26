"""AUTO inpainter: per-batch routing between TBE (STTN) and LaMa with
lazy LaMa load + idle unload (B-5)."""

from __future__ import annotations

import logging
import sys
from typing import List, Optional

import numpy as np

from backend.inpainters._common import BaseInpainter
from backend.inpainters.sttn import STTNInpainter
from backend.inpainters.lama import LAMAInpainter

logger = logging.getLogger(__name__)


class AutoInpainter(BaseInpainter):
    """Per-batch routing between TBE (fast, temporal) and LaMa
    (robust, spatial). LaMa loads lazily on first hard batch and
    unloads after `LAMA_IDLE_UNLOAD_AFTER` consecutive TBE batches
    so long videos don't permanently pin ~1.5 GB VRAM."""

    LAMA_IDLE_UNLOAD_AFTER = 50

    def __init__(self, device: str = "cuda:0", config=None):
        self.device = device
        from backend.config import ProcessingConfig
        self.config = config or ProcessingConfig()
        self._sttn = STTNInpainter(device, self.config)
        self._lama: Optional[LAMAInpainter] = None
        self._tbe_streak: int = 0

    def _ensure_lama(self) -> LAMAInpainter:
        if self._lama is None:
            self._lama = LAMAInpainter(self.device, self.config)
        return self._lama

    def _maybe_unload_lama(self) -> None:
        if self._lama is None:
            return
        if self._tbe_streak < self.LAMA_IDLE_UNLOAD_AFTER:
            return
        logger.info(
            f"AUTO: unloading idle LaMa after {self._tbe_streak} TBE batches"
        )
        self._lama = None
        try:
            import gc as _gc
            _gc.collect()
        except Exception:
            logger.warning("AUTO LaMa idle GC cleanup failed", exc_info=True)
        torch_mod = sys.modules.get("torch")
        if torch_mod is None:
            return
        try:
            if hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
                torch_mod.cuda.empty_cache()
        except Exception:
            logger.warning("AUTO LaMa idle CUDA cleanup failed", exc_info=True)

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

    def inpaint(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        threshold = self.config.auto_exposure_threshold
        score = self._exposure_score(masks)
        if score >= threshold:
            logger.debug(f"AUTO: TBE path (exposure={score:.2f} >= {threshold:.2f})")
            self._tbe_streak += 1
            self._maybe_unload_lama()
            return self._sttn.inpaint(frames, masks)
        logger.debug(f"AUTO: LaMa path (exposure={score:.2f} < {threshold:.2f})")
        self._tbe_streak = 0
        return self._ensure_lama().inpaint(frames, masks)
