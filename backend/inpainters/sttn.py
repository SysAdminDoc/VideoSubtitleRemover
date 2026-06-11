"""STTN-style temporal background exposure inpainter."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from backend.inpainters._common import (
    BaseInpainter,
    _cv2_inpaint,
    _edge_ring_color_correct,
    _feather_blend,
    _temporal_background_expose,
)


class STTNInpainter(BaseInpainter):
    """Temporal-propagation video inpainting via Temporal Background
    Exposure. Falls back to cv2.inpaint only for pixels masked in every
    frame of the batch.
    """

    def __init__(self, device: str = "cuda:0", config=None):
        self.device = device
        from backend.config import ProcessingConfig
        self.config = config or ProcessingConfig()

    def inpaint(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        if self.config.tbe_enable and len(frames) > 1:
            return _temporal_background_expose(
                frames, masks,
                min_coverage=max(1, self.config.tbe_min_coverage),
                use_median=self.config.tbe_use_median,
                feather_px=self.config.mask_feather_px,
                edge_ring_px=self.config.edge_ring_px,
                flow_warp=self.config.tbe_flow_warp,
                scene_cut_split=self.config.tbe_scene_cut_split,
                scene_cut_threshold=self.config.tbe_scene_cut_threshold,
                scene_cut_use_pyscenedetect=self.config.tbe_scene_cut_use_pyscenedetect,
                scene_cut_use_transnetv2=self.config.tbe_scene_cut_use_transnetv2,
            )
        out = []
        for f, m in zip(frames, masks):
            filled = _cv2_inpaint(f, m, 3, cv2.INPAINT_TELEA)
            if self.config.edge_ring_px > 0:
                filled = _edge_ring_color_correct(f, filled, m, self.config.edge_ring_px)
            out.append(_feather_blend(f, filled, m, self.config.mask_feather_px))
        return out
