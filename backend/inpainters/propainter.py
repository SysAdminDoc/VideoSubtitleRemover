"""ProPainter-style hybrid TBE + LaMa-refinement inpainter."""

from __future__ import annotations

import logging
from typing import List

import cv2
import numpy as np

from backend.inpainters.lama import (
    _module_can_import,
    _pytorch_lama_allowed,
)
from backend.inpainters._common import (
    BaseInpainter,
    _cv2_inpaint,
    _feather_blend,
    _temporal_background_expose,
)

logger = logging.getLogger(__name__)


class ProPainterInpainter(BaseInpainter):
    """Motion-robust hybrid: TBE with higher coverage bar + LaMa
    residual refinement. Designed to match ProPainter quality for
    sparse occluders without the 10+ GB VRAM footprint."""

    def __init__(self, device: str = "cuda:0", config=None):
        self.device = device
        from backend.config import ProcessingConfig
        self.config = config or ProcessingConfig()
        self._lama = None
        if not _pytorch_lama_allowed():
            logger.info(
                "ProPainter LaMa refinement skipped; set "
                "VSR_ENABLE_PYTORCH_LAMA=1 to opt in."
            )
            return
        if not _module_can_import(
            "simple_lama_inpainting",
            logger=logger,
            failure_context="ProPainter LaMa refinement disabled",
        ):
            return
        try:
            from simple_lama_inpainting import SimpleLama
            self._lama = SimpleLama()
            logger.info("ProPainter path will use LaMa for residual refinement")
        except Exception:
            pass

    def inpaint(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        feather = self.config.mask_feather_px
        if self.config.tbe_enable and len(frames) > 1:
            results = _temporal_background_expose(
                frames, masks,
                min_coverage=max(2, self.config.tbe_min_coverage + 1),
                use_median=True,
                feather_px=feather,
                edge_ring_px=self.config.edge_ring_px,
                flow_warp=self.config.tbe_flow_warp,
                scene_cut_split=self.config.tbe_scene_cut_split,
                scene_cut_threshold=self.config.tbe_scene_cut_threshold,
                scene_cut_use_pyscenedetect=self.config.tbe_scene_cut_use_pyscenedetect,
                scene_cut_use_transnetv2=self.config.tbe_scene_cut_use_transnetv2,
            )
            if self._lama is not None:
                from PIL import Image
                refined = []
                for frame, inpainted, mask in zip(frames, results, masks):
                    if mask.max() == 0:
                        refined.append(inpainted)
                        continue
                    try:
                        frame_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)
                        pil_mask = Image.fromarray(mask)
                        lama_out = self._lama(pil_image, pil_mask)
                        bgr = cv2.cvtColor(np.array(lama_out), cv2.COLOR_RGB2BGR)
                        # TBE 65 / LaMa 35 -- TBE carries the accurate
                        # background; LaMa kills ringing.
                        blend = cv2.addWeighted(inpainted, 0.65, bgr, 0.35, 0)
                        refined.append(_feather_blend(frame, blend, mask, feather))
                    except Exception:
                        logger.warning(
                            "ProPainter LaMa residual refinement failed",
                            exc_info=True,
                        )
                        refined.append(inpainted)
                return refined
            return results
        out = []
        for f, m in zip(frames, masks):
            filled = _cv2_inpaint(f, m, 5, cv2.INPAINT_TELEA)
            out.append(_feather_blend(f, filled, m, feather))
        return out
