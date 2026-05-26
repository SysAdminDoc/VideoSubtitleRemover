"""Inpainter subpackage.

Re-exports the four built-in inpainters plus the BaseInpainter ABC and
shared mask-conditioning helpers. Registration with the plugin registry
(via ``inpainter_registry.register``) is done from
``backend/processor.py`` at module load time so the order is
deterministic and tests can predict which name resolves where.
"""

from backend.inpainters._common import (
    BaseInpainter,
    _cv2_inpaint,
    _feather_blend,
    _edge_ring_color_correct,
    _expand_mask_by_color,
    _detect_scene_cuts,
    _detect_scene_cuts_pyscenedetect,
    _farneback_winsize,
    _warp_to_reference,
    _warp_mask_to_reference,
    _tbe_single_segment,
    _temporal_background_expose,
)
from backend.inpainters.sttn import STTNInpainter
from backend.inpainters.lama import LAMAInpainter
from backend.inpainters.propainter import ProPainterInpainter
from backend.inpainters.auto import AutoInpainter

__all__ = [
    "BaseInpainter",
    "STTNInpainter",
    "LAMAInpainter",
    "ProPainterInpainter",
    "AutoInpainter",
    "_cv2_inpaint",
    "_feather_blend",
    "_edge_ring_color_correct",
    "_expand_mask_by_color",
    "_detect_scene_cuts",
    "_detect_scene_cuts_pyscenedetect",
    "_farneback_winsize",
    "_warp_to_reference",
    "_warp_mask_to_reference",
    "_tbe_single_segment",
    "_temporal_background_expose",
]
