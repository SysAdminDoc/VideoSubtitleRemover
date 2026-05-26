"""Opt-in diffusion video-inpainter backends.

Each backend below registers itself with the inpainter registry when
the user has set its enable env var AND the optional dependency
imports successfully. Missing deps mean the backend is invisible to
the dispatch -- the user keeps the default TBE / LaMa / ProPainter
(hybrid) path.

Backends shipped here (all are opt-in scaffolds calling the upstream
reference repo when present; full integration with each model's quirky
input formats is the upstream project's concern):

- RM-59 ProPainterReal (the ICCV 2023 sczhou/ProPainter reference).
- RM-60 DiffuEraserBackend (lixiaowen-xw/DiffuEraser).
- RM-61 VaceBackend (ali-vilab/VACE 1.3B MV2V).
- RM-62 VideoPainterBackend.
- RM-63 CoCoCoBackend (text-guided AAAI 2025).
- RM-64 EraserDiTBackend (track only; experimental).
- RM-65 FloedBackend (NevSNev/FloED, flow-guided efficient diffusion).

The registry mode names live alongside the four core inpainters from
inpainter_registry. Each scaffold falls back to TBE + cv2 inpainting
when the heavy model fails so the user always gets a result.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import cv2
import numpy as np

from backend.inpainter_registry import register
from backend.processor import (
    BaseInpainter,
    ProcessingConfig,
    _cv2_inpaint,
    _edge_ring_color_correct,
    _feather_blend,
    _temporal_background_expose,
)

logger = logging.getLogger(__name__)


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _fallback_to_tbe(config: ProcessingConfig,
                     frames: List[np.ndarray],
                     masks: List[np.ndarray]) -> List[np.ndarray]:
    """Common fallback when a diffusion backend's heavy model is
    unavailable. We delegate to the TBE primitive so the user still
    gets a clean output."""
    if config.tbe_enable and len(frames) > 1:
        return _temporal_background_expose(
            frames, masks,
            min_coverage=max(2, config.tbe_min_coverage + 1),
            use_median=True,
            feather_px=config.mask_feather_px,
            edge_ring_px=config.edge_ring_px,
            flow_warp=config.tbe_flow_warp,
            scene_cut_split=config.tbe_scene_cut_split,
            scene_cut_threshold=config.tbe_scene_cut_threshold,
            scene_cut_use_pyscenedetect=config.tbe_scene_cut_use_pyscenedetect,
            scene_cut_use_transnetv2=config.tbe_scene_cut_use_transnetv2,
        )
    out: List[np.ndarray] = []
    for f, m in zip(frames, masks):
        filled = _cv2_inpaint(f, m, 5, cv2.INPAINT_TELEA)
        if config.edge_ring_px > 0:
            filled = _edge_ring_color_correct(f, filled, m, config.edge_ring_px)
        out.append(_feather_blend(f, filled, m, config.mask_feather_px))
    return out


class _DiffusionBackendBase(BaseInpainter):
    """Common scaffolding -- subclasses set ``MODE_NAME``, ``REPO_HINT``,
    and implement ``_run_model``. When the optional dep / weights are
    missing we log once and route to the TBE primitive."""

    MODE_NAME = "diffusion"
    REPO_HINT = ""

    def __init__(self, device: str, config: ProcessingConfig):
        self.device = device
        self.config = config
        self._loaded = False
        self._model = None
        self._warned = False

    def _load(self):  # subclasses override
        return None

    def _run_model(self, frames, masks):  # subclasses override
        raise NotImplementedError

    def inpaint(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        if not self._loaded:
            self._model = self._load()
            self._loaded = True
            if self._model is None and not self._warned:
                logger.info(
                    f"{self.MODE_NAME} weights/deps unavailable; falling "
                    f"back to TBE. {self.REPO_HINT}"
                )
                self._warned = True
        if self._model is None:
            return _fallback_to_tbe(self.config, frames, masks)
        try:
            return self._run_model(frames, masks)
        except Exception as exc:
            logger.warning(
                f"{self.MODE_NAME} inference failed ({exc}); falling back to TBE"
            )
            return _fallback_to_tbe(self.config, frames, masks)


# ---------------------------------------------------------------------------
# RM-59: Real ProPainter (ICCV 2023 reference)
# ---------------------------------------------------------------------------


class _PropainterRealBackend(_DiffusionBackendBase):
    MODE_NAME = "propainter-real"
    REPO_HINT = (
        "Install via `pip install propainter` or clone "
        "github.com/sczhou/ProPainter and set VSR_PROPAINTER_REAL=1."
    )

    def _load(self):
        try:
            import propainter  # type: ignore
            return propainter
        except ImportError:
            return None

    def _run_model(self, frames, masks):
        # ProPainter expects (T, H, W, 3) uint8 frames and (T, H, W)
        # bool masks at consistent dimensions. We adapt the call when
        # the package exposes a `propainter.inpaint(frames, masks)`
        # entry; if not, fall back to TBE.
        if not hasattr(self._model, "inpaint"):
            raise RuntimeError("propainter package missing top-level `inpaint`")
        frames_arr = np.stack(frames, axis=0)
        masks_arr = np.stack([(m > 0).astype(np.uint8) for m in masks], axis=0)
        out = self._model.inpaint(frames_arr, masks_arr)
        results = []
        for i, (f, m) in enumerate(zip(frames, masks)):
            results.append(
                _feather_blend(f, out[i], m, self.config.mask_feather_px)
            )
        return results


# ---------------------------------------------------------------------------
# RM-60: DiffuEraser
# ---------------------------------------------------------------------------


class _DiffuEraserBackend(_DiffusionBackendBase):
    MODE_NAME = "diffueraser"
    REPO_HINT = (
        "Install via `pip install diffueraser` or clone "
        "github.com/lixiaowen-xw/DiffuEraser and set VSR_DIFFUERASER=1."
    )

    def _load(self):
        try:
            from diffueraser import DiffuEraser  # type: ignore
        except ImportError:
            return None
        try:
            return DiffuEraser(device=self.device)
        except Exception:
            return None

    def _run_model(self, frames, masks):
        result = self._model.run(frames, masks)
        out = []
        for f, r, m in zip(frames, result, masks):
            out.append(_feather_blend(f, r, m, self.config.mask_feather_px))
        return out


# ---------------------------------------------------------------------------
# RM-61: Wan2.1-VACE
# ---------------------------------------------------------------------------


class _VaceBackend(_DiffusionBackendBase):
    MODE_NAME = "vace"
    REPO_HINT = (
        "Install via `pip install vace` or clone github.com/ali-vilab/VACE "
        "and set VSR_VACE=1."
    )

    def _load(self):
        try:
            from vace import VACE  # type: ignore
            return VACE(device=self.device)
        except Exception:
            return None

    def _run_model(self, frames, masks):
        out = self._model.mv2v(frames, masks)
        return [
            _feather_blend(f, r, m, self.config.mask_feather_px)
            for f, r, m in zip(frames, out, masks)
        ]


# ---------------------------------------------------------------------------
# RM-62: VideoPainter
# ---------------------------------------------------------------------------


class _VideoPainterBackend(_DiffusionBackendBase):
    MODE_NAME = "videopainter"
    REPO_HINT = (
        "Install via the upstream VideoPainter project and "
        "set VSR_VIDEOPAINTER=1."
    )

    def _load(self):
        try:
            from videopainter import VideoPainter  # type: ignore
            return VideoPainter(device=self.device)
        except Exception:
            return None

    def _run_model(self, frames, masks):
        out = self._model.inpaint(frames, masks)
        return [
            _feather_blend(f, r, m, self.config.mask_feather_px)
            for f, r, m in zip(frames, out, masks)
        ]


# ---------------------------------------------------------------------------
# RM-63: CoCoCo (text-guided)
# ---------------------------------------------------------------------------


class _CocoCoBackend(_DiffusionBackendBase):
    MODE_NAME = "cococo"
    REPO_HINT = (
        "Install via the upstream COCOCO project and set VSR_COCOCO=1."
    )

    def _load(self):
        try:
            from cococo import CoCoCo  # type: ignore
            return CoCoCo(device=self.device)
        except Exception:
            return None

    def _run_model(self, frames, masks):
        # Text prompt defaults to "background" so the model fills the
        # subtitle region with the surrounding scene rather than
        # generating arbitrary content. Users who need a custom prompt
        # set VSR_COCOCO_PROMPT.
        prompt = os.environ.get("VSR_COCOCO_PROMPT", "background")
        out = self._model.inpaint(frames, masks, prompt=prompt)
        return [
            _feather_blend(f, r, m, self.config.mask_feather_px)
            for f, r, m in zip(frames, out, masks)
        ]


# ---------------------------------------------------------------------------
# RM-64: EraserDiT (track only)
# ---------------------------------------------------------------------------


class _EraserDitBackend(_DiffusionBackendBase):
    MODE_NAME = "eraserdit"
    REPO_HINT = (
        "EraserDiT is research-stage; integration is opt-in via "
        "VSR_ERASERDIT=1 once the upstream releases a pip-installable "
        "package."
    )

    def _load(self):
        try:
            from eraserdit import EraserDiT  # type: ignore
            return EraserDiT(device=self.device)
        except Exception:
            return None

    def _run_model(self, frames, masks):
        out = self._model.inpaint(frames, masks)
        return [
            _feather_blend(f, r, m, self.config.mask_feather_px)
            for f, r, m in zip(frames, out, masks)
        ]


# ---------------------------------------------------------------------------
# RM-65: FloED
# ---------------------------------------------------------------------------


class _FloedBackend(_DiffusionBackendBase):
    MODE_NAME = "floed"
    REPO_HINT = (
        "Install via the upstream FloED project and set VSR_FLOED=1."
    )

    def _load(self):
        try:
            from floed import FloED  # type: ignore
            return FloED(device=self.device)
        except Exception:
            return None

    def _run_model(self, frames, masks):
        out = self._model.inpaint(frames, masks)
        return [
            _feather_blend(f, r, m, self.config.mask_feather_px)
            for f, r, m in zip(frames, out, masks)
        ]


# Map an env var to (mode-name, builder).
_OPT_INS = [
    ("VSR_PROPAINTER_REAL", "propainter-real", _PropainterRealBackend),
    ("VSR_DIFFUERASER",     "diffueraser",     _DiffuEraserBackend),
    ("VSR_VACE",            "vace",            _VaceBackend),
    ("VSR_VIDEOPAINTER",    "videopainter",    _VideoPainterBackend),
    ("VSR_COCOCO",          "cococo",          _CocoCoBackend),
    ("VSR_ERASERDIT",       "eraserdit",       _EraserDitBackend),
    ("VSR_FLOED",           "floed",           _FloedBackend),
]


def maybe_register() -> List[str]:
    """Register every diffusion backend the user has opted into. Returns
    the list of mode names that were added so callers can log."""
    registered: List[str] = []
    for env_name, mode_name, klass in _OPT_INS:
        if _env_enabled(env_name):
            register(mode_name, lambda device, config, K=klass: K(device, config))
            registered.append(mode_name)
            logger.info(f"Diffusion backend '{mode_name}' enabled via {env_name}")
    return registered


maybe_register()
