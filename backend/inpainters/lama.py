"""LaMa neural inpainter via simple-lama-inpainting + optional batched
forward pass (RM-40)."""

from __future__ import annotations

import logging
import os
from typing import List

import cv2
import numpy as np

from backend.inpainters._common import (
    BaseInpainter,
    _cv2_inpaint,
    _edge_ring_color_correct,
    _feather_blend,
)

logger = logging.getLogger(__name__)


class LAMAInpainter(BaseInpainter):
    """LAMA-based image inpainting. Uses simple-lama-inpainting when
    available; falls back to cv2.inpaint when not."""

    def __init__(self, device: str = "cuda:0", config=None):
        self.device = device
        from backend.processor import ProcessingConfig
        self.config = config or ProcessingConfig()
        self._lama = None
        self._load_model()

    def _load_model(self):
        try:
            from simple_lama_inpainting import SimpleLama
            self._lama = SimpleLama()
            logger.info("LaMa neural inpainting model loaded (simple-lama-inpainting)")
            # RM-49: best-effort SHA-256 check of the on-disk weights.
            try:
                from backend.adapter_manifest import (
                    log_adapter_verification as _log_adapter,
                    verify_adapter_path as _verify_adapter,
                )
                from backend.model_hashes import (
                    candidate_weight_dirs as _cands,
                )
                verified = True
                for cache_dir in _cands():
                    for path in cache_dir.glob("**/big-lama*.pt"):
                        result = _verify_adapter("simple-lama", str(path))
                        _log_adapter(result)
                        if not result.allowed:
                            verified = False
                        break
                if not verified:
                    self._lama = None
                    logger.warning(
                        "LaMa neural inpainting disabled because cached "
                        "weights failed manifest verification."
                    )
            except Exception as exc:
                logger.debug(f"Weight verification skipped: {exc}")
        except ImportError:
            logger.warning("simple-lama-inpainting not installed, LAMA will use OpenCV fallback. "
                          "Install with: pip install simple-lama-inpainting")
        except Exception as e:
            logger.warning(f"LaMa model load failed: {e}")

    def inpaint(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        feather = self.config.mask_feather_px
        ring = self.config.edge_ring_px
        if self._lama is not None:
            raw = self._inpaint_lama(frames, masks)
        else:
            raw = [_cv2_inpaint(f, m, 7, cv2.INPAINT_NS) for f, m in zip(frames, masks)]
        out = []
        for f, r, m in zip(frames, raw, masks):
            if ring > 0 and m.max() > 0:
                r = _edge_ring_color_correct(f, r, m, ring)
            out.append(_feather_blend(f, r, m, feather))
        return out

    def _inpaint_lama(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        """Per-frame LaMa with an opt-in batched path (RM-40 via
        VSR_LAMA_BATCH=1)."""
        if (os.environ.get("VSR_LAMA_BATCH", "").strip().lower()
                in {"1", "true", "yes", "on"}):
            try:
                return self._inpaint_lama_batched(frames, masks)
            except Exception as exc:
                logger.warning(f"Batched LaMa fell back to per-frame: {exc}")
        from PIL import Image
        results = []
        for frame, mask in zip(frames, masks):
            if mask.max() == 0:
                results.append(frame.copy())
                continue
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                pil_mask = Image.fromarray(mask)
                result_pil = self._lama(pil_image, pil_mask)
                result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
                results.append(result_bgr)
            except Exception as e:
                logger.warning(f"LaMa inpaint failed for frame, falling back to cv2: {e}")
                results.append(_cv2_inpaint(frame, mask, 7, cv2.INPAINT_NS))
        return results

    def _inpaint_lama_batched(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        """RM-40: stack the batch + one forward pass through the underlying
        torch model. Raises on shape mismatch so caller falls back to
        per-frame."""
        import torch  # type: ignore
        model = getattr(self._lama, "model", None) or getattr(self._lama, "_model", None)
        if model is None:
            raise RuntimeError("simple-lama-inpainting model attribute not exposed")
        h, w = frames[0].shape[:2]
        if any(f.shape[:2] != (h, w) for f in frames):
            raise RuntimeError("inconsistent frame shapes in batch")
        ph = ((h + 7) // 8) * 8
        pw = ((w + 7) // 8) * 8
        imgs = []
        msks = []
        had_mask: List[bool] = []
        for f, m in zip(frames, masks):
            had_mask.append(m.max() > 0)
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            if (ph, pw) != (h, w):
                rgb = cv2.copyMakeBorder(rgb, 0, ph - h, 0, pw - w, cv2.BORDER_REFLECT_101)
                m_pad = cv2.copyMakeBorder(m, 0, ph - h, 0, pw - w, cv2.BORDER_CONSTANT, value=0)
            else:
                m_pad = m
            imgs.append((rgb.astype(np.float32) / 255.0).transpose(2, 0, 1))
            msks.append((m_pad.astype(np.float32) / 255.0)[None, ...])
        img_t = torch.from_numpy(np.stack(imgs, axis=0))
        mask_t = torch.from_numpy(np.stack(msks, axis=0))
        device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")
        img_t = img_t.to(device)
        mask_t = mask_t.to(device)
        with torch.no_grad():
            out = model(img_t, mask_t)
        out = out.clamp(0.0, 1.0).cpu().numpy()
        results: List[np.ndarray] = []
        for i, frame in enumerate(frames):
            if not had_mask[i]:
                results.append(frame.copy())
                continue
            rgb_out = (out[i].transpose(1, 2, 0) * 255.0).astype(np.uint8)
            bgr = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
            results.append(bgr[:h, :w])
        return results
