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
        from backend.config import ProcessingConfig
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
        tile_size = self.config.lama_tile_size
        tile_overlap = self.config.lama_tile_overlap
        results = []
        for frame, mask in zip(frames, masks):
            if mask.max() == 0:
                results.append(frame.copy())
                continue
            h, w = frame.shape[:2]
            if h > tile_size or w > tile_size:
                try:
                    results.append(self._inpaint_lama_tiled(
                        frame, mask, tile_size, tile_overlap))
                    continue
                except Exception as exc:
                    logger.warning(f"Tiled LaMa fell back to full-frame: {exc}")
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

    def _inpaint_lama_tiled(self, frame: np.ndarray, mask: np.ndarray,
                            tile_size: int, overlap: int) -> np.ndarray:
        """Tile the masked region into overlapping patches, inpaint each,
        and blend the results with a raised-cosine window."""
        from PIL import Image
        h, w = frame.shape[:2]
        ys = mask.any(axis=1)
        xs = mask.any(axis=0)
        if not ys.any():
            return frame.copy()
        y_indices = np.where(ys)[0]
        x_indices = np.where(xs)[0]
        roi_y1 = max(0, int(y_indices[0]) - overlap)
        roi_y2 = min(h, int(y_indices[-1]) + 1 + overlap)
        roi_x1 = max(0, int(x_indices[0]) - overlap)
        roi_x2 = min(w, int(x_indices[-1]) + 1 + overlap)
        step = max(1, tile_size - overlap)
        result = frame.copy()
        weight_acc = np.zeros((h, w), dtype=np.float32)
        color_acc = np.zeros_like(frame, dtype=np.float32)
        tile_count = 0
        for ty in range(roi_y1, roi_y2, step):
            for tx in range(roi_x1, roi_x2, step):
                ty2 = min(ty + tile_size, h)
                tx2 = min(tx + tile_size, w)
                ty1 = max(0, ty2 - tile_size)
                tx1 = max(0, tx2 - tile_size)
                tile_mask = mask[ty1:ty2, tx1:tx2]
                if tile_mask.max() == 0:
                    continue
                tile_frame = frame[ty1:ty2, tx1:tx2]
                tile_rgb = cv2.cvtColor(tile_frame, cv2.COLOR_BGR2RGB)
                pil_tile = Image.fromarray(tile_rgb)
                pil_mask = Image.fromarray(tile_mask)
                try:
                    pil_out = self._lama(pil_tile, pil_mask)
                    tile_out = cv2.cvtColor(np.array(pil_out), cv2.COLOR_RGB2BGR)
                except Exception:
                    tile_out = _cv2_inpaint(tile_frame, tile_mask, 7, cv2.INPAINT_NS)
                th, tw = tile_out.shape[:2]
                wy = np.ones(th, dtype=np.float32)
                wx = np.ones(tw, dtype=np.float32)
                if overlap > 0:
                    ramp = min(overlap, th // 2, tw // 2)
                    if ramp > 0:
                        taper = 0.5 - 0.5 * np.cos(
                            np.linspace(0, np.pi, ramp, dtype=np.float32))
                        wy[:ramp] *= taper
                        wy[-ramp:] *= taper[::-1]
                        wx[:ramp] *= taper
                        wx[-ramp:] *= taper[::-1]
                win = np.outer(wy, wx)
                color_acc[ty1:ty2, tx1:tx2] += tile_out.astype(np.float32) * win[..., None]
                weight_acc[ty1:ty2, tx1:tx2] += win
                tile_count += 1
        if tile_count > 0:
            blend_mask = weight_acc > 0
            for c in range(3):
                result[:, :, c] = np.where(
                    blend_mask,
                    (color_acc[:, :, c] / np.maximum(weight_acc, 1e-6)).clip(0, 255),
                    frame[:, :, c],
                )
            result = result.astype(np.uint8)
            logger.debug(f"Tiled LaMa: {tile_count} tiles, roi {roi_x2-roi_x1}x{roi_y2-roi_y1}")
        return result

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
