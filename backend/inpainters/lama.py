"""LaMa neural inpainter -- ONNX Runtime preferred, OpenCV 5 DNN second,
opt-in simple-lama-inpainting (PyTorch) fallback, cv2 last resort.

Priority chain:
1. ONNX Runtime   -- fastest, most flexible EP selection (CUDA/DirectML/CPU)
2. OpenCV 5 DNN   -- no torch, no onnxruntime; uses opencv/inpainting_lama
3. PyTorch         -- simple-lama-inpainting; optional opt-in dependency
4. cv2.inpaint     -- always available

The ONNX and DNN paths eliminate the torch.load CVE surface. The DNN path
activates automatically when opencv-python >= 5.0 is installed and an
inpainting_lama ONNX weight file is found.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from backend.import_safety import module_can_import as _module_can_import
from backend.inpainters._common import (
    BaseInpainter,
    _cv2_inpaint,
    _edge_ring_color_correct,
    _feather_blend,
    _temporal_smooth_inpainted,
)

logger = logging.getLogger(__name__)

_ONNX_SEARCH_FILENAMES = ("lama_fp32.onnx", "lama.onnx")

_CV2DNN_LAMA_FILENAMES = (
    "inpainting_lama_2025jan.onnx",
    "lama_fp32.onnx",
    "lama.onnx",
)

_OPENCV_NATIVE_NAMES = frozenset({"inpainting_lama_2025jan.onnx"})

_OPENCV5_MIN = (5, 0, 0)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _pytorch_lama_allowed() -> bool:
    return _env_flag("VSR_ENABLE_PYTORCH_LAMA")


def _find_lama_onnx_weight() -> Optional[str]:
    """Auto-discover a LaMa ONNX weight file. Resolution order:
    1. VSR_LAMA_ONNX env var (explicit)
    2. App model cache (%APPDATA%/VideoSubtitleRemoverPro/models/)
    3. Known weight cache dirs (torch hub, simple_lama)
    """
    explicit = os.environ.get("VSR_LAMA_ONNX", "").strip()
    if explicit and Path(explicit).is_file():
        return explicit

    search_dirs = []
    appdata = os.environ.get("APPDATA")
    if appdata:
        search_dirs.append(
            Path(appdata) / "VideoSubtitleRemoverPro" / "models"
        )
    home = Path.home()
    search_dirs.append(home / ".cache" / "torch" / "hub" / "checkpoints")
    search_dirs.append(home / ".cache" / "simple_lama")
    search_dirs.append(home / ".cache" / "huggingface" / "hub")

    for d in search_dirs:
        if not d.is_dir():
            continue
        for name in _ONNX_SEARCH_FILENAMES:
            candidate = d / name
            if candidate.is_file():
                return str(candidate)
            for match in d.rglob(name):
                if match.is_file():
                    return str(match)
    return None


def _opencv_version_tuple() -> Tuple[int, ...]:
    """Parse cv2.__version__ into a comparable tuple of ints."""
    parts = cv2.__version__.split(".")
    result = []
    for p in parts[:3]:
        digits = ""
        for ch in p:
            if ch.isdigit():
                digits += ch
            else:
                break
        result.append(int(digits) if digits else 0)
    while len(result) < 3:
        result.append(0)
    return tuple(result)


def _opencv5_available() -> bool:
    """Return True when OpenCV >= 5.0 is installed."""
    return _opencv_version_tuple() >= _OPENCV5_MIN


def _find_opencv_lama_weight() -> Optional[str]:
    """Auto-discover an OpenCV-compatible LaMa ONNX weight file.

    Resolution order:
    1. VSR_OPENCV_LAMA env var (explicit)
    2. App model cache (%APPDATA%/VideoSubtitleRemoverPro/models/)
    3. HuggingFace hub cache
    4. OpenCV model cache
    5. Torch hub cache
    """
    explicit = os.environ.get("VSR_OPENCV_LAMA", "").strip()
    if explicit and Path(explicit).is_file():
        return explicit

    search_dirs = []
    appdata = os.environ.get("APPDATA")
    if appdata:
        search_dirs.append(
            Path(appdata) / "VideoSubtitleRemoverPro" / "models"
        )
    home = Path.home()
    search_dirs.append(home / ".cache" / "huggingface" / "hub")
    search_dirs.append(home / ".cache" / "opencv_models")
    search_dirs.append(home / ".cache" / "torch" / "hub" / "checkpoints")
    search_dirs.append(home / ".cache" / "simple_lama")

    for d in search_dirs:
        if not d.is_dir():
            continue
        for name in _CV2DNN_LAMA_FILENAMES:
            candidate = d / name
            if candidate.is_file():
                return str(candidate)
            for match in d.rglob(name):
                if match.is_file():
                    return str(match)
    return None


def _try_cv2dnn_net(
    model_path: str, device: str
) -> Optional["cv2.dnn.Net"]:
    """Load a cv2.dnn.Net from an ONNX LaMa model file.

    Returns the Net on success or None on any failure so the caller can
    fall through to the next backend in the priority chain.
    """
    if not _opencv5_available():
        return None

    filename = Path(model_path).name
    adapter_name = (
        "opencv-lama" if filename in _OPENCV_NATIVE_NAMES else "lama-onnx"
    )
    try:
        from backend.adapter_manifest import (
            log_adapter_verification,
            verify_adapter_path,
        )
        result = verify_adapter_path(adapter_name, model_path)
        log_adapter_verification(result)
        if not result.allowed:
            return None
    except Exception as exc:
        logger.debug("OpenCV LaMa adapter verification skipped: %s", exc)

    try:
        net = cv2.dnn.readNetFromONNX(model_path)
    except Exception as exc:
        logger.info("cv2.dnn.readNetFromONNX failed for LaMa: %s", exc)
        return None

    if "cuda" in device:
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            logger.info("OpenCV DNN LaMa using CUDA backend")
        except Exception:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    return net


def _try_onnx_session(model_path: str, device: str):
    """Attempt to create an ONNX Runtime InferenceSession for LaMa.
    Returns (session, provider_name) or (None, None)."""
    try:
        import onnxruntime as ort
    except ImportError:
        return None, None
    try:
        from backend.adapter_manifest import (
            log_adapter_verification,
            verify_adapter_path,
        )
        result = verify_adapter_path("lama-onnx", model_path)
        log_adapter_verification(result)
        if not result.allowed:
            return None, None
    except Exception as exc:
        logger.debug("LaMa-ONNX adapter verification skipped: %s", exc)

    try:
        from backend.inpainters_onnx import (
            _providers_after_opset_audit,
            _providers_for_device,
        )
        from backend.onnxruntime_cuda import preload_onnxruntime_cuda_dlls_if_needed
        providers = _providers_after_opset_audit(
            model_path, _providers_for_device(device)
        )
        preload_onnxruntime_cuda_dlls_if_needed(ort, providers)
    except Exception:
        providers = ["CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(model_path, providers=providers)
        active = session.get_providers()
        provider = active[0] if active else "unknown"
        return session, provider
    except Exception as exc:
        logger.info("LaMa-ONNX session creation failed: %s", exc)
        return None, None


class LAMAInpainter(BaseInpainter):
    """LaMa inpainter with four-tier backend priority:
    ONNX Runtime > OpenCV 5 DNN > PyTorch > cv2.inpaint."""

    INPUT_NAME = "image"
    MASK_NAME = "mask"

    def __init__(self, device: str = "cuda:0", config=None):
        self.device = device
        from backend.config import ProcessingConfig
        self.config = config or ProcessingConfig()
        self._onnx_session = None
        self._dnn_net = None
        self._lama = None
        self._backend_name = "cv2"
        self._load_model()

    def _load_model(self):
        onnx_path = _find_lama_onnx_weight()
        if onnx_path:
            session, provider = _try_onnx_session(onnx_path, self.device)
            if session is not None:
                self._onnx_session = session
                self._backend_name = "ONNX (%s)" % provider
                logger.info(
                    "LaMa ONNX Runtime inpainting loaded via %s from %s",
                    provider, onnx_path,
                )
                return

        if _opencv5_available():
            opencv_path = _find_opencv_lama_weight()
            if opencv_path:
                net = _try_cv2dnn_net(opencv_path, self.device)
                if net is not None:
                    self._dnn_net = net
                    self._backend_name = "OpenCV DNN"
                    logger.info(
                        "LaMa OpenCV 5 DNN inpainting loaded from %s",
                        opencv_path,
                    )
                    return

        if not _pytorch_lama_allowed():
            logger.warning(
                "No ONNX/OpenCV LaMa backend is available. The PyTorch "
                "simple-lama fallback is disabled by default because broken "
                "native torch wheels can crash the process; set "
                "VSR_ENABLE_PYTORCH_LAMA=1 to opt in."
            )
            return

        if not _module_can_import(
            "simple_lama_inpainting",
            logger=logger,
            failure_context="LaMa PyTorch fallback disabled",
        ):
            logger.warning(
                "No LaMa backend available (onnxruntime, OpenCV 5 DNN, or "
                "simple-lama-inpainting). LAMA will use OpenCV fallback."
            )
            return

        try:
            from simple_lama_inpainting import SimpleLama
            self._lama = SimpleLama()
            self._backend_name = "PyTorch (simple-lama-inpainting)"
            logger.info("LaMa PyTorch inpainting loaded (simple-lama-inpainting)")
            self._verify_pytorch_weights()
        except (ImportError, OSError, RuntimeError) as exc:
            logger.warning(
                "simple-lama-inpainting import failed; LAMA will use OpenCV "
                "fallback: %s",
                exc,
            )
        except Exception as e:
            logger.warning("LaMa model load failed: %s", e)

    def _verify_pytorch_weights(self):
        try:
            from backend.adapter_manifest import (
                log_adapter_verification as _log_adapter,
                verify_adapter_path as _verify_adapter,
            )
            from backend.model_hashes import (
                candidate_weight_dirs as _cands,
            )
            for cache_dir in _cands():
                for path in cache_dir.glob("**/big-lama*.pt"):
                    result = _verify_adapter("simple-lama", str(path))
                    _log_adapter(result)
                    if not result.allowed:
                        self._lama = None
                        self._backend_name = "cv2"
                        logger.warning(
                            "LaMa neural inpainting disabled because cached "
                            "weights failed manifest verification."
                        )
                    return
        except Exception as exc:
            logger.debug("Weight verification skipped: %s", exc)

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def inpaint(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        feather = self.config.mask_feather_px
        ring = self.config.edge_ring_px
        if self._onnx_session is not None:
            raw = self._inpaint_onnx(frames, masks)
        elif self._dnn_net is not None:
            raw = self._inpaint_cv2dnn(frames, masks)
        elif self._lama is not None:
            raw = self._inpaint_pytorch(frames, masks)
        else:
            raw = [_cv2_inpaint(f, m, 7, cv2.INPAINT_NS) for f, m in zip(frames, masks)]
        out = []
        for f, r, m in zip(frames, raw, masks):
            if ring > 0 and m.max() > 0:
                r = _edge_ring_color_correct(f, r, m, ring)
            out.append(_feather_blend(f, r, m, feather))
        smooth = self.config.temporal_smooth_radius
        if smooth > 0 and len(out) > 1:
            out = _temporal_smooth_inpainted(out, masks, radius=smooth)
        return out

    def _inpaint_onnx(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
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
                    results.append(self._inpaint_onnx_tiled(
                        frame, mask, tile_size, tile_overlap))
                    continue
                except Exception as exc:
                    logger.warning("Tiled LaMa-ONNX fell back to full-frame: %s", exc)
            try:
                results.append(self._inpaint_onnx_one(frame, mask))
            except Exception as exc:
                logger.warning("LaMa-ONNX inference failed, falling back to cv2: %s", exc)
                results.append(_cv2_inpaint(frame, mask, 7, cv2.INPAINT_NS))
        return results

    def _inpaint_onnx_one(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        ph = _ensure_multiple_of(h, 8)
        pw = _ensure_multiple_of(w, 8)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if (ph, pw) != (h, w):
            rgb = cv2.copyMakeBorder(rgb, 0, ph - h, 0, pw - w, cv2.BORDER_REFLECT_101)
            mask_padded = cv2.copyMakeBorder(mask, 0, ph - h, 0, pw - w, cv2.BORDER_CONSTANT, value=0)
        else:
            mask_padded = mask
        img_t = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        mask_t = (mask_padded.astype(np.float32) / 255.0)[None, None, ...]
        out = self._onnx_session.run(
            None, {self.INPUT_NAME: img_t, self.MASK_NAME: mask_t}
        )[0]
        bgr = cv2.cvtColor(
            (out[0].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        return bgr[:h, :w]

    def _inpaint_onnx_tiled(self, frame: np.ndarray, mask: np.ndarray,
                            tile_size: int, overlap: int) -> np.ndarray:
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
                try:
                    tile_out = self._inpaint_onnx_one(tile_frame, tile_mask)
                except Exception:
                    logger.warning(
                        "LaMa-ONNX tile inference failed, falling back to cv2",
                        exc_info=True,
                    )
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
        return result

    # ------------------------------------------------------------------
    # OpenCV 5 DNN path
    # ------------------------------------------------------------------

    def _inpaint_cv2dnn(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
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
                    results.append(self._inpaint_cv2dnn_tiled(
                        frame, mask, tile_size, tile_overlap))
                    continue
                except Exception as exc:
                    logger.warning(
                        "Tiled LaMa cv2.dnn fell back to full-frame: %s", exc)
            try:
                results.append(self._inpaint_cv2dnn_one(frame, mask))
            except Exception as exc:
                logger.warning(
                    "LaMa cv2.dnn inference failed, falling back to cv2: %s",
                    exc)
                results.append(_cv2_inpaint(frame, mask, 7, cv2.INPAINT_NS))
        return results

    def _inpaint_cv2dnn_one(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        ph = _ensure_multiple_of(h, 8)
        pw = _ensure_multiple_of(w, 8)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if (ph, pw) != (h, w):
            rgb = cv2.copyMakeBorder(
                rgb, 0, ph - h, 0, pw - w, cv2.BORDER_REFLECT_101)
            mask_padded = cv2.copyMakeBorder(
                mask, 0, ph - h, 0, pw - w, cv2.BORDER_CONSTANT, value=0)
        else:
            mask_padded = mask
        img_blob = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[
            None, ...
        ]
        mask_blob = (mask_padded.astype(np.float32) / 255.0)[None, None, ...]
        self._dnn_net.setInput(img_blob, self.INPUT_NAME)
        self._dnn_net.setInput(mask_blob, self.MASK_NAME)
        out = self._dnn_net.forward()
        bgr = cv2.cvtColor(
            (out[0].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(
                np.uint8
            ),
            cv2.COLOR_RGB2BGR,
        )
        return bgr[:h, :w]

    def _inpaint_cv2dnn_tiled(self, frame: np.ndarray, mask: np.ndarray,
                              tile_size: int, overlap: int) -> np.ndarray:
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
                try:
                    tile_out = self._inpaint_cv2dnn_one(tile_frame, tile_mask)
                except Exception:
                    tile_out = _cv2_inpaint(
                        tile_frame, tile_mask, 7, cv2.INPAINT_NS)
                th, tw = tile_out.shape[:2]
                wy = np.ones(th, dtype=np.float32)
                wx = np.ones(tw, dtype=np.float32)
                if overlap > 0:
                    ramp = min(overlap, th // 2, tw // 2)
                    if ramp > 0:
                        taper = 0.5 - 0.5 * np.cos(
                            np.linspace(
                                0, np.pi, ramp, dtype=np.float32))
                        wy[:ramp] *= taper
                        wy[-ramp:] *= taper[::-1]
                        wx[:ramp] *= taper
                        wx[-ramp:] *= taper[::-1]
                win = np.outer(wy, wx)
                color_acc[ty1:ty2, tx1:tx2] += (
                    tile_out.astype(np.float32) * win[..., None])
                weight_acc[ty1:ty2, tx1:tx2] += win
                tile_count += 1
        if tile_count > 0:
            blend_mask = weight_acc > 0
            for c in range(3):
                result[:, :, c] = np.where(
                    blend_mask,
                    (color_acc[:, :, c] / np.maximum(
                        weight_acc, 1e-6)).clip(0, 255),
                    frame[:, :, c],
                )
            result = result.astype(np.uint8)
        return result

    # ------------------------------------------------------------------
    # PyTorch path (simple-lama-inpainting fallback)
    # ------------------------------------------------------------------

    def _inpaint_pytorch(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        if (os.environ.get("VSR_LAMA_BATCH", "").strip().lower()
                in {"1", "true", "yes", "on"}):
            try:
                return self._inpaint_pytorch_batched(frames, masks)
            except Exception as exc:
                logger.warning(
                    "Batched LaMa fell back to per-frame: %s",
                    exc,
                    exc_info=True,
                )
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
                    results.append(self._inpaint_pytorch_tiled(
                        frame, mask, tile_size, tile_overlap))
                    continue
                except Exception as exc:
                    logger.warning(
                        "Tiled LaMa fell back to full-frame: %s",
                        exc,
                        exc_info=True,
                    )
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                pil_mask = Image.fromarray(mask)
                result_pil = self._lama(pil_image, pil_mask)
                result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
                results.append(result_bgr)
            except Exception as e:
                logger.warning(
                    "LaMa inpaint failed for frame, falling back to cv2: %s",
                    e,
                    exc_info=True,
                )
                results.append(_cv2_inpaint(frame, mask, 7, cv2.INPAINT_NS))
        return results

    def _inpaint_pytorch_tiled(self, frame: np.ndarray, mask: np.ndarray,
                               tile_size: int, overlap: int) -> np.ndarray:
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
                    logger.warning(
                        "LaMa PyTorch tile inference failed, falling back to cv2",
                        exc_info=True,
                    )
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
        return result

    def _inpaint_lama_tiled(self, frame: np.ndarray, mask: np.ndarray,
                            tile_size: int, overlap: int) -> np.ndarray:
        """Compatibility wrapper for callers using the historical name."""
        return self._inpaint_pytorch_tiled(frame, mask, tile_size, overlap)

    def _inpaint_pytorch_batched(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        if not _module_can_import(
            "torch",
            logger=logger,
            failure_context="batched LaMa disabled",
        ):
            raise RuntimeError("torch import failed safety probe")
        import torch
        model = getattr(self._lama, "model", None) or getattr(self._lama, "_model", None)
        if model is None:
            raise RuntimeError("simple-lama-inpainting model attribute not exposed")
        h, w = frames[0].shape[:2]
        if any(f.shape[:2] != (h, w) for f in frames):
            raise RuntimeError("inconsistent frame shapes in batch")
        ph = _ensure_multiple_of(h, 8)
        pw = _ensure_multiple_of(w, 8)
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


def _ensure_multiple_of(value: int, multiple: int) -> int:
    if value % multiple == 0:
        return value
    return ((value // multiple) + 1) * multiple
