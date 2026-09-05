"""LaMa neural inpainter with truthful provider selection.

Priority chain:
1. ONNX Runtime   -- fastest, most flexible EP selection (CUDA/DirectML/CPU)
2. OpenCV 5 DNN   -- no torch, no onnxruntime; uses opencv/inpainting_lama
3. PyTorch         -- simple-lama-inpainting; optional opt-in dependency
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
from backend.execution_provenance import (
    FAILURE_DEPENDENCY_MISSING,
    FAILURE_INITIALIZATION,
    FAILURE_RUNTIME,
    RequestedStageError,
)
from backend.inpainters._common import (
    BaseInpainter,
    _binarize_mask,
    apply_finishing,
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
    except Exception as exc:
        logger.warning(
            "LaMa provider selection for %s failed; falling back to CPU: %s",
            device,
            exc,
        )
        providers = ["CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception as exc:
        logger.info("LaMa-ONNX session creation failed: %s", exc)
        return None, None
    # RM-322: a named accelerator that quietly ran on CPU is a silent
    # substitution. This is the first rung of the LaMa ladder, though, and
    # the rungs below it (OpenCV DNN, PyTorch) may themselves be GPU
    # capable, so report the drop and let the ladder continue rather than
    # ending the whole load here.
    from backend.device_provider import (
        ProviderFellBackError,
        verify_session_provider,
    )

    try:
        verify_session_provider(
            device, session, requested_providers=providers)
    except ProviderFellBackError as exc:
        logger.warning(
            "LaMa-ONNX fell back to CPU for device %s; trying the remaining "
            "providers: %s", device, exc,
        )
        return None, None
    reader = getattr(session, "get_providers", None)
    active = list(reader()) if callable(reader) else []
    provider = active[0] if active else "unknown"
    return session, provider


class LAMAInpainter(BaseInpainter):
    """LaMa inpainter with ONNX, OpenCV DNN, and PyTorch providers."""

    INPUT_NAME = "image"
    MASK_NAME = "mask"

    def __init__(self, device: str = "cuda:0", config=None):
        self.device = device
        from backend.config import ProcessingConfig
        self.config = config or ProcessingConfig()
        self._onnx_session = None
        self._dnn_net = None
        self._lama = None
        self._backend_name = "unavailable"
        self._load_model()

    def _load_model(self):
        failed_provider = None
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
            failed_provider = "ONNX Runtime"

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
                failed_provider = "OpenCV DNN"

        if not _pytorch_lama_allowed():
            if failed_provider is not None:
                raise RequestedStageError(
                    stage="inpaint",
                    requested_implementation="lama",
                    actual_implementation="lama",
                    provider=failed_provider,
                    failure_class=FAILURE_INITIALIZATION,
                    detail=f"the reviewed {failed_provider} provider failed to load",
                    recovery_hint=(
                        "Verify the selected LaMa runtime and model artifact, then "
                        "retry or enable a reviewed alternative LaMa provider."
                    ),
                )
            raise RequestedStageError(
                stage="inpaint",
                requested_implementation="lama",
                failure_class=FAILURE_DEPENDENCY_MISSING,
                detail=(
                    "no reviewed ONNX or OpenCV DNN LaMa provider is available, "
                    "and the PyTorch provider is not enabled"
                ),
                recovery_hint=(
                    # RM-354: the old hint said to install a model and gave
                    # no way to get one, which is what issue #11 ran into.
                    "Download the reviewed weight with "
                    "`--fetch-model opencv-lama`, or point VSR_LAMA_ONNX / "
                    "VSR_OPENCV_LAMA at a model you already have, or set "
                    "VSR_ENABLE_PYTORCH_LAMA=1 with simple-lama-inpainting."
                ),
            )

        if not _module_can_import(
            "simple_lama_inpainting",
            logger=logger,
            failure_context="LaMa PyTorch fallback disabled",
        ):
            raise RequestedStageError(
                stage="inpaint",
                requested_implementation="lama",
                failure_class=FAILURE_DEPENDENCY_MISSING,
                detail="simple-lama-inpainting failed its import safety probe",
                recovery_hint=(
                    "Repair the PyTorch LaMa installation or configure a "
                    "reviewed ONNX model, then retry."
                ),
            )

        try:
            from simple_lama_inpainting import SimpleLama
            self._lama = SimpleLama()
            self._backend_name = "PyTorch (simple-lama-inpainting)"
            logger.info("LaMa PyTorch inpainting loaded (simple-lama-inpainting)")
            self._verify_pytorch_weights()
        except RequestedStageError:
            raise
        except (ImportError, OSError, RuntimeError) as exc:
            raise RequestedStageError(
                stage="inpaint",
                requested_implementation="lama",
                failure_class=FAILURE_INITIALIZATION,
                detail=str(exc),
                recovery_hint=(
                    "Repair the selected LaMa model and PyTorch runtime, then retry."
                ),
                cause=exc,
            ) from exc
        except Exception as exc:
            raise RequestedStageError(
                stage="inpaint",
                requested_implementation="lama",
                failure_class=FAILURE_INITIALIZATION,
                detail=str(exc),
                recovery_hint="Verify the selected LaMa model, then retry.",
                cause=exc,
            ) from exc
        if self._lama is None:
            raise RequestedStageError(
                stage="inpaint",
                requested_implementation="lama",
                failure_class=FAILURE_INITIALIZATION,
                detail="the LaMa weights failed verification",
                recovery_hint=(
                    "Replace the cached weights with a reviewed LaMa artifact, "
                    "then retry."
                ),
            )

    def _verify_pytorch_weights(self):
        try:
            from backend.adapter_manifest import (
                log_adapter_verification as _log_adapter,
                verify_adapter_path as _verify_adapter,
            )
            model_env = os.environ.get("LAMA_MODEL", "").strip()
            if model_env:
                path = Path(model_env)
            else:
                import torch
                model_url = os.environ.get(
                    "LAMA_MODEL_URL",
                    "https://github.com/enesmsahin/simple-lama-inpainting/"
                    "releases/download/v0.1.0/big-lama.pt",
                )
                path = Path(torch.hub.get_dir()) / "checkpoints" / Path(
                    model_url.split("?", 1)[0]).name
            if not path.is_file():
                logger.debug("Simple-Lama weights are not present at %s", path)
                return
            result = _verify_adapter("simple-lama", str(path))
            _log_adapter(result)
            if not result.allowed:
                self._lama = None
                self._backend_name = "unavailable"
                logger.warning(
                    "LaMa neural inpainting disabled because cached "
                    "weights failed manifest verification."
                )
        except Exception as exc:
            logger.debug("Weight verification skipped: %s", exc)

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def _runtime_error(
        self, exc: BaseException, operation: str
    ) -> RequestedStageError:
        return RequestedStageError(
            stage="inpaint",
            requested_implementation="lama",
            actual_implementation="lama",
            provider=self.backend_name,
            failure_class=FAILURE_RUNTIME,
            detail=f"{operation}: {exc}",
            recovery_hint=(
                "Verify the selected LaMa provider and model inputs, then retry."
            ),
            cause=exc,
        )

    def inpaint(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        # Neural models treat the mask as a strict binary indicator; a soft
        # alpha matte (MatAnyone refinement) must be thresholded before the
        # model input. The original soft masks are kept for the feather blend
        # and temporal smoothing below so edges stay seamless.
        model_masks = [_binarize_mask(m) for m in masks]
        if self._onnx_session is not None:
            raw = self._inpaint_onnx(frames, model_masks)
        elif self._dnn_net is not None:
            raw = self._inpaint_cv2dnn(frames, model_masks)
        elif self._lama is not None:
            raw = self._inpaint_pytorch(frames, model_masks)
        else:
            raise RequestedStageError(
                stage="inpaint",
                requested_implementation="lama",
                failure_class=FAILURE_INITIALIZATION,
                detail="no LaMa provider is loaded",
                recovery_hint="Recreate the LaMa provider, then retry.",
            )
        out = apply_finishing(frames, raw, masks, self.config)
        smooth = self.config.temporal_smooth_radius
        if smooth > 0 and len(out) > 1:
            out = _temporal_smooth_inpainted(out, masks, radius=smooth)
        return out

    def _inpaint_onnx(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        tile_size = self.config.lama_tile_size
        tile_overlap = self.config.lama_tile_overlap
        results = []
        for frame, mask in zip(frames, masks, strict=True):
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
                    logger.warning(
                        "Tiled LaMa-ONNX fell back to full-frame: %s",
                        exc,
                        exc_info=True,
                    )
            try:
                results.append(self._inpaint_onnx_one(frame, mask))
            except Exception as exc:
                raise self._runtime_error(
                    exc, "LaMa ONNX inference failed"
                ) from exc
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
                tile_out = self._inpaint_onnx_one(tile_frame, tile_mask)
                th, tw = tile_out.shape[:2]
                wy = np.ones(th, dtype=np.float32)
                wx = np.ones(tw, dtype=np.float32)
                if overlap > 0:
                    ramp = min(overlap, th // 2, tw // 2)
                    if ramp > 0:
                        taper = 0.5 - 0.5 * np.cos(
                            np.linspace(
                                0.5 * np.pi / ramp,
                                np.pi - 0.5 * np.pi / ramp,
                                ramp,
                                dtype=np.float32,
                            ))
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
        for frame, mask in zip(frames, masks, strict=True):
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
                        "Tiled LaMa cv2.dnn fell back to full-frame: %s",
                        exc,
                        exc_info=True,
                    )
            try:
                results.append(self._inpaint_cv2dnn_one(frame, mask))
            except Exception as exc:
                raise self._runtime_error(
                    exc, "LaMa OpenCV DNN inference failed"
                ) from exc
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
                tile_out = self._inpaint_cv2dnn_one(tile_frame, tile_mask)
                th, tw = tile_out.shape[:2]
                wy = np.ones(th, dtype=np.float32)
                wx = np.ones(tw, dtype=np.float32)
                if overlap > 0:
                    ramp = min(overlap, th // 2, tw // 2)
                    if ramp > 0:
                        taper = 0.5 - 0.5 * np.cos(
                            np.linspace(
                                0.5 * np.pi / ramp,
                                np.pi - 0.5 * np.pi / ramp,
                                ramp,
                                dtype=np.float32,
                            ))
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
        for frame, mask in zip(frames, masks, strict=True):
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
                # simple-lama-inpainting pads inputs to a multiple of eight
                # but does not crop its result back to the source geometry.
                # Finishing and output contracts require the original shape.
                result_bgr = result_bgr[:h, :w]
                if result_bgr.shape[:2] != (h, w):
                    raise ValueError(
                        "LaMa returned a frame smaller than the source"
                    )
                results.append(result_bgr)
            except Exception as e:
                raise self._runtime_error(
                    e, "LaMa PyTorch inference failed"
                ) from e
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
                pil_out = self._lama(pil_tile, pil_mask)
                tile_out = cv2.cvtColor(np.array(pil_out), cv2.COLOR_RGB2BGR)
                tile_h, tile_w = tile_frame.shape[:2]
                tile_out = tile_out[:tile_h, :tile_w]
                if tile_out.shape[:2] != (tile_h, tile_w):
                    raise ValueError(
                        "LaMa tile output is smaller than the source tile"
                    )

                th, tw = tile_out.shape[:2]
                wy = np.ones(th, dtype=np.float32)
                wx = np.ones(tw, dtype=np.float32)
                if overlap > 0:
                    ramp = min(overlap, th // 2, tw // 2)
                    if ramp > 0:
                        taper = 0.5 - 0.5 * np.cos(
                            np.linspace(
                                0.5 * np.pi / ramp,
                                np.pi - 0.5 * np.pi / ramp,
                                ramp,
                                dtype=np.float32,
                            ))
                        wy[:ramp] *= taper
                        wy[-ramp:] *= taper[::-1]
                        wx[:ramp] *= taper
                        wx[-ramp:] *= taper[::-1]
                win = np.outer(wy, wx)
                color_acc[ty1:ty2, tx1:tx2] += (
                    tile_out.astype(np.float32) * win[..., None]
                )
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
        for f, m in zip(frames, masks, strict=True):
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
