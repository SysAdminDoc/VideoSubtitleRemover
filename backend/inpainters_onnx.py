"""Opt-in ONNX-based inpainter backends.

RM-25 LaMa-ONNX and RM-26 MI-GAN both ship as ONNX checkpoints and can
be run via `onnxruntime` (CPU or GPU). Each backend defined here:

- Imports lazily so the rest of the codebase keeps working when
  onnxruntime is not installed.
- Registers itself with the inpainter registry under the mode name
  the GUI/CLI uses (`"lama"` for LaMa-ONNX shadowing the default
  PyTorch LaMa, `"migan"` for MI-GAN).
- Falls back to cv2.inpaint when the ONNX session can't be created
  (missing weights, missing onnxruntime, etc.) so the user always
  gets *some* result even on a half-broken install.

To enable LaMa-ONNX:
    pip install onnxruntime
    huggingface-cli download Carve/LaMa-ONNX
    set VSR_LAMA_ONNX=path/to/lama_fp32.onnx

To enable MI-GAN:
    pip install onnxruntime
    Download the MI-GAN ONNX (see IOPaint MI-GAN model card)
    set VSR_MIGAN_ONNX=path/to/migan_pipeline_v2.onnx

When the env var is unset, the backend module imports cleanly but the
registration is skipped, so the existing PyTorch LaMa stays the
default. Setting VSR_LAMA_ONNX forces the LaMa registry slot to be
replaced by the ONNX variant; setting only VSR_MIGAN_ONNX adds a new
`"migan"` mode the GUI/CLI can dispatch to via `--mode migan`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _maybe_session(model_path: str, providers=None, adapter_name: str = "lama-onnx"):
    """Lazy-init an onnxruntime InferenceSession; returns None on any
    failure so the caller falls back to cv2."""
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        logger.warning(
            "onnxruntime is not installed; ONNX inpainter unavailable. "
            "Install with `pip install onnxruntime` or `onnxruntime-gpu`."
        )
        return None
    if not Path(model_path).is_file():
        logger.warning(f"ONNX model not found at {model_path!r}")
        return None
    try:
        from backend.adapter_manifest import (
            log_adapter_verification,
            verify_adapter_path,
        )
        verification = verify_adapter_path(adapter_name, model_path)
        log_adapter_verification(verification)
        if not verification.allowed:
            return None
    except Exception as exc:
        logger.warning(f"Adapter security verification failed for {model_path!r}: {exc}")
        return None
    try:
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = _providers_after_opset_audit(model_path, providers)
        from backend.onnxruntime_cuda import preload_onnxruntime_cuda_dlls_if_needed
        preload_onnxruntime_cuda_dlls_if_needed(ort, providers)
        return ort.InferenceSession(model_path, providers=providers)
    except Exception as exc:
        logger.warning(f"Failed to load ONNX session {model_path!r}: {exc}")
        return None


def _providers_for_device(device: str) -> List:
    """Return ONNX Runtime providers for the requested VSR device token."""
    if device == "windowsml":
        return ["CPUExecutionProvider"]
    if device == "directml":
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    if "cuda" in device:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def windows_ml_probe_decision(run_smoke: bool = False) -> dict:
    """Diagnostic helper for deciding whether Windows ML is migration-ready."""
    try:
        from backend.onnx_model_info import collect_windows_ml_probe
        return collect_windows_ml_probe(run_smoke=run_smoke)
    except Exception as exc:
        return {
            "schema": "vsr.windows_ml_probe.v1",
            "decision": "blocked",
            "reason": f"Windows ML probe failed: {exc}",
        }


def _provider_name(provider) -> str:
    if isinstance(provider, tuple):
        return str(provider[0])
    return str(provider)


def _providers_after_opset_audit(model_path: str, providers: List) -> List:
    """Drop DirectML for ONNX files declaring unsupported default opsets."""
    if not any(_provider_name(p) == "DmlExecutionProvider" for p in providers):
        return providers
    try:
        from backend.onnx_model_info import (
            DIRECTML_MAX_ONNX_OPSET,
            directml_incompatible_opsets,
        )
        incompatible = directml_incompatible_opsets(model_path)
    except Exception as exc:
        logger.warning(
            f"Could not inspect ONNX opsets for {model_path!r}; "
            f"DirectML session creation may fail: {exc}"
        )
        return providers
    if not incompatible:
        return providers
    detail = ", ".join(
        f"{item.domain or 'ai.onnx'}={item.version}" for item in incompatible
    )
    logger.warning(
        f"ONNX model {model_path!r} declares opset {detail}, above "
        f"DirectML EP's supported ONNX opset <= {DIRECTML_MAX_ONNX_OPSET}; "
        "using CPU provider instead."
    )
    filtered = [p for p in providers if _provider_name(p) != "DmlExecutionProvider"]
    return filtered or ["CPUExecutionProvider"]


def _ensure_multiple_of(value: int, multiple: int) -> int:
    """Round `value` up to the next multiple of `multiple`. ONNX
    inpainters typically require dimensions divisible by 8."""
    if value % multiple == 0:
        return value
    return ((value // multiple) + 1) * multiple


class LamaOnnxInpainter:
    """Run LaMa in ONNX Runtime. ~3-5x faster than the PyTorch
    simple-lama-inpainting path and runs without torch entirely.

    Defers to cv2.inpaint per-frame when the ONNX session is
    unavailable so the backend stays usable on partial installs.
    """

    INPUT_NAME = "image"
    MASK_NAME = "mask"

    def __init__(self, device: str = "cpu", config=None):
        self.device = device
        self.config = config
        model_path = os.environ.get("VSR_LAMA_ONNX", "")
        # RM-70: when TensorRT is enabled, prefer the cached engine via
        # the TensorrtExecutionProvider before falling back to CUDA/CPU.
        providers = []
        try:
            from backend.tensorrt_compile import (
                is_tensorrt_enabled, maybe_compile_engine,
            )
            if is_tensorrt_enabled() and model_path and "cuda" in device:
                engine = maybe_compile_engine(model_path)
                if engine is not None:
                    providers.append(("TensorrtExecutionProvider", {
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": str(engine.parent),
                    }))
        except Exception as exc:
            logger.debug(f"TensorRT path skipped: {exc}")
        providers += _providers_for_device(device)
        self._session = (
            _maybe_session(model_path, providers, "lama-onnx")
            if model_path else None
        )

    def inpaint(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        if self._session is None:
            return _cv2_fallback(frames, masks, self.config)
        results: List[np.ndarray] = []
        for frame, mask in zip(frames, masks):
            if mask.max() == 0:
                results.append(frame.copy())
                continue
            try:
                results.append(self._inpaint_one(frame, mask))
            except Exception as exc:
                logger.warning(f"LaMa-ONNX inference failed on frame, falling back: {exc}")
                results.append(_cv2_inpaint_single(frame, mask))
        return _apply_feather_blend(frames, results, masks, self.config)

    def _inpaint_one(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        ph = _ensure_multiple_of(h, 8)
        pw = _ensure_multiple_of(w, 8)
        # LaMa expects RGB float32 [0,1] and a single-channel mask [0,1].
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if (ph, pw) != (h, w):
            rgb = cv2.copyMakeBorder(rgb, 0, ph - h, 0, pw - w, cv2.BORDER_REFLECT_101)
            mask_padded = cv2.copyMakeBorder(mask, 0, ph - h, 0, pw - w, cv2.BORDER_CONSTANT, value=0)
        else:
            mask_padded = mask
        img_t = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        mask_t = (mask_padded.astype(np.float32) / 255.0)[None, None, ...]
        out = self._session.run(
            None, {self.INPUT_NAME: img_t, self.MASK_NAME: mask_t}
        )[0]
        # Out shape: (1, 3, ph, pw), float32 in [0,1].
        bgr = cv2.cvtColor(
            (out[0].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        return bgr[:h, :w]


class MiGanInpainter:
    """Run MI-GAN in ONNX Runtime. Single-frame, mobile-grade speed
    (~10 ms / 512x512 on a modern CPU per the ICCV 2023 paper).
    Falls back to cv2 when the session can't initialise.
    """

    def __init__(self, device: str = "cpu", config=None):
        self.device = device
        self.config = config
        model_path = os.environ.get("VSR_MIGAN_ONNX", "")
        providers = _providers_for_device(device)
        self._session = (
            _maybe_session(model_path, providers, "migan-onnx")
            if model_path else None
        )

    def inpaint(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        if self._session is None:
            return _cv2_fallback(frames, masks, self.config)
        results: List[np.ndarray] = []
        for frame, mask in zip(frames, masks):
            if mask.max() == 0:
                results.append(frame.copy())
                continue
            try:
                results.append(self._inpaint_one(frame, mask))
            except Exception as exc:
                logger.warning(f"MI-GAN inference failed on frame, falling back: {exc}")
                results.append(_cv2_inpaint_single(frame, mask))
        return _apply_feather_blend(frames, results, masks, self.config)

    def _inpaint_one(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        # MI-GAN expects 512x512 input; we tile / pad to that grid then
        # crop the result back. For simplicity at this scale, resize
        # both inputs to the network resolution and resize the output
        # back. This costs detail; tile-based ingest is a follow-up.
        target = 512
        scaled = cv2.resize(frame, (target, target), interpolation=cv2.INTER_AREA)
        smask = cv2.resize(mask, (target, target), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
        img_t = (rgb.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None, ...]
        mask_t = (smask.astype(np.float32) / 255.0)[None, None, ...]
        # MI-GAN's exported ONNX names vary between releases; iterate
        # the session inputs and bind by position.
        inputs = {}
        for inp in self._session.get_inputs():
            if "mask" in inp.name.lower():
                inputs[inp.name] = mask_t
            else:
                inputs[inp.name] = img_t
        out = self._session.run(None, inputs)[0]
        rgb_out = ((out[0].transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        bgr_out = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
        return cv2.resize(bgr_out, (w, h), interpolation=cv2.INTER_LINEAR)


def _cv2_fallback(frames, masks, config):
    """Per-frame cv2.inpaint + feather blend, used when an ONNX session
    isn't available."""
    out: List[np.ndarray] = []
    for f, m in zip(frames, masks):
        out.append(_cv2_inpaint_single(f, m))
    return _apply_feather_blend(frames, out, masks, config)


def _cv2_inpaint_single(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.max() == 0:
        return frame.copy()
    return cv2.inpaint(frame, mask, 7, cv2.INPAINT_NS)


def _apply_feather_blend(original, filled, masks, config):
    """Reuse the shared feather-blend at every boundary."""
    if config is None:
        return list(filled)
    try:
        from backend.inpainters import _feather_blend, _edge_ring_color_correct
    except Exception:
        return list(filled)
    out = []
    feather = getattr(config, "mask_feather_px", 4)
    ring = getattr(config, "edge_ring_px", 2)
    for f, r, m in zip(original, filled, masks):
        if ring > 0 and m.max() > 0:
            r = _edge_ring_color_correct(f, r, m, ring)
        out.append(_feather_blend(f, r, m, feather))
    return out


def maybe_register() -> List[str]:
    """Register ONNX backends when their env vars are set. Returns the
    list of registered mode names so callers can log what landed."""
    from backend.inpainter_registry import register
    registered = []
    if os.environ.get("VSR_LAMA_ONNX"):
        register("lama", lambda device, config: LamaOnnxInpainter(device, config))
        registered.append("lama (ONNX)")
        logger.info("LaMa-ONNX backend registered, shadowing PyTorch LaMa")
    if os.environ.get("VSR_MIGAN_ONNX"):
        register("migan", lambda device, config: MiGanInpainter(device, config))
        registered.append("migan")
        logger.info("MI-GAN ONNX backend registered as mode 'migan'")
    return registered


# Register on import so a user who has the env vars set when the
# processor module loads gets the ONNX backends automatically.
maybe_register()
