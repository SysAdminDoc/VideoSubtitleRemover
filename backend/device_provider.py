"""Injectable runtime-device and inpainter construction policy."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Protocol


logger = logging.getLogger(__name__)


class DeviceProvider(Protocol):
    """Runtime strategy seam used by the processing orchestrator."""

    def probe_available(self) -> str:
        """Return the concrete device that should execute inference."""

    def create_inpainter(self, name: str, device: str, config: Any) -> object:
        """Construct one named inpainter for the selected device."""


def _cuda_available(index: int) -> bool:
    try:
        import torch
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and cuda.is_available():
            count = int(getattr(cuda, "device_count", lambda: 1)())
            return 0 <= index < max(1, count)
    except Exception:
        pass
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def _directml_available() -> bool:
    try:
        import onnxruntime as ort
        return "DmlExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


class RuntimeDeviceProvider:
    """Default device probe, registry factory, and memory-recovery hooks."""

    def __init__(
        self,
        requested_device: str,
        *,
        cuda_probe: Callable[[int], bool] = _cuda_available,
        directml_probe: Callable[[], bool] = _directml_available,
        resolver: Optional[Callable[[str], Callable[..., object]]] = None,
    ) -> None:
        self.requested_device = str(requested_device or "cpu").strip().lower()
        self._cuda_probe = cuda_probe
        self._directml_probe = directml_probe
        self._resolver = resolver

    def probe_available(self) -> str:
        requested = self.requested_device
        if requested == "cpu":
            return "cpu"
        if requested == "directml":
            if self._directml_probe():
                return requested
            logger.warning(
                "DirectML was requested but DmlExecutionProvider is not "
                "available; using CPU inference."
            )
            return "cpu"
        if requested.startswith("cuda:"):
            try:
                index = max(0, int(requested.split(":", 1)[1]))
            except (TypeError, ValueError):
                return "cpu"
            if self._cuda_probe(index):
                return f"cuda:{index}"
            logger.warning(
                "CUDA device %d was requested but no CUDA inference provider "
                "is available; using CPU inference.",
                index,
            )
            return "cpu"
        return "cpu"

    def _resolve(self, name: str) -> Callable[..., object]:
        if self._resolver is not None:
            return self._resolver(name)
        from backend import inpainter_registry
        return inpainter_registry.resolve(name)

    def create_inpainter(self, name: str, device: str, config: Any) -> object:
        try:
            builder = self._resolve(name)
        except KeyError:
            logger.warning(
                "No inpainter registered for %r; falling back to STTN",
                name,
            )
            builder = self._resolve("sttn")
        return builder(device, config)

    @staticmethod
    def is_oom_error(exc: BaseException) -> bool:
        from backend.inpainters import is_oom_error
        return is_oom_error(exc)

    @staticmethod
    def free_inference_memory() -> None:
        from backend.inpainters import free_inference_memory
        free_inference_memory()
