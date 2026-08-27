"""Injectable runtime-device and inpainter construction policy."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Protocol

from backend.execution_provenance import (
    FAILURE_DEPENDENCY_MISSING,
    FAILURE_INITIALIZATION,
    RequestedStageError,
)
from backend.onnxruntime_cuda import TENSORRT_RTX_PROVIDER


logger = logging.getLogger(__name__)


class InpainterUnavailableError(RequestedStageError):
    """Raised when the requested inpainting model has no registered
    backend.

    GitHub issue #7: this case used to silently substitute STTN, so a
    user who selected LaMa/MI-GAN/a diffusion backend that had not been
    registered (missing opt-in env var, failed optional import) got STTN
    output under every model name -- "all the models produce the same
    results" with only a log-file warning as evidence. Substituting a
    different model than the one the user asked for is never the right
    call; fail with an actionable message instead.
    """

    def __init__(self, requested: str, registered: list):
        self.requested = requested
        self.registered = registered
        available = ", ".join(sorted(registered)) if registered else "(none)"
        super().__init__(
            stage="inpaint",
            requested_implementation=requested,
            failure_class=FAILURE_DEPENDENCY_MISSING,
            detail=(
                "No inpainting backend is registered. Registered backends: "
                f"{available}"
            ),
            recovery_hint=(
                "Enable the requested adapter and install its reviewed "
                "dependencies, then retry. Check the log for adapter load "
                "failures."
            ),
        )


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


def windowsml_status() -> dict:
    """Probe whether onnxruntime-windowsml is installed and report its
    version and providers.  DirectML is in maintenance mode; Windows ML
    is the forward path for AMD/Intel GPU inference."""
    try:
        import importlib.metadata as _md
        version = _md.version("onnxruntime-windowsml")
    except Exception:
        return {"available": False, "version": None, "providers": None}
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
    except Exception:
        providers = None
    return {"available": True, "version": version, "providers": providers}


def tensorrtrtx_status() -> dict:
    """Probe the manually installed TensorRT-RTX ONNX Runtime EP."""
    import importlib.metadata as _md

    package = None
    version = None
    for candidate in ("onnxruntime-gpu", "onnxruntime"):
        try:
            version = _md.version(candidate)
            package = candidate
            break
        except Exception:
            continue
    if package is None:
        return {
            "available": False,
            "packageInstalled": False,
            "package": None,
            "version": None,
            "provider": TENSORRT_RTX_PROVIDER,
            "providers": None,
        }
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
    except Exception:
        providers = None
    provider_names = list(providers) if providers is not None else None
    return {
        "available": bool(
            provider_names and TENSORRT_RTX_PROVIDER in provider_names
        ),
        "packageInstalled": True,
        "package": package,
        "version": version,
        "provider": TENSORRT_RTX_PROVIDER,
        "providers": provider_names,
    }


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
        implementation_id = name.strip().lower()
        recovery_hint = ""
        try:
            if self._resolver is None:
                from backend import inpainter_registry
                spec = inpainter_registry.resolve_spec(name)
                builder = spec.builder
                implementation_id = spec.implementation_id
                recovery_hint = spec.recovery_hint
            else:
                builder = self._resolve(name)
        except KeyError:
            from backend import inpainter_registry
            raise InpainterUnavailableError(
                name,
                [mode for mode, _ in inpainter_registry.list_modes()],
            ) from None
        try:
            inpainter = builder(device, config)
        except RequestedStageError:
            raise
        except Exception as exc:
            raise RequestedStageError(
                stage="inpaint",
                requested_implementation=name,
                failure_class=FAILURE_INITIALIZATION,
                detail=str(exc),
                recovery_hint=(
                    recovery_hint
                    or "Verify the selected model files and optional runtime, "
                    "then retry or select Auto."
                ),
                cause=exc,
            ) from exc
        if inpainter is None:
            raise RequestedStageError(
                stage="inpaint",
                requested_implementation=name,
                failure_class=FAILURE_INITIALIZATION,
                detail="the registered builder returned no implementation",
                recovery_hint=(
                    recovery_hint
                    or "Verify the selected adapter installation, then retry "
                    "or select Auto."
                ),
            )
        try:
            setattr(inpainter, "_vsr_requested_implementation", name)
            setattr(inpainter, "_vsr_registered_implementation", implementation_id)
            setattr(inpainter, "_vsr_recovery_hint", recovery_hint)
        except Exception:
            logger.debug(
                "Inpainter %s does not accept execution identity metadata", name)
        return inpainter

    @staticmethod
    def is_oom_error(exc: BaseException) -> bool:
        from backend.inpainters import is_oom_error
        return is_oom_error(exc)

    @staticmethod
    def free_inference_memory() -> None:
        from backend.inpainters import free_inference_memory
        free_inference_memory()
