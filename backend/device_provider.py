"""Injectable runtime-device and inpainter construction policy."""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Optional, Protocol

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


class ProviderFellBackError(RequestedStageError):
    """Raised when a named accelerator silently ran on the CPU.

    RM-322: ONNX Runtime accepts an unavailable provider, prints a warning,
    and runs the session on CPU anyway. A user who chose a GPU device then
    waits through a CPU run believing it is accelerated, which is the same
    silent-substitution failure InpainterUnavailableError exists to stop.
    `device="auto"` and `device="cpu"` are unaffected: nothing was named, so
    nothing was substituted.
    """

    def __init__(self, device: str, requested: str, active: list,
                 *, stage: str = "inpaint"):
        self.device = device
        self.requested = requested
        self.active = list(active)
        ran = self.active[0] if self.active else "no provider"
        super().__init__(
            stage=stage,
            requested_implementation=device,
            failure_class=FAILURE_INITIALIZATION,
            detail=(
                f"{requested} was requested for device {device!r} but "
                f"{ran} ran the session"
            ),
            recovery_hint=(
                "Install the provider's runtime, or select CPU or Auto if a "
                "CPU run is acceptable. ONNX Runtime does not fail on its "
                "own when a requested provider cannot load."
            ),
        )


# Device tokens that name a specific accelerator. "auto" and "cpu" do not,
# so a CPU session under those is the requested behaviour, not a fallback.
_NAMED_ACCELERATOR_PROVIDERS = {
    "cuda": "CUDAExecutionProvider",
    "directml": "DmlExecutionProvider",
}


def named_accelerator_provider(device: object) -> str:
    """Return the provider a device token explicitly asks for, or ""."""
    token = str(device or "").strip().lower()
    if not token or token in {"auto", "cpu", "windowsml"}:
        return ""
    for prefix, provider in _NAMED_ACCELERATOR_PROVIDERS.items():
        if token.startswith(prefix):
            return provider
    return ""


def _provider_name(provider: object) -> str:
    if isinstance(provider, tuple) and provider:
        return str(provider[0])
    return str(provider)


def verify_active_provider(device: object, active: object,
                           *, requested_providers: object = None,
                           stage: str = "inpaint") -> None:
    """Fail loudly when a named accelerator silently ran on the CPU.

    The question is whether the session dropped to the CPU, not whether one
    exact provider is first. Two legitimate cases would fail that stricter
    reading: RM-70 puts TensorrtExecutionProvider ahead of CUDA when a
    cached engine exists, which is the faster accelerated lane rather than a
    fallback, and the DirectML opset audit deliberately drops the DirectML
    provider for models above its supported opset and says so in a warning.
    A deliberate downgrade that the product announced is not a silent one.
    """
    requested = named_accelerator_provider(device)
    if not requested:
        return
    if requested_providers is not None:
        offered = {_provider_name(item) for item in requested_providers}
        if requested not in offered:
            # The provider was removed from the request on purpose upstream.
            return
    names = [_provider_name(item) for item in (active or [])]
    if any(name != "CPUExecutionProvider" for name in names):
        return
    raise ProviderFellBackError(str(device), requested, names, stage=stage)


def verify_session_provider(device: object, session: object,
                            *, requested_providers: object = None,
                            stage: str = "inpaint") -> None:
    """Check a live session, skipping when it cannot report its providers.

    Resolving the provider list lazily matters: a test double or a runtime
    that predates `get_providers` must not turn into an AttributeError on a
    device token that names nothing to verify in the first place.
    """
    if not named_accelerator_provider(device):
        return
    reader = getattr(session, "get_providers", None)
    if not callable(reader):
        logger.debug(
            "Session for device %r cannot report its providers; skipping the "
            "fallback check", device,
        )
        return
    try:
        active = reader()
    except Exception as exc:
        logger.debug("Provider read failed for device %r: %s", device, exc)
        return
    verify_active_provider(
        device, active, requested_providers=requested_providers,
        stage=stage)


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


RELEASES_URL = "https://github.com/SysAdminDoc/VideoSubtitleRemover/releases"


def cpu_build_on_nvidia_hardware(
    *,
    requested_device: str = "",
    env: Optional[Mapping[str, str]] = None,
    gpu_probe: Optional[Callable[[], dict]] = None,
    cuda_probe: Callable[[int], bool] = _cuda_available,
    profile_probe: Optional[Callable[[], dict]] = None,
) -> Optional[dict]:
    """Report a machine that has an NVIDIA card and a build that cannot use it.

    RM-356: the product already knew both halves of this and said neither
    where anyone would look. `gui/utils.detect_gpu` and this module's own
    fallback warning meant a user on the CPU download could watch a twenty
    minute run go by on the CPU with an idle 4070 in the machine, which is
    what issue #10 reports.

    Returns None when there is nothing to say: no NVIDIA adapter, a CUDA
    provider that does load, a build already on the NVIDIA lane, or a user who
    asked for the CPU on purpose.
    """
    requested = str(requested_device or "").strip().lower()
    if requested == "cpu":
        return None

    if profile_probe is None:
        from backend.build_profile import resolve_build_profile

        def profile_probe() -> dict:
            return resolve_build_profile(env=env)
    profile = str((profile_probe() or {}).get("profile") or "").lower()
    if profile == "nvidia":
        return None

    if cuda_probe(0):
        return None

    if gpu_probe is None:
        from backend.provider_benchmark import gpu_host_facts

        gpu_probe = gpu_host_facts
    facts = gpu_probe() or {}
    if not facts.get("present"):
        return None

    return {
        "adapter": str(facts.get("name") or "NVIDIA GPU"),
        "driver": str(facts.get("driver") or ""),
        "profile": profile or "cpu",
        "releasesUrl": RELEASES_URL,
        "assetPrefix": "VideoSubtitleRemoverPro-<version>-nvidia",
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
