"""ONNX Runtime CUDA DLL preload guard and diagnostics."""

from __future__ import annotations

from threading import Lock
from typing import Any, Mapping, Optional, Sequence


CUDA_PRELOAD_STATUS_SCHEMA = "vsr.onnxruntime_cuda_preload.v1"
CUDA_PROVIDER = "CUDAExecutionProvider"

_LOCK = Lock()


def _initial_status() -> dict:
    return {
        "schema": CUDA_PRELOAD_STATUS_SCHEMA,
        "needed": False,
        "attempted": False,
        "available": False,
        "succeeded": False,
        "error": "",
        "callCount": 0,
        "lastProviders": [],
        "lastDirectory": "",
        "lastRuntimeVersion": None,
    }


_STATUS = _initial_status()


def _provider_name(provider: Any) -> str:
    if isinstance(provider, tuple) and provider:
        return str(provider[0])
    return str(provider)


def _provider_names(providers: Optional[Sequence[Any]]) -> list[str]:
    return [_provider_name(provider) for provider in (providers or [])]


def _snapshot_locked() -> dict:
    payload = dict(_STATUS)
    payload["lastProviders"] = list(_STATUS.get("lastProviders", []))
    return payload


def reset_onnxruntime_cuda_preload_status() -> None:
    """Reset process-local preload diagnostics for tests."""
    with _LOCK:
        _STATUS.clear()
        _STATUS.update(_initial_status())


def collect_onnxruntime_cuda_preload_status() -> dict:
    """Return the latest process-local CUDA preload outcome."""
    with _LOCK:
        return _snapshot_locked()


def preload_onnxruntime_cuda_dlls_if_needed(
    ort_module: Any,
    providers: Optional[Sequence[Any]],
    *,
    directory: Optional[str] = None,
) -> dict:
    """Call onnxruntime.preload_dlls() once before CUDA sessions.

    The helper never raises. Session creation still decides whether CUDA can be
    used, while diagnostics record whether preload support was present and
    whether the call succeeded.
    """
    names = _provider_names(providers)
    needs_cuda = CUDA_PROVIDER in names
    with _LOCK:
        _STATUS["lastProviders"] = names
        if not needs_cuda:
            if not _STATUS.get("attempted"):
                _STATUS["needed"] = False
            return _snapshot_locked()
        _STATUS["needed"] = True
        _STATUS["lastRuntimeVersion"] = getattr(ort_module, "__version__", None)
        _STATUS["lastDirectory"] = str(directory or "")
        if _STATUS.get("attempted"):
            return _snapshot_locked()
        preload = getattr(ort_module, "preload_dlls", None)
        _STATUS["attempted"] = True
        _STATUS["available"] = callable(preload)
        if not callable(preload):
            _STATUS["succeeded"] = False
            _STATUS["error"] = "onnxruntime.preload_dlls unavailable"
            return _snapshot_locked()
        _STATUS["callCount"] = int(_STATUS.get("callCount") or 0) + 1
        try:
            if directory:
                preload(directory=directory)
            else:
                preload()
            _STATUS["succeeded"] = True
            _STATUS["error"] = ""
        except Exception as exc:  # pragma: no cover - depends on local CUDA DLLs
            _STATUS["succeeded"] = False
            _STATUS["error"] = str(exc)[:500]
        return _snapshot_locked()


def preload_status_from_mapping(value: Optional[Mapping[str, object]]) -> dict:
    """Coerce external test/status input into a complete preload payload."""
    payload = _initial_status()
    if value:
        payload.update(dict(value))
    payload["lastProviders"] = list(payload.get("lastProviders", []) or [])
    return payload
