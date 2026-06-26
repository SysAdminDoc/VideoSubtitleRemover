"""Runtime checks for dependency major-version ceilings."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata as metadata
import re
import sys
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class DependencyCap:
    package: str
    minimum: str
    maximum: str


OCR_DEPENDENCY_CAPS: Tuple[DependencyCap, ...] = (
    DependencyCap("rapidocr", "2.0.0", "4.0.0"),
    DependencyCap("rapidocr-onnxruntime", "1.4.0", "2.0.0"),
    DependencyCap("paddleocr", "3.0.0", "4.0.0"),
)

ONNXRUNTIME_PROVIDER_STATUS_SCHEMA = "vsr.onnxruntime_providers.v1"
ONNXRUNTIME_GPU_RECOMMENDED_MIN = "1.21.0"
ONNXRUNTIME_GPU_STABLE_CUDA12_MIN = "1.19.0"
ONNXRUNTIME_PACKAGES = (
    "onnxruntime",
    "onnxruntime-gpu",
    "onnxruntime-directml",
)


def _version_key(value: str) -> Tuple[int, int, int]:
    numbers = [int(part) for part in re.findall(r"\d+", value)[:3]]
    while len(numbers) < 3:
        numbers.append(0)
    return tuple(numbers)  # type: ignore[return-value]


def _within_cap(version: str, cap: DependencyCap) -> bool:
    parsed = _version_key(version)
    return _version_key(cap.minimum) <= parsed < _version_key(cap.maximum)


def _version_gte(version: str, floor: str) -> bool:
    return _version_key(version) >= _version_key(floor)


def _installed_package_version(
    package: str,
    package_versions: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    if package_versions is not None:
        for key, value in package_versions.items():
            if key.lower().replace("_", "-") == package:
                return str(value)
        return None
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _onnxruntime_gpu_channel(version: Optional[str]) -> str:
    if not version:
        return "not-installed"
    lowered = str(version).lower()
    if "cuda13" in lowered or "cu13" in lowered:
        return "cuda13-nightly-or-custom"
    if "dev" in lowered or "nightly" in lowered:
        return "nightly-or-custom"
    if _version_gte(version, ONNXRUNTIME_GPU_STABLE_CUDA12_MIN):
        return "cuda12-pypi-stable"
    return "legacy-cuda-package"


def _runtime_provider_probe() -> tuple[list[str], Optional[str], bool, str]:
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:
        return [], None, False, str(exc)
    try:
        providers = list(ort.get_available_providers())
    except Exception as exc:
        return [], getattr(ort, "__version__", None), hasattr(ort, "preload_dlls"), str(exc)
    return (
        providers,
        getattr(ort, "__version__", None),
        hasattr(ort, "preload_dlls"),
        "",
    )


def collect_onnxruntime_provider_status(
    *,
    package_versions: Optional[Mapping[str, str]] = None,
    providers: Optional[Sequence[str]] = None,
    runtime_version: Optional[str] = None,
    preload_dlls_available: Optional[bool] = None,
) -> dict:
    """Return package/provider facts for ONNX Runtime CPU, CUDA, and DirectML."""
    package_status = {
        package: {
            "installed": False,
            "version": None,
        }
        for package in ONNXRUNTIME_PACKAGES
    }
    for package in ONNXRUNTIME_PACKAGES:
        version = _installed_package_version(package, package_versions)
        package_status[package] = {
            "installed": version is not None,
            "version": version,
        }

    probe_error = ""
    if providers is None or runtime_version is None or preload_dlls_available is None:
        probed_providers, probed_version, probed_preload, probe_error = _runtime_provider_probe()
        if providers is None:
            providers = probed_providers
        if runtime_version is None:
            runtime_version = probed_version
        if preload_dlls_available is None:
            preload_dlls_available = probed_preload
    providers = list(providers or [])
    gpu_version = package_status["onnxruntime-gpu"]["version"]
    gpu_channel = _onnxruntime_gpu_channel(gpu_version)
    cuda_provider = "CUDAExecutionProvider" in providers
    directml_provider = "DmlExecutionProvider" in providers
    warnings = []
    if gpu_version and gpu_channel == "legacy-cuda-package":
        warnings.append({
            "id": "ORT-CUDA-LEGACY-PACKAGE",
            "severity": "medium",
            "message": (
                "onnxruntime-gpu is older than the CUDA 12 PyPI package line; "
                f"install onnxruntime-gpu>={ONNXRUNTIME_GPU_RECOMMENDED_MIN} "
                "for the tested NVIDIA ONNX path."
            ),
        })
    if gpu_version and not cuda_provider:
        warnings.append({
            "id": "ORT-CUDA-PROVIDER-MISSING",
            "severity": "medium",
            "message": (
                "onnxruntime-gpu is installed but CUDAExecutionProvider was "
                "not reported; check CUDA/cuDNN DLL availability."
            ),
        })
    if gpu_version and not preload_dlls_available:
        warnings.append({
            "id": "ORT-CUDA-PRELOAD-MISSING",
            "severity": "low",
            "message": (
                "onnxruntime.preload_dlls() is unavailable; upgrade "
                f"onnxruntime-gpu to >= {ONNXRUNTIME_GPU_RECOMMENDED_MIN} "
                "for reliable Windows CUDA DLL loading."
            ),
        })

    return {
        "schema": ONNXRUNTIME_PROVIDER_STATUS_SCHEMA,
        "runtimeVersion": runtime_version,
        "availableProviders": providers,
        "probeError": probe_error,
        "packages": package_status,
        "cuda": {
            "packageInstalled": bool(gpu_version),
            "packageVersion": gpu_version,
            "packageChannel": gpu_channel,
            "providerAvailable": cuda_provider,
            "preloadDllsAvailable": bool(preload_dlls_available),
            "recommendedPackage": (
                f"onnxruntime-gpu>={ONNXRUNTIME_GPU_RECOMMENDED_MIN}"
            ),
            "cuda13Status": "nightly-or-custom-channel",
        },
        "directml": {
            "packageInstalled": bool(
                package_status["onnxruntime-directml"]["installed"]
            ),
            "packageVersion": package_status["onnxruntime-directml"]["version"],
            "providerAvailable": directml_provider,
        },
        "warnings": warnings,
    }


def onnxruntime_release_advisories(status: Optional[Mapping[str, object]] = None) -> list[dict]:
    status = dict(status or collect_onnxruntime_provider_status())
    cuda = status.get("cuda", {}) if isinstance(status.get("cuda"), Mapping) else {}
    warnings = status.get("warnings", [])
    advisories = []
    iterable_warnings = warnings if isinstance(warnings, list) else []
    for warning in iterable_warnings:
        if not isinstance(warning, Mapping):
            continue
        if warning.get("id") not in {
            "ORT-CUDA-LEGACY-PACKAGE",
            "ORT-CUDA-PRELOAD-MISSING",
        }:
            continue
        advisories.append({
            "id": str(warning.get("id")),
            "package": "onnxruntime-gpu",
            "installedVersion": str(cuda.get("packageVersion") or "not installed"),
            "affected": str(cuda.get("packageChannel") or "unknown"),
            "fixedIn": f">={ONNXRUNTIME_GPU_RECOMMENDED_MIN}",
            "severity": str(warning.get("severity") or "medium").lower(),
            "source": "https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html",
            "allowed": True,
            "allowReason": "Provider migration warning; not a security CVE.",
            "mitigation": str(warning.get("message") or ""),
            "blocking": False,
        })
    return advisories


def check_ocr_dependency_caps(
    caps: Iterable[DependencyCap] = OCR_DEPENDENCY_CAPS,
) -> List[str]:
    """Return installed OCR engine packages that exceed supported ranges."""
    problems: List[str] = []
    for cap in caps:
        try:
            installed = metadata.version(cap.package)
        except metadata.PackageNotFoundError:
            continue
        if not _within_cap(installed, cap):
            problems.append(
                f"{cap.package}=={installed} outside supported range "
                f">={cap.minimum},<{cap.maximum}"
            )
    return problems


def enforce_ocr_dependency_caps() -> None:
    problems = check_ocr_dependency_caps()
    if problems:
        joined = "\n- ".join(problems)
        raise RuntimeError(f"OCR dependency version cap check failed:\n- {joined}")


def main() -> int:
    try:
        enforce_ocr_dependency_caps()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("OCR dependency version caps OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
