"""Runtime checks for dependency major-version ceilings."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata as metadata
import json
import re
import subprocess
import sys
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from backend.onnxruntime_cuda import (
    collect_onnxruntime_cuda_preload_status,
    preload_status_from_mapping,
)


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

RAPIDOCR_ENGINE_STATUS_SCHEMA = "vsr.rapidocr_engines.v1"
RAPIDOCR_OPENVINO_MIN = "3.9.0"
RAPIDOCR_ENGINE_PACKAGES = (
    "rapidocr",
    "rapidocr-onnxruntime",
    "openvino",
)

ONNXRUNTIME_PROVIDER_STATUS_SCHEMA = "vsr.onnxruntime_providers.v1"
ONNXRUNTIME_GPU_RECOMMENDED_MIN = "1.21.0"
ONNXRUNTIME_GPU_STABLE_CUDA12_MIN = "1.19.0"
ONNXRUNTIME_PACKAGES = (
    "onnxruntime",
    "onnxruntime-gpu",
    "onnxruntime-directml",
)
OPENCV_WHEEL_STATUS_SCHEMA = "vsr.opencv_wheels.v1"
OPENCV_PACKAGES = (
    "opencv-python",
    "opencv-contrib-python",
    "opencv-python-headless",
    "opencv-contrib-python-headless",
)
OPENCV_REMEDIATION_COMMANDS = (
    "python -m pip uninstall -y opencv-python opencv-contrib-python "
    "opencv-python-headless opencv-contrib-python-headless",
    "python -m pip install \"opencv-python>=4.12.0\"",
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


def _normalise_package_name(value: str) -> str:
    return str(value).lower().replace("_", "-")


def _opencv_owner_candidates(package_versions: Optional[Mapping[str, str]]) -> list[str]:
    installed = [
        package
        for package in OPENCV_PACKAGES
        if _installed_package_version(package, package_versions)
    ]
    if package_versions is not None:
        return installed
    try:
        top_level = metadata.packages_distributions().get("cv2", [])
    except Exception:
        top_level = []
    candidates = []
    installed_set = set(installed)
    for name in top_level:
        normalised = _normalise_package_name(name)
        if normalised in installed_set and normalised not in candidates:
            candidates.append(normalised)
    return candidates or installed


def _opencv_versions_compatible(imported_version: str, package_version: str) -> bool:
    if not imported_version or not package_version:
        return True
    return (
        package_version == imported_version
        or package_version.startswith(imported_version + ".")
        or imported_version.startswith(package_version + ".")
    )


def _opencv_import_probe(timeout: float = 8.0) -> tuple[Optional[str], Optional[str], bool, str]:
    script = (
        "import json\n"
        "try:\n"
        "    import cv2\n"
        "    print(json.dumps({\n"
        "        'version': getattr(cv2, '__version__', '') or '',\n"
        "        'file': getattr(cv2, '__file__', '') or '',\n"
        "        'dnn': hasattr(cv2, 'dnn'),\n"
        "        'error': '',\n"
        "    }))\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'version': None, 'file': None, 'dnn': False, 'error': str(exc)}))\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, None, False, f"cv2 import probe timed out after {timeout:g}s"
    except OSError as exc:
        return None, None, False, str(exc)
    try:
        payload = json.loads((proc.stdout or "").splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        stderr = (proc.stderr or "").strip()
        return None, None, False, stderr or f"cv2 import probe failed with exit {proc.returncode}"
    return (
        payload.get("version"),
        payload.get("file"),
        bool(payload.get("dnn")),
        str(payload.get("error") or ""),
    )


def collect_opencv_wheel_status(
    *,
    package_versions: Optional[Mapping[str, str]] = None,
    imported_version: Optional[str] = None,
    imported_file: Optional[str] = None,
    import_error: Optional[str] = None,
    dnn_available: Optional[bool] = None,
    probe_timeout: float = 8.0,
) -> dict:
    """Return OpenCV wheel ownership/conflict facts for support evidence."""
    package_status = {
        package: {
            "installed": False,
            "version": None,
        }
        for package in OPENCV_PACKAGES
    }
    for package in OPENCV_PACKAGES:
        version = _installed_package_version(package, package_versions)
        package_status[package] = {
            "installed": version is not None,
            "version": version,
        }

    probe_import = (
        imported_version is None
        and imported_file is None
        and import_error is None
        and dnn_available is None
    )
    if probe_import:
        imported_version, imported_file, dnn_available, import_error = (
            _opencv_import_probe(timeout=probe_timeout)
        )
    else:
        import_error = "" if import_error is None else str(import_error)
        dnn_available = bool(dnn_available)

    installed = [
        package for package, status in package_status.items()
        if status["installed"]
    ]
    owner_candidates = _opencv_owner_candidates(package_versions)
    if import_error:
        imported_owner = "not-imported"
    elif len(owner_candidates) == 1:
        imported_owner = owner_candidates[0]
    elif len(owner_candidates) > 1:
        imported_owner = "ambiguous"
    elif imported_version or imported_file:
        imported_owner = "unknown"
    else:
        imported_owner = "not-installed"

    conflict = len(installed) > 1
    warnings = []
    remediation_text = (
        "Run: "
        + " && ".join(OPENCV_REMEDIATION_COMMANDS)
        + " to leave exactly one OpenCV wheel installed."
    )
    if conflict:
        warnings.append({
            "id": "OPENCV-WHEEL-CONFLICT",
            "severity": "high",
            "message": (
                "Multiple OpenCV Python wheels are installed; cv2 imports can "
                "drift between opencv-python, contrib, and headless variants. "
                + remediation_text
            ),
        })
    if import_error and installed:
        warnings.append({
            "id": "OPENCV-IMPORT-FAILED",
            "severity": "high",
            "message": (
                "OpenCV Python wheel metadata is installed, but importing cv2 "
                f"failed: {import_error}. {remediation_text}"
            ),
        })
    if not installed and (imported_version or imported_file):
        warnings.append({
            "id": "OPENCV-UNOWNED-CV2",
            "severity": "medium",
            "message": (
                "cv2 imported, but no known OpenCV wheel metadata was found; "
                + remediation_text
            ),
        })
    if imported_owner in package_status and imported_version:
        owner_version = str(package_status[imported_owner]["version"] or "")
        if not _opencv_versions_compatible(str(imported_version), owner_version):
            warnings.append({
                "id": "OPENCV-VERSION-DRIFT",
                "severity": "medium",
                "message": (
                    f"cv2.__version__ reports {imported_version}, but "
                    f"{imported_owner} metadata reports {owner_version}. "
                    + remediation_text
                ),
            })

    return {
        "schema": OPENCV_WHEEL_STATUS_SCHEMA,
        "packages": package_status,
        "installedDistributions": installed,
        "conflict": conflict,
        "imported": {
            "available": bool((imported_version or imported_file) and not import_error),
            "version": imported_version,
            "file": imported_file,
            "owner": imported_owner,
            "ownerCandidates": owner_candidates,
            "dnnAvailable": bool(dnn_available),
            "error": import_error or "",
        },
        "remediation": {
            "commands": list(OPENCV_REMEDIATION_COMMANDS),
            "summary": remediation_text,
        },
        "warnings": warnings,
    }


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


def _openvino_device_probe() -> tuple[list[str], str]:
    try:
        try:
            from openvino import Core  # type: ignore
        except ImportError:
            from openvino.runtime import Core  # type: ignore
        devices = list(getattr(Core(), "available_devices", []) or [])
        return [str(item) for item in devices], ""
    except Exception as exc:
        return [], str(exc)


def collect_rapidocr_engine_status(
    *,
    package_versions: Optional[Mapping[str, str]] = None,
    openvino_devices: Optional[Sequence[str]] = None,
    openvino_error: Optional[str] = None,
) -> dict:
    """Return RapidOCR runtime-engine facts for ONNX Runtime and OpenVINO."""
    package_status = {
        package: {
            "installed": False,
            "version": None,
        }
        for package in RAPIDOCR_ENGINE_PACKAGES
    }
    for package in RAPIDOCR_ENGINE_PACKAGES:
        version = _installed_package_version(package, package_versions)
        package_status[package] = {
            "installed": version is not None,
            "version": version,
        }

    rapid_version = package_status["rapidocr"]["version"]
    legacy_version = package_status["rapidocr-onnxruntime"]["version"]
    openvino_version = package_status["openvino"]["version"]
    openvino_supported = bool(
        rapid_version and _version_gte(rapid_version, RAPIDOCR_OPENVINO_MIN)
    )
    if openvino_devices is None:
        if openvino_version:
            openvino_devices, probed_error = _openvino_device_probe()
            if openvino_error is None:
                openvino_error = probed_error
        else:
            openvino_devices = []
    openvino_error = "" if openvino_error is None else str(openvino_error)
    openvino_available = bool(
        openvino_supported and openvino_version and not openvino_error
    )
    onnx_available = bool(rapid_version or legacy_version)
    preferred = (
        "openvino" if openvino_available
        else "onnxruntime" if onnx_available
        else "none"
    )
    openvino_provider = (
        "OpenVINO " + "/".join(str(item) for item in openvino_devices)
        if openvino_devices else "OpenVINO"
    )
    warnings = []
    if openvino_version and not openvino_supported:
        warnings.append({
            "id": "RAPIDOCR-OPENVINO-UNSUPPORTED-PACKAGE",
            "severity": "medium",
            "message": (
                "openvino is installed, but RapidOCR OpenVINO routing "
                f"requires rapidocr>={RAPIDOCR_OPENVINO_MIN}; upgrade "
                "rapidocr or use the ONNX Runtime engine."
            ),
        })
    if openvino_supported and openvino_version and openvino_error:
        warnings.append({
            "id": "RAPIDOCR-OPENVINO-PROBE-FAILED",
            "severity": "medium",
            "message": (
                "RapidOCR OpenVINO dependencies are installed, but the "
                f"OpenVINO runtime probe failed: {openvino_error}"
            ),
        })

    return {
        "schema": RAPIDOCR_ENGINE_STATUS_SCHEMA,
        "packages": package_status,
        "engines": {
            "onnxruntime": {
                "available": onnx_available,
                "package": "rapidocr" if rapid_version else (
                    "rapidocr-onnxruntime" if legacy_version else None
                ),
                "provider": "ONNX Runtime",
            },
            "openvino": {
                "available": openvino_available,
                "minimumRapidOCR": RAPIDOCR_OPENVINO_MIN,
                "devices": [str(item) for item in openvino_devices],
                "provider": openvino_provider,
                "probeError": openvino_error,
            },
        },
        "preferredEngine": preferred,
        "preferredProvider": (
            openvino_provider if preferred == "openvino"
            else "ONNX Runtime" if preferred == "onnxruntime"
            else "none"
        ),
        "selectionEnv": "VSR_RAPIDOCR_ENGINE",
        "warnings": warnings,
    }


def collect_onnxruntime_provider_status(
    *,
    package_versions: Optional[Mapping[str, str]] = None,
    providers: Optional[Sequence[str]] = None,
    runtime_version: Optional[str] = None,
    preload_dlls_available: Optional[bool] = None,
    preload_status: Optional[Mapping[str, object]] = None,
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
    cuda_preload_status = preload_status_from_mapping(
        preload_status
        if preload_status is not None
        else collect_onnxruntime_cuda_preload_status()
    )
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
    if (
        gpu_version
        and cuda_preload_status.get("attempted")
        and not cuda_preload_status.get("succeeded")
    ):
        reason = str(cuda_preload_status.get("error") or "unknown error")
        warnings.append({
            "id": "ORT-CUDA-PRELOAD-FAILED",
            "severity": "medium",
            "message": (
                "onnxruntime.preload_dlls() failed before a CUDA provider "
                f"session: {reason}"
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
            "preloadStatus": cuda_preload_status,
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
            "ORT-CUDA-PRELOAD-FAILED",
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


DRIFT_REPORT_SCHEMA = "vsr.dependency_drift.v1"

TRACKED_PACKAGES: Tuple[Tuple[str, str, str], ...] = (
    ("numpy", "1.21.0", ""),
    ("opencv-python", "4.12.0", ""),
    ("Pillow", "12.2.0", ""),
    ("rapidocr", "2.0.0", "4.0.0"),
    ("rapidocr-onnxruntime", "1.4.0", "2.0.0"),
    ("paddleocr", "3.0.0", "4.0.0"),
    ("easyocr", "1.7.0", ""),
    ("onnxruntime", "", ""),
    ("onnxruntime-gpu", "1.21.0", ""),
    ("onnxruntime-directml", "1.18.0", ""),
    ("openvino", "2025.0.0", ""),
    ("simple-lama-inpainting", "0.1.0", ""),
    ("torch", "2.10.0", ""),
    ("torchvision", "0.25.0", ""),
    ("pyinstaller", "", ""),
)

BLOCKED_EXCEPTIONS: Tuple[Tuple[str, str], ...] = (
    ("opencv-python", "libpng < 1.6.54 bundled; CVE-2026-22801 allowed until fixed wheel"),
)


def collect_dependency_drift_report(
    package_versions: Optional[Mapping[str, str]] = None,
) -> dict:
    """Build a local drift report for core and optional stacks."""
    items = []
    for package, minimum, maximum in TRACKED_PACKAGES:
        installed = _installed_package_version(
            _normalise_package_name(package), package_versions,
        )
        status = "not-installed"
        if installed:
            if minimum and not _version_gte(installed, minimum):
                status = "below-minimum"
            elif maximum and _version_gte(installed, maximum):
                status = "above-maximum"
            else:
                status = "ok"
        items.append({
            "package": package,
            "installed": installed or "",
            "minimum": minimum,
            "maximum": maximum,
            "status": status,
        })

    exceptions = []
    for pkg, reason in BLOCKED_EXCEPTIONS:
        exceptions.append({"package": pkg, "reason": reason})

    return {
        "schema": DRIFT_REPORT_SCHEMA,
        "packages": items,
        "blockedExceptions": exceptions,
        "summary": {
            "tracked": len(items),
            "installed": sum(1 for i in items if i["installed"]),
            "ok": sum(1 for i in items if i["status"] == "ok"),
            "belowMinimum": sum(1 for i in items if i["status"] == "below-minimum"),
            "aboveMaximum": sum(1 for i in items if i["status"] == "above-maximum"),
            "notInstalled": sum(1 for i in items if i["status"] == "not-installed"),
        },
    }


def format_drift_report(report: dict) -> str:
    """Format a drift report as a human-readable table."""
    lines = ["Dependency Drift Report", "=" * 60]
    lines.append(
        f"{'Package':<30} {'Installed':<14} {'Min':<10} {'Max':<10} {'Status':<15}"
    )
    lines.append("-" * 79)
    for item in report.get("packages", []):
        lines.append(
            f"{item['package']:<30} "
            f"{item['installed'] or '-':<14} "
            f"{item['minimum'] or '-':<10} "
            f"{item['maximum'] or '-':<10} "
            f"{item['status']:<15}"
        )
    summary = report.get("summary", {})
    lines.append("-" * 79)
    lines.append(
        f"Tracked: {summary.get('tracked', 0)}  "
        f"Installed: {summary.get('installed', 0)}  "
        f"OK: {summary.get('ok', 0)}  "
        f"Below min: {summary.get('belowMinimum', 0)}  "
        f"Above max: {summary.get('aboveMaximum', 0)}"
    )
    blocked = report.get("blockedExceptions", [])
    if blocked:
        lines.append("")
        lines.append("Blocked exceptions:")
        for exc in blocked:
            lines.append(f"  {exc['package']}: {exc['reason']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    try:
        enforce_ocr_dependency_caps()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("OCR dependency version caps OK")
    print()
    report = collect_dependency_drift_report()
    print(format_drift_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
