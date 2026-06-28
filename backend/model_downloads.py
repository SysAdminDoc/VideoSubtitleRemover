"""First-run model download guidance.

VSR deliberately delegates model fetching to the optional libraries that own
those models. This module predicts when a first call is likely to trigger a
large download so the GUI can show an indeterminate progress/status message
before the library blocks on network or cache work.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

from backend.dependency_caps import (
    collect_opencv_wheel_status,
    collect_onnxruntime_provider_status,
    collect_rapidocr_engine_status,
)
from backend.language_support import language_support_status
from backend.remote_model_policy import resolve_remote_model_source


@dataclass(frozen=True)
class ModelDownloadHint:
    label: str
    size_estimate: str
    detail: str
    cache_hint: str


_LAMA_FILENAMES = ("lama_fp32.onnx", "lama.onnx", "inpainting_lama_2025jan.onnx")
_VACE_REPO_ID = "Wan-AI/Wan2.1-VACE-1.3B"
_VIDEOPAINTER_REPO_ID = "TencentARC/VideoPainter"
_VIDEOPAINTER_BASE_REPO_ID = "THUDM/CogVideoX-5b-I2V"
_FLOED_SOURCE_URL = "https://github.com/NevSNev/FloED-main"
_MATANYONE_SOURCE_URL = "https://github.com/pq-yang/MatAnyone2"
_MATANYONE_MODEL_ID = "PeiqingYang/MatAnyone2"
_COTRACKER_SOURCE_URL = "https://github.com/facebookresearch/co-tracker"
_TRUE_FLAG_VALUES = {"1", "true", "yes", "on"}
_WHISPER_SIZES = {
    "tiny": "~75 MB",
    "base": "~145 MB",
    "small": "~460 MB",
    "medium": "~1.5 GB",
    "large": "~3 GB",
    "large-v2": "~3 GB",
    "large-v3": "~3 GB",
}


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _env_truthy(env: Mapping[str, str], name: str) -> bool:
    return str(env.get(name, "") or "").strip().lower() in _TRUE_FLAG_VALUES


def _home(env: Mapping[str, str]) -> Path:
    value = str(env.get("USERPROFILE") or env.get("HOME") or "").strip()
    return Path(value) if value else Path.home()


def _app_model_dir(env: Mapping[str, str]) -> Path:
    root = str(env.get("APPDATA") or "").strip()
    if root:
        return Path(root) / "VideoSubtitleRemoverPro" / "models"
    return _home(env) / ".config" / "VideoSubtitleRemoverPro" / "models"


def _has_any_file(root: Path, names: Sequence[str] = (), suffixes: Sequence[str] = ()) -> bool:
    if not root.exists():
        return False
    try:
        if root.is_file():
            return (not names or root.name in names) and (
                not suffixes or root.suffix.lower() in suffixes
            )
        for item in root.rglob("*"):
            if not item.is_file():
                continue
            if names and item.name in names:
                return True
            if suffixes and item.suffix.lower() in suffixes:
                return True
    except OSError:
        return False
    return False


def _mode_value(config) -> str:
    mode = getattr(config, "mode", "sttn")
    return str(getattr(mode, "value", mode)).lower()


def _lama_weight_present(env: Mapping[str, str]) -> bool:
    for key in ("VSR_LAMA_ONNX", "VSR_OPENCV_LAMA"):
        value = str(env.get(key, "") or "").strip()
        if value and Path(value).is_file():
            return True
    home = _home(env)
    for root in (
        _app_model_dir(env),
        home / ".cache" / "huggingface" / "hub",
        home / ".cache" / "opencv_models",
        home / ".cache" / "torch" / "hub" / "checkpoints",
        home / ".cache" / "simple_lama",
    ):
        if _has_any_file(root, names=_LAMA_FILENAMES):
            return True
    return False


def _vace_checkpoint_present(env: Mapping[str, str]) -> bool:
    for key in ("VSR_VACE_CKPT_DIR", "VSR_VACE_MODEL_DIR", "VSR_VACE_WEIGHTS"):
        value = str(env.get(key, "") or "").strip()
        if value and Path(value).exists():
            return True
    app_cache = _app_model_dir(env) / "vace" / "Wan2.1-VACE-1.3B"
    if _has_any_file(app_cache, suffixes=(".safetensors", ".pth", ".bin", ".json")):
        return True
    repo = str(env.get("VSR_VACE_REPO_ID") or _VACE_REPO_ID)
    return _hf_repo_cached(env, repo)


def _videopainter_checkpoint_present(env: Mapping[str, str]) -> bool:
    for key in (
        "VSR_VIDEOPAINTER_CKPT_DIR",
        "VSR_VIDEOPAINTER_MODEL_DIR",
        "VSR_VIDEOPAINTER_WEIGHTS",
        "VSR_VIDEOPAINTER_BRANCH_DIR",
    ):
        value = str(env.get(key, "") or "").strip()
        if value and Path(value).exists():
            return True
    app_cache = _app_model_dir(env) / "videopainter"
    if _has_any_file(app_cache, suffixes=(".safetensors", ".json", ".pt")):
        return True
    return (
        _hf_repo_cached(env, _VIDEOPAINTER_REPO_ID)
        or _hf_repo_cached(env, _VIDEOPAINTER_BASE_REPO_ID)
    )


def _videopainter_command_present(env: Mapping[str, str]) -> bool:
    command = str(env.get("VSR_VIDEOPAINTER_COMMAND", "") or "").strip()
    if not command:
        return _module_available("videopainter")
    return True


def _floed_checkpoint_present(env: Mapping[str, str]) -> bool:
    for key in ("VSR_FLOED_WEIGHTS", "VSR_FLOED_CKPT", "VSR_FLOED_CKPT_DIR"):
        value = str(env.get(key, "") or "").strip()
        if value and Path(value).exists():
            return True
    app_cache = _app_model_dir(env) / "floed"
    return _has_any_file(app_cache, suffixes=(".ckpt", ".safetensors", ".pth"))


def _floed_command_present(env: Mapping[str, str]) -> bool:
    command = str(env.get("VSR_FLOED_COMMAND", "") or "").strip()
    if not command:
        return _module_available("floed")
    return True


def _matanyone_checkpoint_present(env: Mapping[str, str]) -> bool:
    value = str(env.get("VSR_MATANYONE_PATH", "") or "").strip()
    if value and Path(value).exists():
        return True
    app_cache = _app_model_dir(env) / "matanyone2"
    if _has_any_file(app_cache, suffixes=(".pth", ".safetensors", ".bin", ".json")):
        return True
    model_id = str(env.get("VSR_MATANYONE_MODEL_ID") or _MATANYONE_MODEL_ID)
    return _hf_repo_cached(env, model_id)


def _matanyone_package_present() -> bool:
    return _module_available("matanyone2") or _module_available("matanyone")


def _cotracker_package_present() -> bool:
    return _module_available("torch")


def _hf_repo_cached(env: Mapping[str, str], repo: str) -> bool:
    home = _home(env)
    escaped = "models--" + repo.replace("/", "--")
    return (home / ".cache" / "huggingface" / "hub" / escaped).exists()


def _easyocr_cached(env: Mapping[str, str]) -> bool:
    return _has_any_file(_home(env) / ".EasyOCR" / "model", suffixes=(".pth",))


def _paddleocr_cached(env: Mapping[str, str]) -> bool:
    return _has_any_file(_home(env) / ".paddleocr", suffixes=(".pdparams", ".onnx"))


def _append_detection_hints(hints: list[ModelDownloadHint], env: Mapping[str, str]) -> None:
    if _module_available("rapidocr") or _module_available("rapidocr_onnxruntime"):
        return
    if _module_available("paddleocr") and not _paddleocr_cached(env):
        hints.append(ModelDownloadHint(
            label="PaddleOCR text models",
            size_estimate="~50-200 MB",
            detail="First PaddleOCR detection may download PP-OCR assets.",
            cache_hint="%USERPROFILE%\\.paddleocr",
        ))
        return
    if _module_available("easyocr") and not _easyocr_cached(env):
        hints.append(ModelDownloadHint(
            label="EasyOCR detection models",
            size_estimate="~80-150 MB",
            detail="First EasyOCR fallback may download detection and recognition weights.",
            cache_hint="%USERPROFILE%\\.EasyOCR\\model",
        ))


def _append_vlm_hints(hints: list[ModelDownloadHint], env: Mapping[str, str]) -> None:
    selected = str(env.get("VSR_VLM_OCR", "") or "").strip().lower()
    paddle_vl_selected = (
        _env_truthy(env, "VSR_PADDLEOCR_VL")
        or selected in {
            "paddleocr-vl-llama",
            "paddleocr-vl-llamacpp",
            "paddleocr-vl15",
            "paddleocr-vl-1.5",
        }
    )
    if paddle_vl_selected:
        hints.append(ModelDownloadHint(
            label="PaddleOCR-VL-1.5 GGUF",
            size_estimate="model-size dependent",
            detail=(
                "Start llama-server with a local PaddleOCR-VL-1.5 GGUF "
                "model before enabling VSR_PADDLEOCR_VL."
            ),
            cache_hint="local llama.cpp model directory",
        ))
        return
    if selected != "florence2":
        return
    source = resolve_remote_model_source("florence2", env)
    if source.allowed and source.source_type == "remote" and not _hf_repo_cached(env, source.policy.repo):
        hints.append(ModelDownloadHint(
            label="Florence-2 VLM OCR",
            size_estimate="~450 MB+",
            detail="First Florence-2 OCR use may download model and processor files.",
            cache_hint="%USERPROFILE%\\.cache\\huggingface\\hub",
        ))


def _append_lama_hints(hints: list[ModelDownloadHint], config, env: Mapping[str, str]) -> None:
    mode = _mode_value(config)
    if mode not in {"lama", "auto", "propainter"}:
        return
    if _lama_weight_present(env):
        return
    if not _env_truthy(env, "VSR_ENABLE_PYTORCH_LAMA"):
        return
    if (_module_available("simple_lama_inpainting")
            or _module_available("simple_lama_inpainting.models")):
        hints.append(ModelDownloadHint(
            label="LaMa inpainting weights",
            size_estimate="~200 MB",
            detail="First PyTorch LaMa fallback may download model weights.",
            cache_hint="%USERPROFILE%\\.cache\\torch or %USERPROFILE%\\.cache\\simple_lama",
        ))


def _append_vace_hints(hints: list[ModelDownloadHint], config, env: Mapping[str, str]) -> None:
    mode = _mode_value(config)
    if mode != "vace" and not _env_truthy(env, "VSR_VACE"):
        return
    if _vace_checkpoint_present(env):
        return
    if _env_truthy(env, "VSR_VACE_AUTO_FETCH"):
        hints.append(ModelDownloadHint(
            label="Wan2.1-VACE 1.3B checkpoint",
            size_estimate="multi-GB",
            detail=(
                "VACE auto-fetch will download the Wan-AI/Wan2.1-VACE-1.3B "
                "snapshot through huggingface_hub before inference."
            ),
            cache_hint="%APPDATA%\\VideoSubtitleRemoverPro\\models\\vace",
        ))
    else:
        hints.append(ModelDownloadHint(
            label="Wan2.1-VACE 1.3B checkpoint",
            size_estimate="multi-GB",
            detail=(
                "Set VSR_VACE_CKPT_DIR to a local Wan2.1-VACE-1.3B snapshot "
                "or set VSR_VACE_AUTO_FETCH=1."
            ),
            cache_hint="%APPDATA%\\VideoSubtitleRemoverPro\\models\\vace",
        ))


def _append_videopainter_hints(hints: list[ModelDownloadHint], config, env: Mapping[str, str]) -> None:
    mode = _mode_value(config)
    if mode != "videopainter" and not _env_truthy(env, "VSR_VIDEOPAINTER"):
        return
    if not _videopainter_checkpoint_present(env):
        hints.append(ModelDownloadHint(
            label="VideoPainter and CogVideoX checkpoints",
            size_estimate="multi-GB",
            detail=(
                "Download TencentARC/VideoPainter and THUDM/CogVideoX-5b-I2V "
                "manually, review the research/non-commercial license terms, "
                "then set VSR_VIDEOPAINTER_CKPT_DIR."
            ),
            cache_hint="%APPDATA%\\VideoSubtitleRemoverPro\\models\\videopainter",
        ))
    if not _videopainter_command_present(env):
        hints.append(ModelDownloadHint(
            label="VideoPainter local wrapper",
            size_estimate="local checkout",
            detail=(
                "Set VSR_VIDEOPAINTER_COMMAND to a reviewed local wrapper "
                "that accepts --input-video, --mask-video, and --output-video."
            ),
            cache_hint="local VideoPainter checkout",
        ))


def _append_floed_hints(hints: list[ModelDownloadHint], config, env: Mapping[str, str]) -> None:
    mode = _mode_value(config)
    if mode != "floed" and not _env_truthy(env, "VSR_FLOED"):
        return
    if not _floed_checkpoint_present(env):
        hints.append(ModelDownloadHint(
            label="FloED checkpoint",
            size_estimate="model-size dependent",
            detail=(
                "Download the FloED checkpoint from the upstream Apache-2.0 "
                "repo release instructions, review provenance, then set "
                "VSR_FLOED_WEIGHTS or VSR_FLOED_CKPT_DIR."
            ),
            cache_hint="%APPDATA%\\VideoSubtitleRemoverPro\\models\\floed",
        ))
    if not _floed_command_present(env):
        hints.append(ModelDownloadHint(
            label="FloED local wrapper",
            size_estimate="local checkout",
            detail=(
                "Set VSR_FLOED_COMMAND to a reviewed local wrapper that "
                "accepts --input-dir, --mask-dir, and --output-dir."
            ),
            cache_hint=_FLOED_SOURCE_URL,
        ))


def _append_matanyone_hints(hints: list[ModelDownloadHint], config, env: Mapping[str, str]) -> None:
    if not bool(getattr(config, "matanyone_refine", False)) and not _env_truthy(env, "VSR_MATANYONE"):
        return
    if not _env_truthy(env, "VSR_MATANYONE"):
        hints.append(ModelDownloadHint(
            label="MatAnyone 2 opt-in",
            size_estimate="local setup",
            detail=(
                "Set VSR_MATANYONE=1 only after reviewing the upstream "
                "NTU S-Lab License 1.0 terms."
            ),
            cache_hint=_MATANYONE_SOURCE_URL,
        ))
    if not _matanyone_package_present():
        hints.append(ModelDownloadHint(
            label="MatAnyone 2 package",
            size_estimate="local checkout",
            detail=(
                "Install the reviewed MatAnyone2 package from the upstream "
                "repo before enabling --matanyone-refine."
            ),
            cache_hint=_MATANYONE_SOURCE_URL,
        ))
    if not _matanyone_checkpoint_present(env):
        hints.append(ModelDownloadHint(
            label="MatAnyone 2 checkpoint",
            size_estimate="model-size dependent",
            detail=(
                "Set VSR_MATANYONE_PATH to a reviewed matanyone2.pth or "
                "snapshot; unpinned PyTorch weights require "
                "VSR_ALLOW_UNVERIFIED_MODELS=1."
            ),
            cache_hint="%APPDATA%\\VideoSubtitleRemoverPro\\models\\matanyone2",
        ))


def _append_cotracker_hints(hints: list[ModelDownloadHint], config, env: Mapping[str, str]) -> None:
    if not bool(getattr(config, "cotracker_propagate", False)) and not _env_truthy(env, "VSR_COTRACKER"):
        return
    if not _env_truthy(env, "VSR_COTRACKER"):
        hints.append(ModelDownloadHint(
            label="CoTracker3 opt-in",
            size_estimate="local setup",
            detail=(
                "Set VSR_COTRACKER=1 only after configuring a reviewed local "
                "co-tracker checkout or pinned revision."
            ),
            cache_hint=_COTRACKER_SOURCE_URL,
        ))
    if not _cotracker_package_present():
        hints.append(ModelDownloadHint(
            label="CoTracker3 PyTorch runtime",
            size_estimate="package-size dependent",
            detail="Install PyTorch before enabling --cotracker-propagate.",
            cache_hint="https://pytorch.org/",
        ))
    source = resolve_remote_model_source("cotracker3", env)
    if not source.allowed:
        hints.append(ModelDownloadHint(
            label="CoTracker3 trusted source",
            size_estimate="local checkout or pinned ref",
            detail=source.reason,
            cache_hint=_COTRACKER_SOURCE_URL,
        ))


def _append_whisper_hints(hints: list[ModelDownloadHint], config, env: Mapping[str, str]) -> None:
    if not bool(getattr(config, "whisper_fallback", False)):
        return
    backend = str(getattr(config, "whisper_backend", "faster-whisper") or "").lower()
    if backend != "faster-whisper":
        return
    if str(getattr(config, "whisper_model_path", "") or "").strip():
        return
    size = str(getattr(config, "whisper_model_size", "tiny") or "tiny").lower()
    hints.append(ModelDownloadHint(
        label=f"Whisper {size} model",
        size_estimate=_WHISPER_SIZES.get(size, "model-size dependent"),
        detail="First faster-whisper fallback may download speech model weights.",
        cache_hint="%USERPROFILE%\\.cache\\huggingface\\hub",
    ))


def pending_model_download_hints(
    config,
    env: Optional[Mapping[str, str]] = None,
) -> Tuple[ModelDownloadHint, ...]:
    source_env = os.environ if env is None else env
    hints: list[ModelDownloadHint] = []
    _append_vlm_hints(hints, source_env)
    _append_detection_hints(hints, source_env)
    _append_lama_hints(hints, config, source_env)
    _append_vace_hints(hints, config, source_env)
    _append_videopainter_hints(hints, config, source_env)
    _append_floed_hints(hints, config, source_env)
    _append_cotracker_hints(hints, config, source_env)
    _append_matanyone_hints(hints, config, source_env)
    _append_whisper_hints(hints, config, source_env)
    return tuple(hints)


def summarize_hints(hints: Iterable[ModelDownloadHint]) -> str:
    parts = [f"{h.label} ({h.size_estimate})" for h in hints]
    return "; ".join(parts)


def _dist_version(package_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_status(
    label: str,
    module_name: str,
    *,
    package_name: Optional[str] = None,
    next_action: str = "",
) -> dict:
    available = _module_available(module_name)
    return {
        "name": label,
        "available": available,
        "version": _dist_version(package_name or module_name) if available else None,
        "status": "available" if available else "not_installed",
        "next_action": "" if available else next_action,
    }


def _onnxruntime_provider_status() -> dict:
    status = collect_onnxruntime_provider_status()
    package_versions = [
        str(item.get("version") or "")
        for item in status.get("packages", {}).values()
        if item.get("installed")
    ]
    version = status.get("runtimeVersion") or (package_versions[0] if package_versions else None)
    warnings = list(status.get("warnings", []) or [])
    if not version and not status.get("availableProviders"):
        next_action = (
            "Install onnxruntime, onnxruntime-gpu, or onnxruntime-directml "
            "for ONNX backends."
        )
    elif warnings:
        next_action = str(warnings[0].get("message") or "")
    else:
        next_action = ""
    cuda = status.get("cuda", {}) if isinstance(status.get("cuda"), Mapping) else {}
    preload = (
        cuda.get("preloadStatus", {})
        if isinstance(cuda.get("preloadStatus"), Mapping) else {}
    )
    if preload.get("attempted") and not preload.get("succeeded"):
        next_action = (
            "Repair CUDA/cuDNN DLL visibility; ONNX Runtime preload failed: "
            + str(preload.get("error") or "unknown error")
        )
    return {
        "available": bool(version or status.get("availableProviders")),
        "version": version,
        "providers": list(status.get("availableProviders", []) or []),
        "packages": status.get("packages", {}),
        "cuda": status.get("cuda", {}),
        "directml": status.get("directml", {}),
        "warnings": warnings,
        "next_action": next_action,
    }


def _opencv_runtime_status() -> dict:
    if not _module_available("cv2"):
        wheel_status = collect_opencv_wheel_status(import_error="cv2 module not found")
        return {
            "available": False,
            "version": None,
            "dnn_available": False,
            "opencv5": False,
            "wheel_status": wheel_status,
            "warnings": list(wheel_status.get("warnings", []) or []),
            "next_action": "Install opencv-python; OpenCV fallback is required.",
        }
    try:
        import cv2
        version = getattr(cv2, "__version__", "")
        dnn_available = hasattr(cv2, "dnn")
        wheel_status = collect_opencv_wheel_status(
            imported_version=version,
            imported_file=str(getattr(cv2, "__file__", "") or ""),
            dnn_available=dnn_available,
        )
        warnings = list(wheel_status.get("warnings", []) or [])
        parts = []
        for raw in version.split(".")[:3]:
            digits = ""
            for ch in raw:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            parts.append(int(digits) if digits else 0)
        while len(parts) < 3:
            parts.append(0)
        opencv5 = tuple(parts) >= (5, 0, 0)
        next_action = "" if opencv5 else "OpenCV DNN LaMa needs opencv-python 5.x when those wheels ship."
        if warnings:
            next_action = str(wheel_status.get("remediation", {}).get("summary") or next_action)
        return {
            "available": True,
            "version": version or _dist_version("opencv-python"),
            "dnn_available": dnn_available,
            "opencv5": opencv5,
            "wheel_status": wheel_status,
            "warnings": warnings,
            "next_action": next_action,
        }
    except Exception as exc:
        wheel_status = collect_opencv_wheel_status(import_error=str(exc))
        return {
            "available": False,
            "version": _dist_version("opencv-python"),
            "dnn_available": False,
            "opencv5": False,
            "error": str(exc),
            "wheel_status": wheel_status,
            "warnings": list(wheel_status.get("warnings", []) or []),
            "next_action": "Repair OpenCV; runtime probing failed.",
        }


def _pynv_decode_status(env: Mapping[str, str]) -> dict:
    try:
        from backend.decode_accel import pynv_decode_status
        return pynv_decode_status(env)
    except Exception as exc:
        return {
            "available": False,
            "version": None,
            "enabled": _env_truthy(env, "VSR_PYNVVIDEOCODEC"),
            "status": "probe_failed",
            "error": str(exc),
            "next_action": "Repair PyNvVideoCodec installation; runtime probing failed.",
        }


def _rapidocr_status() -> dict:
    from backend.onnx_model_info import rapidocr_release_provenance

    provenance = rapidocr_release_provenance()
    engine_status = collect_rapidocr_engine_status()
    package = provenance.get("package", {})
    package_name = package.get("name") if isinstance(package, Mapping) else ""
    installed = bool(package_name)
    required = list(provenance.get("required_assets", []))
    model_count = int(provenance.get("model_count") or 0)
    compatible = bool(provenance.get("packaging_compatible"))
    missing_required = [
        item.get("name")
        for item in required
        if item.get("required") and not item.get("present")
    ]
    if not installed:
        status = "not_installed"
        next_action = "Install rapidocr for the fastest OCR path, or use PaddleOCR/EasyOCR/OpenCV fallback."
    elif compatible:
        status = "ready"
        next_action = ""
    else:
        status = "incomplete"
        next_action = "Reinstall RapidOCR or rebuild the bundle with RapidOCR model data collected."
    return {
        "name": "RapidOCR",
        "available": installed and compatible,
        "installed": installed,
        "package": package,
        "status": status,
        "model_count": model_count,
        "model_families": list(provenance.get("model_families", [])),
        "required_assets": required,
        "missing_required_assets": missing_required,
        "engines": engine_status.get("engines", {}),
        "preferred_engine": engine_status.get("preferredEngine"),
        "provider": engine_status.get("preferredProvider"),
        "engine_warnings": list(engine_status.get("warnings", []) or []),
        "hash_status": "hashed" if model_count else "missing",
        "next_action": next_action,
    }


def _adapter_hash_status(adapter_name: str, model_path: Optional[str]) -> dict:
    from backend.adapter_manifest import verify_adapter_path

    if not model_path:
        return {
            "configured": False,
            "exists": False,
            "allowed": False,
            "filename": None,
            "hash_status": "missing",
            "reason": "model file is not configured or discoverable",
        }
    result = verify_adapter_path(adapter_name, model_path)
    payload = result.as_dict(include_path=False)
    return {
        "configured": True,
        "exists": bool(payload.get("exists")),
        "allowed": bool(payload.get("allowed")),
        "filename": payload.get("filename"),
        "hash_status": payload.get("hashStatus"),
        "reason": payload.get("reason"),
        "expected_filenames": payload.get("expectedFilenames", []),
        "license": payload.get("license"),
        "source_url": payload.get("sourceUrl"),
    }


def _discover_lama_weight(
    env: Mapping[str, str],
    env_var: str,
    filenames: Sequence[str],
    *,
    include_opencv_cache: bool = False,
) -> Optional[str]:
    explicit = str(env.get(env_var, "") or "").strip()
    if explicit and Path(explicit).is_file():
        return explicit
    home = _home(env)
    search_dirs = [
        _app_model_dir(env),
        home / ".cache" / "huggingface" / "hub",
    ]
    if include_opencv_cache:
        search_dirs.append(home / ".cache" / "opencv_models")
    search_dirs.extend([
        home / ".cache" / "torch" / "hub" / "checkpoints",
        home / ".cache" / "simple_lama",
    ])
    for root in search_dirs:
        if not root.is_dir():
            continue
        for name in filenames:
            direct = root / name
            if direct.is_file():
                return str(direct)
            try:
                for match in root.rglob(name):
                    if match.is_file():
                        return str(match)
            except OSError:
                continue
    return None


def _lama_model_status(env: Mapping[str, str], providers: Mapping[str, Any]) -> list[dict]:
    onnx_path = _discover_lama_weight(
        env,
        "VSR_LAMA_ONNX",
        _LAMA_FILENAMES[:2],
    )
    opencv_path = _discover_lama_weight(
        env,
        "VSR_OPENCV_LAMA",
        _LAMA_FILENAMES,
        include_opencv_cache=True,
    )
    opencv_adapter = (
        "opencv-lama"
        if opencv_path and Path(opencv_path).name == "inpainting_lama_2025jan.onnx"
        else "lama-onnx"
    )
    onnx_hash = _adapter_hash_status("lama-onnx", onnx_path)
    opencv_hash = _adapter_hash_status(opencv_adapter, opencv_path)
    ort_ready = bool(providers.get("onnxruntime", {}).get("available"))
    opencv_ready = bool(providers.get("opencv", {}).get("opencv5"))
    pytorch_opt_in = _env_truthy(env, "VSR_ENABLE_PYTORCH_LAMA")
    simple_lama_available = _module_available("simple_lama_inpainting")

    return [
        {
            "name": "Temporal Background Exposure",
            "kind": "inpaint",
            "available": True,
            "status": "built_in",
            "model_file": None,
            "hash_status": "not_required",
            "provider": "numpy/OpenCV",
            "next_action": "",
        },
        {
            "name": "LaMa ONNX",
            "kind": "inpaint",
            "available": ort_ready and bool(onnx_hash.get("allowed")),
            "status": "ready" if ort_ready and onnx_hash.get("allowed") else "missing",
            "model_file": onnx_hash.get("filename"),
            "hash_status": onnx_hash.get("hash_status"),
            "provider": "ONNX Runtime",
            "expected_files": onnx_hash.get("expected_filenames", []),
            "next_action": (
                "" if ort_ready and onnx_hash.get("allowed")
                else "Set VSR_LAMA_ONNX or place lama_fp32.onnx/lama.onnx in the app model cache."
            ),
        },
        {
            "name": "OpenCV DNN LaMa",
            "kind": "inpaint",
            "available": opencv_ready and bool(opencv_hash.get("allowed")),
            "status": "ready" if opencv_ready and opencv_hash.get("allowed") else "not_ready",
            "model_file": opencv_hash.get("filename"),
            "hash_status": opencv_hash.get("hash_status"),
            "provider": "OpenCV DNN",
            "expected_files": opencv_hash.get("expected_filenames", []),
            "next_action": (
                "" if opencv_ready and opencv_hash.get("allowed")
                else "Install OpenCV 5.x and set VSR_OPENCV_LAMA for the OpenCV DNN LaMa path."
            ),
        },
        {
            "name": "PyTorch LaMa",
            "kind": "inpaint",
            "available": pytorch_opt_in and simple_lama_available,
            "status": (
                "ready" if pytorch_opt_in and simple_lama_available
                else "disabled" if simple_lama_available
                else "not_installed"
            ),
            "model_file": "big-lama.pt" if simple_lama_available else None,
            "hash_status": "known_hash" if simple_lama_available else "missing",
            "provider": "PyTorch",
            "opt_in_env": "VSR_ENABLE_PYTORCH_LAMA",
            "next_action": (
                "" if pytorch_opt_in and simple_lama_available
                else "Set VSR_ENABLE_PYTORCH_LAMA=1 only if the PyTorch fallback is intentionally needed."
            ),
        },
        {
            "name": "OpenCV inpaint",
            "kind": "inpaint",
            "available": bool(providers.get("opencv", {}).get("available")),
            "status": "built_in",
            "model_file": None,
            "hash_status": "not_required",
            "provider": "OpenCV",
            "next_action": "",
        },
    ]


def _summarize_backend_status(status: Mapping[str, Any]) -> dict:
    detection_items = list(status.get("detection", []))
    inpaint_items = list(status.get("inpainting", []))
    providers = status.get("providers", {})
    selected_mode = str(status.get("selected_mode") or "sttn").lower()
    detection = next(
        (item for item in detection_items if item.get("available")),
        detection_items[-1] if detection_items else {},
    )
    lama_candidates = [
        item for item in inpaint_items
        if item.get("name") in {"LaMa ONNX", "OpenCV DNN LaMa", "PyTorch LaMa"}
    ]
    ready_lama = next((item for item in lama_candidates if item.get("available")), None)
    ort = providers.get("onnxruntime", {})
    provider_text = (
        ", ".join(ort.get("providers", [])[:3])
        if ort.get("available") else "ONNX Runtime not installed"
    )
    cuda = ort.get("cuda", {}) if isinstance(ort.get("cuda"), Mapping) else {}
    if cuda.get("packageInstalled"):
        provider_text += f"; CUDA {cuda.get('packageChannel') or 'unknown'}"
    directml = (
        ort.get("directml", {})
        if isinstance(ort.get("directml"), Mapping) else {}
    )
    if directml.get("packageInstalled"):
        provider_text += "; DirectML package installed"
    pynv = providers.get("pynv_decode", {})
    if isinstance(pynv, Mapping):
        if pynv.get("available"):
            provider_text += "; PyNvVideoCodec available"
        elif pynv.get("enabled"):
            provider_text += "; PyNvVideoCodec not installed"
    rapid = detection_items[0] if detection_items else {}
    rapid_text = (
        f"RapidOCR {rapid.get('model_count', 0)} model file(s)"
        if rapid.get("installed") else "RapidOCR not installed"
    )
    if rapid.get("provider") and rapid.get("provider") != "none":
        rapid_text += f"; {rapid.get('provider')}"
    lama_text = (
        f"{ready_lama.get('name')} {ready_lama.get('hash_status')}"
        if ready_lama else "LaMa neural weights not ready"
    )
    rapid_ready = rapid.get("status") == "ready"
    next_action = str(rapid.get("next_action") or "") if not rapid_ready else ""
    if not next_action and selected_mode in {"lama", "auto", "propainter"}:
        if ready_lama is None:
            onnx = next(
                (item for item in lama_candidates if item.get("name") == "LaMa ONNX"),
                {},
            )
            next_action = str(onnx.get("next_action") or "")
    if not next_action:
        next_action = "No backend setup action needed."
    rapid_hash = (
        "RapidOCR hashes recorded"
        if rapid.get("model_count") else "RapidOCR hashes unavailable"
    )
    lama_hash = (
        str(ready_lama.get("hash_status"))
        if ready_lama else "LaMa missing"
    )
    return {
        "detection": (
            f"{detection.get('name', 'OpenCV fallback')}"
            + (
                f" via {detection.get('provider')}"
                if detection.get("name") == "RapidOCR"
                and detection.get("provider")
                and detection.get("provider") != "none"
                else ""
            )
            + f" ({detection.get('status', 'ready')})"
        ),
        "inpainting": (
            f"{ready_lama.get('name')} ready"
            if ready_lama else "TBE/OpenCV ready; neural LaMa optional"
        ),
        "providers": provider_text,
        "language_support": (
            status.get("language_support", {}).get("summary")
            if isinstance(status.get("language_support"), Mapping)
            else "Language support status unavailable."
        ),
        "model_files": f"{rapid_text}; {lama_text}",
        "hash_status": f"{rapid_hash}; {lama_hash}",
        "next_action": next_action,
        "tone": "success" if ready_lama or detection.get("name") == "RapidOCR" else "warning",
    }


def installed_backend_status(
    config=None,
    env: Optional[Mapping[str, str]] = None,
) -> dict:
    """Return privacy-safe installed backend/model status for UI and support."""
    source_env = os.environ if env is None else env
    providers = {
        "onnxruntime": _onnxruntime_provider_status(),
        "opencv": _opencv_runtime_status(),
        "pynv_decode": _pynv_decode_status(source_env),
        "torch": _module_status(
            "PyTorch",
            "torch",
            next_action="Install torch only for opt-in PyTorch fallback paths.",
        ),
    }
    detection = [
        _rapidocr_status(),
        _module_status(
            "PaddleOCR",
            "paddleocr",
            next_action="Install paddleocr for the secondary OCR path.",
        ),
        _module_status(
            "EasyOCR",
            "easyocr",
            next_action="Install easyocr for the legacy OCR fallback.",
        ),
        {
            "name": "OpenCV fallback",
            "available": bool(providers["opencv"].get("available")),
            "status": "built_in",
            "next_action": "",
        },
    ]
    inpainting = _lama_model_status(source_env, providers)
    hints = [
        {
            "label": hint.label,
            "size_estimate": hint.size_estimate,
            "detail": hint.detail,
            "cache_hint": hint.cache_hint,
        }
        for hint in pending_model_download_hints(config or object(), source_env)
    ]
    status = {
        "schema": "vsr.backend_status.v1",
        "selected_mode": _mode_value(config or object()),
        "providers": providers,
        "detection": detection,
        "language_support": language_support_status(detection),
        "inpainting": inpainting,
        "pending_downloads": hints,
    }
    status["summary"] = _summarize_backend_status(status)
    return status
