"""First-run model download guidance.

VSR deliberately delegates model fetching to the optional libraries that own
those models. This module predicts when a first call is likely to trigger a
large download so the GUI can show an indeterminate progress/status message
before the library blocks on network or cache work.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple

from backend.remote_model_policy import resolve_remote_model_source


@dataclass(frozen=True)
class ModelDownloadHint:
    label: str
    size_estimate: str
    detail: str
    cache_hint: str


_LAMA_FILENAMES = ("lama_fp32.onnx", "lama.onnx", "inpainting_lama_2025jan.onnx")
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
    if _module_available("simple_lama_inpainting") or _module_available("simple_lama_inpainting.models"):
        hints.append(ModelDownloadHint(
            label="LaMa inpainting weights",
            size_estimate="~200 MB",
            detail="First PyTorch LaMa fallback may download model weights.",
            cache_hint="%USERPROFILE%\\.cache\\torch or %USERPROFILE%\\.cache\\simple_lama",
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
    _append_whisper_hints(hints, config, source_env)
    return tuple(hints)


def summarize_hints(hints: Iterable[ModelDownloadHint]) -> str:
    parts = [f"{h.label} ({h.size_estimate})" for h in hints]
    return "; ".join(parts)
