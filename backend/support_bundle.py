"""Redacted support bundle generation for bug reports."""

from __future__ import annotations

import datetime as _dt
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from backend.cache_inventory import discover_caches, model_cache_status
from backend.crash_reporter import _path_scrub
from backend.dependency_caps import (
    collect_onnxruntime_provider_status,
    collect_opencv_wheel_status,
    collect_rapidocr_engine_status,
)
from backend.ffmpeg_profiles import (
    collect_ffmpeg_capability_profiles,
    ffmpeg_profile_entries,
)
from backend.model_downloads import installed_backend_status
from backend.security_checks import opencv_libpng_status


SUPPORT_BUNDLE_SCHEMA = "vsr.support_bundle.v1"
MAX_LOG_BYTES = 256 * 1024
_DEPENDENCY_PACKAGES = (
    "numpy",
    "opencv-python",
    "opencv-contrib-python",
    "Pillow",
    "torch",
    "torchvision",
    "onnxruntime",
    "onnxruntime-gpu",
    "onnxruntime-directml",
    "openvino",
    "paddleocr",
    "rapidocr",
    "rapidocr-onnxruntime",
    "easyocr",
    "sentry-sdk",
)
_SENSITIVE_KEYS = {
    "abs_path",
    "dsn",
    "file_path",
    "filename",
    "input",
    "input_path",
    "log_file",
    "model_path",
    "output",
    "output_path",
    "password",
    "path",
    "settings_file",
    "token",
    "window_geometry",
    "work_directory",
}
_SENSITIVE_SUFFIXES = (
    "_dir",
    "_directory",
    "_dsn",
    "_file",
    "_folder",
    "_path",
    "_secret",
    "_token",
)
_ENV_FLAG_KEYS = (
    "VSR_ALLOW_GPL",
    "VSR_ALLOW_REMOTE_CODE",
    "VSR_CRASH_REPORTS",
    "VSR_DISABLE_UPDATE_CHECK",
    "VSR_FORCE_CPU",
)


def default_appdata_root() -> Path:
    return (
        Path(os.environ.get("APPDATA", Path.home() / ".config"))
        / "VideoSubtitleRemoverPro"
    )


def default_settings_path() -> Path:
    return default_appdata_root() / "settings.json"


def default_log_path() -> Path:
    return default_appdata_root() / "vsr_pro.log"


def create_support_bundle(
    output_path: str | Path,
    *,
    settings_path: Optional[str | Path] = None,
    log_path: Optional[str | Path] = None,
    batch_report_paths: Optional[Iterable[str | Path]] = None,
    app_version: str = "",
    extra_facts: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write a zip file with redacted diagnostics and return its path."""
    output = Path(output_path)
    if output.suffix.lower() != ".zip":
        output = output.with_suffix(".zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    settings = Path(settings_path) if settings_path else default_settings_path()
    log = Path(log_path) if log_path else default_log_path()
    reports = [Path(p) for p in (batch_report_paths or [])]
    included: list[str] = []

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output.stem}.",
        suffix=".tmp",
        dir=str(output.parent),
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with zipfile.ZipFile(
            temp_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as bundle:
            support_payload = _support_payload(
                app_version=app_version,
                extra_facts=extra_facts or {},
            )
            _write_json(bundle, "support.json", support_payload, included)

            settings_payload = _read_json(settings)
            if settings_payload is not None:
                _write_json(
                    bundle,
                    "settings.redacted.json",
                    _redact_json(settings_payload),
                    included,
                )

            log_text = _read_tail_text(log, MAX_LOG_BYTES)
            if log_text is not None:
                _write_text(bundle, "vsr_pro.redacted.log", log_text, included)

            for index, report_path in enumerate(_unique_existing(reports), 1):
                _write_redacted_report(bundle, report_path, index, included)

        os.replace(temp_path, output)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
    return output


def _support_payload(*, app_version: str,
                     extra_facts: Mapping[str, Any]) -> dict:
    return {
        "schema": SUPPORT_BUNDLE_SCHEMA,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "app_version": app_version or "unknown",
        "runtime": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "frozen": bool(getattr(sys, "frozen", False)),
        },
        "env_flags": {
            key: bool(os.environ.get(key))
            for key in _ENV_FLAG_KEYS
        },
        "tools": {
            "ffmpeg": _tool_version("ffmpeg"),
            "ffprobe": _tool_version("ffprobe"),
        },
        "ffmpeg_profiles": collect_ffmpeg_capability_profiles(),
        "dependencies": _dependency_versions(),
        "dependency_diagnostics": {
            "opencv": collect_opencv_wheel_status(),
            "onnxruntime": collect_onnxruntime_provider_status(),
        },
        "backend_status": installed_backend_status(),
        "model_cache": model_cache_status(),
        "security": {
            "opencv_libpng": opencv_libpng_status(),
        },
        "caches": _cache_summary(),
        "facts": _redact_json(dict(extra_facts)),
    }


def _dependency_versions() -> dict:
    versions = {}
    for package in _DEPENDENCY_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _tool_version(executable: str) -> dict:
    if shutil.which(executable) is None:
        return {"available": False, "version": None}
    try:
        result = subprocess.run(
            [executable, "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"available": True, "version": None}
    first_line = (result.stdout or result.stderr).splitlines()[:1]
    return {
        "available": True,
        "version": redact_text(first_line[0]) if first_line else None,
    }


def _cache_summary() -> list[dict]:
    summary = []
    for entry in discover_caches():
        summary.append({
            "label": entry.label,
            "exists": entry.exists,
            "total_bytes": int(entry.total_bytes),
            "file_count": int(entry.file_count),
        })
    return summary


def _unique_existing(paths: Iterable[Path],
                     allowed_suffixes: tuple = (".json", ".md", ".txt",
                                                ".log"),
                     ) -> list[Path]:
    seen = set()
    result = []
    for path in paths:
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            continue
        if resolved.suffix.lower() not in allowed_suffixes:
            continue
        key = str(resolved).casefold()
        if key in seen or not resolved.is_file():
            continue
        seen.add(key)
        result.append(resolved)
    return result


def _write_redacted_report(bundle: zipfile.ZipFile, path: Path,
                           index: int, included: list[str]) -> None:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = _read_json(path)
        if payload is None:
            return
        _write_json(
            bundle,
            f"batch-report-{index}.redacted.json",
            _redact_json(payload),
            included,
        )
        return
    text = _read_tail_text(path, MAX_LOG_BYTES)
    if text is None:
        return
    ext = ".md" if suffix == ".md" else ".txt"
    _write_text(bundle, f"batch-report-{index}.redacted{ext}", text, included)


def _write_json(bundle: zipfile.ZipFile, arcname: str,
                payload: Any, included: list[str]) -> None:
    _write_text(
        bundle,
        arcname,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        included,
    )


def _write_text(bundle: zipfile.ZipFile, arcname: str,
                text: str, included: list[str]) -> None:
    bundle.writestr(arcname, text)
    included.append(arcname)


def _read_json(path: Path) -> Optional[Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _read_tail_text(path: Path, max_bytes: int) -> Optional[str]:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            start = max(0, size - max_bytes)
            handle.seek(start)
            raw = handle.read(max_bytes)
    except OSError:
        return None
    prefix = "[truncated to last %d bytes]\n" % max_bytes if size > max_bytes else ""
    return prefix + redact_text(raw.decode("utf-8", errors="replace"))


def redact_text(text: str) -> str:
    return _path_scrub(text)


def _is_sensitive_key(key: str) -> bool:
    lower = key.strip().lower()
    return lower in _SENSITIVE_KEYS or lower.endswith(_SENSITIVE_SUFFIXES)


def _redact_json(value: Any, key: str = "") -> Any:
    if key and _is_sensitive_key(key):
        return "<redacted>"
    if isinstance(value, dict):
        return {str(k): _redact_json(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_json(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def run_self_test() -> dict:
    """Probe OCR engines, inpaint backends, GPU providers, and codecs.

    Returns a dict of category -> list of {name, available, reason} entries.
    """
    results = {"ocr": [], "inpaint": [], "gpu": [], "codec": [],
               "dependency": [],
               "ffmpeg_profiles": []}

    def _probe(category, name, fn):
        try:
            ok, reason = fn()
            results[category].append({"name": name, "available": ok,
                                       "reason": reason})
        except Exception as exc:
            results[category].append({"name": name, "available": False,
                                       "reason": str(exc)[:200]})

    def _check_rapidocr():
        try:
            from rapidocr import RapidOCR
            r = RapidOCR()
            status = collect_rapidocr_engine_status()
            provider = status.get("preferredProvider") or "ONNX Runtime"
            return True, f"rapidocr loaded ({provider})"
        except ImportError:
            try:
                from rapidocr_onnxruntime import RapidOCR
                return True, "rapidocr_onnxruntime loaded"
            except ImportError:
                return False, "rapidocr not installed"

    def _check_paddleocr():
        try:
            import paddleocr
            return True, f"paddleocr {getattr(paddleocr, '__version__', '?')}"
        except ImportError:
            return False, "paddleocr not installed"

    def _check_easyocr():
        try:
            import easyocr
            return True, "easyocr available"
        except ImportError:
            return False, "easyocr not installed"

    def _check_lama_onnx():
        try:
            import onnxruntime
            return True, f"onnxruntime {onnxruntime.__version__}"
        except ImportError:
            return False, "onnxruntime not installed"

    def _check_lama_cv2dnn():
        try:
            import cv2
            major = int(cv2.__version__.split(".")[0])
            if major >= 5:
                return True, f"opencv {cv2.__version__} (DNN available)"
            return False, f"opencv {cv2.__version__} (DNN requires 5.0+)"
        except Exception:
            return False, "opencv not available"

    def _check_cuda():
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                return True, f"CUDA: {name}"
            return False, "CUDA not available"
        except ImportError:
            return False, "torch not installed"

    def _check_directml():
        try:
            import onnxruntime as ort
            if "DmlExecutionProvider" in ort.get_available_providers():
                return True, "DirectML available"
            return False, "DmlExecutionProvider not in providers"
        except ImportError:
            return False, "onnxruntime not installed"

    def _check_ffmpeg_encoder(codec_name, encoder_name):
        def _fn():
            import shutil as _sh
            import subprocess as _sp
            if _sh.which("ffmpeg") is None:
                return False, "ffmpeg not on PATH"
            try:
                result = _sp.run(
                    ["ffmpeg", "-hide_banner", "-encoders"],
                    capture_output=True, text=True, timeout=10)
                if encoder_name in result.stdout:
                    return True, f"{encoder_name} available"
                return False, f"{encoder_name} not found in ffmpeg"
            except Exception as e:
                return False, str(e)[:100]
        return _fn

    _probe("ocr", "RapidOCR", _check_rapidocr)
    _probe("ocr", "PaddleOCR", _check_paddleocr)
    _probe("ocr", "EasyOCR", _check_easyocr)
    _probe("inpaint", "LaMa-ONNX", _check_lama_onnx)
    _probe("inpaint", "LaMa-CV2DNN", _check_lama_cv2dnn)
    _probe("gpu", "CUDA", _check_cuda)
    _probe("gpu", "DirectML", _check_directml)
    _probe("codec", "H.264", _check_ffmpeg_encoder("h264", "libx264"))
    _probe("codec", "H.265", _check_ffmpeg_encoder("h265", "libx265"))
    _probe("codec", "AV1", _check_ffmpeg_encoder("av1", "libsvtav1"))
    _probe("codec", "VVC", _check_ffmpeg_encoder("vvc", "libvvenc"))
    _probe("codec", "NVENC", _check_ffmpeg_encoder("nvenc", "h264_nvenc"))
    opencv_status = collect_opencv_wheel_status()
    opencv_warnings = list(opencv_status.get("warnings", []) or [])
    imported = opencv_status.get("imported", {})
    owner = imported.get("owner") if isinstance(imported, Mapping) else ""
    version = imported.get("version") if isinstance(imported, Mapping) else ""
    file_path = imported.get("file") if isinstance(imported, Mapping) else ""
    reason = (
        str(opencv_warnings[0].get("message"))
        if opencv_warnings else
        f"cv2 {version or 'unknown'} from {owner or 'unknown'} at {file_path or 'unknown'}"
    )
    results["dependency"].append({
        "name": "OpenCV wheel ownership",
        "available": bool(imported.get("available")) and not opencv_warnings
        if isinstance(imported, Mapping) else False,
        "reason": reason,
    })
    try:
        results["ffmpeg_profiles"] = ffmpeg_profile_entries(
            collect_ffmpeg_capability_profiles()
        )
    except Exception as exc:
        results["ffmpeg_profiles"].append({
            "name": "profiles",
            "available": False,
            "reason": str(exc)[:200],
        })

    return results
