"""TensorRT engine cache helper.

RM-70: when the user runs LaMa-ONNX on NVIDIA hardware, the ONNX path
can be JIT-compiled to a TensorRT engine on first run, then cached in
`%APPDATA%\\VideoSubtitleRemoverPro\\trt_cache` so subsequent runs see
~2-3x speedup. We DO NOT compile here unconditionally -- compilation
takes minutes -- so the helper exposes the cache lookup + compile call
behind `VSR_TENSORRT=1`.

The actual conversion uses `polygraphy` from the TensorRT package
when available; falls back to a no-op when missing.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def is_tensorrt_enabled() -> bool:
    return os.environ.get("VSR_TENSORRT", "").strip().lower() in {"1", "true", "yes", "on"}


def cache_dir() -> Path:
    base = Path(os.environ.get("APPDATA", Path.home() / ".config"))
    out = base / "VideoSubtitleRemoverPro" / "trt_cache"
    out.mkdir(parents=True, exist_ok=True)
    return out


def cached_engine_path(onnx_path: str, suffix: str = "fp16") -> Path:
    """Return the .engine path that would carry a compiled artefact
    for `onnx_path`. The filename is hash-stable so a re-run of the
    same ONNX hits cache."""
    digest = hashlib.sha256(Path(onnx_path).read_bytes()).hexdigest()[:16]
    return cache_dir() / f"{Path(onnx_path).stem}-{digest}-{suffix}.engine"


def maybe_compile_engine(onnx_path: str, precision: str = "fp16") -> Optional[Path]:
    """Compile or fetch the cached engine for `onnx_path`. Returns the
    engine path on success, None when TensorRT is unavailable.

    Uses polygraphy's `convert run` CLI when present (the most
    portable entry point); falls back silently to None otherwise.
    """
    if not is_tensorrt_enabled():
        return None
    cached = cached_engine_path(onnx_path, suffix=precision)
    if cached.is_file():
        logger.info(f"TensorRT engine cache hit: {cached}")
        return cached
    import shutil
    if shutil.which("polygraphy") is None:
        logger.info(
            "polygraphy not on PATH; cannot compile TensorRT engine. "
            "Install via `pip install polygraphy nvidia-tensorrt`."
        )
        return None
    import subprocess
    cmd = [
        "polygraphy", "convert", onnx_path,
        "--convert-to", "trt",
        "-o", str(cached),
    ]
    if precision == "fp16":
        cmd += ["--fp16"]
    elif precision == "int8":
        cmd += ["--int8"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0 and cached.is_file():
            logger.info(f"TensorRT engine compiled: {cached}")
            return cached
        logger.warning(
            f"polygraphy exit {result.returncode}: {(result.stderr or '')[:400]}"
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning(f"TensorRT compile failed: {exc}")
    return None
