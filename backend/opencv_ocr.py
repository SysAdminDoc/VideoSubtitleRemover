"""PP-OCRv6 inference through OpenCV 5 DNN.

RapidOCR supplies the reviewed preprocessing, postprocessing, and bundled
PP-OCRv6 assets. This adapter swaps only the ONNX inference session, allowing
CPU detection and recognition without ONNX Runtime when OpenCV 5 is installed.
The regular RapidOCR providers remain the fallback and the accelerated path.
"""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
from pathlib import Path
import threading
from typing import Any, Mapping

import numpy as np

from backend.onnx_model_info import read_onnx_metadata_props
from backend.security_checks import (
    libpng_fixed_version_str,
    opencv_libpng_status,
)


OPENCV_DNN_OCR_SCHEMA = "vsr.opencv_dnn_ocr.v1"
MINIMUM_OPENCV_VERSION = (5, 0, 0)
MODEL_FILENAMES: Mapping[str, str] = {
    "det": "PP-OCRv6_det_small.onnx",
    "cls": "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
    "rec": "PP-OCRv6_rec_small.onnx",
}
_CONSTRUCTION_LOCK = threading.Lock()


def _version_tuple(value: str) -> tuple[int, int, int]:
    parts = []
    for token in str(value).split("."):
        digits = "".join(char for char in token if char.isdigit())
        if digits:
            parts.append(int(digits))
        if len(parts) == 3:
            break
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)  # type: ignore[return-value]


def _rapidocr_root() -> Path | None:
    try:
        import rapidocr
    except ImportError:
        return None
    package_file = getattr(rapidocr, "__file__", None)
    return Path(package_file).resolve().parent if package_file else None


def _rapidocr_version() -> str | None:
    try:
        return metadata.version("rapidocr")
    except metadata.PackageNotFoundError:
        return None


def collect_opencv_dnn_ocr_status(
    *,
    rapidocr_root: str | Path | None = None,
    rapidocr_version: str | None = None,
    libpng: Mapping[str, Any] | None = None,
    cv_module=None,
) -> dict[str, Any]:
    """Return eligibility and asset evidence without loading the models."""
    import_error = ""
    try:
        cv = cv_module or importlib.import_module("cv2")
    except Exception as exc:
        cv = None
        import_error = str(exc)
    cv_version = str(getattr(cv, "__version__", "") or "")
    root = Path(rapidocr_root) if rapidocr_root is not None else _rapidocr_root()
    model_root = root / "models" if root is not None else None
    models = {
        role: {
            "filename": filename,
            "path": str(model_root / filename) if model_root else "",
            "present": bool(model_root and (model_root / filename).is_file()),
        }
        for role, filename in MODEL_FILENAMES.items()
    }
    png_status = (
        dict(libpng)
        if libpng is not None else opencv_libpng_status()
    )
    dnn_available = bool(
        cv is not None
        and hasattr(cv, "dnn")
        and hasattr(cv.dnn, "readNetFromONNX")
    )
    version_eligible = _version_tuple(cv_version) >= MINIMUM_OPENCV_VERSION
    assets_present = bool(models) and all(
        bool(item["present"]) for item in models.values()
    )
    libpng_fixed = (
        png_status.get("vulnerable") is False
        and bool(png_status.get("libpng_version"))
    )
    errors = []
    if import_error:
        errors.append(f"OpenCV import failed: {import_error}")
    if not version_eligible:
        errors.append("OpenCV 5.0+ is required")
    if not dnn_available:
        errors.append("cv2.dnn.readNetFromONNX is unavailable")
    if root is None:
        errors.append("RapidOCR is not installed")
    elif not assets_present:
        errors.append("RapidOCR PP-OCRv6 detection/recognition assets are missing")
    if not libpng_fixed:
        errors.append(
            "OpenCV build information does not prove libpng >= "
            f"{libpng_fixed_version_str()}"
        )
    return {
        "schema": OPENCV_DNN_OCR_SCHEMA,
        "eligible": not errors,
        "opencvVersion": cv_version or None,
        "minimumOpenCVVersion": ".".join(map(str, MINIMUM_OPENCV_VERSION)),
        "dnnAvailable": dnn_available,
        "rapidocrVersion": rapidocr_version or _rapidocr_version(),
        "models": models,
        "libpng": png_status,
        "fallback": "RapidOCR ONNX Runtime/OpenVINO provider",
        "errors": errors,
    }


class OpenCVDnnSession:
    """Small RapidOCR inference-session contract backed by ``cv2.dnn``."""

    def __init__(self, cfg, *, cv_module=None):
        model_path = cfg.get("model_path", None)
        if not model_path:
            raise ValueError("OpenCV DNN OCR requires an explicit model_path")
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"OCR model does not exist: {self.model_path}")
        self._cv = cv_module or importlib.import_module("cv2")
        self._metadata = read_onnx_metadata_props(self.model_path)
        self._net = self._cv.dnn.readNetFromONNX(str(self.model_path))
        self._lock = threading.Lock()

    def __call__(self, input_content: np.ndarray) -> np.ndarray:
        blob = np.ascontiguousarray(input_content, dtype=np.float32)
        with self._lock:
            self._net.setInput(blob)
            output = self._net.forward()
        return np.asarray(output)

    def have_key(self, key: str = "character") -> bool:
        return key in self._metadata

    def get_character_list(self, key: str = "character") -> list[str]:
        return self._metadata[key].splitlines()


def build_opencv_dnn_rapidocr():
    """Construct RapidOCR with OpenCV sessions, restoring globals immediately."""
    status = collect_opencv_dnn_ocr_status()
    if not status["eligible"]:
        raise RuntimeError("; ".join(status["errors"]))

    import rapidocr
    from rapidocr.ch_ppocr_cls import main as cls_main
    from rapidocr.ch_ppocr_det import main as det_main
    from rapidocr.ch_ppocr_rec import main as rec_main

    root = Path(rapidocr.__file__).resolve().parent
    model_root = root / "models"
    params = {
        "Global.use_cls": False,
        "Det.model_path": str(model_root / MODEL_FILENAMES["det"]),
        "Cls.model_path": str(model_root / MODEL_FILENAMES["cls"]),
        "Rec.model_path": str(model_root / MODEL_FILENAMES["rec"]),
    }
    modules = (det_main, cls_main, rec_main)
    with _CONSTRUCTION_LOCK:
        originals = tuple(module.get_engine for module in modules)
        try:
            for module in modules:
                module.get_engine = lambda _engine_type: OpenCVDnnSession
            engine = rapidocr.RapidOCR(params=params)
        finally:
            for module, original in zip(modules, originals):
                module.get_engine = original
    engine._vsr_engine_provider = "OpenCV 5 DNN"
    return engine
