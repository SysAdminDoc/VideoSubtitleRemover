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

OPENCV_DNN_ENGINE_CONTRACT_SCHEMA = "vsr.opencv_dnn_engines.v1"
# RM-149: OpenCV 5 documents three DNN engine selections. Their operator
# coverage and fallback behaviour differ, so a release must prove the bundled
# PP-OCRv6 graph actually loads and produces the same answer under each one
# that the installed build exposes.
DNN_ENGINE_SELECTIONS: tuple[str, ...] = ("auto", "classic", "new")
_DNN_ENGINE_ATTRIBUTES: Mapping[str, str] = {
    "auto": "ENGINE_AUTO",
    "classic": "ENGINE_CLASSIC",
    "new": "ENGINE_NEW",
}
# Relative L2 distance between two engines' outputs. Above this the engines
# disagree materially and the release must fail rather than ship a silent
# accuracy change.
DNN_ENGINE_DIVERGENCE_TOLERANCE = 1e-3


def dnn_engine_value(name: str, cv_module=None):
    """Return the cv2.dnn constant for a documented engine selection."""
    attribute = _DNN_ENGINE_ATTRIBUTES.get(str(name).strip().lower())
    if attribute is None:
        return None
    try:
        cv = cv_module or importlib.import_module("cv2")
    except Exception:
        return None
    dnn = getattr(cv, "dnn", None)
    if dnn is None:
        return None
    return getattr(dnn, attribute, None)


def available_dnn_engines(cv_module=None) -> list[str]:
    """Documented engine selections the installed OpenCV actually exposes."""
    return [
        name for name in DNN_ENGINE_SELECTIONS
        if dnn_engine_value(name, cv_module) is not None
    ]


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

    def __init__(self, cfg, *, cv_module=None, engine: str = ""):
        model_path = cfg.get("model_path", None)
        if not model_path:
            raise ValueError("OpenCV DNN OCR requires an explicit model_path")
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"OCR model does not exist: {self.model_path}")
        self._cv = cv_module or importlib.import_module("cv2")
        self._metadata = read_onnx_metadata_props(self.model_path)
        self._net = self._cv.dnn.readNetFromONNX(str(self.model_path))
        # RM-149: record the engine that actually applied, not the request.
        self.engine = ""
        requested = str(engine or "").strip().lower()
        if requested:
            value = dnn_engine_value(requested, self._cv)
            setter = getattr(self._net, "setPreferableEngine", None)
            if value is not None and callable(setter):
                setter(value)
                self.engine = requested
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


def _relative_l2(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64).ravel()
    right = np.asarray(second, dtype=np.float64).ravel()
    if left.shape != right.shape:
        return float("inf")
    denominator = float(np.linalg.norm(left)) or 1.0
    return float(np.linalg.norm(left - right) / denominator)


def _deterministic_det_input(shape=(1, 3, 64, 64)) -> np.ndarray:
    grid = np.indices(shape[1:], dtype=np.float32)
    payload = ((grid[0] * 37 + grid[1] * 11 + grid[2] * 3) % 255) / 255.0
    return np.ascontiguousarray(payload[None, ...].astype(np.float32))


def _lama_contract_model() -> Path | None:
    """An advertised local LaMa model, when the user has pointed at one."""
    try:
        from backend.inpainters.lama import _find_opencv_lama_weight
    except Exception:
        return None
    try:
        found = _find_opencv_lama_weight()
    except Exception:
        return None
    return Path(found) if found else None


def _probe_engine(model_path: Path, engine: str, payload, cv) -> dict:
    record = {
        "engine": engine,
        "loaded": False,
        "ran": False,
        "outputShape": [],
        "error": "",
    }
    output = None
    try:
        net = cv.dnn.readNetFromONNX(str(model_path))
        value = dnn_engine_value(engine, cv)
        setter = getattr(net, "setPreferableEngine", None)
        if value is not None and callable(setter):
            setter(value)
        record["loaded"] = True
        net.setInput(np.ascontiguousarray(payload, dtype=np.float32))
        output = np.asarray(net.forward())
        record["ran"] = True
        record["outputShape"] = [int(dim) for dim in output.shape]
    except Exception as exc:
        record["error"] = str(exc)[:300]
    return record, output


def run_opencv_dnn_engine_contract(
    *,
    cv_module=None,
    include_lama: bool = True,
) -> dict:
    """Run one real inference per documented, available DNN engine selection.

    RM-149: PP-OCRv6 (and optionally an advertised local LaMa model) must load
    and agree across OpenCV 5's ``auto`` / ``classic`` / ``new`` engines. Load
    failures, shape changes, and materially divergent outputs all fail the
    contract. Engine selections the installed OpenCV does not expose are
    reported as not applicable rather than assumed to pass, and nothing here
    assumes an ORT-linked DNN build.
    """
    result: dict[str, Any] = {
        "schema": OPENCV_DNN_ENGINE_CONTRACT_SCHEMA,
        "documentedEngines": list(DNN_ENGINE_SELECTIONS),
        "availableEngines": [],
        "models": [],
        "ran": False,
        "passed": None,
        "errors": [],
    }
    try:
        cv = cv_module or importlib.import_module("cv2")
    except Exception as exc:
        result["errors"].append(f"OpenCV import failed: {exc}")
        return result
    result["opencvVersion"] = str(getattr(cv, "__version__", "") or "")
    engines = available_dnn_engines(cv)
    result["availableEngines"] = engines
    if not engines:
        result["errors"].append(
            "This OpenCV build exposes no documented DNN engine selection "
            f"(version {result['opencvVersion'] or 'unknown'}); OpenCV 5 is "
            "required for the engine contract."
        )
        return result

    targets: list[tuple[str, Path, Any]] = []
    status = collect_opencv_dnn_ocr_status(cv_module=cv)
    det = status.get("models", {}).get("det", {})
    if det.get("present"):
        targets.append(
            ("pp-ocrv6-det", Path(str(det["path"])), _deterministic_det_input()))
    else:
        result["errors"].append(
            "bundled PP-OCRv6 detection model is not available")
    if include_lama:
        lama = _lama_contract_model()
        if lama is not None and lama.is_file():
            targets.append(
                ("lama", lama, _deterministic_det_input((1, 4, 64, 64))))

    if not targets:
        return result

    result["ran"] = True
    passed = True
    for name, model_path, payload in targets:
        records = []
        baseline = None
        baseline_engine = ""
        model_ok = True
        for engine in engines:
            record, output = _probe_engine(model_path, engine, payload, cv)
            if not record["ran"]:
                model_ok = False
                result["errors"].append(
                    f"{name}: {engine} engine failed: "
                    f"{record['error'] or 'inference did not run'}")
            elif baseline is None:
                baseline, baseline_engine = output, engine
                record["divergence"] = 0.0
            else:
                divergence = _relative_l2(baseline, output)
                record["divergence"] = (
                    None if divergence == float("inf") else round(divergence, 9))
                if list(record["outputShape"]) != [
                    int(dim) for dim in baseline.shape
                ]:
                    model_ok = False
                    result["errors"].append(
                        f"{name}: {engine} output shape "
                        f"{record['outputShape']} differs from "
                        f"{baseline_engine} {list(baseline.shape)}")
                elif divergence > DNN_ENGINE_DIVERGENCE_TOLERANCE:
                    model_ok = False
                    result["errors"].append(
                        f"{name}: {engine} output diverges from "
                        f"{baseline_engine} by {divergence:.6g} "
                        f"(> {DNN_ENGINE_DIVERGENCE_TOLERANCE})")
            records.append(record)
        passed = passed and model_ok
        result["models"].append({
            "name": name,
            "path": model_path.name,
            "engines": records,
            "passed": model_ok,
        })
    result["passed"] = passed
    return result


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
