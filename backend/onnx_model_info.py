"""Small ONNX metadata readers that avoid a hard dependency on `onnx`.

The app only needs the model-level `opset_import` values before creating
an ONNX Runtime session. Pulling in the full `onnx` package just to read
that metadata would make optional ONNX paths heavier, so this module uses
the protobuf wire format directly.

Model-opset audit (RM-119): PP-OCR ONNX models bundled by RapidOCR use
opset 11 (PaddleOCR v4 ONNX exports). LaMa-ONNX (Carve/LaMa-ONNX on
HuggingFace) uses opset 9. MI-GAN-ONNX uses opset 11. All are well
below the DirectML EP ceiling of opset 20. The guard in
``inpainters_onnx._providers_after_opset_audit`` handles future models
that exceed the ceiling by dropping DML and falling back to CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import contextlib
import importlib
import importlib.metadata
import logging
import mmap
import os
from pathlib import Path
import re
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.model_hashes import hash_file

logger = logging.getLogger(__name__)


DIRECTML_MAX_ONNX_OPSET = 20
DEFAULT_ONNX_DOMAINS = {"", "ai.onnx"}

KNOWN_MODEL_OPSETS: Dict[str, int] = {
    "PP-OCRv4 det (RapidOCR bundled)": 11,
    "PP-OCRv4 cls (RapidOCR bundled)": 11,
    "PP-OCRv4 rec (RapidOCR bundled)": 11,
    "LaMa-ONNX (Carve/LaMa-ONNX)": 9,
    "MI-GAN-ONNX (Picsart)": 11,
}

RAPIDOCR_CONFIG_FILES = ("config.yaml", "default_models.yaml")
WINDOWS_ML_PROBE_SCHEMA = "vsr.windows_ml_probe.v1"
WINDOWS_ML_BRIDGE_MODULE = "winui3.microsoft.windows.ai.machinelearning"
WINDOWS_ML_BOOTSTRAP_MODULE = (
    "winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap"
)


class OnnxModelInfoError(ValueError):
    """Raised when a file is not parseable enough to read ONNX metadata."""


@dataclass(frozen=True)
class OnnxOpsetImport:
    domain: str
    version: int


def read_onnx_opset_imports(path: str | Path) -> List[OnnxOpsetImport]:
    """Return model-level ONNX opset imports from `path`.

    This parses only the fields needed for `ModelProto.opset_import`, skipping
    tensor blobs and graph data without loading the full model into Python
    objects.
    """
    p = Path(path)
    if not p.is_file():
        raise OnnxModelInfoError(f"ONNX model does not exist: {p}")
    if p.stat().st_size == 0:
        raise OnnxModelInfoError(f"ONNX model is empty: {p}")
    with p.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            return _parse_model_opsets(mm)


def directml_incompatible_opsets(
    path: str | Path,
    max_opset: int = DIRECTML_MAX_ONNX_OPSET,
) -> List[OnnxOpsetImport]:
    """Return default-domain opsets newer than DirectML can execute."""
    return [
        item for item in read_onnx_opset_imports(path)
        if item.domain in DEFAULT_ONNX_DOMAINS and item.version > max_opset
    ]


def _parse_model_opsets(buf: Sequence[int]) -> List[OnnxOpsetImport]:
    pos = 0
    found: List[OnnxOpsetImport] = []
    size = len(buf)
    while pos < size:
        key, pos = _read_varint(buf, pos)
        field_number = key >> 3
        wire_type = key & 0x07
        if field_number == 8 and wire_type == 2:
            payload, pos = _read_length_delimited(buf, pos)
            item = _parse_opset_import(payload)
            if item is not None:
                found.append(item)
        else:
            pos = _skip_field(buf, pos, wire_type)
    if not found:
        raise OnnxModelInfoError("ONNX model has no opset_import entries")
    return found


def _parse_opset_import(buf: Sequence[int]) -> OnnxOpsetImport | None:
    pos = 0
    domain = ""
    version = None
    size = len(buf)
    while pos < size:
        key, pos = _read_varint(buf, pos)
        field_number = key >> 3
        wire_type = key & 0x07
        if field_number == 1 and wire_type == 2:
            raw, pos = _read_length_delimited(buf, pos)
            domain = bytes(raw).decode("utf-8", errors="replace")
        elif field_number == 2 and wire_type == 0:
            version, pos = _read_varint(buf, pos)
        else:
            pos = _skip_field(buf, pos, wire_type)
    if version is None:
        return None
    return OnnxOpsetImport(domain=domain, version=int(version))


def _read_varint(buf: Sequence[int], pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    size = len(buf)
    while pos < size:
        byte = int(buf[pos])
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise OnnxModelInfoError("protobuf varint is too long")
    raise OnnxModelInfoError("truncated protobuf varint")


def _read_length_delimited(
    buf: Sequence[int],
    pos: int,
) -> tuple[Sequence[int], int]:
    length, pos = _read_varint(buf, pos)
    end = pos + length
    if end > len(buf):
        raise OnnxModelInfoError("truncated length-delimited protobuf field")
    return buf[pos:end], end


def _skip_field(buf: Sequence[int], pos: int, wire_type: int) -> int:
    if wire_type == 0:
        _, pos = _read_varint(buf, pos)
        return pos
    if wire_type == 1:
        return _skip_fixed(buf, pos, 8)
    if wire_type == 2:
        length, pos = _read_varint(buf, pos)
        return _skip_fixed(buf, pos, length)
    if wire_type == 5:
        return _skip_fixed(buf, pos, 4)
    raise OnnxModelInfoError(f"unsupported protobuf wire type: {wire_type}")


def _skip_fixed(buf: Sequence[int], pos: int, length: int) -> int:
    end = pos + length
    if end > len(buf):
        raise OnnxModelInfoError("truncated protobuf field")
    return end


def _rapidocr_package_root() -> Optional[Tuple[str, Path]]:
    """Return (import name, package root) for an installed RapidOCR package."""
    try:
        import rapidocr
        return "rapidocr", Path(rapidocr.__file__).parent
    except ImportError:
        try:
            import rapidocr_onnxruntime
            return "rapidocr_onnxruntime", Path(rapidocr_onnxruntime.__file__).parent
        except ImportError:
            return None


def _rapidocr_model_dir() -> Optional[Path]:
    """Locate the RapidOCR bundled model directory if installed."""
    package = _rapidocr_package_root()
    if package is None:
        return None
    _import_name, pkg_dir = package
    models = pkg_dir / "models"
    return models if models.is_dir() else None


def _rapidocr_config_root(
    package_root: Optional[Tuple[str, Path]],
    rapid_dir: Optional[Path],
) -> Optional[Path]:
    if package_root is not None:
        return package_root[1]
    if rapid_dir is not None and rapid_dir.name.lower() == "models":
        return rapid_dir.parent
    return None


def _rapidocr_package_version(import_name: str) -> Optional[str]:
    candidates = (
        ("rapidocr", "rapidocr"),
        ("rapidocr_onnxruntime", "rapidocr-onnxruntime"),
    )
    for candidate_import, dist_name in candidates:
        if import_name != candidate_import:
            continue
        try:
            return importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            return None
    return None


def _major_version(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    match = re.search(r"\d+", value)
    return int(match.group(0)) if match else None


def _file_record(root: Path, name: str) -> Dict[str, object]:
    path = root / name
    record: Dict[str, object] = {
        "name": name,
        "path": str(path),
        "exists": path.is_file(),
        "bytes": None,
        "sha256": None,
    }
    if path.is_file():
        try:
            record.update({
                "bytes": path.stat().st_size,
                "sha256": hash_file(path),
            })
        except OSError as exc:
            record["error"] = str(exc)
    return record


def _rapidocr_required_assets(
    models: Sequence[Dict[str, object]],
    config_files: Sequence[Dict[str, object]],
    package_version: Optional[str],
) -> Tuple[List[Dict[str, object]], bool, List[str]]:
    model_paths = [
        str(item.get("relative_path") or item.get("filename") or "").lower()
        for item in models
    ]
    families = sorted({
        match.group(1)
        for path in model_paths
        for match in [re.search(r"(pp-ocrv\d+)", path, re.IGNORECASE)]
        if match
    })
    major = _major_version(package_version)
    requires_ppocrv6 = major is not None and major >= 3

    def has_model(version: str, role: str) -> bool:
        version = version.lower()
        role_token = f"_{role}"
        return any(version in path and role_token in path for path in model_paths)

    required = [
        {
            "name": "RapidOCR config.yaml",
            "required": True,
            "present": any(
                item.get("name") == "config.yaml" and item.get("exists")
                for item in config_files
            ),
        },
        {
            "name": "RapidOCR default_models.yaml",
            "required": True,
            "present": any(
                item.get("name") == "default_models.yaml" and item.get("exists")
                for item in config_files
            ),
        },
        {
            "name": "PP-OCRv6 detection ONNX",
            "required": requires_ppocrv6,
            "present": has_model("pp-ocrv6", "det"),
        },
        {
            "name": "PP-OCRv6 recognition ONNX",
            "required": requires_ppocrv6,
            "present": has_model("pp-ocrv6", "rec"),
        },
    ]
    compatible = bool(models) and all(
        (not item["required"]) or bool(item["present"])
        for item in required
    )
    return required, compatible, families


def rapidocr_release_provenance(
    model_dir: Optional[str | Path] = None,
    *,
    package_name: Optional[str] = None,
    package_version: Optional[str] = None,
) -> Dict[str, object]:
    """Return release evidence for bundled RapidOCR ONNX assets."""
    package_root = None if package_name else _rapidocr_package_root()
    resolved_package = package_name or (package_root[0] if package_root else "")
    resolved_version = (
        package_version
        if package_version is not None
        else (_rapidocr_package_version(resolved_package) if resolved_package else None)
    )
    if model_dir is not None:
        rapid_dir = Path(model_dir)
    else:
        rapid_dir = _rapidocr_model_dir()
    models: List[Dict[str, object]] = []
    if rapid_dir is not None and rapid_dir.is_dir():
        for onnx_file in sorted(rapid_dir.rglob("*.onnx")):
            record = _audit_one(
                f"RapidOCR/{onnx_file.relative_to(rapid_dir)}",
                onnx_file,
            )
            try:
                rel = onnx_file.relative_to(rapid_dir)
            except ValueError:
                rel = onnx_file.name
            try:
                record.update({
                    "filename": onnx_file.name,
                    "relative_path": str(rel).replace(os.sep, "/"),
                    "bytes": onnx_file.stat().st_size,
                    "sha256": hash_file(onnx_file),
                })
            except OSError as exc:
                record.update({
                    "filename": onnx_file.name,
                    "relative_path": str(rel).replace(os.sep, "/"),
                    "bytes": None,
                    "sha256": None,
                    "error": str(exc),
                })
            models.append(record)
    config_root = _rapidocr_config_root(package_root, rapid_dir)
    config_files = (
        [_file_record(config_root, name) for name in RAPIDOCR_CONFIG_FILES]
        if config_root is not None else []
    )
    required_assets, packaging_compatible, model_families = (
        _rapidocr_required_assets(models, config_files, resolved_version)
    )
    return {
        "package": {
            "name": resolved_package,
            "version": resolved_version,
        },
        "model_dir": str(rapid_dir) if rapid_dir else "",
        "config_files": config_files,
        "model_count": len(models),
        "model_families": model_families,
        "required_assets": required_assets,
        "packaging_compatible": packaging_compatible,
        "models": models,
        "missing": not models,
    }


def _adapter_onnx_paths() -> List[Tuple[str, Path]]:
    """Collect configured adapter ONNX paths from environment."""
    pairs: List[Tuple[str, Path]] = []
    for env_var, label in [
        ("VSR_LAMA_ONNX", "LaMa-ONNX"),
        ("VSR_MIGAN_ONNX", "MI-GAN-ONNX"),
    ]:
        value = os.environ.get(env_var, "").strip()
        if value:
            p = Path(value)
            if p.is_file() and p.suffix.lower() == ".onnx":
                pairs.append((label, p))
    return pairs


def audit_onnx_models() -> List[Dict[str, object]]:
    """Audit all discoverable ONNX models for DirectML opset compatibility.

    Returns a list of audit records, one per model file, each containing:
    - ``source``: human label for the model
    - ``path``: filesystem path
    - ``opsets``: list of ``{domain, version}``
    - ``max_default_opset``: highest version among default ONNX domains
    - ``directml_compatible``: True when max opset <= DIRECTML_MAX_ONNX_OPSET
    - ``error``: error string if the model could not be parsed
    """
    records: List[Dict[str, object]] = []

    rapid_dir = _rapidocr_model_dir()
    if rapid_dir is not None:
        for onnx_file in sorted(rapid_dir.rglob("*.onnx")):
            records.append(_audit_one(
                f"RapidOCR/{onnx_file.relative_to(rapid_dir)}",
                onnx_file,
            ))

    for label, path in _adapter_onnx_paths():
        records.append(_audit_one(label, path))

    return records


def _audit_one(source: str, path: Path) -> Dict[str, object]:
    try:
        opsets = read_onnx_opset_imports(path)
    except (OnnxModelInfoError, OSError) as exc:
        return {
            "source": source,
            "path": str(path),
            "opsets": [],
            "max_default_opset": None,
            "directml_compatible": None,
            "error": str(exc),
        }
    max_default = 0
    for op in opsets:
        if op.domain in DEFAULT_ONNX_DOMAINS:
            max_default = max(max_default, op.version)
    return {
        "source": source,
        "path": str(path),
        "opsets": [{"domain": op.domain, "version": op.version} for op in opsets],
        "max_default_opset": max_default,
        "directml_compatible": max_default <= DIRECTML_MAX_ONNX_OPSET,
        "error": None,
    }


def print_audit_report(records: Optional[List[Dict[str, object]]] = None) -> None:
    """Print a human-readable DirectML opset audit to stdout."""
    if records is None:
        records = audit_onnx_models()
    print(f"DirectML ONNX opset audit (ceiling: opset {DIRECTML_MAX_ONNX_OPSET})")
    print(f"Known model opsets (from source repos):")
    for name, opset in sorted(KNOWN_MODEL_OPSETS.items()):
        compat = "OK" if opset <= DIRECTML_MAX_ONNX_OPSET else "EXCEEDS"
        print(f"  {name}: opset {opset} [{compat}]")
    print()
    if not records:
        print("No local ONNX model files found to audit.")
        print("Install rapidocr or set VSR_LAMA_ONNX / VSR_MIGAN_ONNX to audit local files.")
        return
    print(f"Local model audit ({len(records)} files):")
    for rec in records:
        if rec.get("error"):
            print(f"  {rec['source']}: ERROR - {rec['error']}")
            continue
        compat = "OK" if rec["directml_compatible"] else "EXCEEDS CEILING"
        print(f"  {rec['source']}: opset {rec['max_default_opset']} [{compat}]")


def _pb_varint(value: int) -> bytes:
    out = bytearray()
    value = int(value)
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _pb_field_varint(field: int, value: int) -> bytes:
    return _pb_varint((field << 3) | 0) + _pb_varint(value)


def _pb_field_bytes(field: int, payload: bytes) -> bytes:
    return _pb_varint((field << 3) | 2) + _pb_varint(len(payload)) + payload


def _tiny_identity_onnx_bytes() -> bytes:
    """Return a minimal ONNX Identity model: float[1] -> float[1]."""
    dim = _pb_field_varint(1, 1)
    shape = _pb_field_bytes(1, dim)
    tensor_type = _pb_field_varint(1, 1) + _pb_field_bytes(2, shape)
    type_proto = _pb_field_bytes(1, tensor_type)

    input_info = (
        _pb_field_bytes(1, b"x")
        + _pb_field_bytes(2, type_proto)
    )
    output_info = (
        _pb_field_bytes(1, b"y")
        + _pb_field_bytes(2, type_proto)
    )
    node = (
        _pb_field_bytes(1, b"x")
        + _pb_field_bytes(2, b"y")
        + _pb_field_bytes(3, b"identity")
        + _pb_field_bytes(4, b"Identity")
    )
    graph = (
        _pb_field_bytes(1, node)
        + _pb_field_bytes(2, b"vsr_windows_ml_smoke")
        + _pb_field_bytes(11, input_info)
        + _pb_field_bytes(12, output_info)
    )
    opset = _pb_field_varint(2, 13)
    return (
        _pb_field_varint(1, 7)
        + _pb_field_bytes(2, b"VideoSubtitleRemover")
        + _pb_field_bytes(7, graph)
        + _pb_field_bytes(8, opset)
    )


def _module_imported(
    module_name: str,
    importer=importlib.import_module,
):
    try:
        return importer(module_name), None
    except Exception as exc:
        return None, str(exc)


def _distribution_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _complete_async_operation(operation) -> None:
    if operation is None:
        return
    if hasattr(operation, "get"):
        operation.get()
        return
    if hasattr(operation, "result"):
        operation.result()
        return
    try:
        import inspect
        import asyncio
        if inspect.isawaitable(operation):
            asyncio.run(operation)
    except RuntimeError:
        logger.debug("Windows ML async registration skipped inside running loop")


def _ep_devices(ort_module) -> list[dict]:
    get_devices = getattr(ort_module, "get_ep_devices", None)
    if not callable(get_devices):
        return []
    devices = []
    try:
        for item in get_devices():
            hardware = getattr(item, "hardware_device", None)
            devices.append({
                "epName": str(getattr(item, "ep_name", "")),
                "deviceType": str(getattr(hardware, "type", "")),
            })
    except Exception as exc:
        logger.debug(f"Windows ML EP device probe failed: {exc}")
    return devices


@contextlib.contextmanager
def _windows_app_sdk_context(bootstrap_module):
    initialize = getattr(bootstrap_module, "initialize", None)
    options_type = getattr(bootstrap_module, "InitializeOptions", None)
    option = getattr(options_type, "ON_NO_MATCH_SHOW_UI", None)
    if not callable(initialize):
        yield False
        return
    context = initialize(options=option) if option is not None else initialize()
    if hasattr(context, "__enter__") and hasattr(context, "__exit__"):
        with context:
            yield True
    else:
        yield True


def _register_windows_ml_eps(ml_module) -> bool:
    catalog_type = getattr(ml_module, "ExecutionProviderCatalog", None)
    get_default = getattr(catalog_type, "GetDefault", None)
    if not callable(get_default):
        return False
    catalog = get_default()
    register = getattr(catalog, "RegisterCertifiedAsync", None)
    if not callable(register):
        return False
    _complete_async_operation(register())
    return True


def _run_windows_ml_smoke(ort_module,
                          providers: Sequence[str],
                          temp_dir: Optional[str | Path] = None) -> dict:
    with tempfile.TemporaryDirectory(prefix="vsr_winml_",
                                     dir=str(temp_dir) if temp_dir else None) as tmpdir:
        model_path = Path(tmpdir) / "identity.onnx"
        model_path.write_bytes(_tiny_identity_onnx_bytes())
        selected = list(providers or [])
        if not selected:
            selected = ["CPUExecutionProvider"]
        session = ort_module.InferenceSession(str(model_path), providers=selected)
        actual = (
            list(session.get_providers())
            if hasattr(session, "get_providers") else selected
        )
        payload = np.array([3.0], dtype=np.float32)
        output = session.run(None, {"x": payload})[0]
        ok = bool(np.allclose(output, payload))
        return {
            "passed": ok,
            "requestedProviders": selected,
            "activeProviders": actual,
        }


def collect_windows_ml_probe(
    *,
    run_smoke: bool = True,
    importer=importlib.import_module,
    ort_module=None,
    platform_name: Optional[str] = None,
    temp_dir: Optional[str | Path] = None,
) -> dict:
    """Probe whether Windows ML is usable from Python without migrating.

    The probe is guarded at every boundary: non-Windows hosts return
    ``not_applicable``; missing pywinrt/Windows App SDK packages return a
    blocked decision; smoke inference runs only after the bridge imports.
    """
    platform_name = platform_name or ("Windows" if os.name == "nt" else os.name)
    result: Dict[str, object] = {
        "schema": WINDOWS_ML_PROBE_SCHEMA,
        "platform": platform_name,
        "pythonBridgeModule": WINDOWS_ML_BRIDGE_MODULE,
        "bootstrapModule": WINDOWS_ML_BOOTSTRAP_MODULE,
        "pythonBridgeInstalled": False,
        "bootstrapInstalled": False,
        "onnxruntimeInstalled": False,
        "onnxruntimeWindowsMlPackage": _distribution_version(
            "onnxruntime-windowsml"),
        "winui3MlPackage": (
            _distribution_version("winui3-Microsoft.Windows.AI.MachineLearning")
            or _distribution_version("wasdk-Microsoft.Windows.AI.MachineLearning")
        ),
        "availableProviders": [],
        "epDevicesBeforeRegister": [],
        "epDevicesAfterRegister": [],
        "registeredCertifiedProviders": False,
        "smoke": {"attempted": False, "passed": False},
        "decision": "unknown",
        "reason": "",
        "errors": [],
    }
    if str(platform_name).lower() not in {"windows", "nt"}:
        result.update({
            "decision": "not_applicable",
            "reason": "Windows ML is only available on Windows.",
        })
        return result

    bootstrap, bootstrap_error = _module_imported(
        WINDOWS_ML_BOOTSTRAP_MODULE, importer)
    if bootstrap is None:
        result["errors"].append(f"bootstrap import failed: {bootstrap_error}")
    else:
        result["bootstrapInstalled"] = True

    ml_module, ml_error = _module_imported(WINDOWS_ML_BRIDGE_MODULE, importer)
    if ml_module is None:
        result["errors"].append(f"Windows ML bridge import failed: {ml_error}")
        result.update({
            "decision": "blocked",
            "reason": (
                "Python Windows ML bridge is not importable; install the "
                "pywinrt Windows App SDK ML package before migration."
            ),
        })
        return result
    result["pythonBridgeInstalled"] = True

    if ort_module is None:
        ort_module, ort_error = _module_imported("onnxruntime", importer)
        if ort_module is None:
            result["errors"].append(f"onnxruntime import failed: {ort_error}")
            result.update({
                "decision": "blocked",
                "reason": "onnxruntime is not importable from this Python.",
            })
            return result
    result["onnxruntimeInstalled"] = True

    get_providers = getattr(ort_module, "get_available_providers", None)
    providers = list(get_providers() if callable(get_providers) else [])
    result["availableProviders"] = providers
    result["epDevicesBeforeRegister"] = _ep_devices(ort_module)

    try:
        context = (
            _windows_app_sdk_context(bootstrap)
            if bootstrap is not None else contextlib.nullcontext(False)
        )
        with context:
            result["registeredCertifiedProviders"] = _register_windows_ml_eps(
                ml_module)
            result["epDevicesAfterRegister"] = _ep_devices(ort_module)
    except Exception as exc:
        result["errors"].append(f"Windows ML provider registration failed: {exc}")

    if run_smoke:
        result["smoke"] = {"attempted": True, "passed": False}
        try:
            smoke = _run_windows_ml_smoke(ort_module, providers, temp_dir)
            result["smoke"] = {"attempted": True, **smoke}
        except Exception as exc:
            result["errors"].append(f"Windows ML smoke inference failed: {exc}")

    if result.get("smoke", {}).get("passed"):
        result.update({
            "decision": "candidate",
            "reason": (
                "Windows ML Python bridge imported and ONNX Runtime executed "
                "a tiny identity model; migration still needs real-model "
                "benchmarking."
            ),
        })
    else:
        result.update({
            "decision": "blocked",
            "reason": (
                "Windows ML Python path is not yet proven by the smoke model."
            ),
        })
    return result


def print_windows_ml_probe_report(status: Optional[dict] = None) -> None:
    if status is None:
        status = collect_windows_ml_probe()
    print("Windows ML Python probe")
    print(f"  Decision: {status.get('decision')} - {status.get('reason')}")
    print(f"  Bridge: {status.get('pythonBridgeInstalled')} ({status.get('winui3MlPackage')})")
    print(f"  Bootstrap: {status.get('bootstrapInstalled')}")
    print(f"  ONNX Runtime: {status.get('onnxruntimeInstalled')} "
          f"(onnxruntime-windowsml={status.get('onnxruntimeWindowsMlPackage')})")
    print(f"  Providers: {', '.join(status.get('availableProviders') or []) or 'none'}")
    before = status.get("epDevicesBeforeRegister") or []
    after = status.get("epDevicesAfterRegister") or []
    if before:
        print("  EP devices before registration:")
        for item in before:
            print(f"    - {item.get('epName')} ({item.get('deviceType')})")
    if after:
        print("  EP devices after registration:")
        for item in after:
            print(f"    - {item.get('epName')} ({item.get('deviceType')})")
    smoke = status.get("smoke") or {}
    print(f"  Smoke: attempted={smoke.get('attempted')} passed={smoke.get('passed')}")
    if smoke.get("activeProviders"):
        print(f"  Smoke providers: {', '.join(smoke.get('activeProviders') or [])}")
    for error in status.get("errors") or []:
        print(f"  Warning: {error}")
