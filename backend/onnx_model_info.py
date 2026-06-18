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
import importlib.metadata
import logging
import mmap
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
    return {
        "package": {
            "name": resolved_package,
            "version": resolved_version,
        },
        "model_dir": str(rapid_dir) if rapid_dir else "",
        "model_count": len(models),
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
