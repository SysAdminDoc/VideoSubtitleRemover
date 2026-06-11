"""Small ONNX metadata readers that avoid a hard dependency on `onnx`.

The app only needs the model-level `opset_import` values before creating
an ONNX Runtime session. Pulling in the full `onnx` package just to read
that metadata would make optional ONNX paths heavier, so this module uses
the protobuf wire format directly.
"""

from __future__ import annotations

from dataclasses import dataclass
import mmap
from pathlib import Path
from typing import List, Sequence


DIRECTML_MAX_ONNX_OPSET = 20
DEFAULT_ONNX_DOMAINS = {"", "ai.onnx"}


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
