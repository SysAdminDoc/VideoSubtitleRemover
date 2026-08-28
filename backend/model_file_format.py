"""RM-336: identify a model file by its bytes, not by its name.

huggingface_hub 1.29.0 fixed a shard named exactly `.safetensors` being
routed to `torch.load(weights_only=False)`, because `Path.suffix` returns an
empty string for a name that is nothing but an extension. The lesson
generalises past that one client: a filename is attacker-influenced metadata
and a format decision made from it is a decision made from untrusted input.

Every function here reads the first bytes of the file and reports what the
file actually is. A format that cannot be identified is reported as unknown,
never guessed from the name.
"""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_FORMAT_SCHEMA = "vsr.model_file_format.v1"

# safetensors: 8-byte little-endian header length, then that many bytes of
# JSON. A header longer than this is not a real safetensors file.
SAFETENSORS_MAX_HEADER_BYTES = 100 * 1024 * 1024

FORMAT_SAFETENSORS = "safetensors"
FORMAT_TORCH_ZIP = "torch-zip"
FORMAT_TORCH_LEGACY_PICKLE = "torch-legacy-pickle"
FORMAT_ONNX = "onnx"
FORMAT_GGUF = "gguf"
FORMAT_UNKNOWN = "unknown"

# Formats that are safe to load without executing anything the file chose.
NON_EXECUTABLE_FORMATS = frozenset({FORMAT_SAFETENSORS, FORMAT_ONNX,
                                    FORMAT_GGUF})


def _read_head(path: Path, count: int) -> bytes:
    try:
        with open(path, "rb") as handle:
            return handle.read(count)
    except OSError as exc:
        logger.debug("Could not read %s: %s", path, exc)
        return b""


def _is_safetensors(path: Path, head: bytes) -> bool:
    if len(head) < 8:
        return False
    (length,) = struct.unpack("<Q", head[:8])
    if length == 0 or length > SAFETENSORS_MAX_HEADER_BYTES:
        return False
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if 8 + length > size:
        return False
    try:
        with open(path, "rb") as handle:
            handle.seek(8)
            header = handle.read(min(length, SAFETENSORS_MAX_HEADER_BYTES))
    except OSError:
        return False
    if not header.lstrip().startswith(b"{"):
        return False
    try:
        parsed = json.loads(header.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return False
    return isinstance(parsed, dict)


def identify_model_file(path: str | Path) -> str:
    """Return what a file actually is, from its bytes.

    Never consults the filename. A file that cannot be identified is
    ``unknown``, which callers must treat as "do not load", not as "probably
    fine".
    """
    target = Path(path)
    if not target.is_file():
        return FORMAT_UNKNOWN
    head = _read_head(target, 16)
    if not head:
        return FORMAT_UNKNOWN
    # ONNX is a protobuf whose first field is ir_version (field 1, varint).
    if head[:1] == b"\x08":
        return FORMAT_ONNX
    if head[:4] == b"GGUF":
        return FORMAT_GGUF
    # torch.save since 1.6 writes a zip archive.
    if head[:2] == b"PK":
        return FORMAT_TORCH_ZIP
    # Legacy torch.save is a raw pickle: protocol 2 opcode, or the magic
    # long torch wrote ahead of it.
    if head[:2] in (b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05"):
        return FORMAT_TORCH_LEGACY_PICKLE
    if _is_safetensors(target, head):
        return FORMAT_SAFETENSORS
    return FORMAT_UNKNOWN


def describe_model_file(path: str | Path) -> dict:
    """Identify one file and say whether loading it can execute code."""
    target = Path(path)
    fmt = identify_model_file(target)
    return {
        "schema": MODEL_FORMAT_SCHEMA,
        "path": str(target),
        "format": fmt,
        # The extension is recorded as an observation, never as the basis
        # for the decision above.
        "suffix": target.suffix,
        "suffixMatchesBytes": _suffix_matches(target.suffix, fmt),
        "executesOnLoad": fmt in {FORMAT_TORCH_ZIP,
                                  FORMAT_TORCH_LEGACY_PICKLE},
        "identified": fmt != FORMAT_UNKNOWN,
    }


_SUFFIX_FORMATS = {
    ".safetensors": {FORMAT_SAFETENSORS},
    ".onnx": {FORMAT_ONNX},
    ".gguf": {FORMAT_GGUF},
    ".pt": {FORMAT_TORCH_ZIP, FORMAT_TORCH_LEGACY_PICKLE},
    ".pth": {FORMAT_TORCH_ZIP, FORMAT_TORCH_LEGACY_PICKLE},
    ".bin": {FORMAT_TORCH_ZIP, FORMAT_TORCH_LEGACY_PICKLE},
    ".ckpt": {FORMAT_TORCH_ZIP, FORMAT_TORCH_LEGACY_PICKLE},
}


def _suffix_matches(suffix: str, fmt: str) -> Optional[bool]:
    expected = _SUFFIX_FORMATS.get(str(suffix).lower())
    if expected is None:
        return None
    return fmt in expected
