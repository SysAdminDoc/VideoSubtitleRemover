"""Plugin registry for inpainter backends.

RFP-L-2: every inpainter (STTN / LAMA / ProPainter / AUTO) used to be
hardcoded into `SubtitleRemover._create_inpainter` via an if-elif chain.
Adding a new backend (LaMa-ONNX, MI-GAN, real ProPainter, DiffuEraser,
etc.) meant editing both `InpaintMode` and the dispatch.

The registry decouples the two: each backend module calls
`register(name, builder)` once at import time, and the dispatch in
`SubtitleRemover._create_inpainter` looks up the builder by enum name.
External backends (opt-in via `pip install vsr-myinpainter`) can plug
in by importing this module and calling `register` -- no monkey-patch
required.

Builder contract:
    builder(device: str, config: ProcessingConfig) -> BaseInpainter

The registry stays in process; backends are NOT auto-discovered from
disk (that would invite the plugin-marketplace security surface
RM-81's "deferred" gating already rejected). Anything that wants to
ship a backend imports vsr's backend package and calls register().

This module is import-cycle-free: it only depends on standard library
typing primitives. The processor module imports this module after its
own dataclass definitions, so a backend file can safely
`from backend.inpainter_registry import register` at module top.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)


# Builder signature is intentionally loose so the registry does not pull
# in the heavy ProcessingConfig + BaseInpainter types here. The factory
# is called with (device, config) and must return a BaseInpainter
# instance. The processor module enforces the contract at the call site.
_BuilderT = Callable[..., object]


_REGISTRY: Dict[str, _BuilderT] = {}
_ORDER: List[str] = []   # insertion order so listing returns a stable view


def register(name: str, builder: _BuilderT) -> None:
    """Register a builder for a named inpainter mode.

    `name` is matched case-insensitively against `InpaintMode.value`
    (which is the lowercase enum value: "sttn" / "lama" / "propainter"
    / "auto"). Re-registering an existing name replaces the previous
    builder, so opt-in backends can shadow a default if they ship a
    drop-in faster implementation (LaMa-ONNX overriding LAMA, for
    example).
    """
    key = name.strip().lower()
    if not key:
        raise ValueError("inpainter name must be a non-empty string")
    if key in _REGISTRY:
        logger.info(f"Replacing existing inpainter builder for {key!r}")
    else:
        _ORDER.append(key)
    _REGISTRY[key] = builder


def resolve(name: str) -> _BuilderT:
    """Return the builder for the named mode, or KeyError when no
    builder is registered. Caller catches KeyError and falls back to
    the default backend (STTN today)."""
    return _REGISTRY[name.strip().lower()]


def is_registered(name: str) -> bool:
    return name.strip().lower() in _REGISTRY


def list_modes() -> List[Tuple[str, _BuilderT]]:
    """Return [(name, builder), ...] in registration order. Mostly for
    introspection: GUI dropdowns / CLI --list-inpainters."""
    return [(name, _REGISTRY[name]) for name in _ORDER]


def unregister(name: str) -> bool:
    """Remove a previously-registered builder. Returns True when an
    entry was actually removed."""
    key = name.strip().lower()
    if key in _REGISTRY:
        del _REGISTRY[key]
        try:
            _ORDER.remove(key)
        except ValueError:
            pass
        return True
    return False


def clear() -> None:
    """Wipe the registry. Used by tests; production code should never
    call this."""
    _REGISTRY.clear()
    _ORDER.clear()
