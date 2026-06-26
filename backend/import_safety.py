"""Safe probes for optional native Python dependencies.

Some optional engines depend on native wheels that can abort or access-violate
during import on Windows. Probe those packages in a child interpreter before
importing them in the long-running GUI/CLI process.
"""

from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys
from typing import Dict, Optional

_LOGGER = logging.getLogger(__name__)
_IMPORT_PROBE_CACHE: Dict[str, bool] = {}
_FATAL_IMPORT_MARKERS = (
    "fatal exception",
    "fatal python error",
    "access violation",
    "traceback",
    "error loading",
    "dll initialization",
    "0xc0000005",
)


def clear_import_probe_cache() -> None:
    """Clear cached probe results for tests and diagnostics."""
    _IMPORT_PROBE_CACHE.clear()


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS"))


def module_can_import(
    module_name: str,
    *,
    timeout: float = 20.0,
    logger: Optional[logging.Logger] = None,
    failure_context: str = "optional dependency disabled",
) -> bool:
    """Return True when importing an optional module is likely safe.

    Source runs use a subprocess probe so broken native wheels cannot take down
    the main process. Frozen builds skip the probe because ``sys.executable`` is
    the packaged app executable rather than a Python interpreter.
    """
    log = logger or _LOGGER
    if module_name in sys.modules and sys.modules[module_name] is None:
        return False
    if module_name in sys.modules:
        return True
    cached = _IMPORT_PROBE_CACHE.get(module_name)
    if cached is not None:
        return cached
    try:
        if importlib.util.find_spec(module_name) is None:
            _IMPORT_PROBE_CACHE[module_name] = False
            return False
    except (ImportError, ValueError):
        _IMPORT_PROBE_CACHE[module_name] = False
        return False
    if _is_frozen():
        _IMPORT_PROBE_CACHE[module_name] = True
        return True
    try:
        script = "import importlib; importlib.import_module(%r)" % module_name
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        _IMPORT_PROBE_CACHE[module_name] = False
        return False
    stderr = result.stderr or ""
    lowered_stderr = stderr.lower()
    ok = (
        result.returncode == 0
        and not any(marker in lowered_stderr for marker in _FATAL_IMPORT_MARKERS)
    )
    if not ok:
        log.info("%s import probe failed; %s.", module_name, failure_context)
    _IMPORT_PROBE_CACHE[module_name] = ok
    return ok
