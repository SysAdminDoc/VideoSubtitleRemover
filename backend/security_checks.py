"""Runtime security checks for bundled native dependencies."""

from __future__ import annotations

import re
from typing import Optional, Tuple


LIBPNG_FIXED_VERSION = (1, 6, 54)


def parse_libpng_version(build_info: str) -> Optional[Tuple[int, int, int]]:
    """Extract OpenCV's bundled libpng version from getBuildInformation()."""
    match = re.search(r"\bPNG:\s+.*?\(ver\s+(\d+)\.(\d+)\.(\d+)\)", build_info)
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def libpng_is_vulnerable(version: Optional[Tuple[int, int, int]]) -> bool:
    if version is None:
        return False
    return version < LIBPNG_FIXED_VERSION


def format_libpng_version(version: Tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in version)


def opencv_libpng_status() -> dict:
    """Return runtime status for OpenCV's bundled libpng."""
    fixed = format_libpng_version(LIBPNG_FIXED_VERSION)
    try:
        import cv2
    except Exception as exc:
        return {
            "available": False,
            "opencv_version": None,
            "libpng_version": None,
            "fixed_version": fixed,
            "vulnerable": None,
            "warning": None,
            "error": str(exc),
        }

    try:
        version = parse_libpng_version(cv2.getBuildInformation())
    except Exception as exc:
        return {
            "available": True,
            "opencv_version": getattr(cv2, "__version__", None),
            "libpng_version": None,
            "fixed_version": fixed,
            "vulnerable": None,
            "warning": None,
            "error": str(exc),
        }

    current = format_libpng_version(version) if version else None
    vulnerable = libpng_is_vulnerable(version)
    warning = None
    if vulnerable and current:
        warning = (
            f"OpenCV reports bundled libpng {current}; CVE-2026-22801 is "
            f"fixed in libpng {fixed} or newer. Avoid untrusted PNG input "
            "until opencv-python ships a wheel with the fixed library."
        )
    return {
        "available": True,
        "opencv_version": getattr(cv2, "__version__", None),
        "libpng_version": current,
        "fixed_version": fixed,
        "vulnerable": vulnerable,
        "warning": warning,
        "error": None,
    }


def warn_if_vulnerable_opencv_libpng(logger) -> Optional[str]:
    """Log a warning when OpenCV reports a libpng build below the fixed floor."""
    status = opencv_libpng_status()
    message = status.get("warning")
    if not message:
        return None
    logger.warning(str(message))
    return str(message)
