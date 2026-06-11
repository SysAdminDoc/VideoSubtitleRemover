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


def warn_if_vulnerable_opencv_libpng(logger) -> Optional[str]:
    """Log a warning when OpenCV reports a libpng build below the fixed floor."""
    try:
        import cv2
    except Exception:
        return None

    try:
        version = parse_libpng_version(cv2.getBuildInformation())
    except Exception:
        return None
    if not libpng_is_vulnerable(version):
        return None

    current = format_libpng_version(version)
    fixed = format_libpng_version(LIBPNG_FIXED_VERSION)
    message = (
        f"OpenCV reports bundled libpng {current}; CVE-2026-22801 is "
        f"fixed in libpng {fixed} or newer. Avoid untrusted PNG input until "
        "opencv-python ships a wheel with the fixed library."
    )
    logger.warning(message)
    return message
