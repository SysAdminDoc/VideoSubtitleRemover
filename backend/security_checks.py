"""Runtime security checks for bundled native dependencies."""

from __future__ import annotations

import re
from typing import Optional, Tuple


# Single source of truth for the libpng security floor. CVE-2026-22801 is
# fixed in libpng 1.6.54; the reviewed opencv-python 5.0.0.93 wheel bundles a
# newer build (1.6.57) that satisfies this floor. Every consumer (opencv_ocr,
# release_verification, safe_image) must derive its advisory text from these
# constants so the floor cannot drift between modules. Raise the floor only
# alongside a newly cited libpng CVE, never speculatively -- a higher floor
# would falsely flag a genuinely patched build as vulnerable.
LIBPNG_FIXED_VERSION = (1, 6, 54)
LIBPNG_CVE = "CVE-2026-22801"
LIBPNG_AFFECTED_RANGE = ">=1.6.26,<1.6.54"
LIBPNG_ADVISORY_URL = "https://nvd.nist.gov/vuln/detail/CVE-2026-22801"


# Single source of truth for the FFmpeg runtime security floor. FFmpeg 9.0.1
# is the reviewed floor: the 8.x series ended at 8.1.2 (2026-06-17), so the
# 2026-07-24 advisory batch below has no fix on any 8.x branch and the only
# remedy is moving to the 9.0 line. Keep the advisory identifiers and source
# URL beside the floors so diagnostics, release evidence, and cross-source
# tests cannot drift independently. Raise the floor only alongside a newly
# cited FFmpeg CVE, never speculatively.
FFMPEG_SECURITY_FLOOR = (9, 0, 1)
FFMPEG_SECURITY_BRANCH_FLOORS = {
    (9, 0): FFMPEG_SECURITY_FLOOR,
}
# Branches that carry published advisories and will never receive a fix
# because upstream has closed them. Any patch level on these lines is
# vulnerable, so the remedy is a cross-branch upgrade to the floor above.
FFMPEG_SECURITY_EOL_BRANCHES = ((8, 1), (8, 0))
FFMPEG_SECURITY_EOL_FINAL_RELEASES = {
    (8, 1): (8, 1, 2),
    (8, 0): (8, 0, 3),
}
# Reachable from crafted media that VSR decodes: IAMF demuxer allocation,
# LCL/ZLIB uninitialized-heap disclosure, MACE6 decoder overflow, and the
# VobSub subtitle demuxer overflow that sits on the soft-subtitle remux path.
# The two 8.1.2-era identifiers stay listed because builds below that release
# remain exposed to them as well.
FFMPEG_SECURITY_ADVISORY_IDS = (
    "CVE-2026-66037",
    "CVE-2026-66038",
    "CVE-2026-66039",
    "CVE-2026-64830",
    "CVE-2026-12706",
    "CVE-2026-8461",
    "CVE-2026-30999",
)
# Published advisories with no upstream fixed version identified yet. These
# are reported as open risk and must never be presented as remediated by
# meeting the floor above.
FFMPEG_SECURITY_UNFIXED_ADVISORY_IDS = ("CVE-2026-58049",)
FFMPEG_SECURITY_ADVISORY_URL = "https://ffmpeg.org/security.html"
FFMPEG_SECURITY_RELEASE_URL = "https://ffmpeg.org/download.html"


def format_ffmpeg_version(version: Tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in version)


def ffmpeg_security_floor_str() -> str:
    """Return the reviewed current-line FFmpeg floor as dotted text."""
    return format_ffmpeg_version(FFMPEG_SECURITY_FLOOR)


def ffmpeg_security_eol_branch_str() -> str:
    """Return the closed FFmpeg branches as dotted text."""
    return ", ".join(
        f"{major}.{minor}.x" for major, minor in FFMPEG_SECURITY_EOL_BRANCHES
    )


def ffmpeg_security_affected_range() -> str:
    """Return the affected version ranges used by release advisories.

    Covers both the reviewed branches that have a fixed release and the
    closed branches where every patch level stays affected.
    """
    ranges = []
    for (major, minor), floor in FFMPEG_SECURITY_BRANCH_FLOORS.items():
        if floor[2] > 0:
            ranges.append(
                f"{major}.{minor}.0-{major}.{minor}.{floor[2] - 1}"
            )
    for major, minor in FFMPEG_SECURITY_EOL_BRANCHES:
        ranges.append(f"{major}.{minor}.x (no fixed release)")
    return ", ".join(ranges)


def libpng_fixed_version_str() -> str:
    """Return the libpng security floor as a dotted string (single source)."""
    return format_libpng_version(LIBPNG_FIXED_VERSION)


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
            f"OpenCV reports bundled libpng {current}; {LIBPNG_CVE} is "
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
