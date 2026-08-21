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
    "CVE-2026-58049",
)
# Advisories that landed in the current floor but not in the first tag of
# that open branch. 9.0.0 is "outdated" (fail-closed) and must still name
# CVE-2026-58049; the July 2026 batch must not be named against it.
FFMPEG_FLOOR_ONLY_ADVISORY_IDS = ("CVE-2026-58049",)
# Published advisories with no upstream fixed version identified yet. These
# are reported as open risk and must never be presented as remediated by
# meeting the floor above. Empty as of 2026-08-21: CVE-2026-58049's RASC
# DLTA bounds check is in n9.0.1 (commit f8d7795, 2026-07-21).
FFMPEG_SECURITY_UNFIXED_ADVISORY_IDS = ()
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


# Single source of truth for the CPython security floors, one per supported
# release line. The floors match CVE-2026-11940 (tarfile path-traversal,
# including Windows symlink validation) on every line: VSR extracts untrusted
# archives in cache_inventory, release_staging and support_bundle. Other
# advisory IDs are attached per-line from CPYTHON_ADVISORY_FIXES -- never
# name a CVE against a version that already carries its fix.
CPYTHON_SECURITY_FLOORS = {
    (3, 11): (3, 11, 16),
    (3, 12): (3, 12, 14),
    (3, 13): (3, 13, 15),
    (3, 14): (3, 14, 7),
}
# Fixed versions per release line. A build is exposed when it is strictly
# less than the listed triple on its line. CVE-2026-11940 is the tarfile
# identifier that justifies every current floor. The April 2026 three
# (SourcelessFileLoader / expat / http.cookies) shipped in 3.11.16 and
# 3.12.14 with the August releases, but landed earlier on 3.13/3.14.
# CVE-2026-6100 (decompressor UAF) is 3.13.14 / 3.14.5, not the August floor.
CPYTHON_ADVISORY_FIXES = {
    "CVE-2026-11940": {
        (3, 11): (3, 11, 16),
        (3, 12): (3, 12, 14),
        (3, 13): (3, 13, 15),
        (3, 14): (3, 14, 7),
    },
    "CVE-2026-2297": {
        (3, 11): (3, 11, 16),
        (3, 12): (3, 12, 14),
        (3, 13): (3, 13, 13),
        (3, 14): (3, 14, 4),
    },
    "CVE-2026-4224": {
        (3, 11): (3, 11, 16),
        (3, 12): (3, 12, 14),
        (3, 13): (3, 13, 13),
        (3, 14): (3, 14, 4),
    },
    "CVE-2026-3644": {
        (3, 11): (3, 11, 16),
        (3, 12): (3, 12, 14),
        (3, 13): (3, 13, 13),
        (3, 14): (3, 14, 4),
    },
    "CVE-2026-6100": {
        (3, 11): (3, 11, 16),
        (3, 12): (3, 12, 14),
        (3, 13): (3, 13, 14),
        (3, 14): (3, 14, 5),
    },
}
CPYTHON_SECURITY_ADVISORY_IDS = tuple(CPYTHON_ADVISORY_FIXES)
CPYTHON_SECURITY_ADVISORY_URL = (
    "https://blog.python.org/2026/08/python-31214-31116-31021/"
)


def cpython_advisories_for(version: Tuple[int, ...]) -> Tuple[str, ...]:
    """Return CVEs this interpreter is still exposed to on its release line."""
    version = tuple(int(part) for part in version[:3])
    while len(version) < 3:
        version = version + (0,)
    line = (version[0], version[1])
    exposed = []
    for advisory_id, fixes in CPYTHON_ADVISORY_FIXES.items():
        fixed = fixes.get(line)
        if fixed is not None and version < fixed:
            exposed.append(advisory_id)
    return tuple(exposed)


def format_cpython_version(version: Tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in version[:3])


def cpython_security_status(
    version: Optional[Tuple[int, ...]] = None,
) -> dict:
    """Classify an interpreter against the reviewed CPython floors.

    Lines newer than the reviewed table postdate the advisories and are
    reported as unclassified-but-acceptable rather than falsely flagged;
    lines older than the table are below the project's supported floor.
    """
    if version is None:
        import sys

        version = tuple(sys.version_info[:3])
    version = tuple(int(part) for part in version[:3])
    while len(version) < 3:
        version = version + (0,)
    line = (version[0], version[1])
    payload = {
        "version": format_cpython_version(version),
        "line": f"{line[0]}.{line[1]}",
        "classified": False,
        "safe": True,
        "floor": "",
        "advisories": [],
        "advisory_url": CPYTHON_SECURITY_ADVISORY_URL,
        "reason": "",
    }
    floor = CPYTHON_SECURITY_FLOORS.get(line)
    if floor is not None:
        payload["classified"] = True
        payload["floor"] = format_cpython_version(floor)
        if version < floor:
            payload["safe"] = False
            advisories = cpython_advisories_for(version)
            payload["advisories"] = list(advisories)
            named = ", ".join(advisories) if advisories else "reviewed floors"
            payload["reason"] = (
                f"Python {payload['version']} predates the "
                f"{payload['floor']} security release "
                f"({named}); upgrade to {payload['floor']} or newer"
            )
        else:
            payload["reason"] = (
                f"Python {payload['version']} meets the {payload['floor']} "
                "security floor"
            )
        return payload
    known_lines = sorted(CPYTHON_SECURITY_FLOORS)
    if line < known_lines[0]:
        payload["safe"] = False
        payload["reason"] = (
            f"Python {payload['version']} is below VSR's supported "
            f"{known_lines[0][0]}.{known_lines[0][1]} floor"
        )
    else:
        payload["reason"] = (
            f"Python {payload['version']} is newer than VSR's reviewed "
            "release lines; it postdates the tracked advisories but is not "
            "explicitly classified"
        )
    return payload


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
