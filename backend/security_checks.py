"""Runtime security checks for bundled native dependencies."""

from __future__ import annotations

import re
from typing import Mapping, Optional, Sequence, Tuple


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


# OpenCV's Python wheel carries its own FFmpeg libraries and never prints an
# FFmpeg tag, only library ABI versions. RM-320: those ABI numbers do identify
# the release branch the wheel was built from, because libavutil, libavcodec,
# and libavformat all step together at an FFmpeg major release. That mapping
# is what lets the embedded decoder be classified against the same floor the
# external binary must meet, instead of being left permanently unclassified.
OPENCV_FFMPEG_STATUS_SCHEMA = "vsr.opencv_ffmpeg.v1"
OPENCV_FFMPEG_PROVENANCE_SOURCE = (
    "https://github.com/opencv/opencv/releases/tag/5.0.0"
)
OPENCV_FFMPEG_WHEEL_SOURCE = "https://github.com/opencv/opencv-python"
OPENCV_FFMPEG_REQUIRED_LIBRARIES = ("avcodec", "avformat", "avutil")

# Measured, not inferred. Each entry is an (avutil, avcodec, avformat) ABI
# triple read from a build whose FFmpeg release is known:
#   9.0.1  -- `ffmpeg -version` on the reviewed external binary, 2026-08-27
#   7.1    -- opencv-python 5.0.0.93's bundled
#             opencv_videoio_ffmpeg500_64.dll, which OpenCV pins at n7.1
# ABI numbers cannot separate 9.0.0 from 9.0.1, so a triple on the 9.0 branch
# is reported as meeting the floor branch rather than as a specific release.
OPENCV_FFMPEG_ABI_RELEASES: Mapping[Tuple[Tuple[int, ...], ...], str] = {
    ((61, 1, 101), (63, 1, 101), (63, 1, 101)): "9.0.1",
    ((59, 39, 100), (61, 19, 100), (61, 7, 100)): "7.1",
}
# The major of each library on the enforced floor's branch. Anything below
# these majors predates FFmpeg 9.0 and therefore the whole advisory set the
# external floor exists for.
OPENCV_FFMPEG_FLOOR_ABI_MAJORS: Mapping[str, int] = {
    "avutil": 61,
    "avcodec": 63,
    "avformat": 63,
}
OPENCV_FFMPEG_ABI_BRANCHES: Mapping[Tuple[int, int, int], str] = {
    (61, 63, 63): "9.0",
    (59, 61, 61): "7.1",
}
# RM-320: the embedded runtime is now classified against the same floor as the
# external binary. Each rule fires when a component's ABI predates the floor
# branch, which is the condition that puts the build outside every advisory in
# FFMPEG_SECURITY_ADVISORY_IDS.
OPENCV_FFMPEG_ADVISORY_RULES: Tuple[Mapping[str, object], ...] = (
    {
        "id": "VSR-OPENCV-FFMPEG-FLOOR",
        "component": "avcodec",
        "affectedBefore": "63.0.0",
        "affected": "libavcodec below 63, i.e. an FFmpeg branch before 9.0",
        "fixedIn": "libavcodec 63 (FFmpeg 9.0)",
        "advisories": list(FFMPEG_SECURITY_ADVISORY_IDS),
        "source": FFMPEG_SECURITY_ADVISORY_URL,
    },
    {
        "id": "VSR-OPENCV-FFMPEG-FLOOR",
        "component": "avformat",
        "affectedBefore": "63.0.0",
        "affected": "libavformat below 63, i.e. an FFmpeg branch before 9.0",
        "fixedIn": "libavformat 63 (FFmpeg 9.0)",
        "advisories": list(FFMPEG_SECURITY_ADVISORY_IDS),
        "source": FFMPEG_SECURITY_ADVISORY_URL,
    },
)


# RM-320: the embedded runtime is genuinely below the floor and the fix is
# not ours to make -- opencv-python decides which FFmpeg its wheel carries.
# Shipping anyway is therefore a decision, recorded here with a date, the
# exact release it applies to, and the residual exposure. Release
# verification refuses to downgrade the block unless the identified release
# matches this entry exactly, so a wheel that moves to a different old branch
# fails until somebody looks at it again.
OPENCV_FFMPEG_ACKNOWLEDGED_RELEASE = "7.1"
OPENCV_FFMPEG_ACKNOWLEDGEMENT: Mapping[str, str] = {
    "recorded": "2026-08-28",
    "release": OPENCV_FFMPEG_ACKNOWLEDGED_RELEASE,
    "wheel": "opencv-python==5.0.0.93",
    "reason": (
        "opencv-python pins its own FFmpeg at n7.1 and publishes no wheel "
        "built against the 9.0 branch, so this cannot be fixed by pinning. "
        "Every decode the product performs on user-supplied media is being "
        "moved onto the external FFmpeg binary, which does meet the floor; "
        "until that is finished the embedded decoder remains reachable from "
        "the preview, region-selection, and mask-correction paths."
    ),
    "residualExposure": (
        "cv2.VideoCapture calls that still open user-supplied media"
    ),
    "tracking": "ROADMAP.md RM-348",
}


def opencv_ffmpeg_release_from_abi(libraries: Optional[Mapping[str, object]]) -> dict:
    """Name the FFmpeg release or branch behind an embedded ABI triple."""
    payload = {
        "release": "",
        "branch": "",
        "identified": False,
        "abi": {},
        "belowFloor": None,
    }
    if not isinstance(libraries, Mapping):
        return payload
    triple = []
    majors = []
    for name in ("avutil", "avcodec", "avformat"):
        entry = libraries.get(name)
        parts = tuple(
            (entry or {}).get("versionTuple") or ()
        ) if isinstance(entry, Mapping) else ()
        if len(parts) < 3:
            return payload
        triple.append(tuple(int(part) for part in parts[:3]))
        majors.append(int(parts[0]))
        payload["abi"][name] = ".".join(str(part) for part in parts[:3])
    exact = OPENCV_FFMPEG_ABI_RELEASES.get(tuple(triple))
    branch = OPENCV_FFMPEG_ABI_BRANCHES.get(tuple(majors), "")
    payload["release"] = exact or ""
    payload["branch"] = branch
    payload["identified"] = bool(exact or branch)
    payload["belowFloor"] = any(
        major < OPENCV_FFMPEG_FLOOR_ABI_MAJORS[name]
        for name, major in zip(
            ("avutil", "avcodec", "avformat"), majors, strict=True)
    )
    return payload
_OPENCV_FFMPEG_LINE_RE = re.compile(
    r"^\s*(avcodec|avformat|avutil):\s+(YES|NO)"
    r"(?:\s+\(([^)]*)\))?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_OPENCV_FFMPEG_HEADER_RE = re.compile(
    r"^\s*FFMPEG:\s+(YES|NO)(?:\s+\(([^)]*)\))?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _numeric_version(value: object) -> Tuple[int, ...]:
    parts = [int(part) for part in re.findall(r"\d+", str(value or ""))]
    return tuple(parts[:3])


def parse_opencv_ffmpeg_build_info(build_info: object) -> dict:
    """Extract OpenCV's embedded FFmpeg ABI inventory from build text."""
    text = str(build_info or "")
    header = _OPENCV_FFMPEG_HEADER_RE.search(text)
    libraries = {
        name: {
            "available": False,
            "version": None,
            "versionTuple": [],
        }
        for name in OPENCV_FFMPEG_REQUIRED_LIBRARIES
    }
    for match in _OPENCV_FFMPEG_LINE_RE.finditer(text):
        name = match.group(1).lower()
        version = (match.group(3) or "").strip() or None
        parsed = _numeric_version(version)
        libraries[name] = {
            "available": match.group(2).upper() == "YES",
            "version": version,
            "versionTuple": list(parsed),
        }
    ffmpeg_enabled = None
    ffmpeg_build = ""
    if header:
        ffmpeg_enabled = header.group(1).upper() == "YES"
        ffmpeg_build = (header.group(2) or "").strip()
    parseable = bool(
        ffmpeg_enabled is True
        and all(
            item["available"] and item["version"] and item["versionTuple"]
            for item in libraries.values()
        )
    )
    return {
        "ffmpegEnabled": ffmpeg_enabled,
        "ffmpegBuild": ffmpeg_build,
        "libraries": libraries,
        "parseable": parseable,
    }


def _opencv_wheel_provenance(wheel_status: Optional[Mapping[str, object]]) -> dict:
    if not isinstance(wheel_status, Mapping):
        return {
            "schema": "vsr.opencv_wheel_provenance.v1",
            "distribution": "",
            "version": None,
            "importedFile": "",
            "source": OPENCV_FFMPEG_WHEEL_SOURCE,
        }
    recorded = wheel_status.get("provenance")
    if isinstance(recorded, Mapping):
        result = dict(recorded)
        result.setdefault("schema", "vsr.opencv_wheel_provenance.v1")
        result.setdefault("source", OPENCV_FFMPEG_WHEEL_SOURCE)
        return result
    imported = wheel_status.get("imported")
    imported = imported if isinstance(imported, Mapping) else {}
    return {
        "schema": "vsr.opencv_wheel_provenance.v1",
        "distribution": str(imported.get("owner") or ""),
        "version": imported.get("version"),
        "importedFile": str(imported.get("file") or ""),
        "source": OPENCV_FFMPEG_WHEEL_SOURCE,
    }


def opencv_ffmpeg_status(
    *,
    wheel_status: Optional[Mapping[str, object]] = None,
    build_info: Optional[str] = None,
    opencv_version: Optional[str] = None,
    advisory_rules: Optional[Sequence[Mapping[str, object]]] = None,
) -> dict:
    """Inventory OpenCV's embedded FFmpeg and apply only cited mappings.

    OpenCV reports FFmpeg library ABI versions rather than an upstream FFmpeg
    tag. The release gate therefore blocks only when ``advisory_rules`` has a
    component, an affected-before version, an advisory identifier, and a
    source URL. Current OpenCV builds have no such mapping in this repository.
    """
    provenance = {
        "source": OPENCV_FFMPEG_PROVENANCE_SOURCE,
        "wheelSource": OPENCV_FFMPEG_WHEEL_SOURCE,
        "rule": (
            "Record embedded ABI versions and wheel provenance. Block only on "
            "a cited advisory rule that maps a component ABI to an affected "
            "range."
        ),
        "advisoryRules": [
            dict(item)
            for item in (
                advisory_rules if advisory_rules is not None
                else OPENCV_FFMPEG_ADVISORY_RULES
            )
        ],
    }
    payload = {
        "schema": OPENCV_FFMPEG_STATUS_SCHEMA,
        "available": False,
        "opencv_version": opencv_version,
        "wheel": _opencv_wheel_provenance(wheel_status),
        "ffmpeg": {
            "enabled": None,
            "build": "",
        },
        "avcodec": {"available": False, "version": None, "versionTuple": []},
        "avformat": {"available": False, "version": None, "versionTuple": []},
        "avutil": {"available": False, "version": None, "versionTuple": []},
        "libraries": {},
        "provenance": provenance,
        "embeddedRelease": {
            "release": "", "branch": "", "identified": False,
            "abi": {}, "belowFloor": None,
        },
        "securityFloor": ffmpeg_security_floor_str(),
        "classification": "unknown",
        "vulnerable": None,
        "blocking": False,
        "advisories": [],
        "passed": False,
        "warning": "",
        "error": "",
    }
    try:
        if build_info is None:
            import cv2

            build_info = cv2.getBuildInformation()
            if not payload["opencv_version"]:
                payload["opencv_version"] = getattr(cv2, "__version__", None)
        parsed = parse_opencv_ffmpeg_build_info(build_info)
    except Exception as exc:
        payload["error"] = str(exc)
        return payload

    payload["available"] = True
    payload["embeddedRelease"] = opencv_ffmpeg_release_from_abi(
        parsed["libraries"])
    payload["ffmpeg"] = {
        "enabled": parsed["ffmpegEnabled"],
        "build": parsed["ffmpegBuild"],
    }
    payload["libraries"] = parsed["libraries"]
    for name in OPENCV_FFMPEG_REQUIRED_LIBRARIES:
        payload[name] = parsed["libraries"][name]
    if not parsed["parseable"]:
        payload["error"] = (
            "OpenCV build information did not provide enabled, versioned "
            "avcodec, avformat, and avutil entries"
        )
        return payload

    rules = advisory_rules if advisory_rules is not None else OPENCV_FFMPEG_ADVISORY_RULES
    invalid_rules = []
    matches = []
    for rule in rules:
        if not isinstance(rule, Mapping):
            invalid_rules.append("rule is not an object")
            continue
        advisory_id = str(rule.get("id") or "")
        source = str(rule.get("source") or "")
        component = str(rule.get("component") or "").lower()
        affected_before = _numeric_version(rule.get("affectedBefore"))
        if not advisory_id or not source or component not in payload["libraries"] or not affected_before:
            invalid_rules.append(dict(rule))
            continue
        observed = tuple(payload["libraries"][component]["versionTuple"])
        if observed < affected_before:
            named = rule.get("advisories")
            matches.append({
                "id": advisory_id,
                "component": component,
                "version": payload["libraries"][component]["version"],
                "affected": str(rule.get("affected") or f"<{rule.get('affectedBefore')}"),
                "fixedIn": str(rule.get("fixedIn") or rule.get("affectedBefore")),
                "advisories": list(named) if isinstance(named, Sequence)
                and not isinstance(named, (str, bytes)) else [],
                "source": source,
            })
    provenance["invalidRules"] = invalid_rules
    payload["advisories"] = matches
    if matches:
        identity = payload["embeddedRelease"]
        named = (
            identity.get("release")
            or (f"{identity.get('branch')} branch" if identity.get("branch")
                else "an unidentified release")
        )
        payload["classification"] = "vulnerable"
        payload["vulnerable"] = True
        payload["blocking"] = True
        payload["warning"] = (
            f"OpenCV's wheel embeds FFmpeg {named} "
            f"(libavcodec {payload['avcodec']['version']}, "
            f"libavformat {payload['avformat']['version']}), which is below "
            f"the {payload['securityFloor']} floor enforced on the external "
            "binary. Keep untrusted media off cv2.VideoCapture."
        )
    elif rules and not invalid_rules:
        payload["classification"] = "safe"
        payload["vulnerable"] = False
        payload["warning"] = ""
    else:
        payload["classification"] = "unmapped"
        payload["warning"] = (
            "OpenCV's embedded FFmpeg ABI is inventoried, but no cited "
            "advisory mapping exists. No vulnerability claim was made."
        )
    payload["passed"] = not payload["blocking"]
    return payload


def warn_if_vulnerable_opencv_libpng(logger) -> Optional[str]:
    """Log a warning when OpenCV reports a libpng build below the fixed floor."""
    status = opencv_libpng_status()
    message = status.get("warning")
    if not message:
        return None
    logger.warning(str(message))
    return str(message)
