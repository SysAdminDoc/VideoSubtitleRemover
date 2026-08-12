"""Tests for the centralized native-dependency security floors.

The version floors and advisory metadata must live in a single source
(backend.security_checks) so runtime diagnostics and release verification
cannot drift to a different policy than the vulnerability checks enforce.
"""

from __future__ import annotations

import backend.security_checks as sc
from unittest import mock

from backend.security_checks import (
    FFMPEG_SECURITY_ADVISORY_IDS,
    FFMPEG_SECURITY_ADVISORY_URL,
    FFMPEG_SECURITY_BRANCH_FLOORS,
    FFMPEG_SECURITY_FLOOR,
    FFMPEG_SECURITY_LTS_FLOOR,
    LIBPNG_FIXED_VERSION,
    ffmpeg_security_floor_str,
    ffmpeg_security_affected_range,
    ffmpeg_security_lts_floor_str,
    libpng_fixed_version_str,
    libpng_is_vulnerable,
)


def test_runtime_check_rejects_below_floor():
    below = (LIBPNG_FIXED_VERSION[0], LIBPNG_FIXED_VERSION[1],
             LIBPNG_FIXED_VERSION[2] - 1)
    assert libpng_is_vulnerable(below) is True


def test_runtime_check_accepts_floor_and_above():
    at_floor = LIBPNG_FIXED_VERSION
    above = (LIBPNG_FIXED_VERSION[0], LIBPNG_FIXED_VERSION[1],
             LIBPNG_FIXED_VERSION[2] + 3)
    assert libpng_is_vulnerable(at_floor) is False
    assert libpng_is_vulnerable(above) is False


def test_unknown_version_not_flagged():
    assert libpng_is_vulnerable(None) is False


def test_fixed_version_string_matches_tuple():
    assert libpng_fixed_version_str() == ".".join(
        str(part) for part in LIBPNG_FIXED_VERSION
    )


def test_advisory_range_upper_bound_matches_floor():
    # The affected range upper bound must equal the fixed floor so the
    # advisory and the runtime check agree on the same boundary.
    assert sc.LIBPNG_AFFECTED_RANGE.endswith(
        "<" + libpng_fixed_version_str())


def test_opencv_ocr_error_uses_shared_floor():
    from backend import opencv_ocr

    report = opencv_ocr.collect_opencv_dnn_ocr_status(
        libpng={"vulnerable": True, "libpng_version": "1.6.40"},
    )
    joined = " ".join(report["errors"])
    assert libpng_fixed_version_str() in joined


def test_release_advisory_derives_from_constants(monkeypatch):
    from backend import release_verification as rv

    monkeypatch.setattr(
        rv, "opencv_libpng_status",
        lambda: {"vulnerable": True, "libpng_version": "1.6.40",
                 "fixed_version": None},
    )
    result = rv.collect_release_advisories([])
    libpng_findings = [
        f for f in result["advisories"]
        if f.get("package") == "opencv-python bundled libpng"
    ]
    assert libpng_findings, "expected a libpng advisory when vulnerable"
    advisory = libpng_findings[0]
    assert advisory["id"] == sc.LIBPNG_CVE
    assert advisory["fixedIn"] == libpng_fixed_version_str()
    assert advisory["affected"] == sc.LIBPNG_AFFECTED_RANGE
    assert advisory["source"] == sc.LIBPNG_ADVISORY_URL


def test_ffmpeg_policy_is_shared_with_classifier():
    from backend import ffmpeg_profiles

    assert ffmpeg_profiles.FFMPEG_SECURITY_ADVISORY_IDS == FFMPEG_SECURITY_ADVISORY_IDS
    assert ffmpeg_profiles.FFMPEG_SECURITY_SOURCE == FFMPEG_SECURITY_ADVISORY_URL
    assert ffmpeg_profiles.FFMPEG_RELEASE_SOURCE == sc.FFMPEG_SECURITY_RELEASE_URL
    for branch, floor in FFMPEG_SECURITY_BRANCH_FLOORS.items():
        assert ffmpeg_profiles._FFMPEG_VULNERABLE_LINES[branch] == (
            floor[2], ".".join(str(part) for part in floor)
        )
    assert FFMPEG_SECURITY_FLOOR == (8, 1, 2)
    assert FFMPEG_SECURITY_LTS_FLOOR == (8, 0, 3)
    assert ffmpeg_security_floor_str() == "8.1.2"
    assert ffmpeg_security_lts_floor_str() == "8.0.3"
    assert ffmpeg_security_affected_range() == "8.1.0-8.1.1, 8.0.0-8.0.2"


def test_ffmpeg_vulnerability_warning_carries_cves_and_advisory_url():
    from backend import ffmpeg_profiles

    status = ffmpeg_profiles.classify_ffmpeg_security("ffmpeg version 8.1.1")
    assert status["advisories"] == list(FFMPEG_SECURITY_ADVISORY_IDS)
    assert status["advisory_url"] == FFMPEG_SECURITY_ADVISORY_URL
    assert FFMPEG_SECURITY_ADVISORY_URL in status["reason"]
    for advisory_id in FFMPEG_SECURITY_ADVISORY_IDS:
        assert advisory_id in status["reason"]


def test_ffmpeg_probe_records_pass_fail():
    from backend import ffmpeg_profiles

    with mock.patch.object(
        ffmpeg_profiles,
        "_tool_status",
        return_value={
            "available": True,
            "path": "ffmpeg",
            "version": "ffmpeg version 8.1.2",
        },
    ):
        assert ffmpeg_profiles.probe_ffmpeg_security()["passed"] is True

    with mock.patch.object(
        ffmpeg_profiles,
        "_tool_status",
        return_value={
            "available": True,
            "path": "ffmpeg",
            "version": "ffmpeg version 8.1.1",
        },
    ):
        assert ffmpeg_profiles.probe_ffmpeg_security()["passed"] is False
