"""Tests for the centralized libpng security floor.

The libpng version floor and its advisory metadata must live in a single
source (backend.security_checks) so opencv_ocr and release_verification cannot
drift to a different version than the runtime vulnerability check enforces.
"""

from __future__ import annotations

import backend.security_checks as sc
from backend.security_checks import (
    LIBPNG_FIXED_VERSION,
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
