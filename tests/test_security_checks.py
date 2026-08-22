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
    LIBPNG_FIXED_VERSION,
    ffmpeg_security_floor_str,
    ffmpeg_security_affected_range,
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
    assert FFMPEG_SECURITY_FLOOR == (9, 0, 1)
    assert ffmpeg_security_floor_str() == "9.0.1"
    assert sc.FFMPEG_SECURITY_EOL_BRANCHES == ((8, 1), (8, 0))
    assert sc.ffmpeg_security_eol_branch_str() == "8.1.x, 8.0.x"
    assert ffmpeg_security_affected_range() == (
        "9.0.0-9.0.0, 8.1.x (no fixed release), 8.0.x (no fixed release)"
    )
    # The 2026-07-24 batch plus the RASC DLTA bounds check (CVE-2026-58049)
    # which n9.0.1 carries. There is no remaining unfixed FFmpeg advisory
    # in this table.
    for advisory_id in (
        "CVE-2026-66037",
        "CVE-2026-66038",
        "CVE-2026-66039",
        "CVE-2026-64830",
        "CVE-2026-12706",
        "CVE-2026-58049",
    ):
        assert advisory_id in FFMPEG_SECURITY_ADVISORY_IDS
    assert sc.FFMPEG_SECURITY_UNFIXED_ADVISORY_IDS == ()
    assert "CVE-2026-58049" in FFMPEG_SECURITY_ADVISORY_IDS


def test_ffmpeg_vulnerability_warning_carries_cves_and_advisory_url():
    from backend import ffmpeg_profiles

    status = ffmpeg_profiles.classify_ffmpeg_security("ffmpeg version 8.1.1")
    assert status["advisories"] == list(FFMPEG_SECURITY_ADVISORY_IDS)
    assert status["advisory_url"] == FFMPEG_SECURITY_ADVISORY_URL
    assert FFMPEG_SECURITY_ADVISORY_URL in status["reason"]


def test_ffmpeg_open_branch_behind_point_release_names_floor_only_cves():
    """9.0.0 has the July 2026 batch but not the n9.0.1 RASC DLTA check."""
    from backend import ffmpeg_profiles

    status = ffmpeg_profiles.classify_ffmpeg_security("ffmpeg version 9.0.0")
    assert status["classification"] == "outdated"
    assert status["supported"] is True
    assert status["safe"] is False
    assert status["vulnerable"] is False
    assert status["advisories"] == ["CVE-2026-58049"]
    assert status["fixed_in"] == "9.0.1"
    assert "CVE-2026-58049" in status["reason"]
    for advisory_id in FFMPEG_SECURITY_ADVISORY_IDS:
        if advisory_id == "CVE-2026-58049":
            continue
        assert advisory_id not in status["reason"]


def test_ffmpeg_closed_branch_is_vulnerable_at_every_patch_level():
    """The 8.x series ended at 8.1.2, so its final release is still exposed."""
    from backend import ffmpeg_profiles

    for banner in ("ffmpeg version 8.1.2", "ffmpeg version 8.0.3"):
        status = ffmpeg_profiles.classify_ffmpeg_security(banner)
        assert status["classification"] == "vulnerable", banner
        assert status["vulnerable"] is True
        assert status["safe"] is False
        # The remedy is a cross-branch upgrade, not a patch on that branch.
        assert status["fixed_in"] == "9.0.1"
        assert "closed" in status["reason"]
        assert "CVE-2026-64830" in status["reason"]


def test_ffmpeg_safe_build_has_no_open_advisories():
    from backend import ffmpeg_profiles

    status = ffmpeg_profiles.classify_ffmpeg_security("ffmpeg version 9.0.1")
    assert status["classification"] == "safe"
    assert status["open_advisories"] == []


def test_ffmpeg_probe_records_pass_fail():
    from backend import ffmpeg_profiles

    with mock.patch.object(
        ffmpeg_profiles,
        "_tool_status",
        return_value={
            "available": True,
            "path": "ffmpeg",
            "version": "ffmpeg version 9.0.1",
        },
    ):
        assert ffmpeg_profiles.probe_ffmpeg_security()["passed"] is True

    for banner in ("ffmpeg version 9.0.0", "ffmpeg version 8.1.2"):
        with mock.patch.object(
            ffmpeg_profiles,
            "_tool_status",
            return_value={
                "available": True,
                "path": "ffmpeg",
                "version": banner,
            },
        ):
            assert ffmpeg_profiles.probe_ffmpeg_security()["passed"] is False


def test_opencv_ffmpeg_build_info_records_required_abi_versions():
    from backend.security_checks import parse_opencv_ffmpeg_build_info

    status = parse_opencv_ffmpeg_build_info(
        """
    FFMPEG:                      YES (prebuilt binaries)
      avcodec:                   YES (61.19.100)
      avformat:                  YES (61.7.100)
      avutil:                    YES (59.39.100)
        """
    )

    assert status["parseable"] is True
    assert status["ffmpegBuild"] == "prebuilt binaries"
    assert status["libraries"]["avcodec"]["version"] == "61.19.100"
    assert status["libraries"]["avformat"]["version"] == "61.7.100"
    assert status["libraries"]["avutil"]["version"] == "59.39.100"


def test_opencv_ffmpeg_without_advisory_mapping_is_not_called_vulnerable():
    from backend.security_checks import opencv_ffmpeg_status

    status = opencv_ffmpeg_status(
        build_info=(
            "FFMPEG: YES (prebuilt binaries)\n"
            "  avcodec: YES (61.19.100)\n"
            "  avformat: YES (61.7.100)\n"
            "  avutil: YES (59.39.100)\n"
        ),
        opencv_version="5.0.0",
        wheel_status={
            "provenance": {
                "schema": "vsr.opencv_wheel_provenance.v1",
                "distribution": "opencv-python",
                "version": "5.0.0.93",
                "source": "importlib.metadata",
            },
        },
    )

    assert status["passed"] is True
    assert status["classification"] == "unmapped"
    assert status["vulnerable"] is None
    assert status["blocking"] is False
    assert status["wheel"]["distribution"] == "opencv-python"
    assert status["provenance"]["advisoryRules"] == []


def test_opencv_ffmpeg_cited_affected_rule_fails_closed():
    from backend.security_checks import opencv_ffmpeg_status

    status = opencv_ffmpeg_status(
        build_info=(
            "FFMPEG: YES (prebuilt binaries)\n"
            "  avcodec: YES (61.19.100)\n"
            "  avformat: YES (61.7.100)\n"
            "  avutil: YES (59.39.100)\n"
        ),
        advisory_rules=(
            {
                "id": "CVE-TEST-OPENCV-FFMPEG",
                "component": "avcodec",
                "affectedBefore": "62.0.0",
                "fixedIn": "62.0.0",
                "source": "https://example.invalid/advisory",
            },
        ),
    )

    assert status["passed"] is False
    assert status["classification"] == "vulnerable"
    assert status["vulnerable"] is True
    assert status["blocking"] is True
    assert status["advisories"][0]["id"] == "CVE-TEST-OPENCV-FFMPEG"


def test_opencv_ffmpeg_rule_without_citation_cannot_block():
    from backend.security_checks import opencv_ffmpeg_status

    status = opencv_ffmpeg_status(
        build_info=(
            "FFMPEG: YES (prebuilt binaries)\n"
            "  avcodec: YES (61.19.100)\n"
            "  avformat: YES (61.7.100)\n"
            "  avutil: YES (59.39.100)\n"
        ),
        advisory_rules=(
            {
                "id": "UNCITED-RULE",
                "component": "avcodec",
                "affectedBefore": "62.0.0",
            },
        ),
    )

    assert status["passed"] is True
    assert status["blocking"] is False
    assert status["vulnerable"] is None
    assert status["provenance"]["invalidRules"]


def test_cpython_floors_cover_every_supported_release_line():
    """3.11 and 3.12 were previously unguarded by the self-test."""
    assert sc.CPYTHON_SECURITY_FLOORS == {
        (3, 11): (3, 11, 16),
        (3, 12): (3, 12, 14),
        (3, 13): (3, 13, 15),
        (3, 14): (3, 14, 7),
    }
    for advisory_id in (
        "CVE-2026-11940",
        "CVE-2026-2297",
        "CVE-2026-4224",
        "CVE-2026-3644",
        "CVE-2026-6100",
    ):
        assert advisory_id in sc.CPYTHON_SECURITY_ADVISORY_IDS


def test_cpython_advisories_are_per_line_not_a_flat_tuple():
    """A below-floor 3.14 build must not cite CVEs already fixed in 3.14.4."""
    # 3.14.6 is below the 3.14.7 tarfile floor but already has the April
    # three (3.14.4) and the decompressor UAF (3.14.5).
    status = sc.cpython_security_status((3, 14, 6))
    assert status["safe"] is False
    assert status["advisories"] == ["CVE-2026-11940"]
    assert "CVE-2026-2297" not in status["advisories"]
    # 3.11.15 is below 3.11.16, which is the shared fix for every listed CVE.
    old_line = sc.cpython_security_status((3, 11, 15))
    assert old_line["safe"] is False
    for advisory_id in (
        "CVE-2026-11940",
        "CVE-2026-2297",
        "CVE-2026-4224",
        "CVE-2026-3644",
        "CVE-2026-6100",
    ):
        assert advisory_id in old_line["advisories"]


def test_cpython_status_flags_one_stale_and_one_current_per_line():
    for line, floor in sc.CPYTHON_SECURITY_FLOORS.items():
        stale = (floor[0], floor[1], floor[2] - 1)
        status = sc.cpython_security_status(stale)
        assert status["safe"] is False, stale
        assert status["classified"] is True, stale
        assert status["floor"] == sc.format_cpython_version(floor)
        assert "CVE-2026-11940" in status["advisories"], stale
        # The tarfile identifier is why the floor sits where it does.
        assert "CVE-2026-11940" in status["reason"], stale

        current = sc.cpython_security_status(floor)
        assert current["safe"] is True, floor
        assert current["advisories"] == []
        newer = sc.cpython_security_status((floor[0], floor[1], floor[2] + 5))
        assert newer["safe"] is True, floor
        assert line == (floor[0], floor[1])


def test_cpython_status_handles_lines_outside_the_reviewed_table():
    # Newer than the table: postdates the advisories, so no false alarm.
    future = sc.cpython_security_status((3, 15, 0))
    assert future["safe"] is True
    assert future["classified"] is False
    assert "not explicitly classified" in future["reason"]

    # Older than the project's supported floor.
    ancient = sc.cpython_security_status((3, 10, 21))
    assert ancient["safe"] is False
    assert ancient["classified"] is False
    assert "below" in ancient["reason"]


def test_cpython_status_defaults_to_the_running_interpreter():
    import sys

    status = sc.cpython_security_status()
    assert status["version"] == sc.format_cpython_version(
        tuple(sys.version_info[:3])
    )
