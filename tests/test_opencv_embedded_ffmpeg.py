"""RM-320: classify the FFmpeg that ships inside the OpenCV wheel.

The project enforces a 9.0.1 floor on the external FFmpeg binary while the
OpenCV wheel carries its own FFmpeg 7.1 and `cv2.VideoCapture` is still
reachable, so untrusted media could reach a decoder predating the whole
2026 CVE batch. The advisory mapping was left empty "until an advisory
supplies that evidence"; the ABI-to-release mapping supplies it.
"""

from __future__ import annotations

import unittest

from backend.security_checks import (
    FFMPEG_SECURITY_ADVISORY_IDS,
    OPENCV_FFMPEG_ACKNOWLEDGED_RELEASE,
    OPENCV_FFMPEG_ACKNOWLEDGEMENT,
    OPENCV_FFMPEG_ADVISORY_RULES,
    ffmpeg_security_floor_str,
    opencv_ffmpeg_release_from_abi,
    opencv_ffmpeg_status,
)


def _libraries(avutil, avcodec, avformat) -> dict:
    return {
        name: {
            "available": True,
            "version": ".".join(str(part) for part in parts),
            "versionTuple": list(parts),
        }
        for name, parts in (
            ("avutil", avutil),
            ("avcodec", avcodec),
            ("avformat", avformat),
        )
    }


def _build_info(avutil, avcodec, avformat) -> str:
    return (
        "General configuration for OpenCV 5.0.0\n"
        "  Video I/O:\n"
        "    FFMPEG:                      YES (prebuilt binaries)\n"
        f"      avcodec:                   YES ({avcodec})\n"
        f"      avformat:                  YES ({avformat})\n"
        f"      avutil:                    YES ({avutil})\n"
    )


class EmbeddedReleaseIdentityTests(unittest.TestCase):
    def test_the_shipping_wheel_abi_identifies_ffmpeg_7_1(self):
        identity = opencv_ffmpeg_release_from_abi(
            _libraries((59, 39, 100), (61, 19, 100), (61, 7, 100)))
        self.assertTrue(identity["identified"])
        self.assertEqual(identity["release"], "7.1")
        self.assertTrue(identity["belowFloor"])

    def test_a_floor_branch_abi_is_not_below_the_floor(self):
        identity = opencv_ffmpeg_release_from_abi(
            _libraries((61, 1, 101), (63, 1, 101), (63, 1, 101)))
        self.assertEqual(identity["release"], "9.0.1")
        self.assertFalse(identity["belowFloor"])

    def test_an_inferred_branch_is_named_but_marked_unmeasured(self):
        """6.x is covered by the branch table, but was never measured here."""
        identity = opencv_ffmpeg_release_from_abi(
            _libraries((58, 12, 100), (60, 31, 102), (60, 16, 100)))
        self.assertTrue(identity["identified"])
        self.assertEqual(identity["branch"], "6.x")
        self.assertFalse(identity["branchMeasured"])
        # No exact release: the triple names a series, not a point release.
        self.assertEqual(identity["release"], "")
        self.assertTrue(identity["belowFloor"])

    def test_the_two_measured_anchors_say_so(self):
        for libraries, branch in (
            (_libraries((59, 39, 100), (61, 19, 100), (61, 7, 100)), "7.x"),
            (_libraries((61, 1, 101), (63, 1, 101), (63, 1, 101)), "9.x"),
        ):
            with self.subTest(branch=branch):
                identity = opencv_ffmpeg_release_from_abi(libraries)
                self.assertEqual(identity["branch"], branch)
                self.assertTrue(identity["branchMeasured"])

    def test_an_abi_outside_every_known_series_is_unidentified(self):
        identity = opencv_ffmpeg_release_from_abi(
            _libraries((40, 40, 40), (41, 41, 41), (42, 42, 42)))
        self.assertFalse(identity["identified"])
        self.assertEqual(identity["release"], "")
        self.assertEqual(identity["branch"], "")
        # Still measurably below the floor even though the series is unnamed.
        self.assertTrue(identity["belowFloor"])

    def test_missing_or_partial_abi_data_identifies_nothing(self):
        for libraries in ({}, None, {"avcodec": {"versionTuple": [61]}}):
            with self.subTest(libraries=libraries):
                identity = opencv_ffmpeg_release_from_abi(libraries)
                self.assertFalse(identity["identified"])
                self.assertIsNone(identity["belowFloor"])


class EmbeddedClassificationTests(unittest.TestCase):
    def test_the_rules_are_no_longer_empty(self):
        self.assertTrue(OPENCV_FFMPEG_ADVISORY_RULES)
        components = {
            rule["component"] for rule in OPENCV_FFMPEG_ADVISORY_RULES}
        self.assertEqual(components, {"avcodec", "avformat"})

    def test_a_below_floor_build_is_classified_and_names_the_release(self):
        status = opencv_ffmpeg_status(
            build_info=_build_info("59.39.100", "61.19.100", "61.7.100"),
            opencv_version="5.0.0",
        )
        self.assertEqual(status["classification"], "vulnerable")
        self.assertTrue(status["vulnerable"])
        self.assertTrue(status["blocking"])
        self.assertEqual(status["embeddedRelease"]["release"], "7.1")
        self.assertEqual(status["securityFloor"], ffmpeg_security_floor_str())
        self.assertIn("7.1", status["warning"])
        self.assertIn(ffmpeg_security_floor_str(), status["warning"])
        # The ABI numbers that produced the verdict are in the verdict.
        self.assertIn("61.19.100", status["warning"])
        self.assertIn("61.7.100", status["warning"])
        named = {
            advisory
            for match in status["advisories"]
            for advisory in match.get("advisories", [])
        }
        self.assertEqual(named, set(FFMPEG_SECURITY_ADVISORY_IDS))

    def test_a_floor_branch_build_is_classified_safe(self):
        status = opencv_ffmpeg_status(
            build_info=_build_info("61.1.101", "63.1.101", "63.1.101"),
            opencv_version="5.0.0",
        )
        self.assertEqual(status["classification"], "safe")
        self.assertFalse(status["vulnerable"])
        self.assertFalse(status["blocking"])
        self.assertTrue(status["passed"])
        self.assertEqual(status["warning"], "")

    def test_the_installed_wheel_is_classified_rather_than_unmapped(self):
        status = opencv_ffmpeg_status()
        self.assertNotEqual(status["classification"], "unmapped")
        self.assertTrue(status["embeddedRelease"]["identified"])


class ShipDecisionTests(unittest.TestCase):
    def test_the_acknowledgement_is_dated_reasoned_and_scoped(self):
        self.assertRegex(
            OPENCV_FFMPEG_ACKNOWLEDGEMENT["recorded"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertGreater(len(OPENCV_FFMPEG_ACKNOWLEDGEMENT["reason"]), 80)
        self.assertTrue(OPENCV_FFMPEG_ACKNOWLEDGEMENT["residualExposure"])
        self.assertTrue(OPENCV_FFMPEG_ACKNOWLEDGEMENT["tracking"])
        self.assertEqual(
            OPENCV_FFMPEG_ACKNOWLEDGEMENT["release"],
            OPENCV_FFMPEG_ACKNOWLEDGED_RELEASE,
        )

    def _errors_for(self, status):
        from backend.release_verification import _validation_errors

        evidence = {"releaseTools": {"opencvFfmpeg": status}}
        return [
            error for error in _validation_errors(evidence)
            if "OpenCV embedded FFmpeg" in error
        ]

    def test_the_acknowledged_release_does_not_block_the_release(self):
        status = opencv_ffmpeg_status(
            build_info=_build_info("59.39.100", "61.19.100", "61.7.100"),
            opencv_version="5.0.0",
        )
        self.assertEqual(self._errors_for(status), [])

    def test_a_different_old_branch_still_blocks(self):
        status = opencv_ffmpeg_status(
            build_info=_build_info("58.12.100", "60.31.102", "60.16.100"),
            opencv_version="5.0.0",
        )
        self.assertTrue(status["blocking"])
        errors = self._errors_for(status)
        self.assertEqual(len(errors), 1)
        self.assertIn("not the acknowledged", errors[0])


if __name__ == "__main__":
    unittest.main()
