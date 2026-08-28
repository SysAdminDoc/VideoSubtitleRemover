"""RM-336: a declared floor for the client that downloads models.

The opt-in model fetch imported `huggingface_hub` with no version declared
anywhere, in a project that pins and cites a floor for every other
security-relevant dependency. That client writes files to disk under names
the remote supplies, and it shipped two Windows-relevant fixes this year.

Separately, the dependency manifest recorded when it was last reviewed and
nothing ever looked at the date.
"""

from __future__ import annotations

import datetime
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend.dependency_caps import (
    HUGGINGFACE_HUB_ADVISORY_IDS,
    HUGGINGFACE_HUB_MINIMUM_VERSION,
    huggingface_hub_floor_problem,
)
from backend.dependency_profiles import (
    DEPENDENCY_REVIEW_MAX_AGE_DAYS,
    review_age,
)
from backend.model_file_format import (
    FORMAT_ONNX,
    FORMAT_SAFETENSORS,
    FORMAT_TORCH_ZIP,
    FORMAT_UNKNOWN,
    describe_model_file,
    identify_model_file,
)

ROOT = Path(__file__).resolve().parent.parent


def _safetensors_bytes() -> bytes:
    header = json.dumps(
        {"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    ).encode("utf-8")
    return struct.pack("<Q", len(header)) + header + b"\x00\x00\x00\x00"


class HubFloorTests(unittest.TestCase):
    def test_the_floor_is_declared_with_its_advisories(self):
        self.assertEqual(HUGGINGFACE_HUB_MINIMUM_VERSION, "1.29.0")
        self.assertIn("CVE-2026-15717", HUGGINGFACE_HUB_ADVISORY_IDS)

    def test_the_floor_is_tracked_beside_the_other_pinned_packages(self):
        from backend.dependency_caps import TRACKED_PACKAGES

        tracked = {name: minimum for name, minimum, _max in TRACKED_PACKAGES}
        self.assertEqual(
            tracked.get("huggingface-hub"), HUGGINGFACE_HUB_MINIMUM_VERSION)

    def test_an_older_client_is_refused_and_the_message_names_the_floor(self):
        with mock.patch("importlib.metadata.version", return_value="1.25.0"):
            problem = huggingface_hub_floor_problem()
        self.assertIn("1.25.0", problem)
        self.assertIn(HUGGINGFACE_HUB_MINIMUM_VERSION, problem)
        for advisory in HUGGINGFACE_HUB_ADVISORY_IDS:
            self.assertIn(advisory, problem)

    def test_a_current_client_is_accepted(self):
        for installed in ("1.29.0", "1.30.1", "2.0.0"):
            with self.subTest(installed=installed):
                with mock.patch("importlib.metadata.version",
                                return_value=installed):
                    self.assertEqual(huggingface_hub_floor_problem(), "")

    def test_an_absent_client_is_not_an_error(self):
        """Nothing to refuse when the optional package is not installed."""
        from importlib.metadata import PackageNotFoundError

        with mock.patch("importlib.metadata.version",
                        side_effect=PackageNotFoundError("huggingface-hub")):
            self.assertEqual(huggingface_hub_floor_problem(), "")

    def test_the_auto_fetch_consults_the_floor(self):
        import ast

        source = (ROOT / "backend" / "inpainters_diffusion.py").read_text(
            encoding="utf-8")
        self.assertIn("huggingface_hub_floor_problem", source)
        tree = ast.parse(source)
        # The check has to sit between importing the client and calling it.
        found = False
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "huggingface_hub_floor_problem"):
                found = True
        self.assertTrue(found)

    def test_the_requirements_file_explains_the_floor(self):
        text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("huggingface-hub>=1.29.0", text)
        self.assertIn("CVE-2026-15717", text)


class ModelFileFormatTests(unittest.TestCase):
    """A filename is metadata the remote chose; the bytes are not."""

    def test_a_file_is_identified_from_its_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            cases = {
                "a.safetensors": (_safetensors_bytes(), FORMAT_SAFETENSORS),
                "b.onnx": (b"\x08\x07rest of the proto", FORMAT_ONNX),
                "c.pt": (b"PK\x03\x04zip", FORMAT_TORCH_ZIP),
            }
            for name, (payload, expected) in cases.items():
                with self.subTest(name=name):
                    path = folder / name
                    path.write_bytes(payload)
                    self.assertEqual(identify_model_file(path), expected)

    def test_a_name_that_lies_does_not_change_the_verdict(self):
        """The exact defect huggingface_hub 1.29.0 fixed, generalised."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "totally.pt"
            path.write_bytes(_safetensors_bytes())
            self.assertEqual(identify_model_file(path), FORMAT_SAFETENSORS)
            described = describe_model_file(path)
            self.assertFalse(described["suffixMatchesBytes"])
            self.assertFalse(described["executesOnLoad"])

    def test_a_file_named_only_an_extension_is_still_read_correctly(self):
        """`Path.suffix` is empty for a name that is nothing but one."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / ".safetensors"
            path.write_bytes(b"PK\x03\x04this is really a torch archive")
            self.assertEqual(path.suffix, "")
            self.assertEqual(identify_model_file(path), FORMAT_TORCH_ZIP)
            self.assertTrue(describe_model_file(path)["executesOnLoad"])

    def test_an_unidentifiable_file_is_unknown_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "junk.safetensors"
            path.write_bytes(b"hello world, not a model at all")
            self.assertEqual(identify_model_file(path), FORMAT_UNKNOWN)
            self.assertFalse(describe_model_file(path)["identified"])

    def test_a_missing_or_empty_file_is_unknown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            self.assertEqual(
                identify_model_file(folder / "absent.pt"), FORMAT_UNKNOWN)
            empty = folder / "empty.safetensors"
            empty.write_bytes(b"")
            self.assertEqual(identify_model_file(empty), FORMAT_UNKNOWN)

    def test_a_declared_header_longer_than_the_file_is_refused(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lying_header.safetensors"
            path.write_bytes(struct.pack("<Q", 10 ** 6) + b"{}")
            self.assertEqual(identify_model_file(path), FORMAT_UNKNOWN)

    def test_formats_that_execute_on_load_are_flagged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            torch_file = folder / "weights.bin"
            torch_file.write_bytes(b"PK\x03\x04")
            safe_file = folder / "weights.safetensors"
            safe_file.write_bytes(_safetensors_bytes())
            self.assertTrue(describe_model_file(torch_file)["executesOnLoad"])
            self.assertFalse(describe_model_file(safe_file)["executesOnLoad"])


class ReviewAgeTests(unittest.TestCase):
    def test_a_fresh_review_is_not_stale(self):
        today = datetime.date(2026, 8, 28)
        result = review_age("2026-08-27", today=today)
        self.assertEqual(result["days"], 1)
        self.assertFalse(result["stale"])
        self.assertEqual(result["warning"], "")

    def test_an_old_review_warns_and_says_what_to_do(self):
        today = datetime.date(2026, 8, 28)
        result = review_age("2026-01-01", today=today)
        self.assertTrue(result["stale"])
        self.assertIn("reviewedAt", result["warning"])
        self.assertGreater(result["days"], DEPENDENCY_REVIEW_MAX_AGE_DAYS)

    def test_the_boundary_is_inclusive_of_the_interval(self):
        today = datetime.date(2026, 8, 28)
        exactly = today - datetime.timedelta(
            days=DEPENDENCY_REVIEW_MAX_AGE_DAYS)
        self.assertFalse(review_age(exactly.isoformat(), today=today)["stale"])
        one_more = exactly - datetime.timedelta(days=1)
        self.assertTrue(review_age(one_more.isoformat(), today=today)["stale"])

    def test_an_unparseable_date_is_stale_rather_than_ignored(self):
        for value in ("", None, "soon", "2026-13-45"):
            with self.subTest(value=value):
                result = review_age(value)
                self.assertTrue(result["stale"])
                self.assertTrue(result["warning"])

    def test_the_status_payload_carries_the_age(self):
        from backend.dependency_profiles import (
            collect_dependency_profile_status,
        )

        status = collect_dependency_profile_status(profile="cpu")
        self.assertIn("reviewAge", status)
        self.assertEqual(
            status["reviewAge"]["reviewedAt"], status["reviewedAt"])


if __name__ == "__main__":
    unittest.main()
