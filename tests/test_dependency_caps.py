import importlib.metadata as metadata
import unittest
from pathlib import Path
from unittest import mock

from backend import dependency_caps


ROOT = Path(__file__).resolve().parents[1]


class DependencyCapTests(unittest.TestCase):
    def test_requirements_and_setup_use_ocr_major_caps(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        setup = (ROOT / "setup.py").read_text(encoding="utf-8")

        for expected in (
            "rapidocr>=2.0.0,<4.0.0",
            "paddleocr>=3.0.0,<4.0.0",
        ):
            self.assertIn(expected, requirements)
            self.assertIn(expected, setup)

        self.assertIn("rapidocr-onnxruntime>=1.4.0,<2.0.0", requirements)
        self.assertIn("opencv-python>=4.12.0", requirements)
        self.assertNotIn("opencv-python>=4.12.0,<", requirements)
        self.assertNotIn("numpy>=1.21.0,<", requirements)

    def test_checker_passes_missing_or_in_range_engines(self):
        versions = {
            "rapidocr": "2.1.0",
            "paddleocr": "3.7.0",
        }

        def fake_version(package):
            if package not in versions:
                raise metadata.PackageNotFoundError(package)
            return versions[package]

        with mock.patch.object(
            dependency_caps.metadata,
            "version",
            side_effect=fake_version,
        ):
            self.assertEqual(dependency_caps.check_ocr_dependency_caps(), [])

    def test_checker_reports_major_version_overrun(self):
        versions = {
            "rapidocr": "4.0.0",
            "rapidocr-onnxruntime": "2.0.0",
            "paddleocr": "4.0.0",
        }

        def fake_version(package):
            return versions[package]

        with mock.patch.object(
            dependency_caps.metadata,
            "version",
            side_effect=fake_version,
        ):
            problems = dependency_caps.check_ocr_dependency_caps()

        self.assertEqual(len(problems), 3)
        self.assertIn("rapidocr==4.0.0", problems[0])
        self.assertIn("rapidocr-onnxruntime==2.0.0", problems[1])
        self.assertIn("paddleocr==4.0.0", problems[2])


if __name__ == "__main__":
    unittest.main()
