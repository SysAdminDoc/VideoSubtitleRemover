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

    def test_rapidocr_engine_status_prefers_openvino_when_available(self):
        status = dependency_caps.collect_rapidocr_engine_status(
            package_versions={
                "rapidocr": "3.9.0",
                "openvino": "2025.4.0",
            },
            openvino_devices=["CPU", "GPU"],
        )

        self.assertEqual(status["schema"], "vsr.rapidocr_engines.v1")
        self.assertEqual(status["preferredEngine"], "openvino")
        self.assertEqual(status["preferredProvider"], "OpenVINO CPU/GPU")
        self.assertTrue(status["engines"]["openvino"]["available"])
        self.assertEqual(status["warnings"], [])

    def test_rapidocr_engine_status_keeps_onnxruntime_for_legacy_package(self):
        status = dependency_caps.collect_rapidocr_engine_status(
            package_versions={
                "rapidocr-onnxruntime": "1.4.1",
                "openvino": "2025.4.0",
            },
            openvino_devices=["CPU"],
        )

        self.assertEqual(status["preferredEngine"], "onnxruntime")
        self.assertFalse(status["engines"]["openvino"]["available"])
        self.assertEqual(
            status["warnings"][0]["id"],
            "RAPIDOCR-OPENVINO-UNSUPPORTED-PACKAGE",
        )

    def test_onnxruntime_provider_status_distinguishes_cuda_and_directml(self):
        status = dependency_caps.collect_onnxruntime_provider_status(
            package_versions={
                "onnxruntime-gpu": "1.21.0",
                "onnxruntime-directml": "1.22.0",
            },
            providers=[
                "CUDAExecutionProvider",
                "DmlExecutionProvider",
                "CPUExecutionProvider",
            ],
            runtime_version="1.21.0",
            preload_dlls_available=True,
        )

        self.assertEqual(status["schema"], "vsr.onnxruntime_providers.v1")
        self.assertEqual(status["cuda"]["packageChannel"], "cuda12-pypi-stable")
        self.assertTrue(status["cuda"]["providerAvailable"])
        self.assertTrue(status["directml"]["providerAvailable"])
        self.assertEqual(status["warnings"], [])

    def test_onnxruntime_release_advisory_flags_legacy_cuda_package(self):
        status = dependency_caps.collect_onnxruntime_provider_status(
            package_versions={"onnxruntime-gpu": "1.17.3"},
            providers=["CPUExecutionProvider"],
            runtime_version="1.17.3",
            preload_dlls_available=False,
        )
        advisories = dependency_caps.onnxruntime_release_advisories(status)

        ids = {item["id"] for item in advisories}
        self.assertIn("ORT-CUDA-LEGACY-PACKAGE", ids)
        self.assertIn("ORT-CUDA-PRELOAD-MISSING", ids)
        self.assertTrue(all(item["allowed"] for item in advisories))


if __name__ == "__main__":
    unittest.main()
