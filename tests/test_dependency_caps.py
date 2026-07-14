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
            preload_status={
                "needed": True,
                "attempted": True,
                "available": True,
                "succeeded": True,
                "callCount": 1,
                "lastProviders": ["CUDAExecutionProvider"],
            },
        )

        self.assertEqual(status["schema"], "vsr.onnxruntime_providers.v1")
        self.assertEqual(status["cuda"]["packageChannel"], "cuda12-pypi-stable")
        self.assertTrue(status["cuda"]["providerAvailable"])
        self.assertTrue(status["cuda"]["preloadStatus"]["succeeded"])
        self.assertTrue(status["directml"]["providerAvailable"])
        self.assertEqual(status["warnings"], [])

    def test_onnxruntime_provider_status_reports_failed_cuda_preload(self):
        status = dependency_caps.collect_onnxruntime_provider_status(
            package_versions={"onnxruntime-gpu": "1.21.0"},
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            runtime_version="1.21.0",
            preload_dlls_available=True,
            preload_status={
                "needed": True,
                "attempted": True,
                "available": True,
                "succeeded": False,
                "error": "missing cudnn64.dll",
                "lastProviders": ["CUDAExecutionProvider"],
            },
        )

        self.assertEqual(
            status["cuda"]["preloadStatus"]["error"],
            "missing cudnn64.dll",
        )
        self.assertEqual(status["warnings"][0]["id"], "ORT-CUDA-PRELOAD-FAILED")

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
        # Provider-migration warnings remain allowed/non-blocking.
        migration = [a for a in advisories if a["id"].startswith("ORT-CUDA")]
        self.assertTrue(all(item["allowed"] for item in migration))

    def test_onnxruntime_security_floor_blocks_old_runtime(self):
        status = dependency_caps.collect_onnxruntime_provider_status(
            package_versions={"onnxruntime-gpu": "1.21.0"},
            providers=["CPUExecutionProvider"],
            runtime_version="1.21.0",
            preload_dlls_available=True,
        )
        advisories = dependency_caps.onnxruntime_release_advisories(status)
        floor = [a for a in advisories if a["id"] == "ORT-PARSER-OOB-1.25.0"]
        self.assertTrue(floor, "expected a security-floor advisory below 1.25.0")
        self.assertTrue(floor[0]["blocking"])
        self.assertEqual(floor[0]["severity"], "high")
        self.assertFalse(floor[0]["allowed"])

    def test_onnxruntime_security_floor_passes_current_runtime(self):
        status = dependency_caps.collect_onnxruntime_provider_status(
            package_versions={"onnxruntime-gpu": "1.25.0"},
            providers=["CUDAExecutionProvider"],
            runtime_version="1.25.0",
            preload_dlls_available=True,
        )
        advisories = dependency_caps.onnxruntime_release_advisories(status)
        self.assertFalse(
            [a for a in advisories if a["id"] == "ORT-PARSER-OOB-1.25.0"]
        )

    def test_opencv_wheel_status_reports_single_owner(self):
        status = dependency_caps.collect_opencv_wheel_status(
            package_versions={"opencv-python": "4.12.0.88"},
            imported_version="4.12.0",
            imported_file="C:/repo/venv/Lib/site-packages/cv2/__init__.py",
            dnn_available=True,
        )

        self.assertEqual(status["schema"], "vsr.opencv_wheels.v1")
        self.assertFalse(status["conflict"])
        self.assertEqual(status["installedDistributions"], ["opencv-python"])
        self.assertEqual(status["imported"]["owner"], "opencv-python")
        self.assertEqual(status["imported"]["version"], "4.12.0")
        self.assertTrue(status["imported"]["dnnAvailable"])
        self.assertEqual(status["warnings"], [])

    def test_opencv_wheel_status_flags_conflicting_wheels(self):
        status = dependency_caps.collect_opencv_wheel_status(
            package_versions={
                "opencv-python": "4.12.0.88",
                "opencv-contrib-python-headless": "4.12.0.88",
            },
            imported_version="4.12.0",
            imported_file="C:/repo/venv/Lib/site-packages/cv2/__init__.py",
        )

        self.assertTrue(status["conflict"])
        self.assertEqual(status["imported"]["owner"], "ambiguous")
        self.assertEqual(
            status["warnings"][0]["id"],
            "OPENCV-WHEEL-CONFLICT",
        )
        commands = status["remediation"]["commands"]
        self.assertEqual(len(commands), 2)
        self.assertIn("opencv-contrib-python-headless", commands[0])
        self.assertEqual(commands[1], 'python -m pip install "opencv-python>=4.12.0"')


class DriftReportTests(unittest.TestCase):
    def test_drift_report_schema_and_summary(self):
        versions = {
            "numpy": "1.26.4",
            "opencv-python": "4.12.0",
            "pillow": "12.2.0",
            "rapidocr": "3.9.0",
        }
        report = dependency_caps.collect_dependency_drift_report(
            package_versions=versions,
        )
        self.assertEqual(report["schema"], dependency_caps.DRIFT_REPORT_SCHEMA)
        self.assertGreater(report["summary"]["tracked"], 0)
        self.assertEqual(report["summary"]["ok"], 4)
        self.assertGreater(report["summary"]["notInstalled"], 0)

    def test_drift_report_detects_below_minimum(self):
        versions = {"numpy": "1.20.0"}
        report = dependency_caps.collect_dependency_drift_report(
            package_versions=versions,
        )
        numpy_item = next(
            i for i in report["packages"] if i["package"] == "numpy"
        )
        self.assertEqual(numpy_item["status"], "below-minimum")

    def test_drift_report_detects_above_maximum(self):
        versions = {"rapidocr": "5.0.0"}
        report = dependency_caps.collect_dependency_drift_report(
            package_versions=versions,
        )
        item = next(
            i for i in report["packages"] if i["package"] == "rapidocr"
        )
        self.assertEqual(item["status"], "above-maximum")

    def test_drift_report_format_is_human_readable(self):
        versions = {"numpy": "1.26.4", "pillow": "12.2.0"}
        report = dependency_caps.collect_dependency_drift_report(
            package_versions=versions,
        )
        text = dependency_caps.format_drift_report(report)
        self.assertIn("numpy", text)
        self.assertIn("1.26.4", text)
        self.assertIn("Tracked:", text)

    def test_drift_report_includes_blocked_exceptions(self):
        report = dependency_caps.collect_dependency_drift_report(
            package_versions={},
        )
        self.assertGreater(len(report["blockedExceptions"]), 0)
        self.assertEqual(
            report["blockedExceptions"][0]["package"],
            "opencv-python",
        )


if __name__ == "__main__":
    unittest.main()
