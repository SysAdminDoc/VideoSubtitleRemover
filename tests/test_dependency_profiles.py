import json
from pathlib import Path
import tempfile
import types
import unittest

from backend import dependency_caps
from backend import dependency_profiles


ROOT = Path(__file__).resolve().parents[1]


class DependencyProfileTests(unittest.TestCase):
    def test_reviewed_profiles_are_generated_and_current(self):
        manifest = dependency_profiles.load_profile_manifest()
        self.assertEqual(
            set(manifest["profiles"]),
            {"cpu", "nvidia", "directml"},
        )
        self.assertEqual(manifest["python"], ">=3.11,<3.15")
        self.assertEqual(dependency_profiles.profile_diffs(), {})
        for name in dependency_profiles.SUPPORTED_PROFILES:
            text = dependency_profiles.profile_constraint_path(name).read_text(
                encoding="utf-8")
            self.assertIn("Manifest-SHA256:", text)
            self.assertIn("numpy==2.4.6", text)
            self.assertIn("Pillow==12.3.0", text)

        self.assertIn("onnxruntime==1.29.0", (
            dependency_profiles.profile_constraint_path("cpu").read_text(
                encoding="utf-8")))
        self.assertIn("onnxruntime-gpu==1.26.0", (
            dependency_profiles.profile_constraint_path("nvidia").read_text(
                encoding="utf-8")))
        self.assertIn("onnxruntime-directml==1.24.4", (
            dependency_profiles.profile_constraint_path("directml").read_text(
                encoding="utf-8")))
        expected_torch = {
            "cpu": ("2.13.0", "0.28.0"),
            "nvidia": ("2.11.0", "0.26.0"),
            "directml": ("2.13.0", "0.28.0"),
        }
        for name, (torch, torchvision) in expected_torch.items():
            text = dependency_profiles.profile_constraint_path(name).read_text(
                encoding="utf-8")
            self.assertIn(f"torch=={torch}", text)
            self.assertIn(f"torchvision=={torchvision}", text)
        for name in dependency_profiles.SUPPORTED_PROFILES:
            self.assertIn(
                f"protobuf=={dependency_caps.PROTOBUF_TESTED_VERSION}",
                dependency_profiles.profile_constraint_path(name).read_text(
                    encoding="utf-8"),
            )

    def test_nvidia_lock_is_installable_under_the_setup_cuda12_constraint(self):
        """RM-140: setup.py and the generated lock must agree on CUDA 12."""
        setup = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn(
            f'ONNXRUNTIME_GPU_MIN = "{dependency_caps.ONNXRUNTIME_GPU_RECOMMENDED_MIN}"',
            setup,
        )
        self.assertIn(
            f'ONNXRUNTIME_GPU_MAX_EXCLUSIVE = "'
            f'{dependency_caps.ONNXRUNTIME_GPU_MAX_EXCLUSIVE}"',
            setup,
        )
        lock = dependency_profiles.profile_constraint_path("nvidia").read_text(
            encoding="utf-8")
        pinned = [
            line.split("==", 1)[1].strip()
            for line in lock.splitlines()
            if line.startswith("onnxruntime-gpu==")
        ]
        self.assertEqual(len(pinned), 1)
        lane = dependency_caps.provider_lane("cuda12")
        self.assertTrue(dependency_caps.version_in_lane(pinned[0], lane))

    def test_out_of_range_exact_lock_is_rejected(self):
        manifest = json.loads(
            dependency_profiles.MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest["profiles"]["nvidia"]["constraints"] = [
            item.replace("onnxruntime-gpu==1.26.0", "onnxruntime-gpu==1.27.0")
            for item in manifest["profiles"]["nvidia"]["constraints"]
        ]
        problems = dependency_profiles.constraint_range_problems(manifest)
        self.assertTrue(any("onnxruntime-gpu==1.27.0" in item for item in problems))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dependency_profiles.json"
            path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                dependency_profiles.load_profile_manifest(path)

    def test_low_protobuf_pin_is_rejected(self):
        manifest = json.loads(
            dependency_profiles.MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest["commonConstraints"] = [
            item.replace(
                f"protobuf=={dependency_caps.PROTOBUF_TESTED_VERSION}",
                "protobuf==6.33.4")
            for item in manifest["commonConstraints"]
        ]
        problems = dependency_profiles.constraint_range_problems(manifest)
        self.assertTrue(any("protobuf==6.33.4" in item for item in problems))

    def test_provider_lanes_report_separate_tested_and_security_state(self):
        status = dependency_caps.collect_provider_lane_status({
            "onnxruntime": "1.28.0",
            "onnxruntime-gpu": "1.26.0",
            "onnxruntime-directml": "1.24.4",
            "protobuf": "6.33.6",
        })
        lanes = {item["key"]: item for item in status["lanes"]}
        self.assertEqual(
            set(lanes), {
                "cpu", "cuda12", "cuda13", "tensorrt-rtx", "directml",
            })
        self.assertTrue(lanes["cpu"]["tested"])
        self.assertTrue(lanes["cuda12"]["tested"])
        self.assertFalse(lanes["cuda13"]["tested"])
        self.assertFalse(lanes["tensorrt-rtx"]["tested"])
        self.assertEqual(
            lanes["tensorrt-rtx"]["provider"],
            "NvTensorRTRTXExecutionProvider",
        )
        self.assertEqual(lanes["tensorrt-rtx"]["profile"], "")
        self.assertEqual(lanes["tensorrt-rtx"]["securityState"], "ok")
        self.assertTrue(lanes["directml"]["tested"])
        self.assertEqual(lanes["cpu"]["securityState"], "ok")
        self.assertEqual(lanes["cuda12"]["securityState"], "ok")
        # The CUDA 12 pin is outside the CUDA 13 lane and vice versa.
        self.assertEqual(lanes["cuda13"]["securityState"], "below-floor")
        self.assertEqual(lanes["cuda13"]["profile"], "")
        self.assertIs(status["protobuf"]["satisfied"], True)

    def test_protobuf_below_floor_is_a_blocking_advisory(self):
        advisory = dependency_caps.protobuf_release_advisory(
            {"protobuf": "6.33.4"})
        self.assertIsNotNone(advisory)
        self.assertTrue(advisory["blocking"])
        self.assertEqual(advisory["fixedIn"], ">=6.33.5")
        self.assertIsNone(
            dependency_caps.protobuf_release_advisory({"protobuf": "6.33.6"}))

    def test_profile_smoke_fails_when_the_provider_falls_back(self):
        class _Session:
            def __init__(self, path, providers=None):
                self.requested = list(providers or [])

            def get_providers(self):
                return ["CPUExecutionProvider"]

            def run(self, outputs, feeds):
                return [feeds["x"]]

        fake_ort = types.SimpleNamespace(
            get_available_providers=lambda: [
                "CUDAExecutionProvider", "CPUExecutionProvider"],
            InferenceSession=_Session,
        )
        result = dependency_profiles.run_profile_provider_smoke(
            "nvidia", ort_module=fake_ort)
        self.assertTrue(result["ran"])
        self.assertFalse(result["passed"])
        self.assertTrue(result["fellBack"])
        self.assertIn("CUDAExecutionProvider", result["error"])

    def test_profile_smoke_passes_on_the_claimed_provider(self):
        class _Session:
            def __init__(self, path, providers=None):
                self.requested = list(providers or [])

            def get_providers(self):
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]

            def run(self, outputs, feeds):
                return [feeds["x"]]

        fake_ort = types.SimpleNamespace(
            get_available_providers=lambda: [
                "CUDAExecutionProvider", "CPUExecutionProvider"],
            InferenceSession=_Session,
        )
        result = dependency_profiles.run_profile_provider_smoke(
            "nvidia", ort_module=fake_ort)
        self.assertTrue(result["passed"])
        self.assertFalse(result["fellBack"])
        self.assertEqual(result["error"], "")

    def test_profile_smoke_fails_when_provider_is_absent(self):
        fake_ort = types.SimpleNamespace(
            get_available_providers=lambda: ["CPUExecutionProvider"],
            InferenceSession=lambda *a, **k: None,
        )
        result = dependency_profiles.run_profile_provider_smoke(
            "directml", ort_module=fake_ort)
        self.assertFalse(result["passed"])
        self.assertTrue(result["fellBack"])

    def test_real_cpu_provider_smoke_runs_one_inference_session(self):
        result = dependency_profiles.run_profile_provider_smoke("cpu")
        if not result["ran"]:
            self.skipTest(result["error"] or "onnxruntime unavailable")
        self.assertTrue(result["passed"], result["error"])
        self.assertEqual(result["activeProviders"][0], "CPUExecutionProvider")

    def test_hardware_selection_covers_all_supported_profiles(self):
        self.assertEqual(dependency_profiles.select_profile({}), "cpu")
        self.assertEqual(
            dependency_profiles.select_profile({"nvidia": True}),
            "nvidia",
        )
        self.assertEqual(
            dependency_profiles.select_profile({
                "nvidia": True,
                "cuda_disabled_by_python": True,
            }),
            "cpu",
        )
        self.assertEqual(
            dependency_profiles.select_profile({"amd": True}),
            "directml",
        )
        self.assertEqual(
            dependency_profiles.select_profile({"intel": True}),
            "directml",
        )

    def test_release_status_records_profile_hashes_and_exceptions(self):
        status = dependency_profiles.collect_dependency_profile_status(
            profile="nvidia",
            package_versions={"onnxruntime-gpu": "1.27.0"},
        )
        self.assertTrue(status["valid"])
        self.assertEqual(status["profile"], "nvidia")
        self.assertEqual(len(status["manifestSha256"]), 64)
        self.assertEqual(len(status["constraintSha256"]), 64)
        self.assertTrue(status["reviewedArtifactHashes"])
        self.assertTrue(status["intentionalExceptions"])

    def test_update_workflow_emits_diff_then_converges(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = json.loads(
                dependency_profiles.MANIFEST_PATH.read_text(encoding="utf-8"))
            manifest = root / "dependency_profiles.json"
            manifest.write_text(
                json.dumps(source, indent=2) + "\n",
                encoding="utf-8",
            )
            profile_dir = root / "profiles"
            initial = dependency_profiles.update_profiles(
                manifest_path=manifest,
                profile_dir=profile_dir,
            )
            self.assertEqual(
                set(initial),
                set(dependency_profiles.SUPPORTED_PROFILES),
            )
            self.assertEqual(dependency_profiles.profile_diffs(
                manifest_path=manifest,
                profile_dir=profile_dir,
            ), {})
            cpu = profile_dir / "cpu.txt"
            cpu.write_text("stale\n", encoding="utf-8")
            drift = dependency_profiles.update_profiles(
                manifest_path=manifest,
                profile_dir=profile_dir,
                write=False,
            )
            self.assertIn("cpu", drift)
            self.assertIn("-stale", drift["cpu"])

    def test_setup_docker_and_release_build_share_profile_contract(self):
        setup = (ROOT / "setup.py").read_text(encoding="utf-8")
        docker = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        build = (ROOT / "build_exe.bat").read_text(encoding="utf-8")
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

        self.assertIn("_profile_constraint_args", setup)
        self.assertIn('"--profile"', setup)
        main_body = setup.split("def main(argv=None):", 1)[1]
        self.assertNotIn("install_paddlepaddle(gpu_info)", main_body)
        self.assertIn("--constraint dependency_profiles/cpu.txt", docker)
        self.assertIn("VSR_DEPENDENCY_PROFILE=cpu", docker)
        self.assertIn("backend.dependency_profiles check", docker)
        self.assertIn("backend.dependency_profiles check", build)
        self.assertIn("dependency_profiles/{cpu,nvidia,directml}.txt", requirements)
        active_requirements = [
            line.strip() for line in requirements.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        for conflicting_optional in (
            "paddleocr", "easyocr", "simple-lama-inpainting",
        ):
            self.assertFalse(any(
                line.lower().startswith(conflicting_optional)
                for line in active_requirements
            ))


if __name__ == "__main__":
    unittest.main()
