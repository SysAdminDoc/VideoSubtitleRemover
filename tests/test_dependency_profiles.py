import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

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

        manifest = dependency_profiles.load_profile_manifest()
        self.assertEqual(
            set(manifest["commonRequiredPackages"]),
            {
                "numpy",
                "protobuf",
                "opencv-python",
                "Pillow",
                "idna",
                "rapidocr",
                "torch",
                "torchvision",
            },
        )
        expected_provider_package = {
            "cpu": "onnxruntime",
            "nvidia": "onnxruntime-gpu",
            "directml": "onnxruntime-directml",
        }
        for name, package in expected_provider_package.items():
            self.assertEqual(manifest["profiles"][name]["requiredPackages"], [package])
            self.assertTrue(manifest["profiles"][name]["capabilities"])
            required = dependency_profiles.profile_required_packages(name)
            self.assertEqual(len(required), 9)
            self.assertTrue(all(item["expectedVersion"] for item in required))

        self.assertIn("onnxruntime==1.29.0", (
            dependency_profiles.profile_constraint_path("cpu").read_text(
                encoding="utf-8")))
        self.assertIn("onnxruntime-gpu==1.29.0", (
            dependency_profiles.profile_constraint_path("nvidia").read_text(
                encoding="utf-8")))
        self.assertIn("onnxruntime-directml==1.24.4", (
            dependency_profiles.profile_constraint_path("directml").read_text(
                encoding="utf-8")))
        # The pins are stated here rather than read back from the manifest,
        # so moving a reviewed version is a deliberate edit in two places.
        # RM-357 moved them from 2.13.0/0.28.0: torchvision 0.29.0 is where
        # CVE-2026-65918 is closed.
        expected_torch = {
            "cpu": (dependency_caps.TORCH_MINIMUM_VERSION,
                    dependency_caps.TORCHVISION_MINIMUM_VERSION),
            "nvidia": (dependency_caps.TORCH_MINIMUM_VERSION,
                       dependency_caps.TORCHVISION_MINIMUM_VERSION),
            "directml": (dependency_caps.TORCH_MINIMUM_VERSION,
                         dependency_caps.TORCHVISION_MINIMUM_VERSION),
        }
        self.assertEqual(dependency_caps.TORCH_MINIMUM_VERSION, "2.14.0")
        self.assertEqual(dependency_caps.TORCHVISION_MINIMUM_VERSION, "0.29.0")
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

    def test_nvidia_lock_is_installable_under_the_setup_cuda_constraint(self):
        """RM-140/RM-319: setup.py and the lock must agree on the NVIDIA lane.

        The lane is CUDA 13 from RM-319 on, because the default
        onnxruntime-gpu wheel has been the CUDA 13 build since 1.27.0 and
        the cu128 torch index cannot reach a torch outside
        GHSA-rrmf-rvhw-rf47.
        """
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
        lane = dependency_caps.provider_lane("cuda13")
        self.assertTrue(dependency_caps.version_in_lane(pinned[0], lane))

    def test_out_of_range_exact_lock_is_rejected(self):
        manifest = json.loads(
            dependency_profiles.MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest["profiles"]["nvidia"]["constraints"] = [
            item.replace("onnxruntime-gpu==1.29.0", "onnxruntime-gpu==1.26.0")
            for item in manifest["profiles"]["nvidia"]["constraints"]
        ]
        problems = dependency_profiles.constraint_range_problems(manifest)
        self.assertTrue(any("onnxruntime-gpu==1.26.0" in item for item in problems))
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
            "onnxruntime-gpu": "1.29.0",
            "onnxruntime-directml": "1.24.4",
            "protobuf": "6.33.6",
        })
        lanes = {item["key"]: item for item in status["lanes"]}
        self.assertEqual(
            set(lanes), {
                "cpu", "cuda12", "cuda13", "tensorrt-rtx", "directml",
            })
        self.assertTrue(lanes["cpu"]["tested"])
        self.assertTrue(lanes["cuda13"]["tested"])
        self.assertFalse(lanes["cuda12"]["tested"])
        self.assertFalse(lanes["tensorrt-rtx"]["tested"])
        self.assertEqual(
            lanes["tensorrt-rtx"]["provider"],
            "NvTensorRTRTXExecutionProvider",
        )
        self.assertEqual(lanes["tensorrt-rtx"]["profile"], "")
        self.assertEqual(lanes["tensorrt-rtx"]["securityState"], "ok")
        self.assertTrue(lanes["directml"]["tested"])
        self.assertEqual(lanes["cpu"]["securityState"], "ok")
        self.assertEqual(lanes["cuda13"]["securityState"], "ok")
        # The installed CUDA 13 wheel sits outside the legacy CUDA 12 lane.
        self.assertEqual(lanes["cuda12"]["securityState"], "outside-lane")
        self.assertEqual(lanes["cuda12"]["profile"], "")
        self.assertEqual(lanes["cuda13"]["profile"], "nvidia")
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

    def test_runtime_verifier_requires_every_exact_profile_package(self):
        required = dependency_profiles.profile_required_packages("cpu")
        versions = {
            item["name"]: item["expectedVersion"]
            for item in required
        }
        statuses, errors = dependency_profiles.verify_required_package_versions(
            "cpu",
            package_versions=versions,
        )
        self.assertEqual(errors, [])
        self.assertTrue(all(item["satisfied"] for item in statuses))

        missing = dict(versions)
        missing.pop("protobuf")
        _statuses, errors = dependency_profiles.verify_required_package_versions(
            "cpu",
            package_versions=missing,
        )
        self.assertTrue(any("protobuf==6.33.6" in item for item in errors))

        mismatched = dict(versions)
        mismatched["opencv-python"] = "5.0.0.92"
        _statuses, errors = dependency_profiles.verify_required_package_versions(
            "cpu",
            package_versions=mismatched,
        )
        self.assertTrue(any("resolved as 5.0.0.92" in item for item in errors))

    def test_runtime_verifier_accepts_reviewed_local_wheel_suffix(self):
        required = dependency_profiles.profile_required_packages("nvidia")
        versions = {
            item["name"]: item["expectedVersion"]
            for item in required
        }
        versions["torch"] += "+cu128"

        _statuses, errors = dependency_profiles.verify_required_package_versions(
            "nvidia",
            package_versions=versions,
        )

        self.assertEqual(errors, [])

    def test_runtime_verifier_rejects_competing_binary_owners(self):
        required = dependency_profiles.profile_required_packages("cpu")
        versions = {
            item["name"]: item["expectedVersion"]
            for item in required
        }
        versions["opencv-python-headless"] = "5.0.0.93"
        versions["onnxruntime-gpu"] = "1.26.0"

        _statuses, errors = dependency_profiles.verify_required_package_versions(
            "cpu",
            package_versions=versions,
        )

        self.assertEqual(
            sum("Conflicting runtime distributions" in item for item in errors),
            2,
        )

    def test_runtime_import_failure_is_classified(self):
        def importer(name):
            if name == "cv2":
                raise OSError("simulated DLL load failure")
            return object()

        result = dependency_profiles.run_profile_import_smoke(
            "cpu",
            importer=importer,
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["failures"][0]["package"], "opencv-python")
        self.assertIn("DLL load failure", result["failures"][0]["error"])

    def test_verify_command_exits_nonzero_for_an_invalid_runtime(self):
        invalid = {
            "schema": dependency_profiles.PROFILE_STATUS_SCHEMA,
            "profile": "cpu",
            "valid": False,
            "errors": ["simulated missing package"],
        }
        with mock.patch.object(
            dependency_profiles,
            "collect_dependency_profile_status",
            return_value=invalid,
        ) as collect:
            with mock.patch("builtins.print"):
                code = dependency_profiles.main(["verify", "--profile", "cpu"])

        self.assertEqual(code, 1)
        self.assertTrue(collect.call_args.kwargs["verify_runtime"])
        self.assertTrue(collect.call_args.kwargs["run_provider_smoke"])
        self.assertTrue(collect.call_args.kwargs["run_import_checks"])
        self.assertTrue(collect.call_args.kwargs["run_package_check"])

    def test_verify_command_writes_an_atomic_machine_report(self):
        valid = {
            "schema": dependency_profiles.PROFILE_STATUS_SCHEMA,
            "profile": "cpu",
            "valid": True,
            "errors": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "nested" / "verification.json"
            with mock.patch.object(
                dependency_profiles,
                "collect_dependency_profile_status",
                return_value=valid,
            ):
                with mock.patch("builtins.print"):
                    code = dependency_profiles.main([
                        "verify",
                        "--profile",
                        "cpu",
                        "--output",
                        str(target),
                    ])
            payload = json.loads(target.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(payload, valid)
        self.assertFalse(target.with_suffix(".json.tmp").exists())

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
