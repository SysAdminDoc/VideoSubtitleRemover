import json
from pathlib import Path
import tempfile
import unittest

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
            self.assertIn("numpy==2.2.6", text)
            self.assertIn("Pillow==12.3.0", text)

        self.assertIn("onnxruntime==1.27.0", (
            dependency_profiles.profile_constraint_path("cpu").read_text(
                encoding="utf-8")))
        self.assertIn("onnxruntime-gpu==1.27.0", (
            dependency_profiles.profile_constraint_path("nvidia").read_text(
                encoding="utf-8")))
        self.assertIn("onnxruntime-directml==1.24.4", (
            dependency_profiles.profile_constraint_path("directml").read_text(
                encoding="utf-8")))

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
