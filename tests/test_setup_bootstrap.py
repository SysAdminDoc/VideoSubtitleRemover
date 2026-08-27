import importlib.util
import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def _load_setup_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "vsr_setup_for_tests", root / "setup.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PythonCudaWheelGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.setup_mod = _load_setup_module()

    def test_windows_python_314_reports_cuda_wheels_unavailable(self):
        version = SimpleNamespace(major=3, minor=14)
        self.assertTrue(
            self.setup_mod._windows_cuda_wheels_unavailable(version, "Windows")
        )

    def test_python_310_is_rejected_by_security_reviewed_profile_floor(self):
        version = SimpleNamespace(major=3, minor=10, micro=14)
        with mock.patch.object(self.setup_mod.sys, "version_info", version):
            with mock.patch("builtins.print") as printed:
                self.assertFalse(self.setup_mod.check_python())
        output = "\n".join(str(call.args[0]) for call in printed.call_args_list)
        self.assertIn("Python 3.11+ required", output)

    def test_windows_python_313_keeps_cuda_path_available(self):
        version = SimpleNamespace(major=3, minor=13)
        self.assertFalse(
            self.setup_mod._windows_cuda_wheels_unavailable(version, "Windows")
        )

    def test_detect_gpu_uses_cim_for_amd_when_wmic_is_absent(self):
        missing_nvidia = FileNotFoundError()
        cim = SimpleNamespace(
            returncode=0,
            stdout="Name\nAMD Radeon RX 7800 XT\n",
        )
        with mock.patch.object(
            self.setup_mod.subprocess, "run",
            side_effect=[missing_nvidia, cim],
        ) as run:
            gpu = self.setup_mod.detect_gpu()

        self.assertTrue(gpu["amd"])
        self.assertEqual(gpu["name"], "AMD Radeon RX 7800 XT")
        self.assertTrue(any(
            "Get-CimInstance" in arg for arg in run.call_args_list[1].args[0]
        ))

    def test_detect_gpu_falls_back_to_wmic_for_legacy_hosts(self):
        wmic = SimpleNamespace(
            returncode=0,
            stdout="Name\nIntel(R) UHD Graphics 770\n",
        )
        with mock.patch.object(
            self.setup_mod.subprocess, "run",
            side_effect=[FileNotFoundError(), FileNotFoundError(), wmic],
        ):
            gpu = self.setup_mod.detect_gpu()

        self.assertTrue(gpu["intel"])
        self.assertEqual(gpu["name"], "Intel(R) UHD Graphics 770")

    def test_detect_gpu_warns_when_all_non_nvidia_probes_are_unavailable(self):
        with mock.patch.object(
            self.setup_mod.subprocess, "run",
            side_effect=FileNotFoundError(),
        ):
            with mock.patch("builtins.print") as printed:
                gpu = self.setup_mod.detect_gpu()

        self.assertFalse(gpu["amd"] or gpu["intel"])
        output = "\n".join(str(call.args[0]) for call in printed.call_args_list)
        self.assertIn("GPU probe was inconclusive", output)

    def test_nvidia_python_314_fails_without_cpu_override(self):
        gpu_info = {
            "nvidia": True,
            "amd": False,
            "intel": False,
            "blackwell": False,
            "cuda_disabled_by_python": False,
        }
        version = SimpleNamespace(major=3, minor=14, micro=0)

        with mock.patch.object(self.setup_mod.platform, "system", return_value="Windows"):
            with mock.patch.object(self.setup_mod.sys, "version_info", version):
                with mock.patch.dict(os.environ, {self.setup_mod.PY314_CPU_OVERRIDE_ENV: ""}):
                    with mock.patch.object(self.setup_mod.subprocess, "run") as run:
                        ok = self.setup_mod.install_pytorch(gpu_info)

        self.assertFalse(ok)
        self.assertTrue(gpu_info["cuda_disabled_by_python"])
        run.assert_not_called()

    def test_nvidia_python_314_cpu_override_uses_cpu_index(self):
        gpu_info = {
            "nvidia": True,
            "amd": False,
            "intel": False,
            "blackwell": False,
            "cuda_disabled_by_python": False,
        }
        version = SimpleNamespace(major=3, minor=14, micro=0)

        with mock.patch.object(self.setup_mod.platform, "system", return_value="Windows"):
            with mock.patch.object(self.setup_mod.sys, "version_info", version):
                with mock.patch.object(self.setup_mod, "get_pip_command", return_value="pip"):
                    with mock.patch.dict(os.environ, {self.setup_mod.PY314_CPU_OVERRIDE_ENV: "1"}):
                        with mock.patch.object(self.setup_mod.subprocess, "run") as run:
                            ok = self.setup_mod.install_pytorch(gpu_info)

        self.assertTrue(ok)
        self.assertTrue(gpu_info["cuda_disabled_by_python"])
        args = run.call_args.args[0]
        self.assertEqual(
            run.call_args.kwargs["timeout"],
            self.setup_mod.PIP_INSTALL_TIMEOUT_SECONDS,
        )
        self.assertIn("https://download.pytorch.org/whl/cpu", args)
        self.assertNotIn("https://download.pytorch.org/whl/cu118", args)
        self.assertNotIn("https://download.pytorch.org/whl/cu128", args)

    def test_nvidia_cuda_uses_cu128_index_for_torch_floor(self):
        gpu_info = {
            "nvidia": True,
            "amd": False,
            "intel": False,
            "blackwell": False,
            "cuda_disabled_by_python": False,
        }
        version = SimpleNamespace(major=3, minor=13, micro=0)

        with mock.patch.object(self.setup_mod.platform, "system", return_value="Windows"):
            with mock.patch.object(self.setup_mod.sys, "version_info", version):
                with mock.patch.object(self.setup_mod, "get_pip_command", return_value="pip"):
                    with mock.patch.object(self.setup_mod.subprocess, "run") as run:
                        ok = self.setup_mod.install_pytorch(gpu_info)

        self.assertTrue(ok)
        args = run.call_args.args[0]
        self.assertIn("torch>=2.11.0", args)
        self.assertIn("https://download.pytorch.org/whl/cu128", args)
        self.assertNotIn("https://download.pytorch.org/whl/cu118", args)

    def test_create_virtual_env_timeout_fails_with_guidance(self):
        timeout = self.setup_mod.VENV_CREATE_TIMEOUT_SECONDS
        exc = self.setup_mod.subprocess.TimeoutExpired(
            cmd=["python", "-m", "venv", "venv"],
            timeout=timeout,
        )
        with mock.patch.object(self.setup_mod.Path, "exists", return_value=False):
            with mock.patch.object(self.setup_mod.subprocess, "run", side_effect=exc) as run:
                with mock.patch("builtins.print") as printed:
                    ok = self.setup_mod.create_virtual_env()

        self.assertFalse(ok)
        self.assertEqual(run.call_args.kwargs["timeout"], timeout)
        output = "\n".join(str(call.args[0]) for call in printed.call_args_list)
        self.assertIn("Timed out", output)
        self.assertIn("rerun setup.py", output)

    def test_existing_virtual_env_kept_without_prompt_by_default(self):
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                Path("venv").mkdir()
                with mock.patch("builtins.input", side_effect=AssertionError("stdin prompt")):
                    with mock.patch.object(self.setup_mod.shutil, "rmtree") as rmtree:
                        with mock.patch.object(self.setup_mod.subprocess, "run") as run:
                            ok = self.setup_mod.create_virtual_env()
            finally:
                os.chdir(old_cwd)

        self.assertTrue(ok)
        rmtree.assert_not_called()
        run.assert_not_called()

    def test_recreate_virtual_env_removes_only_repo_local_venv(self):
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                Path("venv").mkdir()
                with mock.patch("builtins.input", side_effect=AssertionError("stdin prompt")):
                    with mock.patch.object(self.setup_mod.shutil, "rmtree") as rmtree:
                        with mock.patch.object(self.setup_mod.subprocess, "run") as run:
                            ok = self.setup_mod.create_virtual_env(repair=True)
            finally:
                os.chdir(old_cwd)

        self.assertTrue(ok)
        rmtree.assert_called_once_with(Path("venv"))
        run.assert_called_once()

    def test_recreate_virtual_env_refuses_reparse_point(self):
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                Path("venv").mkdir()
                with mock.patch("builtins.input", side_effect=AssertionError("stdin prompt")):
                    with mock.patch.object(self.setup_mod, "_is_reparse_point", return_value=True):
                        with mock.patch.object(self.setup_mod.shutil, "rmtree") as rmtree:
                            with mock.patch.object(self.setup_mod.subprocess, "run") as run:
                                with mock.patch("builtins.print") as printed:
                                    ok = self.setup_mod.create_virtual_env(repair=True)
            finally:
                os.chdir(old_cwd)

        self.assertFalse(ok)
        rmtree.assert_not_called()
        run.assert_not_called()
        output = "\n".join(str(call.args[0]) for call in printed.call_args_list)
        self.assertIn("Refusing to remove unsafe virtual environment path", output)

    def test_install_dependencies_timeout_fails(self):
        timeout = self.setup_mod.PIP_INSTALL_TIMEOUT_SECONDS
        exc = self.setup_mod.subprocess.TimeoutExpired(
            cmd=["pip", "install", "--upgrade", "pip"],
            timeout=timeout,
        )
        with mock.patch.object(self.setup_mod, "get_pip_command", return_value="pip"):
            with mock.patch.object(self.setup_mod.subprocess, "run", side_effect=exc) as run:
                with mock.patch("builtins.print") as printed:
                    ok = self.setup_mod.install_dependencies()

        self.assertFalse(ok)
        self.assertEqual(run.call_args.kwargs["timeout"], timeout)
        output = "\n".join(str(call.args[0]) for call in printed.call_args_list)
        self.assertIn("Timed out", output)
        self.assertIn("PyPI mirror", output)

    def test_requirements_failure_never_falls_back_to_a_partial_stack(self):
        failure = self.setup_mod.subprocess.CalledProcessError(
            1, ["pip", "install", "-r", "requirements.txt"]
        )
        with mock.patch.object(self.setup_mod, "get_pip_command", return_value="pip"):
            with mock.patch.object(
                self.setup_mod,
                "_run_pip_install",
                side_effect=[None, failure],
            ) as install:
                with mock.patch("builtins.print") as printed:
                    ok = self.setup_mod.install_dependencies({})

        self.assertFalse(ok)
        self.assertEqual(install.call_count, 2)
        output = "\n".join(str(call.args[0]) for call in printed.call_args_list)
        self.assertNotIn("falling back", output.lower())
        self.assertNotIn("All dependencies installed", output)

    def test_nvidia_provider_failure_is_not_reported_as_success(self):
        failure = self.setup_mod.subprocess.CalledProcessError(
            1, ["pip", "install", "onnxruntime-gpu"]
        )
        gpu_info = {
            "nvidia": True,
            "amd": False,
            "intel": False,
            "cuda_disabled_by_python": False,
        }
        with mock.patch.object(self.setup_mod, "get_pip_command", return_value="pip"):
            with mock.patch.object(
                self.setup_mod,
                "_run_pip_install",
                side_effect=[None, None, failure],
            ):
                with mock.patch("builtins.print") as printed:
                    ok = self.setup_mod.install_dependencies(gpu_info)

        self.assertFalse(ok)
        output = "\n".join(str(call.args[0]) for call in printed.call_args_list)
        self.assertNotIn("All dependencies installed", output)

    def test_paddlepaddle_timeout_returns_false(self):
        gpu_info = {
            "nvidia": False,
            "amd": False,
            "intel": False,
            "blackwell": False,
        }
        timeout = self.setup_mod.PIP_INSTALL_TIMEOUT_SECONDS
        exc = self.setup_mod.subprocess.TimeoutExpired(
            cmd=["pip", "install", "paddlepaddle==3.0.0"],
            timeout=timeout,
        )
        with mock.patch.object(self.setup_mod, "get_pip_command", return_value="pip"):
            with mock.patch.object(self.setup_mod.subprocess, "run", side_effect=exc) as run:
                with mock.patch("builtins.print"):
                    ok = self.setup_mod.install_paddlepaddle(gpu_info)

        self.assertFalse(ok)
        self.assertEqual(run.call_args.kwargs["timeout"], timeout)

    def test_amd_intel_branch_keeps_torch_cpu_and_avoids_torch_directml(self):
        gpu_info = {
            "nvidia": False,
            "amd": True,
            "intel": False,
            "blackwell": False,
            "cuda_disabled_by_python": False,
        }
        with mock.patch.object(self.setup_mod, "get_pip_command", return_value="pip"):
            with mock.patch.object(self.setup_mod.subprocess, "run") as run:
                ok = self.setup_mod.install_pytorch(gpu_info)

        self.assertTrue(ok)
        calls = [" ".join(call.args[0]) for call in run.call_args_list]
        self.assertTrue(any("torch>=2.11.0" in call for call in calls))
        self.assertTrue(any("https://download.pytorch.org/whl/cpu" in call for call in calls))
        self.assertFalse(any("torch-directml" in call for call in calls))

    def test_amd_intel_dependencies_install_onnxruntime_directml(self):
        gpu_info = {
            "nvidia": False,
            "amd": True,
            "intel": False,
        }
        with mock.patch.object(self.setup_mod, "get_pip_command", return_value="pip"):
            with mock.patch.object(
                self.setup_mod, "_preflight_directml_distribution", return_value=True
            ):
                with mock.patch.object(self.setup_mod.subprocess, "run") as run:
                    ok = self.setup_mod.install_dependencies(gpu_info)

        self.assertTrue(ok)
        calls = [" ".join(call.args[0]) for call in run.call_args_list]
        self.assertTrue(any("onnxruntime-directml==1.24.4" in call for call in calls))
        self.assertFalse(any("torch-directml" in call for call in calls))
        self.assertFalse(any("openvino" in call for call in calls))

    def test_intel_uses_the_same_locked_directml_core_as_amd(self):
        gpu_info = {
            "nvidia": False,
            "amd": False,
            "intel": True,
        }
        with mock.patch.object(self.setup_mod, "get_pip_command", return_value="pip"):
            with mock.patch.object(
                self.setup_mod, "_preflight_directml_distribution", return_value=True
            ):
                with mock.patch.object(self.setup_mod.subprocess, "run") as run:
                    ok = self.setup_mod.install_dependencies(gpu_info)

        self.assertTrue(ok)
        calls = [" ".join(call.args[0]) for call in run.call_args_list]
        self.assertTrue(any("onnxruntime-directml==1.24.4" in call for call in calls))
        self.assertFalse(any("openvino" in call for call in calls))

    def test_directml_unavailable_fails_before_environment_mutation(self):
        gpu_info = {
            "nvidia": False,
            "amd": True,
            "intel": False,
        }
        result = SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="ERROR: No matching distribution found",
        )
        with mock.patch.object(self.setup_mod, "get_pip_command", return_value="pip"):
            with mock.patch.object(
                self.setup_mod.subprocess, "run", return_value=result
            ) as run:
                with mock.patch("builtins.print") as printed:
                    ok = self.setup_mod.install_dependencies(gpu_info)

        self.assertFalse(ok)
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertIn("--dry-run", command)
        self.assertIn("--only-binary=:all:", command)
        self.assertIn("onnxruntime-directml==1.24.4", command)
        output = "\n".join(str(call.args[0]) for call in printed.call_args_list)
        self.assertIn("No packages were changed", output)
        self.assertIn("CPU setup path", output)
        self.assertIn("Windows ML", output)

    def test_directml_floor_matches_runtime_policy(self):
        from backend import dependency_caps

        self.assertEqual(
            self.setup_mod.DIRECTML_PACKAGE_VERSION,
            dependency_caps.ONNXRUNTIME_DIRECTML_VERSION,
        )

    def test_nvidia_dependencies_install_onnxruntime_gpu(self):
        gpu_info = {
            "nvidia": True,
            "amd": False,
            "intel": False,
            "cuda_disabled_by_python": False,
        }
        with mock.patch.object(self.setup_mod, "get_pip_command", return_value="pip"):
            with mock.patch.object(self.setup_mod.subprocess, "run") as run:
                ok = self.setup_mod.install_dependencies(gpu_info)

        self.assertTrue(ok)
        calls = [" ".join(call.args[0]) for call in run.call_args_list]
        self.assertTrue(any("setuptools<82" in call for call in calls))
        self.assertTrue(
            any("onnxruntime-gpu>=1.26.0,<1.27.0" in call for call in calls)
        )
        self.assertFalse(any("onnxruntime-directml" in call for call in calls))

    def test_repair_argument_enables_unattended_recreate_mode(self):
        self.assertFalse(self.setup_mod.parse_setup_args([]).repair)
        self.assertTrue(self.setup_mod.parse_setup_args(["--repair"]).repair)

    def test_setup_progress_is_atomic_and_temp_directory_bound(self):
        progress = Path(tempfile.gettempdir()) / (
            f"vsr-pro-setup-test-{os.getpid()}.status")
        progress.unlink(missing_ok=True)
        temporary = progress.with_suffix(".tmp")
        temporary.unlink(missing_ok=True)
        try:
            with mock.patch.dict(
                os.environ,
                {self.setup_mod.SETUP_PROGRESS_ENV: str(progress)},
            ):
                self.assertTrue(
                    self.setup_mod.write_setup_progress(
                        "Installing | OCR\npackages", 72))
            self.assertEqual(
                progress.read_text(encoding="utf-8"),
                "RUNNING|Installing OCR packages|72",
            )
            self.assertFalse(temporary.exists())
        finally:
            progress.unlink(missing_ok=True)
            temporary.unlink(missing_ok=True)

    def test_setup_progress_rejects_non_temp_target(self):
        target = Path(__file__).with_name("vsr-pro-setup-test.status")
        with mock.patch.dict(
            os.environ,
            {self.setup_mod.SETUP_PROGRESS_ENV: str(target)},
        ):
            self.assertFalse(
                self.setup_mod.write_setup_progress("blocked", 10))
        self.assertFalse(target.exists())

    def test_generated_launchers_match_tracked_files(self):
        root = Path(__file__).resolve().parents[1]
        launchers = [
            "Run_VSR_Pro.bat",
            "Run_VSR_Pro_Debug.bat",
            "Run_VSR_Pro.ps1",
        ]
        tracked = {
            name: (root / name).read_text(encoding="utf-8")
            for name in launchers
        }
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                with mock.patch("builtins.print"):
                    self.setup_mod.create_launcher()
                for name in launchers:
                    generated = Path(name).read_text(encoding="utf-8")
                    self.assertEqual(generated, tracked[name], name)
            finally:
                os.chdir(old_cwd)

        for name, content in tracked.items():
            self.assertIn("setup.py --repair", content, name)
            self.assertIn("backend.dependency_profiles", content, name)
            self.assertIn("verify", content, name)
            self.assertNotIn("import cv2, PIL, numpy", content, name)
            self.assertNotIn("setup.py\n", content, name)
        self.assertIn("scripts\\setup_splash.py", tracked["Run_VSR_Pro.bat"])
        self.assertIn("VSR_SETUP_PROGRESS_FILE", tracked["Run_VSR_Pro.bat"])

    def test_setup_script_has_no_stdin_prompt(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "setup.py").read_text(encoding="utf-8")
        self.assertNotIn("input(", source)

    def test_profile_verifier_uses_the_venv_and_named_profile(self):
        payload = {
            "schema": "vsr.dependency_profile_status.v1",
            "profile": "cpu",
            "valid": True,
            "requiredPackages": [],
            "capabilities": [],
        }
        result = SimpleNamespace(returncode=0, stdout="import banner", stderr="")
        with tempfile.TemporaryDirectory() as tmpdir:
            report = Path(tmpdir) / "verification.json"

            def write_report(*_args, **_kwargs):
                report.write_text(json.dumps(payload), encoding="utf-8")
                return result

            with mock.patch.object(
                self.setup_mod, "get_python_command", return_value="venv-python"
            ):
                with mock.patch.object(
                    self.setup_mod.subprocess,
                    "run",
                    side_effect=write_report,
                ) as run:
                    actual = self.setup_mod.verify_installed_profile(
                        "cpu",
                        report_path=report,
                    )

            self.assertFalse(report.exists())

        self.assertTrue(actual["valid"])
        self.assertEqual(
            run.call_args.args[0],
            [
                "venv-python",
                "-m",
                "backend.dependency_profiles",
                "verify",
                "--profile",
                "cpu",
                "--output",
                str(report),
            ],
        )

    def test_failed_setup_report_is_atomic_and_rerunnable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "venv" / ".vsr-setup-report.json"
            target.parent.mkdir()
            written = self.setup_mod.write_setup_report(
                "cpu",
                "failed",
                stage="dependency_install",
                message="simulated resolver failure",
                path=target,
            )
            payload = json.loads(target.read_text(encoding="utf-8"))

        self.assertTrue(written)
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["stage"], "dependency_install")
        self.assertEqual(len(payload["requiredPackages"]), 9)
        self.assertEqual(
            payload["repairCommand"],
            "python setup.py --repair --profile cpu",
        )
        self.assertFalse(target.with_suffix(".json.tmp").exists())

    def test_main_cannot_print_complete_after_verification_failure(self):
        invalid = {
            "schema": "vsr.dependency_profile_status.v1",
            "profile": "cpu",
            "valid": False,
            "errors": ["Required package protobuf==6.33.6 is not installed"],
        }
        gpu = {
            "nvidia": False,
            "amd": False,
            "intel": False,
            "blackwell": False,
            "cuda_disabled_by_python": False,
        }
        patches = (
            mock.patch.object(self.setup_mod, "_anchor_working_directory"),
            mock.patch.object(self.setup_mod, "print_banner"),
            mock.patch.object(self.setup_mod, "check_python", return_value=True),
            mock.patch.object(self.setup_mod, "detect_gpu", return_value=gpu),
            mock.patch.object(self.setup_mod, "ensure_profile_current"),
            mock.patch.object(self.setup_mod, "_print_profile_contract"),
            mock.patch.object(self.setup_mod, "create_virtual_env", return_value=True),
            mock.patch.object(self.setup_mod, "install_pytorch", return_value=True),
            mock.patch.object(self.setup_mod, "install_dependencies", return_value=True),
            mock.patch.object(
                self.setup_mod, "verify_installed_profile", return_value=invalid
            ),
            mock.patch.object(self.setup_mod, "write_setup_report", return_value=True),
            mock.patch.object(self.setup_mod, "write_setup_progress"),
            mock.patch.object(self.setup_mod, "check_ffmpeg"),
            mock.patch.object(self.setup_mod, "create_launcher"),
        )
        started = []
        try:
            for item in patches:
                started.append(item.start())
            with mock.patch("builtins.print") as printed:
                code = self.setup_mod.main(["--profile", "cpu"])
        finally:
            for item in reversed(patches):
                item.stop()

        self.assertEqual(code, 1)
        started[-1].assert_not_called()
        started[-2].assert_not_called()
        output = "\n".join(str(call.args[0]) for call in printed.call_args_list)
        self.assertNotIn("SETUP COMPLETE", output)
        self.assertIn("python setup.py --repair --profile cpu", output)

    def test_main_reports_success_only_after_full_profile_verification(self):
        valid = {
            "schema": "vsr.dependency_profile_status.v1",
            "profile": "cpu",
            "valid": True,
            "errors": [],
            "requiredPackages": [],
            "capabilities": ["CPU inference"],
            "providerSmoke": {
                "activeProviders": ["CPUExecutionProvider"],
            },
        }
        gpu = {
            "nvidia": False,
            "amd": False,
            "intel": False,
            "blackwell": False,
            "cuda_disabled_by_python": False,
        }
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                self.setup_mod, "_anchor_working_directory"
            ))
            stack.enter_context(mock.patch.object(self.setup_mod, "print_banner"))
            stack.enter_context(mock.patch.object(
                self.setup_mod, "check_python", return_value=True
            ))
            stack.enter_context(mock.patch.object(
                self.setup_mod, "detect_gpu", return_value=gpu
            ))
            stack.enter_context(mock.patch.object(
                self.setup_mod, "ensure_profile_current"
            ))
            stack.enter_context(mock.patch.object(
                self.setup_mod, "_print_profile_contract"
            ))
            stack.enter_context(mock.patch.object(
                self.setup_mod, "_print_profile_verification"
            ))
            stack.enter_context(mock.patch.object(
                self.setup_mod, "create_virtual_env", return_value=True
            ))
            stack.enter_context(mock.patch.object(
                self.setup_mod, "install_pytorch", return_value=True
            ))
            stack.enter_context(mock.patch.object(
                self.setup_mod, "install_dependencies", return_value=True
            ))
            verify = stack.enter_context(mock.patch.object(
                self.setup_mod,
                "verify_installed_profile",
                return_value=valid,
            ))
            report = stack.enter_context(mock.patch.object(
                self.setup_mod, "write_setup_report", return_value=True
            ))
            progress = stack.enter_context(mock.patch.object(
                self.setup_mod, "write_setup_progress"
            ))
            stack.enter_context(mock.patch.object(
                self.setup_mod, "check_ffmpeg", return_value=True
            ))
            launcher = stack.enter_context(mock.patch.object(
                self.setup_mod, "create_launcher"
            ))
            stack.enter_context(mock.patch.object(
                self.setup_mod.platform, "system", return_value="Windows"
            ))
            with mock.patch("builtins.print") as printed:
                code = self.setup_mod.main(["--profile", "cpu"])

        self.assertEqual(code, 0)
        verify.assert_called_once_with("cpu")
        launcher.assert_called_once_with()
        self.assertEqual(report.call_args_list[-1].args[:2], ("cpu", "verified"))
        self.assertEqual(report.call_args_list[-1].kwargs["verification"], valid)
        self.assertTrue(any(
            call.args[2] == "DONE"
            for call in progress.call_args_list
            if len(call.args) >= 3
        ))
        output = "\n".join(str(call.args[0]) for call in printed.call_args_list)
        self.assertIn("SETUP COMPLETE", output)

    def test_main_install_failure_is_diagnosable_and_rerunnable(self):
        gpu = {
            "nvidia": False,
            "amd": False,
            "intel": False,
            "blackwell": False,
            "cuda_disabled_by_python": False,
        }
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                self.setup_mod, "_anchor_working_directory"
            ))
            stack.enter_context(mock.patch.object(self.setup_mod, "print_banner"))
            stack.enter_context(mock.patch.object(
                self.setup_mod, "check_python", return_value=True
            ))
            stack.enter_context(mock.patch.object(
                self.setup_mod, "detect_gpu", return_value=gpu
            ))
            stack.enter_context(mock.patch.object(
                self.setup_mod, "ensure_profile_current"
            ))
            stack.enter_context(mock.patch.object(
                self.setup_mod, "_print_profile_contract"
            ))
            stack.enter_context(mock.patch.object(
                self.setup_mod, "create_virtual_env", return_value=True
            ))
            stack.enter_context(mock.patch.object(
                self.setup_mod, "install_pytorch", return_value=True
            ))
            stack.enter_context(mock.patch.object(
                self.setup_mod, "install_dependencies", return_value=False
            ))
            partial = {
                "schema": "vsr.dependency_profile_status.v1",
                "profile": "cpu",
                "valid": False,
                "errors": ["simulated missing package"],
            }
            verify = stack.enter_context(mock.patch.object(
                self.setup_mod,
                "verify_installed_profile",
                return_value=partial,
            ))
            report = stack.enter_context(mock.patch.object(
                self.setup_mod, "write_setup_report", return_value=True
            ))
            stack.enter_context(mock.patch.object(
                self.setup_mod, "write_setup_progress"
            ))
            ffmpeg = stack.enter_context(mock.patch.object(
                self.setup_mod, "check_ffmpeg"
            ))
            launcher = stack.enter_context(mock.patch.object(
                self.setup_mod, "create_launcher"
            ))
            stack.enter_context(mock.patch.object(
                self.setup_mod.platform, "system", return_value="Windows"
            ))
            with mock.patch("builtins.print") as printed:
                code = self.setup_mod.main(["--profile", "cpu"])

        self.assertEqual(code, 1)
        verify.assert_called_once_with("cpu")
        ffmpeg.assert_not_called()
        launcher.assert_not_called()
        self.assertEqual(report.call_args_list[-1].args[:2], ("cpu", "failed"))
        self.assertEqual(
            report.call_args_list[-1].kwargs["stage"],
            "dependency_install",
        )
        self.assertEqual(
            report.call_args_list[-1].kwargs["verification"],
            partial,
        )
        output = "\n".join(str(call.args[0]) for call in printed.call_args_list)
        self.assertNotIn("SETUP COMPLETE", output)
        self.assertIn("python setup.py --repair --profile cpu", output)


if __name__ == "__main__":
    unittest.main()


class PowerShellLauncherRobustnessTests(unittest.TestCase):
    """RM-161 / RM-181: both PowerShell launchers failed on a normal path.

    The source launcher probed the venv with a redirected-stderr native call
    under $ErrorActionPreference = "Stop"; on Windows PowerShell 5.1 that
    turns the failing import's traceback into a terminating error, so the
    launcher died in exactly the broken-venv case its repair branch exists
    to handle. The frozen launcher passed an empty $args to
    Start-Process -ArgumentList, which rejects empty collections, so the
    plain no-argument launch failed.
    """

    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]

    def test_source_launcher_probes_without_a_terminating_redirect(self):
        text = (self.root / "Run_VSR_Pro.ps1").read_text(encoding="utf-8")
        self.assertIn("function Invoke-VsrProbe", text)
        self.assertIn('$ErrorActionPreference = "Continue"', text)
        # The raw redirected-native-call form must not come back.
        self.assertNotIn('-c "import cv2, PIL, numpy" 1>$null 2>$null', text)
        self.assertIn("Invoke-VsrProbe", text)

    def test_frozen_launcher_handles_a_no_argument_launch(self):
        text = (
            self.root / "assets" / "frozen" / "Run_VSR_Pro.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("$args.Count -gt 0", text)
        # A bare Start-Process ... -ArgumentList $args would throw on the
        # default launch; the else branch must omit -ArgumentList entirely.
        self.assertIn(
            "Start-Process -FilePath $exe -WorkingDirectory $PSScriptRoot",
            text,
        )


class WorkingDirectoryAnchorTests(unittest.TestCase):
    """setup.py builds venv/, launchers and reads requirements.txt by
    relative path, so it must run from its own directory regardless of
    where the user invoked it from."""

    def setUp(self):
        self.module = _load_setup_module()
        self.root = Path(__file__).resolve().parents[1]
        self.original = os.getcwd()
        self.addCleanup(os.chdir, self.original)

    def test_anchor_moves_into_the_setup_directory(self):
        with tempfile.TemporaryDirectory() as elsewhere:
            os.chdir(elsewhere)
            previous = self.module._anchor_working_directory()
            self.assertEqual(
                os.path.normcase(os.getcwd()),
                os.path.normcase(str(self.root)),
            )
            # The caller's directory is reported back so it can be restored.
            self.assertEqual(
                os.path.normcase(previous),
                os.path.normcase(os.path.realpath(elsewhere)),
            )
        # The real requirements file is now reachable by relative path.
        self.assertTrue(os.path.exists("requirements.txt"))

    def test_anchor_is_a_no_op_when_already_correct(self):
        os.chdir(str(self.root))
        previous = self.module._anchor_working_directory()
        self.assertEqual(
            os.path.normcase(previous), os.path.normcase(str(self.root))
        )
        self.assertEqual(
            os.path.normcase(os.getcwd()), os.path.normcase(str(self.root))
        )

    def test_main_anchors_before_touching_relative_paths(self):
        source = (self.root / "setup.py").read_text(encoding="utf-8")
        main_body = source.split("def main(argv=None):", 1)[1]
        anchor_at = main_body.index("_anchor_working_directory()")
        banner_at = main_body.index("print_banner()")
        self.assertLess(anchor_at, banner_at)
