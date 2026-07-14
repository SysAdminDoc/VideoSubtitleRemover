import importlib.util
import os
import tempfile
import unittest
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

    def test_windows_python_313_keeps_cuda_path_available(self):
        version = SimpleNamespace(major=3, minor=13)
        self.assertFalse(
            self.setup_mod._windows_cuda_wheels_unavailable(version, "Windows")
        )

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
        self.assertIn("torch>=2.10.0", args)
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
        self.assertTrue(any("torch>=2.10.0" in call for call in calls))
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

    def test_intel_dependencies_install_openvino_for_rapidocr(self):
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
        self.assertTrue(any("openvino>=2025.0.0" in call for call in calls))

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
        self.assertTrue(any("onnxruntime-gpu>=1.25.0" in call for call in calls))
        self.assertFalse(any("onnxruntime-directml" in call for call in calls))

    def test_repair_argument_enables_unattended_recreate_mode(self):
        self.assertFalse(self.setup_mod.parse_setup_args([]).repair)
        self.assertTrue(self.setup_mod.parse_setup_args(["--repair"]).repair)

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
            self.assertIn("import cv2, PIL, numpy", content, name)
            self.assertNotIn("setup.py\n", content, name)

    def test_setup_script_has_no_stdin_prompt(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "setup.py").read_text(encoding="utf-8")
        self.assertNotIn("input(", source)


if __name__ == "__main__":
    unittest.main()
