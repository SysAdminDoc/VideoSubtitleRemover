import importlib.util
import os
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
            with mock.patch.object(self.setup_mod.subprocess, "run") as run:
                ok = self.setup_mod.install_dependencies(gpu_info)

        self.assertTrue(ok)
        calls = [" ".join(call.args[0]) for call in run.call_args_list]
        self.assertTrue(any("onnxruntime-directml>=1.18.0" in call for call in calls))
        self.assertFalse(any("torch-directml" in call for call in calls))


if __name__ == "__main__":
    unittest.main()
