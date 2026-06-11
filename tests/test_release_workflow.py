import unittest
from pathlib import Path


class ReleaseWorkflowInstallTests(unittest.TestCase):
    def setUp(self):
        self.workflow = (
            Path(__file__).resolve().parents[1]
            / ".github"
            / "workflows"
            / "build.yml"
        ).read_text(encoding="utf-8")

    def test_required_installs_use_last_exit_code_guard(self):
        self.assertIn("function Install-Required", self.workflow)
        self.assertIn("if ($LASTEXITCODE -ne 0)", self.workflow)
        self.assertIn("throw \"Required dependency install failed: $Name\"", self.workflow)

    def test_release_engines_are_required_or_explicitly_optional(self):
        for group in (
            "torch CPU runtime",
            "core image stack",
            "RapidOCR engines",
            "LaMa inpainter",
            "EasyOCR fallback",
            "PyInstaller",
        ):
            self.assertIn(f'Install-Required "{group}"', self.workflow)

        self.assertIn('Install-Optional "PaddleOCR"', self.workflow)
        self.assertIn('Install-Optional "torch-directml"', self.workflow)

    def test_native_command_try_catch_is_not_used_for_pip_installs(self):
        self.assertNotIn("try {\n            python -m pip install", self.workflow)


if __name__ == "__main__":
    unittest.main()
