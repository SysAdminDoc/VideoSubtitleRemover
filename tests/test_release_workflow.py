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
        self.assertIn('Install-Optional "ONNX Runtime DirectML"', self.workflow)
        self.assertNotIn('Install-Optional "torch-directml"', self.workflow)
        self.assertNotIn("torch_directml", self.workflow)

    def test_native_command_try_catch_is_not_used_for_pip_installs(self):
        self.assertNotIn("try {\n            python -m pip install", self.workflow)

    def test_winget_submission_is_secret_gated_and_non_interactive(self):
        self.assertIn("id: winget", self.workflow)
        self.assertIn("WINGET_CREATE_GITHUB_TOKEN: ${{ secrets.WINGET_PAT }}", self.workflow)
        self.assertIn("steps.winget.outputs.has_token == 'true'", self.workflow)
        self.assertIn("wingetcreate.exe update $packageId", self.workflow)
        self.assertIn('$packageId = "SysAdminDoc.VideoSubtitleRemoverPro"', self.workflow)
        self.assertIn("VideoSubtitleRemoverPro-$versionTag-Setup.exe", self.workflow)
        self.assertIn("releases/download/$versionTag/$installerName", self.workflow)
        self.assertIn('--urls "$installerUrl|x64|machine"', self.workflow)
        self.assertIn("--submit", self.workflow)
        self.assertIn("--token $env:WINGET_CREATE_GITHUB_TOKEN", self.workflow)
        self.assertIn("--no-open", self.workflow)

    def test_strict_release_quality_verifies_artifacts(self):
        self.assertIn("release_quality:", self.workflow)
        self.assertIn("type: choice", self.workflow)
        self.assertIn("- permissive", self.workflow)
        self.assertIn("- strict", self.workflow)
        self.assertIn(
            "continue-on-error: ${{ github.event.inputs.release_quality != 'strict' }}",
            self.workflow,
        )
        self.assertIn("Verify release artifacts", self.workflow)
        self.assertIn("release-verification.json", self.workflow)
        self.assertIn("Get-FileHash $Path -Algorithm SHA256", self.workflow)
        self.assertIn("Required bundled document missing", self.workflow)
        self.assertIn("Strict release requires NSIS installer artifact.", self.workflow)
        self.assertIn("Get-AuthenticodeSignature $target", self.workflow)
        self.assertIn("release-hidden-imports.json", self.workflow)
        self.assertIn("hiddenImports", self.workflow)
        self.assertIn("python -m pip list --format=json", self.workflow)
        self.assertIn("dependencies", self.workflow)
        self.assertIn("APP_VERSION $appVersion does not match release tag", self.workflow)
        self.assertIn("README.md", self.workflow)
        self.assertIn("CHANGELOG.md", self.workflow)
        self.assertIn("ROADMAP.md", self.workflow)
        self.assertIn("smokeLaunch", self.workflow)

    def test_signing_readiness_uses_step_output(self):
        self.assertIn("id: signing", self.workflow)
        self.assertIn("has_signing=", self.workflow)
        self.assertIn("steps.signing.outputs.has_signing == 'true'", self.workflow)
        self.assertNotIn("if: env.AZURE_SIGN_TENANT_ID != ''", self.workflow)


if __name__ == "__main__":
    unittest.main()
