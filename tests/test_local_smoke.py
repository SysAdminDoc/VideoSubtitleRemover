import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from backend import dependency_caps


ROOT = Path(__file__).resolve().parents[1]


def _load_local_smoke_module():
    spec = importlib.util.spec_from_file_location(
        "vsr_local_smoke_for_tests", ROOT / "tools" / "local_smoke.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


local_smoke = _load_local_smoke_module()


class LocalSmokeTests(unittest.TestCase):
    def test_local_smoke_runs_generated_image_cli_path(self):
        result = subprocess.run(
            [sys.executable, "tools/local_smoke.py", "--skip-self-test"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn('[runtime] {"available":', result.stdout)
        self.assertIn('"providers":', result.stdout)
        self.assertIn('"version":', result.stdout)
        self.assertIn("[ok] local CPU smoke output:", result.stdout)

    def test_dockerfile_uses_cpu_smoke_entrypoint(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("opencv-python-headless", dockerfile)
        self.assertIn(
            f'"onnxruntime>={dependency_caps.ONNXRUNTIME_SECURITY_MIN}"',
            dockerfile,
        )
        self.assertNotIn('"onnxruntime>=1.21.0"', dockerfile)
        self.assertIn('CMD ["python", "tools/local_smoke.py"]', dockerfile)
        self.assertNotIn("github", dockerfile.lower())

    def test_runtime_facts_record_resolved_version_and_providers(self):
        fake_ort = SimpleNamespace(
            __version__="1.25.0",
            get_available_providers=lambda: ["CPUExecutionProvider"],
        )
        with mock.patch.dict(sys.modules, {"onnxruntime": fake_ort}):
            facts = local_smoke._onnxruntime_facts()

        self.assertTrue(facts["available"])
        self.assertEqual(facts["version"], "1.25.0")
        self.assertEqual(facts["providers"], ["CPUExecutionProvider"])


if __name__ == "__main__":
    unittest.main()
