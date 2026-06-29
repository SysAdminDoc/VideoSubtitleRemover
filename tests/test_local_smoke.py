import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
        self.assertIn("[ok] local CPU smoke output:", result.stdout)

    def test_dockerfile_uses_cpu_smoke_entrypoint(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("opencv-python-headless", dockerfile)
        self.assertIn('CMD ["python", "tools/local_smoke.py"]', dockerfile)
        self.assertNotIn("github", dockerfile.lower())


if __name__ == "__main__":
    unittest.main()
