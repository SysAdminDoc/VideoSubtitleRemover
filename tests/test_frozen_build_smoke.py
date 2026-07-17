import os
from pathlib import Path
import unittest

from scripts import frozen_build_smoke


ROOT = Path(__file__).resolve().parents[1]


class FrozenBuildSmokeTests(unittest.TestCase):
    def test_committed_spec_collects_numpy_and_disables_upx(self):
        spec = (ROOT / "VideoSubtitleRemoverPro.spec").read_text(encoding="ascii")
        self.assertIn("collect_all('numpy')", spec)
        self.assertIn("binaries=np_binaries", spec)
        self.assertIn("np_hiddenimports", spec)
        self.assertGreaterEqual(spec.count("upx=False"), 2)
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("!VideoSubtitleRemoverPro.spec", ignore)

    def test_build_command_uses_committed_spec_and_isolated_paths(self):
        command = frozen_build_smoke.build_command(
            spec=ROOT / "VideoSubtitleRemoverPro.spec",
            dist_root=ROOT / "build" / "test-dist",
            work_dir=ROOT / "build" / "test-work",
        )
        self.assertIn("PyInstaller", command)
        self.assertIn("--clean", command)
        self.assertEqual(command[-1], str(ROOT / "VideoSubtitleRemoverPro.spec"))

    def test_smoke_payload_requires_numpy_cv2_and_startup_message(self):
        payload = frozen_build_smoke.validate_smoke_payload({
            "schema": "vsr.frozen_import_smoke.v1",
            "passed": True,
            "imports": {"numpy": "2.4.0", "cv2": "5.0.0"},
            "startedMessage": "Video Subtitle Remover Pro v3.18.1 started",
        })
        self.assertTrue(payload["passed"])
        with self.assertRaisesRegex(RuntimeError, "missing cv2"):
            frozen_build_smoke.validate_smoke_payload({
                "schema": "vsr.frozen_import_smoke.v1",
                "passed": True,
                "imports": {"numpy": "2.4.0"},
                "startedMessage": "Video Subtitle Remover Pro v3.18.1 started",
            })

    @unittest.skipUnless(
        os.environ.get("VSR_RUN_FROZEN_SMOKE") == "1",
        "set VSR_RUN_FROZEN_SMOKE=1 for the slow PyInstaller build",
    )
    def test_build_committed_spec_and_run_frozen_imports(self):
        payload = frozen_build_smoke.build_and_smoke()
        self.assertTrue(payload["passed"])


if __name__ == "__main__":
    unittest.main()
