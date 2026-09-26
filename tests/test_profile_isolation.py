"""RM-368: every profile path the product computes lands in the test profile.

tests/conftest.py points APPDATA at a throwaway directory before anything is
imported. These checks read the paths back from the product itself, so a new
module that caches a profile path at import time, or reads a different
variable, fails here instead of writing into a developer's real profile.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _inside_test_profile(path) -> bool:
    root = Path(os.environ["VSR_TEST_PROFILE_ROOT"]).resolve()
    try:
        Path(path).resolve().relative_to(root)
    except ValueError:
        return False
    return True


class ProfileIsolationTests(unittest.TestCase):
    def test_appdata_points_into_the_throwaway_profile(self):
        self.assertIn("VSR_TEST_PROFILE_ROOT", os.environ)
        self.assertTrue(_inside_test_profile(os.environ["APPDATA"]))

    def test_the_product_resolves_its_state_inside_it(self):
        from backend.presets import _user_presets_path
        from backend.resume_checkpoint import _default_checkpoint_dir
        from gui.config import LOG_DIR

        for label, path in (
            ("checkpoints", _default_checkpoint_dir()),
            ("presets", _user_presets_path()),
            ("log and settings", LOG_DIR),
        ):
            with self.subTest(label=label):
                self.assertTrue(_inside_test_profile(path), path)

    def test_child_processes_inherit_it(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "from backend.resume_checkpoint import _default_checkpoint_dir;"
             "print(_default_checkpoint_dir())"],
            cwd=ROOT, capture_output=True, text=True, timeout=120, check=True,
        )
        self.assertTrue(
            _inside_test_profile(result.stdout.strip().splitlines()[-1]),
            result.stdout)


if __name__ == "__main__":
    unittest.main()
