import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


class CacheInventoryTests(unittest.TestCase):
    def test_clean_cache_only_deletes_allowlisted_app_cache_dirs(self):
        from backend.cache_inventory import clean_cache

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            appdata = root / "appdata"
            vsr_root = appdata / "VideoSubtitleRemoverPro"
            checkpoints = vsr_root / "checkpoints"
            checkpoints.mkdir(parents=True)
            (checkpoints / "resume.json").write_text("{}", encoding="utf-8")
            outside = root / "outside"
            outside.mkdir()
            outside_file = outside / "keep.txt"
            outside_file.write_text("keep", encoding="utf-8")

            with mock.patch.dict(os.environ, {"APPDATA": str(appdata)}):
                with mock.patch("backend.cache_inventory.logger.warning") as warning:
                    with contextlib.redirect_stdout(io.StringIO()):
                        freed = clean_cache(
                            dry_run=False,
                            subdirs={
                                "checkpoints",
                                "..",
                                str(outside),
                                Path("proxy_cache"),
                            },
                        )

            self.assertIn("checkpoints", freed)
            self.assertEqual(warning.call_count, 2)
            self.assertTrue(checkpoints.is_dir())
            self.assertFalse((checkpoints / "resume.json").exists())
            self.assertTrue(outside_file.exists())


if __name__ == "__main__":
    unittest.main()
