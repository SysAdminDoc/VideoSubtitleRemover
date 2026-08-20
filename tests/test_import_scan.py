"""RM-218: the folder walk runs off the Tk thread; this tests its logic."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tempfile

from gui.utils import collect_supported_files


class CollectSupportedFilesTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _touch(self, relative: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        return path

    def test_only_supported_media_is_collected_recursively(self):
        keep_a = self._touch("a/b/movie.mp4")
        keep_b = self._touch("frame.png")
        self._touch("notes.txt")
        self._touch("a/readme.md")

        matches, hit_cap = collect_supported_files([self.root], cap=500)

        self.assertEqual(matches, sorted([str(keep_a), str(keep_b)]))
        self.assertFalse(hit_cap)

    def test_output_is_sorted_even_across_multiple_folders(self):
        z = self._touch("first/z.mp4")
        a = self._touch("second/a.mp4")

        matches, _ = collect_supported_files(
            [self.root / "first", self.root / "second"], cap=500)

        # Passing folders in one order must not leak that order into the
        # result; the chunked apply depends on a stable, sorted list.
        self.assertEqual(matches, sorted([str(z), str(a)]))

    def test_enumeration_stops_at_the_cap(self):
        for index in range(8):
            self._touch(f"clip{index:02d}.mp4")

        matches, hit_cap = collect_supported_files([self.root], cap=5)

        self.assertEqual(len(matches), 5)
        self.assertTrue(hit_cap)
        # Sorted within what was collected, so the chunked apply is stable.
        self.assertEqual(matches, sorted(matches))

    def test_progress_fires_on_the_requested_cadence(self):
        for index in range(10):
            self._touch(f"clip{index:02d}.mp4")
        seen = []

        collect_supported_files(
            [self.root], cap=500,
            on_progress=seen.append, progress_every=4)

        self.assertEqual(seen, [4, 8])

    def test_a_missing_folder_is_skipped_not_fatal(self):
        keep = self._touch("ok.mp4")

        matches, hit_cap = collect_supported_files(
            [self.root / "does-not-exist", self.root], cap=500)

        self.assertEqual(matches, [str(keep)])
        self.assertFalse(hit_cap)


if __name__ == "__main__":
    unittest.main()
