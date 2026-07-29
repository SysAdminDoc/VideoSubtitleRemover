"""RM-146: cache and matte replacements are transactionally recoverable."""

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend import atomic_replace, cache_inventory, matte_interchange


class ReplacementJournalTests(unittest.TestCase):
    def _setup(self, tmpdir):
        root = Path(tmpdir)
        a, b = root / "a.bin", root / "b.bin"
        a.write_bytes(b"old-a")
        b.write_bytes(b"old-b")
        new_a, new_b = root / "new-a", root / "new-b"
        new_a.write_bytes(b"new-a")
        new_b.write_bytes(b"new-b")
        return root, a, b, new_a, new_b

    def test_commit_leaves_the_new_set_and_no_journal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root, a, b, new_a, new_b = self._setup(tmpdir)
            journal = atomic_replace.ReplacementJournal(root, "unit")
            backup_a = journal.plan(a)
            backup_b = journal.plan(b)
            journal.begin()
            self.assertTrue(journal.path.is_file())
            os.replace(a, backup_a)
            os.replace(new_a, a)
            os.replace(b, backup_b)
            os.replace(new_b, b)
            journal.commit()
            self.assertEqual(a.read_bytes(), b"new-a")
            self.assertEqual(b.read_bytes(), b"new-b")
            self.assertFalse(backup_a.exists())
            self.assertFalse(backup_b.exists())
            self.assertFalse(journal.path.exists())

    def test_rollback_after_a_partial_move_restores_the_old_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root, a, b, new_a, new_b = self._setup(tmpdir)
            journal = atomic_replace.ReplacementJournal(root, "unit")
            backup_a = journal.plan(a)
            journal.plan(b)
            journal.begin()
            os.replace(a, backup_a)
            os.replace(new_a, a)
            # Second file never lands.
            journal.rollback()
            self.assertEqual(a.read_bytes(), b"old-a")
            self.assertEqual(b.read_bytes(), b"old-b")
            self.assertFalse(journal.path.exists())

    def test_pending_journal_recovery_rolls_back(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root, a, b, new_a, new_b = self._setup(tmpdir)
            journal = atomic_replace.ReplacementJournal(root, "unit")
            backup_a = journal.plan(a)
            journal.plan(b)
            journal.begin()
            os.replace(a, backup_a)
            os.replace(new_a, a)
            # Simulate a crash: the journal is left behind as "pending".
            actions = atomic_replace.recover_pending_replacements(root)
            self.assertTrue(any(
                item["action"] == "rolled-back" for item in actions))
            self.assertEqual(a.read_bytes(), b"old-a")
            self.assertEqual(b.read_bytes(), b"old-b")
            self.assertFalse(journal.path.exists())

    def test_committed_journal_recovery_finishes_forward(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root, a, b, new_a, new_b = self._setup(tmpdir)
            journal = atomic_replace.ReplacementJournal(root, "unit")
            backup_a = journal.plan(a)
            backup_b = journal.plan(b)
            journal.begin()
            os.replace(a, backup_a)
            os.replace(new_a, a)
            os.replace(b, backup_b)
            os.replace(new_b, b)
            journal.mark_committed()
            # Crash before the backups were dropped.
            actions = atomic_replace.recover_pending_replacements(root)
            self.assertTrue(any(
                item["action"] == "completed" for item in actions))
            self.assertEqual(a.read_bytes(), b"new-a")
            self.assertEqual(b.read_bytes(), b"new-b")
            self.assertFalse(backup_a.exists())
            self.assertFalse(backup_b.exists())

    def test_rollback_removes_a_target_that_had_no_original(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "fresh.bin"
            journal = atomic_replace.ReplacementJournal(root, "unit")
            journal.plan(target)
            journal.begin()
            target.write_bytes(b"partial")
            journal.rollback()
            self.assertFalse(target.exists())

    def test_orphaned_backup_is_restored_when_the_target_is_gone(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            backup = root / ("weights.onnx" + atomic_replace.BACKUP_SUFFIX)
            backup.write_bytes(b"original")
            atomic_replace.recover_pending_replacements(root)
            self.assertTrue((root / "weights.onnx").is_file())
            self.assertFalse(backup.exists())

    def test_orphaned_backup_is_dropped_when_the_target_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "weights.onnx"
            target.write_bytes(b"current")
            backup = root / ("weights.onnx" + atomic_replace.BACKUP_SUFFIX)
            backup.write_bytes(b"stale")
            atomic_replace.recover_pending_replacements(root)
            self.assertEqual(target.read_bytes(), b"current")
            self.assertFalse(backup.exists())

    def test_unreadable_journal_is_discarded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bad = root / (".broken" + atomic_replace.JOURNAL_SUFFIX)
            bad.write_text("{not json", encoding="utf-8")
            actions = atomic_replace.recover_pending_replacements(root)
            self.assertEqual(actions[0]["action"], "discarded")
            self.assertFalse(bad.exists())


def _make_bundle(path: Path, payloads: dict) -> None:
    files = []
    with zipfile.ZipFile(path, "w") as bundle:
        for name, blob in payloads.items():
            archive_path = f"files/app-model-cache/{name}"
            bundle.writestr(archive_path, blob)
            files.append({
                "cache": "app-model-cache",
                "relative_path": name,
                "archive_path": archive_path,
                "filename": name,
                "bytes": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(),
                "known_sha256": None,
                "known_hash": False,
            })
        bundle.writestr(cache_inventory.MODEL_CACHE_MANIFEST, json.dumps({
            "schema": cache_inventory.PORTABLE_MODEL_CACHE_SCHEMA,
            "files": files,
            "skipped": [],
        }))


class ModelCacheImportTransactionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.env = {"APPDATA": str(self.root / "appdata")}
        self._patch = mock.patch.object(
            cache_inventory, "app_model_cache_dir", lambda env=None: self.cache)
        self._patch.start()
        self.addCleanup(self._patch.stop)

    def _bundle(self):
        path = self.root / "bundle.zip"
        _make_bundle(path, {"one.onnx": b"NEW-ONE", "two.onnx": b"NEW-TWO"})
        return path

    def test_oversized_manifest_is_rejected_before_reading(self):
        path = self.root / "bomb.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr(
                cache_inventory.MODEL_CACHE_MANIFEST,
                "0" * (cache_inventory._IMPORT_MAX_MANIFEST_BYTES + 1024),
            )
        with self.assertRaisesRegex(ValueError, "ceiling"):
            cache_inventory.import_model_cache_bundle(path, env=self.env)

    def test_successful_import_leaves_no_journal_or_backup(self):
        (self.cache / "one.onnx").write_bytes(b"OLD-ONE")
        cache_inventory.import_model_cache_bundle(self._bundle(), env=self.env)
        self.assertEqual((self.cache / "one.onnx").read_bytes(), b"NEW-ONE")
        self.assertEqual((self.cache / "two.onnx").read_bytes(), b"NEW-TWO")
        leftovers = [
            item.name for item in self.cache.iterdir()
            if item.name.endswith(atomic_replace.JOURNAL_SUFFIX)
            or item.name.endswith(atomic_replace.BACKUP_SUFFIX)
        ]
        self.assertEqual(leftovers, [])

    def test_failure_at_the_second_replace_restores_the_old_set(self):
        (self.cache / "one.onnx").write_bytes(b"OLD-ONE")
        (self.cache / "two.onnx").write_bytes(b"OLD-TWO")
        real_replace = os.replace
        calls = {"n": 0}

        def flaky(src, dst, *args, **kwargs):
            calls["n"] += 1
            # Fail on the third move: one.onnx has already been replaced.
            if calls["n"] == 3:
                raise OSError("injected replace failure")
            return real_replace(src, dst, *args, **kwargs)

        with mock.patch.object(cache_inventory.os, "replace", flaky):
            with self.assertRaisesRegex(ValueError, "rolled back"):
                cache_inventory.import_model_cache_bundle(
                    self._bundle(), env=self.env)

        self.assertEqual((self.cache / "one.onnx").read_bytes(), b"OLD-ONE")
        self.assertEqual((self.cache / "two.onnx").read_bytes(), b"OLD-TWO")

    def test_interrupted_import_is_recovered_at_startup(self):
        target = self.cache / "one.onnx"
        target.write_bytes(b"OLD-ONE")
        backup = atomic_replace.backup_path_for(target)
        journal = atomic_replace.ReplacementJournal(
            self.cache, "model-cache-import")
        journal.plan(target)
        journal.begin()
        os.replace(target, backup)
        # Process dies here: the target is missing and the journal is pending.
        self.assertFalse(target.exists())

        status = cache_inventory.model_cache_status(env=self.env)

        self.assertEqual(target.read_bytes(), b"OLD-ONE")
        self.assertFalse(backup.exists())
        self.assertFalse(journal.path.exists())
        self.assertTrue(status["recovered_replacements"])


class MatteExportTransactionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _paths(self):
        output = self.root / "cleaned.mp4"
        return (output,) + matte_interchange.mask_interchange_paths(
            output, "png")

    def _write(self, value=200):
        output, _artifact, _manifest = self._paths()
        writer = matte_interchange.MaskInterchangeWriter(
            output, "png", width=8, height=8, fps=24.0,
            start_frame=0, end_frame=2,
            timestamps=[0.0, 1 / 24], durations=[1 / 24, 1 / 24],
            is_vfr=False, source_time_base=1 / 24,
        )
        mask = np.full((8, 8), value, dtype=np.uint8)
        writer.write(mask)
        writer.write(mask)
        return writer

    def test_export_writes_a_matching_artifact_and_manifest(self):
        _output, artifact, manifest = self._paths()
        report = self._write().finalize()
        self.assertTrue(artifact.is_dir())
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["artifact_sha256"], report["artifact_sha256"])
        leftovers = [
            item.name for item in self.root.iterdir()
            if item.name.endswith(atomic_replace.JOURNAL_SUFFIX)
            or item.name.endswith(atomic_replace.BACKUP_SUFFIX)
        ]
        self.assertEqual(leftovers, [])

    def test_manifest_write_failure_restores_the_previous_pair(self):
        _output, artifact, manifest = self._paths()
        first = self._write(value=200).finalize()
        old_manifest = manifest.read_text(encoding="utf-8")
        old_frames = sorted(item.name for item in artifact.iterdir())

        writer = self._write(value=99)
        with mock.patch.object(
            matte_interchange, "_write_text_atomic",
            side_effect=OSError("injected manifest failure"),
        ):
            with self.assertRaises(OSError):
                writer.finalize()

        self.assertTrue(artifact.is_dir())
        self.assertEqual(
            sorted(item.name for item in artifact.iterdir()), old_frames)
        self.assertEqual(manifest.read_text(encoding="utf-8"), old_manifest)
        self.assertEqual(
            json.loads(old_manifest)["artifact_sha256"],
            first["artifact_sha256"],
        )
        self.assertEqual(
            matte_interchange._sha256_sequence(artifact, 2),
            first["artifact_sha256"],
        )

    def test_artifact_replace_failure_leaves_the_previous_pair(self):
        _output, artifact, manifest = self._paths()
        first = self._write(value=200).finalize()
        old_manifest = manifest.read_text(encoding="utf-8")

        writer = self._write(value=99)
        real_replace = os.replace
        calls = {"n": 0}

        def flaky(src, dst, *args, **kwargs):
            calls["n"] += 1
            # First call moves the old artifact to its backup; fail the move
            # that would put the new artifact in place.
            if calls["n"] == 2:
                raise OSError("injected artifact promotion failure")
            return real_replace(src, dst, *args, **kwargs)

        with mock.patch.object(matte_interchange.os, "replace", flaky):
            with self.assertRaises(OSError):
                writer.finalize()

        self.assertTrue(artifact.is_dir())
        self.assertEqual(manifest.read_text(encoding="utf-8"), old_manifest)
        self.assertEqual(
            matte_interchange._sha256_sequence(artifact, 2),
            first["artifact_sha256"],
        )

    def test_interrupted_export_is_recoverable(self):
        _output, artifact, manifest = self._paths()
        first = self._write(value=200).finalize()
        old_manifest = manifest.read_text(encoding="utf-8")

        journal = atomic_replace.ReplacementJournal(
            self.root, f"matte-export-{artifact.name}")
        backup = journal.plan(artifact)
        journal.plan(manifest)
        journal.begin()
        os.replace(artifact, backup)
        # Crash here: the artifact is gone and the manifest still describes it.
        self.assertFalse(artifact.exists())

        atomic_replace.recover_pending_replacements(self.root)

        self.assertTrue(artifact.is_dir())
        self.assertEqual(manifest.read_text(encoding="utf-8"), old_manifest)
        self.assertEqual(
            matte_interchange._sha256_sequence(artifact, 2),
            first["artifact_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
