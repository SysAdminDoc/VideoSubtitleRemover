"""RM-144: user-state writes are observable and downgrade-safe."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gui import config as gcfg


class PersistenceResultTests(unittest.TestCase):
    def test_failure_message_carries_retry_guidance(self):
        result = gcfg.PersistenceResult(
            ok=False, kind=gcfg.PERSIST_SETTINGS,
            path="C:/profile/settings.json", error="No space left on device")
        self.assertFalse(result)
        message = result.message()
        self.assertIn("Could not save settings", message)
        self.assertIn("No space left on device", message)
        self.assertIn("free disk space", message)
        self.assertIn("C:/profile/settings.json", message)

    def test_read_only_message_points_at_the_export_path(self):
        result = gcfg.PersistenceResult(
            ok=False, kind=gcfg.PERSIST_SETTINGS, read_only=True)
        self.assertIn("read-only", result.message())
        self.assertIn("Export", result.message())

    def test_success_is_truthy(self):
        self.assertTrue(gcfg.PersistenceResult(ok=True, kind=gcfg.PERSIST_QUEUE))


class SaveOutcomeTests(unittest.TestCase):
    def setUp(self):
        gcfg.allow_settings_overwrite()
        gcfg.consume_settings_load_notice()
        gcfg.consume_preset_import_notice()
        self.seen = []
        gcfg.set_persistence_observer(self.seen.append)
        self.addCleanup(gcfg.set_persistence_observer, None)
        self.addCleanup(gcfg.allow_settings_overwrite)

    def test_settings_save_reports_success_and_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            with mock.patch.object(gcfg, "SETTINGS_FILE", str(path)):
                ok = gcfg.save_settings(gcfg.ProcessingConfig())
                self.assertTrue(ok)
                self.assertTrue(path.is_file())
                self.assertEqual(self.seen, [])

                with mock.patch.object(
                    gcfg, "_write_json_atomic",
                    side_effect=OSError("disk full"),
                ):
                    failed = gcfg.save_settings(gcfg.ProcessingConfig())
        self.assertFalse(failed)
        self.assertEqual(failed.kind, gcfg.PERSIST_SETTINGS)
        self.assertIn("disk full", failed.error)
        self.assertEqual(len(self.seen), 1)
        self.assertIn("Could not save settings", self.seen[0].message())

    def test_preset_save_failure_is_reported_and_not_claimed(self):
        with mock.patch.object(
            gcfg, "_write_json_atomic", side_effect=OSError("locked")
        ):
            with mock.patch.object(gcfg, "_load_user_presets", return_value={}):
                saved = gcfg.save_user_preset(
                    "unit", "desc", gcfg.ProcessingConfig())
        self.assertFalse(saved)
        self.assertEqual(len(self.seen), 1)
        self.assertEqual(self.seen[0].kind, gcfg.PERSIST_PRESETS)

    def test_queue_save_failure_is_reported(self):
        class _Item:
            id = "x"
            file_path = "a.mp4"
            output_path = "b.mp4"
            output_path_locked = False
            config = None
            status = gcfg.ProcessingStatus.IDLE
            progress = 0.0
            message = ""
            error = None

        with mock.patch.object(
            gcfg, "_write_json_atomic", side_effect=OSError("read-only fs")
        ):
            result = gcfg.save_queue_state([_Item()])
        self.assertFalse(result)
        self.assertEqual(result.kind, gcfg.PERSIST_QUEUE)
        self.assertEqual(len(self.seen), 1)


class CorruptBackupHonestyTests(unittest.TestCase):
    def setUp(self):
        gcfg.allow_settings_overwrite()
        gcfg.consume_settings_load_notice()
        self.addCleanup(gcfg.allow_settings_overwrite)

    def test_backup_claim_only_after_the_copy_succeeds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            path.write_text("{not json", encoding="utf-8")
            with mock.patch.object(gcfg, "SETTINGS_FILE", str(path)):
                gcfg.load_settings()
            notice = gcfg.consume_settings_load_notice()
            self.assertIn("backup was saved", notice)
            self.assertTrue(path.with_suffix(".json.bak").is_file())

    def test_failed_backup_is_reported_honestly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            path.write_text("{not json", encoding="utf-8")
            with mock.patch.object(gcfg, "SETTINGS_FILE", str(path)):
                with mock.patch(
                    "shutil.copy2", side_effect=OSError("denied")
                ):
                    gcfg.load_settings()
            notice = gcfg.consume_settings_load_notice()
        self.assertIn("could not be written", notice)
        self.assertNotIn("backup was saved", notice)


class ForwardCompatibilityTests(unittest.TestCase):
    def setUp(self):
        gcfg.allow_settings_overwrite()
        gcfg.consume_settings_load_notice()
        self.addCleanup(gcfg.allow_settings_overwrite)

    def _future_settings(self, tmpdir):
        path = Path(tmpdir) / "settings.json"
        payload = gcfg.ProcessingConfig().to_dict()
        payload["vsr_settings_format"] = gcfg.VSR_SETTINGS_FORMAT + 5
        payload["a_field_this_build_does_not_know"] = "keep me"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def test_future_schema_opens_read_only_and_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._future_settings(tmpdir)
            original = path.read_bytes()
            with mock.patch.object(gcfg, "SETTINGS_FILE", str(path)):
                config = gcfg.load_settings()
                self.assertIsNotNone(gcfg.settings_read_only_version())
                notice = gcfg.consume_settings_load_notice()
                self.assertIn("read-only", notice)

                result = gcfg.save_settings(config)
                self.assertFalse(result)
                self.assertTrue(result.read_only)
                self.assertEqual(path.read_bytes(), original)

                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(
                    payload["a_field_this_build_does_not_know"], "keep me")

    def test_export_copy_is_the_deliberate_way_out(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._future_settings(tmpdir)
            copy_path = Path(tmpdir) / "downgraded.json"
            with mock.patch.object(gcfg, "SETTINGS_FILE", str(path)):
                config = gcfg.load_settings()
                result = gcfg.export_settings_copy(config, copy_path)
                self.assertTrue(result)
                # The original is still untouched and still read-only.
                self.assertIsNotNone(gcfg.settings_read_only_version())
                self.assertFalse(gcfg.save_settings(config))
            payload = json.loads(copy_path.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["vsr_settings_format"], gcfg.VSR_SETTINGS_FORMAT)
        self.assertNotIn("a_field_this_build_does_not_know", payload)

    def test_explicit_overwrite_clears_read_only_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._future_settings(tmpdir)
            with mock.patch.object(gcfg, "SETTINGS_FILE", str(path)):
                config = gcfg.load_settings()
                gcfg.allow_settings_overwrite()
                self.assertIsNone(gcfg.settings_read_only_version())
                self.assertTrue(gcfg.save_settings(config))
                payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["vsr_settings_format"], gcfg.VSR_SETTINGS_FORMAT)

    def test_current_schema_stays_writable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            path.write_text(
                json.dumps(gcfg.ProcessingConfig().to_dict()), encoding="utf-8")
            with mock.patch.object(gcfg, "SETTINGS_FILE", str(path)):
                config = gcfg.load_settings()
                self.assertIsNone(gcfg.settings_read_only_version())
                self.assertTrue(gcfg.save_settings(config))


if __name__ == "__main__":
    unittest.main()
