import json
from pathlib import Path
import tempfile
import unittest

import gui.config as gui_config


class PresetSchemaTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old_presets_file = gui_config.PRESETS_FILE
        gui_config.PRESETS_FILE = Path(self._tmp.name) / "presets.json"
        gui_config.consume_preset_import_notice()

    def tearDown(self):
        gui_config.PRESETS_FILE = self._old_presets_file
        gui_config.consume_preset_import_notice()
        self._tmp.cleanup()

    def _write_import(self, fields):
        path = Path(self._tmp.name) / "incoming.vsr-preset.json"
        path.write_text(
            json.dumps({
                "name": "Shared review preset",
                "description": "Imported",
                "fields": fields,
                "vsr_preset_format": 1,
            }),
            encoding="utf-8",
        )
        return path

    def test_import_preset_filters_unsupported_fields(self):
        path = self._write_import({
            "mask_dilate_px": 12,
            "update_check": True,
            "whisper_model_path": "C:/Users/example/model.bin",
            "window_geometry": "1800x1400+0+0",
        })

        name = gui_config.import_preset(str(path))

        self.assertEqual(name, "Shared review preset")
        stored = json.loads(gui_config.PRESETS_FILE.read_text(encoding="utf-8"))
        fields = stored[name]["fields"]
        self.assertEqual(fields, {"mask_dilate_px": 12})
        notice = gui_config.consume_preset_import_notice()
        self.assertIn("update_check", notice)
        self.assertIn("whisper_model_path", notice)
        self.assertIn("window_geometry", notice)

    def test_import_rejects_presets_without_supported_fields(self):
        path = self._write_import({
            "update_check": True,
            "window_geometry": "1800x1400+0+0",
        })

        self.assertIsNone(gui_config.import_preset(str(path)))
        self.assertFalse(gui_config.PRESETS_FILE.exists())

    def test_import_rejects_oversized_preset_before_parsing(self):
        path = Path(self._tmp.name) / "oversized.vsr-preset.json"
        path.write_text(
            " " * (gui_config.MAX_JSON_OBJECT_BYTES + 1),
            encoding="utf-8",
        )

        with self.assertLogs("gui.config", level="WARNING") as caught:
            self.assertIsNone(gui_config.import_preset(str(path)))

        self.assertFalse(gui_config.PRESETS_FILE.exists())
        self.assertIn("file is too large", "\n".join(caught.output))

    def test_apply_preset_ignores_unsupported_existing_fields(self):
        gui_config.PRESETS_FILE.write_text(
            json.dumps({
                "Legacy": {
                    "description": "Old local payload",
                    "fields": {
                        "mask_feather_px": 9,
                        "update_check": True,
                        "window_geometry": "1x1+0+0",
                    },
                },
            }),
            encoding="utf-8",
        )
        config = gui_config.ProcessingConfig(
            update_check=False,
            window_geometry="1280x720+10+10",
        )

        self.assertTrue(gui_config.apply_preset(config, "Legacy"))

        self.assertEqual(config.mask_feather_px, 9)
        self.assertFalse(config.update_check)
        self.assertEqual(config.window_geometry, "1280x720+10+10")

    def test_save_user_preset_filters_requested_fields(self):
        config = gui_config.ProcessingConfig(update_check=True)

        ok = gui_config.save_user_preset(
            "Review",
            "Local review preset",
            config,
            fields=["mode", "mask_dilate_px", "update_check"],
        )

        self.assertTrue(ok)
        stored = json.loads(gui_config.PRESETS_FILE.read_text(encoding="utf-8"))
        fields = stored["Review"]["fields"]
        self.assertIn("mode", fields)
        self.assertIn("mask_dilate_px", fields)
        self.assertNotIn("update_check", fields)

    def test_export_preset_filters_unsupported_existing_fields(self):
        gui_config.PRESETS_FILE.write_text(
            json.dumps({
                "Legacy": {
                    "description": "Old local payload",
                    "fields": {
                        "mask_feather_px": 7,
                        "update_check": True,
                    },
                },
            }),
            encoding="utf-8",
        )
        export_path = Path(self._tmp.name) / "exported.json"

        self.assertTrue(gui_config.export_preset("Legacy", str(export_path)))

        exported = json.loads(export_path.read_text(encoding="utf-8"))
        self.assertEqual(exported["fields"], {"mask_feather_px": 7})


class IntentParserTests(unittest.TestCase):
    def test_subtitle_intent(self):
        from backend.presets import parse_intent
        result = parse_intent("remove subtitles")
        self.assertIsNotNone(result)
        self.assertTrue(result.get("remove_subtitles"))
        self.assertFalse(result.get("remove_chyrons"))

    def test_logo_intent(self):
        from backend.presets import parse_intent
        result = parse_intent("remove logo")
        self.assertIsNotNone(result)
        self.assertTrue(result.get("remove_chyrons"))

    def test_combined_intent(self):
        from backend.presets import parse_intent
        result = parse_intent("remove everything fast")
        self.assertIsNotNone(result)
        self.assertTrue(result.get("remove_subtitles"))
        self.assertTrue(result.get("remove_chyrons"))
        self.assertTrue(result.get("lama_super_fast"))

    def test_unknown_intent(self):
        from backend.presets import parse_intent
        result = parse_intent("do something random")
        self.assertIsNone(result)

    def test_empty_intent(self):
        from backend.presets import parse_intent
        self.assertIsNone(parse_intent(""))
        self.assertIsNone(parse_intent(None))


if __name__ == "__main__":
    unittest.main()
