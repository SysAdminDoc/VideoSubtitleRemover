from __future__ import annotations

import gettext
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from backend import i18n
from gui.config import ProcessingConfig
from scripts import i18n_catalogs


ROOT = Path(__file__).resolve().parents[1]


class I18nCatalogLifecycleTests(unittest.TestCase):
    def tearDown(self):
        i18n.bind_locale(None)

    def test_catalog_check_and_coverage_command(self):
        result = subprocess.run(
            [sys.executable, "scripts/i18n_catalogs.py", "check"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        self.assertIn("qps-Ploc:", result.stdout)
        self.assertIn("100.0%", result.stdout)

    def test_pseudo_catalog_loads_and_preserves_placeholders(self):
        self.assertIn("qps-Ploc", i18n.available_catalogs())
        self.assertEqual(i18n.bind_locale("qps_ploc"), "qps-Ploc")
        source = "Moving manual regions: {count} track{suffix}"
        translated = i18n.tr(source)
        self.assertNotEqual(translated, source)
        self.assertIn("{count}", translated)
        self.assertIn("{suffix}", translated)
        self.assertIn("3", translated.format(count=3, suffix="s"))

    def test_compiled_catalog_is_valid_gnu_mo(self):
        entries = i18n_catalogs.pseudo_entries({
            "Start batch": i18n_catalogs.Message("Start batch"),
            "{count} file": i18n_catalogs.Message(
                "{count} file", "{count} files"
            ),
        })
        catalog = gettext.GNUTranslations(io.BytesIO(i18n_catalogs.compile_mo(entries)))
        self.assertNotEqual(catalog.gettext("Start batch"), "Start batch")
        self.assertEqual(catalog.gettext("Missing key"), "Missing key")
        self.assertIn(
            "{count}", catalog.ngettext("{count} file", "{count} files", 2)
        )

    def test_placeholder_and_plural_validation_fail_closed(self):
        header = i18n_catalogs.PoEntry(
            "",
            msgstr={0: i18n_catalogs._header("fr")},
        )
        bad_placeholder = i18n_catalogs.PoEntry(
            "Hello {name}",
            msgstr={0: "Bonjour"},
        )
        bad_plural = i18n_catalogs.PoEntry(
            "{count} file",
            msgid_plural="{count} files",
            msgstr={0: "{count} fichier"},
        )
        with self.assertRaisesRegex(ValueError, "placeholder mismatch"):
            i18n_catalogs.validate_po(Path("placeholder.po"), [header, bad_placeholder])
        with self.assertRaisesRegex(ValueError, "needs msgstr indices"):
            i18n_catalogs.validate_po(Path("plural.po"), [header, bad_plural])

    def test_invalid_utf8_catalog_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.po"
            path.write_bytes(b'msgid ""\nmsgstr ""\n"charset=UTF-8\\n"\n\xff')
            with self.assertRaises(UnicodeDecodeError):
                i18n_catalogs.parse_po(path)

    def test_frozen_root_and_language_fallback_find_same_catalog(self):
        with mock.patch.object(sys, "_MEIPASS", str(ROOT), create=True):
            roots = i18n._candidate_locale_dirs()
            self.assertIn((ROOT / "locale").resolve(), [root.resolve() for root in roots])
            self.assertIn("qps-Ploc", i18n.available_catalogs())
        self.assertEqual(i18n.locale_fallback_chain("qps_PlOC"), ("qps-Ploc", "qps"))

    def test_locale_preference_round_trips_and_normalizes(self):
        config = ProcessingConfig(ui_locale="pt_br").normalized()
        self.assertEqual(config.ui_locale, "pt-BR")
        restored = ProcessingConfig.from_dict(config.to_dict())
        self.assertEqual(restored.ui_locale, "pt-BR")
        self.assertEqual(ProcessingConfig(ui_locale="English").normalized().ui_locale, "en")
        self.assertEqual(ProcessingConfig(ui_locale="bad/tag").normalized().ui_locale, "system")

    def test_release_build_checks_and_packages_catalogs(self):
        build = (ROOT / "build_exe.bat").read_text(encoding="ascii")
        self.assertIn("scripts\\i18n_catalogs.py check", build)
        self.assertIn('if exist "locale" set "DATA_ARGS=%DATA_ARGS% --add-data locale;locale"', build)
        self.assertIn('set "VSR_SMOKE_LOCALE=qps-Ploc"', build)
        installer = (ROOT / "installer" / "vsr.nsi").read_text(encoding="utf-8")
        self.assertIn('File /r "${DIST_DIR}\\*.*"', installer)


if __name__ == "__main__":
    unittest.main()
