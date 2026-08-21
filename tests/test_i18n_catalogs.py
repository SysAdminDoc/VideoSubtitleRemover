from __future__ import annotations

import gettext
import io
import os
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

    def test_no_user_visible_string_bypasses_the_catalog(self):
        """RM-152: the runtime-string gate over the real GUI tree."""
        findings = i18n_catalogs.untranslated_literals()
        self.assertEqual(findings, [], "\n".join(
            f"{path}:{line} [{sink}] {literal!r}"
            for path, line, sink, literal in findings
        ))

    def test_the_lint_catches_each_kind_of_caption_sink(self):
        import ast

        cases = {
            "captions": 'tk.Label(root, text="Start the batch")',
            "dialog titles": 'filedialog.askopenfilename(title="Pick a file")',
            "positional captions": 'widget.set_text("Start the batch")',
            "file filters": 'ask(filetypes=[("Video files", "*.mp4")])',
            "interpolated sentences": 'label.config(text=f"{n} of {t} shown")',
            "config caption": 'label.config(text="Detection finished")',
            "configure caption": 'label.configure(text="Detection finished")',
            "configure interpolation":
                'label.configure(text=f"{n} regions found")',
            "menu entries": 'menu.add_command(label="Sort by name")',
            "deferred status": 'self._update_status("A new status")',
            "deferred card title": 'self._card_header(root, "Card", "A new card")',
            "deferred slider label": 'self._create_slider(root, "A new slider", 0, 1, 0, "x")',
            "deferred slider hint": 'self._create_slider(root, "Slider", 0, 1, 0, "x", hint="A new hint")',
        }
        for name, source in cases.items():
            with self.subTest(sink=name):
                self.assertTrue(
                    self._lint_flags(ast.parse(source)),
                    f"{name} sink was not detected",
                )

    def test_the_lint_accepts_translated_and_non_prose_values(self):
        import ast

        cases = {
            "wrapped": 'tk.Label(root, text=tr("Start the batch"))',
            "wrapped and formatted":
                'label.config(text=tr("{n} shown").format(n=n))',
            "wrapped configure":
                'label.configure(text=tr("Detection finished"))',
            "wrapped plural":
                'label.config(text=ntr("{n} item", "{n} items", n))',
            "wrapped filter": 'ask(filetypes=[(tr("Video files"), "*.mp4")])',
            "option value": 'widget.config(text="", state="disabled")',
            "widget anchor token": 'tk.Label(root, text="", anchor="nw")',
            "model record": 'QueueItem(message="Ready to process")',
            "dynamic value": 'label.config(text=some_variable)',
            "deferred marker": 'self._update_status(N_("Marked status"))',
        }
        for name, source in cases.items():
            with self.subTest(value=name):
                self.assertFalse(
                    self._lint_flags(ast.parse(source)),
                    f"{name} was flagged but should not be",
                )

    @staticmethod
    def _lint_flags(tree) -> bool:
        """Run the lint's scan over a parsed snippet in a temp package."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "gui"
            package.mkdir()
            import ast as _ast

            (package / "probe.py").write_text(
                _ast.unparse(tree), encoding="utf-8")
            return bool(i18n_catalogs.untranslated_literals(root))

    def test_deferred_marker_strings_are_extracted(self):
        # `N_()` marks a literal that a later `tr()` call will translate
        # from a variable, so it must still reach the catalog.
        messages = i18n_catalogs.extract_messages()
        self.assertIn("Ready to process", messages)
        self.assertIn("Checking embedded subtitle tracks...", messages)
        self.assertIn("Hardware detected: {}; {}", messages)

    def test_pseudo_catalog_loads_and_preserves_placeholders(self):
        with mock.patch.dict(
            "os.environ", {"VSR_PSEUDO_LOCALE": "1"}, clear=False
        ):
            self.assertIn("qps-Ploc", i18n.available_catalogs())
        self.assertEqual(i18n.bind_locale("qps_ploc"), "qps-Ploc")
        source = "Moving manual regions: {count} track"
        translated = i18n.tr(source)
        self.assertNotEqual(translated, source)
        self.assertIn("{count}", translated)
        self.assertIn("3", translated.format(count=3))
        # The plural form is a separate catalog entry, not an English "s"
        # glued onto the singular by the caller.
        plural = i18n.ntr(source, "Moving manual regions: {count} tracks", 3)
        self.assertNotEqual(plural, translated)
        self.assertIn("{count}", plural)

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
            with mock.patch.dict(
                "os.environ", {"VSR_PSEUDO_LOCALE": "1"}, clear=False
            ):
                self.assertIn("qps-Ploc", i18n.available_catalogs())
        self.assertEqual(i18n.locale_fallback_chain("qps_PlOC"), ("qps-Ploc", "qps"))

    def test_pseudo_catalog_is_hidden_from_end_users_by_default(self):
        with mock.patch.dict(
            "os.environ", {"VSR_PSEUDO_LOCALE": ""}, clear=False
        ):
            self.assertNotIn("qps-Ploc", i18n.available_catalogs())

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
        self.assertIn('set "VSR_SMOKE_LOCALE=qps-Ploc"', build)
        # The catalogs are bundled by the spec, which is what builds the
        # artifact; the build script's old DATA_ARGS copy was never read.
        spec = (ROOT / "VideoSubtitleRemoverPro.spec").read_text(encoding="utf-8")
        self.assertIn("('locale', 'locale')", spec)
        installer = (ROOT / "installer" / "vsr.nsi").read_text(encoding="utf-8")
        self.assertIn('File /r "${DIST_DIR}\\*.*"', installer)



class CatalogCoverageGateTests(unittest.TestCase):
    """RM-284: the picker must not advertise a language the build cannot show."""

    def setUp(self):
        from backend import i18n

        self._saved_dirs = i18n._candidate_locale_dirs
        i18n._coverage_cache.clear()

    def tearDown(self):
        from backend import i18n

        i18n._candidate_locale_dirs = self._saved_dirs
        i18n._coverage_cache.clear()

    @staticmethod
    def _write_catalog(root: Path, tag: str, translated: int, total: int):
        """Write a compiled catalog whose stamped coverage is translated/total."""
        entries = [
            i18n_catalogs.PoEntry(
                msgid="", msgstr={0: i18n_catalogs._header(tag)})
        ]
        for index in range(total):
            entries.append(i18n_catalogs.PoEntry(
                msgid=f"string {index}",
                msgstr={0: f"traduit {index}" if index < translated else ""},
            ))
        percent = i18n_catalogs.catalog_coverage_percent(entries)
        target = root / tag / "LC_MESSAGES"
        target.mkdir(parents=True, exist_ok=True)
        (target / "vsr.mo").write_bytes(
            i18n_catalogs.compile_mo(entries, percent))
        return percent

    def _catalogs_in(self, root: Path):
        from backend import i18n

        i18n._candidate_locale_dirs = lambda: [root]
        i18n._coverage_cache.clear()
        return i18n.available_catalogs()

    def test_a_thin_catalog_is_hidden_from_the_picker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            percent = self._write_catalog(root, "fr", translated=40, total=100)
            self.assertLess(percent, 90.0)
            self.assertEqual(self._catalogs_in(root), ())

    def test_a_covered_catalog_is_offered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_catalog(root, "fr", translated=95, total=100)
            self.assertEqual(self._catalogs_in(root), ("fr",))

    def test_a_catalog_at_the_bar_exactly_is_offered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            percent = self._write_catalog(root, "fr", translated=90, total=100)
            self.assertEqual(percent, 90.0)
            self.assertEqual(self._catalogs_in(root), ("fr",))

    def test_an_unstamped_catalog_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = [
                i18n_catalogs.PoEntry(
                    msgid="", msgstr={0: i18n_catalogs._header("fr")}),
                i18n_catalogs.PoEntry(msgid="one", msgstr={0: "un"}),
            ]
            target = root / "fr" / "LC_MESSAGES"
            target.mkdir(parents=True)
            (target / "vsr.mo").write_bytes(i18n_catalogs.compile_mo(entries))
            self.assertEqual(self._catalogs_in(root), ())

    def test_the_pseudo_locale_is_exempt_from_the_bar(self):
        from backend import i18n

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_catalog(root, "qps-Ploc", translated=1, total=100)
            i18n._candidate_locale_dirs = lambda: [root]
            i18n._coverage_cache.clear()
            with mock.patch.dict(
                os.environ, {"VSR_PSEUDO_LOCALE": "1"}, clear=False
            ):
                self.assertEqual(i18n.available_catalogs(), ("qps-Ploc",))

    def test_a_thin_install_catalog_is_not_unhidden_by_a_fuller_user_copy(self):
        """bind_locale takes the first root, so the gate must judge that one."""
        from backend import i18n

        with tempfile.TemporaryDirectory() as tmp_a, \
                tempfile.TemporaryDirectory() as tmp_b:
            install = Path(tmp_a)
            per_user = Path(tmp_b)
            self._write_catalog(install, "fr", translated=40, total=100)
            self._write_catalog(per_user, "fr", translated=98, total=100)
            i18n._candidate_locale_dirs = lambda: [install, per_user]
            i18n._coverage_cache.clear()
            self.assertEqual(i18n.available_catalogs(), ())

    def test_a_fuller_install_catalog_still_qualifies(self):
        from backend import i18n

        with tempfile.TemporaryDirectory() as tmp_a, \
                tempfile.TemporaryDirectory() as tmp_b:
            install = Path(tmp_a)
            per_user = Path(tmp_b)
            self._write_catalog(install, "fr", translated=98, total=100)
            self._write_catalog(per_user, "fr", translated=10, total=100)
            i18n._candidate_locale_dirs = lambda: [install, per_user]
            i18n._coverage_cache.clear()
            self.assertEqual(i18n.available_catalogs(), ("fr",))

    def test_bind_locale_refuses_a_thin_catalog_and_falls_back_to_english(self):
        """The gate is not a picker cosmetic: the system locale honours it too."""
        from backend import i18n

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_catalog(root, "fr", translated=40, total=100)
            i18n._candidate_locale_dirs = lambda: [root]
            i18n._coverage_cache.clear()
            try:
                self.assertEqual(i18n.bind_locale("fr"), "en")
            finally:
                i18n.bind_locale(None)

    def test_bind_locale_accepts_a_catalog_that_clears_the_bar(self):
        from backend import i18n

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_catalog(root, "fr", translated=95, total=100)
            i18n._candidate_locale_dirs = lambda: [root]
            i18n._coverage_cache.clear()
            try:
                self.assertEqual(i18n.bind_locale("fr"), "fr")
            finally:
                i18n.bind_locale(None)

    def test_the_shipped_catalog_carries_a_coverage_stamp(self):
        from backend import i18n

        stamped = ROOT / "locale" / "qps-Ploc" / "LC_MESSAGES" / "vsr.mo"
        self.assertTrue(stamped.is_file())
        self.assertIsNotNone(
            i18n.catalog_coverage(stamped),
            "compile must stamp coverage; without it every locale is hidden",
        )


if __name__ == "__main__":
    unittest.main()
