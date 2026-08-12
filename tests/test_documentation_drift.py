"""RM-151: published architecture, release, and accessibility claims are true."""

from pathlib import Path
import re
import sys
import unittest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.generate_architecture_map import (
    apply as apply_map,
    undocumented_modules,
)


ARCHITECTURE = _ROOT / "docs" / "architecture.md"
CHANGELOG = _ROOT / "CHANGELOG.md"
README = _ROOT / "README.md"


class ArchitectureMapTests(unittest.TestCase):
    def test_every_module_is_described(self):
        self.assertEqual(undocumented_modules(), [])

    def test_module_map_matches_the_live_tree(self):
        self.assertEqual(
            apply_map(check=True), 0,
            "run scripts/generate_architecture_map.py",
        )

    def test_map_lists_every_gui_and_backend_module(self):
        text = ARCHITECTURE.read_text(encoding="utf-8")
        block = text.split("<!-- module-map:start -->", 1)[1]
        block = block.split("<!-- module-map:end -->", 1)[0]
        listed = set(re.findall(r"([A-Za-z0-9_]+\.py)", block))
        for directory in ("gui", "backend", "backend/inpainters"):
            for path in (_ROOT / directory).glob("*.py"):
                with self.subTest(module=f"{directory}/{path.name}"):
                    self.assertIn(path.name, listed)

    def test_map_does_not_list_files_that_no_longer_exist(self):
        text = ARCHITECTURE.read_text(encoding="utf-8")
        block = text.split("<!-- module-map:start -->", 1)[1]
        block = block.split("<!-- module-map:end -->", 1)[0]
        real = {
            path.name
            for directory in ("gui", "backend", "backend/inpainters", ".")
            for path in (_ROOT / directory).glob("*.py")
        }
        for name in re.findall(r"([A-Za-z0-9_]+\.py)", block):
            with self.subTest(listed=name):
                self.assertIn(name, real)


class ChangelogStructureTests(unittest.TestCase):
    def test_exactly_one_unreleased_section(self):
        text = CHANGELOG.read_text(encoding="utf-8")
        self.assertEqual(text.count("\n## [Unreleased]"), 1)

    def test_unreleased_is_the_first_section(self):
        text = CHANGELOG.read_text(encoding="utf-8")
        headings = re.findall(r"^## \[(.+?)\]", text, flags=re.MULTILINE)
        self.assertEqual(headings[0], "Unreleased")


class UnsignedDistributionWordingTests(unittest.TestCase):
    # The lookbehind keeps "unsigned release" -- the correct wording -- from
    # tripping the "signed release" rule.
    FORBIDDEN = (
        r"(?<!un)signed release",
        r"(?<!un)signed installer",
        r"(?<!un)signed build",
        r"code[- ]signing",
        r"authenticode",
        r"notariz",
    )

    def test_no_document_claims_a_signed_release(self):
        for path in (README, ARCHITECTURE):
            text = path.read_text(encoding="utf-8").lower()
            for pattern in self.FORBIDDEN:
                with self.subTest(document=path.name, pattern=pattern):
                    # "unsigned release" must not trip the "signed release" rule.
                    self.assertIsNone(
                        re.search(pattern, text),
                        f"{path.name} claims a signed distribution",
                    )

    def test_readme_states_the_build_is_unsigned(self):
        text = README.read_text(encoding="utf-8")
        self.assertIn("The build is unsigned", text)
        self.assertIn("SHA256SUMS.txt", text)


class OfflineGuaranteeDocumentationTests(unittest.TestCase):
    def test_readme_names_local_processing_and_network_controls(self):
        text = README.read_text(encoding="utf-8")
        for phrase in (
            "All media processing is local",
            "No account, subscription, or upload is required",
            "opt-in GitHub update check",
            "opt-in crash report",
            "`update_check` setting is `false`",
            "`VSR_GLITCHTIP_DSN`",
            "`VSR_CRASH_REPORTS=1`",
            "`VSR_CRASH_REPORTS=0`",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)

    def test_readme_controls_match_network_modules(self):
        update = (_ROOT / "backend" / "update_check.py").read_text(
            encoding="utf-8"
        )
        crash = (_ROOT / "backend" / "crash_reporter.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("Off by default", update)
        self.assertIn("GITHUB_LATEST_API", update)
        self.assertIn("VSR_GLITCHTIP_DSN", crash)
        self.assertIn("VSR_CRASH_REPORTS", crash)
        self.assertIn("and bool(os.environ.get(\"VSR_GLITCHTIP_DSN\"", crash)


class AccessibilitySupportMatrixTests(unittest.TestCase):
    def test_matrix_separates_tested_from_unsupported_surfaces(self):
        text = ARCHITECTURE.read_text(encoding="utf-8")
        self.assertIn("## Accessibility support matrix", text)
        matrix = text.split("## Accessibility support matrix", 1)[1]
        for tested in (
            "Keyboard reachability",
            "Text scaling 100-200%",
            "High-contrast theme",
            "Pseudo-locale",
            "RTL mirroring",
        ):
            with self.subTest(row=tested):
                self.assertIn(tested, matrix)
        self.assertIn("Not supported", matrix)
        self.assertIn("UI Automation", matrix)

    def test_matrix_points_at_real_proof_files(self):
        matrix = ARCHITECTURE.read_text(encoding="utf-8").split(
            "## Accessibility support matrix", 1)[1]
        for reference in (
            "tools/ui_scaling_probe.py",
            "tests/test_text_scaling.py",
            "gui/dialog_layout.py",
            "scripts/i18n_catalogs.py",
            "backend/a11y.py",
        ):
            with self.subTest(reference=reference):
                self.assertIn(reference, matrix)
                self.assertTrue((_ROOT / reference).is_file())


class BuildGateTests(unittest.TestCase):
    def test_release_build_checks_the_architecture_map(self):
        script = (_ROOT / "build_exe.bat").read_text(encoding="utf-8")
        self.assertIn("generate_architecture_map.py --check", script)


if __name__ == "__main__":
    unittest.main()
