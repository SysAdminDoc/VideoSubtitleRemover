import ast
import json
from pathlib import Path
import subprocess
import sys
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "venv",
}


def _is_excluded(path: Path) -> bool:
    try:
        relative = path.relative_to(ROOT)
    except ValueError:
        return True
    return any(part in EXCLUDED_DIRS for part in relative.parts)


def _source_files():
    for suffix in ("*.py", "*.bat"):
        for path in ROOT.rglob(suffix):
            if not _is_excluded(path) and path.is_file():
                yield path


def _doc_files():
    for rel in ("README.md", "docs/architecture.md"):
        path = ROOT / rel
        if path.is_file():
            yield path


class SourceHygieneTests(unittest.TestCase):
    def test_cli_main_is_a_small_orchestrator_with_named_phases(self):
        tree = ast.parse(
            (ROOT / "backend" / "cli.py").read_text(encoding="utf-8")
        )
        functions = {
            node.name: node for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        for name in (
            "_build_parser",
            "_handle_utility_actions",
            "_prepare_cli_args",
            "_build_processing_config",
            "_apply_cli_config_overlays",
            "_run_soft_subtitle_modes",
            "_run_processing",
        ):
            self.assertIn(name, functions)

        main = functions["main"]
        self.assertLessEqual(main.end_lineno - main.lineno + 1, 80)
        nested = [
            node for node in ast.walk(main)
            if isinstance(node, ast.FunctionDef) and node is not main
        ]
        self.assertEqual(nested, [])

    def test_settings_builder_delegates_to_named_groups(self):
        tree = ast.parse(
            (ROOT / "gui" / "app.py").read_text(encoding="utf-8")
        )
        app = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "VideoSubtitleRemoverApp"
        )
        methods = {
            node.name: node for node in app.body
            if isinstance(node, ast.FunctionDef)
        }
        for name in (
            "_build_profile_settings_group",
            "_build_workflow_settings_group",
            "_build_sttn_settings_group",
            "_build_detection_settings_group",
            "_build_output_settings_group",
            "_build_range_settings_group",
            "_build_performance_settings_groups",
            "_build_accessibility_storage_settings",
        ):
            self.assertIn(name, methods)
        entry = methods["_build_settings_section"]
        self.assertLessEqual(entry.end_lineno - entry.lineno + 1, 60)

    def test_ruff_baseline_and_release_gate_are_explicit(self):
        config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        lint = config["tool"]["ruff"]["lint"]
        self.assertEqual(lint["select"], ["E4", "E7", "E9", "F"])
        self.assertNotIn("ignore", lint)
        self.assertEqual(
            lint["per-file-ignores"],
            {
                "VideoSubtitleRemover.py": ["E402"],
                "backend/processor.py": ["E402"],
            },
        )

        build_script = (ROOT / "build_exe.bat").read_text(encoding="ascii")
        self.assertIn('"ruff==0.15.20"', build_script)
        self.assertIn(
            "-m ruff check backend gui scripts VideoSubtitleRemover.py --no-cache",
            build_script,
        )
        self.assertIn('"%PYTHON%" scripts\\generate_cli_reference.py', build_script)

    def test_generated_cli_and_config_reference_is_current(self):
        check = subprocess.run(
            [sys.executable, "scripts/generate_cli_reference.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(check.returncode, 0, check.stderr or check.stdout)

        dump = subprocess.run(
            [sys.executable, "-m", "backend.processor", "--dump-cli-reference"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(dump.returncode, 0, dump.stderr or dump.stdout)
        payload = json.loads(dump.stdout)
        self.assertEqual(payload["schema"], "vsr.cli_reference.v1")
        internal = [
            option["flags"][0]
            for option in payload["options"]
            if option["internal"]
        ]
        self.assertEqual(internal, ["--dump-cli-reference"])
        for option in payload["options"]:
            self.assertIn(option["category"], payload["categories"])
            self.assertIsInstance(option["deprecated"], bool)
            self.assertIn("default", option)
            self.assertIn("range", option)

        help_result = subprocess.run(
            [sys.executable, "-m", "backend.processor", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        for category in payload["categories"]:
            if category != "Diagnostics and automation":
                self.assertIn(category + ":", help_result.stdout)
        self.assertIn("Diagnostics and automation:", help_result.stdout)

    def test_backend_launches_only_through_subprocess_policy(self):
        policy = ROOT / "backend" / "subprocess_policy.py"
        offenders = []
        for path in sorted((ROOT / "backend").rglob("*.py")):
            if path == policy or _is_excluded(path):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    owner = node.func.value
                    if (
                        isinstance(owner, ast.Name)
                        and owner.id == "subprocess"
                        and node.func.attr in {"run", "Popen"}
                    ):
                        offenders.append(
                            f"{path.relative_to(ROOT)}:{node.lineno}"
                        )
                if isinstance(node, ast.ImportFrom) and node.module == "subprocess":
                    for alias in node.names:
                        if alias.name in {"run", "Popen"}:
                            offenders.append(
                                f"{path.relative_to(ROOT)}:{node.lineno}"
                            )

        self.assertEqual(
            offenders,
            [],
            "Raw backend child-process launch outside subprocess_policy: "
            + ", ".join(offenders),
        )

    def test_python_and_batch_sources_are_ascii_only(self):
        offenders = []
        for path in _source_files():
            data = path.read_bytes()
            for lineno, line in enumerate(data.splitlines(), 1):
                if any(byte > 0x7F for byte in line):
                    rel = path.relative_to(ROOT)
                    offenders.append(f"{rel}:{lineno}")
                    break

        self.assertEqual(
            offenders,
            [],
            "Non-ASCII bytes found in source files: " + ", ".join(offenders),
        )

    def test_language_support_docs_do_not_repeat_legacy_claim(self):
        stale_phrases = ("12-language support", "12 language support")
        offenders = []
        for path in _doc_files():
            text = path.read_text(encoding="utf-8").lower()
            for phrase in stale_phrases:
                if phrase in text:
                    offenders.append(f"{path.relative_to(ROOT)}:{phrase}")
        self.assertEqual(offenders, [])

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("52 selectable OCR language codes", readme)


if __name__ == "__main__":
    unittest.main()
