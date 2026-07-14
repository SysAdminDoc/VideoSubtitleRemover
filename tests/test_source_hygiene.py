import ast
from pathlib import Path
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
            "-m ruff check backend gui VideoSubtitleRemover.py --no-cache",
            build_script,
        )

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
