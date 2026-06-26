from pathlib import Path
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
