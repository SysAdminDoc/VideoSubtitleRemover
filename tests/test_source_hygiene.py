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


if __name__ == "__main__":
    unittest.main()
