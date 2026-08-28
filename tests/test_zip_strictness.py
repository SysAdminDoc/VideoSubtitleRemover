"""RM-323: a frames-to-masks length mismatch must raise, not truncate.

61 `zip()` calls carried no `strict=`, including the inpainting hot path, so
a mismatch processed the shorter prefix, left the remaining frames untouched
in the output, and reported nothing. This is the same silent-degradation
class the truthful-stage-execution work was written to eliminate.
"""

from __future__ import annotations

import ast
import tomllib
import unittest
from pathlib import Path

import numpy as np

from backend.config import ProcessingConfig
from backend.inpainters._common import apply_finishing
from backend.inpainters.sttn import STTNInpainter

ROOT = Path(__file__).resolve().parent.parent


def _frame(value: int = 128) -> np.ndarray:
    return np.full((16, 24, 3), value, dtype=np.uint8)


def _mask(active: bool = True) -> np.ndarray:
    mask = np.zeros((16, 24), dtype=np.uint8)
    if active:
        mask[6:10, 8:16] = 255
    return mask


class InpaintLengthMismatchTests(unittest.TestCase):
    def test_finishing_refuses_a_short_mask_list(self):
        frames = [_frame(), _frame(), _frame()]
        filled = [_frame(90), _frame(90), _frame(90)]
        with self.assertRaises(ValueError):
            apply_finishing(frames, filled, [_mask(), _mask()])

    def test_finishing_refuses_a_short_filled_list(self):
        frames = [_frame(), _frame(), _frame()]
        with self.assertRaises(ValueError):
            apply_finishing(frames, [_frame(90)], [_mask(), _mask(), _mask()])

    def test_the_cv2_fallback_refuses_a_mismatch_instead_of_truncating(self):
        config = ProcessingConfig()
        config.tbe_enable = False
        inpainter = STTNInpainter("cpu", config)

        # The matched call still works, so the raise below is about the
        # mismatch and not about the fixture.
        matched = inpainter.inpaint([_frame(), _frame()], [_mask(), _mask()])
        self.assertEqual(len(matched), 2)

        with self.assertRaises(ValueError):
            inpainter.inpaint([_frame(), _frame(), _frame()],
                              [_mask(), _mask()])

    def test_the_temporal_path_refuses_a_mismatch(self):
        config = ProcessingConfig()
        config.tbe_enable = True
        inpainter = STTNInpainter("cpu", config)

        matched = inpainter.inpaint(
            [_frame(), _frame(120), _frame(140)],
            [_mask(), _mask(), _mask()],
        )
        self.assertEqual(len(matched), 3)

        with self.assertRaises(ValueError):
            inpainter.inpaint(
                [_frame(), _frame(120), _frame(140)], [_mask(), _mask()])


class ZipStrictnessGateTests(unittest.TestCase):
    def test_b905_is_part_of_the_lint_baseline(self):
        config = tomllib.loads(
            (ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertIn("B905", config["tool"]["ruff"]["lint"]["select"])

    def test_every_lenient_pairing_says_why(self):
        """A strict=False must carry a comment, or it reads as an oversight."""
        offenders = []
        # Anchor on the repo, not the cwd: relative roots make this scan
        # zero files and pass vacuously when pytest runs from elsewhere.
        for root in (ROOT / "backend", ROOT / "gui", ROOT / "scripts",
                     ROOT / "tools"):
            self.assertTrue(root.is_dir(), root)
            for path in sorted(root.rglob("*.py")):
                lines = path.read_text(encoding="utf-8").split("\n")
                tree = ast.parse("\n".join(lines), filename=str(path))
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    func = node.func
                    if not isinstance(func, ast.Name) or func.id != "zip":
                        continue
                    lenient = any(
                        keyword.arg == "strict"
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is False
                        for keyword in node.keywords
                    )
                    if not lenient:
                        continue
                    # Look for a comment in the few lines above the call.
                    window = lines[max(0, node.lineno - 6):node.lineno]
                    if not any(
                            "strict=False" in line and line.lstrip().startswith("#")
                            for line in window):
                        offenders.append(f"{path.name}:{node.lineno}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
