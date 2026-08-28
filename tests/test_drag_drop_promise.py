"""RM-343: the drag-and-drop promise has to match the shipped build.

The README and the welcome flow both told users to drag media into the
window. The package that provides it, `tkinterdnd2`, is commented out of
requirements, absent from the dependency profiles and from the frozen spec,
and not installed, so no shipped build could do it. The drop zone already
swapped its own copy when the import failed; nothing else did.

If a future change ships the dependency, these tests fail and the wording
goes back. That is the point: the promise and the capability move together.
"""

from __future__ import annotations

import ast
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = "tkinterdnd2"


def _ships_drag_and_drop() -> dict:
    """Where the optional drag-and-drop package is actually declared."""
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    declared = any(
        line.strip().startswith(PACKAGE)
        for line in requirements.split("\n")
    )
    manifest = json.loads(
        (ROOT / "dependency_profiles.json").read_text(encoding="utf-8"))
    in_profiles = PACKAGE in json.dumps(manifest)
    spec = (ROOT / "VideoSubtitleRemoverPro.spec").read_text(encoding="utf-8")
    return {
        "requirements": declared,
        "profiles": in_profiles,
        "spec": PACKAGE in spec,
    }


class ShippedCapabilityTests(unittest.TestCase):
    def test_the_package_is_not_shipped(self):
        """The premise. If this fails, the wording below must change too."""
        where = _ships_drag_and_drop()
        self.assertEqual(
            where, {"requirements": False, "profiles": False, "spec": False},
            "tkinterdnd2 is now declared somewhere. Drag-and-drop can be "
            "promised again, but update the README, the welcome flow and "
            "this test together.",
        )

    def test_the_import_is_still_guarded(self):
        source = (ROOT / "gui" / "widgets.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        setups = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_setup_dnd"
        ]
        self.assertEqual(len(setups), 1)
        handlers = [
            node for node in ast.walk(setups[0])
            if isinstance(node, ast.ExceptHandler)
        ]
        self.assertTrue(
            handlers, "the optional import must not be able to break startup")


class PromiseTests(unittest.TestCase):
    """No surface may promise the gesture unconditionally."""

    def test_the_readme_qualifies_every_drag_claim(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        lines = text.split("\n")
        offenders = []
        for index, line in enumerate(lines):
            if not re.search(r"\bdrag(ging|ged)?\b", line, re.IGNORECASE):
                continue
            # A claim is qualified when the package is named within the
            # paragraph that makes it.
            window = "\n".join(lines[max(0, index - 3):index + 4])
            if PACKAGE not in window:
                offenders.append(f"README.md:{index + 1}: {line.strip()}")
        self.assertEqual(
            offenders, [],
            "an unqualified drag-and-drop claim: " + "; ".join(offenders),
        )

    def test_the_welcome_flow_does_not_promise_it(self):
        text = (ROOT / "gui" / "onboarding.py").read_text(encoding="utf-8")
        self.assertNotRegex(text, r"\bDrop in\b")
        self.assertNotRegex(text, r"\bdrag\b")

    def test_the_zone_copy_falls_back_to_what_it_can_do(self):
        source = (ROOT / "gui" / "widgets.py").read_text(encoding="utf-8")
        marker = "if not self._dnd_available:"
        self.assertIn(marker, source)
        fallback = source[source.index(marker):source.index(marker) + 700]
        self.assertIn("Choose files below", fallback)
        self.assertNotIn("Drag files here", fallback)

    def test_the_right_click_affordance_is_stated_in_both_copies(self):
        """It opens a folder picker and was documented nowhere."""
        source = (ROOT / "gui" / "widgets.py").read_text(encoding="utf-8")
        texts = [
            node.value
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and "Originals are never modified" in node.value
        ]
        self.assertEqual(len(texts), 2, texts)
        for copy in texts:
            self.assertIn("right-click", copy.lower(), copy)


if __name__ == "__main__":
    unittest.main()
