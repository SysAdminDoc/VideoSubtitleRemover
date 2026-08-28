"""RM-333: contrast for control boundaries, not only for button ink.

The only contrast test in the suite checked three button fills against one
ink token, while the tokens that draw the outline of every control measured
1.18:1 at worst against the surfaces they sit on. WCAG 2.2 1.4.11 asks for
3:1 on any visual boundary needed to identify a control.

Every exemption here names the reason it is exempt, and the two structural
ones are enforced: a divider token may not be used as a control outline.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from gui.theme import Theme, apply_default_theme, apply_high_contrast_theme

ROOT = Path(__file__).resolve().parent.parent

# Surfaces a control or its text is actually drawn on.
SURFACES = (
    "BG_DARK", "BG_SECONDARY", "BG_CARD", "BG_CARD_HOVER",
    "BG_CARD_SELECTED", "BG_TERTIARY", "BG_RAISED", "BG_LOG",
    "BG_OVERLAY", "BG_DISABLED",
)

# Tokens that draw the boundary of a control that has no other boundary.
CONTROL_BOUNDARIES = ("BORDER", "BORDER_STRONG", "BORDER_FOCUS")

# Foreground tokens used as normal-size text.
BODY_TEXT = (
    "TEXT_PRIMARY", "TEXT_SECONDARY", "TEXT_MUTED",
    "SUCCESS", "WARNING", "ERROR", "INFO", "DANGER",
    "BLUE_PRIMARY", "GREEN_PRIMARY", "CYAN",
)

# Ink tokens and the fills they are drawn on.
INK_PAIRS = (
    ("INK_ON_BLUE", ("BLUE_PRIMARY", "BLUE_HOVER", "BLUE_PRESS")),
    ("INK_ON_GREEN", ("GREEN_PRIMARY", "GREEN_HOVER", "GREEN_PRESS")),
    ("INK_ON_DANGER", ("DANGER", "DANGER_HOVER", "DANGER_PRESS")),
)

EXEMPTIONS = {
    "BORDER_SUBTLE": (
        "Divider tone. It draws separators between sections and never the "
        "boundary of a control, so WCAG 2.2 1.4.11 does not apply; the "
        "structural test below enforces that it stays that way."
    ),
    "TEXT_DISABLED": (
        "WCAG 2.2 1.4.3 exempts text that is part of an inactive user "
        "interface component. A disabled control is identified by its "
        "recessed fill as well as its ink."
    ),
}


def _srgb(channel: float) -> float:
    channel /= 255.0
    return (channel / 12.92 if channel <= 0.04045
            else ((channel + 0.055) / 1.055) ** 2.4)


def luminance(value: str) -> float:
    value = str(value).lstrip("#")
    red, green, blue = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _srgb(red) + 0.7152 * _srgb(green) + 0.0722 * _srgb(blue)


def contrast(first: str, second: str) -> float:
    a, b = luminance(first), luminance(second)
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


class ContrastMathTests(unittest.TestCase):
    def test_the_formula_matches_known_values(self):
        self.assertAlmostEqual(contrast("#ffffff", "#000000"), 21.0, places=2)
        self.assertAlmostEqual(contrast("#000000", "#000000"), 1.0, places=2)
        # A published mid-grey pair, so the implementation is not marking
        # its own homework.
        self.assertAlmostEqual(contrast("#777777", "#ffffff"), 4.48, places=2)


class _ThemeCase(unittest.TestCase):
    def tearDown(self):
        apply_default_theme()

    def _themes(self):
        for name, apply in (("default", apply_default_theme),
                            ("high-contrast", apply_high_contrast_theme)):
            apply()
            yield name


class ControlBoundaryContrastTests(_ThemeCase):
    def test_every_control_boundary_clears_three_to_one(self):
        failures = []
        for theme in self._themes():
            for token in CONTROL_BOUNDARIES:
                for surface in SURFACES:
                    value = contrast(
                        getattr(Theme, token), getattr(Theme, surface))
                    if value < 3.0:
                        failures.append(
                            f"{theme}: {token} on {surface} is {value:.2f}")
        self.assertEqual(failures, [], "\n".join(failures))

    def test_every_body_text_pair_clears_four_and_a_half_to_one(self):
        failures = []
        for theme in self._themes():
            for token in BODY_TEXT:
                for surface in SURFACES:
                    value = contrast(
                        getattr(Theme, token), getattr(Theme, surface))
                    if value < 4.5:
                        failures.append(
                            f"{theme}: {token} on {surface} is {value:.2f}")
        self.assertEqual(failures, [], "\n".join(failures))

    def test_ink_on_every_filled_control_clears_four_and_a_half_to_one(self):
        failures = []
        for theme in self._themes():
            for ink, fills in INK_PAIRS:
                for fill in fills:
                    value = contrast(getattr(Theme, ink), getattr(Theme, fill))
                    if value < 4.5:
                        failures.append(
                            f"{theme}: {ink} on {fill} is {value:.2f}")
        self.assertEqual(failures, [], "\n".join(failures))

    def test_status_backgrounds_carry_their_own_foreground(self):
        pairs = (("SUCCESS", "SUCCESS_BG"), ("WARNING", "WARNING_BG"),
                 ("ERROR", "ERROR_BG"), ("INFO", "INFO_BG"))
        failures = []
        for theme in self._themes():
            for fg, bg in pairs:
                value = contrast(getattr(Theme, fg), getattr(Theme, bg))
                if value < 4.5:
                    failures.append(
                        f"{theme}: {fg} on {bg} is {value:.2f}")
        self.assertEqual(failures, [], "\n".join(failures))

    def test_the_progress_fill_is_distinguishable_from_its_track(self):
        failures = []
        for theme in self._themes():
            value = contrast(Theme.PROGRESS_FILL, Theme.PROGRESS_BG)
            if value < 3.0:
                failures.append(f"{theme}: progress fill/track {value:.2f}")
        self.assertEqual(failures, [], "\n".join(failures))


class ExemptionTests(_ThemeCase):
    def test_every_exemption_states_a_reason(self):
        for token, reason in EXEMPTIONS.items():
            with self.subTest(token=token):
                self.assertTrue(hasattr(Theme, token))
                self.assertGreater(len(reason), 60)

    def test_the_divider_token_is_never_a_control_outline(self):
        """The exemption is only true while this stays true.

        `highlightbackground` on a tk widget is the control's own boundary,
        so a divider tone used there is the sole visual boundary of a
        control and the exemption would be false.
        """
        offenders = []
        for path in sorted((ROOT / "gui").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                for keyword in node.keywords:
                    if keyword.arg not in {
                            "highlightbackground", "highlightcolor"}:
                        continue
                    value = keyword.value
                    if (isinstance(value, ast.Attribute)
                            and value.attr == "BORDER_SUBTLE"):
                        offenders.append(f"{path.name}:{node.lineno}")
        self.assertEqual(
            offenders, [],
            "BORDER_SUBTLE is the divider tone and is exempt from the 3:1 "
            "control-boundary rule; using it as a control outline makes that "
            "exemption untrue. Use Theme.BORDER: " + ", ".join(offenders),
        )

    def test_the_disabled_ink_is_still_legible_enough_to_read(self):
        """Exempt from 4.5:1, but it must not vanish entirely."""
        failures = []
        for theme in self._themes():
            for surface in SURFACES:
                value = contrast(Theme.TEXT_DISABLED, getattr(Theme, surface))
                if value < 3.0:
                    failures.append(
                        f"{theme}: TEXT_DISABLED on {surface} is {value:.2f}")
        self.assertEqual(failures, [], "\n".join(failures))


class RegressionGuardTests(_ThemeCase):
    def test_the_gate_would_catch_a_regression(self):
        """Put the old border back and the boundary check must fail."""
        apply_default_theme()
        original = Theme.BORDER
        try:
            Theme.BORDER = "#2a3548"  # the value RM-333 replaced
            worst = min(
                contrast(Theme.BORDER, getattr(Theme, surface))
                for surface in SURFACES
            )
            self.assertLess(worst, 3.0)
        finally:
            Theme.BORDER = original


if __name__ == "__main__":
    unittest.main()
