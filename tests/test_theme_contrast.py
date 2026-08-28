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
    # The selected state of a list row: accent text on the muted accent
    # fill, drawn together and reachable through neither SURFACES nor the
    # ink enumeration above.
    ("BLUE_HOVER", ("BLUE_MUTED",)),
    ("BLUE_PRIMARY", ("BLUE_MUTED",)),
    ("GREEN_PRIMARY", ("GREEN_MUTED",)),
)

# Rest/hover pairs where the hover fill is the only feedback that the
# pointer is over an interactive control, so the step has to be visible.
# Whether it lightens or darkens is a design choice; that it is noticeable
# is not.
MIN_HOVER_STEP = 1.15
HOVER_PAIRS = (
    ("BLUE_PRIMARY", "BLUE_HOVER"),
    ("GREEN_PRIMARY", "GREEN_HOVER"),
    ("DANGER", "DANGER_HOVER"),
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


# Keywords that paint the boundary of a widget rather than a separator.
_OUTLINE_KEYWORDS = frozenset(
    {"highlightbackground", "highlightcolor", "outline", "border_color"})

# Helpers that take the boundary colour as a positional argument, and the
# index that argument sits at.
_POSITIONAL_OUTLINE_HELPERS = {"_apply_surface_state": 1}


def _mentions_divider(node: ast.AST,
                      aliases: frozenset[str] = frozenset()) -> bool:
    """The divider tone, under `Theme.BORDER_SUBTLE` or any local name.

    Matching only the attribute node meant `edge = Theme.BORDER_SUBTLE`
    followed by `highlightbackground=edge` walked straight past.
    """
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and child.attr == "BORDER_SUBTLE":
            return True
        if isinstance(child, ast.Name) and child.id in aliases:
            return True
    return False


def _is_divider_value(value: ast.AST, aliases: frozenset[str]) -> bool:
    """The value IS the divider tone, rather than merely containing it.

    `tk.Frame(bg=Theme.BORDER_SUBTLE)` contains it and is a widget, not a
    colour; treating that as an alias made the gate flag every later use of
    the variable it was assigned to.
    """
    if isinstance(value, ast.Attribute):
        return value.attr == "BORDER_SUBTLE"
    if isinstance(value, ast.Name):
        return value.id in aliases
    return False


def _divider_aliases(scope: ast.AST) -> frozenset[str]:
    """Names in one scope that can only ever be the divider tone.

    A name assigned the divider tone in one branch and something else in
    another is a variable, not an alias: the branch decides, and the
    conditional handling below already reads that. Only a name whose every
    binding is the divider tone stands in for it.
    """
    assigned: dict[str, list[ast.AST]] = {}
    for node in ast.walk(scope):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            targets = [node.target]
        else:
            continue
        value = getattr(node, "value", None)
        for target in targets:
            if isinstance(target, ast.Name):
                assigned.setdefault(target.id, []).append(value)

    names: set[str] = set()
    for _ in range(3):  # resolve a short chain of rebinding
        before = len(names)
        for name, values in assigned.items():
            if name in names:
                continue
            if values and all(
                    _is_divider_value(value, frozenset(names))
                    for value in values):
                names.add(name)
        if len(names) == before:
            break
    return frozenset(names)


def _scopes(tree: ast.AST):
    """Every function body, plus the module body itself."""
    yield tree
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _tests_enabled(node: ast.AST) -> bool:
    return any(
        (isinstance(child, ast.Name) and "enabled" in child.id)
        or (isinstance(child, ast.Attribute) and "enabled" in child.attr)
        for child in ast.walk(node)
    )


def _negates_enabled(test: ast.AST) -> bool:
    """`not enabled`, and the spellings that mean the same thing.

    Only recognising a top-level `ast.Not` let `X if enabled is False else
    SUBTLE` put the divider tone on the live control and read as exempt.
    """
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return True
    if isinstance(test, ast.Compare) and len(test.comparators) == 1:
        operator = test.ops[0]
        comparator = test.comparators[0]
        falsey = (
            isinstance(comparator, ast.Constant)
            and comparator.value in (False, 0, None)
        )
        if falsey and isinstance(operator, (ast.Is, ast.Eq)):
            return True
        truthy = (
            isinstance(comparator, ast.Constant) and comparator.value is True
        )
        if truthy and isinstance(operator, (ast.IsNot, ast.NotEq)):
            return True
    return False


def _draws_live_divider(node: ast.AST,
                        aliases: frozenset[str] = frozenset()) -> bool:
    """The divider tone reaching a control that the user can still operate.

    WCAG 2.2 1.4.11 exempts an inactive component, so `X if enabled else
    SUBTLE` is legitimate: the divider tone only lands on the disabled
    control. Anything else is the sole boundary of a live control.
    """
    if isinstance(node, ast.IfExp) and _tests_enabled(node.test):
        live = (node.orelse if _negates_enabled(node.test) else node.body)
        return _draws_live_divider(live, aliases)
    return _mentions_divider(node, aliases)


def _positional_divider_outline(call: ast.Call,
                                aliases: frozenset[str] = frozenset()) -> bool:
    name = getattr(call.func, "attr", getattr(call.func, "id", ""))
    index = _POSITIONAL_OUTLINE_HELPERS.get(name)
    if index is None or len(call.args) <= index:
        return False
    return _draws_live_divider(call.args[index], aliases)


def _divider_returned_as_outline():
    """Any helper handing the divider tone back to a caller.

    This used to require "border" in the function name, so `_outline_colour`,
    `_edge_colour` and `_ring_color` were all invisible to it. A helper that
    returns a colour is a colour source whatever it is called; the ones that
    legitimately return the divider tone are the separator builders, and they
    return a widget rather than a colour.
    """
    offenders = []
    for path in sorted((ROOT / "gui").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            aliases = _divider_aliases(node)
            for statement in ast.walk(node):
                if (isinstance(statement, ast.Return)
                        and statement.value is not None
                        and _draws_live_divider(statement.value, aliases)):
                    offenders.append(f"{path.name}:{statement.lineno}")
    return offenders


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

    def test_the_hover_state_of_a_filled_control_is_visible(self):
        """Raising BLUE_PRIMARY for contrast left BLUE_HOVER 1.02:1 away
        from it, so hovering an accent button changed nothing a user could
        see."""
        failures = []
        for theme in self._themes():
            for rest, hover in HOVER_PAIRS:
                step = contrast(getattr(Theme, rest), getattr(Theme, hover))
                if step < MIN_HOVER_STEP:
                    failures.append(
                        f"{theme}: {hover} is {step:.3f} from {rest}, "
                        f"under the {MIN_HOVER_STEP} floor")
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

    @staticmethod
    def _check_outline_call(node, aliases, path, offenders):
        for keyword in node.keywords:
            if keyword.arg not in _OUTLINE_KEYWORDS:
                continue
            # The first version of this compared only the top node of the
            # keyword value, so a conditional expression hid a real
            # offender: `highlightbackground=(A if x else SUBTLE)`.
            if _draws_live_divider(keyword.value, aliases):
                offenders.append(f"{path.name}:{node.lineno}")
        if _positional_divider_outline(node, aliases):
            offenders.append(f"{path.name}:{node.lineno}")

    def test_the_divider_token_is_never_a_control_outline(self):
        """The exemption is only true while this stays true.

        `highlightbackground` on a tk widget is the control's own boundary,
        so a divider tone used there is the sole visual boundary of a
        control and the exemption would be false.
        """
        offenders = []
        for path in sorted((ROOT / "gui").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for scope in _scopes(tree):
                # An outline assigned to a local and then handed over is the
                # same outline: `edge = Theme.BORDER_SUBTLE` followed by
                # `highlightbackground=edge`. Aliases are resolved per scope,
                # because a name is only a stand-in for the tone where every
                # one of its bindings is that tone.
                aliases = _divider_aliases(scope)
                for node in ast.walk(scope):
                    if not isinstance(node, ast.Call):
                        continue
                    self._check_outline_call(
                        node, aliases, path, offenders)
        offenders.extend(_divider_returned_as_outline())
        offenders = sorted(set(offenders))
        self.assertEqual(
            offenders, [],
            "BORDER_SUBTLE is the divider tone and is exempt from the 3:1 "
            "control-boundary rule; using it as a control outline makes that "
            "exemption untrue. Use Theme.BORDER: " + ", ".join(offenders),
        )

    def test_the_gate_sees_the_ways_round_it(self):
        """Three spellings that put the divider tone on a live control.

        Each one passed the first version of this gate: it matched only a
        literal `Theme.BORDER_SUBTLE` attribute node, only recognised a
        helper whose name contained "border", and only read negation as a
        top-level `not`.
        """
        import subprocess
        import sys as _sys

        cases = {
            "a local variable holding the tone": (
                "import tkinter as tk\n"
                "from gui.theme import Theme\n"
                "\n\n"
                "class ProbeCard(tk.Frame):\n"
                "    def __init__(self, parent):\n"
                "        edge = Theme.BORDER_SUBTLE\n"
                "        super().__init__(parent, highlightbackground=edge)\n"
            ),
            "a helper whose name lacks 'border'": (
                "import tkinter as tk\n"
                "from gui.theme import Theme\n"
                "\n\n"
                "def _outline_colour():\n"
                "    return Theme.BORDER_SUBTLE\n"
                "\n\n"
                "class ProbeCard(tk.Frame):\n"
                "    def __init__(self, parent):\n"
                "        super().__init__(\n"
                "            parent, highlightbackground=_outline_colour())\n"
            ),
            "`is False` instead of `not`": (
                "import tkinter as tk\n"
                "from gui.theme import Theme\n"
                "\n\n"
                "class ProbeCard(tk.Frame):\n"
                "    enabled = True\n"
                "\n"
                "    def __init__(self, parent):\n"
                "        super().__init__(\n"
                "            parent,\n"
                "            highlightbackground=(\n"
                "                Theme.BORDER_FOCUS if self.enabled is False\n"
                "                else Theme.BORDER_SUBTLE),\n"
                "        )\n"
            ),
        }
        target = (
            "tests/test_theme_contrast.py::ExemptionTests::"
            "test_the_divider_token_is_never_a_control_outline"
        )
        probe = ROOT / "gui" / "_divider_gate_probe.py"
        for label, body in cases.items():
            with self.subTest(case=label):
                probe.write_text(body, encoding="utf-8")
                try:
                    result = subprocess.run(
                        [_sys.executable, "-m", "pytest", target, "-q"],
                        cwd=str(ROOT), capture_output=True, text=True,
                        timeout=300,
                    )
                finally:
                    probe.unlink()
                self.assertNotEqual(
                    result.returncode, 0,
                    f"the gate did not catch {label}:\n{result.stdout[-800:]}",
                )

    def test_a_widget_built_with_the_divider_tone_is_not_an_outline(self):
        """A separator is a Frame whose background is the tone.

        Treating the variable it is assigned to as a stand-in for the tone
        made the gate flag every later use of that name, which is the noise
        that makes a gate get deleted rather than fixed.
        """
        import ast as _ast

        source = (
            "import tkinter as tk\n"
            "from gui.theme import Theme\n"
            "\n\n"
            "def _divider(parent):\n"
            "    line = tk.Frame(parent, bg=Theme.BORDER_SUBTLE, height=1)\n"
            "    line.pack(fill='x')\n"
            "    return line\n"
        )
        tree = _ast.parse(source)
        function = next(
            node for node in _ast.walk(tree)
            if isinstance(node, _ast.FunctionDef)
        )
        self.assertEqual(_divider_aliases(function), frozenset())

    def test_a_name_assigned_two_different_tones_is_not_an_alias(self):
        """`border` takes the divider tone only on the disabled branch."""
        import ast as _ast

        source = (
            "from gui.theme import Theme\n"
            "\n\n"
            "def draw(enabled):\n"
            "    if not enabled:\n"
            "        border = Theme.BORDER_SUBTLE\n"
            "    else:\n"
            "        border = Theme.BORDER\n"
            "    return border\n"
        )
        tree = _ast.parse(source)
        function = next(
            node for node in _ast.walk(tree)
            if isinstance(node, _ast.FunctionDef)
        )
        self.assertEqual(_divider_aliases(function), frozenset())

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
    def test_the_hover_gate_would_catch_the_regression_it_was_written_for(self):
        apply_default_theme()
        original = Theme.BLUE_HOVER
        try:
            Theme.BLUE_HOVER = "#4b8aff"  # the value RM-333 left behind
            self.assertLess(
                contrast(Theme.BLUE_PRIMARY, Theme.BLUE_HOVER),
                MIN_HOVER_STEP,
            )
        finally:
            Theme.BLUE_HOVER = original

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
