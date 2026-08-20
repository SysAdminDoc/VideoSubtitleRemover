import json
from pathlib import Path
import sys
import unittest

from backend.subprocess_policy import run_process
from gui.config import ProcessingConfig, VSR_SETTINGS_FORMAT
from gui.theme import (
    Theme,
    f,
    normalize_text_scale_percent,
    set_text_scale_percent,
)


ROOT = Path(__file__).resolve().parents[1]


class TextScaleConfigTests(unittest.TestCase):
    def tearDown(self):
        set_text_scale_percent(100)
        Theme.RTL_LAYOUT = False

    def test_scale_is_normalized_to_supported_steps(self):
        self.assertEqual(normalize_text_scale_percent(None), 100)
        self.assertEqual(normalize_text_scale_percent(124), 125)
        self.assertEqual(normalize_text_scale_percent(149), 150)
        self.assertEqual(normalize_text_scale_percent(999), 200)

    def test_scaled_font_tuple_preserves_weight(self):
        set_text_scale_percent(200)
        self.assertEqual(
            f(Theme.F_BODY, "bold"),
            (Theme.FONT_FAMILY, Theme.F_BODY * 2, "bold"),
        )

    def test_gui_setting_persists_and_migrates(self):
        config = ProcessingConfig(text_scale_percent=176).normalized()
        self.assertEqual(config.text_scale_percent, 175)
        payload = config.to_dict()
        self.assertEqual(payload["text_scale_percent"], 175)
        self.assertEqual(payload["vsr_settings_format"], VSR_SETTINGS_FORMAT)
        restored = ProcessingConfig.from_dict(payload)
        self.assertEqual(restored.text_scale_percent, 175)


class TextScaleLayoutMatrixTests(unittest.TestCase):
    # RM-148: 125 and 175 are covered too -- the dialog work-area probe runs
    # inside every case, so each step exercises dialog reflow as well.
    CASES = (
        (100, "default", "en"),
        (100, "high-contrast", "pseudo"),
        (125, "default", "en"),
        (150, "default", "pseudo"),
        (150, "high-contrast", "rtl"),
        (175, "high-contrast", "en"),
        (200, "default", "rtl"),
        (200, "high-contrast", "pseudo"),
    )

    @staticmethod
    def _run_probe(scale, theme, locale):
        probe = ROOT / "tools" / "ui_scaling_probe.py"
        result = run_process(
            [
                sys.executable,
                str(probe),
                "--scale",
                str(scale),
                "--theme",
                theme,
                "--locale",
                locale,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = (result.stdout or "").strip().splitlines()
        payload = json.loads(output[-1]) if output else {}
        return result, payload

    def test_hidden_minimum_viewport_matrix(self):
        for scale, theme, locale in self.CASES:
            with self.subTest(scale=scale, theme=theme, locale=locale):
                result, payload = self._run_probe(scale, theme, locale)
                self.assertEqual(
                    result.returncode,
                    0,
                    msg=(result.stderr or "") + "\n" + json.dumps(payload),
                )
                self.assertTrue(payload.get("ok"), payload)

    def test_the_rendered_tree_flips_direction_between_locales(self):
        """RM-152: prove the mirror moved the *whole* tree, not a corner.

        The two runs build the same window at the same scale and theme
        and differ only in direction, so the direction census has to
        invert. Comparing populations rather than exact counts keeps the
        assertion honest: RTL captions are longer, so the responsive
        reflow legitimately stacks a slightly different set of rows.
        """
        _ltr_result, ltr = self._run_probe(150, "default", "en")
        _rtl_result, rtl = self._run_probe(150, "default", "rtl")
        self.assertTrue(ltr.get("ok"), ltr)
        self.assertTrue(rtl.get("ok"), rtl)

        ltr_sides = ltr["packSides"]
        rtl_sides = rtl["packSides"]
        self.assertGreater(ltr_sides["left"], ltr_sides["right"])
        self.assertGreater(rtl_sides["right"], rtl_sides["left"])

        # Not a single label may keep a west anchor once mirrored, and
        # the LTR baseline must actually have some to lose.
        self.assertGreater(ltr["labelAnchors"]["w"], 0)
        self.assertEqual(rtl["labelAnchors"]["w"], 0)
        self.assertGreater(
            rtl["labelAnchors"]["e"], ltr["labelAnchors"]["e"])


if __name__ == "__main__":
    unittest.main()
