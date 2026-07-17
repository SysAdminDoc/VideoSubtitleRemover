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
        self.assertEqual(f(Theme.F_BODY, "bold"), (Theme.FONT_FAMILY, 24, "bold"))

    def test_gui_setting_persists_and_migrates(self):
        config = ProcessingConfig(text_scale_percent=176).normalized()
        self.assertEqual(config.text_scale_percent, 175)
        payload = config.to_dict()
        self.assertEqual(payload["text_scale_percent"], 175)
        self.assertEqual(payload["vsr_settings_format"], VSR_SETTINGS_FORMAT)
        restored = ProcessingConfig.from_dict(payload)
        self.assertEqual(restored.text_scale_percent, 175)


class TextScaleLayoutMatrixTests(unittest.TestCase):
    CASES = (
        (100, "default", "en"),
        (100, "high-contrast", "pseudo"),
        (150, "default", "pseudo"),
        (150, "high-contrast", "rtl"),
        (200, "default", "rtl"),
        (200, "high-contrast", "pseudo"),
    )

    def test_hidden_minimum_viewport_matrix(self):
        probe = ROOT / "tools" / "ui_scaling_probe.py"
        for scale, theme, locale in self.CASES:
            with self.subTest(scale=scale, theme=theme, locale=locale):
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
                self.assertEqual(
                    result.returncode,
                    0,
                    msg=(result.stderr or "") + "\n" + json.dumps(payload),
                )
                self.assertTrue(payload.get("ok"), payload)


if __name__ == "__main__":
    unittest.main()
