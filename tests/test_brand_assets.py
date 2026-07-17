"""Brand asset contracts for source, runtime, README, and Windows packaging."""

from pathlib import Path
import unittest

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


class BrandAssetTests(unittest.TestCase):
    def test_png_assets_are_square_rgba_with_transparent_corners(self):
        for relative, expected_size in (
            ("icon.png", (512, 512)),
            ("assets/branding/vsr-pro-mark-master.png", (1024, 1024)),
        ):
            with self.subTest(relative=relative):
                with Image.open(ROOT / relative) as image:
                    rgba = image.convert("RGBA")
                    alpha = rgba.getchannel("A")
                    self.assertEqual(image.size, expected_size)
                    self.assertEqual(image.mode, "RGBA")
                    self.assertEqual(alpha.getpixel((0, 0)), 0)
                    self.assertEqual(
                        alpha.getpixel((image.width // 2, image.height // 2)),
                        255,
                    )

    def test_windows_icon_contains_small_and_large_variants(self):
        with Image.open(ROOT / "icon.ico") as image:
            sizes = set(image.info.get("sizes", ()))
            expected = {
                (16, 16), (24, 24), (32, 32), (48, 48),
                (64, 64), (128, 128), (256, 256),
            }
            self.assertTrue(expected.issubset(sizes))
            for size in expected:
                frame = image.ico.getimage(size).convert("RGBA")
                self.assertEqual(frame.getpixel((0, 0))[3], 0)

    def test_readme_and_frozen_spec_use_the_brand_assets(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        spec = (ROOT / "VideoSubtitleRemoverPro.spec").read_text(
            encoding="utf-8")
        self.assertIn('src="icon.png"', readme)
        self.assertIn("('icon.png', '.')", spec)
        self.assertIn("icon=['icon.ico']", spec)


if __name__ == "__main__":
    unittest.main()
