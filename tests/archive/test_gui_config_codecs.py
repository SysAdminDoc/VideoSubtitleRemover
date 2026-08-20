import unittest

import gui.config as gui_config


class GuiOutputCodecTests(unittest.TestCase):
    def test_vvc_codec_survives_gui_settings_normalization(self):
        config = gui_config.ProcessingConfig.from_dict({"output_codec": "vvc"})

        self.assertEqual(config.output_codec, "vvc")

    def test_h266_alias_normalizes_to_vvc(self):
        for alias in ("h266", "h.266"):
            with self.subTest(alias=alias):
                config = gui_config.ProcessingConfig.from_dict({
                    "output_codec": alias,
                })

                self.assertEqual(config.output_codec, "vvc")

    def test_hevc_alias_normalizes_to_h265(self):
        for alias in ("hevc", "h.265"):
            with self.subTest(alias=alias):
                config = gui_config.ProcessingConfig.from_dict({
                    "output_codec": alias,
                })

                self.assertEqual(config.output_codec, "h265")

    def test_unknown_codec_falls_back_to_h264(self):
        config = gui_config.ProcessingConfig.from_dict({"output_codec": "vp9"})

        self.assertEqual(config.output_codec, "h264")


if __name__ == "__main__":
    unittest.main()
