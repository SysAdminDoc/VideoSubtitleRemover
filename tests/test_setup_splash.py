import unittest

from scripts.setup_splash import parse_progress


class SetupSplashTests(unittest.TestCase):
    def test_parse_progress_clamps_and_normalizes(self):
        self.assertEqual(
            parse_progress("done|Setup complete|240"),
            ("DONE", "Setup complete", 100),
        )
        self.assertEqual(
            parse_progress("invalid|Working|not-a-number"),
            ("RUNNING", "Working", 2),
        )

    def test_parse_progress_uses_safe_default_for_malformed_payload(self):
        self.assertEqual(
            parse_progress("partial"),
            ("RUNNING", "Preparing the local runtime...", 2),
        )


if __name__ == "__main__":
    unittest.main()
