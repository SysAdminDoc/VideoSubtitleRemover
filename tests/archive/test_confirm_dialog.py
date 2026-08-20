import unittest

from gui.widgets import _confirm_default_focus


class ConfirmDialogTests(unittest.TestCase):
    def test_danger_confirmations_default_to_cancel_focus(self):
        self.assertEqual(_confirm_default_focus("danger"), "cancel")

    def test_non_destructive_confirmations_default_to_confirm_focus(self):
        self.assertEqual(_confirm_default_focus("primary"), "confirm")
        self.assertEqual(_confirm_default_focus("accent"), "confirm")

    def test_explicit_default_focus_override_is_honored(self):
        self.assertEqual(_confirm_default_focus("danger", "confirm"), "confirm")
        self.assertEqual(_confirm_default_focus("primary", "cancel"), "cancel")
        self.assertEqual(_confirm_default_focus("danger", "invalid"), "cancel")


if __name__ == "__main__":
    unittest.main()
