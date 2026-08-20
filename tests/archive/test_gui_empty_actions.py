import unittest

from gui.app import VideoSubtitleRemoverApp


class _DropAreaStub:
    def __init__(self):
        self.file_calls = 0
        self.folder_calls = 0

    def _open_file_dialog(self):
        self.file_calls += 1

    def _open_folder_dialog(self):
        self.folder_calls += 1


class GuiEmptyActionsTests(unittest.TestCase):
    def test_empty_queue_import_actions_reuse_drop_area_dialogs(self):
        app = VideoSubtitleRemoverApp.__new__(VideoSubtitleRemoverApp)
        app.drop_area = _DropAreaStub()

        app._open_file_picker()
        app._open_folder_picker()

        self.assertEqual(app.drop_area.file_calls, 1)
        self.assertEqual(app.drop_area.folder_calls, 1)

    def test_empty_queue_import_actions_ignore_missing_drop_area(self):
        app = VideoSubtitleRemoverApp.__new__(VideoSubtitleRemoverApp)

        app._open_file_picker()
        app._open_folder_picker()


if __name__ == "__main__":
    unittest.main()
