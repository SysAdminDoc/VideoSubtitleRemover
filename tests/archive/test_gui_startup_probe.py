import unittest
from types import SimpleNamespace
from unittest import mock

from gui.app import VideoSubtitleRemoverApp


class _FakeRoot:
    def __init__(self):
        self.after_calls = []

    def after(self, delay, callback):
        self.after_calls.append((delay, callback))


class _RuntimeErrorRoot:
    def after(self, delay, callback):
        raise RuntimeError("main thread is not in main loop")


class _FakeVar:
    def __init__(self):
        self.value = ""

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


class StartupHardwareProbeTests(unittest.TestCase):
    def test_probe_results_are_marshaled_to_tk_thread(self):
        app = object.__new__(VideoSubtitleRemoverApp)
        app.root = _FakeRoot()
        app._apply_startup_hardware_probe = mock.Mock()
        gpus = [{"index": 0, "name": "GPU", "memory": "8 GB"}]
        engines = {"detection": ["RapidOCR"], "inpainting": ["OpenCV"]}
        backend_status = {
            "schema": "vsr.backend_status.v1",
            "summary": {"detection": "RapidOCR (ready)"},
        }
        ffmpeg_profiles = {
            "schema": "vsr.ffmpeg_profiles.v1",
            "profiles": [{"name": "basic", "available": True}],
        }

        with mock.patch("gui.app.detect_gpu", return_value=gpus):
            with mock.patch("gui.app.detect_ai_engines", return_value=engines):
                with mock.patch("gui.app.detect_ffmpeg", return_value=True):
                    with mock.patch(
                        "gui.app.installed_backend_status",
                        return_value=backend_status,
                    ):
                        with mock.patch(
                            "gui.app.collect_ffmpeg_capability_profiles",
                            return_value=ffmpeg_profiles,
                        ):
                            app._probe_startup_hardware()

        self.assertEqual(len(app.root.after_calls), 1)
        delay, callback = app.root.after_calls[0]
        self.assertEqual(delay, 0)
        app._apply_startup_hardware_probe.assert_not_called()
        callback()
        app._apply_startup_hardware_probe.assert_called_once_with(
            gpus, engines, True, backend_status, ffmpeg_profiles
        )

    def test_probe_drops_results_when_tk_loop_is_not_available(self):
        app = object.__new__(VideoSubtitleRemoverApp)
        app.root = _RuntimeErrorRoot()
        app._apply_startup_hardware_probe = mock.Mock()

        with mock.patch("gui.app.detect_gpu", return_value=[]):
            with mock.patch("gui.app.detect_ai_engines", return_value={}):
                with mock.patch("gui.app.detect_ffmpeg", return_value=False):
                    with mock.patch(
                        "gui.app.installed_backend_status",
                        return_value={},
                    ):
                        with mock.patch(
                            "gui.app.collect_ffmpeg_capability_profiles",
                            return_value={"schema": "vsr.ffmpeg_profiles.v1", "profiles": []},
                        ):
                            app._probe_startup_hardware()

        app._apply_startup_hardware_probe.assert_not_called()

    def test_gpu_selection_uses_saved_gpu_when_available(self):
        app = object.__new__(VideoSubtitleRemoverApp)
        app.gpus = [
            {"index": 0, "name": "First GPU", "memory": "8 GB"},
            {"index": 2, "name": "Saved GPU", "memory": "16 GB"},
        ]
        app.config = SimpleNamespace(gpu_id=2, use_gpu=False)
        app.gpu_var = _FakeVar()

        app._apply_gpu_selection_from_config()

        self.assertEqual(app.gpu_var.get(), "Saved GPU (16 GB)")
        self.assertEqual(app.config.gpu_id, 2)
        self.assertTrue(app.config.use_gpu)

    def test_gpu_selection_falls_back_to_cpu_without_gpus(self):
        app = object.__new__(VideoSubtitleRemoverApp)
        app.gpus = []
        app.config = SimpleNamespace(gpu_id=3, use_gpu=True)
        app.gpu_var = _FakeVar()

        app._apply_gpu_selection_from_config()

        self.assertEqual(app.gpu_var.get(), "CPU mode")
        self.assertFalse(app.config.use_gpu)


if __name__ == "__main__":
    unittest.main()
