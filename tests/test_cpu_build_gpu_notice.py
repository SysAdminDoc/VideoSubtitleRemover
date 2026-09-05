"""RM-356: a CPU build on an NVIDIA machine has to say so before the run.

The product already knew there was an NVIDIA card and already knew it had no
CUDA execution provider, and put both facts in the log as a warning nobody
reads. Issue #10 is a user who watched a twenty minute run go by on the CPU
with an idle card in the machine.

The probes are injected here rather than patched globally: the function takes
its GPU, CUDA and build-profile probes as arguments precisely so the four
cases can be driven without a particular machine.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend.device_provider import RELEASES_URL, cpu_build_on_nvidia_hardware


def _nvidia_present() -> dict:
    return {
        "present": True,
        "name": "NVIDIA GeForce RTX 4070 SUPER",
        "driver": "610.88",
        "memoryTotalMiB": 12282,
    }


def _no_gpu() -> dict:
    return {"present": False, "name": "", "driver": "", "memoryTotalMiB": None}


def _profile(name: str):
    return lambda: {"profile": name, "provider": "", "source": "stamp"}


class CpuBuildOnNvidiaNoticeTests(unittest.TestCase):
    def test_a_cpu_build_on_an_nvidia_machine_is_reported(self):
        notice = cpu_build_on_nvidia_hardware(
            requested_device="cuda:0",
            gpu_probe=_nvidia_present,
            cuda_probe=lambda index: False,
            profile_probe=_profile("cpu"),
        )
        self.assertIsNotNone(notice)
        self.assertEqual(notice["adapter"], "NVIDIA GeForce RTX 4070 SUPER")
        self.assertEqual(notice["releasesUrl"], RELEASES_URL)
        self.assertIn("nvidia", notice["assetPrefix"])
        self.assertTrue(notice["releasesUrl"].startswith("https://"))

    def test_no_notice_on_the_nvidia_lane(self):
        self.assertIsNone(cpu_build_on_nvidia_hardware(
            requested_device="cuda:0",
            gpu_probe=_nvidia_present,
            # A stamped nvidia build settles it before any probe runs, which
            # matters because a driver hiccup must not advertise a download
            # the user already has.
            cuda_probe=lambda index: False,
            profile_probe=_profile("nvidia"),
        ))

    def test_no_notice_when_the_machine_has_no_nvidia_adapter(self):
        self.assertIsNone(cpu_build_on_nvidia_hardware(
            requested_device="cuda:0",
            gpu_probe=_no_gpu,
            cuda_probe=lambda index: False,
            profile_probe=_profile("cpu"),
        ))

    def test_no_notice_when_the_user_asked_for_the_cpu(self):
        self.assertIsNone(cpu_build_on_nvidia_hardware(
            requested_device="cpu",
            gpu_probe=_nvidia_present,
            cuda_probe=lambda index: False,
            profile_probe=_profile("cpu"),
        ))

    def test_no_notice_when_cuda_actually_loads(self):
        self.assertIsNone(cpu_build_on_nvidia_hardware(
            requested_device="cuda:0",
            gpu_probe=_nvidia_present,
            cuda_probe=lambda index: True,
            profile_probe=_profile("cpu"),
        ))

    def test_the_gpu_probe_is_not_run_when_cuda_already_works(self):
        calls = []

        def _counting_gpu_probe() -> dict:
            calls.append(1)
            return _nvidia_present()

        cpu_build_on_nvidia_hardware(
            requested_device="cuda:0",
            gpu_probe=_counting_gpu_probe,
            cuda_probe=lambda index: True,
            profile_probe=_profile("cpu"),
        )
        self.assertEqual(
            calls, [],
            "nvidia-smi is a subprocess; do not shell it when there is "
            "nothing the answer could change",
        )

    def test_no_notice_on_a_directml_build(self):
        # DirectML is a supported profile that runs on this very card through
        # DmlExecutionProvider, and it ships onnxruntime-directml, so the CUDA
        # probe is correctly False. Telling that user their build runs on the
        # CPU would be wrong, and pointing them at a CUDA download redundant.
        self.assertIsNone(cpu_build_on_nvidia_hardware(
            requested_device="cuda:0",
            gpu_probe=_nvidia_present,
            cuda_probe=lambda index: False,
            directml_probe=lambda: True,
            profile_probe=_profile("directml"),
        ))

    def test_no_notice_when_directml_is_loadable_on_an_unstamped_build(self):
        self.assertIsNone(cpu_build_on_nvidia_hardware(
            requested_device="cuda:0",
            gpu_probe=_nvidia_present,
            cuda_probe=lambda index: False,
            directml_probe=lambda: True,
            profile_probe=_profile(""),
        ))

    def test_a_failing_probe_does_not_take_the_run_down(self):
        """The notice is advisory. A raise from it must not end a render."""
        import subprocess

        repo_root = Path(__file__).resolve().parents[1]
        clip = repo_root / "tests" / "clips" / "static_dialogue.mkv"
        with tempfile.TemporaryDirectory(prefix="vsr-probe-raise-") as tmp:
            output = Path(tmp) / "out.mp4"
            script = (
                "import backend.device_provider as dp\n"
                "def _boom(*a, **k):\n"
                "    raise RuntimeError('build stamp corrupted')\n"
                "dp.cpu_build_on_nvidia_hardware = _boom\n"
                "import backend.cli as cli\n"
                "cli.cpu_build_on_nvidia_hardware = _boom\n"
                "import sys\n"
                f"sys.argv = ['cli', '--input', {str(clip)!r},"
                f" '--output', {str(output)!r}, '--mode', 'sttn']\n"
                "cli.main()\n"
            )
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=str(repo_root), capture_output=True, text=True,
            )
            self.assertEqual(
                result.returncode, 0,
                f"a raising notice probe ended the run:\n{result.stderr[-1500:]}",
            )
            self.assertTrue(output.is_file())

    def test_an_empty_requested_device_still_reports(self):
        # The GUI has no device string before the config loads, and that is
        # not the same as the user choosing the CPU.
        self.assertIsNotNone(cpu_build_on_nvidia_hardware(
            requested_device="",
            gpu_probe=_nvidia_present,
            cuda_probe=lambda index: False,
            profile_probe=_profile("cpu"),
        ))


class CliNoticeTests(unittest.TestCase):
    """The command line prints the equivalent, once."""

    def test_the_run_banner_carries_the_notice(self):
        import subprocess

        repo_root = Path(__file__).resolve().parents[1]
        clip = repo_root / "tests" / "clips" / "static_dialogue.mkv"
        self.assertTrue(clip.is_file())

        with tempfile.TemporaryDirectory(prefix="vsr-notice-") as tmp:
            output = Path(tmp) / "out.mp4"
            env = dict(os.environ)
            env["VSR_DEPENDENCY_PROFILE"] = "cpu"
            result = subprocess.run(
                [sys.executable, "-m", "backend.cli", "--input", str(clip),
                 "--output", str(output), "--mode", "sttn", "--gpu", "0"],
                cwd=str(repo_root), env=env, capture_output=True, text=True,
            )

        from backend.device_provider import cpu_build_on_nvidia_hardware as probe

        expected = probe(requested_device="cuda:0")
        if expected is None:
            self.skipTest(
                "this machine is not a CPU build with an NVIDIA card, so "
                "there is no notice to assert"
            )
        self.assertIn("[note]", result.stdout)
        self.assertIn(expected["adapter"], result.stdout)
        self.assertIn(RELEASES_URL, result.stdout)
        self.assertEqual(
            result.stdout.count(RELEASES_URL), 1,
            "the notice is once per invocation, not once per file",
        )


def _have_display() -> bool:
    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


@unittest.skipUnless(_have_display(), "GUI notice test needs a display")
class GuiNoticeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tkinter as tk

        cls.tk = tk
        cls._tmpdir = tempfile.TemporaryDirectory()
        import VideoSubtitleRemover as app_exports
        from gui import app as gui_app_module
        from gui import config as gui_config

        cls._app_exports = app_exports
        cls._gui_app_module = gui_app_module
        cls._gui_config = gui_config
        cls._shared_root = tk.Tk()
        cls._shared_root.withdraw()
        cls._originals = (
            app_exports.SETTINGS_FILE,
            gui_config.SETTINGS_FILE,
            gui_config.QUEUE_STATE_FILE,
        )
        settings_path = Path(cls._tmpdir.name) / "settings.json"
        app_exports.SETTINGS_FILE = settings_path
        gui_config.SETTINGS_FILE = settings_path
        gui_config.QUEUE_STATE_FILE = Path(cls._tmpdir.name) / "queue.json"

    @classmethod
    def tearDownClass(cls):
        (cls._app_exports.SETTINGS_FILE,
         cls._gui_config.SETTINGS_FILE,
         cls._gui_config.QUEUE_STATE_FILE) = cls._originals
        cls._shared_root.destroy()
        try:
            cls.tk._default_root = None
        except AttributeError:
            pass
        cls._tmpdir.cleanup()

    def _make_app(self):
        self._gui_config.save_settings(self._gui_config.ProcessingConfig(
            onboarding_seen=True, adv_panel_open=False, log_panel_open=False))
        with mock.patch.object(
            self._app_exports.VideoSubtitleRemoverApp,
            "_start_startup_hardware_probe",
        ), mock.patch.object(
            self._app_exports.VideoSubtitleRemoverApp, "_maybe_restore_queue",
        ), mock.patch.object(
            self._gui_app_module.tk, "Tk",
            side_effect=lambda: self.tk.Toplevel(self._shared_root),
        ):
            app = self._app_exports.VideoSubtitleRemoverApp()
        app._live_region_ocr_enabled = False
        app.root.withdraw()
        self.addCleanup(self._destroy_app, app)
        return app

    def _destroy_app(self, app):
        app._shutdown_started = True
        try:
            app._shutdown_ui_resources()
        finally:
            try:
                app.root.destroy()
            except self.tk.TclError:
                pass

    def test_the_notice_and_its_link_appear_before_any_run(self):
        app = self._make_app()
        app._cpu_build_gpu_notice = {
            "adapter": "NVIDIA GeForce RTX 4070 SUPER",
            "driver": "610.88",
            "profile": "cpu",
            "releasesUrl": RELEASES_URL,
            "assetPrefix": "VideoSubtitleRemoverPro-<version>-nvidia",
        }
        app._refresh_cpu_build_gpu_notice()
        app.root.update_idletasks()

        self.assertTrue(app.cpu_build_gpu_row.winfo_manager())
        self.assertIn("RTX 4070 SUPER", app.cpu_build_gpu_label.cget("text"))
        self.assertTrue(app.cpu_build_gpu_btn.winfo_manager())

    def test_the_notice_stays_hidden_when_there_is_nothing_to_say(self):
        app = self._make_app()
        app._cpu_build_gpu_notice = None
        app._refresh_cpu_build_gpu_notice()
        app.root.update_idletasks()
        self.assertFalse(app.cpu_build_gpu_row.winfo_manager())

    def test_the_link_button_opens_the_releases_page_and_nothing_else(self):
        app = self._make_app()
        app._cpu_build_gpu_notice = {
            "adapter": "NVIDIA", "driver": "", "profile": "cpu",
            "releasesUrl": RELEASES_URL,
            "assetPrefix": "VideoSubtitleRemoverPro-<version>-nvidia",
        }
        opened = []
        with mock.patch("webbrowser.open", side_effect=lambda url, new=0:
                        opened.append(url)):
            app._open_releases_page()
        self.assertEqual(opened, [RELEASES_URL])

    def test_a_non_https_url_is_refused(self):
        app = self._make_app()
        app._cpu_build_gpu_notice = {
            "adapter": "NVIDIA", "driver": "", "profile": "cpu",
            "releasesUrl": "http://example.com/releases",
            "assetPrefix": "x",
        }
        opened = []
        with mock.patch("webbrowser.open", side_effect=lambda url, new=0:
                        opened.append(url)):
            app._open_releases_page()
        self.assertEqual(opened, [], "the opener must stay https-only")


if __name__ == "__main__":
    unittest.main()
