import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest import mock

from backend.support_bundle import create_support_bundle
from backend import ffmpeg_profiles

ROOT = Path(__file__).resolve().parents[1]


class SupportBundleTests(unittest.TestCase):
    def test_bundle_redacts_settings_logs_and_batch_report_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = root / "settings.json"
            settings.write_text(
                json.dumps({
                    "output_codec": "vvc",
                    "whisper_model_path": "C:/Users/example/models/ggml.bin",
                    "window_geometry": "1800x1200+0+0",
                }),
                encoding="utf-8",
            )
            log = root / "vsr_pro.log"
            log.write_text(
                "failed at C:/Users/example/Videos/source.mp4\n",
                encoding="utf-8",
            )
            report = root / "vsr-batch-summary.json"
            report.write_text(
                json.dumps({
                    "files": [{
                        "input": "C:/Users/example/Videos/source.mp4",
                        "output": "D:/Exports/source_clean.mp4",
                        "input_name": "source.mp4",
                        "status": "failed",
                    }],
                }),
                encoding="utf-8",
            )
            out = root / "support.zip"

            created = create_support_bundle(
                out,
                settings_path=settings,
                log_path=log,
                batch_report_paths=[report],
                app_version="9.9.9",
                extra_facts={
                    "settings_file": str(settings),
                    "ffmpeg_ready": True,
                },
            )

            self.assertEqual(created, out)
            with zipfile.ZipFile(out) as bundle:
                names = set(bundle.namelist())
                self.assertIn("support.json", names)
                self.assertIn("settings.redacted.json", names)
                self.assertIn("vsr_pro.redacted.log", names)
                self.assertIn("batch-report-1.redacted.json", names)

                support = json.loads(bundle.read("support.json"))
                self.assertEqual(support["schema"], "vsr.support_bundle.v1")
                self.assertEqual(support["app_version"], "9.9.9")
                self.assertIn("ffmpeg", support["tools"])
                self.assertIn("ffprobe", support["tools"])
                self.assertEqual(
                    support["ffmpeg_profiles"]["schema"],
                    "vsr.ffmpeg_profiles.v1",
                )
                self.assertEqual(
                    support["backend_status"]["schema"],
                    "vsr.backend_status.v1",
                )
                self.assertIn("summary", support["backend_status"])
                self.assertEqual(
                    support["model_cache"]["schema"],
                    "vsr.model_cache_status.v1",
                )
                self.assertIn("opencv_libpng", support["security"])
                self.assertEqual(
                    support["security"]["opencv_libpng"]["fixed_version"],
                    "1.6.54",
                )
                self.assertEqual(support["facts"]["settings_file"], "<redacted>")
                self.assertTrue(support["facts"]["ffmpeg_ready"])

                redacted_settings = json.loads(
                    bundle.read("settings.redacted.json")
                )
                self.assertEqual(redacted_settings["output_codec"], "vvc")
                self.assertEqual(
                    redacted_settings["whisper_model_path"],
                    "<redacted>",
                )
                self.assertEqual(
                    redacted_settings["window_geometry"],
                    "<redacted>",
                )

                redacted_log = bundle.read("vsr_pro.redacted.log").decode()
                self.assertNotIn("C:/Users/example", redacted_log)
                self.assertIn("source.mp4", redacted_log)

                redacted_report = json.loads(
                    bundle.read("batch-report-1.redacted.json")
                )
                record = redacted_report["files"][0]
                self.assertEqual(record["input"], "<redacted>")
                self.assertEqual(record["output"], "<redacted>")
                self.assertEqual(record["input_name"], "source.mp4")

    def test_bundle_adds_zip_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            created = create_support_bundle(
                Path(tmp) / "support",
                app_version="1.0",
            )

            self.assertEqual(created.suffix, ".zip")
            self.assertTrue(created.exists())

    def test_cli_support_bundle_entrypoint_is_dependency_light(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "support.zip"

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "backend.cli",
                    "--support-bundle",
                    str(out),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(out.exists())
            self.assertIn("[support] wrote", result.stdout)

    def test_bug_report_form_requests_support_bundle(self):
        form = (
            ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml"
        ).read_text(encoding="utf-8")

        self.assertIn("Support bundle", form)
        self.assertIn("python -m backend.cli --support-bundle", form)


class FfmpegProfileTests(unittest.TestCase):
    def _profiles(self, *, filters="", encoders="", ffmpeg=True, ffprobe=True):
        def which(name):
            if name == "ffmpeg" and ffmpeg:
                return "ffmpeg"
            if name == "ffprobe" and ffprobe:
                return "ffprobe"
            return None

        def run_text(command, timeout):
            if "-filters" in command:
                return filters, ""
            if "-encoders" in command:
                return encoders, ""
            return "ffmpeg version test", ""

        with mock.patch.object(ffmpeg_profiles.shutil, "which", which):
            with mock.patch.object(ffmpeg_profiles, "_run_ffmpeg_text", run_text):
                return ffmpeg_profiles.collect_ffmpeg_capability_profiles()

    def test_profiles_report_exact_missing_filters_and_encoder_groups(self):
        payload = self._profiles(
            filters=(
                " ..C loudnorm         A->A       EBU R128 loudness\n"
                " ..C whisper          A->A       Whisper filter\n"
            ),
            encoders=(
                " V..... libx264       H.264\n"
                " V..... libx265       H.265\n"
                " V..... libsvtav1     AV1\n"
            ),
        )
        by_name = {entry["name"]: entry for entry in payload["profiles"]}

        self.assertFalse(by_name["advanced_quality"]["available"])
        self.assertEqual(
            by_name["advanced_quality"]["missing"]["filters"],
            ["libvmaf"],
        )
        self.assertFalse(by_name["modern_codec"]["available"])
        self.assertEqual(
            by_name["modern_codec"]["missing"]["encoder_groups"][0]["name"],
            "vvc",
        )
        self.assertIn("libvvenc", by_name["modern_codec"]["reason"])

    def test_config_preflight_is_scoped_to_selected_options(self):
        payload = self._profiles(
            filters=" ..C loudnorm         A->A       EBU R128 loudness\n",
            encoders=(
                " V..... libx264       H.264\n"
                " V..... libx265       H.265\n"
            ),
        )
        from backend.config import ProcessingConfig

        h265 = ProcessingConfig(output_codec="h265", preserve_audio=False)
        self.assertEqual(
            ffmpeg_profiles.missing_profile_requirements_for_config(h265, payload),
            [],
        )
        vvc = ProcessingConfig(output_codec="vvc", preserve_audio=False)
        missing = ffmpeg_profiles.missing_profile_requirements_for_config(
            vvc,
            payload,
        )
        self.assertEqual(missing[0]["profile"], "modern_codec")
        self.assertIn("libvvenc", missing[0]["reason"])

    def test_self_test_entries_include_four_profiles(self):
        payload = self._profiles(filters="", encoders="", ffmpeg=False, ffprobe=False)
        entries = ffmpeg_profiles.ffmpeg_profile_entries(payload)

        self.assertEqual(
            [entry["name"] for entry in entries],
            ["basic", "advanced_quality", "speech_fallback", "modern_codec"],
        )
        self.assertIn("ffmpeg", entries[0]["reason"])


if __name__ == "__main__":
    unittest.main()
