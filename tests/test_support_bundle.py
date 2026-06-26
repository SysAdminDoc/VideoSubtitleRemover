import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
import zipfile

from backend.support_bundle import create_support_bundle

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


if __name__ == "__main__":
    unittest.main()
