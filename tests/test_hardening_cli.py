import io
import json
import datetime
import os
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace


from backend import processor


def _has_display() -> bool:
    """Return True if a GUI display is available."""
    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class CliSoftSubtitleTests(unittest.TestCase):
    def _run_cli(self, args):
        from unittest import mock
        from backend import cli as _cli

        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(sys, "argv", ["vsr"] + args):
            with mock.patch("sys.stdout", stdout), mock.patch("sys.stderr", stderr):
                with self.assertRaises(SystemExit) as caught:
                    _cli.main()
        return caught.exception.code, stdout.getvalue(), stderr.getvalue()

    def test_soft_subtitle_dry_run_does_not_construct_remover(self):
        from unittest import mock
        from backend import cli as _cli

        stream = processor.SubtitleStreamInfo(
            index=2,
            codec_name="subrip",
            language="eng",
            title="SDH",
            default=True,
            forced=False,
        )
        with mock.patch.object(_cli, "_probe_subtitle_streams", return_value=[stream]):
            with mock.patch(
                "backend.processor.SubtitleRemover",
                side_effect=AssertionError("heavy backend should not load"),
            ):
                code, stdout, _stderr = self._run_cli([
                    "--input", "movie.mkv",
                    "--soft-subtitle-dry-run",
                ])

        self.assertEqual(code, 0)
        self.assertIn("action=inspect", stdout)
        self.assertIn("stream=2", stdout)
        self.assertIn("codec=subrip", stdout)
        self.assertIn("lang=eng", stdout)
        self.assertIn("title=SDH", stdout)
        self.assertIn("default=yes", stdout)

    def test_strip_soft_subtitles_remuxes_without_remover(self):
        from unittest import mock
        from backend import cli as _cli
        from backend.remux import SoftSubtitleAction

        with mock.patch.object(_cli, "_probe_subtitle_streams", return_value=[]):
            with mock.patch.object(_cli, "remux_soft_subtitles") as remux:
                with mock.patch(
                    "backend.processor.SubtitleRemover",
                    side_effect=AssertionError("heavy backend should not load"),
                ):
                    code, stdout, _stderr = self._run_cli([
                        "--input", "movie.mkv",
                        "--output", "out.mkv",
                        "--strip-soft-subtitles",
                    ])

        self.assertEqual(code, 0)
        remux.assert_called_once_with(
            "movie.mkv",
            "out.mkv",
            action=SoftSubtitleAction.STRIP,
        )
        self.assertIn("action=strip", stdout)

    def test_soft_subtitle_dry_run_writes_json_plan(self):
        from unittest import mock
        from backend import cli as _cli

        stream = processor.SubtitleStreamInfo(
            index=2,
            codec_name="subrip",
            language="eng",
            title="SDH",
            default=True,
            forced=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            first = work / "first.mkv"
            second = work / "second.mkv"
            plan = work / "soft-plan.json"
            first.write_bytes(b"not a real video")
            second.write_bytes(b"not a real video")

            with mock.patch.object(
                _cli, "_probe_subtitle_streams", return_value=[stream],
            ):
                with mock.patch(
                    "backend.processor.SubtitleRemover",
                    side_effect=AssertionError("heavy backend should not load"),
                ):
                    code, stdout, _stderr = self._run_cli([
                        "--pattern", str(work / "*.mkv"),
                        "--soft-subtitle-dry-run",
                        "--strip-soft-subtitles",
                        "--soft-subtitle-plan-json", str(plan),
                    ])

            payload = json.loads(plan.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertIn("wrote plan", stdout)
        self.assertEqual(payload["schema"], "vsr.soft_subtitle_preflight.v1")
        self.assertEqual(payload["action"], "strip")
        self.assertEqual(payload["count"], 2)
        self.assertEqual(
            [record["input_name"] for record in payload["files"]],
            ["first.mkv", "second.mkv"],
        )
        self.assertTrue(payload["files"][0]["has_soft_subtitles"])
        self.assertEqual(payload["files"][0]["subtitle_stream_count"], 1)
        self.assertEqual(
            payload["files"][0]["subtitle_streams"][0]["language"],
            "eng",
        )

    def test_soft_subtitle_plan_json_requires_dry_run(self):
        code, _stdout, stderr = self._run_cli([
            "--input", "movie.mkv",
            "--output", "out.mkv",
            "--strip-soft-subtitles",
            "--soft-subtitle-plan-json", "plan.json",
        ])
        self.assertEqual(code, 2)
        self.assertIn("requires --soft-subtitle-dry-run", stderr)

    def test_soft_subtitle_modes_are_mutually_exclusive(self):
        code, _stdout, stderr = self._run_cli([
            "--input", "movie.mkv",
            "--output", "out.mkv",
            "--strip-soft-subtitles",
            "--keep-soft-subtitles",
        ])
        self.assertEqual(code, 2)
        self.assertIn("mutually exclusive", stderr)


class CliBatchReportTests(unittest.TestCase):
    def _run_cli(self, args):
        from unittest import mock
        from backend import cli as _cli

        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(sys, "argv", ["vsr"] + args):
            with mock.patch("sys.stdout", stdout), mock.patch("sys.stderr", stderr):
                with self.assertRaises(SystemExit) as caught:
                    _cli.main()
        return caught.exception.code, stdout.getvalue(), stderr.getvalue()

    def _patch_preflight_probes(self):
        from unittest import mock
        from backend import batch_report as _br

        return mock.patch.multiple(
            _br,
            _probe_codec_for_log=mock.Mock(return_value="h264,640,360,30/1"),
            _probe_duration_seconds=mock.Mock(return_value=10.0),
            _probe_subtitle_streams=mock.Mock(return_value=[]),
        )

    def test_pattern_skip_existing_writes_report_without_alt_processing(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            src = work / "clip.mp4"
            out_dir = work / "out"
            ckpt = work / "ckpt"
            src.write_bytes(b"video")
            out_dir.mkdir()
            (out_dir / "clip_no_sub.mp4").write_bytes(b"done")
            fake_remover = SimpleNamespace(
                config=processor.ProcessingConfig(),
                process_video=mock.Mock(return_value=True),
                process_image=mock.Mock(return_value=True),
            )
            with self._patch_preflight_probes():
                with mock.patch("backend.processor.SubtitleRemover", return_value=fake_remover):
                    code, stdout, stderr = self._run_cli([
                        "--pattern", str(work / "*.mp4"),
                        "--out-dir", str(out_dir),
                        "--checkpoint-dir", str(ckpt),
                        "--skip-existing",
                    ])
            payload = json.loads((out_dir / "vsr-batch-summary.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0, stderr)
        fake_remover.process_video.assert_not_called()
        self.assertIn("[skip] clip.mp4 (output exists)", stdout)
        self.assertEqual(payload["files"][0]["status"], "skipped-existing")
        self.assertEqual(payload["files"][0]["output_name"], "clip_no_sub.mp4")

    def test_pattern_success_writes_processed_report(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            src = work / "clip.mp4"
            out_dir = work / "out"
            ckpt = work / "ckpt"
            src.write_bytes(b"video")
            out_dir.mkdir()
            fake_remover = SimpleNamespace(
                config=processor.ProcessingConfig(),
                process_video=mock.Mock(return_value=True),
                process_image=mock.Mock(return_value=True),
            )
            with self._patch_preflight_probes():
                with mock.patch("backend.processor.SubtitleRemover", return_value=fake_remover):
                    code, stdout, stderr = self._run_cli([
                        "--pattern", str(work / "*.mp4"),
                        "--out-dir", str(out_dir),
                        "--checkpoint-dir", str(ckpt),
                        "--gpu", "-1",
                    ])
            payload = json.loads((out_dir / "vsr-batch-summary.json").read_text(encoding="utf-8"))
            markdown = (out_dir / "vsr-batch-summary.md").read_text(encoding="utf-8")

        self.assertEqual(code, 0, stderr)
        fake_remover.process_video.assert_called_once()
        self.assertIn("[batch] wrote report", stdout)
        self.assertEqual(payload["counts"], {"hardcoded-processed": 1})
        self.assertEqual(payload["files"][0]["status"], "hardcoded-processed")
        self.assertIn("clip_no_sub.mp4", markdown)

    def test_pattern_retries_false_result_and_records_attempt(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            src = work / "clip.mp4"
            out_dir = work / "out"
            ckpt = work / "ckpt"
            src.write_bytes(b"video")
            out_dir.mkdir()

            calls = {"count": 0}

            def process_video(*_args, **_kwargs):
                calls["count"] += 1
                if calls["count"] == 1:
                    fake_remover.last_error_message = "CUDA out of memory"
                    fake_remover.last_error_reason = "video_processing_error"
                    return False
                fake_remover.last_error_message = None
                fake_remover.last_error_reason = None
                return True

            fake_remover = SimpleNamespace(
                config=processor.ProcessingConfig(),
                process_video=mock.Mock(side_effect=process_video),
                process_image=mock.Mock(return_value=True),
                last_mask_export={
                    "requested": False,
                    "status": "not-requested",
                    "path": "",
                },
            )
            with self._patch_preflight_probes():
                with mock.patch(
                    "backend.processor.SubtitleRemover",
                    return_value=fake_remover,
                ):
                    code, stdout, stderr = self._run_cli([
                        "--pattern", str(work / "*.mp4"),
                        "--out-dir", str(out_dir),
                        "--checkpoint-dir", str(ckpt),
                        "--gpu", "-1",
                        "--max-retries", "1",
                        "--retry-backoff", "0",
                    ])
            payload = json.loads(
                (out_dir / "vsr-batch-summary.json").read_text(
                    encoding="utf-8")
            )

        self.assertEqual(code, 0, stderr)
        self.assertEqual(fake_remover.process_video.call_count, 2)
        self.assertIn("[retry] clip.mp4: attempt 1/1", stdout)
        self.assertEqual(payload["files"][0]["retry_attempts"], 1)
        self.assertIn("CUDA out of memory", payload["files"][0]["retry_errors"][0])

    def test_paused_status_is_reported_distinctly(self):
        from backend import batch_report as _br

        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            src = work / "clip.mp4"
            out = work / "clip_no_sub.mp4"
            src.write_bytes(b"video")
            with self._patch_preflight_probes():
                record = _br.make_batch_item_record(
                    str(src),
                    str(out),
                    config=processor.ProcessingConfig(),
                )
            _br.finish_batch_item(
                record,
                _br.STATUS_PAUSED,
                message="Processing paused at frame 4/10",
                elapsed_seconds=1.25,
                stage_timings={"decode": 0.5, "inpaint": 0.75},
            )
            json_path, md_path = _br.write_batch_reports(
                work,
                [record],
                kind="hardcoded-cleanup",
                started_at=datetime.datetime.now(datetime.timezone.utc),
            )
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = md_path.read_text(encoding="utf-8")

        self.assertEqual(payload["counts"], {"paused": 1})
        self.assertEqual(payload["files"][0]["status"], "paused")
        self.assertIn("paused", markdown)

    def test_pattern_pause_writes_paused_report_and_exit_130(self):
        from unittest import mock
        from backend.resume_checkpoint import ProcessingPaused

        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            src = work / "clip.mp4"
            out_dir = work / "out"
            ckpt = work / "ckpt"
            src.write_bytes(b"video")
            out_dir.mkdir()
            fake_remover = SimpleNamespace(
                config=processor.ProcessingConfig(),
                process_video=mock.Mock(
                    side_effect=ProcessingPaused("Processing paused at frame 4/10")
                ),
                process_image=mock.Mock(return_value=True),
                last_stage_timings={"decode": 0.25},
            )
            with self._patch_preflight_probes():
                with mock.patch("backend.processor.SubtitleRemover", return_value=fake_remover):
                    code, stdout, stderr = self._run_cli([
                        "--pattern", str(work / "*.mp4"),
                        "--out-dir", str(out_dir),
                        "--checkpoint-dir", str(ckpt),
                        "--gpu", "-1",
                    ])
            payload = json.loads((out_dir / "vsr-batch-summary.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 130, stderr)
        self.assertIn("[batch] Paused", stdout)
        self.assertEqual(payload["files"][0]["status"], "paused")

    def test_soft_subtitle_pattern_writes_report(self):
        from unittest import mock
        from backend import cli as _cli

        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            src = work / "clip.mkv"
            out_dir = work / "out"
            src.write_bytes(b"video")
            out_dir.mkdir()
            with self._patch_preflight_probes():
                with mock.patch.object(_cli, "_probe_subtitle_streams", return_value=[]):
                    with mock.patch.object(_cli, "remux_soft_subtitles") as remux:
                        with mock.patch(
                            "backend.processor.SubtitleRemover",
                            side_effect=AssertionError("heavy backend should not load"),
                        ):
                            code, stdout, stderr = self._run_cli([
                                "--pattern", str(work / "*.mkv"),
                                "--out-dir", str(out_dir),
                                "--strip-soft-subtitles",
                            ])
            payload = json.loads((out_dir / "vsr-batch-summary.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0, stderr)
        remux.assert_called_once()
        self.assertIn("[batch] wrote report", stdout)
        self.assertEqual(payload["counts"], {"soft-subtitle-remuxed": 1})
        self.assertEqual(payload["files"][0]["soft_action"], "strip")


class LoadJsonConfigTests(unittest.TestCase):
    def test_load_json_config_rejects_oversized_file(self):
        """Files larger than 1 MB should raise ValueError without being parsed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            big = Path(tmpdir) / "big.json"
            # Write >1 MB of valid JSON; use enough entries to exceed the cap
            big.write_text("{" + ", ".join(f'"{i}": {i}' for i in range(150_000)) + "}",
                           encoding="utf-8")
            self.assertGreater(big.stat().st_size, 1 * 1024 * 1024,
                               "test fixture must be >1 MB")
            with self.assertRaises(ValueError):
                processor._load_json_config(str(big))


class CliNumericRangeTests(unittest.TestCase):
    def test_out_of_range_crf_is_clamped(self):
        from backend.config import ProcessingConfig, normalize_processing_config
        cfg = ProcessingConfig(output_quality=100)
        cfg = normalize_processing_config(cfg)
        self.assertLessEqual(cfg.output_quality, 51)

    def test_negative_mask_dilate_is_clamped(self):
        from backend.config import ProcessingConfig, normalize_processing_config
        cfg = ProcessingConfig(mask_dilate_px=-5)
        cfg = normalize_processing_config(cfg)
        self.assertGreaterEqual(cfg.mask_dilate_px, 0)

    def test_extreme_frame_skip_is_clamped(self):
        from backend.config import ProcessingConfig, normalize_processing_config
        cfg = ProcessingConfig(detection_frame_skip=9999)
        cfg = normalize_processing_config(cfg)
        self.assertLessEqual(cfg.detection_frame_skip, 240)


class DryRunCliTests(unittest.TestCase):
    """P2: full-pipeline --dry-run and machine-readable --json output."""

    def test_dry_run_plan_probes_without_encoding(self):
        import shutil as _sh
        import subprocess as _sp
        if _sh.which("ffmpeg") is None:
            self.skipTest("ffmpeg not installed")
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "clip.mp4"
            out = Path(tmp) / "out.mp4"
            _sp.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=1",
                 "-pix_fmt", "yuv420p", str(video)],
                check=True, timeout=60,
            )
            proc = _sp.run(
                [sys.executable, "-m", "backend.cli", "-i", str(video),
                 "-o", str(out), "--gpu", "-1", "--dry-run", "--json"],
                capture_output=True, text=True, timeout=300,
                cwd=str(Path(__file__).resolve().parents[1]),
            )
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            # stdout has log noise before JSON; parse the JSON object at the end
            start = proc.stdout.index("{")
            payload = json.loads(proc.stdout[start:])
            self.assertTrue(payload["dry_run"])
            self.assertEqual(len(payload["plans"]), 1)
            plan = payload["plans"][0]
            self.assertTrue(plan["is_video"])
            self.assertEqual(plan["frames"], 10)
            self.assertTrue(plan["codec_ok"])
            self.assertFalse(out.exists())  # dry-run never encodes



if __name__ == "__main__":
    unittest.main()
