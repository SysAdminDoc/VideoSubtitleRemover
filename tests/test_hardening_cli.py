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


class CliWatchFolderTests(unittest.TestCase):
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

    def test_watch_requires_stable_file_version_before_claiming(self):
        from backend import cli as _cli

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "clip.mp4"
            source.write_bytes(b"first")
            state = {}
            processed = set()

            ready, candidates = _cli._watch_ready_files(
                root,
                {".mp4"},
                state,
                processed,
                now=10.0,
                stable_seconds=2.0,
            )
            self.assertEqual(ready, [])
            self.assertEqual(candidates, 1)

            source.write_bytes(b"second-version")
            ready, candidates = _cli._watch_ready_files(
                root,
                {".mp4"},
                state,
                processed,
                now=11.0,
                stable_seconds=2.0,
            )
            self.assertEqual(ready, [])
            self.assertEqual(candidates, 1)

            ready, candidates = _cli._watch_ready_files(
                root,
                {".mp4"},
                state,
                processed,
                now=12.9,
                stable_seconds=2.0,
            )
            self.assertEqual(ready, [])
            self.assertEqual(candidates, 1)

            ready, candidates = _cli._watch_ready_files(
                root,
                {".mp4"},
                state,
                processed,
                now=13.0,
                stable_seconds=2.0,
            )
            self.assertEqual([path.name for path, _fingerprint in ready], ["clip.mp4"])
            self.assertEqual(candidates, 1)

    def test_watch_once_processes_file_dropped_during_drain_exactly_once(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            watch_dir = work / "incoming"
            out_dir = work / "out"
            checkpoint_dir = work / "checkpoints"
            watch_dir.mkdir()
            first = watch_dir / "first.mp4"
            second = watch_dir / "second.mp4"
            first.write_bytes(b"first")
            calls = []

            def process_video(input_path, *_args, **_kwargs):
                calls.append(Path(input_path).name)
                if len(calls) == 1:
                    second.write_bytes(b"second")
                return True

            fake_remover = SimpleNamespace(
                config=processor.ProcessingConfig(),
                process_video=mock.Mock(side_effect=process_video),
                process_image=mock.Mock(return_value=True),
            )
            with self._patch_preflight_probes():
                with mock.patch(
                    "backend.processor.SubtitleRemover",
                    return_value=fake_remover,
                ):
                    code, stdout, stderr = self._run_cli([
                        "--watch", str(watch_dir),
                        "--watch-once",
                        "--watch-stable-seconds", "0",
                        "--watch-interval", "0.1",
                        "--out-dir", str(out_dir),
                        "--checkpoint-dir", str(checkpoint_dir),
                        "--gpu", "-1",
                    ])
            payload = json.loads(
                (out_dir / "vsr-batch-summary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(code, 0, stderr)
        self.assertEqual(calls, ["first.mp4", "second.mp4"])
        self.assertEqual(fake_remover.process_video.call_count, 2)
        self.assertEqual(payload["kind"], "watch-folder")
        self.assertEqual(payload["counts"], {"hardcoded-processed": 2})
        self.assertIn("[watch] drain complete: 2/2 succeeded", stdout)

    def test_watch_failure_is_recorded_and_later_files_still_run(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            watch_dir = work / "incoming"
            out_dir = work / "out"
            watch_dir.mkdir()
            (watch_dir / "bad.mp4").write_bytes(b"bad")
            (watch_dir / "good.mp4").write_bytes(b"good")
            calls = []

            def process_video(input_path, *_args, **_kwargs):
                calls.append(Path(input_path).name)
                if Path(input_path).name == "bad.mp4":
                    raise ValueError("unsupported fixture")
                return True

            fake_remover = SimpleNamespace(
                config=processor.ProcessingConfig(),
                process_video=mock.Mock(side_effect=process_video),
                process_image=mock.Mock(return_value=True),
            )
            with self._patch_preflight_probes():
                with mock.patch(
                    "backend.processor.SubtitleRemover",
                    return_value=fake_remover,
                ):
                    code, _stdout, stderr = self._run_cli([
                        "--watch", str(watch_dir),
                        "--watch-once",
                        "--watch-stable-seconds", "0",
                        "--out-dir", str(out_dir),
                        "--gpu", "-1",
                    ])
            payload = json.loads(
                (out_dir / "vsr-batch-summary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(code, 1, stderr)
        self.assertEqual(calls, ["bad.mp4", "good.mp4"])
        self.assertEqual(
            payload["counts"],
            {"failed": 1, "hardcoded-processed": 1},
        )


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

    def test_auto_dilate_is_disabled_by_an_explicit_manual_radius(self):
        from backend.cli import (
            _build_parser,
            _build_processing_config,
            _prepare_cli_args,
        )
        from backend.config import (
            InpaintMode,
            ProcessingConfig,
            _coerce_backend_mode,
            normalize_processing_config,
        )

        parser = _build_parser([mode.value for mode in InpaintMode])
        argv = [
            "--input", "source.mp4", "--output", "clean.mp4",
            "--gpu", "-1", "--auto-dilate", "--mask-dilate", "12",
        ]
        args = parser.parse_args(argv)
        _prepare_cli_args(args, parser, argv)
        config = _build_processing_config(
            args,
            False,
            ProcessingConfig,
            _coerce_backend_mode,
            normalize_processing_config,
        )

        self.assertEqual(config.mask_dilate_px, 12)
        self.assertFalse(config.auto_dilate_enable)

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


class FrozenMatteCliErrorTests(unittest.TestCase):
    """RM-153: a refused freeze is a clean CLI error, not a traceback."""

    def test_a_bad_frozen_matte_manifest_is_a_parser_error(self):
        import subprocess as _sp
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "not-there.mask.json"
            proc = _sp.run(
                [sys.executable, "-m", "backend.cli",
                 "-i", str(Path(tmp) / "in.mp4"),
                 "-o", str(Path(tmp) / "out.mp4"),
                 "--gpu", "-1", "--frozen-matte", str(missing)],
                capture_output=True, text=True, timeout=300,
                cwd=str(Path(__file__).resolve().parents[1]),
            )
            self.assertEqual(proc.returncode, 2, proc.stderr[-2000:])
            self.assertIn("--frozen-matte", proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)


class ExecutedSurfaceTests(unittest.TestCase):
    """RM-295: surfaces an audit claimed to cover but never actually ran.

    Every other watch-folder test mocks process_video, so none of them
    proved a drain writes a real file. These run the surfaces for real on
    a tiny clip.
    """

    @staticmethod
    def _write_clip(path, frames=8, size=(160, 120), fps=10.0):
        import cv2
        import numpy as np

        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        for index in range(frames):
            frame = np.full((size[1], size[0], 3), 40, dtype=np.uint8)
            cv2.putText(frame, f"SUB {index}", (12, size[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            writer.write(frame)
        writer.release()
        return path

    def _run_real_cli(self, args):
        import subprocess
        import sys

        return subprocess.run(
            [sys.executable, "-m", "backend.cli", *args],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True, text=True, timeout=900,
        )

    def test_a_watch_drain_actually_writes_an_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            watch_dir = work / "incoming"
            out_dir = work / "out"
            watch_dir.mkdir()
            self._write_clip(watch_dir / "clip.mp4")
            proc = self._run_real_cli([
                "--watch", str(watch_dir), "--watch-once",
                "--watch-stable-seconds", "0", "--watch-interval", "0.1",
                "--out-dir", str(out_dir), "--mode", "sttn", "--gpu", "-1",
                "--no-audio", "--skip-detection", "--end", "0.4",
            ])
            outputs = sorted(path.name for path in out_dir.glob("*.mp4"))
            summary = out_dir / "vsr-batch-summary.json"
            self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
            self.assertEqual(outputs, ["clip_no_sub.mp4"])
            self.assertTrue(summary.is_file())
            payload = json.loads(summary.read_text(encoding="utf-8"))
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["counts"].get("hardcoded-processed"), 1)

    def test_the_nle_sidecar_round_trips_through_nle_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            clip = self._write_clip(work / "clip.mp4")
            output = work / "clip_clean.mp4"
            export = self._run_real_cli([
                "-i", str(clip), "-o", str(output), "--nle-sidecar", "edl",
                "--mode", "sttn", "--gpu", "-1", "--no-audio",
                "--skip-detection", "--end", "0.4",
            ])
            self.assertEqual(
                export.returncode, 0, export.stderr or export.stdout)
            edls = sorted(work.glob("*.edl"))
            self.assertEqual(
                len(edls), 1, f"expected one EDL, found {edls}")
            reimport = self._run_real_cli([
                "-i", str(clip), "--nle-input", str(edls[0]),
                "--validate-config",
            ])
        self.assertEqual(
            reimport.returncode, 0, reimport.stderr or reimport.stdout)
        self.assertIn("resolved_config", reimport.stdout)

    def test_the_vapoursynth_bridge_declines_cleanly_when_absent(self):
        """It must return None, not raise, when VapourSynth is not installed."""
        from backend import vapoursynth_bridge

        with tempfile.TemporaryDirectory() as tmpdir:
            script = Path(tmpdir) / "probe.vpy"
            script.write_text("import vapoursynth as vs\n", encoding="utf-8")
            try:
                import vapoursynth  # noqa: F401
            except ImportError:
                self.assertIsNone(vapoursynth_bridge.try_open_vpy(str(script)))
            else:
                self.skipTest("VapourSynth is installed on this host")

    def test_the_whisper_helpers_convert_segments_to_frame_spans(self):
        from backend import whisper_fallback

        spans = whisper_fallback.segments_to_frame_spans(
            [(0.0, 1.0, "hello"), (2.0, 3.0, "world")], fps=10.0)
        self.assertEqual(spans, [(0, 10), (20, 30)])
        self.assertIsInstance(whisper_fallback.is_available(), bool)
        self.assertIsInstance(
            whisper_fallback.ffmpeg_whisper_available(), bool)


if __name__ == "__main__":
    unittest.main()


class PresetFlagPrecedenceTests(unittest.TestCase):
    """RM-165 / RM-184: an explicitly typed flag must beat a preset.

    _explicitly_provided_dests exists so a preset cannot discard a value the
    user actually typed, but the field-to-dest maps covered only 14 fields.
    Anything else -- --keep-chyrons, --no-tbe, --max-retries, --vertical --
    fell into preset_backend_overrides and was applied unconditionally, so
    `--preset "Logo / Watermark removal" --keep-chyrons` removed chyrons
    anyway. Abbreviated and attached option forms were not recognised as
    explicit either.
    """

    def _parser(self):
        from backend.cli import _build_parser
        from backend.config import InpaintMode

        return _build_parser([mode.value for mode in InpaintMode])

    def test_every_preset_settable_field_with_a_flag_is_protected(self):
        from backend.cli import _build_parser  # noqa: F401
        from backend.config_schema import processing_field_names
        from gui.config import SAFE_PRESET_FIELDS

        parser = self._parser()
        dests = {action.dest for action in parser._actions}

        # Mirrors the maps in _prepare_cli_args; kept in sync by this test.
        field_to_attr = {
            "mode": "mode", "detection_threshold": "threshold",
            "mask_dilate_px": "mask_dilate", "mask_feather_px": "mask_feather",
            "edge_ring_px": "edge_ring", "tbe_flow_warp": "flow_warp",
            "tbe_flow_estimator": "flow_estimator",
            "paddleocr_variant": "paddleocr_variant",
            "poisson_seam_enable": "poisson_seam",
            "colour_tune_enable": "colour_tune",
            "colour_tune_tolerance": "colour_tolerance",
            "phash_skip_distance": "phash_distance", "auto_band": "auto_band",
            "detection_frame_skip": "frame_skip",
            "detection_vertical": "vertical",
            "confidence_weighted_dilation": "confidence_dilate",
            "temporal_smooth_radius": "temporal_smooth",
            "detection_denoise": "denoise_detect",
            "tbe_scene_cut_use_pyscenedetect": "pyscenedetect",
            "batch_max_retries": "max_retries",
            "batch_retry_backoff_seconds": "retry_backoff",
            "keyframe_detection": "keyframe_detect",
            "karaoke_x_gap_px": "karaoke_x_gap",
        }
        inverted_flags = {
            "tbe_global_motion_align": "no_global_motion_align",
            "tbe_scene_cut_split": "no_scene_split",
            "kalman_tracking": "no_kalman",
            "phash_skip_enable": "no_phash",
            "tbe_enable": "no_tbe",
            "adaptive_batch": "no_adaptive_batch",
            "deinterlace_auto": "no_deinterlace_detect",
            "remove_chyrons": "keep_chyrons",
            "remove_subtitles": "keep_subtitles",
        }

        for field, dest in {**field_to_attr, **inverted_flags}.items():
            with self.subTest(field=field):
                self.assertIn(
                    dest, dests,
                    f"{field} maps to dest {dest!r}, which the parser does "
                    "not define",
                )

        fields = set(processing_field_names())
        protected = (
            set(field_to_attr)
            | set(inverted_flags)
            | {name for name in fields if name in dests}
        )
        unprotected = sorted((fields & SAFE_PRESET_FIELDS) - protected)
        # Fields with no CLI flag at all legitimately ride
        # preset_backend_overrides; only flag-backed ones must be protected.
        flag_backed = {
            field for field in unprotected
            if field.replace("_", "-") in {
                opt.lstrip("-")
                for action in parser._actions
                for opt in action.option_strings
            }
        }
        self.assertEqual(
            sorted(flag_backed), [],
            "these preset-settable fields have a CLI flag but no "
            "explicit-flag protection, so a preset overrides what the user "
            "typed",
        )

    def test_explicit_inverted_flag_beats_a_preset(self):
        from backend.cli import _explicitly_provided_dests

        parser = self._parser()
        argv = ["--preset", "Logo / Watermark removal", "--keep-chyrons"]
        explicit = _explicitly_provided_dests(parser, argv)
        self.assertIn("keep_chyrons", explicit)

    def test_attached_short_option_counts_as_explicit(self):
        from backend.cli import _explicitly_provided_dests

        parser = self._parser()
        explicit = _explicitly_provided_dests(parser, ["-msttn"])
        self.assertIn("mode", explicit)

    def test_abbreviated_flags_are_rejected_rather_than_silently_ignored(self):
        parser = self._parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--thresho", "0.8"])


class CliConfigOverlayTests(unittest.TestCase):
    def test_unknown_config_field_fails_closed(self):
        import argparse
        from unittest import mock

        from backend import cli as _cli
        from backend.config import ProcessingConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cfg.json"
            path.write_text(
                '{"not_a_real_field": 1, "output_quality": 20}',
                encoding="utf-8",
            )
            args = SimpleNamespace(
                config=str(path),
                config_schema_version=None,
                config_overrides=None,
                validate_config=False,
                plan_in="",
                _preset_backend_overrides=None,
            )
            parser = argparse.ArgumentParser(prog="vsr")
            stderr = io.StringIO()
            with mock.patch("sys.stderr", stderr):
                with self.assertRaises(SystemExit) as caught:
                    _cli._apply_cli_config_overlays(
                        args, parser, ProcessingConfig())
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("not_a_real_field", stderr.getvalue())
        self.assertIn("unknown config field", stderr.getvalue())
