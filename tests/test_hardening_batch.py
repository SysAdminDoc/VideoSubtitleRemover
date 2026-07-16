import io
import json
import os
import sys
import tempfile
import threading
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace


import VideoSubtitleRemover as gui
from backend import processor


def _has_display() -> bool:
    """Return True if a GUI display is available."""
    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class BatchReportTests(unittest.TestCase):
    def test_output_quality_preflight_warns_for_high_crf_source_risk(self):
        from backend.output_quality_preflight import (
            evaluate_output_quality_preflight,
            output_quality_preflight_messages,
        )

        cfg = SimpleNamespace(output_codec="h264", output_quality=31)
        preflight = evaluate_output_quality_preflight(
            "clip.mp4",
            cfg,
            source={
                "ok": True,
                "codec": "h264",
                "width": 1920,
                "height": 1080,
                "bitrate_bps": 14_000_000,
                "bitrate_source": "stream",
                "frame_rate": "30000/1001",
            },
        )

        self.assertEqual(preflight["schema"], "vsr.output_quality_preflight.v1")
        self.assertEqual(preflight["status"], "warning")
        self.assertTrue(preflight["overrideRequired"])
        self.assertTrue(preflight["overridden"])
        self.assertEqual(preflight["source"]["bitrate_bps"], 14_000_000)
        self.assertIn("CRF", output_quality_preflight_messages(preflight)[0])
        self.assertIn("Suggested safer output setting", preflight["recommendation"])

    def test_output_quality_preflight_ignores_remux_copy(self):
        from backend.output_quality_preflight import evaluate_output_quality_preflight

        preflight = evaluate_output_quality_preflight(
            "clip.mkv",
            {"output_codec": "copy", "output_quality": 23},
            source={
                "ok": True,
                "codec": "h264",
                "width": 1920,
                "height": 1080,
                "bitrate_bps": 10_000_000,
            },
        )

        self.assertEqual(preflight["status"], "not_applicable")
        self.assertEqual(preflight["warnings"], [])

    def test_batch_report_marks_soft_remux_quality_preflight_not_applicable(self):
        from unittest import mock
        from backend import batch_report as _br

        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            source = work / "clip.mkv"
            output = work / "out" / "clip_no_sub.mkv"
            source.write_bytes(b"video")
            cfg = SimpleNamespace(output_codec="h264", output_quality=31)

            with mock.patch.object(_br, "_probe_codec_for_log", return_value="h264,1920,1080,30000/1001"):
                with mock.patch.object(_br, "_probe_duration_seconds", return_value=12.5):
                    with mock.patch.object(_br, "_probe_subtitle_streams", return_value=[]):
                        with mock.patch.object(_br, "evaluate_output_quality_preflight") as evaluate:
                            record = _br.make_batch_item_record(
                                str(source),
                                str(output),
                                config=cfg,
                                soft_action="strip",
                            )

        evaluate.assert_not_called()
        self.assertEqual(
            record["output_quality_preflight"]["status"],
            "not_applicable",
        )
        self.assertIn("remux", record["output_quality_preflight"]["reason"])

    def test_choose_batch_output_path_honors_skip_existing(self):
        from backend.batch_report import choose_batch_output_path

        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            source = work / "clip.mp4"
            out_dir = work / "out"
            out_dir.mkdir()
            source.write_bytes(b"video")
            existing = out_dir / "clip_no_sub.mp4"
            existing.write_bytes(b"done")

            collision_safe = choose_batch_output_path(
                str(source),
                out_dir,
                "_no_sub",
                set(),
                skip_existing=False,
            )
            skip_target = choose_batch_output_path(
                str(source),
                out_dir,
                "_no_sub",
                set(),
                skip_existing=True,
            )

        self.assertEqual(collision_safe.name, "clip_no_sub(2).mp4")
        self.assertEqual(skip_target.name, "clip_no_sub.mp4")

    def test_write_batch_reports_includes_preflight_and_result_status(self):
        import datetime as _dt
        from unittest import mock
        from backend import batch_report as _br

        stream = processor.SubtitleStreamInfo(
            index=3,
            codec_name="subrip",
            language="eng",
            title="CC",
            default=True,
            forced=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            source = work / "clip.mkv"
            output = work / "out" / "clip_no_sub.mkv"
            source.write_bytes(b"video")
            cfg = SimpleNamespace(
                mode=processor.InpaintMode.AUTO,
                device="cuda:0",
                output_codec="h265",
            )
            preflight = {
                "schema": "vsr.output_quality_preflight.v1",
                "status": "warning",
                "source": {
                    "codec": "h264",
                    "width": 1920,
                    "height": 1080,
                    "bitrate_bps": 14_000_000,
                },
                "output": {"codec": "h265", "crf": 30},
                "warnings": [{
                    "id": "OUTPUT-CRF-SOURCE-RISK",
                    "message": "CRF 30 is above the source-aware recommendation.",
                }],
                "recommendation": "Suggested safer output setting: CRF 22 or lower.",
                "overrideRequired": True,
                "overridden": True,
                "reason": "",
            }
            with mock.patch.object(_br, "_probe_codec_for_log", return_value="h264,1920,1080,30000/1001"):
                with mock.patch.object(_br, "_probe_duration_seconds", return_value=12.5):
                    with mock.patch.object(_br, "_probe_subtitle_streams", return_value=[stream]):
                        with mock.patch.object(_br, "evaluate_output_quality_preflight", return_value=preflight):
                            record = _br.make_batch_item_record(
                                str(source),
                                str(output),
                                config=cfg,
                            )
            _br.finish_batch_item(
                record,
                _br.STATUS_HARDCODED_PROCESSED,
                message="Processed",
                elapsed_seconds=3.25,
                stage_timings={
                    "decode": 0.5,
                    "ocr": 1.25,
                    "mask": 0.25,
                    "inpaint": 2.0,
                    "encode": 0.75,
                    "mux": 0.4,
                    "quality": 0.2,
                },
                quality_report={
                    "tag": "Review",
                    "samples": 3,
                    "psnr": 31.0,
                    "ssim": 0.99,
                    "roi_ssim": 0.90,
                    "sheet": "clip.qualitysheet.png",
                },
            )
            started = _dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc)
            json_path, md_path = _br.write_batch_reports(
                output.parent,
                [record],
                kind="hardcoded-cleanup",
                started_at=started,
                completed_at=started + _dt.timedelta(seconds=4),
            )
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = md_path.read_text(encoding="utf-8")

        self.assertEqual(payload["schema"], "vsr.batch_summary.v1")
        self.assertEqual(payload["counts"], {"review-needed": 1})
        self.assertEqual(payload["files"][0]["status"], "review-needed")
        self.assertIn("quality gate: manual-review", payload["files"][0]["message"])
        self.assertEqual(payload["files"][0]["source_width"], 1920)
        self.assertEqual(payload["files"][0]["subtitle_stream_count"], 1)
        self.assertGreater(payload["files"][0]["estimated_seconds"], 0)
        self.assertEqual(payload["stage_summary"]["slowest_stage"]["name"], "inpaint")
        self.assertEqual(payload["files"][0]["dominant_stage"]["name"], "inpaint")
        self.assertEqual(payload["files"][0]["stage_timings"]["ocr"], 1.25)
        self.assertEqual(
            payload["files"][0]["output_quality_preflight"]["status"],
            "warning",
        )
        self.assertTrue(payload["files"][0]["output_quality_preflight"]["overridden"])
        self.assertIn("Stage timing summary", markdown)
        self.assertIn("Per-item stage timings", markdown)
        self.assertIn("slowest inpaint 2.0s", markdown)
        self.assertEqual(payload["files"][0]["quality_gate"]["status"], "review")
        self.assertEqual(
            payload["files"][0]["quality_gate"]["ladderStep"],
            "manual-review",
        )
        self.assertEqual(
            payload["files"][0]["quality_gate"]["previewFramePaths"],
            ["clip.qualitysheet.png"],
        )
        self.assertIn("remediation", payload["files"][0]["quality_gate"])
        self.assertIn("reasons", payload["files"][0]["quality_gate"])
        self.assertIsInstance(payload["files"][0]["quality_gate"]["reasons"], list)
        self.assertIn("degradedMetrics", payload["files"][0]["quality_gate"])
        self.assertIn("| review-needed | clip.mkv | clip_no_sub.mkv |", markdown)
        self.assertIn("Output quality preflight notes", markdown)
        self.assertIn("Suggested safer output setting", markdown)
        self.assertIn("review (manual-review)", markdown)
        self.assertIn("Quality review notes", markdown)


class NleSidecarTests(unittest.TestCase):
    """RM-76: EDL and FCPXML writers must produce well-formed sidecars
    with the source / cleaned filenames and the processed time range."""

    def test_edl_round_trip(self):
        from backend import nle_sidecar
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "out.edl")
            written = nle_sidecar.write_edl(
                path,
                source="C:/clips/source.mp4",
                cleaned="C:/clips/source_no_sub.mp4",
                fps=24.0, start_s=0.0, end_s=10.0,
            )
            text = Path(written).read_text(encoding="ascii")
        self.assertIn("TITLE: VSR cleanup", text)
        self.assertIn("FROM CLIP NAME: source.mp4", text)
        self.assertIn("TO CLIP NAME:   source_no_sub.mp4", text)
        self.assertIn("00:00:00:00", text)

    def test_fcpxml_round_trip(self):
        from backend import nle_sidecar
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "out.fcpxml")
            written = nle_sidecar.write_fcpxml(
                path,
                source="/clips/source.mp4",
                cleaned=str(Path(tmpdir) / "cleaned.mp4"),
                fps=24.0, start_s=0.0, end_s=10.0,
            )
            text = Path(written).read_text(encoding="utf-8")
        self.assertIn("<fcpxml version=\"1.10\">", text)
        self.assertIn("frameDuration=\"1/24s\"", text)


class CrashReporterScaffoldTests(unittest.TestCase):
    """RM-52: opt-in crash reporting must be OFF unless both env vars
    are set, and the path scrubber must hide local layout info."""

    def setUp(self):
        self._saved = {
            "VSR_GLITCHTIP_DSN": os.environ.pop("VSR_GLITCHTIP_DSN", None),
            "VSR_CRASH_REPORTS": os.environ.pop("VSR_CRASH_REPORTS", None),
        }

    def tearDown(self):
        for k, v in self._saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def test_disabled_by_default(self):
        from backend.crash_reporter import is_enabled, install
        self.assertFalse(is_enabled())
        self.assertFalse(install())

    def test_partial_consent_is_not_enough(self):
        from backend.crash_reporter import is_enabled
        os.environ["VSR_CRASH_REPORTS"] = "1"
        self.assertFalse(is_enabled(), "DSN missing -> still disabled")
        os.environ.pop("VSR_CRASH_REPORTS", None)
        os.environ["VSR_GLITCHTIP_DSN"] = "https://example/0"
        self.assertFalse(is_enabled(), "consent flag missing -> still disabled")

    def test_path_scrub_drops_windows_paths(self):
        from backend.crash_reporter import _path_scrub
        sample = "File \"C:\\Users\\xxx\\repos\\VSR\\backend\\processor.py\", line 1"
        scrubbed = _path_scrub(sample)
        self.assertNotIn("Users", scrubbed)
        self.assertIn("<path>", scrubbed)

    def test_before_send_scrubs_nested_paths_and_drops_locals(self):
        from backend.crash_reporter import _before_send
        home = "C:\\Users\\xxx"
        event = {
            "message": home + "\\top.mp4",
            "exception": {"values": [{
                "type": "ValueError",
                "value": "Cannot open " + home + "\\secret\\clip.mp4",
                "stacktrace": {"frames": [{
                    "abs_path": home + "\\app.py",
                    "vars": {"frame": "huge-array"},
                }]},
            }]},
            "breadcrumbs": {"values": [{"message": "read " + home + "\\a.srt"}]},
            "extra": {"path": home + "\\out.mp4"},
        }
        out = _before_send(event, {})
        import json
        blob = json.dumps(out)
        self.assertNotIn("Users", blob)          # no user path leaks anywhere
        self.assertNotIn("huge-array", blob)     # frame locals dropped
        self.assertIn("<path>", blob)


class UpdateCheckTests(unittest.TestCase):
    """RM-116: optional startup update check."""

    def _join(self, thread):
        if thread is not None:
            thread.join(timeout=5)
        if thread is not None and thread.is_alive():
            self.fail("update check thread did not finish")

    def _response(self, payload, headers=None):
        fake_resp = io.BytesIO(json.dumps(payload).encode())
        fake_resp.status = 200
        fake_resp.headers = headers or {}
        fake_resp.__enter__ = lambda s: s
        fake_resp.__exit__ = lambda s, *a: None
        return fake_resp

    def test_parse_version_standard(self):
        from backend.update_check import _parse_version
        self.assertEqual(_parse_version("v3.16.1"), (3, 16, 1))
        self.assertEqual(_parse_version("3.16.1"), (3, 16, 1))

    def test_parse_version_garbage(self):
        from backend.update_check import _parse_version
        self.assertIsNone(_parse_version(""))
        self.assertIsNone(_parse_version("latest"))

    def test_no_callback_when_current_is_latest(self):
        from unittest.mock import MagicMock, patch
        from backend.update_check import check_for_update
        cb = MagicMock()
        fake_resp = self._response({
            "tag_name": "v3.16.1",
            "html_url": "https://example.com/releases/v3.16.1",
        })
        with patch("backend.update_check.urlopen", return_value=fake_resp):
            t = check_for_update("3.16.1", cb)
            self._join(t)
        cb.assert_not_called()

    def test_callback_when_newer_available(self):
        from unittest.mock import MagicMock, patch
        from backend.update_check import check_for_update
        cb = MagicMock()
        fake_resp = self._response({
            "tag_name": "v4.0.0",
            "html_url": "https://example.com/releases/v4.0.0",
        })
        with patch("backend.update_check.urlopen", return_value=fake_resp):
            t = check_for_update("3.16.1", cb)
            self._join(t)
        cb.assert_called_once_with("v4.0.0", "https://example.com/releases/v4.0.0")

    def test_no_crash_on_network_error(self):
        from unittest.mock import MagicMock, patch
        from backend.update_check import check_for_update
        cb = MagicMock()
        with patch("backend.update_check.urlopen", side_effect=OSError("offline")):
            t = check_for_update("3.16.1", cb)
            self._join(t)
        cb.assert_not_called()

    def test_request_headers_and_conditional_state_are_used(self):
        from unittest.mock import MagicMock, patch
        from backend.update_check import check_for_update
        cb = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "update_check.json"
            state_path.write_text(json.dumps({
                "etag": '"abc123"',
                "last_modified": "Wed, 17 Jun 2026 12:00:00 GMT",
            }), encoding="utf-8")
            captured = {}

            def fake_urlopen(req, timeout):
                captured["req"] = req
                captured["timeout"] = timeout
                return self._response({
                    "tag_name": "v3.16.1",
                    "html_url": "https://example.com/releases/v3.16.1",
                }, headers={
                    "ETag": '"def456"',
                    "Last-Modified": "Thu, 18 Jun 2026 12:00:00 GMT",
                })

            with patch("backend.update_check.urlopen", side_effect=fake_urlopen):
                t = check_for_update("3.16.1", cb, state_path=state_path)
                self._join(t)

            req = captured["req"]
            self.assertEqual(req.get_header("Accept"), "application/vnd.github+json")
            self.assertEqual(req.get_header("X-github-api-version"), "2022-11-28")
            self.assertIn("VideoSubtitleRemover/3.16.1", req.get_header("User-agent"))
            self.assertEqual(req.get_header("If-none-match"), '"abc123"')
            self.assertEqual(
                req.get_header("If-modified-since"),
                "Wed, 17 Jun 2026 12:00:00 GMT",
            )
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["etag"], '"def456"')
            self.assertEqual(
                saved["last_modified"],
                "Thu, 18 Jun 2026 12:00:00 GMT",
            )

    def test_not_modified_response_is_no_update(self):
        from urllib.error import HTTPError
        from unittest.mock import MagicMock, patch
        from backend.update_check import check_for_update
        cb = MagicMock()
        err = HTTPError("url", 304, "Not Modified", {}, None)
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "update_check.json"
            with patch("backend.update_check.urlopen", side_effect=err):
                t = check_for_update("3.16.1", cb, state_path=state_path)
                self._join(t)
        cb.assert_not_called()

    def test_rate_limit_sets_backoff_and_skips_next_request(self):
        from urllib.error import HTTPError
        from unittest.mock import MagicMock, patch
        from backend.update_check import check_for_update
        cb = MagicMock()
        err = HTTPError("url", 429, "Too Many Requests", {"Retry-After": "60"}, None)
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "update_check.json"
            with patch("backend.update_check.time.time", return_value=1000):
                with patch("backend.update_check.urlopen", side_effect=err) as mocked:
                    t = check_for_update("3.16.1", cb, state_path=state_path)
                    self._join(t)
                    self.assertEqual(mocked.call_count, 1)
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["backoff_until"], 1060)
            with patch("backend.update_check.time.time", return_value=1001):
                with patch("backend.update_check.urlopen") as mocked:
                    t = check_for_update("3.16.1", cb, state_path=state_path)
                    self._join(t)
                    mocked.assert_not_called()
        cb.assert_not_called()

    def test_config_field_defaults_off(self):
        cfg = gui.ProcessingConfig()
        self.assertFalse(cfg.update_check)

    def test_config_round_trip(self):
        cfg = gui.ProcessingConfig()
        cfg.update_check = True
        d = cfg.to_dict()
        self.assertTrue(d["update_check"])
        cfg2 = gui.ProcessingConfig.from_dict(d)
        self.assertTrue(cfg2.update_check)


class QueueAutosaveRoundTripTests(unittest.TestCase):
    def setUp(self):
        import gui.config as gui_config
        self._tmp = tempfile.TemporaryDirectory()
        self._old = gui_config.QUEUE_STATE_FILE
        gui_config.QUEUE_STATE_FILE = Path(self._tmp.name) / "queue.json"

    def tearDown(self):
        import gui.config as gui_config
        gui_config.QUEUE_STATE_FILE = self._old
        self._tmp.cleanup()

    def test_save_load_clear_round_trip(self):
        import gui.config as gui_config
        item = gui_config.QueueItem(
            id="rt-1",
            file_path="video.mp4",
            output_path="cleaned.mp4",
            output_path_locked=True,
            config=gui_config.ProcessingConfig(
                detection_lang="ja",
                output_quality=18,
            ),
            soft_subtitle_streams=[{"index": 2, "codec": "ass"}],
            soft_subtitle_probe_done=True,
            soft_subtitle_action="strip",
            retry_config={"source": "quality_gate"},
        )
        gui_config.save_queue_state([item])
        loaded = gui_config.load_queue_state()
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["id"], "rt-1")
        self.assertEqual(loaded[0]["file_path"], "video.mp4")
        self.assertTrue(loaded[0]["output_path_locked"])
        self.assertEqual(loaded[0]["config"]["detection_lang"], "ja")
        self.assertEqual(loaded[0]["soft_subtitle_action"], "strip")
        self.assertTrue(loaded[0]["soft_subtitle_probe_done"])
        self.assertEqual(loaded[0]["soft_subtitle_streams"][0]["codec"], "ass")
        self.assertEqual(loaded[0]["retry_config"]["source"], "quality_gate")
        self.assertEqual(
            json.loads(gui_config.QUEUE_STATE_FILE.read_text(encoding="utf-8"))["schema"],
            gui_config.QUEUE_STATE_SCHEMA,
        )
        gui_config.clear_queue_state()
        self.assertIsNone(gui_config.load_queue_state())

    def test_completed_items_are_not_saved(self):
        import gui.config as gui_config
        idle = gui_config.QueueItem(
            id="idle-1", file_path="a.mp4", output_path="a_out.mp4",
            config=gui_config.ProcessingConfig(),
        )
        done = gui_config.QueueItem(
            id="done-1", file_path="b.mp4", output_path="b_out.mp4",
            config=gui_config.ProcessingConfig(),
            status=gui_config.ProcessingStatus.COMPLETE,
        )
        gui_config.save_queue_state([idle, done])
        loaded = gui_config.load_queue_state()
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["file_path"], "a.mp4")

    def test_inflight_item_is_saved_as_resumable(self):
        import gui.config as gui_config
        running = gui_config.QueueItem(
            id="running-1", file_path="a.mp4", output_path="a_out.mp4",
            config=gui_config.ProcessingConfig(),
            status=gui_config.ProcessingStatus.PROCESSING,
            progress=0.25,
        )
        gui_config.save_queue_state([running])
        loaded = gui_config.load_queue_state()
        self.assertEqual(loaded[0]["status"], "idle")
        self.assertIn("interrupted session", loaded[0]["message"])

    def test_paused_items_are_saved_for_resume(self):
        import gui.config as gui_config
        paused = gui_config.QueueItem(
            id="paused-1", file_path="a.mp4", output_path="a_out.mp4",
            config=gui_config.ProcessingConfig(),
            status=gui_config.ProcessingStatus.PAUSED,
            progress=0.4,
            message="Paused at checkpoint",
            pause_checkpoint_path="checkpoints/demo.pause.json",
        )
        gui_config.save_queue_state([paused])
        loaded = gui_config.load_queue_state()
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["status"], "paused")
        self.assertEqual(loaded[0]["progress"], 0.4)
        self.assertEqual(
            loaded[0]["pause_checkpoint_path"],
            "checkpoints/demo.pause.json",
        )

    def test_corrupt_state_is_quarantined(self):
        import gui.config as gui_config
        gui_config.QUEUE_STATE_FILE.write_text("{broken", encoding="utf-8")

        self.assertIsNone(gui_config.load_queue_state())
        self.assertFalse(gui_config.QUEUE_STATE_FILE.exists())
        quarantined = list(
            Path(self._tmp.name).glob("queue.corrupt-*.json"))
        self.assertEqual(len(quarantined), 1)

    def test_two_restore_cycles_preserve_order_outputs_and_behavior(self):
        import gui.config as gui_config

        work = Path(self._tmp.name)
        first_source = work / "first.mp4"
        second_source = work / "second.mp4"
        first_source.write_bytes(b"first")
        second_source.write_bytes(b"second")
        first = gui_config.QueueItem(
            id="first-id",
            file_path=str(first_source),
            output_path=str(work / "custom-first.mkv"),
            output_path_locked=True,
            config=gui_config.ProcessingConfig(output_codec="h265"),
            soft_subtitle_streams=[{"index": 4, "codec": "ass"}],
            soft_subtitle_probe_done=True,
            soft_subtitle_action="strip",
        )
        second = gui_config.QueueItem(
            id="second-id",
            file_path=str(second_source),
            output_path=str(work / "custom-second.mp4"),
            output_path_locked=True,
            config=gui_config.ProcessingConfig(detection_lang="ja"),
            status=gui_config.ProcessingStatus.PAUSED,
            progress=0.4,
            message="Paused at checkpoint",
            pause_checkpoint_path=str(work / "second.pause.json"),
            soft_subtitle_probe_done=True,
        )
        gui_config.save_queue_state([first, second])

        def make_app():
            app = gui.VideoSubtitleRemoverApp.__new__(
                gui.VideoSubtitleRemoverApp)
            app.queue = []
            app.queue_lock = threading.Lock()
            app.queue_widgets = {}
            app.root = unittest.mock.Mock()
            app._selected_queue_item_id = None
            app._update_queue_display = unittest.mock.Mock()
            app._update_status = unittest.mock.Mock()
            app._start_soft_subtitle_probe = unittest.mock.Mock()
            return app

        with unittest.mock.patch("gui.app.show_confirm", return_value=True):
            first_app = make_app()
            first_app._maybe_restore_queue()
            self.assertTrue(gui_config.QUEUE_STATE_FILE.exists())
            second_app = make_app()
            second_app._maybe_restore_queue()

        self.assertEqual(
            [item.id for item in second_app.queue],
            ["first-id", "second-id"],
        )
        self.assertEqual(
            [item.output_path for item in second_app.queue],
            [str(work / "custom-first.mkv"), str(work / "custom-second.mp4")],
        )
        self.assertTrue(all(item.output_path_locked for item in second_app.queue))
        self.assertEqual(second_app.queue[0].soft_subtitle_action, "strip")
        self.assertEqual(second_app.queue[0].config.output_codec, "h265")
        self.assertEqual(second_app.queue[1].status, gui_config.ProcessingStatus.PAUSED)
        self.assertEqual(second_app.queue[1].config.detection_lang, "ja")


class BatchRetryTests(unittest.TestCase):
    """P2: automatic bounded retry for transient batch failures."""

    def test_retriable_classification(self):
        from backend.batch_report import is_retriable_error
        import subprocess as sp
        self.assertTrue(is_retriable_error(RuntimeError("CUDA out of memory")))
        self.assertTrue(is_retriable_error(sp.TimeoutExpired("ffmpeg", 10)))
        self.assertTrue(is_retriable_error(BrokenPipeError()))
        self.assertTrue(is_retriable_error(MemoryError()))
        # permanent
        self.assertFalse(is_retriable_error(FileNotFoundError("missing.mp4")))
        self.assertFalse(is_retriable_error(ValueError("insufficient disk space")))
        self.assertFalse(is_retriable_error(ValueError("unsupported codec")))
        self.assertFalse(is_retriable_error(RuntimeError("invalid config")))

    def test_config_default_off_and_clamped(self):
        from backend.config import ProcessingConfig, normalize_processing_config
        cfg = ProcessingConfig()
        self.assertEqual(cfg.batch_max_retries, 0)
        cfg.batch_max_retries = 99
        normalize_processing_config(cfg)
        self.assertLessEqual(cfg.batch_max_retries, 10)

    def test_gui_retries_false_result_and_preserves_attempt_evidence(self):
        from unittest import mock

        app = gui.VideoSubtitleRemoverApp.__new__(gui.VideoSubtitleRemoverApp)
        app._update_item_display = mock.Mock()
        app._process_soft_subtitle_item = mock.Mock(return_value=False)
        app._announce_model_download_guidance = mock.Mock()
        app._gui_to_backend_mode = mock.Mock(
            return_value=processor.InpaintMode.STTN)
        app._gui_to_backend_device = mock.Mock(return_value="cpu")
        app._cached_remover = None
        app._cached_remover_key = None
        app._active_remover = None
        app._batch_report_records = {"retry-item": {}}
        app.cancel_event = threading.Event()
        app.pause_event = threading.Event()
        app._batch_times = []

        class FakeBackendRemover:
            calls = 0

            def __init__(self, config):
                self.config = config
                self.last_error_message = None
                self.last_error_reason = None
                self.last_quality_report = None
                self.last_output_path = None
                self.last_mask_export = {
                    "requested": False,
                    "status": "not-requested",
                    "path": "",
                }

            def process_video(self, *_args, **_kwargs):
                type(self).calls += 1
                if type(self).calls == 1:
                    self.last_error_message = "CUDA out of memory"
                    self.last_error_reason = "video_processing_error"
                    return False
                self.last_error_message = None
                self.last_error_reason = None
                return True

        cfg = gui.ProcessingConfig()
        cfg.batch_max_retries = 1
        cfg.batch_retry_backoff_seconds = 0.0
        item = gui.QueueItem(
            id="retry-item",
            file_path="clip.mp4",
            output_path="clip_no_sub.mp4",
            config=cfg,
        )
        with mock.patch(
            "backend.processor.SubtitleRemover", FakeBackendRemover,
        ):
            app._process_item(item)

        self.assertEqual(FakeBackendRemover.calls, 2)
        self.assertEqual(item.status, gui.ProcessingStatus.COMPLETE)
        self.assertEqual(item.retry_attempts, 1)
        self.assertIn("CUDA out of memory", item.retry_errors[0])
        self.assertEqual(
            app._batch_report_records[item.id]["retry_attempts"], 1)



if __name__ == "__main__":
    unittest.main()
