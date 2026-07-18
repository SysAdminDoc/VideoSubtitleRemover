import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import VideoSubtitleRemover as gui
from backend import processor


def _has_display() -> bool:
    """Return True if a GUI display is available."""
    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class MediaInputFailureTests(unittest.TestCase):
    def _minimal_remover(self, work: Path):
        remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        remover.config = processor.ProcessingConfig(
            preserve_audio=False,
            preserve_color_metadata=False,
            deinterlace=False,
            deinterlace_auto=False,
            keyframe_detection=False,
            prefetch_decode=False,
            sttn_max_load_num=2,
        )
        remover.last_output_path = None
        remover.last_error_message = None
        remover.last_error_reason = None
        remover._srt_entries = []
        remover._quality_mask_bbox = None
        remover._color_metadata = None
        remover._active_writer = None
        remover._active_subprocess = None
        remover._teardown_requested = False
        remover.on_preview_frame = None
        remover.live_preview_stride = 6
        remover._report_progress = lambda *_args, **_kwargs: None
        remover._set_active_subprocess = lambda *_args, **_kwargs: None
        remover._is_teardown_requested = lambda: False

        def make_temp_dir():
            path = work / "vsr-temp"
            path.mkdir()
            return str(path)

        remover._make_temp_dir = make_temp_dir

        class FakeDetector:
            def detect(self, *_args, **_kwargs):
                return []

            def detect_with_confidence(self, *_args, **_kwargs):
                return []

        class FakeInpainter:
            def inpaint(self, frames, _masks):
                return frames

        remover.detector = FakeDetector()
        remover.inpainter = FakeInpainter()
        return remover

    def test_zero_byte_video_fails_cleanly_without_temp_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            source = work / "empty.mp4"
            output = work / "empty_no_sub.mp4"
            source.write_bytes(b"")
            remover = self._minimal_remover(work)

            with self.assertLogs("backend.processor", level="WARNING") as logs:
                ok = remover.process_video(str(source), str(output))

            self.assertFalse(ok)
            self.assertEqual(remover.last_error_reason, "empty_file")
            self.assertIn("empty", remover.last_error_message.lower())
            self.assertFalse(output.exists())
            self.assertFalse((work / "vsr-temp").exists())
            self.assertNotIn("Traceback", "\n".join(logs.output))

    def test_partial_decode_fails_and_removes_work_dir(self):
        import numpy as _np
        from unittest import mock

        class FakeCapture:
            def __init__(self):
                self.reads = 0
                self.released = False

            def isOpened(self):
                return True

            def get(self, prop):
                if prop == processor.cv2.CAP_PROP_FPS:
                    return 24.0
                if prop == processor.cv2.CAP_PROP_FRAME_WIDTH:
                    return 32
                if prop == processor.cv2.CAP_PROP_FRAME_HEIGHT:
                    return 24
                if prop == processor.cv2.CAP_PROP_FRAME_COUNT:
                    return 4
                return 0

            def set(self, *_args):
                return True

            def read(self):
                self.reads += 1
                if self.reads == 1:
                    return True, _np.zeros((24, 32, 3), dtype=_np.uint8)
                return False, None

            def release(self):
                self.released = True

        class FakeWriter:
            def __init__(self, path, *_args, **_kwargs):
                self.path = path
                self.writes = 0
                self.released = False

            def isOpened(self):
                return True

            def write(self, _frame):
                self.writes += 1

            def release(self):
                self.released = True

        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            source = work / "partial.mp4"
            output = work / "partial_no_sub.mp4"
            source.write_bytes(b"not a real video, validation is mocked")
            remover = self._minimal_remover(work)
            fake_capture = FakeCapture()

            with mock.patch("backend.processor._validate_video_input_file"):
                with mock.patch("backend.processor._open_capture", return_value=fake_capture):
                    with mock.patch(
                        "backend.processor._LosslessIntermediateWriter",
                        FakeWriter,
                    ):
                        ok = remover.process_video(str(source), str(output))

            self.assertFalse(ok)
            self.assertEqual(remover.last_error_reason, "truncated_decode")
            self.assertIn("truncated", remover.last_error_message.lower())
            self.assertFalse(output.exists())
            self.assertFalse((work / "vsr-temp").exists())
            self.assertTrue(fake_capture.released)

    def test_unsupported_codec_is_actionable(self):
        from unittest import mock

        class ClosedCapture:
            def isOpened(self):
                return False

            def release(self):
                return None

        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            source = work / "unsupported.mp4"
            output = work / "unsupported_no_sub.mp4"
            source.write_bytes(b"video bytes")
            remover = self._minimal_remover(work)

            with mock.patch("backend.processor._validate_video_input_file"):
                with mock.patch("backend.processor._open_capture", return_value=ClosedCapture()):
                    with mock.patch(
                        "backend.io._probe_video_stream_status",
                        return_value={
                            "available": True,
                            "ok": True,
                            "hasVideo": True,
                            "codec": "fictional_codec",
                            "width": 1920,
                            "height": 1080,
                            "error": "",
                        },
                    ):
                        ok = remover.process_video(str(source), str(output))

            self.assertFalse(ok)
            self.assertEqual(remover.last_error_reason, "unsupported_codec")
            self.assertIn("fictional_codec", remover.last_error_message)
            self.assertIn("Convert it", remover.last_error_message)

    def test_process_image_records_stage_timings(self):
        class FakeDetector:
            def detect(self, *_args, **_kwargs):
                return [(4, 4, 28, 18)]

        class FakeInpainter:
            def inpaint(self, frames, _masks):
                return frames

        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            source = work / "source.png"
            output = work / "source_clean.png"
            image = np.full((32, 48, 3), 128, dtype=np.uint8)
            self.assertTrue(processor.cv2.imwrite(str(source), image))

            remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
            remover.config = processor.ProcessingConfig()
            remover.detector = FakeDetector()
            remover.inpainter = FakeInpainter()
            remover.on_progress = None
            remover.last_stage_timings = remover._empty_stage_timings()

            ok = remover.process_image(str(source), str(output))
            output_exists = output.exists()

        self.assertTrue(ok)
        self.assertTrue(output_exists)
        for stage in ("decode", "ocr", "mask", "inpaint", "encode"):
            self.assertIn(stage, remover.last_stage_timings)
            self.assertGreaterEqual(remover.last_stage_timings[stage], 0.0)
        self.assertEqual(remover.last_detection_stats["frames_total"], 1)
        self.assertEqual(remover.last_detection_stats["frames_ocr"], 1)
        self.assertEqual(remover.last_detection_stats["frames_skipped"], 0)
        self.assertEqual(
            remover.last_detection_stats["unique_regions_detected"], 1)

    def test_process_image_filters_masks_by_selected_language_script(self):
        class FakeDetector:
            def detect_with_text(self, *_args, **_kwargs):
                return [
                    (4, 4, 16, 16, 0.95, "English"),
                    (28, 4, 44, 16, 0.93, chr(0x5B57) + chr(0x5E55)),
                ]

        class CapturingInpainter:
            def __init__(self):
                self.mask = None

            def inpaint(self, frames, masks):
                self.mask = masks[0].copy()
                return frames

        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            source = work / "mixed.png"
            output = work / "mixed_clean.png"
            image = np.full((24, 52, 3), 128, dtype=np.uint8)
            self.assertTrue(processor.cv2.imwrite(str(source), image))

            remover = processor.SubtitleRemover.__new__(
                processor.SubtitleRemover)
            remover.config = processor.ProcessingConfig(
                detection_lang="ja",
                language_mask_filter=True,
                mask_dilate_px=0,
            )
            remover.detector = FakeDetector()
            inpainter = CapturingInpainter()
            remover.inpainter = inpainter
            remover.on_progress = None
            remover.last_stage_timings = remover._empty_stage_timings()

            ok = remover.process_image(str(source), str(output))

        self.assertTrue(ok)
        self.assertEqual(int(inpainter.mask[10, 10]), 0)
        self.assertEqual(int(inpainter.mask[10, 34]), 255)

    def test_detection_stats_cluster_stable_boxes_and_count_skip_reasons(self):
        remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        remover._reset_detection_stats()

        remover._record_ocr_detection([(10, 10, 110, 40)])
        remover._record_ocr_detection([(12, 11, 112, 41), (200, 20, 260, 50)])
        remover._record_detection_skip("phash")
        remover._record_detection_skip("frame_skip")

        self.assertEqual(remover.last_detection_stats["frames_ocr"], 2)
        self.assertEqual(remover.last_detection_stats["frames_skipped"], 2)
        self.assertEqual(
            remover.last_detection_stats["unique_regions_detected"], 2)
        self.assertEqual(
            remover.last_detection_stats["skip_reasons"],
            {"phash": 1, "frame_skip": 1},
        )

    def test_gui_failed_queue_item_uses_media_input_message(self):
        from unittest import mock

        app = gui.VideoSubtitleRemoverApp.__new__(gui.VideoSubtitleRemoverApp)
        app._update_item_display = mock.Mock()
        app._process_soft_subtitle_item = mock.Mock(return_value=False)
        app._announce_model_download_guidance = mock.Mock()
        app._gui_to_backend_mode = mock.Mock(return_value=processor.InpaintMode.STTN)
        app._gui_to_backend_device = mock.Mock(return_value="cpu")
        app._cached_remover = None
        app._cached_remover_key = None
        app._active_remover = None
        app.cancel_event = threading.Event()
        app._batch_times = []

        class FakeBackendRemover:
            def __init__(self, _config):
                self.last_error_message = (
                    "The selected video appears corrupt or incomplete."
                )
                self.last_quality_report = None
                self.last_output_path = None

            def process_video(self, *_args, **_kwargs):
                return False

        item = gui.QueueItem(
            id="bad-media",
            file_path="bad.mp4",
            output_path="bad_no_sub.mp4",
            config=gui.ProcessingConfig(),
        )
        with mock.patch("backend.processor.SubtitleRemover", FakeBackendRemover):
            app._process_item(item)

        self.assertEqual(item.status, gui.ProcessingStatus.ERROR)
        self.assertEqual(
            item.message,
            "The selected video appears corrupt or incomplete.",
        )
        self.assertEqual(item.error, item.message)


class DecodeHwAccelCoerceTests(unittest.TestCase):
    """decode_hw_accel must clamp to the allowed token set; anything else
    silently disables the hint so we never pass garbage to cv2."""

    def test_default_is_off(self):
        cfg = processor.normalize_processing_config(processor.ProcessingConfig())
        self.assertEqual(cfg.decode_hw_accel, "off")

    def test_known_tokens_kept(self):
        for token in ("off", "auto", "any", "d3d11", "vaapi", "mfx", "pynv", "nvdec"):
            cfg = processor.normalize_processing_config(
                processor.ProcessingConfig(decode_hw_accel=token)
            )
            self.assertEqual(cfg.decode_hw_accel, token)

    def test_unknown_token_becomes_off(self):
        cfg = processor.normalize_processing_config(
            processor.ProcessingConfig(decode_hw_accel="cuda-experimental")
        )
        self.assertEqual(cfg.decode_hw_accel, "off")

    def test_mixed_case_token_normalised(self):
        cfg = processor.normalize_processing_config(
            processor.ProcessingConfig(decode_hw_accel="D3D11")
        )
        self.assertEqual(cfg.decode_hw_accel, "d3d11")


class MultiAudioPassthroughTests(unittest.TestCase):
    def test_default_is_on(self):
        cfg = processor.normalize_processing_config(processor.ProcessingConfig())
        self.assertTrue(cfg.multi_audio_passthrough)

    def test_explicit_off(self):
        cfg = processor.normalize_processing_config(
            processor.ProcessingConfig(multi_audio_passthrough=False)
        )
        self.assertFalse(cfg.multi_audio_passthrough)


class LoudnormCoerceTests(unittest.TestCase):
    """normalize_processing_config must clamp loudnorm_target to valid
    LUFS, with 0.0 reserved as 'disabled'."""

    def test_zero_passes_through_as_disabled(self):
        cfg = processor.normalize_processing_config(
            processor.ProcessingConfig(loudnorm_target=0.0)
        )
        self.assertEqual(cfg.loudnorm_target, 0.0)

    def test_in_range_youtube_target_kept(self):
        cfg = processor.normalize_processing_config(
            processor.ProcessingConfig(loudnorm_target=-14.0)
        )
        self.assertEqual(cfg.loudnorm_target, -14.0)

    def test_in_range_broadcast_target_kept(self):
        cfg = processor.normalize_processing_config(
            processor.ProcessingConfig(loudnorm_target=-23.0)
        )
        self.assertEqual(cfg.loudnorm_target, -23.0)

    def test_out_of_range_silently_disables(self):
        """A value outside ffmpeg's loudnorm range (-70 to -5) is rejected
        as 0.0 (off) rather than crashing the encode."""
        for bad in (5.0, -100.0, -2.0):
            cfg = processor.normalize_processing_config(
                processor.ProcessingConfig(loudnorm_target=bad)
            )
            self.assertEqual(cfg.loudnorm_target, 0.0, f"bad={bad}")

    def test_nan_and_inf_become_zero(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            cfg = processor.normalize_processing_config(
                processor.ProcessingConfig(loudnorm_target=bad)
            )
            self.assertEqual(cfg.loudnorm_target, 0.0, f"bad={bad}")


class FrameSequenceCaptureTests(unittest.TestCase):
    """_FrameSequenceCapture must mirror cv2.VideoCapture closely enough
    that process_video does not notice the swap."""

    def _make_seq_dir(self, n: int, size=(32, 48)):
        """Returns a TemporaryDirectory holding `n` PNG frames numbered
        00.png ... (n-1).png, each filled with the frame index value."""
        import numpy as _np
        import cv2 as _cv2
        tmp = tempfile.mkdtemp(prefix="vsr-seq-")
        h, w = size
        for i in range(n):
            arr = _np.full((h, w, 3), i + 10, dtype=_np.uint8)
            ok = _cv2.imwrite(str(Path(tmp) / f"{i:03d}.png"), arr)
            assert ok, f"could not write {i:03d}.png in {tmp}"
        return tmp

    def test_open_capture_routes_dir_to_frame_sequence_adapter(self):
        tmp = self._make_seq_dir(5)
        try:
            cap = processor._open_capture(tmp, "off", input_fps=12.0)
            self.assertIsInstance(cap, processor._FrameSequenceCapture)
            self.assertTrue(cap.isOpened())
            self.assertEqual(int(cap.get(processor.cv2.CAP_PROP_FRAME_COUNT)), 5)
            self.assertEqual(cap.get(processor.cv2.CAP_PROP_FPS), 12.0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_read_walks_files_in_sorted_order(self):
        tmp = self._make_seq_dir(4)
        try:
            cap = processor._FrameSequenceCapture(tmp, fps=24.0)
            seen = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                seen.append(int(frame.flat[0]))
            self.assertEqual(seen, [10, 11, 12, 13])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_set_pos_frames_supports_seek(self):
        tmp = self._make_seq_dir(6)
        try:
            cap = processor._FrameSequenceCapture(tmp, fps=24.0)
            cap.set(processor.cv2.CAP_PROP_POS_FRAMES, 4)
            ok, frame = cap.read()
            self.assertTrue(ok)
            self.assertEqual(int(frame.flat[0]), 14)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_empty_dir_raises(self):
        tmp = tempfile.mkdtemp(prefix="vsr-empty-")
        try:
            with self.assertRaises(ValueError):
                processor._FrameSequenceCapture(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class PrefetchReaderTests(unittest.TestCase):
    """_PrefetchReader contract:
    - returns the same frames in the same order as the underlying cap
    - read() returns (False, None) after exhaustion or release
    - release() stops the worker even when the queue is full
    """

    class _FakeCap:
        """Minimal cv2.VideoCapture stand-in for unit tests. Returns
        deterministic 'frames' (small numpy arrays) one per read until
        n_frames is reached."""

        def __init__(self, n_frames: int):
            self._n = n_frames
            self._i = 0
            self._released = False
            self._lock = __import__("threading").Lock()

        def isOpened(self):
            return not self._released

        def read(self):
            with self._lock:
                if self._released or self._i >= self._n:
                    return False, None
                import numpy as _np
                frame = _np.full((4, 4, 3), self._i, dtype=_np.uint8)
                self._i += 1
                return True, frame

        def get(self, _prop):
            return 0

        def release(self):
            with self._lock:
                self._released = True

    def test_read_yields_every_frame_in_order(self):
        cap = self._FakeCap(n_frames=20)
        reader = processor._PrefetchReader(cap, max_frames=20, queue_size=4)
        try:
            seen = []
            while True:
                ret, frame = reader.read()
                if not ret:
                    break
                seen.append(int(frame.flat[0]))
            self.assertEqual(seen, list(range(20)))
        finally:
            reader.release()

    def test_release_stops_worker_with_full_queue(self):
        # A worker that has filled the queue must still exit on release().
        cap = self._FakeCap(n_frames=1000)
        reader = processor._PrefetchReader(cap, max_frames=1000, queue_size=4)
        # Don't consume; let the queue fill, then release.
        import time as _time
        _time.sleep(0.05)
        reader.release()
        # Thread must have stopped within the release() join window.
        self.assertFalse(reader._thread.is_alive())

    def test_read_after_exhaustion_is_idempotent(self):
        cap = self._FakeCap(n_frames=3)
        reader = processor._PrefetchReader(cap, max_frames=3, queue_size=2)
        try:
            for _ in range(3):
                ret, _ = reader.read()
                self.assertTrue(ret)
            # After exhaustion, repeated reads keep returning (False, None).
            for _ in range(5):
                ret, frame = reader.read()
                self.assertFalse(ret)
                self.assertIsNone(frame)
        finally:
            reader.release()


class LosslessIntermediateWriterTests(unittest.TestCase):
    """I-1: the intermediate writer must roundtrip frames losslessly when
    ffmpeg is available (FFV1 in .mkv) and degrade gracefully to the
    legacy mp4v writer when it is not."""

    def _have_ffmpeg(self):
        return shutil.which("ffmpeg") is not None

    def test_writer_round_trips_frames_losslessly(self):
        if not self._have_ffmpeg():
            self.skipTest("ffmpeg not on PATH")
        import numpy as _np
        import cv2 as _cv2
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "intermediate.mkv")
            w, h, fps = 32, 24, 12.0
            writer = processor._LosslessIntermediateWriter(path, w, h, fps)
            self.assertTrue(writer.isOpened())
            self.assertTrue(writer.lossless,
                            "FFV1 path should engage when ffmpeg is present")
            frames = []
            for i in range(10):
                # Each frame is uniformly coloured with (i, i*2, i*3) so a
                # lossless round-trip yields bit-identical values back.
                arr = _np.empty((h, w, 3), dtype=_np.uint8)
                arr[:] = (i, (i * 2) % 256, (i * 3) % 256)
                frames.append(arr)
                writer.write(arr)
            writer.release()
            self.assertTrue(Path(path).exists())
            cap = _cv2.VideoCapture(path)
            seen = []
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    seen.append(frame)
            finally:
                cap.release()
            self.assertEqual(len(seen), len(frames))
            # Lossless: per-frame max channel delta is 0 for FFV1 + bgr24.
            for i, (src, decoded) in enumerate(zip(frames, seen)):
                delta = int(_np.abs(src.astype(_np.int16) - decoded.astype(_np.int16)).max())
                self.assertEqual(delta, 0,
                                 f"frame {i} expected lossless roundtrip, got delta={delta}")

    def test_writer_can_emit_bgr48le_intermediate(self):
        if not self._have_ffmpeg() or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg/ffprobe not on PATH")
        import numpy as _np
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "intermediate_16.mkv")
            w, h, fps = 8, 6, 6.0
            writer = processor._LosslessIntermediateWriter(
                path,
                w,
                h,
                fps,
                pixel_format="bgr48le",
            )
            self.assertTrue(writer.isOpened())
            self.assertEqual(writer.pixel_format, "bgr48le")
            frame = _np.full((h, w, 3), 42000, dtype=_np.uint16)
            writer.write(frame)
            writer.release()

            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=pix_fmt",
                    "-of", "default=nw=1:nk=1",
                    path,
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(result.stdout.strip(), {"bgr48le", "gbrp16le"})

    def test_writer_fallback_when_ffmpeg_path_is_blank(self):
        # Simulate a missing ffmpeg by patching shutil.which inside the
        # processor module. The writer must open the cv2 fallback and stay
        # functional rather than raising.
        import shutil as _shutil
        original_which = _shutil.which
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "intermediate.mkv")
            try:
                _shutil.which = lambda name: None
                writer = processor._LosslessIntermediateWriter(path, 16, 12, 24.0)
                # Fallback path renames .mkv to .mp4 because mp4v in .mkv
                # is rarely playable on consumer builds.
                self.assertFalse(writer.lossless)
                self.assertTrue(writer.path.endswith(".mp4"))
                writer.release()
            finally:
                _shutil.which = original_which


class MultiTrackLoudnormFilterTests(unittest.TestCase):
    """B-4: when both loudnorm and multi-track passthrough are active and
    the source has multiple audio streams, _merge_audio must build a
    -filter_complex pipeline instead of relying on the single-pass
    `-af loudnorm`. We exercise the audio-stream probe helper here
    (the full _merge_audio orchestration needs real ffmpeg + a video)."""

    def test_audio_stream_count_falls_back_to_one_when_ffprobe_missing(self):
        # The helper must not crash when ffprobe is absent. Returning 1
        # means _merge_audio takes the legacy single-stream path.
        import shutil as _shutil
        original = _shutil.which
        try:
            _shutil.which = lambda name: None
            count = processor._probe_audio_stream_count("/non-existent.mkv")
            # ffprobe absent -> falls back to 1.
            self.assertEqual(count, 1)
        finally:
            _shutil.which = original


class SubtitleStreamProbeTests(unittest.TestCase):
    """#103 first pass: probe embedded subtitle tracks without loading OCR."""

    def test_probe_subtitle_streams_parses_ffprobe_json(self):
        from unittest import mock

        payload = {
            "streams": [
                {
                    "index": 2,
                    "codec_name": "subrip",
                    "tags": {"language": "eng", "title": "SDH"},
                    "disposition": {"default": 1, "forced": 0},
                },
                {
                    "index": 4,
                    "codec_name": "ass",
                    "tags": {"language": "jpn"},
                    "disposition": {"default": "0", "forced": "1"},
                },
            ]
        }
        completed = SimpleNamespace(
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )
        with mock.patch("backend.io.run_process", return_value=completed) as run:
            streams = processor._probe_subtitle_streams("movie.mkv")

        cmd = run.call_args.args[0]
        self.assertIn("-select_streams", cmd)
        self.assertIn("s", cmd)
        self.assertIn("-of", cmd)
        self.assertIn("json", cmd)
        self.assertEqual(len(streams), 2)
        self.assertEqual(streams[0].index, 2)
        self.assertEqual(streams[0].codec_name, "subrip")
        self.assertEqual(streams[0].language, "eng")
        self.assertEqual(streams[0].title, "SDH")
        self.assertTrue(streams[0].default)
        self.assertFalse(streams[0].forced)
        self.assertEqual(streams[1].index, 4)
        self.assertEqual(streams[1].codec_name, "ass")
        self.assertTrue(streams[1].forced)

    def test_probe_subtitle_streams_falls_back_to_empty_list(self):
        from unittest import mock

        with mock.patch(
            "backend.io.run_process",
            side_effect=FileNotFoundError,
        ):
            self.assertEqual(processor._probe_subtitle_streams("missing.mkv"), [])

        completed = SimpleNamespace(returncode=0, stdout="{bad", stderr="")
        with mock.patch("backend.io.run_process", return_value=completed):
            self.assertEqual(processor._probe_subtitle_streams("bad.mkv"), [])


class SoftSubtitleRemuxTests(unittest.TestCase):
    """#103 remux primitive: explicit stream-copy mapping only."""

    def test_build_strip_cmd_removes_subtitle_streams(self):
        from backend.remux import SoftSubtitleAction, build_soft_subtitle_remux_cmd

        cmd = build_soft_subtitle_remux_cmd(
            "input.mkv",
            "output.mkv",
            action=SoftSubtitleAction.STRIP,
        )
        self.assertIn("-map", cmd)
        self.assertIn("0", cmd)
        self.assertIn("-0:s?", cmd)
        self.assertIn("-c", cmd)
        self.assertIn("copy", cmd)
        self.assertLess(cmd.index("-0:s?"), cmd.index("-c"))

    def test_build_keep_selected_cmd_maps_selected_global_streams(self):
        from backend.remux import SoftSubtitleAction, build_soft_subtitle_remux_cmd

        cmd = build_soft_subtitle_remux_cmd(
            "input.mkv",
            "output.mkv",
            action=SoftSubtitleAction.KEEP_SELECTED,
            keep_stream_indices=[4, 2, 4],
        )
        self.assertIn("-0:s?", cmd)
        maps = [
            cmd[i + 1] for i, token in enumerate(cmd[:-1])
            if token == "-map"
        ]
        self.assertEqual(maps, ["0", "-0:s?", "0:2", "0:4"])

    def test_keep_selected_requires_stream_index(self):
        from backend.remux import SoftSubtitleAction, build_soft_subtitle_remux_cmd

        with self.assertRaises(ValueError):
            build_soft_subtitle_remux_cmd(
                "input.mkv",
                "output.mkv",
                action=SoftSubtitleAction.KEEP_SELECTED,
            )

    def test_remux_uses_atomic_temp_output(self):
        from unittest import mock
        from backend import remux as _remux

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir) / ".out.tmp.mkv"
            final_path = Path(tmpdir) / "out.mkv"
            with mock.patch.object(_remux.shutil, "which", return_value="ffmpeg"):
                with mock.patch.object(
                    _remux, "_allocate_temp_output_path", return_value=temp_path,
                ) as allocate:
                    with mock.patch.object(_remux, "_promote_temp_output") as promote:
                        with mock.patch.object(_remux, "_cleanup_temp_output") as clean:
                            with mock.patch.object(
                                _remux, "_probe_duration_seconds", return_value=2.0,
                            ):
                                with mock.patch.object(
                                    _remux,
                                    "_run_subprocess_checked",
                                ) as run:
                                    _remux.remux_soft_subtitles(
                                        "input.mkv",
                                        str(final_path),
                                    )

        allocate.assert_called_once_with(str(final_path))
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[-1], str(temp_path))
        self.assertIn("-0:s?", cmd)
        self.assertIn("timeout", run.call_args.kwargs)
        promote.assert_called_once_with(temp_path, final_path)
        clean.assert_called_once_with(temp_path)

    def test_remux_rejects_same_input_and_output_path(self):
        from unittest import mock
        from backend import remux as _remux

        with tempfile.TemporaryDirectory() as tmpdir:
            media = Path(tmpdir) / "movie.mkv"
            media.write_bytes(b"not a real video")
            with mock.patch.object(_remux, "_allocate_temp_output_path") as allocate:
                with self.assertRaises(ValueError):
                    _remux.remux_soft_subtitles(str(media), str(media))
        allocate.assert_not_called()

    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe"),
        "ffmpeg/ffprobe unavailable",
    )
    def test_strip_remux_integration_removes_subtitle_streams(self):
        from backend.io import _probe_codec_for_log, _probe_subtitle_streams
        from backend.remux import remux_soft_subtitles

        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            srt = work / "captions.srt"
            source = work / "source.mkv"
            output = work / "stripped.mkv"
            srt.write_text(
                "1\n00:00:00,000 --> 00:00:00,200\nHello\n",
                encoding="utf-8",
            )
            subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "color=c=black:s=32x24:d=0.25:r=1",
                    "-f", "srt", "-i", str(srt),
                    "-map", "0:v", "-map", "1:s",
                    "-c:v", "ffv1", "-c:s", "srt", str(source),
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(len(_probe_subtitle_streams(str(source))), 1)
            source_codec = _probe_codec_for_log(str(source))

            remux_soft_subtitles(str(source), str(output))

            self.assertEqual(_probe_subtitle_streams(str(output)), [])
            self.assertEqual(_probe_codec_for_log(str(output)), source_codec)


class HdrPipelineTests(unittest.TestCase):
    """RM-73 partial: probe_color_metadata returns None without ffprobe;
    hdr_encode_args produces empty list for None/empty metadata and the
    expected -color_primaries / -color_trc / -colorspace flags otherwise."""

    def test_hdr_encode_args_empty_for_none(self):
        from backend.hdr import hdr_encode_args
        self.assertEqual(hdr_encode_args(None), [])

    def test_hdr_encode_args_emits_tags(self):
        from backend.hdr import hdr_encode_args, ColorMetadata
        meta = ColorMetadata(
            color_primaries="bt2020",
            color_transfer="smpte2084",
            color_space="bt2020nc",
            color_range="tv",
        )
        args = hdr_encode_args(meta)
        self.assertIn("-color_primaries", args)
        self.assertIn("bt2020", args)
        self.assertIn("-color_trc", args)
        self.assertIn("smpte2084", args)
        self.assertIn("-colorspace", args)
        self.assertIn("-color_range", args)
        self.assertTrue(meta.is_hdr)

    def test_hdr_encode_args_skip_rgb_matrix_tags(self):
        from backend.hdr import hdr_encode_args, ColorMetadata
        meta = ColorMetadata(color_space="gbr", color_range="pc")
        args = hdr_encode_args(meta)
        self.assertNotIn("-colorspace", args)
        self.assertIn("-color_range", args)

    def test_hdr_safe_codec_promotes_h264_to_h265(self):
        from backend.hdr import ColorMetadata, hdr_safe_codec
        meta = ColorMetadata(
            color_primaries="bt2020",
            color_transfer="smpte2084",
            color_space="bt2020nc",
        )
        self.assertEqual(hdr_safe_codec("h264", meta), "h265")
        self.assertEqual(hdr_safe_codec("av1", meta), "av1")

    def test_hdr_pixel_format_args_require_10bit_surface(self):
        from backend.hdr import ColorMetadata, hdr_pixel_format_args
        meta = ColorMetadata(
            color_primaries="bt2020",
            color_transfer="smpte2084",
            color_space="bt2020nc",
        )
        self.assertEqual(
            hdr_pixel_format_args(meta, "h265"),
            ["-pix_fmt", "yuv420p10le"],
        )
        self.assertEqual(
            hdr_pixel_format_args(meta, "h265", hardware=True),
            ["-pix_fmt", "p010le"],
        )
        self.assertEqual(hdr_pixel_format_args(meta, "h264"), [])

    def test_hdr_encoder_private_args_emit_x265_color_params(self):
        from backend.hdr import ColorMetadata, hdr_encoder_private_args
        meta = ColorMetadata(
            color_primaries="bt2020",
            color_transfer="smpte2084",
            color_space="bt2020nc",
        )
        args = hdr_encoder_private_args(meta, "h265")
        self.assertEqual(args[0], "-x265-params")
        self.assertIn("colorprim=9", args[1])
        self.assertIn("transfer=16", args[1])
        self.assertIn("colormatrix=9", args[1])
        self.assertEqual(hdr_encoder_private_args(meta, "av1"), [])

    def test_static_and_dynamic_hdr_metadata_policy(self):
        from backend import hdr

        payload = {
            "streams": [{
                "color_primaries": "bt2020",
                "color_transfer": "smpte2084",
                "color_space": "bt2020nc",
                "color_range": "tv",
                "codec_tag_string": "dvh1",
                "side_data_list": [{
                    "side_data_type": "Mastering display metadata",
                    "red_x": "34000/50000",
                    "red_y": "16000/50000",
                    "green_x": "13250/50000",
                    "green_y": "34500/50000",
                    "blue_x": "7500/50000",
                    "blue_y": "3000/50000",
                    "white_point_x": "15635/50000",
                    "white_point_y": "16450/50000",
                    "max_luminance": "10000000/10000",
                    "min_luminance": "1/10000",
                }, {
                    "side_data_type": "Content light level metadata",
                    "max_content": 1000,
                    "max_average": 400,
                }, {
                    "side_data_type": "HDR Dynamic Metadata SMPTE2094-40",
                }],
            }],
        }
        completed = SimpleNamespace(
            returncode=0, stdout=json.dumps(payload), stderr="")
        with unittest.mock.patch(
            "backend.hdr.shutil.which", return_value="ffprobe"
        ), unittest.mock.patch(
            "backend.hdr.run_process", return_value=completed
        ):
            meta = hdr.probe_color_metadata("hdr.mkv")

        self.assertEqual(
            meta.mastering_display,
            "G(13250,34500)B(7500,3000)R(34000,16000)"
            "WP(15635,16450)L(10000000,1)",
        )
        self.assertEqual((meta.max_cll, meta.max_fall), (1000, 400))
        self.assertIn("HDR Dynamic Metadata SMPTE2094-40", meta.dynamic_metadata)
        self.assertIn("Dolby Vision Configuration Record", meta.dynamic_metadata)
        self.assertIn("master-display=", hdr.hdr_encoder_private_args(meta, "h265")[1])
        self.assertIn("mastering-display=", hdr.hdr_encoder_private_args(meta, "av1")[1])
        self.assertIn(
            "MasteringDisplayColourVolume=",
            hdr.hdr_encoder_private_args(meta, "vvc")[1],
        )

        from backend.output_contract import build_output_contract
        with unittest.mock.patch(
            "backend.output_contract._probe_source_audio", return_value=True
        ):
            contract = build_output_contract(
                input_path="hdr.mkv",
                output_path="cleaned.mkv",
                codec="h265",
                preserve_audio=True,
                preserve_color_metadata=True,
                color_metadata=meta,
                hardware_requested=True,
            )
        report = contract.report()
        self.assertEqual(report["container"], "mkv")
        self.assertIsNone(report["color_preserved"])
        self.assertEqual(report["color"]["max_cll"], 1000)
        self.assertEqual(len(report["warnings"]), 2)
        self.assertTrue(all("Dropped stale" in item for item in report["warnings"]))
        self.assertTrue(contract.color_preserved([]))
        self.assertTrue(contract.color_preserved(["source audio is missing"]))
        self.assertFalse(contract.color_preserved([
            "color transfer is not preserved",
        ]))

    def test_deinterlace_uses_lossless_contract_intermediate(self):
        from backend.io import _deinterlace_to_temp
        from backend.output_contract import OutputContract

        contract = OutputContract(
            output_suffix=".mkv",
            codec="h265",
            preserve_audio=False,
            source_has_audio=True,
            preserve_color_metadata=True,
            color_metadata=__import__(
                "backend.hdr", fromlist=["ColorMetadata"]
            ).ColorMetadata(
                color_primaries="bt2020",
                color_transfer="smpte2084",
                color_space="bt2020nc",
                color_range="tv",
            ),
            hardware_requested=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir, unittest.mock.patch(
            "backend.io._run_subprocess_checked"
        ) as run, unittest.mock.patch(
            "backend.io._probe_duration_seconds", return_value=1.0
        ):
            result = _deinterlace_to_temp(
                "source.mkv", tmpdir, output_contract=contract)
        command = run.call_args.args[0]
        self.assertTrue(result.endswith("deinterlaced.mkv"))
        self.assertIn("ffv1", command)
        self.assertIn("yuv420p10le", command)
        self.assertIn("-an", command)
        self.assertNotIn("libx264", command)

    def test_d3d12_deinterlace_falls_back_to_yadif(self):
        from backend.io import _deinterlace_to_temp

        failure = subprocess.CalledProcessError(1, ["ffmpeg"])
        with tempfile.TemporaryDirectory() as tmpdir, unittest.mock.patch(
            "backend.io._run_subprocess_checked",
            side_effect=[failure, None],
        ) as run, unittest.mock.patch(
            "backend.io._probe_duration_seconds", return_value=1.0
        ):
            result = _deinterlace_to_temp(
                "source.mkv", tmpdir, prefer_d3d12=True)

        self.assertTrue(result.endswith("deinterlaced.mkv"))
        first = run.call_args_list[0].args[0]
        second = run.call_args_list[1].args[0]
        self.assertIn("deinterlace_d3d12=mode=field", first[first.index("-vf") + 1])
        self.assertEqual(second[second.index("-vf") + 1], "yadif=1")

    def test_processing_frame_downconverts_uint16_to_uint8(self):
        frame = np.array([[[0, 257, 65535]]], dtype=np.uint16)
        out = processor.SubtitleRemover._processing_frame(frame)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.tolist(), [[[0, 1, 255]]])

    def test_high_bit_merge_preserves_unmasked_source(self):
        remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        remover.config = processor.ProcessingConfig(mask_feather_px=0)
        source = np.full((2, 2, 3), 40000, dtype=np.uint16)
        cleaned = np.full((2, 2, 3), 10, dtype=np.uint8)
        mask = np.zeros((2, 2), dtype=np.uint8)
        mask[0, 1] = 255

        out = remover._merge_high_bit_output(source, cleaned, mask)

        self.assertEqual(out.dtype, np.uint16)
        self.assertEqual(int(out[0, 0, 0]), 40000)
        self.assertEqual(int(out[0, 1, 0]), 2570)

    def test_probe_color_metadata_falls_back(self):
        from backend.hdr import probe_color_metadata
        # ffprobe absent or path missing -- helper returns None.
        result = probe_color_metadata("/nonexistent.mp4")
        # Cannot guarantee ffprobe is missing in CI; accept None or a
        # ColorMetadata so the test is environment-tolerant.
        self.assertTrue(result is None or hasattr(result, "label"))


class PostRestoreTests(unittest.TestCase):
    """RM-78 / RM-80: optional post-restore adapters must return None
    when their dependency is missing. The pipeline never crashes on a
    half-broken install."""

    def test_realesrgan_skip_when_binary_missing(self):
        from backend import post_restore as _pr
        import shutil as _shutil
        original = _shutil.which
        try:
            _shutil.which = lambda name: None
            with tempfile.TemporaryDirectory() as tmpdir:
                src = Path(tmpdir) / "in.mp4"
                src.write_bytes(b"\x00" * 16)  # placeholder
                dst = str(Path(tmpdir) / "out.mp4")
                result = _pr.realesrgan_upscale(str(src), dst, scale=2)
            self.assertIsNone(result)
        finally:
            _shutil.which = original

    def test_film_grain_skip_when_ffmpeg_missing(self):
        from backend import post_restore as _pr
        import shutil as _shutil
        original = _shutil.which
        try:
            _shutil.which = lambda name: None
            with tempfile.TemporaryDirectory() as tmpdir:
                src = Path(tmpdir) / "in.mp4"
                src.write_bytes(b"\x00" * 16)
                dst = str(Path(tmpdir) / "out.mp4")
                result = _pr.add_film_grain(str(src), dst, strength=0.04)
            self.assertIsNone(result)
        finally:
            _shutil.which = original

    def test_av1_native_grain_skips_additive_post_pass(self):
        remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        remover.config = processor.ProcessingConfig(
            output_codec="av1",
            film_grain_strength=0.04,
            use_hw_encode=False,
        )
        remover._hw_encoder = None
        remover._color_metadata = None
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "out.mp4"
            output.write_bytes(b"placeholder")
            with unittest.mock.patch(
                "backend.post_restore.add_film_grain"
            ) as add_grain:
                remover._run_post_restore_passes(str(output), tmpdir)
            add_grain.assert_not_called()

    def test_film_grain_uses_selected_encode_and_audio_contract(self):
        from backend import post_restore

        completed = SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        with unittest.mock.patch(
            "backend.post_restore.shutil.which", return_value="ffmpeg"
        ), unittest.mock.patch(
            "backend.post_restore.run_process", return_value=completed
        ) as run:
            produced = post_restore.add_film_grain(
                "source.mkv",
                "grain.mkv",
                video_encode_args=(
                    "-c:v", "libx265", "-pix_fmt", "yuv420p10le"
                ),
                preserve_audio=False,
            )
        self.assertEqual(produced, "grain.mkv")
        command = run.call_args.args[0]
        self.assertIn("libx265", command)
        self.assertIn("yuv420p10le", command)
        self.assertIn("-an", command)
        self.assertNotIn("libx264", command)


class OutputCodecTests(unittest.TestCase):
    """F-8: output_codec must coerce to h264 / h265 / av1 / vvc and
    drive the right software encoder when no HW encoder is available."""

    def test_default_is_h264(self):
        cfg = processor.normalize_processing_config(processor.ProcessingConfig())
        self.assertEqual(cfg.output_codec, "h264")

    def test_hevc_alias_normalises_to_h265(self):
        cfg = processor.normalize_processing_config(
            processor.ProcessingConfig(output_codec="hevc")
        )
        self.assertEqual(cfg.output_codec, "h265")

    def test_unknown_codec_resets_to_h264(self):
        cfg = processor.normalize_processing_config(
            processor.ProcessingConfig(output_codec="vp9")
        )
        self.assertEqual(cfg.output_codec, "h264")

    def test_vvc_codec_normalises(self):
        cfg = processor.normalize_processing_config(
            processor.ProcessingConfig(output_codec="vvc")
        )
        self.assertEqual(cfg.output_codec, "vvc")

    def test_h266_alias_normalises_to_vvc(self):
        cfg = processor.normalize_processing_config(
            processor.ProcessingConfig(output_codec="h266")
        )
        self.assertEqual(cfg.output_codec, "vvc")

    def test_software_encoder_args_match_codec(self):
        remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        remover._hw_encoder = None
        remover._color_metadata = None  # RM-73 init slot for _hdr_encode_args
        remover.config = processor.ProcessingConfig(output_codec="h265",
                                                     output_quality=22,
                                                     use_hw_encode=False)
        args = remover._get_encode_args()
        self.assertIn("libx265", args)
        remover.config.output_codec = "av1"
        args = remover._get_encode_args()
        self.assertIn("libsvtav1", args)
        remover.config.output_codec = "vvc"
        args = remover._get_encode_args()
        self.assertIn("libvvenc", args)

    def test_hdr_h264_output_promotes_to_software_hevc_10bit(self):
        from backend.hdr import ColorMetadata
        remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        remover._hw_encoder = "h264_nvenc"
        remover._color_metadata = ColorMetadata(
            color_primaries="bt2020",
            color_transfer="smpte2084",
            color_space="bt2020nc",
            color_range="tv",
        )
        remover._hdr_codec_warning_logged = False
        remover._hdr_software_warning_logged = False
        remover.config = processor.ProcessingConfig(
            output_codec="h264",
            output_quality=22,
            use_hw_encode=True,
        )

        args = remover._get_encode_args()

        self.assertIn("libx265", args)
        self.assertNotIn("h264_nvenc", args)
        self.assertIn("-pix_fmt", args)
        self.assertIn("yuv420p10le", args)
        self.assertIn("-x265-params", args)
        self.assertIn("-color_trc", args)
        self.assertIn("smpte2084", args)

    def test_hdr_preserves_explicit_av1_10bit_output(self):
        from backend.hdr import ColorMetadata
        remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        remover._hw_encoder = None
        remover._color_metadata = ColorMetadata(
            color_primaries="bt2020",
            color_transfer="arib-std-b67",
            color_space="bt2020nc",
        )
        remover._hdr_codec_warning_logged = False
        remover._hdr_software_warning_logged = False
        remover.config = processor.ProcessingConfig(
            output_codec="av1",
            output_quality=28,
            use_hw_encode=False,
        )

        args = remover._get_encode_args()

        self.assertIn("libsvtav1", args)
        self.assertIn("-pix_fmt", args)
        self.assertIn("yuv420p10le", args)

    def test_hdr_hevc_keeps_compatible_hardware_encoder(self):
        from backend.hdr import ColorMetadata
        remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        remover._hw_encoder = "hevc_nvenc"
        remover._output_contract = None
        remover._color_metadata = ColorMetadata(
            color_primaries="bt2020",
            color_transfer="smpte2084",
            color_space="bt2020nc",
            color_range="tv",
        )
        remover.config = processor.ProcessingConfig(
            output_codec="h265", use_hw_encode=True)

        args = remover._get_encode_args()

        self.assertIn("hevc_nvenc", args)
        self.assertIn("p010le", args)
        self.assertNotIn("libx265", args)

    def test_static_hdr_uses_metadata_capable_software_encoder(self):
        from backend.hdr import ColorMetadata
        remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        remover._hw_encoder = "hevc_nvenc"
        remover._output_contract = None
        remover._hdr_software_warning_logged = False
        remover._color_metadata = ColorMetadata(
            color_primaries="bt2020",
            color_transfer="smpte2084",
            color_space="bt2020nc",
            mastering_display=(
                "G(13250,34500)B(7500,3000)R(34000,16000)"
                "WP(15635,16450)L(10000000,1)"
            ),
        )
        remover.config = processor.ProcessingConfig(
            output_codec="h265", use_hw_encode=True)

        args = remover._get_encode_args()

        self.assertIn("libx265", args)
        self.assertIn("master-display=", args[args.index("-x265-params") + 1])
        self.assertNotIn("hevc_nvenc", args)

    def test_av1_film_grain_uses_svtav1_native_param(self):
        remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        remover._hw_encoder = None
        remover._color_metadata = None
        remover.config = processor.ProcessingConfig(
            output_codec="av1",
            output_quality=30,
            film_grain_strength=0.04,
            use_hw_encode=False,
        )

        args = remover._get_encode_args()

        self.assertIn("libsvtav1", args)
        self.assertIn("-svtav1-params", args)
        self.assertIn("film-grain=10", args)


class MediaExtensionParityTests(unittest.TestCase):
    """Verify that GUI and backend media extension sets stay in sync."""

    def test_gui_image_extensions_include_tif(self):
        from gui.utils import IMAGE_EXTENSIONS
        self.assertIn(".tif", IMAGE_EXTENSIONS)
        self.assertIn(".tiff", IMAGE_EXTENSIONS)

    def test_backend_frame_capture_matches_gui_image_extensions(self):
        from gui.utils import IMAGE_EXTENSIONS
        from backend.io import _FrameSequenceCapture
        self.assertEqual(
            _FrameSequenceCapture.SUPPORTED_EXTS,
            set(IMAGE_EXTENSIONS),
        )

    def test_filepicker_pattern_covers_all_extensions(self):
        from gui.utils import (
            SUPPORTED_EXTENSIONS, VIDEO_EXTENSIONS, IMAGE_EXTENSIONS,
            filepicker_pattern,
        )
        pattern = filepicker_pattern(SUPPORTED_EXTENSIONS)
        for ext in VIDEO_EXTENSIONS | IMAGE_EXTENSIONS:
            self.assertIn(f"*{ext}", pattern)


class TempCleanupOnExceptionTests(unittest.TestCase):
    def test_temp_output_removed_after_simulated_error(self):
        from backend.io import (
            _allocate_temp_output_path,
            _cleanup_temp_output,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "output.mp4")
            temp = _allocate_temp_output_path(output)
            self.assertTrue(os.path.exists(temp))
            _cleanup_temp_output(temp)
            self.assertFalse(os.path.exists(temp))


class OutputSidecarTests(unittest.TestCase):
    def test_build_output_sidecar_schema_and_fields(self):
        from backend.batch_report import build_output_sidecar, SIDECAR_SCHEMA
        from backend.config import ProcessingConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "input.mp4")
            out = os.path.join(tmpdir, "output.mp4")
            Path(inp).write_bytes(b"fake video content")
            Path(out).write_bytes(b"fake output content")

            config = ProcessingConfig(detection_lang="en", output_quality=20)
            sidecar = build_output_sidecar(
                input_path=inp,
                output_path=out,
                config=config,
                status="processed",
                elapsed_seconds=12.345,
                stage_timings={"decode": 1.0, "ocr": 2.5, "inpaint": 5.0, "encode": 3.0},
                detection_stats={
                    "frames_total": 10,
                    "frames_ocr": 4,
                    "frames_skipped": 6,
                    "unique_regions_detected": 2,
                },
                quality_report={"psnr": 25.0, "ssim": 0.95, "samples": 10, "tag": "Good"},
                quality_gate={"status": "passed"},
                checkpoint_resumed=False,
                app_version="3.17.3",
            )

        self.assertEqual(sidecar["schema"], SIDECAR_SCHEMA)
        self.assertEqual(sidecar["appVersion"], "3.17.3")
        self.assertEqual(sidecar["status"], "processed")
        self.assertFalse(sidecar["checkpointResumed"])
        self.assertAlmostEqual(sidecar["elapsedSeconds"], 12.345, places=2)
        self.assertEqual(sidecar["source"]["name"], "input.mp4")
        self.assertGreater(sidecar["source"]["bytes"], 0)
        self.assertEqual(len(sidecar["source"]["sha256"]), 64)
        self.assertEqual(sidecar["output"]["name"], "output.mp4")
        self.assertEqual(sidecar["config"]["detection_lang"], "en")
        self.assertEqual(sidecar["config"]["output_quality"], 20)
        self.assertIn("engine", sidecar)
        self.assertEqual(sidecar["stageTimings"]["ocr"], 2.5)
        self.assertEqual(sidecar["detectionStats"]["frames_skipped"], 6)
        self.assertEqual(sidecar["qualityReport"]["psnr"], 25.0)
        self.assertEqual(sidecar["qualityGate"]["status"], "passed")

    def test_sidecar_snapshot_contains_every_canonical_config_field(self):
        from backend.batch_report import build_output_sidecar
        from backend.config import ProcessingConfig
        from backend.config_schema import CONFIG_SCHEMA_VERSION, processing_field_names

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "input.mp4")
            out = os.path.join(tmpdir, "output.mp4")
            Path(inp).write_bytes(b"source")
            Path(out).write_bytes(b"output")
            sidecar = build_output_sidecar(
                input_path=inp,
                output_path=out,
                config=ProcessingConfig(),
                status="processed",
            )

        self.assertEqual(sidecar["configSchemaVersion"], CONFIG_SCHEMA_VERSION)
        self.assertEqual(set(sidecar["config"]), set(processing_field_names()))

    def test_source_hash_has_no_legacy_large_file_cutoff(self):
        import hashlib
        from backend.batch_report import _sha256_file

        class VirtualLargePath:
            def stat(self):
                return SimpleNamespace(st_size=513 * 1024 * 1024)

            def open(self, mode):
                self.assert_mode = mode
                return io.BytesIO(b"streamed payload")

        virtual_path = VirtualLargePath()
        digest = _sha256_file(virtual_path)  # type: ignore[arg-type]

        self.assertEqual(virtual_path.assert_mode, "rb")
        self.assertEqual(digest, hashlib.sha256(b"streamed payload").hexdigest())

    def test_write_output_sidecar_creates_file(self):
        from backend.batch_report import write_output_sidecar, SIDECAR_SCHEMA
        from backend.config import ProcessingConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "input.mp4")
            out = os.path.join(tmpdir, "output.mp4")
            Path(inp).write_bytes(b"source")
            Path(out).write_bytes(b"output")

            config = ProcessingConfig()
            result = write_output_sidecar(
                input_path=inp,
                output_path=out,
                config=config,
                status="processed",
                app_version="3.17.3",
            )

            self.assertIsNotNone(result)
            self.assertTrue(result.exists())
            self.assertEqual(result.name, "output.mp4.vsr.json")
            payload = json.loads(result.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], SIDECAR_SCHEMA)

    def test_sidecar_omits_optional_fields_when_not_provided(self):
        from backend.batch_report import build_output_sidecar
        from backend.config import ProcessingConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "input.mp4")
            out = os.path.join(tmpdir, "output.mp4")
            Path(inp).write_bytes(b"source")
            Path(out).write_bytes(b"output")

            sidecar = build_output_sidecar(
                input_path=inp,
                output_path=out,
                config=ProcessingConfig(),
                status="skipped-existing",
            )

        self.assertNotIn("elapsedSeconds", sidecar)
        self.assertNotIn("stageTimings", sidecar)
        self.assertNotIn("qualityReport", sidecar)
        self.assertNotIn("qualityGate", sidecar)


class WorkDirectoryPolicyTests(unittest.TestCase):
    def test_selected_work_root_is_created_probed_and_used_for_temp(self):
        from backend.work_directory import make_work_temp_dir, resolve_work_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            requested = Path(tmpdir) / "scratch" / "nested"
            resolution = resolve_work_directory(requested)
            work_temp = make_work_temp_dir(resolution, prefix="policy-")
            try:
                self.assertEqual(resolution.path, requested.resolve())
                self.assertFalse(resolution.used_fallback)
                self.assertEqual(work_temp.parent, requested.resolve())
            finally:
                shutil.rmtree(work_temp, ignore_errors=True)

    def test_unavailable_work_root_falls_back_with_actionable_warning(self):
        from backend.work_directory import resolve_work_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            unavailable = root / "not-a-directory"
            unavailable.write_text("file", encoding="utf-8")
            fallback = root / "fallback"
            resolution = resolve_work_directory(unavailable, fallback=fallback)

            self.assertTrue(resolution.used_fallback)
            self.assertEqual(resolution.path, fallback.resolve())
            self.assertIn("unavailable or read-only", resolution.warning)
            self.assertIn("Choose a writable work folder", resolution.warning)

    def test_default_checkpoint_directory_follows_selected_work_root(self):
        from backend.cli import _default_checkpoint_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir) / "work"
            checkpoint = _default_checkpoint_dir(str(work))

            self.assertEqual(checkpoint, work.resolve() / "checkpoints")
            self.assertTrue(checkpoint.is_dir())

    def test_processor_staging_output_uses_selected_work_root(self):
        from backend.config import ProcessingConfig
        from backend.processor import SubtitleRemover

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            work = root / "work"
            remover = SubtitleRemover.__new__(SubtitleRemover)
            remover.config = ProcessingConfig(work_directory=str(work))
            remover._work_directory_resolution = None
            remover.last_work_directory_warning = None
            staged = remover._allocate_work_output(str(root / "out" / "movie.mp4"))
            try:
                self.assertEqual(staged.parent, work.resolve())
                self.assertEqual(staged.suffix, ".mp4")
            finally:
                staged.unlink(missing_ok=True)

    def test_cross_volume_promotion_stages_atomically_on_destination(self):
        import errno
        from backend import io as backend_io

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "work" / "render.mp4"
            destination = root / "output" / "final.mp4"
            source.parent.mkdir()
            source.write_bytes(b"complete-output")
            real_replace = os.replace

            def replace_with_cross_volume_once(src, dst):
                if Path(src) == source:
                    raise OSError(errno.EXDEV, "cross-volume test")
                return real_replace(src, dst)

            with unittest.mock.patch.object(
                backend_io.os, "replace", side_effect=replace_with_cross_volume_once,
            ):
                backend_io._promote_temp_output(source, destination)

            self.assertEqual(destination.read_bytes(), b"complete-output")
            self.assertFalse(source.exists())


class DiskSpaceAndLogRotationTests(unittest.TestCase):
    """P2: pre-encode free-space preflight and bounded JSON log."""

    def _remover(self):
        from backend.processor import SubtitleRemover
        return SubtitleRemover.__new__(SubtitleRemover)

    def test_aborts_when_free_space_critically_low(self):
        import collections
        remover = self._remover()
        du = collections.namedtuple("du", "total used free")(0, 0, 1024 * 1024)
        with unittest.mock.patch("backend.work_directory.shutil.disk_usage",
                                 return_value=du):
            with self.assertRaises(ValueError) as ctx:
                remover._check_encode_disk_space(
                    "out.mp4", width=1920, height=1080,
                    frames=100000, high_bit=False)
        self.assertIn("disk space", str(ctx.exception).lower())

    def test_passes_when_space_is_ample(self):
        import collections
        remover = self._remover()
        du = collections.namedtuple("du", "total used free")(
            0, 0, 500 * 1024 ** 3)  # 500 GB free
        with unittest.mock.patch("backend.work_directory.shutil.disk_usage",
                                 return_value=du):
            remover._check_encode_disk_space(
                "out.mp4", width=640, height=480, frames=300, high_bit=False)

    def test_preflight_includes_work_output_and_checkpoint_requirements(self):
        from backend.config import ProcessingConfig
        from backend.work_directory import StorageVolumeStatus

        remover = self._remover()
        remover.config = ProcessingConfig(work_directory="C:/work")
        remover._work_directory_resolution = SimpleNamespace(
            requested="C:/work", path=Path("C:/work"), warning="")
        statuses = [StorageVolumeStatus(
            path=Path("C:/work"),
            free_bytes=500 * 1024 ** 3,
            required_bytes=1024,
            purposes=("temporary processing files",),
        )]
        with unittest.mock.patch(
            "backend._encode_mixin.assess_storage_volumes",
            return_value=statuses,
        ) as assess:
            remover._check_encode_disk_space(
                "D:/output/out.mp4",
                width=1920,
                height=1080,
                frames=300,
                high_bit=False,
                checkpoint_dir=Path("C:/work/checkpoints"),
            )

        requirements = assess.call_args.args[0]
        self.assertEqual(len(requirements), 3)
        self.assertEqual(
            {requirement.purpose for requirement in requirements},
            {
                "temporary processing files",
                "final output",
                "checkpoint and resume frames",
            },
        )

    def test_zero_frames_is_noop(self):
        remover = self._remover()
        # must not raise or probe when there is nothing to encode
        remover._check_encode_disk_space(
            "out.mp4", width=640, height=480, frames=0, high_bit=False)

    def test_json_log_rolls_over_when_large(self):
        from backend import processor
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "vsr.jsonl"
            log.write_bytes(b"x" * (10 * 1024 * 1024 + 10))
            handler = processor.attach_json_log(str(log))
            try:
                self.assertTrue((Path(tmp) / "vsr.jsonl.1").exists())
                self.assertLess(log.stat().st_size, 1024)
            finally:
                if handler is not None:
                    logging.getLogger().removeHandler(handler)
                    try:
                        handler.close()
                    except Exception:
                        pass


class EncodeStageCheckpointTests(unittest.TestCase):
    """P1: encode/mux-phase resume marker."""

    def _write_frames(self, frame_dir, n):
        import cv2
        frame_dir.mkdir(parents=True, exist_ok=True)
        img = np.zeros((8, 8, 3), np.uint8)
        for i in range(n):
            cv2.imwrite(str(frame_dir / f"frame_{i:06d}.png"), img)

    def _args(self, tmp, frame_dir):
        return dict(
            input_path=str(tmp / "in.mp4"),
            output_path=str(tmp / "out.mp4"),
            config_hash="abc",
            frame_dir=frame_dir,
            total_frames=5,
            width=8, height=8, fps=24.0,
        )

    def test_marker_persists_stage_and_flag(self):
        from backend import resume_checkpoint as rc
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            frame_dir = tmp / "job.frames"
            payload = rc.write_pause_checkpoint(
                tmp, "job", next_frame=5, status="running",
                stage="encoding", inpaint_complete=True,
                **self._args(tmp, frame_dir))
            self.assertEqual(payload["stage"], "encoding")
            self.assertTrue(payload["inpaint_complete"])

    def test_resume_detects_inpaint_complete(self):
        from backend import resume_checkpoint as rc
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "in.mp4").write_bytes(b"x")
            frame_dir = tmp / "job.frames"
            self._write_frames(frame_dir, 5)
            args = self._args(tmp, frame_dir)
            rc.write_pause_checkpoint(
                tmp, "job", next_frame=5, status="running",
                stage="encoding", inpaint_complete=True, **args)
            state = rc.load_pause_checkpoint(
                tmp, "job",
                input_path=args["input_path"], output_path=args["output_path"],
                config_hash="abc", total_frames=5, width=8, height=8, fps=24.0)
            self.assertEqual(state.next_frame, 5)
            self.assertTrue(state.inpaint_complete)

    def test_partial_inpaint_is_not_complete(self):
        from backend import resume_checkpoint as rc
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "in.mp4").write_bytes(b"x")
            frame_dir = tmp / "job.frames"
            self._write_frames(frame_dir, 3)  # only 3 of 5
            args = self._args(tmp, frame_dir)
            rc.write_pause_checkpoint(
                tmp, "job", next_frame=3, status="running",
                stage="inpainting", inpaint_complete=False, **args)
            state = rc.load_pause_checkpoint(
                tmp, "job",
                input_path=args["input_path"], output_path=args["output_path"],
                config_hash="abc", total_frames=5, width=8, height=8, fps=24.0)
            self.assertEqual(state.next_frame, 3)
            self.assertFalse(state.inpaint_complete)

    def test_resume_reports_mid_sequence_gap_and_orphaned_frames(self):
        from backend import resume_checkpoint as rc
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "in.mp4").write_bytes(b"x")
            frame_dir = tmp / "job.frames"
            self._write_frames(frame_dir, 5)
            (frame_dir / "frame_000002.png").unlink()
            args = self._args(tmp, frame_dir)
            rc.write_pause_checkpoint(
                tmp, "job", next_frame=5, status="paused",
                stage="inpainting", inpaint_complete=False, **args)

            state = rc.load_pause_checkpoint(
                tmp, "job",
                input_path=args["input_path"], output_path=args["output_path"],
                config_hash="abc", total_frames=5, width=8, height=8,
                fps=24.0)

            self.assertEqual(state.next_frame, 2)
            self.assertIn("gap at frame 2", state.warning)
            self.assertIn("2 later orphaned frame file(s)", state.warning)
            self.assertIn("reset to the first gap", state.warning)

    def test_resume_reports_gap_before_first_frame(self):
        from backend import resume_checkpoint as rc
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "in.mp4").write_bytes(b"x")
            frame_dir = tmp / "job.frames"
            self._write_frames(frame_dir, 3)
            (frame_dir / "frame_000000.png").unlink()
            args = self._args(tmp, frame_dir)
            rc.write_pause_checkpoint(
                tmp, "job", next_frame=3, status="paused",
                stage="inpainting", inpaint_complete=False, **args)

            state = rc.load_pause_checkpoint(
                tmp, "job",
                input_path=args["input_path"], output_path=args["output_path"],
                config_hash="abc", total_frames=5, width=8, height=8,
                fps=24.0)

            self.assertEqual(state.next_frame, 0)
            self.assertIn("gap starts at frame 0", state.warning)
            self.assertIn("2 later orphaned frame file(s)", state.warning)


class OutputIntegrityValidationTests(unittest.TestCase):
    """P0: validate final video integrity before destination promotion."""

    def _probe(self, **kwargs):
        base = {
            "path": "x", "has_video": True, "duration": 3.0, "frames": 90,
            "width": 320, "height": 240, "error": "", "prober": "ffprobe",
        }
        base.update(kwargs)
        return base

    def test_passes_when_output_matches_reference_envelope(self):
        from backend import io as vio
        with unittest.mock.patch.object(
            vio, "probe_video_integrity",
            side_effect=[self._probe(duration=3.0, frames=90),
                         self._probe(duration=3.0, frames=90)],
        ):
            ok, reason, _ = vio.validate_video_output("cand.mp4", reference="ref.mp4")
        self.assertTrue(ok, reason)

    def test_fails_closed_on_shortest_audio_truncation(self):
        from backend import io as vio
        with unittest.mock.patch.object(
            vio, "probe_video_integrity",
            side_effect=[self._probe(duration=1.021, frames=30),
                         self._probe(duration=3.0, frames=90)],
        ):
            ok, reason, _ = vio.validate_video_output("cand.mp4", reference="ref.mp4")
        self.assertFalse(ok)
        self.assertIn("shorter", reason)

    def test_fails_closed_on_fixed_frame_limit_stall(self):
        from backend import io as vio
        with unittest.mock.patch.object(
            vio, "probe_video_integrity",
            side_effect=[self._probe(duration=8.0, frames=240),
                         self._probe(duration=40.0, frames=1200)],
        ):
            ok, reason, _ = vio.validate_video_output("cand.mp4", reference="ref.mp4")
        self.assertFalse(ok)

    def test_fails_closed_when_no_video_stream(self):
        from backend import io as vio
        with unittest.mock.patch.object(
            vio, "probe_video_integrity",
            return_value=self._probe(has_video=False, error="no video stream"),
        ):
            ok, reason, _ = vio.validate_video_output("cand.mp4")
        self.assertFalse(ok)
        self.assertIn("video", reason)

    def test_longer_valid_output_still_passes(self):
        from backend import io as vio
        with unittest.mock.patch.object(
            vio, "probe_video_integrity",
            side_effect=[self._probe(duration=3.2, frames=96),
                         self._probe(duration=3.0, frames=90)],
        ):
            ok, _, _ = vio.validate_video_output("cand.mp4", reference="ref.mp4")
        self.assertTrue(ok)

    def test_inconclusive_probe_does_not_invent_failure(self):
        from backend import io as vio
        with unittest.mock.patch.object(
            vio, "probe_video_integrity",
            return_value={"path": "x", "has_video": False, "duration": 0.0,
                          "frames": 0, "error": "", "prober": ""},
        ):
            ok, _, _ = vio.validate_video_output("cand.mp4")
        self.assertTrue(ok)

    def test_promote_video_output_raises_and_preserves_destination(self):
        from backend import processor as proc
        remover = proc.SubtitleRemover.__new__(proc.SubtitleRemover)
        promoted = []
        with unittest.mock.patch.object(
            proc, "validate_video_output",
            return_value=(False, "output duration shorter", {}),
        ), unittest.mock.patch.object(
            proc, "_promote_temp_output",
            side_effect=lambda *a, **k: promoted.append(a),
        ):
            with self.assertRaises(proc.OutputIntegrityError):
                remover._promote_video_output("cand.mp4", "out.mp4", reference="ref.mp4")
        self.assertEqual(promoted, [])  # destination never replaced

    def test_end_to_end_short_audio_mux_is_caught(self):
        """Integration: real ffmpeg -shortest mux of a 3s video + 1s audio
        must fail validation against the 3s processed video."""
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg/ffprobe not installed")
        from backend import io as vio
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "video.mp4"
            merged = root / "merged.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-f", "lavfi", "-i", "testsrc=size=64x64:rate=30:duration=3",
                 "-pix_fmt", "yuv420p", str(video)],
                check=True, timeout=60,
            )
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-i", str(video),
                 "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                 "-shortest", "-map", "0:v:0", "-map", "1:a:0", str(merged)],
                check=True, timeout=60,
            )
            ok, reason, _ = vio.validate_video_output(str(merged), reference=str(video))
            self.assertFalse(ok, "short-audio -shortest mux should fail closed")
            self.assertIn("shorter", reason)



if __name__ == "__main__":
    unittest.main()
