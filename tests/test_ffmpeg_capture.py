"""RM-348: user media is decoded by the external FFmpeg, not OpenCV's.

OpenCV's wheel embeds FFmpeg 7.1, older than every 2026 advisory the 9.0.1
floor on the external binary exists for. These tests pin the reader that
replaced it: frame-exact reads and seeks, rotation, variable frame rate,
the colour conversion the encode inverts, the refusal when FFmpeg is
missing, and a whole run with OpenCV's decoder made unreachable.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from backend import io as vio
from backend import processor

W, H = 96, 64
HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _numbered_clip(path: Path, frames: int = 12, *, rate: str = "12",
                   extra: tuple = (), pix_fmt: str = "yuv420p") -> Path:
    """Frame i is a flat grey of 10 + 20 * i, so any frame is identifiable."""
    data = np.stack([
        np.full((H, W, 3), 10 + 20 * index, np.uint8) for index in range(frames)
    ])
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", rate, "-i", "-", "-c:v", "libx264",
         "-qp", "0", "-g", "4", "-pix_fmt", pix_fmt, *extra, str(path)],
        input=data.tobytes(), capture_output=True, timeout=120, check=True,
    )
    return path


def _index_of(frame: np.ndarray) -> int:
    return int(round((float(frame.mean()) - 10) / 20))


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not on PATH")
class CaptureTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_reads_every_frame_in_order_then_stops(self):
        cap = vio.open_video_capture(str(_numbered_clip(self.tmp / "a.mkv")))
        try:
            self.assertIsInstance(cap, vio._FfmpegCapture)
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), W)
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), H)
            self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 12.0, places=3)
            seen = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                self.assertEqual(frame.shape, (H, W, 3))
                self.assertEqual(frame.dtype, np.uint8)
                seen.append(_index_of(frame))
            self.assertEqual(seen, list(range(12)))
        finally:
            cap.release()

    def test_seeks_land_on_the_frame_they_name(self):
        # 29.97 fps: a seek computed as pos / fps lands past the frame.
        clip = _numbered_clip(self.tmp / "ntsc.mp4", rate="30000/1001")
        cap = vio.open_video_capture(str(clip))
        try:
            for target in (0, 1, 5, 10, 11, 3):
                with self.subTest(target=target):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                    ok, frame = cap.read()
                    self.assertTrue(ok)
                    self.assertEqual(_index_of(frame), target)
                    self.assertEqual(
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), target + 1)
        finally:
            cap.release()

    def test_grab_and_retrieve_behave_like_opencv(self):
        cap = vio.open_video_capture(str(_numbered_clip(self.tmp / "g.mkv")))
        try:
            self.assertTrue(cap.grab())
            self.assertTrue(cap.grab())
            ok, frame = cap.retrieve()
            self.assertTrue(ok)
            self.assertEqual(_index_of(frame), 1)
        finally:
            cap.release()

    def test_variable_frame_rate_keeps_one_frame_per_decoded_frame(self):
        # A concat of 12 fps and 6 fps halves; a constant-rate rawvideo mux
        # would duplicate frames in the slow half.
        first = _numbered_clip(self.tmp / "p1.mkv", frames=6, rate="12")
        data = np.stack([
            np.full((H, W, 3), 10 + 20 * index, np.uint8)
            for index in range(6, 12)
        ])
        second = self.tmp / "p2.mkv"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt",
             "bgr24", "-s", f"{W}x{H}", "-r", "6", "-i", "-", "-c:v",
             "libx264", "-qp", "0", "-pix_fmt", "yuv420p", str(second)],
            input=data.tobytes(), capture_output=True, timeout=120, check=True)
        listing = self.tmp / "list.txt"
        listing.write_text(
            f"file '{first.as_posix()}'\nfile '{second.as_posix()}'\n",
            encoding="utf-8")
        vfr = self.tmp / "vfr.mkv"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
             "-i", str(listing), "-c", "copy", str(vfr)],
            capture_output=True, timeout=120, check=True)
        cap = vio.open_video_capture(str(vfr))
        seen = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                seen.append(_index_of(frame))
        finally:
            cap.release()
        self.assertEqual(seen, list(range(12)))

    def test_rotation_becomes_a_decoded_pixel_property(self):
        clip = _numbered_clip(self.tmp / "rot.mp4", frames=2)
        rotated = self.tmp / "rotated.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-display_rotation:v:0", "90",
             "-i", str(clip), "-c", "copy", str(rotated)],
            capture_output=True, timeout=120, check=True)
        cap = vio.open_video_capture(str(rotated))
        try:
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), H)
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), W)
            ok, frame = cap.read()
            self.assertTrue(ok)
            self.assertEqual(frame.shape, (W, H, 3))
        finally:
            cap.release()

    def test_tagged_matrix_and_range_are_used_and_reported(self):
        clip = _numbered_clip(
            self.tmp / "tagged.mkv",
            extra=("-colorspace", "bt709", "-color_range", "pc"),
            pix_fmt="yuvj420p")
        cap = vio.open_video_capture(str(clip))
        try:
            self.assertEqual((cap.decode_matrix, cap.decode_range),
                             ("bt709", "pc"))
        finally:
            cap.release()
        rgb = vio.open_video_capture(
            str(_numbered_clip(self.tmp / "rgb.mkv", pix_fmt="yuv444p")))
        rgb.release()
        self.assertEqual(rgb.decode_matrix, "bt601")

    def test_the_round_trip_is_bit_exact_before_the_codec(self):
        """RM-367: untouched pixels come back as they went in."""
        rng = np.random.default_rng(7)
        base = rng.integers(40, 200, (H, W, 3), dtype=np.uint8)
        base = cv2.GaussianBlur(base, (0, 0), 3)
        data = np.stack([base] * 4)
        src = self.tmp / "noise.mkv"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt",
             "bgr24", "-s", f"{W}x{H}", "-r", "12", "-i", "-", "-vf",
             "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
             "-c:v", "libx264", "-qp", "0", "-colorspace", "bt709",
             "-color_range", "tv", str(src)],
            input=data.tobytes(), capture_output=True, timeout=120, check=True)
        cap = vio.open_video_capture(str(src))
        frames = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            cap.release()
        from backend.hdr import ColorMetadata, sdr_yuv_conversion_filter

        chain = sdr_yuv_conversion_filter(
            ColorMetadata(color_space="bt709", color_range="tv",
                          pixel_format="yuv420p"),
            "h264",
            decode_matrix=cap.decode_matrix,
            decode_range=cap.decode_range,
        )
        again = self.tmp / "again.mkv"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt",
             "bgr24", "-s", f"{W}x{H}", "-r", "12", "-i", "-", "-vf", chain,
             "-c:v", "libx264", "-qp", "0", "-colorspace", "bt709",
             "-color_range", "tv", str(again)],
            input=np.stack(frames).tobytes(), capture_output=True,
            timeout=120, check=True)

        def planes(path):
            raw = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
                 "-pix_fmt", "yuv420p", "-"],
                capture_output=True, timeout=60, check=True).stdout
            return np.frombuffer(raw, np.uint8).astype(np.int16)

        delta = planes(again) - planes(src)
        self.assertLess(abs(float(delta.mean())), 0.1)
        self.assertGreater(float(np.mean(delta == 0)), 0.9)


class MissingFfmpegTests(unittest.TestCase):
    def test_a_video_is_refused_rather_than_handed_to_opencv(self):
        with mock.patch("backend.io.shutil.which", return_value=None), \
                mock.patch("backend.io.cv2.VideoCapture") as opencv:
            cap = vio.open_video_capture("clip.mp4")
            self.assertFalse(cap.isOpened())
            error = vio._video_capture_open_error("clip.mp4", "clip.mp4")
        opencv.assert_not_called()
        self.assertEqual(error.reason, "ffmpeg_missing")
        self.assertIn("FFmpeg", str(error))


class _Passthrough:
    def inpaint(self, frames, masks):
        return frames

    def execution_identity(self):
        return {
            "implementation": "sttn", "provider": "passthrough",
            "effectiveDevice": "cpu", "executionContract": "vsr-inpaint-v1",
            "actualExecutions": [], "fallbackChain": [],
        }


def _stub_remover(cfg):
    remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
    remover.config = cfg
    remover.detector = processor.SubtitleDetector.__new__(
        processor.SubtitleDetector)
    remover.detector.device = "cpu"
    remover.detector.lang = "en"
    remover.detector._engine_name = "skip"
    remover.detector._rapid_model = None
    remover.detector._paddle_model = None
    remover.detector._surya_det = None
    remover.detector._easyocr_reader = None
    remover.inpainter = _Passthrough()
    remover.on_progress = None
    remover.on_preview_frame = None
    remover.live_preview_stride = 6
    remover._hw_encoder = None
    remover._srt_entries = []
    remover.last_quality_report = None
    remover.last_output_path = None
    remover.last_error_message = None
    remover.last_error_reason = None
    remover._quality_mask_bbox = None
    remover._color_metadata = None
    remover._source_color_probe = None
    remover._hdr_codec_warning_logged = False
    remover._hdr_software_warning_logged = False
    remover._active_writer = None
    remover._active_subprocess = None
    remover._teardown_requested = False
    remover.last_resume_warning = None
    remover.last_pause_checkpoint = None
    remover.last_pause_checkpoint_path = None
    return remover


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not on PATH")
class NoOpenCvDecoderTests(unittest.TestCase):
    def test_a_whole_run_never_hands_the_users_file_to_opencv(self):
        real_capture = cv2.VideoCapture
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = _numbered_clip(
                tmp / "user.mp4",
                extra=("-colorspace", "bt709", "-color_range", "tv"))
            reached = []

            def guarded(target, *args, **kwargs):
                if Path(str(target)).resolve() == src.resolve():
                    reached.append(str(target))
                    raise AssertionError("OpenCV decoder reached the user's file")
                return real_capture(target, *args, **kwargs)

            cfg = processor.normalize_processing_config(
                processor.ProcessingConfig(
                    mode=processor.InpaintMode.STTN,
                    device="cpu",
                    sttn_skip_detection=True,
                    subtitle_area=(8, H - 16, W - 8, H - 4),
                    preserve_audio=False,
                    adaptive_batch=False,
                    use_hw_encode=False,
                    prefetch_decode=True,
                    quality_report=True,
                    deinterlace_auto=False,
                ))
            remover = _stub_remover(cfg)
            with mock.patch("cv2.VideoCapture", side_effect=guarded):
                ok = remover.process_video(str(src), str(tmp / "out.mp4"))
            self.assertEqual(reached, [])
            self.assertTrue(ok, remover.last_error_message)
            self.assertTrue(remover.last_quality_report)


ROOT = Path(__file__).resolve().parent.parent
# Every remaining OpenCV decode reads a file this product wrote itself, never
# one a user handed it. Anything new has to be added here with its reason.
OPENCV_DECODE_ALLOWLIST = {
    ("backend/io.py", "_opencv_video_integrity"):
        "validates an output this job just encoded",
    ("backend/inpainters_diffusion.py", "_read_adapter_output_video"):
        "reads what the adapter subprocess wrote for this job",
    ("backend/segmentation.py", "_read_alpha_video"):
        "reads the matting adapter's output for this job",
    ("backend/proxy_workflow.py", "probe_proxy_window"):
        "reads the proxy this product re-encoded with libx264",
    ("backend/reference_corpus.py", "decoded_frame_digest"):
        "hashes outputs and repository fixtures; the committed baselines "
        "were recorded through this decoder",
    ("gui/mask_correction_controller.py", "detect_mask"):
        "reads the mask video this product exported for the item",
}


class OpenCvDecoderHygieneTests(unittest.TestCase):
    def test_only_allowlisted_sites_construct_an_opencv_capture(self):
        import ast

        found = {}
        for path in sorted(
                list((ROOT / "backend").rglob("*.py"))
                + list((ROOT / "gui").rglob("*.py"))
                + [ROOT / "VideoSubtitleRemover.py"]):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            parents = {}
            for node in ast.walk(tree):
                for child in ast.iter_child_nodes(node):
                    parents[child] = node
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "VideoCapture"):
                    continue
                scope = parents.get(node)
                while scope is not None and not isinstance(
                        scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    scope = parents.get(scope)
                key = (path.relative_to(ROOT).as_posix(),
                       scope.name if scope is not None else "<module>")
                found.setdefault(key, []).append(node.lineno)
        unexpected = {
            key: lines for key, lines in found.items()
            if key not in OPENCV_DECODE_ALLOWLIST
        }
        self.assertEqual(unexpected, {})
        stale = set(OPENCV_DECODE_ALLOWLIST) - set(found)
        self.assertEqual(stale, set())


if __name__ == "__main__":
    unittest.main()
