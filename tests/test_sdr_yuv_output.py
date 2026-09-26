"""RM-351: SDR YUV sources come back as YUV with their own tags and colours.

The benchmark clip is ordinary 4:2:0 footage tagged bt709 and tv range, and
every run on it failed the output contract on a host with NVENC. The final
encode read the pipeline's BGR frames and let FFmpeg choose the pixel format
and the matrix: NVENC took the RGB frames as they were and wrote RGB H.264
(gbrp, matrix gbr, range pc), and libx264 wrote High 4:4:4 from a 4:2:0
source. Worse, FFmpeg converted with the *tagged* matrix while OpenCV had
decoded with BT.601, so every untouched pixel of a BT.709 source shifted
colour, which the quality gate cannot see because it decodes both files
through the same OpenCV path. These tests pin the explicit conversion that
replaced FFmpeg's choice.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from backend import processor
from backend.hdr import (
    OPENCV_DECODE_MATRIX,
    ColorMetadata,
    sdr_yuv_conversion_filter,
    yuv_chroma_layout,
)

W, H, FRAMES = 96, 64, 12
# Four flat bands, moderately saturated so neither matrix clips them.
BANDS = ((60, 120, 200), (200, 80, 60), (90, 180, 90), (180, 160, 40))
BAND_W = W // len(BANDS)


def _filters(chain: str) -> dict:
    """Map each filter in a -vf chain to its options."""
    parsed = {}
    for item in chain.split(","):
        name, _, options = item.partition("=")
        parsed[name] = dict(
            part.split("=", 1) for part in options.split(":") if "=" in part
        ) if name != "format" else {"pix_fmt": options}
    return parsed


def _probe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=pix_fmt,color_space,color_range,profile", "-of", "json",
         str(path)],
        capture_output=True, text=True, timeout=30, check=True,
    )
    return json.loads(result.stdout)["streams"][0]


def _planes(path: Path) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", str(FRAMES),
         "-fps_mode", "passthrough",
         "-f", "rawvideo", "-pix_fmt", "yuv444p", "-"],
        capture_output=True, timeout=60, check=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, 3, H, W).astype(np.int16)


def _write_yuv_clip(path: Path, *, full_range: bool = False) -> Path:
    """Flat bands encoded with the real BT.709 matrix, as a camera would."""
    frames = []
    for index in range(FRAMES):
        frame = np.zeros((H, W, 3), np.uint8)
        for band, colour in enumerate(BANDS):
            frame[:, band * BAND_W:(band + 1) * BAND_W] = np.clip(
                np.array(colour) + 2 * index, 0, 255)
        frames.append(frame)
    colour_range = "pc" if full_range else "tv"
    pix_fmt = "yuvj420p" if full_range else "yuv420p"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", "12", "-i", "-",
         "-vf", f"scale=out_color_matrix=bt709:out_range={colour_range},"
                f"format={pix_fmt}",
         "-c:v", "libx264", "-qp", "0",
         "-colorspace", "bt709", "-color_range", colour_range, str(path)],
        input=np.stack(frames).tobytes(),
        capture_output=True, timeout=120, check=True,
    )
    return path


def _band_interiors() -> np.ndarray:
    """Flat pixels above the fixed removal region, away from band edges."""
    inner = np.zeros((H, W), bool)
    for band in range(len(BANDS)):
        inner[4:H - 20, band * BAND_W + 4:(band + 1) * BAND_W - 4] = True
    return inner


def _nvenc_works() -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
             "testsrc2=size=256x144:rate=12", "-frames:v", "2",
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


class _PassthroughInpainter:
    def inpaint(self, frames, masks):
        return frames

    def execution_identity(self):
        return {
            "implementation": "sttn",
            "provider": "passthrough",
            "effectiveDevice": "cpu",
            "executionContract": "vsr-inpaint-v1",
            "actualExecutions": [],
            "fallbackChain": [],
        }


def _remover(cfg: processor.ProcessingConfig, hw_encoder=None):
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
    remover.inpainter = _PassthroughInpainter()
    remover.on_progress = None
    remover.on_preview_frame = None
    remover.live_preview_stride = 6
    remover._hw_encoder = hw_encoder
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


def _config(**overrides) -> processor.ProcessingConfig:
    fields = dict(
        mode=processor.InpaintMode.STTN,
        device="cpu",
        sttn_skip_detection=True,
        subtitle_area=(8, H - 16, W - 8, H - 4),
        preserve_audio=False,
        adaptive_batch=False,
        use_hw_encode=False,
        output_quality=0,
        prefetch_decode=False,
        quality_report=False,
        # idet reads small synthetic patterns as interlaced, and yadif's
        # field-rate output would double the frame count under the test.
        deinterlace_auto=False,
    )
    fields.update(overrides)
    return processor.normalize_processing_config(
        processor.ProcessingConfig(**fields))


class ConversionChoiceTests(unittest.TestCase):
    def test_chroma_layouts(self):
        for name, layout in {
            "yuv420p": "420", "yuvj420p": "420", "yuv420p10le": "420",
            "yuv411p": "420", "nv12": "420", "p010le": "420", "gray": "420",
            "yuv422p": "422", "yuv422p10le": "422", "yuyv422": "422",
            "nv16": "422", "yuv444p": "444", "yuvj444p": "444",
            "bgr0": "", "gbrp": "", "rgb24": "", "rgba": "", "": "",
        }.items():
            self.assertEqual(yuv_chroma_layout(name), layout, name)

    def test_target_format_follows_the_encoder_that_will_take_it(self):
        meta = ColorMetadata(color_space="bt709", color_range="tv",
                             pixel_format="yuv422p")

        def target(codec, **kwargs):
            chain = sdr_yuv_conversion_filter(meta, codec, **kwargs)
            return _filters(chain)["format"]["pix_fmt"]

        self.assertEqual(target("h264"), "yuv422p")
        self.assertEqual(target("h265"), "yuv422p")
        # libsvtav1 is 4:2:0 only, libvvenc takes only yuv420p10le, and
        # every hardware encoder takes nv12.
        self.assertEqual(target("av1"), "yuv420p")
        self.assertEqual(target("vvc"), "yuv420p10le")
        self.assertEqual(target("h264", hardware=True), "nv12")

    def test_inverse_uses_the_decode_matrix_and_the_output_keeps_the_tags(self):
        tagged = ColorMetadata(color_space="bt709", color_range="pc",
                               pixel_format="yuv420p")
        chain = _filters(sdr_yuv_conversion_filter(tagged, "h264"))
        self.assertEqual(
            chain["scale"]["out_color_matrix"], OPENCV_DECODE_MATRIX)
        # OpenCV decodes a non-yuvj full-range stream as limited, so the
        # inverse is limited too, while the frames keep the source's tags.
        self.assertEqual(chain["scale"]["out_range"], "tv")
        self.assertEqual(chain["setparams"], {
            "colorspace": "bt709", "range": "pc",
            "color_primaries": "unknown", "color_trc": "unknown"})
        full = _filters(sdr_yuv_conversion_filter(
            ColorMetadata(color_range="pc", pixel_format="yuvj420p"), "h264"))
        self.assertEqual(full["scale"]["out_range"], "pc")
        # Untagged matrix stays untagged; an untagged range describes the
        # data the conversion produced.
        self.assertEqual(full["setparams"]["colorspace"], "unknown")
        self.assertEqual(full["setparams"]["range"], "pc")

    def test_untagged_output_when_tags_are_not_preserved(self):
        meta = ColorMetadata(color_space="bt709", color_range="tv",
                             pixel_format="yuv420p")
        chain = _filters(sdr_yuv_conversion_filter(
            meta, "h264", preserve_tags=False))
        self.assertEqual(chain["setparams"]["colorspace"], "unknown")

    def test_hdr_rgb_and_unknown_sources_keep_their_existing_path(self):
        hdr = ColorMetadata(color_transfer="smpte2084", color_space="bt2020nc",
                            pixel_format="yuv420p10le")
        self.assertEqual(sdr_yuv_conversion_filter(hdr, "h265"), "")
        self.assertEqual(sdr_yuv_conversion_filter(
            ColorMetadata(pixel_format="bgr0"), "h264"), "")
        self.assertEqual(sdr_yuv_conversion_filter(ColorMetadata(), "h264"), "")
        self.assertEqual(sdr_yuv_conversion_filter(None, "h264"), "")


class EncodeArgumentTests(unittest.TestCase):
    def _remover(self, hw_encoder=None, **cfg):
        remover = _remover(_config(use_hw_encode=bool(hw_encoder), **cfg),
                           hw_encoder)
        remover._output_contract = None
        remover._color_metadata = ColorMetadata(
            color_space="bt709", color_range="tv", pixel_format="yuv420p")
        return remover

    def test_only_rgb_frame_encodes_are_converted(self):
        remover = self._remover()
        converted = remover._get_encode_args(rgb_frames=True)
        self.assertIn("-filter:v:0", converted)
        chain = _filters(converted[converted.index("-filter:v:0") + 1])
        self.assertEqual(chain["format"]["pix_fmt"], "yuv420p")
        # Post-restore passes re-read a finished YUV file and must not be
        # converted a second time.
        self.assertNotIn("-filter:v:0", remover._get_encode_args())

    def test_nvenc_gets_nv12_with_the_source_tags(self):
        args = self._remover("h264_nvenc")._get_encode_args(rgb_frames=True)
        self.assertIn("h264_nvenc", args)
        chain = _filters(args[args.index("-filter:v:0") + 1])
        self.assertEqual(chain["format"]["pix_fmt"], "nv12")
        self.assertEqual(args[args.index("-colorspace") + 1], "bt709")
        self.assertEqual(args[args.index("-color_range") + 1], "tv")

    def test_d3d12_chain_starts_with_the_conversion(self):
        args = self._remover("h264_d3d12va")._get_encode_args(rgb_frames=True)
        self.assertEqual(args.count("-filter:v:0"), 1)
        chain = args[args.index("-filter:v:0") + 1]
        self.assertTrue(chain.startswith("scale=out_color_matrix=bt601"), chain)
        self.assertTrue(chain.endswith(",hwupload,scale_d3d12=w=iw:h=ih"), chain)

    def test_disabled_colour_tagging_still_restores_a_yuv_layout(self):
        remover = self._remover(preserve_color_metadata=False)
        remover._source_color_probe = remover._color_metadata
        remover._color_metadata = None
        args = remover._get_encode_args(rgb_frames=True)
        self.assertIn("-filter:v:0", args)
        self.assertNotIn("-colorspace", args)


@unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg not on PATH")
class OpenCvDecodeMatrixTests(unittest.TestCase):
    """Positive control for OPENCV_DECODE_MATRIX. If OpenCV starts honouring
    the stream's matrix tag, this fails and the inverse must change too."""

    def test_opencv_decodes_a_bt709_tagged_stream_with_bt601(self):
        import cv2

        with tempfile.TemporaryDirectory() as tmpdir:
            clip = _write_yuv_clip(Path(tmpdir) / "tagged.mkv")
            cap = cv2.VideoCapture(str(clip))
            ok, frame = cap.read()
            cap.release()
            self.assertTrue(ok)
            error = {}
            for matrix in ("bt601", "bt709"):
                raw = subprocess.run(
                    ["ffmpeg", "-v", "error", "-i", str(clip), "-frames:v",
                     "1", "-vf",
                     f"scale=in_color_matrix={matrix}:in_range=tv:"
                     "out_range=pc,format=bgr24",
                     "-f", "rawvideo", "-"],
                    capture_output=True, timeout=60, check=True).stdout
                decoded = np.frombuffer(raw, np.uint8).reshape(H, W, 3)
                error[matrix] = float(np.abs(
                    frame.astype(int) - decoded.astype(int)).mean())
            self.assertEqual(OPENCV_DECODE_MATRIX, "bt601")
            self.assertLess(error["bt601"], 0.5, error)
            self.assertGreater(error["bt709"], 3.0, error)


@unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg not on PATH")
class PipelineOutputTests(unittest.TestCase):
    # The FFmpeg reader and the inverse agree exactly (RM-348, RM-367): the
    # bands come back with no luma error and under a level of chroma, where
    # OpenCV's decode lost 1.15 levels of luma and a wrong inverse matrix
    # costs about seven.
    LUMA_CEILING = 0.5
    CHROMA_CEILING = 1.0

    def _run(self, *, hw_encoder=None, full_range=False, checkpoint=False):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = _write_yuv_clip(tmp / "source.mkv", full_range=full_range)
            output = tmp / "cleaned.mp4"
            remover = _remover(
                _config(use_hw_encode=bool(hw_encoder)), hw_encoder)
            kwargs = {}
            if checkpoint:
                kwargs = {"checkpoint_dir": tmp / "checkpoints",
                          "checkpoint_key": "rm351"}
            ok = remover.process_video(str(src), str(output), **kwargs)
            self.assertTrue(ok, remover.last_error_message)
            self.assertEqual(
                remover.last_output_contract.get("status"), "preserved",
                remover.last_output_contract)
            produced = Path(remover.last_output_path or output)
            stream = _probe(produced)
            source, cleaned = _planes(src), _planes(produced)
            error = np.abs(cleaned - source)[:, :, _band_interiors()]
            return stream, error.mean(axis=(0, 2))

    def _assert_neutral(self, error):
        luma, cb, cr = (float(value) for value in error)
        self.assertLess(luma, self.LUMA_CEILING, error)
        self.assertLess(max(cb, cr), self.CHROMA_CEILING, error)

    def test_software_encode_keeps_4_2_0_the_tags_and_the_colours(self):
        stream, error = self._run()
        self.assertEqual(stream["pix_fmt"], "yuv420p")
        self.assertEqual(stream["color_space"], "bt709")
        self.assertEqual(stream["color_range"], "tv")
        self._assert_neutral(error)

    def test_a_wrong_inverse_matrix_is_caught_by_the_same_ceilings(self):
        # Negative control: without it the ceilings above could be loose
        # enough to pass the colour shift this change removed. The capture
        # decodes these bands with their own BT.709 tag, so an encode that
        # inverts BT.601 instead is exactly the old mismatch.
        from backend import hdr

        real = hdr.sdr_yuv_conversion_filter

        def mismatched(meta, codec, **kwargs):
            kwargs["decode_matrix"] = "bt601"
            return real(meta, codec, **kwargs)

        with mock.patch.object(
                hdr, "sdr_yuv_conversion_filter", side_effect=mismatched):
            _stream, error = self._run()
        self.assertGreater(float(error[0]), self.LUMA_CEILING * 2, error)

    def test_checkpoint_frames_take_the_same_conversion(self):
        stream, error = self._run(checkpoint=True)
        self.assertEqual(stream["pix_fmt"], "yuv420p")
        self.assertEqual(stream["color_space"], "bt709")
        self._assert_neutral(error)

    def test_full_range_source_stays_full_range(self):
        stream, error = self._run(full_range=True)
        self.assertIn(stream["pix_fmt"], {"yuvj420p", "yuv420p"})
        self.assertEqual(stream["color_range"], "pc")
        self._assert_neutral(error)

    def test_full_range_vp9_keeps_its_blacks_and_whites(self):
        """RM-367: OpenCV read a full-range stream without the yuvj flag as
        limited range and clipped it to 16..235 before the job began."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            frame = np.zeros((H, W, 3), np.uint8)
            frame[:, W // 2:] = 255
            src = tmp / "full.webm"
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt",
                 "bgr24", "-s", f"{W}x{H}", "-r", "12", "-i", "-", "-vf",
                 "scale=out_color_matrix=bt709:out_range=pc,format=yuv420p",
                 "-c:v", "libvpx-vp9", "-lossless", "1", "-colorspace",
                 "bt709", "-color_range", "pc", str(src)],
                input=np.stack([frame] * FRAMES).tobytes(),
                capture_output=True, timeout=120, check=True)
            self.assertEqual(_probe(src)["pix_fmt"], "yuv420p")
            output = tmp / "cleaned.mp4"
            remover = _remover(_config())
            self.assertTrue(
                remover.process_video(str(src), str(output)),
                remover.last_error_message)
            produced = Path(remover.last_output_path or output)
            self.assertEqual(_probe(produced)["color_range"], "pc")
            luma = _planes(produced)[:, 0, : H - 20]
            self.assertLessEqual(int(luma[:, :, 4: W // 2 - 4].max()), 2)
            self.assertGreaterEqual(int(luma[:, :, W // 2 + 4: -4].min()), 253)

    @unittest.skipUnless(_nvenc_works(), "h264_nvenc is not usable here")
    def test_nvenc_no_longer_writes_rgb_h264(self):
        for checkpoint in (False, True):
            with self.subTest(checkpoint=checkpoint):
                stream, error = self._run(hw_encoder="h264_nvenc",
                                          checkpoint=checkpoint)
                self.assertEqual(stream["pix_fmt"], "yuv420p")
                self.assertEqual(stream["color_space"], "bt709")
                self.assertEqual(stream["color_range"], "tv")
                self._assert_neutral(error)


def _tagged_bt709_clip(path: Path) -> Path:
    """Flat bands tagged the way camera and phone footage is: primaries,
    transfer, matrix and range all BT.709 limited. setparams puts the tags
    on the frames; FFmpeg 9 drops -color_primaries/-color_trc otherwise."""
    frames = np.stack([
        np.full((H, W, 3), (60 + 2 * index, 120, 200), np.uint8)
        for index in range(FRAMES)
    ])
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", "12", "-i", "-", "-vf",
         "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,"
         "setparams=colorspace=bt709:range=tv:color_primaries=bt709:"
         "color_trc=bt709",
         "-c:v", "libx264", "-qp", "0", str(path)],
        input=frames.tobytes(), capture_output=True, timeout=120, check=True,
    )
    return path


def _stream_tags(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=color_primaries,color_transfer,color_space,color_range",
         "-of", "json", str(path)],
        capture_output=True, text=True, timeout=30, check=True)
    return json.loads(result.stdout)["streams"][0]


@unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg not on PATH")
class ReviewRegressionTests(unittest.TestCase):
    """Defects an adversarial review found in the first version of this fix."""

    def test_a_fully_tagged_bt709_source_keeps_every_tag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = _tagged_bt709_clip(tmp / "camera.mp4")
            expected = {
                "color_primaries": "bt709", "color_transfer": "bt709",
                "color_space": "bt709", "color_range": "tv",
            }
            self.assertEqual(_stream_tags(src), expected)
            for checkpoint in (False, True):
                with self.subTest(checkpoint=checkpoint):
                    remover = _remover(_config())
                    kwargs = {}
                    if checkpoint:
                        kwargs = {"checkpoint_dir": tmp / f"ck{checkpoint}",
                                  "checkpoint_key": "tags"}
                    output = tmp / f"out_{checkpoint}.mp4"
                    self.assertTrue(
                        remover.process_video(str(src), str(output), **kwargs),
                        remover.last_error_message)
                    self.assertEqual(
                        _stream_tags(Path(remover.last_output_path or output)),
                        expected)

    def test_cover_art_and_soft_subtitles_survive_the_conversion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            srt = tmp / "subs.srt"
            srt.write_text(
                "1\n00:00:00,000 --> 00:00:00,500\nHello\n\n", encoding="utf-8")
            cover = tmp / "cover.png"
            import cv2

            cv2.imwrite(str(cover), np.full((32, 32, 3), 200, np.uint8))
            video = _tagged_bt709_clip(tmp / "video.mp4")
            src = tmp / "full.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", str(video),
                 "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                 "-i", str(srt), "-i", str(cover),
                 "-map", "0:v", "-map", "1:a", "-map", "2:s", "-map", "3:v",
                 "-c:v:0", "copy", "-c:a", "aac", "-c:s", "mov_text",
                 "-c:v:1", "png", "-disposition:v:1", "attached_pic",
                 "-t", "1", str(src)],
                capture_output=True, timeout=120, check=True)
            kinds = json.loads(subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
                 "-of", "json", str(src)],
                capture_output=True, text=True, timeout=30,
                check=True).stdout)["streams"]
            self.assertEqual(
                sorted(item["codec_type"] for item in kinds),
                ["audio", "subtitle", "video", "video"])
            remover = _remover(_config(preserve_audio=True))
            output = tmp / "out.mp4"
            self.assertTrue(
                remover.process_video(str(src), str(output)),
                remover.last_error_message)
            streams = json.loads(subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries",
                 "stream=codec_type:stream_disposition=attached_pic",
                 "-of", "json", str(Path(remover.last_output_path or output))],
                capture_output=True, text=True, timeout=30,
                check=True).stdout)["streams"]
            kinds = [item["codec_type"] for item in streams]
            self.assertIn("audio", kinds)
            self.assertIn("subtitle", kinds)
            self.assertTrue(any(
                (item.get("disposition") or {}).get("attached_pic")
                for item in streams), streams)

    def test_a_gray_source_keeps_its_luma(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            values = (16, 60, 180, 235)
            frame = np.zeros((H, W), np.uint8)
            for band, value in enumerate(values):
                frame[:, band * BAND_W:(band + 1) * BAND_W] = value
            src = tmp / "gray.mkv"
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt",
                 "gray", "-s", f"{W}x{H}", "-r", "12", "-i", "-",
                 "-c:v", "ffv1", str(src)],
                input=np.stack([frame] * FRAMES).tobytes(),
                capture_output=True, timeout=120, check=True)
            remover = _remover(_config())
            output = tmp / "out.mp4"
            self.assertTrue(
                remover.process_video(str(src), str(output)),
                remover.last_error_message)
            luma = _planes(Path(remover.last_output_path or output))[:, 0]
            for band, value in enumerate(values):
                inner = luma[:, 4:H - 20, band * BAND_W + 4:(band + 1) * BAND_W - 4]
                with self.subTest(value=value):
                    self.assertLessEqual(
                        abs(float(inner.mean()) - value), 1.0, inner.mean())

    def test_an_integrity_retry_stays_a_stream_copy(self):
        """A contract-ready file retried after a failed promotion must not
        be re-encoded through the BGR inverse (a second conversion)."""
        from backend.processor import OutputIntegrityError

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = _tagged_bt709_clip(tmp / "src.mp4")
            remover = _remover(_config(), hw_encoder="h264_nvenc")
            remover._color_metadata = ColorMetadata(
                color_space="bt709", color_range="tv", pixel_format="yuv420p")
            remover._output_contract = None
            remover._decode_colorimetry = ("bt709", "tv")
            commands = []
            promotions = iter([OutputIntegrityError("truncated", {}), None])

            def promote(*_args, **_kwargs):
                outcome = next(promotions)
                if outcome is not None:
                    raise outcome

            with mock.patch.object(
                    remover, "_run_checked_ffmpeg",
                    side_effect=lambda cmd, _timeout: commands.append(list(cmd))), \
                    mock.patch.object(remover, "_promote_video_output",
                                      side_effect=promote), \
                    mock.patch.object(remover, "_fallback_after_hw_failure",
                                      return_value=True), \
                    mock.patch("backend._encode_mixin.validate_container_payload",
                               return_value={}):
                remover._merge_audio(
                    str(src), str(src), str(tmp / "out.mp4"),
                    video_is_contract_ready=True)
            self.assertEqual(len(commands), 2)
            for command in commands:
                self.assertEqual(
                    command[command.index("-c:v") + 1], "copy", command)
                self.assertNotIn("-filter:v:0", command)


if __name__ == "__main__":
    unittest.main()
