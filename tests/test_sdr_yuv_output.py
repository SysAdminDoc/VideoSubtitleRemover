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
        self.assertEqual(chain["setparams"],
                         {"colorspace": "bt709", "range": "pc"})
        full = _filters(sdr_yuv_conversion_filter(
            ColorMetadata(color_range="pc", pixel_format="yuvj420p"), "h264"))
        self.assertEqual(full["scale"]["out_range"], "pc")
        # Untagged matrix stays untagged; an untagged range describes the
        # data the conversion produced.
        self.assertEqual(full["setparams"],
                         {"colorspace": "unknown", "range": "pc"})

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
        self.assertIn("-vf", converted)
        chain = _filters(converted[converted.index("-vf") + 1])
        self.assertEqual(chain["format"]["pix_fmt"], "yuv420p")
        # Post-restore passes re-read a finished YUV file and must not be
        # converted a second time.
        self.assertNotIn("-vf", remover._get_encode_args())

    def test_nvenc_gets_nv12_with_the_source_tags(self):
        args = self._remover("h264_nvenc")._get_encode_args(rgb_frames=True)
        self.assertIn("h264_nvenc", args)
        chain = _filters(args[args.index("-vf") + 1])
        self.assertEqual(chain["format"]["pix_fmt"], "nv12")
        self.assertEqual(args[args.index("-colorspace") + 1], "bt709")
        self.assertEqual(args[args.index("-color_range") + 1], "tv")

    def test_d3d12_chain_starts_with_the_conversion(self):
        args = self._remover("h264_d3d12va")._get_encode_args(rgb_frames=True)
        self.assertEqual(args.count("-vf"), 1)
        chain = args[args.index("-vf") + 1]
        self.assertTrue(chain.startswith("scale=out_color_matrix=bt601"), chain)
        self.assertTrue(chain.endswith(",hwupload,scale_d3d12=w=iw:h=ih"), chain)

    def test_disabled_colour_tagging_still_restores_a_yuv_layout(self):
        remover = self._remover(preserve_color_metadata=False)
        remover._source_color_probe = remover._color_metadata
        remover._color_metadata = None
        args = remover._get_encode_args(rgb_frames=True)
        self.assertIn("-vf", args)
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
    # Luma carries OpenCV's own decode bias of about one level (RM-367);
    # a wrong inverse matrix costs about seven on these bands.
    LUMA_CEILING = 2.5
    CHROMA_CEILING = 1.5

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
        # enough to pass the colour shift this change removed.
        with mock.patch("backend.hdr.OPENCV_DECODE_MATRIX", "bt709"):
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


if __name__ == "__main__":
    unittest.main()
