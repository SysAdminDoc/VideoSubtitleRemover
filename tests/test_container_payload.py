import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend import container_payload, io, processor


class ContainerPayloadPlanTests(unittest.TestCase):
    def _manifest(self):
        return {
            "available": True,
            "format": {"name": "matroska", "tags": {"title": "Source title"}},
            "rotationDegrees": 90,
            "chapters": [{"id": 0}],
            "streams": [
                {"index": 0, "codecType": "video", "codecName": "h264", "tags": {}, "disposition": {}, "rotationDegrees": 90},
                {"index": 1, "codecType": "audio", "codecName": "pcm_s16le", "tags": {"language": "eng", "title": "Main"}, "disposition": {"default": True}},
                {"index": 2, "codecType": "audio", "codecName": "aac", "tags": {"language": "jpn"}, "disposition": {}},
                {"index": 3, "codecType": "subtitle", "codecName": "subrip", "tags": {"language": "spa"}, "disposition": {"forced": True}},
                {"index": 4, "codecType": "attachment", "codecName": "ttf", "tags": {"filename": "font.ttf"}, "disposition": {}},
                {"index": 5, "codecType": "data", "codecName": "klv", "tags": {}, "disposition": {}},
            ],
        }

    def test_matroska_copies_every_compatible_payload_stream(self):
        plan = container_payload.build_container_mux_plan(
            self._manifest(),
            "output.mkv",
            preserve_audio=True,
            multi_audio=True,
        )
        actions = {
            (item["type"], item["sourceIndex"]): item["action"]
            for item in plan["streams"]
        }
        self.assertEqual(actions[("audio", 1)], "copy")
        self.assertEqual(actions[("audio", 2)], "copy")
        self.assertEqual(actions[("subtitle", 3)], "copy")
        self.assertEqual(actions[("attachment", 4)], "copy")
        self.assertEqual(actions[("data", 5)], "drop")
        self.assertEqual(plan["chapters"]["action"], "copy")
        self.assertEqual(plan["rotation"]["action"], "bake-and-clear")

    def test_mp4_transcodes_only_incompatible_streams_and_records_losses(self):
        plan = container_payload.build_container_mux_plan(
            self._manifest(),
            "output.mp4",
            preserve_audio=True,
            multi_audio=True,
        )
        streams = {item["sourceIndex"]: item for item in plan["streams"]}
        self.assertEqual((streams[1]["action"], streams[1]["outputCodec"]), ("transcode", "aac"))
        self.assertEqual(streams[2]["action"], "copy")
        self.assertEqual((streams[3]["action"], streams[3]["outputCodec"]), ("transcode", "mov_text"))
        self.assertEqual(streams[4]["action"], "drop")
        self.assertEqual(streams[5]["action"], "drop")
        self.assertTrue(any("Dropped attachment stream 4" in item for item in plan["warnings"]))
        self.assertTrue(any("Styling may change" in item for item in plan["warnings"]))
        args = container_payload.build_container_mux_args(plan)
        self.assertIn("1:1", args)
        self.assertIn("-c:a:0", args)
        self.assertIn("-c:s:0", args)
        self.assertIn("-map_chapters", args)
        self.assertIn("rotate=0", args)

    def test_disabling_audio_does_not_drop_other_payload(self):
        plan = container_payload.build_container_mux_plan(
            self._manifest(),
            "output.mkv",
            preserve_audio=False,
            multi_audio=True,
        )
        audio = [item for item in plan["streams"] if item["type"] == "audio"]
        subtitle = next(item for item in plan["streams"] if item["type"] == "subtitle")
        self.assertTrue(all(item["action"] == "drop" for item in audio))
        self.assertEqual(subtitle["action"], "copy")

    def test_probe_reads_display_rotation(self):
        payload = {
            "format": {"format_name": "mov,mp4", "tags": {"title": "Rotated"}},
            "chapters": [],
            "streams": [{
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "side_data_list": [{"rotation": -90}],
                "tags": {},
                "disposition": {},
            }],
        }
        completed = mock.Mock(returncode=0, stdout=json.dumps(payload), stderr="")
        with mock.patch("backend.container_payload.shutil.which", return_value="ffprobe"), mock.patch(
            "backend.container_payload.run_process", return_value=completed
        ):
            manifest = container_payload.probe_container_manifest("rotated.mp4")
        self.assertTrue(manifest["available"])
        self.assertEqual(manifest["rotationDegrees"], 270)

    def test_opencv_capture_enables_orientation_auto(self):
        cap = mock.Mock()
        with mock.patch("backend.io.cv2.VideoCapture", return_value=cap):
            opened = io._open_capture("rotated.mp4", "off")
        self.assertIs(opened, cap)
        prop = getattr(io.cv2, "CAP_PROP_ORIENTATION_AUTO", None)
        if prop is not None:
            cap.set.assert_called_with(prop, 1)

    def test_output_contract_fails_closed_on_unpreserved_payload(self):
        remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        remover._output_contract = mock.Mock()
        remover._output_contract.validate.return_value = (True, [])
        remover._output_contract.report.return_value = {"codec": "h264"}
        remover.last_container_payload = {
            "status": "failed",
            "issues": ["subtitle stream missing"],
        }
        with self.assertRaisesRegex(processor.OutputIntegrityError, "container payload"):
            remover._validate_output_contract("output.mp4")


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe required")
class ContainerPayloadIntegrationTests(unittest.TestCase):
    def _run(self, command):
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def _make_source(self, root: Path) -> tuple[Path, Path]:
        subtitle = root / "captions.srt"
        subtitle.write_text("1\n00:00:00,100 --> 00:00:00,800\nHola\n", encoding="utf-8")
        attachment = root / "note.txt"
        attachment.write_text("fixture attachment", encoding="utf-8")
        metadata = root / "chapters.ffmeta"
        metadata.write_text(
            ";FFMETADATA1\ntitle=Container Contract\nartist=VSR Tests\n"
            "[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=500\ntitle=Intro\n"
            "[CHAPTER]\nTIMEBASE=1/1000\nSTART=500\nEND=1000\ntitle=Outro\n",
            encoding="utf-8",
        )
        source = root / "source.mkv"
        self._run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=660:sample_rate=48000:duration=1",
            "-i", str(subtitle),
            "-f", "ffmetadata", "-i", str(metadata),
            "-map", "0:v:0", "-map", "1:a:0", "-map", "2:a:0", "-map", "3:s:0",
            "-map_metadata", "4", "-map_chapters", "4",
            "-metadata:s:a:0", "language=eng", "-metadata:s:a:0", "title=Main",
            "-metadata:s:a:1", "language=jpn", "-metadata:s:a:1", "title=Alternate",
            "-metadata:s:s:0", "language=spa", "-metadata:s:s:0", "title=Captions",
            "-disposition:a:0", "default", "-disposition:a:1", "0",
            "-disposition:s:0", "forced",
            "-c:v", "ffv1", "-c:a:0", "pcm_s16le", "-c:a:1", "aac", "-c:s", "srt",
            "-attach", str(attachment),
            "-metadata:s:t:0", "filename=note.txt",
            "-metadata:s:t:0", "mimetype=text/plain",
            str(source),
        ])
        processed = root / "processed.mkv"
        self._run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn",
            "-c:v", "ffv1", str(processed),
        ])
        return source, processed

    def _remover(self, *, preserve_audio: bool = True):
        remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        remover.config = processor.ProcessingConfig(
            preserve_audio=preserve_audio,
            multi_audio_passthrough=True,
            loudnorm_target=0.0,
            output_codec="h264",
            output_quality=30,
            use_hw_encode=False,
            preserve_color_metadata=False,
        )
        remover._hw_encoder = None
        remover._color_metadata = None
        remover._output_contract = None
        remover._active_subprocess = None
        remover._teardown_requested = False
        remover.last_container_payload = {}
        return remover

    def test_mkv_and_mp4_preserve_compatible_payload_and_record_losses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, processed = self._make_source(root)

            mkv = root / "cleaned.mkv"
            mkv_remover = self._remover()
            self.assertEqual(mkv_remover._merge_audio(str(source), str(processed), str(mkv)), str(mkv))
            mkv_manifest = container_payload.probe_container_manifest(mkv)
            mkv_audio = [stream for stream in mkv_manifest["streams"] if stream["codecType"] == "audio"]
            mkv_subtitles = [stream for stream in mkv_manifest["streams"] if stream["codecType"] == "subtitle"]
            self.assertEqual([stream["codecName"] for stream in mkv_audio], ["pcm_s16le", "aac"])
            self.assertEqual([stream["tags"].get("language") for stream in mkv_audio], ["eng", "jpn"])
            self.assertEqual([stream["tags"].get("title") for stream in mkv_audio], ["Main", "Alternate"])
            self.assertTrue(mkv_audio[0]["disposition"]["default"])
            self.assertEqual(len(mkv_subtitles), 1)
            self.assertEqual(mkv_subtitles[0]["tags"].get("language"), "spa")
            self.assertEqual(mkv_subtitles[0]["tags"].get("title"), "Captions")
            self.assertTrue(mkv_subtitles[0]["disposition"]["forced"])
            self.assertEqual(len([stream for stream in mkv_manifest["streams"] if stream["codecType"] == "attachment"]), 1)
            self.assertEqual(len(mkv_manifest["chapters"]), 2)
            mkv_format_tags = {key.lower(): value for key, value in mkv_manifest["format"]["tags"].items()}
            self.assertEqual(mkv_format_tags.get("title"), "Container Contract")
            self.assertEqual(mkv_format_tags.get("artist"), "VSR Tests")
            self.assertEqual(mkv_remover.last_container_payload["status"], "preserved")

            mp4 = root / "cleaned.mp4"
            mp4_remover = self._remover()
            self.assertEqual(mp4_remover._merge_audio(str(source), str(processed), str(mp4)), str(mp4))
            mp4_manifest = container_payload.probe_container_manifest(mp4)
            mp4_audio = [stream for stream in mp4_manifest["streams"] if stream["codecType"] == "audio"]
            mp4_subtitles = [stream for stream in mp4_manifest["streams"] if stream["codecType"] == "subtitle"]
            self.assertEqual([stream["codecName"] for stream in mp4_audio], ["aac", "aac"])
            self.assertEqual([stream["tags"].get("language") for stream in mp4_audio], ["eng", "jpn"])
            self.assertEqual([stream["tags"].get("handler_name") for stream in mp4_audio], ["Main", "Alternate"])
            self.assertTrue(mp4_audio[0]["disposition"]["default"])
            self.assertEqual([stream["codecName"] for stream in mp4_subtitles], ["mov_text"])
            self.assertEqual(mp4_subtitles[0]["tags"].get("language"), "spa")
            self.assertEqual(mp4_subtitles[0]["tags"].get("handler_name"), "Captions")
            self.assertTrue(mp4_subtitles[0]["disposition"]["forced"])
            self.assertEqual(len([stream for stream in mp4_manifest["streams"] if stream["codecType"] == "attachment"]), 0)
            self.assertEqual(len(mp4_manifest["chapters"]), 2)
            mp4_format_tags = {key.lower(): value for key, value in mp4_manifest["format"]["tags"].items()}
            self.assertEqual(mp4_format_tags.get("title"), "Container Contract")
            self.assertEqual(mp4_format_tags.get("artist"), "VSR Tests")
            self.assertEqual(mp4_remover.last_container_payload["status"], "preserved")
            self.assertTrue(
                any("Dropped attachment stream" in warning for warning in mp4_remover.last_container_payload["warnings"])
            )

    def test_audio_disabled_still_preserves_subtitles_chapters_and_attachments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, processed = self._make_source(root)
            output = root / "silent.mkv"
            remover = self._remover(preserve_audio=False)
            remover._merge_audio(str(source), str(processed), str(output))
            manifest = container_payload.probe_container_manifest(output)
            self.assertEqual(len([stream for stream in manifest["streams"] if stream["codecType"] == "audio"]), 0)
            self.assertEqual(len([stream for stream in manifest["streams"] if stream["codecType"] == "subtitle"]), 1)
            self.assertEqual(len([stream for stream in manifest["streams"] if stream["codecType"] == "attachment"]), 1)
            self.assertEqual(len(manifest["chapters"]), 2)

    def test_selected_range_preserves_only_overlapping_chapters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _processed = self._make_source(root)
            processed = root / "partial-video.mkv"
            self._run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", "0.55", "-i", str(source), "-t", "0.45",
                "-map", "0:v:0", "-an", "-sn", "-dn", "-map_chapters", "-1",
                "-c:v", "ffv1", str(processed),
            ])
            output = root / "partial.mkv"
            remover = self._remover()
            remover._merge_audio(
                str(source), str(processed), str(output),
                start_seconds=0.55, end_seconds=1.0,
            )
            manifest = container_payload.probe_container_manifest(output)
            self.assertEqual(remover.last_container_payload["status"], "preserved")
            self.assertEqual(len(manifest["chapters"]), 1)
            self.assertEqual(manifest["chapters"][0]["tags"].get("title"), "Outro")
            self.assertTrue(
                any("outside the selected processing range" in warning for warning in remover.last_container_payload["warnings"])
            )

    def test_rotation_is_baked_once_and_stale_display_metadata_is_cleared(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.mp4"
            rotated = root / "rotated.mp4"
            processed = root / "processed.mp4"
            output = root / "cleaned.mp4"
            self._run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=1",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                str(base),
            ])
            self._run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-display_rotation:v:0", "90", "-i", str(base),
                "-map", "0", "-c", "copy", str(rotated),
            ])
            source_manifest = container_payload.probe_container_manifest(rotated)
            self.assertNotEqual(source_manifest["rotationDegrees"], 0)
            self._run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(rotated), "-map", "0:v:0", "-an",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(processed),
            ])
            processed_manifest = container_payload.probe_container_manifest(processed)
            remover = self._remover()
            remover._merge_audio(str(rotated), str(processed), str(output))
            output_manifest = container_payload.probe_container_manifest(output)
            processed_video = next(
                stream for stream in processed_manifest["streams"]
                if stream["codecType"] == "video"
            )
            output_video = next(
                stream for stream in output_manifest["streams"]
                if stream["codecType"] == "video"
            )
            self.assertEqual(output_manifest["rotationDegrees"], 0)
            self.assertEqual(
                (output_video["width"], output_video["height"]),
                (processed_video["width"], processed_video["height"]),
            )
            self.assertEqual(remover.last_container_payload["rotation"]["action"], "bake-and-clear")

    def test_post_restore_remux_restores_the_complete_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, processed = self._make_source(root)
            initial = root / "initial.mkv"
            transformed = root / "transformed.mkv"
            restored = root / "restored.mkv"
            remover = self._remover()
            remover._merge_audio(str(source), str(processed), str(initial))
            self._run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(initial), "-map", "0:v:0", "-an", "-sn", "-dn",
                "-vf", "hflip", "-c:v", "libx264", str(transformed),
            ])
            report = remover._remux_transformed_video(
                str(transformed), str(initial), str(restored)
            )
            manifest = container_payload.probe_container_manifest(restored)
            self.assertEqual(report["status"], "preserved")
            self.assertEqual(len([stream for stream in manifest["streams"] if stream["codecType"] == "audio"]), 2)
            self.assertEqual(len([stream for stream in manifest["streams"] if stream["codecType"] == "subtitle"]), 1)
            self.assertEqual(len([stream for stream in manifest["streams"] if stream["codecType"] == "attachment"]), 1)
            self.assertEqual(len(manifest["chapters"]), 2)

    def test_mp4_timecode_data_track_is_recreated_with_its_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "timecode.mp4"
            processed = root / "processed.mkv"
            output = root / "cleaned.mp4"
            self._run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-timecode", "00:00:00:00", str(source),
            ])
            self._run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(source), "-map", "0:v:0", "-c:v", "ffv1", str(processed),
            ])
            remover = self._remover()
            remover._merge_audio(str(source), str(processed), str(output))
            manifest = container_payload.probe_container_manifest(output)
            data = [stream for stream in manifest["streams"] if stream["codecType"] == "data"]
            source_data = next(
                stream for stream in remover.last_container_payload["streams"]
                if stream["type"] == "data"
            )
            self.assertEqual(source_data["action"], "recreate")
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["codecTag"], "tmcd")
            self.assertEqual(data[0]["tags"].get("timecode"), "00:00:00:0")
            self.assertTrue(
                any("container-generated handler metadata may differ" in warning for warning in remover.last_container_payload["warnings"])
            )


if __name__ == "__main__":
    unittest.main()
