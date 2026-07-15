from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from backend.config import ProcessingConfig, normalize_processing_config
from backend.subtitle_translation import (
    SubtitleTranslationError,
    TRANSLATION_REQUEST_SCHEMA,
    TRANSLATION_RESPONSE_SCHEMA,
    parse_srt,
    provided_translation_evidence,
    read_srt,
    register_translation_provider,
    render_segments_srt,
    translate_srt_file,
    validate_translation_config,
)


SRT_TEXT = (
    "1\n"
    "00:00:00,000 --> 00:00:00,500\n"
    "Hello world\n\n"
    "cue-two\n"
    "00:00:00,500 --> 00:00:01,000 align:start\n"
    "Second line\n"
)


class SubtitleTranslationUnitTests(unittest.TestCase):
    def test_parse_preserves_identifier_timing_and_settings(self):
        cues = parse_srt(SRT_TEXT)

        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0].identifier, "1")
        self.assertEqual(cues[1].identifier, "cue-two")
        self.assertIn("align:start", cues[1].timing)

    def test_parser_rejects_malformed_or_reversed_cues(self):
        with self.assertRaises(SubtitleTranslationError):
            parse_srt("1\nnot timing\nText\n")
        with self.assertRaises(SubtitleTranslationError):
            parse_srt("1\n00:00:02,000 --> 00:00:01,000\nText\n")

    def test_registered_provider_preserves_cue_timing_and_writes_hashes(self):
        def provider(texts, _source, target, _options):
            return [f"[{target}] {text}" for text in texts]

        register_translation_provider("unit-local", provider, replace=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.srt"
            output = Path(tmpdir) / "translated.srt"
            source.write_text(SRT_TEXT, encoding="utf-8")

            report = translate_srt_file(
                source,
                output,
                provider_name="unit-local",
                source_language="en",
                target_language="fr",
            )
            cues = read_srt(output)

        self.assertEqual(report["status"], "translated")
        self.assertEqual(report["cueCount"], 2)
        self.assertEqual(len(report["source"]["sha256"]), 64)
        self.assertEqual(len(report["translated"]["sha256"]), 64)
        self.assertEqual(cues[0].timing, "00:00:00,000 --> 00:00:00,500")
        self.assertEqual(cues[0].text, "[fr] Hello world")

    def test_local_command_protocol_is_hidden_json_not_shell_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.srt"
            output = root / "translated.srt"
            command = root / "translator.py"
            source.write_text(SRT_TEXT, encoding="utf-8")
            command.write_text(
                "import json, sys\n"
                "request = json.load(sys.stdin)\n"
                f"assert request['schema'] == {TRANSLATION_REQUEST_SCHEMA!r}\n"
                "translations = ['LOCAL ' + cue['text'] for cue in request['cues']]\n"
                "print(json.dumps({'schema': "
                f"{TRANSLATION_RESPONSE_SCHEMA!r}, 'translations': translations}}))\n",
                encoding="utf-8",
            )

            report = translate_srt_file(
                source,
                output,
                provider_name="command",
                source_language="en",
                target_language="de",
                provider_options={"command": str(command), "timeout": 30},
            )

        self.assertEqual(report["provider"], "command")
        self.assertEqual(report["providerMode"], "local")

    def test_config_validation_fails_before_processing(self):
        missing_target = normalize_processing_config(ProcessingConfig(
            translation_enabled=True,
            translation_provider="command",
            translation_command=sys.executable,
        ))
        with self.assertRaisesRegex(SubtitleTranslationError, "target language"):
            validate_translation_config(missing_target)

        with tempfile.TemporaryDirectory() as tmpdir:
            translated = Path(tmpdir) / "ready.srt"
            translated.write_text(SRT_TEXT, encoding="utf-8")
            provided = normalize_processing_config(ProcessingConfig(
                translation_enabled=True,
                translation_srt=str(translated),
            ))
            validate_translation_config(provided)

    def test_whisper_segment_renderer_produces_valid_srt(self):
        text = render_segments_srt([(0.0, 1.25, "Spoken line")])
        cues = parse_srt(text)

        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].text, "Spoken line")
        self.assertIn("00:00:01,250", cues[0].timing)


class SubtitleTranslationProcessorTests(unittest.TestCase):
    @staticmethod
    def _provider(texts, _source, target, _options):
        return [f"{target}: {text}" for text in texts]

    def test_ocr_cues_generate_translated_srt_and_provenance(self):
        from backend.processor import SubtitleRemover

        register_translation_provider(
            "processor-local", self._provider, replace=True)
        remover = SubtitleRemover.__new__(SubtitleRemover)
        remover.config = normalize_processing_config(ProcessingConfig(
            translation_enabled=True,
            translation_provider="processor-local",
            translation_target_lang="fr",
            use_hw_encode=False,
        ))
        remover._srt_entries = [(0, "Hello"), (1, "Hello"), (20, "World")]
        remover._whisper_segments = []
        remover.last_translation = {"requested": True, "status": "pending"}
        remover._translation_burn_path = ""

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "cleaned.mp4"
            remover._prepare_translation_workflow(
                "source.mp4", str(output), 10.0)
            translated = Path(remover._translation_burn_path)
            translated_cues = read_srt(translated)
            source_exists = (Path(tmpdir) / "cleaned.source.srt").is_file()

        self.assertTrue(source_exists)
        self.assertEqual(remover.last_translation["sourceKind"], "ocr-srt")
        self.assertEqual(remover.last_translation["provider"], "processor-local")
        self.assertEqual(translated_cues[0].text, "fr: Hello")

    def test_provided_translation_bypasses_provider(self):
        from backend.processor import SubtitleRemover

        with tempfile.TemporaryDirectory() as tmpdir:
            translated = Path(tmpdir) / "ready.srt"
            translated.write_text(SRT_TEXT, encoding="utf-8")
            remover = SubtitleRemover.__new__(SubtitleRemover)
            remover.config = normalize_processing_config(ProcessingConfig(
                translation_enabled=True,
                translation_srt=str(translated),
                translation_target_lang="es",
                use_hw_encode=False,
            ))
            remover._srt_entries = []
            remover._whisper_segments = []
            remover.last_translation = {"requested": True, "status": "pending"}
            remover._translation_burn_path = ""

            remover._prepare_translation_workflow(
                "source.mp4", str(Path(tmpdir) / "cleaned.mp4"), 24.0)

        self.assertEqual(remover._translation_burn_path, str(translated))
        self.assertEqual(remover.last_translation["provider"], "provided")
        self.assertEqual(remover.last_translation["cueCount"], 2)

    def test_sidecar_records_translation_evidence(self):
        from backend.batch_report import build_output_sidecar

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.mp4"
            output = Path(tmpdir) / "output.mp4"
            source.write_bytes(b"source")
            output.write_bytes(b"output")
            evidence = {
                "schema": "vsr.subtitle_translation.v1",
                "requested": True,
                "status": "embedded",
                "provider": "provided",
                "sourceKind": "provided-translated-srt",
            }
            sidecar = build_output_sidecar(
                input_path=str(source),
                output_path=str(output),
                config=ProcessingConfig(),
                status="processed",
                translation=evidence,
            )

        self.assertEqual(sidecar["translation"], evidence)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required")
    def test_translated_srt_is_burned_into_byte_valid_video(self):
        from backend.io import validate_video_output
        from backend.processor import SubtitleRemover
        from backend.subprocess_policy import run_process

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "cleaned.mp4"
            translated = root / "ready.srt"
            translated.write_text(SRT_TEXT, encoding="utf-8")
            run_process(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i",
                    "color=c=black:size=320x180:rate=10:duration=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output),
                ],
                timeout=30,
                capture_output=True,
                check=True,
            )
            remover = SubtitleRemover.__new__(SubtitleRemover)
            remover.config = normalize_processing_config(ProcessingConfig(
                translation_enabled=True,
                translation_srt=str(translated),
                preserve_audio=False,
                use_hw_encode=False,
            ))
            remover._translation_burn_path = str(translated)
            remover.last_translation = {
                "requested": True,
                "status": "validated",
            }
            remover._output_contract = None
            remover._hw_encoder = None
            remover._color_metadata = None
            remover.last_container_payload = {}

            remover._run_post_restore_passes(str(output), tmpdir)
            valid, reason, details = validate_video_output(
                output, expected_frames=10)

        self.assertTrue(valid, (reason, details))
        self.assertEqual(remover.last_translation["status"], "embedded")

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required")
    def test_cli_workflow_writes_embedded_translation_sidecar(self):
        from backend.io import validate_video_output
        from backend.subprocess_policy import run_process

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.mp4"
            output = root / "localized.mp4"
            translated = root / "ready.srt"
            translated.write_text(SRT_TEXT, encoding="utf-8")
            run_process(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i",
                    "color=c=black:size=160x90:rate=8:duration=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
                ],
                timeout=30,
                capture_output=True,
                check=True,
            )
            result = run_process(
                [
                    sys.executable, "-m", "backend.processor",
                    "-i", str(source), "-o", str(output),
                    "--translated-srt", str(translated),
                    "--translation-style", "FontSize=18",
                    "--skip-detection", "--no-audio", "--no-hw-encode",
                    "--set", "subtitle_area=[0,0,2,2]",
                ],
                cwd=Path(__file__).resolve().parents[1],
                timeout=60,
                capture_output=True,
                text=True,
                check=False,
            )
            valid, reason, details = validate_video_output(
                output, expected_frames=8)
            sidecar = json.loads(
                Path(str(output) + ".vsr.json").read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        self.assertTrue(valid, (reason, details))
        self.assertEqual(sidecar["translation"]["status"], "embedded")
        self.assertEqual(sidecar["translation"]["provider"], "provided")


class ProvidedTranslationEvidenceTests(unittest.TestCase):
    def test_provided_evidence_uses_name_hash_and_no_absolute_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "translated.srt"
            path.write_text(SRT_TEXT, encoding="utf-8")
            report = provided_translation_evidence(path, target_language="fr")

        self.assertEqual(report["translated"]["name"], "translated.srt")
        self.assertNotIn(tmpdir, json.dumps(report))
        self.assertEqual(len(report["translated"]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
