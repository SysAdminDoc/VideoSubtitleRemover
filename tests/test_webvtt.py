"""RM-154: loss-aware WebVTT translation and re-embed interchange."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend import webvtt as W
from backend.subtitle_translation import (
    SubtitleTranslationError,
    provided_translation_evidence,
    read_subtitles,
    register_translation_provider,
    subtitle_format,
    translate_srt_file,
    translated_srt_path,
)


# A document that exercises every structural feature the item names:
# NOTE, STYLE, REGION, cue identifiers, positioning settings, vertical
# writing mode, voice/class/lang spans, ruby, and karaoke timestamp tags.
FULL_VTT = (
    "WEBVTT - Sample with everything\n"
    "\n"
    "NOTE\n"
    "Translator comments live here and must survive untouched.\n"
    "\n"
    "STYLE\n"
    "::cue(.narrator) {\n"
    "  color: yellow;\n"
    "}\n"
    "\n"
    "REGION\n"
    "id:top\n"
    "width:40%\n"
    "lines:3\n"
    "regionanchor:0%,100%\n"
    "viewportanchor:10%,90%\n"
    "scroll:up\n"
    "\n"
    "intro\n"
    "00:00:01.000 --> 00:00:04.000 region:top align:start line:0\n"
    "<v Narrator><c.narrator>Once upon a time</c></v>\n"
    "\n"
    "2\n"
    "00:00:05.000 --> 00:00:07.500 vertical:rl position:90%\n"
    "<ruby>\u6f22\u5b57<rt>kanji</rt></ruby> on screen\n"
    "\n"
    "00:00:08.000 --> 00:00:10.000\n"
    "<lang ja>Plain &amp; simple</lang> text with <b>bold</b>\n"
    "\n"
    "karaoke\n"
    "00:00:11.000 --> 00:00:14.000\n"
    "<00:00:11.500>First <00:00:12.500>second\n"
)

MINIMAL_VTT = (
    "WEBVTT\n"
    "\n"
    "00:00:01.000 --> 00:00:02.000\n"
    "Hello\n"
)


class TimestampTests(unittest.TestCase):
    def test_hours_are_optional_per_the_webvtt_grammar(self):
        self.assertEqual(W.parse_timestamp("00:01.500"), 1.5)
        self.assertEqual(W.parse_timestamp("01:02:03.004"), 3723.004)
        # Real files mix in SRT's comma; accept it on input.
        self.assertEqual(W.parse_timestamp("00:00:01,250"), 1.25)

    def test_output_always_uses_the_webvtt_period(self):
        self.assertEqual(W.format_timestamp(3723.004), "01:02:03.004")
        self.assertEqual(W.format_timestamp(0), "00:00:00.000")

    def test_malformed_timestamps_are_rejected(self):
        for value in ("", "1:2:3.4", "00:60:00.000", "00:00:60.000",
                      "00:00:01.1", "abc", "00:00:01"):
            with self.subTest(value=value):
                with self.assertRaises(W.WebVttError):
                    W.parse_timestamp(value)


class ParseAndRoundTripTests(unittest.TestCase):
    def test_a_full_document_round_trips_byte_for_byte(self):
        # The strongest guarantee available: parse then render must not
        # perturb a single character of a document using every feature.
        document = W.parse_vtt(FULL_VTT)
        self.assertEqual(W.render_vtt(document), FULL_VTT)

    def test_a_minimal_document_round_trips(self):
        self.assertEqual(W.render_vtt(W.parse_vtt(MINIMAL_VTT)), MINIMAL_VTT)

    def test_structure_is_captured_not_flattened(self):
        document = W.parse_vtt(FULL_VTT)
        self.assertEqual(document.header, "WEBVTT - Sample with everything")
        self.assertEqual(
            [block.kind for block in document.blocks],
            ["note", "style", "region"],
        )
        self.assertEqual(len(document.cues), 4)
        first = document.cues[0]
        self.assertEqual(first.identifier, "intro")
        self.assertEqual(first.settings, "region:top align:start line:0")
        self.assertEqual(first.start, 1.0)
        self.assertEqual(first.end, 4.0)
        # A cue with no identifier line keeps an empty one rather than
        # inventing a number that would then be written out.
        self.assertEqual(document.cues[2].identifier, "")
        self.assertEqual(document.cues[1].settings, "vertical:rl position:90%")

    def test_block_order_is_preserved_even_when_interleaved(self):
        source = (
            "WEBVTT\n\n"
            "00:00:01.000 --> 00:00:02.000\nfirst\n\n"
            "NOTE mid-document comment\n\n"
            "00:00:03.000 --> 00:00:04.000\nsecond\n"
        )
        document = W.parse_vtt(source)
        self.assertEqual(
            document.order, (("cue", 0), ("block", 0), ("cue", 1)))
        self.assertEqual(W.render_vtt(document), source)

    def test_crlf_and_bom_input_normalize_without_losing_cues(self):
        source = "\ufeff" + MINIMAL_VTT.replace("\n", "\r\n")
        document = W.parse_vtt(source)
        self.assertEqual(len(document.cues), 1)
        self.assertEqual(W.render_vtt(document), MINIMAL_VTT)

    def test_bad_documents_fail_closed(self):
        cases = {
            "no signature": "00:00:01.000 --> 00:00:02.000\nhi\n",
            "empty": "   \n",
            "nul byte": "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n\x00\n",
            "no cues": "WEBVTT\n\nNOTE only a comment\n",
            "reversed timing": (
                "WEBVTT\n\n00:00:05.000 --> 00:00:02.000\nhi\n"),
            "equal timing": (
                "WEBVTT\n\n00:00:02.000 --> 00:00:02.000\nhi\n"),
            "empty payload": "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n\n",
            "bad setting": (
                "WEBVTT\n\n00:00:01.000 --> 00:00:02.000 not a setting\nhi\n"),
            "unknown block": "WEBVTT\n\nGARBAGE\nmore garbage\n",
            "missing arrow": "WEBVTT\n\n00:00:01.000 00:00:02.000\nhi\n",
        }
        for name, source in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(W.WebVttError):
                    W.parse_vtt(source)

    def test_a_webvtt_signature_with_a_trailing_word_is_accepted(self):
        # "WEBVTT" may be followed by a space/tab and free text, but
        # "WEBVTTX" is a different (invalid) token.
        self.assertTrue(W.parse_vtt("WEBVTT - x\n\n" + MINIMAL_VTT.split("\n\n", 1)[1]))
        with self.assertRaises(W.WebVttError):
            W.parse_vtt("WEBVTTX\n\n00:00:01.000 --> 00:00:02.000\nhi\n")

    def test_unknown_cue_settings_are_preserved_not_rejected(self):
        # A setting this build has never heard of is still the author's
        # data; dropping it would be the silent flattening this avoids.
        source = (
            "WEBVTT\n\n00:00:01.000 --> 00:00:02.000 futuresetting:7\nhi\n")
        document = W.parse_vtt(source)
        self.assertEqual(document.cues[0].settings, "futuresetting:7")
        self.assertEqual(W.render_vtt(document), source)

    def test_reading_a_file_enforces_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.vtt"
            path.write_text(MINIMAL_VTT, encoding="utf-8")
            self.assertEqual(len(W.read_vtt(path).cues), 1)
            path.write_bytes(b"")
            with self.assertRaises(W.WebVttError):
                W.read_vtt(path)
            with self.assertRaises(W.WebVttError):
                W.read_vtt(Path(tmp) / "missing.vtt")
            path.write_bytes(b"WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n\xff\xfe\n")
            with self.assertRaises(W.WebVttError):
                W.read_vtt(path)


class EscapeTests(unittest.TestCase):
    def test_entities_decode_and_reencode(self):
        self.assertEqual(W.unescape("a &amp; b &lt;c&gt;"), "a & b <c>")
        self.assertEqual(W.escape("a & b <c>"), "a &amp; b &lt;c&gt;")

    def test_reencoding_cannot_double_encode_an_ampersand(self):
        self.assertEqual(W.escape(W.unescape("&amp;lt;")), "&amp;lt;")

    def test_nbsp_and_bidi_marks_decode(self):
        self.assertEqual(W.unescape("a&nbsp;b"), "a\u00a0b")
        self.assertIn("\u200e", W.unescape("&lrm;"))
        self.assertIn("\u200f", W.unescape("&rlm;"))


class VisibleRunTests(unittest.TestCase):
    def test_only_visible_text_is_offered_for_translation(self):
        document = W.parse_vtt(FULL_VTT)
        self.assertEqual(
            W.document_runs(document),
            [
                "Once upon a time",
                "\u6f22\u5b57",
                " on screen",
                "Plain & simple",
                " text with ",
                "bold",
                "First ",
                "second",
            ],
        )

    def test_ruby_annotations_are_excluded_on_purpose(self):
        # An <rt> annotation is a reading aid for the source script;
        # translating it produces a phonetic guide to a language the
        # reader is no longer looking at.
        runs = W.translatable_runs(
            "<ruby>\u6f22\u5b57<rt>kanji</rt></ruby>")
        self.assertEqual(runs, ["\u6f22\u5b57"])

    def test_translating_rewrites_text_and_nothing_else(self):
        document = W.parse_vtt(FULL_VTT)
        runs = W.document_runs(document)
        translated = W.apply_document_runs(document, [r.upper() for r in runs])

        # Structure is byte-identical apart from the payload text.
        self.assertEqual(translated.header, document.header)
        self.assertEqual(
            [b.text for b in translated.blocks],
            [b.text for b in document.blocks],
        )
        for before, after in zip(document.cues, translated.cues):
            self.assertEqual(before.identifier, after.identifier)
            self.assertEqual(before.timing_text, after.timing_text)
            self.assertEqual(before.settings, after.settings)
            self.assertEqual((before.start, before.end), (after.start, after.end))

        payloads = [cue.payload for cue in translated.cues]
        self.assertEqual(
            payloads[0], "<v Narrator><c.narrator>ONCE UPON A TIME</c></v>")
        # Ruby annotation untranslated, base translated.
        self.assertIn("<rt>kanji</rt>", payloads[1])
        # The entity is re-escaped, not emitted raw.
        self.assertIn("PLAIN &amp; SIMPLE", payloads[2])
        self.assertNotIn("PLAIN & SIMPLE", payloads[2])
        # Karaoke timestamp tags stay put.
        self.assertEqual(payloads[3], "<00:00:11.500>FIRST <00:00:12.500>SECOND")

    def test_surrounding_whitespace_keeps_markup_welded_to_its_words(self):
        payload = "one <b>two</b> three"
        rebuilt = W.apply_translated_runs(payload, ["EIN", "ZWEI", "DREI"])
        self.assertEqual(rebuilt, "EIN <b>ZWEI</b> DREI")

    def test_a_wrong_run_count_fails_closed(self):
        document = W.parse_vtt(FULL_VTT)
        for count in (0, 1, 99):
            with self.subTest(count=count):
                with self.assertRaises(W.WebVttError):
                    W.apply_document_runs(document, ["x"] * count)

    def test_empty_or_nul_translations_are_refused(self):
        for bad in ("", "   ", "a\x00b"):
            with self.subTest(value=bad):
                with self.assertRaises(W.WebVttError):
                    W.apply_translated_runs("hello", [bad])

    def test_a_provider_cannot_inject_markup_that_breaks_structure(self):
        # Provider output is escaped, so returned angle brackets become
        # literal text instead of new tags.
        rebuilt = W.apply_translated_runs("hi", ["<script>evil</script>"])
        self.assertNotIn("<script>", rebuilt)
        self.assertIn("&lt;script&gt;", rebuilt)
        # The result is still a parsable document.
        document = W.parse_vtt(
            f"WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n{rebuilt}\n")
        self.assertEqual(len(document.cues), 1)

    def test_an_oversized_translation_is_refused(self):
        with self.assertRaises(W.WebVttError):
            W.apply_translated_runs("hi", ["x" * (W.MAX_CUE_TEXT + 1)])


class LossReportTests(unittest.TestCase):
    def test_a_webvtt_to_webvtt_translation_reports_itself_lossless(self):
        # "No report" and "nothing was lost" must not look the same.
        report = W.loss_report(W.parse_vtt(FULL_VTT), target_format="vtt")
        self.assertEqual(report["schema"], W.WEBVTT_SCHEMA)
        self.assertTrue(report["lossless"])
        self.assertEqual(report["losses"], [])
        self.assertEqual(report["cueCount"], 4)
        self.assertEqual(report["blockCount"], 3)
        self.assertTrue(report["rubyAnnotationsPreservedUntranslated"])

    def test_flattening_to_srt_enumerates_every_dropped_feature(self):
        report = W.loss_report(W.parse_vtt(FULL_VTT), target_format="srt")
        self.assertFalse(report["lossless"])
        features = {item["feature"]: item for item in report["losses"]}
        for expected in (
            "region", "style", "note", "cue_settings", "cue_identifier",
            "voice_span", "language_span", "class_span", "ruby",
            "timestamp_tag",
        ):
            with self.subTest(feature=expected):
                self.assertIn(expected, features)
                self.assertGreater(features[expected]["count"], 0)
                self.assertTrue(features[expected]["detail"].strip())

    def test_a_plain_document_loses_nothing_going_to_srt(self):
        report = W.loss_report(W.parse_vtt(MINIMAL_VTT), target_format="srt")
        self.assertTrue(report["lossless"])
        self.assertEqual(report["losses"], [])

    def test_an_unsupported_target_format_is_refused(self):
        for fmt in ("ttml", "ass", "dfxp", ""):
            with self.subTest(fmt=fmt):
                with self.assertRaises(W.WebVttError):
                    W.loss_report(W.parse_vtt(MINIMAL_VTT), target_format=fmt)

    def test_counts_are_accurate_not_merely_nonzero(self):
        report = W.loss_report(W.parse_vtt(FULL_VTT), target_format="srt")
        counts = report["features"]
        self.assertEqual(counts["region"], 1)
        self.assertEqual(counts["style"], 1)
        self.assertEqual(counts["note"], 1)
        self.assertEqual(counts["cue_settings"], 2)
        self.assertEqual(counts["cue_identifier"], 3)
        self.assertEqual(counts["voice_span"], 1)
        self.assertEqual(counts["timestamp_tag"], 2)


class SrtInteropTests(unittest.TestCase):
    def test_flattening_keeps_ruby_annotations_legible(self):
        text = W.to_srt_text(W.parse_vtt(FULL_VTT))
        # Welding the annotation onto its base would corrupt the word.
        self.assertIn("\u6f22\u5b57 (kanji) on screen", text)
        self.assertNotIn("\u6f22\u5b57kanji", text)

    def test_flattening_uses_srt_comma_timestamps_and_numbering(self):
        text = W.to_srt_text(W.parse_vtt(FULL_VTT))
        self.assertIn("1\n00:00:01,000 --> 00:00:04,000\n", text)
        self.assertIn("4\n00:00:11,000 --> 00:00:14,000\n", text)
        self.assertNotIn("-->" + " " + "00:00:04.000", text)

    def test_srt_cues_convert_into_a_webvtt_document(self):
        from backend.subtitle_translation import parse_srt

        cues = parse_srt(
            "1\n00:00:01,000 --> 00:00:02,000\nHello <world>\n")
        document = W.from_srt_cues(cues)
        self.assertEqual(len(document.cues), 1)
        # Angle brackets from SRT text must be escaped, not become tags.
        self.assertEqual(document.cues[0].payload, "Hello &lt;world&gt;")
        self.assertIn("WEBVTT", W.render_vtt(document))
        self.assertEqual(W.document_runs(document), ["Hello <world>"])

    def test_converting_no_cues_fails_closed(self):
        with self.assertRaises(W.WebVttError):
            W.from_srt_cues([])

    def test_path_detection(self):
        self.assertTrue(W.is_webvtt_path("a.VTT"))
        self.assertFalse(W.is_webvtt_path("a.srt"))


class TranslationPipelineTests(unittest.TestCase):
    """The seam that decides SRT vs WebVTT, end to end."""

    @classmethod
    def setUpClass(cls):
        def upper_provider(texts, _source, _target, _options):
            return [str(text).upper() for text in texts]

        register_translation_provider(
            "rm154-upper", upper_provider, replace=True)
        cls.provider = "rm154-upper"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_format_detection_claims_only_srt_and_vtt(self):
        self.assertEqual(subtitle_format("a.vtt"), "vtt")
        self.assertEqual(subtitle_format("a.VTT"), "vtt")
        self.assertEqual(subtitle_format("a.srt"), "srt")
        self.assertEqual(subtitle_format("a.txt"), "srt")

    def test_rejection_rationale_names_current_imsc_recommendation(self):
        import backend.subtitle_translation as translation

        self.assertIn("21 May 2026", translation.subtitle_format.__doc__ or "")
        self.assertIn("21 May 2026", W.__doc__ or "")
        self.assertIn(
            "w3.org/TR/ttml-imsc1.3",
            translation.subtitle_format.__doc__ or "",
        )
        self.assertIn("w3.org/TR/ttml-imsc1.3", W.__doc__ or "")

    def test_ttml_and_imsc_are_refused_loudly(self):
        # The item scopes TTML/IMSC out. Silently parsing one as SRT is
        # exactly the flattening this work exists to prevent.
        for name in ("a.ttml", "a.dfxp", "a.itt", "a.xml"):
            with self.subTest(name=name):
                with self.assertRaises(SubtitleTranslationError) as ctx:
                    subtitle_format(name)
                self.assertIn("TTML", str(ctx.exception))

    def test_translating_a_vtt_source_preserves_the_document(self):
        source = self.root / "in.vtt"
        source.write_text(FULL_VTT, encoding="utf-8")
        destination = self.root / "out.vtt"
        report = translate_srt_file(
            source, destination,
            provider_name=self.provider,
            source_language="ja",
            target_language="en",
        )
        self.assertEqual(report["sourceFormat"], "vtt")
        self.assertEqual(report["targetFormat"], "vtt")
        self.assertEqual(report["cueCount"], 4)
        self.assertEqual(report["runCount"], 8)
        self.assertTrue(report["loss"]["lossless"])
        self.assertTrue(report["source"]["sha256"])
        self.assertNotEqual(
            report["source"]["sha256"], report["translated"]["sha256"])

        written = destination.read_text(encoding="utf-8")
        rewritten = W.parse_vtt(written)
        original = W.parse_vtt(FULL_VTT)
        self.assertEqual(rewritten.header, original.header)
        self.assertEqual(
            [b.text for b in rewritten.blocks],
            [b.text for b in original.blocks],
        )
        self.assertEqual(
            [c.settings for c in rewritten.cues],
            [c.settings for c in original.cues],
        )
        self.assertIn("ONCE UPON A TIME", written)
        self.assertIn("<rt>kanji</rt>", written)
        self.assertIn("region:top align:start line:0", written)
        self.assertIn("vertical:rl position:90%", written)

    def test_translating_an_srt_source_still_reports_srt(self):
        source = self.root / "in.srt"
        source.write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nhello\n", encoding="utf-8")
        report = translate_srt_file(
            source, self.root / "out.srt",
            provider_name=self.provider,
            source_language="en",
            target_language="de",
        )
        self.assertEqual(report["sourceFormat"], "srt")
        self.assertEqual(report["targetFormat"], "srt")
        self.assertEqual(report["cueCount"], 1)

    def test_a_provider_returning_the_wrong_run_count_fails_closed(self):
        def short_provider(texts, _source, _target, _options):
            return [str(text).upper() for text in texts][:-1]

        register_translation_provider(
            "rm154-short", short_provider, replace=True)
        source = self.root / "in.vtt"
        source.write_text(FULL_VTT, encoding="utf-8")
        destination = self.root / "out.vtt"
        with self.assertRaises(SubtitleTranslationError) as ctx:
            translate_srt_file(
                source, destination,
                provider_name="rm154-short",
                source_language="ja",
                target_language="en",
            )
        self.assertIn("unusable", str(ctx.exception))
        self.assertFalse(destination.exists())

    def test_the_translated_path_keeps_the_source_extension(self):
        self.assertEqual(
            translated_srt_path("clip.mp4", "de").name, "clip.de.srt")
        self.assertEqual(
            translated_srt_path("clip.mp4", "de", suffix=".vtt").name,
            "clip.de.vtt",
        )
        # A bare extension is accepted too.
        self.assertEqual(
            translated_srt_path("clip.mp4", "de", suffix="vtt").name,
            "clip.de.vtt",
        )

    def test_a_provided_vtt_is_validated_and_reported_as_vtt(self):
        source = self.root / "given.vtt"
        source.write_text(FULL_VTT, encoding="utf-8")
        report = provided_translation_evidence(source, target_language="en")
        self.assertEqual(report["sourceFormat"], "vtt")
        self.assertEqual(report["sourceKind"], "provided-translated-vtt")
        self.assertEqual(report["cueCount"], 4)
        self.assertTrue(report["loss"]["lossless"])

    def test_a_provided_broken_vtt_is_rejected(self):
        source = self.root / "broken.vtt"
        source.write_text("not a vtt at all\n", encoding="utf-8")
        with self.assertRaises(SubtitleTranslationError):
            provided_translation_evidence(source)

    def test_read_subtitles_dispatches_on_extension(self):
        vtt = self.root / "a.vtt"
        vtt.write_text(MINIMAL_VTT, encoding="utf-8")
        srt = self.root / "a.srt"
        srt.write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nhi\n", encoding="utf-8")
        self.assertIsInstance(read_subtitles(vtt), W.VttDocument)
        self.assertIsInstance(read_subtitles(srt), list)

    def test_a_vtt_with_no_translatable_text_is_refused(self):
        source = self.root / "ruby_only.vtt"
        source.write_text(
            "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n"
            "<ruby><rt>kanji</rt></ruby>\n",
            encoding="utf-8",
        )
        with self.assertRaises(SubtitleTranslationError) as ctx:
            translate_srt_file(
                source, self.root / "out.vtt",
                provider_name=self.provider,
                source_language="ja",
                target_language="en",
            )
        self.assertIn("no translatable", str(ctx.exception))

    def test_config_validation_accepts_a_vtt_source(self):
        from backend.subtitle_translation import validate_translation_config

        source = self.root / "src.vtt"
        source.write_text(FULL_VTT, encoding="utf-8")

        class Config:
            translation_enabled = True
            restyle_subtitle = ""
            translation_srt = ""
            translation_target_lang = "en"
            translation_provider = "rm154-upper"
            translation_command = ""
            translation_source_srt = str(source)

        validate_translation_config(Config())  # must not raise

        Config.translation_source_srt = str(self.root / "missing.vtt")
        with self.assertRaises(SubtitleTranslationError):
            validate_translation_config(Config())


class ContainerReembedTests(unittest.TestCase):
    @staticmethod
    def _decide(suffix: str, codec: str = "webvtt"):
        from backend.container_payload import _stream_action

        return _stream_action(
            {"codecType": "subtitle", "codecName": codec},
            suffix,
            preserve_audio=True,
            multi_audio=False,
            audio_number=0,
            loudnorm_target=0.0,
            include_auxiliary=True,
            force_audio_transcode=False,
            primary_video_index=0,
        )

    def test_matroska_and_webm_copy_webvtt_but_mp4_names_its_loss(self):
        self.assertEqual(self._decide(".mkv")[0], "copy")
        self.assertEqual(self._decide(".webm")[0], "copy")

        action, codec, reason = self._decide(".mp4")
        self.assertEqual((action, codec), ("transcode", "mov_text"))
        # The downgrade must say what it costs, not just that it happened.
        for token in ("regions", "cue", "STYLE"):
            with self.subTest(token=token):
                self.assertIn(token, reason)

    def test_an_srt_source_still_gets_the_generic_mp4_conversion_note(self):
        # Only the WebVTT downgrade needs the extra detail; SRT never had
        # regions or cue settings to lose.
        action, codec, reason = self._decide(".mp4", codec="subrip")
        self.assertEqual((action, codec), ("transcode", "mov_text"))
        self.assertNotIn("regions", reason)


class ParseLimitTests(unittest.TestCase):
    """The parse-side DoS bounds had no coverage: only read_vtt's byte cap
    and the translated-payload cap were tested, so MAX_CUES, MAX_CUE_TEXT
    and MAX_BLOCKS could be deleted without a red test. The module
    constants are patched down so each case stays fast."""

    def _vtt(self, cues):
        parts = ["WEBVTT", ""]
        for index, text in enumerate(cues):
            parts.append("00:00:%02d.000 --> 00:00:%02d.500" % (index, index))
            parts.append(text)
            parts.append("")
        return chr(10).join(parts)

    def test_cue_count_limit_is_enforced(self):
        text = self._vtt(["one", "two", "three"])
        with mock.patch.object(W, "MAX_CUES", 2):
            with self.assertRaises(W.WebVttError) as ctx:
                W.parse_vtt(text)
        self.assertIn("cue count", str(ctx.exception))
        # At the limit it still parses, so the bound is off-by-one safe.
        with mock.patch.object(W, "MAX_CUES", 3):
            self.assertEqual(len(W.parse_vtt(text).cues), 3)

    def test_cue_text_length_limit_is_enforced(self):
        text = self._vtt(["x" * 40])
        with mock.patch.object(W, "MAX_CUE_TEXT", 10):
            with self.assertRaises(W.WebVttError) as ctx:
                W.parse_vtt(text)
        self.assertIn("cue text", str(ctx.exception))
        with mock.patch.object(W, "MAX_CUE_TEXT", 40):
            self.assertEqual(len(W.parse_vtt(text).cues), 1)

    def test_block_count_limit_is_enforced(self):
        text = chr(10).join([
            "WEBVTT",
            "",
            "NOTE first",
            "",
            "NOTE second",
            "",
            "NOTE third",
            "",
            "00:00:00.000 --> 00:00:01.000",
            "hello",
            "",
        ])
        with mock.patch.object(W, "MAX_BLOCKS", 2):
            with self.assertRaises(W.WebVttError) as ctx:
                W.parse_vtt(text)
        self.assertIn("block count", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
