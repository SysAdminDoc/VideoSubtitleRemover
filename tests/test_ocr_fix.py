"""Tests for the SRT OCR-fix replace-list mechanism."""

from __future__ import annotations

import json
from pathlib import Path

from backend.ocr_fix import (
    _BUNDLED_DIR,
    apply_ocr_fixes,
    load_ocr_fix_replacements,
)


def test_whole_word_replacement_only_on_boundaries():
    reps = {"l": "I"}
    # Standalone mis-read L becomes I; inside a word it is left alone.
    assert apply_ocr_fixes("l am here", reps) == "I am here"
    assert apply_ocr_fixes("already", reps) == "already"
    assert apply_ocr_fixes("well, l think", reps) == "well, I think"


def test_literal_replacement_for_non_word_keys():
    assert apply_ocr_fixes("|t was |", {"|": "I"}) == "It was I"


def test_empty_inputs_are_safe():
    assert apply_ocr_fixes("", {"l": "I"}) == ""
    assert apply_ocr_fixes("hello", {}) == "hello"


def test_builtin_english_defaults_present():
    reps = load_ocr_fix_replacements("en")
    assert reps.get("l") == "I"
    assert reps.get("|") == "I"


def test_user_file_overrides_and_extends(tmp_path):
    (tmp_path / "en.json").write_text(
        json.dumps({"l": "1", "cornpany": "company"}), encoding="utf-8")
    reps = load_ocr_fix_replacements("en", base_dir=tmp_path)
    # User file overrides the built-in value...
    assert reps["l"] == "1"
    # ...and adds new entries.
    assert reps["cornpany"] == "company"
    assert apply_ocr_fixes("the cornpany", reps) == "the company"


def test_malformed_user_file_falls_back_to_builtin(tmp_path):
    (tmp_path / "en.json").write_text("{ not valid json", encoding="utf-8")
    reps = load_ocr_fix_replacements("en", base_dir=tmp_path)
    assert reps.get("l") == "I"


def test_longer_keys_apply_before_shorter_overlapping_keys():
    # Insertion order puts the short, general key first; the longer, more
    # specific key must still win so "rnm" is not pre-empted by "rn".
    reps = {"rn": "m", "rnm": "X"}
    assert apply_ocr_fixes("rnm", reps) == "X"
    # Determinism does not depend on dict order.
    assert apply_ocr_fixes("rnm", {"rnm": "X", "rn": "m"}) == "X"


def test_locale_variant_normalization(tmp_path):
    (tmp_path / "ch.json").write_text(json.dumps({"x": "y"}), encoding="utf-8")
    reps = load_ocr_fix_replacements("ch_sim", base_dir=tmp_path)
    assert reps.get("x") == "y"


def test_bundled_lists_ship_for_five_languages():
    # The clean-room bundled lists satisfy the "at least 5 languages" goal and
    # load through the same entry point as user files.
    langs = ["en", "es", "fr", "de", "pt"]
    for lang in langs:
        assert (_BUNDLED_DIR / f"{lang}.json").is_file(), lang
        reps = load_ocr_fix_replacements(lang)
        assert reps, f"{lang} bundled list is empty"


def test_bundled_english_repairs_h_as_b_artifacts():
    reps = load_ocr_fix_replacements("en")
    assert apply_ocr_fixes("tbe otber montb", reps) == "the other month"
    # Valid words that merely resemble a source key are untouched because every
    # bundled source key is a non-word applied on boundaries.
    assert apply_ocr_fixes("breathe together", reps) == "breathe together"


def test_bundled_sources_never_corrupt_valid_words():
    # Safety invariant: no bundled source key is itself a valid target word, so a
    # whole-word replacement can only repair a garbled token, never damage a
    # correct one. Verify no source maps to another entry's target verbatim.
    for lang in ["en", "es", "fr", "de", "pt"]:
        data = json.loads((_BUNDLED_DIR / f"{lang}.json").read_text(encoding="utf-8"))
        targets = set(data.values())
        for src in data:
            assert src not in targets, f"{lang}: source {src!r} collides with a target"


def test_user_file_overrides_bundled_list(tmp_path):
    (tmp_path / "en.json").write_text(json.dumps({"tbe": "TBE"}), encoding="utf-8")
    reps = load_ocr_fix_replacements("en", base_dir=tmp_path)
    # User layer wins over the bundled layer.
    assert reps["tbe"] == "TBE"


def test_import_script_converts_wholewords(tmp_path):
    import scripts.import_se_ocrfix as imp

    xml = tmp_path / "eng_OCRFixReplaceList.xml"
    xml.write_text(
        "<OCRFixReplaceList><WholeWords>"
        '<WholeWord from="tbe" to="the" />'
        '<WholeWord from="same" to="same" />'
        "</WholeWords><PartialWordsAlways>"
        '<Partial from="rn" to="m" />'
        "</PartialWordsAlways></OCRFixReplaceList>",
        encoding="utf-8",
    )
    out = tmp_path / "en.json"
    assert imp.main(["--lang", "eng", "--input", str(xml), "--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    # WholeWords imported; no-op (same->same) and partials dropped by default.
    assert data == {"tbe": "the"}
    # Partial import is opt-in.
    assert imp.parse_se_xml(Path(xml), include_partial=True) == {"tbe": "the", "rn": "m"}


def test_import_script_lang_normalization():
    import scripts.import_se_ocrfix as imp

    assert imp.normalize_lang("eng") == "en"
    assert imp.normalize_lang("deu") == "de"
    assert imp.normalize_lang("pt_BR") == "pt"
