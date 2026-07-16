"""Tests for the SRT OCR-fix replace-list mechanism."""

from __future__ import annotations

import json

from backend.ocr_fix import apply_ocr_fixes, load_ocr_fix_replacements


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


def test_locale_variant_normalization(tmp_path):
    (tmp_path / "ch.json").write_text(json.dumps({"x": "y"}), encoding="utf-8")
    reps = load_ocr_fix_replacements("ch_sim", base_dir=tmp_path)
    assert reps.get("x") == "y"
