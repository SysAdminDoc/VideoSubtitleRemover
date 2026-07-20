"""Convert Subtitle Edit OCRFixReplaceList XML dictionaries to VSR JSON.

Subtitle Edit (GPL) ships per-language ``{lang}_OCRFixReplaceList.xml`` files
whose ``<WholeWords>`` section is a large list of ``from``/``to`` corrections for
common OCR mis-reads. VSR's OCR-fix engine (``backend/ocr_fix.py``) consumes a
flat ``{"from": "to"}`` JSON map where any key matching ``\\w+`` is applied on
word boundaries -- which is exactly the semantics of Subtitle Edit's WholeWords.

This script reads one or more SE XML files (point it at your own Subtitle Edit
installation's ``Dictionaries`` folder, or at files you have downloaded) and
writes a VSR JSON list. Only the WholeWords section is imported by default:
SE's PartialWords are substring rewrites (e.g. ``rn`` -> ``m``) that VSR's
whole-word engine cannot represent faithfully, so they are skipped unless
``--include-partial`` is passed (which then treats them as whole-word keys).

VSR does NOT bundle Subtitle Edit's data (GPL, incompatible with VSR's MIT
license). The lists shipped under ``backend/ocr_fix_data/`` are clean-room
authored. Use this script to build your own extended lists locally.

Usage:
    py -3.12 scripts/import_se_ocrfix.py --lang eng \
        --input path/to/eng_OCRFixReplaceList.xml \
        --out backend/ocr_fix_data/en.json
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

# Subtitle Edit uses ISO 639-2/3 three-letter codes; VSR normalizes to the
# two-letter base key (see backend/ocr_fix.py:_normalize_lang).
_LANG_ALIASES: Dict[str, str] = {
    "eng": "en", "en": "en",
    "spa": "es", "es": "es",
    "fra": "fr", "fre": "fr", "fr": "fr",
    "deu": "de", "ger": "de", "de": "de",
    "por": "pt", "pt": "pt",
    "ita": "it", "it": "it",
    "nld": "nl", "dut": "nl", "nl": "nl",
    "pol": "pl", "pl": "pl",
    "rus": "ru", "ru": "ru",
}


def normalize_lang(code: str) -> str:
    code = (code or "").strip().lower()
    if code in _LANG_ALIASES:
        return _LANG_ALIASES[code]
    return code.split("_", 1)[0].split("-", 1)[0] or code


def _collect(root: ET.Element, section: str) -> Dict[str, str]:
    """Return {from: to} for every child element under ``section`` with both."""
    out: Dict[str, str] = {}
    node = root.find(section)
    if node is None:
        return out
    for child in list(node):
        src = child.get("from")
        dst = child.get("to")
        if src is None or dst is None:
            continue
        src = src.strip()
        dst = dst.strip()
        if src and dst and src != dst:
            out[src] = dst
    return out


def parse_se_xml(path: Path, *, include_partial: bool = False) -> Dict[str, str]:
    """Parse one SE OCRFixReplaceList XML into a {from: to} map."""
    tree = ET.parse(path)
    root = tree.getroot()
    result: Dict[str, str] = {}
    result.update(_collect(root, "WholeWords"))
    if include_partial:
        # PartialWordsAlways are the least-risky partials (always applied).
        result.update(_collect(root, "PartialWordsAlways"))
    return result


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--input", "-i", nargs="+", required=True,
                    help="one or more SE OCRFixReplaceList XML files")
    ap.add_argument("--lang", "-l", required=True,
                    help="language code (SE 3-letter or 2-letter; normalized)")
    ap.add_argument("--out", "-o", required=True,
                    help="output JSON path (e.g. backend/ocr_fix_data/en.json)")
    ap.add_argument("--include-partial", action="store_true",
                    help="also import PartialWordsAlways as whole-word keys")
    ap.add_argument("--merge", action="store_true",
                    help="merge into an existing output file instead of replacing")
    args = ap.parse_args(argv)

    key = normalize_lang(args.lang)
    merged: Dict[str, str] = {}
    out_path = Path(args.out)
    if args.merge and out_path.is_file():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                merged.update({str(k): str(v) for k, v in existing.items()})
        except (OSError, ValueError) as exc:
            print(f"warning: could not read existing {out_path}: {exc}",
                  file=sys.stderr)

    for raw in args.input:
        p = Path(raw)
        if not p.is_file():
            print(f"error: input not found: {p}", file=sys.stderr)
            return 2
        try:
            merged.update(parse_se_xml(p, include_partial=args.include_partial))
        except ET.ParseError as exc:
            print(f"error: cannot parse {p}: {exc}", file=sys.stderr)
            return 2

    ordered = {k: merged[k] for k in sorted(merged)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(ordered, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {len(ordered)} entries for '{key}' -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
