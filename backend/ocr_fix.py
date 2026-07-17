"""Per-language OCR-fix replace lists for exported SRT text.

Hardsub OCR frequently makes the same substitutions (a capital ``I`` read as a
lowercase ``l``, a pipe ``|`` read as ``I``, ...). Subtitle Edit ships language
``OCRFixReplaceList`` dictionaries for exactly this; VSR offers a light
equivalent that a user can extend without touching code.

A small, deliberately-conservative built-in default set is merged with an
optional user file at ``%APPDATA%/VideoSubtitleRemoverPro/ocr_fix/{lang}.json``
(a flat ``{"from": "to"}`` JSON object). Whole-word keys (``\\w+``) are replaced
only on word boundaries so ``l -> I`` never rewrites ``already``; other keys are
literal substring replacements.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Mapping, Optional

# Conservative, high-confidence defaults only. Whole-word keys are boundary
# matched; anything ambiguous (0<->O, 1<->l inside numbers) is intentionally
# left to user files.
_BUILTIN_REPLACEMENTS: Dict[str, Dict[str, str]] = {
    "en": {
        "l": "I",     # standalone lowercase L is virtually always a mis-read I
        "|": "I",     # pipe mis-read as I
        "ll": "II",   # e.g. roman numerals / initialisms mis-read
    },
}


def _appdata_root(env: Optional[Mapping[str, str]] = None) -> Path:
    source = env if env is not None else os.environ
    root = str(source.get("APPDATA") or "").strip()
    if root:
        return Path(root) / "VideoSubtitleRemoverPro"
    home = Path(str(source.get("USERPROFILE") or source.get("HOME") or Path.home()))
    return home / ".config" / "VideoSubtitleRemoverPro"


def ocr_fix_dir(env: Optional[Mapping[str, str]] = None) -> Path:
    return _appdata_root(env) / "ocr_fix"


def _normalize_lang(lang: Optional[str]) -> str:
    lang = (lang or "en").strip().lower()
    # Map detector locale variants onto a base list key.
    if lang in {"ch", "ch_sim", "zh", "zh-cn", "chinese"}:
        return "ch"
    return lang.split("_", 1)[0].split("-", 1)[0] or "en"


def load_ocr_fix_replacements(
    lang: Optional[str],
    *,
    env: Optional[Mapping[str, str]] = None,
    base_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """Return the merged replace map for ``lang`` (built-in then user file)."""
    key = _normalize_lang(lang)
    result: Dict[str, str] = dict(_BUILTIN_REPLACEMENTS.get(key, {}))
    directory = base_dir if base_dir is not None else ocr_fix_dir(env)
    path = directory / f"{key}.json"
    try:
        if path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for src, dst in raw.items():
                    result[str(src)] = str(dst)
    except (OSError, ValueError):
        # A malformed user file must never break SRT export.
        pass
    return result


def apply_ocr_fixes(text: str, replacements: Mapping[str, str]) -> str:
    """Apply the replace map to ``text``.

    Whole-word (``\\w+``) keys are replaced on word boundaries; every other key
    is a literal substring replacement. Keys are applied longest-first (then
    alphabetically) so a shorter, more general key cannot pre-empt a longer,
    more specific one, and the result stays deterministic regardless of the
    source dict's insertion order.
    """
    if not text or not replacements:
        return text
    for src, dst in sorted(replacements.items(), key=lambda kv: (-len(kv[0]), kv[0])):
        if not src:
            continue
        if re.fullmatch(r"\w+", src, flags=re.UNICODE):
            text = re.sub(rf"(?<!\w){re.escape(src)}(?!\w)", dst, text)
        else:
            text = text.replace(src, dst)
    return text
