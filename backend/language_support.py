"""Language support facts shared by diagnostics and the GUI."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Tuple


CURATED_LANGUAGE_NAMES: Tuple[Tuple[str, str], ...] = (
    ("en", "English"),
    ("ch", "Chinese"),
    ("japan", "Japanese"),
    ("ja", "Japanese"),
    ("manga", "Manga / Anime (vertical JP via manga-ocr)"),
    ("ko", "Korean"),
    ("korean", "Korean"),
    ("fr", "French"),
    ("french", "French"),
    ("de", "German"),
    ("german", "German"),
    ("es", "Spanish"),
    ("spanish", "Spanish"),
    ("pt", "Portuguese"),
    ("portuguese", "Portuguese"),
    ("ru", "Russian"),
    ("ar", "Arabic"),
    ("arabic", "Arabic"),
    ("hi", "Hindi"),
    ("it", "Italian"),
    ("italian", "Italian"),
    ("nl", "Dutch"),
    ("pl", "Polish"),
    ("tr", "Turkish"),
    ("vi", "Vietnamese"),
    ("th", "Thai"),
    ("uk", "Ukrainian"),
    ("sv", "Swedish"),
    ("no", "Norwegian"),
    ("da", "Danish"),
    ("fi", "Finnish"),
    ("cs", "Czech"),
    ("hu", "Hungarian"),
    ("ro", "Romanian"),
    ("el", "Greek"),
    ("he", "Hebrew"),
    ("id", "Indonesian"),
    ("ms", "Malay"),
    ("fil", "Filipino"),
)

ENGINE_COMPATIBLE_LANGUAGE_CODES: Tuple[str, ...] = (
    "en", "ch", "chinese_cht", "japan", "korean", "ka",
    "fr", "german", "it", "es", "pt", "ru", "ar", "hi",
    "nl", "no", "pl", "tr", "th", "vi", "uk", "be",
    "bg", "hr", "cs", "da", "et", "fi", "hu", "is",
    "lv", "lt", "mt", "ro", "sk", "sl", "sv", "id", "ms",
    "fa", "he", "el",
)

ENGINE_LANGUAGE_CAPACITY: Tuple[Mapping[str, str], ...] = (
    {
        "engine": "RapidOCR",
        "capacity": "100+",
        "detail": "PP-OCR model family language coverage depends on installed assets.",
    },
    {
        "engine": "PaddleOCR",
        "capacity": "106",
        "detail": "PaddleOCR PP-OCR language table coverage.",
    },
    {
        "engine": "Surya",
        "capacity": "90+",
        "detail": "Opt-in GPL OCR path; not bundled by default.",
    },
    {
        "engine": "EasyOCR",
        "capacity": "80+",
        "detail": "Legacy fallback OCR path.",
    },
)


def engine_supported_languages() -> List[str]:
    return list(ENGINE_COMPATIBLE_LANGUAGE_CODES)


def build_language_list() -> List[Tuple[str, str]]:
    pretty: Dict[str, str] = {}
    for code, name in CURATED_LANGUAGE_NAMES:
        pretty.setdefault(code, name)
    out: List[Tuple[str, str]] = []
    seen: set[str] = set()
    out.append(("en", "English"))
    seen.add("en")
    for code, name in CURATED_LANGUAGE_NAMES:
        if code in seen:
            continue
        seen.add(code)
        out.append((code, name))
    for code in engine_supported_languages():
        if code in seen:
            continue
        seen.add(code)
        out.append((code, pretty.get(code, code.upper())))
    return out


def language_support_status(
    detection_items: Iterable[Mapping[str, object]] = (),
) -> dict:
    """Return GUI picker scope and installed OCR engine capacity facts."""
    installed = {
        str(item.get("name") or "")
        for item in detection_items
        if bool(item.get("available") or item.get("installed"))
    }
    capacity_rows = []
    installed_broad = []
    for item in ENGINE_LANGUAGE_CAPACITY:
        name = str(item["engine"])
        installed_here = name in installed
        row = {
            "engine": name,
            "capacity": str(item["capacity"]),
            "detail": str(item["detail"]),
            "installed": installed_here,
        }
        capacity_rows.append(row)
        if installed_here:
            installed_broad.append(f"{name} {row['capacity']}")

    gui_count = len(build_language_list())
    engine_code_count = len(set(ENGINE_COMPATIBLE_LANGUAGE_CODES))
    if installed_broad:
        engine_text = "installed OCR capacity: " + ", ".join(installed_broad)
    else:
        engine_text = (
            "no broad OCR engine installed; OpenCV fallback is "
            "language-agnostic thresholding"
        )
    return {
        "schema": "vsr.language_support.v1",
        "gui_selectable_count": gui_count,
        "curated_entry_count": len(CURATED_LANGUAGE_NAMES),
        "engine_compatible_code_count": engine_code_count,
        "engine_capacity": capacity_rows,
        "summary": (
            f"GUI picker: {gui_count} selectable OCR codes; {engine_text}."
        ),
    }
