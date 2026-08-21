"""Gettext runtime for source, frozen, and per-user locale catalogs.

The runtime provides:

- A `_()` helper that returns the input string unchanged when no
  catalog is bound, and the translated string otherwise.
- A `tr()` alias for GUI modules where `_` would collide with common
  throwaway loop variables. Extraction commands should include
  `-k_ -ktr`.
- A `bind_locale(lang)` entry point the GUI calls at startup. The
  lookup walks `locale/<lang>/LC_MESSAGES/vsr.mo` and falls back to
  English when the catalog is missing.
- A deterministic catalog lifecycle through `scripts/i18n_catalogs.py`.

The module imports cleanly when `gettext` is missing because gettext
is stdlib; the only way it disappears is on broken-up CPython builds
where users typically have bigger problems.
"""

from __future__ import annotations

import gettext
import locale as _locale
import logging
import os
import re
import struct
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DOMAIN = "vsr"

# RM-284: a catalog is only offered once it can carry the interface. Below
# this bar a user picks a language, most of the window stays English, and
# the feature reads as broken. The percentage is stamped into the compiled
# catalog's metadata by scripts/i18n_catalogs.py; a catalog with no stamp
# predates the gate and is treated as unmeasured, which fails closed.
COVERAGE_HEADER = "x-vsr-coverage"
MINIMUM_CATALOG_COVERAGE = 90.0

_coverage_cache: dict = {}


def catalog_coverage(path) -> Optional[float]:
    """Return a compiled catalog's translated percentage, or None."""
    key = str(path)
    if key in _coverage_cache:
        return _coverage_cache[key]
    percent = None
    try:
        with open(path, "rb") as handle:
            info = gettext.GNUTranslations(handle).info()
        raw = info.get(COVERAGE_HEADER)
        if raw is not None:
            percent = float(str(raw).strip().rstrip("%"))
    except Exception as exc:
        logger.debug(f"Could not read catalog coverage from {path}: {exc}")
        percent = None
    _coverage_cache[key] = percent
    return percent


def _candidate_locale_dirs() -> list:
    """Return directories that might hold compiled `.mo` catalogs.

    Order matters: in-repo `locale/` comes first so a developer build
    overrides a system-wide install; per-user `%APPDATA%/VSR/locale`
    next so power users can drop a translation without touching the
    install dir; system locations last."""
    candidates = []
    here = Path(__file__).resolve().parent.parent
    bundle_root = Path(getattr(sys, "_MEIPASS", here))
    candidates.append(bundle_root / "locale")
    if bundle_root != here:
        candidates.append(here / "locale")
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(Path(appdata) / "VideoSubtitleRemoverPro" / "locale")
    unique = []
    seen = set()
    for candidate in candidates:
        try:
            key = str(candidate.resolve()).lower()
        except OSError:
            key = str(candidate).lower()
        if key not in seen and candidate.exists():
            unique.append(candidate)
            seen.add(key)
    return unique


_active_translation: Optional[gettext.NullTranslations] = None
_active_locale = "en"


def normalise_locale_tag(value: Optional[str]) -> str:
    """Return a conservative BCP-47-style tag or an empty string."""
    raw = str(value or "").strip().split(".", 1)[0].split("@", 1)[0]
    if not raw:
        return ""
    parts = [part for part in raw.replace("_", "-").split("-") if part]
    if not parts or not re.fullmatch(r"[A-Za-z]{2,8}", parts[0]):
        return ""
    result = [parts[0].lower()]
    for part in parts[1:]:
        if not re.fullmatch(r"[A-Za-z0-9]{1,8}", part):
            return ""
        if len(part) == 4 and part.isalpha():
            result.append(part.title())
        elif len(part) in {2, 3} and part.isalpha():
            result.append(part.upper())
        else:
            result.append(part.lower())
    return "-".join(result)


def system_locale_tag() -> str:
    """Return the OS locale without collapsing territory/script subtags."""
    override = normalise_locale_tag(os.environ.get("VSR_UI_LOCALE"))
    if override:
        return override
    if sys.platform == "win32":
        try:
            import ctypes

            buffer = ctypes.create_unicode_buffer(85)
            if ctypes.windll.kernel32.GetUserDefaultLocaleName(
                buffer, len(buffer),
            ):
                detected = normalise_locale_tag(buffer.value)
                if detected:
                    return detected
        except Exception:
            pass
    try:
        detected = normalise_locale_tag(_locale.getlocale()[0])
        if detected:
            return detected
    except Exception:
        pass
    return "en"


def locale_fallback_chain(value: Optional[str]) -> tuple[str, ...]:
    tag = normalise_locale_tag(value)
    if not tag:
        return ()
    parts = tag.split("-")
    chain = [tag]
    if len(parts) >= 3 and len(parts[1]) == 4:
        chain.append("-".join(parts[:2]))
    if parts[0] not in chain:
        chain.append(parts[0])
    return tuple(chain)


def available_catalogs() -> tuple[str, ...]:
    pseudo_enabled = os.environ.get("VSR_PSEUDO_LOCALE", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    smoke_locale = normalise_locale_tag(os.environ.get("VSR_SMOKE_LOCALE"))
    found = {}
    for root in _candidate_locale_dirs():
        try:
            children = root.iterdir()
        except OSError:
            continue
        for child in children:
            if not child.is_dir():
                continue
            tag = normalise_locale_tag(child.name)
            catalog = child / "LC_MESSAGES" / f"{DOMAIN}.mo"
            if tag.lower().startswith("qps-") and not (
                pseudo_enabled or smoke_locale == tag
            ):
                continue
            if not (tag and catalog.is_file()):
                continue
            # First root wins, exactly as bind_locale resolves it. Gating a
            # later root's copy would advertise a language while binding a
            # different, possibly thinner, catalog.
            if tag.lower() in found:
                continue
            if catalog_meets_coverage(tag, catalog, smoke_locale):
                found[tag.lower()] = tag
            else:
                found[tag.lower()] = None
    return tuple(sorted(
        (tag for tag in found.values() if tag), key=str.lower))


def catalog_meets_coverage(tag: str, catalog, smoke_locale: str = "") -> bool:
    """True when this catalog is complete enough to offer and to bind."""
    if tag.lower().startswith("qps-") or (smoke_locale and smoke_locale == tag):
        return True
    percent = catalog_coverage(catalog)
    if percent is None or percent < MINIMUM_CATALOG_COVERAGE:
        logger.info(
            f"Hiding locale {tag}: coverage "
            f"{'unmeasured' if percent is None else f'{percent:.1f}%'}"
            f" is below the {MINIMUM_CATALOG_COVERAGE:.0f}% bar"
        )
        return False
    return True


def bind_locale(lang: Optional[str]) -> str:
    """Bind the active gettext catalog. `lang` is a BCP-47 / ISO 639-1
    code (`en`, `ja`, `pt_BR`). Passing None / empty keeps the
    NullTranslations -- every string returns unchanged.

    Logs a single line stating which catalog (if any) bound."""
    global _active_translation, _active_locale
    requested = (
        system_locale_tag()
        if str(lang or "").lower() == "system"
        else normalise_locale_tag(lang)
    )
    if not requested or requested.split("-", 1)[0].lower() == "en":
        _active_translation = None
        _active_locale = "en"
        return _active_locale
    locale_dirs = _candidate_locale_dirs()
    translations = []
    loaded_paths = []
    for fallback_tag in locale_fallback_chain(requested):
        wanted = fallback_tag.lower()
        for root in locale_dirs:
            try:
                directories = {
                    normalise_locale_tag(child.name).lower(): child
                    for child in root.iterdir()
                    if child.is_dir() and normalise_locale_tag(child.name)
                }
            except OSError:
                continue
            locale_dir = directories.get(wanted)
            catalog = (
                locale_dir / "LC_MESSAGES" / f"{DOMAIN}.mo"
                if locale_dir is not None else None
            )
            if catalog is None or not catalog.is_file():
                continue
            # RM-284: the coverage bar is not a picker cosmetic. A system
            # locale that resolves to a thin catalog must fall back to
            # English too, or the gate only works for people who use the
            # picker.
            if not catalog_meets_coverage(
                fallback_tag, catalog,
                normalise_locale_tag(os.environ.get("VSR_SMOKE_LOCALE")),
            ):
                break
            try:
                with catalog.open("rb") as handle:
                    translations.append(gettext.GNUTranslations(handle))
                loaded_paths.append(str(catalog))
            except (OSError, EOFError, UnicodeError, struct.error) as exc:
                logger.warning("Ignoring unreadable locale catalog %s: %s", catalog, exc)
            break
    if translations:
        primary = translations[0]
        for fallback in translations[1:]:
            primary.add_fallback(fallback)
        _active_translation = primary
        _active_locale = requested
        logger.info("Bound locale '%s' from %s", requested, loaded_paths)
        return _active_locale
    logger.info(
        f"No locale catalog for '{requested}' (searched: "
        f"{[str(d) for d in locale_dirs]}); falling back to source strings."
    )
    _active_translation = None
    _active_locale = "en"
    return _active_locale


def _(text: str) -> str:
    """Translate `text` via the bound catalog. Returns `text` unchanged
    when no catalog is bound or the catalog doesn't carry the key.

    The function is named `_` to match the gettext convention so
    extractors like `xgettext` and `pybabel` find every call site."""
    if _active_translation is None:
        return text
    return _active_translation.gettext(text)


def gettext_passthrough(text: str) -> str:
    """Public alias of `_` for places where `_` would clash with a
    local variable name."""
    return _(text)


def tr(text: str) -> str:
    """Documented GUI alias for gettext extraction (`xgettext -ktr`)."""
    return _(text)


def N_(text: str) -> str:
    """Mark a string for extraction without translating it yet.

    Deferred-translation sites need this: a model field that persists
    across a locale change, or a module-level table built at import
    time before `bind_locale()` has run, must store stable English and
    call `tr()` at render time. Marking the literal here keeps it in the
    catalog even though the `tr()` call site sees only a variable."""
    return text


def ntr(singular: str, plural: str, count: int) -> str:
    """Translate a plural pair through the active catalog."""
    if _active_translation is None:
        return singular if count == 1 else plural
    return _active_translation.ngettext(singular, plural, count)


def current_locale() -> str:
    return _active_locale


def is_translation_active() -> bool:
    return _active_translation is not None
