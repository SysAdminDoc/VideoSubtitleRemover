"""GUI localisation scaffold.

RM-97: ship the infrastructure for a translatable GUI before any
actual translations land. The scaffold provides:

- A `_()` helper that returns the input string unchanged when no
  catalog is bound, and the translated string otherwise.
- A `tr()` alias for GUI modules where `_` would collide with common
  throwaway loop variables. Extraction commands should include
  `-k_ -ktr`.
- A `bind_locale(lang)` entry point the GUI calls at startup. The
  lookup walks `locale/<lang>/LC_MESSAGES/vsr.mo` and falls back to
  English when the catalog is missing.
- A `messages.pot`-style template at `locale/vsr.pot` that is refreshed
  from the GUI call sites using the same `_`/`tr` keywords.

The module imports cleanly when `gettext` is missing because gettext
is stdlib; the only way it disappears is on broken-up CPython builds
where users typically have bigger problems.
"""

from __future__ import annotations

import gettext
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DOMAIN = "vsr"


def _candidate_locale_dirs() -> list:
    """Return directories that might hold compiled `.mo` catalogs.

    Order matters: in-repo `locale/` comes first so a developer build
    overrides a system-wide install; per-user `%APPDATA%/VSR/locale`
    next so power users can drop a translation without touching the
    install dir; system locations last."""
    candidates = []
    here = Path(__file__).resolve().parent.parent
    candidates.append(here / "locale")
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(Path(appdata) / "VideoSubtitleRemoverPro" / "locale")
    return [c for c in candidates if c.exists()]


_active_translation: Optional[gettext.NullTranslations] = None


def bind_locale(lang: Optional[str]) -> None:
    """Bind the active gettext catalog. `lang` is a BCP-47 / ISO 639-1
    code (`en`, `ja`, `pt_BR`). Passing None / empty keeps the
    NullTranslations -- every string returns unchanged.

    Logs a single line stating which catalog (if any) bound."""
    global _active_translation
    if not lang:
        _active_translation = None
        return
    locale_dirs = _candidate_locale_dirs()
    for d in locale_dirs:
        try:
            t = gettext.translation(DOMAIN, localedir=str(d), languages=[lang])
            _active_translation = t
            logger.info(f"Bound locale '{lang}' from {d}")
            return
        except FileNotFoundError:
            continue
    logger.info(
        f"No locale catalog for '{lang}' (searched: "
        f"{[str(d) for d in locale_dirs]}); falling back to source strings."
    )
    _active_translation = None


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


def is_translation_active() -> bool:
    return _active_translation is not None
