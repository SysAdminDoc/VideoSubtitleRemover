"""Local-first SRT translation providers and strict cue preservation."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from backend.io import _write_text_atomic
from backend.subprocess_policy import run_process


TRANSLATION_SCHEMA = "vsr.subtitle_translation.v1"
TRANSLATION_REQUEST_SCHEMA = "vsr.translation_request.v1"
TRANSLATION_RESPONSE_SCHEMA = "vsr.translation_response.v1"
MAX_SRT_BYTES = 16 * 1024 * 1024
MAX_CUES = 100_000
MAX_CUE_TEXT = 20_000

_TIMING_RE = re.compile(
    r"^(?P<start>\d{2,}:\d{2}:\d{2}[,.]\d{3})\s+-->\s+"
    r"(?P<end>\d{2,}:\d{2}:\d{2}[,.]\d{3})(?P<settings>.*)$"
)
_PROVIDER_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_LANGUAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,31}$")


class SubtitleTranslationError(ValueError):
    """Raised when translation input, provider output, or execution is invalid."""


@dataclass(frozen=True)
class SrtCue:
    identifier: str
    timing: str
    text: str


TranslationProvider = Callable[
    [Sequence[str], str, str, Mapping[str, object]],
    Sequence[str],
]

_PROVIDERS: dict[str, TranslationProvider] = {}


def normalize_provider_name(value: object) -> str:
    name = str(value or "command").strip().lower()
    return name if _PROVIDER_RE.fullmatch(name) else "command"


def normalize_language_tag(value: object, default: str = "") -> str:
    tag = str(value or default).strip()
    if not tag:
        return default
    return tag if _LANGUAGE_RE.fullmatch(tag) else default


def register_translation_provider(
    name: str,
    provider: TranslationProvider,
    *,
    replace: bool = False,
) -> None:
    normalized = normalize_provider_name(name)
    if normalized != str(name or "").strip().lower():
        raise ValueError(f"invalid translation provider name: {name!r}")
    if not callable(provider):
        raise TypeError("translation provider must be callable")
    if normalized in _PROVIDERS and not replace:
        raise ValueError(f"translation provider already registered: {normalized}")
    _PROVIDERS[normalized] = provider


def translation_provider_names() -> tuple[str, ...]:
    return tuple(sorted(_PROVIDERS))


def validate_translation_config(config: object) -> None:
    """Fail before media processing when an enabled workflow is incomplete."""
    if not bool(getattr(config, "translation_enabled", False)):
        return
    if str(getattr(config, "restyle_subtitle", "") or "").strip():
        raise SubtitleTranslationError(
            "translation workflow cannot be combined with restyle_subtitle")
    translated = str(getattr(config, "translation_srt", "") or "").strip()
    if translated:
        read_srt(translated)
        return
    target = normalize_language_tag(
        getattr(config, "translation_target_lang", ""))
    if not target:
        raise SubtitleTranslationError("translation target language is required")
    provider = normalize_provider_name(
        getattr(config, "translation_provider", "command"))
    if provider not in _PROVIDERS:
        raise SubtitleTranslationError(
            f"translation provider is not registered: {provider}")
    if provider == "command":
        _resolved_command(getattr(config, "translation_command", ""))
    source = str(getattr(config, "translation_source_srt", "") or "").strip()
    if source:
        read_srt(source)


def _timestamp_seconds(value: str) -> float:
    normalized = value.replace(".", ",")
    hh, mm, tail = normalized.split(":", 2)
    ss, ms = tail.split(",", 1)
    if int(mm) >= 60 or int(ss) >= 60:
        raise SubtitleTranslationError(f"invalid SRT timestamp: {value}")
    return int(hh) * 3600.0 + int(mm) * 60.0 + int(ss) + int(ms) / 1000.0


def parse_srt(text: str) -> list[SrtCue]:
    """Parse a bounded UTF-8 SRT while preserving identifiers and timing text."""
    if "\x00" in text:
        raise SubtitleTranslationError("SRT contains a NUL byte")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise SubtitleTranslationError("SRT contains no cues")
    cues: list[SrtCue] = []
    for block in re.split(r"\n[ \t]*\n", normalized):
        lines = block.splitlines()
        timing_index = next(
            (index for index, line in enumerate(lines[:2]) if "-->" in line),
            -1,
        )
        if timing_index < 0:
            raise SubtitleTranslationError("SRT cue is missing a timing line")
        match = _TIMING_RE.fullmatch(lines[timing_index].strip())
        if not match:
            raise SubtitleTranslationError(
                f"invalid SRT timing line: {lines[timing_index]!r}")
        if _timestamp_seconds(match.group("end")) <= _timestamp_seconds(
            match.group("start")
        ):
            raise SubtitleTranslationError("SRT cue end must be after its start")
        identifier = (
            "\n".join(lines[:timing_index]).strip()
            or str(len(cues) + 1)
        )
        cue_text = "\n".join(lines[timing_index + 1:]).strip()
        if not cue_text:
            raise SubtitleTranslationError("SRT cue contains no text")
        if len(cue_text) > MAX_CUE_TEXT:
            raise SubtitleTranslationError("SRT cue text exceeds the safety limit")
        cues.append(SrtCue(identifier, lines[timing_index].strip(), cue_text))
        if len(cues) > MAX_CUES:
            raise SubtitleTranslationError("SRT cue count exceeds the safety limit")
    return cues


def read_srt(path: str | Path) -> list[SrtCue]:
    source = Path(path)
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise SubtitleTranslationError(f"SRT file is unavailable: {source}") from exc
    if size <= 0 or size > MAX_SRT_BYTES:
        raise SubtitleTranslationError(
            f"SRT size must be between 1 and {MAX_SRT_BYTES} bytes")
    try:
        text = source.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise SubtitleTranslationError("SRT must be valid UTF-8 text") from exc
    return parse_srt(text)


def render_srt(cues: Sequence[SrtCue], translations: Sequence[str]) -> str:
    if len(cues) != len(translations):
        raise SubtitleTranslationError(
            "translation provider returned a different cue count")
    blocks = []
    for cue, translated in zip(cues, translations):
        text = str(translated or "").strip()
        if not text or "\x00" in text or len(text) > MAX_CUE_TEXT:
            raise SubtitleTranslationError(
                "translation provider returned invalid cue text")
        blocks.append(f"{cue.identifier}\n{cue.timing}\n{text}")
    return "\n\n".join(blocks) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_command(value: object) -> list[str]:
    command = str(value or "").strip()
    if not command:
        raise SubtitleTranslationError(
            "local command translation requires --translation-command")
    candidate = Path(command).expanduser()
    if candidate.is_file():
        resolved = str(candidate.resolve())
    else:
        found = shutil.which(command)
        if not found:
            raise SubtitleTranslationError(
                f"local translation command was not found: {command}")
        resolved = found
    if Path(resolved).suffix.lower() == ".py":
        return [sys.executable, resolved]
    return [resolved]


def _command_provider(
    texts: Sequence[str],
    source_language: str,
    target_language: str,
    options: Mapping[str, object],
) -> Sequence[str]:
    request = {
        "schema": TRANSLATION_REQUEST_SCHEMA,
        "sourceLanguage": source_language,
        "targetLanguage": target_language,
        "cues": [
            {"index": index + 1, "text": text}
            for index, text in enumerate(texts)
        ],
    }
    try:
        timeout = float(options.get("timeout", 300.0) or 300.0)
    except (TypeError, ValueError):
        timeout = 300.0
    result = run_process(
        _resolved_command(options.get("command")),
        input=json.dumps(request, ensure_ascii=False),
        capture_output=True,
        text=True,
        timeout=max(5.0, min(3600.0, timeout)),
        check=False,
        max_output_bytes=MAX_SRT_BYTES,
    )
    if result.returncode != 0:
        error = str(result.stderr or "").strip()[-600:]
        raise SubtitleTranslationError(
            "local translation command failed"
            + (f": {error}" if error else f" with exit {result.returncode}")
        )
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise SubtitleTranslationError(
            "local translation command returned invalid JSON") from exc
    if payload.get("schema") != TRANSLATION_RESPONSE_SCHEMA:
        raise SubtitleTranslationError(
            "local translation command returned an unsupported response schema")
    translations = payload.get("translations")
    if not isinstance(translations, list):
        raise SubtitleTranslationError(
            "local translation command response is missing translations")
    return [str(value or "") for value in translations]


def translated_srt_path(output_path: str | Path, target_language: str) -> Path:
    output = Path(output_path)
    tag = normalize_language_tag(target_language, "translated")
    safe_tag = re.sub(r"[^A-Za-z0-9_.-]", "-", tag)
    return output.with_name(f"{output.stem}.{safe_tag}.srt")


def translate_srt_file(
    source_path: str | Path,
    output_path: str | Path,
    *,
    provider_name: str,
    source_language: str,
    target_language: str,
    provider_options: Mapping[str, object] | None = None,
    source_kind: str = "provided-source-srt",
) -> dict:
    provider_key = normalize_provider_name(provider_name)
    provider = _PROVIDERS.get(provider_key)
    if provider is None:
        raise SubtitleTranslationError(
            f"translation provider is not registered: {provider_key}")
    target = normalize_language_tag(target_language)
    if not target:
        raise SubtitleTranslationError("translation target language is required")
    source = normalize_language_tag(source_language, "auto")
    cues = read_srt(source_path)
    try:
        translations = provider(
            [cue.text for cue in cues],
            source,
            target,
            dict(provider_options or {}),
        )
    except SubtitleTranslationError:
        raise
    except Exception as exc:
        raise SubtitleTranslationError(
            f"translation provider {provider_key} failed: {exc}") from exc
    destination = Path(output_path)
    _write_text_atomic(destination, render_srt(cues, translations))
    return {
        "schema": TRANSLATION_SCHEMA,
        "requested": True,
        "status": "translated",
        "provider": provider_key,
        "providerMode": "local",
        "sourceKind": source_kind,
        "sourceLanguage": source,
        "targetLanguage": target,
        "cueCount": len(cues),
        "source": {
            "name": Path(source_path).name,
            "sha256": _sha256(Path(source_path)),
        },
        "translated": {
            "name": destination.name,
            "sha256": _sha256(destination),
        },
    }


def provided_translation_evidence(
    path: str | Path,
    *,
    target_language: str = "",
) -> dict:
    source = Path(path)
    cues = read_srt(source)
    return {
        "schema": TRANSLATION_SCHEMA,
        "requested": True,
        "status": "validated",
        "provider": "provided",
        "providerMode": "provided",
        "sourceKind": "provided-translated-srt",
        "sourceLanguage": "",
        "targetLanguage": normalize_language_tag(target_language),
        "cueCount": len(cues),
        "translated": {
            "name": source.name,
            "sha256": _sha256(source),
        },
    }


def render_segments_srt(
    segments: Sequence[tuple[float, float, str]],
) -> str:
    def timestamp(seconds: float) -> str:
        milliseconds = max(0, int(round(float(seconds) * 1000.0)))
        hours, remainder = divmod(milliseconds, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        secs, millis = divmod(remainder, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    blocks = []
    for start, end, text in segments:
        clean = str(text or "").strip()
        if clean and float(end) > float(start):
            blocks.append(
                f"{len(blocks) + 1}\n{timestamp(start)} --> {timestamp(end)}\n{clean}"
            )
    if not blocks:
        raise SubtitleTranslationError("Whisper produced no translatable cues")
    return "\n\n".join(blocks) + "\n"


register_translation_provider("command", _command_provider)
