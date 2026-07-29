"""RM-154: loss-aware WebVTT parsing, translation, and serialization.

WebVTT carries a great deal that SRT has no way to express: cue
identifiers, per-cue positioning settings, named regions, a CSS `STYLE`
block, `NOTE` comments, voice and language spans, classed text, ruby
annotations, and vertical writing mode. Routing a `.vtt` file through the
SRT model flattens all of it, and the flattening is silent -- the file
still looks like subtitles, so nobody notices the positioning is gone
until it is on screen.

This module keeps a WebVTT document intact. Parsing preserves the header,
every block in its original order, and each cue's identifier, timing text,
and settings verbatim. Translation rewrites only the visible text runs of
a cue payload, leaving every tag, entity, and annotation exactly where it
was. Serializing a parsed-then-unmodified document reproduces the input.

Where a conversion genuinely cannot be lossless -- writing a positioned,
regioned WebVTT document out as SRT -- `loss_report()` enumerates what
will be dropped instead of dropping it quietly.

TTML and IMSC are deliberately not handled here. They are a different
model (XML, nested styling, layout regions with inheritance) and pretending
otherwise inside a WebVTT parser would be the same silent-flattening
mistake one format further along.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace as _dataclass_replace
from pathlib import Path
from typing import Iterable, Sequence


WEBVTT_SCHEMA = "vsr.webvtt.v1"

MAX_VTT_BYTES = 16 * 1024 * 1024
MAX_CUES = 100_000
MAX_CUE_TEXT = 20_000
MAX_BLOCKS = 10_000

# `HH:MM:SS.mmm` with the hours component optional, per the WebVTT
# timestamp grammar. SRT's comma separator is accepted on input because
# real-world files mix them, but output always uses the WebVTT period.
_TIMESTAMP_RE = re.compile(
    r"^(?:(?P<hours>\d{2,}):)?(?P<minutes>[0-5]\d):"
    r"(?P<seconds>[0-5]\d)[.,](?P<millis>\d{3})$"
)

_ARROW_RE = re.compile(r"\s+-->\s+")

# A cue-settings list is space-separated `key:value` pairs. Unknown keys
# are preserved untouched rather than rejected: a newer WebVTT setting is
# still text this module has no business discarding.
_SETTING_RE = re.compile(r"^[A-Za-z-]+:[^\s]+$")

_TAG_RE = re.compile(r"<[^>]*>")

# WebVTT's character escapes. `&amp;` is applied last on the way out so a
# literal ampersand cannot be double-encoded.
_ESCAPES = (
    ("&nbsp;", "\u00a0"),
    ("&lrm;", "\u200e"),
    ("&rlm;", "\u200f"),
    ("&lt;", "<"),
    ("&gt;", ">"),
    ("&amp;", "&"),
)

# Structural features SRT cannot carry. Used to build the loss report.
SRT_UNSUPPORTED = (
    "region", "style", "note", "cue_settings", "cue_identifier",
    "voice_span", "language_span", "class_span", "ruby", "timestamp_tag",
)


class WebVttError(ValueError):
    """Raised when WebVTT input or a translated payload is invalid."""


@dataclass(frozen=True)
class VttBlock:
    """A non-cue block: `NOTE`, `STYLE`, or `REGION`."""

    kind: str            # "note" | "style" | "region"
    text: str            # verbatim block body, including its keyword line


@dataclass(frozen=True)
class VttCue:
    identifier: str      # "" when the cue had no identifier line
    start: float
    end: float
    timing_text: str     # the original "a --> b" text, reproduced verbatim
    settings: str        # the original settings tail, verbatim
    payload: str         # cue text with all markup intact

    def with_payload(self, payload: str) -> "VttCue":
        return _dataclass_replace(self, payload=payload)


@dataclass(frozen=True)
class VttDocument:
    header: str = "WEBVTT"
    # Blocks and cues interleave in the source; `order` records the
    # sequence so serializing round-trips the original layout.
    blocks: tuple[VttBlock, ...] = ()
    cues: tuple[VttCue, ...] = ()
    order: tuple[tuple[str, int], ...] = ()
    trailing_notes: tuple[str, ...] = field(default=())

    def with_cues(self, cues: Sequence[VttCue]) -> "VttDocument":
        if len(cues) != len(self.cues):
            raise WebVttError("cue count must not change")
        return _dataclass_replace(self, cues=tuple(cues))


def unescape(text: str) -> str:
    """Turn WebVTT character escapes into the characters they denote."""
    result = str(text)
    for entity, char in _ESCAPES:
        result = result.replace(entity, char)
    return result


def escape(text: str) -> str:
    """Escape the three characters WebVTT payload text must not contain."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def parse_timestamp(value: str) -> float:
    match = _TIMESTAMP_RE.fullmatch(str(value).strip())
    if not match:
        raise WebVttError(f"invalid WebVTT timestamp: {value!r}")
    hours = int(match.group("hours") or 0)
    return (
        hours * 3600.0
        + int(match.group("minutes")) * 60.0
        + int(match.group("seconds"))
        + int(match.group("millis")) / 1000.0
    )


def format_timestamp(seconds: float) -> str:
    millis = max(0, int(round(float(seconds) * 1000.0)))
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def _block_kind(first_line: str) -> str:
    keyword = first_line.strip().split(None, 1)[0].upper() if first_line.strip() else ""
    if keyword == "NOTE":
        return "note"
    if keyword == "STYLE":
        return "style"
    if keyword == "REGION":
        return "region"
    return ""


def parse_vtt(text: str) -> VttDocument:
    """Parse a bounded WebVTT document, preserving everything it carries."""
    if "\x00" in text:
        raise WebVttError("WebVTT contains a NUL byte")
    # Strip a UTF-8 BOM, then normalize line endings only -- never trim
    # payload whitespace, which is significant inside a cue.
    normalized = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.strip():
        raise WebVttError("WebVTT contains no content")

    lines = normalized.split("\n")
    header = lines[0].rstrip()
    if header != "WEBVTT" and not header.startswith(("WEBVTT ", "WEBVTT\t")):
        raise WebVttError("WebVTT must start with a WEBVTT signature line")

    blocks: list[VttBlock] = []
    cues: list[VttCue] = []
    order: list[tuple[str, int]] = []

    for raw_block in re.split(r"\n[ \t]*\n", "\n".join(lines[1:])):
        block = raw_block.strip("\n")
        if not block.strip():
            continue
        if len(blocks) + len(cues) >= MAX_BLOCKS + MAX_CUES:
            raise WebVttError("WebVTT block count exceeds the safety limit")
        block_lines = block.split("\n")
        kind = _block_kind(block_lines[0])
        if kind:
            if len(blocks) >= MAX_BLOCKS:
                raise WebVttError("WebVTT block count exceeds the safety limit")
            order.append(("block", len(blocks)))
            blocks.append(VttBlock(kind, block))
            continue

        timing_index = next(
            (index for index, line in enumerate(block_lines[:2])
             if "-->" in line),
            -1,
        )
        if timing_index < 0:
            raise WebVttError(
                f"WebVTT block is neither a cue nor NOTE/STYLE/REGION: "
                f"{block_lines[0]!r}"
            )
        timing_line = block_lines[timing_index].strip()
        parts = _ARROW_RE.split(timing_line, maxsplit=1)
        if len(parts) != 2:
            raise WebVttError(f"invalid WebVTT timing line: {timing_line!r}")
        start_text = parts[0].strip()
        tail = parts[1].split(None, 1)
        end_text = tail[0].strip() if tail else ""
        settings = tail[1].strip() if len(tail) > 1 else ""
        start = parse_timestamp(start_text)
        end = parse_timestamp(end_text)
        if end <= start:
            raise WebVttError("WebVTT cue end must be after its start")
        for setting in settings.split():
            if not _SETTING_RE.fullmatch(setting):
                raise WebVttError(
                    f"invalid WebVTT cue setting: {setting!r}")
        identifier = "\n".join(block_lines[:timing_index]).strip()
        payload = "\n".join(block_lines[timing_index + 1:]).strip("\n")
        if not payload.strip():
            raise WebVttError("WebVTT cue contains no text")
        if len(payload) > MAX_CUE_TEXT:
            raise WebVttError("WebVTT cue text exceeds the safety limit")
        if len(cues) >= MAX_CUES:
            raise WebVttError("WebVTT cue count exceeds the safety limit")
        order.append(("cue", len(cues)))
        cues.append(VttCue(
            identifier=identifier,
            start=start,
            end=end,
            timing_text=f"{start_text} --> {end_text}",
            settings=settings,
            payload=payload,
        ))

    if not cues:
        raise WebVttError("WebVTT contains no cues")
    return VttDocument(
        header=header,
        blocks=tuple(blocks),
        cues=tuple(cues),
        order=tuple(order),
    )


def read_vtt(path: str | Path) -> VttDocument:
    source = Path(path)
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise WebVttError(f"WebVTT file is unavailable: {source}") from exc
    if size <= 0 or size > MAX_VTT_BYTES:
        raise WebVttError(
            f"WebVTT size must be between 1 and {MAX_VTT_BYTES} bytes")
    try:
        text = source.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise WebVttError("WebVTT must be valid UTF-8 text") from exc
    return parse_vtt(text)


def render_vtt(document: VttDocument) -> str:
    """Serialize a document. A parsed, unmodified document round-trips."""
    chunks: list[str] = [document.header]
    for kind, index in document.order or _default_order(document):
        if kind == "block":
            chunks.append(document.blocks[index].text)
            continue
        cue = document.cues[index]
        timing = cue.timing_text or (
            f"{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}")
        if cue.settings:
            timing = f"{timing} {cue.settings}"
        head = f"{cue.identifier}\n" if cue.identifier else ""
        chunks.append(f"{head}{timing}\n{cue.payload}")
    return "\n\n".join(chunks) + "\n"


def _default_order(document: VttDocument) -> tuple[tuple[str, int], ...]:
    return (
        tuple(("block", index) for index in range(len(document.blocks)))
        + tuple(("cue", index) for index in range(len(document.cues)))
    )


def tokenize_payload(payload: str) -> list[tuple[str, str]]:
    """Split a cue payload into ordered ("tag"|"text", value) tokens."""
    tokens: list[tuple[str, str]] = []
    position = 0
    for match in _TAG_RE.finditer(payload):
        if match.start() > position:
            tokens.append(("text", payload[position:match.start()]))
        tokens.append(("tag", match.group(0)))
        position = match.end()
    if position < len(payload):
        tokens.append(("text", payload[position:]))
    return tokens


def _tag_name(tag: str) -> str:
    inner = tag[1:-1].strip()
    if inner.startswith("/"):
        inner = inner[1:].strip()
    name = inner.split(None, 1)[0] if inner else ""
    return name.split(".", 1)[0].lower()


def _is_close(tag: str) -> bool:
    return tag[1:].lstrip().startswith("/")


def translatable_runs(payload: str) -> list[str]:
    """Return the visible text runs a translator should rewrite.

    Ruby annotation text (`<rt>` content) is excluded on purpose. It is a
    pronunciation guide for the *source* script, so translating it yields
    a phonetic reading of a language the reader is no longer looking at.
    It is preserved verbatim and reported by `loss_report`.
    """
    runs: list[str] = []
    ruby_depth = 0
    for kind, value in tokenize_payload(payload):
        if kind == "tag":
            if _tag_name(value) == "rt":
                ruby_depth += -1 if _is_close(value) else 1
                ruby_depth = max(0, ruby_depth)
            continue
        if ruby_depth == 0 and value.strip():
            runs.append(unescape(value))
    return runs


def apply_translated_runs(payload: str, translations: Sequence[str]) -> str:
    """Rebuild a payload with its visible runs replaced, markup intact."""
    pending = list(translations)
    expected = len(translatable_runs(payload))
    if len(pending) != expected:
        raise WebVttError(
            f"expected {expected} translated run(s) for this cue, "
            f"got {len(pending)}"
        )
    out: list[str] = []
    ruby_depth = 0
    for kind, value in tokenize_payload(payload):
        if kind == "tag":
            if _tag_name(value) == "rt":
                ruby_depth += -1 if _is_close(value) else 1
                ruby_depth = max(0, ruby_depth)
            out.append(value)
            continue
        if ruby_depth or not value.strip():
            out.append(value)
            continue
        replacement = str(pending.pop(0) or "").strip()
        if not replacement or "\x00" in replacement:
            raise WebVttError("translated run is empty or contains a NUL byte")
        # Preserve the run's surrounding whitespace so markup stays
        # welded to the words it was wrapped around.
        leading = value[:len(value) - len(value.lstrip())]
        trailing = value[len(value.rstrip()):]
        out.append(f"{leading}{escape(replacement)}{trailing}")
    rebuilt = "".join(out)
    if len(rebuilt) > MAX_CUE_TEXT:
        raise WebVttError("translated cue text exceeds the safety limit")
    return rebuilt


def document_runs(document: VttDocument) -> list[str]:
    """Every translatable run in the document, in cue order."""
    runs: list[str] = []
    for cue in document.cues:
        runs.extend(translatable_runs(cue.payload))
    return runs


def apply_document_runs(
    document: VttDocument,
    translations: Sequence[str],
) -> VttDocument:
    """Distribute a flat translation list back across the document's cues."""
    pending = list(translations)
    expected = len(document_runs(document))
    if len(pending) != expected:
        raise WebVttError(
            f"translation provider returned {len(pending)} run(s); "
            f"expected {expected}"
        )
    cues = []
    for cue in document.cues:
        count = len(translatable_runs(cue.payload))
        chunk, pending = pending[:count], pending[count:]
        cues.append(cue.with_payload(
            apply_translated_runs(cue.payload, chunk)))
    return document.with_cues(cues)


def _feature_counts(document: VttDocument) -> dict:
    counts = {name: 0 for name in SRT_UNSUPPORTED}
    for block in document.blocks:
        if block.kind in counts:
            counts[block.kind] += 1
    for cue in document.cues:
        if cue.settings:
            counts["cue_settings"] += 1
        if cue.identifier:
            counts["cue_identifier"] += 1
        for kind, value in tokenize_payload(cue.payload):
            if kind != "tag" or _is_close(value):
                continue
            name = _tag_name(value)
            if name == "v":
                counts["voice_span"] += 1
            elif name == "lang":
                counts["language_span"] += 1
            elif name == "c":
                counts["class_span"] += 1
            elif name in {"ruby", "rt"}:
                counts["ruby"] += 1
            elif _TIMESTAMP_RE.fullmatch(value[1:-1].strip()):
                counts["timestamp_tag"] += 1
    return counts


_LOSS_DETAIL = {
    "region": "named REGION blocks position cues; SRT has no regions",
    "style": "the STYLE block's CSS has no SRT equivalent",
    "note": "NOTE comments cannot be represented in SRT",
    "cue_settings": (
        "per-cue line/position/size/align/vertical settings are dropped"
    ),
    "cue_identifier": "cue identifiers become sequential SRT numbers",
    "voice_span": "<v Name> speaker attribution is not carried by SRT",
    "language_span": "<lang> spans are not carried by SRT",
    "class_span": "<c.class> styling hooks are not carried by SRT",
    "ruby": (
        "ruby markup is lost; the annotation is parenthesized after its base"
    ),
    "timestamp_tag": "karaoke timestamp tags are dropped",
}


def loss_report(document: VttDocument, *, target_format: str = "vtt") -> dict:
    """Describe what a conversion to `target_format` cannot preserve.

    A WebVTT-to-WebVTT translation is lossless by construction, so the
    report says so explicitly rather than being omitted -- "no report"
    and "nothing was lost" must not look the same to a caller.
    """
    fmt = str(target_format).strip().lower().lstrip(".")
    if fmt not in {"vtt", "srt"}:
        raise WebVttError(f"unsupported target format: {target_format!r}")
    counts = _feature_counts(document)
    ruby_preserved = counts["ruby"] > 0
    losses = []
    if fmt == "srt":
        losses = [
            {
                "feature": name,
                "count": counts[name],
                "detail": _LOSS_DETAIL[name],
            }
            for name in SRT_UNSUPPORTED
            if counts[name]
        ]
    return {
        "schema": WEBVTT_SCHEMA,
        "targetFormat": fmt,
        "lossless": not losses,
        "cueCount": len(document.cues),
        "blockCount": len(document.blocks),
        "features": counts,
        "losses": losses,
        # Ruby annotations survive a VTT round-trip untranslated. That is
        # a deliberate choice, not an oversight, so it is stated either way.
        "rubyAnnotationsPreservedUntranslated": ruby_preserved,
    }


def _plain_text(payload: str) -> str:
    """Strip markup to plain text, keeping ruby annotations legible.

    Concatenating a ruby base with its annotation would weld two scripts
    into one corrupt word ("kanji" glued onto its own characters), so the
    annotation is parenthesized instead. Nothing is discarded, and the
    result still reads as a sentence.
    """
    out: list[str] = []
    ruby_depth = 0
    for kind, value in tokenize_payload(payload):
        if kind == "tag":
            if _tag_name(value) == "rt":
                if _is_close(value):
                    ruby_depth = max(0, ruby_depth - 1)
                    out.append(")")
                else:
                    ruby_depth += 1
                    out.append(" (")
            continue
        out.append(unescape(value))
    while ruby_depth > 0:
        out.append(")")
        ruby_depth -= 1
    return "".join(out).strip()


def to_srt_text(document: VttDocument) -> str:
    """Flatten a document to SRT. Pair with `loss_report(..., 'srt')`."""
    blocks = []
    for index, cue in enumerate(document.cues, start=1):
        plain = _plain_text(cue.payload)
        if not plain:
            raise WebVttError(
                f"cue {index} has no plain text to write as SRT")
        start = format_timestamp(cue.start).replace(".", ",")
        end = format_timestamp(cue.end).replace(".", ",")
        blocks.append(f"{index}\n{start} --> {end}\n{plain}")
    return "\n\n".join(blocks) + "\n"


def from_srt_cues(cues: Iterable) -> VttDocument:
    """Build a minimal WebVTT document from parsed SRT cues."""
    vtt_cues = []
    for cue in cues:
        start_text, end_text = [
            part.strip() for part in _ARROW_RE.split(cue.timing, maxsplit=1)
        ]
        start = parse_timestamp(start_text)
        end = parse_timestamp(end_text)
        vtt_cues.append(VttCue(
            identifier=str(cue.identifier or "").strip(),
            start=start,
            end=end,
            timing_text=(
                f"{format_timestamp(start)} --> {format_timestamp(end)}"),
            settings="",
            payload=escape(cue.text),
        ))
    if not vtt_cues:
        raise WebVttError("no SRT cues to convert")
    return VttDocument(cues=tuple(vtt_cues))


def is_webvtt_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".vtt"


__all__ = [
    "MAX_CUES",
    "MAX_CUE_TEXT",
    "MAX_VTT_BYTES",
    "SRT_UNSUPPORTED",
    "WEBVTT_SCHEMA",
    "VttBlock",
    "VttCue",
    "VttDocument",
    "WebVttError",
    "apply_document_runs",
    "apply_translated_runs",
    "document_runs",
    "escape",
    "format_timestamp",
    "from_srt_cues",
    "is_webvtt_path",
    "loss_report",
    "parse_timestamp",
    "parse_vtt",
    "read_vtt",
    "render_vtt",
    "to_srt_text",
    "tokenize_payload",
    "translatable_runs",
    "unescape",
]
