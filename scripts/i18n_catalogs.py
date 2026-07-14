"""Deterministic gettext extraction, merge, validation, and compilation."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import string
import struct
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
LOCALE_DIR = ROOT / "locale"
POT_PATH = LOCALE_DIR / "vsr.pot"
DOMAIN = "vsr"
PSEUDO_LOCALE = "qps-Ploc"
SOURCE_GLOBS = ("gui/**/*.py", "backend/i18n.py")


@dataclass
class Message:
    msgid: str
    msgid_plural: str = ""
    references: set[str] = field(default_factory=set)


@dataclass
class PoEntry:
    msgid: str
    msgid_plural: str = ""
    msgstr: dict[int, str] = field(default_factory=dict)
    references: tuple[str, ...] = ()
    comments: tuple[str, ...] = ()


def _source_files(root: Path = ROOT) -> list[Path]:
    files: set[Path] = set()
    for pattern in SOURCE_GLOBS:
        files.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(files, key=lambda path: path.as_posix().lower())


def extract_messages(root: Path = ROOT) -> dict[str, Message]:
    messages: dict[str, Message] = {}
    for path in _source_files(root):
        relative = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            name = node.func.id
            if name not in {"_", "tr", "ntr"} or not node.args:
                continue
            singular_node = node.args[0]
            if not isinstance(singular_node, ast.Constant) or not isinstance(
                singular_node.value, str
            ):
                continue
            singular = singular_node.value
            plural = ""
            if name == "ntr":
                if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant):
                    continue
                if not isinstance(node.args[1].value, str):
                    continue
                plural = node.args[1].value
            message = messages.setdefault(singular, Message(singular, plural))
            if message.msgid_plural != plural:
                raise ValueError(f"inconsistent plural definition for {singular!r}")
            message.references.add(f"{relative}:{node.lineno}")
    return messages


def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _header(language: str = "") -> str:
    lines = [
        "Project-Id-Version: VideoSubtitleRemoverPro\\n",
        "Report-Msgid-Bugs-To: https://github.com/SysAdminDoc/VideoSubtitleRemover/issues\\n",
        "MIME-Version: 1.0\\n",
        "Content-Type: text/plain; charset=UTF-8\\n",
        "Content-Transfer-Encoding: 8bit\\n",
    ]
    if language:
        lines.extend(
            (
                f"Language: {language}\\n",
                "Plural-Forms: nplurals=2; plural=(n != 1);\\n",
            )
        )
    return "".join(lines).replace("\\n", "\n")


def _format_header(value: str) -> list[str]:
    return ['msgid ""', 'msgstr ""'] + [_quote(line) for line in value.splitlines(True)]


def render_pot(messages: dict[str, Message]) -> str:
    blocks = [
        "# Video Subtitle Remover Pro gettext template\n"
        "# This file is distributed under the same license as the project.\n"
        + "\n".join(_format_header(_header()))
    ]
    for message in sorted(messages.values(), key=lambda item: item.msgid):
        lines = ["#: " + " ".join(sorted(message.references))]
        lines.append("msgid " + _quote(message.msgid))
        if message.msgid_plural:
            lines.append("msgid_plural " + _quote(message.msgid_plural))
            lines.extend(('msgstr[0] ""', 'msgstr[1] ""'))
        else:
            lines.append('msgstr ""')
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def _parse_quoted(value: str, *, path: Path, line_number: int) -> str:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"{path}:{line_number}: invalid PO string") from exc
    if not isinstance(parsed, str):
        raise ValueError(f"{path}:{line_number}: PO value is not a string")
    return parsed


def parse_po(path: Path) -> list[PoEntry]:
    text = path.read_text(encoding="utf-8")
    entries: list[PoEntry] = []
    current = PoEntry("")
    state: tuple[str, int] | None = None
    started = False

    def finish() -> None:
        nonlocal current, state, started
        if started:
            entries.append(current)
        current = PoEntry("")
        state = None
        started = False

    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            finish()
            continue
        started = True
        if line.startswith("#:"):
            current.references = tuple(line[2:].strip().split())
            continue
        if line.startswith("#"):
            current.comments += (raw,)
            continue
        match = re.match(r"msgstr\[(\d+)\]\s+(.*)$", line)
        if match:
            index = int(match.group(1))
            current.msgstr[index] = _parse_quoted(
                match.group(2), path=path, line_number=line_number
            )
            state = ("msgstr", index)
            continue
        for keyword in ("msgid_plural", "msgid", "msgstr"):
            prefix = keyword + " "
            if not line.startswith(prefix):
                continue
            value = _parse_quoted(
                line[len(prefix):], path=path, line_number=line_number
            )
            if keyword == "msgid":
                current.msgid = value
                state = ("msgid", 0)
            elif keyword == "msgid_plural":
                current.msgid_plural = value
                state = ("msgid_plural", 0)
            else:
                current.msgstr[0] = value
                state = ("msgstr", 0)
            break
        else:
            if line.startswith('"') and state is not None:
                value = _parse_quoted(line, path=path, line_number=line_number)
                field_name, index = state
                if field_name == "msgid":
                    current.msgid += value
                elif field_name == "msgid_plural":
                    current.msgid_plural += value
                else:
                    current.msgstr[index] = current.msgstr.get(index, "") + value
                continue
            raise ValueError(f"{path}:{line_number}: unsupported PO syntax: {raw}")
    finish()
    if not entries or entries[0].msgid != "":
        raise ValueError(f"{path}: missing gettext header")
    return entries


def _render_po_entry(entry: PoEntry) -> str:
    lines = list(entry.comments)
    if entry.references:
        lines.append("#: " + " ".join(sorted(entry.references)))
    lines.append("msgid " + _quote(entry.msgid))
    if entry.msgid_plural:
        lines.append("msgid_plural " + _quote(entry.msgid_plural))
        for index in sorted(entry.msgstr):
            lines.append(f"msgstr[{index}] " + _quote(entry.msgstr[index]))
    else:
        lines.append("msgstr " + _quote(entry.msgstr.get(0, "")))
    return "\n".join(lines)


def render_po(entries: Iterable[PoEntry]) -> str:
    entries = list(entries)
    header = entries[0].msgstr.get(0, "")
    blocks = ["\n".join(_format_header(header))]
    blocks.extend(_render_po_entry(entry) for entry in entries[1:])
    return "\n\n".join(blocks) + "\n"


_PROTECTED_TOKEN = re.compile(
    r"\{[^{}]+\}|%\([^)]+\)[#0\- +]?(?:\d+|\*)?(?:\.\d+)?[diouxXeEfFgGcrs]"
    r"|%(?:[#0\- +]?(?:\d+|\*)?(?:\.\d+)?[diouxXeEfFgGcrs%])"
)
_ACCENTS = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    "\u00c5\u0181\u00c7\u00d0\u00c9\u0191\u011c\u0126\u00ce\u0134"
    "\u0136\u013f\u1e3e\u0143\u00d6\u01a4\u024a\u0154\u0160\u0162"
    "\u00db\u1e7c\u0174\u1e8a\u00dd\u017d\u00e5\u0180\u00e7\u00f0"
    "\u00e9\u0192\u011d\u0127\u00ee\u0135\u0137\u0140\u1e3f\u0144"
    "\u00f6\u01a5\u024b\u0155\u0161\u0163\u00fb\u1e7d\u0175\u1e8b"
    "\u00fd\u017e",
)


def pseudo_translate(value: str) -> str:
    protected: list[str] = []

    def hold(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"__VSRPH{len(protected) - 1}__"

    transformed = _PROTECTED_TOKEN.sub(hold, value).translate(_ACCENTS)
    for index, token in enumerate(protected):
        transformed = transformed.replace(
            f"__\u1e7c\u0160\u0154\u01a4\u0126{index}__", token
        )
        transformed = transformed.replace(f"__VSRPH{index}__", token)
    padding = " !" * max(1, len(value) // 12)
    return f"[!! {transformed}{padding} !!]"


def pseudo_entries(messages: dict[str, Message]) -> list[PoEntry]:
    entries = [PoEntry("", msgstr={0: _header(PSEUDO_LOCALE)})]
    for message in sorted(messages.values(), key=lambda item: item.msgid):
        translations = {0: pseudo_translate(message.msgid)}
        if message.msgid_plural:
            translations[1] = pseudo_translate(message.msgid_plural)
        entries.append(
            PoEntry(
                message.msgid,
                msgid_plural=message.msgid_plural,
                msgstr=translations,
                references=tuple(sorted(message.references)),
            )
        )
    return entries


def merge_entries(path: Path, messages: dict[str, Message]) -> list[PoEntry]:
    existing = parse_po(path)
    by_id = {entry.msgid: entry for entry in existing[1:]}
    merged = [existing[0]]
    for message in sorted(messages.values(), key=lambda item: item.msgid):
        previous = by_id.get(message.msgid)
        if previous is None or previous.msgid_plural != message.msgid_plural:
            translations = {0: "", 1: ""} if message.msgid_plural else {0: ""}
            comments: tuple[str, ...] = ()
        else:
            translations = dict(previous.msgstr)
            comments = previous.comments
        merged.append(
            PoEntry(
                message.msgid,
                msgid_plural=message.msgid_plural,
                msgstr=translations,
                references=tuple(sorted(message.references)),
                comments=comments,
            )
        )
    return merged


def _placeholder_set(value: str) -> set[str]:
    fields = set()
    try:
        for _literal, field_name, _format_spec, _conversion in string.Formatter().parse(value):
            if field_name is not None:
                fields.add("{" + field_name.split("!", 1)[0].split(":", 1)[0] + "}")
    except ValueError as exc:
        raise ValueError(f"invalid Python format string {value!r}") from exc
    fields.update(_PROTECTED_TOKEN.findall(value))
    return fields


def _plural_count(header: str) -> int:
    match = re.search(r"^Plural-Forms:\s*nplurals=(\d+);", header, re.MULTILINE)
    return int(match.group(1)) if match else 0


def validate_po(path: Path, entries: list[PoEntry] | None = None) -> None:
    entries = entries or parse_po(path)
    header = entries[0].msgstr.get(0, "")
    if "charset=UTF-8" not in header:
        raise ValueError(f"{path}: Content-Type must declare UTF-8")
    plural_count = _plural_count(header)
    for entry in entries[1:]:
        if entry.msgid_plural:
            if plural_count < 1:
                raise ValueError(f"{path}: plural entry requires a Plural-Forms header")
            expected_indices = set(range(plural_count))
            if set(entry.msgstr) != expected_indices:
                raise ValueError(
                    f"{path}: plural {entry.msgid!r} needs msgstr indices "
                    f"{sorted(expected_indices)}"
                )
        elif set(entry.msgstr) != {0}:
            raise ValueError(f"{path}: singular {entry.msgid!r} must have one msgstr")
        for index, translation in entry.msgstr.items():
            if not translation:
                continue
            source = entry.msgid if index == 0 or not entry.msgid_plural else entry.msgid_plural
            if _placeholder_set(source) != _placeholder_set(translation):
                raise ValueError(
                    f"{path}: placeholder mismatch for {entry.msgid!r} msgstr[{index}]"
                )


def compile_mo(entries: list[PoEntry]) -> bytes:
    pairs: list[tuple[str, str]] = []
    for entry in entries:
        original = entry.msgid
        translated = entry.msgstr.get(0, "")
        if entry.msgid_plural:
            original += "\0" + entry.msgid_plural
            translated = "\0".join(
                entry.msgstr[index] for index in sorted(entry.msgstr)
            )
        pairs.append((original, translated))
    pairs.sort(key=lambda pair: pair[0])
    originals = [pair[0].encode("utf-8") for pair in pairs]
    translations = [pair[1].encode("utf-8") for pair in pairs]
    count = len(pairs)
    original_table = 28
    translation_table = original_table + count * 8
    string_offset = translation_table + count * 8
    original_blob = b""
    original_descriptors = []
    for value in originals:
        original_descriptors.append((len(value), string_offset + len(original_blob)))
        original_blob += value + b"\0"
    translation_offset = string_offset + len(original_blob)
    translation_blob = b""
    translation_descriptors = []
    for value in translations:
        translation_descriptors.append(
            (len(value), translation_offset + len(translation_blob))
        )
        translation_blob += value + b"\0"
    header = struct.pack(
        "<7I", 0x950412DE, 0, count, original_table, translation_table, 0, 0
    )
    tables = b"".join(struct.pack("<2I", *item) for item in original_descriptors)
    tables += b"".join(
        struct.pack("<2I", *item) for item in translation_descriptors
    )
    return header + tables + original_blob + translation_blob


def _catalog_paths(locale_dir: Path = LOCALE_DIR) -> list[Path]:
    return sorted(
        locale_dir.glob("*/LC_MESSAGES/vsr.po"),
        key=lambda path: path.as_posix().lower(),
    )


def _coverage(entries: list[PoEntry]) -> tuple[int, int]:
    total = len(entries) - 1
    translated = sum(
        1 for entry in entries[1:] if entry.msgstr and all(entry.msgstr.values())
    )
    return translated, total


def report_coverage(locale_dir: Path = LOCALE_DIR) -> None:
    for path in _catalog_paths(locale_dir):
        entries = parse_po(path)
        translated, total = _coverage(entries)
        percent = 100.0 if total == 0 else translated * 100.0 / total
        print(f"{path.parents[1].name}: {translated}/{total} ({percent:.1f}%)")


def update_catalogs(locale_dir: Path = LOCALE_DIR) -> None:
    messages = extract_messages()
    locale_dir.mkdir(parents=True, exist_ok=True)
    (locale_dir / "vsr.pot").write_text(render_pot(messages), encoding="utf-8", newline="\n")
    pseudo_path = locale_dir / PSEUDO_LOCALE / "LC_MESSAGES" / "vsr.po"
    for path in _catalog_paths(locale_dir):
        if path == pseudo_path:
            continue
        path.write_text(
            render_po(merge_entries(path, messages)), encoding="utf-8", newline="\n"
        )
    pseudo_path.parent.mkdir(parents=True, exist_ok=True)
    pseudo_path.write_text(
        render_po(pseudo_entries(messages)), encoding="utf-8", newline="\n"
    )
    compile_catalogs(locale_dir)
    report_coverage(locale_dir)


def compile_catalogs(locale_dir: Path = LOCALE_DIR) -> None:
    for path in _catalog_paths(locale_dir):
        entries = parse_po(path)
        validate_po(path, entries)
        path.with_suffix(".mo").write_bytes(compile_mo(entries))


def check_catalogs(locale_dir: Path = LOCALE_DIR) -> None:
    messages = extract_messages()
    expected_pot = render_pot(messages)
    actual_pot = (locale_dir / "vsr.pot").read_text(encoding="utf-8")
    if actual_pot != expected_pot:
        raise ValueError(
            "locale/vsr.pot drifted; run `python scripts/i18n_catalogs.py update`"
        )
    pseudo_path = locale_dir / PSEUDO_LOCALE / "LC_MESSAGES" / "vsr.po"
    expected_pseudo = render_po(pseudo_entries(messages))
    if not pseudo_path.is_file() or pseudo_path.read_text(encoding="utf-8") != expected_pseudo:
        raise ValueError("pseudo-locale catalog drifted; run the i18n update command")
    expected_ids = set(messages)
    for path in _catalog_paths(locale_dir):
        entries = parse_po(path)
        validate_po(path, entries)
        actual_ids = {entry.msgid for entry in entries[1:]}
        if actual_ids != expected_ids:
            raise ValueError(f"{path}: catalog keys do not match the POT template")
        mo_path = path.with_suffix(".mo")
        expected_mo = compile_mo(entries)
        if not mo_path.is_file() or mo_path.read_bytes() != expected_mo:
            raise ValueError(f"{mo_path}: compiled catalog drifted")
    report_coverage(locale_dir)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract, merge, validate, compile, and audit gettext catalogs."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="check",
        choices=("check", "update", "extract", "compile", "coverage"),
    )
    args = parser.parse_args()
    try:
        if args.command == "update":
            update_catalogs()
        elif args.command == "extract":
            POT_PATH.write_text(
                render_pot(extract_messages()), encoding="utf-8", newline="\n"
            )
        elif args.command == "compile":
            compile_catalogs()
            report_coverage()
        elif args.command == "coverage":
            for path in _catalog_paths():
                validate_po(path)
            report_coverage()
        else:
            check_catalogs()
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"i18n catalog error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
