"""Generate the README CLI and canonical processing-config references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
README_PATH = ROOT / "README.md"
CLI_BEGIN = "<!-- BEGIN GENERATED CLI REFERENCE -->"
CLI_END = "<!-- END GENERATED CLI REFERENCE -->"
CONFIG_BEGIN = "<!-- BEGIN GENERATED CONFIG REFERENCE -->"
CONFIG_END = "<!-- END GENERATED CONFIG REFERENCE -->"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _cli_payload() -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "backend.processor", "--dump-cli-reference"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "CLI reference probe failed: "
            + ((result.stderr or result.stdout).strip() or f"exit {result.returncode}")
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("CLI reference probe returned invalid JSON") from exc
    if payload.get("schema") != "vsr.cli_reference.v1":
        raise RuntimeError("CLI reference probe returned an unsupported schema")
    return payload


def _table_text(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|").strip()


def _format_default(value: Any) -> str:
    if value is None or value == "" or value == [] or value == argparse.SUPPRESS:
        return "-"
    if value is True:
        return "On"
    if value is False:
        return "Off"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    return str(value)


def render_cli_reference(payload: dict[str, Any]) -> str:
    options = [option for option in payload["options"] if not option["internal"]]
    lines = [
        CLI_BEGIN,
        "This table is generated from the live argparse actions and their category,",
        "default, range, visibility, and deprecation metadata. Regenerate it with",
        "`python scripts/generate_cli_reference.py --write`.",
        "",
    ]
    for category in payload["categories"]:
        rows = [option for option in options if option["category"] == category]
        if not rows:
            continue
        lines.extend(
            [
                f"#### {category}",
                "",
                "| Flag | Description | Default | Range/choices | Status |",
                "|------|-------------|---------|---------------|--------|",
            ]
        )
        for option in rows:
            flags = ", ".join(f"`{flag}`" for flag in option["flags"])
            status = "Deprecated" if option["deprecated"] else "Public"
            lines.append(
                "| "
                + " | ".join(
                    (
                        flags,
                        _table_text(option["description"]) or "-",
                        _table_text(_format_default(option["default"])),
                        _table_text(option["range"]) or "-",
                        status,
                    )
                )
                + " |"
            )
        lines.append("")
    lines.append(CLI_END)
    return "\n".join(lines)


def render_config_reference() -> str:
    from backend.config_schema import processing_field_registry

    lines = [
        CONFIG_BEGIN,
        "### Canonical processing fields",
        "",
        "These fields are accepted by `--set FIELD=JSON` and JSON config overlays.",
        "The table is generated directly from `ProcessingConfig` in registry order.",
        "",
        "| Field | Type | Default |",
        "|-------|------|---------|",
    ]
    for spec in processing_field_registry():
        lines.append(
            "| `"
            + _table_text(spec.name)
            + "` | `"
            + _table_text(spec.type_name)
            + "` | `"
            + _table_text(_format_default(spec.default))
            + "` |"
        )
    lines.extend(("", CONFIG_END))
    return "\n".join(lines)


def _replace_section(text: str, begin: str, end: str, replacement: str) -> str:
    start = text.find(begin)
    finish = text.find(end)
    if start < 0 or finish < 0 or finish < start:
        raise RuntimeError(f"README is missing generated-section markers: {begin}")
    finish += len(end)
    return text[:start] + replacement + text[finish:]


def render_readme(current: str) -> str:
    updated = _replace_section(
        current,
        CLI_BEGIN,
        CLI_END,
        render_cli_reference(_cli_payload()),
    )
    return _replace_section(
        updated,
        CONFIG_BEGIN,
        CONFIG_END,
        render_config_reference(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate or verify README CLI/config reference sections."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Update README.md in place; the default checks for drift.",
    )
    args = parser.parse_args()

    with README_PATH.open("r", encoding="utf-8", newline="") as readme:
        current = readme.read()
    expected = render_readme(current)
    if args.write:
        if current != expected:
            with README_PATH.open("w", encoding="utf-8", newline="") as readme:
                readme.write(expected)
            print("Updated README CLI/config reference sections.")
        else:
            print("README CLI/config reference sections are current.")
        return 0
    if current != expected:
        print(
            "README CLI/config reference drift detected; run "
            "`python scripts/generate_cli_reference.py --write`.",
            file=sys.stderr,
        )
        return 1
    print("README CLI/config reference sections are current.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
