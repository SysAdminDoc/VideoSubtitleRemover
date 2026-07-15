"""Versioned processing-config registry and reproducibility helpers.

``backend.config.ProcessingConfig`` is the canonical field definition.  This
module projects that definition into GUI conversion, JSON serialization, CLI
overrides, settings migrations, and output sidecars so those surfaces cannot
quietly drift apart again.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, Iterable, Mapping


CONFIG_SCHEMA_VERSION = 8
GUI_SETTINGS_FORMAT = 14
CONFIG_SCHEMA_VERSION_KEY = "config_schema_version"
GUI_SETTINGS_VERSION_KEY = "vsr_settings_format"


@dataclass(frozen=True)
class ConfigFieldSpec:
    """A stable, inspectable description of one canonical backend field."""

    name: str
    default: Any
    type_name: str


def _json_value(value: Any) -> Any:
    """Return a deterministic JSON-safe representation of ``value``."""
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def processing_field_registry() -> tuple[ConfigFieldSpec, ...]:
    """Build the versioned registry from the canonical backend dataclass."""
    from backend.config import ProcessingConfig

    defaults = ProcessingConfig()
    return tuple(
        ConfigFieldSpec(
            name=field_def.name,
            default=_json_value(getattr(defaults, field_def.name)),
            type_name=str(field_def.type),
        )
        for field_def in fields(ProcessingConfig)
    )


def processing_field_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in processing_field_registry())


def serialize_dataclass_config(config: Any) -> dict[str, Any]:
    """Serialize every declared field of a dataclass without hand lists."""
    if not is_dataclass(config):
        raise TypeError("config must be a dataclass instance")
    return {
        field_def.name: _json_value(getattr(config, field_def.name))
        for field_def in fields(config)
    }


def serialize_backend_config(config: Any) -> dict[str, Any]:
    """Serialize all canonical processing fields in registry order."""
    payload: dict[str, Any] = {}
    for spec in processing_field_registry():
        payload[spec.name] = _json_value(getattr(config, spec.name, spec.default))
    return payload


def _coerce_schema_version(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def migrate_gui_settings(data: Any) -> dict[str, Any]:
    """Migrate GUI settings while preserving payloads from newer builds."""
    if not isinstance(data, dict):
        return {}
    result = dict(data)
    version = _coerce_schema_version(result.get(GUI_SETTINGS_VERSION_KEY))
    if version > GUI_SETTINGS_FORMAT:
        return result
    # Formats 1-4 added fields with backward-compatible defaults.  Format 5
    # binds settings to the canonical schema; format 6 adds work-directory
    # policy with an empty/system-default migration value. Format 7 adds
    # moving-region keyframes while preserving legacy timed rectangles.
    # Format 8 adds a persisted UI text scale with a 100 percent default.
    # Format 9 adds the persisted System/English/catalog locale preference.
    # Format 10 adds ordered add/subtract mask-correction semantics while
    # legacy correction entries continue to mean additive polygons.
    # Format 11 adds manifest-backed lossless matte export and import fields.
    # Format 12 adds the opt-in FFmpeg D3D12 acceleration policy.
    # Format 13 adds local-first subtitle translation and re-embedding.
    # Format 14 allows a confidence-gated clean plate on each timed region.
    result[GUI_SETTINGS_VERSION_KEY] = GUI_SETTINGS_FORMAT
    result[CONFIG_SCHEMA_VERSION_KEY] = CONFIG_SCHEMA_VERSION
    return result


def gui_config_kwargs(config_type: type, data: Any) -> dict[str, Any]:
    """Select declared GUI fields from a migrated settings payload."""
    if not isinstance(data, dict):
        return {}
    allowed = {field_def.name for field_def in fields(config_type)}
    return {name: value for name, value in data.items() if name in allowed}


def gui_to_backend_payload(gui_config: Any) -> dict[str, Any]:
    """Project one per-item GUI config onto every canonical backend field."""
    payload: dict[str, Any] = {}
    missing: list[str] = []
    for spec in processing_field_registry():
        if spec.name == "device":
            use_gpu = bool(getattr(gui_config, "use_gpu", True))
            gpu_id = max(0, int(getattr(gui_config, "gpu_id", 0)))
            payload[spec.name] = f"cuda:{gpu_id}" if use_gpu else "cpu"
            continue
        if not hasattr(gui_config, spec.name):
            missing.append(spec.name)
            continue
        value = getattr(gui_config, spec.name)
        if spec.name == "mode":
            value = getattr(value, "value", value)
            value = str(value).strip().lower().replace(" ", "")
        payload[spec.name] = _json_value(value)
    if missing:
        raise ValueError(
            "GUI processing schema is missing backend fields: "
            + ", ".join(missing)
        )
    return payload


def gui_to_backend_config(gui_config: Any):
    """Create and normalize a backend config without a manual field map."""
    from backend.config import ProcessingConfig, normalize_processing_config

    return normalize_processing_config(ProcessingConfig(**gui_to_backend_payload(gui_config)))


def apply_backend_payload(config: Any, payload: Mapping[str, Any]) -> Any:
    """Apply a validated config mapping and return the normalized instance."""
    from backend.config import normalize_processing_config

    allowed = set(processing_field_names())
    unknown = sorted(str(name) for name in payload if name not in allowed)
    if unknown:
        raise ValueError("unknown processing config field(s): " + ", ".join(unknown))
    for name, value in payload.items():
        setattr(config, name, value)
    return normalize_processing_config(config)


def parse_cli_assignments(assignments: Iterable[str]) -> dict[str, Any]:
    """Parse repeatable ``FIELD=JSON`` CLI overrides against the registry."""
    allowed = set(processing_field_names())
    result: dict[str, Any] = {}
    for assignment in assignments:
        name, separator, raw = str(assignment).partition("=")
        name = name.strip()
        if not separator or not name:
            raise ValueError(f"invalid --set value {assignment!r}; expected FIELD=JSON")
        if name not in allowed:
            raise ValueError(f"unknown processing config field: {name}")
        try:
            result[name] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSON for processing config field {name}: {exc.msg}"
            ) from exc
    return result


def backend_config_cli_args(
    config: Any,
    *,
    include_defaults: bool = False,
) -> list[str]:
    """Generate lossless versioned CLI overrides from a backend config."""
    current = serialize_backend_config(config)
    args = ["--config-schema-version", str(CONFIG_SCHEMA_VERSION)]
    for spec in processing_field_registry():
        value = current[spec.name]
        if not include_defaults and value == spec.default:
            continue
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        args.extend(["--set", f"{spec.name}={encoded}"])
    return args


def validate_gui_schema(gui_config_type: type) -> None:
    """Fail fast when a new backend field has no GUI/per-item projection."""
    gui_names = {field_def.name for field_def in fields(gui_config_type)}
    missing = set(processing_field_names()) - {"device"} - gui_names
    if missing:
        raise ValueError("GUI schema missing: " + ", ".join(sorted(missing)))


def ensure_supported_schema_version(version: Any) -> None:
    parsed = _coerce_schema_version(version)
    if parsed != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported config schema version {version!r}; "
            f"this build requires {CONFIG_SCHEMA_VERSION}"
        )
