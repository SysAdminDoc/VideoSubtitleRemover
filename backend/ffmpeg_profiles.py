"""FFmpeg feature profiles for preflight diagnostics.

The app can run with a minimal FFmpeg build, but several advertised
features depend on optional filters or encoders. This module centralizes
those probes so self-test, support bundles, release evidence, and GUI
preflight warnings explain the same missing capability in the same terms.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from typing import Any, Mapping, Optional, Sequence


FFMPEG_PROFILE_SCHEMA = "vsr.ffmpeg_profiles.v1"

_FILTER_RE = re.compile(r"^\s*[.A-Z|]+\s+([A-Za-z0-9_][A-Za-z0-9_.-]*)\s+", re.M)
_ENCODER_RE = re.compile(r"^\s*[A-Z.]{6}\s+([A-Za-z0-9_][A-Za-z0-9_.-]*)\s+", re.M)

_MODERN_ENCODER_GROUPS: dict[str, tuple[str, ...]] = {
    "h265": ("libx265", "hevc_nvenc", "hevc_qsv", "hevc_amf"),
    "av1": ("libsvtav1", "libaom-av1", "av1_nvenc", "av1_qsv", "av1_amf"),
    "vvc": ("libvvenc", "vvc_nvenc", "vvc_qsv", "vvc_amf"),
}

_PROFILE_DEFS: dict[str, dict[str, Any]] = {
    "basic": {
        "label": "Basic media",
        "tools": ("ffmpeg", "ffprobe"),
        "filters": (),
        "encoders": ("libx264",),
        "encoder_groups": {},
    },
    "advanced_quality": {
        "label": "Advanced quality",
        "tools": ("ffmpeg",),
        "filters": ("loudnorm", "libvmaf"),
        "encoders": (),
        "encoder_groups": {},
    },
    "speech_fallback": {
        "label": "Speech fallback",
        "tools": ("ffmpeg",),
        "filters": ("whisper",),
        "encoders": (),
        "encoder_groups": {},
    },
    "modern_codec": {
        "label": "Modern codec",
        "tools": ("ffmpeg",),
        "filters": (),
        "encoders": (),
        "encoder_groups": _MODERN_ENCODER_GROUPS,
    },
}


def _run_ffmpeg_text(command: Sequence[str], timeout: float) -> tuple[str, str]:
    try:
        proc = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return "", str(exc)
    return f"{proc.stdout or ''}\n{proc.stderr or ''}", ""


def _tool_status(name: str, *, timeout: float) -> dict:
    path = shutil.which(name)
    payload = {
        "available": bool(path),
        "path": path or "",
        "version": "",
        "error": "",
    }
    if not path:
        return payload
    text, error = _run_ffmpeg_text([path, "-version"], timeout)
    if error:
        payload["error"] = error
        return payload
    first = text.splitlines()[:1]
    payload["version"] = first[0] if first else ""
    return payload


def parse_ffmpeg_filters(text: str) -> set[str]:
    return {match.group(1) for match in _FILTER_RE.finditer(text or "")}


def parse_ffmpeg_encoders(text: str) -> set[str]:
    return {match.group(1) for match in _ENCODER_RE.finditer(text or "")}


def _missing_for_profile(
    name: str,
    *,
    tools: Mapping[str, Mapping[str, Any]],
    filters: set[str],
    encoders: set[str],
) -> dict:
    definition = _PROFILE_DEFS[name]
    missing_tools = [
        tool for tool in definition["tools"]
        if not tools.get(tool, {}).get("available")
    ]
    missing_filters = [
        item for item in definition["filters"]
        if item not in filters
    ]
    missing_encoders = [
        item for item in definition["encoders"]
        if item not in encoders
    ]
    missing_groups = []
    for group_name, alternatives in definition["encoder_groups"].items():
        if not any(encoder in encoders for encoder in alternatives):
            missing_groups.append({
                "name": group_name,
                "any_of": list(alternatives),
            })
    return {
        "tools": missing_tools,
        "filters": missing_filters,
        "encoders": missing_encoders,
        "encoder_groups": missing_groups,
    }


def _reason_from_missing(missing: Mapping[str, Any]) -> str:
    parts = []
    if missing.get("tools"):
        parts.append("missing tools: " + ", ".join(missing["tools"]))
    if missing.get("filters"):
        parts.append("missing filters: " + ", ".join(missing["filters"]))
    if missing.get("encoders"):
        parts.append("missing encoders: " + ", ".join(missing["encoders"]))
    groups = missing.get("encoder_groups") or []
    if groups:
        rendered = []
        for group in groups:
            rendered.append(
                f"{group['name']} ({'/'.join(group.get('any_of') or [])})"
            )
        parts.append("missing encoder groups: " + ", ".join(rendered))
    return "; ".join(parts) if parts else "ready"


def _profile_entry(
    name: str,
    *,
    tools: Mapping[str, Mapping[str, Any]],
    filters: set[str],
    encoders: set[str],
) -> dict:
    definition = _PROFILE_DEFS[name]
    missing = _missing_for_profile(
        name,
        tools=tools,
        filters=filters,
        encoders=encoders,
    )
    available = not any(missing.values())
    return {
        "name": name,
        "label": definition["label"],
        "available": available,
        "requires": {
            "tools": list(definition["tools"]),
            "filters": list(definition["filters"]),
            "encoders": list(definition["encoders"]),
            "encoder_groups": {
                key: list(value)
                for key, value in definition["encoder_groups"].items()
            },
        },
        "missing": missing,
        "reason": _reason_from_missing(missing),
    }


def collect_ffmpeg_capability_profiles(*, timeout: float = 10.0) -> dict:
    """Return deterministic FFmpeg profile evidence."""
    tools = {
        "ffmpeg": _tool_status("ffmpeg", timeout=timeout),
        "ffprobe": _tool_status("ffprobe", timeout=timeout),
    }
    filters_text = ""
    filters_error = ""
    encoders_text = ""
    encoders_error = ""
    ffmpeg_path = tools["ffmpeg"].get("path")
    if ffmpeg_path:
        filters_text, filters_error = _run_ffmpeg_text(
            [str(ffmpeg_path), "-hide_banner", "-filters"],
            timeout,
        )
        encoders_text, encoders_error = _run_ffmpeg_text(
            [str(ffmpeg_path), "-hide_banner", "-encoders"],
            timeout,
        )
    filters = parse_ffmpeg_filters(filters_text)
    encoders = parse_ffmpeg_encoders(encoders_text)
    profiles = [
        _profile_entry(
            name,
            tools=tools,
            filters=filters,
            encoders=encoders,
        )
        for name in ("basic", "advanced_quality", "speech_fallback", "modern_codec")
    ]
    return {
        "schema": FFMPEG_PROFILE_SCHEMA,
        "tools": tools,
        "profiles": profiles,
        "feature_counts": {
            "filters": len(filters),
            "encoders": len(encoders),
        },
        "probe_errors": {
            "filters": filters_error,
            "encoders": encoders_error,
        },
    }


def _profile_map(payload: Optional[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    profiles = (payload or {}).get("profiles") or []
    result: dict[str, Mapping[str, Any]] = {}
    if isinstance(profiles, list):
        for entry in profiles:
            if isinstance(entry, Mapping) and entry.get("name"):
                result[str(entry["name"])] = entry
    return result


def ffmpeg_profile_entries(payload: Optional[Mapping[str, Any]]) -> list[dict]:
    """Return self-test style entries for the four user-facing profiles."""
    entries = []
    for name in ("basic", "advanced_quality", "speech_fallback", "modern_codec"):
        entry = _profile_map(payload).get(name)
        if entry is None:
            entries.append({
                "name": name,
                "available": False,
                "reason": "profile not probed",
            })
        else:
            entries.append({
                "name": name,
                "available": bool(entry.get("available")),
                "reason": str(entry.get("reason") or "unknown"),
            })
    return entries


def _missing_filter(payload: Mapping[str, Any], profile: str, name: str) -> bool:
    entry = _profile_map(payload).get(profile) or {}
    missing = entry.get("missing") if isinstance(entry, Mapping) else {}
    return name in ((missing or {}).get("filters") or [])


def _missing_encoder_group(
    payload: Mapping[str, Any],
    profile: str,
    group_name: str,
) -> Optional[Mapping[str, Any]]:
    entry = _profile_map(payload).get(profile) or {}
    missing = entry.get("missing") if isinstance(entry, Mapping) else {}
    for group in (missing or {}).get("encoder_groups") or []:
        if isinstance(group, Mapping) and group.get("name") == group_name:
            return group
    return None


def missing_profile_requirements_for_config(
    config: object,
    profiles: Optional[Mapping[str, Any]] = None,
) -> list[dict]:
    """Return profile requirements that the selected config cannot satisfy."""
    payload = profiles or collect_ffmpeg_capability_profiles()
    missing: list[dict] = []
    if getattr(config, "preserve_audio", False):
        basic = _profile_map(payload).get("basic", {})
        if not basic.get("available"):
            missing.append({
                "profile": "basic",
                "feature": "audio preservation",
                "missing": basic.get("missing") or {},
                "reason": basic.get("reason") or "basic FFmpeg profile unavailable",
            })
    if getattr(config, "loudnorm_target", 0.0) != 0.0:
        if _missing_filter(payload, "advanced_quality", "loudnorm"):
            missing.append({
                "profile": "advanced_quality",
                "feature": "loudness normalisation",
                "missing": {"filters": ["loudnorm"]},
                "reason": "missing filters: loudnorm",
            })
    if getattr(config, "quality_report", False):
        if _missing_filter(payload, "advanced_quality", "libvmaf"):
            missing.append({
                "profile": "advanced_quality",
                "feature": "VMAF quality metric",
                "missing": {"filters": ["libvmaf"]},
                "reason": "missing filters: libvmaf",
            })
    if (getattr(config, "whisper_fallback", False)
            and getattr(config, "whisper_backend", "faster-whisper") == "ffmpeg"):
        if _missing_filter(payload, "speech_fallback", "whisper"):
            missing.append({
                "profile": "speech_fallback",
                "feature": "FFmpeg Whisper fallback",
                "missing": {"filters": ["whisper"]},
                "reason": "missing filters: whisper",
            })
    codec = str(getattr(config, "output_codec", "h264") or "h264").lower()
    if codec in _MODERN_ENCODER_GROUPS:
        group = _missing_encoder_group(payload, "modern_codec", codec)
        if group:
            missing.append({
                "profile": "modern_codec",
                "feature": f"{codec} output codec",
                "missing": {"encoder_groups": [dict(group)]},
                "reason": (
                    "missing encoder group: "
                    f"{codec} ({'/'.join(group.get('any_of') or [])})"
                ),
            })
    return missing


def summarize_missing_profile_requirements(items: Sequence[Mapping[str, Any]]) -> str:
    parts = []
    for item in items:
        feature = str(item.get("feature") or item.get("profile") or "feature")
        reason = str(item.get("reason") or "missing requirement")
        parts.append(f"{feature}: {reason}")
    return "; ".join(parts)
