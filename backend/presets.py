"""Shared preset library for the GUI and CLI.

Both `VideoSubtitleRemover.py` (GUI) and `backend/processor.py` (CLI)
need to look up the same preset by name. The GUI used to own
`BUILTIN_PRESETS` privately so the CLI could not apply them; moving the
table here closes that gap and lets `python -m backend.processor
--preset "YouTube (default)"` work.

User presets persist to `%APPDATA%\\VideoSubtitleRemoverPro\\presets.json`
under the same v1 envelope (`{"name": ..., "fields": {...}}`).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple


BUILTIN_PRESETS: Dict[str, Dict[str, object]] = {
    "YouTube (default)": {
        "description": "Balanced defaults for typical YouTube / streaming footage.",
        "fields": {
            "mode": "STTN",
            "detection_threshold": 0.5,
            "mask_dilate_px": 8,
            "mask_feather_px": 4,
            "edge_ring_px": 2,
            "tbe_flow_warp": False,
            "tbe_scene_cut_split": True,
            "colour_tune_enable": False,
            "kalman_tracking": True,
            "phash_skip_enable": True,
        },
    },
    "Anime / Animation": {
        "description": "Flat backgrounds benefit from LAMA + tight feather.",
        "fields": {
            "mode": "LAMA",
            "detection_threshold": 0.55,
            "mask_dilate_px": 10,
            "mask_feather_px": 3,
            "edge_ring_px": 0,
            "colour_tune_enable": True,
            "colour_tune_tolerance": 30,
        },
    },
    "Motion-heavy / Action": {
        "description": "Enables flow-warped TBE + ProPainter for fast pans.",
        "fields": {
            "mode": "ProPainter",
            "detection_threshold": 0.45,
            "mask_dilate_px": 12,
            "mask_feather_px": 6,
            "edge_ring_px": 3,
            "tbe_flow_warp": True,
            "tbe_scene_cut_split": True,
            "kalman_tracking": True,
        },
    },
    "TikTok / Vertical short": {
        "description": "9:16 short-form with bold burned-in captions.",
        "fields": {
            "mode": "STTN",
            "detection_threshold": 0.4,
            "mask_dilate_px": 14,
            "mask_feather_px": 5,
            "colour_tune_enable": True,
            "auto_band": True,
        },
    },
    "VHS / Low-res restore": {
        "description": "Noisy SD footage; higher feather and tolerant pHash.",
        "fields": {
            "mode": "STTN",
            "detection_threshold": 0.4,
            "mask_dilate_px": 10,
            "mask_feather_px": 6,
            "edge_ring_px": 4,
            "phash_skip_enable": True,
            "phash_skip_distance": 8,
            "kalman_tracking": True,
        },
    },
    "News / Chyron (bottom-third)": {
        "description": "Lower-third graphics; auto-band + STTN + tight mask.",
        "fields": {
            "mode": "STTN",
            "detection_threshold": 0.5,
            "auto_band": True,
            "mask_dilate_px": 6,
            "mask_feather_px": 3,
            "kalman_tracking": True,
        },
    },
    "Logo / Watermark removal": {
        "description": "Remove persistent logos and watermarks. Uses LaMa "
                       "for always-visible overlays that TBE cannot recover "
                       "behind. Keeps dialogue subtitles.",
        "fields": {
            "mode": "LAMA",
            "detection_threshold": 0.45,
            "mask_dilate_px": 6,
            "mask_feather_px": 6,
            "edge_ring_px": 3,
            "remove_subtitles": False,
            "remove_chyrons": True,
            "chyron_min_hits": 30,
            "kalman_tracking": True,
            "colour_tune_enable": True,
            "colour_tune_tolerance": 20,
        },
    },
}


BENCHMARK_PRESETS: Dict[str, Dict[str, object]] = {
    "static_logo_cleanup": {
        "description": "Benchmark-only static-logo removal profile.",
        "fields": {
            "mode": "LAMA",
            "remove_subtitles": False,
            "remove_chyrons": True,
            "chyron_min_hits": 1,
            "mask_dilate_px": 6,
            "mask_feather_px": 6,
            "edge_ring_px": 3,
            "temporal_smooth_radius": 2,
            "quality_report": True,
        },
    },
}


def _user_presets_path() -> Path:
    base = Path(os.environ.get("APPDATA", Path.home() / ".config")) / "VideoSubtitleRemoverPro"
    base.mkdir(parents=True, exist_ok=True)
    return base / "presets.json"


def load_user_presets(path: Optional[Path] = None) -> Dict[str, dict]:
    p = Path(path) if path is not None else _user_presets_path()
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            return {}
        return payload
    except (OSError, json.JSONDecodeError):
        return {}


def list_preset_names(user_presets: Optional[Dict[str, dict]] = None) -> List[str]:
    """All built-in + user preset names, deduplicated and sorted by source."""
    user = user_presets if user_presets is not None else load_user_presets()
    names = list(BUILTIN_PRESETS.keys())
    for name in user.keys():
        if name not in BUILTIN_PRESETS:
            names.append(name)
    return names


def resolve_preset(name: str,
                    user_presets: Optional[Dict[str, dict]] = None) -> Optional[Dict[str, object]]:
    """Return the named preset payload (built-in or user). None if unknown."""
    if name in BUILTIN_PRESETS:
        return BUILTIN_PRESETS[name]
    user = user_presets if user_presets is not None else load_user_presets()
    candidate = user.get(name)
    if isinstance(candidate, dict):
        return candidate
    return None


def preset_fields(name: str,
                   user_presets: Optional[Dict[str, dict]] = None) -> Optional[Dict[str, object]]:
    """Return the bare `fields` dict (mode + knobs) for a preset, or None."""
    payload = resolve_preset(name, user_presets)
    if not payload:
        return None
    fields = payload.get("fields")
    if isinstance(fields, dict):
        return fields
    return None


def benchmark_preset_fields(name: str) -> Optional[Dict[str, object]]:
    """Return fields for an internal benchmark profile, or None if unknown."""
    payload = BENCHMARK_PRESETS.get(name)
    if not payload:
        return None
    fields = payload.get("fields")
    if isinstance(fields, dict):
        return dict(fields)
    return None


_INTENT_RULES = [
    ({"subtitle", "subtitles", "sub", "subs", "caption", "captions",
      "dialogue", "dialog"},
     {"remove_subtitles": True, "remove_chyrons": False}),
    ({"logo", "watermark", "stamp", "branding", "channel"},
     {"remove_subtitles": False, "remove_chyrons": True,
      "chyron_min_hits": 1}),
    ({"all", "everything", "text", "overlay"},
     {"remove_subtitles": True, "remove_chyrons": True}),
    ({"chyron", "lower-third", "lower third", "ticker", "banner"},
     {"remove_subtitles": False, "remove_chyrons": True}),
    ({"karaoke", "lyrics", "sing"},
     {"remove_subtitles": True, "karaoke_grouping": True}),
    ({"fast", "quick", "speed"},
     {"lama_super_fast": True, "detection_frame_skip": 3}),
    ({"quality", "best", "careful", "thorough"},
     {"lama_super_fast": False, "detection_frame_skip": 0,
      "quality_report": True}),
]


def parse_intent(phrase: str) -> Optional[Dict[str, object]]:
    """Map a natural-language cleanup phrase to config field overrides.

    Returns a dict of field changes, or None when the phrase does not
    match any known pattern. Purely local and deterministic -- no cloud
    or LLM calls.
    """
    if not phrase or not phrase.strip():
        return None
    words = set(phrase.lower().split())
    merged = {}
    matched = False
    for triggers, fields in _INTENT_RULES:
        if words & triggers:
            merged.update(fields)
            matched = True
    return merged if matched else None
