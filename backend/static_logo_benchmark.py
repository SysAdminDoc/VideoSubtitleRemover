"""Benchmark helpers for static-logo cleanup research.

This module is deliberately inert: no CLI command, no subprocess execution, and
no bundled datasets. Callers must pass rights-cleared clips listed in
``tests/clips/manifest.json`` or synthetic frames created by tests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np

from backend.inpainters._common import _cv2_inpaint, _feather_blend
from backend.inpainters.external import deterministic_static_logo_cleanup
from backend.quality import compare_static_logo_cleanup

STATIC_LOGO_BENCHMARK_SCHEMA = "vsr.static_logo_benchmark.v1"
STATIC_LOGO_CATEGORY = "static_logo"
REFERENCE_LICENSE_ALLOWLIST = {
    "cc0",
    "cc0-1.0",
    "public-domain",
    "public domain",
    "mit",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc-by-4.0",
}


class StaticLogoBenchmarkError(ValueError):
    """Raised when benchmark inputs violate the reference-clip policy."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_license(value: object) -> str:
    return str(value or "").strip().lower()


def load_reference_manifest(path: Path | str) -> dict:
    manifest_path = Path(path)
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise StaticLogoBenchmarkError(
            f"reference manifest could not be read: {manifest_path}"
        ) from exc
    if not isinstance(data, dict):
        raise StaticLogoBenchmarkError("reference manifest must be a JSON object")
    if not isinstance(data.get("clips"), list):
        raise StaticLogoBenchmarkError("reference manifest clips must be a list")
    return data


def iter_static_logo_manifest_entries(
    manifest_path: Path | str,
    clips_dir: Path | str,
) -> List[dict]:
    """Return verified static-logo clip entries from a reference manifest.

    A clip must be listed, rights-cleared via the allowlist, present on disk,
    and hash-matched before benchmark code will use it.
    """
    data = load_reference_manifest(manifest_path)
    root = Path(clips_dir)
    entries: List[dict] = []
    for index, entry in enumerate(data.get("clips", [])):
        if not isinstance(entry, dict):
            raise StaticLogoBenchmarkError(f"clip entry {index} must be an object")
        if entry.get("failure_category") != STATIC_LOGO_CATEGORY:
            continue
        missing = {
            "filename", "license", "contributor", "sha256",
            "config", "metric_floors",
        } - set(entry.keys())
        if missing:
            raise StaticLogoBenchmarkError(
                f"static-logo clip {index} missing fields: {sorted(missing)}"
            )
        license_name = _normalize_license(entry.get("license"))
        if license_name not in REFERENCE_LICENSE_ALLOWLIST:
            raise StaticLogoBenchmarkError(
                f"static-logo clip {entry.get('filename')} has unsupported "
                f"license {entry.get('license')!r}"
            )
        filename = str(entry.get("filename") or "").replace("\\", "/")
        if filename.startswith("/") or ".." in Path(filename).parts:
            raise StaticLogoBenchmarkError(
                f"static-logo clip has unsafe filename: {filename!r}"
            )
        path = root / filename
        if not path.is_file():
            raise StaticLogoBenchmarkError(
                f"static-logo clip file is missing: {filename}"
            )
        expected = str(entry.get("sha256") or "").strip().lower()
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise StaticLogoBenchmarkError(
                f"static-logo clip {filename} has invalid sha256"
            )
        actual = _sha256_file(path)
        if actual != expected:
            raise StaticLogoBenchmarkError(
                f"static-logo clip {filename} sha256 mismatch"
            )
        enriched = dict(entry)
        enriched["path"] = str(path)
        entries.append(enriched)
    return entries


def current_cv2_static_logo_cleanup(
    frames: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    *,
    radius: int = 7,
    feather_px: int = 4,
) -> List[np.ndarray]:
    """Current no-weight static-logo baseline: per-frame OpenCV inpaint."""
    out: List[np.ndarray] = []
    for frame, mask in zip(frames, masks):
        filled = _cv2_inpaint(frame, mask, radius, cv2.INPAINT_NS)
        out.append(_feather_blend(frame, filled, mask, feather_px))
    return out


def run_static_logo_cleanup_benchmark(
    frames: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    *,
    reference_frames: Optional[Sequence[np.ndarray]] = None,
    methods: Optional[Iterable[str]] = None,
) -> dict:
    """Compare current cv2 cleanup with a deterministic static-logo candidate."""
    frames = list(frames)
    masks = list(masks)
    if len(frames) != len(masks):
        raise StaticLogoBenchmarkError("frames and masks must have the same length")
    if not frames:
        raise StaticLogoBenchmarkError("benchmark needs at least one frame")
    selected = {m.lower() for m in (methods or ("current_cv2", "deterministic_static"))}
    outputs = {}
    if "current_cv2" in selected:
        outputs["current_cv2"] = current_cv2_static_logo_cleanup(frames, masks)
    if "deterministic_static" in selected:
        outputs["deterministic_static"] = deterministic_static_logo_cleanup(
            frames, masks)
    if not outputs:
        raise StaticLogoBenchmarkError("no benchmark methods selected")
    metrics = compare_static_logo_cleanup(
        frames,
        outputs,
        masks,
        reference_frames=reference_frames,
    )
    return {
        "schema": STATIC_LOGO_BENCHMARK_SCHEMA,
        "category": STATIC_LOGO_CATEGORY,
        "frameCount": len(frames),
        "methods": sorted(outputs.keys()),
        "metrics": metrics,
    }
