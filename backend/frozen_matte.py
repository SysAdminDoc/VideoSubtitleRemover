"""RM-153: freeze a reviewed matte into a durable, reusable queue input.

A matte that a human has looked at frame by frame and approved is the most
expensive artifact this pipeline produces. Re-running the same source
normally throws it away and re-derives it from OCR and tracking, which is
slow and -- because detection is not bit-reproducible across engine,
driver, and threading changes -- not guaranteed to land on the same mask.

Freezing captures the approved matte as an explicit input: the artifact
and manifest hashes, a fingerprint of the source it was approved against,
plus the geometry, timing, and frame range it was authored for. A later
run that matches on every one of those may skip detection entirely and
paint the approved mask directly.

Everything here fails closed. A frozen matte is a promise that these
exact pixels belong to these exact frames of this exact file; if any part
of that no longer holds, the answer is a specific error naming what moved
and a request to revalidate -- never a silent re-derivation, and never a
matte quietly applied to the wrong frames.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional

from backend.io import (
    _normalize_time_base,
    _seconds_to_ticks,
    timing_ticks_digest,
)
from backend.matte_interchange import (
    MASK_INTERCHANGE_SCHEMA,
    _artifact_from_manifest,
    _load_manifest,
    _sha256_file,
    _sha256_sequence,
    normalize_mask_export_format,
)


FROZEN_MATTE_SCHEMA = "vsr.frozen_matte.v1"

# Hashing a multi-gigabyte source on every queue rerun would cost more
# than the detection pass the freeze exists to skip. The head and tail
# samples plus the exact byte length catch re-encodes, truncations, and
# same-name substitutions, which is what this fingerprint is defending
# against -- it is an integrity check, not a security boundary.
SOURCE_DIGEST_SCHEME = "vsr.sampled-sha256.v1"
SOURCE_SAMPLE_BYTES = 8 * 1024 * 1024


class FrozenMatteError(ValueError):
    """A frozen matte no longer matches the job it is attached to."""

    def __init__(self, user_message: str, *, reason: str,
                 needs_revalidation: bool = True):
        super().__init__(user_message)
        self.user_message = user_message
        self.reason = reason
        self.needs_revalidation = needs_revalidation


def source_fingerprint(path: str | Path) -> dict:
    """Return a cheap but substitution-resistant fingerprint of a source."""
    source = Path(path)
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise FrozenMatteError(
            "The source file for this frozen matte is missing.",
            reason="source_missing",
        ) from exc
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    digest.update(b"\0")
    try:
        with source.open("rb") as handle:
            digest.update(handle.read(SOURCE_SAMPLE_BYTES))
            if size > SOURCE_SAMPLE_BYTES * 2:
                handle.seek(-SOURCE_SAMPLE_BYTES, 2)
                digest.update(handle.read(SOURCE_SAMPLE_BYTES))
    except OSError as exc:
        raise FrozenMatteError(
            "The source file for this frozen matte could not be read.",
            reason="source_unreadable",
        ) from exc
    return {
        "path": str(source),
        "size_bytes": int(size),
        "digest": digest.hexdigest(),
        "digest_scheme": SOURCE_DIGEST_SCHEME,
    }


def timing_digest(timestamps: Iterable[float],
                  durations: Iterable[float]) -> str:
    """Digest a frame range's presentation timing.

    Storing the digest instead of the raw arrays keeps a frozen record
    small enough to sit in the queue-state file for a feature-length
    clip, while still detecting a single shifted frame.
    """
    digest = hashlib.sha256()
    for label, values in (("t", timestamps), ("d", durations)):
        digest.update(label.encode("ascii"))
        for value in values:
            digest.update(f"{float(value):.9f}".encode("ascii"))
            digest.update(b",")
        digest.update(b";")
    return digest.hexdigest()


def _artifact_digest(artifact: Path, export_format: str,
                     frame_count: int) -> str:
    if export_format == "png":
        return _sha256_sequence(artifact, frame_count)
    return _sha256_file(artifact)


def freeze_matte(manifest_path: str | Path,
                 source_path: str | Path) -> dict:
    """Validate an exported matte and build its frozen queue record.

    Raises `FrozenMatteError` when the manifest, the artifact, or the
    source cannot support the promise a freeze makes.
    """
    try:
        manifest, payload = _load_manifest(manifest_path)
        artifact = _artifact_from_manifest(manifest, payload)
    except ValueError as exc:
        raise FrozenMatteError(
            f"This matte manifest cannot be frozen: {exc}",
            reason="manifest_invalid",
        ) from exc

    export_format = normalize_mask_export_format(payload.get("format"))
    frame_count = int(payload.get("frame_count", 0) or 0)
    if frame_count <= 0:
        raise FrozenMatteError(
            "This matte manifest records no frames.",
            reason="empty_matte",
        )
    if export_format == "png":
        if not artifact.is_dir():
            raise FrozenMatteError(
                "The PNG matte sequence for this manifest is missing.",
                reason="artifact_missing",
            )
    elif not artifact.is_file():
        raise FrozenMatteError(
            "The matte video for this manifest is missing.",
            reason="artifact_missing",
        )

    current = _artifact_digest(artifact, export_format, frame_count)
    exported = str(payload.get("artifact_sha256") or "")
    if exported and exported != current:
        # The matte was edited after export without a re-export, so the
        # manifest's own hash no longer describes the pixels on disk.
        # Freezing that pair would bake in a contradiction.
        raise FrozenMatteError(
            "This matte was edited after it was exported. Re-export it, "
            "then freeze the new manifest.",
            reason="artifact_edited_since_export",
        )

    timestamps = payload.get("timestamps_seconds") or []
    durations = payload.get("durations_seconds") or []
    timestamp_ticks = payload.get("timestamp_ticks")
    duration_ticks = payload.get("duration_ticks")
    if not isinstance(timestamp_ticks, list):
        timestamp_ticks = None
    if not isinstance(duration_ticks, list):
        duration_ticks = None
    time_base_num, time_base_den = _normalize_time_base(
        payload.get("source_time_base_num"),
        payload.get("source_time_base_den"),
        fallback_seconds=payload.get("source_time_base_seconds", 0.0),
    )
    if timestamp_ticks is None:
        timestamp_ticks = [
            _seconds_to_ticks(value, time_base_num, time_base_den)
            for value in timestamps
        ]
    if duration_ticks is None:
        duration_ticks = [
            _seconds_to_ticks(value, time_base_num, time_base_den)
            for value in durations
        ]
    return {
        "schema": FROZEN_MATTE_SCHEMA,
        "manifest": str(manifest),
        "manifest_sha256": _sha256_file(manifest),
        "artifact": str(artifact),
        "artifact_sha256": current,
        "format": export_format,
        "width": int(payload.get("width", 0) or 0),
        "height": int(payload.get("height", 0) or 0),
        "frame_count": frame_count,
        "source_start_frame": int(payload.get("source_start_frame", 0) or 0),
        "source_end_frame": int(payload.get("source_end_frame", 0) or 0),
        "source_is_vfr": bool(payload.get("source_is_vfr", False)),
        "source_time_base_seconds": float(
            payload.get("source_time_base_seconds", 0.0) or 0.0),
        "source_time_base_num": time_base_num,
        "source_time_base_den": time_base_den,
        "source_start_ticks": (
            int(payload["source_start_ticks"])
            if payload.get("source_start_ticks") is not None else None
        ),
        "stream_start_ticks": (
            int(payload["stream_start_ticks"])
            if payload.get("stream_start_ticks") is not None else None
        ),
        "timestamp_ticks": [int(value) for value in timestamp_ticks],
        "duration_ticks": [int(value) for value in duration_ticks],
        "timing_ticks_sha256": timing_ticks_digest(
            [int(value) for value in timestamp_ticks],
            [int(value) for value in duration_ticks],
        ),
        "timing_sha256": timing_digest(timestamps, durations),
        "source": source_fingerprint(source_path),
    }


def normalize_frozen_matte(value: object) -> dict:
    """Coerce persisted state into a frozen record, or an empty dict.

    Anything unrecognised becomes `{}` rather than a half-populated
    record: a freeze that cannot be read back is not a freeze.
    """
    if not isinstance(value, dict) or not value:
        return {}
    if value.get("schema") != FROZEN_MATTE_SCHEMA:
        return {}
    required = (
        "manifest", "manifest_sha256", "artifact", "artifact_sha256",
        "format", "timing_sha256", "source",
    )
    if any(not value.get(key) for key in required):
        return {}
    if not isinstance(value.get("source"), dict):
        return {}
    record = dict(value)
    record["format"] = normalize_mask_export_format(record.get("format"))
    for key in ("width", "height", "frame_count",
                "source_start_frame", "source_end_frame"):
        try:
            record[key] = int(record.get(key, 0) or 0)
        except (TypeError, ValueError):
            return {}
    for key in ("source_start_ticks", "stream_start_ticks"):
        if record.get(key) is not None:
            try:
                record[key] = int(record[key])
            except (TypeError, ValueError, OverflowError):
                return {}
    try:
        record["source_time_base_seconds"] = float(
            record.get("source_time_base_seconds", 0.0) or 0.0)
    except (TypeError, ValueError):
        return {}
    record["source_is_vfr"] = bool(record.get("source_is_vfr", False))
    try:
        base_num, base_den = _normalize_time_base(
            record.get("source_time_base_num"),
            record.get("source_time_base_den"),
            fallback_seconds=record["source_time_base_seconds"],
        )
        record["source_time_base_num"] = base_num
        record["source_time_base_den"] = base_den
        if isinstance(record.get("timestamp_ticks"), list):
            record["timestamp_ticks"] = [
                int(value) for value in record["timestamp_ticks"]
            ]
        if isinstance(record.get("duration_ticks"), list):
            record["duration_ticks"] = [
                int(value) for value in record["duration_ticks"]
            ]
    except (TypeError, ValueError, OverflowError):
        return {}
    record["source"] = dict(record["source"])
    return record


def _require(condition: bool, message: str, reason: str) -> None:
    if not condition:
        raise FrozenMatteError(message, reason=reason)


def validate_frozen_matte(
    record: dict,
    *,
    source_path: str | Path,
    width: int,
    height: int,
    start_frame: int,
    end_frame: int,
    timestamps: Iterable[float],
    durations: Iterable[float],
    is_vfr: bool,
    source_time_base: float,
    timestamp_ticks: Optional[Iterable[int]] = None,
    duration_ticks: Optional[Iterable[int]] = None,
    source_time_base_num: Optional[int] = None,
    source_time_base_den: Optional[int] = None,
    source_start_ticks: Optional[int] = None,
    stream_start_ticks: Optional[int] = None,
    verify_artifact: bool = True,
) -> dict:
    """Re-verify a frozen matte against the job about to run.

    Returns the evidence recorded in the job report. Raises
    `FrozenMatteError` naming the first thing that moved.
    """
    frozen = normalize_frozen_matte(record)
    _require(
        bool(frozen),
        "This item's frozen matte record is unreadable. Freeze it again.",
        "record_invalid",
    )

    manifest = Path(frozen["manifest"])
    _require(
        manifest.is_file(),
        "The frozen matte manifest is missing. Freeze the matte again.",
        "manifest_missing",
    )
    _require(
        _sha256_file(manifest) == frozen["manifest_sha256"],
        "The frozen matte manifest changed on disk. Freeze it again.",
        "manifest_changed",
    )

    artifact = Path(frozen["artifact"])
    if frozen["format"] == "png":
        _require(
            artifact.is_dir(),
            "The frozen PNG matte sequence is missing. Freeze it again.",
            "artifact_missing",
        )
    else:
        _require(
            artifact.is_file(),
            "The frozen matte video is missing. Freeze it again.",
            "artifact_missing",
        )
    if verify_artifact:
        current = _artifact_digest(
            artifact, frozen["format"], frozen["frame_count"])
        _require(
            current == frozen["artifact_sha256"],
            "The frozen matte changed on disk. Freeze the new version "
            "before reusing it.",
            "artifact_changed",
        )

    stored_source = frozen["source"]
    current_source = source_fingerprint(source_path)
    _require(
        current_source["size_bytes"] == int(stored_source.get("size_bytes", -1)),
        "This file is not the source the matte was approved against "
        "(its size changed). Freeze the matte again.",
        "source_size_changed",
    )
    _require(
        stored_source.get("digest_scheme") == SOURCE_DIGEST_SCHEME,
        "The frozen matte used an older source fingerprint. Freeze it again.",
        "source_scheme_changed",
    )
    _require(
        current_source["digest"] == stored_source.get("digest"),
        "This file is not the source the matte was approved against. "
        "Freeze the matte again.",
        "source_changed",
    )

    _require(
        (frozen["width"], frozen["height"]) == (int(width), int(height)),
        f"The frozen matte is {frozen['width']}x{frozen['height']}; this job "
        f"is {int(width)}x{int(height)}. Freeze the matte again.",
        "geometry_changed",
    )
    _require(
        frozen["source_start_frame"] == int(start_frame)
        and frozen["source_end_frame"] == int(end_frame),
        f"The frozen matte covers frames {frozen['source_start_frame']}-"
        f"{frozen['source_end_frame']}; this job runs "
        f"{int(start_frame)}-{int(end_frame)}. Freeze the matte again.",
        "range_changed",
    )
    _require(
        frozen["source_is_vfr"] == bool(is_vfr),
        "The frozen matte was approved with a different CFR/VFR timing "
        "mode. Freeze the matte again.",
        "timing_mode_changed",
    )
    tolerance = max(1e-6, abs(float(source_time_base or 0.0)) * 0.51)
    _require(
        abs(frozen["source_time_base_seconds"]
            - float(source_time_base or 0.0)) <= tolerance,
        "The frozen matte was approved against a different time base. "
        "Freeze the matte again.",
        "time_base_changed",
    )
    expected_num, expected_den = _normalize_time_base(
        source_time_base_num,
        source_time_base_den,
        fallback_seconds=source_time_base,
    )
    if (
        source_time_base_num is None
        and source_time_base_den is None
        and frozen.get("source_time_base_num") is not None
        and frozen.get("source_time_base_den") is not None
    ):
        expected_num, expected_den = _normalize_time_base(
            frozen.get("source_time_base_num"),
            frozen.get("source_time_base_den"),
            fallback_seconds=source_time_base,
        )
    for key, expected in (
        ("source_start_ticks", source_start_ticks),
        ("stream_start_ticks", stream_start_ticks),
    ):
        actual = frozen.get(key)
        if expected is None or actual is None:
            continue
        _require(
            int(actual) == int(expected),
            "This job's edit-list timing no longer matches the frozen matte. "
            "Freeze it again.",
            "timing_changed",
        )
    expected_timestamp_ticks = (
        [int(value) for value in timestamp_ticks]
        if timestamp_ticks is not None else [
            _seconds_to_ticks(value, expected_num, expected_den)
            for value in timestamps
        ]
    )
    expected_duration_ticks = (
        [int(value) for value in duration_ticks]
        if duration_ticks is not None else [
            _seconds_to_ticks(value, expected_num, expected_den)
            for value in durations
        ]
    )
    if "timing_ticks_sha256" in frozen:
        _require(
            frozen.get("source_time_base_num") == expected_num
            and frozen.get("source_time_base_den") == expected_den
            and frozen.get("timing_ticks_sha256") == timing_ticks_digest(
                expected_timestamp_ticks, expected_duration_ticks),
            "This job's exact frame timing no longer matches the frozen matte. "
            "Freeze it again.",
            "timing_changed",
        )
    _require(
        timing_digest(timestamps, durations) == frozen["timing_sha256"],
        "This job's frame timing no longer matches the frozen matte. "
        "Freeze the matte again.",
        "timing_changed",
    )

    return {
        "schema": FROZEN_MATTE_SCHEMA,
        "status": "validated",
        "manifest": str(manifest),
        "artifact": str(artifact),
        "format": frozen["format"],
        "frame_count": frozen["frame_count"],
        "artifact_sha256": frozen["artifact_sha256"],
        "manifest_sha256": frozen["manifest_sha256"],
        "source_sha256": current_source["digest"],
        "source_digest_scheme": SOURCE_DIGEST_SCHEME,
        "timing_sha256": frozen["timing_sha256"],
        "source_start_frame": frozen["source_start_frame"],
        "source_end_frame": frozen["source_end_frame"],
        "source_time_base_num": frozen.get("source_time_base_num"),
        "source_time_base_den": frozen.get("source_time_base_den"),
        "source_start_ticks": frozen.get("source_start_ticks"),
        "stream_start_ticks": frozen.get("stream_start_ticks"),
        "artifact_verified": bool(verify_artifact),
        "bypassed_stages": ["ocr", "tracking", "mask_refiners"],
    }


def frozen_matte_summary(record: dict) -> str:
    """One-line description for a queue card or tooltip."""
    frozen = normalize_frozen_matte(record)
    if not frozen:
        return ""
    return (
        f"{frozen['frame_count']} frames, {frozen['width']}x"
        f"{frozen['height']}, {frozen['format'].upper()}, frames "
        f"{frozen['source_start_frame']}-{frozen['source_end_frame']}"
    )


def default_manifest_for_output(output_path: str | Path) -> Optional[Path]:
    """Locate the matte manifest a completed job wrote beside its output."""
    base = Path(output_path).with_suffix("")
    candidate = base.with_name(base.name + ".mask.json")
    return candidate if candidate.is_file() else None


def describe_frozen_matte(record: dict) -> str:
    """Render a frozen record as stable JSON for reports and bundles."""
    return json.dumps(normalize_frozen_matte(record), indent=2, sort_keys=True)


__all__ = [
    "FROZEN_MATTE_SCHEMA",
    "SOURCE_DIGEST_SCHEME",
    "FrozenMatteError",
    "MASK_INTERCHANGE_SCHEMA",
    "default_manifest_for_output",
    "describe_frozen_matte",
    "freeze_matte",
    "frozen_matte_summary",
    "normalize_frozen_matte",
    "source_fingerprint",
    "timing_digest",
    "validate_frozen_matte",
]
