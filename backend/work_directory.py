"""Validated processing-work storage policy and volume preflight helpers."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class WorkDirectoryResolution:
    requested: str
    path: Path
    used_fallback: bool
    warning: str = ""


@dataclass(frozen=True)
class StorageRequirement:
    path: Path
    required_bytes: int
    purpose: str


@dataclass(frozen=True)
class StorageVolumeStatus:
    path: Path
    free_bytes: int
    required_bytes: int
    purposes: tuple[str, ...]


def _writable_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise NotADirectoryError(str(path))
    fd = -1
    probe = None
    try:
        fd, name = tempfile.mkstemp(prefix=".vsr-write-probe-", dir=str(path))
        probe = Path(name)
        os.write(fd, b"vsr")
        os.fsync(fd)
    finally:
        if fd >= 0:
            os.close(fd)
        if probe is not None:
            try:
                probe.unlink()
            except OSError:
                pass
    return path


def resolve_work_directory(
    requested: str | os.PathLike[str] | None,
    *,
    fallback: str | os.PathLike[str] | None = None,
) -> WorkDirectoryResolution:
    """Resolve a configured work root, probing writes before accepting it."""
    raw = str(requested or "").strip()
    fallback_path = Path(fallback or tempfile.gettempdir()).expanduser().resolve()
    if not raw:
        return WorkDirectoryResolution("", _writable_directory(fallback_path), False)
    candidate = Path(os.path.expandvars(raw)).expanduser().resolve(strict=False)
    try:
        return WorkDirectoryResolution(raw, _writable_directory(candidate), False)
    except OSError as exc:
        resolved_fallback = _writable_directory(fallback_path)
        warning = (
            f"Work directory '{candidate}' is unavailable or read-only ({exc}). "
            f"Using '{resolved_fallback}' for this run. Choose a writable work "
            "folder in Settings to keep temporary and resume files together."
        )
        return WorkDirectoryResolution(raw, resolved_fallback, True, warning)


def make_work_temp_dir(
    resolution: WorkDirectoryResolution,
    *,
    prefix: str = "vsr_",
) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(resolution.path)))


def checkpoint_directory(
    work_directory: str | os.PathLike[str] | None,
    *,
    default: str | os.PathLike[str],
) -> tuple[Path, WorkDirectoryResolution | None]:
    """Put checkpoints below a selected work root; retain the legacy default."""
    if not str(work_directory or "").strip():
        path = _writable_directory(Path(default).expanduser().resolve(strict=False))
        return path, None
    resolution = resolve_work_directory(work_directory)
    path = _writable_directory(resolution.path / "checkpoints")
    return path, resolution


def _existing_probe_path(path: Path) -> Path:
    current = path.expanduser().resolve(strict=False)
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def _volume_key(path: Path) -> str:
    resolved = path.expanduser().resolve(strict=False)
    drive = os.path.splitdrive(str(resolved))[0]
    return (drive or resolved.anchor or str(resolved)).casefold()


def assess_storage_volumes(
    requirements: Iterable[StorageRequirement],
) -> list[StorageVolumeStatus]:
    """Aggregate byte requirements per filesystem volume and probe each once."""
    grouped: dict[str, dict] = {}
    for requirement in requirements:
        required = max(0, int(requirement.required_bytes))
        key = _volume_key(requirement.path)
        group = grouped.setdefault(key, {
            "path": Path(requirement.path),
            "required": 0,
            "purposes": [],
        })
        group["required"] += required
        if requirement.purpose not in group["purposes"]:
            group["purposes"].append(requirement.purpose)
    statuses: list[StorageVolumeStatus] = []
    for group in grouped.values():
        usage = shutil.disk_usage(_existing_probe_path(group["path"]))
        statuses.append(StorageVolumeStatus(
            path=group["path"],
            free_bytes=int(usage.free),
            required_bytes=int(group["required"]),
            purposes=tuple(group["purposes"]),
        ))
    return statuses
