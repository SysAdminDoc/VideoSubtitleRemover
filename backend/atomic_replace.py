"""Journalled multi-file replacement.

RM-146: replacing a set of files with ``os.replace`` is atomic per file but not
across the set. A crash (or an OSError on the second file) could leave the
model cache with a target stranded in ``.vsrbak``, or leave a matte artifact
promoted without its manifest.

A journal is written and fsynced *before* the first move, so recovery is
deterministic on the next run:

* journal in ``pending`` state  -> roll back to the complete old set
* journal in ``committed`` state -> finish forward (drop the backups)

Either way the caller ends up with one complete, self-consistent set.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

JOURNAL_SCHEMA = "vsr.replacement_journal.v1"
JOURNAL_SUFFIX = ".vsrjournal.json"
BACKUP_SUFFIX = ".vsrbak"

STATE_PENDING = "pending"
STATE_COMMITTED = "committed"


def _write_json_synced(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def _remove(path: Path) -> None:
    try:
        if path.is_dir() and not path.is_symlink():
            import shutil

            shutil.rmtree(path, ignore_errors=True)
        elif path.exists() or path.is_symlink():
            path.unlink()
    except OSError as exc:
        logger.debug("Could not remove %s: %s", path, exc)


def backup_path_for(target: Path) -> Path:
    return target.with_name(target.name + BACKUP_SUFFIX)


class ReplacementJournal:
    """Record a planned set of replacements before any original is moved."""

    def __init__(self, directory: str | Path, name: str):
        self.directory = Path(directory)
        self.path = self.directory / f".{name}{JOURNAL_SUFFIX}"
        self._entries: list[dict[str, str]] = []
        self._started = False
        self._finished = False

    @property
    def entries(self) -> list[dict[str, str]]:
        return list(self._entries)

    def plan(self, target: str | Path, *, backup: Optional[str | Path] = None):
        """Declare one target that will be replaced."""
        if self._started:
            raise RuntimeError("cannot plan after the journal has begun")
        target_path = Path(target)
        backup_path = (
            Path(backup) if backup is not None else backup_path_for(target_path)
        )
        self._entries.append({
            "target": str(target_path),
            "backup": str(backup_path),
            # Recorded now so rollback can tell "this run created it" from
            # "this run never got to it".
            "existed": bool(target_path.exists()),
        })
        return backup_path

    def begin(self) -> None:
        """Persist the plan. Nothing may move before this returns."""
        if self._started:
            return
        _write_json_synced(self.path, {
            "schema": JOURNAL_SCHEMA,
            "state": STATE_PENDING,
            "entries": self._entries,
        })
        self._started = True

    def mark_committed(self) -> None:
        """All targets are in place; recovery should now finish forward."""
        if not self._started:
            return
        _write_json_synced(self.path, {
            "schema": JOURNAL_SCHEMA,
            "state": STATE_COMMITTED,
            "entries": self._entries,
        })

    def commit(self) -> None:
        """Mark committed, drop the backups, and clear the journal."""
        if self._finished:
            return
        self.mark_committed()
        for entry in self._entries:
            _remove(Path(entry["backup"]))
        _remove(self.path)
        self._finished = True

    def rollback(self) -> None:
        """Restore every backup over its target and clear the journal."""
        if self._finished:
            return
        _rollback_entries(self._entries)
        _remove(self.path)
        self._finished = True

    def __enter__(self) -> "ReplacementJournal":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self.rollback()
        elif not self._finished:
            self.commit()
        return False


def _rollback_entries(entries: Iterable[Any]) -> None:
    for entry in reversed(list(entries)):
        if not isinstance(entry, dict):
            continue
        target = Path(str(entry.get("target") or ""))
        backup = Path(str(entry.get("backup") or ""))
        if not str(target):
            continue
        if backup.exists():
            _remove(target)
            try:
                os.replace(backup, target)
            except OSError as exc:
                logger.warning(
                    "Could not restore %s from %s: %s", target, backup, exc)
        elif not entry.get("existed", True):
            # There was no original, so anything at the target was produced by
            # the failed run. Removing it restores the old (empty) state.
            _remove(target)
        # Otherwise the original is still in place -- this entry was never
        # reached, so leave it alone.


def _finish_forward(entries: Iterable[Any]) -> None:
    for entry in entries:
        if isinstance(entry, dict):
            _remove(Path(str(entry.get("backup") or "")))


def recover_pending_replacements(directory: str | Path) -> list[dict]:
    """Resolve every leftover journal in ``directory``. Safe to call anytime."""
    root = Path(directory)
    results: list[dict] = []
    if not root.is_dir():
        return results
    for path in sorted(root.glob(f"*{JOURNAL_SUFFIX}")):
        action = "rolled-back"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Discarding unreadable journal %s: %s", path, exc)
            _remove(path)
            results.append({
                "journal": str(path), "action": "discarded", "entries": 0})
            continue
        entries = payload.get("entries")
        entries = entries if isinstance(entries, list) else []
        if payload.get("state") == STATE_COMMITTED:
            _finish_forward(entries)
            action = "completed"
        else:
            _rollback_entries(entries)
        _remove(path)
        logger.info(
            "Recovered interrupted replacement journal %s (%s)", path, action)
        results.append({
            "journal": str(path),
            "action": action,
            "entries": len(entries),
        })
    # Any orphaned backup with no journal is a leftover from an older build or
    # a rollback that could not delete it; the target is authoritative.
    for stale in sorted(root.glob(f"*{BACKUP_SUFFIX}")):
        target = stale.with_name(stale.name[: -len(BACKUP_SUFFIX)])
        if target.exists():
            _remove(stale)
            results.append({
                "journal": "", "action": "dropped-stale-backup",
                "entries": 0, "path": str(stale),
            })
        else:
            try:
                os.replace(stale, target)
            except OSError as exc:
                logger.warning("Could not restore %s: %s", target, exc)
                continue
            results.append({
                "journal": "", "action": "restored-orphan-backup",
                "entries": 0, "path": str(target),
            })
    return results
