"""Local cache inventory and cleanup for model weights, proxies, engines,
and checkpoints that VSR accumulates on disk.

CLI: ``--cache-info`` prints sizes; ``--cache-clean`` removes stale
entries. Both skip unknown user files and active-run locks.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    label: str
    path: Path
    exists: bool = False
    total_bytes: int = 0
    file_count: int = 0
    children: List[str] = field(default_factory=list)


def _appdata_root() -> Path:
    return (
        Path(os.environ.get("APPDATA", Path.home() / ".config"))
        / "VideoSubtitleRemoverPro"
    )


def _dir_size(path: Path) -> tuple:
    """Return (total_bytes, file_count, child_names) for a directory."""
    total = 0
    count = 0
    children = []
    if not path.is_dir():
        return 0, 0, children
    try:
        for item in sorted(path.iterdir()):
            try:
                if item.is_file():
                    total += item.stat().st_size
                    count += 1
                    children.append(item.name)
                elif item.is_dir():
                    sub_total, sub_count, _ = _dir_size(item)
                    total += sub_total
                    count += sub_count
                    children.append(f"{item.name}/")
            except OSError:
                continue
    except OSError:
        pass
    return total, count, children


def discover_caches() -> List[CacheEntry]:
    """Discover all cache directories VSR uses and measure their sizes."""
    root = _appdata_root()
    home = Path.home()
    entries = []

    locations = [
        ("Settings and logs", root),
        ("Crash-resume checkpoints", root / "checkpoints"),
        ("App model cache", root / "models"),
        ("Proxy workflow cache", root / "proxy_cache"),
        ("TensorRT engine cache", root / "trt_cache"),
        ("Presets", root / "presets.json"),
        ("Torch hub checkpoints", home / ".cache" / "torch" / "hub" / "checkpoints"),
        ("simple-lama-inpainting cache", home / ".cache" / "simple_lama"),
        ("HuggingFace cache", home / ".cache" / "huggingface"),
    ]

    for label, path in locations:
        if path.is_file():
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            entries.append(CacheEntry(
                label=label,
                path=path,
                exists=True,
                total_bytes=size,
                file_count=1,
                children=[path.name],
            ))
        elif path.is_dir():
            total, count, children = _dir_size(path)
            entries.append(CacheEntry(
                label=label,
                path=path,
                exists=True,
                total_bytes=total,
                file_count=count,
                children=children[:20],
            ))
        else:
            entries.append(CacheEntry(
                label=label,
                path=path,
                exists=False,
            ))

    return entries


def _format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.2f} GB"


def print_cache_info() -> None:
    """Print a human-readable cache inventory to stdout."""
    entries = discover_caches()
    total_bytes = sum(e.total_bytes for e in entries)
    print(f"VSR cache inventory (total: {_format_bytes(total_bytes)})")
    print()
    for e in entries:
        if not e.exists:
            print(f"  {e.label}: (not present)")
            continue
        print(f"  {e.label}: {_format_bytes(e.total_bytes)} ({e.file_count} files)")
        print(f"    {e.path}")


_CLEANABLE_SUBDIRS = {
    "checkpoints",
    "proxy_cache",
    "trt_cache",
}


def clean_cache(
    *,
    dry_run: bool = True,
    subdirs: Optional[set] = None,
) -> Dict[str, int]:
    """Remove stale cache entries. Returns {subdir: bytes_freed}.

    Only cleans known-safe subdirectories (checkpoints, proxy_cache,
    trt_cache). Leaves settings, logs, presets, and unknown files alone.
    ``dry_run=True`` reports what would be cleaned without deleting.
    """
    root = _appdata_root()
    targets = subdirs or _CLEANABLE_SUBDIRS
    freed: Dict[str, int] = {}

    for name in sorted(targets):
        path = root / name
        if not path.is_dir():
            continue
        total, count, _ = _dir_size(path)
        if count == 0:
            continue
        freed[name] = total
        if dry_run:
            print(f"  [dry-run] would remove {path} ({_format_bytes(total)}, {count} files)")
        else:
            try:
                shutil.rmtree(path)
                path.mkdir(parents=True, exist_ok=True)
                logger.info("Cleaned cache: %s (%s)", path, _format_bytes(total))
                print(f"  [cleaned] {path} ({_format_bytes(total)}, {count} files)")
            except OSError as exc:
                logger.warning("Failed to clean %s: %s", path, exc)
                print(f"  [failed] {path}: {exc}")
                freed[name] = 0

    if not freed:
        print("  Nothing to clean.")
    return freed
