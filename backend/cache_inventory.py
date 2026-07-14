"""Local cache inventory and cleanup for model weights, proxies, engines,
and checkpoints that VSR accumulates on disk.

CLI: ``--cache-info`` prints sizes; ``--cache-clean`` removes stale
entries. Both skip unknown user files and active-run locks.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Mapping, Optional

logger = logging.getLogger(__name__)

PORTABLE_MODEL_CACHE_SCHEMA = "vsr.model_cache_bundle.v1"
MODEL_CACHE_STATUS_SCHEMA = "vsr.model_cache_status.v1"
MODEL_CACHE_MANIFEST = "manifest.json"

# Bounds for importing an untrusted model-cache bundle. A ZIP archive is
# untrusted input, so the importer preflights these ceilings before writing
# anything and stages every member before committing, guarding against
# decompression-bomb disk exhaustion and partial multi-file imports.
# Source: https://docs.python.org/3/library/zipfile.html#decompression-pitfalls
_IMPORT_MAX_MEMBERS = 1024
_IMPORT_MAX_MEMBER_BYTES = 8 * 1024 ** 3            # 8 GiB per file
_IMPORT_MAX_TOTAL_BYTES = 32 * 1024 ** 3            # 32 GiB total expansion
_IMPORT_MAX_COMPRESSION_RATIO = 200.0              # uncompressed / compressed
_IMPORT_FREE_SPACE_SLACK_BYTES = 256 * 1024 ** 2    # keep 256 MiB headroom

_MODEL_CACHE_ROOTS = {
    "app-model-cache": "",
    "torch-hub-checkpoints": "torch-hub-checkpoints",
    "simple-lama-cache": "simple-lama-cache",
    "huggingface-hub": "huggingface-hub",
    "opencv-model-cache": "opencv-model-cache",
}
_SAFE_MODEL_SUFFIXES = {
    ".bin",
    ".json",
    ".model",
    ".onnx",
    ".pdiparams",
    ".pdmodel",
    ".pdparams",
    ".safetensors",
    ".yaml",
    ".yml",
}
_PICKLE_LIKE_MODEL_SUFFIXES = {".ckpt", ".pkl", ".pt", ".pth"}
_EXECUTABLE_SUFFIXES = {
    ".bat",
    ".cmd",
    ".dll",
    ".dylib",
    ".exe",
    ".js",
    ".msi",
    ".ps1",
    ".py",
    ".pyc",
    ".pyd",
    ".sh",
    ".so",
    ".vbs",
}


@dataclass
class CacheEntry:
    label: str
    path: Path
    exists: bool = False
    total_bytes: int = 0
    file_count: int = 0
    children: List[str] = field(default_factory=list)


def _env_mapping(env: Optional[Mapping[str, str]] = None) -> Mapping[str, str]:
    return os.environ if env is None else env


def _home(env: Optional[Mapping[str, str]] = None) -> Path:
    source = _env_mapping(env)
    value = str(source.get("USERPROFILE") or source.get("HOME") or "").strip()
    return Path(value) if value else Path.home()


def _appdata_root(env: Optional[Mapping[str, str]] = None) -> Path:
    source = _env_mapping(env)
    root = str(source.get("APPDATA") or "").strip()
    if root:
        return Path(root) / "VideoSubtitleRemoverPro"
    return _home(source) / ".config" / "VideoSubtitleRemoverPro"


def app_model_cache_dir(env: Optional[Mapping[str, str]] = None) -> Path:
    return _appdata_root(env) / "models"


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


def _model_cache_roots(
    env: Optional[Mapping[str, str]] = None,
) -> list[tuple[str, Path]]:
    home = _home(env)
    return [
        ("app-model-cache", app_model_cache_dir(env)),
        ("torch-hub-checkpoints", home / ".cache" / "torch" / "hub" / "checkpoints"),
        ("simple-lama-cache", home / ".cache" / "simple_lama"),
        ("huggingface-hub", home / ".cache" / "huggingface" / "hub"),
        ("opencv-model-cache", home / ".cache" / "opencv_models"),
    ]


def _safe_posix_path(value: object) -> Optional[PurePosixPath]:
    text = str(value or "").replace("\\", "/").strip()
    if not text:
        return None
    path = PurePosixPath(text)
    if path.is_absolute():
        return None
    if any(part in {"", ".", ".."} or ":" in part for part in path.parts):
        return None
    return path


def _relative_posix(path: Path, root: Path) -> Optional[str]:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    parts = relative.parts
    if not parts or any(part in {"", ".", ".."} or ":" in part for part in parts):
        return None
    return PurePosixPath(*parts).as_posix()


def _model_file_rejection(path: Path, *, known_hash: bool) -> Optional[str]:
    suffix = path.suffix.lower()
    if suffix in _EXECUTABLE_SUFFIXES:
        return "executable code is not allowed in model-cache bundles"
    if known_hash:
        return None
    if suffix in _PICKLE_LIKE_MODEL_SUFFIXES:
        return "pickle-like model weights require a vendored hash"
    if suffix not in _SAFE_MODEL_SUFFIXES:
        return f"unsupported model file suffix: {suffix or '(none)'}"
    return None


def _iter_export_files(
    env: Optional[Mapping[str, str]] = None,
) -> Iterable[tuple[str, Path, Path]]:
    from backend.model_hashes import KNOWN_WEIGHT_HASHES

    known_names = set(KNOWN_WEIGHT_HASHES)
    for cache_name, root in _model_cache_roots(env):
        if not root.is_dir():
            continue
        try:
            iterator = root.rglob("*")
            for path in iterator:
                if not path.is_file():
                    continue
                if cache_name != "app-model-cache" and path.name not in known_names:
                    continue
                yield cache_name, root, path
        except OSError:
            continue


def _target_path_for_import(
    cache_name: str,
    relative_path: PurePosixPath,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> Optional[Path]:
    prefix = _MODEL_CACHE_ROOTS.get(cache_name)
    if prefix is None:
        return None
    root = app_model_cache_dir(env)
    parts = list(relative_path.parts)
    if prefix:
        parts = [prefix] + parts
    target = root.joinpath(*parts)
    try:
        root_resolved = root.resolve()
        target_resolved = target.resolve()
    except OSError:
        return None
    if target_resolved != root_resolved and root_resolved not in target_resolved.parents:
        return None
    return target


def model_cache_status(env: Optional[Mapping[str, str]] = None) -> dict:
    """Report known optional model assets without exposing full paths."""
    from backend.model_hashes import KNOWN_WEIGHT_HASHES, hash_file

    app_cache = app_model_cache_dir(env)
    roots = [path for _name, path in _model_cache_roots(env) if path.is_dir()]
    weights = []
    for filename, expected in sorted(KNOWN_WEIGHT_HASHES.items()):
        matches = []
        for root in roots:
            try:
                for match in root.rglob(filename):
                    if match.is_file():
                        matches.append(match)
            except OSError:
                continue
        if not matches:
            weights.append({
                "filename": filename,
                "status": "missing",
                "bytes": None,
                "sha256": None,
            })
            continue
        first = matches[0]
        try:
            actual = hash_file(first)
            size = first.stat().st_size
        except OSError:
            weights.append({
                "filename": filename,
                "status": "unreadable",
                "bytes": None,
                "sha256": None,
            })
            continue
        weights.append({
            "filename": filename,
            "status": "verified" if actual.lower() == expected.lower() else "mismatch",
            "bytes": size,
            "sha256": actual,
        })
    return {
        "schema": MODEL_CACHE_STATUS_SCHEMA,
        "app_model_cache_exists": app_cache.is_dir(),
        "known_weights": weights,
        "missing_known_filenames": [
            item["filename"] for item in weights
            if item["status"] == "missing"
        ],
        "verified_known_count": sum(
            1 for item in weights if item["status"] == "verified"
        ),
        "mismatch_known_filenames": [
            item["filename"] for item in weights
            if item["status"] == "mismatch"
        ],
    }


def export_model_cache_bundle(
    output_path: str | Path,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> dict:
    """Export safe model-cache files with a SHA-256 manifest."""
    from backend.model_hashes import KNOWN_WEIGHT_HASHES, hash_file

    output = Path(output_path)
    if output.suffix.lower() != ".zip":
        output = output.with_suffix(".zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    files = []
    skipped = []
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output.stem}.",
        suffix=".tmp",
        dir=str(output.parent),
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with zipfile.ZipFile(
            temp_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as bundle:
            for cache_name, root, path in _iter_export_files(env):
                relative = _relative_posix(path, root)
                if relative is None:
                    skipped.append({
                        "path": path.name,
                        "reason": "unsafe relative path",
                    })
                    continue
                known_expected = KNOWN_WEIGHT_HASHES.get(path.name)
                reason = _model_file_rejection(
                    path,
                    known_hash=known_expected is not None,
                )
                if reason:
                    skipped.append({"path": relative, "reason": reason})
                    continue
                try:
                    actual = hash_file(path)
                    size = path.stat().st_size
                except OSError as exc:
                    skipped.append({"path": relative, "reason": str(exc)})
                    continue
                if known_expected and actual.lower() != known_expected.lower():
                    skipped.append({
                        "path": relative,
                        "reason": "vendored hash mismatch",
                    })
                    continue
                archive_path = (
                    PurePosixPath("files") / cache_name / PurePosixPath(relative)
                ).as_posix()
                bundle.write(path, archive_path)
                files.append({
                    "cache": cache_name,
                    "relative_path": relative,
                    "archive_path": archive_path,
                    "filename": path.name,
                    "bytes": size,
                    "sha256": actual,
                    "known_sha256": known_expected,
                    "known_hash": known_expected is not None,
                })
            manifest = {
                "schema": PORTABLE_MODEL_CACHE_SCHEMA,
                "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "files": files,
                "skipped": skipped,
            }
            bundle.writestr(
                MODEL_CACHE_MANIFEST,
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            )
        os.replace(temp_path, output)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
    manifest["output"] = str(output)
    manifest["status_after_export"] = model_cache_status(env)
    return manifest


def _cleanup_path(path: Path) -> None:
    try:
        Path(path).unlink()
    except OSError:
        pass


def _target_key(target: Path) -> str:
    """Case/normalized identity for duplicate-target detection."""
    return os.path.normcase(os.path.abspath(str(target)))


def _stream_member_capped(
    bundle: zipfile.ZipFile,
    archive_name: str,
    dest: Path,
    *,
    cap_bytes: int,
) -> tuple:
    """Stream one member to ``dest`` with a hard byte cap.

    Returns ``(ok, sha256_hex, bytes_written, reason)``. Aborts (ok=False)
    when the decompressed stream exceeds ``cap_bytes`` so a member whose real
    size exceeds its declared size cannot exhaust the disk.
    """
    import hashlib

    digest = hashlib.sha256()
    written = 0
    limit = max(0, int(cap_bytes))
    try:
        with bundle.open(archive_name, "r") as src, dest.open("wb") as out:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit:
                    return False, "", written, "member exceeds declared size"
                digest.update(chunk)
                out.write(chunk)
    except (OSError, zipfile.BadZipFile, EOFError) as exc:
        return False, "", written, f"member extraction failed: {exc}"
    return True, digest.hexdigest(), written, ""


def import_model_cache_bundle(
    bundle_path: str | Path,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> dict:
    """Import a portable model-cache bundle into the app model cache.

    Two-phase and transactional: the whole bundle is preflighted against
    member-count / per-member size / compression-ratio / total-expansion /
    free-space ceilings, every accepted member is staged and hash-verified in
    a scratch directory, and only then are all staged files committed to their
    targets. A commit failure rolls back every file already moved so the cache
    is never left in a half-imported state.
    """
    from backend.model_hashes import KNOWN_WEIGHT_HASHES

    bundle_path = Path(bundle_path)
    imported: List[dict] = []
    rejected: List[dict] = []
    target_root = app_model_cache_dir(env)
    target_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(bundle_path, "r") as bundle:
        try:
            manifest = json.loads(bundle.read(MODEL_CACHE_MANIFEST))
        except (KeyError, json.JSONDecodeError) as exc:
            raise ValueError("model-cache bundle is missing a valid manifest") from exc
        if manifest.get("schema") != PORTABLE_MODEL_CACHE_SCHEMA:
            raise ValueError("unsupported model-cache bundle schema")
        entries = manifest.get("files", [])
        if not isinstance(entries, list):
            raise ValueError("model-cache manifest files must be a list")

        # ---- aggregate preflight (fail the whole import, commit nothing) ----
        if len(entries) > _IMPORT_MAX_MEMBERS:
            raise ValueError(
                f"bundle declares {len(entries)} members, exceeding the "
                f"{_IMPORT_MAX_MEMBERS} member ceiling"
            )
        infos = {info.filename: info for info in bundle.infolist()}
        names = set(infos)
        declared_total = 0
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            arch = _safe_posix_path(entry.get("archive_path"))
            info = infos.get(arch.as_posix()) if arch is not None else None
            if info is not None:
                declared_total += max(0, int(info.file_size))
        if declared_total > _IMPORT_MAX_TOTAL_BYTES:
            raise ValueError(
                f"bundle total expansion {declared_total} bytes exceeds the "
                f"{_IMPORT_MAX_TOTAL_BYTES}-byte ceiling"
            )
        try:
            free_bytes = shutil.disk_usage(target_root).free
        except OSError:
            free_bytes = None
        if (free_bytes is not None
                and free_bytes < declared_total + _IMPORT_FREE_SPACE_SLACK_BYTES):
            raise ValueError(
                f"insufficient free space to import: need about "
                f"{declared_total + _IMPORT_FREE_SPACE_SLACK_BYTES} bytes, "
                f"have {free_bytes}"
            )

        # ---- stage each accepted member into a scratch dir ----
        staged: List[dict] = []
        seen_targets: set = set()
        staging_dir = Path(tempfile.mkdtemp(prefix=".vsr-import-", dir=str(target_root)))
        try:
            for index, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    rejected.append({"path": "", "reason": "manifest entry is not an object"})
                    continue
                cache_name = str(entry.get("cache") or "")
                relative = _safe_posix_path(entry.get("relative_path"))
                archive_path = _safe_posix_path(entry.get("archive_path"))
                filename = str(entry.get("filename") or "")
                if relative is None or archive_path is None:
                    rejected.append({"path": filename, "reason": "unsafe manifest path"})
                    continue
                info = infos.get(archive_path.as_posix())
                if info is None:
                    rejected.append({
                        "path": archive_path.as_posix(),
                        "reason": "archive member is missing",
                    })
                    continue
                if PurePosixPath(filename).name != filename or filename != relative.name:
                    rejected.append({"path": relative.as_posix(), "reason": "filename mismatch"})
                    continue
                known_expected = KNOWN_WEIGHT_HASHES.get(filename)
                reason = _model_file_rejection(
                    Path(filename),
                    known_hash=known_expected is not None,
                )
                if reason:
                    rejected.append({"path": relative.as_posix(), "reason": reason})
                    continue
                declared = max(0, int(info.file_size))
                if declared > _IMPORT_MAX_MEMBER_BYTES:
                    rejected.append({
                        "path": relative.as_posix(),
                        "reason": "member exceeds per-file size ceiling",
                    })
                    continue
                ratio = declared / max(1, int(info.compress_size))
                if ratio > _IMPORT_MAX_COMPRESSION_RATIO:
                    rejected.append({
                        "path": relative.as_posix(),
                        "reason": "suspicious compression ratio (possible zip bomb)",
                    })
                    continue
                target = _target_path_for_import(cache_name, relative, env=env)
                if target is None:
                    rejected.append({"path": relative.as_posix(), "reason": "unsupported cache target"})
                    continue
                key = _target_key(target)
                if key in seen_targets:
                    rejected.append({"path": relative.as_posix(), "reason": "duplicate import target"})
                    continue
                manifest_hash = str(entry.get("sha256") or "").lower()
                expected_hash = (known_expected or manifest_hash).lower()
                if len(expected_hash) != 64:
                    rejected.append({"path": relative.as_posix(), "reason": "missing SHA-256"})
                    continue
                temp_path = staging_dir / f"{index:04d}_{filename}"
                cap = min(declared, _IMPORT_MAX_MEMBER_BYTES)
                ok, actual, _written, fail = _stream_member_capped(
                    bundle, archive_path.as_posix(), temp_path, cap_bytes=cap,
                )
                if not ok:
                    _cleanup_path(temp_path)
                    rejected.append({"path": relative.as_posix(), "reason": fail})
                    continue
                if actual.lower() != expected_hash:
                    _cleanup_path(temp_path)
                    rejected.append({"path": relative.as_posix(), "reason": "SHA-256 mismatch"})
                    continue
                seen_targets.add(key)
                staged.append({
                    "cache": cache_name,
                    "relative": relative,
                    "target": target,
                    "temp_path": temp_path,
                    "sha256": actual,
                    "known_hash": known_expected is not None,
                })

            # ---- commit phase (all-or-nothing with rollback) ----
            committed: List[tuple] = []
            try:
                for item in staged:
                    target = item["target"]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    backup = None
                    if target.exists():
                        backup = target.with_name(target.name + ".vsrbak")
                        _cleanup_path(backup)
                        os.replace(target, backup)
                    os.replace(item["temp_path"], target)
                    committed.append((target, backup))
                    imported.append({
                        "cache": item["cache"],
                        "relative_path": item["relative"].as_posix(),
                        "target": str(target),
                        "bytes": target.stat().st_size,
                        "sha256": item["sha256"],
                        "known_hash": item["known_hash"],
                    })
            except OSError as exc:
                for target, backup in reversed(committed):
                    _cleanup_path(target)
                    if backup is not None and backup.exists():
                        try:
                            os.replace(backup, target)
                        except OSError:
                            pass
                imported.clear()
                raise ValueError(
                    f"model-cache import failed and was rolled back: {exc}"
                ) from exc
            else:
                for _target, backup in committed:
                    if backup is not None:
                        _cleanup_path(backup)
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

    return {
        "schema": PORTABLE_MODEL_CACHE_SCHEMA,
        "source": str(bundle_path),
        "imported": imported,
        "rejected": rejected,
        "limits": {
            "max_members": _IMPORT_MAX_MEMBERS,
            "max_member_bytes": _IMPORT_MAX_MEMBER_BYTES,
            "max_total_bytes": _IMPORT_MAX_TOTAL_BYTES,
            "max_compression_ratio": _IMPORT_MAX_COMPRESSION_RATIO,
        },
        "status_after_import": model_cache_status(env),
    }


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

    for name in sorted(str(target) for target in targets):
        if name not in _CLEANABLE_SUBDIRS:
            logger.warning("Skipping unsupported cache-clean target: %s", name)
            continue
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
