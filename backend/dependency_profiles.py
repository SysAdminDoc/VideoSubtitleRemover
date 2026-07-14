"""Reviewed, reproducible dependency profiles and release evidence."""

from __future__ import annotations

import argparse
import difflib
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "dependency_profiles.json"
PROFILE_DIR = ROOT / "dependency_profiles"
PROFILE_SCHEMA = "vsr.dependency_profiles.v1"
PROFILE_STATUS_SCHEMA = "vsr.dependency_profile_status.v1"
PROFILE_ENV = "VSR_DEPENDENCY_PROFILE"
SUPPORTED_PROFILES = ("cpu", "nvidia", "directml")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def load_profile_manifest(path: str | Path = MANIFEST_PATH) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != PROFILE_SCHEMA:
        raise ValueError(f"Unsupported dependency profile manifest: {manifest_path}")
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict) or set(profiles) != set(SUPPORTED_PROFILES):
        raise ValueError("Dependency manifest must define CPU, NVIDIA, and DirectML profiles")
    common = payload.get("commonConstraints")
    if not isinstance(common, list) or not all(isinstance(item, str) for item in common):
        raise ValueError("commonConstraints must be a list of requirement strings")
    hashes = payload.get("reviewedArtifactHashes", {})
    if not isinstance(hashes, dict) or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value.lower())
        for value in hashes.values()
    ):
        raise ValueError("Reviewed artifact hashes must be lowercase SHA-256 digests")
    for name in SUPPORTED_PROFILES:
        profile = profiles[name]
        if not isinstance(profile, dict):
            raise ValueError(f"Profile {name!r} must be an object")
        constraints = profile.get("constraints")
        indexes = profile.get("indexes")
        if not isinstance(constraints, list) or not all(
            isinstance(item, str) for item in constraints
        ):
            raise ValueError(f"Profile {name!r} constraints are invalid")
        if not isinstance(indexes, list) or not all(
            isinstance(item, str) and item.startswith("https://") for item in indexes
        ):
            raise ValueError(f"Profile {name!r} indexes are invalid")
    return payload


def manifest_sha256(path: str | Path = MANIFEST_PATH) -> str:
    return _sha256_file(Path(path))


def profile_constraint_path(
    name: str,
    profile_dir: str | Path = PROFILE_DIR,
) -> Path:
    if name not in SUPPORTED_PROFILES:
        raise ValueError(f"Unsupported dependency profile: {name!r}")
    return Path(profile_dir) / f"{name}.txt"


def select_profile(gpu_info: Mapping[str, object] | None = None) -> str:
    info = dict(gpu_info or {})
    if info.get("nvidia") and not info.get("cuda_disabled_by_python"):
        return "nvidia"
    if info.get("amd") or info.get("intel"):
        return "directml"
    return "cpu"


def render_profile(
    name: str,
    manifest: Mapping[str, Any] | None = None,
    *,
    manifest_path: str | Path = MANIFEST_PATH,
) -> str:
    payload = dict(manifest or load_profile_manifest(manifest_path))
    profile = payload["profiles"][name]
    digest = manifest_sha256(manifest_path)
    lines = [
        "# Generated file. Edit dependency_profiles.json, then run:",
        "#   python -m backend.dependency_profiles update",
        f"# Profile: {name}",
        f"# Provider: {profile['provider']}",
        f"# Reviewed: {payload['reviewedAt']}",
        f"# Manifest-SHA256: {digest}",
        "#",
        "# Universal artifacts reviewed with a stable SHA-256:",
    ]
    for requirement, artifact_hash in sorted(
        payload.get("reviewedArtifactHashes", {}).items(),
        key=lambda item: item[0].lower(),
    ):
        lines.append(f"#   {requirement} sha256={artifact_hash}")
    lines.extend(("", "# Exact reviewed direct-dependency constraints."))
    constraints = list(payload["commonConstraints"]) + list(profile["constraints"])
    seen: set[str] = set()
    for constraint in constraints:
        normalized = constraint.strip().lower()
        if normalized and normalized not in seen:
            lines.append(constraint.strip())
            seen.add(normalized)
    return "\n".join(lines) + "\n"


def profile_diffs(
    *,
    manifest_path: str | Path = MANIFEST_PATH,
    profile_dir: str | Path = PROFILE_DIR,
) -> dict[str, str]:
    manifest = load_profile_manifest(manifest_path)
    changes: dict[str, str] = {}
    for name in SUPPORTED_PROFILES:
        path = profile_constraint_path(name, profile_dir)
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        expected = render_profile(name, manifest, manifest_path=manifest_path)
        if current == expected:
            continue
        changes[name] = "".join(difflib.unified_diff(
            current.splitlines(keepends=True),
            expected.splitlines(keepends=True),
            fromfile=str(path),
            tofile=f"{path} (generated)",
        ))
    return changes


def update_profiles(
    *,
    manifest_path: str | Path = MANIFEST_PATH,
    profile_dir: str | Path = PROFILE_DIR,
    write: bool = True,
) -> dict[str, str]:
    changes = profile_diffs(
        manifest_path=manifest_path,
        profile_dir=profile_dir,
    )
    if write and changes:
        target_dir = Path(profile_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        manifest = load_profile_manifest(manifest_path)
        for name in changes:
            profile_constraint_path(name, target_dir).write_text(
                render_profile(name, manifest, manifest_path=manifest_path),
                encoding="utf-8",
                newline="\n",
            )
    return changes


def ensure_profile_current(
    name: str,
    *,
    manifest_path: str | Path = MANIFEST_PATH,
    profile_dir: str | Path = PROFILE_DIR,
) -> Path:
    path = profile_constraint_path(name, profile_dir)
    changes = profile_diffs(
        manifest_path=manifest_path,
        profile_dir=profile_dir,
    )
    if name in changes:
        raise RuntimeError(
            f"Dependency profile {name!r} is missing or stale. Run "
            "`python -m backend.dependency_profiles update`."
        )
    return path


def _installed_provider_profile(
    package_versions: Mapping[str, str] | None = None,
) -> str:
    versions = {
        str(name).lower().replace("_", "-"): str(version)
        for name, version in (package_versions or {}).items()
    }
    if not versions:
        for package in ("onnxruntime-directml", "onnxruntime-gpu"):
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                pass
    if versions.get("onnxruntime-directml"):
        return "directml"
    if versions.get("onnxruntime-gpu"):
        return "nvidia"
    return "cpu"


def collect_dependency_profile_status(
    *,
    profile: str | None = None,
    env: Mapping[str, str] | None = None,
    package_versions: Mapping[str, str] | None = None,
    manifest_path: str | Path = MANIFEST_PATH,
    profile_dir: str | Path = PROFILE_DIR,
) -> dict[str, Any]:
    environment = os.environ if env is None else env
    requested = str(profile or environment.get(PROFILE_ENV, "")).strip().lower()
    source = "explicit" if requested else "installed-provider"
    name = requested or _installed_provider_profile(package_versions)
    errors: list[str] = []
    if name not in SUPPORTED_PROFILES:
        errors.append(f"Unsupported dependency profile {name!r}")
        name = "cpu"
    try:
        manifest = load_profile_manifest(manifest_path)
        expected = render_profile(name, manifest, manifest_path=manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "schema": PROFILE_STATUS_SCHEMA,
            "profile": name,
            "source": source,
            "valid": False,
            "errors": [str(exc)],
            "manifestSha256": "",
            "constraintSha256": "",
            "intentionalExceptions": [],
        }
    path = profile_constraint_path(name, profile_dir)
    actual = path.read_text(encoding="utf-8") if path.exists() else ""
    if not path.exists():
        errors.append(f"Constraint file is missing: {path}")
    elif actual != expected:
        errors.append(f"Constraint file is stale: {path}")
    profile_data = manifest["profiles"][name]
    return {
        "schema": PROFILE_STATUS_SCHEMA,
        "profile": name,
        "source": source,
        "provider": profile_data["provider"],
        "valid": not errors,
        "errors": errors,
        "manifest": _display_path(Path(manifest_path)),
        "manifestSha256": manifest_sha256(manifest_path),
        "constraintFile": _display_path(path),
        "constraintSha256": _sha256_file(path) if path.exists() else "",
        "reviewedAt": manifest["reviewedAt"],
        "indexes": list(profile_data["indexes"]),
        "reviewedArtifactHashes": dict(manifest.get("reviewedArtifactHashes", {})),
        "intentionalExceptions": list(manifest.get("intentionalExceptions", [])),
    }


def _print_diffs(changes: Mapping[str, str]) -> None:
    for name in SUPPORTED_PROFILES:
        diff = changes.get(name)
        if diff:
            print(diff, end="" if diff.endswith("\n") else "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check, regenerate, or inspect reviewed dependency profiles."
    )
    parser.add_argument("command", choices=("check", "update", "status"))
    parser.add_argument("--profile", choices=SUPPORTED_PROFILES)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--profile-dir", default=str(PROFILE_DIR))
    args = parser.parse_args(argv)

    if args.command == "status":
        print(json.dumps(collect_dependency_profile_status(
            profile=args.profile,
            manifest_path=args.manifest,
            profile_dir=args.profile_dir,
        ), indent=2, sort_keys=True))
        return 0

    changes = update_profiles(
        manifest_path=args.manifest,
        profile_dir=args.profile_dir,
        write=args.command == "update" and not args.dry_run,
    )
    _print_diffs(changes)
    if not changes:
        print("Dependency profiles are current.")
        return 0
    if args.command == "check" or args.dry_run:
        return 1
    print("Dependency profiles updated.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
