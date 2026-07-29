"""Atomic, version-derived staging for a local release artifact set.

RM-141: `build/release/` used to be a reusable scratch directory. Every build
overwrote a subset of it, so a newer installer could sit beside an older
portable ZIP and a `SHA256SUMS.txt` that described neither -- which makes the
published "verify the checksums" instructions actively misleading.

This module stages a release into a fresh temporary directory, derives every
filename from `APP_VERSION`, refuses evidence that describes a different
version or a failed smoke, hashes exactly the staged set, and only then
promotes the whole directory into `build/release/<version>/` in one move.

Releases remain explicitly unsigned; see `publication_guidance()`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence
import zipfile


RELEASE_STAGE_SCHEMA = "vsr.release_stage.v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE_ROOT = ROOT / "build" / "release"
CHECKSUM_NAME = "SHA256SUMS.txt"

# Evidence produced by backend.release_verification. Every one of these is
# required: a release whose audit or SBOM is missing cannot be verified.
EVIDENCE_FILES: tuple[str, ...] = (
    "release-verification.json",
    "release-advisories.json",
    "release-hidden-imports.json",
    "pip-audit.json",
    "sbom.cdx.json",
)


class ReleaseStagingError(RuntimeError):
    """A release stage could not be built or did not verify."""


def installer_asset_name(version: str) -> str:
    return f"VideoSubtitleRemoverPro-{version}-Setup.exe"


def portable_asset_name(version: str) -> str:
    return f"VideoSubtitleRemoverPro-{version}-Windows-x64.zip"


def expected_assets(version: str) -> tuple[str, ...]:
    """Every filename a complete release directory must contain, and no more."""
    return tuple(sorted((
        installer_asset_name(version),
        portable_asset_name(version),
        CHECKSUM_NAME,
        *EVIDENCE_FILES,
    )))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_evidence(evidence_dir: Path) -> dict[str, Any]:
    path = evidence_dir / "release-verification.json"
    if not path.is_file():
        raise ReleaseStagingError(f"Release evidence is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseStagingError(f"Release evidence is unreadable: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ReleaseStagingError("Release evidence is not an object")
    return dict(payload)


def evidence_problems(evidence: Mapping[str, Any], version: str) -> list[str]:
    """Reject evidence that does not describe this exact, fully smoked build."""
    problems: list[str] = []
    app = evidence.get("app")
    recorded = str(app.get("version") or "") if isinstance(app, Mapping) else ""
    if recorded != version:
        problems.append(
            f"release evidence records version {recorded or 'unknown'!r}, "
            f"expected {version!r}"
        )
    checks = evidence.get("versionChecks")
    if isinstance(checks, Mapping):
        checked = str(checks.get("appVersion") or "")
        if checked and checked != version:
            problems.append(
                f"version checks ran against {checked!r}, expected {version!r}"
            )
    errors = evidence.get("errors")
    if isinstance(errors, list) and errors:
        problems.append(
            f"release evidence reports {len(errors)} verification error(s)"
        )
    for key, label in (
        ("installerSmoke", "installer payload smoke"),
        ("smokeLaunch", "frozen launch smoke"),
    ):
        section = evidence.get(key)
        if not isinstance(section, Mapping):
            problems.append(f"{label} evidence is missing")
            continue
        if not section.get("passed"):
            problems.append(f"{label} did not pass")
    installer = evidence.get("installer")
    if not isinstance(installer, Mapping) or not installer.get(
            "validPortableExecutable"):
        problems.append("installer artifact evidence is missing or invalid")
    return problems


def build_portable_zip(dist_dir: str | Path, target: str | Path) -> Path:
    """Zip the frozen distribution folder as the portable release asset."""
    source = Path(dist_dir)
    if not source.is_dir():
        raise ReleaseStagingError(f"Distribution folder is missing: {source}")
    destination = Path(target)
    files = sorted(
        (item for item in source.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(source).as_posix(),
    )
    if not files:
        raise ReleaseStagingError(f"Distribution folder is empty: {source}")
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True,
    ) as archive:
        for item in files:
            arcname = Path(source.name) / item.relative_to(source)
            archive.write(item, arcname.as_posix())
    return destination


def write_checksums(stage: Path, names: Sequence[str]) -> dict[str, str]:
    """Hash exactly ``names`` and write the checksum manifest beside them."""
    digests = {name: _sha256_file(stage / name) for name in sorted(names)}
    (stage / CHECKSUM_NAME).write_text(
        "".join(f"{digest}  {name}\n" for name, digest in digests.items()),
        encoding="utf-8",
        newline="\n",
    )
    return digests


def parse_checksums(text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        digest, _, name = stripped.partition("  ")
        if not name:
            digest, _, name = stripped.partition(" ")
        if digest and name:
            parsed[name.strip()] = digest.strip().lower()
    return parsed


def verify_release_dir(directory: str | Path, version: str) -> dict[str, Any]:
    """Confirm a promoted release directory is exactly one versioned set."""
    path = Path(directory)
    problems: list[str] = []
    if not path.is_dir():
        return {
            "schema": RELEASE_STAGE_SCHEMA,
            "version": version,
            "directory": str(path),
            "valid": False,
            "problems": [f"release directory is missing: {path}"],
            "assets": [],
        }
    expected = set(expected_assets(version))
    present = {item.name for item in path.iterdir()}
    for missing in sorted(expected - present):
        problems.append(f"missing release asset: {missing}")
    for extra in sorted(present - expected):
        problems.append(f"unexpected asset in release directory: {extra}")
    checksum_path = path / CHECKSUM_NAME
    digests: dict[str, str] = {}
    if checksum_path.is_file():
        digests = parse_checksums(checksum_path.read_text(encoding="utf-8"))
        hashed = expected - {CHECKSUM_NAME}
        for missing in sorted(hashed - set(digests)):
            problems.append(f"{CHECKSUM_NAME} does not cover {missing}")
        for extra in sorted(set(digests) - hashed):
            problems.append(f"{CHECKSUM_NAME} lists an unknown asset: {extra}")
        for name in sorted(hashed & set(digests)):
            candidate = path / name
            if not candidate.is_file():
                continue
            actual = _sha256_file(candidate)
            if actual != digests[name]:
                problems.append(f"checksum mismatch for {name}")
    return {
        "schema": RELEASE_STAGE_SCHEMA,
        "version": version,
        "directory": str(path),
        "valid": not problems,
        "problems": problems,
        "assets": sorted(present),
        "checksums": digests,
    }


def stale_release_artifacts(
    release_root: str | Path,
    version: str,
) -> list[str]:
    """Loose files in the release root left over from the pre-versioned layout.

    Promoted sets live in ``<release root>/<version>/``. Anything else sitting
    loose in the root is a leftover from an older build, except the evidence
    inputs the current build just produced.
    """
    root = Path(release_root)
    if not root.is_dir():
        return []
    keep = set(EVIDENCE_FILES)
    return sorted(
        item.name for item in root.iterdir()
        if item.is_file() and item.name not in keep
    )


def stage_release(
    version: str,
    *,
    dist_dir: str | Path,
    installer_path: str | Path,
    evidence_dir: str | Path,
    release_root: str | Path = DEFAULT_RELEASE_ROOT,
    stage_root: str | Path | None = None,
    prune_stale: bool = False,
) -> dict[str, Any]:
    """Stage, hash, verify, and promote one complete versioned release set."""
    version = str(version).strip()
    if not version:
        raise ReleaseStagingError("A release version is required")
    installer = Path(installer_path)
    if not installer.is_file():
        raise ReleaseStagingError(f"Installer artifact is missing: {installer}")
    evidence_path = Path(evidence_dir)
    evidence = _read_evidence(evidence_path)
    problems = evidence_problems(evidence, version)
    if problems:
        raise ReleaseStagingError(
            "Release evidence does not describe a promotable build:\n- "
            + "\n- ".join(problems)
        )
    missing_evidence = [
        name for name in EVIDENCE_FILES if not (evidence_path / name).is_file()
    ]
    if missing_evidence:
        raise ReleaseStagingError(
            "Release evidence set is incomplete: " + ", ".join(missing_evidence)
        )

    root = Path(release_root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / version
    stage_parent = Path(stage_root) if stage_root else root
    stage_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".stage-{version}-", dir=str(stage_parent)))
    try:
        shutil.copy2(installer, stage / installer_asset_name(version))
        build_portable_zip(dist_dir, stage / portable_asset_name(version))
        for name in EVIDENCE_FILES:
            shutil.copy2(evidence_path / name, stage / name)
        hashed = [name for name in expected_assets(version) if name != CHECKSUM_NAME]
        digests = write_checksums(stage, hashed)
        report = verify_release_dir(stage, version)
        if not report["valid"]:
            raise ReleaseStagingError(
                "Staged release did not verify:\n- "
                + "\n- ".join(report["problems"])
            )
        if target.exists():
            replaced = Path(tempfile.mkdtemp(
                prefix=f".replaced-{version}-", dir=str(stage_parent)))
            os.replace(target, replaced / "previous")
            shutil.rmtree(replaced, ignore_errors=True)
        os.replace(stage, target)
        stage = None  # type: ignore[assignment]
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)

    stale = stale_release_artifacts(root, version)
    if prune_stale:
        for name in stale:
            try:
                (root / name).unlink()
            except OSError:
                pass
    return {
        "schema": RELEASE_STAGE_SCHEMA,
        "version": version,
        "directory": str(target),
        "assets": list(expected_assets(version)),
        "checksums": digests,
        "prunedStaleArtifacts": stale if prune_stale else [],
        "staleArtifacts": [] if prune_stale else stale,
        "valid": True,
    }


def publication_guidance(version: str) -> list[str]:
    """Publication steps for an unsigned, immutable GitHub release."""
    return [
        f"gh release create v{version} --draft --title \"v{version}\" "
        "--notes-file <changelog excerpt>",
        f"gh release upload v{version} build/release/{version}/* --clobber",
        "Review the draft, then publish it. Enable immutable releases on the "
        "repository so a published tag's assets can never be replaced.",
        "Artifacts are intentionally UNSIGNED. Do not acquire or apply a "
        "code-signing certificate; publish the SHA256SUMS.txt from the same "
        "staged set as the only integrity reference.",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage and verify one versioned release artifact set."
    )
    parser.add_argument("command", choices=("stage", "verify", "guidance"))
    parser.add_argument("--version", default="")
    parser.add_argument("--dist-dir", default="dist/VideoSubtitleRemoverPro")
    parser.add_argument("--installer-path", default="")
    parser.add_argument("--evidence-dir", default=str(DEFAULT_RELEASE_ROOT))
    parser.add_argument("--release-root", default=str(DEFAULT_RELEASE_ROOT))
    parser.add_argument(
        "--prune-stale",
        action="store_true",
        help="Delete loose pre-versioned artifacts from the release root.",
    )
    args = parser.parse_args(argv)

    version = args.version.strip()
    if not version:
        from gui.config import APP_VERSION

        version = APP_VERSION

    if args.command == "guidance":
        for line in publication_guidance(version):
            print(line)
        return 0

    if args.command == "verify":
        report = verify_release_dir(Path(args.release_root) / version, version)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["valid"] else 1

    try:
        report = stage_release(
            version,
            dist_dir=args.dist_dir,
            installer_path=args.installer_path,
            evidence_dir=args.evidence_dir,
            release_root=args.release_root,
            prune_stale=args.prune_stale,
        )
    except ReleaseStagingError as exc:
        print(f"Release staging failed: {exc}")
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
