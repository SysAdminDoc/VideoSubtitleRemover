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


# RM-350: the download used to be one generically named build that was
# effectively CPU-only while the product recommended NVIDIA. The lane is now
# part of the filename, and `evidence_problems` refuses to promote an
# artifact whose recorded provider does not match the lane its name claims.
DEFAULT_PROFILE = "cpu"

# The lane published as a tested bundle. DirectML stays a supported profile
# for a local install and is deliberately not shipped as an artifact: nothing
# here has ever measured it on the hardware it targets.
PUBLISHED_PROFILES: tuple[str, ...] = ("cpu", "nvidia")

# Lanes that ship an NSIS installer as well as a portable ZIP.
#
# The NSIS compiler is a 32-bit program and maps its payload into a 32-bit
# address space, so an installer cannot exceed about 2 GB. Compiling the CUDA
# lane fails there with "Internal compiler error #12345: error mmapping file
# (2078463402, 33554432) is out of range" against a 3.1 GB payload. That size
# is not slack: the CUDA execution provider loads its runtime out of the
# torch cu130 wheel, where cuBLASLt is 456 MB and cuFFT is 272 MB on their
# own. The lane therefore ships as a portable ZIP, which uses ZIP64 and has
# no such ceiling. This is a property of the packager, not a missing step, so
# it is declared here rather than discovered by a failed build.
INSTALLER_PROFILES: tuple[str, ...] = ("cpu",)


def ships_installer(profile: str) -> bool:
    return normalize_release_profile(profile) in INSTALLER_PROFILES


def normalize_release_profile(profile: object) -> str:
    from backend.build_profile import normalize_profile

    name = normalize_profile(profile)
    if not name:
        raise ReleaseStagingError(
            f"Unsupported release profile: {profile!r}")
    return name


def installer_asset_name(version: str,
                         profile: str = DEFAULT_PROFILE) -> str:
    return (f"VideoSubtitleRemoverPro-{version}-"
            f"{normalize_release_profile(profile)}-Setup.exe")


def portable_asset_name(version: str,
                        profile: str = DEFAULT_PROFILE) -> str:
    return (f"VideoSubtitleRemoverPro-{version}-"
            f"{normalize_release_profile(profile)}-Windows-x64.zip")


def expected_assets(version: str,
                    profile: str = DEFAULT_PROFILE) -> tuple[str, ...]:
    """Every filename a complete release directory must contain, and no more."""
    names = [
        portable_asset_name(version, profile),
        CHECKSUM_NAME,
        *EVIDENCE_FILES,
    ]
    if ships_installer(profile):
        names.append(installer_asset_name(version, profile))
    return tuple(sorted(names))


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


def profile_problems(evidence: Mapping[str, Any], profile: str) -> list[str]:
    """Reject evidence whose provider does not match the name being claimed.

    RM-350: a filename is a claim about what is inside. Two independent
    records have to agree with it before the artifact is promoted: the
    profile stamped into the frozen bundle at freeze time, and the provider
    ONNX Runtime actually activated when that frozen bundle ran. A CPU
    payload with an NVIDIA name fails here rather than on a user's machine.
    """
    from backend.build_profile import declared_provider

    name = normalize_release_profile(profile)
    problems: list[str] = []
    expected_provider = declared_provider(name)

    frozen = evidence.get("frozenProviderSmoke")
    if not isinstance(frozen, Mapping):
        return [
            f"the artifact claims the {name} lane but carries no frozen "
            "provider evidence to support it"
        ]
    stamped = str(frozen.get("profile") or "")
    if stamped != name:
        problems.append(
            f"the artifact name claims the {name} lane but the frozen build "
            f"is stamped {stamped or 'unstamped'!r}"
        )
    if str(frozen.get("profileSource") or "") != "stamp":
        problems.append(
            "the frozen build's profile was inferred rather than stamped in "
            "at freeze time, so it is not evidence of what was built"
        )
    if not frozen.get("ran"):
        problems.append(
            "the frozen provider smoke did not run, so nothing measured "
            "which provider this artifact selects"
        )
    active = [str(item) for item in (frozen.get("activeProviders") or [])]
    if expected_provider and expected_provider not in active:
        problems.append(
            f"the artifact claims the {name} lane, whose provider is "
            f"{expected_provider}, but the frozen build activated "
            + (", ".join(active) if active else "no provider")
        )
    if frozen.get("fellBack"):
        problems.append(
            f"the frozen build fell back off {expected_provider}, so the "
            f"{name} name would be untrue"
        )
    if not frozen.get("passed"):
        problems.append("the frozen provider smoke did not pass")
    return problems


def evidence_problems(evidence: Mapping[str, Any], version: str,
                      profile: str = DEFAULT_PROFILE) -> list[str]:
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
    required_smokes = [("smokeLaunch", "frozen launch smoke")]
    if ships_installer(profile):
        required_smokes.insert(
            0, ("installerSmoke", "installer payload smoke"))
    for key, label in required_smokes:
        section = evidence.get(key)
        if not isinstance(section, Mapping):
            problems.append(f"{label} evidence is missing")
            continue
        if not section.get("passed"):
            problems.append(f"{label} did not pass")
    if ships_installer(profile):
        installer = evidence.get("installer")
        if not isinstance(installer, Mapping) or not installer.get(
                "validPortableExecutable"):
            problems.append(
                "installer artifact evidence is missing or invalid")
    problems.extend(profile_problems(evidence, profile))
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


# GitHub refuses a single release asset at or above 2 GiB. The NVIDIA lane's
# portable ZIP is past it (2.119 GiB at 3.41.0) and cannot be compressed
# under: deflate level 9 buys 0.55 percent over level 6 on these DLLs, and the
# payload cannot be trimmed because torch loads its whole CUDA set eagerly at
# import. So the asset is uploaded in parts and rejoined by the user.
GITHUB_ASSET_LIMIT_BYTES = 2 * 1024 ** 3
SPLIT_PART_BYTES = 1536 * 1024 ** 2


def split_asset_for_upload(
    path: str | Path,
    *,
    limit: int = GITHUB_ASSET_LIMIT_BYTES,
    part_bytes: int = SPLIT_PART_BYTES,
) -> list[Path]:
    """Split one oversized asset into ``.001``, ``.002`` upload parts.

    Returns an empty list when the file is already small enough, so a caller
    can treat "no parts" as "upload the file itself". Concatenating the parts
    in name order reproduces the original byte for byte, which is what the
    whole-file digest in ``SHA256SUMS.txt`` still refers to.
    """
    source = Path(path)
    if not source.is_file():
        raise ReleaseStagingError(f"Asset to split is missing: {source}")
    size = source.stat().st_size
    if size < limit:
        return []
    if part_bytes <= 0 or part_bytes >= limit:
        raise ReleaseStagingError(
            f"Split part size {part_bytes} must be below the {limit} limit"
        )

    for stale in sorted(source.parent.glob(f"{source.name}.[0-9][0-9][0-9]")):
        stale.unlink()

    parts: list[Path] = []
    with source.open("rb") as handle:
        index = 1
        while True:
            chunk = handle.read(part_bytes)
            if not chunk:
                break
            part = source.with_name(f"{source.name}.{index:03d}")
            part.write_bytes(chunk)
            parts.append(part)
            index += 1
    return parts


def rejoin_split_asset(first_part: str | Path, target: str | Path) -> Path:
    """Concatenate ``.001``, ``.002`` parts back into one file.

    The product does not need this at runtime; it exists so the split has a
    tested inverse rather than a documented hope.
    """
    start = Path(first_part)
    base = start.with_suffix("")
    parts = sorted(base.parent.glob(f"{base.name}.[0-9][0-9][0-9]"))
    if not parts:
        raise ReleaseStagingError(f"No split parts found beside {start}")
    destination = Path(target)
    with destination.open("wb") as out:
        for part in parts:
            with part.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    out.write(chunk)
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


def verify_release_dir(directory: str | Path, version: str,
                       profile: str = DEFAULT_PROFILE) -> dict[str, Any]:
    """Confirm a promoted release directory is exactly one versioned set."""
    path = Path(directory)
    name = normalize_release_profile(profile)
    problems: list[str] = []
    if not path.is_dir():
        return {
            "schema": RELEASE_STAGE_SCHEMA,
            "version": version,
            "profile": name,
            "directory": str(path),
            "valid": False,
            "problems": [f"release directory is missing: {path}"],
            "assets": [],
        }
    expected = set(expected_assets(version, name))
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
        "profile": name,
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

    Promoted sets live in ``<release root>/<version>/<profile>/``. Anything
    else sitting loose in the root is a leftover from an older build, except
    the evidence inputs the current build just produced.
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
    profile: str = DEFAULT_PROFILE,
) -> dict[str, Any]:
    """Stage, hash, verify, and promote one complete versioned release set."""
    version = str(version).strip()
    if not version:
        raise ReleaseStagingError("A release version is required")
    name = normalize_release_profile(profile)
    installer = Path(installer_path) if installer_path else None
    if ships_installer(name):
        if installer is None or not installer.is_file():
            raise ReleaseStagingError(
                f"Installer artifact is missing: {installer or '(none given)'}")
    elif installer is not None and installer.is_file():
        raise ReleaseStagingError(
            f"The {name} lane ships no installer, but one was passed: "
            f"{installer}. See INSTALLER_PROFILES for why."
        )
    evidence_path = Path(evidence_dir)
    evidence = _read_evidence(evidence_path)
    problems = evidence_problems(evidence, version, name)
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
    # One directory per lane, so a CPU set and a CUDA set of the same version
    # cannot overwrite each other or share a checksum manifest.
    target = root / version / name
    target.parent.mkdir(parents=True, exist_ok=True)
    stage_parent = Path(stage_root) if stage_root else root
    stage_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(
        prefix=f".stage-{version}-{name}-", dir=str(stage_parent)))
    try:
        if ships_installer(name):
            shutil.copy2(
                installer, stage / installer_asset_name(version, name))
        build_portable_zip(dist_dir, stage / portable_asset_name(version, name))
        for evidence_name in EVIDENCE_FILES:
            shutil.copy2(evidence_path / evidence_name, stage / evidence_name)
        hashed = [item for item in expected_assets(version, name)
                  if item != CHECKSUM_NAME]
        digests = write_checksums(stage, hashed)
        report = verify_release_dir(stage, version, name)
        if not report["valid"]:
            raise ReleaseStagingError(
                "Staged release did not verify:\n- "
                + "\n- ".join(report["problems"])
            )
        if target.exists():
            replaced = Path(tempfile.mkdtemp(
                prefix=f".replaced-{version}-{name}-", dir=str(stage_parent)))
            os.replace(target, replaced / "previous")
            shutil.rmtree(replaced, ignore_errors=True)
        os.replace(stage, target)
        stage = None  # type: ignore[assignment]
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)

    stale = stale_release_artifacts(root, version)
    if prune_stale:
        # `name` is the profile, and rebinding it here left the return below
        # reading a filename as the lane. The release was already promoted at
        # that point, so the operator saw "Release staging failed" for a
        # release that had in fact succeeded.
        for stale_name in stale:
            try:
                (root / stale_name).unlink()
            except OSError:
                pass
    return {
        "schema": RELEASE_STAGE_SCHEMA,
        "version": version,
        "profile": name,
        "directory": str(target),
        "assets": list(expected_assets(version, name)),
        "checksums": digests,
        "prunedStaleArtifacts": stale if prune_stale else [],
        "staleArtifacts": [] if prune_stale else stale,
        "valid": True,
    }


def published_release_dirs(version: str,
                           release_root: str | Path = DEFAULT_RELEASE_ROOT,
                           ) -> list[Path]:
    """The per-lane directories a complete publication has to cover."""
    root = Path(release_root) / version
    return [root / name for name in PUBLISHED_PROFILES]


def missing_published_profiles(
    version: str,
    release_root: str | Path = DEFAULT_RELEASE_ROOT,
) -> list[str]:
    """Lanes that have not been staged for this version.

    RM-350: the CPU build alone used to be the whole release, published
    under a name that said nothing while the README recommended NVIDIA.
    """
    root = Path(release_root) / version
    missing = []
    for name in PUBLISHED_PROFILES:
        report = verify_release_dir(root / name, version, name)
        if not report["valid"]:
            missing.append(name)
    return missing


def publication_guidance(version: str) -> list[str]:
    """Publication steps for an unsigned, immutable GitHub release."""
    lanes = ", ".join(PUBLISHED_PROFILES)
    zip_only = [name for name in PUBLISHED_PROFILES
                if not ships_installer(name)]
    return [
        f"Build every published lane before releasing: {lanes}. Each is "
        "built from its own locked dependency profile with "
        "`build_exe.bat <profile>`, and each carries the provider its name "
        "claims.",
        (("These lanes ship a portable ZIP and no installer, because their "
          "payload is past the 2 GB ceiling of the 32-bit NSIS compiler: "
          + ", ".join(zip_only) + ".")
         if zip_only else
         "Every published lane ships both an installer and a portable ZIP."),
        f"gh release create v{version} --draft --title \"v{version}\" "
        "--notes-file <changelog excerpt>",
        f"gh release upload v{version} build/release/{version}/*/* --clobber",
        "Review the draft, then publish it. Enable immutable releases on the "
        "repository so a published tag's assets can never be replaced.",
        "Artifacts are intentionally UNSIGNED. Do not acquire or apply a "
        "code-signing certificate; publish the SHA256SUMS.txt from each "
        "staged lane as the only integrity reference.",
        "DirectML is a supported local install profile and is deliberately "
        "not published as a tested bundle: nothing here has measured it on "
        "the hardware it targets.",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage and verify one versioned release artifact set."
    )
    parser.add_argument(
        "command",
        choices=("stage", "verify", "guidance", "check-lanes",
                 "split-oversized"))
    parser.add_argument("--version", default="")
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help=(
            "Dependency profile this artifact set was built from. It becomes "
            "part of every filename, and staging refuses evidence whose "
            "recorded provider does not match it."
        ),
    )
    parser.add_argument("--dist-dir", default="dist/VideoSubtitleRemoverPro")
    parser.add_argument(
        "--installer-path",
        default="",
        help=(
            "The compiled installer for this lane. Leave empty for a lane "
            "that ships only a portable ZIP; see INSTALLER_PROFILES."
        ),
    )
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

    if args.command == "split-oversized":
        root = Path(args.release_root) / version
        if not root.is_dir():
            print(f"No staged release at {root}")
            return 1
        split_any = False
        for lane in sorted(p for p in root.iterdir() if p.is_dir()):
            for asset in sorted(lane.iterdir()):
                if not asset.is_file() or asset.suffix == ".txt":
                    continue
                parts = split_asset_for_upload(asset)
                if not parts:
                    continue
                split_any = True
                print(f"{asset.name} exceeds the GitHub asset limit; "
                      f"split into {len(parts)} parts:")
                for part in parts:
                    print(f"  {part.name}  {part.stat().st_size} bytes")
        if not split_any:
            print("No staged asset is over the GitHub limit.")
        return 0

    if args.command == "check-lanes":
        missing = missing_published_profiles(version, args.release_root)
        if missing:
            print(
                "Release is incomplete. These lanes have not been staged "
                f"for {version}: " + ", ".join(missing)
            )
            return 1
        print(f"All published lanes are staged for {version}: "
              + ", ".join(PUBLISHED_PROFILES))
        return 0

    if args.command == "verify":
        report = verify_release_dir(
            Path(args.release_root) / version / args.profile,
            version,
            args.profile,
        )
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
            profile=args.profile,
        )
    except ReleaseStagingError as exc:
        print(f"Release staging failed: {exc}")
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
