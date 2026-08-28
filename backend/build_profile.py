"""RM-350: the dependency profile a frozen build was actually made from.

The download was one generically named installer that recommended NVIDIA and
shipped a CPU-only payload, and nothing in the artifact said which it was.
Two things have to travel with the build for that to be fixable: the profile
name, stamped in at freeze time, and the provider ONNX Runtime really
activates when the frozen executable runs. Release verification compares both
against the name on the artifact, so a CPU payload cannot be published under
a CUDA filename.

The stamp is written by `VideoSubtitleRemoverPro.spec` during the build and
read back from inside the frozen bundle. Reading from a source checkout falls
back to the profile environment variable, and then to whichever provider
package is installed, so the same call works in a test and in the field.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Optional

BUILD_PROFILE_SCHEMA = "vsr.build_profile.v1"
BUILD_PROFILE_FILE = "vsr-build-profile.json"

ROOT = Path(__file__).resolve().parents[1]


def _supported() -> tuple[str, ...]:
    from backend.dependency_profiles import SUPPORTED_PROFILES

    return tuple(SUPPORTED_PROFILES)


def normalize_profile(value: object) -> str:
    """A supported profile name, or "" when the value names none."""
    name = str(value or "").strip().lower()
    return name if name in _supported() else ""


def declared_provider(profile: str,
                      *, manifest_path: str | Path | None = None) -> str:
    """The execution provider the manifest says this profile delivers."""
    from backend.dependency_profiles import MANIFEST_PATH, load_profile_manifest

    name = normalize_profile(profile)
    if not name:
        return ""
    manifest = load_profile_manifest(manifest_path or MANIFEST_PATH)
    return str(manifest["profiles"][name]["provider"])


def write_build_profile(directory: str | Path, profile: str,
                        *, app_version: str = "") -> Path:
    """Stamp the profile into a directory the freeze will bundle."""
    name = normalize_profile(profile)
    if not name:
        raise ValueError(f"Unsupported build profile: {profile!r}")
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / BUILD_PROFILE_FILE
    payload = {
        "schema": BUILD_PROFILE_SCHEMA,
        "profile": name,
        "provider": declared_provider(name),
        "appVersion": str(app_version or ""),
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _bundle_roots() -> list[Path]:
    """Every directory a stamped profile could be read back from."""
    roots: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", "")
    if meipass:
        roots.append(Path(meipass))
    executable = Path(sys.executable).resolve().parent
    roots.append(executable)
    roots.append(executable / "_internal")
    roots.append(ROOT)
    seen: set[Path] = set()
    unique = []
    for root in roots:
        if root not in seen:
            seen.add(root)
            unique.append(root)
    return unique


def read_stamped_profile(
    roots: Optional[list[Path]] = None,
) -> Optional[dict[str, Any]]:
    """The stamp written at freeze time, or None when there is none."""
    for root in (roots if roots is not None else _bundle_roots()):
        path = Path(root) / BUILD_PROFILE_FILE
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, Mapping):
            continue
        name = normalize_profile(payload.get("profile"))
        if not name:
            continue
        return {
            "schema": BUILD_PROFILE_SCHEMA,
            "profile": name,
            "provider": str(payload.get("provider") or ""),
            "appVersion": str(payload.get("appVersion") or ""),
            "source": "stamp",
            "path": str(path),
        }
    return None


def resolve_build_profile(
    *,
    env: Mapping[str, str] | None = None,
    package_versions: Mapping[str, str] | None = None,
    roots: Optional[list[Path]] = None,
) -> dict[str, Any]:
    """The profile this build carries, and how that was established.

    `source` says which of the three answers was used, because a stamp is
    evidence and a guess from the installed packages is not.
    """
    stamped = read_stamped_profile(roots)
    if stamped is not None:
        return stamped

    from backend.dependency_profiles import (
        PROFILE_ENV,
        _installed_provider_profile,
    )

    environment = os.environ if env is None else env
    requested = normalize_profile(environment.get(PROFILE_ENV, ""))
    if requested:
        return {
            "schema": BUILD_PROFILE_SCHEMA,
            "profile": requested,
            "provider": declared_provider(requested),
            "appVersion": "",
            "source": "environment",
            "path": "",
        }
    detected = normalize_profile(
        _installed_provider_profile(package_versions)) or "cpu"
    return {
        "schema": BUILD_PROFILE_SCHEMA,
        "profile": detected,
        "provider": declared_provider(detected),
        "appVersion": "",
        "source": "installed-provider",
        "path": "",
    }
