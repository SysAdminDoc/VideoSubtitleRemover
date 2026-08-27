"""Reviewed, reproducible dependency profiles and release evidence."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

from backend.dependency_caps import (
    PROTOBUF_SECURITY_MIN,
    PROVIDER_LANES,
    collect_provider_lane_status,
    provider_lane_for_profile,
    version_in_lane,
)
from backend.subprocess_policy import run_process


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "dependency_profiles.json"
PROFILE_DIR = ROOT / "dependency_profiles"
PROFILE_SCHEMA = "vsr.dependency_profiles.v1"
PROFILE_STATUS_SCHEMA = "vsr.dependency_profile_status.v1"
PROFILE_ENV = "VSR_DEPENDENCY_PROFILE"
PROFILE_SMOKE_SCHEMA = "vsr.dependency_profile_smoke.v1"
SUPPORTED_PROFILES = ("cpu", "nvidia", "directml")
_EXACT_PIN = re.compile(r"^\s*([A-Za-z0-9._-]+)\s*==\s*([0-9][0-9A-Za-z.+-]*)\s*$")
_PACKAGE_IMPORTS = {
    "numpy": "numpy",
    "protobuf": "google.protobuf",
    "opencv-python": "cv2",
    "pillow": "PIL",
    "idna": "idna",
    "rapidocr": "rapidocr",
    "torch": "torch",
    "torchvision": "torchvision",
    "onnxruntime": "onnxruntime",
    "onnxruntime-gpu": "onnxruntime",
    "onnxruntime-directml": "onnxruntime",
}
_CONFLICT_GROUPS = (
    (
        "opencv-python",
        "opencv-contrib-python",
        "opencv-python-headless",
        "opencv-contrib-python-headless",
    ),
    ("onnxruntime", "onnxruntime-gpu", "onnxruntime-directml"),
)


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
    common_required = payload.get("commonRequiredPackages")
    if not isinstance(common_required, list) or not all(
        isinstance(item, str) and item.strip() for item in common_required
    ):
        raise ValueError("commonRequiredPackages must be a list of package names")
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
        required = profile.get("requiredPackages")
        capabilities = profile.get("capabilities")
        if not isinstance(constraints, list) or not all(
            isinstance(item, str) for item in constraints
        ):
            raise ValueError(f"Profile {name!r} constraints are invalid")
        if not isinstance(indexes, list) or not all(
            isinstance(item, str) and item.startswith("https://") for item in indexes
        ):
            raise ValueError(f"Profile {name!r} indexes are invalid")
        if not isinstance(required, list) or not all(
            isinstance(item, str) and item.strip() for item in required
        ):
            raise ValueError(f"Profile {name!r} requiredPackages are invalid")
        if not isinstance(capabilities, list) or not all(
            isinstance(item, str) and item.strip() for item in capabilities
        ):
            raise ValueError(f"Profile {name!r} capabilities are invalid")
        required_names = [
            _normalize_package_name(item)
            for item in [*common_required, *required]
        ]
        if len(required_names) != len(set(required_names)):
            raise ValueError(f"Profile {name!r} repeats a required package")
        pins = _exact_pins([*common, *constraints])
        missing_pins = sorted(set(required_names) - set(pins))
        if missing_pins:
            raise ValueError(
                f"Profile {name!r} required packages lack exact pins: "
                + ", ".join(missing_pins)
            )
    problems = constraint_range_problems(payload)
    if problems:
        raise ValueError(
            "Dependency manifest pins a version outside its reviewed range: "
            + "; ".join(problems)
        )
    return payload


def _exact_pins(requirements: Sequence[str]) -> dict[str, str]:
    pins: dict[str, str] = {}
    for requirement in requirements:
        match = _EXACT_PIN.match(str(requirement))
        if match:
            pins[_normalize_package_name(match.group(1))] = match.group(2)
    return pins


def _normalize_package_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", str(value).strip().lower())


def profile_required_packages(
    name: str,
    manifest: Mapping[str, Any] | None = None,
    *,
    manifest_path: str | Path = MANIFEST_PATH,
) -> tuple[dict[str, str], ...]:
    payload = dict(manifest or load_profile_manifest(manifest_path))
    if name not in SUPPORTED_PROFILES:
        raise ValueError(f"Unsupported dependency profile: {name!r}")
    profile = payload["profiles"][name]
    required = [
        *payload["commonRequiredPackages"],
        *profile["requiredPackages"],
    ]
    pins = _exact_pins([
        *payload["commonConstraints"],
        *profile["constraints"],
    ])
    return tuple({
        "name": _normalize_package_name(package),
        "expectedVersion": pins[_normalize_package_name(package)],
    } for package in required)


def profile_capabilities(
    name: str,
    manifest: Mapping[str, Any] | None = None,
    *,
    manifest_path: str | Path = MANIFEST_PATH,
) -> tuple[str, ...]:
    payload = dict(manifest or load_profile_manifest(manifest_path))
    if name not in SUPPORTED_PROFILES:
        raise ValueError(f"Unsupported dependency profile: {name!r}")
    return tuple(str(item) for item in payload["profiles"][name]["capabilities"])


def constraint_range_problems(manifest: Mapping[str, Any]) -> list[str]:
    """Reject exact locks that fall outside their reviewed provider window.

    RM-140: the NVIDIA lock previously pinned an onnxruntime-gpu build that the
    setup path explicitly excludes (CUDA 13 only), so the profile advertised a
    CUDA 12 provider it could never install. The reviewed windows live in
    ``backend.dependency_caps.PROVIDER_LANES`` so setup, the locks, and the
    advisories cannot drift apart.
    """
    problems: list[str] = []
    common = _exact_pins(manifest.get("commonConstraints") or [])
    protobuf = common.get("protobuf")
    if protobuf is None:
        problems.append(
            f"commonConstraints must pin protobuf >={PROTOBUF_SECURITY_MIN}"
        )
    elif not _version_gte(protobuf, PROTOBUF_SECURITY_MIN):
        problems.append(
            f"protobuf=={protobuf} is below the security floor "
            f"{PROTOBUF_SECURITY_MIN}"
        )
    lane_packages = {lane.package for lane in PROVIDER_LANES}
    profiles = manifest.get("profiles") or {}
    for name in sorted(profiles):
        profile = profiles[name]
        if not isinstance(profile, Mapping):
            continue
        lane = provider_lane_for_profile(name)
        pins = _exact_pins(profile.get("constraints") or [])
        for package, version in sorted(pins.items()):
            if package not in lane_packages:
                continue
            if lane is None or package != lane.package:
                problems.append(
                    f"profile {name!r} pins {package}=={version}, which does "
                    "not belong to its provider lane"
                )
                continue
            if not version_in_lane(version, lane):
                problems.append(
                    f"profile {name!r} pins {package}=={version} outside the "
                    f"{lane.label} lane range >={lane.minimum},"
                    f"<{lane.maximum_exclusive}"
                )
        if lane is not None and lane.package not in pins:
            problems.append(
                f"profile {name!r} must pin {lane.package} for the "
                f"{lane.label} lane"
            )
    return problems


def _version_key(value: str) -> tuple[int, int, int, int]:
    numbers = [int(part) for part in re.findall(r"\d+", value)[:4]]
    while len(numbers) < 4:
        numbers.append(0)
    return tuple(numbers)  # type: ignore[return-value]


def _version_gte(version: str, floor: str) -> bool:
    return _version_key(version) >= _version_key(floor)


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


def run_profile_provider_smoke(
    name: str,
    *,
    manifest_path: str | Path = MANIFEST_PATH,
    ort_module: Any = None,
    importer=importlib.import_module,
    temp_dir: str | Path | None = None,
) -> dict:
    """Create one real inference session on the profile's claimed provider.

    RM-140: a profile that advertises ``CUDAExecutionProvider`` is only
    meaningful if ONNX Runtime actually activates it. ONNX Runtime silently
    falls back to CPU when a requested provider cannot load, so this smoke
    fails when the active provider list does not lead with the claimed one.
    """
    manifest = load_profile_manifest(manifest_path)
    provider = str(manifest["profiles"][name]["provider"])
    result: dict[str, Any] = {
        "schema": PROFILE_SMOKE_SCHEMA,
        "profile": name,
        "requestedProvider": provider,
        "availableProviders": [],
        "activeProviders": [],
        "ran": False,
        "passed": None,
        "fellBack": None,
        "error": "",
    }
    module = ort_module
    if module is None:
        try:
            module = importer("onnxruntime")
        except Exception as exc:  # pragma: no cover - import-environment specific
            result["error"] = f"onnxruntime is not importable: {exc}"
            return result
    try:
        available = list(module.get_available_providers())
    except Exception as exc:
        result["error"] = f"provider enumeration failed: {exc}"
        return result
    result["availableProviders"] = available
    if provider not in available:
        result["ran"] = True
        result["passed"] = False
        result["fellBack"] = True
        result["error"] = (
            f"{provider} is not offered by the installed onnxruntime build"
        )
        return result
    from backend.onnx_model_info import _tiny_identity_onnx_bytes

    try:
        import numpy as np

        with tempfile.TemporaryDirectory(
            prefix="vsr_profile_smoke_",
            dir=str(temp_dir) if temp_dir else None,
        ) as tmpdir:
            model_path = Path(tmpdir) / "identity.onnx"
            model_path.write_bytes(_tiny_identity_onnx_bytes())
            session = module.InferenceSession(
                str(model_path), providers=[provider])
            active = (
                list(session.get_providers())
                if hasattr(session, "get_providers") else [provider]
            )
            payload = np.array([3.0], dtype=np.float32)
            output = session.run(None, {"x": payload})[0]
            correct = bool(np.allclose(output, payload))
    except Exception as exc:
        result["ran"] = True
        result["passed"] = False
        result["error"] = f"inference session failed: {exc}"
        return result
    result["ran"] = True
    result["activeProviders"] = active
    fell_back = not active or active[0] != provider
    result["fellBack"] = fell_back
    result["passed"] = bool(correct and not fell_back)
    if fell_back:
        result["error"] = (
            f"{provider} was requested but {active[0] if active else 'no provider'} "
            "ran the session"
        )
    elif not correct:
        result["error"] = "identity model produced an unexpected result"
    return result


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


def _installed_distribution_version(package: str) -> str | None:
    try:
        return str(metadata.version(package))
    except metadata.PackageNotFoundError:
        return None


def _runtime_package_versions(
    required: Sequence[Mapping[str, str]],
    supplied: Mapping[str, str] | None,
) -> dict[str, str]:
    if supplied is not None:
        return {
            _normalize_package_name(name): str(version)
            for name, version in supplied.items()
        }
    names = {
        str(item["name"])
        for item in required
    }
    for group in _CONFLICT_GROUPS:
        names.update(group)
    versions: dict[str, str] = {}
    for name in sorted(names):
        version = _installed_distribution_version(name)
        if version is not None:
            versions[name] = version
    return versions


def _matches_exact_version(installed: str, expected: str) -> bool:
    actual = str(installed).strip()
    reviewed = str(expected).strip()
    return actual == reviewed or actual.split("+", 1)[0] == reviewed


def verify_required_package_versions(
    name: str,
    *,
    package_versions: Mapping[str, str] | None = None,
    manifest: Mapping[str, Any] | None = None,
    manifest_path: str | Path = MANIFEST_PATH,
) -> tuple[list[dict[str, Any]], list[str]]:
    payload = dict(manifest or load_profile_manifest(manifest_path))
    required = profile_required_packages(
        name,
        payload,
        manifest_path=manifest_path,
    )
    installed = _runtime_package_versions(required, package_versions)
    statuses: list[dict[str, Any]] = []
    errors: list[str] = []
    for item in required:
        package = item["name"]
        expected = item["expectedVersion"]
        actual = installed.get(package)
        matched = bool(
            actual is not None and _matches_exact_version(actual, expected)
        )
        statuses.append({
            "name": package,
            "expectedVersion": expected,
            "installedVersion": actual,
            "satisfied": matched,
        })
        if actual is None:
            errors.append(
                f"Required package {package}=={expected} is not installed"
            )
        elif not matched:
            errors.append(
                f"Required package {package}=={expected} resolved as {actual}"
            )
    for group in _CONFLICT_GROUPS:
        present = [package for package in group if installed.get(package)]
        if len(present) > 1:
            errors.append(
                "Conflicting runtime distributions are installed: "
                + ", ".join(present)
            )
    return statuses, errors


def run_profile_import_smoke(
    name: str,
    *,
    importer=None,
    manifest_path: str | Path = MANIFEST_PATH,
) -> dict[str, Any]:
    load = importlib.import_module if importer is None else importer
    required = profile_required_packages(name, manifest_path=manifest_path)
    modules = []
    failures = []
    seen = set()
    for item in required:
        package = item["name"]
        module_name = _PACKAGE_IMPORTS.get(package)
        if not module_name or module_name in seen:
            continue
        seen.add(module_name)
        try:
            load(module_name)
        except Exception as exc:  # noqa: BLE001
            failures.append({
                "package": package,
                "module": module_name,
                "error": str(exc),
            })
        else:
            modules.append(module_name)
    return {
        "passed": not failures,
        "modules": modules,
        "failures": failures,
    }


def run_pip_check(timeout: float = 120.0) -> dict[str, Any]:
    try:
        result = run_process(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "passed": False,
            "returnCode": None,
            "output": str(exc),
        }
    output = "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part.strip()
    )
    return {
        "passed": result.returncode == 0,
        "returnCode": int(result.returncode),
        "output": output,
    }


def collect_dependency_profile_status(
    *,
    profile: str | None = None,
    env: Mapping[str, str] | None = None,
    package_versions: Mapping[str, str] | None = None,
    manifest_path: str | Path = MANIFEST_PATH,
    profile_dir: str | Path = PROFILE_DIR,
    run_provider_smoke: bool = False,
    verify_runtime: bool = False,
    run_import_checks: bool = False,
    run_package_check: bool = False,
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
            "providerLanes": collect_provider_lane_status(package_versions),
            "providerSmoke": None,
            "requiredPackages": [],
            "capabilities": [],
            "importSmoke": None,
            "pipCheck": None,
        }
    path = profile_constraint_path(name, profile_dir)
    actual = path.read_text(encoding="utf-8") if path.exists() else ""
    if not path.exists():
        errors.append(f"Constraint file is missing: {path}")
    elif actual != expected:
        errors.append(f"Constraint file is stale: {path}")
    profile_data = manifest["profiles"][name]
    required_packages = [
        dict(item) for item in profile_required_packages(
            name,
            manifest,
            manifest_path=manifest_path,
        )
    ]
    if verify_runtime:
        required_packages, runtime_errors = verify_required_package_versions(
            name,
            package_versions=package_versions,
            manifest=manifest,
            manifest_path=manifest_path,
        )
        errors.extend(runtime_errors)
    lanes = collect_provider_lane_status(package_versions)
    protobuf = lanes.get("protobuf", {})
    if protobuf.get("satisfied") is False:
        errors.append(
            f"protobuf {protobuf.get('installedVersion')} is below the "
            f"{protobuf.get('securityFloor')} security floor"
        )
    smoke = None
    if run_provider_smoke:
        smoke = run_profile_provider_smoke(name, manifest_path=manifest_path)
        if smoke.get("passed") is not True:
            errors.append(
                "Provider inference smoke failed for profile "
                f"{name!r}: {smoke.get('error') or 'unknown error'}"
            )
    import_smoke = None
    if run_import_checks:
        import_smoke = run_profile_import_smoke(
            name,
            manifest_path=manifest_path,
        )
        for failure in import_smoke.get("failures", []):
            errors.append(
                f"Runtime import failed for {failure.get('package')}: "
                f"{failure.get('error') or 'unknown error'}"
            )
    pip_check = None
    if run_package_check:
        pip_check = run_pip_check()
        if pip_check.get("passed") is not True:
            errors.append(
                "pip check failed: "
                + str(pip_check.get("output") or "unknown error")
            )
    return {
        "schema": PROFILE_STATUS_SCHEMA,
        "profile": name,
        "source": source,
        "provider": profile_data["provider"],
        "capabilities": list(profile_data["capabilities"]),
        "requiredPackages": required_packages,
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
        "providerLanes": lanes,
        "providerSmoke": smoke,
        "importSmoke": import_smoke,
        "pipCheck": pip_check,
    }


def _print_diffs(changes: Mapping[str, str]) -> None:
    for name in SUPPORTED_PROFILES:
        diff = changes.get(name)
        if diff:
            print(diff, end="" if diff.endswith("\n") else "\n")


def _write_status_report(path: str | Path, status: Mapping[str, Any]) -> None:
    """Write a verifier report atomically for setup and support tooling."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(dict(status), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check, regenerate, or inspect reviewed dependency profiles."
    )
    parser.add_argument(
        "command",
        choices=("check", "update", "status", "smoke", "verify"),
    )
    parser.add_argument("--profile", choices=SUPPORTED_PROFILES)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--profile-dir", default=str(PROFILE_DIR))
    parser.add_argument(
        "--output",
        help="Atomically write status, smoke, or verification JSON to this path.",
    )
    args = parser.parse_args(argv)

    if args.command in ("status", "smoke", "verify"):
        verify = args.command == "verify"
        status = collect_dependency_profile_status(
            profile=args.profile,
            manifest_path=args.manifest,
            profile_dir=args.profile_dir,
            run_provider_smoke=args.command in {"smoke", "verify"},
            verify_runtime=verify,
            run_import_checks=verify,
            run_package_check=verify,
        )
        print(json.dumps(status, indent=2, sort_keys=True))
        if args.output:
            try:
                _write_status_report(args.output, status)
            except OSError as exc:
                print(f"Could not write dependency profile report: {exc}", file=sys.stderr)
                return 2
        return 0 if status.get("valid") or args.command == "status" else 1

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
