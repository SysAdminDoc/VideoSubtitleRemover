"""Local release evidence generation.

The old GitHub Actions release job emitted release-verification.json,
release-hidden-imports.json, and an SBOM. Local-build mode keeps those
contracts here so the PyInstaller build can be audited without a workflow
file.
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import hashlib
import importlib.metadata as metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from backend.adapter_manifest import (
    collect_adapter_conformance_matrix,
    release_manifest_status,
)
from backend.dependency_caps import (
    PILLOW_MINIMUM_VERSION,
    collect_dependency_drift_report,
    collect_opencv_wheel_status,
    collect_onnxruntime_provider_status,
    collect_rapidocr_engine_status,
    onnxruntime_release_advisories,
)
from backend.ffmpeg_profiles import (
    FFMPEG_RELEASE_SOURCE,
    FFMPEG_SECURITY_SOURCE,
    classify_ffmpeg_security,
    collect_ffmpeg_capability_profiles,
    probe_ffmpeg_security,
)
from backend.onnx_model_info import rapidocr_release_provenance
from backend.remote_model_policy import release_remote_model_status
from backend.security_checks import opencv_libpng_status


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ("README.md", "LICENSE", "CHANGELOG.md")
LAUNCHERS = ("Run_VSR_Pro.bat", "Run_VSR_Pro_Debug.bat", "Run_VSR_Pro.ps1")
FROZEN_LAUNCHER_SOURCE_DIR = ROOT / "assets" / "frozen"
VERSIONED_DOCS = ("README.md", "CHANGELOG.md")
HIDDEN_IMPORT_RE = re.compile(r"--hidden-import(?:=|\s+)([^\s]+)")
RUNTIME_HOOK_RE = re.compile(r"--runtime-hook(?:=|\s+)([^\s]+)")
EXCLUDE_MODULE_RE = re.compile(r"--exclude-module(?:=|\s+)([^\s]+)")
STRICT_BLOCKING_SEVERITIES = {"high", "critical"}
REQUIRED_RUNTIME_HOOK = "assets/runtime_hook_mp.py"
NSIS_MINIMUM_VERSION = "3.12"
BUILD_ONLY_DISTRIBUTIONS = {
    "pip",
    "pip-audit",
    "pyinstaller",
    "pyinstaller-hooks-contrib",
    "setuptools",
    "wheel",
}


def _read_config_constant(name: str, default: str) -> str:
    config_path = ROOT / "gui" / "config.py"
    try:
        tree = ast.parse(config_path.read_text(encoding="utf-8"))
    except OSError:
        return default
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                try:
                    value = ast.literal_eval(node.value)
                except (TypeError, ValueError, SyntaxError):
                    return default
                return str(value)
    return default


APP_NAME = _read_config_constant("APP_NAME", "Video Subtitle Remover Pro")
APP_VERSION = _read_config_constant("APP_VERSION", "0.0.0")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_hidden_imports(value: str | Sequence[str]) -> tuple[str, ...]:
    """Return sorted PyInstaller hidden-import module names."""
    if isinstance(value, str):
        text = value
    else:
        text = " ".join(str(part) for part in value)
    found = HIDDEN_IMPORT_RE.findall(text)
    return tuple(sorted(dict.fromkeys(name.strip("\"'") for name in found if name)))


def parse_runtime_hooks(value: str | Sequence[str]) -> tuple[str, ...]:
    """Return sorted PyInstaller runtime-hook paths."""
    if isinstance(value, str):
        text = value
    else:
        text = " ".join(str(part) for part in value)
    found = RUNTIME_HOOK_RE.findall(text)
    normalised = []
    for raw in found:
        item = raw.strip("\"'").replace("\\", "/")
        if item and item not in normalised:
            normalised.append(item)
    return tuple(sorted(normalised))


def parse_excluded_modules(value: str | Sequence[str]) -> tuple[str, ...]:
    text = value if isinstance(value, str) else " ".join(str(part) for part in value)
    found = EXCLUDE_MODULE_RE.findall(text)
    return tuple(sorted(dict.fromkeys(
        name.strip("\"'") for name in found if name
    )))


def _dist_file_status(root: Path, name: str) -> dict:
    source = ROOT / name
    bundled = root / name
    payload = {
        "name": name,
        "sourceExists": source.exists(),
        "bundled": bundled.exists(),
        "sourceSha256": None,
        "bundledSha256": None,
    }
    if source.is_file():
        payload["sourceSha256"] = sha256_file(source)
    if bundled.is_file():
        payload["bundledSha256"] = sha256_file(bundled)
    return payload


def _source_file_status(name: str) -> dict:
    normalised = str(name or "").replace("\\", "/")
    source = ROOT / normalised
    payload = {
        "name": normalised,
        "sourceExists": source.is_file(),
        "sourceSha256": None,
    }
    if source.is_file():
        payload["sourceSha256"] = sha256_file(source)
    return payload


def _doc_version_status(path: Path) -> dict:
    exists = path.is_file()
    text = path.read_text(encoding="utf-8", errors="replace") if exists else ""
    return {
        "name": path.name,
        "exists": exists,
        "containsAppVersion": APP_VERSION in text,
    }


def collect_dependency_versions() -> list[dict]:
    items = []
    for dist in sorted(metadata.distributions(),
                       key=lambda d: (d.metadata.get("Name") or "").lower()):
        name = dist.metadata.get("Name") or ""
        if not name:
            continue
        items.append({
            "name": name,
            "version": dist.version,
        })
    return items


def _normalise_distribution_name(value: object) -> str:
    return re.sub(r"[-_.]+", "-", str(value or "").strip().lower())


def _pyinstaller_analysis_entries(path: Path) -> tuple[list[tuple[str, str, str]], str]:
    """Read PyInstaller's literal TOC without executing build output."""
    if not path.is_file():
        return [], "PyInstaller Analysis TOC not found"
    try:
        payload = ast.literal_eval(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError) as exc:
        return [], f"PyInstaller Analysis TOC could not be parsed: {exc}"
    entries: list[tuple[str, str, str]] = []
    seen = set()
    if not isinstance(payload, (tuple, list)):
        return [], "PyInstaller Analysis TOC has an unexpected root type"
    for group in payload:
        if not isinstance(group, (tuple, list)):
            continue
        for item in group:
            if not isinstance(item, (tuple, list)) or len(item) < 3:
                continue
            name, source, kind = (str(item[0]), str(item[1]), str(item[2]))
            if kind not in {"BINARY", "DATA", "EXTENSION", "PYMODULE", "PYSOURCE"}:
                continue
            key = (name, source, kind)
            if key not in seen:
                seen.add(key)
                entries.append(key)
    return entries, "" if entries else "PyInstaller Analysis TOC contains no artifact entries"


def _site_package_top_level(source: str) -> str:
    parts = [part for part in re.split(r"[\\/]", source) if part]
    lowered = [part.lower() for part in parts]
    try:
        index = lowered.index("site-packages")
    except ValueError:
        return ""
    if index + 1 >= len(parts):
        return ""
    candidate = parts[index + 1]
    if candidate.endswith((".dist-info", ".egg-info")):
        candidate = re.split(r"-(?=\d)", candidate, maxsplit=1)[0]
    return candidate.split(".")[0]


def _packaged_file(dist_dir: Path, destination: str) -> Optional[Path]:
    relative = Path(destination.replace("\\", os.sep).replace("/", os.sep))
    for candidate in (dist_dir / relative, dist_dir / "_internal" / relative):
        if candidate.is_file():
            return candidate
    return None


def build_cyclonedx_sbom(
    dependencies: Sequence[Mapping[str, object]],
    *,
    dist_dir: str | Path | None = None,
    analysis_path: str | Path | None = None,
) -> dict:
    """Build an SBOM from the frozen artifact, with an explicit fallback.

    PyInstaller's Analysis TOC is the source of truth for Python modules and
    native binaries.  Environment-wide metadata is used only to attach names
    and versions to entries proven present in that TOC.
    """
    dependencies_by_name = {
        _normalise_distribution_name(dep.get("name")): dep
        for dep in dependencies
        if dep.get("name")
    }
    entries: list[tuple[str, str, str]] = []
    analysis_error = "PyInstaller Analysis TOC was not supplied"
    analysis = Path(analysis_path) if analysis_path else None
    if analysis is not None:
        entries, analysis_error = _pyinstaller_analysis_entries(analysis)
    artifact_derived = bool(entries and dist_dir is not None)
    dist = Path(dist_dir) if dist_dir is not None else None
    components = []

    if artifact_derived:
        top_levels = set()
        for name, source, kind in entries:
            if kind in {"PYMODULE", "PYSOURCE"}:
                top_levels.add(name.split(".")[0])
            source_top = _site_package_top_level(source)
            if source_top:
                top_levels.add(source_top)

        distribution_names = set()
        try:
            package_map = metadata.packages_distributions()
        except Exception:
            package_map = {}
        for top_level in top_levels:
            distribution_names.update(package_map.get(top_level, ()) or ())
            normalised_top = _normalise_distribution_name(top_level)
            if normalised_top in dependencies_by_name:
                distribution_names.add(str(
                    dependencies_by_name[normalised_top].get("name") or ""
                ))

        runtime_names = {
            _normalise_distribution_name(name)
            for name in distribution_names
        } - BUILD_ONLY_DISTRIBUTIONS
        for normalised in sorted(runtime_names):
            dep = dependencies_by_name.get(normalised)
            if not dep:
                continue
            name = str(dep.get("name") or "")
            version = str(dep.get("version") or "")
            components.append({
                "type": "library",
                "name": name,
                "version": version,
                "scope": "required",
                "purl": f"pkg:pypi/{name.lower()}@{version}" if version else "",
                "properties": [
                    {"name": "vsr:evidence", "value": "pyinstaller-analysis"},
                ],
            })

        for normalised in sorted(BUILD_ONLY_DISTRIBUTIONS):
            dep = dependencies_by_name.get(normalised)
            if not dep:
                continue
            name = str(dep.get("name") or "")
            version = str(dep.get("version") or "")
            components.append({
                "type": "library",
                "name": name,
                "version": version,
                "scope": "excluded",
                "purl": f"pkg:pypi/{name.lower()}@{version}" if version else "",
                "properties": [
                    {"name": "vsr:evidence", "value": "build-tool-only"},
                ],
            })

        native_seen = set()
        for destination, source, kind in entries:
            if kind not in {"BINARY", "EXTENSION"}:
                continue
            relative = destination.replace("\\", "/")
            key = relative.lower()
            if key in native_seen:
                continue
            native_seen.add(key)
            packaged = _packaged_file(dist, destination) if dist else None
            component = {
                "type": "file",
                "name": relative,
                "scope": "required",
                "properties": [
                    {"name": "vsr:evidence", "value": "pyinstaller-analysis"},
                    {"name": "vsr:sourceName", "value": Path(source).name},
                ],
            }
            if packaged is not None:
                component["hashes"] = [
                    {"alg": "SHA-256", "content": sha256_file(packaged)},
                ]
            components.append(component)
    else:
        # Compatibility mode for source-only diagnostics. Strict releases
        # reject this mode; it remains useful before a frozen artifact exists.
        for dep in dependencies:
            name = str(dep.get("name") or "")
            version = str(dep.get("version") or "")
            if not name:
                continue
            components.append({
                "type": "library",
                "name": name,
                "version": version,
                "scope": "required",
                "purl": f"pkg:pypi/{name.lower()}@{version}" if version else "",
                "properties": [
                    {"name": "vsr:evidence", "value": "environment-fallback"},
                ],
            })

    app_component = {
        "type": "application",
        "name": APP_NAME,
        "version": APP_VERSION,
    }
    if dist is not None:
        executable = dist / "VideoSubtitleRemoverPro.exe"
        if executable.is_file():
            app_component["hashes"] = [
                {"alg": "SHA-256", "content": sha256_file(executable)},
            ]
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": (
            "urn:uuid:"
            + str(uuid.UUID(hashlib.sha256(
                f"{APP_NAME}:{APP_VERSION}".encode()
            ).hexdigest()[:32]))
        ),
        "version": 1,
        "metadata": {
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "component": app_component,
            "properties": [
                {
                    "name": "vsr:artifactDerived",
                    "value": str(artifact_derived).lower(),
                },
                {
                    "name": "vsr:analysisSource",
                    "value": analysis.name if analysis is not None else "",
                },
                {"name": "vsr:analysisError", "value": analysis_error},
            ],
        },
        "components": components,
    }


def _version_parts(value: object) -> tuple[int, ...]:
    parts = []
    for match in re.finditer(r"\d+", str(value or "")):
        parts.append(int(match.group(0)))
    return tuple(parts)


def _version_lt(value: object, floor: str) -> bool:
    current = _version_parts(value)
    target = _version_parts(floor)
    if not current or not target:
        return False
    width = max(len(current), len(target))
    return current + (0,) * (width - len(current)) < target + (0,) * (width - len(target))


def _version_gte(value: object, floor: str) -> bool:
    current = _version_parts(value)
    target = _version_parts(floor)
    if not current or not target:
        return False
    width = max(len(current), len(target))
    return current + (0,) * (width - len(current)) >= target + (0,) * (width - len(target))


def _dependency_version(
    dependencies: Sequence[Mapping[str, object]],
    *names: str,
) -> Optional[str]:
    wanted = {name.lower().replace("_", "-") for name in names}
    for dep in dependencies:
        name = str(dep.get("name") or "").lower().replace("_", "-")
        if name in wanted:
            return str(dep.get("version") or "")
    return None


def _advisory(
    *,
    advisory_id: str,
    package: str,
    installed_version: str,
    affected: str,
    fixed_in: str,
    severity: str,
    source: str,
    allowed: bool = False,
    allow_reason: str = "",
    mitigation: str = "",
) -> dict:
    severity_norm = severity.lower()
    blocking = (
        severity_norm in STRICT_BLOCKING_SEVERITIES
        and not allowed
    )
    return {
        "id": advisory_id,
        "package": package,
        "installedVersion": installed_version,
        "affected": affected,
        "fixedIn": fixed_in,
        "severity": severity_norm,
        "source": source,
        "allowed": bool(allowed),
        "allowReason": allow_reason,
        "mitigation": mitigation,
        "blocking": blocking,
    }


def collect_release_advisories(
    dependencies: Sequence[Mapping[str, object]],
    *,
    env: Optional[Mapping[str, str]] = None,
) -> dict:
    """Collect deterministic release advisory evidence for strict builds."""
    findings = []
    torch_version = _dependency_version(dependencies, "torch", "pytorch")
    if torch_version and _version_lt(torch_version, "2.6.0"):
        findings.append(_advisory(
            advisory_id="CVE-2025-32434",
            package="torch",
            installed_version=torch_version,
            affected="<=2.5.1",
            fixed_in="2.6.0",
            severity="critical",
            source="https://nvd.nist.gov/vuln/detail/CVE-2025-32434",
            mitigation=(
                "Do not ship builds with vulnerable torch; PyTorch LaMa is "
                "packaging opt-in and should use torch >= 2.6.0."
            ),
        ))

    pillow_version = _dependency_version(dependencies, "pillow", "pil")
    if pillow_version and _version_lt(pillow_version, PILLOW_MINIMUM_VERSION):
        findings.append(_advisory(
            advisory_id="PILLOW-12.3-SECURITY-FLOOR",
            package="Pillow",
            installed_version=pillow_version,
            affected=f"<{PILLOW_MINIMUM_VERSION}",
            fixed_in=PILLOW_MINIMUM_VERSION,
            severity="high",
            source=(
                "https://pillow.readthedocs.io/en/stable/"
                "releasenotes/12.3.0.html"
            ),
            mitigation=(
                f"Upgrade Pillow to >= {PILLOW_MINIMUM_VERSION} before "
                "producing a strict release; 12.3.0 contains multiple "
                "input-decoder, memory-safety, and decompression-bomb fixes."
            ),
        ))

    libpng = opencv_libpng_status()
    if libpng.get("vulnerable") is True:
        findings.append(_advisory(
            advisory_id="CVE-2026-22801",
            package="opencv-python bundled libpng",
            installed_version=str(libpng.get("libpng_version") or "unknown"),
            affected=">=1.6.26,<1.6.54",
            fixed_in=str(libpng.get("fixed_version") or "1.6.54"),
            severity="medium",
            source="https://nvd.nist.gov/vuln/detail/CVE-2026-22801",
            allowed=True,
            allow_reason=(
                "opencv-python has not shipped a fixed bundled libpng wheel; "
                "user-supplied PNG still-image reads are routed through Pillow "
                "while the runtime remains vulnerable."
            ),
            mitigation="Update opencv-python once it bundles libpng >= 1.6.54.",
        ))
    package_versions = {
        str(dep.get("name") or "").lower().replace("_", "-"): str(dep.get("version") or "")
        for dep in dependencies
        if dep.get("name")
    }
    ort_status = collect_onnxruntime_provider_status(
        package_versions=package_versions,
    )
    findings.extend(onnxruntime_release_advisories(ort_status))

    ffmpeg_advisory = ffmpeg_security_advisory()
    if ffmpeg_advisory is not None:
        findings.append(ffmpeg_advisory)

    pyinstaller_version = _dependency_version(dependencies, "pyinstaller")
    if pyinstaller_version and _version_lt(pyinstaller_version, "6.10.0"):
        findings.append(_advisory(
            advisory_id="CVE-2025-59042",
            package="pyinstaller",
            installed_version=pyinstaller_version,
            affected="<6.10.0",
            fixed_in="6.10.0",
            severity="high",
            source="https://github.com/pyinstaller/pyinstaller/security/advisories/GHSA-9w2p-rh8c-v9g5",
            mitigation=(
                "Build with PyInstaller >= 6.10.0; older bootloaders allow a "
                "writable-CWD sys.path injection (local privilege escalation)."
            ),
        ))

    nsis_advisory = nsis_security_advisory()
    if nsis_advisory is not None:
        findings.append(nsis_advisory)

    blocking = [item for item in findings if item.get("blocking")]
    return {
        "schema": "vsr.release_advisories.v1",
        "generatedAt": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "policy": {
            "strictBlocks": sorted(STRICT_BLOCKING_SEVERITIES),
            "allowedRuntimeExceptions": [
                item["id"] for item in findings if item.get("allowed")
            ],
        },
        "advisories": findings,
        "summary": {
            "total": len(findings),
            "blocking": len(blocking),
            "allowed": sum(1 for item in findings if item.get("allowed")),
        },
    }


def probe_makensis_version() -> str:
    """Return the makensis version string (e.g. '3.12'), '' if unavailable."""
    path = shutil.which("makensis")
    if not path:
        return ""
    try:
        proc = subprocess.run(
            [path, "-VERSION"], capture_output=True, text=True,
            timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (proc.stdout or proc.stderr or "").strip()


def nsis_security_advisory(version: Optional[str] = None) -> Optional[dict]:
    """Block installer builds that predate the NSIS 3.12 SYSTEM LPE fix.

    Returns None when makensis is absent (not every build machine installs it)
    or when the version is current. NSIS 3.12 stopped elevated installers from
    using the Low IL temp directory, preventing a possible SYSTEM escalation.
    """
    if version is None:
        version = probe_makensis_version()
    if not version:
        return None
    if not _version_lt(version, NSIS_MINIMUM_VERSION):
        return None
    return _advisory(
        advisory_id="NSIS-2026-LOW-IL-TEMP",
        package="nsis (makensis)",
        installed_version=version,
        affected=f"<{NSIS_MINIMUM_VERSION}",
        fixed_in=NSIS_MINIMUM_VERSION,
        severity="high",
        source="https://nsis.sourceforge.io/Docs/AppendixF.html#v3.12",
        mitigation=(
            f"Build the installer with NSIS >= {NSIS_MINIMUM_VERSION}; older "
            "releases can use a Low IL temp directory while elevated, enabling "
            "a possible privilege escalation for SYSTEM installers."
        ),
    )


def ffmpeg_security_advisory(
    status: Optional[Mapping[str, object]] = None,
) -> Optional[dict]:
    """Return a blocker unless FFmpeg is on an explicitly reviewed safe line.

    Accepts a pre-computed classification (from ``classify_ffmpeg_security`` or
    ``probe_ffmpeg_security``) for testability; probes the local runtime when
    not supplied. Legacy callers that only provide ``vulnerable=False`` retain
    their earlier safe meaning, but live probes always include classification.
    """
    if status is None:
        status = probe_ffmpeg_security()
    classification = str(status.get("classification") or "").lower()
    if not classification:
        classification = "vulnerable" if status.get("vulnerable") else "safe"
    if classification == "safe":
        return None
    version = str(status.get("version") or "unknown")
    if classification == "vulnerable":
        fixed_in = str(status.get("fixed_in") or "8.1.2")
        advisories = status.get("advisories") or []
        advisory_id = advisories[0] if advisories else "CVE-2026-8461"
        affected = "8.1.0-8.1.1, 8.0.0-8.0.2"
        source = FFMPEG_SECURITY_SOURCE
    elif classification == "unsupported":
        advisory_id = "FFMPEG-UNSUPPORTED-BRANCH"
        affected = "outside VSR's reviewed 8.0/8.1 branches"
        fixed_in = "8.1.2 (reviewed stable branch)"
        source = FFMPEG_RELEASE_SOURCE
    else:
        advisory_id = "FFMPEG-UNCLASSIFIED-VERSION"
        affected = "snapshot, missing, or unclassified future branch"
        fixed_in = "8.1.2 (reviewed stable branch)"
        source = FFMPEG_RELEASE_SOURCE
    return _advisory(
        advisory_id=str(advisory_id),
        package="ffmpeg (external runtime)",
        installed_version=version,
        affected=affected,
        fixed_in=fixed_in,
        severity="high",
        source=source,
        mitigation=(
            "VSR decodes untrusted media through FFmpeg; install a reviewed "
            "stable FFmpeg 8.1.2+ build (or 8.0.3+ on the 8.0 branch) before "
            f"shipping. Probe result: {status.get('reason') or classification}."
        ),
    )


def _tool_version(command: Sequence[str], timeout: float = 10.0) -> dict:
    exe = command[0]
    path = shutil.which(exe)
    payload = {
        "command": list(command),
        "path": path or "",
        "available": bool(path),
        "version": "",
        "error": "",
    }
    if not path:
        return payload
    try:
        proc = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        output = (proc.stdout or proc.stderr or "").splitlines()
        payload["version"] = output[0] if output else ""
        payload["returncode"] = proc.returncode
    except Exception as exc:  # pragma: no cover - platform/tool dependent
        payload["error"] = str(exc)
    return payload


def _ffmpeg_encoder_status() -> dict:
    ffmpeg = shutil.which("ffmpeg")
    payload = {
        "available": bool(ffmpeg),
        "path": ffmpeg or "",
        "hasLibvvenc": False,
        "error": "",
    }
    if not ffmpeg:
        return payload
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        text = (proc.stdout or "") + (proc.stderr or "")
        payload["hasLibvvenc"] = "libvvenc" in text
        payload["returncode"] = proc.returncode
    except Exception as exc:  # pragma: no cover - platform/tool dependent
        payload["error"] = str(exc)
    return payload


def _ffmpeg_subprocess_smoke(timeout: float = 30.0) -> dict:
    """Exercise ffprobe and ffmpeg with a tiny synthetic fixture.

    Records command, path, env, return code, and output so release
    evidence proves the packaged app can safely launch external tools.
    """
    schema = "vsr.ffmpeg_subprocess_smoke.v1"
    ffmpeg_path = shutil.which("ffmpeg") or ""
    ffprobe_path = shutil.which("ffprobe") or ""
    payload = {
        "schema": schema,
        "ran": False,
        "passed": False,
        "ffmpegPath": ffmpeg_path,
        "ffprobePath": ffprobe_path,
        "ffmpegAvailable": bool(ffmpeg_path),
        "ffprobeAvailable": bool(ffprobe_path),
        "generate": {"ran": False, "passed": False, "error": ""},
        "probe": {"ran": False, "passed": False, "error": "", "codec": "", "width": None, "height": None, "frames": None},
        "transcode": {"ran": False, "passed": False, "error": ""},
        "env": {
            "PATH": os.environ.get("PATH", "")[:2000],
            "frozen": bool(getattr(sys, "frozen", False)),
        },
        "error": "",
    }
    if not ffmpeg_path or not ffprobe_path:
        payload["error"] = "ffmpeg or ffprobe not on PATH"
        return payload

    tmpdir = None
    try:
        tmpdir = tempfile.mkdtemp(prefix="vsr-ffsmoke-")
        fixture = os.path.join(tmpdir, "smoke_fixture.avi")
        transcoded = os.path.join(tmpdir, "smoke_out.mkv")

        gen_cmd = [
            ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=black:s=32x32:r=2:d=0.5",
            "-c:v", "rawvideo", "-pix_fmt", "bgr24",
            fixture,
        ]
        gen_proc = subprocess.run(
            gen_cmd, capture_output=True, text=True, timeout=timeout,
            check=False,
        )
        payload["generate"]["ran"] = True
        payload["generate"]["command"] = gen_cmd
        if gen_proc.returncode != 0:
            payload["generate"]["error"] = (gen_proc.stderr or "")[-1000:]
            payload["error"] = "ffmpeg fixture generation failed"
            payload["ran"] = True
            return payload
        payload["generate"]["passed"] = True

        probe_cmd = [
            ffprobe_path, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,nb_frames",
            "-of", "json", fixture,
        ]
        probe_proc = subprocess.run(
            probe_cmd, capture_output=True, text=True, timeout=timeout,
            check=False,
        )
        payload["probe"]["ran"] = True
        payload["probe"]["command"] = probe_cmd
        if probe_proc.returncode != 0:
            payload["probe"]["error"] = (probe_proc.stderr or "")[-1000:]
            payload["error"] = "ffprobe failed on generated fixture"
            payload["ran"] = True
            return payload
        try:
            probe_data = json.loads(probe_proc.stdout or "{}")
            streams = probe_data.get("streams") or []
            stream = streams[0] if streams else {}
            payload["probe"]["codec"] = str(stream.get("codec_name") or "")
            payload["probe"]["width"] = stream.get("width")
            payload["probe"]["height"] = stream.get("height")
            nb = stream.get("nb_frames")
            payload["probe"]["frames"] = int(nb) if nb and str(nb).isdigit() else None
        except (json.JSONDecodeError, IndexError, TypeError) as exc:
            payload["probe"]["error"] = str(exc)
            payload["error"] = "ffprobe returned invalid JSON"
            payload["ran"] = True
            return payload
        payload["probe"]["passed"] = True

        tc_cmd = [
            ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error",
            "-i", fixture,
            "-c:v", "ffv1", "-g", "1",
            transcoded,
        ]
        tc_proc = subprocess.run(
            tc_cmd, capture_output=True, text=True, timeout=timeout,
            check=False,
        )
        payload["transcode"]["ran"] = True
        payload["transcode"]["command"] = tc_cmd
        if tc_proc.returncode != 0:
            payload["transcode"]["error"] = (tc_proc.stderr or "")[-1000:]
            payload["error"] = "ffmpeg transcode failed"
            payload["ran"] = True
            return payload
        payload["transcode"]["passed"] = os.path.isfile(transcoded)
        if not payload["transcode"]["passed"]:
            payload["error"] = "transcode output file missing"
            payload["ran"] = True
            return payload

        payload["ran"] = True
        payload["passed"] = True
    except Exception as exc:
        payload["error"] = str(exc)
        payload["ran"] = True
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
    return payload


def _frozen_launcher_status(dist_dir: Path, name: str) -> dict:
    status = _dist_file_status(dist_dir, name)
    status.update({
        "nativeExecutable": False,
        "forbiddenReferences": [],
    })
    path = dist_dir / name
    if not path.is_file():
        return status
    try:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
    except OSError as exc:
        status["readError"] = str(exc)
        return status
    forbidden = [
        token for token in ("setup.py", "videosubtitleremover.py", "venv\\")
        if token in text
    ]
    status["forbiddenReferences"] = forbidden
    status["nativeExecutable"] = (
        "videosubtitleremoverpro.exe" in text and not forbidden
    )
    return status


def _smoke_entry_command(path: Path) -> list[str]:
    if path.suffix.lower() == ".bat":
        comspec = os.environ.get("COMSPEC") or shutil.which("cmd.exe") or "cmd.exe"
        return [comspec, "/d", "/c", str(path), "--smoke-test"]
    if path.suffix.lower() == ".ps1":
        powershell = shutil.which("pwsh") or shutil.which("powershell") or "powershell.exe"
        return [
            powershell,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(path),
            "--smoke-test",
        ]
    return [str(path), "--smoke-test"]


def _link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def _run_smoke(dist_dir: Path, timeout: float = 45.0) -> dict:
    source_exe = dist_dir / "VideoSubtitleRemoverPro.exe"
    payload = {
        "schema": "vsr.frozen_entrypoint_smoke.v1",
        "exe": str(source_exe),
        "available": source_exe.is_file(),
        "ran": False,
        "passed": False,
        "temporaryCopy": "",
        "entryPoints": [],
        "error": "",
    }
    if not source_exe.is_file():
        payload["error"] = "frozen executable not found"
        return payload
    try:
        with tempfile.TemporaryDirectory(prefix="vsr-frozen-smoke-") as tmp:
            smoke_dist = Path(tmp) / "Video Subtitle Remover Pro"
            shutil.copytree(dist_dir, smoke_dist, copy_function=_link_or_copy)
            payload["temporaryCopy"] = str(smoke_dist)
            smoke_env = os.environ.copy()
            smoke_env["VSR_LAUNCHER_WAIT"] = "1"
            smoke_env["VSR_LAUNCHER_SMOKE"] = "1"
            for name in ("VideoSubtitleRemoverPro.exe", *LAUNCHERS):
                path = smoke_dist / name
                command = _smoke_entry_command(path)
                result = {
                    "name": name,
                    "path": str(path),
                    "command": command,
                    "available": path.is_file(),
                    "ran": False,
                    "passed": False,
                    "returncode": None,
                    "stdout": "",
                    "stderr": "",
                    "error": "",
                }
                if not path.is_file():
                    result["error"] = "entry point not found"
                    payload["entryPoints"].append(result)
                    continue
                try:
                    proc = subprocess.run(
                        command,
                        cwd=smoke_dist,
                        env=smoke_env,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        check=False,
                    )
                    result.update({
                        "ran": True,
                        "passed": proc.returncode == 0,
                        "returncode": proc.returncode,
                        "stdout": (proc.stdout or "")[-4000:],
                        "stderr": (proc.stderr or "")[-4000:],
                    })
                except Exception as exc:  # pragma: no cover - frozen runtime
                    result["error"] = str(exc)
                payload["entryPoints"].append(result)
            payload["ran"] = any(item["ran"] for item in payload["entryPoints"])
            payload["passed"] = bool(payload["entryPoints"]) and all(
                item["passed"] for item in payload["entryPoints"]
            )
    except Exception as exc:  # pragma: no cover - depends on frozen build
        payload["error"] = str(exc)
    return payload


def _installer_status(installer_path: str | Path | None) -> dict:
    path = Path(installer_path) if installer_path else Path()
    available = bool(installer_path) and path.is_file()
    valid_pe = False
    if available:
        try:
            valid_pe = path.read_bytes()[:2] == b"MZ"
        except OSError:
            valid_pe = False
    return {
        "schema": "vsr.installer_artifact.v1",
        "path": str(path) if installer_path else "",
        "available": available,
        "validPortableExecutable": valid_pe,
        "sha256": sha256_file(path) if available else "",
        "sizeBytes": path.stat().st_size if available else 0,
    }


def _run_installer_smoke(
    installed_executable: str | Path | None,
    timeout: float = 45.0,
) -> dict:
    path = Path(installed_executable) if installed_executable else Path()
    payload = {
        "schema": "vsr.installer_smoke.v1",
        "path": str(path) if installed_executable else "",
        "available": bool(installed_executable) and path.is_file(),
        "ran": False,
        "passed": False,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "error": "",
    }
    if not payload["available"]:
        payload["error"] = "installer smoke executable not found"
        return payload
    env = os.environ.copy()
    env["VSR_LAUNCHER_WAIT"] = "1"
    env["VSR_LAUNCHER_SMOKE"] = "1"
    try:
        proc = subprocess.run(
            [str(path), "--smoke-test"],
            cwd=path.parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        payload.update({
            "ran": True,
            "passed": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "")[-4000:],
            "stderr": (proc.stderr or "")[-4000:],
        })
    except Exception as exc:  # pragma: no cover - depends on frozen runtime
        payload["error"] = str(exc)
    return payload


def _component_property(component: Mapping[str, object], name: str) -> str:
    for item in component.get("properties", []):
        if isinstance(item, Mapping) and item.get("name") == name:
            return str(item.get("value") or "")
    return ""


def _audit_frozen_dependencies(sbom: Mapping[str, object], timeout: float = 300.0) -> dict:
    requirements = []
    for component in sbom.get("components", []):
        if not isinstance(component, Mapping):
            continue
        if component.get("type") != "library" or component.get("scope") != "required":
            continue
        name = str(component.get("name") or "").strip()
        version = str(component.get("version") or "").strip()
        if name and version:
            requirements.append(f"{name}=={version}")
    requirements = sorted(set(requirements), key=str.lower)
    payload = {
        "schema": "vsr.pip_audit.v1",
        "scope": "frozen-runtime-components",
        "requirements": requirements,
        "ran": False,
        "passed": False,
        "returncode": None,
        "vulnerabilityCount": 0,
        "skippedCount": 0,
        "dependencies": [],
        "stderr": "",
        "error": "",
    }
    if not requirements:
        payload["error"] = "artifact SBOM contains no auditable runtime libraries"
        return payload
    try:
        with tempfile.TemporaryDirectory(prefix="vsr-pip-audit-") as tmp:
            requirement_path = Path(tmp) / "frozen-requirements.txt"
            requirement_path.write_text("\n".join(requirements) + "\n", encoding="utf-8")
            command = [
                sys.executable,
                "-m",
                "pip_audit",
                "--requirement",
                str(requirement_path),
                "--no-deps",
                "--strict",
                "--progress-spinner",
                "off",
                "--format",
                "json",
            ]
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        report = json.loads(proc.stdout or "{}")
        dependencies = report.get("dependencies", [])
        vulnerabilities = sum(
            len(item.get("vulns", []))
            for item in dependencies
            if isinstance(item, Mapping)
        )
        skipped = sum(
            1 for item in dependencies
            if isinstance(item, Mapping) and item.get("skip_reason")
        )
        payload.update({
            "ran": True,
            "passed": proc.returncode == 0 and vulnerabilities == 0 and skipped == 0,
            "returncode": proc.returncode,
            "vulnerabilityCount": vulnerabilities,
            "skippedCount": skipped,
            "dependencies": dependencies,
            "stderr": (proc.stderr or "")[-8000:],
        })
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        payload["ran"] = True
        payload["error"] = str(exc)
    return payload


def _run_reference_corpus(timeout_note: str = "local release") -> dict:
    payload = {
        "schema": "vsr.reference_corpus.v1",
        "ran": True,
        "passed": False,
        "clipCount": 0,
        "failures": [],
        "context": timeout_note,
        "error": "",
    }
    try:
        from backend.reference_corpus import run_reference_corpus

        with tempfile.TemporaryDirectory(prefix="vsr-release-corpus-") as tmpdir:
            result = run_reference_corpus(output_dir=tmpdir)
        payload.update({
            "passed": bool(result.get("passed")),
            "clipCount": int(result.get("clipCount", 0)),
            "failures": result.get("failures", []),
        })
    except Exception as exc:  # pragma: no cover - exercises installed env
        payload["error"] = str(exc)
    return payload


def build_release_evidence(
    *,
    dist_dir: str | Path,
    analysis_path: str | Path | None = None,
    installer_path: str | Path | None = None,
    installer_smoke_executable: str | Path | None = None,
    hidden_imports: str | Sequence[str] = (),
    runtime_hooks: str | Sequence[str] = (),
    excludes: str | Sequence[str] = (),
    collect_data: str | Sequence[str] = (),
    run_smoke: bool = True,
    run_reference_corpus: bool = False,
    run_dependency_audit: bool = False,
    env: Optional[Mapping[str, str]] = None,
) -> tuple[dict, dict, dict, dict]:
    dist = Path(dist_dir)
    hidden = parse_hidden_imports(hidden_imports)
    hooks = parse_runtime_hooks(runtime_hooks)
    excluded = parse_excluded_modules(excludes)
    collected = tuple(str(collect_data).split()) if isinstance(collect_data, str) else tuple(collect_data)
    dependencies = collect_dependency_versions()
    package_versions = {
        str(dep.get("name") or "").lower().replace("_", "-"): str(dep.get("version") or "")
        for dep in dependencies
        if dep.get("name")
    }
    sbom = build_cyclonedx_sbom(
        dependencies,
        dist_dir=dist,
        analysis_path=analysis_path,
    )
    sbom_metadata = sbom.get("metadata", {})
    artifact_derived = _component_property(
        sbom_metadata if isinstance(sbom_metadata, Mapping) else {},
        "vsr:artifactDerived",
    ) == "true"
    dependency_audit = (
        _audit_frozen_dependencies(sbom)
        if run_dependency_audit else
        {
            "schema": "vsr.pip_audit.v1",
            "scope": "frozen-runtime-components",
            "requirements": [],
            "ran": False,
            "passed": None,
            "returncode": None,
            "vulnerabilityCount": 0,
            "skippedCount": 0,
            "dependencies": [],
            "stderr": "",
            "error": "dependency audit skipped",
        }
    )
    onnxruntime_providers = collect_onnxruntime_provider_status(
        package_versions=package_versions,
    )
    opencv_wheels = collect_opencv_wheel_status(
        package_versions=package_versions,
    )
    rapidocr_engines = collect_rapidocr_engine_status(
        package_versions=package_versions,
    )
    advisories = collect_release_advisories(dependencies, env=env)
    hidden_payload = {
        "schema": "vsr.release_hidden_imports.v1",
        "hiddenImports": list(hidden),
        "runtimeHooks": list(hooks),
        "collectData": list(collected),
        "excludedModules": list(excluded),
    }
    documents = [_dist_file_status(dist, name) for name in DOCUMENTS]
    launchers = [_frozen_launcher_status(dist, name) for name in LAUNCHERS]
    version_docs = [_doc_version_status(ROOT / name) for name in VERSIONED_DOCS]
    evidence = {
        "schema": "vsr.release_verification.v1",
        "app": {
            "name": APP_NAME,
            "version": APP_VERSION,
        },
        "generatedAt": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "distDir": str(dist),
        "documents": documents,
        "launchers": launchers,
        "versionChecks": {
            "appVersion": APP_VERSION,
            "documents": version_docs,
        },
        "dependencies": dependencies,
        "hiddenImports": list(hidden),
        "runtimeHooks": list(hooks),
        "excludedModules": list(excluded),
        "adapterSecurity": list(release_manifest_status(env=env)),
        "remoteModelSecurity": list(release_remote_model_status(env=env)),
        "rapidocrModels": rapidocr_release_provenance(),
        "releaseTools": {
            "python": {
                "executable": sys.executable,
                "version": sys.version.split()[0],
            },
            "pyinstaller": _tool_version([sys.executable, "-m", "PyInstaller", "--version"]),
            "pyinstallerRuntimeHooks": [
                _source_file_status(item) for item in hooks
            ],
            "ffmpeg": _tool_version(["ffmpeg", "-version"]),
            "ffmpegEncoders": _ffmpeg_encoder_status(),
            "ffmpegProfiles": collect_ffmpeg_capability_profiles(),
            "opencvWheels": opencv_wheels,
            "onnxRuntimeProviders": onnxruntime_providers,
            "rapidocrEngines": rapidocr_engines,
            "referenceCorpus": (
                _run_reference_corpus()
                if run_reference_corpus else
                {
                    "schema": "vsr.reference_corpus.v1",
                    "ran": False,
                    "passed": None,
                    "clipCount": 0,
                    "failures": [],
                    "error": "reference corpus skipped",
                }
            ),
            "wingetcreate": _tool_version(["wingetcreate.exe", "--version"]),
            "ffmpegSubprocessSmoke": _ffmpeg_subprocess_smoke(),
            "dependencyDrift": collect_dependency_drift_report(
                package_versions=package_versions,
            ),
            "adapterConformance": collect_adapter_conformance_matrix(env=env),
            "dependencyAudit": dependency_audit,
        },
        "smokeLaunch": _run_smoke(dist) if run_smoke else {
            "ran": False,
            "passed": None,
            "error": "smoke launch skipped",
        },
        "sbom": {
            "file": "sbom.cdx.json",
            "componentCount": len(sbom.get("components", [])),
            "runtimeComponentCount": sum(
                1 for item in sbom.get("components", [])
                if isinstance(item, Mapping)
                and item.get("type") == "library"
                and item.get("scope") == "required"
            ),
            "nativeComponentCount": sum(
                1 for item in sbom.get("components", [])
                if isinstance(item, Mapping) and item.get("type") == "file"
            ),
            "artifactDerived": artifact_derived,
            "analysisPath": str(analysis_path or ""),
        },
        "installer": _installer_status(installer_path),
        "installerSmoke": _run_installer_smoke(installer_smoke_executable),
        "advisories": {
            "file": "release-advisories.json",
            "total": advisories["summary"]["total"],
            "blocking": advisories["summary"]["blocking"],
            "allowed": advisories["summary"]["allowed"],
        },
    }
    evidence["errors"] = list(_validation_errors(evidence))
    return evidence, hidden_payload, sbom, advisories


def _validation_errors(evidence: Mapping[str, object]) -> Iterable[str]:
    for item in evidence.get("documents", []):
        if isinstance(item, Mapping) and not item.get("bundled"):
            yield f"Required bundled document missing: {item.get('name')}"
    for item in evidence.get("launchers", []):
        if not isinstance(item, Mapping):
            continue
        if not item.get("bundled"):
            yield f"Required bundled launcher missing: {item.get('name')}"
        elif not item.get("nativeExecutable"):
            refs = ", ".join(str(ref) for ref in item.get("forbiddenReferences", []))
            detail = f" (source references: {refs})" if refs else ""
            yield f"Bundled launcher is not frozen-native: {item.get('name')}{detail}"
    for item in evidence.get("versionChecks", {}).get("documents", []):
        if isinstance(item, Mapping) and not item.get("containsAppVersion"):
            yield f"APP_VERSION {APP_VERSION} missing from {item.get('name')}"
    smoke = evidence.get("smokeLaunch", {})
    if isinstance(smoke, Mapping) and smoke.get("ran") and not smoke.get("passed"):
        yield "Smoke launch failed"
    reference = evidence.get("releaseTools", {}).get("referenceCorpus", {})
    if (isinstance(reference, Mapping)
            and reference.get("ran") and not reference.get("passed")):
        yield "Reference corpus regression failed"
    ffsmoke = evidence.get("releaseTools", {}).get("ffmpegSubprocessSmoke", {})
    if (isinstance(ffsmoke, Mapping)
            and ffsmoke.get("ran") and not ffsmoke.get("passed")):
        yield f"FFmpeg subprocess smoke failed: {ffsmoke.get('error', 'unknown')}"
    runtime_hooks = evidence.get("releaseTools", {}).get("pyinstallerRuntimeHooks", [])
    if isinstance(runtime_hooks, list):
        hook_names = {
            str(item.get("name") or "").replace("\\", "/")
            for item in runtime_hooks
            if isinstance(item, Mapping) and item.get("sourceExists")
        }
        if REQUIRED_RUNTIME_HOOK not in hook_names:
            yield f"Required PyInstaller runtime hook missing: {REQUIRED_RUNTIME_HOOK}"
    rapid = evidence.get("rapidocrModels", {})
    if isinstance(rapid, Mapping):
        package = rapid.get("package", {})
        package_name = (
            package.get("name") if isinstance(package, Mapping) else ""
        )
        if package_name and not rapid.get("packaging_compatible"):
            yield "RapidOCR packaged model evidence is incomplete"


def write_release_evidence(
    *,
    dist_dir: str | Path,
    evidence_dir: str | Path | None = None,
    analysis_path: str | Path | None = None,
    installer_path: str | Path | None = None,
    installer_smoke_executable: str | Path | None = None,
    hidden_imports: str | Sequence[str] = (),
    runtime_hooks: str | Sequence[str] = (),
    excludes: str | Sequence[str] = (),
    collect_data: str | Sequence[str] = (),
    run_smoke: bool = True,
    run_reference_corpus: bool = False,
    run_dependency_audit: bool = False,
    strict: bool = False,
    env: Optional[Mapping[str, str]] = None,
) -> dict:
    out_dir = Path(evidence_dir) if evidence_dir is not None else Path(dist_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence, hidden_payload, sbom, advisories = build_release_evidence(
        dist_dir=dist_dir,
        analysis_path=analysis_path,
        installer_path=installer_path,
        installer_smoke_executable=installer_smoke_executable,
        hidden_imports=hidden_imports,
        runtime_hooks=runtime_hooks,
        excludes=excludes,
        collect_data=collect_data,
        run_smoke=run_smoke,
        run_reference_corpus=run_reference_corpus,
        run_dependency_audit=run_dependency_audit,
        env=env,
    )
    outputs = {
        "release-verification.json": evidence,
        "release-hidden-imports.json": hidden_payload,
        "release-advisories.json": advisories,
        "sbom.cdx.json": sbom,
        "pip-audit.json": evidence.get("releaseTools", {}).get(
            "dependencyAudit", {}
        ),
    }
    for filename, payload in outputs.items():
        (out_dir / filename).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    strict_errors = [str(err) for err in evidence.get("errors", [])]
    strict_errors.extend(
        f"{item.get('id')} {item.get('package')} {item.get('installedVersion')}"
        for item in advisories.get("advisories", [])
        if isinstance(item, Mapping) and item.get("blocking")
    )
    if strict:
        if not evidence.get("sbom", {}).get("artifactDerived"):
            strict_errors.append(
                "SBOM is not derived from a PyInstaller Analysis TOC"
            )
        audit = evidence.get("releaseTools", {}).get("dependencyAudit", {})
        if not isinstance(audit, Mapping) or not audit.get("ran"):
            strict_errors.append("Frozen-runtime dependency audit did not run")
        elif not audit.get("passed"):
            strict_errors.append(
                "Frozen-runtime dependency audit found vulnerabilities or skipped packages"
            )
        installer = evidence.get("installer", {})
        if (not isinstance(installer, Mapping)
                or not installer.get("validPortableExecutable")):
            strict_errors.append("NSIS installer artifact is missing or invalid")
        installer_smoke = evidence.get("installerSmoke", {})
        if (not isinstance(installer_smoke, Mapping)
                or not installer_smoke.get("ran")
                or not installer_smoke.get("passed")):
            strict_errors.append("Installer payload smoke did not pass")
    if strict and strict_errors:
        raise SystemExit(
            "Strict release verification failed:\n- "
            + "\n- ".join(strict_errors)
        )
    return evidence


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate local VSR release evidence."
    )
    parser.add_argument("--dist-dir", default="dist/VideoSubtitleRemoverPro")
    parser.add_argument("--evidence-dir", default="")
    parser.add_argument("--analysis-path", default="")
    parser.add_argument("--installer-path", default="")
    parser.add_argument("--installer-smoke-executable", default="")
    parser.add_argument("--hidden-imports", default="")
    parser.add_argument("--runtime-hooks", default="")
    parser.add_argument("--excludes", default="")
    parser.add_argument("--collect-data", default="")
    parser.add_argument(
        "--quality",
        choices=("permissive", "strict"),
        default="permissive",
    )
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--run-reference-corpus", action="store_true")
    parser.add_argument("--run-dependency-audit", action="store_true")
    args = parser.parse_args(argv)
    write_release_evidence(
        dist_dir=args.dist_dir,
        evidence_dir=args.evidence_dir or None,
        analysis_path=args.analysis_path or None,
        installer_path=args.installer_path or None,
        installer_smoke_executable=args.installer_smoke_executable or None,
        hidden_imports=args.hidden_imports,
        runtime_hooks=args.runtime_hooks,
        excludes=args.excludes,
        collect_data=args.collect_data,
        run_smoke=not args.skip_smoke,
        run_reference_corpus=args.run_reference_corpus,
        run_dependency_audit=args.run_dependency_audit,
        strict=args.quality == "strict",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
