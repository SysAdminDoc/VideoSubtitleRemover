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
    collect_dependency_drift_report,
    collect_opencv_wheel_status,
    collect_onnxruntime_provider_status,
    collect_rapidocr_engine_status,
    onnxruntime_release_advisories,
)
from backend.ffmpeg_profiles import collect_ffmpeg_capability_profiles
from backend.onnx_model_info import rapidocr_release_provenance
from backend.remote_model_policy import release_remote_model_status
from backend.security_checks import opencv_libpng_status


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ("README.md", "LICENSE", "CHANGELOG.md")
LAUNCHERS = ("Run_VSR_Pro.bat", "Run_VSR_Pro_Debug.bat", "Run_VSR_Pro.ps1")
VERSIONED_DOCS = ("README.md", "CHANGELOG.md")
HIDDEN_IMPORT_RE = re.compile(r"--hidden-import(?:=|\s+)([^\s]+)")
RUNTIME_HOOK_RE = re.compile(r"--runtime-hook(?:=|\s+)([^\s]+)")
STRICT_BLOCKING_SEVERITIES = {"high", "critical"}
REQUIRED_RUNTIME_HOOK = "assets/runtime_hook_mp.py"


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


def build_cyclonedx_sbom(dependencies: Sequence[Mapping[str, object]]) -> dict:
    components = []
    for dep in dependencies:
        name = str(dep.get("name") or "")
        version = str(dep.get("version") or "")
        if not name:
            continue
        components.append({
            "type": "library",
            "name": name,
            "version": version,
            "purl": f"pkg:pypi/{name.lower()}@{version}" if version else "",
        })
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
            "component": {
                "type": "application",
                "name": APP_NAME,
                "version": APP_VERSION,
            },
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
    if (pillow_version and _version_gte(pillow_version, "11.2.0")
            and _version_lt(pillow_version, "11.3.0")):
        findings.append(_advisory(
            advisory_id="CVE-2025-48379",
            package="Pillow",
            installed_version=pillow_version,
            affected=">=11.2.0,<11.3.0",
            fixed_in="11.3.0",
            severity="high",
            source="https://nvd.nist.gov/vuln/detail/CVE-2025-48379",
            mitigation="Upgrade Pillow before producing a strict release.",
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


def _run_smoke(dist_dir: Path, timeout: float = 45.0) -> dict:
    exe = dist_dir / "VideoSubtitleRemoverPro.exe"
    payload = {
        "command": [str(exe), "--smoke-test"],
        "exe": str(exe),
        "available": exe.is_file(),
        "ran": False,
        "passed": False,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "error": "",
    }
    if not exe.is_file():
        payload["error"] = "frozen executable not found"
        return payload
    try:
        proc = subprocess.run(
            [str(exe), "--smoke-test"],
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
    except Exception as exc:  # pragma: no cover - depends on frozen build
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
    hidden_imports: str | Sequence[str] = (),
    runtime_hooks: str | Sequence[str] = (),
    collect_data: str | Sequence[str] = (),
    run_smoke: bool = True,
    run_reference_corpus: bool = False,
    env: Optional[Mapping[str, str]] = None,
) -> tuple[dict, dict, dict]:
    dist = Path(dist_dir)
    hidden = parse_hidden_imports(hidden_imports)
    hooks = parse_runtime_hooks(runtime_hooks)
    collected = tuple(str(collect_data).split()) if isinstance(collect_data, str) else tuple(collect_data)
    dependencies = collect_dependency_versions()
    package_versions = {
        str(dep.get("name") or "").lower().replace("_", "-"): str(dep.get("version") or "")
        for dep in dependencies
        if dep.get("name")
    }
    sbom = build_cyclonedx_sbom(dependencies)
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
    }
    documents = [_dist_file_status(dist, name) for name in DOCUMENTS]
    launchers = [_dist_file_status(dist, name) for name in LAUNCHERS]
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
        },
        "smokeLaunch": _run_smoke(dist) if run_smoke else {
            "ran": False,
            "passed": None,
            "error": "smoke launch skipped",
        },
        "sbom": {
            "file": "sbom.cdx.json",
            "componentCount": len(sbom.get("components", [])),
        },
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
        if isinstance(item, Mapping) and not item.get("bundled"):
            yield f"Required bundled launcher missing: {item.get('name')}"
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
    hidden_imports: str | Sequence[str] = (),
    runtime_hooks: str | Sequence[str] = (),
    collect_data: str | Sequence[str] = (),
    run_smoke: bool = True,
    run_reference_corpus: bool = False,
    strict: bool = False,
    env: Optional[Mapping[str, str]] = None,
) -> dict:
    out_dir = Path(evidence_dir) if evidence_dir is not None else Path(dist_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence, hidden_payload, sbom, advisories = build_release_evidence(
        dist_dir=dist_dir,
        hidden_imports=hidden_imports,
        runtime_hooks=runtime_hooks,
        collect_data=collect_data,
        run_smoke=run_smoke,
        run_reference_corpus=run_reference_corpus,
        env=env,
    )
    outputs = {
        "release-verification.json": evidence,
        "release-hidden-imports.json": hidden_payload,
        "release-advisories.json": advisories,
        "sbom.cdx.json": sbom,
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
    parser.add_argument("--hidden-imports", default="")
    parser.add_argument("--runtime-hooks", default="")
    parser.add_argument("--collect-data", default="")
    parser.add_argument(
        "--quality",
        choices=("permissive", "strict"),
        default="permissive",
    )
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--run-reference-corpus", action="store_true")
    args = parser.parse_args(argv)
    write_release_evidence(
        dist_dir=args.dist_dir,
        evidence_dir=args.evidence_dir or None,
        hidden_imports=args.hidden_imports,
        runtime_hooks=args.runtime_hooks,
        collect_data=args.collect_data,
        run_smoke=not args.skip_smoke,
        run_reference_corpus=args.run_reference_corpus,
        strict=args.quality == "strict",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
