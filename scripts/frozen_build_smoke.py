"""Build and smoke-test the committed PyInstaller specification.

This check is intentionally opt-in because a clean PyInstaller build is slow.
It never constructs the GUI: the frozen process imports NumPy and OpenCV,
writes a versioned JSON result marker, and exits.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from backend.subprocess_policy import run_process


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "VideoSubtitleRemoverPro.spec"
DEFAULT_DIST_ROOT = ROOT / "build" / "frozen-spec-smoke" / "dist"
DEFAULT_WORK_DIR = ROOT / "build" / "frozen-spec-smoke" / "work"


def build_command(
    *,
    spec: Path,
    dist_root: Path,
    work_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--distpath",
        str(dist_root),
        "--workpath",
        str(work_dir),
        str(spec),
    ]


def validate_smoke_payload(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise RuntimeError("frozen smoke marker is not a JSON object")
    if payload.get("schema") != "vsr.frozen_import_smoke.v1":
        raise RuntimeError("frozen smoke marker has an unexpected schema")
    if payload.get("passed") is not True:
        raise RuntimeError(str(payload.get("error") or "frozen import smoke failed"))
    imports = payload.get("imports")
    if not isinstance(imports, dict):
        raise RuntimeError("frozen smoke marker is missing import evidence")
    for name in ("numpy", "cv2"):
        if not str(imports.get(name) or "").strip():
            raise RuntimeError(f"frozen smoke marker is missing {name}")
    started = str(payload.get("startedMessage") or "")
    if not started.startswith("Video Subtitle Remover Pro v"):
        raise RuntimeError("frozen smoke marker is missing the startup message")
    return payload


def run_frozen_import_smoke(executable: Path, *, timeout: float) -> dict:
    if not executable.is_file():
        raise RuntimeError(f"frozen executable not found: {executable}")
    with tempfile.TemporaryDirectory(prefix="vsr-frozen-import-smoke-") as tmp:
        marker = Path(tmp) / "result.json"
        result = run_process(
            [str(executable), "--frozen-import-smoke", str(marker)],
            cwd=executable.parent,
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                f"frozen import smoke exited {result.returncode}: {detail}"
            )
        if not marker.is_file():
            raise RuntimeError("frozen import smoke did not write its result marker")
        return validate_smoke_payload(
            json.loads(marker.read_text(encoding="utf-8"))
        )


def build_and_smoke(
    *,
    spec: Path = DEFAULT_SPEC,
    dist_root: Path = DEFAULT_DIST_ROOT,
    work_dir: Path = DEFAULT_WORK_DIR,
    skip_build: bool = False,
    timeout: float = 600.0,
) -> dict:
    spec = spec.resolve()
    dist_root = dist_root.resolve()
    work_dir = work_dir.resolve()
    if not spec.is_file():
        raise RuntimeError(f"PyInstaller spec not found: {spec}")
    if not skip_build:
        if dist_root.exists():
            shutil.rmtree(dist_root)
        if work_dir.exists():
            shutil.rmtree(work_dir)
        result = run_process(
            build_command(spec=spec, dist_root=dist_root, work_dir=work_dir),
            cwd=ROOT,
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"PyInstaller build failed: {detail[-8000:]}")
    executable = dist_root / "VideoSubtitleRemoverPro" / "VideoSubtitleRemoverPro.exe"
    return run_frozen_import_smoke(executable, timeout=min(timeout, 120.0))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--dist-root", type=Path, default=DEFAULT_DIST_ROOT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = build_and_smoke(
            spec=args.spec,
            dist_root=args.dist_root,
            work_dir=args.work_dir,
            skip_build=args.skip_build,
            timeout=args.timeout,
        )
    except Exception as exc:
        print(f"Frozen build smoke failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
