#!/usr/bin/env python3
"""Local CPU smoke for isolated installs and container builds."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, cwd: Path) -> None:
    print("[run] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _write_smoke_ppm(path: Path) -> None:
    width, height = 64, 40
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            base = 44 + (x * 80 // width)
            if 18 <= y <= 28 and 10 <= x <= 54:
                pixels.extend((238, 238, 238))
            else:
                pixels.extend((base, 70 + (y * 60 // height), 112))
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + bytes(pixels))


def _write_smoke_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "mode": "sttn",
                "device": "cpu",
                "sttn_skip_detection": True,
                "subtitle_area": [10, 18, 54, 29],
                "mask_dilate_px": 2,
                "mask_feather_px": 0,
                "tbe_enable": False,
                "preserve_audio": False,
                "prefetch_decode": False,
                "use_hw_encode": False,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def run_smoke(*, skip_self_test: bool, work_dir: Path | None, keep_artifacts: bool) -> Path:
    if not skip_self_test:
        _run([sys.executable, "-m", "backend.processor", "--self-test"], cwd=ROOT)

    owned_temp = None
    if work_dir is None:
        if keep_artifacts:
            work = Path(tempfile.mkdtemp(prefix="vsr-local-smoke-"))
        else:
            owned_temp = tempfile.TemporaryDirectory(prefix="vsr-local-smoke-")
            work = Path(owned_temp.name)
    else:
        work = work_dir
        work.mkdir(parents=True, exist_ok=True)

    try:
        source = work / "smoke_input.ppm"
        output = work / "smoke_output.png"
        config = work / "smoke_config.json"
        _write_smoke_ppm(source)
        _write_smoke_config(config)
        if output.exists():
            output.unlink()
        _run(
            [
                sys.executable,
                "-m",
                "backend.processor",
                "--input",
                str(source),
                "--output",
                str(output),
                "--config",
                str(config),
                "--mode",
                "sttn",
                "--gpu",
                "-1",
                "--no-audio",
                "--no-prefetch",
                "--no-hw-encode",
                "--no-color-preserve",
            ],
            cwd=ROOT,
        )
        if not output.is_file() or output.stat().st_size <= 0:
            raise RuntimeError(f"smoke output was not written: {output}")
        print(f"[ok] local CPU smoke output: {output}", flush=True)
        if keep_artifacts:
            return work
        return output
    finally:
        if owned_temp is not None:
            owned_temp.cleanup()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the local CPU self-test and a tiny CLI image cleanup smoke."
    )
    parser.add_argument(
        "--skip-self-test",
        action="store_true",
        help="Run only the generated-image CLI smoke.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Optional directory for smoke artifacts.",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep generated input/config/output files for inspection.",
    )
    args = parser.parse_args(argv)
    if shutil.which("ffmpeg") is None:
        print("[warn] ffmpeg is not on PATH; image smoke will still run.", flush=True)
    run_smoke(
        skip_self_test=args.skip_self_test,
        work_dir=args.work_dir,
        keep_artifacts=args.keep_artifacts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
