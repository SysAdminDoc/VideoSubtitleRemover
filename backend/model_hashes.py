"""Vendored SHA-256 hashes for model weights downloaded by opt-in features.

RM-49: when a feature fetches model weights from the internet on first
use, we verify the downloaded file's hash against a known-good value
shipped with the source tree. Catches:

- Silent supply-chain substitution at the upstream CDN.
- Half-downloaded weight files (truncated tensors crash hours into a
  long render).
- Local on-disk corruption that would otherwise present as nondescript
  runtime errors deep inside the model loader.

`verify_weight_file()` reads in 1 MiB chunks so a multi-GB file does
not need to fit in RAM. Returns True on match; False AND logs a
warning on mismatch. Callers should treat a False return as "fall back
to the cv2 inpainter" rather than crash hard, because hash mismatches
often arise from PyTorch hub cache corruption that resolves on a
manual re-download.

Add a new entry to KNOWN_WEIGHT_HASHES when wiring a new opt-in
inpainter / detector that pulls weights at runtime.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


KNOWN_WEIGHT_HASHES: Dict[str, str] = {
    # simple-lama-inpainting 0.1.x PyTorch checkpoint.
    # https://github.com/enesmsahin/simple-lama-inpainting
    "big-lama.pt": (
        "344c77bbcb158f17dd143070d1e789f38a66c04202311ae"
        "3a258ef66667a9ea9"
    ),
    # LaMa-ONNX FP32 -- Carve/LaMa-ONNX on HuggingFace.
    # https://huggingface.co/Carve/LaMa-ONNX
    "lama_fp32.onnx": (
        "1faef5301d78db7dda502fe59966957ec4b79dd64e16f0"
        "3ed96913c7a4eb68d6"
    ),
    "lama.onnx": (
        "351e481e287f345b7fbfd026068cfb9ec0c7f24b440e65"
        "01458ebe54a833d1a1"
    ),
    # OpenCV inpainting_lama (opencv/inpainting_lama on HuggingFace).
    # https://huggingface.co/opencv/inpainting_lama
    "inpainting_lama_2025jan.onnx": (
        "7df918ac3921d3daf0aae1d219776cf0dc4e4935f035af"
        "81841b40adcf74fdf2"
    ),
}


_CHUNK = 1 * 1024 * 1024  # 1 MiB streaming reads


def hash_file(path: Path) -> str:
    """Return the lowercase hex SHA-256 of the file at `path`."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_weight_file(path: Path, expected_filename: Optional[str] = None,
                       expected_hash: Optional[str] = None) -> bool:
    """Verify that `path` matches the expected SHA-256.

    Resolution order:
    1. If `expected_hash` is given, compare against that.
    2. Else, look up `expected_filename or path.name` in
       KNOWN_WEIGHT_HASHES.
    3. If neither path produces an expected hash, log an info message
       and return True -- absence of a hash entry should not block
       loading of weights for engines we have not pinned yet.

    Returns False only when there IS an expected hash and the file
    does not match it. Logs a clear warning on mismatch.
    """
    p = Path(path)
    if not p.exists():
        logger.warning(f"Weight verification skipped: {p} does not exist")
        return False
    if expected_hash is None:
        key = expected_filename or p.name
        expected_hash = KNOWN_WEIGHT_HASHES.get(key)
    if expected_hash is None:
        logger.debug(f"No vendored hash for {p.name}; skipping verification")
        return True
    actual = hash_file(p)
    if actual.lower() != expected_hash.lower():
        logger.warning(
            f"Weight file {p.name} failed SHA-256 verification "
            f"(expected {expected_hash[:16]}..., got {actual[:16]}...). "
            f"This typically means the file was truncated or the upstream "
            f"checkpoint was rotated. Re-download or delete {p} to retry."
        )
        return False
    logger.info(f"Verified weight file {p.name} via vendored SHA-256")
    return True


def scan_and_verify_weights() -> list:
    """Scan all candidate weight directories and verify known files.

    Returns a list of dicts: {filename, path, expected, actual, status}
    where status is "verified", "mismatch", "unknown", or "missing".
    """
    results = []
    dirs = candidate_weight_dirs()
    seen: set = set()
    for d in dirs:
        for f in sorted(d.iterdir()):
            if not f.is_file():
                continue
            key = f.name
            if key in seen:
                continue
            seen.add(key)
            expected = KNOWN_WEIGHT_HASHES.get(key)
            if expected is None:
                continue
            actual = hash_file(f)
            match = actual.lower() == expected.lower()
            results.append({
                "filename": key,
                "path": str(f),
                "expected": expected,
                "actual": actual,
                "status": "verified" if match else "mismatch",
            })
    for key in sorted(KNOWN_WEIGHT_HASHES):
        if key not in seen:
            results.append({
                "filename": key,
                "path": None,
                "expected": KNOWN_WEIGHT_HASHES[key],
                "actual": None,
                "status": "missing",
            })
    return results


def print_weight_report() -> None:
    """Print a human-readable weight verification report."""
    results = scan_and_verify_weights()
    print(f"Weight verification report ({len(KNOWN_WEIGHT_HASHES)} known hashes)")
    dirs = candidate_weight_dirs()
    print(f"Scanned directories: {len(dirs)}")
    for d in dirs:
        print(f"  {d}")
    print()
    if not results:
        print("No known weight files found.")
        return
    for r in results:
        tag = r["status"].upper()
        if r["path"]:
            print(f"  [{tag}] {r['filename']} -> {r['path']}")
        else:
            print(f"  [{tag}] {r['filename']}")


def candidate_weight_dirs() -> list:
    """Best-effort list of directories where model weight caches live.
    Used by verification scans that need to find a known weight file
    without depending on the loader's private resolution logic."""
    candidates = []
    # torch.hub default
    home = Path.home()
    candidates.append(home / ".cache" / "torch" / "hub" / "checkpoints")
    # simple-lama-inpainting expanded cache
    candidates.append(home / ".cache" / "simple_lama")
    # Per-app cache
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(Path(appdata) / "VideoSubtitleRemoverPro" / "models")
    return [c for c in candidates if c.exists()]
