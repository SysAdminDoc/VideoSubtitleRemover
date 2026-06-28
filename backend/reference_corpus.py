"""Deterministic reference-clip corpus runner.

The committed fixtures in ``tests/clips`` are tiny generated videos with
manifested source hashes and output-frame baselines. This module verifies the
fixture license/hash policy, runs the backend with fixed manual masks, and
compares decoded output-frame hashes plus PSNR/SSIM floors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Mapping, Optional, Sequence
from urllib.parse import urlparse

import cv2
import numpy as np

from backend import processor


REFERENCE_CORPUS_SCHEMA = "vsr.reference_corpus.v1"
REFERENCE_CORPUS_CATEGORY = "core_reference"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "tests" / "clips" / "manifest.json"
REFERENCE_LICENSE_ALLOWLIST = {
    "cc0",
    "cc0-1.0",
    "public-domain",
    "public domain",
    "mit",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc-by-4.0",
}
REQUIRED_FIELDS = {
    "filename",
    "license",
    "contributor",
    "sha256",
    "failure_category",
    "config",
    "metric_floors",
    "baseline",
}
SYNTHETIC_SOURCE_TYPES = {
    "",
    "generated",
    "synthetic",
    "deterministic",
}
REAL_SOURCE_TYPES = {
    "real",
    "real-world",
    "community",
    "public-domain",
    "cc0",
}
REAL_SOURCE_REQUIRED_FIELDS = {
    "url",
    "license_url",
    "retrieved_at",
    "rights_confirmation",
}


class ReferenceCorpusError(ValueError):
    """Raised when the reference corpus cannot be evaluated."""


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(path: Path | str) -> dict:
    manifest_path = Path(path)
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReferenceCorpusError(
            f"reference manifest could not be read: {manifest_path}"
        ) from exc
    if not isinstance(data, dict):
        raise ReferenceCorpusError("reference manifest must be a JSON object")
    if not isinstance(data.get("clips"), list):
        raise ReferenceCorpusError("reference manifest clips must be a list")
    return data


def _safe_manifest_filename(value: object) -> str:
    filename = str(value or "").replace("\\", "/")
    if not filename or filename.startswith("/") or ".." in Path(filename).parts:
        raise ReferenceCorpusError(f"unsafe reference clip filename: {filename!r}")
    return filename


def _valid_sha256(value: object) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(c in "0123456789abcdef" for c in text)


def _normalise_license(value: object) -> str:
    return str(value or "").strip().lower()


def _normalise_source_type(value: object) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _valid_http_url(value: object) -> bool:
    text = str(value or "").strip()
    parsed = urlparse(text)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _validate_real_clip_source(entry: Mapping[str, object], filename: str) -> None:
    source_type = _normalise_source_type(entry.get("source_type"))
    if source_type in SYNTHETIC_SOURCE_TYPES:
        return
    if source_type not in REAL_SOURCE_TYPES:
        raise ReferenceCorpusError(
            f"reference clip {filename} has unsupported source_type "
            f"{entry.get('source_type')!r}"
        )
    source = entry.get("source")
    if not isinstance(source, Mapping):
        raise ReferenceCorpusError(
            f"real reference clip {filename} needs source metadata"
        )
    missing = [
        field for field in sorted(REAL_SOURCE_REQUIRED_FIELDS)
        if not str(source.get(field) or "").strip()
    ]
    if missing:
        raise ReferenceCorpusError(
            f"real reference clip {filename} missing source fields: {missing}"
        )
    if not _valid_http_url(source.get("url")):
        raise ReferenceCorpusError(
            f"real reference clip {filename} needs an http(s) source url"
        )
    if not _valid_http_url(source.get("license_url")):
        raise ReferenceCorpusError(
            f"real reference clip {filename} needs an http(s) license url"
        )
    source_license = _normalise_license(source.get("license"))
    clip_license = _normalise_license(entry.get("license"))
    if source_license and source_license != clip_license:
        raise ReferenceCorpusError(
            f"real reference clip {filename} source license "
            f"{source.get('license')!r} does not match clip license "
            f"{entry.get('license')!r}"
        )
    confirmation = str(source.get("rights_confirmation") or "").strip().lower()
    if "redistribut" not in confirmation and "public domain" not in confirmation:
        raise ReferenceCorpusError(
            f"real reference clip {filename} needs redistribution confirmation"
        )


def reference_manifest_entries(
    manifest_path: Path | str = DEFAULT_MANIFEST,
    clips_dir: Path | str | None = None,
) -> list[dict]:
    """Return verified core reference clip entries."""
    manifest = _load_manifest(manifest_path)
    root = Path(clips_dir) if clips_dir is not None else Path(manifest_path).parent
    entries: list[dict] = []
    for index, entry in enumerate(manifest.get("clips", [])):
        if not isinstance(entry, dict):
            raise ReferenceCorpusError(f"clip entry {index} must be an object")
        if entry.get("failure_category") != REFERENCE_CORPUS_CATEGORY:
            continue
        missing = REQUIRED_FIELDS - set(entry.keys())
        if missing:
            raise ReferenceCorpusError(
                f"reference clip {index} missing fields: {sorted(missing)}"
            )
        license_name = str(entry.get("license") or "").strip().lower()
        if license_name not in REFERENCE_LICENSE_ALLOWLIST:
            raise ReferenceCorpusError(
                f"reference clip {entry.get('filename')} has unsupported "
                f"license {entry.get('license')!r}"
            )
        filename = _safe_manifest_filename(entry.get("filename"))
        _validate_real_clip_source(entry, filename)
        path = root / filename
        if not path.is_file():
            raise ReferenceCorpusError(f"reference clip file is missing: {filename}")
        expected_sha = str(entry.get("sha256") or "").strip().lower()
        if not _valid_sha256(expected_sha):
            raise ReferenceCorpusError(
                f"reference clip {filename} has invalid sha256"
            )
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ReferenceCorpusError(
                f"reference clip {filename} sha256 mismatch"
            )
        if not isinstance(entry.get("config"), dict):
            raise ReferenceCorpusError(f"reference clip {filename} needs config")
        if not isinstance(entry.get("metric_floors"), dict):
            raise ReferenceCorpusError(
                f"reference clip {filename} needs metric_floors"
            )
        baseline = entry.get("baseline")
        if not isinstance(baseline, dict):
            raise ReferenceCorpusError(f"reference clip {filename} needs baseline")
        if not _valid_sha256(baseline.get("output_frames_sha256")):
            raise ReferenceCorpusError(
                f"reference clip {filename} has invalid output frame baseline"
            )
        enriched = dict(entry)
        enriched["path"] = str(path)
        entries.append(enriched)
    return entries


def decoded_frame_digest(path: Path | str) -> dict:
    """Hash decoded frame pixels, avoiding container metadata drift."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ReferenceCorpusError(f"could not open output for frame digest: {path}")
    h = hashlib.sha256()
    h.update(b"vsr.decoded-frame-sha256.v1\n")
    frame_count = 0
    width = height = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None:
                continue
            height, width = frame.shape[:2]
            h.update(f"{frame_count}:{height}x{width}:{frame.dtype}\n".encode("ascii"))
            h.update(np.ascontiguousarray(frame).tobytes())
            frame_count += 1
    finally:
        cap.release()
    if frame_count <= 0:
        raise ReferenceCorpusError(f"output decoded no frames: {path}")
    return {
        "sha256": h.hexdigest(),
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }


def _coerce_rect(value: object) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ReferenceCorpusError(f"invalid subtitle rect: {value!r}")
    return tuple(int(v) for v in value)  # type: ignore[return-value]


def _apply_config_overrides(overrides: Mapping[str, object]) -> processor.ProcessingConfig:
    defaults = {
        "mode": "sttn",
        "device": "cpu",
        "sttn_skip_detection": True,
        "tbe_enable": True,
        "quality_report": True,
        "preserve_audio": False,
        "output_quality": 0,
        "adaptive_batch": False,
        "use_hw_encode": False,
        "decode_hw_accel": "off",
        "prefetch_decode": False,
    }
    cfg = processor.ProcessingConfig()
    for key, value in {**defaults, **dict(overrides)}.items():
        if key == "mode":
            cfg.mode = processor.InpaintMode(str(value).lower())
        elif key == "subtitle_area" and value is not None:
            cfg.subtitle_area = _coerce_rect(value)
        elif key == "subtitle_areas" and value is not None:
            cfg.subtitle_areas = [_coerce_rect(rect) for rect in value]  # type: ignore[arg-type]
        elif key == "subtitle_region_spans" and value is not None:
            spans = []
            for span in value:  # type: ignore[union-attr]
                spans.append({
                    "rect": _coerce_rect(span["rect"]),
                    "start": float(span.get("start", 0.0)),
                    "end": float(span.get("end", 0.0)),
                })
            cfg.subtitle_region_spans = spans
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
    return processor.normalize_processing_config(cfg)


def _deterministic_remover(config: processor.ProcessingConfig):
    if config.mode != processor.InpaintMode.STTN:
        raise ReferenceCorpusError("reference corpus currently supports STTN only")
    remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
    remover.config = config
    remover.detector = processor.SubtitleDetector.__new__(processor.SubtitleDetector)
    remover.detector.device = "cpu"
    remover.detector.lang = "en"
    remover.detector.vertical = False
    remover.detector._engine_name = "reference-corpus"
    remover.detector._rapid_model = None
    remover.detector._paddle_model = None
    remover.detector._surya_det = None
    remover.detector._surya_processor = None
    remover.detector._easyocr_reader = None
    remover.inpainter = processor.STTNInpainter("cpu", config)
    remover.on_progress = None
    remover.on_preview_frame = None
    remover.live_preview_stride = 0
    remover._hw_encoder = None
    remover._srt_entries = []
    remover.last_quality_report = None
    remover.last_output_path = None
    remover.last_error_message = None
    remover.last_error_reason = None
    remover._quality_mask_bbox = None
    remover._color_metadata = None
    remover._active_writer = None
    remover._active_subprocess = None
    remover._teardown_requested = False
    return remover


def _metric_value(metrics: Mapping[str, object], name: str) -> Optional[float]:
    value = metrics.get(name)
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    return None


def run_reference_clip(entry: Mapping[str, object], output_dir: Path | str) -> dict:
    source = Path(str(entry["path"]))
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{source.stem}_cleaned.mp4"
    config = _apply_config_overrides(entry.get("config", {}))  # type: ignore[arg-type]
    remover = _deterministic_remover(config)
    ok = remover.process_video(str(source), str(output_path))
    actual_output = Path(remover.last_output_path or output_path)
    metrics = remover.last_quality_report or {}
    digest = decoded_frame_digest(actual_output) if ok else None

    failures: list[str] = []
    if not ok:
        failures.append(str(remover.last_error_message or "processing failed"))
    if digest is not None:
        expected_digest = entry["baseline"]["output_frames_sha256"]  # type: ignore[index]
        if digest["sha256"] != expected_digest:
            failures.append("decoded output frame hash mismatch")
        expected_count = entry["baseline"].get("frame_count")  # type: ignore[index]
        if expected_count is not None and digest["frame_count"] != int(expected_count):
            failures.append("decoded output frame count mismatch")

    floors = entry.get("metric_floors", {})
    if isinstance(floors, Mapping):
        for metric_name, floor in sorted(floors.items()):
            value = _metric_value(metrics, str(metric_name))
            if value is None:
                failures.append(f"metric {metric_name} missing")
                continue
            if value < float(floor):
                failures.append(
                    f"metric {metric_name} {value:.6f} below {float(floor):.6f}"
                )

    return {
        "filename": entry.get("filename"),
        "passed": not failures,
        "failures": failures,
        "output": str(actual_output),
        "outputFrames": digest,
        "metrics": {
            key: metrics.get(key)
            for key in (
                "psnr", "ssim", "roi_psnr", "roi_ssim",
                "temporal_flicker_score", "temporal_consistency",
                "residual_text_score", "samples", "tag",
            )
        },
    }


def run_reference_corpus(
    manifest_path: Path | str = DEFAULT_MANIFEST,
    *,
    clips_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> dict:
    entries = reference_manifest_entries(manifest_path, clips_dir)
    if not entries:
        raise ReferenceCorpusError("reference corpus has no core_reference clips")
    temp_dir = None
    if output_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="vsr-reference-corpus-")
        output_root = Path(temp_dir)
    else:
        output_root = Path(output_dir)
    try:
        results = [run_reference_clip(entry, output_root) for entry in entries]
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
    failures = [
        {
            "filename": result["filename"],
            "failures": result["failures"],
        }
        for result in results
        if not result["passed"]
    ]
    return {
        "schema": REFERENCE_CORPUS_SCHEMA,
        "manifest": str(Path(manifest_path)),
        "clipCount": len(results),
        "passed": not failures,
        "failures": failures,
        "clips": results,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the VSR reference clip regression corpus."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--clips-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = run_reference_corpus(
        args.manifest,
        clips_dir=args.clips_dir or None,
        output_dir=args.output_dir or None,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.json:
        print(payload)
    else:
        status = "passed" if result["passed"] else "failed"
        print(f"Reference corpus {status}: {result['clipCount']} clips")
        if result["failures"]:
            print(payload)
    return 0 if result["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
