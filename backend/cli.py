"""CLI entrypoint, checkpointing, and JSON config overlay loader.

Extracted from processor.py as part of RFP-L-1. Provides:

- ``main()``: the ``python -m backend.processor`` argparse + dispatch.
- ``_default_checkpoint_dir`` / ``_checkpoint_key`` /
  ``_checkpoint_is_done`` / ``_checkpoint_mark_done``: crash-resume
  marker bookkeeping under ``%APPDATA%/VSR/checkpoints/``.
- ``_load_json_config``: JSON config overlay loader.
- ``_apply_auto_band_override``: per-file region reset + auto-band probe.
"""

from __future__ import annotations

import hashlib
import datetime
import json
import logging
import os
import signal
import shutil
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)
_RUNTIME_HELPERS_LOADED = False
_path_key = None
_probe_subtitle_streams = None
_write_text_atomic = None
write_output_sidecar = None
SoftSubtitleAction = None
remux_soft_subtitles = None


def _load_runtime_helpers() -> None:
    """Import processing helpers only after diagnostics-only exits run."""
    global _RUNTIME_HELPERS_LOADED
    global STATUS_CANCELLED, STATUS_CHECKPOINT_DONE, STATUS_FAILED
    global STATUS_HARDCODED_PROCESSED, STATUS_PENDING
    global STATUS_PAUSED, STATUS_REVIEW_NEEDED, STATUS_SKIPPED_EXISTING
    global STATUS_SOFT_REMUXED
    global choose_batch_output_path, finish_batch_item
    global make_batch_item_record, write_batch_reports, write_output_sidecar
    global _path_key, _probe_subtitle_streams, _write_text_atomic
    global SoftSubtitleAction, remux_soft_subtitles

    if _RUNTIME_HELPERS_LOADED:
        return

    from backend.batch_report import (
        STATUS_CANCELLED,
        STATUS_CHECKPOINT_DONE,
        STATUS_FAILED,
        STATUS_HARDCODED_PROCESSED,
        STATUS_PENDING,
        STATUS_PAUSED,
        STATUS_REVIEW_NEEDED,
        STATUS_SKIPPED_EXISTING,
        STATUS_SOFT_REMUXED,
        choose_batch_output_path,
        finish_batch_item,
        make_batch_item_record,
        write_batch_reports,
        write_output_sidecar,
    )
    from backend.io import (
        _path_key as _io_path_key,
        _probe_subtitle_streams as _io_probe_subtitle_streams,
        _write_text_atomic as _io_write_text_atomic,
    )
    from backend.remux import (
        SoftSubtitleAction as _remux_soft_subtitle_action,
        remux_soft_subtitles as _remux_soft_subtitles,
    )
    if _path_key is None:
        _path_key = _io_path_key
    if _probe_subtitle_streams is None:
        _probe_subtitle_streams = _io_probe_subtitle_streams
    if _write_text_atomic is None:
        _write_text_atomic = _io_write_text_atomic
    if SoftSubtitleAction is None:
        SoftSubtitleAction = _remux_soft_subtitle_action
    if remux_soft_subtitles is None:
        remux_soft_subtitles = _remux_soft_subtitles
    _RUNTIME_HELPERS_LOADED = True


def _ensure_runtime_helpers() -> None:
    if not _RUNTIME_HELPERS_LOADED:
        _load_runtime_helpers()


def _app_version() -> str:
    try:
        from gui.config import APP_VERSION
        return APP_VERSION
    except Exception:
        return ""


def _default_checkpoint_dir() -> Path:
    """Where to store per-file crash-resume markers."""
    base = Path(os.environ.get("APPDATA", Path.home() / ".config")) / "VideoSubtitleRemoverPro" / "checkpoints"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _checkpoint_key(input_path: str, output_path: str) -> str:
    """Stable identifier for a (input, output, size, mtime) pair. A
    size/mtime change on the input invalidates the checkpoint so users
    do not skip a freshly re-downloaded file by accident."""
    try:
        stat = os.stat(input_path)
        fingerprint = f"{input_path}|{output_path}|{stat.st_size}|{int(stat.st_mtime)}"
    except OSError:
        fingerprint = f"{input_path}|{output_path}"
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:24]


def _checkpoint_is_done(ckpt_dir: Path, key: str, output_path: str) -> bool:
    marker = ckpt_dir / f"{key}.done"
    return marker.exists() and Path(output_path).exists()


def _checkpoint_mark_done(ckpt_dir: Path, key: str):
    _ensure_runtime_helpers()
    marker = ckpt_dir / f"{key}.done"
    try:
        _write_text_atomic(marker, "ok")
    except Exception as exc:
        logger.warning(f"Could not write checkpoint {marker}: {exc}")


def _load_json_config(path: str) -> dict:
    """Load a JSON config file of {field: value} pairs for ProcessingConfig."""
    size = os.path.getsize(path)
    if size > 1 * 1024 * 1024:
        raise ValueError(f"config file is too large ({size:,} bytes); expected a small JSON object")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("config file must contain a top-level JSON object")
    return payload


def _apply_auto_band_override(remover, input_path: str, *, auto_band: bool,
                              base_subtitle_area, base_subtitle_areas,
                              base_subtitle_region_spans=None):
    """Reset per-file region overrides before optionally probing a fresh band."""
    remover.config.subtitle_area = base_subtitle_area
    remover.config.subtitle_areas = list(base_subtitle_areas) if base_subtitle_areas else None
    remover.config.subtitle_region_spans = (
        list(base_subtitle_region_spans)
        if base_subtitle_region_spans else None
    )
    if (not auto_band or base_subtitle_area or base_subtitle_areas
            or base_subtitle_region_spans):
        return base_subtitle_area
    band = remover.detect_subtitle_band(input_path)
    remover.config.subtitle_area = band
    return band


def _soft_subtitle_action(args):
    _ensure_runtime_helpers()
    if args.strip_soft_subtitles:
        return SoftSubtitleAction.STRIP
    if args.keep_soft_subtitles:
        return SoftSubtitleAction.KEEP_ALL
    return None


def _soft_subtitle_stream_record(stream) -> dict:
    return {
        "index": stream.index,
        "codec_name": stream.codec_name or "",
        "language": stream.language or "",
        "title": stream.title or "",
        "default": bool(stream.default),
        "forced": bool(stream.forced),
    }


def _build_soft_subtitle_plan_record(input_path: str, action_label: str) -> dict:
    _ensure_runtime_helpers()
    streams = _probe_subtitle_streams(input_path)
    return {
        "input": str(input_path),
        "input_name": Path(input_path).name,
        "action": action_label,
        "has_soft_subtitles": bool(streams),
        "subtitle_stream_count": len(streams),
        "subtitle_streams": [
            _soft_subtitle_stream_record(stream)
            for stream in streams
        ],
    }


def _write_soft_subtitle_plan_json(path: str, action_label: str,
                                   records: list[dict]) -> None:
    payload = {
        "schema": "vsr.soft_subtitle_preflight.v1",
        "action": action_label,
        "count": len(records),
        "files": records,
    }
    _write_text_atomic(
        Path(path),
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def _print_soft_subtitle_plan_record(record: dict) -> None:
    streams = record["subtitle_streams"]
    source = record["input_name"]
    action_label = record["action"]
    if not streams:
        print(f"[soft-subtitles] {source}: no embedded subtitle streams | action={action_label}")
        return
    print(f"[soft-subtitles] {source}: {len(streams)} stream(s) | action={action_label}")
    for stream in streams:
        language = stream["language"] or "-"
        title = stream["title"] or "-"
        default = "yes" if stream["default"] else "no"
        forced = "yes" if stream["forced"] else "no"
        print(
            "  "
            f"stream={stream['index']} | codec={stream['codec_name'] or '-'} | "
            f"lang={language} | title={title} | default={default} | forced={forced}"
        )


def _print_soft_subtitle_plan(input_path: str, action_label: str) -> dict:
    record = _build_soft_subtitle_plan_record(input_path, action_label)
    _print_soft_subtitle_plan_record(record)
    return record


def _run_soft_subtitle_only(input_path: str, output_path: str,
                            action: SoftSubtitleAction) -> bool:
    _ensure_runtime_helpers()
    _print_soft_subtitle_plan(input_path, action.value)
    remux_soft_subtitles(input_path, output_path, action=action)
    print(f"[soft-subtitles] wrote {output_path}")
    return True


def _cancel_pending_records(records: list[dict]) -> None:
    _ensure_runtime_helpers()
    for record in records:
        if record.get("status") == STATUS_PENDING:
            finish_batch_item(record, STATUS_CANCELLED, message="Interrupted")


def _write_cli_batch_reports(out_dir: Path, records: list[dict], *,
                             kind: str,
                             started_at: datetime.datetime) -> None:
    _ensure_runtime_helpers()
    if not records:
        return
    json_path, md_path = write_batch_reports(
        out_dir,
        records,
        kind=kind,
        started_at=started_at,
        completed_at=datetime.datetime.now(datetime.timezone.utc),
    )
    print(f"[batch] wrote report {json_path}")
    print(f"[batch] wrote summary {md_path}")


def _print_output_quality_preflight(preflight: dict) -> None:
    from backend.output_quality_preflight import output_quality_preflight_messages

    for message in output_quality_preflight_messages(preflight):
        print(f"[quality-preflight] {message}")


def _update_record_output_path(record: dict, actual_output_path: str) -> None:
    """Keep batch evidence aligned when processing salvages to another path."""
    actual = Path(actual_output_path)
    record["output"] = str(actual)
    record["output_name"] = actual.name
    record["output_exists"] = actual.exists()
    try:
        record["output_parent_free_bytes"] = shutil.disk_usage(actual.parent).free
    except OSError:
        record["output_parent_free_bytes"] = None


def main():
    """CLI entry point."""
    import argparse
    early_parser = argparse.ArgumentParser(add_help=False)
    early_parser.add_argument("--support-bundle", metavar="PATH")
    early_args, _remaining = early_parser.parse_known_args()
    if early_args.support_bundle:
        from backend.support_bundle import create_support_bundle
        bundle = create_support_bundle(
            early_args.support_bundle,
            app_version=os.environ.get("VSR_APP_VERSION", ""),
            extra_facts={"surface": "cli"},
        )
        print(f"[support] wrote {bundle}")
        sys.exit(0)

    # Import here so the heavy backend (SubtitleRemover + cv2 + numpy)
    # loads only when the CLI actually runs.
    _load_runtime_helpers()
    from backend.processor import (
        ProcessingConfig, SubtitleRemover,
        attach_json_log, normalize_processing_config, _coerce_backend_mode,
        is_known_backend_mode,
    )
    from backend.resume_checkpoint import ProcessingPaused
    from backend import inpainter_registry

    # Built-in modes first, then whatever opt-in backends registered at
    # import time (ONNX / diffusion scaffolds gated by env vars).
    mode_choices = ["sttn", "lama", "propainter", "auto", "migan"]
    for _name, _builder in inpainter_registry.list_modes():
        if _name not in mode_choices:
            mode_choices.append(_name)

    parser = argparse.ArgumentParser(
        description="Video Subtitle Remover Pro CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m backend.processor -i input.mp4 -o output.mp4 -m sttn --lang en\n"
            "  python -m backend.processor --pattern \"inputs/*.mp4\" --out-dir cleaned --mode auto"
        ),
    )
    parser.add_argument("--input", "-i", help="Input file path")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--pattern", help="Glob pattern for batch mode (e.g. 'inputs/*.mp4')")
    parser.add_argument("--out-dir", help="Output directory for batch mode")
    parser.add_argument("--config", help="JSON config file (key=value pairs overriding CLI defaults)")
    parser.add_argument("--preset", metavar="NAME",
                       help="Apply a built-in or user preset by name.")
    parser.add_argument("--list-presets", action="store_true",
                       help="Print every known preset and exit.")
    parser.add_argument("--checkpoint-dir", default=None,
                       help=("Checkpoint dir for crash-resume and pause/resume "
                             "(default: %%APPDATA%%/.../checkpoints)"))
    parser.add_argument("--no-resume", action="store_true",
                       help=("Ignore existing checkpoints and reprocess every file; "
                             "pause checkpoints are still written for this run"))
    parser.add_argument("--mode", "-m", default="sttn",
                       choices=mode_choices,
                       help="Inpainting algorithm.")
    parser.add_argument("--gpu", "-g", type=int, default=0, help="GPU device ID (-1 for CPU)")
    parser.add_argument("--lang", "-l", default="en", help="Detection language")
    parser.add_argument("--skip-detection", action="store_true",
                       help="Skip automatic detection (STTN only)")
    parser.add_argument("--fast", action="store_true", help="Fast mode (LAMA only)")
    parser.add_argument("--no-audio", action="store_true", help="Don't preserve audio")
    parser.add_argument("--crf", type=int, default=23, help="Output CRF quality (15-35)")
    parser.add_argument("--start", type=float, default=0, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=0, help="End time in seconds (0=full)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold (0.1-1.0)")
    parser.add_argument("--vertical", action="store_true",
                       help="Vertical-text mode (rotate frames 90 CCW before OCR).")
    parser.add_argument("--whisper-fallback", action="store_true",
                       help="Whisper-driven bottom-band default mask on OCR-empty frames.")
    parser.add_argument("--whisper-backend", default="faster-whisper",
                       choices=["faster-whisper", "ffmpeg"],
                       help="Whisper fallback backend.")
    parser.add_argument("--upscale", type=int, default=0, choices=[0, 2, 3, 4],
                       help="Post-cleanup upscale (Real-ESRGAN).")
    parser.add_argument("--no-color-preserve", action="store_true",
                       help="Do not re-tag the output with the source's color signalling.")
    parser.add_argument("--nle-sidecar", default="off",
                       choices=["off", "edl", "fcpxml"],
                       help="Emit an EDL or FCPXML sidecar next to the output.")
    parser.add_argument("--swinir", action="store_true",
                       help="Post-cleanup SwinIR restoration pass.")
    parser.add_argument("--seedvr2", action="store_true",
                       help="Post-cleanup SeedVR2 restoration pass.")
    parser.add_argument("--film-grain", type=float, default=0.0, metavar="STRENGTH",
                       help="Additive film grain after cleanup (0..0.5; 0 disables).")
    parser.add_argument("--watermark", default="", metavar="PATH",
                       help="Burn a PNG watermark onto the output after cleanup.")
    parser.add_argument("--watermark-position", default="bottom-right",
                       choices=["top-left", "top-right", "bottom-left",
                                "bottom-right", "center"],
                       help="Watermark corner position (default bottom-right).")
    parser.add_argument("--watermark-opacity", type=float, default=1.0,
                       help="Watermark opacity 0.0-1.0 (default 1.0).")
    parser.add_argument("--watermark-margin", type=int, default=16,
                       help="Watermark margin from edge in pixels (default 16).")
    parser.add_argument("--nle-input", default="", metavar="PATH",
                       help="Parse an EDL/FCPXML to extract time segments for processing.")
    parser.add_argument("--restyle", default="", metavar="PATH",
                       help="Re-burn an .srt or .ass subtitle file onto the cleaned output.")
    parser.add_argument("--restyle-style", default="", metavar="ASS_STYLE",
                       help="ASS force_style override for --restyle (e.g. 'FontSize=24,PrimaryColour=&H00FFFFFF').")
    parser.add_argument("--whisper-model", default="tiny",
                       choices=["tiny", "base", "small", "medium",
                                "large", "large-v2", "large-v3"],
                       help="faster-whisper model size.")
    parser.add_argument("--ffmpeg-whisper-model", default="",
                       help="Path to a local whisper.cpp ggml model for --whisper-backend ffmpeg.")
    parser.add_argument("--ffmpeg-whisper-queue", type=float, default=3.0,
                       metavar="SECONDS",
                       help="FFmpeg whisper filter queue size in seconds.")
    parser.add_argument("--ffmpeg-whisper-vad-model", default="",
                       help="Path to a Silero VAD ONNX model for FFmpeg Whisper.")
    parser.add_argument("--ffmpeg-whisper-vad-threshold", type=float, default=0.5,
                       metavar="FLOAT",
                       help="VAD confidence threshold (0.0-1.0, default 0.5).")
    parser.add_argument("--ffmpeg-whisper-min-speech", type=float, default=0.0,
                       metavar="SECONDS",
                       help="Minimum speech duration for VAD segments (default 0).")
    parser.add_argument("--frame-skip", type=int, default=0,
                       help="Reuse detection mask for N frames between detections")
    parser.add_argument("--rife-fast-stride", type=int, default=0,
                       help=("Inpaint every Nth frame and synthesize skipped "
                             "frames with Practical-RIFE (0 disables)."))
    parser.add_argument("--mask-dilate", type=int, default=8,
                       help="Mask dilation in pixels (0=off)")
    parser.add_argument("--confidence-dilate", action="store_true",
                       help="Scale mask dilation inversely with OCR confidence")
    parser.add_argument("--no-hw-encode", action="store_true",
                       help="Disable hardware encoding (force libx264)")
    parser.add_argument("--codec", default="h264",
                       choices=["h264", "h265", "av1", "vvc"],
                       help="Output video codec (vvc requires FFmpeg with libvvenc).")
    parser.add_argument("--mask-feather", type=int, default=4,
                       help="Gaussian edge feathering in pixels (0=off)")
    parser.add_argument("--temporal-smooth", type=int, default=0,
                       metavar="RADIUS",
                       help="Post-inpaint temporal smoothing radius for LaMa (0=off, 1-5)")
    parser.add_argument("--edge-ring", type=int, default=2,
                       help="Edge-ring colour match width in pixels (0=off)")
    parser.add_argument("--flow-warp", action="store_true",
                       help="Farneback flow-warp TBE frames before aggregation")
    parser.add_argument("--no-scene-split", action="store_true",
                       help="Disable scene-cut splitting inside TBE batches")
    parser.add_argument("--pyscenedetect", action="store_true",
                       help="Prefer PySceneDetect AdaptiveDetector for scene cuts.")
    parser.add_argument("--transnetv2", action="store_true",
                       help="Prefer TransNetV2 (deep CNN) for scene-cut detection.")
    parser.add_argument("--denoise-detect", action="store_true",
                       help="Run a denoise pass on the detection-frame stream.")
    parser.add_argument("--sam2-refine", action="store_true",
                       help="SAM 2 mask refinement of detected boxes.")
    parser.add_argument("--matanyone-refine", action="store_true",
                       help="MatAnyone 2 alpha-matte refinement of masks.")
    parser.add_argument("--cotracker-propagate", action="store_true",
                       help="Use CoTracker3 to fill OCR-empty masks in a batch.")
    parser.add_argument("--no-tbe", action="store_true",
                       help="Disable Temporal Background Exposure (STTN/ProPainter use cv2)")
    parser.add_argument("--no-adaptive-batch", action="store_true",
                       help="Disable VRAM-probe-driven batch sizing")
    parser.add_argument("--export-srt", action="store_true",
                       help="Write an .srt sidecar with detected text")
    parser.add_argument("--export-mask", action="store_true",
                       help="Write a B/W .mask.mp4 debug video")
    parser.add_argument("--auto-band", action="store_true",
                       help="Auto-detect the dominant subtitle band before processing")
    parser.add_argument("--no-kalman", action="store_true",
                       help="Disable Kalman detection smoothing")
    parser.add_argument("--no-phash", action="store_true",
                       help="Disable perceptual-hash adaptive mask reuse")
    parser.add_argument("--phash-distance", type=int, default=4,
                       help="pHash Hamming distance threshold for mask reuse (0-64)")
    parser.add_argument("--colour-tune", action="store_true",
                       help="Grow the mask by dominant-colour match inside each box")
    parser.add_argument("--colour-tolerance", type=int, default=25,
                       help="Lab-space colour distance tolerance for colour-tune")
    parser.add_argument("--auto-threshold", type=float, default=0.55,
                       help="AUTO-mode exposure threshold (0-1)")
    parser.add_argument("--deinterlace", action="store_true",
                       help="Force ffmpeg yadif deinterlace before processing")
    parser.add_argument("--no-deinterlace-detect", action="store_true",
                       help="Skip the automatic ffprobe interlacing detection")
    parser.add_argument("--keyframe-detect", action="store_true",
                       help="OCR only at video I-frames (ffprobe-probed)")
    parser.add_argument("--quality-report", action="store_true",
                       help="Compute PSNR/SSIM on a random frame sample after run")
    parser.add_argument("--quality-sheet", action="store_true",
                       help="Render a side-by-side comparison PNG alongside the report.")
    parser.add_argument("--input-fps", type=float, default=24.0, metavar="FPS",
                       help="FPS for directory-of-images input.")
    parser.add_argument("--output-frames", action="store_true",
                       help="Write cleaned frames as individual PNGs instead of a video.")
    parser.add_argument("--keep-chyrons", action="store_true",
                       help="Leave persistent text (logos, lower-thirds, tickers).")
    parser.add_argument("--keep-subtitles", action="store_true",
                       help="Leave non-persistent text (dialogue captions).")
    parser.add_argument("--chyron-min-hits", type=int, default=90, metavar="N",
                       help="Kalman-track frame count to classify as chyron.")
    parser.add_argument("--karaoke-grouping", action="store_true",
                       help="Fuse per-syllable OCR boxes on the same line.")
    parser.add_argument("--karaoke-x-gap", type=int, default=20, metavar="PX",
                       help="Max horizontal gap (px) between karaoke boxes.")
    parser.add_argument("--karaoke-y-overlap", type=float, default=0.5,
                       metavar="RATIO",
                       help="Min vertical overlap ratio for karaoke line fusion.")
    parser.add_argument("--loudnorm", type=float, default=0.0, metavar="LUFS",
                       help="EBU R128 loudness target in LUFS.")
    parser.add_argument("--decode-accel", default="off",
                       choices=[
                           "off", "auto", "any", "d3d11", "vaapi", "mfx",
                           "pynv", "nvdec",
                       ],
                       help="Hardware-decode hint (OpenCV or PyNvVideoCodec).")
    parser.add_argument("--single-audio", action="store_true",
                       help="Mux only the first audio stream.")
    parser.add_argument("--no-prefetch", action="store_true",
                       help="Disable the worker-thread frame prefetcher.")
    parser.add_argument("--prefetch-queue", type=int, default=0, metavar="N",
                       help="Bounded prefetch queue size in frames.")
    parser.add_argument("--skip-existing", action="store_true",
                       help="Skip inputs whose output path already exists.")
    parser.add_argument("--soft-subtitle-dry-run", action="store_true",
                       help="Print embedded subtitle tracks and planned action, then exit.")
    parser.add_argument("--soft-subtitle-plan-json", metavar="PATH",
                       help="Write soft-subtitle dry-run preflight details as JSON.")
    parser.add_argument("--strip-soft-subtitles", action="store_true",
                       help="Fast remux that removes embedded subtitle tracks without OCR.")
    parser.add_argument("--keep-soft-subtitles", action="store_true",
                       help="Fast remux that keeps embedded subtitle tracks without OCR.")
    parser.add_argument("--burned-in-only", action="store_true",
                       help="Ignore embedded subtitle tracks and run burned-in cleanup normally.")
    parser.add_argument("--audit-onnx", action="store_true",
                       help="Audit all discoverable ONNX models for DirectML opset compatibility and exit.")
    parser.add_argument("--audit-windows-ml", action="store_true",
                       help="Probe the Windows ML Python path with a tiny ONNX smoke model and exit.")
    parser.add_argument("--scan-weights", action="store_true",
                       help="Scan cached model weights and verify SHA-256 against known hashes, then exit.")
    parser.add_argument("--cache-info", action="store_true",
                       help="Print cache directory inventory with sizes and exit.")
    parser.add_argument("--cache-clean", action="store_true",
                       help="Remove stale cache entries (checkpoints, proxies, TRT engines) and exit.")
    parser.add_argument("--model-cache-export", metavar="PATH",
                       help="Write a portable model-cache zip with SHA-256 manifest and exit.")
    parser.add_argument("--model-cache-import", metavar="PATH",
                       help="Import a verified portable model-cache zip into the app model cache and exit.")
    parser.add_argument("--support-bundle", metavar="PATH",
                       help="Write a redacted diagnostics zip and exit.")
    parser.add_argument("--validate-config", action="store_true",
                       help="Print the resolved ProcessingConfig as JSON and exit.")
    parser.add_argument("--self-test", action="store_true",
                       help="Probe OCR engines, inpaint backends, GPU providers, "
                            "and codecs, then print results and exit.")
    parser.add_argument("--auto-lang-probe", action="store_true",
                       help="Probe the first frame for script/language and print "
                            "a suggestion, then exit. Requires -i.")
    parser.add_argument("--intent", metavar="PHRASE",
                       help="Natural-language cleanup intent (e.g. 'remove subtitles',"
                            " 'remove logo'). Prints config changes and exits.")
    parser.add_argument("--json-log", metavar="PATH",
                       help="Append a structured JSON-line log at PATH.")

    args = parser.parse_args()

    if args.json_log:
        attach_json_log(args.json_log)

    if args.list_presets:
        from backend.presets import BUILTIN_PRESETS as _BUILTIN, load_user_presets as _load_user
        rows = []
        for name, payload in _BUILTIN.items():
            rows.append(("built-in", name, payload.get("description", "")))
        for name, payload in _load_user().items():
            if name in _BUILTIN:
                continue
            desc = payload.get("description", "") if isinstance(payload, dict) else ""
            rows.append(("user", name, desc))
        width = max((len(n) for _, n, _ in rows), default=4)
        for source, name, desc in rows:
            print(f"[{source:<8}] {name.ljust(width)}  {desc}")
        sys.exit(0)

    if args.audit_onnx:
        from backend.onnx_model_info import print_audit_report
        print_audit_report()
        sys.exit(0)

    if args.audit_windows_ml:
        from backend.onnx_model_info import print_windows_ml_probe_report
        print_windows_ml_probe_report()
        sys.exit(0)

    if args.scan_weights:
        from backend.model_hashes import print_weight_report
        print_weight_report()
        sys.exit(0)

    if args.cache_info:
        from backend.cache_inventory import print_cache_info
        print_cache_info()
        sys.exit(0)

    if args.cache_clean:
        from backend.cache_inventory import clean_cache
        print("Cleaning stale VSR caches:")
        clean_cache(dry_run=False)
        sys.exit(0)

    if args.model_cache_export and args.model_cache_import:
        parser.error("--model-cache-export and --model-cache-import are mutually exclusive")

    if args.model_cache_export:
        from backend.cache_inventory import export_model_cache_bundle
        try:
            result = export_model_cache_bundle(args.model_cache_export)
        except Exception as exc:
            print(f"[model-cache] export failed: {exc}", file=sys.stderr)
            sys.exit(1)
        print(
            f"[model-cache] exported {len(result['files'])} file(s) "
            f"to {result['output']}"
        )
        missing = result["status_after_export"].get("missing_known_filenames", [])
        if missing:
            print(
                "[model-cache] missing optional known assets: "
                + ", ".join(missing)
            )
        if result.get("skipped"):
            print(f"[model-cache] skipped {len(result['skipped'])} unsafe or invalid file(s)")
        sys.exit(0)

    if args.model_cache_import:
        from backend.cache_inventory import import_model_cache_bundle
        try:
            result = import_model_cache_bundle(args.model_cache_import)
        except Exception as exc:
            print(f"[model-cache] import failed: {exc}", file=sys.stderr)
            sys.exit(1)
        print(
            f"[model-cache] imported {len(result['imported'])} file(s) "
            f"from {result['source']}"
        )
        if result.get("rejected"):
            print(f"[model-cache] rejected {len(result['rejected'])} unsafe or invalid file(s)")
        missing = result["status_after_import"].get("missing_known_filenames", [])
        if missing:
            print(
                "[model-cache] missing optional known assets: "
                + ", ".join(missing)
            )
        sys.exit(1 if result.get("rejected") and not result.get("imported") else 0)

    if args.self_test:
        from backend.support_bundle import run_self_test
        results = run_self_test()
        for category, entries in results.items():
            print(f"\n{category.upper()}")
            for entry in entries:
                mark = "OK" if entry["available"] else "  "
                print(f"  [{mark}] {entry['name']}: {entry['reason']}")
        sys.exit(0)

    if args.intent:
        from backend.presets import parse_intent
        changes = parse_intent(args.intent)
        if changes is None:
            print(f"No config changes matched for: {args.intent!r}",
                  file=sys.stderr)
            sys.exit(1)
        print("Intent config changes:")
        for key, value in sorted(changes.items()):
            print(f"  {key}: {value}")
        sys.exit(0)

    if args.auto_lang_probe:
        if not args.input:
            print("--auto-lang-probe requires -i <input file>", file=sys.stderr)
            sys.exit(1)
        import cv2 as _cv2
        cap = _cv2.VideoCapture(args.input)
        try:
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok or frame is None:
            from backend.safe_image import safe_imread
            frame = safe_imread(args.input)
        if frame is None:
            print("Could not read input file", file=sys.stderr)
            sys.exit(1)
        from backend.detection import probe_language
        lang, conf, script = probe_language(frame)
        print(f"Detected script: {script}")
        print(f"Suggested language: {lang}")
        print(f"Confidence: {conf:.2f}")
        sys.exit(0)

    soft_mode_count = sum(
        1 for enabled in (
            args.strip_soft_subtitles,
            args.keep_soft_subtitles,
            args.burned_in_only,
        ) if enabled
    )
    if soft_mode_count > 1:
        parser.error(
            "--strip-soft-subtitles, --keep-soft-subtitles, and "
            "--burned-in-only are mutually exclusive"
        )
    if args.soft_subtitle_plan_json and not args.soft_subtitle_dry_run:
        parser.error("--soft-subtitle-plan-json requires --soft-subtitle-dry-run")
    soft_action = _soft_subtitle_action(args)
    dry_run_only = args.soft_subtitle_dry_run

    if args.preset:
        from backend.presets import preset_fields as _preset_fields
        fields = _preset_fields(args.preset)
        if fields is None:
            parser.error(
                f"unknown preset {args.preset!r}; run --list-presets to see options"
            )
        attr_to_default = {a.dest: a.default for a in parser._actions}
        field_to_attr = {
            "mode": "mode",
            "detection_threshold": "threshold",
            "mask_dilate_px": "mask_dilate",
            "mask_feather_px": "mask_feather",
            "edge_ring_px": "edge_ring",
            "tbe_flow_warp": "flow_warp",
            "tbe_scene_cut_split": None,
            "colour_tune_enable": "colour_tune",
            "colour_tune_tolerance": "colour_tolerance",
            "kalman_tracking": None,
            "phash_skip_enable": None,
            "phash_skip_distance": "phash_distance",
            "auto_band": None,
            "detection_frame_skip": "frame_skip",
        }
        for fname, value in fields.items():
            if fname == "mode":
                if getattr(args, "mode", None) == attr_to_default.get("mode"):
                    args.mode = str(value).lower().replace(" ", "")
                continue
            attr = field_to_attr.get(fname, fname)
            if attr is None or not hasattr(args, attr):
                continue
            if getattr(args, attr) == attr_to_default.get(attr):
                setattr(args, attr, value)
        logger.info(f"Applied preset: {args.preset}")

    if not args.validate_config:
        if not args.input and not args.pattern:
            parser.error("one of --input or --pattern is required")
        if args.input and args.pattern:
            parser.error("--input and --pattern are mutually exclusive")
        if args.pattern and not args.out_dir and not dry_run_only:
            parser.error("--pattern requires --out-dir")
        if args.input and not args.output and not dry_run_only:
            parser.error("--input requires --output")
    if not 0.1 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0.1 and 1.0")
    if not 15 <= args.crf <= 35:
        parser.error("--crf must be between 15 and 35")
    if args.start < 0 or args.end < 0:
        parser.error("--start and --end must be zero or positive")
    if args.end and args.end < args.start:
        parser.error("--end must be greater than or equal to --start")
    if args.frame_skip < 0:
        parser.error("--frame-skip must be zero or positive")
    if args.rife_fast_stride < 0:
        parser.error("--rife-fast-stride must be zero or positive")
    if args.mask_dilate < 0:
        parser.error("--mask-dilate must be zero or positive")
    if args.mask_feather < 0:
        parser.error("--mask-feather must be zero or positive")
    if args.edge_ring < 0:
        parser.error("--edge-ring must be zero or positive")
    if not 0.0 <= args.auto_threshold <= 1.0:
        parser.error("--auto-threshold must be between 0 and 1")
    if not 0 <= args.phash_distance <= 64:
        parser.error("--phash-distance must be between 0 and 64")
    if args.colour_tolerance < 0:
        parser.error("--colour-tolerance must be zero or positive")
    if args.loudnorm != 0.0 and not -70.0 <= args.loudnorm <= -5.0:
        parser.error("--loudnorm must be 0 (off) or between -70 and -5 LUFS")
    if args.ffmpeg_whisper_queue < 0.02:
        parser.error("--ffmpeg-whisper-queue must be at least 0.02 seconds")

    config = ProcessingConfig(
        mode=_coerce_backend_mode(args.mode),
        device=f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu",
        sttn_skip_detection=args.skip_detection,
        lama_super_fast=args.fast,
        preserve_audio=not args.no_audio,
        detection_lang=args.lang,
        detection_threshold=args.threshold,
        detection_vertical=args.vertical,
        whisper_fallback=args.whisper_fallback,
        whisper_backend=args.whisper_backend,
        whisper_model_size=args.whisper_model,
        whisper_model_path=args.ffmpeg_whisper_model,
        whisper_queue_seconds=args.ffmpeg_whisper_queue,
        whisper_vad_model=args.ffmpeg_whisper_vad_model,
        whisper_vad_threshold=args.ffmpeg_whisper_vad_threshold,
        whisper_min_speech_duration=args.ffmpeg_whisper_min_speech,
        upscale_factor=args.upscale,
        film_grain_strength=args.film_grain,
        watermark_image=args.watermark,
        watermark_position=args.watermark_position,
        watermark_opacity=args.watermark_opacity,
        watermark_margin=args.watermark_margin,
        restyle_subtitle=args.restyle,
        restyle_style=args.restyle_style,
        swinir_restore=args.swinir,
        seedvr2_restore=args.seedvr2,
        preserve_color_metadata=not args.no_color_preserve,
        nle_sidecar=args.nle_sidecar,
        output_quality=args.crf,
        time_start=args.start,
        time_end=args.end,
        detection_frame_skip=args.frame_skip,
        rife_fast_stride=args.rife_fast_stride,
        mask_dilate_px=args.mask_dilate,
        confidence_weighted_dilation=args.confidence_dilate,
        mask_feather_px=args.mask_feather,
        temporal_smooth_radius=args.temporal_smooth,
        edge_ring_px=args.edge_ring,
        tbe_enable=not args.no_tbe,
        tbe_flow_warp=args.flow_warp,
        tbe_scene_cut_split=not args.no_scene_split,
        tbe_scene_cut_use_pyscenedetect=args.pyscenedetect,
        tbe_scene_cut_use_transnetv2=args.transnetv2,
        detection_denoise=args.denoise_detect,
        sam2_refine=args.sam2_refine,
        matanyone_refine=args.matanyone_refine,
        cotracker_propagate=args.cotracker_propagate,
        adaptive_batch=not args.no_adaptive_batch,
        export_srt=args.export_srt,
        export_mask_video=args.export_mask,
        kalman_tracking=not args.no_kalman,
        phash_skip_enable=not args.no_phash,
        phash_skip_distance=args.phash_distance,
        colour_tune_enable=args.colour_tune,
        colour_tune_tolerance=args.colour_tolerance,
        auto_exposure_threshold=args.auto_threshold,
        deinterlace=args.deinterlace,
        deinterlace_auto=not args.no_deinterlace_detect,
        keyframe_detection=args.keyframe_detect,
        quality_report=args.quality_report,
        use_hw_encode=not args.no_hw_encode,
        output_codec=args.codec,
        loudnorm_target=args.loudnorm,
        decode_hw_accel=args.decode_accel,
        multi_audio_passthrough=not args.single_audio,
        prefetch_decode=not args.no_prefetch,
        prefetch_queue_size=args.prefetch_queue,
        input_fps=args.input_fps,
        output_frames=args.output_frames,
        quality_report_sheet=args.quality_sheet,
        remove_subtitles=not args.keep_subtitles,
        remove_chyrons=not args.keep_chyrons,
        chyron_min_hits=args.chyron_min_hits,
        karaoke_grouping=args.karaoke_grouping,
        karaoke_x_gap_px=args.karaoke_x_gap,
        karaoke_y_overlap=args.karaoke_y_overlap,
    )
    config = normalize_processing_config(config)

    ffmpeg_ready = shutil.which("ffmpeg") is not None

    if args.config:
        try:
            overlay = _load_json_config(args.config)
            from dataclasses import fields as _dc_fields
            _allowed = {f.name for f in _dc_fields(config)}
            for k, v in overlay.items():
                if k == "mode":
                    if is_known_backend_mode(v):
                        config.mode = _coerce_backend_mode(v)
                    else:
                        logger.warning(f"Ignoring unknown mode in config: {v}")
                    continue
                if k in _allowed:
                    setattr(config, k, v)
                else:
                    logger.warning(f"Ignoring unknown config field: {k}")
            config = normalize_processing_config(config)
            logger.info(f"Loaded config overlay from {args.config}")
        except Exception as exc:
            parser.error(f"Could not load --config {args.config}: {exc}")

    if args.validate_config:
        resolved = {
            "mode": config.mode.value,
            "device": config.device,
            "detection_lang": config.detection_lang,
            "detection_threshold": config.detection_threshold,
            "detection_frame_skip": config.detection_frame_skip,
            "rife_fast_stride": config.rife_fast_stride,
            "subtitle_area": list(config.subtitle_area) if config.subtitle_area else None,
            "subtitle_areas": (
                [list(r) for r in config.subtitle_areas]
                if config.subtitle_areas else None
            ),
            "subtitle_region_spans": (
                [
                    {
                        "rect": list(span["rect"]),
                        "start": float(span.get("start", 0.0)),
                        "end": float(span.get("end", 0.0)),
                    }
                    for span in (config.subtitle_region_spans or [])
                ] or None
            ),
            "mask_dilate_px": config.mask_dilate_px,
            "mask_feather_px": config.mask_feather_px,
            "edge_ring_px": config.edge_ring_px,
            "tbe_enable": config.tbe_enable,
            "tbe_min_coverage": config.tbe_min_coverage,
            "tbe_flow_warp": config.tbe_flow_warp,
            "tbe_scene_cut_split": config.tbe_scene_cut_split,
            "tbe_scene_cut_threshold": config.tbe_scene_cut_threshold,
            "kalman_tracking": config.kalman_tracking,
            "kalman_iou_threshold": config.kalman_iou_threshold,
            "kalman_max_age": config.kalman_max_age,
            "phash_skip_enable": config.phash_skip_enable,
            "phash_skip_distance": config.phash_skip_distance,
            "colour_tune_enable": config.colour_tune_enable,
            "colour_tune_tolerance": config.colour_tune_tolerance,
            "auto_exposure_threshold": config.auto_exposure_threshold,
            "deinterlace": config.deinterlace,
            "deinterlace_auto": config.deinterlace_auto,
            "keyframe_detection": config.keyframe_detection,
            "quality_report": config.quality_report,
            "adaptive_batch": config.adaptive_batch,
            "sttn_skip_detection": config.sttn_skip_detection,
            "sttn_neighbor_stride": config.sttn_neighbor_stride,
            "sttn_reference_length": config.sttn_reference_length,
            "sttn_max_load_num": config.sttn_max_load_num,
            "lama_super_fast": config.lama_super_fast,
            "time_start": config.time_start,
            "time_end": config.time_end,
            "preserve_audio": config.preserve_audio,
            "output_format": config.output_format,
            "output_quality": config.output_quality,
            "use_hw_encode": config.use_hw_encode,
            "loudnorm_target": config.loudnorm_target,
            "decode_hw_accel": config.decode_hw_accel,
            "multi_audio_passthrough": config.multi_audio_passthrough,
            "prefetch_decode": config.prefetch_decode,
            "prefetch_queue_size": config.prefetch_queue_size,
            "input_fps": config.input_fps,
            "quality_report_sheet": config.quality_report_sheet,
            "whisper_fallback": config.whisper_fallback,
            "whisper_backend": config.whisper_backend,
            "whisper_model_size": config.whisper_model_size,
            "whisper_model_path": config.whisper_model_path,
            "whisper_queue_seconds": config.whisper_queue_seconds,
            "remove_subtitles": config.remove_subtitles,
            "remove_chyrons": config.remove_chyrons,
            "chyron_min_hits": config.chyron_min_hits,
            "karaoke_grouping": config.karaoke_grouping,
            "karaoke_x_gap_px": config.karaoke_x_gap_px,
            "karaoke_y_overlap": config.karaoke_y_overlap,
            "export_srt": config.export_srt,
            "export_mask_video": config.export_mask_video,
        }
        print(json.dumps({"resolved_config": resolved}, indent=2, sort_keys=True))
        sys.exit(0)

    video_exts = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpeg', '.mpg'}

    if args.soft_subtitle_dry_run:
        planned = (
            soft_action.value if soft_action is not None
            else ("burned-in-cleanup" if args.burned_in_only else "inspect")
        )
        records = []
        if args.pattern:
            from glob import glob
            inputs = sorted(glob(args.pattern, recursive=True))
            inputs = [p for p in inputs if Path(p).is_file()]
            if not inputs:
                parser.error(f"No files matched pattern: {args.pattern}")
            for inp in inputs:
                records.append(_print_soft_subtitle_plan(inp, planned))
        else:
            records.append(_print_soft_subtitle_plan(args.input, planned))
        if args.soft_subtitle_plan_json:
            _write_soft_subtitle_plan_json(
                args.soft_subtitle_plan_json,
                planned,
                records,
            )
            print(f"[soft-subtitles] wrote plan {args.soft_subtitle_plan_json}")
        sys.exit(0)

    if soft_action is not None:
        if args.pattern:
            from glob import glob
            inputs = sorted(glob(args.pattern, recursive=True))
            inputs = [p for p in inputs if Path(p).is_file()]
            if not inputs:
                parser.error(f"No files matched pattern: {args.pattern}")
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            batch_started_at = datetime.datetime.now(datetime.timezone.utc)
            records: list[dict] = []
            interrupted = False
            reserved_outputs: set = set()
            try:
                for i, inp in enumerate(inputs, 1):
                    src = Path(inp)
                    outp = choose_batch_output_path(
                        inp,
                        out_dir,
                        "_soft_subtitles",
                        reserved_outputs,
                        skip_existing=args.skip_existing,
                    )
                    reserved_outputs.add(_path_key(outp))
                    record = make_batch_item_record(
                        inp,
                        str(outp),
                        config={
                            "mode": "soft-subtitles",
                            "device": "cpu",
                            "output_codec": "copy",
                            "output_quality": config.output_quality,
                        },
                        skip_existing=args.skip_existing,
                        soft_action=soft_action.value,
                    )
                    records.append(record)
                    print(f"\n[soft-subtitles] ({i}/{len(inputs)}) {src.name}")
                    if record["planned_result"] == STATUS_SKIPPED_EXISTING:
                        print(f"[skip] {src.name} (output exists)")
                        finish_batch_item(
                            record,
                            STATUS_SKIPPED_EXISTING,
                            message="Output already exists",
                        )
                        continue
                    started = time.monotonic()
                    try:
                        _run_soft_subtitle_only(inp, str(outp), soft_action)
                        elapsed = time.monotonic() - started
                        finish_batch_item(
                            record,
                            STATUS_SOFT_REMUXED,
                            message=f"Soft subtitles {soft_action.value}",
                            elapsed_seconds=elapsed,
                            stage_timings={"mux": elapsed},
                        )
                        write_output_sidecar(
                            input_path=inp, output_path=str(outp),
                            config=config, status="soft-subtitle-remuxed",
                            elapsed_seconds=elapsed,
                            stage_timings={"mux": elapsed},
                            app_version=_app_version(),
                        )
                    except Exception as exc:
                        logger.error(f"Soft-subtitle remux failed on {src.name}: {exc}")
                        finish_batch_item(
                            record,
                            STATUS_FAILED,
                            message=str(exc),
                            elapsed_seconds=time.monotonic() - started,
                            stage_timings={"mux": time.monotonic() - started},
                        )
            except KeyboardInterrupt:
                print("\n[soft-subtitles] Interrupted by user -- partial results kept on disk.")
                _cancel_pending_records(records)
                interrupted = True
            finally:
                _write_cli_batch_reports(
                    out_dir,
                    records,
                    kind="soft-subtitles",
                    started_at=batch_started_at,
                )
            if interrupted:
                sys.exit(130)
            failures = sum(1 for record in records if record.get("status") == STATUS_FAILED)
            sys.exit(0 if failures == 0 else 1)
        try:
            if args.skip_existing and Path(args.output).exists():
                print(f"[skip] {Path(args.input).name} (output exists)")
                sys.exit(0)
            _run_soft_subtitle_only(args.input, args.output, soft_action)
            sys.exit(0)
        except KeyboardInterrupt:
            print("\n[soft-subtitles] Interrupted by user.")
            sys.exit(130)
        except Exception as exc:
            logger.error(f"Soft-subtitle remux failed: {exc}")
            sys.exit(1)

    remover = SubtitleRemover(config)
    remover.on_progress = lambda p, m: print(f"[{int(p*100):3d}%] {m}")

    print(
        "[run] "
        f"mode={config.mode.value} | device={config.device} | lang={config.detection_lang} | "
        f"audio={'on' if config.preserve_audio else 'off'} | hw_encode={'on' if config.use_hw_encode else 'off'}"
    )
    if config.preserve_audio and not ffmpeg_ready:
        print("[note] FFmpeg is not available, so outputs will be saved without original audio.")

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else _default_checkpoint_dir()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pause_requested = {"value": False}

    def _request_pause(_signum=None, _frame=None):
        if not pause_requested["value"]:
            pause_requested["value"] = True
            print("\n[pause] Requested. Waiting for the next safe frame checkpoint...")

    previous_sigint = signal.getsignal(signal.SIGINT)
    try:
        signal.signal(signal.SIGINT, _request_pause)
    except (ValueError, OSError):
        previous_sigint = None

    def _pause_requested() -> bool:
        return bool(pause_requested["value"])

    base_subtitle_area = config.subtitle_area
    base_subtitle_areas = list(config.subtitle_areas) if config.subtitle_areas else None
    base_subtitle_region_spans = (
        list(config.subtitle_region_spans)
        if config.subtitle_region_spans else None
    )

    def _process_one(inp: str, outp: str) -> bool:
        if args.skip_existing and Path(outp).exists():
            print(f"[skip] {Path(inp).name} (output exists)")
            return True
        key = _checkpoint_key(inp, outp)
        if not args.no_resume and _checkpoint_is_done(ckpt_dir, key, outp):
            print(f"[skip] {Path(inp).name} (checkpoint)")
            return True
        _apply_auto_band_override(
            remover,
            inp,
            auto_band=False,
            base_subtitle_area=base_subtitle_area,
            base_subtitle_areas=base_subtitle_areas,
            base_subtitle_region_spans=base_subtitle_region_spans,
        )
        ext = Path(inp).suffix.lower()
        if Path(inp).is_dir() or ext in video_exts:
            if args.auto_band:
                band = _apply_auto_band_override(
                    remover,
                    inp,
                    auto_band=True,
                    base_subtitle_area=base_subtitle_area,
                    base_subtitle_areas=base_subtitle_areas,
                    base_subtitle_region_spans=base_subtitle_region_spans,
                )
                if band:
                    print(f"[auto-band] {Path(inp).name}: {band}")
                elif not (
                        base_subtitle_area or base_subtitle_areas
                        or base_subtitle_region_spans):
                    print(f"[auto-band] {Path(inp).name}: no dominant band, full-frame")
            ok = remover.process_video(
                inp,
                outp,
                checkpoint_dir=ckpt_dir,
                checkpoint_key=key,
                resume_checkpoint=not args.no_resume,
                pause_check=_pause_requested,
            )
        else:
            ok = remover.process_image(inp, outp)
        if ok:
            _checkpoint_mark_done(ckpt_dir, key)
        return ok

    if args.pattern:
        from glob import glob
        inputs = sorted(glob(args.pattern, recursive=True))
        inputs = [p for p in inputs if Path(p).is_file()]
        if not inputs:
            parser.error(f"No files matched pattern: {args.pattern}")
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[batch] {len(inputs)} file(s) queued | out={out_dir} | resume={'on' if not args.no_resume else 'off'}")
        batch_started_at = datetime.datetime.now(datetime.timezone.utc)
        records: list[dict] = []
        interrupted = False
        paused = False
        reserved_outputs: set = set()
        try:
            for i, inp in enumerate(inputs, 1):
                src = Path(inp)
                outp = choose_batch_output_path(
                    inp,
                    out_dir,
                    "_no_sub",
                    reserved_outputs,
                    skip_existing=args.skip_existing,
                )
                reserved_outputs.add(_path_key(outp))
                key = _checkpoint_key(inp, str(outp))
                checkpoint_done = (
                    not args.no_resume
                    and _checkpoint_is_done(ckpt_dir, key, str(outp))
                )
                record = make_batch_item_record(
                    inp,
                    str(outp),
                    config=config,
                    skip_existing=args.skip_existing,
                    checkpoint_done=checkpoint_done,
                )
                records.append(record)
                _print_output_quality_preflight(
                    record.get("output_quality_preflight") or {}
                )
                print(f"\n[batch] ({i}/{len(inputs)}) {src.name}")
                if record["planned_result"] == STATUS_SKIPPED_EXISTING:
                    print(f"[skip] {src.name} (output exists)")
                    finish_batch_item(
                        record,
                        STATUS_SKIPPED_EXISTING,
                        message="Output already exists",
                    )
                    write_output_sidecar(
                        input_path=inp, output_path=str(outp),
                        config=config, status="skipped-existing",
                        app_version=_app_version(),
                    )
                    continue
                if record["planned_result"] == STATUS_CHECKPOINT_DONE:
                    print(f"[skip] {src.name} (checkpoint)")
                    finish_batch_item(
                        record,
                        STATUS_CHECKPOINT_DONE,
                        message="Checkpoint already complete",
                    )
                    write_output_sidecar(
                        input_path=inp, output_path=str(outp),
                        config=config, status="checkpoint-done",
                        checkpoint_resumed=True,
                        app_version=_app_version(),
                    )
                    continue
                started = time.monotonic()
                try:
                    ok = _process_one(inp, str(outp))
                except ProcessingPaused as exc:
                    print(f"\n[pause] {exc}")
                    finish_batch_item(
                        record,
                        STATUS_PAUSED,
                        message=str(exc),
                        elapsed_seconds=time.monotonic() - started,
                        stage_timings=getattr(remover, "last_stage_timings", None),
                    )
                    _cancel_pending_records(records)
                    paused = True
                    interrupted = True
                    break
                except Exception as exc:
                    logger.error(f"Failed on {src.name}: {exc}")
                    ok = False
                    finish_batch_item(
                        record,
                        STATUS_FAILED,
                        message=str(exc),
                        elapsed_seconds=time.monotonic() - started,
                        stage_timings=getattr(remover, "last_stage_timings", None),
                    )
                else:
                    quality_report = (
                        getattr(remover, "last_quality_report", None)
                        if ok else None
                    )
                    failure_message = (
                        getattr(remover, "last_error_message", None)
                        or "Processing failed"
                    )
                    actual_output = getattr(remover, "last_output_path", None)
                    if ok and actual_output and _path_key(actual_output) != _path_key(record["output"]):
                        _update_record_output_path(record, actual_output)
                    finish_batch_item(
                        record,
                        STATUS_HARDCODED_PROCESSED if ok else STATUS_FAILED,
                        message="Processed" if ok else failure_message,
                        elapsed_seconds=time.monotonic() - started,
                        quality_report=quality_report,
                        stage_timings=getattr(remover, "last_stage_timings", None),
                    )
        except KeyboardInterrupt:
            print("\n[batch] Interrupted by user -- partial results kept on disk.")
            _cancel_pending_records(records)
            interrupted = True
        finally:
            _write_cli_batch_reports(
                out_dir,
                records,
                kind="hardcoded-cleanup",
                started_at=batch_started_at,
            )
        if paused:
            print("[batch] Paused. Re-run the same command to resume the current item.")
            sys.exit(130)
        if interrupted:
            sys.exit(130)
        failures = sum(1 for record in records if record.get("status") == STATUS_FAILED)
        reviews = sum(
            1 for record in records
            if record.get("status") == STATUS_REVIEW_NEEDED
        )
        succeeded = len(inputs) - failures
        suffix = f", {reviews} review-needed" if reviews else ""
        print(f"\n{'='*60}")
        print(f"  BATCH COMPLETE: {succeeded}/{len(inputs)} succeeded{suffix}")
        print(f"{'='*60}")
        if failures:
            print("[batch] Some items need attention. Review the errors above before retrying.")
        if reviews:
            print("[batch] Some outputs need manual review. See vsr-batch-summary for quality-gate details.")
        sys.exit(0 if failures == 0 else 1)

    if args.nle_input:
        from backend.nle_sidecar import parse_nle_input
        raw_fps = cap_fps = 24.0
        try:
            from backend.io import _probe_duration_seconds
            import cv2 as _cv2
            _c = _cv2.VideoCapture(args.input)
            if _c.isOpened():
                cap_fps = _c.get(_cv2.CAP_PROP_FPS) or 24.0
                _c.release()
        except Exception:
            pass
        segments = parse_nle_input(args.nle_input, cap_fps)
        if not segments:
            parser.error(f"No time segments found in: {args.nle_input}")
        print(f"[nle] {len(segments)} segment(s) from {Path(args.nle_input).name}")
        nle_preflight = make_batch_item_record(
            args.input,
            args.output,
            config=config,
        ).get("output_quality_preflight") or {}
        _print_output_quality_preflight(nle_preflight)
        out_base = Path(args.output)
        failures = 0
        for idx, (seg_start, seg_end) in enumerate(segments, 1):
            config.time_start = seg_start
            config.time_end = seg_end
            if len(segments) == 1:
                seg_out = str(out_base)
            else:
                seg_out = str(
                    out_base.parent
                    / f"{out_base.stem}_seg{idx}{out_base.suffix}"
                )
            print(f"[nle] segment {idx}/{len(segments)}: "
                  f"{seg_start:.2f}s - {seg_end:.2f}s -> {Path(seg_out).name}")
            try:
                ok = _process_one(args.input, seg_out)
            except ProcessingPaused as exc:
                print(f"\n[nle] Paused: {exc}")
                print("[nle] Re-run the same command to resume the current segment.")
                sys.exit(130)
            except KeyboardInterrupt:
                print("\n[nle] Interrupted by user.")
                sys.exit(130)
            if not ok:
                failures += 1
        print(f"[nle] {len(segments) - failures}/{len(segments)} segments completed")
        sys.exit(0 if failures == 0 else 1)

    print(f"[file] source={Path(args.input).name}")
    print(f"[file] output={args.output}")
    single_preflight = make_batch_item_record(
        args.input,
        args.output,
        config=config,
    ).get("output_quality_preflight") or {}
    _print_output_quality_preflight(single_preflight)
    try:
        success = _process_one(args.input, args.output)
    except ProcessingPaused as exc:
        print(f"\n[file] Paused: {exc}")
        print("[file] Re-run the same command to resume this file.")
        sys.exit(130)
    except KeyboardInterrupt:
        print("\n[file] Interrupted by user.")
        sys.exit(130)
    actual_output = getattr(remover, "last_output_path", None)
    if success and actual_output and _path_key(actual_output) != _path_key(args.output):
        print(f"[file] actual-output={actual_output}")
    if not success:
        message = getattr(remover, "last_error_message", None)
        if message:
            print(f"[file] error={message}")
    print(f"[file] {'completed' if success else 'failed'}")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
