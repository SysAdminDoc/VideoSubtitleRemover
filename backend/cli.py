"""CLI parser, configuration assembly, and batch dispatch.

Extracted from processor.py as part of RFP-L-1. Provides:

- ``main()``: the ``python -m backend.processor`` argparse + dispatch.
Checkpoint and configuration helpers live in their focused modules so importing
the processing orchestrator never imports this CLI module.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import signal
import shutil
import sys
import time
from pathlib import Path

from backend.config import _apply_auto_band_override, _load_json_config
from backend.resume_checkpoint import (
    _checkpoint_is_done,
    _checkpoint_key,
    _checkpoint_mark_done,
    _default_checkpoint_dir,
)

logger = logging.getLogger(__name__)
_RUNTIME_HELPERS_LOADED = False
_path_key = None
_probe_subtitle_streams = None
_write_text_atomic = None
write_output_sidecar = None
SoftSubtitleAction = None
remux_soft_subtitles = None


_CLI_CATEGORY_OPTIONS = (
    (
        "General",
        ("--help",),
    ),
    (
        "Inputs, batches, and reproducibility",
        (
            "--input", "--output", "--pattern", "--out-dir", "--config",
            "--config-schema-version", "--set", "--preset", "--list-presets",
            "--checkpoint-dir", "--work-dir", "--no-resume", "--start", "--end",
            "--input-fps", "--output-frames", "--nle-input", "--skip-existing",
        ),
    ),
    (
        "Removal, detection, and masks",
        (
            "--mode", "--gpu", "--lang", "--language-filter",
            "--skip-detection", "--fast",
            "--threshold", "--vertical", "--frame-skip", "--mask-dilate",
            "--confidence-dilate", "--mask-feather", "--temporal-smooth",
            "--edge-ring", "--flow-warp", "--no-scene-split", "--pyscenedetect",
            "--transnetv2", "--denoise-detect", "--sam2-refine",
            "--matanyone-refine", "--cotracker-propagate", "--no-tbe",
            "--no-adaptive-batch", "--temporal-mask-union",
            "--temporal-mask-window", "--auto-band", "--no-kalman", "--no-phash",
            "--phash-distance", "--colour-tune", "--colour-tolerance",
            "--auto-threshold", "--keep-chyrons", "--keep-subtitles",
            "--chyron-min-hits", "--karaoke-grouping", "--karaoke-x-gap",
            "--karaoke-y-overlap",
        ),
    ),
    (
        "Speech and subtitle tracks",
        (
            "--whisper-fallback", "--whisper-backend", "--whisper-model",
            "--ffmpeg-whisper-model", "--ffmpeg-whisper-queue",
            "--ffmpeg-whisper-vad-model", "--ffmpeg-whisper-vad-threshold",
            "--ffmpeg-whisper-min-speech", "--export-srt", "--ocr-fix",
            "--soft-subtitle-dry-run",
            "--soft-subtitle-plan-json", "--strip-soft-subtitles",
            "--keep-soft-subtitles", "--burned-in-only", "--restyle",
            "--restyle-style", "--translate", "--translated-srt",
            "--translation-source-srt", "--translation-provider",
            "--translation-source-lang", "--translation-target-lang",
            "--translation-command", "--translation-style",
            "--translation-timeout",
        ),
    ),
    (
        "Output and post-processing",
        (
            "--no-audio", "--crf", "--upscale", "--no-color-preserve",
            "--nle-sidecar", "--swinir", "--seedvr2", "--film-grain", "--watermark",
            "--watermark-position", "--watermark-opacity", "--watermark-margin",
            "--no-hw-encode", "--d3d12-accel", "--codec", "--export-mask",
            "--mask-export-format", "--import-mask", "--mask-import-mode",
            "--deinterlace",
            "--no-deinterlace-detect", "--keyframe-detect", "--quality-report",
            "--quality-sheet", "--loudnorm", "--decode-accel", "--single-audio",
        ),
    ),
    (
        "Performance and recovery",
        (
            "--rife-fast-stride", "--max-retries", "--retry-backoff",
            "--no-prefetch", "--prefetch-queue",
        ),
    ),
    (
        "Diagnostics and automation",
        (
            "--audit-onnx", "--audit-windows-ml", "--scan-weights", "--cache-info",
            "--cache-clean", "--model-cache-export", "--model-cache-import",
            "--support-bundle", "--validate-config", "--self-test",
            "--inference-smoke", "--ocr-benchmark", "--ocr-engine", "--dry-run",
            "--json", "--auto-lang-probe", "--intent", "--json-log",
            "--dump-cli-reference",
        ),
    ),
)

_CLI_VALUE_RANGES = {
    "--gpu": "-1 or >=0",
    "--crf": "15..35",
    "--start": ">=0 seconds",
    "--end": "0 or >= start",
    "--threshold": "0.1..1.0",
    "--film-grain": "0..0.5",
    "--watermark-opacity": "0..1",
    "--watermark-margin": "0..500 pixels",
    "--ffmpeg-whisper-queue": "0.02..3600 seconds",
    "--ffmpeg-whisper-vad-threshold": "0..1",
    "--ffmpeg-whisper-min-speech": "0..30 seconds",
    "--frame-skip": "0..240 frames",
    "--rife-fast-stride": "0..60 frames",
    "--mask-dilate": "0..100 pixels",
    "--mask-feather": "0..100 pixels",
    "--temporal-smooth": "0..5 frames",
    "--edge-ring": "0..32 pixels",
    "--temporal-mask-window": "1..15 frames",
    "--max-retries": "0..10",
    "--retry-backoff": "0..600 seconds",
    "--phash-distance": "0..64",
    "--colour-tolerance": "0..255",
    "--auto-threshold": "0..1",
    "--input-fps": "1..240",
    "--chyron-min-hits": "1..100000 frames",
    "--karaoke-x-gap": "0..1024 pixels",
    "--karaoke-y-overlap": "0..1",
    "--loudnorm": "0 (off) or -70..-5 LUFS",
    "--prefetch-queue": "0..512 frames",
    "--translation-timeout": "5..3600 seconds",
}

# There are currently no deprecated public options. Keeping the set explicit
# makes the generated reference fail closed when a compatibility flag is added.
_CLI_DEPRECATED_OPTIONS = frozenset()
_CLI_INTERNAL_OPTIONS = frozenset({"--dump-cli-reference"})


def _primary_option(action: argparse.Action) -> str:
    return next(
        (flag for flag in action.option_strings if flag.startswith("--")),
        action.option_strings[0] if action.option_strings else action.dest,
    )


def _apply_cli_option_metadata(parser: argparse.ArgumentParser) -> None:
    """Attach complete option metadata and group ``--help`` from that source."""
    category_by_option: dict[str, str] = {}
    for category, options in _CLI_CATEGORY_OPTIONS:
        for option in options:
            if option in category_by_option:
                raise RuntimeError(f"duplicate CLI metadata for {option}")
            category_by_option[option] = category

    actions_by_category = {category: [] for category, _ in _CLI_CATEGORY_OPTIONS}
    seen: set[str] = set()
    for action in parser._actions:
        if not action.option_strings:
            continue
        option = _primary_option(action)
        category = category_by_option.get(option)
        if category is None:
            raise RuntimeError(f"CLI option has no metadata: {option}")
        metadata = {
            "category": category,
            "value_range": _CLI_VALUE_RANGES.get(option, ""),
            "deprecated": option in _CLI_DEPRECATED_OPTIONS,
            "internal": option in _CLI_INTERNAL_OPTIONS,
        }
        action.vsr_metadata = metadata
        actions_by_category[category].append(action)
        seen.add(option)

    stale = sorted(set(category_by_option) - seen)
    if stale:
        raise RuntimeError("CLI metadata refers to missing options: " + ", ".join(stale))

    parser._optionals.title = _CLI_CATEGORY_OPTIONS[0][0]
    parser._optionals._group_actions = actions_by_category[_CLI_CATEGORY_OPTIONS[0][0]]
    for category, _options in _CLI_CATEGORY_OPTIONS[1:]:
        group = parser.add_argument_group(category)
        group._group_actions = actions_by_category[category]


def _cli_reference_payload(parser: argparse.ArgumentParser) -> dict:
    """Return deterministic JSON-safe reference data from live parser actions."""
    options = []
    for action in parser._actions:
        if not action.option_strings:
            continue
        metadata = getattr(action, "vsr_metadata", None)
        if metadata is None:
            raise RuntimeError(f"CLI option metadata was not attached: {_primary_option(action)}")
        choices = list(action.choices) if action.choices is not None else []
        value_range = metadata["value_range"]
        if not value_range and choices:
            value_range = " | ".join(str(choice) for choice in choices)
        help_text = "" if action.help is argparse.SUPPRESS else str(action.help or "")
        options.append(
            {
                "flags": list(action.option_strings),
                "dest": action.dest,
                "category": metadata["category"],
                "description": help_text.replace("%%", "%").strip(),
                "default": action.default,
                "range": value_range,
                "metavar": action.metavar,
                "deprecated": metadata["deprecated"],
                "internal": metadata["internal"],
            }
        )
    return {
        "schema": "vsr.cli_reference.v1",
        "categories": [category for category, _options in _CLI_CATEGORY_OPTIONS],
        "options": options,
    }


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


def _dry_run_plan_for(remover, config, inp: str, video_exts) -> dict:
    """Build a no-encode plan for one input: probe, detect, codec check."""
    import cv2 as _cv2
    from backend.ffmpeg_profiles import missing_profile_requirements_for_config

    plan = {
        "input": inp,
        "is_video": False,
        "frames": None,
        "fps": None,
        "sampled": 0,
        "frames_with_text": 0,
        "detected_regions": [],
        "codec_ok": True,
        "warnings": [],
    }
    ext = Path(inp).suffix.lower()
    is_video = Path(inp).is_dir() or ext in video_exts
    plan["is_video"] = bool(is_video)

    try:
        missing = missing_profile_requirements_for_config(config)
        if missing:
            plan["codec_ok"] = False
            plan["warnings"].append(
                "codec/profile requirements unmet: "
                + "; ".join(m.get("reason", "") for m in missing)
            )
    except Exception as exc:  # noqa: BLE001
        plan["warnings"].append(f"codec probe failed: {exc}")

    if not is_video:
        try:
            from backend.safe_image import safe_imread
            img = safe_imread(inp)
            if img is not None:
                boxes = remover.detector.detect(img, config.detection_threshold)
                plan["sampled"] = 1
                plan["frames_with_text"] = 1 if boxes else 0
                plan["detected_regions"] = [list(b) for b in (boxes or [])][:8]
            else:
                plan["warnings"].append("could not read image")
        except Exception as exc:  # noqa: BLE001
            plan["warnings"].append(f"detection failed: {exc}")
        return plan

    cap = _cv2.VideoCapture(inp)
    try:
        if not cap.isOpened():
            plan["warnings"].append("could not open video")
            return plan
        total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(_cv2.CAP_PROP_FPS) or 0.0)
        plan["frames"] = total or None
        plan["fps"] = round(fps, 3) if fps else None
        sample_count = 5 if total else 0
        indices = ([int(total * i / (sample_count + 1)) for i in range(1, sample_count + 1)]
                   if total else [])
        for idx in indices:
            cap.set(_cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            plan["sampled"] += 1
            try:
                boxes = remover.detector.detect(frame, config.detection_threshold)
            except Exception as exc:  # noqa: BLE001
                plan["warnings"].append(f"detection failed at frame {idx}: {exc}")
                break
            if boxes:
                plan["frames_with_text"] += 1
                if not plan["detected_regions"]:
                    plan["detected_regions"] = [list(b) for b in boxes][:8]
    finally:
        cap.release()
    return plan


def _run_dry_run_and_exit(remover, config, args, video_exts) -> None:
    """Resolve inputs, build no-encode plans, print, and exit."""
    if args.pattern:
        from glob import glob
        inputs = [p for p in sorted(glob(args.pattern, recursive=True))
                  if Path(p).is_file()]
    else:
        inputs = [args.input] if args.input else []
    plans = [_dry_run_plan_for(remover, config, inp, video_exts)
             for inp in inputs]

    if getattr(args, "json_output", False):
        print(json.dumps({
            "dry_run": True,
            "mode": config.mode.value,
            "device": config.device,
            "plans": plans,
        }, indent=2))
    else:
        print(f"[dry-run] {len(plans)} input(s); no files will be written")
        for plan in plans:
            name = Path(plan["input"]).name
            kind = "video" if plan["is_video"] else "image"
            frames = plan.get("frames")
            hit = plan["frames_with_text"]
            sampled = plan["sampled"]
            codec = "ok" if plan["codec_ok"] else "MISSING"
            print(f"  - {name} [{kind}] frames={frames} "
                  f"text-in {hit}/{sampled} sampled, codec={codec}")
            for warn in plan["warnings"]:
                print(f"      ! {warn}")
    any_codec_missing = any(not p["codec_ok"] for p in plans)
    sys.exit(1 if (not plans or any_codec_missing) else 0)


def _build_parser(mode_choices):
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
    parser.add_argument(
        "--config-schema-version",
        type=int,
        default=None,
        help="Canonical processing-config schema version for reproducible commands.",
    )
    parser.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        default=[],
        metavar="FIELD=JSON",
        help="Override any canonical processing field; repeat for multiple values.",
    )
    parser.add_argument("--preset", metavar="NAME",
                       help="Apply a built-in or user preset by name.")
    parser.add_argument("--list-presets", action="store_true",
                       help="Print every known preset and exit.")
    parser.add_argument("--checkpoint-dir", default=None,
                       help=("Checkpoint dir for crash-resume and pause/resume "
                             "(default: %%APPDATA%%/.../checkpoints)"))
    parser.add_argument(
        "--work-dir",
        default="",
        help=("Writable root for temporary, mask, checkpoint, and resume "
              "artifacts; falls back with a warning when unavailable."),
    )
    parser.add_argument("--no-resume", action="store_true",
                       help=("Ignore existing checkpoints and reprocess every file; "
                             "pause checkpoints are still written for this run"))
    parser.add_argument("--mode", "-m", default="sttn",
                       choices=mode_choices,
                       help="Inpainting algorithm.")
    parser.add_argument("--gpu", "-g", type=int, default=0, help="GPU device ID (-1 for CPU)")
    parser.add_argument("--lang", "-l", default="en", help="Detection language")
    parser.add_argument(
        "--language-filter",
        action="store_true",
        help="Only mask OCR text matching the selected language's script.",
    )
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
    parser.add_argument(
        "--translate", action="store_true",
        help="Erase subtitles, translate a source SRT locally, and re-embed it.")
    parser.add_argument(
        "--translated-srt", default="", metavar="PATH",
        help="Validated UTF-8 SRT that is already translated; bypasses a provider.")
    parser.add_argument(
        "--translation-source-srt", default="", metavar="PATH",
        help="Source-language SRT to translate; otherwise OCR/Whisper cues are used.")
    parser.add_argument(
        "--translation-provider", default="command", metavar="NAME",
        help="Registered local translation provider name (default: command).")
    parser.add_argument(
        "--translation-source-lang", default="auto", metavar="LANG",
        help="Source language tag passed to the local translation provider.")
    parser.add_argument(
        "--translation-target-lang", default="", metavar="LANG",
        help="Required target language tag when generating translated subtitles.")
    parser.add_argument(
        "--translation-command", default="", metavar="PATH",
        help="Local executable or Python script using the VSR translation JSON protocol.")
    parser.add_argument(
        "--translation-style", default="", metavar="ASS_STYLE",
        help="ASS force_style override for the translated subtitle burn pass.")
    parser.add_argument(
        "--translation-timeout", type=float, default=300.0, metavar="SECONDS",
        help="Timeout for the local translation provider command.")
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
    parser.add_argument(
        "--d3d12-accel",
        action="store_true",
        help=(
            "Opt into FFmpeg 8.1+ D3D12 filters and encoding after a "
            "byte-valid runtime smoke; falls back automatically."
        ),
    )
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
    parser.add_argument("--temporal-mask-union", action="store_true",
                       help="Scene-cut-safe temporal mask stabilization: OR each "
                            "frame's mask with a short trailing window (auto "
                            "detection only) to retain pixels missed on single "
                            "frames or moving overlays; resets at scene cuts")
    parser.add_argument("--temporal-mask-window", type=int, default=3,
                       help="Trailing window size for --temporal-mask-union (1-15)")
    parser.add_argument("--max-retries", type=int, default=0,
                       help="Automatically re-attempt a batch item that fails with "
                            "a transient error (GPU glitch, ffmpeg hiccup, timeout) "
                            "up to N times with backoff (0=off, max 10)")
    parser.add_argument("--retry-backoff", type=float, default=5.0,
                       help="Base seconds between transient retries (0-600; "
                            "each later attempt waits a multiple of this value)")
    parser.add_argument("--export-srt", action="store_true",
                       help="Write an .srt sidecar with detected text")
    parser.add_argument("--ocr-fix", action="store_true",
                       help=("Apply a per-language OCR-fix replace list to the "
                             "exported SRT text (built-in defaults plus "
                             "%%APPDATA%%/VideoSubtitleRemoverPro/ocr_fix/"
                             "{lang}.json)."))
    parser.add_argument("--export-mask", action="store_true",
                       help="Export a lossless grayscale matte plus timing manifest")
    parser.add_argument(
        "--mask-export-format", choices=["ffv1", "png"], default="ffv1",
        help="Lossless matte export as FFV1 video or a PNG sequence.")
    parser.add_argument(
        "--import-mask", default="", metavar="MANIFEST",
        help="Import an edited .mask.json timing manifest before inpainting.")
    parser.add_argument(
        "--mask-import-mode", choices=["replace", "add", "subtract"],
        default="replace",
        help="Compose the imported matte after native mask generation.")
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
    parser.add_argument("--inference-smoke", action="store_true",
                       help="Run a generated text image and masked frame through "
                            "the OCR and inpaint backends to prove they actually "
                            "execute (records provider/timing), then exit. No model "
                            "downloads. Uses --gpu to pick the device.")
    parser.add_argument("--ocr-benchmark", action="store_true",
                       help="Benchmark the active OCR detector on synthetic "
                            "ground-truth subtitle fixtures (recall, latency, "
                            "and memory) "
                            "and print JSON evidence, then exit. Use --gpu to "
                            "pick the device. Gate any default-detector swap on "
                            "the meets_floors verdict.")
    parser.add_argument(
        "--ocr-engine",
        choices=(
            "auto", "rapidocr", "opencv-dnn", "paddleocr", "easyocr", "opencv"
        ),
        default="auto",
        help=("Select the OCR detector for processing or --ocr-benchmark; "
              "auto uses the best available engine."),
    )
    parser.add_argument("--dry-run", action="store_true",
                       help="Validate the run without encoding: probe each input, "
                            "run detection on a few sampled frames, check the "
                            "requested codec is available, and print a per-file "
                            "plan, then exit. Combine with --json for machine "
                            "output.")
    parser.add_argument("--json", action="store_true", dest="json_output",
                       help="Emit a machine-readable JSON result to stdout "
                            "(the --dry-run plan, or the batch/file result).")
    parser.add_argument("--auto-lang-probe", action="store_true",
                       help="Probe the first frame for script/language and print "
                            "a suggestion, then exit. Requires -i.")
    parser.add_argument("--intent", metavar="PHRASE",
                       help="Natural-language cleanup intent (e.g. 'remove subtitles',"
                            " 'remove logo'). Prints config changes and exits.")
    parser.add_argument("--json-log", metavar="PATH",
                       help="Append a structured JSON-line log at PATH.")
    parser.add_argument(
        "--dump-cli-reference",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    _apply_cli_option_metadata(parser)
    return parser


def _handle_utility_actions(args, parser, attach_json_log) -> bool:
    if args.dump_cli_reference:
        print(json.dumps(_cli_reference_payload(parser), ensure_ascii=True, sort_keys=True))
        return True

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

    if getattr(args, "ocr_benchmark", False):
        from backend.ocr_benchmark import run_default_detector_benchmark
        device = f"cuda:{args.gpu}" if getattr(args, "gpu", 0) >= 0 else "cpu"
        result = run_default_detector_benchmark(
            device=device,
            engine=args.ocr_engine,
        )
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["meets_floors"] else 1)

    if args.inference_smoke:
        from backend.support_bundle import run_inference_smoke
        device = f"cuda:{args.gpu}" if getattr(args, "gpu", 0) >= 0 else "cpu"
        results = run_inference_smoke(device=device)
        failed = False
        for category in ("ocr", "inpaint"):
            print(f"\n{category.upper()} (device={results['device']})")
            for entry in results.get(category, []):
                if entry.get("ran") and not entry.get("passed"):
                    failed = True
                mark = "OK" if entry.get("passed") else ("--" if entry.get("ran") else "  ")
                ms = entry.get("ms")
                timing = f" {ms:.1f}ms" if isinstance(ms, (int, float)) else ""
                detail = entry.get("provider") or entry.get("reason") or ""
                print(f"  [{mark}] {entry['name']}: {detail}{timing}")
        sys.exit(1 if failed else 0)

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
    return False


def _explicitly_provided_dests(parser, argv):
    """Return the set of argument dests the user actually typed on the CLI.

    argparse cannot distinguish an omitted flag from one passed with a value
    that happens to equal the parser default, so preset merging must not rely
    on ``value == default``. Inspecting the raw tokens recovers intent:
    ``--threshold 0.5`` and ``--threshold=0.5`` both mark ``threshold`` as
    explicit, so a preset can no longer silently discard it.
    """
    tokens = list(argv or [])
    provided = set()
    for action in parser._actions:
        for opt in action.option_strings:
            if any(tok == opt or tok.startswith(opt + "=") for tok in tokens):
                provided.add(action.dest)
                break
    return provided


def _prepare_cli_args(args, parser, argv=None):
    if argv is None:
        argv = sys.argv[1:]
    explicit_dests = _explicitly_provided_dests(parser, argv)
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
        field_to_attr = {
            "mode": "mode",
            "detection_threshold": "threshold",
            "mask_dilate_px": "mask_dilate",
            "mask_feather_px": "mask_feather",
            "edge_ring_px": "edge_ring",
            "tbe_flow_warp": "flow_warp",
            "colour_tune_enable": "colour_tune",
            "colour_tune_tolerance": "colour_tolerance",
            "phash_skip_distance": "phash_distance",
            "auto_band": "auto_band",
            "detection_frame_skip": "frame_skip",
        }
        # Preset booleans exposed only as inverted "--no-*" store_true flags.
        # A preset value of True means "enabled" (the parser default), so map
        # it back onto the negative flag; the user's explicit --no-* wins.
        inverted_flags = {
            "tbe_scene_cut_split": "no_scene_split",
            "kalman_tracking": "no_kalman",
            "phash_skip_enable": "no_phash",
        }
        for fname, value in fields.items():
            if fname == "mode":
                if "mode" not in explicit_dests:
                    args.mode = str(value).lower().replace(" ", "")
                continue
            if fname in inverted_flags:
                neg = inverted_flags[fname]
                if hasattr(args, neg) and neg not in explicit_dests:
                    setattr(args, neg, not bool(value))
                continue
            attr = field_to_attr.get(fname, fname)
            if attr is None or not hasattr(args, attr):
                continue
            if attr not in explicit_dests:
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
    if not 0.0 <= args.retry_backoff <= 600.0:
        parser.error("--retry-backoff must be between 0 and 600 seconds")
    if not 5.0 <= args.translation_timeout <= 3600.0:
        parser.error("--translation-timeout must be between 5 and 3600 seconds")
    translation_enabled = bool(
        args.translate or args.translated_srt or args.translation_source_srt)
    if translation_enabled and args.restyle:
        parser.error("--translate/--translated-srt cannot be combined with --restyle")
    if translation_enabled and not args.translated_srt:
        if not args.translation_target_lang:
            parser.error(
                "--translation-target-lang is required unless --translated-srt is used")
        if args.translation_provider == "command" and not args.translation_command:
            parser.error(
                "--translation-command is required for the command provider")
    return soft_action, dry_run_only, translation_enabled


def _build_processing_config(
    args, translation_enabled, ProcessingConfig, _coerce_backend_mode,
    normalize_processing_config,
):
    config = ProcessingConfig(
        mode=_coerce_backend_mode(args.mode),
        device=f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu",
        work_directory=args.work_dir,
        sttn_skip_detection=args.skip_detection,
        lama_super_fast=args.fast,
        preserve_audio=not args.no_audio,
        detection_lang=args.lang,
        detection_engine=args.ocr_engine,
        language_mask_filter=args.language_filter,
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
        translation_enabled=translation_enabled,
        translation_srt=args.translated_srt,
        translation_source_srt=args.translation_source_srt,
        translation_provider=args.translation_provider,
        translation_source_lang=args.translation_source_lang,
        translation_target_lang=args.translation_target_lang,
        translation_command=args.translation_command,
        translation_style=args.translation_style,
        translation_timeout_seconds=args.translation_timeout,
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
        temporal_mask_union=args.temporal_mask_union,
        temporal_mask_window=args.temporal_mask_window,
        batch_max_retries=args.max_retries,
        batch_retry_backoff_seconds=args.retry_backoff,
        export_srt=args.export_srt,
        ocr_fix_enable=args.ocr_fix,
        export_mask_video=args.export_mask,
        mask_export_format=args.mask_export_format,
        mask_import_path=args.import_mask,
        mask_import_mode=args.mask_import_mode,
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
        d3d12_accel=args.d3d12_accel,
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
    return config


def _apply_cli_config_overlays(args, parser, config):
    from backend.config_schema import (
        CONFIG_SCHEMA_VERSION_KEY,
        apply_backend_payload,
        ensure_supported_schema_version,
        parse_cli_assignments,
        processing_field_names,
        serialize_backend_config,
    )

    ffmpeg_ready = shutil.which("ffmpeg") is not None

    if args.config:
        try:
            overlay = _load_json_config(args.config)
            schema_version = overlay.pop(CONFIG_SCHEMA_VERSION_KEY, None)
            if schema_version is not None:
                ensure_supported_schema_version(schema_version)
            allowed = set(processing_field_names())
            for name in sorted(set(overlay) - allowed):
                logger.warning(f"Ignoring unknown config field: {name}")
            config = apply_backend_payload(
                config,
                {name: value for name, value in overlay.items() if name in allowed},
            )
            logger.info(f"Loaded config overlay from {args.config}")
        except Exception as exc:
            parser.error(f"Could not load --config {args.config}: {exc}")

    try:
        if args.config_schema_version is not None:
            ensure_supported_schema_version(args.config_schema_version)
        if args.config_overrides:
            config = apply_backend_payload(
                config, parse_cli_assignments(args.config_overrides))
    except ValueError as exc:
        parser.error(str(exc))

    if args.validate_config:
        resolved = serialize_backend_config(config)
        print(json.dumps({"resolved_config": resolved}, indent=2, sort_keys=True))
        sys.exit(0)
    return config, ffmpeg_ready


def _run_soft_subtitle_modes(args, parser, config, soft_action) -> bool:
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
    return False


def _run_processing(
    args, parser, config, SubtitleRemover, ProcessingPaused,
    ffmpeg_ready, video_exts,
):
    remover = SubtitleRemover(config)
    remover.on_progress = lambda p, m: print(f"[{int(p*100):3d}%] {m}")

    if getattr(args, "dry_run", False):
        _run_dry_run_and_exit(remover, config, args, video_exts)

    print(
        "[run] "
        f"mode={config.mode.value} | device={config.device} | lang={config.detection_lang} | "
        f"audio={'on' if config.preserve_audio else 'off'} | "
        f"hw_encode={'on' if config.use_hw_encode else 'off'} | "
        f"d3d12={'on' if config.d3d12_accel else 'off'} | "
        f"translation={'on' if config.translation_enabled else 'off'}"
    )
    if config.preserve_audio and not ffmpeg_ready:
        print("[note] FFmpeg is not available, so outputs will be saved without original audio.")

    ckpt_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else _default_checkpoint_dir(config.work_directory)
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pause_requested = {"value": False}

    def _request_pause(_signum=None, _frame=None):
        if not pause_requested["value"]:
            pause_requested["value"] = True
            print("\n[pause] Requested. Waiting for the next safe frame checkpoint...")

    try:
        signal.signal(signal.SIGINT, _request_pause)
    except (ValueError, OSError):
        pass

    def _pause_requested() -> bool:
        return bool(pause_requested["value"])

    base_subtitle_area = config.subtitle_area
    base_subtitle_areas = list(config.subtitle_areas) if config.subtitle_areas else None
    base_subtitle_region_spans = (
        list(config.subtitle_region_spans)
        if config.subtitle_region_spans else None
    )
    base_subtitle_region_keyframes = (
        list(config.subtitle_region_keyframes)
        if config.subtitle_region_keyframes else None
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
            base_subtitle_region_keyframes=base_subtitle_region_keyframes,
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
                    base_subtitle_region_keyframes=(
                        base_subtitle_region_keyframes),
                )
                if band:
                    print(f"[auto-band] {Path(inp).name}: {band}")
                elif not (
                        base_subtitle_area or base_subtitle_areas
                        or base_subtitle_region_spans
                        or base_subtitle_region_keyframes):
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

    def _process_one_with_retry(inp: str, outp: str, record: dict) -> bool:
        """Run _process_one, retrying transient failures up to the configured
        limit with backoff. Permanent errors and ProcessingPaused propagate.

        The processor deliberately converts most failures to ``False`` so it
        can retain a user-facing error message. Treat that result exactly like
        a raised exception for retry classification; otherwise the common
        failure path can never reach this loop.
        """
        from backend.batch_report import is_retriable_error
        max_retries = max(0, int(getattr(config, "batch_max_retries", 0)))
        backoff = float(getattr(config, "batch_retry_backoff_seconds", 5.0))
        attempt = 0
        while True:
            raised_error = False
            try:
                ok = _process_one(inp, outp)
            except ProcessingPaused:
                raise
            except Exception as exc:  # noqa: BLE001
                failure = exc
                raised_error = True
                ok = False
            else:
                mask_export = getattr(remover, "last_mask_export", None)
                if isinstance(mask_export, dict):
                    record["mask_export"] = dict(mask_export)
                mask_import = getattr(remover, "last_mask_import", None)
                if isinstance(mask_import, dict):
                    record["mask_import"] = dict(mask_import)
                timing_report = getattr(remover, "last_timing_report", None)
                if isinstance(timing_report, dict):
                    record["source_timing"] = dict(timing_report)
                output_contract = getattr(
                    remover, "last_output_contract", None)
                if isinstance(output_contract, dict):
                    record["output_contract"] = dict(output_contract)
                if ok:
                    return True
                failure_message = (
                    getattr(remover, "last_error_message", None)
                    or "Processing failed"
                )
                failure_reason = (
                    getattr(remover, "last_error_reason", None) or ""
                )
                detail = ": ".join(
                    part for part in (failure_reason, failure_message) if part
                )
                failure = RuntimeError(detail or "Processing failed")

            record.setdefault("retry_errors", []).append(str(failure))
            if attempt >= max_retries or not is_retriable_error(failure):
                record["retry_attempts"] = attempt
                if raised_error:
                    raise failure
                return False

            if _pause_requested():
                raise ProcessingPaused("Processing paused before retry")
            attempt += 1
            record["retry_attempts"] = attempt
            wait = round(backoff * attempt, 2)
            logger.warning(
                "Transient failure on %s (attempt %d/%d): %s; retrying in %.1fs",
                Path(inp).name, attempt, max_retries, failure, wait,
            )
            print(f"[retry] {Path(inp).name}: attempt {attempt}/{max_retries} "
                  f"after transient error; waiting {wait:.1f}s")
            deadline = time.monotonic() + wait
            while time.monotonic() < deadline:
                if _pause_requested():
                    raise ProcessingPaused("Processing paused before retry")
                time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

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
                    ok = _process_one_with_retry(inp, str(outp), record)
                except ProcessingPaused as exc:
                    print(f"\n[pause] {exc}")
                    finish_batch_item(
                        record,
                        STATUS_PAUSED,
                        message=str(exc),
                        elapsed_seconds=time.monotonic() - started,
                        stage_timings=getattr(remover, "last_stage_timings", None),
                        detection_stats=getattr(remover, "last_detection_stats", None),
                        output_contract=getattr(remover, "last_output_contract", None),
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
                        detection_stats=getattr(remover, "last_detection_stats", None),
                        output_contract=getattr(remover, "last_output_contract", None),
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
                        detection_stats=getattr(remover, "last_detection_stats", None),
                        output_contract=getattr(remover, "last_output_contract", None),
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
        if getattr(args, "json_output", False):
            print(json.dumps({
                "batch": True,
                "total": len(inputs),
                "succeeded": succeeded,
                "failed": failures,
                "review_needed": reviews,
                "items": [
                    {
                        "input": r.get("input"),
                        "output": r.get("output"),
                        "status": r.get("status"),
                        "message": r.get("message"),
                        "elapsed_seconds": r.get("elapsed_seconds"),
                    }
                    for r in records
                ],
            }, indent=2))
        sys.exit(0 if failures == 0 else 1)

    if args.nle_input:
        from backend.nle_sidecar import parse_nle_input
        cap_fps = 24.0
        try:
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
    if getattr(args, "json_output", False):
        print(json.dumps({
            "status": "completed" if success else "failed",
            "input": args.input,
            "output": actual_output or args.output,
            "error": (None if success
                      else getattr(remover, "last_error_message", None)),
            "stage_timings": getattr(remover, "last_stage_timings", None),
            "detection_stats": getattr(remover, "last_detection_stats", None),
            "quality_report": getattr(remover, "last_quality_report", None),
            "source_timing": getattr(remover, "last_timing_report", None),
            "output_contract": getattr(remover, "last_output_contract", None),
        }, indent=2))
    sys.exit(0 if success else 1)


def main():
    """CLI entry point."""
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
    )
    from backend.resume_checkpoint import ProcessingPaused
    from backend import inpainter_registry

    # Built-in modes first, then whatever opt-in backends registered at
    # import time (ONNX / diffusion scaffolds gated by env vars).
    mode_choices = ["sttn", "lama", "propainter", "auto", "migan"]
    for _name, _builder in inpainter_registry.list_modes():
        if _name not in mode_choices:
            mode_choices.append(_name)
    if "--dump-cli-reference" in sys.argv:
        mode_choices = ["sttn", "lama", "propainter", "auto", "migan"]

    parser = _build_parser(mode_choices)
    args = parser.parse_args()

    if _handle_utility_actions(args, parser, attach_json_log):
        return

    soft_action, dry_run_only, translation_enabled = _prepare_cli_args(args, parser)

    config = _build_processing_config(
        args, translation_enabled, ProcessingConfig, _coerce_backend_mode,
        normalize_processing_config,
    )

    config, ffmpeg_ready = _apply_cli_config_overlays(
        args, parser, config,
    )

    video_exts = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpeg', '.mpg'}

    if _run_soft_subtitle_modes(args, parser, config, soft_action):
        return

    _run_processing(
        args, parser, config, SubtitleRemover, ProcessingPaused,
        ffmpeg_ready, video_exts,
    )


if __name__ == "__main__":
    main()
