"""CLI parser, configuration assembly, and batch dispatch.

Extracted from processor.py as part of RFP-L-1. Provides:

- ``main()``: the ``python -m backend.processor`` argparse + dispatch.
Checkpoint and configuration helpers live in their focused modules so importing
the processing orchestrator never imports this CLI module.
"""

from __future__ import annotations

import argparse
import atexit
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
    file_stability_signature,
    normalized_config_snapshot,
)

logger = logging.getLogger(__name__)
_RUNTIME_HELPERS_LOADED = False
_path_key = None
_probe_subtitle_streams = None
_write_text_atomic = None
write_output_sidecar = None
evaluate_skip_existing = None
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
            "--skip-existing-policy",
            "--watch", "--watch-interval", "--watch-stable-seconds", "--watch-once",
        ),
    ),
    (
        "Removal, detection, and masks",
        (
            "--mode", "--gpu", "--lang", "--language-filter",
            "--skip-detection", "--fast",
            "--threshold", "--vertical", "--frame-skip", "--mask-dilate",
            "--auto-dilate", "--confidence-dilate", "--mask-feather",
            "--temporal-smooth",
            "--edge-ring", "--flow-warp", "--flow-estimator",
            "--poisson-seam", "--no-translucency",
            "--no-global-motion-align",
            "--no-scene-split", "--pyscenedetect",
            "--transnetv2", "--denoise-detect", "--sam2-refine",
            "--matanyone-refine", "--cotracker-propagate", "--no-tbe",
            "--no-adaptive-batch", "--temporal-mask-union",
            "--temporal-mask-window", "--fade-in", "--fade-out",
            "--auto-band", "--no-kalman", "--no-phash",
            "--phash-distance", "--colour-tune", "--colour-tolerance",
            "--auto-threshold", "--keep-chyrons", "--keep-subtitles",
            "--chyron-min-hits", "--karaoke-grouping", "--karaoke-x-gap",
            "--karaoke-y-overlap",
            "--clean-reference", "--clean-reference-offset",
            "--clean-reference-alignment", "--clean-reference-confidence",
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
            "--frozen-matte",
            "--deinterlace",
            "--no-deinterlace-detect", "--keyframe-detect", "--quality-report",
            "--quality-sheet", "--no-verify-removal", "--loudnorm",
            "--decode-accel", "--single-audio",
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
            "--audit-onnx", "--audit-windows-ml", "--scan-weights",
            "--list-fetchable-models", "--fetch-model", "--cache-info",
            "--cache-clean", "--model-cache-export", "--model-cache-import",
            "--support-bundle", "--validate-config", "--self-test",
            "--inference-smoke", "--ocr-benchmark", "--ocr-engine",
            "--rapidocr-variant", "--paddleocr-variant",
            "--ocr-compare-variants", "--dry-run",
            "--plan-out", "--plan-in",
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
    "--clean-reference-confidence": "0.05..0.99",
    "--fade-in": "0..15 frames",
    "--fade-out": "0..15 frames",
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
    "--watch-interval": ">=0.1 seconds",
    "--watch-stable-seconds": ">=0 seconds",
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
    global choose_batch_output_path, evaluate_skip_existing, finish_batch_item
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
        evaluate_skip_existing,
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


def _provenance_dict(remover):
    """RM-147: serialize how the last job actually executed, if recorded."""
    provenance = getattr(remover, "execution_provenance", None)
    if provenance is None:
        return None
    try:
        return provenance.to_dict()
    except Exception:
        logger.debug("Execution provenance serialization failed", exc_info=True)
        return None


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


def _reset_item_failure_state(remover) -> None:
    """Clear the shared remover's last-failure fields before a new item.

    One SubtitleRemover serves the whole batch, and only process_video()
    resets these. Work that runs before it (auto-band detection, checkpoint
    probing) can raise while item A's reason is still recorded, which then
    outranks item B's actual exception in classify_failure_reason.
    """
    for name in ("last_error_message", "last_error_reason"):
        try:
            setattr(remover, name, None)
        except Exception:
            pass


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


_WATCH_IMAGE_EXTENSIONS = frozenset({
    ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp",
})


def _watch_path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _watch_discover_files(
    root: Path,
    media_extensions: set[str] | frozenset[str],
    *,
    excluded_roots: tuple[Path, ...] = (),
) -> list[Path]:
    """Return sorted media files below ``root``, excluding worker artifacts."""
    root = Path(root).resolve()
    extensions = {str(ext).lower() for ext in media_extensions}
    excluded = tuple(Path(path).resolve() for path in excluded_roots)
    found: list[Path] = []
    try:
        candidates = root.rglob("*")
        for candidate in candidates:
            try:
                if not candidate.is_file():
                    continue
                resolved = candidate.resolve()
                if any(_watch_path_is_within(resolved, parent) for parent in excluded):
                    continue
                if resolved.suffix.lower() not in extensions:
                    continue
                found.append(resolved)
            except OSError:
                # A file can disappear or be locked between enumeration and stat.
                continue
    except OSError:
        return []
    return sorted(set(found), key=lambda path: (str(path).casefold(), str(path)))


def _watch_ready_files(
    root: Path,
    media_extensions: set[str] | frozenset[str],
    state: dict[str, tuple[tuple[int, int], float]],
    processed: set[tuple[str, int, int]],
    *,
    now: float | None = None,
    stable_seconds: float = 0.0,
    excluded_roots: tuple[Path, ...] = (),
) -> tuple[list[tuple[Path, tuple[str, int, int]]], int]:
    """Return stable, not-yet-processed files and the active candidate count.

    ``state`` is deliberately supplied by the caller so a long-lived watch
    loop can observe a file growing across polls without writing another
    state file beside user media. The processed key includes the current
    size/mtime, so an edited source becomes a new work item while an item
    that failed remains exactly-once for that stable file version.
    """
    if now is None:
        now = time.monotonic()
    stable_seconds = max(0.0, float(stable_seconds))
    ready: list[tuple[Path, tuple[str, int, int]]] = []
    active_paths: set[str] = set()
    candidate_count = 0
    for path in _watch_discover_files(
        root,
        media_extensions,
        excluded_roots=excluded_roots,
    ):
        path_key = str(path)
        active_paths.add(path_key)
        try:
            size, mtime_ns = file_stability_signature(path)
        except OSError:
            continue
        fingerprint = (path_key, size, mtime_ns)
        if fingerprint in processed:
            continue
        candidate_count += 1
        previous = state.get(path_key)
        signature = (size, mtime_ns)
        if previous is None or previous[0] != signature:
            state[path_key] = (signature, now)
            if stable_seconds > 0.0:
                continue
        if now - state[path_key][1] >= stable_seconds:
            ready.append((path, fingerprint))
    for path_key in set(state) - active_paths:
        state.pop(path_key, None)
    return ready, candidate_count


def _wait_for_watch_interval(seconds: float, pause_check) -> bool:
    """Sleep in short slices so SIGINT can stop an idle watcher promptly."""
    deadline = time.monotonic() + max(0.0, float(seconds))
    while time.monotonic() < deadline:
        if pause_check():
            return False
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    return not pause_check()


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
        # Abbreviated flags cannot be recognised as "the user typed this"
        # by _explicitly_provided_dests, which matches whole tokens, so a
        # preset silently overwrote a value passed as e.g. --thresho 0.8.
        # Every documented flag is spelled in full in --help and the README.
        allow_abbrev=False,
        epilog=(
            "Examples:\n"
            "  python -m backend.processor -i input.mp4 -o output.mp4 -m sttn --lang en\n"
            "  python -m backend.processor --pattern \"inputs/*.mp4\" --out-dir cleaned --mode auto\n"
            "  python -m backend.processor --watch incoming --out-dir cleaned"
        ),
    )
    parser.add_argument("--input", "-i", help="Input file path")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--pattern", help="Glob pattern for batch mode (e.g. 'inputs/*.mp4')")
    parser.add_argument("--out-dir", help="Output directory for batch mode")
    parser.add_argument(
        "--watch",
        metavar="DIR",
        help="Watch DIR recursively for new media files and process them continuously.",
    )
    parser.add_argument(
        "--watch-interval",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="Seconds between watch-folder polls.",
    )
    parser.add_argument(
        "--watch-stable-seconds",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="Require a file's size and mtime to stay unchanged this long before processing.",
    )
    parser.add_argument(
        "--watch-once",
        action="store_true",
        help="Process stable files currently in the watch folder, including files dropped during the drain, then exit.",
    )
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
                       help=("Use configured manual regions as the complete "
                             "mask with any inpainting mode."))
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
                       help="Restore masked texture and add film grain after cleanup (0..0.5; 0 disables).")
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
        help=("Erase subtitles, translate a source SRT or WebVTT file "
              "locally, and re-embed it."))
    parser.add_argument(
        "--translated-srt", default="", metavar="PATH",
        help=("Validated UTF-8 .srt or .vtt that is already translated; "
              "bypasses a provider."))
    parser.add_argument(
        "--translation-source-srt", default="", metavar="PATH",
        help=("Source-language .srt or .vtt to translate; otherwise "
              "OCR/Whisper cues are used. A .vtt source keeps its cue "
              "identifiers, settings, regions, styles, and markup."))
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
    parser.add_argument(
        "--auto-dilate",
        action="store_true",
        help=(
            "Measure outlined and shadowed glyph falloff and build a "
            "continuous mask; an explicit --mask-dilate overrides it."
        ),
    )
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
                       help="Flow-warp TBE frames before aggregation")
    parser.add_argument(
        "--flow-estimator",
        choices=["dis", "farneback"],
        default="dis",
        help="Dense flow estimator for --flow-warp (DIS FAST or Farneback).",
    )
    parser.add_argument(
        "--poisson-seam",
        action="store_true",
        help="Use opt-in gradient-domain seam correction before feathering.",
    )
    parser.add_argument(
        "--no-translucency",
        action="store_true",
        help="Disable fitted semi-transparent overlay recovery.",
    )
    parser.add_argument("--no-global-motion-align", action="store_true",
                       help="Disable affine global-motion alignment before TBE aggregation")
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
                       help="Write an .srt sidecar from tracked OCR consensus")
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
    parser.add_argument(
        "--frozen-matte", default="", metavar="MANIFEST",
        help=("Reuse an approved .mask.json matte as this job's mask, "
              "skipping OCR, tracking, and the mask refiners. Fails "
              "closed if the source, geometry, range, or timing no "
              "longer match what the matte was approved against."))
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
    parser.add_argument("--no-verify-removal", action="store_true",
                       help=(
                           "Skip re-running the detector over the repaired "
                           "region of sampled frames. The check is the "
                           "standard removal-success test and costs two "
                           "detector passes per sampled frame, one over the "
                           "output region and one over the same region in "
                           "the source. It runs on the frames "
                           "--quality-report samples, so without "
                           "--quality-report there is nothing for it to "
                           "run on."
                       ))
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
                       help="Skip only outputs whose identity sidecar matches.")
    parser.add_argument(
        "--skip-existing-policy",
        choices=("verified", "any"),
        default="verified",
        help=(
            "Identity policy for --skip-existing and watch outputs. "
            "Use 'any' only for legacy existence-only behavior."
        ),
    )
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
    parser.add_argument("--list-fetchable-models", action="store_true",
                       help="List optional model weights this build can download, then exit.")
    parser.add_argument("--fetch-model", metavar="ADAPTER[:FILE]",
                       help="Download one pinned optional model weight, verify its SHA-256, and exit.")
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
            "auto", "rapidocr", "opencv-dnn", "paddleocr",
            "opencv", "surya", "vlm-florence2", "vlm-qwen25vl",
            "vlm-paddleocr-vl", "vlm-paddleocr-vl-llama",
        ),
        default="auto",
        help=("Select the OCR detector for processing or --ocr-benchmark; "
              "auto uses the best available engine. surya needs the GPL "
              "opt-in (VSR_ALLOW_GPL=1); vlm-* engines need their optional "
              "dependencies installed; a pinned engine fails with repair "
              "guidance when it cannot run."),
    )
    parser.add_argument(
        "--rapidocr-variant",
        choices=("v6", "v5"),
        default="v6",
        help="Select RapidOCR PP-OCR generation (v6 default, v5 fallback).",
    )
    parser.add_argument(
        "--paddleocr-variant",
        choices=("mobile", "server", "tiny", "small", "medium"),
        default="mobile",
        help=("Select PaddleOCR models: PP-OCRv5 mobile (default, "
              "smaller/faster) or server, or a PP-OCRv6 tier "
              "(tiny/small/medium) from paddleocr 3.7.0."),
    )
    parser.add_argument(
        "--ocr-compare-variants",
        action="store_true",
        help="Benchmark RapidOCR PP-OCRv6 and PP-OCRv5 on the same fixtures.",
    )
    parser.add_argument(
        "--fade-in", metavar="N", type=int, default=0,
        help=("Hold the first confident mask of each text track for N frames "
              "before it, so a subtitle that fades in is covered while it is "
              "still too faint to recognise. 0 disables it."),
    )
    parser.add_argument(
        "--fade-out", metavar="N", type=int, default=0,
        help=("Hold the last confident mask of each text track for N frames "
              "after it, covering the frames where a subtitle fades out. "
              "0 disables it."),
    )
    parser.add_argument(
        "--clean-reference", metavar="PATH", default="",
        help=("Attach a clean plate or a donor video to every timed region "
              "that does not already have one. When the background exists "
              "somewhere (a clean release, a differently-subbed cut) it is "
              "used directly instead of being invented; frames whose "
              "alignment falls below the confidence floor fall back to the "
              "normal inpaint path."),
    )
    parser.add_argument(
        "--clean-reference-offset", metavar="SECONDS", type=float, default=0.0,
        help=("Seconds to add to the source timestamp when looking up a "
              "donor frame. Use a negative value when the donor starts "
              "later than the source. Ignored for a still plate."),
    )
    parser.add_argument(
        "--clean-reference-alignment", default="auto",
        choices=("auto", "translation", "homography"),
        help="How a reference frame is aligned to the source frame.",
    )
    parser.add_argument(
        "--clean-reference-confidence", metavar="FLOAT", type=float,
        default=0.75,
        help=("Alignment confidence a reference frame must reach before it "
              "is used (0.05-0.99). Below it, the frame is inpainted."),
    )
    parser.add_argument(
        "--plan-out", metavar="PATH", default="",
        help=("Scan the input for temporal text tracks and write a "
              "reviewable track plan JSON (frame span, sample text, "
              "thumbnail per track), then exit. Edit the plan's keep flags "
              "and pass it back with --plan-in. Requires -i."),
    )
    parser.add_argument(
        "--plan-in", metavar="PATH", default="",
        help=("Apply an edited track plan: every track marked keep is "
              "excluded from the inpaint mask for exactly its frame span. "
              "A plan-driven run with --export-mask yields a matte "
              "manifest reusable via --frozen-matte."),
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

    if args.list_fetchable_models:
        from backend.model_fetch import fetchable_weights
        for item in fetchable_weights():
            print(
                f"{item['adapter']}:{item['filename']}  "
                f"{item['repository']}@{item['revision'][:12]}  "
                f"{item['license']}"
            )
        sys.exit(0)

    if args.fetch_model:
        sys.exit(_run_model_fetch(args.fetch_model))

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

    if getattr(args, "plan_out", ""):
        if not args.input:
            parser.error("--plan-out requires -i INPUT")
        from backend.track_plan import save_track_plan, scan_track_plan
        device = f"cuda:{args.gpu}" if getattr(args, "gpu", 0) >= 0 else "cpu"
        try:
            plan = scan_track_plan(
                args.input,
                device=device,
                lang=args.lang,
                threshold=args.threshold,
            )
        except ValueError as exc:
            parser.error(str(exc))
        save_track_plan(plan, args.plan_out)
        tracks = plan["tracks"]
        print(f"Track plan written to {args.plan_out}: "
              f"{len(tracks)} track(s)")
        for track in tracks:
            text = track.get("sample_text") or "(no text)"
            # RM-361: a track the scan already decided to keep has to say so
            # here, or the plan silently disagrees with this summary.
            note = ""
            if track.get("persistent_overlay"):
                note = (f"  [kept: {float(track.get('coverage') or 0):.0%} of "
                        "the runtime, outside the subtitle area]")
            print(f"  #{track['id']} frames {track['start_frame']}-"
                  f"{track['end_frame']}  {text[:60]}{note}")
        sys.exit(0)

    if getattr(args, "ocr_benchmark", False):
        from backend.ocr_benchmark import (
            run_default_detector_benchmark,
            run_rapidocr_variant_benchmark,
        )
        device = f"cuda:{args.gpu}" if getattr(args, "gpu", 0) >= 0 else "cpu"
        if args.ocr_compare_variants:
            if args.ocr_engine not in {"auto", "rapidocr"}:
                parser.error("--ocr-compare-variants requires --ocr-engine auto or rapidocr")
            result = run_rapidocr_variant_benchmark(device=device)
        else:
            result = run_default_detector_benchmark(
                device=device,
                engine=args.ocr_engine,
                variant=args.rapidocr_variant,
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


def _run_model_fetch(spec: str) -> int:
    """Download one pinned optional weight. RM-354.

    *spec* is ``adapter`` or ``adapter:filename``. Returns a process exit
    code: 0 when the weight is present and verified, 1 otherwise. Progress is
    printed on one rewritten line so a 208 MB download does not look hung.
    """
    import time

    from backend.model_downloads import format_download_progress
    from backend.model_fetch import fetch_weight

    adapter, _, filename = spec.partition(":")
    last = [-1]
    started = time.monotonic()

    def _progress(read: int, total) -> None:
        # RM-328: the same line the interface shows, so the two surfaces
        # cannot drift into describing one download differently.
        percent = int(read * 100 / total) if total else -1
        if total and percent == last[0]:
            return
        last[0] = percent
        line = format_download_progress(
            filename or adapter, read, total, time.monotonic() - started)
        print(f"\r[fetch] {line}", end="", flush=True)
    try:
        result = fetch_weight(
            adapter.strip(), filename.strip(), progress=_progress
        )
    except KeyboardInterrupt:
        print("\n[fetch] cancelled", file=sys.stderr)
        return 1
    if result.bytes_read:
        print()
    if result.ok:
        print(f"[fetch] {result.filename}: {result.reason} -> {result.path}")
        return 0
    print(
        f"[fetch] {result.filename or adapter}: {result.reason}: "
        f"{result.detail}",
        file=sys.stderr,
    )
    return 1


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
            matched = any(
                tok == opt or tok.startswith(opt + "=") for tok in tokens
            )
            if not matched and len(opt) == 2 and opt.startswith("-"):
                # Short options take an attached value: -msttn, -g0. Those
                # are the user typing the flag just as surely as "-m sttn".
                matched = any(
                    tok.startswith(opt) and len(tok) > 2 and not tok.startswith("--")
                    for tok in tokens
                )
            if matched:
                provided.add(action.dest)
                break
    return provided


def _preset_field_to_dest(parser) -> dict:
    """Map every backend config field name to the argparse dest that sets it.

    Built from the parser itself rather than a hand-kept list: any field with
    a CLI flag must be protected from preset overwrite, and the two
    hand-maintained dicts this replaces covered 14 of them, so a preset
    quietly won over flags like --keep-chyrons, --no-tbe and --max-retries.
    """
    mapping = {}
    for action in parser._actions:
        dest = getattr(action, "dest", "")
        if not dest or dest in {"help", "version"}:
            continue
        mapping.setdefault(dest, dest)
    return mapping


def _prepare_cli_args(args, parser, argv=None):
    if argv is None:
        argv = sys.argv[1:]
    explicit_dests = _explicitly_provided_dests(parser, argv)
    args._explicit_dests = explicit_dests
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
    # --dry-run writes nothing, so requiring --output/--out-dir
    # contradicted the flag's own help text.
    dry_run_only = bool(args.soft_subtitle_dry_run or getattr(args, "dry_run", False))

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
            "auto_dilate_enable": "auto_dilate",
            "mask_feather_px": "mask_feather",
            "edge_ring_px": "edge_ring",
            "tbe_flow_warp": "flow_warp",
            "tbe_flow_estimator": "flow_estimator",
            "poisson_seam_enable": "poisson_seam",
            "colour_tune_enable": "colour_tune",
            "colour_tune_tolerance": "colour_tolerance",
            "phash_skip_distance": "phash_distance",
            "auto_band": "auto_band",
            "detection_frame_skip": "frame_skip",
            # Renamed dests below this line were missing, so a preset
            # carrying any of them silently beat an explicitly typed flag.
            "detection_vertical": "vertical",
            "confidence_weighted_dilation": "confidence_dilate",
            "temporal_smooth_radius": "temporal_smooth",
            "detection_denoise": "denoise_detect",
            "tbe_scene_cut_use_pyscenedetect": "pyscenedetect",
            "batch_max_retries": "max_retries",
            "batch_retry_backoff_seconds": "retry_backoff",
            "keyframe_detection": "keyframe_detect",
            "karaoke_x_gap_px": "karaoke_x_gap",
        }
        # Preset booleans exposed only as inverted "--no-*" store_true flags.
        # A preset value of True means "enabled" (the parser default), so map
        # it back onto the negative flag; the user's explicit --no-* wins.
        inverted_flags = {
            "tbe_global_motion_align": "no_global_motion_align",
            "tbe_scene_cut_split": "no_scene_split",
            "kalman_tracking": "no_kalman",
            "phash_skip_enable": "no_phash",
            "tbe_enable": "no_tbe",
            "adaptive_batch": "no_adaptive_batch",
            "deinterlace_auto": "no_deinterlace_detect",
            "remove_chyrons": "keep_chyrons",
            "remove_subtitles": "keep_subtitles",
        }
        # Preset fields with no CLI flag but that name a real backend config
        # field are applied to the built config later (via apply_backend_payload
        # in _apply_cli_config_overlays) so user presets round-trip losslessly
        # instead of silently dropping unmapped fields.
        from backend.config_schema import processing_field_names as _pfn
        backend_fields = set(_pfn())
        preset_backend_overrides: dict = {}
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
            if attr is None:
                continue
            if not hasattr(args, attr):
                if fname in backend_fields:
                    preset_backend_overrides[fname] = value
                else:
                    logger.warning(
                        "Preset %r field %r has no CLI mapping and is not a "
                        "known config field; ignoring", args.preset, fname)
                continue
            if attr not in explicit_dests:
                setattr(args, attr, value)
        args._preset_backend_overrides = preset_backend_overrides
        logger.info(f"Applied preset: {args.preset}")

    watch_path = getattr(args, "watch", None)
    if getattr(args, "watch_once", False) and not watch_path:
        parser.error("--watch-once requires --watch")
    if watch_path and (soft_mode_count or args.soft_subtitle_dry_run):
        parser.error("--watch cannot be combined with soft-subtitle modes")
    if watch_path and args.nle_input:
        parser.error("--watch cannot be combined with --nle-input")
    if watch_path and args.frozen_matte:
        parser.error("--watch cannot be combined with --frozen-matte")
    if not 0.1 <= args.watch_interval:
        parser.error("--watch-interval must be at least 0.1 seconds")
    if not 0.0 <= args.watch_stable_seconds:
        parser.error("--watch-stable-seconds must be zero or positive")
    if not args.validate_config:
        source_count = sum(bool(value) for value in (args.input, args.pattern, watch_path))
        if source_count == 0:
            parser.error("one of --input, --pattern, or --watch is required")
        if source_count > 1:
            parser.error("--input, --pattern, and --watch are mutually exclusive")
        if watch_path:
            if dry_run_only:
                parser.error("--watch cannot be combined with --dry-run")
            if args.output:
                parser.error("--watch uses --out-dir, not --output")
            watch_root = Path(watch_path).resolve()
            if not watch_root.is_dir():
                parser.error(f"--watch directory does not exist: {watch_path}")
            if not args.out_dir:
                parser.error("--watch requires --out-dir")
            if watch_root == Path(args.out_dir).resolve():
                parser.error("--out-dir must differ from the --watch directory")
            # Watch mode is an unattended batch drain. Existing canonical
            # outputs are always skipped instead of receiving a collision suffix.
            args.skip_existing = True
        if args.pattern and not args.out_dir and not dry_run_only:
            parser.error("--pattern requires --out-dir")
        if args.input and not args.output and not dry_run_only:
            parser.error("--input requires --output")
        # RM-153: a frozen matte is pinned to one source file, one frame
        # range, and one timing table, so it cannot describe a glob of
        # inputs and cannot coexist with a second matte source.
        if args.frozen_matte and args.pattern:
            parser.error("--frozen-matte applies to a single --input, not --pattern")
        if args.frozen_matte and args.import_mask:
            parser.error("--frozen-matte and --import-mask are mutually exclusive")
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


def _apply_track_plan_args(args, parser, config) -> None:
    """RM-275: merge an edited --plan-in track plan into the run config.

    Every track marked keep becomes a frame-bounded subtract correction, so
    exactly its span and region are excluded from the inpaint mask.
    """
    if not getattr(args, "plan_in", ""):
        return
    from backend.track_plan import load_track_plan, plan_to_mask_corrections
    try:
        plan = load_track_plan(args.plan_in)
    except (OSError, ValueError) as exc:
        parser.error(f"--plan-in: {exc}")
    exclusions = plan_to_mask_corrections(plan)
    existing = list(getattr(config, "manual_mask_corrections", None) or [])
    config.manual_mask_corrections = existing + exclusions
    kept = sum(1 for track in plan["tracks"] if track.get("keep"))
    print(f"Track plan applied: {kept} track(s) kept "
          f"(excluded from cleanup), {len(plan['tracks']) - kept} to remove")


def _apply_clean_reference_args(args, parser, config) -> None:
    """RM-283: attach `--clean-reference` to every timed region.

    Attaching to spans rather than introducing a new global field keeps one
    representation of the feature: the GUI already stores the reference on
    the span, and the sidecar provenance is written per span.
    """
    path = str(getattr(args, "clean_reference", "") or "").strip()
    if not path:
        return
    if not Path(path).is_file():
        parser.error(f"--clean-reference: file not found: {path}")
    spans = list(getattr(config, "subtitle_region_spans", None) or [])
    if not spans:
        parser.error(
            "--clean-reference needs at least one timed region; define "
            "subtitle_region_spans in --config or draw them in the GUI"
        )
    from backend.reference_fill import normalize_clean_reference

    spec = normalize_clean_reference({
        "path": path,
        "offset_seconds": getattr(args, "clean_reference_offset", 0.0),
        "alignment": getattr(args, "clean_reference_alignment", "auto"),
        "min_confidence": getattr(args, "clean_reference_confidence", 0.75),
    })
    if spec is None:
        parser.error(f"--clean-reference: unusable reference: {path}")
    attached = 0
    for span in spans:
        if not isinstance(span, dict) or span.get("clean_reference"):
            continue
        span["clean_reference"] = dict(spec)
        attached += 1
    config.subtitle_region_spans = spans
    print(
        f"Clean reference attached to {attached} timed region(s) "
        f"as a {spec['kind']}"
    )


def _frozen_matte_from_args(args) -> dict:
    """RM-153: build a frozen-matte record from `--frozen-matte MANIFEST`.

    Freezing here is deliberately eager: the manifest, artifact, and the
    source are checked against each other now, so the reason a matte
    cannot be reused is reported before any decoding starts rather than
    surfacing as an opaque mid-run failure.
    """
    manifest = str(getattr(args, "frozen_matte", "") or "").strip()
    if not manifest:
        return {}
    from backend.frozen_matte import freeze_matte

    return freeze_matte(manifest, args.input)


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
        rapidocr_variant=args.rapidocr_variant,
        paddleocr_variant=args.paddleocr_variant,
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
        auto_dilate_enable=(
            bool(args.auto_dilate)
            and "mask_dilate" not in getattr(args, "_explicit_dests", set())
        ),
        confidence_weighted_dilation=args.confidence_dilate,
        mask_feather_px=args.mask_feather,
        temporal_smooth_radius=args.temporal_smooth,
        edge_ring_px=args.edge_ring,
        tbe_enable=not args.no_tbe,
        tbe_flow_warp=args.flow_warp,
        tbe_flow_estimator=args.flow_estimator,
        tbe_global_motion_align=not args.no_global_motion_align,
        poisson_seam_enable=args.poisson_seam,
        translucency_enable=not args.no_translucency,
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
        mask_fade_in_frames=args.fade_in,
        mask_fade_out_frames=args.fade_out,
        batch_max_retries=args.max_retries,
        batch_retry_backoff_seconds=args.retry_backoff,
        export_srt=args.export_srt,
        ocr_fix_enable=args.ocr_fix,
        export_mask_video=args.export_mask,
        mask_export_format=args.mask_export_format,
        mask_import_path=args.import_mask,
        frozen_matte=_frozen_matte_from_args(args),
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
        verify_removal=not getattr(args, "no_verify_removal", False),
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

    # Preset fields that have no dedicated CLI flag are applied first, so an
    # explicit --config file or --set override still wins over the preset.
    preset_overrides = getattr(args, "_preset_backend_overrides", None)
    if preset_overrides:
        try:
            config = apply_backend_payload(config, preset_overrides)
        except ValueError as exc:
            parser.error(str(exc))

    if args.config:
        try:
            overlay = _load_json_config(args.config)
            schema_version = overlay.pop(CONFIG_SCHEMA_VERSION_KEY, None)
            if schema_version is not None:
                ensure_supported_schema_version(schema_version)
            allowed = set(processing_field_names())
            unknown = sorted(set(overlay) - allowed)
            if unknown:
                parser.error(
                    "unknown config field: " + ", ".join(unknown)
                )
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
    _apply_track_plan_args(args, parser, config)
    _apply_clean_reference_args(args, parser, config)
    return config, ffmpeg_ready


def _print_existing_output_decision(input_name: str, decision: dict) -> None:
    if not isinstance(decision, dict) or not decision.get("requested"):
        return
    action = decision.get("action")
    if action == "skip":
        if decision.get("reason_code") == "legacy-any":
            detail = "legacy any policy; identity not verified"
        else:
            detail = "verified output identity"
        print(f"[skip] {input_name} ({detail})")
    elif decision.get("output_exists"):
        print(f"[reprocess] {input_name} ({decision.get('message', '')})")


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
        soft_identity_config = {
            "mode": "soft-subtitles",
            "device": "cpu",
            "output_codec": "copy",
            "output_quality": config.output_quality,
            "soft_action": soft_action.value,
        }
        if args.pattern:
            from glob import glob
            inputs = sorted(glob(args.pattern, recursive=True))
            inputs = [p for p in inputs if Path(p).is_file()]
            if not inputs:
                parser.error(f"No files matched pattern: {args.pattern}")
            out_dir = Path(args.out_dir)
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                parser.error(f"--out-dir is not usable: {exc}")
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
                        config=soft_identity_config,
                        skip_existing=args.skip_existing,
                        skip_existing_policy=args.skip_existing_policy,
                        soft_action=soft_action.value,
                    )
                    records.append(record)
                    print(f"\n[soft-subtitles] ({i}/{len(inputs)}) {src.name}")
                    if record["planned_result"] == STATUS_SKIPPED_EXISTING:
                        _print_existing_output_decision(
                            src.name, record["skip_existing"])
                        finish_batch_item(
                            record,
                            STATUS_SKIPPED_EXISTING,
                            message=record["skip_existing"]["message"],
                        )
                        continue
                    _print_existing_output_decision(
                        src.name, record["skip_existing"])
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
                            config=soft_identity_config,
                            status="soft-subtitle-remuxed",
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
                            error=exc,
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
            if args.skip_existing:
                decision = evaluate_skip_existing(
                    args.input,
                    args.output,
                    soft_identity_config,
                    policy=args.skip_existing_policy,
                )
                _print_existing_output_decision(
                    Path(args.input).name, decision)
                if decision["action"] == "skip":
                    sys.exit(0)
            started = time.monotonic()
            _run_soft_subtitle_only(args.input, args.output, soft_action)
            elapsed = time.monotonic() - started
            write_output_sidecar(
                input_path=args.input,
                output_path=args.output,
                config=soft_identity_config,
                status="soft-subtitle-remuxed",
                elapsed_seconds=elapsed,
                stage_timings={"mux": elapsed},
                app_version=_app_version(),
            )
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
    # RM-316: hold the system awake for the run and release it however
    # the run ends.
    from backend import keep_awake

    keep_awake.acquire()
    atexit.register(keep_awake.release_all)

    remover = SubtitleRemover(config)
    identity_config = normalized_config_snapshot(config)
    remover.output_identity_config = identity_config
    remover.on_progress = lambda p, m: print(f"[{int(p*100):3d}%] {m}")

    if getattr(args, "dry_run", False):
        _run_dry_run_and_exit(remover, config, args, video_exts)

    # RM-321: a fixed manual region gives the temporal engines nothing to
    # recover from, so say so before the run rather than reporting a clean
    # success over a cv2 result. This sits after the dry-run branch: a dry
    # run performs no removal, so there is no fallback to warn about.
    from backend.config import (
        STATIC_REGION_DEGRADES_MESSAGE,
        static_region_degrades_to_cv2,
    )

    if static_region_degrades_to_cv2(config):
        print(
            "WARNING: "
            + STATIC_REGION_DEGRADES_MESSAGE.format(mode=config.mode.value),
            file=sys.stderr,
        )

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

    # RM-356: this build knows there is an NVIDIA card and knows it cannot use
    # it, and used to put both facts in the log as a warning nobody reads.
    from backend.device_provider import cpu_build_on_nvidia_hardware

    # args.gpu is what the user asked for. config.device is already past the
    # fallback, so "cpu" there cannot tell a deliberate CPU run from one that
    # dropped to the CPU, which is the whole case this notice exists for.
    try:
        gpu_notice = cpu_build_on_nvidia_hardware(
            requested_device=f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu",
        )
    except Exception:  # noqa: BLE001 - an advisory note must not end the run
        # This shells nvidia-smi and reads the build stamp. Neither is worth
        # aborting a render for, and the GUI already guards the same call.
        logger.debug("CPU-build GPU notice probe failed", exc_info=True)
        gpu_notice = None
    if gpu_notice:
        print(
            f"[note] {gpu_notice['adapter']} is present but this build runs "
            f"on the CPU. The {gpu_notice['assetPrefix']} download uses it: "
            f"{gpu_notice['releasesUrl']}"
        )

    ckpt_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else _default_checkpoint_dir(config.work_directory)
    )
    try:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        parser.error(f"--checkpoint-dir is not usable: {exc}")
    pause_requested = {"value": False}

    def _request_pause(_signum=None, _frame=None):
        if not pause_requested["value"]:
            pause_requested["value"] = True
            print("\n[pause] Requested. Waiting for the next safe frame checkpoint...")
        else:
            # The first Ctrl+C is a cooperative pause. A second one is an
            # explicit escalation and must reach the existing KeyboardInterrupt
            # failure path instead of being silently ignored.
            signal.signal(signal.SIGINT, signal.default_int_handler)
            raise KeyboardInterrupt

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

    def _verified_checkpoint_done(inp: str, outp: str, key: str) -> bool:
        if args.no_resume or not _checkpoint_is_done(ckpt_dir, key, outp):
            return False
        decision = evaluate_skip_existing(
            inp,
            outp,
            identity_config,
            policy="verified",
        )
        if decision["action"] == "skip":
            return True
        print(
            f"[resume] {Path(inp).name}: completed checkpoint ignored; "
            f"{decision['message']}"
        )
        return False

    def _process_one(
        inp: str,
        outp: str,
        *,
        preflight_done: bool = False,
    ) -> bool:
        key = _checkpoint_key(inp, outp, identity_config)
        if not preflight_done:
            if args.skip_existing:
                decision = evaluate_skip_existing(
                    inp,
                    outp,
                    identity_config,
                    policy=args.skip_existing_policy,
                )
                _print_existing_output_decision(Path(inp).name, decision)
                if decision["action"] == "skip":
                    return True
            if _verified_checkpoint_done(inp, outp, key):
                print(f"[skip] {Path(inp).name} (verified checkpoint output)")
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
                ok = _process_one(inp, outp, preflight_done=True)
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

    if args.watch:
        out_dir = Path(args.out_dir).resolve()
        watch_dir = Path(args.watch).resolve()
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            parser.error(f"--out-dir is not usable: {exc}")
        excluded_roots = [out_dir, ckpt_dir.resolve()]
        work_dir = str(getattr(config, "work_directory", "") or "").strip()
        if work_dir:
            excluded_roots.append(Path(work_dir).resolve())
        media_extensions = set(video_exts) | _WATCH_IMAGE_EXTENSIONS
        watch_state: dict[str, tuple[tuple[int, int], float]] = {}
        processed: set[tuple[str, int, int]] = set()
        records: list[dict] = []
        reserved_outputs: set = set()
        watch_started_at = datetime.datetime.now(datetime.timezone.utc)
        interrupted = False
        paused = False

        def _run_watch_item(inp: str, outp: str, record: dict) -> tuple[bool, bool]:
            """Process one discovered file; return (success, paused)."""
            src = Path(inp)
            _print_output_quality_preflight(
                record.get("output_quality_preflight") or {}
            )
            print(f"\n[watch] {src.name}")
            _reset_item_failure_state(remover)
            planned = record["planned_result"]
            if planned == STATUS_SKIPPED_EXISTING:
                _print_existing_output_decision(
                    src.name, record["skip_existing"])
                finish_batch_item(
                    record,
                    STATUS_SKIPPED_EXISTING,
                    message=record["skip_existing"]["message"],
                )
                return True, False
            if planned == STATUS_CHECKPOINT_DONE:
                print(f"[skip] {src.name} (verified checkpoint output)")
                finish_batch_item(
                    record,
                    STATUS_CHECKPOINT_DONE,
                    message="Checkpoint and output identity are complete",
                )
                return True, False

            _print_existing_output_decision(
                src.name, record["skip_existing"])

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
                    execution_provenance=_provenance_dict(remover),
                    output_contract=getattr(remover, "last_output_contract", None),
                )
                return False, True
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Failed on {src.name}: {exc}")
                finish_batch_item(
                    record,
                    STATUS_FAILED,
                    message=str(exc),
                    elapsed_seconds=time.monotonic() - started,
                    stage_timings=getattr(remover, "last_stage_timings", None),
                    detection_stats=getattr(remover, "last_detection_stats", None),
                    execution_provenance=_provenance_dict(remover),
                    output_contract=getattr(remover, "last_output_contract", None),
                    failure_reason=getattr(remover, "last_error_reason", None),
                    error=exc,
                )
                return False, False

            quality_report = (
                getattr(remover, "last_quality_report", None) if ok else None
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
                execution_provenance=_provenance_dict(remover),
                output_contract=getattr(remover, "last_output_contract", None),
                failure_reason=getattr(remover, "last_error_reason", None),
            )
            return bool(ok), False

        print(
            f"[watch] directory={watch_dir} | out={out_dir} | "
            f"interval={args.watch_interval:g}s | "
            f"stable={args.watch_stable_seconds:g}s | "
            f"once={'on' if args.watch_once else 'off'}"
        )
        try:
            while True:
                ready, candidate_count = _watch_ready_files(
                    watch_dir,
                    media_extensions,
                    watch_state,
                    processed,
                    stable_seconds=args.watch_stable_seconds,
                    excluded_roots=tuple(excluded_roots),
                )
                for path, fingerprint in ready:
                    if _pause_requested():
                        paused = True
                        break
                    canonical = Path(out_dir) / f"{path.stem}_no_sub{path.suffix}"
                    collision = _path_key(canonical) in reserved_outputs
                    outp = choose_batch_output_path(
                        str(path),
                        out_dir,
                        "_no_sub",
                        reserved_outputs,
                        skip_existing=not collision,
                    )
                    reserved_outputs.add(_path_key(outp))
                    key = _checkpoint_key(
                        str(path), str(outp), identity_config)
                    checkpoint_done = _verified_checkpoint_done(
                        str(path), str(outp), key)
                    record = make_batch_item_record(
                        str(path),
                        str(outp),
                        config=config,
                        skip_existing=True,
                        skip_existing_policy=args.skip_existing_policy,
                        identity_config=identity_config,
                        checkpoint_done=checkpoint_done,
                    )
                    records.append(record)
                    _run_watch_item(str(path), str(outp), record)
                    processed.add(fingerprint)
                    watch_state.pop(str(path), None)
                    _write_cli_batch_reports(
                        out_dir,
                        records,
                        kind="watch-folder",
                        started_at=watch_started_at,
                    )
                    if record.get("status") == STATUS_PAUSED:
                        paused = True
                        break
                if paused or _pause_requested():
                    interrupted = True
                    break
                if args.watch_once and candidate_count == 0:
                    break
                if not _wait_for_watch_interval(args.watch_interval, _pause_requested):
                    interrupted = True
                    break
        except KeyboardInterrupt:
            print("\n[watch] Interrupted by user -- partial results kept on disk.")
            _cancel_pending_records(records)
            interrupted = True
        finally:
            _write_cli_batch_reports(
                out_dir,
                records,
                kind="watch-folder",
                started_at=watch_started_at,
            )
        if paused:
            print("[watch] Paused. Re-run the same command to resume pending files.")
            sys.exit(130)
        if interrupted:
            sys.exit(130)
        failures = sum(1 for record in records if record.get("status") == STATUS_FAILED)
        reviews = sum(
            1 for record in records
            if record.get("status") == STATUS_REVIEW_NEEDED
        )
        succeeded = len(records) - failures
        suffix = f", {reviews} review-needed" if reviews else ""
        print(
            f"\n[watch] drain complete: {succeeded}/{len(records)} succeeded{suffix}"
        )
        if failures:
            print("[watch] Some items failed; review vsr-batch-summary before retrying.")
        if getattr(args, "json_output", False):
            print(json.dumps({
                "watch": True,
                "total": len(records),
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

    if args.pattern:
        from glob import glob
        inputs = sorted(glob(args.pattern, recursive=True))
        inputs = [p for p in inputs if Path(p).is_file()]
        if not inputs:
            parser.error(f"No files matched pattern: {args.pattern}")
        out_dir = Path(args.out_dir)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            parser.error(f"--out-dir is not usable: {exc}")
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
                key = _checkpoint_key(inp, str(outp), identity_config)
                checkpoint_done = _verified_checkpoint_done(
                    inp, str(outp), key)
                record = make_batch_item_record(
                    inp,
                    str(outp),
                    config=config,
                    skip_existing=args.skip_existing,
                    skip_existing_policy=args.skip_existing_policy,
                    identity_config=identity_config,
                    checkpoint_done=checkpoint_done,
                )
                records.append(record)
                _print_output_quality_preflight(
                    record.get("output_quality_preflight") or {}
                )
                print(f"\n[batch] ({i}/{len(inputs)}) {src.name}")
                _reset_item_failure_state(remover)
                if record["planned_result"] == STATUS_SKIPPED_EXISTING:
                    _print_existing_output_decision(
                        src.name, record["skip_existing"])
                    finish_batch_item(
                        record,
                        STATUS_SKIPPED_EXISTING,
                        message=record["skip_existing"]["message"],
                    )
                    continue
                if record["planned_result"] == STATUS_CHECKPOINT_DONE:
                    print(f"[skip] {src.name} (verified checkpoint output)")
                    finish_batch_item(
                        record,
                        STATUS_CHECKPOINT_DONE,
                        message="Checkpoint and output identity are complete",
                    )
                    continue
                _print_existing_output_decision(
                    src.name, record["skip_existing"])
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
                        execution_provenance=_provenance_dict(remover),
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
                        execution_provenance=_provenance_dict(remover),
                        output_contract=getattr(remover, "last_output_contract", None),
                        failure_reason=getattr(remover, "last_error_reason", None),
                        error=exc,
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
                        execution_provenance=_provenance_dict(remover),
                        output_contract=getattr(remover, "last_output_contract", None),
                        failure_reason=getattr(remover, "last_error_reason", None),
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
        from backend.io import probe_video_fps
        probed_fps = probe_video_fps(args.input)
        if probed_fps and probed_fps > 0:
            cap_fps = float(probed_fps)
        else:
            cap_fps = 24.0
            logger.warning(
                "Could not probe source frame rate for %s; assuming %.3f fps "
                "for NLE timecode->frame conversion. Non-24fps sources may "
                "misalign -- install ffmpeg/ffprobe or check the file.",
                args.input, cap_fps,
            )
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
            identity_config = normalized_config_snapshot(config)
            remover.output_identity_config = identity_config
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
            "execution_provenance": _provenance_dict(remover),
            "quality_report": getattr(remover, "last_quality_report", None),
            "source_timing": getattr(remover, "last_timing_report", None),
            "output_contract": getattr(remover, "last_output_contract", None),
        }, indent=2))
    sys.exit(0 if success else 1)


def main():
    """CLI entry point."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass
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

    # RM-153: a frozen matte that no longer matches its manifest, artifact,
    # or source is a *user-facing* refusal with a curated message -- present
    # it as a CLI error, not a traceback.
    from backend.frozen_matte import FrozenMatteError

    try:
        config = _build_processing_config(
            args, translation_enabled, ProcessingConfig, _coerce_backend_mode,
            normalize_processing_config,
        )
    except FrozenMatteError as exc:
        parser.error(f"--frozen-matte: {exc.user_message}")
    except (ValueError, TypeError) as exc:
        # Preset values reach args unvalidated (unlike --set, which is
        # coerced), so an unregistered mode or a string-typed numeric in a
        # user preset used to surface as a raw traceback.
        parser.error(str(exc))

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
