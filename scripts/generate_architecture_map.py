"""Render the architecture module map from the live source tree.

RM-151: the hand-maintained map in ``docs/architecture.md`` drifted -- it was
missing nine GUI modules and thirty-three backend modules and still listed a
test file that had been split apart. The map is now generated between markers
so it cannot claim ownership that does not exist, and ``--check`` fails the
release build when the tree and the document disagree.

Usage:
    python scripts/generate_architecture_map.py          # rewrite in place
    python scripts/generate_architecture_map.py --check  # fail on drift
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "architecture.md"
START = "<!-- module-map:start -->"
END = "<!-- module-map:end -->"

DESCRIPTIONS: dict[str, str] = {
    "gui/__init__.py": "GUI subpackage re-exports.",
    "gui/app.py": "Tk shell, shared state, queue model, settings.",
    "gui/config.py": "APP_VERSION, QueueItem, GUI ProcessingConfig, settings I/O.",
    "gui/dialog_layout.py": "Work-area fitting and scrollable dialog bodies.",
    "gui/failure_copy.py": "Stable English queue-row failure and status copy.",
    "gui/job_supervisor.py": "Parent-side supervisor for isolated queue jobs.",
    "gui/process_job.py": "Windows job object containment for worker process trees.",
    "gui/track_plan_controller.py": "Pre-run track plan scan and review dialog.",
    "gui/direction.py": "Logical-to-physical RTL mirror for Tk options.",
    "gui/layout_build.py": "Builder mixin: header, settings, queue, preview.",
    "gui/layout_helpers.py": "Shared layout primitives for the builder mixins.",
    "gui/layout_responsive.py": "Responsive / stacked layout mixin.",
    "gui/mask_correction_controller.py": "Mask paint/erase review and selective rerun.",
    "gui/onboarding.py": "First-run onboarding modal mixin.",
    "gui/preview_controller.py": "Preview timeline, proxy planning, A/B compare, live frames, zoom.",
    "gui/processing_controller.py": "Queue worker, pause/stop, reports, notifications.",
    "gui/quality_controller.py": "Quality review, retry, batch-report helpers.",
    "gui/queue_view.py": "Queue table rendering and row state mixin.",
    "gui/release_probe.py": "Packaged scaling, contrast, RTL, and dialog release probe.",
    "gui/region_controller.py": "Region editor: rects, spans, keyframes, polygons.",
    "gui/settings_controller.py": "Settings widgets, presets, mode selection.",
    "gui/support_controller.py": "Support bundle, model cache, log panel, About.",
    "gui/theme.py": "Design tokens, colors, spacing, typography, text scale.",
    "gui/utils.py": "File helpers, media type checks, formatting.",
    "gui/widgets.py": "Custom controls (ModernButton/Toggle/Slider/Picker/...).",
    "backend/__init__.py": "Lazy re-exports SubtitleRemover and friends.",
    "backend/_clean_ref_mixin.py": "Clean-reference plate handling.",
    "backend/_encode_mixin.py": "Encode / mux / audio stages of the processor.",
    "backend/_finalize_mixin.py": "Finalize, output contract, post-restore, sidecar.",
    "backend/_quality_mixin.py": "Quality-report stages of the processor.",
    "backend/_srt_mixin.py": "Tracked OCR consensus and SRT export stages.",
    "backend/a11y.py": "Accessibility metadata helpers.",
    "backend/adapter_manifest.py": "Optional model pins, artifact hashes, provenance records.",
    "backend/atomic_replace.py": "Journalled multi-file replacement and recovery.",
    "backend/batch_report.py": "JSON + Markdown batch summary and output sidecars.",
    "backend/cache_inventory.py": "Cache info/clean and portable model-cache bundles.",
    "backend/cli.py": "argparse entry point and batch driver.",
    "backend/config.py": "Backend ProcessingConfig, InpaintMode, coercers.",
    "backend/config_schema.py": "Canonical config schema and settings migration.",
    "backend/container_payload.py": "Container metadata/chapters/attachment mapping.",
    "backend/crash_reporter.py": "Opt-in crash reporter (allowlisted minimal events).",
    "backend/decode_accel.py": "Hardware decode hints (D3D11/VAAPI/MFX/PyNv).",
    "backend/dependency_caps.py": "Dependency ceilings and execution-provider lanes.",
    "backend/dependency_profiles.py": "Locked profiles plus package, import, pip, and provider verification.",
    "backend/detection.py": "OCR cascade, selectable engines, execution provenance.",
    "backend/detection_geometry.py": "OCR boxes, polygons, text, confidence, track IDs.",
    "backend/device_provider.py": "Device strategy and inpainter construction.",
    "backend/encoder.py": "Output codec probing and HW encoder selection.",
    "backend/execution_provenance.py": "Execution and loaded-model identity record.",
    "backend/failure_reason.py": "Closed-set failure classification for rows and reports.",
    "backend/ffmpeg_profiles.py": "FFmpeg capability profiles and security probe.",
    "backend/hdr.py": "Color metadata preservation and HDR handling.",
    "backend/i18n.py": "gettext localisation runtime.",
    "backend/import_safety.py": "Crash-safe optional-module import probes.",
    "backend/inpainter_registry.py": "In-process inpainter discovery registry.",
    "backend/inpainters_diffusion.py": "Opt-in diffusion adapter scaffolds.",
    "backend/inpainters_onnx.py": "ONNX Runtime inpaint session helpers.",
    "backend/io.py": "Capture, ffprobe, intermediate writers, PrefetchReader.",
    "backend/karaoke_flow.py": "Karaoke optical-flow grouping helper.",
    "backend/language_support.py": "GUI picker scope vs. OCR engine language facts.",
    "backend/mask_corrections.py": "Ordered add/subtract mask corrections.",
    "backend/mask_free_benchmark.py": "Mask-free removal benchmark harness.",
    "backend/job_worker.py": "Child-process entry for one isolated queue job.",
    "backend/frozen_matte.py": "Freeze an approved matte as a reusable input.",
    "backend/webvtt.py": "Loss-aware WebVTT parse / translate / serialize.",
    "backend/matte_interchange.py": "Lossless matte export / import / compose.",
    "backend/model_downloads.py": "First-run guidance and outbound-model inventory.",
    "backend/model_hashes.py": "Vendored SHA-256 hashes and chunked verifier.",
    "backend/nle_sidecar.py": "EDL / FCPXML sidecar export.",
    "backend/ocr_benchmark.py": "OCR engine recall / precision benchmark.",
    "backend/ocr_fix.py": "Per-language OCR replace lists for exported SRT.",
    "backend/ocr_vlm.py": "Optional VLM detectors (Florence-2, Qwen2.5-VL).",
    "backend/onnx_model_info.py": "ONNX opset audit and Windows ML probe.",
    "backend/onnxruntime_cuda.py": "CUDA preload status for ONNX Runtime.",
    "backend/opencv_ocr.py": "PP-OCRv6 via OpenCV 5 DNN and the engine contract.",
    "backend/output_contract.py": "Frozen per-job output policy.",
    "backend/output_quality_preflight.py": "Pre-run output quality warnings.",
    "backend/ocr_variants.py": "Canonical PaddleOCR model families and aliases.",
    "backend/paddle_compat.py": "PaddleOCR 2.x / 3.x API compatibility layer.",
    "backend/post_restore.py": "Post-inpaint temporal smoothing and burn-in.",
    "backend/preprocess.py": "Deinterlacing and keyframe enumeration.",
    "backend/presets.py": "Shared preset library (GUI + CLI).",
    "backend/processor.py": "Frame loop plus the legacy re-export / CLI shim.",
    "backend/proxy_workflow.py": "Proxy-encode workflow for large files.",
    "backend/quality.py": "PSNR / SSIM / VMAF and temporal metrics.",
    "backend/quality_gate.py": "Graduated quality gate with a remediation ladder.",
    "backend/reference_corpus.py": "Exact-profile reference-clip regression harness.",
    "backend/reference_fill.py": "Clean-plate reference fill.",
    "backend/region_editing.py": "Region geometry edit / undo primitives.",
    "backend/region_keyframes.py": "Interpolated moving-region keyframe tracks.",
    "backend/release_staging.py": "Atomic, version-derived release artifact set.",
    "backend/release_verification.py": "Local PyInstaller release evidence writer.",
    "backend/remote_model_policy.py": "Gate for trust_remote_code / torch.hub.",
    "backend/remux.py": "Soft-subtitle strip / keep remux paths.",
    "backend/resume_checkpoint.py": "Crash-resume and pause checkpoints.",
    "backend/safe_image.py": "Bounded image reads.",
    "backend/security_checks.py": "Runtime safety checks (libpng and OpenCV FFmpeg inventory).",
    "backend/segmentation.py": "Optional SAM 2 / MatAnyone / CoTracker adapters.",
    "backend/static_logo_benchmark.py": "Static-logo removal benchmark harness.",
    "backend/subprocess_policy.py": "Hidden, bounded, cancellable child processes.",
    "backend/subtitle_translation.py": "SRT parsing, providers, translated export.",
    "backend/support_bundle.py": "Redacted diagnostics zip export.",
    "backend/temporal_profile.py": "Mask-aware temporal regression metrics and fixtures.",
    "backend/tensorrt_compile.py": "Optional TensorRT engine compilation.",
    "backend/track_plan.py": "Reviewable pre-run text track plans.",
    "backend/tracking.py": "Stable OCR identities, pHash reuse, karaoke grouping.",
    "backend/update_check.py": "Startup version check (opt-in).",
    "backend/vapoursynth_bridge.py": "VapourSynth bridge (opt-in).",
    "backend/whisper_fallback.py": "Whisper-based timing for OCR-empty speech.",
    "backend/work_directory.py": "End-to-end scratch / storage policy.",
    "backend/inpainters/__init__.py": "Mode routing and shared inpainter exports.",
    "backend/inpainters/_common.py": "BaseInpainter, feathering, edge-ring match.",
    "backend/inpainters/auto.py": "Per-scene STTN / ProPainter motion routing.",
    "backend/inpainters/external.py": "VSR_EXTERNAL_INPAINTER bridge.",
    "backend/inpainters/lama.py": "ONNX > OpenCV 5 DNN > PyTorch opt-in > cv2.",
    "backend/inpainters/propainter.py": "TBE plus LaMa residual refinement.",
    "backend/inpainters/sttn.py": "TBE (Temporal Background Exposure).",
}

TOP_LEVEL = (
    ("VideoSubtitleRemover.py", "Entry point (thin launcher -> gui.app)."),
    ("setup.py", "Validated venv bootstrap with atomic setup reports."),
    ("build_exe.bat", "Local PyInstaller build, evidence, release staging."),
    ("requirements.txt", "Pinned and advisory dependency floors."),
    ("dependency_profiles.json", "Reviewed CPU/NVIDIA/DirectML profile manifest."),
    ("Run_VSR_Pro.bat", "Windows launcher with profile verification and repair."),
    ("Run_VSR_Pro_Debug.bat", "Visible-console launcher with profile repair."),
    ("Run_VSR_Pro.ps1", "PowerShell launcher with profile verification and repair."),
)


def _modules(directory: str) -> list[str]:
    return sorted(item.name for item in (ROOT / directory).glob("*.py"))


def _render_group(directory: str, indent: str) -> list[str]:
    names = _modules(directory)
    width = max((len(name) for name in names), default=0) + 1
    lines = []
    for index, name in enumerate(names):
        branch = "`--" if index == len(names) - 1 else "|--"
        description = DESCRIPTIONS.get(f"{directory}/{name}", "")
        lines.append(
            f"{indent}{branch} {name.ljust(width)}# {description}".rstrip())
    return lines


def render_map() -> str:
    lines = ["```", "."]
    for name, description in TOP_LEVEL:
        lines.append(f"|-- {name.ljust(28)}# {description}")
    lines.append("|-- gui/")
    lines.extend(_render_group("gui", "|   "))
    lines.append("|-- backend/")
    lines.extend(_render_group("backend", "|   "))
    lines.append("|   `-- inpainters/")
    lines.extend(_render_group("backend/inpainters", "|       "))
    lines.append("|-- scripts/                     # Build, i18n, and doc tooling.")
    lines.append("|-- tools/                       # Local probes (smoke, UI scaling).")
    lines.append("|-- installer/                   # NSIS installer sources.")
    lines.append("|-- tests/                       # Unit, hardening, and GUI suites.")
    lines.append("`-- docs/                        # Architecture and corpus guides.")
    lines.append("```")
    return "\n".join(lines)


def apply(check: bool = False) -> int:
    text = DOC.read_text(encoding="utf-8")
    if START not in text or END not in text:
        print(f"{DOC} is missing the module-map markers", file=sys.stderr)
        return 2
    head, rest = text.split(START, 1)
    _old, tail = rest.split(END, 1)
    rendered = f"{START}\n\n{render_map()}\n\n{END}"
    updated = head + rendered + tail
    if updated == text:
        print("Architecture module map is current.")
        return 0
    if check:
        print(
            "Architecture module map drifted; run "
            "`python scripts/generate_architecture_map.py`.",
            file=sys.stderr,
        )
        return 1
    DOC.write_text(updated, encoding="utf-8", newline="\n")
    print("Architecture module map updated.")
    return 0


def undocumented_modules() -> list[str]:
    """Modules with no description entry -- a new file nobody described."""
    missing = []
    for directory in ("gui", "backend", "backend/inpainters"):
        for name in _modules(directory):
            if f"{directory}/{name}" not in DESCRIPTIONS:
                missing.append(f"{directory}/{name}")
    return missing


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    missing = undocumented_modules()
    if missing:
        print(
            "Undescribed modules (add them to DESCRIPTIONS): "
            + ", ".join(missing),
            file=sys.stderr,
        )
        return 1
    return apply(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
