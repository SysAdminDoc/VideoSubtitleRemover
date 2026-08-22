# Architecture Map

This document walks the pipeline a frame travels through, names every
module it touches, and points new contributors at the right file for a
given change. Pairs with [ROADMAP.md](../ROADMAP.md) and
[CHANGELOG.md](../CHANGELOG.md).

> Concrete and up to date as of the v3.37.0 pass.
> Keep this in sync when modules move.

---

## Module map

<!-- module-map:start -->

```
.
|-- VideoSubtitleRemover.py     # Entry point (thin launcher -> gui.app).
|-- setup.py                    # First-run venv bootstrap and dependency profiles.
|-- build_exe.bat               # Local PyInstaller build, evidence, release staging.
|-- requirements.txt            # Pinned and advisory dependency floors.
|-- dependency_profiles.json    # Reviewed CPU/NVIDIA/DirectML profile manifest.
|-- Run_VSR_Pro.bat             # Windows launcher.
|-- Run_VSR_Pro_Debug.bat       # Windows launcher with a visible console.
|-- Run_VSR_Pro.ps1             # PowerShell launcher.
|-- gui/
|   |-- __init__.py                   # GUI subpackage re-exports.
|   |-- app.py                        # Tk shell, shared state, queue model, settings.
|   |-- config.py                     # APP_VERSION, QueueItem, GUI ProcessingConfig, settings I/O.
|   |-- dialog_layout.py              # Work-area fitting and scrollable dialog bodies.
|   |-- direction.py                  # Logical-to-physical RTL mirror for Tk options.
|   |-- failure_copy.py               # Stable English queue-row failure and status copy.
|   |-- job_supervisor.py             # Parent-side supervisor for isolated queue jobs.
|   |-- layout_build.py               # Builder mixin: header, settings, queue, preview.
|   |-- layout_helpers.py             # Shared layout primitives for the builder mixins.
|   |-- layout_responsive.py          # Responsive / stacked layout mixin.
|   |-- mask_correction_controller.py # Mask paint/erase review and selective rerun.
|   |-- onboarding.py                 # First-run onboarding modal mixin.
|   |-- preview_controller.py         # Preview timeline, proxy planning, A/B compare, live frames, zoom.
|   |-- process_job.py                # Windows job object containment for worker process trees.
|   |-- processing_controller.py      # Queue worker, pause/stop, reports, notifications.
|   |-- quality_controller.py         # Quality review, retry, batch-report helpers.
|   |-- queue_view.py                 # Queue table rendering and row state mixin.
|   |-- region_controller.py          # Region editor: rects, spans, keyframes, polygons.
|   |-- release_probe.py              # Packaged scaling, contrast, RTL, and dialog release probe.
|   |-- settings_controller.py        # Settings widgets, presets, mode selection.
|   |-- support_controller.py         # Support bundle, model cache, log panel, About.
|   |-- theme.py                      # Design tokens, colors, spacing, typography, text scale.
|   |-- track_plan_controller.py      # Pre-run track plan scan and review dialog.
|   |-- utils.py                      # File helpers, media type checks, formatting.
|   `-- widgets.py                    # Custom controls (ModernButton/Toggle/Slider/Picker/...).
|-- backend/
|   |-- __init__.py                 # Lazy re-exports SubtitleRemover and friends.
|   |-- _clean_ref_mixin.py         # Clean-reference plate handling.
|   |-- _encode_mixin.py            # Encode / mux / audio stages of the processor.
|   |-- _finalize_mixin.py          # Finalize, output contract, post-restore, sidecar.
|   |-- _quality_mixin.py           # Quality-report stages of the processor.
|   |-- _srt_mixin.py               # SRT export and clean-reference export stages.
|   |-- a11y.py                     # Accessibility metadata helpers.
|   |-- adapter_manifest.py         # Optional model provenance and hash policy.
|   |-- atomic_replace.py           # Journalled multi-file replacement and recovery.
|   |-- batch_report.py             # JSON + Markdown batch summary and output sidecars.
|   |-- cache_inventory.py          # Cache info/clean and portable model-cache bundles.
|   |-- cli.py                      # argparse entry point and batch driver.
|   |-- config.py                   # Backend ProcessingConfig, InpaintMode, coercers.
|   |-- config_schema.py            # Canonical config schema and settings migration.
|   |-- container_payload.py        # Container metadata/chapters/attachment mapping.
|   |-- crash_reporter.py           # Opt-in crash reporter (allowlisted minimal events).
|   |-- decode_accel.py             # Hardware decode hints (D3D11/VAAPI/MFX/PyNv).
|   |-- dependency_caps.py          # Dependency ceilings and execution-provider lanes.
|   |-- dependency_profiles.py      # Reviewed CPU/NVIDIA/DirectML dependency locks.
|   |-- detection.py                # OCR cascade, selectable engines, execution provenance.
|   |-- detection_geometry.py       # Normalized OCR boxes and polygon geometry.
|   |-- device_provider.py          # Device strategy and inpainter construction.
|   |-- encoder.py                  # Output codec probing and HW encoder selection.
|   |-- execution_provenance.py     # Requested vs. effective device/engine record.
|   |-- failure_reason.py           # Closed-set failure classification for rows and reports.
|   |-- ffmpeg_profiles.py          # FFmpeg capability profiles and security probe.
|   |-- frozen_matte.py             # Freeze an approved matte as a reusable input.
|   |-- hdr.py                      # Color metadata preservation and HDR handling.
|   |-- i18n.py                     # gettext localisation runtime.
|   |-- import_safety.py            # Crash-safe optional-module import probes.
|   |-- inpainter_registry.py       # In-process inpainter discovery registry.
|   |-- inpainters_diffusion.py     # Opt-in diffusion adapter scaffolds.
|   |-- inpainters_onnx.py          # ONNX Runtime inpaint session helpers.
|   |-- io.py                       # Capture, ffprobe, intermediate writers, PrefetchReader.
|   |-- job_worker.py               # Child-process entry for one isolated queue job.
|   |-- karaoke_flow.py             # Karaoke optical-flow grouping helper.
|   |-- language_support.py         # GUI picker scope vs. OCR engine language facts.
|   |-- mask_corrections.py         # Ordered add/subtract mask corrections.
|   |-- mask_free_benchmark.py      # Mask-free removal benchmark harness.
|   |-- matte_interchange.py        # Lossless matte export / import / compose.
|   |-- model_downloads.py          # First-run model download guidance.
|   |-- model_hashes.py             # Vendored SHA-256 hashes and chunked verifier.
|   |-- nle_sidecar.py              # EDL / FCPXML sidecar export.
|   |-- ocr_benchmark.py            # OCR engine recall / precision benchmark.
|   |-- ocr_fix.py                  # Per-language OCR replace lists for exported SRT.
|   |-- ocr_variants.py             # Canonical PaddleOCR model families and aliases.
|   |-- ocr_vlm.py                  # Optional VLM detectors (Florence-2, Qwen2.5-VL).
|   |-- onnx_model_info.py          # ONNX opset audit and Windows ML probe.
|   |-- onnxruntime_cuda.py         # CUDA preload status for ONNX Runtime.
|   |-- opencv_ocr.py               # PP-OCRv6 via OpenCV 5 DNN and the engine contract.
|   |-- output_contract.py          # Frozen per-job output policy.
|   |-- output_quality_preflight.py # Pre-run output quality warnings.
|   |-- paddle_compat.py            # PaddleOCR 2.x / 3.x API compatibility layer.
|   |-- post_restore.py             # Post-inpaint temporal smoothing and burn-in.
|   |-- preprocess.py               # Deinterlacing and keyframe enumeration.
|   |-- presets.py                  # Shared preset library (GUI + CLI).
|   |-- processor.py                # Frame loop plus the legacy re-export / CLI shim.
|   |-- proxy_workflow.py           # Proxy-encode workflow for large files.
|   |-- quality.py                  # PSNR / SSIM / VMAF and temporal metrics.
|   |-- quality_gate.py             # Graduated quality gate with a remediation ladder.
|   |-- reference_corpus.py         # Synthetic reference-clip regression harness.
|   |-- reference_fill.py           # Clean-plate reference fill.
|   |-- region_editing.py           # Region geometry edit / undo primitives.
|   |-- region_keyframes.py         # Interpolated moving-region keyframe tracks.
|   |-- release_staging.py          # Atomic, version-derived release artifact set.
|   |-- release_verification.py     # Local PyInstaller release evidence writer.
|   |-- remote_model_policy.py      # Gate for trust_remote_code / torch.hub.
|   |-- remux.py                    # Soft-subtitle strip / keep remux paths.
|   |-- resume_checkpoint.py        # Crash-resume and pause checkpoints.
|   |-- safe_image.py               # Bounded image reads.
|   |-- security_checks.py          # Runtime safety checks (libpng and OpenCV FFmpeg inventory).
|   |-- segmentation.py             # Optional SAM 2 / MatAnyone / CoTracker adapters.
|   |-- static_logo_benchmark.py    # Static-logo removal benchmark harness.
|   |-- subprocess_policy.py        # Hidden, bounded, cancellable child processes.
|   |-- subtitle_translation.py     # SRT parsing, providers, translated export.
|   |-- support_bundle.py           # Redacted diagnostics zip export.
|   |-- temporal_profile.py         # Mask-aware temporal regression metrics and fixtures.
|   |-- tensorrt_compile.py         # Optional TensorRT engine compilation.
|   |-- track_plan.py               # Reviewable pre-run text track plans.
|   |-- tracking.py                 # Kalman tracking, pHash reuse, karaoke grouping.
|   |-- update_check.py             # Startup version check (opt-in).
|   |-- vapoursynth_bridge.py       # VapourSynth bridge (opt-in).
|   |-- webvtt.py                   # Loss-aware WebVTT parse / translate / serialize.
|   |-- whisper_fallback.py         # Whisper-based timing for OCR-empty speech.
|   `-- work_directory.py           # End-to-end scratch / storage policy.
|   `-- inpainters/
|       |-- __init__.py   # Mode routing and shared inpainter exports.
|       |-- _common.py    # BaseInpainter, feathering, edge-ring match.
|       |-- auto.py       # Per-scene STTN / ProPainter motion routing.
|       |-- external.py   # VSR_EXTERNAL_INPAINTER bridge.
|       |-- lama.py       # ONNX > OpenCV 5 DNN > PyTorch opt-in > cv2.
|       |-- propainter.py # TBE plus LaMa residual refinement.
|       `-- sttn.py       # TBE (Temporal Background Exposure).
|-- scripts/                     # Build, i18n, and doc tooling.
|-- tools/                       # Local probes (smoke, UI scaling).
|-- installer/                   # NSIS installer sources.
|-- tests/                       # Unit, hardening, and GUI suites.
`-- docs/                        # Architecture and corpus guides.
```

<!-- module-map:end -->

### Why this layout (and where new code should land)

- **`gui/app.py`** owns the Tk shell, shared state, settings variables,
  queue model, and the public `VideoSubtitleRemoverApp` surface. The widget
  construction itself lives in the layout mixins (`gui/layout_build.py`,
  `gui/layout_responsive.py`, `gui/layout_helpers.py`, `gui/queue_view.py`,
  `gui/onboarding.py`), which compose onto the app exactly like the controller
  mixins -- put new widget-building code in a mixin, not in `app.py`.
  The default shell is a command-first workbench: one compact command strip
  sits above the preview/inspector split, and the persistent queue is rendered
  as a dense table below it. Advanced controls remain progressively disclosed
  in the inspector rather than competing with the primary workflow.
- **`gui/processing_controller.py`** owns queue processing, pause/stop
  orchestration, per-item backend dispatch, progress/taskbar updates,
  report preparation, and completion notifications.
- **`gui/preview_controller.py`** owns preview placeholders, the selected-time
  timeline, proxy scene planning, live frames, mask review, A/B compare,
  test-cleanup previews, and preview zoom. The
  region editor lives in `gui/region_controller.py` and mask correction in
  `gui/mask_correction_controller.py`; both build into a scrollable dialog
  body from `gui/dialog_layout.py` so they reflow at high text scale.
- **`gui/quality_controller.py`** owns batch-summary dialogs, source-aware
  quality warnings, quality-review worklists, retry-with-suggested-settings,
  and batch report file opening/writing.
- **`gui/support_controller.py`** owns the log panel, support bundle
  export, model-cache import/export, backend-status panel, and About dialog.
- **`gui/widgets.py`** contains all custom controls: `ModernButton`,
  `ModernToggle`, `ModernSlider`, `SegmentedPicker`,
  `ModernProgressBar`, `DragDropFrame`, `QueueItemWidget`, `Toast`,
  `Tooltip`, and themed utility functions.
- **`gui/config.py`** is the single source of truth for `APP_VERSION`,
  the GUI `ProcessingConfig` dataclass, `QueueItem`, settings
  load/save/migrate, and preset import/export.
- **`backend/config.py`** owns the backend `ProcessingConfig`,
  `InpaintMode` enum, coercers, and `normalize_processing_config`.
  Inpainters import `backend.config` directly.
- **`backend/detection_geometry.py`** owns the normalized OCR record. It keeps
  the legacy bounding box beside optional polygon vertices, clips and remaps
  them for the current frame, and rasterizes each shape with local expansion.
- **`backend/processor.py`** preserves the legacy public import surface
  and delegates `python -m backend.processor` to `backend.cli.main`.
- **`backend/detection.py`**, **`backend/tracking.py`**,
  **`backend/io.py`** (capture, ffprobe, and exact rational timing),
  **`backend/quality.py`**, and
  **`backend/inpainters/`** own the focused pipeline pieces.
- **`backend/encoder.py`** probes hardware encoders and selects the
  output codec (H.264 / H.265 / AV1 / VVC).
- **`backend/presets.py`** holds the one place a preset definition is
  allowed to live.
- **`backend/model_hashes.py`** owns vendored weight hashes and the
  chunked SHA-256 verifier.
- **`backend/language_support.py`** owns the distinction between the
  GUI's selectable OCR language codes and broader OCR engine language
  capacity reported in support/backend status.

---

## Pipeline walkthrough

1. **Ingest.** `gui.app.VideoSubtitleRemoverApp._on_files_dropped` ->
   `_add_to_queue` builds `QueueItem` entries, each carrying its own
   `ProcessingConfig` snapshot. Queue capped at 500.
2. **Per-item dispatch.** `_process_queue` walks the queue;
   `_process_item` translates the GUI `ProcessingConfig` to the
   backend `ProcessingConfig` and instantiates a `SubtitleRemover`
   (or reuses a cached one when mode/device/lang match).
3. **Backend constructor.**
   `backend.processor.SubtitleRemover.__init__`:
   - Normalises the config via `normalize_processing_config`.
   - Builds the OCR `SubtitleDetector` (cascade resolution).
   - Picks the inpainter (`STTNInpainter` /
     `LAMAInpainter` / `ProPainterInpainter` / `AutoInpainter`).
   - Probes the matching HW encoder family for `output_codec`.
   - Optional NVML free-VRAM probe scales `sttn_max_load_num`.
4. **Optional preprocessing.** `process_video`:
   - ffprobe `idet` -> `ffmpeg yadif` deinterlace when auto-detected.
   - ffprobe keyframe enumeration when `keyframe_detection`.
   - ffprobe timing fields are retained as integer PTS and duration ticks with
     the stream's rational time base. Missing, repeated, and non-monotonic PTS
     are repaired with warning records, while edit-list starts stay available
     for validation.
5. **Decode.** `_open_capture` either opens a `cv2.VideoCapture` (with
   optional `decode_hw_accel`) or a `_FrameSequenceCapture` for an
   image-directory input. When `prefetch_decode` is on, the cap is
   wrapped in a `_PrefetchReader` daemon worker that feeds a bounded
   queue. Tagged PQ and HLG sources use a `bgr48le` surface. OCR and model
   inputs receive a separate tone-mapped 8-bit proxy, while the high-bit source
   remains attached to the batch for final repair.
6. **Per-frame detect.** Inside the main loop:
   - `pHash` skip + keyframe gating short-circuit when content is
     unchanged.
   - `SubtitleDetector.detect_with_geometry(frame)` calls the active engine
     (RapidOCR / PaddleOCR / Surya / EasyOCR / OpenCV) and keeps polygon
     vertices beside the compatibility boxes. `detect(frame)` remains the
     rectangle API for older callers.
   - `_group_horizontal_line` fuses karaoke syllables.
   - `SubtitleTracker.update_with_geometry` smooths boxes and remaps polygon
     vertices with the tracked translation and scale. `update` remains the
     rectangle API.
   - `categorize` filters chyron vs subtitle when either
     `remove_chyrons` / `remove_subtitles` is off.
   - `_create_mask` produces the binary mask (with dilation). Polygon records
     are filled and expanded independently, so one rotated caption cannot
     widen another caption's mask.
   - `_expand_mask_by_color` extends to dominant-colour pixels.
   - `_accumulate_quality_bbox` widens the union-mask bbox used by
     the ROI quality metric.
7. **Per-batch inpaint.** The current batch of `(frame, mask)` pairs
   is passed to the inpainter chosen above:
   - `STTNInpainter`: `_temporal_background_expose` reconstructs the
     true background from temporally-exposed neighbours.
   - `LAMAInpainter`: ONNX Runtime > OpenCV 5 DNN > PyTorch
     (`simple-lama-inpainting`, only when `VSR_ENABLE_PYTORCH_LAMA=1`)
     > cv2.inpaint four-tier chain.
   - `ProPainterInpainter`: TBE with a higher coverage bar + LaMa
     residual blend (MIT-licensed hybrid, not the ICCV 2023 model).
   - `AutoInpainter`: per-batch routing on the exposure score;
     idle-LaMa is unloaded after `LAMA_IDLE_UNLOAD_AFTER` TBE
     batches.
   All paths terminate in `apply_finishing` (edge-ring colour match
   then feather blend). HDR output converts only the active mask ROI to bounded
   linear light, lifts the proxy result with high-bit boundary detail, and
   reapplies the source PQ or HLG transfer function. Outside-mask pixels stay
   byte-identical to the decoded source surface.
8. **Intermediate write.** `_LosslessIntermediateWriter` pipes raw
   BGR frames through `ffmpeg -c:v ffv1` so the final encode is the
   only lossy step. Falls back to legacy `mp4v` when ffmpeg is
   absent.
9. **Mux + finalise.** `_merge_audio` re-encodes the FFV1 temp into
   the user-visible H.264 / H.265 / AV1 / VVC (H.266) output (HW
   encoder when available, software fallback per `output_codec`).
   Audio path honours:
   - Time-range trim.
   - Multi-track passthrough (`-map 1:a?`).
   - Per-stream loudness normalisation (`-filter_complex` branch).
   - Adaptive `_ffmpeg_subprocess_timeout` scaled to source
     duration.
   - Exact frame-range ticks are converted to FFmpeg boundary strings only at
     the tool boundary, so audio, SRT, EDL, and FCPXML share one clock.
10. **Quality report.** When `quality_report` is on,
    `_compute_quality_report` samples N frames from input and output,
    computes both whole-frame and ROI-cropped PSNR/SSIM (the ROI
    is the union mask bbox), and reopens the final encoded output against
    persisted per-frame masks before calculating mask-local evidence. The
    temporal pass estimates motion from untouched pixels, excludes scene cuts,
    gates on the worst valid pair, and records it with a timestamp and PNG
    overlay. It also measures outside-mask CIELAB drift for SDR or linear-light
    drift for tagged HDR. Optional
    `_write_quality_sheet` renders the side-by-side PNG. The quality gate
    ladder escalates through increase-dilation, temporal-smooth,
    alternate-inpainter, and manual-review. Color drift always remains a
    review signal and never triggers automatic recoloring.

### Release runtime inventory

`backend/release_verification.py` records the external FFmpeg banner,
configuration, compiler line, and configuration hash in `ffmpegRuntime`. It
also records OpenCV wheel provenance and the embedded `avcodec`, `avformat`,
and `avutil` ABI versions in `opencvFfmpeg`. Those ABI values are not treated
as upstream FFmpeg release tags. A release blocks only when a cited advisory
rule maps a component ABI to an affected range. The current OpenCV build has
no such mapping and is reported as `unmapped` without a vulnerability claim.

11. **Batch report.** `backend/batch_report.py` writes
    `vsr-batch-summary.json` and `vsr-batch-summary.md` with
    per-item status, codec/duration data, quality gate results,
    and remediation suggestions.
12. **Progress, preview, cancel.** During every batch:
    - `on_progress(progress, message)` ticks the GUI progress bar
      and Windows taskbar.
    - `on_preview_frame(frame, idx, total)` marshals an
      inpainted frame to the Tk preview pane.
    - `cancel_event` (global) and the per-item
      `cancel_requested` flag each raise `InterruptedError` from
      the progress callback so the batch can stop cleanly.

---

## Configuration data flow

```
[settings.json]
        |        load_settings + _migrate_settings (schema backfill)
        v
[GUI ProcessingConfig]   (single source of truth in gui/config.py)
        |        per QueueItem snapshot via to_dict / from_dict
        v
[QueueItem.config]       (immutable from this point unless re-snapshotted)
        |        _process_item builds the BackendConfig and passes it down
        v
[backend ProcessingConfig]  (backend/config.py)
        |        normalize_processing_config (idempotent, runs on hot-swap)
        v
[runtime: SubtitleRemover.config]
```

- Persistence is dataclass-driven via `dataclasses.fields(self)` so a
  new field lands in settings.json automatically.
- The GUI `_sync_config_from_ui` is the one place a tk variable maps
  to a config field; every new toggle adds a `hasattr` guard here.
- Hot-swap of a cached remover re-runs
  `normalize_processing_config(backend_config)` to defang any NaN/inf
  or out-of-range per-item override.

---

## Adding a new feature: checklist

For a new ProcessingConfig field:
1. Declare the dataclass field with a backend-default value in
   **both** `gui/config.py:ProcessingConfig` and
   `backend/config.py:ProcessingConfig`.
2. Add a coercion entry to `normalize_processing_config` (backend) and
   `ProcessingConfig.normalized()` (GUI) with safe bounds.
3. Pass the field through `_process_item` -> `BackendConfig(...)`.
4. Surface it in the GUI: add a tk variable + an Advanced card widget,
   then sync it in `_sync_config_from_ui`.
5. Surface it on the CLI: add `parser.add_argument(...)` and map it in
   the `config = ProcessingConfig(...)` block.
6. Add a regression test that round-trips the field through to_dict +
   from_dict.
7. Bump `VSR_SETTINGS_FORMAT` only when the new field's semantics
   require a migration -- a backend-default new key does not.

For a new inpainter:
1. Subclass `BaseInpainter` with an `inpaint(frames, masks)` method.
2. Add the mode to `InpaintMode` (both GUI and backend enums).
3. Add a branch to `SubtitleRemover._create_inpainter`.
4. Add the GUI label to `InpaintMode` and `mode_map` in
   `_process_item`.
5. If the inpainter loads weights at runtime, register the
   vendored SHA-256 in `backend/model_hashes.py:KNOWN_WEIGHT_HASHES`
   and call `verify_weight_file` from the loader.

For a new OCR detector:
1. Add a probe + lazy-load block in
   `SubtitleDetector._load_model` (respect the cascade priority).
2. Add a `_detect_<engine>` method that returns a list of
   `(x1, y1, x2, y2)` tuples.
3. Register the engine name in `detect_ai_engines()` for the About
   dialog.
4. If GPL-licensed, gate behind `VSR_ALLOW_GPL` like Surya.

---

## DirectML to Windows ML migration

Microsoft has placed DirectML in maintenance mode (explicit README
banner on github.com/microsoft/DirectML as of 2026). New ONNX Runtime
GPU development for AMD/Intel targets is moving to Windows ML.

Current VSR state:
- AMD/Intel GPU inference uses `onnxruntime-directml==1.24.4`
  (latest published, March 2026). It receives security patches
  but no new features.
- `onnxruntime-windowsml` 1.27.1 is available on PyPI (Python
  3.11-3.14) and provides automatic execution-provider selection.
- `--audit-windows-ml` probes the Windows ML Python path.
- `backend/device_provider.py:windowsml_status()` reports
  whether `onnxruntime-windowsml` is installed (surfaced in the
  support bundle).

Migration prerequisites:
1. Confirm `onnxruntime-windowsml` provides equivalent EP selection
   for the OCR and inpaint ONNX models on AMD/Intel GPUs.
2. Benchmark latency versus the current DirectML path.
3. Update `dependency_profiles/directml.txt` to pin
   `onnxruntime-windowsml` instead of `onnxruntime-directml`.
4. Update `setup.py` to install the new package on AMD/Intel
   hardware.

No urgent action is needed: DirectML continues to receive security
patches and functions correctly. Track `onnxruntime-windowsml`
releases and confirm inference parity before switching the default.

---

## Known trade-offs

- `process_video` is monolithic (~250 lines). Splitting it into
  detect / inpaint / mux phases is on the roadmap but every existing
  call site assumes the current state machine, so the split needs
  careful test coverage first.
- The cached-remover reuse in `_process_item` saves model-load
  time across batch items but means a config change that needs a
  different detector engine triggers a full reload (mode + device +
  lang form the cache key).
- `_PrefetchReader` requires strict ownership: once it wraps a cap,
  the main thread cannot touch the underlying object directly.
  Cleanup goes through `reader.release()` so a mid-batch crash never
  leaks the worker thread.
- The FFV1 intermediate (`_LosslessIntermediateWriter`) needs
  ffmpeg on PATH; falls back to `mp4v` automatically. The fallback
  reverts to v3.12 behaviour (lossy intermediate).

---

## Accessibility support matrix

What is actually tested here, and what is explicitly not supported. Keep this
honest: an untested claim is worse than a documented gap.

| Surface | State | Proof |
|---------|-------|-------|
| Keyboard reachability of major actions | Supported and tested | `tools/ui_scaling_probe.py` asserts every major action is focusable, has non-zero geometry, and is not clipped, across the scale/theme/locale matrix; `tests/test_text_scaling.py` keeps the probe in active collection |
| Text scaling 100-200% | Supported and tested | Same probe at 100/125/150/175/200%; fonts, control heights, and wrap lengths must all scale |
| Dialog reflow and scrolling at high scale | Supported and tested | `gui/dialog_layout.py`; the probe opens the dialogs at 980x720 and 2752x1152 work areas and requires an internal scroll path |
| High-contrast theme | Supported and tested | Probe runs the whole matrix under the high-contrast palette |
| Pseudo-locale (qps-Ploc) string expansion | Supported and tested | `scripts/i18n_catalogs.py`; probe renders pseudo-localised strings |
| RTL mirroring | Supported and tested (pseudo-RTL) | Probe checks theme direction, label justification, and mirrored toggle geometry |
| Accessible names/roles on standard and custom widgets | Supported as an MSAA bridge | `backend/a11y.py` annotates names, roles, values, descriptions, and help text on native widget HWNDs; the active release probes cover metadata and focus behavior |
| **Screen readers / UI Automation on custom controls** | **Live client proof blocked** | The Canvas-based controls expose the MSAA bridge but do not implement native UIA patterns. Narrator/NVDA validation requires an isolated virtual monitor or user session. The current headless release build records the limitation instead of claiming a live reader result |

If you need screen-reader support today, the CLI (`python -m backend.cli`) is
the accessible surface: it is plain text, fully keyboard driven, and every
option in the GUI has a CLI equivalent.
