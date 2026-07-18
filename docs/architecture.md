# Architecture Map

This document walks the pipeline a frame travels through, names every
module it touches, and points new contributors at the right file for a
given change. Pairs with [ROADMAP.md](../ROADMAP.md) and
[CHANGELOG.md](../CHANGELOG.md).

> Concrete and up to date as of the v3.25.0 audit pass.
> Keep this in sync when modules move.

---

## Module map

```
.
|-- VideoSubtitleRemover.py     # Entry point (thin launcher -> gui.app).
|-- gui/
|   |-- __init__.py             # GUI subpackage.
|   |-- app.py                  # Main Tk shell, shared state, layout, settings.
|   |-- processing_controller.py # Queue worker, pause/stop, reports, notify.
|   |-- preview_controller.py   # Preview pane, A/B compare, region editor.
|   |-- quality_controller.py   # Quality review, retry, batch-report helpers.
|   |-- support_controller.py   # Support bundle, model cache, About panels.
|   |-- widgets.py              # Custom controls: ModernButton, ModernToggle,
|   |                           # ModernSlider, SegmentedPicker, ProgressBar,
|   |                           # DragDropFrame, QueueItemWidget, Tooltip, Toast.
|   |-- theme.py                # Design tokens, colors, spacing, typography.
|   |-- config.py               # APP_VERSION, APP_NAME, QueueItem dataclass,
|   |                           # ProcessingConfig (GUI-side), InpaintMode enum,
|   |                           # settings load/save/migrate, preset import.
|   `-- utils.py                # File helpers, media type checks, formatting.
|-- backend/
|   |-- __init__.py             # Lazy re-exports SubtitleRemover etc.
|   |-- processor.py            # Legacy re-export shim + CLI delegation.
|   |-- config.py               # Backend ProcessingConfig, InpaintMode enum,
|   |                           # coercers, normalize_processing_config.
|   |-- detection.py            # OCR cascade, selectable engines, confidence,
|   |                           # recognized-text/script filtering.
|   |-- tracking.py             # Kalman, pHash, karaoke grouping.
|   |-- io.py                   # Capture, ffprobe, intermediate writers,
|   |                           # PrefetchReader.
|   |-- cli.py                  # argparse entry point.
|   |-- encoder.py              # Output codec probing and HW encoder selection.
|   |-- remux.py                # Soft-subtitle strip/keep remux paths.
|   |-- quality.py              # PSNR/SSIM/VMAF helpers.
|   |-- quality_gate.py         # Graduated quality gate with remediation ladder.
|   |-- batch_report.py         # JSON + Markdown batch summary output.
|   |-- inpainters/             # Built-in STTN/LaMa/ProPainter/AUTO paths.
|   |   |-- __init__.py         # BaseInpainter, mode routing.
|   |   |-- sttn.py             # TBE (Temporal Background Exposure).
|   |   |-- lama.py             # ONNX > OpenCV 5 DNN > PyTorch opt-in > cv2.
|   |   |-- propainter.py       # TBE + LaMa refinement hybrid.
|   |   |-- auto.py             # Per-scene STTN/ProPainter motion routing.
|   |   `-- _common.py          # Feathering, edge-ring color match.
|   |-- inpainters_onnx.py      # ONNX Runtime inpaint session helpers.
|   |-- inpainters_diffusion.py # Opt-in diffusion adapter scaffolds.
|   |-- inpainter_registry.py   # In-process inpainter discovery registry.
|   |-- presets.py              # Shared preset library (GUI + CLI).
|   |-- model_hashes.py         # Vendored SHA-256 hashes + verifier.
|   |-- adapter_manifest.py     # Optional model provenance and hash policy.
|   |-- onnx_model_info.py      # ONNX opset audit and DirectML compat check.
|   |-- remote_model_policy.py  # Gate for trust_remote_code / torch.hub.
|   |-- model_downloads.py      # First-run model download guidance.
|   |-- language_support.py     # GUI picker scope + OCR engine language facts.
|   |-- paddle_compat.py        # PaddleOCR 2.x/3.x API compatibility layer.
|   |-- hdr.py                  # Color metadata preservation.
|   |-- decode_accel.py         # Hardware decode hints (D3D11/VAAPI/MFX).
|   |-- preprocess.py           # Deinterlacing, keyframe enumeration.
|   |-- post_restore.py         # Post-inpaint temporal smoothing.
|   |-- whisper_fallback.py     # Whisper-based timing for OCR-empty speech.
|   |-- karaoke_flow.py         # Karaoke optical-flow grouping helper.
|   |-- proxy_workflow.py       # Proxy-encode workflow for large files.
|   |-- nle_sidecar.py          # EDL/FCPXML sidecar export.
|   |-- tensorrt_compile.py     # Optional TensorRT engine compilation.
|   |-- cache_inventory.py      # --cache-info / --cache-clean.
|   |-- update_check.py         # Startup version check (opt-in).
|   |-- security_checks.py      # Runtime safety checks.
|   |-- crash_reporter.py       # Opt-in GlitchTip reporter (path-scrubbed).
|   |-- support_bundle.py       # Redacted diagnostics zip export.
|   |-- dependency_caps.py      # Dependency version ceiling enforcement.
|   |-- release_verification.py # Local PyInstaller release evidence writer.
|   |-- ocr_vlm.py              # Optional VLM detectors (Florence-2, Qwen2.5).
|   |-- segmentation.py         # Optional SAM 2 / CoTracker adapters.
|   |-- i18n.py                 # Localisation scaffold.
|   |-- a11y.py                 # Accessibility scaffold.
|   `-- vapoursynth_bridge.py   # VapourSynth bridge scaffold.
|-- tests/
|   |-- test_hardening.py       # Core regression + fuzz suite.
|   |-- test_detection_pipeline.py  # OCR cascade unit tests.
|   |-- test_tracking_pipeline.py   # Kalman/pHash tracking tests.
|   |-- test_io_pipeline.py         # Capture, intermediate, prefetch tests.
|   |-- test_gui_smoke.py           # Off-screen GUI construction smoke test.
|   |-- test_gui_*.py               # GUI widget / state / feedback tests.
|   |-- test_release_workflow.py    # Local release evidence validation.
|   |-- test_support_bundle.py      # Support bundle export tests.
|   |-- test_preset_schema.py       # Preset round-trip validation.
|   |-- test_reference_clips.py     # Reference-clip quality baselines.
|   `-- test_*.py                   # Additional focused test modules.
|-- .github/
|   `-- ISSUE_TEMPLATE/         # Bug report + feature request forms.
|-- docs/
|   |-- architecture.md         # This file.
|   |-- edge_case_corpus.md     # Community regression-corpus guide.
|   `-- archive/                # Retired audits and completed checklists.
|-- setup.py                    # First-run venv bootstrap.
|-- scripts/setup_splash.py     # Dependency-free first-run progress splash.
|-- Run_VSR_Pro.bat             # Windows launcher.
|-- Run_VSR_Pro_Debug.bat       # Windows launcher with visible console.
|-- Run_VSR_Pro.ps1             # PowerShell launcher (generated by setup).
|-- build_exe.bat               # Local PyInstaller build + evidence script.
`-- requirements.txt            # Pinned + advisory deps.
```

### Why this layout (and where new code should land)

- **`gui/app.py`** owns the Tk shell, shared state, layout, settings
  variables, queue model, and public `VideoSubtitleRemoverApp` surface.
  The default shell is a command-first workbench: one compact command strip
  sits above the preview/inspector split, and the persistent queue is rendered
  as a dense table below it. Advanced controls remain progressively disclosed
  in the inspector rather than competing with the primary workflow.
- **`gui/processing_controller.py`** owns queue processing, pause/stop
  orchestration, per-item backend dispatch, progress/taskbar updates,
  report preparation, and completion notifications.
- **`gui/preview_controller.py`** owns preview placeholders, live frames,
  mask review, A/B compare, test-cleanup previews, preview zoom, and the
  inline region editor.
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
- **`backend/processor.py`** preserves the legacy public import surface
  and delegates `python -m backend.processor` to `backend.cli.main`.
- **`backend/detection.py`**, **`backend/tracking.py`**,
  **`backend/io.py`**, **`backend/quality.py`**, and
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
5. **Decode.** `_open_capture` either opens a `cv2.VideoCapture` (with
   optional `decode_hw_accel`) or a `_FrameSequenceCapture` for an
   image-directory input. When `prefetch_decode` is on, the cap is
   wrapped in a `_PrefetchReader` daemon worker that feeds a bounded
   queue.
6. **Per-frame detect.** Inside the main loop:
   - `pHash` skip + keyframe gating short-circuit when content is
     unchanged.
   - `SubtitleDetector.detect(frame)` calls the active engine
     (RapidOCR / PaddleOCR / Surya / EasyOCR / OpenCV).
   - `_group_horizontal_line` fuses karaoke syllables.
   - `SubtitleTracker.update` smooths via Kalman.
   - `categorize` filters chyron vs subtitle when either
     `remove_chyrons` / `remove_subtitles` is off.
   - `_create_mask` produces the binary mask (with dilation).
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
   All paths terminate in `_edge_ring_color_correct` then
   `_feather_blend`.
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
10. **Quality report.** When `quality_report` is on,
    `_compute_quality_report` samples N frames from input + output,
    computes both whole-frame and ROI-cropped PSNR/SSIM (the ROI
    is the union mask bbox). Optional `_write_quality_sheet`
    renders the side-by-side PNG. The quality gate ladder escalates
    through increase-dilation, temporal-smooth,
    alternate-inpainter, and manual-review.
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
