# Architecture Map

This document walks the pipeline a frame travels through, names every
module it touches, and points new contributors at the right file for a
given change. Pairs with [ROADMAP.md](../ROADMAP.md) and
[COMPLETED.md](../COMPLETED.md).

> Concrete and up to date as of the post-`eae6672` autonomous pass.
> Keep this in sync when modules move.

---

## Module map

```
.
|-- VideoSubtitleRemover.py     # Tk GUI: window, queue, settings, preview.
|-- backend/
|   |-- __init__.py             # Lazy re-exports SubtitleRemover etc.
|   |-- processor.py            # Legacy re-export shim + CLI delegation.
|   |-- detection.py            # OCR cascade and detector routing.
|   |-- tracking.py             # Kalman, pHash, karaoke grouping.
|   |-- io.py                   # Capture, ffprobe, intermediates.
|   |-- cli.py                  # argparse entry point.
|   |-- quality.py              # Shared quality helpers.
|   |-- inpainters/             # Built-in STTN/LaMa/ProPainter/AUTO paths.
|   |-- presets.py              # Shared preset library (GUI <-> CLI).
|   `-- model_hashes.py         # Vendored SHA-256 hashes + verifier.
|-- tests/
|   `-- test_hardening.py       # Regression + fuzz suite.
|-- .github/workflows/build.yml # Release builder + pip-audit gate.
|-- setup.py                    # First-run venv bootstrap.
|-- Run_VSR_Pro.bat             # Windows launcher.
`-- requirements.txt            # Pinned + advisory deps.
```

### Why this layout (and where new code should land)

- **`VideoSubtitleRemover.py`** owns everything tkinter-shaped: the
  Theme tokens, the custom widgets (ModernButton / ModernToggle /
  ModernSlider / SegmentedPicker / ModernProgressBar), the queue
  model, the preview pane, the region selector, the settings dialog,
  and the progress / batch wiring. Pure GUI. The only "backend"
  reach-in is the BackendConfig adapter inside `_process_item`.
- **`backend/processor.py`** preserves the legacy public import surface and
  delegates `python -m backend.processor` to `backend.cli.main`.
- **`backend/detection.py`**, **`backend/tracking.py`**, **`backend/io.py`**,
  **`backend/quality.py`**, and **`backend/inpainters/`** own the focused
  pipeline pieces that used to live in the processor monolith.
- **`backend/presets.py`** holds the one place a preset definition
  is allowed to live. The GUI imports `BUILTIN_PRESETS` from here;
  the CLI's `--preset NAME` flag resolves through the same table.
  Anything that wants to add a preset edits this file only.
- **`backend/model_hashes.py`** owns vendored weight hashes and the
  chunked SHA-256 verifier. Any future opt-in inpainter / detector
  that fetches weights at runtime adds its hash here.
- **`tests/test_hardening.py`** runs in CI via `python -m unittest
  discover -s tests`. New regressions belong here, grouped by their
  TestCase class.

---

## Pipeline walkthrough

1. **Ingest.** `VideoSubtitleRemoverApp._on_files_dropped` -> `_add_to_queue`
   builds `QueueItem` entries, each carrying its own
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
   - `LAMAInpainter`: per-frame neural fill via
     `simple-lama-inpainting`.
   - `ProPainterInpainter`: TBE with a higher coverage bar + LaMa
     residual blend.
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
   the user-visible H.264 / H.265 / AV1 output (HW encoder when
   available, software fallback per `output_codec`). Audio path
   honours:
   - Time-range trim.
   - Multi-track passthrough (`-map 1:a?`).
   - Per-stream loudness normalisation (`-filter_complex` branch).
   - Adaptive `_ffmpeg_subprocess_timeout` scaled to source
     duration.
10. **Quality report.** When `quality_report` is on,
    `_compute_quality_report` samples N frames from input + output,
    computes both whole-frame and ROI-cropped PSNR/SSIM (the ROI
    is the union mask bbox). Optional `_write_quality_sheet`
    renders the side-by-side PNG.
11. **Progress, preview, cancel.** During every batch:
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
        |        load_settings + _migrate_settings (schema 1 -> 2 backfill)
        v
[GUI ProcessingConfig]   (single source of truth in the running app)
        |        per QueueItem snapshot via to_dict / from_dict
        v
[QueueItem.config]       (immutable from this point unless re-snapshotted)
        |        _process_item builds the BackendConfig and passes it down
        v
[backend ProcessingConfig]
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
   **both** `VideoSubtitleRemover.py:ProcessingConfig` and
   `backend/processor.py:ProcessingConfig`.
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
  detect / inpaint / mux phases is on the roadmap (RFP-L-1) but
  every existing call site assumes the current state machine, so
  the split needs careful test coverage first.
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
