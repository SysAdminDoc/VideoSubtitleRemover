# Changelog

All notable changes to VideoSubtitleRemover will be documented in this file.

## [Unreleased]

### Security

- **Pin `torch >= 2.10.0`** -- CVE-2026-24747 / CVE-2025-32434 are
  `torch.load` `weights_only` RCEs reachable on PyTorch 2.9.1 and earlier.
  `simple-lama-inpainting` loads weights via `torch.load`, so this is a
  runtime concern. The CUDA + CPU install paths in `setup.py` and the
  GitHub Actions workflow are bumped to `>=2.10.0` / `torchvision>=0.25.0`.
  The torch-directml path stays on torch 2.4.x because no patched
  torch-directml wheel exists yet; `setup.py` now warns the user when that
  path is selected.
- **Pin `opencv-python >= 4.12.0`** -- CVE-2025-53644 is an uninitialised
  pointer in the JPEG reader that can become an arbitrary heap write on a
  crafted file. Bumped in `requirements.txt` and the GHA build matrix.
- **Pin `Pillow >= 11.3.0`** -- CVE-2026-25990 is a PSD loader out-of-bounds
  write. We do not currently open PSDs, but Pillow is on the transitive
  closure and a future image-format feature could expose it; pinning now is
  cheap insurance.

### Migration

- **Settings-schema versioning (`vsr_settings_format`)** -- `settings.json`
  now carries an integer schema version stamped by `to_dict()`. A new
  `_migrate_settings()` shim runs before `from_dict` on load so a future
  field rename can upgrade legacy payloads in place rather than silently
  dropping user state. Settings written by a newer build are honoured as-is
  (we don't downgrade). Format starts at `1`; `_migrate_settings()` learns
  one new case per bump.

### Fixed

- **Shutdown race condition**: `_shutdown_started` is now set only *after* the
  user confirms the "Close while processing?" dialog, preventing a racing
  `_on_processing_complete` callback from destroying the root window while the
  confirmation modal was still open.
- **Duplicate queue-item IDs**: replaced the millisecond-timestamp ID with
  `uuid.uuid4()` so IDs are collision-proof even when many files are added
  simultaneously.
- **`_coerce_int` / `_coerce_float` NaN/inf**: both coercers now reject
  non-finite floats (NaN, ±Inf) and fall back to the specified default, matching
  the stricter guard already present in the backend.
- **`from_dict` pre-sanitisation crash**: `subtitle_area` and `subtitle_areas`
  in `ProcessingConfig.from_dict` now go through `_coerce_rect` /
  `_coerce_rect_list` directly instead of raw `tuple()`/list-comprehension
  conversions that could raise before `.normalized()` ran.
- **pHash double-computation**: the perceptual hash is now computed once per
  detection frame; the value is reused for `last_hash` instead of being
  recomputed immediately after.
- **`_write_srt` unsafe fps guard**: `fps or 30.0` replaced with
  `fps if fps and fps > 1.0 else 30.0` to prevent absurd SRT timestamps from
  a near-zero but non-falsy fps value.
- **`_load_json_config` size guard**: added a 1 MB cap before parsing so an
  accidentally large file cannot be loaded in full before the type check.
- **CLI `KeyboardInterrupt`**: Ctrl-C during a batch or single-file run now
  prints a clean message and exits with code 130 instead of showing a raw
  traceback.
- **`detect_ai_engines()` Surya detection**: broadened from `ImportError` to
  `Exception` so a partially-installed Surya (runtime import errors) no longer
  crashes engine probing.
- **`TextWidgetHandler._append` after-destroy safety**: added a
  `winfo_exists()` guard at the start of `_append` so log records scheduled
  with `after(0, ...)` just before root teardown are silently dropped rather
  than raising `TclError`.
- **`_reveal_output` silent failure**: `os.startfile` errors are now logged as
  warnings instead of being swallowed silently.
- **`_start_elapsed_timer` double-start**: calls `_stop_elapsed_timer()` first
  to cancel any existing tick loop before starting a new one.
- **Headless CI guard**: `test_on_processing_complete_during_shutdown_skips_summary_ui`
  is now skipped automatically on systems without a display.

### UI/UX improvements

- **Workflow step pills**: the header guidance panel now renders three compact
  step pills (Import → Inspect → Run) that highlight the current stage as the
  user moves through the workflow. The pills were wired to `_set_workflow_stage`
  but never built; they are now constructed at startup.
- **Section eyebrow labels**: `_section_title` now renders the `eyebrow`
  parameter as a small-caps meta-label above the section title, adding the
  intended two-level hierarchy (e.g. "WORKSPACE / Import media",
  "PROCESSING / Settings"). Previously the parameter was accepted but silently
  ignored.
- **Settings section title deduplication**: the Processing settings section was
  titled "PROCESSING / Processing". It is now "PROCESSING / Settings".
- **Log badge ordering**: warn/error count badges in the activity-log header now
  appear between the section title and the toggle button, where they belong.
  Previously they were packed after the toggle button, putting them in the wrong
  visual position.
- **Log badge pluralization**: "1 warn" is now "1 warning"; counts > 1 use the
  correct plural form for both warnings and errors.
- **Queue item default message**: new queue items show "Ready to process" instead
  of "Waiting…" for consistency with the `update_item()` fallback text.
- **Status badge padding tokens**: `padx=10, pady=4` on queue item status badges
  replaced with `padx=Theme.S_SM, pady=Theme.S_XS` (8 and 4 respectively) for
  design-system consistency.
- **Slider hint indent**: `padx=(Theme.S_LG + 128, Theme.S_LG)` (magic-number
  approximation) simplified to `padx=(Theme.S_LG, Theme.S_LG)`.
- **Queue empty-state spacing**: `pady=(Theme.S_3XL + 20, …)` magic number
  replaced with `pady=(Theme.S_3XL, …)`.
- **Footer hint text**: shortened from a 13-word instructional sentence to
  "Add files, review a sample frame, then start." — concise and calm.
- **Activity log height**: increased from 5 to 6 text rows for better context
  visibility without dominating the layout.
- **Progress bar height parity**: batch-level progress bar harmonised to
  `height=5` to match item-level bars (was `height=6`).

### Tests

- Added `CoerceHardeningTests`: NaN/inf for `_coerce_int` and `_coerce_float`,
  and non-iterable `subtitle_area` / `subtitle_areas` in `from_dict`.
- Added `BackendWriteSrtTests`: zero fps and near-zero fps fallback to 30.
- Added `LoadJsonConfigTests`: oversized config file is rejected before parsing.

## [v3.12.0] - 2026-04-17

Smart routing, legacy-source preprocessing, and self-testing. Four more
near-term roadmap items; 20 total shipped from the v3.9 plan.

**Smart routing**
- **AUTO inpaint mode** -- new `InpaintMode.AUTO` routes each TBE batch
  between temporal (STTN) and spatial (LaMa) inpainting based on how
  many masked pixels are temporally exposed. Controlled by
  `auto_exposure_threshold` (default 0.55). Lazy-loads LaMa only when
  the routing actually calls for it, so users who never hit a hard
  batch pay nothing.

**Preprocessing**
- **Deinterlace on ingest** -- `ffprobe -vf idet` probe detects
  interlaced sources; `ffmpeg -vf yadif=1` produces a temp
  progressive-scan copy before the main pass. Opt-in manual mode
  plus automatic detection (default on). Fixes comb-artefact-induced
  OCR junk on DV / broadcast rips.
- **Keyframe-driven detection** -- `ffprobe` lists every I-frame in
  the source; OCR runs only at keyframes while Kalman-smoothed
  masks propagate between them. Large speedup for long streams with
  stable subtitles. Gracefully falls back to pHash skip when
  ffprobe is unavailable.

**Quality self-test**
- **PSNR / SSIM report** -- after each run, samples 10 random frames
  (deterministic seed) and compares input vs output. Logs the
  per-run PSNR (dB) and SSIM (0-1). Catches regressions from
  mis-configured dilation / feather / encoder settings. Available
  via config + CLI `--quality-report`.
- Pure-numpy / cv2 SSIM implementation -- no new dependency.

**CLI**
- `--mode auto` is a new choice alongside the three legacy algorithms.
- New flags: `--auto-threshold`, `--deinterlace`,
  `--no-deinterlace-detect`, `--keyframe-detect`, `--quality-report`.

**GUI**
- Algorithm picker grows an "Auto" segment; new description tag.
- Output card: Auto-deinterlace toggle, Keyframe-driven detection
  toggle, PSNR / SSIM report toggle.
- All new settings persist to `settings.json`.
- Version bumped to 3.12.0.

## [v3.11.0] - 2026-04-17

Automation + visibility release. Four more near-term roadmap items
landed, rounding the total shipped from the v3.9 plan to 16.

**CLI**
- **Glob + config-file batch mode** -- `python -m backend.processor
  --pattern "inputs/*.mp4" --out-dir outputs/ --config recipe.json`.
  Users no longer need to shell-loop for bulk jobs. A JSON config
  file overlays fields that aren't exposed as flags
  (kalman_max_age, tbe_scene_cut_threshold, colour_tune_tolerance,
  etc.).
- **Crash-resume checkpointing** -- every completed file writes a
  SHA-256 marker to `%APPDATA%\VideoSubtitleRemoverPro\checkpoints\`.
  Re-running the same batch skips already-finished files. Input
  size/mtime is part of the fingerprint, so re-encoded inputs are
  correctly reprocessed. Disable with `--no-resume`.
- Mutual validation of `--input` / `--output` vs `--pattern` /
  `--out-dir`; clear error messages instead of silent no-ops.

**Workflow**
- **Preset JSON export / import** -- share a tuned recipe as a
  single JSON file. The Profile card grows Export / Import ghost
  buttons next to "Save as...". Import collisions with built-in
  names auto-rename with an "(imported)" suffix; imports carry a
  `vsr_preset_format` version tag for future compatibility.
- **Live processing preview** -- the preview pane now renders the
  latest inpainted frame during processing. Backend emits via an
  `on_preview_frame` callback every Nth frame (default 6); GUI
  marshals to the Tk main loop with a 15 FPS throttle. No separate
  window, no extra flag -- shows automatically whenever a batch is
  running.

**GUI**
- Profile card: Export / Import preset buttons.
- Preview pane: switches to "Live preview" title during
  processing, shows "Frame N/M (X%)" in the meta line.
- Version bumped to 3.11.0 across banner, header, logs.

## [v3.10.0] - 2026-04-17

Detection intelligence + preset workflow. Four more near-term roadmap
items landed. Big win for temporally-noisy footage and stylized text.

**Quality**
- **Kalman box tracking** -- per-frame OCR bounding boxes jitter a few
  pixels even when the rendered text is identical. A constant-velocity
  Kalman filter per subtitle "line" smooths that jitter, absorbs
  single-frame misses (cuts mask flicker), and stabilises the set of
  pixels TBE treats as masked across a batch. Pure numpy + cv2,
  default on.
- **Colour-tuned mask expansion** -- sample the dominant text colour
  inside each detected box (Lab-space two-cluster split), then extend
  the mask to every pixel within a tolerance radius of that colour.
  Catches decorative serifs, drop shadows, strokes, and karaoke
  bloom that the OCR bbox clips. Opt-in toggle.

**Speed**
- **Perceptual-hash adaptive mask reuse** -- instead of fixed
  `frame_skip`, compute a pHash of each frame and skip OCR when the
  Hamming distance from the last detected frame is below a threshold.
  Adapts to scene content: dense detection on motion / cuts, sparse on
  static shots. Default on, threshold 4/64 bits.

**Workflow**
- **Preset library** -- six built-in presets (YouTube default, Anime /
  Animation, Motion-heavy, TikTok vertical, VHS restore, News chyron)
  plus save/load of user-defined presets to
  `%APPDATA%\VideoSubtitleRemoverPro\presets.json`. Preset picker in
  the Profile card with a "Save as..." ghost button. Applying a
  preset updates every toggle in the UI and persists the settings.

**CLI**
- New flags: `--no-kalman`, `--no-phash`, `--phash-distance`,
  `--colour-tune`, `--colour-tolerance`.

**GUI**
- Detection card: Kalman tracking toggle, Adaptive mask reuse
  toggle, Colour-tuned expansion toggle.
- Profile card: Preset combobox + Save as button.
- All new settings persist to `settings.json` and restore on launch.
- Version bumped to 3.10.0 across banner, header, and logs.

## [v3.9.0] - 2026-04-17

Quality + workflow release. Every item here came directly off the v3.9
near-term roadmap; no external model weights were added. Focus is on
making TBE dramatically better on real-world footage (motion, cuts,
gradients) and shipping the workflow features users keep asking for
(SRT export, mask debug, auto-band, multi-region).

**Quality**
- **Scene-cut-aware TBE** -- the TBE batch is now split at histogram
  scene cuts before aggregation. Previously a cut mid-batch polluted
  the temporal median; now each segment is handled independently.
- **Flow-warped TBE** (opt-in) -- Farneback dense optical flow aligns
  every frame in a TBE segment to the middle reference frame before
  aggregating. Directly addresses camera pans / zooms / handheld
  motion. Slower but dramatically cleaner on motion-heavy footage.
  Toggle in Detection card.
- **Edge-ring colour match** -- after inpainting, sample a thin ring
  immediately outside the mask in both original and filled frames,
  compute the mean colour delta, and apply the offset to the filled
  region. Kills the faint colour seam that sometimes appeared on
  gradient backgrounds. Applies to every inpainter path.

**Workflow**
- **SRT sidecar export** -- detected text is transcribed during the
  removal pass and written as an `.srt` next to the output. Cues are
  collapsed when consecutive frames share text (gaps up to 0.5s
  bridged). Works with RapidOCR / PaddleOCR / EasyOCR.
- **Debug mask video** -- optional `.mask.mp4` written alongside the
  output containing the per-frame binary mask. Useful for tuning
  detection threshold / dilation / feather without full re-runs.
- **Auto subtitle-band detection** -- scans the first 30 frames,
  clusters detected text by vertical band, pins the dominant band as
  the `subtitle_area`. Saves a manual region-draw for the 90% common
  case where the subtitle lives in one horizontal strip.
- **Multi-region masks** -- `ProcessingConfig.subtitle_areas` now
  accepts a list of rectangles (top banner + bottom banner + logo).
  When combined with auto detection the rects are unioned with
  per-frame detections.

**Robustness**
- **Adaptive batch sizing** -- on CUDA init, probe free VRAM via
  NVML (`pynvml`) and scale `sttn_max_load_num` to match. Defaults
  to on, clamped to [8, 512]. Prevents OOM on 4K, unlocks headroom
  on 24 GB cards.

**CLI**
- New flags: `--mask-feather`, `--edge-ring`, `--flow-warp`,
  `--no-scene-split`, `--no-tbe`, `--no-adaptive-batch`,
  `--export-srt`, `--export-mask`, `--auto-band`.

**GUI**
- Detection card: Mask feather slider, Colour-match ring slider,
  Auto-band toggle, Flow-warp toggle, Scene-split toggle.
- Output card: Adaptive batch toggle, Export SRT toggle, Export
  mask video toggle.
- All new settings persist to `settings.json` and restore on launch.
- Version bumped to 3.9.0 across banner, header, and logs.

## [v3.8.0] - 2026-04-17

Real video inpainting, faster detection, seamless boundaries. This is the
first release where STTN and ProPainter actually do something meaningfully
different from `cv2.inpaint` -- we keep the STTN / ProPainter names because
they describe the user-facing niche (temporal propagation, motion-robust),
but the implementations are now homegrown and do not require external
model weight downloads.

- **RapidOCR detection (default)**: drops in as the top-priority OCR backend.
  Runs PP-OCR via ONNX Runtime -- roughly 4-5x faster than the paddlepaddle
  build, smaller install, and free of PaddleOCR's well-known memory-leak
  behaviour on long batch runs. Auto-selected when available; otherwise the
  existing PaddleOCR > Surya > EasyOCR > OpenCV chain still applies.
- **Temporal Background Exposure (TBE)**: the STTN inpainter is now a real
  video-inpainting primitive. For every pixel inside a frame's mask it
  looks across the batch for frames where the same pixel is unmasked and
  reconstructs the true background from those exposures (median when the
  batch is small, otherwise mean). Only pixels that are masked in every
  frame fall back to cv2 inpainting. This is the principle behind
  transformer-based video inpainting for sparse occluders (subtitles,
  watermarks, logos) -- the background is literally revealed in adjacent
  frames.
- **Hybrid ProPainter path**: the ProPainter mode now runs TBE with a
  higher coverage bar plus LaMa refinement on the exposed background.
  Produces noticeably smoother results on motion-heavy footage without
  ProPainter's 10+ GB VRAM footprint.
- **Mask edge feathering**: a configurable Gaussian alpha blend
  (`mask_feather_px`, default 4) softens the boundary between original and
  inpainted pixels so there is no visible cut line at the edge of the
  removal region. Applies to every inpainter path (STTN / LAMA /
  ProPainter / fallback).
- **Engine probe**: About dialog and the in-GUI badge now report
  "Temporal BG (TBE)" as an always-available inpainting engine and
  prefer RapidOCR in the detection chain when installed.
- **Docs**: requirements.txt pins the new RapidOCR dependency. CLAUDE.md
  and README.md reflect the new temporal pipeline and detection priority.

## [v3.7.0] - 2026-04-17

Premium polish pass. No behavioral changes; dramatic UX/UI refinement.

- **Design system**: unified typography, spacing, radii, and color tokens on the `Theme` class
- **Refined palette**: tighter tonal ladder (BG_DARK -> BG_SECONDARY -> BG_CARD -> BG_TERTIARY -> BG_RAISED) and more vibrant emerald primary / sky-blue accent
- **Custom widgets**:
  - `ModernToggle` replaces `tk.Checkbutton` -- canvas-rendered checkmark, focus ring, hover, proper disabled state
  - `ModernSlider` replaces `tk.Scale` -- rounded track, emerald fill, prominent thumb, keyboard and wheel support
  - `ModernButton` gains size variants (sm/md/lg), style variants (primary/accent/ghost/secondary/danger/success), icon support, and crisp focus rings
  - `ModernProgressBar` thinner default track (5-6px) with rounded corners
- **Section structure**: `tk.LabelFrame` usage replaced with consistent card pattern (eyebrow + title + content) for Profile, Workflow, STTN Motion, Detection, Output, Video Range
- **Header**: status chips now include a color-coded status dot; tighter vertical rhythm; PRO pill outlined instead of filled
- **Queue section**: illustrated empty state with film-strip icon; count shown as pill; ghost-style row actions; refined progress bar tinting; **selected item now shows a left-edge blue accent stripe**
- **Preview card**: clearer hierarchy with eyebrow + title + meta; **PIL-rendered placeholder illustration** so the card never collapses; refined detection loading state
- **Footer**: status indicator now has a color dot matching the tone (success/warning/error/info)
- **Tooltips**: 380ms hover delay, raised-surface look, subtle 1px border
- **Region selector modal**: cross-cursor, translucent selection fill, two-line hint
- **Log panel**: eyebrow/title header, slim scrollbar, bordered body, visible by default
- **ttk styles**: slimmer scrollbar, better combobox focus/hover/disabled states, popup listbox uses raised surface tone
- **Microcopy**: all button labels and helper lines tightened for a calmer, more confident tone
- **Custom confirm dialog**: `show_confirm()` themed modal replaces the native Windows messagebox for Clear Queue and Close-while-processing; proper title/message/detail hierarchy with Escape/Enter support
- **Toast notifications**: lightweight `Toast` popups anchor to bottom-right for batch completion; stacked so multiple toasts don't overlap
- **Language picker**: shows friendly names ("English (en)", "Japanese (ja)") instead of raw language codes
- **Action buttons**: Start/Stop batch and Open output now ship leading ASCII glyphs for faster visual recognition
- **Batch progress**: now uses full-width bar with a fraction label ("3 of 10 complete") on the left and a percent pill on the right
- **Close confirmation**: asks before quitting if a batch is still running, preventing accidental cancellation
- **Right-click menu on queue items**: themed context menu with Preview, Detect, Open result, Reveal folder, Copy source path, Remove
- **Windows taskbar progress**: Windows 7+ taskbar integration via ITaskbarList3 reflects batch progress on the taskbar icon
- **About dialog**: themed About panel showing version, detected engines, compute summary, and quick links to the log and settings folder
- **Log level badges**: live counts of warnings and errors appear as colored pills in the log panel header and hide when zero
- **Auto-scroll to active item**: the queue now auto-scrolls so the currently processing item is always in view
- **ETA**: batch progress now shows a rolling-average ETA ("2m 14s left") based on completed item timings
- **Throbber animation**: detection preview shows animated pulsing dots in a shimmer placeholder rather than a static label
- **Tweened progress bar**: ModernProgressBar eases to target values instead of jumping, making small backend updates feel continuous
- **Window state persistence**: last window size/position, advanced-panel expanded state, and log-panel visibility are saved to settings.json and restored on next launch (off-screen positions are rejected)
- **Queue summary chips**: queue header now shows a total pill plus green "N done" and red "N failed" chips that auto-hide when zero
- **Batch completion summary modal**: a themed modal now appears when a batch finishes, showing COMPLETED/FAILED/STOPPED counts as large stat pills plus total elapsed time, with an Open-output shortcut
- **Toast fade-out**: toasts now fade to zero alpha over ~200ms before destroying, and remaining toasts restack upward so there are never orphaned gaps
- **Segmented algorithm picker**: the algorithm combobox is replaced by a three-segment Canvas-based radio control with hover/focus/selected states for faster recognition of STTN/LAMA/ProPainter
- **Preview zoom**: double-clicking the preview opens a full-size themed viewer with a compact header (filename + pixel dimensions) and Escape-to-close
- **Active-item pulse**: the currently processing queue card pulses its border and accent stripe between emerald tones so the eye finds the working item instantly
- **Queue filter**: a themed filter input appears above the queue once there are 6+ items; filters by filename or full path, with a Clear button
- **Per-item rename**: right-click menu now offers "Rename output..." for idle items, opening a themed save-as dialog seeded with the current output path
- **First-run onboarding**: on first launch, a themed welcome modal presents three numbered cue cards (Import, Inspect, Run) and persists `onboarding_seen` so it never re-appears
- **Queue sort menu**: a Sort button appears in the queue header once there are 3+ items, opening a themed menu for filename/size/status sorts plus reverse; disabled while a batch is running
- **Version**: bumped to 3.7.0; all version strings and badges aligned

## [v3.6.0] - 2026-03-28

- v3.5.0
- Polish GUI spacing and professional layout
- Simplify drop zone text to just 'Drag & Drop Files Here'
- Changed: Update badge to v3.4.0
- v3.4.0: DPI-safe responsive GUI overhaul
- Changed: Update README badge and CLAUDE.md for v3.3.0
- v3.3.0: Real AI inpainting, multi-engine detection, comprehensive GUI overhaul
- Added: Add files via upload
