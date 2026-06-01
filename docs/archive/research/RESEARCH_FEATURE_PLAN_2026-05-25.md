# Project Research and Feature Plan

> Archived 2026-06-01. This audit was consolidated into
> [RESEARCH_REPORT.md](../../../RESEARCH_REPORT.md) and
> [ROADMAP.md](../../../ROADMAP.md). It is retained as historical evidence
> for the 2026-05-25 backlog-drain work and is not the active planning source.

> Companion to [ROADMAP.md](../../../ROADMAP.md). The roadmap is already a strong
> ordered backlog; this document does NOT re-list its items. Instead it
> records what an end-to-end audit (2026-05-25, against `e057c41`) found
> that the roadmap does **not** already track: bugs, wiring gaps between
> the GUI and backend, architectural smells, test gaps, and ten or so new
> feature directions that fit the offline-first / single-binary philosophy.
>
> Anchored to verifiable evidence from `backend/processor.py` (3,006 lines)
> and `VideoSubtitleRemover.py` (6,072 lines). Roadmap-tracked items are
> referenced by their roadmap number (e.g. "ROADMAP #29") rather than
> repeated.

---

## Executive Summary

Video Subtitle Remover Pro is a Windows-first PyInstaller desktop app that
removes burned-in subtitles via a 4-tier OCR detector chain
(RapidOCR -> PaddleOCR -> Surya -> EasyOCR -> OpenCV) feeding a 3-mode
inpainter (STTN / LAMA / ProPainter, with v3.12 `AUTO`). The codebase is
unusually mature for a project of this size: TBE, Kalman tracking, scene-
cut splitting, edge-ring colour match, adaptive batch, deinterlace-on-
ingest, keyframe-driven detection, PSNR/SSIM self-test, chyron classifier,
karaoke grouping, frame-sequence ingest, prefetch decoder, JSON log,
loudness norm, and crash-resume are all in `e057c41`. The shipped
`ROADMAP.md` clearly maps the v3.13 -> v5 backlog and is already strong.

The highest-value direction for this cycle is **closing the visible gap
between what the backend can do and what the GUI can drive**. v3.13 added
seven non-trivial CLI features (multi-audio passthrough, HW-decode hint,
loudnorm, karaoke grouping, chyron classifier, prefetch, JSON log,
quality sheet) and not a single one is reachable from the GUI today.
That makes them invisible to the dominant Windows / "drag-and-drop the
.bat" user base. Wiring them in is largely additive work in the GUI
config dataclass + a single Advanced card; the underlying backend
plumbing is already test-covered.

Beyond that, there is one **silent quality regression** in the pipeline:
the intermediate temp file uses `mp4v` lossy encoding, so every video
output is double-lossy-encoded (mp4v -> libx264/NVENC). This is a free
quality lift for every user once the intermediate is moved to raw / lossless.

### Top 10 opportunities, priority order

1. **P0 -- Lossless intermediate codec.** Today every video output is
   re-encoded twice (mp4v intermediate + ffmpeg final). Fix by piping raw
   frames directly to ffmpeg over stdin, or writing FFV1. Free quality
   win; touches one file. ([Item I-1](#i-1))
2. **P0 -- GUI / backend config drift.** 13 backend `ProcessingConfig`
   fields are unreachable from the GUI (loudnorm, decode_hw_accel,
   multi_audio_passthrough, prefetch, input_fps, quality_report_sheet,
   chyron / karaoke). Cached remover hot-swap also bypasses
   `normalize_processing_config`. ([Item B-1, I-2](#b-1))
3. **P0 -- Surya GPL auto-load without warning.** OCR engine cascade picks
   Surya silently after PaddleOCR fails; violates the MIT-clean
   philosophy if a user redistributes their venv. Add explicit opt-in +
   loud GUI banner. ([Item B-2](#b-2))
4. **P0 -- Quality report metric is dominated by unchanged pixels.**
   PSNR/SSIM is computed over the entire frame, so the unchanged 90%
   masks a bad inpaint. Compute the metric inside the union of detected
   masks instead. ([Item B-3](#b-3))
5. **P1 -- Region selector cannot scrub frames or draw multiple rects.**
   Always reads frame 0 (often black on intros); only one rectangle. The
   backend already supports `subtitle_areas`. ([Item F-1, F-2](#f-1))
6. **P1 -- Live OCR-preview at any frame, no full run required.** Today
   the user can only mask-preview frame 0 (right-click on queue item) or
   wait for the full live preview during processing. Add a frame
   scrubber + inpaint-on-demand for N=1 sample. ([Item F-3](#f-3))
7. **P1 -- Per-track loudness + multi-audio loudnorm.** Single-pass
   loudnorm only normalises the first selected audio stream. For Bluray
   rips with 3-5 language tracks this leaves the other tracks unnormalised.
   ([Item B-4](#b-4))
8. **P1 -- AutoInpainter VRAM doesn't shrink back.** Once LaMa is lazily
   loaded for one hard batch, it never unloads. On a long batch where 95%
   of frames go through TBE this is permanently held VRAM. ([Item B-5](#b-5))
9. **P2 -- Lang picker is hardcoded to 12 languages.** Backend OCR
   engines support 80-106. ([Item F-5](#f-5))
10. **P2 -- No reference clip regression harness.** Pure-unit tests run
    against synthetic frames; no test fails when an inpainter regression
    cosmetically breaks output. ([Item T-1](#t-1)) -- this is on the roadmap
    (#54) but unscoped; this plan adds concrete clip selection + commit
    criteria.

---

## Evidence Reviewed

Local files and directories inspected (verbatim file count + line count
from a `wc -l` equivalent against `e057c41`):

- `README.md` (266 lines)
- `CLAUDE.md` (79 lines)
- `AGENTS.md` (10 lines)
- `ROADMAP.md` (988 lines -- read end-to-end)
- `CHANGELOG.md` (1,000+ lines -- read through v3.9.0)
- `requirements.txt` (45 lines)
- `setup.py` (496 lines)
- `Run_VSR_Pro.bat`, `Run_VSR_Pro_Debug.bat`, `build_exe.bat`
- `VideoSubtitleRemover.py` (6,072 lines -- mapped by symbol + read in
  ~400-line chunks across:
   - lines 1-500 (logging, Theme, ProcessingConfig, _coerce_* helpers)
   - lines 644-940 (settings persistence + presets)
   - lines 1000-1170 (utilities, detect_ai_engines, detect_gpu, etc.)
   - lines 2750-3000 (app init / shutdown / sync_config_from_ui)
   - lines 3000-3500 (UI helpers, throbber, preview)
   - lines 3707-4125 (settings UI section -- key for the GUI gap audit)
   - lines 4910-5050 (mode option enablement + onboarding)
   - lines 5427-5570 (region selector)
   - lines 6082-6250 (preview / mask preview)
   - lines 6313-6745 (start_processing / process_queue / process_item)
   )
- `backend/__init__.py`, `backend/processor.py` (3,006 lines -- mapped by
  symbol + read end-to-end through the CLI parser)
- `tests/test_hardening.py` (802 lines)
- `.github/workflows/build.yml`
- Git history: `git log --oneline -40` covering 25 commits from initial
  to `e057c41`. Top of tree commits show backend hardening, chyron
  classifier, frame-sequence input, prefetch reader, JSON log + fuzz
  pass, HW-accel decode + multi-track audio + pip-audit, validate-config
  + skip-existing + loudnorm, CVE pins / settings versioning.

External sources reviewed in the roadmap (not re-listed; see
[ROADMAP.md "Appendix: sources"](../../../ROADMAP.md#appendix-sources)).

Areas that could not be verified without runtime:

- Actual end-to-end output quality on a real clip (no sample assets in
  repo; running PyInstaller / setup.py + a long video was not practical
  for a static audit). All quality claims marked "Needs live validation".
- Whether `cv2.VideoCapture(path, cv2.CAP_FFMPEG, [...HW_ACCELERATION...])`
  actually works on this Windows + OpenCV 4.12 build (depends on the
  shipped FFmpeg bindings).
- Whether ProPainter's hybrid TBE+LaMa blend at 65/35 was tuned against
  representative footage (no documented benchmark).

---

## Current Product Map

### Core workflows (verified by reading the code paths)

1. **Drag-and-drop or folder import** -- `DragDropFrame` widget +
   `_on_files_dropped` (line 5683 in VideoSubtitleRemover.py) routes
   files and folders through `_add_to_queue`. Queue is capped at 500.
2. **Per-item config snapshot** -- every `QueueItem` carries its own
   `ProcessingConfig`. `_apply_current_settings_to_idle_items` (line
   2998) refreshes idle items when the user changes settings; running
   items keep their snapshot.
3. **Preset library** -- `BUILTIN_PRESETS` (line 718) defines six
   recipes; user presets persist in `%APPDATA%\VSR\presets.json`.
   Import / export through `_import_preset_dialog` /
   `_export_preset_dialog`.
4. **Region selection** -- `_open_region_selector` (line 5427) draws on
   frame 0 only, single rectangle.
5. **Mask preview** -- right-click queue item -> "Detect" runs the
   detector on frame 0 in a background thread and overlays red boxes.
6. **Batch processing** -- `_start_processing` -> `_process_queue` ->
   `_process_item` (line 6451). One worker thread; `cancel_event` aborts
   the whole batch.
7. **CLI batch** -- `python -m backend.processor --pattern ... --out-dir
   ...` runs the same pipeline without the GUI. Crash-resume via
   `_checkpoint_key` (line 2839).

### Existing features (full inventory)

Detection chain (backend.processor.SubtitleDetector):
- RapidOCR (1.x tuple API + 2.x object API both handled)
- PaddleOCR PP-OCRv5
- Surya (GPL, opt-in pip install)
- EasyOCR (with PaddleOCR<->EasyOCR lang code mapping)
- OpenCV bright/dark threshold + contour fallback

Inpainting:
- STTNInpainter (TBE-based, falls back to cv2.inpaint for fully-masked
  pixels)
- LAMAInpainter (simple-lama-inpainting, falls back to cv2)
- ProPainterInpainter (TBE with higher coverage bar + LaMa residual at
  65/35 blend)
- AutoInpainter (per-batch routing on exposure_score)
- All paths terminate in `_feather_blend` + optional `_edge_ring_color_correct`

Pre / post pipeline:
- ffprobe `idet` deinterlace probe + auto-yadif (v3.12)
- ffprobe keyframe enumeration -> OCR only at I-frames (v3.12)
- Histogram scene-cut split inside TBE (v3.9)
- Farneback flow-warped TBE (v3.9 opt-in)
- Kalman SubtitleTracker (v3.10)
- pHash adaptive mask reuse (v3.10)
- Colour-tuned mask expansion (v3.10)
- Auto subtitle-band detection (v3.9)
- Multi-region masks (`subtitle_areas`, v3.9)
- Karaoke grouping `_group_horizontal_line` (v3.13)
- Chyron classifier `_KalmanBox.is_chyron` (v3.13)
- Adaptive batch sizing via NVML probe (v3.9)
- Mask dilation (v3.6)
- Mask edge feathering (v3.8)
- SRT sidecar export (v3.9)
- Debug mask video export (v3.9)
- PSNR / SSIM quality report (v3.12)
- Quality sheet PNG (v3.13)

Output:
- Hardware encoder probe (NVENC / QSV / AMF) + libx264 fallback
- CRF quality slider 15-35
- Audio passthrough via ffmpeg
- Multi-track audio passthrough (v3.13 default on)
- EBU R128 loudness normalisation (v3.13 CLI-only)
- Atomic temp-file -> os.replace promotion

Workflow / UX:
- Premium dark theme with design tokens
- Custom widgets (ModernButton / ModernToggle / ModernSlider /
  SegmentedPicker / ModernProgressBar)
- Onboarding modal (3 cue cards, persists `onboarding_seen`)
- Workflow stage pills (Import -> Inspect -> Run)
- Toast notifications with fade-out
- Themed about dialog / confirmation dialogs
- Live processing preview (15 FPS throttle)
- Right-click context menu on queue items
- Queue filter (appears at 6+ items), sort menu (3+ items)
- Stat chips (total / done / failed)
- Empty-state illustrations
- Windows taskbar progress (ITaskbarList3 via ctypes)
- Activity log panel (collapsible, warn/error badges)
- Window geometry / panel state persistence
- Crash handler -> MessageBox + log

CLI:
- All of the above settings via 30+ argparse flags
- `--pattern` / `--out-dir` for glob batch
- `--config` JSON overlay
- `--validate-config` dry run (v3.13)
- `--skip-existing` (v3.13)
- `--no-resume` (bypass checkpoint)
- `--json-log PATH` (v3.13)
- Frame-sequence directory input (v3.13)

Distribution:
- PyInstaller `--onedir --windowed` build
- GitHub Actions release workflow
- `Run_VSR_Pro.bat` first-run bootstrap (creates venv, installs deps)
- `Run_VSR_Pro_Debug.bat` (same but visible console)

### Personas

- **Solo creator / YouTuber.** Removing burned-in subs from
  reaction-video sources; the dominant use case. Wants drag-drop +
  default settings to "just work".
- **Anime / fansub re-flow.** Vertical CJK text, decorative fonts,
  karaoke captions, animation backgrounds. Specialised, but the chyron
  classifier and karaoke grouping target them.
- **VHS / restoration hobbyist.** Noisy SD footage with rolling
  subtitle bands. The "VHS / Low-res restore" preset exists.
- **Power user / scripter.** CLI batch + JSON config + crash-resume.
  Wants reproducibility.
- **One-shot operator.** Drops one file, hits Start, walks away. Lives
  in the AUTO mode + default preset.

Personas absent from current product:
- **Video editor (DaVinci / Premiere round-trip).** No NLE format
  support (XML / EDL / OTIO). ROADMAP #76.
- **Mobile content creator.** No Android / iOS. ROADMAP #93 / #94.
- **Localisation / dubbing studio.** SRT export exists but no
  translation. ROADMAP #85.

### Platforms / distribution

- Windows 10 / 11 desktop (only platform with a build matrix).
- PyInstaller onedir + ZIP through GHA.
- Workflow runs `python -m unittest discover -s tests -v` + pip-audit
  (non-fatal) before bundling.

---

## Feature Inventory (gaps + improvement opportunities)

Only features where I can spot a concrete improvement opportunity beyond
what ROADMAP.md tracks. Roadmap-tracked features without new findings are
omitted.

### Detection chain (SubtitleDetector)

- **User value**: Auto-detect burned-in text across 12 surfaced languages
  (lang picker).
- **Code**: `backend/processor.py` lines 515-746.
- **Maturity**: complete + production-tested.
- **Tests / docs**: `ChyronClassifierTests`, `KaraokeGroupingTests`,
  `BackendHardeningTests` -- but no end-to-end test feeds a frame
  through the cascade. `_fallback_detection` (OpenCV) has zero tests.
- **Improvement opportunities**:
  - Surya GPL auto-load risks license bleed (Item B-2).
  - The cascade is hardcoded by import-availability; users can't pin a
    specific engine without uninstalling everything else.
  - `_merge_boxes` (line 748) is O(n^2 * iters) and runs per-frame on
    every OCR result. Cheap to cap or short-circuit for n <= 2.
  - `_fallback_detection` uses a hardcoded threshold pair (200 / 55)
    that fails on grey or mid-tone subtitles. Should be Otsu.

### Region selector (`_open_region_selector`)

- **User value**: Drag a rect on the first frame to pin a mask region.
- **Code**: `VideoSubtitleRemover.py` lines 5427-5550.
- **Maturity**: partial. Reads only frame 0, only one rectangle, only at
  the source resolution scaled to fit 80%/70% of screen.
- **Improvement opportunities**:
  - Cannot scrub to a non-zero frame (Item F-1).
  - Cannot draw multiple rectangles, though backend
    `subtitle_areas: List[Tuple[int,int,int,int]]` exists (Item F-2).
  - Cannot pan/zoom or use the keyboard arrows for sub-pixel tweaks.
  - No undo / clear-individual-rect.

### Mask preview / detect

- **User value**: Verify detection before running the full batch.
- **Code**: lines 6132-6198 in `_show_preview`.
- **Maturity**: complete but only operates on frame 0.
- **Improvement opportunities**:
  - Frame scrubbing + N-frame sample preview (Item F-3).
  - No way to preview the actual inpaint result before committing the
    batch -- the live preview only fires during processing.

### AUTO inpaint routing (AutoInpainter)

- **User value**: Per-batch TBE vs LaMa routing on exposure score.
- **Code**: lines 1501-1543.
- **Maturity**: complete.
- **Improvement opportunities**:
  - LaMa, once lazily loaded, is never unloaded. Persistent VRAM
    holdover (Item B-5).
  - Single threshold (`auto_exposure_threshold=0.55`) doesn't adapt to
    batch size; smaller batches naturally have lower exposure scores.

### Quality report (`_compute_quality_report`)

- **User value**: PSNR / SSIM self-test.
- **Code**: lines 2116-2247.
- **Maturity**: complete but metric is whole-frame.
- **Improvement opportunities**: see Item B-3.

### Output writer pipeline

- **User value**: Audio-preserving, hardware-accelerated video output.
- **Code**: `process_video` (line 2350), `_get_encode_args` (line 2724),
  `_merge_audio` (line 2763), `_reencode_or_copy` (line 2741).
- **Maturity**: partial. Uses `mp4v` for the intermediate -- see Item
  I-1.
- **Improvement opportunities**:
  - Lossy intermediate is double-encoded (Item I-1).
  - HEVC / AV1 output not surfaced.
  - 10-minute audio merge timeout caps long videos silently.

### Cached remover reuse

- **User value**: Avoid re-loading OCR models for every queue item.
- **Code**: `_process_item` (line 6560).
- **Maturity**: complete but bypasses normalisation on reuse (Item
  B-1).

### Settings persistence + migration

- **User value**: Window geometry, panel state, all knobs persist.
- **Code**: `_migrate_settings` (line 648), `to_dict` /
  `from_dict` / `normalized`.
- **Maturity**: complete + schema versioning shipped in `e057c41`.
- **Improvement opportunities**:
  - `to_dict` enumerates fields manually -- adding a new field requires
    touching `to_dict`, `from_dict`, and `normalized`. A `dataclasses.asdict`
    + ignore-list would make it impossible to forget. Three of the new
    backend fields (loudnorm, decode_hw_accel, multi_audio_passthrough)
    weren't propagated to the GUI dataclass at all because of this
    triple-edit requirement.

### CLI

- **Status**: complete, all backend fields surfaced.
- **Improvement opportunities**: no `--list-engines` / `--print-version`
  / `--probe-encoders` introspection flags. Documented presets are not
  selectable by name via `--preset NAME`.

---

## Competitive and Ecosystem Research

The roadmap already cites:
- YaoFANGUK upstream
- HitPaw / Vmake / Media.io (commercial)
- KrillinAI (subtitle re-flow)
- ProPainter / DiffuEraser / VACE / VideoPainter / COCOCO / EraserDiT /
  FloED (inpainting research)
- LaMa-ONNX / MI-GAN / Real-ESRGAN / SwinIR / SeedVR2 (acceleration +
  post-processing)
- SAM 2 / SAM 3 / MatAnyone2 / CoTracker3 (segmentation)

Adding three competitors / analogues the roadmap does not explicitly call
out:

- **[IOPaint](https://github.com/Sanster/IOPaint)** (formerly Lama
  Cleaner). The strongest open-source local inpainting tool. Supports
  LaMa, MI-GAN, ControlNet, Stable Diffusion, RemBG, Real-ESRGAN. Has a
  proper plugin architecture. **What to learn from it**: the plugin
  loader pattern (each inpainter is a self-contained class registered at
  import); the `--device` / `--model` / `--port` CLI signature; the way
  it streams inpainted frames to the browser via WebSocket. **What to
  avoid**: web UI bloat + Stable Diffusion dependency we don't need.
- **[Bezel](https://github.com/jpr-graphics/sub-cleaner)** (small open
  repo). Single-script subtitle remover. Useful as a baseline for "what
  does the bare-minimum competitor look like" -- our 9,000 lines vs
  their 500 illustrates how far up the value chain we already are.
- **[BiRefNet + RVM](https://github.com/PeterL1n/RobustVideoMatting)**.
  RVM is the standard for "matte a moving object out of video"; would
  be a strong fit if we ever offer "remove the watermark logo, not just
  subtitle" mode. Already mentioned in ROADMAP appendix but not as an
  action item.

The ecosystem winner pattern is clear: **batched local processing,
explicit model picker, plug-and-play model installation**. IOPaint is the
template. ROADMAP #81 (plugin architecture) already commits to this
direction; this plan upgrades it from "speculative v5" to "concrete v4"
with a specific interface (see Item L-3 below).

---

## Highest-Value New Features

Numbered F-* for new feature, B-* for bug, I-* for infrastructure /
architecture improvement, L-* for "larger bet", T-* for testing.

### <a id="i-1"></a>I-1. Lossless intermediate codec

- **Title**: Eliminate the double-lossy-encode generation loss.
- **User problem**: The pipeline writes the inpainted frames to a temp
  `mp4v`-encoded file, then re-encodes to libx264/NVENC. Every output
  is gen-2 lossy compressed before the user gets it. Visible mostly on
  flat colour regions and the inpainted patches themselves -- exactly
  the regions users will be looking at.
- **Evidence**: `backend/processor.py:2463`
  `fourcc = cv2.VideoWriter_fourcc(*'mp4v')` followed by
  `_merge_audio` (line 2763) re-encoding through ffmpeg.
- **Proposed behaviour**: Replace the intermediate writer with one of:
  (a) ffmpeg subprocess in `-c:v rawvideo -pix_fmt bgr24` mode reading
  raw bytes from stdin (fastest, no intermediate file); (b)
  `cv2.VideoWriter` with `FFV1` fourcc (lossless, smaller than raw but
  needs FFV1 codec compiled into the cv2 build); (c)
  `cv2.VideoWriter_fourcc(*'HFYU')` (Huffyuv, lossless and broadly
  compatible). Recommend (a) -- the cleanest path is to merge the
  inpaint-output write with the final encode in one ffmpeg invocation.
- **Implementation areas**: `backend/processor.py` `process_video`
  body, `_reencode_or_copy`, `_merge_audio`. The hand-off from the
  decode/inpaint loop to the encoder becomes a single Popen stdin pipe.
- **Data model / API / UI**: no surface change.
- **Risks**: Raw stdin pipe loses cv2's container-aware retries. If
  ffmpeg errors mid-write, we lose the partial output (currently we
  lose the partial `mp4v` too, so no regression).
- **Verification plan**: Run a 30-frame TBE pass against a fixed
  synthetic source; compute PSNR(input, output) on unmasked pixels --
  it should rise by 1-2 dB after the change. Also confirm bit-for-bit
  identical `output.mp4` when the inpainter is bypassed (mask all zero,
  preserve-audio off).
- **Complexity**: M
- **Priority**: P0

### <a id="b-1"></a>B-1. Wire missing backend fields through the GUI

- **Title**: Surface the seven v3.13 backend features in the GUI.
- **User problem**: A GUI user cannot enable loudness normalisation,
  multi-track passthrough toggling, hardware-decode hints, prefetch
  control, frame-sequence FPS, quality sheet output, or chyron / karaoke
  modes. All exist in the backend, all are CLI-only. ROADMAP
  acknowledges this is "pending" but doesn't list each as an item.
- **Evidence**: `VideoSubtitleRemover.py` `_process_item` (line 6451)
  constructs `BackendConfig(...)` without passing through:
  `loudnorm_target`, `decode_hw_accel`, `multi_audio_passthrough`,
  `prefetch_decode`, `prefetch_queue_size`, `input_fps`,
  `quality_report_sheet`, `remove_subtitles`, `remove_chyrons`,
  `chyron_min_hits`, `karaoke_grouping`, `karaoke_x_gap_px`,
  `karaoke_y_overlap`. GUI `ProcessingConfig.to_dict` (line 336)
  omits the same fields. Also: a cached remover assignment at
  `remover.config = backend_config` (line 6562) bypasses
  `normalize_processing_config`.
- **Proposed behaviour**:
  1. Add the 13 fields to GUI `ProcessingConfig` dataclass.
  2. Persist them in `to_dict` / `from_dict` / `normalized`.
  3. Pass them through to `BackendConfig` in `_process_item`.
  4. After hot-swap, call `normalize_processing_config(remover.config)`.
  5. New "Audio" Advanced card with the loudnorm dropdown + multi-track
     toggle.
  6. New "Performance" Advanced card with decode-accel dropdown +
     prefetch toggle + queue-size slider + input-fps slider (hidden
     unless input is a directory).
  7. New "Editorial" Advanced card with chyron / karaoke / quality-sheet
     toggles.
  8. Replace the manual `to_dict` enumeration with `dataclasses.asdict`
     + an `_excluded_fields` set so new fields are persisted by default.
- **Implementation areas**: `VideoSubtitleRemover.py` dataclass (~50
  lines), settings persistence (~30 lines), `_process_item` (~30
  lines), `_build_settings_section` (~150 lines).
- **Data model / API / UI**: schema version bump from 1 -> 2 (with the
  default-add migration: existing settings read in with all new fields
  at their backend defaults).
- **Risks**: Settings file size grows by ~13 keys; well under 1 MB cap.
  Two `to_dict` paths (one GUI / one backend) drifting is the existing
  problem this fixes.
- **Verification plan**: With a fresh user profile, open the GUI,
  toggle each new control, close, reopen -- every toggle restored.
  Then run `python -m backend.processor --validate-config --config <the
  saved settings.json>` after a `to_dict` -> JSON dump -- resolved
  config must match.
- **Complexity**: M
- **Priority**: P0

### <a id="i-2"></a>I-2. Normalise on hot-swap

- **Title**: Cached-remover hot-swap must re-normalise the config.
- **User problem**: Same root cause as B-1 in a different surface. If a
  user changes per-file region or device between queue items and the
  remover gets reused (same mode/device/lang cache key), the new
  `backend_config` skips `normalize_processing_config` and could carry
  unsafe values (NaN floats, out-of-range ints) into the pipeline. The
  fuzz tests proved the normaliser handles every payload safely; the
  hot-swap bypasses it.
- **Evidence**: `_process_item` line 6562
  `remover.config = backend_config` directly assigns. The constructor
  path (line 1911) does call `normalize_processing_config`; hot-swap
  does not.
- **Proposed behaviour**: Replace the direct assignment with
  `remover.config = normalize_processing_config(backend_config)`.
- **Risks**: None. The normaliser is idempotent.
- **Complexity**: S
- **Priority**: P0

### <a id="b-2"></a>B-2. Surya GPL silent auto-load

- **Title**: Surya is auto-selected even though it's GPL.
- **User problem**: VSR is MIT. If PaddleOCR isn't installed but Surya
  is, the cascade silently uses Surya at runtime. A user redistributing
  their venv-bundled binary unknowingly distributes a GPL-derived
  product. ROADMAP says Surya is "opt-in" but the loader doesn't
  enforce it.
- **Evidence**: `backend/processor.py:572-582` `_load_model` falls
  through to Surya with no warning. `detect_ai_engines` (GUI line
  1017) lists it as a normal detection option. `requirements.txt:37`
  comments Surya out but if the user pip-installs it manually, the
  loader picks it up automatically.
- **Proposed behaviour**:
  1. Add `VSR_ALLOW_GPL=1` env var or `--allow-gpl` CLI flag.
  2. Refuse Surya in `_load_model` unless the gate is set; log a clear
     warning that names it as GPL-licensed and points to the env var.
  3. In `detect_ai_engines`, mark Surya as "Surya (GPL -- opt-in)".
  4. About dialog displays the active engine and its license.
- **Implementation areas**: `_load_model` (5 lines), `detect_ai_engines`
  (5 lines), About dialog (10 lines).
- **Verification plan**: With Surya installed and the gate unset,
  `SubtitleDetector(device="cpu").detect(frame)._engine_name` is one
  of `RapidOCR | PaddleOCR | EasyOCR | OpenCV fallback` -- never Surya.
  With the gate set, it can be Surya.
- **Complexity**: S
- **Priority**: P0

### <a id="b-3"></a>B-3. Quality report measures unchanged pixels

- **Title**: PSNR / SSIM is dominated by the 80-95% of frame that was
  never masked.
- **User problem**: A great PSNR / SSIM score is reported even when the
  inpainted region looks awful, because most of the frame is unchanged.
  The metric was supposed to catch regressions; today it can't.
- **Evidence**: `_compute_quality_report` (line 2116) reads whole
  frames from input / output and computes PSNR / SSIM over the entire
  frame. The masked region is typically <20% of pixels, so its
  contribution is averaged out.
- **Proposed behaviour**: During processing, accumulate a per-frame
  union mask (the same one already passed to the inpainter). On metric
  computation, crop to the dilated bbox of the cumulative mask plus a
  16-pixel ring; compute PSNR / SSIM there. Surface the unchanged
  whole-frame metric too, so users can see both ("inpaint quality 32 dB
  / frame quality 48 dB").
- **Implementation areas**: `_compute_quality_report`, plus a
  cumulative-mask member on `SubtitleRemover` populated inside the main
  loop.
- **Risks**: For videos with very small masks (e.g. corner logo), the
  cropped region may be too small for a stable SSIM. Fall back to
  whole-frame if the cropped area is < 64 x 64.
- **Verification plan**: Synthesize a test where the inpainted region
  is deliberately corrupted (e.g. replace with random noise). The
  current whole-frame SSIM stays >0.9; the new masked-region SSIM should
  drop below 0.5. Add a regression test that asserts the gap.
- **Complexity**: M
- **Priority**: P0

### <a id="b-4"></a>B-4. Multi-track loudnorm

- **Title**: Loudness normalisation only normalises the first selected
  audio stream.
- **User problem**: Bluray / DVD rips with 3-5 audio tracks: only the
  first track is normalised. The other tracks ship through unchanged.
  For a user processing a film with a 5.1 main track + 2.0 commentary
  + stereo Spanish dub, the commentary and dub will be 10 dB quieter
  than expected.
- **Evidence**: `_merge_audio` (line 2763)
  `cmd += ['-af', f'loudnorm=I={target}:TP=-1.5:LRA=11']` is a single
  filter on the implicit stream-0. ffmpeg's `-filter_complex` is needed
  to apply per-stream.
- **Proposed behaviour**: Build a `-filter_complex` with one loudnorm
  branch per audio stream, label each output, then `-map "[a0]" -map
  "[a1]"` ... Pre-pass `ffprobe -select_streams a` to count audio
  streams. If only one stream, keep the simple single-pass path.
- **Implementation areas**: `_merge_audio` (~30 lines + a helper).
- **Risks**: Single-pass loudnorm is approximate; two-pass would be
  broadcast-grade but adds 2x audio decode time. Defer two-pass to a
  follow-on.
- **Verification plan**: ffprobe the output -- track count must equal
  input track count; `ebur128` filter applied to each output track
  should report integrated loudness within 1 LU of target.
- **Complexity**: M
- **Priority**: P1

### <a id="b-5"></a>B-5. AutoInpainter does not unload LaMa

- **Title**: Lazy-loaded LaMa stays resident for the rest of the
  process.
- **User problem**: On a 4-hour video where one early hard batch (low
  exposure) routes to LaMa, LaMa is loaded into VRAM and held for the
  remaining ~4 hours of TBE-friendly content. Wastes ~1.5 GB VRAM
  permanently.
- **Evidence**: `AutoInpainter._ensure_lama` (line 1515) lazy-loads;
  no `_release_lama`. The `LAMAInpainter` instance is kept on
  `self._lama` forever.
- **Proposed behaviour**: Track consecutive-TBE-batch count; if it
  exceeds N=50, release LaMa and clear `_lama = None`. If a later hard
  batch arrives, re-load. This trades a bit of re-load cost for
  reclaimed VRAM. Add a `lama_unload_after_n_tbe` config knob (default
  50; 0 disables).
- **Risks**: Re-load cost (~3 s on CPU, ~1 s on GPU). The threshold
  default is conservative; for typical short clips the unload never
  triggers.
- **Verification plan**: Smoke-test on a synthetic batch list:
  alternating hard / easy batches with N=2 unload threshold. Assert
  `gc.collect(); torch.cuda.memory_allocated()` drops by the LaMa
  weight size after each unload.
- **Complexity**: M
- **Priority**: P1

### <a id="f-1"></a>F-1. Region selector frame scrubbing

- **Title**: Allow drawing the region on any frame, not just frame 0.
- **User problem**: Many sources have a black intro card or a 5-second
  splash where no subtitle exists. The region selector loads frame 0
  and forces the user to draw on a useless frame.
- **Evidence**: `_open_region_selector` (line 5447) only reads the
  first frame via `cap.read()`.
- **Proposed behaviour**: Add a frame slider below the region image.
  Dragging seeks via `cap.set(CAP_PROP_POS_FRAMES)`. Show timecode.
  Slider range: 0 to `frame_count - 1`.
- **Implementation areas**: `_open_region_selector` (+ 40 lines, no
  new file).
- **Risks**: Variable-bitrate sources have inaccurate seeks. Snap to
  the nearest keyframe.
- **Verification plan**: Open the selector on a 1-hour clip, drag to
  ~30 minutes, verify the displayed frame matches `ffplay -ss 30:00`.
- **Complexity**: M
- **Priority**: P1

### <a id="f-2"></a>F-2. Multi-rectangle region drawing

- **Title**: Surface the existing `subtitle_areas` (multi-region) field
  in the GUI.
- **User problem**: Some sources have multiple persistent text bands
  (top + bottom, or chyron + lower-third). Backend already supports
  this; GUI doesn't.
- **Evidence**: `ProcessingConfig.subtitle_areas: Optional[List[...]]`
  declared at backend line 132 and GUI line 302; `_create_mask` (line
  2040) accepts a list of boxes. But `_open_region_selector` (line
  5427) only writes `self.config.subtitle_area` (singular).
- **Proposed behaviour**: Shift-click in the region selector adds a
  new rect to the list; existing single-rect drag still sets the
  primary. List visible in a side panel; click to delete. Persist as
  `subtitle_areas`.
- **Implementation areas**: `_open_region_selector` (~80 lines), new
  list widget.
- **Risks**: UX clutter on the selector; keep it behind a single
  "Add another region" toggle. Single-rect drag stays the default.
- **Verification plan**: Define top + bottom rects, run on a chyron-
  rich clip, confirm both bands are masked in the output.
- **Complexity**: M
- **Priority**: P1

### <a id="f-3"></a>F-3. Inpaint-preview on a sample frame

- **Title**: "Show me what 1 frame would look like after cleanup,
  before I commit the batch."
- **User problem**: The mask preview shows OCR boxes but not the
  result. The live preview during processing only fires once the batch
  starts. Users have no way to A/B their settings without a full run.
- **Evidence**: `_show_preview` (line 6082) only renders the source
  frame or the before/after of an already-completed item.
- **Proposed behaviour**: Add a "Preview cleanup" button that runs the
  full detect+inpaint pipeline on the currently selected frame (re-uses
  cached detector + inpainter) and renders the result inline. For TBE,
  also fetch the N=5 frames around the current frame so the temporal
  signal works.
- **Implementation areas**: `_show_preview` (+ 60 lines), reuse
  `SubtitleRemover.inpainter`.
- **Risks**: Pulls the full inpainter cost for one frame; 1-3 s on GPU
  for LaMa, faster for TBE. Use a background thread with the existing
  throbber.
- **Verification plan**: Click "Preview cleanup" -- the preview pane
  updates within 5 seconds; the rendered frame matches the first frame
  of the full output.
- **Complexity**: M
- **Priority**: P1

### <a id="f-5"></a>F-5. Open up the language picker

- **Title**: Replace the hardcoded 12-language dropdown with a
  searchable picker that respects engine availability.
- **User problem**: PaddleOCR ships 106 languages, RapidOCR ships
  100+, Surya 90+. We expose 12. Users with Thai / Vietnamese /
  Greek / Polish / etc. footage hit a wall.
- **Evidence**: `VideoSubtitleRemover.py:3806-3819` hardcodes 12
  `(code, friendly_name)` pairs.
- **Proposed behaviour**: Pull the full language list from the active
  engine at startup (RapidOCR / PaddleOCR / Surya each expose a list),
  union them, deduplicate, sort by name. Use a search-as-you-type
  combobox. Map to EasyOCR codes only when EasyOCR is the active
  engine.
- **Implementation areas**: New `_load_supported_languages()` helper;
  lang combobox swap in `_build_settings_section`.
- **Risks**: Engine-specific lang codes diverge (already partly handled
  in `_load_model`'s `easyocr_lang_map`). Need to fall back gracefully
  when a chosen lang isn't supported by the current engine.
- **Complexity**: M
- **Priority**: P2

### <a id="f-6"></a>F-6. Long-video audio merge timeout escalation

- **Title**: Make the 10-minute ffmpeg audio merge timeout adaptive.
- **User problem**: For >1-hour videos with hardware-encoder
  re-mux + loudnorm + multi-track, the merge can exceed 600 s. The code
  silently falls back to "copy video without audio" and the user gets
  a silent output.
- **Evidence**: `_merge_audio` (line 2763)
  `subprocess.run(..., timeout=600)` -- hardcoded.
- **Proposed behaviour**: Compute timeout from `duration * factor +
  base` where factor=5 and base=120. For an 8-hour film, that's 8 *
  3600 * 5 + 120 = 144,120 s. Cap at 24 h. Stream ffmpeg stderr to the
  log panel via line-buffered Popen so users see progress instead of
  "stuck" UI.
- **Implementation areas**: `_merge_audio`, `_reencode_or_copy`,
  `_deinterlace_to_temp` (also uses `timeout=600`).
- **Risks**: A runaway ffmpeg now runs for hours before cancellation.
  Mitigation: respect the `cancel_event` between Popen polls.
- **Complexity**: M
- **Priority**: P1

### <a id="f-7"></a>F-7. Per-item cancellation

- **Title**: Let users cancel one item without aborting the batch.
- **User problem**: Today's `cancel_event` is process-wide. If item 5
  of 50 hits a bad source and locks up, the user's only option is
  cancel everything.
- **Evidence**: `_stop_processing` (line 6367) -> `cancel_event.set()`
  -> `_process_item` raises `InterruptedError` -> all remaining items
  marked CANCELLED.
- **Proposed behaviour**: Replace the single `Event` with a
  per-`QueueItem` `cancel_event`. The batch worker sets the current
  item's event when the user clicks "Cancel item" on a row. The shared
  `cancel_event` stays for "stop batch".
- **Implementation areas**: `QueueItem` dataclass, `_process_item`
  callbacks, `_stop_processing`, new context-menu entry on
  `QueueItemWidget`.
- **Risks**: Subtle threading -- the worker calls into the backend
  which polls only one event. Need to check both events at every poll
  site.
- **Complexity**: M
- **Priority**: P2

### <a id="f-8"></a>F-8. HEVC / AV1 output

- **Title**: Surface an output codec dropdown.
- **User problem**: All output is H.264 today, even when the input was
  4K HDR HEVC. Users have to re-encode to HEVC themselves to keep the
  bitrate manageable.
- **Evidence**: `_get_encode_args` (line 2724) hardcodes `h264_nvenc`,
  `h264_qsv`, `h264_amf`, `libx264`.
- **Proposed behaviour**: Add a `output_codec` config field with
  choices `h264 / h265 / av1 / prores`. `_get_encode_args` resolves to
  the corresponding hw / sw encoder. Default stays `h264` for
  compatibility.
- **Implementation areas**: `ProcessingConfig`, `_get_encode_args`,
  CLI `--codec`, GUI dropdown.
- **Risks**: HEVC + audio mux may need a different container; stick
  with `.mp4` (HEVC is fine in mp4; AV1 in mp4 needs a recent FFmpeg).
- **Complexity**: M
- **Priority**: P2

### <a id="f-9"></a>F-9. ETA estimate before the batch starts

- **Title**: Pre-run time estimate.
- **User problem**: Today the only ETA appears mid-batch from the
  rolling-average of completed items. A user starting a 50-file batch
  doesn't know if it's 30 minutes or 30 hours.
- **Evidence**: `_compute_eta` (line 6684) uses only
  `self._batch_times`, which is empty at start.
- **Proposed behaviour**: On batch start, sample 30 frames from the
  first queued video, run a single-frame inpaint pass, measure
  wall-time, multiply by total frames * file count. Surface as
  "Estimated runtime: ~2 h 15 min (probe-based)".
- **Implementation areas**: `_start_processing`, `_compute_eta`.
- **Risks**: Probe adds 5-15 s to startup. Worth it for >1-min batches.
- **Complexity**: M
- **Priority**: P2

### <a id="f-10"></a>F-10. `--preset NAME` CLI flag

- **Title**: Apply a built-in or user preset from the CLI.
- **User problem**: CLI users currently have to manually translate a
  preset name to its individual flags or export the preset to a JSON
  file and pass `--config`.
- **Evidence**: Preset library at `BUILTIN_PRESETS` (GUI line 718) +
  `_load_user_presets` (line 797). `main()` in `backend/processor.py`
  has no `--preset` flag.
- **Proposed behaviour**: Add `--preset NAME`. On parse, load the
  preset's `fields` dict, then overlay CLI flags on top (CLI wins).
- **Implementation areas**: `backend/processor.py` `main()` (~20
  lines). Preset library needs to live in a shared module (today it's
  GUI-only); move `BUILTIN_PRESETS` to `backend/presets.py`.
- **Risks**: Built-in presets reference `mode` as a string ("STTN" /
  "Anime / Animation") -- need to coerce to the backend enum at apply
  time.
- **Complexity**: M
- **Priority**: P2

---

## Existing Feature Improvements

### EI-1. `_fallback_detection` uses fixed bright/dark thresholds

- **Current**: `cv2.threshold(gray, 200/55, 255, ...)` against frame
  luma.
- **Problem**: Grey or mid-tone subtitles (e.g. semi-transparent banner
  text) fall outside both thresholds; the OpenCV fallback misses them.
- **Recommended**: Use Otsu (`cv2.THRESH_OTSU`) to pick the bright
  threshold adaptively. Add a second adaptive threshold for dark text.
- **Locations**: `backend/processor.py:722-746`.
- **Backcompat**: None -- this is the last-resort fallback.
- **Verification**: Create a synthetic grey-on-grey frame; assert the
  fallback finds the text (`len(boxes) > 0`).
- **Complexity**: S
- **Priority**: P2

### EI-2. `_compute_quality_report` opens the output via HW-accel hint

- **Current**: `cap_out = _open_capture(output_path, self.config.decode_hw_accel)`.
- **Problem**: HW-decode on the output (which is the just-written
  H.264 mp4) is unnecessary and may fall back inconsistently. The
  metric sample is 10 frames; software decode is fine.
- **Recommended**: Pass `"off"` for `cap_out`. Keep `cap_in` honouring
  the user's hint.
- **Locations**: `backend/processor.py:2138`.
- **Complexity**: S
- **Priority**: P2

### EI-3. Detection threshold slider scale

- **Current**: Slider is in percent (10-90) and stored as
  `_detection_threshold_pct`, divided by 100 at sync time.
- **Problem**: The threshold is a confidence value, not a percent.
  Users see "30%" and think it means "30% sensitivity" when it actually
  means "accept boxes with >=0.3 model confidence". The mapping is
  inverted from the user's intuition (lower confidence = more boxes
  detected = "more sensitive").
- **Recommended**: Rename to "Detection sensitivity"; invert so high
  sensitivity = low confidence threshold. Or display as a 1-9 step
  with "more sensitive <-> stricter" copy.
- **Locations**: `VideoSubtitleRemover.py:3940-3943`, sync at
  `_sync_config_from_ui:2953-2955`.
- **Backcompat**: Preserve the underlying float in settings; only the
  UI label / direction changes.
- **Complexity**: S
- **Priority**: P3

### EI-4. Live preview throttle conflicts with 4K

- **Current**: 15 FPS throttle, BGR->RGB->PIL.thumbnail->PhotoImage
  every Nth frame.
- **Problem**: For 4K (3840x2160), the resize + PhotoImage step on the
  Tk thread blocks for ~50 ms. At 15 FPS that's 75% CPU time on the
  main thread.
- **Recommended**: Down-sample on the backend thread (already done in
  `on_preview_frame` -- max_w=520, max_h=320) BEFORE marshalling to Tk.
  Move the PIL conversion off the Tk thread.
- **Locations**: `VideoSubtitleRemover.py:6591-6615`,
  `_push_live_preview:3196`.
- **Complexity**: S
- **Priority**: P3

### EI-5. Settings persistence drift via manual to_dict

- **Current**: `ProcessingConfig.to_dict` (line 336) manually
  enumerates every field. Three v3.13 backend fields are missing here
  (`loudnorm_target`, etc.).
- **Recommended**: Replace with
  `{f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name not in _NOT_PERSISTED}`.
  Handle the enum (`mode.value`) and the rect-tuple specially. Same
  for `from_dict` -- use a single `_excluded_fields` set.
- **Locations**: lines 336-388, 443-495.
- **Complexity**: M
- **Priority**: P2

### EI-6. `_choose_available_output_path` race

- **Current**: Inside a batch loop, the GUI uses
  `_make_unique_output_path` (line 5598) which doesn't reserve the
  output. Two queue items added at nearly the same time could both
  pick the same `_no_sub.mp4`.
- **Recommended**: The backend CLI batch path already uses a `reserved`
  set (`_choose_available_output_path` in `backend/processor.py`); GUI
  should do the same with the queue.
- **Locations**: `_make_unique_output_path:5598`,
  `_suggest_output_path:5609`. Today the GUI does use the
  `_occupied_output_paths` set but only at suggest-time; if a user
  manually edits one item's output path to collide with another, no
  guard.
- **Complexity**: S
- **Priority**: P3

### EI-7. Preset description text is sometimes wrong

- **Current**: `BUILTIN_PRESETS["TikTok / Vertical short"]` (line 759)
  description: "9:16 short-form with bold burned-in captions" + sets
  `mask_dilate_px=14` + `auto_band=True`. But TikTok captions are
  almost always centred / top, not bottom -- `auto_band` will lock to
  whatever it sees most, which on TikTok is variable. Worth a
  follow-up A/B test with real TikTok rips.
- **Locations**: `BUILTIN_PRESETS:759`.
- **Complexity**: S (test + tune)
- **Priority**: P3

---

## Reliability, Security, Privacy, and Data Safety

### Confirmed (Verified)

- **Atomic file writes**: outputs go through
  `_allocate_temp_output_path` -> `os.replace`. No partial files on
  disk. Crash-safe.
- **Checkpoint store**: SHA-256 keyed on input + output + size + mtime.
  Resists silent re-runs of edited sources.
- **CVE pinning**: `torch>=2.10.0`, `Pillow>=11.3.0`,
  `opencv-python>=4.12.0` (verified in requirements.txt).
- **pip-audit in CI** (non-fatal). Tracked in roadmap.
- **NaN/inf guards in coercers** + 1500-iteration fuzz pass against
  both GUI and backend normalisers (verified in
  `ConfigFuzzTests:692-783`).
- **JSON config 1 MB cap** (verified in `_load_json_config:2865`).
- **No outbound network calls** in the runtime path (model weights are
  pulled by `simple-lama-inpainting` at first use via its own
  downloader; no analytics, no telemetry).

### Risks found (Likely / Needs live validation)

- **Surya GPL silent load** -- see Item B-2. (Verified.)
- **Double lossy encode** -- see Item I-1. (Verified -- direct grep
  of `VideoWriter_fourcc(*'mp4v')`.)
- **Loudnorm single-stream only** -- see Item B-4. (Verified.)
- **`simple-lama-inpainting` torch.load weight loading**: requirements
  pins `torch>=2.10.0` which patches the RCE; valid only as long as the
  pin holds. The DirectML install path pins `torch==2.4.1` (no
  patched torch-directml wheel). `setup.py` does warn the user but
  doesn't block install. Recommended: harder gate -- print a banner at
  app startup on the DirectML path that the CVE is unpatched on this
  build.
- **PyInstaller pickle exposure**: not specific to VSR, but the
  PyInstaller bundle includes pickled imports. Any future feature that
  reads pickled data from untrusted sources is a CVE waiting to happen.
  Recommended: add a CONTRIBUTING.md rule "never load pickle from
  user input".
- **Region selector temp file leakage**: `_open_region_selector`
  doesn't write temp files. Safe.
- **Settings file world-readable**: `%APPDATA%\VSR\settings.json` is
  written with default OS permissions. Contains paths to user media
  but no credentials. Acceptable.

### Missing guardrails

- **No max video duration / file size**: A pathological 50 GB input
  will consume all available temp space (decode + intermediate +
  output). Recommended: pre-flight check on input size; warn at >20 GB.
- **No max output dimension**: A 16K-by-9K input would consume all
  VRAM via TBE's `np.stack`. Adaptive batch sizing helps but won't
  catch the single-frame case.
- **No model weight hash verification**: ROADMAP #49 already on the
  list; called out here as concrete: `simple-lama-inpainting` fetches
  weights to `~/.cache/torch/hub/checkpoints/`; we don't verify the
  hash on first use.

### Recovery / rollback needs

- **Crash during processing**: temp file cleanup is in `finally`
  blocks; verified.
- **Crash during settings save**: `_write_json_atomic` does
  temp-write + rename. Verified.
- **No undo on settings reset**: clicking "Reset region" / changing
  preset has no undo. Acceptable for a desktop tool.

### Logging / diagnostics

- **Active log channels**: rotating file 5 MB / 2 backups (line 50),
  text widget in the GUI, optional JSON-line via `--json-log`.
- **Missing**: no `--log-level DEBUG` flag (currently hardcoded INFO).
  Should be a CLI arg + GUI setting.

---

## UX, Accessibility, and Trust

### Onboarding

- First-run welcome modal with 3 cue cards (Import / Inspect / Run) --
  verified. Good.
- The modal does not check whether FFmpeg is installed -- which is the
  single most common silent-failure on a fresh Windows box. Add a
  fourth card: "FFmpeg detected: yes/no" with a one-click `winget
  install ffmpeg` hint.

### Empty / loading / error states

- Empty queue: illustrated film-strip glyph + helper text. Good.
- Detection loading: animated 3-dot throbber. Good.
- Mask preview success: badge with engine name + box count. Good.
- Mask preview no detection: helper text steering the user to lower
  the threshold. Good.
- Preview load failure: generic "Preview unavailable". Improvement:
  show the underlying cv2 error code so users can diagnose codec
  issues.

### Destructive / irreversible actions

- `_clear_queue` (line 5890): no confirmation dialog. The clear button
  is enabled until a batch is running -- one accidental click wipes a
  50-file batch.
- `_remove_from_queue` (line 5874): silent. Should at least show a
  toast that an item can be re-added.

### Settings clarity

- The 35+ knobs are split across cards (Profile / Workflow / STTN
  motion / Detection / Output / Video range) -- this is solid IA.
- BUT: the relationship between knobs is invisible. Example: enabling
  `keyframe_detection` makes `detection_frame_skip` mostly a no-op,
  but both are still presented as independent sliders.
- Recommended: a small "Active in this mode" subtle indicator next to
  each setting that's actually used by the selected mode / inpainter.

### Accessibility

- **Verified**: keyboard shortcuts for the major actions (Ctrl+O,
  Ctrl+Enter, F5, Ctrl+L, Ctrl+F).
- **Gap**: custom widgets (ModernButton / ModernToggle / ModernSlider)
  are tk.Canvas-based and have no UI Automation provider. NVDA /
  Narrator cannot announce their state. ROADMAP #95 acknowledges this
  is non-trivial.
- **Gap**: dark theme has no high-contrast variant. ROADMAP #96
  acknowledges.
- **Gap**: no GUI localisation. ROADMAP #97 acknowledges.

### Microcopy

- Generally good. The status hint and footer text use full sentences.
- Found one drift: the algorithm description (`_get_algo_description`,
  line 4677) refers to "ProPainter" without mentioning it's our hybrid
  TBE + LaMa, not the ICCV 2023 reference. CLAUDE.md is explicit about
  this; the user-facing text isn't. Fix: rename the user-facing label
  to "ProPainter (hybrid)" with a tooltip explaining the difference.

### Trust signals

- About dialog shows active detection engine + inpainting engine +
  GPU type. Good.
- The version chip in the header shows the current version. Good.
- Missing: link to the ROADMAP from the About dialog so users can see
  what's planned. Currently they have to find GitHub.

---

## Architecture and Maintainability

### Module boundaries

- `backend/__init__.py` lazy-loads `processor` symbols. Clean.
- `backend/processor.py` is 3,000 lines -- it does detection,
  inpainting, encoding, audio mux, CLI parsing, preset application,
  checkpointing. Recommended split:
  - `backend/detection.py` (SubtitleDetector + `_merge_boxes` +
    fallbacks)
  - `backend/inpainters/{sttn,lama,propainter,auto}.py` (each is ~100
    lines; the shared TBE primitive stays in
    `backend/inpainters/_tbe.py`)
  - `backend/io.py` (`_open_capture`, `_PrefetchReader`,
    `_FrameSequenceCapture`, `_deinterlace_to_temp`)
  - `backend/encoder.py` (`_detect_hw_encoder`, `_get_encode_args`,
    `_merge_audio`, `_reencode_or_copy`, atomic write helpers)
  - `backend/quality.py` (`_compute_quality_report`,
    `_write_quality_sheet`, SSIM)
  - `backend/tracking.py` (`_KalmanBox`, `SubtitleTracker`,
    `_group_horizontal_line`, `_phash`)
  - `backend/cli.py` (the argparse + dispatch)
  - `backend/processor.py` shrinks to `SubtitleRemover` + thin
    re-exports for backward compat.

  This isn't strictly necessary today; CLAUDE.md notes the project
  intentionally stays in two files. But once the file passes ~3K lines
  ASCII-only review starts being a tax. Tracked as L-1.

### Refactor candidates

- **L-1**: `backend/processor.py` split (above).
- **L-2**: `VideoSubtitleRemover.py` ProcessingConfig dataclass field
  enumeration (EI-5).
- **L-3**: Pluggable inpainter interface. ROADMAP #81 acknowledges; my
  proposed shape: each inpainter exposes a `register()` function that
  takes a `Registry`; `_create_inpainter` reads from the registry. Then
  a future LaMa-ONNX or MI-GAN backend lands as a new file, no
  modification to the dispatch.

### Test gaps

Existing tests (`tests/test_hardening.py`, 802 lines): coerce / fuzz /
karaoke / chyron / frame-sequence / quality sheet / prefetch / json log /
write-srt / settings migration / config load. All synthetic-input.

Missing:
- **No end-to-end test on a real video clip** (T-1 below).
- **No test for `process_video` outside of helpers**. The integration
  surface is untested.
- **No test for the OCR cascade selection** -- which engine ends up
  loaded when N engines are installed.
- **No test for the live preview callback** -- the marshalling +
  throttle logic.
- **No GUI smoke test** -- one test creates the app but skips on no
  display. Could be extended to walk through the major flows headlessly
  via Tk's `update_idletasks`.

### Documentation gaps

- **CHANGELOG.md** is up-to-date and detailed.
- **README.md** is current as of v3.12.0 but doesn't mention v3.13
  CLI features (multi-audio, loudnorm, chyron, karaoke, prefetch).
- **CLAUDE.md** is current.
- **ROADMAP.md** is the most complete planning doc I've seen in any
  project of this size. No gap.
- **No `docs/` directory**. For a project this size, a `docs/architecture.md`
  walking through the pipeline (detection -> tracker -> mask -> TBE ->
  refinement -> mux) would help contributors more than splitting the
  source files would.

### Release / build gaps

- GHA workflow runs unittest + pip-audit + PyInstaller. Solid.
- **No automated changelog stamp**: the workflow's `--generate-notes`
  pulls from GitHub PR titles; the rich CHANGELOG.md is not auto-pulled.
  Recommended: a workflow step that uses the latest `## [Unreleased]`
  block as the GH release body.
- **No code signing** (ROADMAP #50, on the list).
- **No installer** (ROADMAP #51, on the list).

---

## Prioritized Roadmap

> Phases are sized in chunks of work a single coding agent can ship in
> one focused session. Each item is independently verifiable.

### Phase 0 -- Tagged in v3.13 (already shipped, listed for completeness)

See [ROADMAP.md "Closed v3.13 hardening set"](../../../ROADMAP.md#closed-v313-hardening-set)
items 1-20, 36-41, 44, 46-48, 53, 56-57. No work needed here.

### Phase 1 -- Quality + correctness lift (P0 items)

- [ ] P0 - **I-1: Lossless intermediate codec**
  - Why: Every output is gen-2 lossy. Free quality lift.
  - Evidence: `backend/processor.py:2463`
  - Touches: `process_video`, `_reencode_or_copy`, `_merge_audio`.
  - Acceptance: PSNR(input, output) on unmasked pixels rises >=1 dB on
    a synthetic clip; a no-op pipeline (zero mask, no audio) is
    bit-identical to the input H.264 output.
  - Verify: `python tests/integration/test_lossless_intermediate.py`
    (new). Manual: encode a flat-colour clip, eyeball banding on the
    output before and after.

- [ ] P0 - **B-1: Wire missing backend fields through the GUI**
  - Why: Seven v3.13 features are CLI-only; GUI users get none.
  - Evidence: `_process_item:6451-6537` omits loudnorm /
    decode_hw_accel / multi_audio_passthrough / prefetch_decode /
    prefetch_queue_size / input_fps / quality_report_sheet /
    remove_subtitles / remove_chyrons / chyron_min_hits /
    karaoke_grouping / karaoke_x_gap_px / karaoke_y_overlap. GUI
    dataclass `to_dict` (line 336) omits the same.
  - Touches: GUI `ProcessingConfig` dataclass + `to_dict` +
    `from_dict` + `normalized`; `_build_settings_section` (new cards);
    `_process_item` config build; settings schema version 1 -> 2.
  - Acceptance: a fresh user can toggle each new control, close, reopen,
    and find the toggle preserved.
  - Verify: `python tests/test_hardening.py SettingsMigrationTests`
    (extend) + manual: change each new field; settings.json contains
    it; restart; field restored.

- [ ] P0 - **I-2: Normalise on cached-remover hot-swap**
  - Why: hot-swap path bypasses NaN/inf coercion.
  - Evidence: `_process_item:6562` `remover.config = backend_config`.
  - Touches: 1 line in `_process_item`.
  - Acceptance: a fuzz-test passing pathological values into the GUI
    queue then re-using the cached remover never raises.
  - Verify: extend `ConfigFuzzTests` to inject through the GUI path.

- [ ] P0 - **B-2: Surya GPL opt-in gate**
  - Why: MIT-clean philosophy.
  - Evidence: `backend/processor.py:572-582` auto-loads Surya.
  - Touches: `_load_model`, `detect_ai_engines`, About dialog.
  - Acceptance: With Surya installed but `VSR_ALLOW_GPL` unset,
    `_load_model()._engine_name` is never "Surya". With the env var
    set, it can be.
  - Verify: a new `SuryaOptInTests` class with two cases.

- [ ] P0 - **B-3: Quality report masked-region metric**
  - Why: whole-frame PSNR/SSIM masks regressions in the inpainted area.
  - Evidence: `_compute_quality_report:2116-2199`.
  - Touches: `_compute_quality_report`, new mask accumulator on
    `SubtitleRemover`.
  - Acceptance: an adversarial test that replaces the inpaint output
    with noise inside the mask drops `metrics['ssim']` below 0.5;
    today's whole-frame metric stays >0.9.
  - Verify: new `QualityReportMaskedRegionTests`.

### Phase 2 -- High-impact UX + format wins (P1 items)

- [ ] P1 - **F-1: Region selector frame scrubbing**
  - Why: frame 0 is useless for sources with intro cards.
  - Evidence: `_open_region_selector:5447`.
  - Touches: `_open_region_selector` (slider + seek).
  - Acceptance: opening selector on a 1-hour clip and dragging to
    30:00 shows the matching frame.
  - Verify: manual on a known clip + a Tk smoke test that asserts the
    slider widget exists.

- [ ] P1 - **F-2: Multi-rectangle region drawing**
  - Why: backend supports `subtitle_areas` since v3.9; GUI doesn't.
  - Evidence: `_open_region_selector:5525` writes only `subtitle_area`.
  - Touches: `_open_region_selector` (shift-click handler, list panel),
    `_reset_region`, `_update_region_label_display`.
  - Acceptance: drawing 2 rects writes `subtitle_areas`; mask preview
    shows both.

- [ ] P1 - **F-3: Inpaint-preview on a sample frame**
  - Why: today users can't A/B settings without a full run.
  - Evidence: `_show_preview:6082` only shows source or completed
    before/after.
  - Touches: `_show_preview`, new "Preview cleanup" button on the
    preview panel, background thread.
  - Acceptance: click "Preview cleanup" on a queued item, preview
    pane updates within 5 s with the inpainted result.

- [ ] P1 - **B-4: Multi-track loudnorm**
  - Why: Bluray rips with 3-5 tracks get partial normalisation.
  - Evidence: `_merge_audio:2800-2804` applies single-pass loudnorm.
  - Touches: `_merge_audio` (filter_complex), new pre-pass
    `ffprobe -select_streams a`.
  - Acceptance: `ffprobe -show_streams` on output confirms N audio
    tracks; `ffmpeg -af ebur128` on each track reports integrated
    loudness within 1 LU of target.

- [ ] P1 - **B-5: AutoInpainter unload after N consecutive TBE batches**
  - Why: lazily-loaded LaMa is held forever; ~1.5 GB VRAM permanently
    pinned on long videos.
  - Evidence: `AutoInpainter:1501-1543` no unload path.
  - Touches: `AutoInpainter` + new `lama_unload_after_n_tbe` field.
  - Acceptance: a synthetic alternating-batches test confirms
    `torch.cuda.memory_allocated()` drops after the threshold.

- [ ] P1 - **F-6: Adaptive ffmpeg timeout**
  - Why: 10-minute timeout silently strips audio on long videos.
  - Evidence: `_merge_audio:2809`, `_deinterlace_to_temp:1665`.
  - Touches: ffmpeg timeout calculation + Popen polling.
  - Acceptance: an 8-hour clip mux finishes with audio preserved.

### Phase 3 -- Polish, completeness (P2 / P3 items)

- [ ] P2 - **F-5: Open lang picker**
  - Touches: `_build_settings_section` lang combobox + a
    `_load_supported_languages` helper.

- [ ] P2 - **F-7: Per-item cancellation**
  - Touches: `QueueItem`, `_stop_processing`, `_process_item`, queue
    item context menu.

- [ ] P2 - **F-8: HEVC / AV1 output**
  - Touches: `ProcessingConfig.output_codec`, `_get_encode_args`, CLI
    `--codec`, GUI dropdown.

- [ ] P2 - **F-9: ETA estimate before batch starts**
  - Touches: `_start_processing`, `_compute_eta`, new probe step.

- [ ] P2 - **F-10: `--preset NAME` CLI**
  - Touches: move presets to `backend/presets.py`; `main()` parser.

- [ ] P2 - **EI-1: Otsu fallback detection thresholds**
  - Touches: `_fallback_detection`.

- [ ] P2 - **EI-2: Software decode for quality report output**
  - Touches: `_compute_quality_report` line 2138.

- [ ] P2 - **EI-5: Replace manual to_dict with dataclasses.asdict**
  - Touches: GUI `ProcessingConfig.to_dict`, `from_dict`. Stops the
    triple-edit-required problem permanently.

- [ ] P3 - **EI-3: Detection sensitivity slider direction + rename**
- [ ] P3 - **EI-4: Live preview off-thread PIL conversion**
- [ ] P3 - **EI-6: Output-path collision guard on manual edit**
- [ ] P3 - **EI-7: TikTok preset auto-band tuning** (needs real-source
  validation)

### Phase 4 -- Tests + infrastructure

- [ ] P0 - <a id="t-1"></a>**T-1: Reference-clip regression harness**
  - Why: ROADMAP #54 acknowledges; this plan scopes it.
  - Concrete clip selection (8 clips, ~10 s each, total <80 MB so the
    repo can ship them):
    1. `static_subtitle_dialogue_eng.mp4` -- 720p, English, fixed
       lower-third
    2. `karaoke_burnin_jp.mp4` -- 1080p, animated per-syllable JP
    3. `vertical_japanese.mp4` -- vertical CJK
    4. `motion_pan_fast.mp4` -- fast horizontal pan, subtitle in lower
       third
    5. `dissolve_cuts.mp4` -- cross-dissolve between scenes
    6. `chyron_news.mp4` -- persistent ticker + dialogue subtitles
    7. `vhs_grain.mp4` -- noisy SD source
    8. `thin_font_arabic.mp4` -- RTL thin-font Arabic
  - Each clip ships with a `*.expected.json` carrying baseline PSNR /
    SSIM / detected-box count.
  - Acceptance: nightly GHA run compares against baselines; tolerance
    +/-0.5 dB PSNR, +/-0.01 SSIM, exact box count.
  - Verify: `python tests/integration/test_reference_clips.py`.

- [ ] P1 - **T-2: End-to-end test with a synthesised clip**
  - Generates a 30-frame 320x180 BGR video with burned-in subtitle on
    half the frames; runs `SubtitleRemover.process_video` against it;
    asserts the output exists, has the right frame count, and a
    masked-region SSIM >0.85.
  - Verify: `python tests/integration/test_pipeline_end_to_end.py`.

- [ ] P1 - **T-3: OCR cascade selection test**
  - With each combination of (rapidocr / paddleocr / surya / easyocr)
    installed, assert the picked engine matches the documented
    priority. Skip combos that need a real install; use
    `unittest.mock.patch` on `import` so the cascade can be
    unit-tested.

- [ ] P2 - **T-4: GUI smoke test (Tk update_idletasks walk)**
  - Drives the major flows headlessly: open app -> add file (mocked)
    -> toggle each Advanced control -> close. Asserts no exceptions.

### Phase 5 -- Larger bets (L items, longer-horizon)

- [ ] L-1 - **Split `backend/processor.py` into 7 modules**
  - Done module-by-module so each split is a separate PR.

- [ ] L-2 - **Plugin architecture for inpainters** (concretises
  ROADMAP #81)
  - Each backend file exposes `register(registry: dict)`; the
    `SubtitleRemover` constructor picks from `registry[mode]`. Adding
    LaMa-ONNX (ROADMAP #25) or MI-GAN (ROADMAP #26) lands as a single
    new file + one-line registration.

- [ ] L-3 - **`docs/architecture.md`** -- contributors map of the
  pipeline. Pairs with L-1.

---

## Quick Wins

Items that can ship in one focused session:

- **I-2** (one-line fix in `_process_item`).
- **B-2** (Surya opt-in gate: <30 lines total).
- **EI-2** (one-line change in `_compute_quality_report`).
- **EI-3** (rename + invert one slider).
- **F-6** (timeout calculation, ~10 lines).
- **EI-5** (replace `to_dict` with `dataclasses.asdict`).

---

## Larger Bets

- **I-1** (lossless intermediate) -- touches the encode pipeline; needs
  a real benchmark.
- **B-1** (full GUI gap closure) -- 13 fields, 3 new Advanced cards,
  settings schema bump, integration tests.
- **L-1** (file split) -- multi-PR refactor.
- **L-2** (plugin architecture) -- prerequisite for ROADMAP #25 / #26 /
  #59 / #60 to land cleanly.
- **T-1** (reference-clip regression harness) -- requires curating
  clips + committing baselines; the clips themselves need rights
  cleared (use CC0 / public-domain sources).

---

## Explicit Non-Goals

Ideas considered and consciously rejected. The ROADMAP "Explicitly not
on the roadmap" section already lists cloud APIs, Docker, telemetry,
subscriptions, GPL defaults, multi-user, and mock weights. Adding:

- **Real-time / live preview during region selection at full
  resolution.** Considered: render the detection overlay live as the
  user drags. Rejected: the cost of running OCR per drag-event is too
  high; the existing "Detect" right-click flow gives the same signal on
  demand.
- **Auto-clean output filename suggestions ("DocumentaryClean.mp4"
  instead of `MyDoc_no_sub.mp4`).** Considered: NLP-based or
  language-detection-derived filename suggestions. Rejected:
  user-controlled filenames are a feature, not a bug. The current
  `_no_sub` suffix is unambiguous and reversible.
- **Built-in subtitle translation (translate the SRT export).**
  ROADMAP #85 (v5+) considers this; this plan does not promote it.
  Adds dependencies (Whisper, llama.cpp) that violate the
  no-heavy-mandatory-models principle.
- **Plugin marketplace / hot-load on running app.** Considered as a
  follow-on to L-2. Rejected for now: dynamic plugin loading is a
  security risk we don't want to inherit. Plugins load at startup
  only.
- **"Smart undo" of a completed batch (restore originals).** Considered.
  Rejected: outputs go to a separate path (`_no_sub` suffix); the
  original is never modified. The "undo" is already "delete the
  output". No need for an extra layer.

---

## Open Questions

Only the items that genuinely block prioritisation or implementation:

1. **DirectML torch pin: is there a 2.6+ wheel on the horizon?** The
   roadmap notes `torch-directml==0.2.5.dev240914` blocks the
   `torch>=2.10` CVE fix. If Microsoft has released a newer wheel
   compatible with `torch>=2.10`, the DirectML CVE gap closes
   immediately. Verify via
   `pip index versions torch-directml`.
2. **What's the user's median input resolution?** Adaptive batch
   sizing (`adaptive_batch=True`) defaults to a 50 MB-per-frame
   estimate assuming 1080p. If most users are at 4K, the default
   `sttn_max_load_num=30` is already too big on 8 GB GPUs. A telemetry
   pass would resolve this, but telemetry is off-roadmap; the next-best
   answer is to add a per-launch log line "first decoded frame:
   1920x1080, batch=30" so support requests carry the signal.
3. **Are users running Surya in practice?** If yes, B-2 should ship
   with a one-shot migration that detects an existing Surya install
   and asks the user to confirm the GPL opt-in rather than silently
   downgrading the cascade. If no, the simpler hard-gate is fine.
4. **Does the live preview throttle (EI-4) measurably slow 4K
   processing?** Static analysis says yes (~50 ms / frame on the Tk
   thread at 15 FPS = 75% utilisation). Live profiling on real
   hardware would confirm before committing to the off-thread fix.
5. **Lossless intermediate (I-1): how much disk does the raw stdin
   path use during processing?** A 1080p, 30 fps, 10-minute video is
   ~108 GB raw. The stdin pipe approach is fine because the data is
   never written to disk; the FFV1 / HFYU alternatives ARE written and
   could exhaust temp space. Verify the stdin-pipe path is the right
   default.

---

*Audit completed 2026-05-25 against `e057c41`. No code was modified;
this is research only.*
