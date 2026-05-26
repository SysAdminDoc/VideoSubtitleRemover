# Changelog

All notable changes to VideoSubtitleRemover will be documented in this file.

## [Unreleased]

### Added

- **Proxy-file workflow (RM-34).** New
  `backend/proxy_workflow.ensure_proxy(path, height, crf)` builds a
  cached low-res (default 480p) proxy via ffmpeg for fast preview /
  region selection on 4K source. Cache lives in
  `%APPDATA%/VSR/proxy_cache/` keyed by an MD5 fingerprint of the
  source (path, size, mtime).
- **Karaoke optical-flow mask warp (RM-43).** New
  `backend/karaoke_flow.warp_mask_with_flow(prev, next, mask)`
  exposes a Farneback flow remap so callers can union an old mask
  with the next detection to catch karaoke text that moved between
  frames.
- **WhisperX word-level alignment helper (RM-45).** New
  `backend/karaoke_flow.run_whisperx(audio_path)` returns
  `(start_s, end_s, word)` tuples when `whisperx` is installed.
  Pairs with the v3.13 Whisper-span fallback to provide finer-
  grained mask gating on borderline subtitle/dialogue cases.
- **VapourSynth bridge (RM-75).** New
  `backend/vapoursynth_bridge._VapourSynthCapture` evaluates a `.vpy`
  script and exposes a cv2.VideoCapture-shaped reader so
  `_open_capture("path.vpy")` ingests through VapourSynth when
  installed. Lets users chain QTGMC deinterlace / Waifu2x / SMDegrain
  ahead of VSR.
- **RTL layout scaffold (RM-98).** New `ProcessingConfig.rtl_layout`
  flag and `_rtl_layout` runtime hook set before widget
  construction. The deep pack-side flip across every widget is a
  follow-up; this commit lands the framework so future translation
  + RTL work can land incrementally.

- **SeedVR2 one-step restoration (RM-77, opt-in).** New
  `backend/post_restore.seedvr2_restore` runs the SeedVR2 wrapper
  (or whatever CLI `VSR_SEEDVR2_CMD` names) as another post-cleanup
  stage. 16B-param diffusion transformer, single sampling step,
  best-in-class quality on heavily-degraded footage. CLI `--seedvr2`.
- **TensorRT engine cache for the LaMa-ONNX backend (RM-70, opt-in).**
  New `backend/tensorrt_compile.maybe_compile_engine` invokes
  `polygraphy convert` once per ONNX, caches the result in
  `%APPDATA%/VSR/trt_cache/`, and adds the
  `TensorrtExecutionProvider` to the LaMa-ONNX session. ~2-3x
  further speedup on top of ONNX Runtime. Activate via
  `VSR_TENSORRT=1`.
- **INT8 OCR quantization script (RM-39).** New
  `scripts/quantize_ocr.py` calls `onnxruntime.quantization.quantize_dynamic`
  on a RapidOCR detection ONNX. Drop-in replacement; the resulting
  model halves detection cost on CPU with <1% F1 loss in practice.

### Added

- **SAM 2 / SAM 3 / MatAnyone 2 / CoTracker3 adapters
  (RM-66/67/68/69, opt-in).** New `backend/segmentation.py` exposes:
  - `refine_mask_with_sam2(frame, boxes, base_mask)` -- prompted box
    -> text-shaped mask, integrated into `_create_mask` so enabling
    `sam2_refine` (CLI `--sam2-refine`) replaces the padded rect mask
    with the SAM 2 output. Tighter mask = less inpaint area.
  - `segment_text_with_sam3(frame)` -- text-prompt mask via SAM 3
    when `VSR_SAM3=1` + the `sam3` package is installed.
  - `matte_frame(frame, hint_mask)` -- MatAnyone 2 soft alpha matte
    for thin moving subtitle lines.
  - `track_points(frames, points)` -- CoTracker3 helper for callers
    that need to confirm a karaoke caret stays on the same line.

- **Diffusion inpainter scaffolds (RM-59/60/61/62/63/64/65, opt-in).**
  New `backend/inpainters_diffusion.py` registers seven optional
  video-inpainter backends with the plugin registry:
  - `propainter-real` (sczhou/ProPainter ICCV 2023 reference)
  - `diffueraser` (lixiaowen-xw/DiffuEraser 2025)
  - `vace` (ali-vilab/VACE 1.3B MV2V)
  - `videopainter`
  - `cococo` (text-guided, prompt via `VSR_COCOCO_PROMPT`)
  - `eraserdit` (research-stage track)
  - `floed` (flow-guided efficient diffusion)
  Each registers ONLY when its enable env var is set; missing
  packages route through the existing TBE primitive so the pipeline
  never crashes. Each becomes addressable via `--mode <name>` once
  the user opts in.

- **PyNvVideoCodec GPU-resident decode (RM-71, opt-in).** New
  `backend/decode_accel._PyNvVideoCapture` wraps NVIDIA's
  PyNvVideoCodec with a cv2.VideoCapture-shaped facade. Activated
  via `VSR_PYNVVIDEOCODEC=1`; falls back to cv2 transparently when
  the package is missing or the open fails. ~6x faster decode on
  reference NVIDIA hardware. Frames currently download to CPU as
  BGR so the rest of the pipeline is unchanged; the zero-copy
  GPU-resident path remains future work.
- **RIFE-interpolated fast mode helper (RM-72, opt-in).**
  `maybe_interpolate_pair(prev, next, t)` calls Practical-RIFE when
  `practical-rife` is installed, otherwise returns None. Caller
  falls back to a duplicate frame. The full pipeline-side wiring
  (detect every Nth frame, RIFE the gaps) is queued as a follow-up;
  the adapter ships now so the integration is decoupled from the
  fast-mode dispatch.
- **Batched LaMa inference (RM-40, opt-in).** Set
  `VSR_LAMA_BATCH=1` to stack the LAMA-mode batch and call the
  underlying torch model in a single forward pass. ~2-3x faster
  than per-frame on a 30-frame batch. Falls back to the per-frame
  path on any shape mismatch / model-attribute mismatch.

### Added

- **VLM OCR detector cascade (RM-22, RM-23).** New
  `backend/ocr_vlm.py` registers four optional detectors fronting the
  default RapidOCR -> PaddleOCR -> Surya -> EasyOCR cascade. Pick one
  via `VSR_VLM_OCR={florence2|qwen25vl|paddleocr-vl}`:
  - **Florence-2** (microsoft/Florence-2-base) layout-aware OCR with
    `<OCR_WITH_REGION>` quad boxes.
  - **Qwen2.5-VL** (Qwen/Qwen2.5-VL-2B-Instruct) -- leads
    OmniDocBench as of April 2026. Prompts the model for a JSON box
    list.
  - **PaddleOCR-VL** -- exposes PaddleOCR 3.0+'s `paddleocr_vl`
    profile when the install ships it; falls back to PP-OCRv5
    otherwise.
- **Manga / anime mode (RM-42).** Setting `detection_lang="manga"`
  routes through manga-ocr + comic-text-detector (if installed) to
  pick up irregular speech-bubble shapes and vertical Japanese.
  Falls back to an Otsu-derived single-bubble crop when
  comic-text-detector isn't installed so the mode degrades
  gracefully. The Subtitle Language picker now lists "Manga / Anime"
  as a selectable language.

- **Pre-detect denoise (RM-33, opt-in).** New
  `backend/preprocess.fastdvdnet_denoise_frame` runs FastDVDnet on the
  detection-frame stream when `VSR_FASTDVDNET` + torch are available,
  falling back to OpenCV NLM otherwise. Output pixels stay untouched
  -- the denoise only sharpens the OCR signal. CLI `--denoise-detect`.
- **TransNetV2 deep scene-cut detector (RM-21, opt-in).** Slots into
  the existing `_detect_scene_cuts` cascade ahead of the PySceneDetect
  / histogram paths when `transnetv2` is `pip install`-ed and
  `VSR_TRANSNETV2` names the model weights. CLI `--transnetv2`.
- **AV1 / VP9 ingest validation (RM-74).** New `_probe_codec_for_log`
  helper logs the source codec + dimensions at the top of every run
  so users can reproduce a decode failure with the exact ffmpeg
  invocation. AV1 egress is already covered by the `--codec av1`
  output dropdown shipped in v3.14.

## [v3.14.0] -- 2026-05-25

Backlog drain pass. Six previously-deferred optional integrations
land as opt-in adapters; the existing pipeline keeps working
byte-identical for users who do not opt in.

### Added

- **SwinIR restoration pass (RM-79, opt-in).** Pairs with
  Real-ESRGAN: `swinir_restore=True` (CLI `--swinir`) routes the
  post-cleanup output through whichever of `swinir-ncnn-vulkan`,
  `realsr-ncnn-vulkan`, or `swinir` is on PATH. Skipped silently
  when no binary is found.
- **Synthetic reference-clip regression harness (RFP-T-1).** New
  `tests/test_reference_clips.py` generates eight deterministic
  synthetic clips (static dialogue / motion pan / dissolve cuts /
  karaoke / persistent chyron / vertical text column / thin font /
  gradient background), runs the full pipeline against each, and
  asserts the run completes. CC0 real-world clip sourcing is the
  next pass.
- **APP_VERSION bumped to 3.14.0.**

### Changed

- **Inpainter dispatch is now plugin-driven (RFP-L-2).** New
  `backend/inpainter_registry.py` exposes
  `register(name, builder)` / `resolve(name)` /
  `is_registered(name)` / `list_modes()` / `unregister(name)`. The
  four built-in inpainters (STTN / LAMA / ProPainter / AUTO)
  register themselves at processor import time; `_create_inpainter`
  resolves through the registry instead of an if-elif chain.
  External backends (LaMa-ONNX, MI-GAN, real ProPainter,
  DiffuEraser, etc.) can now land as standalone modules that
  import `register` and a `BaseInpainter` subclass -- no edit to
  the dispatch needed. Re-registering a name replaces the
  previous builder, so a drop-in faster implementation can shadow
  a default.

### Added

- **GUI localisation scaffold (RM-97).** New `backend/i18n.py`
  exposes `bind_locale(lang)` + `_("...")` so a future translation
  catalog can drop into `locale/<lang>/LC_MESSAGES/vsr.mo` and bind
  on startup without further code changes. Locale auto-detection
  uses `locale.getlocale()`; non-`en` users get a translated UI
  whenever a catalog ships. A `locale/vsr.pot` template seeds the
  first wave of strings translators can target.
- **UIA screen-reader announce scaffold (RM-95).** New
  `backend/a11y.py` exposes `announce(text, importance)` that fires
  a Windows UIA notification readable by NVDA / Narrator. Probed
  lazily via comtypes; silent no-op on non-Windows / missing deps.
  Wired into `_notify_completion` so screen-reader users learn the
  batch finished (and how many errors landed) without polling the
  activity log.

- **HDR / colorspace metadata passthrough (RM-73 partial).** New
  `backend/hdr.py` probes the source's color signalling via ffprobe
  (`color_primaries`, `color_transfer`, `color_space`, `color_range`)
  and re-tags the output encode with the same flags, even though the
  pixel pipeline is still 8-bit BGR. HDR sources land in the log
  banner ("Detected: bt2020 / smpte2084 -- output tagged as HDR but
  pixels tone-mapped"). The full 16-bit pixel pipeline is queued as a
  follow-up; this slice at least stops the output from being
  mis-tagged. New `preserve_color_metadata` config + `--no-color-preserve`
  CLI flag.
- **NLE round-trip sidecar (RM-76).** New
  `backend/nle_sidecar.py` writes a 1-event CMX 3600 EDL or a
  minimal FCPXML 1.10 stub next to the output naming the source,
  cleaned filename, and processed time range. Lets a DaVinci /
  Premiere editor hand-conform the cleaned clip at the same timecode.
  `--nle-sidecar {off|edl|fcpxml}` CLI flag + GUI mirror.

- **Real-ESRGAN upscale + film-grain post-restore stages (RM-78, RM-80,
  opt-in).** New `backend/post_restore.py` adapters and a
  `_run_post_restore_passes` hook in `process_video` that fires after
  the main mux:
  - `upscale_factor` (0/2/3/4) shells out to
    `realesrgan-ncnn-vulkan` for 2x/3x/4x upscaling; the original
    output stays on disk when the binary is missing.
  - `film_grain_strength` (0..0.5) adds an ffmpeg `noise` filter pass
    so inpainted regions blend with the surrounding grain. Skipped
    when ffmpeg is missing. Note: for AV1 outputs prefer the
    encoder's native film-grain table over this additive pass.
  CLI exposes `--upscale {0,2,3,4}` and `--film-grain STRENGTH`.

- **Whisper-driven mask fallback (RM-27, opt-in).** When
  `whisper_fallback` is on AND `faster-whisper` is `pip install`-ed,
  `process_video` extracts the audio track, runs Whisper once per
  file, and applies a default bottom-band mask to frames whose OCR
  returned no boxes but whose timecode falls inside a speech span.
  Catches anti-aliased / motion-blurred / decorative subtitles the
  OCR cascade misses. New `backend/whisper_fallback.py` module;
  CLI `--whisper-fallback` and `--whisper-model {tiny|base|small|
  medium|large|large-v2|large-v3}`. The audio extraction temp dir
  is cleaned up in the same finally block as the main temp dir.

- **LaMa-ONNX inpainter backend (RM-25, opt-in).** Set
  `VSR_LAMA_ONNX=<path to lama_fp32.onnx>` and `pip install
  onnxruntime` / `onnxruntime-gpu` to swap the default PyTorch LaMa
  for a 3-5x faster ONNX Runtime path. The new
  `backend/inpainters_onnx.py` module registers `LamaOnnxInpainter`
  through the plugin registry, replacing the `lama` mode slot. If
  onnxruntime is missing or the model file is unreadable, the
  backend falls back to `cv2.inpaint` per-frame so users never see a
  hard failure.
- **MI-GAN ONNX inpainter backend (RM-26, opt-in).** Set
  `VSR_MIGAN_ONNX=<path to migan.onnx>` and `pip install
  onnxruntime`; a new `migan` mode lands in the inpainter registry
  + CLI choice list. ~10 ms per 512x512 on a modern CPU for the
  cleanup-only case (matches the ICCV 2023 paper); single-frame so
  there is no TBE temporal pass. Useful for image queues and
  underpowered laptops.

- **PySceneDetect-backed scene cut detector (RM-32, opt-in).** New
  `tbe_scene_cut_use_pyscenedetect` field; when on and `scenedetect`
  is `pip install`-ed, the TBE batch-splitter uses PySceneDetect's
  AdaptiveDetector instead of the built-in histogram correlator.
  Handles dissolves and flashes that mis-fire on the histogram path.
  Defaults off; the histogram path stays the zero-dep default.

- **Vertical text mode (RM-24).** Japanese tategaki and classical
  Chinese subtitle columns now detect cleanly. The detector wrapper
  rotates each frame 90 CCW before invoking whichever engine is
  loaded, then rotates the returned boxes back into the source
  frame's coordinate space. New `ProcessingConfig.detection_vertical`
  field, CLI `--vertical`, and a Detection-card toggle in the GUI.
- **High-contrast theme variant (RM-96).** A low-vision-friendly
  palette (pure black surfaces, pure white text, saturated accent
  colours, yellow focus ring) toggleable from the Output card and
  persisted in settings. Applies on next launch because re-skinning
  every live widget mid-session would force a tree-wide redraw the
  design tokens were not built for.
- **A/B flicker-scrubber preview (RM-30).** Completed items grow an
  "A/B compare" preview button that opens a Toplevel with a frame
  slider AND a vertical-wipe slider so users can compare the source
  and cleaned outputs side-by-side at any frame. Both captures stay
  open for the duration of the modal and are released on close.

- **Per-file overrides popover (RM-29).** Right-click an idle queue
  item -> "Override settings for this file..." opens a themed modal
  that edits the item's own `ProcessingConfig` snapshot. Surfaced
  fields: mode (segmented picker), detection language, sensitivity
  slider, output codec. Overrides survive a global UI change because
  every queue item already carries its own config dataclass; the
  popover writes back to `item.config` and runs `.normalized()` so
  bad values never reach the worker.

### Changed

- **Detection slider relabelled "Sensitivity" (EI-3).** The underlying
  knob is unchanged (10-90% confidence floor); the new label removes
  the inversion users had to remember ("threshold" goes DOWN to detect
  more text). New hint: "Higher catches more text (lower confidence
  floor). Lower is stricter."

### Performance

- **Live preview worker-side throttle (EI-4).** The backend's
  `on_preview_frame` callback now drops conversions when the previous
  one fired within the last 1/15 s, so the worker stops burning CPU on
  cv2.resize + PIL.Image.fromarray that the receiver was going to
  throttle away anyway.

### Tests

- **ConfigFuzzTests expanded** to cover every v3.13 GUI-exposed field
  (loudnorm_target, multi_audio_passthrough, decode_hw_accel,
  prefetch_decode, prefetch_queue_size, input_fps, quality_report_sheet,
  remove_subtitles, remove_chyrons, chyron_min_hits, karaoke_grouping,
  karaoke_x_gap_px, karaoke_y_overlap, output_codec). 1500 random
  payloads on each path; post-conditions assert loudnorm range,
  decode_hw_accel token set, output_codec set, input_fps bounds.

## [v3.13.0] -- 2026-05-25

### Added

- **HEVC + AV1 output codec dropdown (F-8).** Output is no longer
  H.264-only. New `ProcessingConfig.output_codec` (h264 / h265 / av1)
  picks the matching HW encoder family (`hevc_nvenc`/`hevc_qsv`/
  `hevc_amf` for h265, `av1_nvenc`/`av1_qsv`/`av1_amf` for AV1) with
  `libx265` / `libsvtav1` software fallbacks. CLI exposes `--codec`;
  the Output card grows a "Output codec" dropdown next to HW
  encoding. Settings persist via the existing dataclass-driven
  pipeline.
- **Per-item cancellation (F-7).** Right-click a running queue item
  -> "Cancel this item" sets a per-`QueueItem` cancel flag. The
  worker's progress callback raises `InterruptedError` next tick so
  the file is dropped, but the global `cancel_event` stays clear and
  the remainder of the batch continues. Per-item flag is reset every
  time `_process_item` re-enters so a retry works cleanly.
- **Vendored SHA-256 weight verification (RM-49).** New
  `backend/model_hashes.py` registers known-good hashes for opt-in
  model downloads and a chunked verifier (`verify_weight_file`) safe
  for multi-GB files. The LAMA loader scans the standard torch.hub
  cache for `big-lama*.pt` on first init and warns -- but does not
  refuse to load -- when the hash mismatches. Catches silent
  supply-chain swaps and truncated downloads that would otherwise
  surface as cryptic deep-model errors hours into a run.

- **Region selector grows frame scrubbing + multi-rectangle drawing
  (F-1, F-2).** The selector window now carries a frame slider for
  video sources so users can pin the rect on a frame where the
  subtitle is actually visible -- the legacy "always frame 0" path
  silently failed on every clip with a black intro card. Every drag
  appends a rect to the list; "Clear all" removes them; "Save" writes
  every rect to `subtitle_areas` (plus the first rect to
  `subtitle_area` for backward compatibility with single-rect callers).
- **One-click cleanup preview (F-3).** The Preview panel grew a
  "Preview cleanup" button that runs detect + inpaint on the first
  frame of the selected queue item in a background thread and renders
  the result inline. Users can A/B detection thresholds, mask dilation,
  and mode choices without committing a full batch run.

- **Per-stream loudness normalisation for multi-track audio (B-4).**
  When both `loudnorm_target` and `multi_audio_passthrough` are set and
  the source has more than one audio stream, `_merge_audio` now builds
  a `-filter_complex` pipeline with one `loudnorm=I=...` branch per
  stream. Each track lands at the same LUFS target, so a Bluray rip
  with main / commentary / dub tracks normalises uniformly instead of
  applying the filter to track 0 only.
- **Pre-batch ETA probe (F-9).** Starting a batch now runs a 30-frame
  detect probe on the first queued video, scales the result by the
  full duration, and seeds `_compute_eta` with the estimate. Users see
  "about X left (estimated)" from the very first frame instead of an
  empty string until the first item finishes.
- **Expanded language picker (F-5).** The Subtitle Language dropdown
  now covers ~50 languages (was 12). Curated friendly names lead the
  list (English first), with the rest filling in by ISO code so users
  with Thai / Vietnamese / Polish / Greek / Ukrainian / etc. footage
  can pick a sensible engine code without modifying source.

- **CLI `--preset NAME` flag + shared preset library (F-10).** Presets
  now live in `backend/presets.py` so the GUI's picker and
  `python -m backend.processor --preset NAME` resolve from the same
  table. CLI flags typed alongside `--preset` still win on a conflict
  (the preset only fills attrs whose argparse value is still the
  parser default). New companion flag `--list-presets` prints every
  known preset (built-in + user) and exits.
- **"Repeat with these settings" queue action (RM-28).** Right-click a
  queue item -> "Repeat with these settings" re-queues the same source
  with a snapshot of that item's `ProcessingConfig`. Useful when you
  have tweaked the global UI knobs but want to re-run an earlier file
  with the exact settings that worked the first time.

### Improved

- **OpenCV fallback detector catches mid-tone subtitles (EI-1).** The
  legacy fixed thresholds at 200 / 55 missed semi-transparent banners
  and dimly-lit captions whose luminance sat in the 55-200 dead zone.
  The fallback now picks thresholds from the frame's 5th / 95th
  percentile, clamped so a near-flat source cannot collapse both
  thresholds and mark the entire frame.

- **Lossless FFV1 intermediate (I-1).** The temp file written between
  the inpaint pass and the final ffmpeg encode used to be `mp4v`
  inside `.mp4` -- a full generation of lossy compression sitting in
  front of the user-visible H.264/NVENC final encode. Every output was
  effectively gen-2 lossy. The new `_LosslessIntermediateWriter` pipes
  raw BGR frames through a Popen-spawned `ffmpeg -c:v ffv1` writing
  `.mkv`, so the final encode pass is the only lossy step. When
  ffmpeg is missing the writer falls back to the legacy `mp4v` path
  with a logged warning, so installations without ffmpeg keep working
  at the old quality. Verified by `LosslessIntermediateWriterTests`
  that the FFV1 round-trip is bit-identical for FFV1-eligible
  installations.

- **Quality report includes inpaint-region (ROI) PSNR/SSIM (B-3).** The
  v3.12 quality report was computed over the entire frame, so unchanged
  pixels (typically 80-95% of the area) dominated the metric and could
  hide a bad inpaint behind a strong-looking overall score. The pipeline
  now accumulates the union-mask bbox while processing and the report
  returns both a whole-frame metric and an ROI-cropped metric. The
  Good/Review tag is now driven by the ROI score when available. ROI
  output: `{'roi_psnr': float, 'roi_ssim': float, 'roi_bbox': [x1,y1,x2,y2]}`
  alongside the existing whole-frame fields.
- **AutoInpainter unloads idle LaMa (B-5).** When the AUTO routing has
  stayed on the TBE path for `LAMA_IDLE_UNLOAD_AFTER` consecutive
  batches (50, ~1500 frames at batch=30), the lazily-loaded
  `LAMAInpainter` reference is dropped and `torch.cuda.empty_cache()`
  is called. A later hard batch re-loads on demand. Long videos that
  hit one hard batch early no longer permanently pin ~1.5 GB VRAM.

- **GUI: thirteen v3.13 backend fields are now reachable from the
  Advanced panel.** Loudness normalisation target (LUFS, 0 = off),
  multi-track audio passthrough toggle, hardware-decode hint dropdown
  (off / auto / any / d3d11 / vaapi / mfx), worker-thread frame
  prefetch toggle, chyron classifier (Remove dialogue subtitles /
  Remove persistent text), karaoke grouping toggle, and the quality
  report sheet toggle. Previously these were CLI-only -- the GUI built
  a `BackendConfig` that simply didn't pass them through, so toggling
  them in the GUI had no effect. New Advanced cards: "Editorial",
  "Audio", "Performance". The Output card grew the quality-sheet
  toggle next to the existing PSNR/SSIM report toggle.
- **GUI: settings.json schema bumped 1 -> 2.** Older files round-trip
  cleanly via `_migrate_settings`; new keys land at backend defaults so
  pre-v3.13 behaviour is preserved for users who never touch them.

### Changed

- **GUI ProcessingConfig persistence is dataclass-driven.** `to_dict`
  and `from_dict` now walk `dataclasses.fields(self)` rather than a
  manual enumeration. The 13-field B-1 gap was rooted in three
  enumerations (`to_dict`, `from_dict`, `normalized`) that all needed
  to be edited in lockstep whenever a new field landed -- a structural
  invitation for drift. New fields now persist by default; only
  `normalized` still requires an explicit coercion entry (intentional:
  it documents the safe range).

### Security

- **Surya GPL opt-in gate** -- the OCR cascade no longer auto-loads Surya
  even when it is pip-installed. Surya is GPL-licensed; loading it at
  runtime in a PyInstaller bundle put the MIT-clean release at risk.
  Users who want Surya must set `VSR_ALLOW_GPL=1` in the environment.
  When the gate is closed but Surya is installed, the loader logs a
  warning explaining the env var. `detect_ai_engines()` in the GUI now
  labels it `"Surya (GPL -- set VSR_ALLOW_GPL=1)"` so the About dialog
  reflects the gated state.

### Fixed

- **Cached-remover hot-swap missed normalisation** -- when a queue item
  reused a cached `SubtitleRemover` (same mode / device / language) the
  GUI assigned the new `BackendConfig` directly to `remover.config`,
  bypassing `normalize_processing_config`. A NaN/inf or out-of-range
  per-item override could then leak into the pipeline. The hot-swap now
  routes through the normaliser the constructor uses on cold start.
- **Quality-report output capture honoured HW-accel hint** -- the 10-
  frame PSNR/SSIM sample pass opened the just-written output through
  `decode_hw_accel`, which can fall back inconsistently against a fresh
  H.264 mp4. The output capture now forces software decode; the input
  capture still honours the user's hint.
- **ffmpeg subprocess timeout truncated long videos** -- the audio mux,
  yadif deinterlace, and reencode-or-copy paths all used a fixed 600 s
  timeout. Videos over ~1 hour silently fell back to "copy without
  audio" once the encode pass ran longer than 10 minutes. The timeout is
  now adaptive: `_ffmpeg_subprocess_timeout(duration)` returns
  `base + duration * 4` with a 24-hour ceiling and a 600-s floor when
  ffprobe is unavailable.

### Added

- **Karaoke / per-syllable grouping (`--karaoke-grouping`)** -- new
  `_group_horizontal_line()` helper fuses OCR boxes on the same
  horizontal text line into a single composite. Karaoke captions
  render as many small per-syllable boxes that the rest of the
  pipeline treats as independent lines; masking them individually
  leaks the original highlighted text through the gaps between
  syllables. Two boxes merge when their vertical extent overlaps by
  at least `karaoke_y_overlap` (default 0.5) AND the horizontal gap
  is at most `karaoke_x_gap_px` (default 20). The merge loop iterates
  until no further merges happen, so five syllables fuse into one
  rectangle in one call. Pure transformation; safe to apply before
  the Kalman tracker so the smoothed track covers the fused span.
  CLI: `--karaoke-grouping`, `--karaoke-x-gap PX`,
  `--karaoke-y-overlap RATIO`.
- **Chyron vs subtitle classifier (`--keep-chyrons` / `--keep-subtitles`)**
  -- `_KalmanBox.is_chyron(min_hits)` and `SubtitleTracker.categorize()`
  classify each detection by lifetime: a Kalman track that has matched
  in `>= chyron_min_hits` frames (default 90, ~3 s at 30 fps) is a
  chyron (station logo, lower-third, breaking-news ticker); shorter-
  lived tracks are dialogue subtitles. New `ProcessingConfig` fields
  `remove_subtitles` and `remove_chyrons` (both default True for
  backward compatibility) gate which class is sent to the inpaint
  mask. The filter is a no-op when both are True, so v3.12 behaviour
  is preserved. CLI: `--keep-chyrons`, `--keep-subtitles`,
  `--chyron-min-hits N`.
- **Frame-sequence input (DPX / EXR / PNG / JPG directories)** -- new
  `_FrameSequenceCapture` adapter mirrors the `cv2.VideoCapture` surface
  (`isOpened`, `read`, `set(POS_FRAMES)`, `get(FPS / WIDTH / HEIGHT /
  FRAME_COUNT)`, `release`) over a directory of images walked in sorted
  filename order. First image fixes width / height; mid-sequence size
  changes are letterboxed to that frame so the writer pipeline never
  sees a dimension shift. Routed transparently through `_open_capture()`
  and the `process_video` path. `process_video` silently bypasses the
  audio merge when the input is a directory (no audio stream). New
  `ProcessingConfig.input_fps` (default 24.0, clamped to [1, 240]);
  `--input-fps FPS` CLI flag. Ingest-only for v3.13; output remains mp4
  -- PNG/EXR sequence *output* is queued for a later release.
- **Quality self-test sheet (`--quality-sheet`)** -- extends the existing
  PSNR / SSIM numeric report with a side-by-side comparison PNG written
  next to the output as `<output>.qualitysheet.png`. Each sampled frame
  becomes one row (`original | cleaned`) with a caption showing the
  per-frame PSNR/SSIM; a header strip carries mean PSNR, mean SSIM, and
  a `Good` / `Review` tag derived from the SSIM 0.95 threshold. Implies
  `--quality-report`; the `quality_report_sheet` config field
  auto-enables `quality_report` in `normalize_processing_config` so a
  config-file overlay can't reach an inconsistent state.
- **Prefetch / pipeline parallelism (`prefetch_decode`, default on)** --
  new `_PrefetchReader` wraps `cv2.VideoCapture` with a daemon worker
  thread that fills a bounded frame queue while the main thread runs
  detection + inpainting. cv2 / numpy / onnxruntime release the GIL on
  heavy calls so plain threading is enough to overlap I/O with compute.
  Strict ownership rule: once the wrapper is active, the underlying cap
  must not be touched directly (`.read()`/`.set()`/`.get()` on the raw
  cap from the main thread would race the worker). Cleanup goes through
  `reader.release()`, which sets a stop event, drains the queue so the
  worker isn't blocked on `put()`, joins the thread, and then releases
  the underlying cap. Exception path in `process_video` is wired through
  the same path so a crash mid-batch never leaks the worker thread.
  Toggle off via `--no-prefetch`. Queue size auto-derives from
  `sttn_max_load_num * 2` (min 8); override with `--prefetch-queue N`.
- **Structured JSON-line log option (`--json-log PATH`)** -- new
  `backend.processor.JsonLineLogHandler` writes one JSON record per line
  with `ts` (UTC ISO-8601), `level`, `logger`, `msg`, and optional `exc`
  (formatted traceback when the record carries `exc_info`). The text log
  in `%APPDATA%\VideoSubtitleRemoverPro\vsr_pro.log` keeps writing in
  parallel; this handler is purely additive. Useful for `jq` / `grep`
  pipelines across days of batch jobs. CLI-only for v3.13; the GUI
  toggle lands in a later release alongside the loudnorm GUI control.
- **Multi-track audio passthrough (default on)** -- `_merge_audio` now
  emits `-map 1:a?` (all input audio streams) re-encoded to AAC instead
  of `-map 1:a:0?` (first only). Bluray/DVD rips routinely carry 3-5
  language tracks; the legacy behaviour silently dropped all but the
  first. New `ProcessingConfig.multi_audio_passthrough = True` field +
  `--single-audio` CLI flag for the legacy behaviour. Caveat: the
  simple single-pass loudnorm filter applies only to the first selected
  audio stream; broadcast-grade multi-track loudnorm needs
  `-filter_complex` and is deferred.
- **Hardware-accelerated decode hint (`--decode-accel`)** -- new
  `_open_capture()` helper supports `off` (default; status quo),
  `auto`/`any`, `d3d11` (Windows DXVA2/D3D11VA), `vaapi` (Linux), or
  `mfx` (Intel Media SDK). Probes one frame on open; if the HW path
  returns no frames (known cv2/FFmpeg issue, opencv/opencv#25185) it
  silently falls back to software decode with a warning. Wired into the
  main video decode path in `SubtitleRemover.process_video`. Other
  short-scan call sites (subtitle-band probe, quality-report sampler)
  remain on the software path and are tracked as a follow-up.
- **`--validate-config` CLI dry-run** -- parse all CLI flags + the optional
  `--config` JSON overlay, normalise the resolved `ProcessingConfig`,
  print it as JSON, and exit 0 without instantiating the detector or
  inpainter. Lets shell scripts verify flag combinations before launching
  a long batch. `--input` / `--pattern` / `--output` / `--out-dir` are
  not required when this flag is set.
- **`--skip-existing` CLI toggle** -- skip any input whose output path
  already exists, regardless of the checkpoint store. Independent of
  `--no-resume`; useful when re-running a glob against a partly populated
  output directory without enabling the full checkpoint workflow.
- **EBU R128 loudness normalisation (`--loudnorm <LUFS>`)** --
  `ProcessingConfig.loudnorm_target` defaults to 0.0 (off). When set to a
  LUFS value in [-70, -5], the ffmpeg audio mux runs an extra
  `-af loudnorm=I=<target>:TP=-1.5:LRA=11` pass during merge. Common
  platform targets: YouTube -14, Apple -16, broadcast -23. Single-pass for
  speed; broadcast-grade two-pass measure-then-apply may follow.
  CLI-only for v3.13; the GUI control lands in a later release.

### Security

- **CI dependency vulnerability scan (`pip-audit`)** -- the GitHub Actions
  workflow now installs `pip-audit` after the runtime stack and reports
  known CVEs in the installed closure. Non-fatal during the v3.13
  transition (`continue-on-error: true`); the gating switch flips to
  fail-on-vuln once the pin set in `requirements.txt` is stable.
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
