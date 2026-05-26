# Active Backlog (single source of truth)

> Consolidates the open items from [ROADMAP.md](./ROADMAP.md) and
> [RESEARCH_FEATURE_PLAN.md](./RESEARCH_FEATURE_PLAN.md). Shipped items
> live in [CHANGELOG.md](./CHANGELOG.md). When an item closes, check it
> off here AND log the ship details in CHANGELOG under `[Unreleased]`.
>
> Priority key: P0 (next release blocker) > P1 (high value) > P2 (nice) > P3 (polish).
> Source tag: `RFP-<id>` = from RESEARCH_FEATURE_PLAN, `RM-<n>` = from ROADMAP.

---

## Phase 1 -- Correctness + GUI/backend gap (P0)

- [x] P0 RFP-I-2 Normalize on cached-remover hot-swap
- [x] P0 RFP-B-2 Surya GPL opt-in gate
- [x] P0 RFP-EI-2 Software decode for quality-report output capture
- [x] P0 RFP-F-6 Adaptive ffmpeg timeout for long videos
- [x] P0 RFP-B-1 Wire 13 missing backend fields through the GUI
- [x] P0 RFP-B-3 Quality report measures masked region only
- [x] P0 RFP-B-5 AutoInpainter unload LaMa after N consecutive TBE batches
- [x] P0 RFP-I-1 Lossless intermediate codec (eliminate double-encode)

## Phase 2 -- Workflow features (P1)

- [x] P1 RFP-F-1 Region selector frame scrubbing
- [x] P1 RFP-F-2 Multi-rectangle region drawing (surface subtitle_areas)
- [x] P1 RFP-F-3 Inpaint-preview on a sample frame
- [x] P1 RFP-B-4 Multi-track loudness normalisation (filter_complex)
- [x] P1 RM-28 "Repeat with these settings" queue action
- [x] P1 RM-29 Per-file overrides popover (mode / language / sensitivity / codec)
- [x] P1 RM-30 A/B flicker-scrubber in preview pane
- [x] P1 RM-31 Subtitle-area drag-refinement in-GUI (shipped with F-1/F-2)
- [x] P1 RM-49 Model-weight hash verification on first download

## Phase 3 -- Polish (P2)

- [x] P2 RFP-F-5 Open up the language picker
- [x] P2 RFP-F-7 Per-item cancellation
- [x] P2 RFP-F-8 HEVC / AV1 output codec dropdown
- [x] P2 RFP-F-9 Pre-batch ETA probe
- [x] P2 RFP-F-10 `--preset NAME` CLI flag
- [x] P2 RFP-EI-1 Percentile fallback detection thresholds (was Otsu in plan; percentile is more robust on low-contrast)
- [x] P2 RFP-EI-5 dataclasses-driven ProcessingConfig persistence (shipped with B-1)
- [x] P2 RM-21 TransNetV2 deep scene-cut detector (opt-in via transnetv2 + VSR_TRANSNETV2)
- [x] P2 RM-24 Vertical-text mode
- [ ] P2 RM-39 INT8 quantisation of OCR detector
- [x] P2 RM-40 Batched LaMa inference (opt-in via VSR_LAMA_BATCH=1)

## Phase 4 -- Tests + observability (P0 for T-1, P1 elsewhere)

- [x] P0 RFP-T-1 Synthetic regression harness (8 deterministic clips; CC0 sourcing follow-up tracked)
- [x] P1 RFP-T-2 End-to-end test with synthesised clip
- [x] P1 RFP-T-3 OCR cascade selection test
- [ ] P2 RFP-T-4 GUI smoke test (Tk update_idletasks walk) -- needs headless display
- [ ] P2 RM-52 Opt-in crash reporting via GlitchTip (deferred -- privacy review needed)

## Phase 5 -- Larger bets (P1 long-horizon)

- [ ] P1 RFP-L-1 Split backend/processor.py into 7 modules
- [x] P1 RFP-L-2 Plugin architecture for inpainters (concretises RM-81)
- [x] P1 RFP-L-3 docs/architecture.md contributor map
- [x] P1 RM-25 LaMa via ONNX Runtime (opt-in via VSR_LAMA_ONNX)
- [x] P1 RM-26 MI-GAN fast mode (opt-in via VSR_MIGAN_ONNX)
- [x] P1 RM-27 Whisper fallback when OCR returns no boxes inside a speech span
- [x] P1 RM-32 PySceneDetect-backed scene splitter (opt-in)

## Phase 6 -- Polish (P3)

- [x] P3 RFP-EI-3 Detection threshold slider direction + rename
- [x] P3 RFP-EI-4 Live preview worker-side throttle (PIL conversion already off-thread)
- [x] P3 RFP-EI-6 Output-path collision guard already covered via `_make_unique_output_path`
- [ ] P3 RFP-EI-7 TikTok preset auto-band tuning (needs real-source A/B)

## Phase 7 -- Distribution + accessibility (RM, longer-horizon)

- [ ] P1 RM-50 Code-signed release via Azure Trusted Signing
- [ ] P1 RM-51 NSIS or MSI installer
- [ ] P2 RM-58 Drag-drop onto app icon (depends on RM-51)
- [x] P2 RM-95 Screen-reader / UIA announce scaffold (batch-complete + fatal-error notifications)
- [x] P2 RM-96 High-contrast theme variant
- [x] P2 RM-97 GUI localisation scaffold (gettext binding + vsr.pot template)
- [ ] P2 RM-98 Right-to-left UI support (depends on RM-97)

## Phase 8 -- Detection / inpainting future (P2 research-heavy)

- [x] P2 RM-22 Florence-2 / Qwen2.5-VL detector (opt-in via VSR_VLM_OCR)
- [x] P2 RM-23 PaddleOCR-VL 0.9B detector tier (opt-in via VSR_VLM_OCR=paddleocr-vl)
- [x] P2 RM-33 Pre-detect denoise (FastDVDnet + cv2 NLM fallback, opt-in)
- [ ] P2 RM-34 Proxy-file workflow
- [x] P2 RM-42 Manga / anime mode (lang="manga" routes to manga-ocr + comic-text-detector when installed)
- [ ] P2 RM-43 Karaoke / animated subtitle tracking
- [ ] P2 RM-45 WhisperX-aided chyron classifier
- [x] P2 RM-54 alias closed with RFP-T-1 (synthetic harness)
- [ ] P3 RM-55 Community edge-case corpus (non-code)

## Phase 9 -- Heavier inpainters (P3 v4.x targets)

- [x] P3 RM-59 Real ProPainter scaffold (opt-in via VSR_PROPAINTER_REAL)
- [x] P3 RM-60 DiffuEraser scaffold (opt-in via VSR_DIFFUERASER)
- [x] P3 RM-61 Wan2.1-VACE scaffold (opt-in via VSR_VACE)
- [x] P3 RM-62 VideoPainter scaffold (opt-in via VSR_VIDEOPAINTER)
- [x] P3 RM-63 CoCoCo scaffold (opt-in via VSR_COCOCO)
- [x] P3 RM-64 EraserDiT scaffold (opt-in via VSR_ERASERDIT)
- [x] P3 RM-65 FloED scaffold (opt-in via VSR_FLOED)

## Phase 10 -- Mask / segmentation (P3 v4.x)

- [ ] P3 RM-66 SAM 2 mask refinement
- [ ] P3 RM-67 SAM 3 text-prompt segmentation
- [ ] P3 RM-68 MatAnyone 2 (video matting)
- [ ] P3 RM-69 CoTracker3 point tracking

## Phase 11 -- Acceleration (P3 v4.x)

- [ ] P3 RM-70 TensorRT inpainter path
- [x] P3 RM-71 PyNvVideoCodec hardware decode (opt-in via VSR_PYNVVIDEOCODEC)
- [x] P3 RM-72 RIFE-interpolated fast mode (opt-in via practical-rife)

## Phase 12 -- Format support (P3 v4.x)

- [x] P3 RM-73 10-bit / HDR pipeline (color-metadata passthrough; 16-bit pixel path tracked as follow-up)
- [x] P3 RM-74 AV1 + VP9 ingest validation (codec banner) + AV1 egress via RFP-F-8
- [ ] P3 RM-75 VapourSynth bridge
- [x] P3 RM-76 NLE round-trip (EDL/XML)

## Phase 13 -- Post-processing (P3 v4.x)

- [ ] P3 RM-77 SeedVR2 one-step video restoration
- [x] P3 RM-78 Real-ESRGAN output upscale (opt-in via realesrgan-ncnn-vulkan)
- [x] P3 RM-79 SwinIR restoration pass (opt-in via swinir/realsr-ncnn-vulkan binary)
- [x] P3 RM-80 Film-grain re-synthesis (ffmpeg noise filter, opt-in)

## Phase 14 -- v5+ research bench (deferred)

ROADMAP items 81-100 stay deferred (plugin marketplace, multi-GPU,
REST server, Tauri shell, translation pipeline, streaming mode,
WebGPU, font-aware inpainting, logo mode, AI chat, performance
dashboard, mobile ports). All tracked in
[ROADMAP.md "Under consideration"](./ROADMAP.md#under-consideration-v5);
not duplicated here.

---

## Execution rules

- Each item closes with: code + test + CHANGELOG entry + TODO checkbox.
- Pure-ASCII source preserved (no em-dashes, no smart quotes).
- No new mandatory dependencies; opt-in installs are fine.
- Backward-compatible settings.json (bump `vsr_settings_format` only when
  semantics change).
- Each batch is one commit with conventional-commit subject.
- Push to `SysAdminDoc/VideoSubtitleRemover` is pre-authorized (see
  global CLAUDE.md).
