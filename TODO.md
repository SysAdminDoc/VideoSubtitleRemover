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
- [ ] P0 RFP-B-1 Wire 13 missing backend fields through the GUI
- [ ] P0 RFP-B-3 Quality report measures masked region only
- [ ] P0 RFP-B-5 AutoInpainter unload LaMa after N consecutive TBE batches
- [ ] P0 RFP-I-1 Lossless intermediate codec (eliminate double-encode)

## Phase 2 -- Workflow features (P1)

- [ ] P1 RFP-F-1 Region selector frame scrubbing
- [ ] P1 RFP-F-2 Multi-rectangle region drawing (surface subtitle_areas)
- [ ] P1 RFP-F-3 Inpaint-preview on a sample frame
- [ ] P1 RFP-B-4 Multi-track loudness normalisation (filter_complex)
- [ ] P1 RM-28 "Repeat last job" shortcut
- [ ] P1 RM-29 Per-file overrides surface
- [ ] P1 RM-30 A/B flicker-scrubber in preview pane
- [ ] P1 RM-31 Subtitle-area drag-refinement in-GUI (overlaps F-1/F-2)
- [ ] P1 RM-49 Model-weight hash verification on first download

## Phase 3 -- Polish (P2)

- [ ] P2 RFP-F-5 Open up the language picker
- [ ] P2 RFP-F-7 Per-item cancellation
- [ ] P2 RFP-F-8 HEVC / AV1 output codec dropdown
- [ ] P2 RFP-F-9 Pre-batch ETA probe
- [ ] P2 RFP-F-10 `--preset NAME` CLI flag
- [ ] P2 RFP-EI-1 Otsu fallback detection thresholds
- [ ] P2 RFP-EI-5 dataclasses.asdict for ProcessingConfig
- [ ] P2 RM-21 TransNetV2 deep scene-cut detector (opt-in)
- [ ] P2 RM-24 Vertical-text mode
- [ ] P2 RM-39 INT8 quantisation of OCR detector
- [ ] P2 RM-40 Batched LaMa inference

## Phase 4 -- Tests + observability (P0 for T-1, P1 elsewhere)

- [ ] P0 RFP-T-1 Reference-clip regression harness (8 synthetic clips)
- [ ] P1 RFP-T-2 End-to-end test with synthesised clip
- [ ] P1 RFP-T-3 OCR cascade selection test
- [ ] P2 RFP-T-4 GUI smoke test (Tk update_idletasks walk)
- [ ] P2 RM-52 Opt-in crash reporting via GlitchTip (deferred -- privacy review needed)

## Phase 5 -- Larger bets (P1 long-horizon)

- [ ] P1 RFP-L-1 Split backend/processor.py into 7 modules
- [ ] P1 RFP-L-2 Plugin architecture for inpainters (concretises RM-81)
- [ ] P1 RFP-L-3 docs/architecture.md contributor map
- [ ] P1 RM-25 LaMa via ONNX Runtime (opt-in, smaller install)
- [ ] P1 RM-26 MI-GAN fast mode (mobile-grade single-frame)
- [ ] P1 RM-27 Whisper fallback when OCR confidence floor is hit
- [ ] P1 RM-32 PySceneDetect-backed scene splitter (opt-in)

## Phase 6 -- Polish (P3)

- [ ] P3 RFP-EI-3 Detection threshold slider direction + rename
- [ ] P3 RFP-EI-4 Live preview off-thread PIL conversion
- [ ] P3 RFP-EI-6 Output-path collision guard on manual edit
- [ ] P3 RFP-EI-7 TikTok preset auto-band tuning (needs real-source A/B)

## Phase 7 -- Distribution + accessibility (RM, longer-horizon)

- [ ] P1 RM-50 Code-signed release via Azure Trusted Signing
- [ ] P1 RM-51 NSIS or MSI installer
- [ ] P2 RM-58 Drag-drop onto app icon (depends on RM-51)
- [ ] P2 RM-95 Screen-reader / UIA support on Windows
- [ ] P2 RM-96 High-contrast theme variant
- [ ] P2 RM-97 GUI localisation (gettext)
- [ ] P2 RM-98 Right-to-left UI support (depends on RM-97)

## Phase 8 -- Detection / inpainting future (P2 research-heavy)

- [ ] P2 RM-22 Florence-2 / Qwen2.5-VL experimental detector (opt-in)
- [ ] P2 RM-23 PaddleOCR-VL 0.9B detector tier (opt-in)
- [ ] P2 RM-33 Pre-detect denoise (FastDVDnet)
- [ ] P2 RM-34 Proxy-file workflow
- [ ] P2 RM-42 Manga / anime mode
- [ ] P2 RM-43 Karaoke / animated subtitle tracking
- [ ] P2 RM-45 WhisperX-aided chyron classifier
- [ ] P2 RM-54 (alias of RFP-T-1; closes when T-1 closes)
- [ ] P3 RM-55 Community edge-case corpus (non-code)

## Phase 9 -- Heavier inpainters (P3 v4.x targets)

- [ ] P3 RM-59 Real ProPainter (ICCV 2023 reference)
- [ ] P3 RM-60 DiffuEraser (2025 diffusion)
- [ ] P3 RM-61 Wan2.1-VACE (1.3B variant)
- [ ] P3 RM-62 VideoPainter
- [ ] P3 RM-63 CoCoCo (research bench)
- [ ] P3 RM-64 EraserDiT (track, integrate when stable)
- [ ] P3 RM-65 FloED (flow-guided efficient diffusion)

## Phase 10 -- Mask / segmentation (P3 v4.x)

- [ ] P3 RM-66 SAM 2 mask refinement
- [ ] P3 RM-67 SAM 3 text-prompt segmentation
- [ ] P3 RM-68 MatAnyone 2 (video matting)
- [ ] P3 RM-69 CoTracker3 point tracking

## Phase 11 -- Acceleration (P3 v4.x)

- [ ] P3 RM-70 TensorRT inpainter path
- [ ] P3 RM-71 PyNvVideoCodec hardware decode
- [ ] P3 RM-72 RIFE-interpolated fast mode

## Phase 12 -- Format support (P3 v4.x)

- [ ] P3 RM-73 10-bit / HDR pipeline
- [ ] P3 RM-74 AV1 + VP9 ingest/egress (overlaps RFP-F-8)
- [ ] P3 RM-75 VapourSynth bridge
- [ ] P3 RM-76 NLE round-trip (EDL/XML)

## Phase 13 -- Post-processing (P3 v4.x)

- [ ] P3 RM-77 SeedVR2 one-step video restoration
- [ ] P3 RM-78 Real-ESRGAN output upscale
- [ ] P3 RM-79 SwinIR restoration pass
- [ ] P3 RM-80 Film-grain re-synthesis

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
