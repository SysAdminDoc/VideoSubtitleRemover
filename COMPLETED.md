# Video Subtitle Remover Pro -- Completed Work

This file is the durable shipped-work summary. Detailed release notes stay in
[CHANGELOG.md](./CHANGELOG.md); future planning stays in
[ROADMAP.md](./ROADMAP.md).

## Current release

- **v3.15.0, released 2026-05-25.** Backlog-drain release with the GUI smoke
  test, opt-in GlitchTip crash reporting, NSIS installer, Azure Trusted
  Signing workflow, Explorer "Send to VSR" verb, community edge-case corpus
  guide, proxy workflow, karaoke optical-flow helper, WhisperX helper,
  VapourSynth bridge, RTL scaffold, and a broad set of opt-in research
  adapters.
- **Unreleased after v3.15.0.** `backend/processor.py` was split into focused
  modules while preserving the legacy import and `python -m backend.processor`
  entry points. The unreleased changelog records the full module map and the
  164-test verification claim from that split.

## Product surface shipped

- Windows-first Tk desktop app with drag-and-drop queueing, per-item settings,
  dark design-token UI, live preview, before/after review, mask preview, and
  persisted settings.
- OCR cascade through RapidOCR, PaddleOCR, Surya opt-in, EasyOCR, OpenCV
  fallback, plus optional VLM and manga/anime detector tiers.
- Inpainting paths for STTN/TBE, LaMa, ProPainter hybrid, AUTO routing, ONNX
  LaMa, MI-GAN, and opt-in heavier research inpainter scaffolds.
- Video workflow support for lossless intermediate writing, audio passthrough,
  multi-track audio, loudness normalisation, HEVC/AV1 output, HDR metadata
  passthrough, frame-sequence and VapourSynth ingest, NLE sidecars, proxy
  previews, crash-resume, and JSON logs.
- Quality and testing surfaces including masked-region PSNR/SSIM, quality
  sheets, synthetic reference clips, GUI smoke coverage, TikTok preset A/B
  coverage, and edge-case corpus contributor docs.
- Distribution surfaces including PyInstaller release flow, NSIS installer,
  optional Azure code signing, installer shell verbs, and dependency audit
  coverage.

## Retired planning inputs

- The 2026-05-25 audit now lives at
  [docs/archive/research/RESEARCH_FEATURE_PLAN_2026-05-25.md](./docs/archive/research/RESEARCH_FEATURE_PLAN_2026-05-25.md).
- The exhausted active checklist now lives at
  [docs/archive/roadmap/TODO_legacy_2026-05-25.md](./docs/archive/roadmap/TODO_legacy_2026-05-25.md).

Both files are historical evidence, not active work queues.
