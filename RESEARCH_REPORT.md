# Video Subtitle Remover Pro -- Research Report

This is the current research synthesis. The full 2026-05-25 audit is archived
at [docs/archive/research/RESEARCH_FEATURE_PLAN_2026-05-25.md](./docs/archive/research/RESEARCH_FEATURE_PLAN_2026-05-25.md).

## Product thesis

Video Subtitle Remover Pro is an offline-first Windows desktop app for removing
burned-in subtitles and text overlays from videos and images. Its strongest
position is a local pipeline that combines OCR, tracking, mask generation,
temporal background exposure, optional neural inpainting, and ffmpeg-based
audio/video muxing without requiring cloud inference.

## Research conclusions

- The highest-value gaps from the 2026-05-25 audit were closed in the v3.14.0
  and v3.15.0 backlog-drain releases: GUI/backend config drift, lossless
  intermediate quality, Surya opt-in, masked-region quality metrics, region
  selector upgrades, per-item controls, format controls, reference tests,
  installer/signing surfaces, and contributor architecture docs.
- Optional model integrations are deliberately gated. Heavy or GPL-adjacent
  detectors/inpainters remain opt-in so the default install stays MIT-clean,
  offline-first, and practical for Windows users.
- The processor split reduced the risk of future feature work by moving
  detection, tracking, I/O, quality helpers, inpainters, and CLI code into
  focused modules while preserving the old public import surface.

## Current risks

- Real-source validation is still needed for the TikTok preset `auto_band`
  decision; synthetic A/B coverage documents the no-OCR behavior but cannot
  prove real OCR clustering performance.
- Optional research adapters should continue to fail closed when dependencies,
  model weights, or command-line tools are missing.
- The signing and installer flow depends on external secrets and Windows build
  runner state, so release verification should always confirm artifacts rather
  than trusting docs alone.
- Accessibility/i18n work has scaffolding, but broad screen-reader and
  right-to-left UI behavior still needs live Windows validation.
- Processor-module changes should keep import compatibility tests in place
  because downstream scripts may still import from `backend.processor`.

## Active planning rule

Use [ROADMAP.md](./ROADMAP.md) for new work and [CHANGELOG.md](./CHANGELOG.md)
for shipped details. Do not reopen the archived audit or legacy checklist as
active planning documents.
