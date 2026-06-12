# Research - Video Subtitle Remover Pro

## Executive Summary

Video Subtitle Remover Pro is a mature Windows-first Python/tkinter desktop
tool for removing burned-in subtitles and text-like overlays with an offline OCR
cascade, Temporal Background Exposure, LaMa refinement, FFmpeg muxing, soft
subtitle remuxing, quality reports, and a recent GUI/backend module split.
The highest-value direction is trust hardening around the image/dependency
surface: the project now pins Pillow at `>=12.1.1`, but newer April-May 2026
Pillow advisories require `>=12.2.0`; this matters because the app accepts
image inputs and uses Pillow for preview/region workflows. Top opportunities:
1. raise the Pillow floor everywhere and test the release/install pins;
2. remove global keyboard accelerators that conflict with local product rules;
3. expose FFmpeg Whisper VAD controls so speech-derived masking is less coarse;
4. prune stale `torch_directml` packaging leftovers after the ONNX Runtime
DirectML migration; 5. centralize media extension allowlists; 6. add a cache
inspector for model/proxy/TensorRT/Whisper storage; 7. refresh existing
PaddleOCR-VL planning from 1.5 to 1.6 before implementation.

## Product Map

- Core workflows: queue videos/images; detect burned-in text; inpaint masked
  regions; remux audio/subtitle streams; write batch summaries and optional
  quality reports.
- User personas: Windows creators repurposing social video, archivists cleaning
  local files, localizers extracting/replacing captions, and batch operators
  running unattended folders.
- Platforms and distribution: Windows desktop source checkout, PyInstaller
  onedir build, NSIS installer, GitHub Actions release workflow, optional
  winget submission, CUDA/DirectML/CPU paths.
- Key integrations and data flows: FFmpeg/ffprobe for media probing/remuxing,
  RapidOCR/PaddleOCR/Surya/EasyOCR for detection, OpenCV/Pillow for image I/O
  and preview, ONNX Runtime DirectML for AMD/Intel acceleration, optional
  external adapters gated by local environment variables and manifests.

## Competitive Landscape

- **YaoFANGUK/video-subtitle-remover** does broad local subtitle/text removal,
  Docker/CUDA variants, and upstream community discovery well. Learn from its
  cross-platform packaging matrix and issue volume; avoid hardware-specific
  install complexity and weaker release verification.
- **VideOCR** focuses on hard-subtitle extraction with a simple GUI and 200+
  language claim. Learn from language breadth and OCR-first workflow clarity;
  avoid splitting extraction/removal into a separate user journey.
- **GhostCut** combines auto OCR and video inpainting for subtitle, watermark,
  logo, and moving text removal. Learn from scenario-specific positioning and
  language/format landing pages; avoid cloud-first processing.
- **IOPaint** presents model choice and object-removal UX cleanly. Learn from
  model-zoo discoverability and before/after review patterns; avoid image-only
  assumptions in a temporal video pipeline.
- **HitPaw, Media.io, AniEraser, CapCut, GiliSoft** sell "no blur" AI cleanup,
  browser/mobile access, preview/export steps, and subscription packaging.
  VSR should compete on offline privacy, no upload caps, batch transparency,
  and measurable quality gates rather than chasing cloud convenience.
- **ProPainter, DiffuEraser, VACE, VideoPainter** remain the strongest adjacent
  video-inpainting references. Keep them opt-in/research-backed because VRAM,
  weights, licensing, and runtime remain the practical barriers for the default
  Windows binary.
- **SAM 3 / MatAnyone 2 / CoTracker3** validate promptable mask refinement and
  temporal mask propagation. Existing roadmap items cover these; the product
  fit is refinement after OCR, not default full-scene segmentation.

## Security, Privacy, and Reliability

- Verified: `requirements.txt:7-10`, `setup.py:333-337`, and
  `.github/workflows/build.yml:65-67` still allow/install `Pillow>=12.1.1`.
  GitHub advisories GHSA-wjx4-4jcj-g98j, GHSA-pwv6-vv43-88gr, and
  GHSA-whj4-6x5x-4v2j require 12.2.0 for later font, PSD, and FITS decoder
  fixes. VSR does not support PSD/PDF/FITS as user media, but the dependency
  floor and release tests should still move to `>=12.2.0`.
- Verified: supported image lists are duplicated and slightly inconsistent:
  `gui/utils.py:108-110` lacks `.tif`, while `backend/io.py:332-342` accepts
  it for image directories. Centralizing the allowlist reduces decoder and UX
  drift.
- Verified: `build_exe.bat:49-55` still probes `torch_directml` as an optional
  PyInstaller hidden import, while `setup.py:254`, requirements, and workflow
  tests have moved DirectML to ONNX Runtime. This is a stale packaging path.
- Verified: the app binds global accelerators in `gui/app.py:685-694` and the
  README advertises `Ctrl+O` at `README.md:129-130`, conflicting with the
  local product rule that this repo should not add keyboard shortcuts. Focused
  Enter/Space activation for controls should remain for accessibility.
- Verified: optional model/proxy caches can accumulate under `%APPDATA%` and
  user-level model caches (`backend/proxy_workflow.py`, `backend/tensorrt_compile.py`,
  `backend/model_hashes.py`, `backend/whisper_fallback.py`). The app exposes
  settings/log folders but not cache size, provenance, or cleanup.
- Verified: startup update checks are opt-in and use GitHub Releases; this fits
  the privacy posture because no telemetry or frame content is sent.
- Likely: FFmpeg Whisper masking will be more reliable with explicit VAD model
  controls. VSR exposes model path and queue seconds, while FFmpeg's filter also
  exposes `vad_model`, threshold, and minimum speech duration.

## Architecture Assessment

- `gui/app.py` is now the main GUI module at 5,113 lines after RM-114, with
  high-value seams around settings cards, preview/region flows, queue handling,
  and batch completion. Further extraction is already roadmapped; do not add a
  duplicate item.
- Backend split is holding: `backend/processor.py` orchestrates, while focused
  modules handle I/O, quality, config, adapters, remuxing, and optional
  detectors. New work should follow those module boundaries.
- Release and install dependency floors are spread across `requirements.txt`,
  `setup.py`, `.github/workflows/build.yml`, README snippets, and tests. Any
  dependency security fix needs all five surfaces updated in the same pass.
- UI policy drift is concentrated in `_bind_shortcuts`, README usage copy, and
  the About dialog shortcut fact. Removing global accelerators can be surgical.
- Media-type handling should be a shared constant imported by GUI file pickers,
  drag/drop validation, CLI validation, and frame-directory capture.
- Existing items already cover OpenCV 5 DNN migration, libpng mitigation,
  DirectML opset audit, reference clips, UIA, i18n, RTL, real ProPainter,
  diffusion inpainting, SAM, NLE round-trip, mobile, WebGPU, and plugin
  discovery. The additions below avoid re-listing those.
- Category audit: security gets the new Pillow P0; accessibility/i18n/RTL,
  observability, testing, docs, distribution, plugin ecosystem, mobile,
  offline/resilience, migration, and upgrade strategy are either covered by
  existing roadmap items or by the targeted additions here. Multi-user remains
  intentionally rejected for this single-user desktop app.

## Rejected Ideas

- **Default cloud cleanup API** (HitPaw/Media.io/GhostCut commercial pages) -
  conflicts with the offline/no-upload product stance.
- **Make PaddleOCR-VL-1.6 the default detector immediately** (Hugging Face) -
  useful target refresh for existing VLM roadmap items, but too heavy for the
  default Windows path.
- **Expand image support to PSD/PDF/FITS** (Pillow advisories) - adds decoder
  risk with no user evidence; keep the current common-image allowlist.
- **Adopt SAM 3 as the primary detector** (Meta/SAM 3 sources) - better as
  opt-in mask refinement after OCR; full-scene promptable segmentation is
  already represented in existing roadmap items.
- **Rewrite the GUI to Tauri/webview now** (existing roadmap #84) - not a
  root-cause fix for current security, packaging, or quality risks.
- **Filesystem plugin auto-discovery** (existing roadmap #81) - intentionally
  deferred because the in-process registry avoids loading untrusted code.

## Sources

OSS competitors and adjacent projects:
- https://github.com/YaoFANGUK/video-subtitle-remover
- https://github.com/timminator/VideOCR
- https://github.com/devmaxxing/videocr-PaddleOCR
- https://github.com/JollyToday/GhostCut_Remove_Video_Text
- https://github.com/Rats20/EraseSubtitles
- https://github.com/Sanster/IOPaint
- https://github.com/sczhou/ProPainter
- https://github.com/lixiaowen-xw/DiffuEraser
- https://github.com/TencentARC/VideoPainter
- https://github.com/suhwan-cho/awesome-video-inpainting

Commercial and community signal:
- https://online.hitpaw.com/remove-subtitles-from-video.html
- https://www.media.io/video-watermark-remover.html
- https://anieraser.media.io/remove-subtitles-from-video.html
- https://jollytoday.com/subtitle-removal/
- https://www.capcut.com/resource/remove-subtitles-from-video
- https://forum.videohelp.com/threads/418726-Is-there-a-way-to-remove-hardcoded-subtitles-without-cropping
- https://forum.videohelp.com/threads/418629-Which-tool-has-the-best-accuracy-for-extracting-hardsubs-from-video
- https://stackoverflow.com/questions/78191202/opencv-subtitle-remove-smoothly

Platform, dependency, and research:
- https://github.com/advisories/GHSA-wjx4-4jcj-g98j
- https://github.com/advisories/GHSA-pwv6-vv43-88gr
- https://github.com/python-pillow/Pillow/security/advisories/GHSA-whj4-6x5x-4v2j
- https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
- https://ayosec.github.io/ffmpeg-filters-docs/8.0/Filters/Audio/whisper.html
- https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6
- https://github.com/RapidAI/RapidOCR
- https://github.com/PaddlePaddle/PaddleOCR/releases
- https://github.com/pytorch/pytorch/issues/169929
- https://github.com/opencv/opencv-python/issues/1186
- https://github.com/facebookresearch/sam3
- https://github.com/pq-yang/MatAnyone2

## Open Questions

- Which Windows FFmpeg build should be recommended for `whisper` filter support
  once VAD options are exposed? The filter is build-flag dependent.
- Do target PaddleOCR-VL-1.6 dependencies run locally without remote-code
  execution or large surprise downloads in the planned optional adapter path?
- Should cache cleanup be GUI-only or also exposed as a CLI command for
  headless batch operators?
