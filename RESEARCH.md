# Research — Video Subtitle Remover Pro

## Executive Summary

Video Subtitle Remover Pro is a Windows-first, local-first Python/tkinter desktop app for removing hardcoded subtitles with a multi-engine OCR cascade, mask preview/refinement, FFmpeg-backed encode/remux workflows, and quality/support evidence. Its strongest current shape is trustable local batch processing: codec breadth, recovery controls, quality reports, and release verification are stronger than most OSS peers and avoid the privacy tradeoffs of SaaS tools. The highest-value direction is to harden provenance, queue recovery, workspace preflight, and replayability while selectively adopting OCR/runtime advances that preserve the local-first model.

Top opportunities in priority order:

1. Extend one remote-model provenance policy across every opt-in HF/torchhub/research adapter.
2. Add GUI queue autosave/crash restore so desktop batches recover like CLI processing does.
3. Add a temp/work directory selector with free-space preflight for FFV1 intermediates, caches, and outputs.
4. Add "copy replayable CLI command" from queue items for support, reruns, and reproducibility.
5. Add a one-shot acceleration/codec self-test for DirectML/CUDA/CPU and FFmpeg feature proof.
6. Require license/hash manifests for real reference clips before expanding the regression corpus.
7. Add an OCR language probe for manual regions, backed by RapidOCR/PaddleOCR current multilingual models.
8. Spike Windows ML as the future AMD/Intel inference path while keeping DirectML as the proven default.
9. Keep research-model adapters opt-in and license-gated; do not bundle ProPainter-family or noncommercial defaults.

## Product Map

- Core workflows: import video/images; choose automatic, manual, or soft-subtitle flows; preview/refine masks; process through AUTO/TBE/LaMa-ONNX/OpenCV/Torch routes; review quality reports, batch summaries, support bundles, and restored checkpoints.
- User personas: Windows video editors, archive/restoration users with large batches, nontechnical GUI users, technical CLI/evidence users, and privacy-sensitive users who reject cloud upload.
- Platforms and distribution: Python/tkinter source; Windows `.bat`/PowerShell launchers; PyInstaller/NSIS release path; GitHub Actions build, SBOM, pip-audit, release verification, and winget preparation; limited source-install use on Linux/macOS.
- Key integrations and data flows: FFmpeg/FFprobe decode/encode/remux; RapidOCR/PaddleOCR/Surya/EasyOCR/OpenCV/VLM detection; ONNX Runtime DirectML/CUDA/CPU and PyTorch optional acceleration; JSON settings/checkpoints/reports; cache inventory and scrubbed support bundles.

## Competitive Landscape

### YaoFANGUK/video-subtitle-remover
- Does well: large user base, Apache-2.0 upstream, simple hard-subtitle removal, custom subtitle areas, Docker and downloadable CPU/DirectML/CUDA package matrix.
- Learn from: packaging clarity for users who cannot assemble Python/GPU stacks, and explicit progress/pause/time-range requests in upstream issues.
- Avoid: weaker evidence/recovery surface than VSR Pro; no comparable quality reports, support bundle, or multi-engine local fallback ladder.

### GhostCut / JollyToday
- Does well: automatic text/subtitle/logo removal, batch scale, API workflow, and adjacent translation/dubbing product packaging.
- Learn from: target-mode clarity and integration-ready job semantics.
- Avoid: cloud-only processing and credit/subscription economics, which conflict with VSR Pro's privacy and free local desktop stance.

### EchoSubs
- Does well: offline desktop positioning, simple import/mask/process workflow, MP4/MKV/AVI/MOV support, and low-friction paid packaging.
- Learn from: clear offline value proposition and plain recovery expectations for nontechnical users.
- Avoid: opaque quality claims without open benchmarks, CLI reproducibility, or support evidence.

### videowipe
- Does well: PyPI/API/CLI distribution, natural-language target intent, preview/confirm, Docker CPU/GPU images, and an external model command contract.
- Learn from: replayable command surface, intent-to-config helper, and process-isolated experimental model bridge.
- Avoid: GPL-by-default pressure and ProPainter-family licensing/VRAM assumptions as bundled defaults.

### IOPaint
- Does well: one-click installers, local web UI, CPU/GPU/Apple Silicon support, model manager, and plugins for SAM, RemoveBG, RealESRGAN, and GFPGAN.
- Learn from: explicit model/plugin lifecycle and user-visible dependency status.
- Avoid: drifting into a general image editor; VSR Pro should stay video-subtitle/workflow focused.

### WatermarkRemover-AI
- Does well: Florence-2 + LaMA pipeline, fade mask handling, video support, batch mode, preview, themes, and multilingual UI.
- Learn from: moving/fading watermark UX and community demand for portable installs, GPU diagnostics, and PyPI packaging.
- Avoid: dependency failures and ambiguous GPU state; VSR Pro should prove available providers before users start long jobs.

### RapidVideOCR / VideOCR
- Does well: hard-subtitle extraction to SRT/ASS/TXT, local PaddleOCR, CLI, hybrid cloud OCR option, and active user requests for queue autosave, temp-dir selection, copy-CLI-command, profiles, timestamps, and auto language selection.
- Learn from: supportability features around reproducible commands and persistent queues.
- Avoid: cloud OCR fallback as a default; VSR Pro's extraction/removal path should remain local unless users explicitly install opt-in tools.

### InpaintDelogo / VideoSubFinder / Subtitle Edit workflow
- Does well: power-user mask adjustment, OCR extraction loops, and dynamic subtitle/logo cleanup patterns.
- Learn from: dynamic mask preview/review and text-correction loops remain valuable even when AI inpainting improves.
- Avoid: forcing users through slow multi-tool/manual pipelines when VSR Pro can keep detection, review, inpaint, and evidence in one desktop flow.

## Security, Privacy, and Reliability

- Verified: `requirements.txt` already pins around recent OpenCV/Pillow/RapidOCR/PaddleOCR risk areas, and `.github/workflows/build.yml` runs pip-audit, SBOM generation, libpng advisory checks, signing/verification, and release evidence.
- Verified: `Roadmap_Blocked.md` correctly keeps the OpenCV/libpng CVE floor blocked until a fixed wheel is available; do not claim that risk closed until the dependency ecosystem moves.
- Verified risk: `backend/remote_model_policy.py` protects Florence-2 and CoTracker-style remote-code paths, but optional remote model provenance is not centralized for every enabled heavy adapter in `backend/ocr_vlm.py` and `backend/segmentation.py` (Qwen2.5-VL, PaddleOCR-VL, SAM2, MatAnyone-family probes). A single policy should record local path, pinned revision, license status, and release-verification evidence.
- Verified risk already on ROADMAP: `_preview_detector` concurrency, main-thread preview reads, `_LosslessIntermediateWriter.write()` blocking pipe writes, stale `STATUS_UI` colors after theme mutation, GUI normalization gaps, and legacy adapter hash holes are already captured; do not duplicate them.
- Verified reliability gap: GUI queue state is volatile even though CLI processing has checkpoint/resume concepts; upstream and VideOCR issue traffic shows progress save and queue autosave are recurring user needs.
- Verified reliability gap: FFV1 lossless intermediates, model caches, and output encodes can exceed default system-drive free space; current workflows need an explicit work-dir selector and preflight estimate before starting long jobs.
- Verified privacy stance: local processing, no telemetry by default, and scrubbed support bundles align with user trust; cloud API integrations remain rejected unless future users explicitly opt into separate tooling.

## Architecture Assessment

- The module split is healthy: GUI state lives in `gui/`, pipeline/codec/model work lives in `backend/`, and `docs/architecture.md` now maps the main boundaries. The next improvements should consolidate existing surfaces, not rewrite the stack.
- Boundary improvement: centralize model provenance in `backend/remote_model_policy.py` and route optional detector/segmenter/inpainter adapters through it before loading files or remote refs.
- Boundary improvement: add a small persisted GUI queue/session layer around `QueueItem` in `gui/config.py` and orchestration in `gui/app.py`, separate from processor checkpoints.
- Boundary improvement: make temp/work/cache/output preflight a backend service used by both CLI and GUI, likely spanning `backend/io.py`, `backend/processor.py`, `backend/cache_inventory.py`, and settings.
- Refactor candidate: add a replayable config/CLI-command builder so GUI queue items, support bundles, and bug reports share one serialization path instead of hand-reconstructing arguments.
- Test gaps: no real redistributable reference-clip corpus with license/hash manifests; no queue restore tests; no CLI-command round-trip test from GUI config; no one-shot DirectML/CUDA/FFmpeg feature self-test.
- Documentation gaps: release/support docs should show how to run the planned self-test, how to select a work directory, and how to attach a replay command plus scrubbed support bundle.
- Category review: security, accessibility, i18n, observability, testing, docs, distribution, plugin ecosystem, mobile, offline/resilience, multi-user, migration, and upgrade strategy were considered. Accessibility/i18n/mobile/server/plugin ideas already exist as later/under-consideration roadmap lanes; multi-user remains intentionally rejected for a single-user desktop tool.

## Rejected Ideas

- Cloud/SaaS removal or OCR integration (GhostCut, Media.io, HitPaw, Vmake, Creatok): conflicts with local-first privacy and no-telemetry stance.
- Docker-first distribution (YaoFANGUK, videowipe): useful for some Linux/GPU users but explicitly not the dominant Windows desktop release path for this repo.
- Bundled Real ProPainter, DiffuEraser, MatAnyone2, or CoTracker3 defaults: license, noncommercial, dependency, or VRAM constraints make them opt-in/research-only, not shipped defaults.
- Full image-editor plugin marketplace (IOPaint): strong adjacent design, but too broad for a subtitle-removal desktop app.
- REST/multi-user server mode now (GhostCut API, videowipe API): integration value exists, but current architecture and roadmap prioritize local single-user reliability first.
- Mobile ports now (commercial web tools): packaging FFmpeg, OCR, ONNX/PyTorch, and video files on mobile is a separate product; keep as later investigation only.
- PyPI full-GUI distribution as a priority (videowipe, WatermarkRemover-AI community asks): useful for developers, but the current user base is better served by reproducible CLI support plus Windows installer quality.
- Mask-free subtitle-removal models as near-term defaults (CLEAR, SEDiT): promising research, but current public paths depend on large generative stacks and need licensing, speed, and quality validation before roadmap promotion.

## Sources

### OSS and Direct Competitors

- https://github.com/YaoFANGUK/video-subtitle-remover/blob/main/README_en.md
- https://github.com/KKenny0/videowipe
- https://github.com/JollyToday/GhostCut_Remove_Video_Text
- https://github.com/JollyToday/GhostCut-auto_video_translation
- https://github.com/D-Ogi/WatermarkRemover-AI/blob/main/README.md
- https://github.com/Sanster/IOPaint
- https://github.com/Purfview/InpaintDelogo
- https://github.com/SWHL/RapidVideOCR
- https://github.com/timminator/VideOCR

### Commercial and Community Signal

- https://jollytoday.com/subtitle-removal/
- https://www.echosubs.com/remove-hardcoded-subtitles-offline
- https://anieraser.media.io/remove-subtitles-from-video.html
- https://online.hitpaw.com/watermark-remover-online.html
- https://forum.videohelp.com/threads/418629-Which-tool-has-the-best-accuracy-for-extracting-hardsubs-from-video
- https://github.com/SubtitleEdit/subtitleedit/discussions/9562
- https://www.reddit.com/r/StableDiffusion/comments/1nko1w4/open_source_models_for_video_inpainting_removing/

### Dependencies, Platform, and Security

- https://github.com/RapidAI/RapidOCR/releases
- https://github.com/PaddlePaddle/PaddleOCR/releases
- https://github.com/opencv/opencv/wiki/OpenCV-5-DNN-Benchmarks
- https://onnxruntime.ai/docs/get-started/with-windows.html
- https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview
- https://ayosec.github.io/ffmpeg-filters-docs/8.0/Filters/Audio/whisper.html
- https://github.com/opencv/opencv-python/issues/1186
- https://nvd.nist.gov/vuln/detail/CVE-2025-32434
- https://securitylab.github.com/advisories/GHSL-2025-057_OpenCV/

### Research and SOTA

- https://huggingface.co/charlesw09/CLEAR-mask-free-video-subtitle-removal
- https://arxiv.org/html/2605.14894v1
- https://arxiv.org/html/2503.05639v2
- https://github.com/lixiaowen-xw/diffueraser
- https://rose2025-inpaint.github.io/

## Open Questions

- Needs live validation: which CC0/public-domain clips can be redistributed in `tests/clips/` for #54 without legal friction?
- Needs live validation: does Windows ML expose a Python-packagable ONNX inference path suitable for VSR Pro today, or should DirectML remain the only AMD/Intel GPU path until official Python support matures?
- Needs live validation: which Windows FFmpeg distribution available to target users includes both `libvvenc` and `--enable-whisper`, so VVC/Whisper support can move from optional evidence to installer-default claims?
