# Research - Video Subtitle Remover Pro

## Executive Summary
Video Subtitle Remover Pro is a Windows-first local desktop and CLI tool for removing burned-in subtitles, text watermarks, and logo-like overlays from videos and images. Verified: the project is already strong in offline/privacy-preserving processing, OCR/inpaint fallback breadth, FFmpeg capability reporting, release evidence, quality gates, support bundles, timed regions, and deterministic reference clips. The highest-value direction is still trust and recovery: ship current artifacts, harden the frozen-build path, make setup repair unattended, preload/report CUDA runtime state before ONNX sessions, turn corpus intake into enforceable data, and continue splitting the largest GUI/backend orchestration surfaces.

Top opportunities in priority order:
- Verified: publish current `v3.17.3` release artifacts; public GitHub releases still show `v3.17.1` as latest while local `APP_VERSION`, README, and CHANGELOG are at `3.17.3`.
- Verified: add PyInstaller multiprocessing/frozen-build guards; `VideoSubtitleRemover.py` does not call `multiprocessing.freeze_support()` before imports, `build_exe.bat` passes no runtime hook, and `VideoSubtitleRemoverPro.spec` has `runtime_hooks=[]`.
- Verified: make setup repair non-interactive; `setup.py` prompts on an existing `venv`, while launchers can leave users with a broken environment and no automated repair path.
- Verified: preload and record ONNX Runtime CUDA DLL state before CUDA provider sessions; dependency status detects `preload_dlls`, but `backend/inpainters_onnx.py` creates sessions without invoking it.
- Verified: detect conflicting OpenCV wheel installs and imported `cv2` ownership before runtime drift.
- Verified: add edge-case issue intake plus manifest license validation for real redistributable clips.
- Verified: add source-aware output quality preflight warnings before long runs.
- Verified: record per-stage timings in batch reports and support bundles.
- Verified: extend accessibility and gettext coverage through the main GUI.
- Verified: extract focused controllers from `gui/app.py` and the long media state machine.

## Product Map
- Core workflows: queue files/folders, choose cleanup preset/mode/device/language, set automatic or timed manual masks, preview masks/inpainted frames, run batch cleanup, review quality gates, export SRT/mask/NLE/cache/support artifacts.
- User personas: Windows video editors, archivists/restoration users, privacy-sensitive local users, batch operators, and CLI users who need reproducible diagnostics.
- Platforms and distribution: Python 3.10-3.13, Tkinter GUI, CLI, Windows launchers, local PyInstaller/NSIS build scripts, winget-ready metadata, optional CUDA/DirectML/OpenVINO/ONNX/Paddle/RapidOCR paths.
- Key integrations and data flows: FFmpeg/FFprobe ingest/encode/profiles, OpenCV/Pillow media I/O, RapidOCR/PaddleOCR/EasyOCR/Surya/OpenCV detection, TBE/LaMa/registered inpainting, JSON settings/presets/checkpoints/reports, redacted support bundles, release SBOM/advisories.

## Competitive Landscape
- YaoFANGUK/video-subtitle-remover: ships large Windows/macOS packages and has active demand around GPU packaging, install clarity, pause/progress saving, dynamic watermark quality, bitrate/blur, and system requirements. Learn from its package matrix and issue signals; avoid unresolved startup/install ambiguity.
- VideOCR and RapidVideOCR: specialize in hard-subtitle extraction to SRT, language breadth, crop/time controls, and isolated install paths. Learn from explicit extraction/review controls; avoid making cloud OCR the default.
- Subtitle Edit: mature subtitle correction/OCR workflow with broad format literacy and review UX. Learn from review/edit affordances for extracted text; avoid turning VSR into a general subtitle editor before cleanup/recovery gaps are closed.
- IOPaint: strong local inpainting/model visibility and optional backend management. Learn from clear model state and batch image cleanup; avoid general image-editing drift.
- ProPainter, VACE, SAM2, and related research projects: strong temporal/mask quality but heavy weights, license, and adapter-trust constraints matter. Keep these as opt-in, hash/path-gated research adapters.
- PaddleOCR and RapidOCR: active OCR stacks with PP-OCRv6, OpenVINO, and packaging churn. Keep dependency caps, provider reporting, model-file evidence, and fallback messaging explicit.
- Media.io/AniEraser-style commercial tools: sell one-click removal, manual brush workflows, browser/mobile access, and quality claims. Learn confidence-building previews and quality language; avoid upload-first workflows.

## Security, Privacy, and Reliability
- Verified: GitHub latest release is `v3.17.1`, but local docs/config are `3.17.3`; this weakens installer trust after prior public issues about false-positive trust, missing RapidOCR packaged data, and broken GUI basics.
- Verified: `VideoSubtitleRemover.py` lacks an early `multiprocessing.freeze_support()` call, `build_exe.bat` has no `--runtime-hook`, and `VideoSubtitleRemoverPro.spec` has `runtime_hooks=[]`; PyInstaller documents recursive spawn loops for frozen apps that use multiprocessing through dependencies.
- Verified: PyInstaller also warns that frozen Windows apps can taint child-process DLL search paths; VSR shells out to FFmpeg/FFprobe extensively, so release smoke should cover sanitized external-process launches, not only GUI construction.
- Verified: `setup.py` still asks `Recreate? (y/N)` when `venv` exists; this can block launcher-driven setup or leave a stale environment unrepaired.
- Verified: `backend/dependency_caps.py` detects whether `onnxruntime.preload_dlls` exists, but ONNX session creation in `backend/inpainters_onnx.py` and `backend/inpainters/lama.py` does not preload CUDA/cuDNN DLLs before `CUDAExecutionProvider`.
- Verified: OpenCV/libpng mitigation and safe PNG routing exist; remaining OpenCV risk is install drift between `opencv-python`, `opencv-contrib-python`, and headless variants.
- Verified: `docs/edge_case_corpus.md` requires CC0/public-domain real clips, settings, screenshots, and license declarations, but `.github/ISSUE_TEMPLATE/` only has bug and feature forms.
- Verified: quality gates produce review/retry suggestions, but no preflight warns when selected codec/quality is likely below the source before a long run.

## Architecture Assessment
- `gui/app.py` is about 7,556 lines and owns onboarding, queue, preview, settings, dialogs, cache import/export, support bundle, quality review, region selector, and processing orchestration; controller extraction remains the safest risk reducer.
- `backend/processor.py`, `backend/cli.py`, and `backend/io.py` still hold long-running media orchestration and subprocess edges; stage timing and external-process smoke evidence would make failures diagnosable.
- `VideoSubtitleRemoverPro.spec` is stale relative to `build_exe.bat`: it omits optional imports/data collection, release evidence arguments, and runtime hooks, so contributors can build a weaker package by using the spec directly.
- `backend/a11y.py` provides UIA announcements and `gui/widgets.py` has focusable custom controls, but major dialogs and Canvas widgets need broader state/focus tests.
- `backend/i18n.py` and `locale/vsr.pot` exist, but most GUI strings are not wrapped and there is no catalog smoke proving a translated main screen.
- Test gaps: frozen-build multiprocessing/runtime-hook evidence, ONNX CUDA preload behavior, non-interactive setup repair, OpenCV wheel-conflict diagnosis, real-clip intake schema, stage-timing aggregation, output-quality preflight decisions, i18n extraction, and accessibility/focus traversal.
- Coverage note: security, accessibility, i18n/l10n, observability, testing, docs, distribution/packaging, plugin/adapter ecosystem, mobile, offline resilience, multi-user service, migration paths, and upgrade strategy were either promoted to ROADMAP or rejected below.

## Rejected Ideas
- Cloud upload/API cleanup from Media.io/AniEraser-style tools: conflicts with the local privacy model and current offline architecture.
- Default bundled ProPainter, DiffuEraser, CLEAR, SEDiT, CoCoCo, SAM 3, ROSE, or MiniMax-Remover paths: blocked by non-commercial terms, missing weights/code, unverified assets, or unavailable releases.
- Google Lens/cloud hybrid OCR from VideOCR-style workflows: useful for extraction accuracy but contradicts local/offline defaults.
- Full GUI framework rewrite: the Tk app is large but working; controller extraction has lower regression risk.
- General plugin marketplace: the registry/adapter boundary already exists; marketplace work should wait until release and adapter trust are quieter.
- Mobile app or hosted multi-user service: current leverage is Windows desktop reliability, artifact trust, and local batch recovery.
- GitHub Actions/Dependabot-style automation: project rules keep builds, tests, dependency updates, and releases local.

## Sources
### Project and Issues
- https://github.com/SysAdminDoc/VideoSubtitleRemover
- https://api.github.com/repos/SysAdminDoc/VideoSubtitleRemover/releases?per_page=5
- https://api.github.com/repos/SysAdminDoc/VideoSubtitleRemover/issues?state=all&per_page=20

### OSS and Adjacent Tools
- https://github.com/YaoFANGUK/video-subtitle-remover
- https://api.github.com/repos/YaoFANGUK/video-subtitle-remover/releases?per_page=5
- https://github.com/YaoFANGUK/video-subtitle-remover/issues/200
- https://github.com/YaoFANGUK/video-subtitle-remover/issues/228
- https://github.com/YaoFANGUK/video-subtitle-remover/issues/231
- https://github.com/timminator/VideOCR
- https://github.com/SWHL/RapidVideOCR
- https://github.com/SubtitleEdit/subtitleedit
- https://github.com/Sanster/IOPaint
- https://github.com/sczhou/ProPainter
- https://github.com/facebookresearch/sam2
- https://github.com/ali-vilab/VACE

### Dependencies, Platform, and Security
- https://pyinstaller.org/en/stable/common-issues-and-pitfalls.html
- https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- https://github.com/PaddlePaddle/PaddleOCR/releases/tag/v3.7.0
- https://github.com/RapidAI/RapidOCR/releases/tag/v3.9.0
- https://github.com/opencv/opencv-python/issues/1186
- https://nvd.nist.gov/vuln/detail/CVE-2026-22801
- https://nvd.nist.gov/vuln/detail/CVE-2025-32434
- https://pillow.readthedocs.io/en/stable/releasenotes/12.2.0.html

### Commercial and Research
- https://anieraser.media.io/remove-subtitles-from-video.html
- https://huggingface.co/charlesw09/CLEAR-mask-free-video-subtitle-removal
- https://arxiv.org/abs/2605.14894
- https://arxiv.org/abs/2503.05639

## Open Questions
None that block prioritization. Real-clip selection still needs license validation during implementation.
