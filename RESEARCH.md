# Research — Video Subtitle Remover Pro

## Executive Summary
Video Subtitle Remover Pro is a Windows-first, local/offline desktop and CLI workstation for removing hard subtitles, text watermarks, and static-logo artifacts from videos/images. Verified: the project is already strongest in privacy-preserving local processing, a broad OCR/inpaint backend matrix, FFmpeg capability reporting, release evidence, quality gates, support bundles, manual timed regions, and deterministic reference clips. The highest-value direction is to convert that technical depth into user trust and recoverability: publish current release artifacts for `3.17.3`, detect dependency drift before imports fail, make real-world clip intake enforceable, expose stage timing, extend accessibility/i18n scaffolds, and reduce risk in the largest UI/backend modules.

Top opportunities in priority order:
- Verified: publish a current `v3.17.3` release because local docs/config are ahead of the latest GitHub release `v3.17.1`.
- Verified: detect conflicting OpenCV wheels before `cv2` imports a different distribution than release/support evidence reports.
- Verified: add an edge-case issue template plus real-clip manifest validation; `docs/edge_case_corpus.md` asks for GitHub Discussions, but the repo only has bug/feature issue forms.
- Verified: add source-aware output-quality warnings; competitor users report blur/bitrate loss, and VSR has quality metrics but no preflight bitrate guard.
- Verified: record per-stage timing in batch reports/support bundles so slow OCR, decode, inpaint, or encode paths are diagnosable.
- Verified: wire gettext extraction through GUI strings; `backend/i18n.py` is a scaffold and `locale/vsr.pot` exists, but most user-facing strings remain literals.
- Verified: extend screen-reader announcements and keyboard/focus smoke coverage beyond the current UIA announcement scaffold.
- Verified: split the highest-risk GUI/backend orchestration paths; `gui/app.py` is very large and `docs/architecture.md` still flags `process_video` as monolithic.
- Likely: local container/isolated install recipes would reduce setup failures for non-NVIDIA/GPU users without making cloud processing the default.

## Product Map
- Core workflows: queue files/folders, choose cleanup preset/mode/device/language, define automatic or timed manual masks, preview inpainted frames, run batch cleanup, inspect quality reports/support bundles, export SRT/mask/NLE/cache artifacts.
- User personas: Windows video editors, archivists/restoration users, privacy-sensitive local users, batch operators, and CLI users who need reproducible diagnostics.
- Platforms and distribution: Python 3.10-3.13, Tkinter GUI, CLI, Windows launchers, PyInstaller/NSIS local builds, winget-ready metadata, optional CUDA/DirectML/OpenVINO/ONNX/Paddle/RapidOCR paths.
- Key integrations and data flows: FFmpeg/FFprobe ingest/encode/profiles, OpenCV/Pillow media I/O, RapidOCR/PaddleOCR/EasyOCR/Surya/OpenCV detection, TBE/LaMa/registered inpainting, JSON settings/presets/checkpoints/reports, redacted support bundles, release SBOM/advisories.

## Competitive Landscape
- YaoFANGUK/video-subtitle-remover: does local hard-subtitle/text-watermark removal with prebuilt CPU/DirectML/CUDA packages and active user demand for pause, progress saving, GPU clarity, time ranges, and less blur. Learn from its packaging matrix and issue signals; avoid copying unclear startup behavior or making Docker/conda the main Windows path.
- VideOCR and RapidVideOCR: focus on hard-subtitle extraction to SRT with GUI/CLI flows, language breadth, crop boxes, time ranges, and Docker builds. Learn from explicit extraction/review controls and install isolation; avoid cloud OCR or Google Lens-style defaults in this privacy-first app.
- Subtitle Edit: mature subtitle correction/OCR workflow with broad format literacy. Learn from review/edit affordances for extracted text and subtitles; avoid turning VSR into a full subtitle editor before cleanup/recovery gaps are closed.
- IOPaint: strong local model visibility, plugin-style optional backends, file/model management, and batch image cleanup. Learn from clear model state and backend management; avoid becoming a general image editor or adding a marketplace before adapter trust gates stay boring.
- ProPainter and related video-inpainting research: strong temporal quality but licensing/weight size constraints matter. Learn through opt-in benchmarks and local adapters; keep non-commercial or unverified weights out of the default path.
- PaddleOCR and RapidOCR: active OCR ecosystems with frequent releases and hardware-provider churn. Learn from their backend breadth; keep VSR's dependency caps, provider status, and fallback messaging explicit.
- Media.io/AniEraser and similar commercial tools: sell one-click browser/mobile cleanup, manual brush selection, broad format handling, and no-blur claims. Learn confidence-building previews and quality language; avoid upload-first workflows and opaque claims.

## Security, Privacy, and Reliability
- Verified: local `gui/config.py` reports `APP_VERSION = "3.17.3"` and README/CHANGELOG match, but GitHub's latest release is `v3.17.1`; this weakens installer trust after prior repo issues about AV false positives and broken release loads.
- Verified: release evidence now covers SBOM, `release-advisories.json`, FFmpeg profiles, OpenCV/libpng state, RapidOCR packaging, and reference corpus hooks in `backend/release_verification.py`; do not duplicate old advisory/FFmpeg tasks.
- Verified: OpenCV/libpng PNG reads are routed through `backend/safe_image.py`, and `tests/test_hardening.py` enforces production `cv2.imread` hygiene; remaining risk is conflicting OpenCV wheel installs (`opencv-python`, `opencv-contrib-python`, headless variants) shadowing each other.
- Verified: support bundles report many dependency versions in `backend/support_bundle.py`, but do not yet explain when multiple OpenCV distributions are installed or how to repair them safely.
- Verified: `docs/edge_case_corpus.md` requires CC0/public-domain real clips, settings, screenshots, and license declarations, but `.github/ISSUE_TEMPLATE/` only contains bug and feature forms.
- Verified: deterministic fixtures in `tests/clips/manifest.json` cover subtitle motion, karaoke, vertical text, thin fonts, HDR-like ramps, drop shadows, and moving lower thirds; real redistributable clips remain the main coverage gap.
- Verified: current quality gates emit remediation and retry patches, but no preflight warns when a selected codec/quality combination is likely to reduce source detail or bitrate before a long run.
- Verified: release/model adapters are local/opt-in and path/hash-gated; keep this privacy/trust policy.

## Architecture Assessment
- `gui/app.py` owns too many surfaces: onboarding, queue, preview, settings, dialogs, cache import/export, support bundle, quality review, region selector, and processing orchestration. Extracting focused controllers will lower regression risk.
- `backend/processor.py` still contains the hardest-to-change long-running media state machine; `docs/architecture.md` identifies `process_video` as monolithic and sensitive to call-site assumptions.
- `backend/a11y.py` provides UIA announcements, and `gui/widgets.py` has focusable custom widgets, but custom Canvas controls still lack broad semantic/state test coverage.
- `backend/i18n.py` and `locale/vsr.pot` exist, but GUI strings are mostly not wrapped in `_()` and no catalog smoke test proves non-English replacement through the main screens.
- `backend/batch_report.py` records per-item elapsed time, but not decode/OCR/mask/inpaint/encode/mux stage timing; support bundles cannot yet identify the slow stage of a failed or slow run.
- Test gaps: OpenCV wheel-conflict diagnosis, real-clip intake schema, stage-timing aggregation, output-quality preflight decisions, i18n extraction coverage, accessibility/focus traversal for major dialogs, and pause/resume checkpoints.
- Documentation gaps: GitHub contribution flow for edge clips points to Discussions that are not present; isolated install/container guidance is weaker than direct competitors; release-download trust should point at the current shipped tag.
- Coverage note: security, accessibility, i18n/l10n, observability, testing, docs, distribution/packaging, plugin ecosystem, mobile, offline resilience, multi-user service, migration paths, and upgrade strategy were either promoted to the roadmap or explicitly rejected above.

## Rejected Ideas
- Cloud upload/API cleanup from Media.io/AniEraser-style tools: conflicts with local privacy and the current offline architecture.
- Default bundled ProPainter, DiffuEraser, CLEAR, SEDiT, CoCoCo, SAM 3, ROSE, or MiniMax-Remover paths: current blockers are licensing, unreleased weights, missing code, or large unverified assets.
- Google Lens/cloud hybrid OCR from VideOCR-style workflows: useful for extraction accuracy but contradicts local/offline defaults.
- Full GUI framework rewrite: the Tk app is large, but incremental controller extraction is safer than replacing a working release surface.
- General plugin marketplace from IOPaint-style ecosystems: VSR already has a registry/adapter boundary; marketplace work should wait until release and adapter trust are quieter.
- Mobile app or hosted multi-user service from commercial competitors: the best current leverage remains Windows desktop reliability, artifact trust, and local batch recovery.
- GitHub Actions/Dependabot-style automation: project rules keep builds, tests, dependency updates, and releases local on this machine.

## Sources
### Project and Issues
- https://github.com/SysAdminDoc/VideoSubtitleRemover
- https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/2
- https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/3
- https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/5

### OSS and Adjacent Tools
- https://github.com/YaoFANGUK/video-subtitle-remover
- https://github.com/YaoFANGUK/video-subtitle-remover/issues/200
- https://github.com/YaoFANGUK/video-subtitle-remover/issues/222
- https://github.com/YaoFANGUK/video-subtitle-remover/issues/224
- https://github.com/YaoFANGUK/video-subtitle-remover/issues/218
- https://github.com/YaoFANGUK/video-subtitle-remover/issues/232
- https://github.com/timminator/VideOCR
- https://github.com/SWHL/RapidVideOCR
- https://github.com/SubtitleEdit/subtitleedit
- https://github.com/Sanster/IOPaint
- https://github.com/sczhou/ProPainter
- https://github.com/PaddlePaddle/PaddleOCR
- https://github.com/RapidAI/RapidOCR
- https://github.com/facebookresearch/sam2

### Commercial, Platform, and Security
- https://anieraser.media.io/remove-subtitles-from-video.html
- https://ffmpeg.org/ffmpeg-filters.html
- https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- https://github.com/opencv/opencv-python/issues/1186
- https://nvd.nist.gov/vuln/detail/CVE-2026-22801
- https://nvd.nist.gov/vuln/detail/CVE-2025-32434
- https://pyinstaller.org/en/stable/CHANGES.html
- https://pillow.readthedocs.io/en/stable/releasenotes/12.2.0.html

### Research
- https://huggingface.co/charlesw09/CLEAR-mask-free-video-subtitle-removal
- https://arxiv.org/abs/2605.14894
- https://arxiv.org/abs/2503.05639
- https://github.com/ali-vilab/VACE

## Open Questions
None that block prioritization. Real-clip selection still needs license validation during implementation.
