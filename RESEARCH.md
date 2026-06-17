# Research -- Video Subtitle Remover

## Executive Summary
Video Subtitle Remover is a Windows-first, local-first Python/tkinter desktop app for removing hardcoded subtitles, text overlays, and text watermarks with a multi-engine OCR/detection cascade, video-aware inpainting, batch reports, and release verification. Its strongest current shape is not a cloud-style one-click tool; it is an inspectable local workflow with many codec, OCR, GPU, and recovery controls. The highest-value direction is to make that power more trustworthy and operable before adding more experimental model breadth: finish the CVE-2026-22801 release blocker, pin or localize runtime remote-code model adapters, harden release/update supply-chain evidence, capture RapidOCR bundled-model provenance, surface quality-gate failures as a GUI review worklist, refresh stale architecture and launcher docs, and keep heavier SOTA adapters behind opt-in benches until they beat TBE/LaMa on the reference corpus.

## Product Map
- Core workflows: select one or more videos; choose automatic/manual subtitle regions; preview masks; process with AUTO/TBE/LaMa or remux-only paths; review output, logs, quality reports, and batch reports.
- User personas: Windows video editors who need a local app; archive/restoration users processing many files; nontechnical users who prefer GUI setup; technical users who need CLI/batch evidence; privacy-sensitive users avoiding cloud upload.
- Platforms and distribution: Python/tkinter source app, Windows launcher/setup scripts, PyInstaller/NSIS release workflow, optional Windows Store/winget distribution, limited Linux/macOS source use.
- Key integrations and data flows: FFmpeg/FFprobe for decode/encode/remux; RapidOCR/PaddleOCR/EasyOCR/Surya/OpenCV/VLM detection; ONNX Runtime DirectML and PyTorch CUDA/CPU acceleration; model caches and hash checks; JSON/Markdown quality/batch reports.

## Competitive Landscape
- YaoFANGUK/video-subtitle-remover -- Direct OSS competitor with prebuilt CPU/DirectML/CUDA packages, Docker variants, text-watermark removal, and a large user base. Learn from its packaging matrix and "works without local Python expertise" stance; avoid copying its broader fork identity or hiding advanced failure evidence behind simple presets.
- GhostCut/JollyToday -- Commercial cloud remover with batch upload, quality tiers, API, translation/dubbing, and NLE-oriented localization. Learn from explicit quality modes, batch affordances, and localization workflow packaging; avoid a cloud-first upload model that contradicts VSR's local privacy advantage.
- RecCloud / Vmake / CreatOK / AniEraser -- Commercial tools emphasize one-click upload, hardcoded-versus-soft subtitle explanations, mobile/web availability, privacy deletion claims, and easy continuation into editing. Learn from their calm user education and confidence-building states; avoid overpromising autonomous results for complex archives where human review is still required.
- VideOCR / RapidVideOCR / VideoSubFinder -- Adjacent extraction tools show continued demand for hard-sub OCR, SRT/ASS output, and simple setup. Learn from extraction/export workflows and language discoverability; avoid making OCR-only workflows the main product, because VSR's differentiator is removal plus local quality evidence.
- IOPaint -- Adjacent local inpainting app with one-click installers, CPU/GPU/Apple Silicon support, model directories, plugins, batch CLI, and a self-hosted UI. Learn from model-manager ergonomics and plugin isolation; avoid expanding into a general image editor that dilutes video subtitle removal.
- InpaintDelogo / Subtitle Edit community workflows -- Community discussions still value robust temporal logo/subtitle removal but complain about workflow friction. Learn from temporal refinement, trash/false-positive controls, and review loops; avoid AviSynth-style setup complexity for mainstream users.
- PaddleOCR / RapidOCR ecosystem -- OCR dependency ecosystem is moving quickly: PaddleOCR 3.7.0 / PP-OCRv6 claims stronger 50-language recognition, while RapidOCR discussion #667 flags fragile bundled-model release-asset provenance. Learn from new accuracy and deployment options, but add release evidence before floating dependencies silently change results.

## Security, Privacy, and Reliability
- Verified: `.github/workflows/build.yml` still has the libpng CVE-2026-22801 gate as the current release blocker in `ROADMAP.md`; keep this as P0 because user-facing binaries should not ship with known vulnerable image-stack evidence.
- Verified: optional VLM/segmentation paths can execute remote repository code or moving refs: `backend/ocr_vlm.py:99-100` uses `trust_remote_code=True`, and `backend/segmentation.py:207` calls `torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")` without a pinned ref. Hugging Face and PyTorch both describe these as trust boundaries; VSR should require pinned revisions or local paths before enabling them.
- Verified: RapidOCR model discovery exists in `backend/onnx_model_info.py:167-214`, but the current ONNX audit captures opsets/DirectML compatibility rather than dependency-package version, file hashes, or source provenance. RapidOCR discussion #667 makes this roadmap-worthy because current wheels reportedly depend on old v1.1.0 release assets.
- Verified: release workflow hardening remains important: `.github/workflows/build.yml:482-483` downloads and executes `wingetcreate.exe` from a mutable "latest" URL, while action references use major tags. Existing roadmap items already cover this; do not duplicate.
- Verified: strict release verification still checks `ROADMAP.md` for the expected version in `.github/workflows/build.yml:370`, which conflicts with the project rule that ROADMAP contains only incomplete work. Existing roadmap items already cover this; do not duplicate.
- Likely: the GUI writes quality/batch reports and sets `review-needed` status via `backend/batch_report.py:37` and `backend/batch_report.py:140`, but `gui/app.py:3333-3349` only exposes a generic "Open report" action in the completion dialog. A dedicated review worklist would reduce missed failed-gate outputs.
- Verified: `setup.py:431-583` generates `Run_VSR_Pro.ps1`, but the tracked root launchers and README focus on `.bat` files. Either track/test/document the PowerShell launcher or stop generating it to prevent release drift.

## Architecture Assessment
- Module boundaries are healthier than older docs imply: GUI code now lives under `gui/`, detection/inpainting/quality/cache/release helpers are split under `backend/`, and smoke/reference/release tests exist. `docs/architecture.md` is still stale and should be refreshed before new contributors extend workflows.
- Refactor candidate: centralize optional model loading policy for Hugging Face, torch.hub, ONNX, and local adapter paths. Touches `backend/ocr_vlm.py`, `backend/segmentation.py`, `backend/adapter_manifest.py`, `backend/model_hashes.py`, and `backend/onnx_model_info.py`.
- Refactor candidate: promote quality-gate state from report-only evidence into a first-class GUI review queue. Touches `gui/app.py`, `gui/widgets.py`, `backend/batch_report.py`, and `backend/quality_gate.py`.
- Test gaps: add mocked tests for remote-code loader rejection/pinned acceptance, RapidOCR bundled-model hash/provenance capture, PowerShell launcher bundle parity, and GUI review-needed action visibility.
- Documentation gaps: README codec/launcher tables and `docs/architecture.md` should be checked against the live CLI/GUI choices after VVC and launcher changes. Existing roadmap items already cover VVC sync and architecture refresh.
- Category coverage: security, observability, testing, docs, distribution, accessibility, i18n/l10n, plugin ecosystem, mobile, offline/resilience, multi-user, migration paths, and upgrade strategy were reviewed. New additions focus on security/provenance/workflow/packaging because accessibility/i18n/mobile/plugin/migration lanes are already tracked in `ROADMAP.md` and no new public source changed their priority.

## Additive Pass -- Missed Gaps
- Verified: imported presets are trusted too broadly. `gui/config.py:765-784` stores imported `fields` without a schema allowlist, and `gui/config.py:683-700` applies any field that matches `ProcessingConfig`. This is not remote code execution, but a shared preset can unexpectedly flip network/update, output, reporting, Whisper, or destructive workflow settings without a preview.
- Verified: support intake is under-built for a local media tool. README links directly to generic GitHub issues, `.github/` contains only the build workflow, and there is no single redacted diagnostics bundle even though `backend/crash_reporter.py` already scrubs paths and `backend/batch_report.py:161-168` already redacts report paths. GitHub issue forms support structured required fields, and OWASP logging guidance backs explicit redaction before sharing logs.
- Verified: soft-subtitle remux, cache inventory, quality sheets, JSON logs, crash reporting, and batch reports exist, so the missed support gap is not "add logging"; it is packaging the already-available local evidence into a privacy-preserving handoff.
- Rechecked but not added: soft-subtitle track handling, cache-clean safety, reference-clip coverage, accessibility, i18n, mobile, plugin architecture, and cloud/API workflows are already implemented, already on `ROADMAP.md`, or explicitly rejected.

## Rejected Ideas
- Cloud upload/default SaaS processing -- commercial tools prove demand, but it conflicts with VSR's local privacy and large-file control advantage.
- Mobile app as a near-term goal -- AniEraser/Media.io show mobile demand, but VSR's current architecture is desktop Python/tkinter plus FFmpeg/model dependencies; packaging and reliability work should land first.
- Add more default SOTA video diffusion removers now -- awesome-lists and papers show many candidates, but existing roadmap already has adapter benches; adding more before reference-clip evidence would increase maintenance and GPU burden.
- Make VLM/OCR remote models the default detector -- PaddleOCR/RapidOCR releases and VSR's fallback cascade support local-first OCR; remote-code VLMs should remain opt-in until pinned, reviewed, and benchmarked.
- General-purpose image editor or full NLE scope -- IOPaint and GhostCut show adjacent value, but VSR should keep video subtitle/text removal as the primary workflow.
- Multi-user/team collaboration -- commercial APIs suggest enterprise workflows, but there is no repo architecture for accounts, shared projects, or server state; exportable reports are the better local-first path.
- Mandatory Docker-first distribution -- direct competitors offer Docker, but VSR's target user and current setup are Windows GUI-first; Docker belongs as optional advanced documentation only.

## Sources
OSS and adjacent:
- https://github.com/YaoFANGUK/video-subtitle-remover
- https://github.com/timminator/VideOCR
- https://github.com/voun7/VidSubX
- https://github.com/SWHL/RapidVideOCR
- https://github.com/SWHL/VideoSubFinder
- https://github.com/Purfview/InpaintDelogo
- https://github.com/Sanster/IOPaint
- https://www.iopaint.com/batch_process
- https://www.iopaint.com/install/download_model

Commercial and community:
- https://jollytoday.com/subtitle-removal/
- https://reccloud.com/remove-subtitle
- https://reccloud.com/ai-burned-in-subtitle-removal-large-video-archives.html
- https://anieraser.media.io/remove-subtitles-from-video.html
- https://github.com/SubtitleEdit/subtitleedit/discussions/9562
- https://github.com/SubtitleEdit/docs
- https://forum.videohelp.com/threads/418629-Which-tool-has-the-best-accuracy-for-extracting-hardsubs-from-video
- https://stackoverflow.com/questions/58283967/tesseract-ocr-pre-processing-for-subtitles-extraction

Dependencies, standards, and advisories:
- https://github.com/PaddlePaddle/PaddleOCR/releases
- https://github.com/RapidAI/RapidOCR/releases
- https://github.com/RapidAI/RapidOCR/discussions/667
- https://huggingface.co/docs/transformers/en/model_doc/auto
- https://docs.pytorch.org/docs/2.12/hub.html
- https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
- https://github.com/opencv/opencv-python/issues/1186
- https://nvd.nist.gov/vuln/detail/CVE-2026-22801
- https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/configuring-issue-templates-for-your-repository
- https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html
- https://docs.github.com/en/actions/reference/security/secure-use
- https://docs.github.com/en/actions/concepts/security/artifact-attestations
- https://www.cisa.gov/resources-tools/resources/2025-minimum-elements-software-bill-materials-sbom

## Open Questions
- Needs live validation: which FFmpeg binary is bundled or recommended for releases after the libpng fix, and does it expose `libvvenc` for VVC/H.266?
- Needs live validation: which optional VLM/CoTracker model revisions have actually been smoke-tested on Windows hardware, and which should be blessed in an adapter allowlist?
