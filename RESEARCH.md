# Research - Video Subtitle Remover Pro

## Executive Summary
Video Subtitle Remover Pro is a Windows-first local desktop and CLI tool for removing burned-in subtitles, text watermarks, and logo-like overlays without uploading media. Verified: the project is now strongest in offline processing, OCR/inpaint fallback breadth, quality gates, support bundles, pause/resume checkpoints, local model policy, release evidence, and deterministic synthetic reference clips. The highest-value direction is to turn its advanced claims into portable proof: add frozen-app FFmpeg smoke evidence, per-output reproducibility sidecars, real redistributable reference clips, GUI screenshot regression checks, dependency/security drift reporting, pseudo-locale/RTL render tests, stronger NLE interchange, adapter conformance dry-runs, and optional brush-level mask correction.

## Product Map
- Core workflows: import files/folders, choose mode/device/language/preset, set automatic or timed mask regions, preview masks/inpainted frames, run or pause a batch, review quality gates, export SRT/mask/NLE/cache/support artifacts.
- User personas: Windows video editors, archivists/restoration users, privacy-sensitive local users, batch operators, and CLI users who need reproducible diagnostics.
- Platforms and distribution: Python 3.10-3.13, Tkinter GUI, CLI, Windows launchers, PyInstaller/NSIS build scripts, winget-ready metadata, optional CUDA/DirectML/OpenVINO/ONNX/Paddle/RapidOCR paths.
- Key integrations and data flows: FFmpeg/FFprobe ingest/encode/profiles, OpenCV/Pillow media I/O, RapidOCR/PaddleOCR/EasyOCR/Surya/OpenCV detection, TBE/LaMa/registered inpainting, JSON settings/presets/checkpoints/reports, redacted support bundles, release SBOM/advisories.

## Competitive Landscape
- YaoFANGUK/video-subtitle-remover: does local hard-subtitle/text-watermark removal with large prebuilt packages and broad community demand. Learn from its simple packaged entry points and issue volume; avoid opaque backend/package failures.
- VideOCR and RapidVideOCR: focus on hard-subtitle extraction to SRT with language, crop, and time controls. Learn from explicit review/extraction workflows; avoid making cloud OCR the default.
- Subtitle Edit: mature subtitle review/editing with broad format literacy and a polished correction loop. Learn from review affordances and format confidence; avoid becoming a general subtitle editor before cleanup proof improves.
- IOPaint: strong local inpainting UX, model visibility, and manual correction affordances. Learn from model-state clarity and brush workflows; avoid drifting into full image-editor complexity.
- ProPainter, STTN, E2FGVI, CoTracker, SAM2, SEDiT, and CLEAR: show that temporal propagation, tracking, and mask-free subtitle erasure are the quality frontier. Keep integrations gated by license, weight provenance, hash policy, and local benchmarks.
- PaddleOCR, RapidOCR, ONNX Runtime, OpenVINO, and DirectML: active dependency churn creates capability and packaging wins, but also release-risk. Keep local provider reporting, pinned exceptions, and fallback messaging explicit.
- Media.io/AniEraser, HitPaw, and Kapwing-style commercial tools: sell one-click cleanup, manual brush correction, preview confidence, and browser/mobile convenience. Learn preview/proof language and targeted correction UX; avoid upload-first workflows that conflict with the privacy model.

## Security, Privacy, and Reliability
- Verified: `backend/release_verification.py` records package evidence, but the strongest frozen-app risk left is an actual packaged EXE invoking external `ffmpeg`/`ffprobe`; PyInstaller documents child-process DLL search-path hazards and VSR shells out extensively through `backend/io.py`, `backend/remux.py`, and `backend/processor.py`.
- Verified: `backend/batch_report.py` creates batch-level JSON/Markdown summaries, but outputs do not yet carry their own reproducibility sidecar tying source fingerprint, config, engine/provider, model hashes, stage timings, quality gate, checkpoint state, and command provenance to the file.
- Verified: `tests/clips/manifest.json` contains deterministic MIT fixtures, while `tests/test_mask_free_benchmark.py` and `tests/test_static_logo_benchmark.py` still prove benchmark schemas with placeholder bytes; real-world quality claims need public-domain/CC clips with source metadata and baselines.
- Verified: `tests/test_gui_smoke.py` exercises Tk widgets headlessly, but there is no committed visual regression gate for empty queue, queued item, preview-unavailable, quality-review, or backend-status states.
- Verified: `backend/i18n.py`, `locale/vsr.pot`, and `rtl_layout` exist; tests cover pass-through catalog behavior, but no pseudo-locale or RTL rendered smoke proves translated text still fits.
- Verified: `backend/nle_sidecar.py` explicitly emits a one-event EDL/FCPXML stub and does not round-trip transitions, audio tracks, dimensions, or multiple processed ranges.
- Verified: `backend/adapter_manifest.py` and many opt-in adapter tests fail closed, but adapter conformance is spread across tests/support/release evidence instead of one dry-run matrix that operators can inspect.
- Likely: current offline/local posture is a competitive advantage against commercial upload-first tools; recommendations preserve that default.

## Architecture Assessment
- `gui/app.py` remains a large shell/shared-state surface even after controller extraction; visual regression and pseudo-locale tests are safer next steps than a framework rewrite.
- `backend/processor.py`, `backend/cli.py`, and `backend/io.py` still concentrate long-running media orchestration, subprocess edges, and final artifact writes; per-output sidecars and frozen external-process smoke target these boundaries directly.
- `backend/batch_report.py` is the right source of truth for planned/final status, quality preflight, stage timings, and quality-gate fields; sidecar work should reuse it rather than inventing parallel provenance.
- `backend/reference_corpus.py`, `backend/static_logo_benchmark.py`, and `backend/mask_free_benchmark.py` already enforce hash/license schema gates; the missing work is ingesting real redistributable clips and baseline outputs.
- `backend/adapter_manifest.py`, `backend/remote_model_policy.py`, and `backend/release_verification.py` form a good trust boundary for optional models; a single adapter conformance command would make regressions visible without importing untrusted code.
- Coverage audit: security, accessibility, i18n/l10n, observability, testing, docs, distribution/packaging, plugin/adapter ecosystem, offline resilience, migration paths, and upgrade strategy are represented in the roadmap additions. Mobile and multi-user service work are rejected because they weaken the local Windows batch-tool focus.

## Rejected Ideas
- Cloud upload/API cleanup from Media.io/AniEraser-style tools: conflicts with the local privacy model and current offline architecture.
- Default bundled ProPainter, DiffuEraser, CLEAR, SEDiT, SAM2/SAM3-family, or other heavy research models: license, weight, hardware, or provenance constraints make them adapter/benchmark candidates only.
- Google Lens/cloud hybrid OCR from VideOCR-style workflows: useful for extraction accuracy, but it would make local/offline behavior conditional on external services.
- Full GUI framework rewrite: Tk is large but working; targeted screenshot, state, and controller tests reduce regression risk faster.
- General plugin marketplace: adapter manifests already provide a trust boundary; marketplace UX should wait until dry-run conformance and release proof are stronger.
- Mobile app or hosted multi-user service: commercial competitors cover convenience there, but this project's edge is local Windows processing, artifact trust, and batch recovery.
- GitHub Actions/Dependabot-style automation: project rules keep builds, tests, dependency updates, and releases local.

## Sources
### Project
- https://github.com/SysAdminDoc/VideoSubtitleRemover

### OSS and Adjacent Tools
- https://github.com/YaoFANGUK/video-subtitle-remover
- https://github.com/Sanster/IOPaint
- https://github.com/SWHL/RapidVideOCR
- https://github.com/timminator/VideOCR
- https://github.com/SubtitleEdit/subtitleedit
- https://github.com/sczhou/ProPainter
- https://github.com/MCG-NKU/E2FGVI
- https://github.com/facebookresearch/co-tracker
- https://github.com/facebookresearch/sam2
- https://github.com/advimman/lama

### Commercial Tools
- https://anieraser.media.io/remove-subtitles-from-video.html
- https://www.hitpaw.com/remove-watermark.html
- https://www.kapwing.com/tools/remove-subtitles-from-video

### Research
- https://arxiv.org/abs/2605.14894
- https://arxiv.org/abs/2603.21901
- https://huggingface.co/charlesw09/CLEAR-mask-free-video-subtitle-removal
- https://arxiv.org/abs/2501.10018
- https://arxiv.org/abs/2007.10247

### Dependencies, Platform, Security, and Fixtures
- https://pyinstaller.org/en/stable/common-issues-and-pitfalls.html
- https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
- https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html
- https://ffmpeg.org/ffmpeg-filters.html
- https://github.com/PaddlePaddle/PaddleOCR/releases
- https://github.com/RapidAI/RapidOCR/releases
- https://github.com/microsoft/onnxruntime/releases
- https://github.com/opencv/opencv-python/issues/1186
- https://nvd.nist.gov/vuln/detail/CVE-2026-22801
- https://www.nasa.gov/nasa-brand-center/images-and-media/
- https://www.loc.gov/free-to-use/

## Open Questions
None that block prioritization. Real-clip ingestion still requires per-clip license proof during implementation.
