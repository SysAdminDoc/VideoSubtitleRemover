# Research — Video Subtitle Remover Pro

## Executive Summary

Video Subtitle Remover Pro is a Windows-first, offline desktop workstation for hardcoded subtitle/text/logo cleanup, with a Tk GUI, CLI, FFmpeg I/O, OCR cascade, TBE/LaMa/OpenCV inpainting, local release evidence, support bundles, quality gates, and opt-in research adapters. Verified: the strongest current direction is not another default model, but trust and recovery hardening around native media decode, dependency drift, FFmpeg capability truth, and batch review loops. Highest-value opportunities: safe PNG decode while OpenCV ships vulnerable libpng, release-time advisory gating, FFmpeg feature profiles for VVC/VMAF/Whisper, one-click quality-gate retry settings, time-ranged manual masks, portable offline model caches, OpenVINO OCR acceleration, and NVIDIA ONNX Runtime provider strategy.

## Product Map

- Core workflows: queue files/folders, choose cleanup preset/mode/device/language, set or auto-detect subtitle regions, review masks/inpaint previews, process batches, inspect quality reports/support bundles, export SRT/mask/NLE sidecars.
- User personas: Windows video editors, archivists/restoration users, privacy-sensitive local users, batch operators, and CLI users needing reproducible diagnostics.
- Platforms and distribution: Python 3.10-3.13 source, Windows launchers, PyInstaller onedir, NSIS installer, winget-ready metadata, local release evidence; no current GitHub Actions workflow.
- Key integrations and data flows: FFmpeg/FFprobe, OpenCV, Pillow, RapidOCR/PaddleOCR/EasyOCR/Surya/OpenCV detection, ONNX Runtime/DirectML/OpenCV DNN/PyTorch opt-in inpainting, JSON settings/presets/checkpoints/reports, opt-in external/model adapters.

## Competitive Landscape

- YaoFANGUK/video-subtitle-remover: strong upstream package matrix and simple hard-subtitle removal UX. Learn from its CPU/CUDA/DirectML build clarity; avoid its heavier Docker/conda path as the primary Windows story.
- InpaintDelogo + VideoHelp workflows: expert users value dynamic masks, previewable extraction, and deterministic delogo/deblend control. Learn from mask preview and static-logo handling; avoid script-first setup friction and unsafe arbitrary filter execution.
- RapidVideOCR / VideoSubFinder / SubtitleEdit: extraction-first tools make SRT/OCR workflows explicit. Learn from OCR review/export clarity; avoid making cloud OCR or Google Lens-style flows part of the offline default.
- IOPaint: excellent local model/backend visibility and model-manager UX for inpainting. Learn from clear installed-model state; avoid becoming a general image editor.
- GhostCut / AniEraser / HitPaw / Vmake: commercial tools sell automatic detection, batch processing, API access, mobile/browser convenience, and tiered quality. Learn task-mode clarity and retry guidance; avoid cloud uploads, subscriptions, and opaque quality claims.
- ProPainter / E2FGVI / VACE / VideoPainter / VOID / SAM2 / CoTracker: strong temporal/mask research. Learn through benchmark-only and opt-in adapters; avoid bundling noncommercial or multi-GB weights by default.
- CLEAR / SEDiT: mask-free subtitle erasure is the leapfrog direction. Keep benchmarking it, but do not expose it as default until VSR reference clips prove quality, runtime, licensing, and safety.

## Security, Privacy, and Reliability

- Verified: OpenCV/libpng is only warned about in `backend/security_checks.py`; PNG decode still goes through `cv2.imread` in `backend/processor.py`, `backend/io.py`, `backend/cli.py`, and `gui/app.py`. Route untrusted PNG reads through Pillow or fail closed while OpenCV reports vulnerable libpng.
- Verified: `backend/release_verification.py` emits dependency versions and CycloneDX SBOM, but it does not run an advisory database gate. Add a local release vulnerability report with explicit allowlisting for known unavoidable native-bundle issues.
- Verified: FFmpeg-dependent features are build-sensitive. `backend/support_bundle.py` checks common encoders, but README/CLI also expose libvmaf, libvvenc, loudnorm, and FFmpeg Whisper; users need a profile-level capability result before long batches.
- Verified: quality gates in `backend/quality_gate.py` already produce remediation steps, but GUI retry flows do not turn those steps into per-item config changes.
- Verified: manual regions are global (`subtitle_area` / `subtitle_areas`), so OCR-failure cases with moving subtitle placement still need repeated batches or overbroad masks.
- Verified: `backend/model_downloads.py` and `backend/cache_inventory.py` can describe caches, but cannot export/import a verified portable model cache for air-gapped or slow-network systems.
- Verified safe: PyTorch LaMa packaging is opt-in, corrupt/truncated video handling is covered, support bundles redact paths, remote-code/model adapters are gated, and scaled mask-selector regressions have tests.

## Architecture Assessment

- Centralize image I/O before adding more image features: `cv2.imread` is duplicated across GUI preview, CLI probes, image-directory capture, and `process_image`.
- Keep release evidence local-first, but extend it from inventory to decision support: dependency versions, SBOM, native-library status, and advisory results should be one strict artifact.
- Split future manual-mask work out of `gui/app.py`; the file remains the largest risk surface and time-ranged region editing will otherwise increase coupling.
- Quality reporting is mature, but it needs a closed loop: review-needed outputs should offer reproducible retry configs instead of only prose remediation.
- Keep current adapter policy: `backend/adapter_manifest.py`, `backend/remote_model_policy.py`, and `backend/inpainters_diffusion.py` are the right boundary for research models.
- Test gaps: safe image decode, advisory gate fixtures, FFmpeg profile parsing, quality-gate retry mutations, time-ranged region serialization, and verified model-cache import/export.
- Documentation gaps: README says "12-language support" in the overview while current feature/docs describe roughly 50+ engine-supported languages; FFmpeg reference-build guidance is still unresolved.
- Coverage: security, observability, testing, docs, distribution, offline resilience, migration, and upgrade strategy get new roadmap items; accessibility, i18n/RTL, plugin ecosystem, mobile, WebGPU, and multi-user already have existing roadmap/rejection coverage and are not duplicated.

## Rejected Ideas

- Cloud upload/API cleanup from GhostCut/Media.io/HitPaw/Vmake: conflicts with offline privacy.
- Subscription/paywall tiers from commercial tools: conflicts with MIT/local identity.
- Docker-first distribution from upstream VSR: useful for some Linux users, but weaker fit than Windows installer/winget for this repo.
- Default bundled ProPainter/VACE/CLEAR/SEDiT weights: size, license, and trust risk outweigh default-user value.
- Plugin marketplace now from IOPaint-style ecosystems: existing env-gated registry/adapters are enough until release/advisory gates are stronger.
- Mobile ports now from commercial web/mobile competitors: desktop reliability and packaging evidence are higher leverage.
- Mask-free erasure as default now from CLEAR/SEDiT: promising but unproven against VSR clips and operational constraints.
- Full GUI framework rewrite: `gui/app.py` is large, but incremental extraction is safer than replacing a working Tk application.

## Sources

### OSS and Adjacent

- https://github.com/YaoFANGUK/video-subtitle-remover
- https://github.com/SWHL/RapidVideOCR
- https://github.com/Sanster/IOPaint
- https://github.com/Purfview/InpaintDelogo
- https://github.com/sczhou/ProPainter
- https://github.com/MCG-NKU/E2FGVI
- https://github.com/facebookresearch/sam2
- https://github.com/facebookresearch/co-tracker
- https://github.com/kba/awesome-ocr

### Commercial and Community

- https://jollytoday.com/subtitle-removal/
- https://anieraser.media.io/remove-subtitles-from-video.html
- https://www.hitpaw.com/remove-watermark/remove-subtitles-from-video.html
- https://vmake.ai/remove-subtitle-from-video
- https://forum.videohelp.com/threads/418629-Which-tool-has-the-best-accuracy-for-extracting-hardsubs-from-video
- https://forum.videohelp.com/threads/415678-Can-t-get-good-logo-removal-results-with-InpaintDelogo

### Dependencies, Platform, Security

- https://github.com/RapidAI/RapidOCR/releases/tag/v3.9.0
- https://github.com/PaddlePaddle/PaddleOCR/releases/tag/v3.7.0
- https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
- https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview
- https://github.com/microsoft/onnxruntime/releases/tag/v1.27.0
- https://pyinstaller.org/en/v6.21.0/CHANGES.html
- https://pillow.readthedocs.io/en/stable/releasenotes/12.2.0.html
- https://github.com/opencv/opencv-python/issues/1186
- https://nvd.nist.gov/vuln/detail/CVE-2026-22801
- https://nvd.nist.gov/vuln/detail/CVE-2025-32434
- https://ffmpeg.org/ffmpeg-all.html

### Research

- https://huggingface.co/charlesw09/CLEAR-mask-free-video-subtitle-removal
- https://arxiv.org/abs/2605.14894
- https://arxiv.org/abs/2503.05639
- https://github.com/ali-vilab/VACE

## Open Questions

- Needs live validation: which redistributable real clips can populate `tests/clips/manifest.json` without license risk?
- Needs live validation: which Windows FFmpeg build should be named as the reference for libvmaf, libvvenc, loudnorm, and Whisper support?
- Needs live validation: whether OpenVINO OCR materially improves VSR's CPU/Intel path after RapidOCR PP-OCRv6 without adding unacceptable install friction.
