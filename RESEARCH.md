# Research -- Video Subtitle Remover Pro

## Executive Summary

Video Subtitle Remover Pro is a Windows-first local desktop tool for removing hardcoded subtitles, chyrons, logos, and text watermarks from videos/images with a Python/tkinter GUI, CLI, FFmpeg I/O, OCR detection, TBE/LaMa/OpenCV inpainting, support bundles, and local PyInstaller/NSIS packaging. Verified locally: the product is now strongest as a privacy-preserving batch workstation with unusually strong recovery/evidence surfaces for an OSS media tool; the highest-value direction is release-truth hardening after GitHub Actions removal, mask-selection regression depth, dependency drift control, and selective model UX rather than broad new model sprawl. Top opportunities: local release verifier replacing skipped workflow checks; env-gated PyTorch LaMa bundling; mask-selector visual/coordinate regression suite; corrupt-media failure UX tests; RapidOCR PP-OCRv6 compatibility; OpenCV/libpng fixed-wheel tracking; installed model/backend status UI; Windows ML vs DirectML strategy; benchmark-only mask-free subtitle erasure adapters.

## Product Map

- Core workflows: queue videos/images, choose preset/mode/device/language, select or auto-detect subtitle regions, preview masks/inpainted frames, process batches, review quality reports/logs/support bundles, export SRT/mask/NLE sidecars.
- User personas: Windows video editors, archival/restoration users, privacy-sensitive local users, batch operators, CLI users who need reproducible commands and diagnostic artifacts.
- Platforms and distribution: local Python source, Windows launchers, PyInstaller `--onedir`, NSIS installer path, winget-ready metadata; `.github/workflows` is intentionally absent in the current local-build checkout.
- Key integrations and data flows: FFmpeg/FFprobe decode/encode/remux, OpenCV image/video I/O, RapidOCR/PaddleOCR/EasyOCR/OpenCV detection cascade, optional VLM/OCR and segmentation adapters, ONNX Runtime DirectML/CUDA/CPU, optional PyTorch fallback, JSON settings/presets/checkpoints/batch reports.

## Competitive Landscape

- YaoFANGUK/video-subtitle-remover: mature upstream with simple GUI/package matrix. Learn from its low-friction user path; avoid weaker diagnostics and less explicit recovery evidence.
- IOPaint: strong local model manager, plugin-like workflow, and installer story. Learn from model/backend status visibility; avoid becoming a general image editor.
- Purfview/InpaintDelogo and VideoHelp workflows: practical FFmpeg/VapourSynth delogo approaches remain useful for static logos. Learn from deterministic local filters; avoid exposing script execution without the existing trust gates.
- RapidVideOCR/VideOCR: focused hard-subtitle extraction to SRT/ASS. Learn from extraction-first workflows and language clarity; avoid cloud OCR defaults.
- GhostCut, AniEraser/Media.io, HitPaw, Vmake: commercial tools sell convenience, batch text/logo removal, and clear tiers. Learn from task-mode clarity; avoid cloud-only processing, subscriptions, and opaque quality claims.
- ProPainter/E2FGVI/VACE/VideoPainter/VOID/SAM2 family: stronger temporal coherence and segmentation research. Learn through opt-in adapters and benchmarks; avoid bundling heavy or license-unclear weights by default.

## Security, Privacy, and Reliability

- Verified risk: `build_exe.bat` always includes `--hidden-import simple_lama_inpainting` while README says PyTorch LaMa is disabled unless `VSR_ENABLE_PYTORCH_LAMA=1`; this can pull optional native/PyTorch code into release bundles contrary to the runtime trust posture.
- Verified risk: `.github/workflows/build.yml` was removed in `c4a4617`, but `tests/test_release_workflow.py` now skips most release-evidence assertions when the workflow is absent; the local build path needs equivalent verifiable artifacts.
- Verified risk: OpenCV currently cannot be forced to a fixed bundled libpng until opencv-python publishes a wheel with libpng >= 1.6.54; `backend/security_checks.py` warns, but release tooling should keep tracking the fixed-wheel state.
- Verified risk: `docs/architecture.md` still describes removed GitHub Actions release paths and old automatic PyTorch fallback behavior, so implementation agents can reintroduce stale assumptions.
- Verified safe: support bundles now include `rapidocr` and redact `work_directory` in `backend/support_bundle.py`.
- Verified safe: `backend/config.py` normalizes invalid backend device strings to `cpu`, and `backend/cli.py` warns on unknown JSON config fields.
- Verified safe: remote-code/model-adapter policy exists in `backend/remote_model_policy.py` and `backend/adapter_manifest.py`; keep new adapters behind the same policy.

## Architecture Assessment

- The `gui/` and `backend/` split is serviceable: GUI owns interaction state, backend owns processing/config, and adapters are lazy/gated.
- The most fragile boundary is release verification: older strict-release checks lived in a now-removed workflow, while the current local scripts do not emit the same single evidence bundle.
- The mask selector remains a high-risk UX path because it converts displayed canvas coordinates to image-space rectangles, persists both current config and queued snapshots, and has had recent regressions; current tests cover save flow but not enough scaled/resized/video/image combinations.
- `backend/inpainters_diffusion.py` already has opt-in scaffolds; future research models should land first as benchmark/adapters with no default dependency.
- Test gaps worth filling before more features: corrupt/truncated media, mask selector scaling, local release evidence, and packaging hidden-import policy.
- Documentation gaps: `docs/architecture.md` and `CLAUDE.md` still mention GitHub Actions/old backend assumptions; README is closer to current truth.
- Coverage: security/reliability and distribution are active roadmap work; accessibility/i18n/RTL already have scaffolds and remain in ROADMAP; observability is covered by logs, reports, and support bundles; plugin/mobile/multi-user/cloud are intentionally deferred or rejected; migration/upgrade work is handled through settings normalization, dependency caps, and release evidence.

## Rejected Ideas

- Cloud upload/API processing from GhostCut/Media.io/HitPaw/Vmake: conflicts with the offline privacy promise.
- Subscription/paywall model from commercial competitors: conflicts with the current MIT/local tool identity.
- Docker-first distribution from upstream-style projects: useful for Linux power users, but secondary to Windows installer/winget.
- Default bundled noncommercial/heavy research weights: licensing, size, and trust risk outweigh parity value.
- Plugin marketplace now: `backend/inpainter_registry.py` and env-gated adapters are enough until local release verification is solid.
- Mobile ports now: Android/iOS are plausible later, but current desktop reliability and packaging evidence are higher leverage.
- Mask-free subtitle erasure as default now: CLEAR/SEDiT are promising but need VSR-specific benchmark evidence before user-facing placement.

## Sources

### OSS and Adjacent

- https://github.com/YaoFANGUK/video-subtitle-remover
- https://github.com/KKenny0/videowipe
- https://github.com/Purfview/InpaintDelogo
- https://github.com/SWHL/RapidVideOCR
- https://github.com/timminator/VideOCR
- https://github.com/Sanster/IOPaint
- https://github.com/sczhou/ProPainter
- https://github.com/MCG-NKU/E2FGVI
- https://github.com/facebookresearch/sam2

### Commercial and Community

- https://jollytoday.com/subtitle-removal/
- https://anieraser.media.io/remove-subtitles-from-video.html
- https://www.hitpaw.com/remove-watermark/remove-subtitles-from-video.html
- https://vmake.ai/remove-subtitle-from-video
- https://www.echosubs.com/remove-hardcoded-subtitles-offline
- https://forum.videohelp.com/threads/418629

### Dependencies, Platform, Security

- https://github.com/RapidAI/RapidOCR/releases
- https://github.com/PaddlePaddle/PaddleOCR/releases
- https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
- https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview
- https://github.com/opencv/opencv-python/issues/1186
- https://ffmpeg.org/
- https://pyinstaller.org/en/stable/CHANGES.html
- https://nvd.nist.gov/vuln/detail/CVE-2025-32434
- https://nvd.nist.gov/vuln/detail/CVE-2026-22801
- https://nvd.nist.gov/vuln/detail/CVE-2026-25990

### Research

- https://huggingface.co/charlesw09/CLEAR-mask-free-video-subtitle-removal
- https://arxiv.org/abs/2605.14894
- https://arxiv.org/abs/2503.05639
- https://github.com/ali-vilab/VACE
- https://void-model.github.io/

## Open Questions

- Needs live validation: which redistributable clips can be added to `tests/clips/` for mask-selection and corrupt-media regression without license friction?
- Needs live validation: does Windows ML expose a Python-packagable inference path that should replace or sit beside ONNX Runtime DirectML for AMD/Intel users?
- Needs live validation: which Windows FFmpeg distribution should be documented as the reference build for VVC, whisper filter, and libvmaf support?
