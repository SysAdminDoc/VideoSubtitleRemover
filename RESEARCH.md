# Research -- Video Subtitle Remover

## Executive Summary

Video Subtitle Remover is a Windows-first, local-first Python/tkinter desktop app for removing hardcoded subtitles via a multi-engine OCR cascade (RapidOCR > PaddleOCR > Surya > EasyOCR > OpenCV), video-aware inpainting (TBE / LaMa-ONNX / ProPainter-hybrid), batch processing with quality reports, and release verification. Its strongest current shape is as an inspectable local workflow with deep codec, GPU, and recovery controls -- a clear differentiator over cloud-only commercial tools. The highest-value direction is to raise detection accuracy and pipeline reliability before adding more experimental model breadth.

Top opportunities in priority order:

1. **Lift the RapidOCR 3.x cap** -- 9 months of improvements blocked by `<3.0.0` pin; API changes need validation.
2. **Evaluate PP-OCRv6 models** -- +5.1% recognition accuracy, 5.2x CPU speedup, 50-language single model; ships inside PaddleOCR 3.7+ within the existing pip floor.
3. **Add GitHub issue templates** -- support bundle prompt, structured reproduction, version fields; the support-bundle infrastructure exists but intake is unstructured.
4. **Bundle or guide TkinterDnD2** -- native drag-drop silently fails without it; issue #4/#5 reporter hit this gap.
5. **Evaluate FFmpeg 8 Whisper filter** -- built-in whisper in FFmpeg 8 could simplify or replace `whisper_fallback.py`.
6. **Refresh architecture docs** -- stale module map (already on ROADMAP); new contributors are sent to wrong files.
7. **Monitor DirectML-to-WinML transition** -- ONNX Runtime DirectML EP is "sustained engineering" only; WinML is the forward path for AMD/Intel GPU inference.

## Product Map

- **Core workflows:** import video/images; set automatic or manual subtitle regions; preview masks; process with AUTO/TBE/LaMa/ProPainter-hybrid routes or soft-sub remux; review outputs via quality reports, batch reports, and A/B scrubber.
- **User personas:** Windows video editors needing local subtitle cleanup; archive/restoration users batch-processing large collections; nontechnical GUI users; technical CLI/batch-evidence users; privacy-sensitive users avoiding cloud upload.
- **Platforms and distribution:** Python/tkinter source, Windows launchers (.bat/.ps1), PyInstaller/NSIS releases, GitHub Actions CI/CD, planned winget submission. Limited Linux/macOS source-install use.
- **Key integrations:** FFmpeg/FFprobe for decode/encode/remux; RapidOCR/PaddleOCR/EasyOCR/Surya/OpenCV/VLM detection; ONNX Runtime (DirectML, CUDA, CPU) and PyTorch acceleration; JSON/Markdown quality/batch reports; crash-resume checkpoints; optional Whisper fallback.

## Competitive Landscape

### YaoFANGUK/video-subtitle-remover (upstream, 11.5k stars, Apache-2.0)
- Strengths: Docker packaging (CUDA 11.8/12.6/12.8/DirectML/CPU variants), pre-built 7z archives, cross-platform (Windows/macOS/Linux), PP-OCRv5 detection.
- Learn from: packaging matrix that works without local Python expertise; Docker variants for cross-platform reach.
- Avoid: no quality reports, no multi-engine OCR cascade, no batch progress/recovery, no review worklist -- these remain VSR Pro's advantages.

### GhostCut / JollyToday (commercial SaaS)
- Strengths: batch scale (100 videos), API for integration, combined translation+dubbing+removal pipeline. Credit-based pricing (CN28-30000).
- Learn from: explicit quality tiers, batch affordances, localization workflow packaging.
- Avoid: cloud-only model (189s for a 15s clip in testing), privacy concerns, 30s minimum billing granularity.

### EchoSubs (commercial desktop, $5.99/mo or $49 lifetime)
- Strengths: 100% offline, unlimited batch (folder drag-drop), ~2 min for 10-min 1080p on M1/dedicated GPU, cheapest subscription in the space.
- Learn from: offline-first messaging, folder batch UX, clear quality expectations (85-90/100 score, 90-95 on static backgrounds).
- Avoid: desktop-only with no CLI/evidence output, no quality reports or review worklists.

### RecCloud (commercial cloud)
- Strengths: rated best overall quality in 2026 comparisons, frame-by-frame reconstruction, minimal blur.
- Learn from: quality reputation built on careful reconstruction; Android app availability.
- Avoid: weekly billing ($9-15/week), cloud-only, internet-dependent.

### IOPaint / Lama Cleaner (23.2k stars, archived Aug 2025)
- Strengths: plugin architecture (gold standard for extensibility), web UI, model-switching capability, multi-model support.
- Learn from: plugin architecture and model-manager ergonomics. IOPaint's death creates a vacuum in the image inpainting space.
- Avoid: expanding into a general image editor; IOPaint never had strong video support.

### videowipe (GPL-3.0, new Jun 2026)
- Strengths: PyPI install, Docker images, natural-language intent parsing for target selection, pluggable inpainting architecture.
- Learn from: PyPI distribution, pluggable model interface. Watch for traction -- could absorb the subtitle-removal segment if it matures.
- Avoid: GPL-3.0 license is incompatible with MIT redistribution.

### WatermarkRemover-AI (1.5k stars, MIT)
- Strengths: Florence-2 for detection + LaMa for inpainting, multi-language UI (EN/FR/ZH/JA/PT), audio preservation.
- Learn from: validates Florence-2 as a viable detection engine (already on ROADMAP as #22).

## Security, Privacy, and Reliability

- **Verified:** dependency floors in `requirements.txt` are current. All known CVEs (Pillow CVE-2026-25990/40192/42309, OpenCV CVE-2025-53644, PyTorch CVE-2025-32434/CVE-2026-24747) are addressed by stated minimums.
- **Verified:** `simple-lama-inpainting` uses `torch.load` internally. Project correctly mitigates by preferring ONNX Runtime and OpenCV 5 DNN backends. Documentation in `requirements.txt` is clear.
- **Verified:** optional VLM/segmentation paths execute remote code: `backend/ocr_vlm.py:99-100` uses `trust_remote_code=True`; `backend/segmentation.py:207` calls `torch.hub.load` without pinned ref. Commit `3f9e9b4` added gating, but pinned revisions or local paths should be required before enabling.
- **Verified:** `backend/remote_model_policy.py` exists and gates remote-code adapters. Existing ROADMAP items cover further hardening.
- **Verified:** release workflow downloads `wingetcreate.exe` from a mutable "latest" URL (`.github/workflows/build.yml:482-483`). Existing ROADMAP items cover this.
- **Verified:** preset import field allowlist shipped in commit `e3619d7` -- imported fields are now schema-filtered.
- **Verified:** support bundles shipped in commit `4806ba0` -- About dialog and CLI `--support-bundle` produce redacted diagnostics zips. But no GitHub issue template prompts users to attach them.
- **Likely:** SAM 3 license (Custom SAM License) has copyleft-like redistribution constraints. ROADMAP items #67/#115 should note this when implementing.
- **Likely:** ProPainter name overlap creates license confusion. VSR's "ProPainter" mode is TBE+LaMa (MIT-clean), NOT the ICCV 2023 ProPainter (NTU S-Lab non-commercial). `CLAUDE.md` documents this but user-facing docs do not.

## Architecture Assessment

- **Module boundaries are healthy:** GUI under `gui/` (6 files), 47 backend modules with clear separation, 31 test files covering smoke/regression/pipeline/release paths.
- **Stale architecture docs:** `docs/architecture.md` still references `VideoSubtitleRemover.py` as the GUI owner (moved to `gui/app.py`), lists a narrow test map (`test_hardening.py` only), omits VVC/cache/update/release/security modules, and carries a stale "current as of" marker. Already on ROADMAP.
- **Refactor candidate:** centralize optional model loading policy across `backend/ocr_vlm.py`, `backend/segmentation.py`, `backend/adapter_manifest.py`, `backend/model_hashes.py`, `backend/onnx_model_info.py`.
- **RapidOCR version cap:** `requirements.txt` pins `rapidocr>=2.0.0,<3.0.0`. RapidOCR v3.8.0+ has breaking API changes (constructor, return types). The current `_detect_rapid` function in `backend/detection.py` handles v1.x/v2.x output differences but has no v3.x path. Nine months of improvements (unified package, CoreML, TensorRT engine support) are blocked.
- **PP-OCRv6 opportunity:** PaddleOCR 3.7+ ships PP-OCRv6 (medium: +5.1% accuracy, 5.2x CPU speedup, 50-language single model). The current `paddleocr>=3.0.0,<4.0.0` floor allows it, but model selection and API output handling in `backend/detection.py` and `backend/paddle_compat.py` may need updates.
- **Drag-drop gap:** `gui/widgets.py:1252-1259` wraps TkinterDnD2 setup in a silent try/except. Without the package, drag-drop is entirely disabled and the drop area only responds to clicks. Issue #4/#5 reporter was affected.
- **Test gaps:** no tests for RapidOCR v3.x API surface; no test for drag-drop degradation feedback; no issue template validation test.

## Rejected Ideas

- **Cloud upload / SaaS processing** -- commercial tools prove demand, but local privacy and large-file control are VSR's core differentiator. Source: GhostCut/RecCloud feature analysis.
- **Docker-first distribution** -- upstream offers Docker, but VSR's target user is Windows GUI-first. Docker adds friction for the dominant user base. Already in ROADMAP "Explicitly not" section.
- **PyPI distribution of the full GUI** -- videowipe does this, but a tkinter app with model weights, FFmpeg dependency, and GPU setup is complex to ship via pip. The Run_VSR_Pro.bat auto-setup flow is better for the target user.
- **Add more default SOTA video diffusion inpainters now** -- awesome-lists and papers show many candidates, but existing ROADMAP already has adapter benches (#59-#65, #101). Adding more before reference-clip evidence would increase maintenance burden.
- **Make VLM/OCR remote models the default detector** -- PaddleOCR/RapidOCR releases support local-first OCR. Remote-code VLMs should remain opt-in until pinned, reviewed, and benchmarked. Source: HuggingFace trust_remote_code docs.
- **General-purpose image editor scope** -- IOPaint's plugin architecture is worth studying, but VSR should keep video subtitle/text removal as the primary workflow.
- **SAM 3 as default segmentation** -- Custom SAM License has copyleft-like redistribution clause; incompatible with MIT as a default dependency. Keep as opt-in. Source: facebookresearch/sam3 LICENSE.
- **Multi-user/collaboration** -- no architecture for shared state. Exportable reports are the better local-first path.
- **Subscription model** -- free MIT tool; EchoSubs shows $5.99/mo works commercially but VSR's value is in being the free alternative with better evidence output.

## Sources

OSS and adjacent:
- https://github.com/YaoFANGUK/video-subtitle-remover
- https://github.com/Sanster/IOPaint
- https://github.com/KKenny0/videowipe
- https://github.com/D-Ogi/WatermarkRemover-AI
- https://github.com/timminator/VideOCR
- https://github.com/SWHL/RapidVideOCR
- https://github.com/Purfview/InpaintDelogo
- https://github.com/suhwan-cho/awesome-video-inpainting

Commercial and community:
- https://jollytoday.com/subtitle-removal/
- https://reccloud.com/remove-subtitle
- https://echosubs.com/best-hardcode-subtitle-remover/
- https://anieraser.media.io/remove-subtitles-from-video.html
- https://forum.videohelp.com/threads/418726
- https://github.com/SubtitleEdit/subtitleedit/discussions/9562

Dependencies, standards, and advisories:
- https://github.com/RapidAI/RapidOCR/releases
- https://github.com/PaddlePaddle/PaddleOCR/releases
- https://github.com/opencv/opencv/releases/tag/5.0.0
- https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
- https://github.com/pyinstaller/pyinstaller/releases
- https://ffmpeg.org/download.html
- https://nvd.nist.gov/vuln/detail/CVE-2025-32434

Research and SOTA:
- https://arxiv.org/abs/2605.14894 (SEDiT -- mask-free one-step subtitle erasure)
- https://arxiv.org/abs/2603.21901 (CLEAR -- context-aware mask-free removal)
- https://arxiv.org/abs/2501.10018 (DiffuEraser)
- https://arxiv.org/abs/2503.05639 (VideoPainter, SIGGRAPH 2025)
- https://arxiv.org/abs/2412.00857 (FloED)
- https://arxiv.org/abs/2506.12853 (EraserDiT)
- https://arxiv.org/html/2606.13108 (PP-OCRv6)
- https://huggingface.co/microsoft/Florence-2-large

## Open Questions

- Needs live validation: does RapidOCR 3.x constructor change (`model_root_dir` required in some configs) break the current `_detect_rapid` flow, and does the new unified return type need a third handler path?
- Needs live validation: which FFmpeg build ships with current Windows releases (winget/scoop) -- does it include libvvenc for VVC and the Whisper filter?
- Needs live validation: has PP-OCRv6 (medium tier, 34.5M params) been tested on subtitle-class text at typical video resolutions, or only on document/scene-text benchmarks?
