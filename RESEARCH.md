# Research -- Video Subtitle Remover Pro

## Executive Summary

Video Subtitle Remover Pro is a mature (v3.17.1, 32K Python LOC, 386 tests) Windows-first desktop tool for removing hardcoded subtitles via multi-engine OCR + TBE/LaMa-ONNX inpainting. Its strongest current shape is trustable local batch processing with codec breadth, quality reports, crash-resume, support bundles, and release verification. The codebase is well-split (gui/ + backend/ + 20 focused modules), and the feature set exceeds every OSS competitor surveyed.

The highest-value direction is correctness hardening: a deep code audit surfaced 14 concrete bugs and validation gaps across GUI state management, CLI command generation, config validation, and subprocess lifecycle. These are P1/P2 fixes that improve every user's experience without new dependencies or model integrations.

Top opportunities in priority order:

1. Fix Toast cleanup race condition (list mutation during iteration).
2. Disable import buttons during processing (prevents duplicate queue entries).
3. Complete the CLI command builder (missing ~15 config fields, making "Copy CLI command" produce incorrect commands).
4. Clear SRT entries between consecutive process_video() calls (cross-run contamination).
5. Add graceful handling of corrupt/truncated video input.
6. Validate ProcessingConfig device strings and enforce subtitle_area/subtitle_areas mutual exclusion.
7. Add CLI numeric range validation and unknown-field warnings for JSON config overlays.
8. Fix elapsed timer leak on processing error.
9. Add missing test coverage: full cascade failure, temp cleanup on exception, subprocess termination on cancel, queue autosave round-trip.
10. Update support bundle dependency list to include `rapidocr` and scrub `work_directory`.

## Product Map

- Core workflows: import video/images, choose automatic/manual/soft-subtitle flows, preview/refine masks, process through AUTO/TBE/LaMa-ONNX/OpenCV/Torch routes, review quality reports and batch summaries, restore from crash checkpoints.
- User personas: Windows video editors, archive/restoration users with large batches, nontechnical GUI users, technical CLI/evidence users, privacy-sensitive users who reject cloud upload.
- Platforms and distribution: Python/tkinter source, Windows .bat/PowerShell launchers, PyInstaller/NSIS release path, GitHub Actions CI/CD with SBOM/pip-audit/release verification, winget preparation, limited source-install on Linux/macOS.
- Key integrations: FFmpeg/FFprobe decode/encode/remux, RapidOCR/PaddleOCR/Surya/EasyOCR/OpenCV/VLM detection cascade, ONNX Runtime DirectML/CUDA/CPU, PyTorch optional, JSON settings/checkpoints/reports, cache inventory, scrubbed support bundles.

## Competitive Landscape

### YaoFANGUK/video-subtitle-remover (upstream)
- Strengths: large user base, Apache-2.0, Docker/CPU/DirectML/CUDA package matrix, simple subtitle removal.
- Learn from: packaging clarity for users who cannot build Python/GPU environments.
- Avoid: weaker recovery/evidence surface; no quality reports, support bundles, or multi-engine fallback.

### GhostCut / JollyToday
- Strengths: automatic text/logo removal, batch scale, API workflow, adjacent translation/dubbing.
- Learn from: target-mode clarity and integration-ready job semantics.
- Avoid: cloud-only processing, credit/subscription economics.

### EchoSubs
- Strengths: offline desktop positioning, simple workflow, low-friction paid packaging.
- Learn from: plain recovery expectations for nontechnical users.
- Avoid: opaque quality claims without open benchmarks or CLI reproducibility.

### videowipe
- Strengths: PyPI/API/CLI distribution, natural-language intent, Docker CPU/GPU, external model command contract.
- Learn from: replayable command surface and intent-to-config helper.
- Avoid: GPL-by-default pressure and ProPainter-family bundled defaults.

### IOPaint
- Strengths: one-click installers, local web UI, CPU/GPU/Apple Silicon, model manager, plugin system.
- Learn from: explicit model/plugin lifecycle and dependency status UI.
- Avoid: drifting into a general image editor.

### WatermarkRemover-AI
- Strengths: Florence-2 + LaMA pipeline, fade mask handling, themes, multilingual UI.
- Learn from: moving/fading watermark UX, community demand for GPU diagnostics and portable installs.
- Avoid: dependency failures and ambiguous GPU state.

### RapidVideOCR / VideOCR
- Strengths: hard-subtitle extraction to SRT/ASS, local PaddleOCR, CLI.
- Learn from: community requests for queue autosave, temp-dir selection, copy-CLI-command, auto language selection.
- Avoid: cloud OCR fallback as default.

## Security, Privacy, and Reliability

### Active Risks

- **Toast cleanup race** (`gui/widgets.py`): `Toast._active` list is mutated during iteration in `_position()` when a concurrent fade-out completes. Can skip toasts or index out-of-bounds. Severity: low (cosmetic), but affects perceived polish.
- **DragDropFrame buttons active during processing** (`gui/widgets.py`): `add_files_btn` and `add_folder_btn` in the empty queue state are never disabled by `set_enabled()`, so users can add files mid-batch, creating race conditions with the queue lock.
- **SRT cross-run contamination** (`backend/processor.py`): `_srt_entries` is appended to but never cleared between calls to `process_video()`. A second run produces an SRT file containing the previous run's entries.
- **Elapsed timer leak** (`gui/app.py`): `_elapsed_timer_id` is set on processing start but only cancelled on close confirmation. If processing errors out early, the timer keeps firing until shutdown.
- **Selected queue item unprotected** (`gui/app.py`): `_selected_queue_item_id` is written from UI callbacks and read from preview logic without lock guards.

### Verified Safe

- `requirements.txt` pins around recent OpenCV/Pillow/RapidOCR/PaddleOCR CVEs.
- CI runs pip-audit, SBOM generation, libpng advisory check, release evidence.
- `Roadmap_Blocked.md` correctly tracks the OpenCV/libpng CVE-2026-22801 floor.
- `backend/remote_model_policy.py` gates Florence-2, Qwen2.5-VL, SAM2, MatAnyone, CoTracker3.
- Support bundles scrub sensitive keys and paths.

### Missing Guards

- **Support bundle dependency list** (`backend/support_bundle.py`): `_DEPENDENCY_PACKAGES` does not include `rapidocr` (only `rapidocr-onnxruntime`), so the primary OCR engine version is not captured in bug reports.
- **Support bundle work_directory leak**: `_SENSITIVE_KEYS` does not include `work_directory` (new field from `38200b8`), so user-chosen work directories could appear unscrubbed.
- **CLI command builder incomplete** (`gui/widgets.py:_build_cli_command`): Only emits ~8 of 30+ config fields. Missing: `mask_feather_px`, `whisper_fallback`, `karaoke_grouping`, `remove_chyrons`, `keep_subtitles`, `detection_vertical`, `time_start`, `time_end`, `loudnorm_target`, `edge_ring_px`, `colour_tune_enable`, `phash_skip_enable`, `kalman_tracking`, `temporal_smooth_radius`, `export_srt`, `export_mask_video`, `output_frames`, `auto_band`. Users copying CLI commands get incomplete reproduction instructions.
- **No validation of `device` string**: `ProcessingConfig.device` accepts any string (e.g., "gpu", "1"); invalid values fail late in CUDA init instead of at config parse time.
- **No mutual exclusion of `subtitle_area`/`subtitle_areas`**: Both can be set simultaneously; mask logic chooses one arbitrarily.

## Architecture Assessment

### Module Boundaries

The backend module split (RFP-L-1) is healthy. GUI state in `gui/`, pipeline in `backend/`, clear re-export shim in `backend/processor.py`. The 7 backend submodules (detection, tracking, io, encoder, quality, inpainters, cli) have clean ownership.

### Refactor Candidates

- **`_build_cli_command`** (`gui/widgets.py:28-68`): Should walk `ProcessingConfig` fields systematically (like `to_dict` does) instead of a manual enumeration that drifts. Every new config field silently becomes invisible in copied commands.
- **Normalization completeness** (`backend/config.py`): `normalized()` has explicit coercion for ~40 fields, but `work_directory` is normalized only in `gui/config.py`, not in the backend normalizer. CLI paths bypass the GUI normalizer.

### Test Gaps

- No test for full OCR cascade failure (all engines absent + no OpenCV fallback).
- No test for `_cleanup_temp_output` invocation on mid-inpaint exceptions.
- No test for FFmpeg subprocess termination behavior on cancel (SIGTERM vs SIGKILL).
- No test for queue autosave/restore round-trip (feature landed at `36737c4`).
- No test for corrupt/truncated video input (malformed MP4, missing codecs).
- No test for CLI numeric flag out-of-range values flowing through to ProcessingConfig.
- No test for JSON config overlay with unknown/typo field names.

### Category Coverage Review

- **Security**: Active CVE monitoring, pip-audit, model provenance -- strong.
- **Accessibility**: High-contrast theme, keyboard selection, UIA scaffold -- adequate.
- **i18n**: Scaffold exists but no `.mo` files -- future work (on ROADMAP).
- **Observability**: JSON-line log, batch reports, support bundles -- strong.
- **Testing**: 386 tests, 7.4K lines -- good coverage, specific gaps noted above.
- **Distribution**: PyInstaller/NSIS/winget/GitHub Actions -- strong.
- **Plugin ecosystem**: Inpainter registry exists; no filesystem auto-discovery -- intentional.
- **Mobile**: Android/iOS ports on "under consideration" -- correct placement.
- **Offline/resilience**: Checkpoint/resume, queue autosave -- recently improved.
- **Multi-user**: Intentionally rejected (single-user desktop).
- **Migration**: Settings schema versioning with `_migrate_settings()` -- adequate.

## Rejected Ideas

- **Cloud/SaaS integration** (GhostCut, Media.io, HitPaw, Vmake, Creatok): conflicts with local-first privacy. Source: competitive survey.
- **Docker-first distribution**: useful for some Linux users but not the dominant Windows release path. Source: YaoFANGUK upstream.
- **Bundled noncommercial model defaults** (Real ProPainter, DiffuEraser, COCOCO): license constraints. Source: Roadmap_Blocked.md.
- **Full image-editor plugin marketplace**: too broad for a subtitle-removal tool. Source: IOPaint architecture.
- **REST/multi-user server mode now**: local single-user reliability is higher priority. Source: architecture review.
- **PyPI full-GUI distribution as priority**: Windows installer + winget serves the user base better. Source: community feedback.
- **Mask-free removal as near-term default** (CLEAR, SEDiT): promising but unvalidated on VSR's workloads. Source: arxiv.org/abs/2603.21901, arxiv.org/abs/2605.14894.
- **VOID as bundled default**: best-in-class results (Netflix, HuggingFace weights available) but targets general object removal, not subtitle-specific; large model, unclear license terms for redistribution. Track as opt-in research adapter. Source: void-model.github.io.
- **Automatic `subtitle_area`/`subtitle_areas` mutual exclusion error**: enforcement would break existing settings files that carried both. Prefer silent precedence with logged warning.

## Sources

### OSS and Direct Competitors

- https://github.com/YaoFANGUK/video-subtitle-remover
- https://github.com/KKenny0/videowipe
- https://github.com/JollyToday/GhostCut_Remove_Video_Text
- https://github.com/D-Ogi/WatermarkRemover-AI
- https://github.com/Sanster/IOPaint
- https://github.com/Purfview/InpaintDelogo
- https://github.com/SWHL/RapidVideOCR
- https://github.com/timminator/VideOCR

### Commercial and Community Signal

- https://jollytoday.com/subtitle-removal/
- https://www.echosubs.com/remove-hardcoded-subtitles-offline
- https://anieraser.media.io/remove-subtitles-from-video.html
- https://forum.videohelp.com/threads/418629

### Dependencies and Platform

- https://github.com/RapidAI/RapidOCR/releases (v3.9.0: PP-OCRv6 default models)
- https://github.com/PaddlePaddle/PaddleOCR/releases (PP-OCRv6 June 2026)
- https://onnxruntime.ai/docs/reference/releases-servicing.html (v1.27.0)
- https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview
- https://github.com/opencv/opencv-python/issues/1186
- https://nvd.nist.gov/vuln/detail/CVE-2025-32434
- https://nvd.nist.gov/vuln/detail/CVE-2026-24747 (PyTorch torch.load RCE)
- https://cnx-software.com/2026/06/10/opencv-5-release-new-dnn-engine-with-enhanced-onnx-and-llm-vlm-support-intel-arm-and-risc-v-hardware-optimizations/
- https://pytorch.org/blog/pytorch-2-7/
- https://pyinstaller.org/en/stable/CHANGES.html (6.21.0)

### Research and SOTA

- https://huggingface.co/charlesw09/CLEAR-mask-free-video-subtitle-removal
- https://arxiv.org/abs/2605.14894 (SEDiT mask-free subtitle erasure)
- https://arxiv.org/abs/2503.05639 (VideoPainter)
- https://rose2025-inpaint.github.io/
- https://void-model.github.io/ (VOID, Netflix, HuggingFace weights available)
- https://arxiv.org/abs/2506.12853 (EraserDiT, Mango TV)
- https://arxiv.org/abs/2505.24873 (MiniMax-Remover)

### Ecosystem Intelligence (June 2026)

- RapidOCR v3.9.0 defaults to PP-OCRv6 det/rec models; package size doubled 15 MB to 29 MB; min Python bumped to 3.8.
- OpenCV 5 released at CVPR June 2026: DNN fully rewritten, 22% to 80%+ ONNX operator coverage, CPU-only for now.
- ONNX Runtime v1.27.0: WebGPU EP v0.1.0 shipped as standalone plugin; DirectML EP in sustained engineering (WinML is forward path).
- PyTorch 2.7 stable: CUDA 12.8 wheels for Blackwell; Python 3.14 still has no CUDA wheels.
- FFmpeg 8.0 "Huffman": whisper audio filter shipped for on-device ASR, VVC VA-API decode.
- VOID (Netflix, April 2026): open-sourced on HuggingFace, removes objects + physical interactions, won 64.8% user preference vs Runway/ROSE/MiniMax/ProPainter.

## Open Questions

- Needs live validation: which CC0/public-domain clips can be redistributed in `tests/clips/` for #54 without legal friction?
- Needs live validation: does Windows ML expose a Python-packagable ONNX inference path suitable for VSR Pro today, or should DirectML remain the only AMD/Intel GPU path? DirectML EP is now in sustained engineering; WinML is the Microsoft-strategic forward path.
- Needs live validation: which Windows FFmpeg distribution includes both `libvvenc` and `--enable-whisper` for VVC/Whisper default claims?
- Needs live validation: does RapidOCR v3.9.0 PP-OCRv6 default model change break any existing detection path in VSR Pro? Model filenames and config paths may differ.
