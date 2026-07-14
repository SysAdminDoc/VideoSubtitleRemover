# Research — Video Subtitle Remover Pro

## Executive Summary
Video Subtitle Remover Pro is a Windows-first, local-only Tkinter/CLI application for removing burned-in subtitles, text watermarks, and logo-like overlays. Its strongest current shape is an unusually broad offline pipeline: multiple OCR and inpaint fallbacks, timed/manual masks, checkpoints, quality reports, NLE/SRT/mask exports, reproducibility sidecars, support bundles, and license-guarded optional-model adapters, backed by ~596 passing tests. The security posture is already strong on the dependencies most researchers flag first (verified: `requirements.txt` pins `Pillow>=12.2.0`, `torch>=2.10.0`, `opencv-python>=4.12.0`, and OCR engines to tested majors), so the highest-value direction is now *reliability and provenance of the final artifact under long/adverse runs*, plus closing the second tier of dependency-floor gaps (ONNX Runtime, PyInstaller, NSIS) that the pinned-media stack does not yet cover.

Top opportunities, in priority order:

1. Checkpoint and resume the encode/mux phase so multi-hour jobs survive a crash after inpainting (verified gap: `backend/resume_checkpoint.py` tracks frames only).
2. Recover gracefully from GPU OOM mid-batch instead of crashing (verified: no OOM handler in `backend/processor.py`).
3. Raise the ONNX Runtime floor to `>=1.25.0` to pick up parser heap-OOB fixes (verified pins are `1.21.0`/`1.18.0`).
4. Pin the build toolchain versions that carry local-privilege-escalation fixes (PyInstaller `>=6.10.0`, NSIS `>=3.11`).
5. Preflight free disk space before encode, and rotate the growing `vsr_pro.log`.
6. Add automatic bounded retry for transient batch-item failures.
7. Add a full-pipeline `--dry-run` and machine-readable batch result on stdout for automation.
8. Benchmark PP-OCRv6 (via the existing RapidOCR/ONNX path) against the reference corpus before any default swap.
9. Evaluate FFmpeg 8.1 D3D12 filters/encoders as a Windows-native GPU accel path.
10. (Under consideration) An erase -> translate -> re-embed workflow that reuses existing OCR + Whisper + SRT export.

(Prior-cycle items — output-integrity validation, ZIP-bomb-safe cache import, FFmpeg CVE gate, temporal mask stabilization, residual/seam/flicker quality checks, provider-execution self-test, UIA providers, region keyframes, locale packaging, clean-reference-frame fill — remain valid and are already tracked in ROADMAP.md; they are not repeated here.)

## Product Map
- Core workflows: queue files/folders; choose OCR, inpaint, device, language, codec, and preset; review/correct automatic/timed masks; process, pause/resume, and inspect output/quality.
- Core workflows: strip or retain soft subtitles; export SRT/mask/frame-sequence/NLE sidecars; import NLE ranges; run repeatable CLI/glob batches.
- Core workflows: inspect backend/provider/model health, move verified model caches, export redacted support bundles, reproduce a run from `<output>.vsr.json`.
- User personas: privacy-sensitive Windows editors, restoration/archival users, multilingual subtitle operators, batch/CLI users, and maintainers validating local GPU/model stacks.
- Platforms and distribution: Windows 10/11; Python 3.10+ source/venv (3.12-3.13 for the documented CUDA path); Tkinter GUI + CLI; PyInstaller/NSIS/portable packaging; no hosted service or mandatory network path.
- Integrations/data flows: FFmpeg/FFprobe + OpenCV ingest/encode; RapidOCR/PaddleOCR/EasyOCR/Surya detection; TBE/LaMa/guarded adapters for fill; JSON settings, queue/checkpoint/report/sidecar state; optional CUDA, DirectML, OpenVINO, ONNX Runtime providers.

## Competitive Landscape
- YaoFANGUK/video-subtitle-remover (leading OSS): low-friction install and a large user base, but its loudest 2026 issue cluster is unreliable GPU acceleration (#240/#242/#237, incl. ROCm-on-Windows demand) and moving-watermark residue (#232/#236/#243). Learn: make the multi-vendor GPU path provably working and self-reporting; that is a direct wedge. Avoid its recurring GPU/package ambiguity.
- EchoSubs (echosubs.com, closed, ~$69 lifetime, Win/Mac local inpaint): the closest direct commercial competitor to "offline local subtitle removal," marketed explicitly as "no smudge blur box." Learn: residue-free proof clips are the headline buyers respond to. Avoid its closed/mac-first framing.
- chenwr727/SubErase-Translate-Embed (NEW): PaddleOCR + STTN erase, then translate and re-embed. Learn: the full localization loop (erase -> translate -> burn-in) is an unmet adjacent workflow this tool could reach by reusing its OCR, Whisper, and SRT layers. Avoid becoming a general translation suite.
- Rats20/EraseSubtitles + JollyToday/Extract-Subtitles-by-OCR (NEW): E2FGVI inpainting (a lighter-VRAM flow-guided option than ProPainter) and LLM post-OCR calibration for stylized fonts. Learn: cheaper inpainters and OCR-error correction before masking. Avoid Google-Drive-weight distribution and cloud LLM defaults.
- IOPaint: mature manual-mask reuse, brush/rectangle selection, explicit lossless controls. Learn: fast correction/retry loops. Avoid becoming a general image editor.
- After Effects / DaVinci Resolve Studio: tracking, keyframed masks, work ranges, reference clean frames, non-destructive review. Learn: guided correction. Avoid project-editor breadth and subscription coupling.
- ProPainter / MiniMax-Remover / Netflix VOID / 2026 removal research: define the temporal-propagation + mask-defect-tolerant frontier. Learn the scene-cut-safe mask ideas; avoid bundling non-commercial or 40GB-VRAM weights by default (see Rejected).

## Security, Privacy, and Reliability
- Verified (already mitigated): `requirements.txt:19-21,25-28` pins `Pillow>=12.2.0` (CVE-2026-25990 and related), `torch>=2.10.0` (CVE-2025-32434 / torch.load weights_only RCE), and `opencv-python>=4.12.0`. These common flags do not need new roadmap items.
- Verified (open, NEW): ONNX Runtime floors are `onnxruntime-gpu>=1.21.0` (`requirements.txt:37,44`, `setup.py:450`) and `onnxruntime-directml>=1.18.0` (`setup.py:430`), below the `1.25.0` release that hardened parser integer-truncation heap-OOB. VSR runs untrusted ONNX/OCR models through this runtime.
- Verified (open, NEW): `build_exe.bat:26-29` installs PyInstaller with no version floor; builds with `<6.10.0` carry CVE-2025-59042 (local privilege escalation via writable-CWD bootstrap). `installer/vsr.nsi` documents no minimum NSIS version; `<3.11` carries CVE-2025-43715 (temp-plugin-dir SYSTEM LPE). Both are build-time, unsigned-Windows-relevant.
- Verified (open, NEW): FFmpeg security page lists CVE-2026-8461 "PixelSmash" (MagicYUV heap OOB write, RCE via crafted media, fixed 8.1.2) plus 8.1-line fixes (CVE-2026-40962, CVE-2025-69693). The existing ROADMAP FFmpeg-gate item covers *detecting* a vulnerable version; if any FFmpeg is bundled/shipped it must be `>=8.1.2`.
- Verified (open, NEW): `backend/resume_checkpoint.py` checkpoints frame-by-frame OCR/inpaint state only; the encode+mux phase has no resume. A crash after all frames are inpainted re-runs the entire encode — hours lost on long videos.
- Verified (open, NEW): `backend/processor.py` has no OutOfMemory handler around adaptive batch sizing (`grep` for OOM/CUDA-memory recovery returns nothing); a batch that still OOMs crashes with a raw CUDA error instead of downscaling.
- Verified (open, NEW): free disk space is *recorded* (`backend/batch_report.py:268`, `backend/cli.py:288`) but never *gated* before encode; a mid-encode disk-full dies with OSError and can leave a half-written temp file. `vsr_pro.log` has no rotation handler (no `RotatingFileHandler` in the tree).
- Verified (correction to record): the `Roadmap_Blocked.md` "MiniMax-Remover ... weights not published" reason is out of date — weights are now on HuggingFace (`zibojia/minimax-remover`, updated Apr 2026) under **CC-BY-NC-4.0**. It remains blocked, but the true blocker is the non-commercial license, not availability.
- Verified: the local/privacy boundary stays strong — remote-model loading is opt-in, source/hash-gated (`backend/model_hashes.py` verifies SHA-256 and falls back to cv2 on mismatch by design), support data is redacted, outputs go through temp paths, and cloud processing is not required. Preserve this default.

## Architecture Assessment
- `backend/processor.py` (~123 KB) owns decode, mask, inpaint, encode, mux, quality, and finalization. Extract a thin encode/finalize boundary to host the encode-phase checkpoint and OOM-retry logic rather than growing the monolith.
- `backend/resume_checkpoint.py` models only `frame_dir`/`next_frame`; adding an encode-stage marker there (or a sibling) is the low-risk seam for resumable encoding.
- `backend/cli.py` (~63 KB) has `--validate-config` and `--soft-subtitle-dry-run` but no whole-pipeline `--dry-run` and no structured machine-readable result on stdout; keep new logic in helpers and let CLI stay parse/print.
- `gui/config.py` bumps `VSR_SETTINGS_FORMAT` (0->4) but the migration path transforms nothing and corrupt-settings backup can fail silently on a read-only parent — fragile for future key removals; not urgent, note for the next settings change.
- Existing foundations to reuse: `backend/model_hashes.py` (hash gate), `backend/batch_report.py` (free-space + report schema), `backend/reference_corpus.py`/`static_logo_benchmark.py`/`mask_free_benchmark.py` (fixture harness for a PP-OCRv6 A/B), `backend/ffmpeg_profiles.py` (version parse for the D3D12/CVE work), `backend/io.py` atomic helpers.
- Test status is strong (~596 passing) but has no coverage for encode-phase resume, GPU-OOM downscale, disk-full preflight, or log rotation; each new item below carries its own tests.
- Coverage decision: security, reliability/resilience, observability, testing, docs, packaging, i18n, and offline resilience are represented. A general third-party plugin marketplace, mobile clients, and a multi-user/cloud service remain intentionally excluded.

## Rejected Ideas
- Bundle MiniMax-Remover as a default eraser (github.com/zibojia/MiniMax-Remover): weights are CC-BY-NC-4.0 — incompatible with MIT redistribution and any commercial tier. Belongs in the license-gated/blocked lane, not the actionable roadmap.
- Bundle Netflix VOID (Apache-2.0, github.com/Netflix/void-model): permissive but needs ~40GB VRAM and text-prompt quadmasks — impractical for consumer Windows GPUs and subtitle strips.
- Bundle EraserDiT / LoVoRA / OmnimatteZero: no published permissive weights/code as of mid-2026 — watch-items, not implementable now.
- Switch the default OCR stack to PP-OCRv6 immediately (RapidOCR/PaddleOCR 2026 releases): accuracy/speed claims are real but unvalidated on this repo's subtitle fixtures; benchmark first (roadmapped), do not swap defaults blind.
- Migrate off DirectML to Windows ML now: strategically correct (DirectML EP is in maintenance; Windows ML auto-selects NVIDIA/AMD/Intel/NPU EPs) but requires Windows ML SDK + AMD/Intel test hardware — already tracked in Roadmap_Blocked.md, keep it there.
- Full erase->translate->re-embed as a core feature (chenwr727/SubErase-Translate-Embed): valuable but a large scope jump toward a translation suite; kept Under Consideration in ROADMAP, not committed.
- Hosted/cloud cleanup or cloud OCR: contradicts the local/offline privacy position.
- macOS/Linux packaging now: launchers, installer, CUDA/DirectML, and release proof are Windows-specific; stabilize artifact trust and correction quality first.
- General third-party plugin marketplace: arbitrary-code discovery would weaken the existing manifest/hash/license boundary.
- GitHub Actions/Dependabot migration: repo policy requires local builds/tests/releases.

## Sources
### Project
- https://github.com/SysAdminDoc/VideoSubtitleRemover
- https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/6

### OSS / Commercial Competitors and Adjacent Tools
- https://github.com/YaoFANGUK/video-subtitle-remover/issues
- https://echosubs.com/best-ai-subtitle-remover-2026
- https://github.com/chenwr727/SubErase-Translate-Embed
- https://github.com/Rats20/EraseSubtitles
- https://github.com/JollyToday/Extract-Subtitles-by-OCR
- https://github.com/Sanster/IOPaint
- https://github.com/timminator/VideOCR/issues/140

### Models and Research (2026)
- https://github.com/zibojia/MiniMax-Remover
- https://huggingface.co/zibojia/minimax-remover
- https://github.com/Netflix/void-model
- https://arxiv.org/abs/2506.12853
- https://arxiv.org/abs/2512.02933
- https://github.com/dvirsamuel/OmnimatteZero

### OCR / Runtime (2026)
- https://huggingface.co/blog/baidu/ppocrv5
- https://github.com/PaddlePaddle/PaddleOCR/releases
- https://github.com/RapidAI/RapidOCR/issues/686
- https://github.com/datalab-to/surya
- https://github.com/microsoft/onnxruntime/releases/tag/v1.25.0
- https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers

### FFmpeg (2026)
- https://ffmpeg.org/security.html
- https://jfrog.com/blog/pixelsmash-critical-ffmpeg-vulnerability-turns-media-files-into-weapons/
- https://9to5linux.com/ffmpeg-8-1-hoare-multimedia-framework-brings-d3d12-h-264-av1-encoding

### Security Advisories (build/runtime)
- https://nvd.nist.gov/vuln/detail/CVE-2025-59042
- https://github.com/advisories/GHSA-g9m2-7jc6-pmvf
- https://nvd.nist.gov/vuln/detail/CVE-2025-32434
- https://github.com/opencv/opencv-python/issues/1186

## Open Questions
- Can the reporter of `SysAdminDoc/VideoSubtitleRemover#6` supply a redistributable failing clip plus a redacted support bundle? Without it, the residue mechanism and real-world acceptance threshold remain Needs live validation.
- Does any shipped/portable build carry a bundled FFmpeg binary, or is FFmpeg always external/user-supplied? This decides whether the `>=8.1.2` floor is a build gate or only a self-test advisory.
