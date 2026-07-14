# Video Subtitle Remover Pro -- Roadmap

Remaining-work backlog only: every item below is NOT yet implemented.
Completed items are deleted from this file; history lives in CHANGELOG.md and git.

## Audit-Driven Fixes

- [ ] P2 -- Drain PrefetchReader stderr pipe to prevent FFmpeg deadlock
  Why: LosslessIntermediateWriter creates FFmpeg with stderr=PIPE but never drains it; if FFmpeg emits enough warnings the pipe buffer fills and the pipeline deadlocks.
  Where: `backend/io.py` lines 1069-1072, 1131-1137.

- [ ] P2 -- Inline region editor replaces all regions instead of adding
  Why: _on_preview_region_release always replaces all existing regions and clears timed spans; users who configured multiple subtitle regions lose them when they use the inline editor.
  Where: `gui/preview_controller.py` lines 967-969.

- [ ] P2 -- Queue iteration in ETA probe not guarded by queue_lock
  Why: _probe_batch_eta iterates self.queue on the worker thread without holding queue_lock, risking RuntimeError if the user adds items during the probe.
  Where: `gui/processing_controller.py` lines 830-834.

- [ ] P3 -- Onboarding seen flag set before dialog is shown
  Why: If _show_onboarding fails after the flag is set, the user never sees onboarding.
  Where: `gui/app.py` line 2838.

- [ ] P3 -- Toast._active class-level list persists across test runs
  Why: Failed toast cleanup leaves references in the class-level list, potential memory leak in long sessions.
  Where: `gui/widgets.py` line 1199.

## Research-Driven Additions

### P0 — Now

- [ ] P0 — Validate final video integrity before destination promotion
  Why: `-shortest` can truncate video to a shorter audio stream, and successful FFmpeg exits are currently trusted without checking output duration, frames, or a decodable video stream.
  Evidence: `backend/processor.py:2558-2654`; https://github.com/allenk/VeoWatermarkRemover/issues/29
  Touches: `backend/processor.py`, `backend/io.py`, `backend/remux.py`, `tests/test_hardening.py`, `tests/test_reference_clips.py`
  Acceptance: FFprobe, with a bounded OpenCV fallback when absent, validates a video stream plus expected duration/frame envelope before any destination is replaced; short-audio and deliberately truncated fixtures fail closed while preserving an existing destination, and valid CFR/VFR/time-range outputs pass documented tolerances.
  Complexity: M

- [ ] P0 — Make model-cache ZIP imports bounded and transactional
  Why: The importer accepts unbounded manifests/members and commits files incrementally, exposing disk-exhaustion and partial-import failure modes.
  Evidence: `backend/cache_inventory.py:420-520`; https://docs.python.org/3/library/zipfile.html#decompression-pitfalls
  Touches: `backend/cache_inventory.py`, `backend/cli.py`, `gui/support_controller.py`, `tests/test_model_cache.py`
  Acceptance: Import preflights duplicate targets, member count, declared and actual sizes, compression ratio, total expansion, and free-space headroom; it stages and hashes every member before commit, rolls back on any failure, and has adversarial ZIP-bomb/truncation/duplicate/partial-write tests.
  Complexity: M

- [ ] P0 — Gate known-vulnerable FFmpeg release floors
  Why: VSR processes untrusted media through an external runtime, and the inspected FFmpeg 8.1.1 predates the 8.1.2 backports for CVE-2026-8461 and CVE-2026-30999.
  Evidence: `backend/ffmpeg_profiles.py:84-91`, `backend/release_verification.py:658`; https://ffmpeg.org/security.html
  Touches: `backend/ffmpeg_profiles.py`, `backend/release_verification.py`, `backend/support_bundle.py`, `tests/test_release_workflow.py`, `README.md`
  Acceptance: Self-test/support/release evidence parse the external version, mark 8.1.0-8.1.1 and 8.0.0-8.0.2 blocking, accept 8.1.2+/8.0.3+, explain upgrades, and strict release validation fails on known-vulnerable versions.
  Complexity: S

### P1 — Next

- [ ] P1 — Add scene-cut-safe temporal mask stabilization
  Why: Per-frame OCR misses and abrupt motion leave residue; short-window mask union is a research-backed way to retain observed target pixels without adopting a new diffusion model.
  Evidence: `backend/processor.py:1716-1923`; https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/6; https://arxiv.org/abs/2603.09283
  Touches: `backend/processor.py`, `backend/config.py`, `gui/config.py`, `gui/app.py`, `backend/reference_corpus.py`, `tests/test_hardening.py`
  Acceptance: An opt-in/configurable rolling mask union runs after propagation/refinement, resets at scene cuts and inactive timed spans, reduces missed-mask pixels on moving/shadow/dissolve fixtures, and never expands a mask into adjacent scenes or outside configured regions.
  Complexity: M

- [ ] P1 — Add residual-text, seam, and temporal-flicker quality checks
  Why: PSNR/SSIM/VMAF cannot tell users that glyph edges remain or a filled region flickers, which is the only current first-party quality complaint.
  Evidence: `backend/processor.py:733-932`; https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/6; https://arxiv.org/abs/2601.06391
  Touches: `backend/quality.py`, `backend/processor.py`, `backend/batch_report.py`, `gui/quality_controller.py`, `backend/reference_corpus.py`, `tests/test_reference_clips.py`
  Acceptance: Quality reports compare input/output OCR confidence inside the final mask, score mask-boundary discontinuity and inter-frame fill variance, warn on deterministic residual/flicker fixtures, avoid blocking clean/textured-background controls, and expose thresholds in sidecars/batch reports.
  Complexity: M

- [ ] P1 — Prove OCR and inpaint provider execution in self-test
  Why: Constructor/provider availability does not prove inference ran on the selected backend, leaving GPU fallback and broken-model reports ambiguous.
  Evidence: `backend/support_bundle.py:347-427`; https://github.com/YaoFANGUK/video-subtitle-remover/issues/242
  Touches: `backend/support_bundle.py`, `backend/model_downloads.py`, `backend/detection.py`, `backend/inpainters/lama.py`, `backend/cli.py`, `tests/test_support_bundle.py`
  Acceptance: An explicit no-download inference-smoke option runs a generated text image and masked frame through each locally ready selected backend, records actual provider/fallback/timing/model hash, returns failure when a claimed provider cannot execute, and remains skipped with a precise reason when weights are absent.
  Complexity: M

- [ ] P1 — Implement UI Automation providers for custom controls
  Why: In-process metadata and notifications do not let Narrator/NVDA enumerate or operate Canvas-based buttons, toggles, sliders, and queue rows.
  Evidence: `backend/a11y.py:1-19`, `gui/widgets.py`; https://learn.microsoft.com/en-us/windows/win32/winauto/uiauto-providersoverview
  Touches: `backend/a11y.py`, `gui/widgets.py`, `gui/app.py`, `setup.py`, `build_exe.bat`, `tests/test_hardening.py`
  Acceptance: Windows UIA clients can discover stable name/role/state/value, keyboard focus, and Invoke/Toggle/RangeValue/Selection behavior for every primary custom control; state changes raise UIA events, frozen builds retain the provider, and an automated external-tree smoke plus Narrator checklist passes.
  Complexity: XL

- [ ] P1 — Checkpoint and resume the encode/mux phase
  Why: `resume_checkpoint.py` saves only frame-by-frame OCR/inpaint state, so a crash after all frames are inpainted re-runs the entire encode; multi-hour jobs can lose hours of completed work.
  Evidence: `backend/resume_checkpoint.py` (tracks `frame_dir`/`next_frame` only); RESEARCH.md Security/Reliability.
  Touches: `backend/resume_checkpoint.py`, `backend/processor.py`, `backend/io.py`, `backend/cli.py`, `gui/processing_controller.py`, `tests/test_hardening.py`
  Acceptance: an encode-stage marker records that inpainting completed and where the encode/mux left off; resuming a job whose encode was interrupted skips re-inpainting and either resumes or restarts only the encode, verified by a fixture that kills the process mid-encode and re-runs to a byte-valid output without redoing detection.
  Complexity: L

- [ ] P1 — Recover gracefully from GPU out-of-memory mid-batch
  Why: Adaptive batch sizing probes VRAM/host RAM up front but has no handler if a batch still OOMs during inference, so borderline-VRAM users crash with a raw CUDA error instead of degrading.
  Evidence: `backend/processor.py` (no OutOfMemory/CUDA-memory recovery path); YaoFANGUK issues #240/#242 (GPU-accel reliability).
  Touches: `backend/processor.py`, `backend/config.py`, `backend/inpainters/_common.py`, `tests/test_hardening.py`
  Acceptance: a simulated OOM during a batch triggers cache clear + batch-size halving + retry down to size 1, then a logged CPU fallback; the run completes or fails with an actionable message and never leaves a corrupt partial output, covered by an injected-OOM test.
  Complexity: M

- [ ] P1 — Raise the ONNX Runtime dependency floor to >=1.25.0
  Why: Current floors (`onnxruntime-gpu>=1.21.0`, `onnxruntime-directml>=1.18.0`) predate the 1.25.0 parser integer-truncation heap-OOB hardening, and VSR runs untrusted OCR/inpaint ONNX models through this runtime.
  Evidence: `requirements.txt:37,44`, `setup.py:430,450`; https://github.com/microsoft/onnxruntime/releases/tag/v1.25.0
  Touches: `requirements.txt`, `setup.py`, `README.md`, `backend/dependency_caps.py`, `backend/release_verification.py`, `tests/test_dependency_caps.py`
  Acceptance: all onnxruntime install lines and dependency-cap checks require `>=1.25.0`, release/self-test evidence flags an installed runtime below 1.25.0 as vulnerable, and docs note the floor.
  Complexity: S

### P2 — Later

- [ ] P2 — Pin build-toolchain versions carrying LPE fixes (PyInstaller, NSIS)
  Why: `build_exe.bat` installs PyInstaller with no floor (<6.10.0 carries CVE-2025-59042 writable-CWD LPE) and `installer/vsr.nsi` documents no minimum NSIS (<3.11 carries CVE-2025-43715 temp-plugin-dir SYSTEM LPE); both matter for an unsigned Windows build.
  Evidence: `build_exe.bat:26-29`, `installer/vsr.nsi:1-17`; https://nvd.nist.gov/vuln/detail/CVE-2025-59042; https://github.com/advisories/GHSA-g9m2-7jc6-pmvf
  Touches: `build_exe.bat`, `installer/vsr.nsi`, `backend/release_verification.py`, `README.md`, `tests/test_release_workflow.py`
  Acceptance: the build installs/requires PyInstaller `>=6.10.0`, the installer build documents and (where detectable) checks NSIS `>=3.11`, and release verification records both tool versions and fails strict validation on known-vulnerable ones.
  Complexity: S

- [ ] P2 — Preflight free disk space before encode and rotate the log file
  Why: Free space is recorded but never gated, so a mid-encode disk-full dies with a raw OSError and can leave a half-written temp file; `vsr_pro.log` has no rotation and grows unbounded in long sessions.
  Evidence: `backend/batch_report.py:268`, `backend/cli.py:288` (records only); no `RotatingFileHandler` in the tree.
  Touches: `backend/processor.py`, `backend/io.py`, `backend/cli.py`, `gui/app.py`, `tests/test_hardening.py`
  Acceptance: before encode, an estimated-output-size vs free-space check warns/aborts cleanly (removing any temp file) instead of crashing, and the log handler rotates by size with a bounded backup count; both are covered by tests.
  Complexity: S

- [ ] P2 — Automatic bounded retry for transient batch-item failures
  Why: A single flaky failure (temporary GPU glitch, FFmpeg hiccup) marks a batch item FAILED with no retry, so long unattended batches finish with avoidable gaps.
  Evidence: `backend/batch_report.py:168-204` (FAILED-then-review, no retry); RESEARCH.md Reliability.
  Touches: `gui/processing_controller.py`, `backend/cli.py`, `backend/config.py`, `backend/batch_report.py`, `tests/test_hardening.py`
  Acceptance: an opt-in max-retry setting re-attempts a failed item up to N times with backoff, records each attempt in the batch report, distinguishes retriable from permanent errors, and defaults preserve current behavior.
  Complexity: M

- [ ] P2 — Add a full-pipeline --dry-run and machine-readable batch result on stdout
  Why: Only `--soft-subtitle-dry-run` and `--validate-config` exist; users must fully encode to validate detection/inpaint settings, and scripts have no structured result to consume.
  Evidence: `backend/cli.py:519-522,545-546,1097`; RESEARCH.md Architecture.
  Touches: `backend/cli.py`, `backend/processor.py`, `backend/batch_report.py`, `tests/test_hardening.py`
  Acceptance: `--dry-run` runs detection + mask + codec-availability checks and prints a per-file plan without encoding, and a `--json` flag emits a structured batch result (status, timings, output paths, warnings) to stdout; both are tested.
  Complexity: M

- [ ] P2 — Benchmark PP-OCRv6 on the reference corpus before any default swap
  Why: PP-OCRv6 (reachable through the existing RapidOCR/ONNX path) claims detection/speed gains, but the default must not change without validation on this repo's subtitle fixtures.
  Evidence: `backend/reference_corpus.py`, `backend/static_logo_benchmark.py`; https://github.com/RapidAI/RapidOCR/issues/686; https://github.com/PaddlePaddle/PaddleOCR/releases
  Touches: `backend/detection.py`, `backend/reference_corpus.py`, `backend/static_logo_benchmark.py`, `tests/test_detection_pipeline.py`, `README.md`
  Acceptance: a benchmark run scores PP-OCRv6 det/rec against the current default on the reference corpus (accuracy + wall-clock), records results as evidence, and the default detector changes only if v6 meets documented accuracy/speed floors on those fixtures.
  Complexity: M

- [ ] P2 — Add interpolated keyframes for moving manual regions
  Why: Timed regions are static rectangles, while moving watermarks require users to over-mask the whole path or depend on optional tracking models.
  Evidence: `backend/config.py:185-190`, `gui/app.py:3028-3431`; https://github.com/YaoFANGUK/video-subtitle-remover/issues/236; https://github.com/timminator/VideOCR/issues/140
  Touches: `gui/app.py`, `gui/preview_controller.py`, `backend/config.py`, `gui/config.py`, `backend/processor.py`, `backend/batch_report.py`, `tests/test_gui_smoke.py`
  Acceptance: Users can set two or more rectangle/polygon keyframes on a scrubbed timeline; masks interpolate deterministically only inside the span, preview matches processing, presets/CLI/sidecars round-trip the data, and settings format 4 migrates without losing existing timed regions.
  Complexity: L

- [ ] P2 — Make locale catalogs selectable and package-safe
  Why: The gettext layer drops territory/script subtags, has no locale setting, and frozen builds omit the catalog directory, so a real translation cannot ship reliably.
  Evidence: `backend/i18n.py:35-77`, `gui/app.py:144-152`, `build_exe.bat:40-46`, `locale/vsr.pot`
  Touches: `backend/i18n.py`, `gui/app.py`, `gui/config.py`, `locale/`, `build_exe.bat`, `installer/`, `tests/test_hardening.py`, `tests/test_release_workflow.py`
  Acceptance: Settings offers System/English/discovered catalogs, persists a full BCP-47-compatible locale with fallback chain, reloads translated widgets predictably, packages `.mo` files in portable/installer builds, and frozen-build tests prove a supplied territory-specific catalog is selected with source-string fallback.
  Complexity: M

### P3 — Under Consideration

- [ ] P3 — Evaluate FFmpeg 8.1 D3D12 filters/encoders for Windows-native GPU accel
  Why: FFmpeg 8.1 adds D3D12 H.264/AV1 encode and `scale_d3d12`/`deinterlace_d3d12`, a Windows-native GPU path for decode/encode legs that does not require CUDA and complements the multi-vendor GPU story.
  Evidence: `backend/ffmpeg_profiles.py`, `backend/decode_accel.py`, `backend/encoder.py`; https://9to5linux.com/ffmpeg-8-1-hoare-multimedia-framework-brings-d3d12-h-264-av1-encoding
  Touches: `backend/ffmpeg_profiles.py`, `backend/decode_accel.py`, `backend/encoder.py`, `backend/processor.py`, `tests/test_hardening.py`
  Acceptance: when the detected FFmpeg exposes D3D12 filters/encoders, an opt-in path uses them with automatic fallback to the current libx264/NVENC/QSV chain, validated on a fixture that confirms byte-valid output and clean fallback when D3D12 is absent. Needs live validation on FFmpeg >=8.1 hardware.
  Complexity: M

- [ ] P3 — Add an erase -> translate -> re-embed subtitle workflow
  Why: Erase-only leaves a localization gap that competitors close; the tool already has OCR, Whisper transcription, and SRT export to reuse, so re-embedding translated subtitles is an adjacent workflow rather than a new pipeline.
  Evidence: `backend/whisper_fallback.py`, `backend/nle_sidecar.py`, SRT export; https://github.com/chenwr727/SubErase-Translate-Embed
  Touches: `backend/cli.py`, `backend/processor.py`, `gui/app.py`, `backend/config.py`, `tests/test_hardening.py`
  Acceptance: an opt-in mode erases burned-in text, accepts or generates translated SRT, and re-burns it with configurable styling in one pass; translation providers stay pluggable/local-first, the feature is off by default, and reproducibility sidecars capture the chosen translation source.
  Complexity: L

- [ ] P3 — Add a clean-reference-frame fill override
  Why: A user-supplied clean plate is a deterministic, fast recovery path for static-camera overlays when temporal estimation or neural fill leaves residue.
  Evidence: https://helpx.adobe.com/after-effects/desktop/remove-objects-from-your-videos/content-aware-fill.html; https://news.ycombinator.com/item?id=45988018
  Touches: `gui/preview_controller.py`, `gui/config.py`, `backend/config.py`, `backend/processor.py`, `backend/tracking.py`, `backend/batch_report.py`, `tests/test_reference_clips.py`
  Acceptance: Users can choose a clean reference frame per timed region, preview translation/homography alignment and per-frame color matching, apply only inside the final mask, fall back when alignment confidence is low, and reproduce the result from persisted config/sidecar evidence.
  Complexity: L
