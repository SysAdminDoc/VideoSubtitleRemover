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

### P1 — Next

- [ ] P1 — Implement UI Automation providers for custom controls
  Why: In-process metadata and notifications do not let Narrator/NVDA enumerate or operate Canvas-based buttons, toggles, sliders, and queue rows.
  Evidence: `backend/a11y.py:1-19`, `gui/widgets.py`; https://learn.microsoft.com/en-us/windows/win32/winauto/uiauto-providersoverview
  Touches: `backend/a11y.py`, `gui/widgets.py`, `gui/app.py`, `setup.py`, `build_exe.bat`, `tests/test_hardening.py`
  Acceptance: Windows UIA clients can discover stable name/role/state/value, keyboard focus, and Invoke/Toggle/RangeValue/Selection behavior for every primary custom control; state changes raise UIA events, frozen builds retain the provider, and an automated external-tree smoke plus Narrator checklist passes.
  Complexity: XL

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
