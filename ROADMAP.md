# Video Subtitle Remover Pro -- Roadmap

Remaining-work backlog only: every item below is NOT yet implemented.
Completed items are deleted from this file; history lives in CHANGELOG.md and git.

## Research-Driven Additions

### P0 — Now

### P1 — Next

### P2 — Later

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

## Audit Findings (2026-07-14 deep audit)

- [ ] P2 — Confine VapourSynth `.vpy` execution to an allowlisted script directory
  Why: `try_open_vpy` runs `exec()` on any `.vpy` reached by extension when `VSR_VAPOURSYNTH=1`; a batch/folder sweep could pull in an untrusted script and execute arbitrary code.
  Where: `backend/vapoursynth_bridge.py` (exec of script), `backend/io.py` (`_open_capture` dispatch by extension).

- [ ] P2 — Require a pinned commit SHA for CoTracker `torch.hub.load(trust_repo=True)`
  Why: `trust_repo=True` executes the remote repo's `hubconf.py`; if the resolved revision is a mutable branch/tag rather than a pinned SHA, a moved/compromised upstream can run code (gated behind `VSR_COTRACKER`).
  Where: `backend/segmentation.py` lines ~679-690; `resolve_remote_model_source` revision handling.

- [ ] P3 — Stabilize GUI tests under a full-suite run
  Why: A couple of Tk tests pass in isolation but fail with "Tcl wasn't installed properly" when the whole suite creates many Tk roots in one process; this is test-harness resource exhaustion, not a product bug, but it makes the suite flaky.
  Where: `tests/test_gui_smoke.py`, `tests/test_hardening.py` (GUI test roots); needs per-test Tk teardown or subprocess isolation.

- [ ] P3 — First-run-friendly copy for advanced-setting tooltips
  Why: Several advanced toggles surface backend jargon without explaining the tradeoff (e.g. "Kalman box tracking", "Flow-warped temporal exposure (motion-heavy)", "remuxing"); a copy pass would help first-time users choose settings with confidence.
  Where: `gui/app.py` advanced-settings tooltips; `gui/processing_controller.py` status strings.
