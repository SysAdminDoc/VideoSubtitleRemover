# Video Subtitle Remover Pro -- Roadmap

Actionable work only. Historical and completed roadmap material is archived in CHANGELOG.md; blocked work is kept in Roadmap_Blocked.md.

## Remaining work

Source research: `RESEARCH.md`, dated 2026-08-21. There is no P0.

Rejected on purpose (do not re-file): CLEAR/ROSE/EraserDiT/VOID adapters, Qt rewrite,
REST/Gradio, Mac/ROCm, re-adding GitHub Actions, Sigstore/attestations for SmartScreen,
winget/Store submission. Reasons in RESEARCH.md Rejected Ideas.

Implementer index -- drain in this order.

P2:
- RM-281 S -- quality report harmonic mean + worst-frame index, seek preview to it
- RM-282 M -- HWND Dynamic Annotation on custom Canvas controls (not a UIA provider)
- RM-284 S -- gate language picker on catalog coverage; document translation policy
- RM-283 L -- donor-video clean reference (still-plate path already exists)
- RM-292 M -- fade-in / fade-out mask temporal extension
- RM-294 S -- wrap remaining `N_(f"...")` status toasts so they extract and translate

P3:
- RM-286 S -- Last FFmpeg commands view + support-bundle ring buffer
- RM-288 S -- record `SOURCE_DATE_EPOCH` / `PYTHONHASHSEED`; claim semantic rebuilds only
- RM-289 S -- PyInstaller floor 6.22.2 + onedir tripwire (advisory does not apply today)
- RM-293 S -- README Detection / Troubleshooting copy still describes the 3.8 picker
- RM-295 S -- unaudited surfaces from the 2026-08-21 pass (watch folder, Whisper, NLE, Docker)

### P2

- [ ] P2 -- RM-281: Report harmonic mean and the worst-frame index in the quality report
  Why: this product's characteristic defect is a handful of bad frames on an otherwise sharp clip, and a mean is structurally blind to exactly that.
  Evidence: libvmaf's `pooled_metrics` reports min/max/mean/harmonic_mean per metric; harmonic mean weights low frames more heavily.
  Touches: `backend/quality.py`, `backend/_quality_mixin.py` (`_compute_quality_report`), `backend/quality_gate.py`, `gui/quality_controller.py`, `tests/test_hardening_quality.py`.
  Acceptance: the report carries `harmonic_mean` alongside `mean` for PSNR and SSIM plus the index and score of the worst sampled frame; the GUI quality pane shows that frame number as a control that seeks the preview to it (not a dead label); a synthetic clip with one badly-filled frame shows a measurable mean/harmonic-mean gap in a test; existing sidecar consumers tolerate the added keys.
  Confidence: Verified
  Complexity: S

- [ ] P2 -- RM-282: Annotate custom Canvas controls for Windows accessibility
  Why: every custom control is an anonymous pane to a screen reader, and the cheap annotation route was never separated from the blocked full-provider route.
  Evidence: Tk 8.6.15 is on this machine, so `tk accessible` (TIP 733, Tk 9.1) is unreachable. Each Tk widget is a real HWND via `winfo_id()`, and `IAccPropServices::SetHwndProp` can set role, name, help and a LiveRegion. Distinct from the blocked full UIA provider item.
  Touches: `backend/a11y.py`, `gui/widgets.py`, `gui/layout_build.py`.
  Acceptance: each custom control annotates a role and name on its own HWND; tooltip text is attached as help; the footer/progress surface is a LiveRegion; annotation failure degrades silently.
  Confidence: Verified (mechanism) / Needs live validation (speech quality)
  Complexity: M

- [ ] P2 -- RM-283: Accept a donor video as a clean reference, not just a still plate
  Why: when a clean or differently-subbed release exists the background does not need to be invented.
  Evidence: `backend/reference_fill.py` already ships alignment modes under `CLEAN_REFERENCE_SCHEMA`, but `normalize_clean_reference` only reads a still plate per timed region.
  Touches: `backend/reference_fill.py`, `backend/_clean_ref_mixin.py`, `backend/config.py` with the `gui/config.py` mirror, `backend/cli.py` (via `_apply_cli_config_overlays`, not `main()`), `gui/region_controller.py`.
  Acceptance: a donor video path can be attached as a clean reference; frames are matched by timestamp with a configurable offset; a per-frame alignment-confidence floor falls back to the normal inpaint path; provenance records donor hash and offset; schema version bumps rather than mutating v1.
  Confidence: Verified (gap and foundation) / Needs live validation
  Complexity: L

- [ ] P2 -- RM-284: Gate the language picker on catalog coverage and document the translation policy
  Why: the picker advertises localization the build does not have -- only the `qps-Ploc` pseudo-locale ships -- so choosing a language appears to do nothing.
  Evidence: `gui/layout_build.py` builds the picker from `available_catalogs()`, and `locale/` contains only `vsr.pot` and `qps-Ploc`.
  Touches: `scripts/i18n_catalogs.py`, `gui/layout_build.py`, `backend/i18n.py`, README localization section.
  Acceptance: a catalog below a documented coverage threshold is excluded from the picker (pseudo-locale exempt); the picker shows English only when no catalog qualifies.
  Confidence: Verified
  Complexity: S

- [ ] P2 -- RM-292: Extend detected subtitle masks across fade-in and fade-out frames
  Why: fading hardsubs are the case where per-frame detection and steady-state alpha unmix both miss.
  Evidence: WatermarkRemover-AI `--fade-in`/`--fade-out`; VSR has zero `fade_in`/`fade_out` config keys.
  Touches: `backend/config.py` + `gui/config.py` mirror, `backend/cli.py` (hang flags on `_apply_cli_config_overlays`, not `main()`), mask compose path, `gui/layout_build.py`.
  Acceptance: `--fade-in N` / `--fade-out N` (and matching GUI sliders, default 0) hold the last confident mask N frames before first detection and N frames after last detection of that track; N=0 remains byte-identical on the reference corpus.
  Confidence: Verified (gap) / Needs live validation
  Complexity: M

- [ ] P2 -- RM-294: Replace remaining `N_(f"...")` status toasts with extractable `tr()`/`ntr()` templates
  Why: `N_` is an extraction marker, not a translator, and an f-string inside it never becomes a catalog msgid.
  Evidence: still present in `gui/app.py` (about 20 sites), `gui/settings_controller.py` (about 12), `gui/quality_controller.py` (about 7), `gui/preview_controller.py` (1). Region editor save/keyframe copy was wrapped in the 2026-08-21 audit.
  Touches: those GUI files, `locale/vsr.pot`.
  Acceptance: `rg "N_\\(f" gui` is empty; each former site uses `tr("...{name}...").format(...)` or `ntr()`; catalogs updated last.
  Confidence: Verified
  Complexity: S

### P3

- [ ] P3 -- RM-286: Add a "Last FFmpeg commands" diagnostics view
  Why: FFmpeg failures are the most opaque error class in the app, and the argv is already built.
  Evidence: LosslessCut's Last FFmpeg commands; VSR invokes FFmpeg from `_encode_mixin.py`, `backend/io.py`, and deinterlace, all through `backend/subprocess_policy.py`.
  Touches: `backend/subprocess_policy.py`, `gui/support_controller.py`, `backend/support_bundle.py`.
  Acceptance: the last N FFmpeg/ffprobe invocations are viewable and copyable from Help and included in the support bundle, quoted so each line runs as-is; paths honour the existing redaction policy; the buffer is bounded.
  Confidence: Verified
  Complexity: S

- [ ] P3 -- RM-288: Pin the reproducible-build envelope and state it honestly
  Why: the release publishes checksums and an SBOM but not the two settings that make a rebuild comparable.
  Evidence: PyInstaller honours `SOURCE_DATE_EPOCH` for the PE timestamp; `PYTHONHASHSEED` affects the build run; output is not build-path invariant.
  Touches: `build_exe.bat`, `backend/release_verification.py`, `backend/release_staging.py`, README release section.
  Acceptance: the build sets and records `SOURCE_DATE_EPOCH` and `PYTHONHASHSEED` in the release evidence; the README describes rebuild verification as semantic rather than bit-for-bit.
  Confidence: Verified
  Complexity: S

- [ ] P3 -- RM-289: Raise the PyInstaller floor to 6.22.2 and guard the onedir assumption
  Why: the toolchain floor predates GHSA-9fxf-4qw3-ghmr (patched in 6.22.1; 6.22.2 is current). The advisory does not apply to this onedir/`asInvoker` build, so this is floor hygiene plus a tripwire.
  Evidence: current floor is 6.10.0 in `build_exe.bat`, `backend/release_verification.py`, and `README.md`. `VideoSubtitleRemoverPro.spec` builds onedir.
  Touches: `build_exe.bat`, `backend/release_verification.py`, `README.md`, `tests/test_release_workflow.py`.
  Acceptance: the floor reads 6.22.2 in all three places; verification names both GHSA-9w2p-rh8c-v9g5 and GHSA-9fxf-4qw3-ghmr with onedir/`asInvoker` non-applicability stated; a test fails if the spec ever stops being onedir while the floor sits below 6.22.1.
  Confidence: Verified
  Complexity: S

- [ ] P3 -- RM-293: Bring README Detection and Troubleshooting in line with the engine picker
  Why: the docs still tell users to install PaddleOCR for best accuracy and to activate VLM via env vars.
  Evidence: README pin list omits Surya/VLM; Troubleshooting still leads with PaddleOCR; `backend/detection.py` module doc still says VLM is env-only.
  Touches: `README.md` (byte-precise; mixed CRLF/LF -- do not whole-file rewrite), `backend/detection.py` module docstring. Do not hand-edit the generated CLI table.
  Acceptance: Detection pin list names every engine the picker offers; VLM section says the GUI/`--ocr-engine` path first; Troubleshooting recommends RapidOCR (default) and names PaddleOCR as opt-in; `tests/test_documentation_drift.py` still passes.
  Confidence: Verified
  Complexity: S

- [ ] P3 -- RM-295: Surfaces this audit did not execute
  Why: a pass that never opened these paths should not claim they were reviewed.
  Evidence: no live run of the watch-folder ingest, Whisper/WhisperX helpers, `--nle-input` NLE sidecar, VapourSynth `.vpy` ingest, Docker image build (Linux engine absent on this host), or a real NSIS silent uninstall. Dark theme only; there is no light theme to switch.
  Touches: `backend/watch_folder.py` (if present), Whisper helpers, `backend/nle_sidecar.py`, `backend/vapoursynth_bridge.py`, `Dockerfile`, `installer/vsr.nsi`.
  Acceptance: each named surface has a recorded smoke (command, expected result) or is moved to Roadmap_Blocked with the actual blocker.
  Confidence: Verified (gap)
  Complexity: S
