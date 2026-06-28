# Video Subtitle Remover Pro -- Roadmap

Remaining-work backlog only: every item below is NOT yet implemented.
Completed items are deleted from this file; history lives in CHANGELOG.md and git.



## Research-Driven Additions

### P1 -- Trust and release readiness

- [ ] P1 — Detect conflicting OpenCV wheel installs before runtime import drift
  Why: `opencv-python`, `opencv-contrib-python`, and headless variants can shadow each other, but support/release evidence currently reports versions without diagnosing which wheel actually owns `cv2`.
  Evidence: `backend/support_bundle.py`, `backend/dependency_caps.py`, `requirements.txt`, OpenCV/libpng advisory tracking in `backend/security_checks.py`
  Touches: `backend/dependency_caps.py`, `backend/support_bundle.py`, `backend/release_verification.py`, `backend/model_downloads.py`, `tests/test_dependency_caps.py`, `tests/test_support_bundle.py`
  Acceptance: Self-test, support bundle, backend status, and release evidence flag multiple installed OpenCV distributions, show the imported `cv2.__file__`/version owner, and print exact uninstall/install remediation without breaking normal single-wheel installs.
  Complexity: M

- [ ] P1 — Add real edge-case corpus intake and license validator
  Why: The project asks for CC0/public-domain real clips but only ships synthetic fixtures and has no dedicated GitHub intake form or manifest guard for real-source provenance.
  Evidence: `docs/edge_case_corpus.md`, `.github/ISSUE_TEMPLATE/`, `tests/clips/manifest.json`, `tests/test_reference_clips.py`, YaoFANGUK issues #200 and #232
  Touches: `.github/ISSUE_TEMPLATE/edge_case.yml`, `backend/reference_corpus.py`, `tests/test_reference_clips.py`, `tests/clips/manifest.json`, `README.md`
  Acceptance: Contributors can file an edge-case clip issue with URL/license/settings/screenshots; manifest validation rejects missing license/source/fixture metadata for real clips; at least one tiny CC0/public-domain fixture path is documented or wired without inflating default test runtime.
  Complexity: M

- [ ] P1 — Add source-aware output quality and blur-risk preflight
  Why: Competing-tool users report blurred/lower-bitrate outputs, and VSR has quality metrics but no preflight warning when selected codec/quality settings are likely below the source.
  Evidence: YaoFANGUK issue #200, `backend/quality_gate.py`, `backend/batch_report.py`, `backend/encoder.py`, README quality-report docs
  Touches: `backend/encoder.py`, `backend/quality_gate.py`, `backend/batch_report.py`, `gui/app.py`, `backend/cli.py`, tests for bitrate/CRF decision paths
  Acceptance: For each input, CLI/GUI preflight compares source bitrate/resolution/codec against selected output quality and warns or suggests safer codec/quality settings before processing; batch reports persist the recommendation and whether the user overrode it.
  Complexity: M

- [ ] P1 — Expand accessibility announcements and focus traversal coverage
  Why: UIA announcements and focusable custom widgets exist, but custom Canvas controls and major dialogs still need broader semantic/state coverage for screen-reader and keyboard users.
  Evidence: `backend/a11y.py`, `gui/widgets.py`, `gui/app.py`, `tests/test_gui_settings_lock.py`, `tests/test_confirm_dialog.py`
  Touches: `backend/a11y.py`, `gui/widgets.py`, `gui/app.py`, `tests/test_gui_smoke.py`, `tests/test_gui_settings_lock.py`
  Acceptance: Major dialogs, queue rows, toggles, sliders, segmented picker, quality review, and cache import/export expose stable focus order, visible focus, state announcements for important changes, and automated smoke coverage for enabled/disabled/focused states.
  Complexity: M

### P2 -- Dependency, documentation, and UX hardening

- [ ] P2 — Record per-stage timings in batch reports and support bundles
  Why: Per-item elapsed time exists, but users and maintainers cannot tell whether decode, OCR, mask creation, inpaint, encode, mux, or quality analysis caused a slow or failed run.
  Evidence: `backend/batch_report.py`, `backend/processor.py`, `backend/support_bundle.py`, `gui/widgets.py`, YaoFANGUK issues #224 and #222
  Touches: `backend/processor.py`, `backend/batch_report.py`, `backend/support_bundle.py`, `gui/app.py`, `tests/test_hardening.py`, `tests/test_support_bundle.py`
  Acceptance: Batch summary JSON/Markdown and support bundles include per-stage duration totals per item plus run-level slowest-stage summaries; GUI completion/review surfaces show the dominant slow stage without changing successful processing output.
  Complexity: M

- [ ] P2 — Wire gettext extraction through user-facing GUI strings
  Why: `backend/i18n.py` and `locale/vsr.pot` are only scaffolding until main GUI strings are wrapped and extraction is testable.
  Evidence: `backend/i18n.py`, `locale/vsr.pot`, `gui/app.py`, `gui/widgets.py`, `tests/test_hardening.py`
  Touches: `gui/app.py`, `gui/widgets.py`, `backend/i18n.py`, `locale/vsr.pot`, tests for extraction and fallback behavior
  Acceptance: Core visible strings in onboarding, queue, settings, dialogs, status, errors, and About/backend status use `_()` or a documented alias; extraction refresh updates `locale/vsr.pot`; a test catalog proves at least one non-English string renders while missing keys fall back to source text.
  Complexity: L

- [ ] P2 — Add cooperative pause/resume checkpoints for long videos
  Why: VSR can skip completed files, but competitor issue streams show users expect long-running removals to pause and resume without restarting the current video.
  Evidence: YaoFANGUK issues #222 and #224, `README.md` crash-resume note, `backend/processor.py`, `gui/config.py` queue state
  Touches: `backend/processor.py`, `backend/cli.py`, `gui/app.py`, `gui/config.py`, `backend/batch_report.py`, checkpoint tests
  Acceptance: GUI and CLI can pause a running job at safe frame/batch boundaries, persist enough checkpoint state to resume the current item, report paused status in queue/batch summaries, and handle stale/incompatible checkpoints by falling back to normal processing with a clear warning.
  Complexity: XL

- [ ] P2 — Extract GUI and processor orchestration controllers
  Why: `gui/app.py` and the long media state machine remain the highest-risk files for future polish and reliability work.
  Evidence: `docs/architecture.md`, `gui/app.py`, `backend/processor.py`, recent commits adding many UI/backend surfaces
  Touches: `gui/app.py`, new focused `gui/*` controller modules, `backend/processor.py`, backend pipeline helper modules, GUI/backend smoke tests
  Acceptance: Queue processing, preview/quality review, support/cache dialogs, and processor stage orchestration move into focused modules with no behavior change; existing public imports and tests still pass; architecture docs name the new boundaries.
  Complexity: L

### P3 -- Research bench

- [ ] P3 — Add a local container/isolated install smoke path
  Why: Direct competitors ship Docker or isolated install options, and user reports in adjacent projects show GPU/package setup remains a recurring failure point even when the main Windows launcher is preferred.
  Evidence: VideOCR Docker distribution, YaoFANGUK package/GPU install issues #218/#221/#226/#228, current `setup.py` and README install flow
  Touches: `Dockerfile` or local container recipe, `.dockerignore`, `README.md`, setup smoke command, release docs
  Acceptance: A documented local-only CPU container or isolated install recipe launches `python -m backend.processor --self-test` and a tiny CLI smoke without GitHub Actions, cloud upload, or replacing the Windows launcher as the primary path.
  Complexity: M

### P1 -- Additional release hardening

- [ ] P1 - Add frozen-build multiprocessing guards and runtime-hook evidence
  Why: PyInstaller documents recursive spawn loops when frozen apps or dependencies use multiprocessing without early `freeze_support()`, and this repo's entry point/build path does not currently install that guard before heavy imports.
  Evidence: `VideoSubtitleRemover.py`, `build_exe.bat`, `VideoSubtitleRemoverPro.spec`, PyInstaller multiprocessing guidance
  Touches: `VideoSubtitleRemover.py`, `build_exe.bat`, `VideoSubtitleRemoverPro.spec` or its replacement, new runtime hook asset, `backend/release_verification.py`, `tests/test_release_workflow.py`
  Acceptance: `multiprocessing.freeze_support()` runs before GUI/OpenCV/ML imports in frozen launches; PyInstaller builds include a runtime hook that also calls it; release evidence records the hook; frozen smoke proves one GUI process and no recursive child storm.
  Complexity: M

- [ ] P1 - Preload ONNX Runtime CUDA DLLs before CUDA provider sessions
  Why: ONNX Runtime documents CUDA/cuDNN DLL preload support, while VSR only reports whether `preload_dlls` exists and then creates CUDA sessions without using it.
  Evidence: ONNX Runtime CUDA Execution Provider docs, `backend/dependency_caps.py`, `backend/inpainters_onnx.py`, `backend/inpainters/lama.py`
  Touches: `backend/inpainters_onnx.py`, `backend/inpainters/lama.py`, `backend/dependency_caps.py`, `backend/support_bundle.py`, `backend/release_verification.py`, `tests/test_dependency_caps.py`, `tests/test_hardening.py`
  Acceptance: Before the first CUDA `InferenceSession`, VSR calls `onnxruntime.preload_dlls()` when available, records success/failure in backend status/support/release evidence, and falls back cleanly to current provider order when unavailable.
  Complexity: M

### P2 -- Setup repair

- [ ] P2 - Make setup and launcher repair non-interactive
  Why: `setup.py` prompts on an existing `venv`, but launcher-driven first-run and repair flows should not block on stdin or leave users manually deleting broken environments.
  Evidence: `setup.py`, `Run_VSR_Pro.bat`, `Run_VSR_Pro.ps1`, SysAdminDoc issue #3, YaoFANGUK install issues #228 and #231
  Touches: `setup.py`, `Run_VSR_Pro.bat`, `Run_VSR_Pro.ps1`, `Run_VSR_Pro_Debug.bat`, `tests/test_setup_bootstrap.py`, README troubleshooting
  Acceptance: Launchers detect missing/broken venv state and run setup in unattended repair mode; `setup.py --repair` safely recreates only the repo-local venv after boundary checks; interactive prompts are never used by launcher paths; tests cover keep, repair, unsafe-path refusal, and timeout messaging.
  Complexity: M
