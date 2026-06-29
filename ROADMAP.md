# Video Subtitle Remover Pro -- Roadmap

Remaining-work backlog only: every item below is NOT yet implemented.
Completed items are deleted from this file; history lives in CHANGELOG.md and git.



## Research-Driven Additions

### P2 -- Dependency, documentation, and UX hardening

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
