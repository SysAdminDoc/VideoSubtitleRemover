# Video Subtitle Remover Pro -- Roadmap

Remaining-work backlog only: every item below is NOT yet implemented.
Completed items are deleted from this file; history lives in CHANGELOG.md and git.

## Research-Driven Additions

### P0 — Now

### P1 — Next

### P2 — Later

### P3 — Under Consideration

## Audit Findings (2026-07-14 deep audit)

- [ ] P3 — Stabilize GUI tests under a full-suite run
  Why: A couple of Tk tests pass in isolation but fail with "Tcl wasn't installed properly" when the whole suite creates many Tk roots in one process; this is test-harness resource exhaustion, not a product bug, but it makes the suite flaky.
  Where: `tests/test_gui_smoke.py`, `tests/test_hardening.py` (GUI test roots); needs per-test Tk teardown or subprocess isolation.

- [ ] P3 — First-run-friendly copy for advanced-setting tooltips
  Why: Several advanced toggles surface backend jargon without explaining the tradeoff (e.g. "Kalman box tracking", "Flow-warped temporal exposure (motion-heavy)", "remuxing"); a copy pass would help first-time users choose settings with confidence.
  Where: `gui/app.py` advanced-settings tooltips; `gui/processing_controller.py` status strings.

## Audit Findings (2026-07-17 deep audit)

## Audit Findings (2026-07-17 deep audit, second pass)

