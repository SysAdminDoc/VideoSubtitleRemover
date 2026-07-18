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

## Audit Findings (2026-07-17 deep audit)

## Audit Findings (2026-07-17 deep audit, second pass)

