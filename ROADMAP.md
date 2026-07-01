# Video Subtitle Remover Pro -- Roadmap

Remaining-work backlog only: every item below is NOT yet implemented.
Completed items are deleted from this file; history lives in CHANGELOG.md and git.

## Research-Driven Additions

- [ ] P3 — Add frame-level manual mask correction for hard clips
  Why: Rectangular/timed regions and OCR masks miss curved, animated, or partial overlays; commercial tools and IOPaint show that brush-level correction is table-stakes for stubborn cases.
  Evidence: `gui/preview_controller.py`, `gui/app.py`, `backend/processor.py`, IOPaint manual inpainting UX, Media.io/HitPaw remover workflows.
  Touches: `gui/preview_controller.py`, `gui/app.py`, `gui/config.py`, `backend/config.py`, `backend/processor.py`, queue-state/preset serialization, tests.
  Acceptance: users can draw, erase, save, and re-open freehand or polygon mask corrections tied to a time range; processing merges corrections with automatic masks; queue persistence, preset export/import, and CLI-compatible config tests pass.
  Complexity: L
