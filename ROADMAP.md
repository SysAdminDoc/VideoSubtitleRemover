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

