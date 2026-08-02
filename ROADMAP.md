# Video Subtitle Remover Pro -- Roadmap

Remaining-work backlog only: every item below is NOT yet implemented.
Completed items are deleted from this file; history lives in CHANGELOG.md and git.

- [ ] P3 -- Mirror the remaining Canvas widgets under RTL
  Why: RM-152's option-funnel mirror excludes Canvas items by design, and
  only ModernToggle hand-mirrors its geometry today. Under an RTL locale
  the slider fill origin, progress bar direction, ModernButton icon/text
  order, the queue item's selected-edge accent stripe, and SegmentedPicker
  segment order all remain LTR, giving mixed direction cues. Needs a
  per-widget geometry pass plus visual verification in an RTL session.
  Where: gui/widgets.py (2026-08-02 audit)

- [ ] P3 -- Make JobSupervisor.run(timeout=...) a wall-clock bound
  Why: the timeout only bounds the post-EOF process wait; the event pump
  blocks on child stdout until EOF, so a child that stops emitting while
  keeping the pipe open outlives any timeout the caller passed. The GUI
  path is covered by the watchdog's cancel escalation, but the API
  contract misleads other callers (tests pass timeouts expecting a bound).
  Where: gui/job_supervisor.py (2026-08-02 audit)
