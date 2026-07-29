# Video Subtitle Remover Pro -- Roadmap

Remaining-work backlog only: every item below is NOT yet implemented.
Completed items are deleted from this file; history lives in CHANGELOG.md and git.

## Research-Driven Additions

### P2

### P3

- [ ] P3 -- RM-155 Supervise each queue job in an isolated child process
  Why: a fatal native OpenCV/ORT/model crash in the single in-process worker can
    terminate the GUI and every queued job despite checkpoint/retry support.
  Evidence: `gui/processing_controller.py:121,340-539`; HandBrake process
    isolation documentation.
  Touches: `gui/processing_controller.py`; CLI/processor job protocol;
    `backend/subprocess_policy.py`; progress/cancel/pause IPC; checkpoints,
    reports, and integration tests.
  Acceptance: each item runs under a versioned local child protocol; progress,
    preview, pause, cancel, and checkpoints still work; a forced access
    violation/abrupt child exit marks only that item failed with retained logs,
    leaves the GUI responsive, and continues the remaining queue.
  Complexity: XL
