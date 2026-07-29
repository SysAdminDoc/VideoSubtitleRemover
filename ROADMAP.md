# Video Subtitle Remover Pro -- Roadmap

Remaining-work backlog only: every item below is NOT yet implemented.
Completed items are deleted from this file; history lives in CHANGELOG.md and git.

## Research-Driven Additions

### P2

- [ ] P2 -- RM-154 Add loss-aware WebVTT translation and re-embed interchange
  Why: WebVTT cue IDs, settings, regions, vertical text, ruby, voice/language
    spans, STYLE, and NOTE cannot be safely flattened through the SRT-only model.
  Evidence: `backend/subtitle_translation.py`; `backend/container_payload.py`;
    W3C WebVTT and Matroska subtitle mappings.
  Touches: subtitle parser/model/serializer; translation provider protocol; CLI
    and GUI file filters; container mapping; fixtures/tests.
  Acceptance: `.vtt` input/output preserves timing, cue IDs/settings, and supported
    inline/block structure while translating only visible cue text; unsupported
    regions/styles produce an explicit loss report; vertical, ruby, positioned,
    NOTE, and STYLE fixtures round-trip; TTML/IMSC remains out of scope.
  Complexity: M

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
