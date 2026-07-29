# Video Subtitle Remover Pro -- Roadmap

Remaining-work backlog only: every item below is NOT yet implemented.
Completed items are deleted from this file; history lives in CHANGELOG.md and git.

## Research-Driven Additions

### P2

- [ ] P2 -- RM-150 Add a mask-aware temporal regression profile
  Why: per-frame image scores miss camera-motion, background-motion, mask-motion,
    boundary leakage, and flicker failures that dominate video inpainting quality.
  Evidence: MichiganCOG/DEVIL benchmark and CVPR 2022 paper;
    `backend/quality.py`; `tests/test_reference_clips.py`.
  Touches: reference-corpus/quality modules; deterministic synthetic fixtures;
    release verification and tests.
  Acceptance: synthetic fixtures independently vary camera, background, and mask
    motion; lightweight masked-warp/flicker/edge thresholds fail strict release
    verification on seeded temporal regressions; no learned metric download or
    real-media licensing is required (the blocked real-corpus item remains separate).
  Complexity: M

- [ ] P2 -- RM-152 Complete runtime-string extraction and bidirectional layout coverage
  Why: gettext/pseudo-locale infrastructure is complete, but untranslated runtime
    strings and left/right layout assumptions still block trustworthy human catalogs.
  Evidence: `scripts/i18n_catalogs.py coverage` reports qps-Ploc 563/563 as the
    only bundled catalog; runtime strings/layouts in `gui/app.py`,
    `gui/preview_controller.py`, `gui/region_controller.py`, and settings flows;
    Windows globalization/RTL guidance.
  Touches: GUI controllers/layout helpers; `backend/i18n.py`; locale extraction
    and UI scaling probes; i18n tests.
  Acceptance: all user-visible runtime strings in GUI/controller paths pass
    through `tr()`; direction-sensitive packing, anchors, arrows, and ordering
    mirror under RTL; qps-Ploc plus a pseudo-RTL rendered probe passes every major
    workflow; non-pseudo catalogs remain contribution-driven with enforced coverage.
  Complexity: M

- [ ] P2 -- RM-153 Freeze an approved matte as a reusable queue input
  Why: reviewed masks are valuable durable work, but reruns can still repeat
    detection/tracking or depend on disposable cache state.
  Evidence: `backend/matte_interchange.py`; `gui/mask_correction_controller.py`;
    DaVinci Resolve “Render in Place”; Resolve cache-loss/matte-export reports.
  Touches: quality/mask-correction UI; queue item/state schema; matte interchange;
    processor mask-input path; reports/sidecars.
  Acceptance: quality review offers “Freeze approved matte”; the exact
    artifact/manifest hash, source fingerprint, geometry, timing, and range persist
    with the queue item; a matching rerun can explicitly bypass OCR/tracking, while
    any source/hash/timing mismatch fails closed and requests revalidation.
  Complexity: M

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
