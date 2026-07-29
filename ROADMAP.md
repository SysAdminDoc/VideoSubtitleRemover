# Video Subtitle Remover Pro -- Roadmap

Remaining-work backlog only: every item below is NOT yet implemented.
Completed items are deleted from this file; history lives in CHANGELOG.md and git.

## Research-Driven Additions

Added 2026-07-21 (deep line-level audit of ~25 previously-unaudited backend
modules). All three items are fixable and unit-testable headlessly (no GPU
required to implement or verify), and sit on opt-in / edge paths.

### P2

- [ ] P2 -- RM-136 MatAnyone alpha read must tolerate empty (subtitle-gap) frames
  Why: `_read_alpha_video` discards the ENTIRE MatAnyone alpha result if any
    single frame is fully transparent, which is exactly what a subtitle-gap
    frame produces -- so the opt-in refinement silently no-ops on nearly every
    real clip.
  Evidence: `backend/segmentation.py:371-388` returns None when any
    `_normalize_alpha_matte` (line 318 returns None for an all-zero frame) is
    None; the sibling `_normalize_alpha_sequence` (line 338-343) already handles
    this by falling back to the per-frame hint mask.
  Touches: `backend/segmentation.py` (`_read_alpha_video`); `tests/test_segmentation.py`.
  Acceptance: an alpha video containing one all-zero frame yields a full-length
    result where the empty frame becomes an all-zero matte (or the hint) instead
    of returning None; a unit test with synthetic frames covers the gap case.
  Complexity: S

### P3

- [ ] P3 -- RM-137 Scale the Whisper audio-extraction ffmpeg timeout by source duration
  Why: `extract_audio_to_temp` hardcodes a 600 s ffmpeg timeout while the sibling
    `run_ffmpeg_whisper_segments` scales it by probed duration; a long source on
    a slow disk/CPU can exceed 10 min to demux+resample and get the Whisper
    fallback silently killed.
  Evidence: `backend/whisper_fallback.py:302` (`timeout=600`) vs lines 203-207
    which use `_ffmpeg_subprocess_timeout(duration, base=600.0, factor=12.0)`
    (`backend/io.py:832`, `_probe_duration_seconds` at `io.py:525`).
  Touches: `backend/whisper_fallback.py`; `tests/test_whisper_fallback.py`.
  Acceptance: `extract_audio_to_temp` probes duration and passes a duration-scaled
    timeout (falling back to a safe default when probing fails); a test asserts a
    long-duration input yields a timeout above the 600 s floor.
  Complexity: S

- [ ] P3 -- RM-138 decode_accel legacy seek must fail loudly when it cannot reposition
  Why: legacy-mode `set(CAP_PROP_POS_FRAMES)` advances `_pos` and returns True
    even when the decoder has no `SeekFrame`, so a seek-then-read silently
    returns the wrong frame while advertising success on the cv2.VideoCapture
    drop-in contract (resume / ROI re-scan misalignment).
  Evidence: `backend/decode_accel.py:209-235` -- `SeekFrame` is called only
    `if hasattr(self._decoder, "SeekFrame")`, but `_pos = idx; return True` runs
    unconditionally, and `read()` uses sequential `GetNextFrame()` in legacy mode.
  Touches: `backend/decode_accel.py` (`_PyNvVideoCapture.set`); a new/updated test
    with a fake legacy decoder lacking `SeekFrame`.
  Acceptance: `set(CAP_PROP_POS_FRAMES)` returns False (leaving `_pos` unchanged)
    when a legacy decoder cannot seek; a unit test with a fake decoder verifies it.
  Complexity: S

### P0

### P1

- [ ] P1 -- RM-143 Reject unknown GUI, settings, and preset inpaint modes
  Why: an invalid value currently becomes STTN while import/apply can report
    success, so user intent and the effective algorithm diverge silently.
  Evidence: `gui/config.py:304-320`; SysAdminDoc/VideoSubtitleRemover#7; the
    backend registry at 2026-07-29 fails on unavailable selected models but this
    GUI edge remains.
  Touches: `gui/config.py`; `gui/settings_controller.py`; preset/settings import
    paths; config and feedback tests.
  Acceptance: unknown values such as `banana` are rejected/quarantined with an
    actionable notice and leave the prior setting unchanged; valid backend-only
    modes receive their existing explicit compatibility notice; no invalid value
    can produce a successful preset/import result.
  Complexity: S

- [ ] P1 -- RM-144 Make user-state persistence observable and downgrade-safe
  Why: settings, queue, and preset write failures are swallowed, and opening then
    closing a future settings schema can erase fields the current version ignores.
  Evidence: `gui/config.py:861-870,921-938,944-1031,1207-1221`; Subtitle Edit's
    concurrent autosave guard (PR #12976).
  Touches: `gui/config.py`; app shutdown/startup and preset callers; persistence
    result types; GUI/config tests.
  Acceptance: every save returns or raises a typed outcome surfaced in the
    activity UI with retry guidance; corrupt-backup success is reported only
    after the copy succeeds; a future schema opens in explicit read-only
    compatibility mode and the original file is not overwritten unless the user
    deliberately exports a current-format copy.
  Complexity: M

- [ ] P1 -- RM-145 Make opt-in crash-report privacy fail closed
  Why: a scrubber exception returns the original event, and path scrubbing can
    retain the processed filename despite the documented privacy contract.
  Evidence: `backend/crash_reporter.py:1-26,43-52,110-126`.
  Touches: `backend/crash_reporter.py`; crash-reporter privacy tests.
  Acceptance: telemetry is built from an allowlisted minimal event or returns
    `None` on any scrub failure; nested Windows/POSIX/UNC/file-URL paths,
    basenames, locals, breadcrumbs, OCR text, and request data cannot leave the
    process in tests.
  Complexity: S

- [ ] P1 -- RM-146 Make cache and matte replacements transactionally recoverable
  Why: a cache move failure can strand the current target in `.vsrbak`, while
    matte export can delete/promote one half of the artifact/manifest pair.
  Evidence: `backend/cache_inventory.py:499-503,626-658`;
    `backend/matte_interchange.py:172-220`.
  Touches: `backend/cache_inventory.py`; `backend/matte_interchange.py`;
    shared atomic-replacement helper if introduced; cache/matte tests.
  Acceptance: model-cache manifest bytes are capped before reading; a journal or
    equivalent records rollback before any original moves; injected failure at
    every replace/write step leaves either the complete old set or complete new
    hash-matching set, with deterministic startup cleanup/recovery.
  Complexity: M

- [ ] P1 -- RM-147 Persist requested and effective execution provenance
  Why: CUDA/DirectML requests can execute RapidOCR or LaMa on CPU/cv2 while UI,
    smoke, and reports expose only ambiguous engine/backend labels.
  Evidence: `backend/detection.py:168-193,258-293`;
    `backend/support_bundle.py:619-699`; Topaz processor preferences; HitPaw
    hardware/output troubleshooting; RapidOCR's CPU recommendation.
  Touches: detector/inpainter result contracts; queue item schema; batch reports;
    output sidecars; support bundle/inference smoke; queue and Help UI.
  Acceptance: every job records requested/effective device, OCR engine +
    provider, inpaint mode + backend, fallback reason, and observed throughput;
    a CUDA request that runs RapidOCR CPU and LaMa cv2 is visibly labeled as such
    in the queue, JSON report, sidecar, and support bundle.
  Complexity: L

- [ ] P1 -- RM-148 Reflow and scroll all major dialogs at 125-200% text scale
  Why: fixed non-resizable onboarding and editor windows can obscure actions at
    high text scale even though the main workbench is responsive.
  Evidence: `gui/onboarding.py:30-34`; `gui/region_controller.py:177-182`;
    `gui/mask_correction_controller.py:142-145`; 2026-07-29 probes measured
    1106x969 onboarding and 1192x1144 region editor at 200%.
  Touches: those three dialog modules; shared dialog/work-area helper;
    `tools/ui_scaling_probe.py`; scaling/accessibility tests.
  Acceptance: at 980x720 and 2752x1152 work areas, 100/125/150/175/200% scale,
    default/high-contrast themes, and qps-Ploc/RTL, every control is reachable by
    keyboard and scrolling, focus is never obscured, and no dialog exceeds the
    work area without an internal scroll path.
  Complexity: M

### P2

- [ ] P2 -- RM-149 Contract-test selectable OpenCV 5 DNN inference engines
  Why: PP-OCRv6 and LaMa now depend on OpenCV 5 DNN, whose new/classic/default
    engines have different coverage and fallback behavior not proven by current tests.
  Evidence: `backend/opencv_ocr.py`; `backend/inpainters/lama.py`; OpenCV 5
    migration/status documentation.
  Touches: OpenCV OCR/LaMa adapters; inference smoke; release verification;
    OpenCV and release tests.
  Acceptance: release evidence runs one real bundled PP-OCRv6 inference under
    each applicable documented engine selection, conditionally does the same for
    an advertised local LaMa model, records the actual engine, and fails on load,
    shape, or materially divergent-output regressions without assuming ORT-linked DNN.
  Complexity: M

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

- [ ] P2 -- RM-151 Publish accurate architecture, release, and accessibility support docs
  Why: current contributor/release claims contradict live module ownership,
    unsigned distribution, changelog structure, and the known UIA limitation.
  Evidence: `docs/architecture.md:8,20,35,117,144`; `README.md:85-104`;
    duplicate `CHANGELOG.md` Unreleased sections; Adobe After Effects ACR;
    UI Automation work remains in `Roadmap_Blocked.md`.
  Touches: `README.md`; `docs/architecture.md`; `CLAUDE.md`; `CHANGELOG.md`;
    documentation drift tests/checks.
  Acceptance: module ownership and sizes match live code; one Unreleased section
    remains; unsigned installer/Winget wording is consistent; a support matrix
    distinguishes tested keyboard, high-contrast, scaling, and pseudo-locale
    behavior from the explicitly unsupported custom-control screen-reader/UIA surface.
  Complexity: S

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
