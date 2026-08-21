# Video Subtitle Remover Pro -- Roadmap

Actionable work only. Historical and completed roadmap material is archived in CHANGELOG.md; blocked work is kept in Roadmap_Blocked.md.

## Remaining work

Source research: RESEARCH.md, dated 2026-08-21. P0 safety work is ordered first.

Rejected on purpose (do not re-file): production ROSE/EraserDiT/VOID adapters, Qt rewrite,
REST/Gradio, Mac/ROCm, re-adding GitHub Actions, Sigstore/attestations for SmartScreen,
winget/Store submission. Reasons in RESEARCH.md Rejected Ideas.

CLEAR now has released code and weights. Keep it research-only until redistribution
rights, consumer GPU cost, large-font quality, and color drift pass written gates.

Implementer index -- drain in this order. Blocked work is in Roadmap_Blocked.md.

## Research-Driven Additions

- [ ] P1 — RM-299 Restore GUI workflow and accessibility release proof
  - Why: Active tests prove GUI imports but not the queue, manual-region, Test
    Cleanup, scaling, high-contrast, RTL, or collapsed-control workflows. The
    relevant interaction tests are archived and ignored. One local UIA inspection
    found Advanced controls discoverable while collapsed and needs live validation.
  - Evidence: tests/test_gui_import_smoke.py, tests/archive/conftest.py,
    tests/archive/test_gui_smoke.py, tools/ui_scaling_probe.py,
    gui/settings_controller.py:99-117, backend/a11y.py,
    https://learn.microsoft.com/en-us/windows/apps/design/accessibility/accessibility-testing,
    and
    https://learn.microsoft.com/en-us/accessibility-tools-docs/items/wpf/control_iscontrolelement.
  - Touches: selected tests under tests/archive/, new active GUI tests,
    gui/settings_controller.py, backend/a11y.py, tools/ui_scaling_probe.py,
    and build_exe.bat.
  - Acceptance: Promote a minimal stable interaction suite into active collection;
    prove queue selection, region propagation, Test Cleanup dispatch, cancellation,
    and clean shutdown; prove collapsed Advanced descendants are absent from the
    control view and tab order; run 100 and 200 percent scaling, high-contrast, and
    RTL probes in the local release build; record packaged-build Narrator and NVDA
    results as live evidence; keep every Tk call on the UI thread.
  - Complexity: Medium. Regression risk is medium because native accessibility
    trees and display scaling require packaged Windows validation.

- [ ] P1 — RM-300 Make Test Cleanup temporally representative
  - Why: Test Cleanup always reads frame zero and runs one frame, even when the user
    is viewing another time or has selected a temporal mode. The low-resolution
    planning proxy exists but is dormant.
  - Evidence: gui/preview_controller.py:704-740,
    backend/proxy_workflow.py:45-92, and README.md:55.
  - Touches: gui/preview_controller.py, backend/proxy_workflow.py,
    preview state in gui/, and active GUI or preview tests.
  - Acceptance: Use the selected preview timestamp for single-frame modes; use a
    scene-bounded before/current/after window for temporal modes; display the tested
    timestamp, frame range, and proxy resolution; call the existing cached proxy for
    planning when it reduces latency; never use proxy pixels for final output;
    cancel stale preview work safely; prove a nonzero selected timestamp and a scene
    cut with deterministic fixtures.
  - Complexity: Medium. Regression risk is medium. Depends on RM-299 for stable GUI
    interaction coverage.

- [ ] P1 — RM-301 Preserve OCR polygon geometry through mask creation
  - Why: OCR engines can return quadrilaterals, but VSR converts them to
    axis-aligned rectangles. Rotated text then removes more valid content and can
    miss the intended stroke shape.
  - Evidence: backend/ocr_vlm.py:376-405, backend/detection.py:654, and
    https://github.com/D-Ogi/WatermarkRemover-AI/issues/38.
  - Touches: backend/detection.py, OCR adapter result types, tracking and track-plan
    serialization, backend/segmentation.py, mask rendering, preview overlays, and
    geometry tests.
  - Acceptance: Carry a normalized polygon beside the backward-compatible bounding
    box; preserve polygon vertices through tracking, scaling, clipping, scene cuts,
    serialization, and reload; rasterize polygon masks with configured expansion in
    the polygon's local geometry; show editable polygon overlays; prove a 45-degree
    text fixture removes the target without the old rectangle's excess area; keep
    existing manual rectangles and old plans valid.
  - Complexity: Medium to high. Regression risk is medium because geometry crosses
    detection, tracking, storage, and rendering.

- [ ] P1 — RM-302 Preserve exact rational VFR timing
  - Why: Current probes and manifests store decimal timestamp seconds. Long or
    irregular timelines can accumulate conversion error and lose the source clock
    needed for exact checkpoints, mattes, subtitles, NLE exports, and validation.
  - Evidence: backend/io.py:52-90, backend/processor.py:2779-2810,
    https://ffmpeg.org/ffprobe.html,
    https://ffmpeg.org/ffmpeg.html, and
    https://www.rfc-editor.org/rfc/rfc9559.html.
  - Touches: backend/io.py, timing data classes, backend/processor.py,
    checkpoints, matte and track plans, subtitle and NLE exports, sidecars, and VFR
    tests.
  - Acceptance: Store integer best-effort timestamps and durations with rational
    time-base numerator and denominator; convert to seconds only for display or a
    format that requires it; preserve exact ticks through resume and plan reload;
    log every missing, repeated, non-monotonic, or repaired timestamp; prove a long
    1001-based timeline stays within half a source frame and keeps audio alignment;
    validate repeated PTS, missing PTS, edit lists, and muxer rounding.
  - Complexity: High. Regression risk is high because timing is shared across
    decoding, processing, exports, and final validation.

- [ ] P1 — RM-303 Preserve linear-light high-bit HDR repairs
  - Why: HDR input reaches the processor as uint16, but the repair surface is divided
    to uint8 and expanded again. Pixels inside the repaired mask are therefore
    limited to 256 code values and are blended in transfer-encoded space.
  - Evidence: backend/processor.py:1520-1565,
    https://www.itu.int/rec/R-REC-BT.2100-3-202502-I/en,
    https://www.itu.int/pub/R-REP-BT.2446-1-2021, and
    https://helpx.adobe.com/ca/after-effects/desktop/remove-objects-from-your-videos/content-aware-fill.html.
  - Touches: HDR decode and merge helpers in backend/processor.py, color metadata
    probing, ROI buffers, finishing, quality checks, release fixtures, and HDR tests.
  - Acceptance: Keep the original high-bit source surface; generate a separate
    tone-mapped 8-bit proxy only for OCR and models that require it; decode tagged PQ
    or HLG repair regions to bounded linear float, composite there, and reapply the
    transfer function; fail closed or require an explicit override for missing or
    conflicting transfer tags; keep outside-mask high-bit pixels exact; prove
    synthetic PQ and HLG ramps retain more than 256 repaired code levels without
    banding regressions; preserve HDR metadata and bound memory to active regions.
  - Complexity: High. Regression risk is high. Requires trustworthy color tags and
    dedicated HDR fixtures.

- [ ] P1 — RM-304 Add mask-local temporal and color-drift quality gates
  - Why: The current temporal score averages raw adjacent-frame ROI SSIM, which
    confuses valid motion with flicker and can hide localized defects. The pipeline
    also lacks an outside-mask color-drift guard.
  - Evidence: backend/quality.py:252-268, backend/_quality_mixin.py,
    https://arxiv.org/abs/2605.14534,
    https://github.com/silent-commit/CLEAR/issues/5, and
    https://docs.telestream.dev/docs/qualify-user-guide.
  - Touches: backend/quality.py, backend/_quality_mixin.py,
    backend/quality_gate.py, quality reports, reference corpus fixtures, and
    quality tests.
  - Acceptance: Add a scene-aware, mask-local, motion-compensated temporal score;
    exclude cuts and report the worst frame pair with timestamp and overlay; add an
    outside-mask CIELAB color-drift measure for SDR and an appropriate linear-light
    measure for tagged HDR; calibrate thresholds on static, moving, flicker,
    occlusion, and global-cast fixtures; prove intentional camera motion does not
    fail by itself; warn or gate without automatically recoloring valid footage.
  - Complexity: Medium. Regression risk is medium because thresholds need a licensed
    calibration corpus. Benefits from RM-302 and RM-303.

- [ ] P1 — RM-305 Inventory both FFmpeg runtimes
  - Why: Release evidence validates the external FFmpeg executable, but untrusted
    media also enters through OpenCV's embedded FFmpeg libraries. The Docker image
    installs an unversioned distribution package from an unpinned base.
  - Evidence: backend/io.py:1477, backend/security_checks.py:262-335,
    backend/release_verification.py:1388-1430, Dockerfile:1-14,
    https://github.com/opencv/opencv/releases/tag/5.0.0, and
    https://ffmpeg.org/download.html.
  - Touches: backend/security_checks.py,
    backend/release_verification.py, release evidence schema, Dockerfile, and
    security or release tests.
  - Acceptance: Parse and record OpenCV's avcodec, avformat, avutil, and wheel
    provenance beside the external FFmpeg version and build configuration; maintain
    a cited version or provenance rule before blocking a release; fail closed when a
    known affected embedded build is identified; pin the container base by digest;
    install or verify a reviewed FFmpeg version in the image; prove both runtime
    records appear in strict release evidence. Do not label the current embedded
    build vulnerable without an advisory mapping.
  - Complexity: Medium. Regression risk is low to medium. The hard part is mapping
    library ABI versions to a defensible upstream provenance.

- [ ] P2 — RM-306 Publish the requested beginner workflow
  - Why: Discussion #8 asks for a beginner video covering import, translation,
    inpainting, and output. It is the only unresolved concrete request in this
    repository's public tracker.
  - Evidence:
    https://github.com/SysAdminDoc/VideoSubtitleRemover/discussions/8#discussioncomment-17848058.
  - Touches: hosted tutorial media, a versioned transcript and screenshot walkthrough
    under docs/, README.md, and discussion #8.
  - Acceptance: Record the packaged current release from a fresh install; show file
    import, region review, Test Cleanup, model availability, optional translation,
    processing, quality review, and output location; explain local processing and
    optional network paths; provide accurate captions, timestamps, transcript,
    keyboard-free mouse steps, and a tested sample; state the exact app version and
    recording date; link the stable video and text fallback from README and reply to
    discussion #8.
  - Complexity: Low to medium. Regression risk is low. Record after RM-299 and RM-300
    stabilize the demonstrated workflow.
