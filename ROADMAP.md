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
