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
