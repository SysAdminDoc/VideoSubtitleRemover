# Video Subtitle Remover Pro -- Roadmap

Remaining-work backlog only: every item below is NOT yet implemented.
Completed items are deleted from this file; history lives in CHANGELOG.md and git.

## Research-Driven Additions

- [ ] P2 — Add a local upstream dependency and security drift report
  Why: PaddleOCR, RapidOCR, ONNX Runtime, OpenCV, PyInstaller, and related packages move quickly, while this repo intentionally avoids Dependabot/GitHub Actions.
  Evidence: `requirements.txt`, `backend/dependency_caps.py`, `backend/release_verification.py`, PaddleOCR/RapidOCR/ONNX Runtime releases, OpenCV libpng CVE issue.
  Touches: `backend/dependency_caps.py`, `backend/security_checks.py`, `backend/release_verification.py`, `tests/test_hardening.py`, README usage notes.
  Acceptance: a local command reports installed, pinned/minimum, latest-known, blocked exceptions, and security-advisory status for core and optional stacks without auto-updating dependencies; release verification embeds the report.
  Complexity: M

- [ ] P2 — Add pseudo-locale and RTL rendered smoke tests
  Why: Gettext and RTL scaffolds exist, but no rendered smoke proves translated or expanded strings fit the main GUI.
  Evidence: `backend/i18n.py`, `locale/vsr.pot`, `gui/config.py`, `gui/app.py`, `tests/test_hardening.py`, Subtitle Edit localization precedent.
  Touches: `backend/i18n.py`, `locale/vsr.pot`, `gui/app.py`, `tests/test_gui_smoke.py`, `tests/test_hardening.py`.
  Acceptance: tests create or load a temporary pseudo-locale catalog with expanded strings, start the GUI with RTL layout enabled, verify translated sentinel labels, and assert major buttons/status chips are visible and not clipped.
  Complexity: M

- [ ] P2 — Enrich NLE sidecars for multi-segment editorial handoff
  Why: Current EDL/FCPXML exports are intentionally one-event stubs; editor handoff improves if timed regions, source timecode, audio state, dimensions, and multiple cleaned ranges survive.
  Evidence: `backend/nle_sidecar.py`, `backend/processor.py`, `tests/test_nle_sidecar.py`, Subtitle Edit format breadth.
  Touches: `backend/nle_sidecar.py`, `backend/processor.py`, `backend/cli.py`, `gui/config.py`, `gui/widgets.py`, `tests/test_nle_sidecar.py`.
  Acceptance: EDL and FCPXML exports represent every processed time span, preserve source dimensions/timecode where available, include cleaned/source clip metadata, parse back through existing import helpers, and pass round-trip tests.
  Complexity: M

- [ ] P2 — Add optional adapter conformance dry-run matrix
  Why: Model adapters are intentionally gated, but trust evidence is spread across manifests, support bundles, and individual tests instead of one operator-readable matrix.
  Evidence: `backend/adapter_manifest.py`, `backend/remote_model_policy.py`, `backend/release_verification.py`, `tests/test_vace_adapter.py`, `tests/test_videopainter_adapter.py`, `tests/test_void_adapter.py`, SEDiT/CLEAR research churn.
  Touches: `backend/adapter_manifest.py`, `backend/inpainter_registry.py`, `backend/model_downloads.py`, `backend/release_verification.py`, `backend/support_bundle.py`, tests.
  Acceptance: a local dry-run command lists every production and benchmark adapter with env vars, license/source, expected weight paths, hash policy, import-before-trust status, and availability result without loading untrusted model code; support bundles and release evidence include the matrix.
  Complexity: M

- [ ] P3 — Add frame-level manual mask correction for hard clips
  Why: Rectangular/timed regions and OCR masks miss curved, animated, or partial overlays; commercial tools and IOPaint show that brush-level correction is table-stakes for stubborn cases.
  Evidence: `gui/preview_controller.py`, `gui/app.py`, `backend/processor.py`, IOPaint manual inpainting UX, Media.io/HitPaw remover workflows.
  Touches: `gui/preview_controller.py`, `gui/app.py`, `gui/config.py`, `backend/config.py`, `backend/processor.py`, queue-state/preset serialization, tests.
  Acceptance: users can draw, erase, save, and re-open freehand or polygon mask corrections tied to a time range; processing merges corrections with automatic masks; queue persistence, preset export/import, and CLI-compatible config tests pass.
  Complexity: L
