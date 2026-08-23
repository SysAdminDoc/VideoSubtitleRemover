# Video Subtitle Remover Pro Roadmap

Actionable work only. Historical and completed roadmap material is archived in CHANGELOG.md; blocked work is kept in Roadmap_Blocked.md.

## Remaining work

Source research: RESEARCH.md, dated 2026-08-23. P0 safety work is ordered first.

Rejected on purpose (do not re-file): production ROSE/EraserDiT/VOID adapters, Qt rewrite,
REST/Gradio, Mac/ROCm, re-adding GitHub Actions, Sigstore/attestations for SmartScreen,
winget/Store submission. Reasons in RESEARCH.md Rejected Ideas.

CLEAR now has released code and weights. Keep it research-only until redistribution
rights, consumer GPU cost, large-font quality, and color drift pass written gates.

Implementer index. Drain in this order. Blocked work is in Roadmap_Blocked.md.

## Research-Driven Additions

### P0

- [ ] P0 | RM-307: Enforce truthful requested-stage execution
  Why: A named OCR, inpainting, segmentation, tracking, or restoration choice can fail and silently run a different implementation or return unchanged pixels while provenance still names the requested stage.
  Evidence: `backend/inpainters_diffusion.py:126-165`, `backend/detection.py:411-440`, `backend/detection.py:503-652`, `backend/processor.py:1376-1389`, `backend/processor.py:1501-1531`, `backend/post_restore.py`, `tests/test_hardening_inpaint.py:298-370`, and issue #7.
  Touches: `backend/device_provider.py`, `backend/inpainter_registry.py`, `backend/inpainters_diffusion.py`, `backend/detection.py`, `backend/processor.py`, `backend/post_restore.py`, `backend/batch_report.py`, failure-injection tests, architecture docs.
  Acceptance: Every explicit stage request either executes that implementation and reports its actual provider or stops with a classified error and recovery hint. Only Auto may cross implementation boundaries, its full fallback chain is recorded, and injected load and runtime failures cannot produce false stage provenance.
  Complexity: L

- [ ] P0 | RM-308: Make Manual region a real cross-engine mode
  Why: The command-bar choice currently resets without opening or applying a region, and settings clear fixed-region mode for LaMa, ProPainter, and Auto despite backend support for an engine-independent fixed mask.
  Evidence: `gui/layout_build.py:175`, `gui/app.py:1381-1386`, `gui/settings_controller.py:603-608`, `backend/processor.py:2117-2146`, and https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/7.
  Touches: `gui/layout_build.py`, `gui/app.py`, `gui/settings_controller.py`, `gui/config.py`, `gui/region_controller.py`, `backend/processor.py`, GUI workflow and processor tests.
  Acceptance: Selecting Manual region opens the isolated region editor when no region exists, preserves and displays an existing region when one does, bypasses automatic detection in manual-only mode, and applies the same fixed mask across Auto, STTN, LaMa, and ProPainter. Clear-region feedback and private-desktop regression tests pass without touching the active display.
  Complexity: M

### P1

- [ ] P1 | RM-309: Restore authoritative FFmpeg 9 frame durations
  Why: The current ffprobe request asks for legacy packet-duration fields that FFmpeg 9 does not return for frames, which loses final-frame and irregular-duration evidence.
  Evidence: `backend/io.py:953`, a local FFmpeg 9.0.1 probe on 2026-08-23, https://ffmpeg.org/doxygen/trunk/ffprobe_8c_source.html, and https://ffmpeg.org/ffprobe.html.
  Touches: `backend/io.py`, timing fixtures, parser tests, FFmpeg integration tests, release runtime evidence.
  Acceptance: The probe requests and parses `duration` and `duration_time`, retains `pkt_duration` fields only as legacy fallbacks, and preserves exact integer timestamps. A real FFmpeg 9.0.1 fixture with an irregular final frame round-trips its authoritative duration and passes post-mux clock validation.
  Complexity: S

- [ ] P1 | RM-310: Build SRT cues from tracked OCR consensus
  Why: SRT extraction ignores the supplied boxes, reruns OCR across the full frame, and exact-string merging fragments cues when recognition fluctuates.
  Evidence: `backend/_srt_mixin.py:21-174`, `backend/detection.py`, `backend/tracking.py`, https://github.com/SubtitleEdit/subtitleedit/blob/main/docs/features/video-ocr.md, and https://www.unicode.org/reports/tr29/.
  Touches: `backend/detection.py`, `backend/tracking.py`, `backend/_srt_mixin.py`, translation cue flow, SRT and multilingual fixture tests.
  Acceptance: Detection text and confidence remain associated with stable tracks, cue text uses confidence-weighted near-equivalent consensus, and no second whole-frame OCR call occurs when tracked text exists. Exact source timing is retained, a fallback handles missing recognized text, and Latin, CJK, combining-mark, and RTL regressions pass without merging genuinely changed captions.
  Complexity: M

- [ ] P1 | RM-311: Verify output identity before skip-existing
  Why: Existence-only skip logic can accept stale or truncated outputs and then write current provenance beside bytes that were never validated against the current source or configuration.
  Evidence: `backend/cli.py:2021`, `backend/cli.py:2253-2262`, `backend/cli.py:2498-2507`, `backend/batch_report.py:904-998`, and the stronger fingerprint checks in `backend/resume_checkpoint.py:307-352`.
  Touches: `backend/cli.py`, `backend/batch_report.py`, `backend/resume_checkpoint.py`, sidecar schema and migration, batch and watch tests, CLI documentation.
  Acceptance: The default skip policy requires matching source SHA-256, normalized processing configuration, output path, byte size, and output SHA-256 in a versioned sidecar. Missing or mismatched evidence reprocesses the item and never relabels the old output. An explicit legacy `any` policy remains available and is visibly recorded; CLI, batch, and watch tests cover every branch.
  Complexity: M

- [ ] P1 | RM-312: Pin every auto-fetched model to immutable artifacts
  Why: VACE can download the repository's moving default revision, and its manifest has no artifact hashes, so two runs with the same settings can execute different weights.
  Evidence: `backend/inpainters_diffusion.py:330-370`, `backend/adapter_manifest.py:170-190`, `backend/adapter_manifest.py:298-430`, https://huggingface.co/docs/huggingface_hub/guides/download, and https://huggingface.co/docs/hub/security-malware.
  Touches: `backend/inpainters_diffusion.py`, `backend/adapter_manifest.py`, `backend/model_downloads.py`, reproducibility sidecars, support bundles, model security tests, privacy and offline documentation.
  Acceptance: Auto-fetch resolves an allowlisted commit before download, verifies every required artifact hash, and records repository, commit, files, hashes, and cache path in provenance. Missing or mismatched identity fails before model load. Any unsafe override is explicit, non-default, and recorded. Documentation lists every optional outbound model path.
  Complexity: M

- [ ] P1 | RM-313: Replace setup's silent partial fallback with validated profiles
  Why: A failed full install silently switches to four packages, omits declared security floors, and still reports that all dependencies installed.
  Evidence: `setup.py:568-656`, `requirements.txt:27`, `dependency_profiles.json`, `Run_VSR_Pro.bat:14`, and `Run_VSR_Pro.ps1`.
  Touches: `setup.py`, requirements and dependency profiles, launchers, setup reports, dependency verification tests, setup documentation.
  Acceptance: Setup installs a named locked profile or exits nonzero with a repair command. A supported core profile includes every runtime and security dependency it needs, prints its exact capabilities, passes the same dependency verifier used by launchers, and cannot emit a full-success message after a partial failure. Failure-simulation tests leave the environment diagnosable and rerunnable.
  Complexity: M

- [ ] P1 | RM-314: Prevent same-user settings and queue races
  Why: Multiple GUI processes share files, use only process-local locks, and ignore whether the named Windows mutex already existed.
  Evidence: `gui/config.py:141-148`, `gui/app.py:159`, settings persistence paths, and queue persistence paths.
  Touches: `gui/app.py`, `gui/config.py`, queue persistence, startup status handling, multiprocessing tests, architecture documentation.
  Acceptance: A second GUI process detects the existing per-user instance, reports the condition without foreground activation, and exits without writing state. Any CLI and GUI paths that may legitimately coexist use cross-process atomic locking or isolated state. Multiprocessing tests prove settings and queue files remain valid and no newer write is lost.
  Complexity: M

- [ ] P1 | RM-315: Ship provider-labeled CPU and CUDA bundles with benchmark evidence
  Why: The generic v3.38.0 Windows artifact is effectively CPU-only while product guidance recommends NVIDIA, and users cannot predict install size, provider use, throughput, RAM, VRAM, or output fidelity before downloading.
  Evidence: `VideoSubtitleRemoverPro.spec`, `dependency_profiles.json`, v3.38.0 release SBOM, the current RTX 4070 SUPER host that reactivates stale GPU-validation entries in `Roadmap_Blocked.md`, https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/1, https://github.com/YaoFANGUK/video-subtitle-remover, and https://github.com/KKenny0/videowipe/releases/tag/v0.8.0.
  Touches: release build scripts and specs, dependency profiles, runtime inventory, SBOM and audit generation, packaged smoke tests, benchmark tooling, installer names, README release matrix.
  Acceptance: Local release tooling produces separately named CPU and NVIDIA CUDA 12.8 artifacts from dedicated locked profiles. Each artifact carries checksums, SBOM, dependency audit, actual-provider inventory, and a frozen inference smoke result. Machine-readable benchmark evidence records input and config hashes, cold and warm time, FPS, peak RAM and VRAM, exact-timing verification, and quality gates on CPU and the current RTX 4070 SUPER. DirectML is marked unverified and is not published as a tested bundle until equivalent hardware evidence exists.
  Complexity: XL

### P2

- [ ] P2 | RM-316: Hold the Windows system awake only while processing
  Why: Long GUI and CLI jobs can be interrupted by idle sleep even though Windows provides a scoped processing power request that does not require the display to stay on.
  Evidence: no current `SetThreadExecutionState` use, https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate, and https://learn.microsoft.com/en-us/windows/powertoys/awake.
  Touches: shared platform utility, GUI processing lifecycle, CLI processing lifecycle, cancellation and crash cleanup, platform tests, user documentation.
  Acceptance: Active processing acquires one reference-counted `ES_SYSTEM_REQUIRED | ES_CONTINUOUS` hold and releases it on success, failure, cancellation, pause, and process exit. The implementation never requests `ES_DISPLAY_REQUIRED` or away mode, does not override user-initiated sleep, is a non-Windows no-op, and passes mocked lifecycle tests for overlapping jobs.
  Complexity: S
