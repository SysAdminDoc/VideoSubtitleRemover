# Research — Video Subtitle Remover Pro
Date: 2026-07-29 — replaces all prior research.

## Executive Summary

[Verified] Video Subtitle Remover Pro 3.29.0 is a mature Windows-first, local-only Python/tkinter application and CLI for hard-subtitle and text-watermark removal. Its strongest shape is the complete workflow around detection, timed/tracked masks, local inpainting, review, pause/resume, lossless matte interchange, quality reports, and reproducibility sidecars; the 2026-07-29 suite passes 913 tests with two environment-dependent skips. The highest-value direction is to make that breadth demonstrably trustworthy before adding more models: fail closed on writer errors, repair the contradictory NVIDIA dependency profile, make release artifacts atomic, report requested versus effective inference, and make user-state/artifact replacement recoverable.

Top opportunities, in priority order:

1. [Verified] Fail closed when FFV1 finalization or frame-sequence writes fail (`backend/io.py:1681-1706,1757-1760`).
2. [Verified] Unify setup, profile locks, provider smoke, and provider-specific security floors; the current NVIDIA profile requires both ORT `<1.27` and `==1.27.0`.
3. [Verified] Build installer, portable ZIP, evidence, and checksums in one versioned release transaction; `build/release/` currently mixes v3.22 evidence with a later installer.
4. [Verified] Put stdin writes under the shared subprocess timeout/cancellation policy (`backend/subprocess_policy.py:183-202`).
5. [Verified] Reject unknown GUI/preset modes and expose requested/effective OCR, device, provider, inpaint backend, fallback reason, and throughput.
6. [Verified] Make settings, queue, presets, model-cache imports, and matte replacement observable and rollback-safe.
7. [Verified] Make opt-in crash reporting fail closed if privacy scrubbing fails.
8. [Verified] Make onboarding and editor dialogs scrollable/resizable at 125-200% text scale, then finish runtime-string and RTL coverage.
9. [Likely] Add OpenCV-engine and mask-aware temporal regression contracts before changing inference defaults.
10. [Likely] Turn reviewed mattes into durable queue inputs and add loss-aware WebVTT interchange without expanding into a full subtitle editor.

## Product Map

- **Core workflows:** [Verified] add images/videos/folders; select automatic, scripted, or timed/keyframed masks; preview detection and cleanup; process a resumable batch; review quality failures and selectively rerun corrected spans.
- **Output workflows:** [Verified] preserve or transcode audio/video/container payloads; export SRT, comparison/quality evidence, reproducibility sidecars, NLE interchange, and exact FFV1/PNG matte artifacts.
- **User personas:** [Likely] privacy-sensitive creators, archivists, localization/dubbing operators, and automation users who need repeatable local processing rather than upload/credit services.
- **Platforms and distribution:** [Verified] Windows 10/11 GUI and CLI from Python 3.11-3.14-compatible source profiles, plus unsigned PyInstaller/NSIS artifacts; CPU, NVIDIA CUDA, and AMD/Intel DirectML lanes are declared.
- **Key data flow:** [Verified] media probe/decode -> OCR/manual regions -> tracking/refinement -> mask composition -> inpaint/fallback -> FFV1 intermediate/final encode -> payload restoration -> output contract/quality report/sidecar.
- **Key integrations:** [Verified] FFmpeg/ffprobe, OpenCV, RapidOCR/PaddleOCR/EasyOCR/Surya/VLM adapters, ONNX Runtime/OpenVINO/PyTorch, local translation commands, and optional hash-gated model adapters.

## Competitive Landscape

- **YaoFANGUK/video-subtitle-remover:** [Verified] Does well: broad cross-platform packages, automatic OCR, regions, and multiple inpainting modes. Learn: installation/provider proof and output-fidelity contracts answer recurring provider, blur, truncation, headless, and recovery complaints. Avoid: its monolithic runtime and implicit fallback behavior.
- **IOPaint:** [Verified] Does well: direct mask editing, batch/API workflows, model switching, feathering, and lifecycle controls. Learn: keep correction immediate and unload heavyweight models deterministically. Avoid: an archived WebUI/plugin surface and its dependency breadth.
- **ProPainter and DEVIL:** [Verified] Do well: temporal propagation and diagnostics stratified by camera motion, background motion, and mask dynamics. Learn: add lightweight masked-warp/flicker regression fixtures. Avoid: ProPainter's non-commercial redistribution and DEVIL's full learned-metric dependency stack.
- **Subtitle Edit:** [Verified] Does well: focused local OCR correction, autosave, quality summaries, translation catalogs, and self-contained distribution. Learn: safe concurrent persistence and measurable localization completeness. Avoid: turning VSR into a full subtitle authoring application.
- **Adobe After Effects:** [Verified] Does well: tracked masks, range controls, fill modes, lighting correction, and clean-reference frames. Learn: publish an honest accessibility support/limitation matrix. Avoid: subscription/cloud assumptions and broad compositing scope.
- **DaVinci Resolve Studio:** [Verified] Does well: mask tracking/object removal and “Render in Place” traveling mattes. Learn: make an approved matte a durable, reusable queue artifact. Avoid: opaque, disposable cache dependence.
- **Topaz Video and HandBrake:** [Verified] Do well: effective processor/resource visibility, restart-resumable queues, local activity logs, and child-process isolation. Learn: expose actual execution and later isolate native-crash-prone jobs. Avoid: adding cloud execution or retaining unnecessary user media.
- **Runway, Filmora, HitPaw, and CapCut:** [Verified] Do well: low-friction brush/box correction, preview, tracking, and multiple fill modes. Learn: preserve fast feedback and clear speed/fidelity expectations. Avoid: credits, uploads, account dependence, provider ambiguity, and general-object-removal scope.

## Security, Privacy, and Reliability

- [Verified] `_LosslessIntermediateWriter.release()` kills a timed-out FFmpeg process and discards nonzero exit status; `_FrameSequenceWriter.write()` ignores `cv2.imwrite()`'s Boolean result and advances its index. Both can let processing/checkpointing continue after incomplete output (`backend/io.py:1681-1706,1757-1760`).
- [Verified] The reviewed NVIDIA dependency contract is unsatisfiable: `setup.py:35-46` and `backend/dependency_caps.py:43-50` require `onnxruntime-gpu>=1.26,<1.27`, while `dependency_profiles.json:31-44` and `dependency_profiles/nvidia.txt:23` lock 1.27.0. Setup catches that resolver failure and continues without the promised CUDA ORT path (`setup.py:592-605`).
- [Verified] ORT 1.28.0 adds malformed-model memory-safety hardening, but the app's generic 1.26.0 CPU/GPU floor cannot express newer CPU fixes alongside constrained CUDA-12, CUDA-13, and separately pinned DirectML channels. The 2026-07-29 environment audit also finds protobuf 6.33.4 affected by CVE-2026-0994; 6.33.5 is fixed.
- [Verified] `build_exe.bat:169-238` reuses `build/release/`, overwrites only the installer, and does not regenerate the portable ZIP or `SHA256SUMS.txt`. The staged checksum names a different installer hash; as of 2026-07-29, the published release is v3.22.0 while source metadata is 3.29.0. Release publication needs clean versioned staging and immutable-asset verification, never code signing.
- [Verified] `_before_send()` returns the original event after any scrub exception, and `_path_scrub()` can retain the processed file basename despite the module's “no file names” promise (`backend/crash_reporter.py:43-52,110-126`). Opt-in telemetry must drop the event on scrub failure.
- [Verified] Settings, queue, and user-preset saves suppress write errors (`gui/config.py:932-938,944-1031,1207-1221`); a future `vsr_settings_format` is loaded with unknown keys discarded and may then be overwritten (`gui/config.py:861-870`). Save failures and downgrade protection need explicit UI-visible outcomes.
- [Verified] Model-cache import reads `manifest.json` without a size cap and moves an existing target to `.vsrbak` before the failing move is enrolled in rollback (`backend/cache_inventory.py:499-503,626-658`). Matte export deletes/promotes the artifact before its matching manifest is safely committed (`backend/matte_interchange.py:172-220`). Failure injection can leave lost or mismatched state.
- [Verified] `run_process()` performs a potentially large synchronous stdin write before entering its timeout/cancel loop (`backend/subprocess_policy.py:183-202`); a child that never reads can hang indefinitely. Some support-bundle FFmpeg probes also bypass the shared process policy (`backend/support_bundle.py:436-450`).
- [Verified] Existing guardrails remain strong: local processing by default, no account requirement, hash-gated optional adapters, bounded model ZIP extraction, source-preserving output paths, pause checkpoints, redacted support bundles, and strict output contracts.

## Architecture Assessment

- [Verified] **Configuration boundary:** `gui/config.py:313-320` still converts any unknown mode to STTN. Settings/preset import must distinguish valid GUI modes, valid backend-only modes, and invalid values; invalid values must never become a successful preset application.
- [Verified] **Runtime provenance boundary:** `backend/detection.py:168-193` can satisfy a CUDA request with RapidOCR CPU, while `backend/support_bundle.py:619-699` labels an engine/backend string as “provider.” Queue items, reports, sidecars, and smoke tests need one requested/effective execution record.
- [Verified] **User-state boundary:** atomic file writes exist, but callers discard their outcome. A small persistence result/error contract should cover settings, queue, presets, corrupt-file backup, and future-schema read-only handling.
- [Verified] **Artifact transaction boundary:** cache import and matte export need a shared journaled replacement primitive or equivalent per-module two-phase protocol, with recovery tests at every move/write boundary.
- [Verified] **UI boundary:** onboarding, region editor, and mask correction use fixed non-resizable top-level windows (`gui/onboarding.py:30-34`, `gui/region_controller.py:177-182`, `gui/mask_correction_controller.py:142-145`). At 200% text scale, measured dialog heights approach or exceed the 1152-pixel work area; all actions need scroll/reflow and keyboard reachability.
- [Verified] **Localization boundary:** gettext tooling and qps-Ploc cover 563/563 strings, but qps-Ploc is the only bundled catalog and several runtime strings/layout directions still bypass `tr()` or retain left/right assumptions. Engineering coverage and RTL probes should land before soliciting human catalogs.
- [Verified] **Test boundary:** the full 913-test suite passes, but it does not fail on FFV1 nonzero finalization, `cv2.imwrite(False)`, a non-reading stdin child, transaction failure at each promotion step, actual requested/effective providers, OpenCV 5 DNN engine selection, or temporal motion strata.
- [Likely] **Workflow improvement:** quality review should be able to “freeze” the approved matte, bind its source fingerprint/hash/manifest to the queue item, and rerun without OCR/tracking. `backend/matte_interchange.py` and queue persistence already supply most primitives.
- [Likely] **Standards improvement:** add a WebVTT parser/serializer beside the SRT-shaped translation model, preserving cue IDs/settings/spans and reporting losses. Defer TTML/IMSC until real demand justifies its XML and image-profile surface.
- [Likely] **Failure containment:** `gui/processing_controller.py:121,340-539` runs the queue in one in-process worker thread. A later supervised child-per-item boundary would let native OpenCV/ORT/model crashes fail one item while the GUI and remaining queue survive.
- [Verified] **Documentation drift:** `docs/architecture.md` claims 3.29.0 currency while describing `processor.py` as a shim and `app.py` as layout owner; both still contain substantial implementation. `README.md:85-104` calls artifacts unsigned and then promises signed Winget installers, `CHANGELOG.md` has two Unreleased sections, and `CLAUDE.md` retains stale counts/timeout statements.

## Rejected Ideas

- [Verified] **Bundle/default-enable ProPainter, E2FGVI, MatAnyone, CLEAR, ROSE, SEDiT, or other research models** — rejected or already blocked by non-commercial/absent licenses, unpublished artifacts, transitive model terms, VRAM, or missing live GPU validation. Sources: upstream repositories/papers and `Roadmap_Blocked.md`.
- [Verified] **Force RapidOCR onto CUDA** — rejected: RapidOCR's own dynamic-input benchmarks recommend CPU; accelerate only after a subtitle-frame benchmark proves a benefit. Source: RapidOCR inference-engine documentation.
- [Likely] **Cloud processing, mobile clients, multi-user workspaces, accounts, or plugin marketplace** — rejected: they contradict the local/privacy-first Windows product, introduce authentication/storage/abuse surfaces, and do not solve a verified current gap. Sources: Runway pricing/workflow and YaoFANGUK/IOPaint comparisons.
- [Likely] **General object, face, or license-plate removal and text-prompt targeting** — rejected: Filmora and IOPaint show demand, but the scope duplicates dedicated editors and expands misuse and model/licensing risk; VSR already covers subtitle/script filtering and tracked manual regions.
- [Verified] **Full subtitle editor** — rejected: Subtitle Edit already owns waveform authoring, correction, and conversion; VSR should preserve focused SRT/WebVTT translation/re-embed interoperability.
- [Verified] **TTML/IMSC import in this plan** — rejected: no project/community demand was found, and safe XML plus region/image-profile fidelity is materially larger than the verified WebVTT gap. Source: IMSC 1.2.
- [Verified] **Learned LPIPS/VFID benchmark dependencies by default** — rejected: DEVIL's diagnostic dimensions are useful, but deterministic masked-warp/flicker fixtures provide regression value without model downloads and release bloat.
- [Verified] **Code signing as a release fix** — rejected by repository policy. Repair staging, hashes, unsigned-install guidance, and immutable GitHub releases instead. Sources: `AGENTS.md`, GitHub immutable-release documentation.

## Sources

### Project and Open Source

- https://github.com/SysAdminDoc/VideoSubtitleRemover
- https://github.com/SysAdminDoc/VideoSubtitleRemover/releases/tag/v3.22.0
- https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/7
- https://github.com/YaoFANGUK/video-subtitle-remover
- https://github.com/YaoFANGUK/video-subtitle-remover/issues/194
- https://github.com/YaoFANGUK/video-subtitle-remover/issues/224
- https://github.com/YaoFANGUK/video-subtitle-remover/issues/244
- https://github.com/Sanster/IOPaint
- https://github.com/SubtitleEdit/subtitleedit
- https://github.com/sczhou/ProPainter
- https://github.com/MichiganCOG/devil
- https://github.com/geekyutao/Inpaint-Anything
- https://github.com/allenk/VeoWatermarkRemover
- https://github.com/allenk/VeoWatermarkRemover/issues/29
- https://github.com/SubtitleEdit/subtitleedit/pull/12976
- https://github.com/suhwan-cho/awesome-video-inpainting
- https://github.com/zengyh1900/Awesome-Image-Inpainting
- https://github.com/topics/video-inpainting

### Commercial and Adjacent Products

- https://helpx.adobe.com/after-effects/using/content-aware-fill.html
- https://www.adobe.com/accessibility/compliance/adobe-after-effects-win-2024-07-acr.html
- https://www.blackmagicdesign.com/products/davinciresolve/studio
- https://www.blackmagicdesign.com/products/davinciresolve/whatsnew
- https://filmora.wondershare.com/ai-video-object-remover.html
- https://www.hitpaw.com/remove-watermark.html
- https://docs.topazlabs.com/video-ai/reference-guide/importing-previewing-and-exporting
- https://docs.topazlabs.com/video-ai/reference-guide/preferences
- https://handbrake.fr/docs/en/1.9.0/technical/process-isolation.html
- https://help.runwayml.com/hc/en-us/articles/19155664495379-Inpainting
- https://runway.com/pricing
- https://www.capcut.com/resource/how-to-remove-objects-from-video

### Community Signal

- https://www.reddit.com/r/VideoEditors/comments/1rchdtn/hardcoded_subtitle_removal_from_short_videos/
- https://www.reddit.com/r/davinciresolve/comments/18fdsue
- https://www.reddit.com/r/davinciresolve/comments/18w9jc6
- https://news.ycombinator.com/item?id=45988018
- https://news.ycombinator.com/item?id=27766655
- https://forum.videohelp.com/threads/418726-Is-there-a-way-to-remove-hardcoded-subtitles-without-cropping
- https://stackoverflow.com/questions/55832273/inpaint-function-is-not-working-for-all-kind-of-image-to-remove-watermark-usin

### Standards and Platform Guidance

- https://www.w3.org/TR/webvtt1/
- https://www.w3.org/TR/ttml-imsc1.2/
- https://www.matroska.org/technical/subtitles.html
- https://ffmpeg.org/ffmpeg-formats.html
- https://www.w3.org/WAI/WCAG22/Understanding/reflow
- https://www.w3.org/WAI/WCAG22/Understanding/focus-not-obscured-minimum
- https://learn.microsoft.com/en-us/windows/apps/design/globalizing/globalizing-portal
- https://learn.microsoft.com/en-us/windows/uwp/design/globalizing/adjust-layout-and-fonts--and-support-rtl

### Dependencies, Security, and Research

- https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- https://github.com/microsoft/onnxruntime/releases/tag/v1.28.0
- https://github.com/microsoft/onnxruntime/releases/tag/v1.27.0
- https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/how_to_use_infer_engine/
- https://github.com/opencv/opencv/wiki/OpenCV-5
- https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases
- https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/secure-your-dependencies/verify-release-integrity
- https://nvd.nist.gov/vuln/detail/CVE-2026-0994
- https://test.osv.dev/vulnerability/GHSA-7gcm-g887-7qv7
- https://ffmpeg.org/security.html
- https://openaccess.thecvf.com/content/ICCV2023/html/Zhou_ProPainter_Improving_Propagation_and_Transformer_for_Video_Inpainting_ICCV_2023_paper.html
- https://openaccess.thecvf.com/content/CVPR2022/html/Szeto_The_DEVIL_Is_in_the_Details_A_Diagnostic_Evaluation_Benchmark_CVPR_2022_paper.html

## Open Questions

- None. The prioritized work is implementable and testable from the current repository and cited public evidence; hardware-specific adapter work remains explicitly parked in `Roadmap_Blocked.md`.
