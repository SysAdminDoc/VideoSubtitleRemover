# Research: Video Subtitle Remover Pro

Date: 2026-08-23. Replaces all prior research.

Repository state: v3.38.0 at [7ee0ce1](https://github.com/SysAdminDoc/VideoSubtitleRemover/commit/7ee0ce1a316e07c3853f2cd9565a5d2d6bc6998a).

Confidence labels:

- **Verified:** confirmed in the v3.38.0 source, tracker, release artifacts, or a local read-only probe.
- **Corroborated:** supported by project evidence and at least one independent primary source.
- **Needs live validation:** requires hardware, licensed media, or assistive technology that was not exercised during this research pass.

## Executive Summary

Video Subtitle Remover Pro is a local-first Windows desktop and command-line tool for detecting, tracking, removing, exporting, translating, and reburning visible video text. Its strongest current shape is unusually broad for an open source remover: exact timestamp handling, HDR-aware processing, polygon masks, temporal quality gates, recoverable jobs, donor references, review tools, batch preflight, and detailed release evidence are already present in v3.38.0 (`README.md`, `backend/processor.py`, `backend/quality.py`, `backend/resume_checkpoint.py`). The highest-value direction is not another model. It is making every user choice truthful, every reused result verifiable, and every distributed runtime measurable.

Priority opportunities:

1. **RM-307, requested-stage execution contract.** [Verified] Explicit diffusion, OCR, segmentation, tracking, and restoration choices can still catch a load or runtime failure and silently use another implementation or unchanged pixels. Provenance can then disagree with the work actually performed (`backend/inpainters_diffusion.py:126-165`, `backend/detection.py:411-440`, `backend/processor.py:1376-1389`, `backend/post_restore.py`).
2. **RM-308, manual-region contract.** [Verified] The command bar's Manual region choice is a no-op, and fixed-region mode is cleared for LaMa, ProPainter, and Auto even though the processor can build a fixed mask independently of the inpainting mode (`gui/layout_build.py:175`, `gui/app.py:1381-1386`, `gui/settings_controller.py:603-608`, `backend/processor.py:2117-2146`). This is the unresolved part of [issue #7](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/7).
3. **RM-309, FFmpeg 9 frame-duration compatibility.** [Verified] VSR requests legacy `pkt_duration` fields, while FFmpeg 9 exposes frame `duration` fields. A local FFmpeg 9.0.1 probe returned no values for the legacy request. Adjacent timestamps hide most failures, but final and irregular frame durations lose authoritative evidence (`backend/io.py:953`, [ffprobe source](https://ffmpeg.org/doxygen/trunk/ffprobe_8c_source.html)).
4. **RM-310, tracked OCR consensus for SRT and translation.** [Verified] `_read_text_for_boxes` ignores its `boxes` argument, reruns OCR over the full frame, and merges only exact adjacent strings. This wastes work and fragments cues when OCR fluctuates (`backend/_srt_mixin.py:21-174`). [Subtitle Edit](https://github.com/SubtitleEdit/subtitleedit/blob/main/docs/features/video-ocr.md) demonstrates near-equivalent temporal collapsing and representative text selection.
5. **RM-311, verified skip-existing behavior.** [Verified] CLI, batch, and watch paths treat file existence as success, then can write current provenance beside an unverified older output (`backend/cli.py:2021`, `backend/cli.py:2253-2262`, `backend/cli.py:2498-2507`). Existing checkpoint fingerprints provide a stronger pattern (`backend/resume_checkpoint.py:307-352`).
6. **RM-312, immutable model snapshots.** [Verified] VACE auto-fetch can call `snapshot_download(revision=None)` and its manifest has no artifact hashes (`backend/inpainters_diffusion.py:330-370`, `backend/adapter_manifest.py:170-190`). Hugging Face documents that an omitted revision resolves the current repository state, while commit revisions provide immutable identity ([download guide](https://huggingface.co/docs/huggingface_hub/guides/download)).
7. **RM-313, validated setup profiles.** [Verified] A failed full dependency install silently falls back to four packages, omits declared security floors, and still prints “All dependencies installed” (`setup.py:568-656`, `requirements.txt:27`, `Run_VSR_Pro.bat:14`).
8. **RM-314, same-user state concurrency.** [Verified] GUI settings and queue state are shared by all processes for one Windows user, their locks are process-local, and the named mutex result is never checked (`gui/config.py:141-148`, `gui/app.py:159`). Two instances can race on persisted state.
9. **RM-315, provider-labeled release bundles with benchmark evidence.** [Corroborated] The v3.38.0 public bundle contains CPU ONNX Runtime and no default Torch payload, while the README recommends an NVIDIA GPU. Focused competitors publish provider-specific bundles, and their trackers repeatedly expose throughput, VRAM, and output-fidelity uncertainty ([v3.38.0 release](https://github.com/SysAdminDoc/VideoSubtitleRemover/releases/tag/v3.38.0), [Yao VSR](https://github.com/YaoFANGUK/video-subtitle-remover), [VideoWipe v0.8.0](https://github.com/KKenny0/videowipe/releases/tag/v0.8.0)).
10. **RM-316, Windows processing power hold.** [Corroborated] Long jobs have no execution-state request. Windows provides a scoped system-required flag that prevents idle sleep without forcing the display on, and PowerToys uses the same behavior for long-running work ([SetThreadExecutionState](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate), [PowerToys Awake](https://learn.microsoft.com/en-us/windows/powertoys/awake)).

## Product Map

### Core workflows

- **Remove visible text:** import a file or queue, choose automatic or guided detection, track regions, inpaint frames, preserve audio and media timing, then review quality evidence (`gui/app.py`, `backend/processor.py`, `backend/batch_report.py`).
- **Plan and review:** preview detections, edit masks, use donor references or clean plates, export and import track plans, inspect contact sheets, and rerun failed regions (`gui/preview_controller.py`, `gui/region_controller.py`, `backend/track_plan.py`).
- **Subtitle localization:** export SRT, translate cues through configured providers, reburn localized text, or export timed masks and NLE sidecars (`backend/_srt_mixin.py`, `backend/subtitle_translation.py`, `backend/post_restore.py`, `backend/nle_sidecar.py`).
- **Operate long jobs:** batch preflight, disk forecasting, pause and resume, atomic output publication, watch folders, failure classification, support bundles, and crash recovery (`backend/batch_report.py`, `backend/work_directory.py`, `backend/resume_checkpoint.py`, `backend/atomic_replace.py`, `backend/cli.py`).
- **Validate media fidelity:** preserve exact frame timing, metadata, HDR transfer behavior, alpha masks, subtitle streams, and audio while producing temporal and color quality signals (`backend/io.py`, `backend/hdr.py`, `backend/quality.py`).

### User personas

- Editors and localization teams removing burned-in subtitles before translation or reburning.
- Archivists and restoration operators repairing overlays while preserving timing, audio, color, and metadata.
- Privacy-sensitive users who need local CPU or GPU processing without uploading footage.
- Technical batch users who need CLI, watch-folder, report, and reproducibility controls.

These personas are [Corroborated] by VSR's own tracker and by the guided removal, batch, clean-plate, and localization features sold by [Adobe](https://helpx.adobe.com/after-effects/using/content-aware-fill.html), [DaVinci Resolve Studio](https://www.blackmagicdesign.com/products/davinciresolve/studio), [Mocha Pro](https://borisfx.com/products/mocha-pro/), and [NukeX](https://www.foundry.com/products/nuke-family/nukex).

### Platforms and distribution

- [Verified] Windows 11 is the primary product surface. The repository builds a portable onedir bundle and NSIS installer, with a Linux CPU container as a secondary CLI path (`VideoSubtitleRemoverPro.spec`, `installer/vsr.nsi`, `Dockerfile`).
- [Verified] Source setup supports CPU, NVIDIA CUDA 12.8, and DirectML dependency profiles. The v3.38.0 downloadable Windows artifact is not provider-labeled and its release SBOM shows the CPU ONNX Runtime profile (`setup.py`, `dependency_profiles.json`, `build/release/3.38.0`).
- [Needs live validation] NVIDIA source inference has not yet been captured as frozen-build evidence on the current RTX 4070 SUPER host. DirectML also lacks current hardware evidence (`Roadmap_Blocked.md`).

### Key integrations and data flows

- FFmpeg and ffprobe handle probing, frame clocks, audio and subtitle stream handling, encoding, and verification (`backend/io.py`, `backend/processor.py`).
- RapidOCR and optional PaddleOCR, EasyOCR, Surya, and VLM paths feed normalized text geometry into tracking (`backend/detection.py`, `backend/ocr_vlm.py`).
- STTN, LaMa, VSR's ProPainter hybrid, external commands, and optional diffusion adapters repair masks through a shared registry (`backend/inpainter_registry.py`, `backend/inpainters_diffusion.py`).
- Optional model downloads cross the local-only boundary. Update checks, crash uploads, VLM endpoints, VACE snapshots, and speech-model fetches must remain explicit and observable (`backend/model_downloads.py`, `backend/ocr_vlm.py`, `README.md`).
- JSON settings, queue state, checkpoint fingerprints, reports, and reproducibility sidecars form the durable control plane (`gui/config.py`, `backend/resume_checkpoint.py`, `backend/batch_report.py`).

## Competitive Landscape

| Product or project | What it does well | Learn and avoid |
|---|---|---|
| [YaoFANGUK Video Subtitle Remover](https://github.com/YaoFANGUK/video-subtitle-remover) | Provider-specific CPU, DirectML, and CUDA downloads; task-specific regions; PP-OCR; image batches | Learn from clear provider choices. Avoid ambiguous GPU claims and unmeasured quality; issues [#3](https://github.com/YaoFANGUK/video-subtitle-remover/issues/3), [#64](https://github.com/YaoFANGUK/video-subtitle-remover/issues/64), and [#200](https://github.com/YaoFANGUK/video-subtitle-remover/issues/200) report throughput, multi-GPU, blur, and bitrate problems. |
| [VideoWipe](https://github.com/KKenny0/videowipe) | Streaming segments, long-video memory controls, repeatable timing, RSS, environment, and source identity in v0.8.0 | Copy the evidence shape, not GPL-3.0 code. Add cold and warm timing, RAM, VRAM, input identity, and actual provider to release benchmarks. |
| [WatermarkRemover-AI](https://github.com/D-Ogi/WatermarkRemover-AI) | Guided mask workflow and a focused desktop surface | Keep manual guidance dependable. Avoid weak hardware communication and uncapped memory paths reported in [#38](https://github.com/D-Ogi/WatermarkRemover-AI/issues/38) and [#48](https://github.com/D-Ogi/WatermarkRemover-AI/issues/48). |
| [IOPaint](https://github.com/Sanster/IOPaint) | Mature image-inpainting UI, model selection, plugins, and local operation | Reuse the clarity of model status and guided masks. Do not import an image-first plugin surface that bypasses VSR's video timing and provenance contracts. |
| [ProPainter](https://github.com/sczhou/ProPainter) | Strong propagation and transformer-based video inpainting | Retain its temporal ideas as benchmark references. Do not bundle its noncommercial implementation, and do not assume 12 GB cards avoid OOM failures reported in [#76](https://github.com/sczhou/ProPainter/issues/76). |
| [DiffuEraser](https://github.com/lixiaowen-xw/DiffuEraser) | Modern propagation plus diffusion repair | Benchmark only. Its dependency weight, runtime cost, and acceleration requests such as [#34](https://github.com/lixiaowen-xw/DiffuEraser/issues/34) do not support a default desktop backend. |
| [CLEAR](https://github.com/silent-commit/CLEAR) | Apache-2.0 mask-free subtitle removal with released code and weights | It is the strongest new experimental candidate. Require redistribution review, 12 GB VRAM evidence, timing, color, and large-font tests before any adapter work. |
| [SVOR](https://github.com/xiaomi-research/SVOR) | 2026 mask-free removal and degradation-aware training ideas | Use its mask-union and benchmark ideas. Avoid a default integration because the documented 24 GB to 33 GB VRAM requirement conflicts with the target desktop. |
| [VOID](https://github.com/netflix/void-model) | Apache-2.0 object removal with mask reasoning | Its reasoning and benchmark design are useful. The documented 40 GB-plus VRAM path belongs in high-end experiments, not the normal release. |
| [EffectErase](https://github.com/FudanCVL/EffectErase) | Explicit treatment of shadows, reflections, and associated effects | Add those failure classes to quality research. Do not bundle CC BY-NC assets or a SAM2.1 and Wan dependency chain. |
| [Subtitle Edit](https://github.com/SubtitleEdit/subtitleedit/blob/main/docs/features/video-ocr.md) | Samples frames, collapses near-identical images and text, and selects a representative OCR result | Apply temporal consensus to VSR's existing tracked detections. Avoid a second whole-frame OCR pass per cue. |
| [Adobe Content-Aware Fill](https://helpx.adobe.com/after-effects/using/content-aware-fill.html) | Guided masks, reference frames, fill ranges, and professional review | Keep donor frames, operator correction, and scoped reprocessing first-class. Avoid subscription or cloud dependence. |
| [DaVinci Resolve Studio](https://www.blackmagicdesign.com/products/davinciresolve/studio) | Integrated object removal, tracking, color management, and finishing | Preserve color and timeline evidence as part of removal. Avoid claiming parity with a full NLE or expanding the minimal FCPXML sidecar into a fragile pseudo-editor. |
| [Mocha Pro](https://borisfx.com/products/mocha-pro/) | Planar tracking, clean plates, occlusion controls, and guided removal | Keep clean plates and tracking visible to advanced operators. Avoid hiding failures behind an automatic result. |
| [NukeX](https://www.foundry.com/products/nuke-family/nukex) | SmartVector, CopyCat, paint, and node-level inspectability | Learn from explicit stage identity and debuggable intermediate data. Avoid a node system that would overwhelm the focused desktop workflow. |

Market conclusions:

- [Corroborated] VSR already exceeds focused open source peers in media fidelity, recovery, and review breadth. The clearest missing proof is provider-specific performance and frozen-build behavior, not another checkbox (`README.md`, [VideoWipe v0.8.0](https://github.com/KKenny0/videowipe/releases/tag/v0.8.0)).
- [Corroborated] Long-video consistency, defective masks, occlusions, shadows, and reflections remain unsolved across current tools and research. Existing VSR temporal gates are the correct foundation; new model families should compete on a fixed corpus before integration ([BeyondMasks](https://yigitekin.github.io/BeyondMasks/), [PROVE](https://xiaomi-research.github.io/prove/), [DEVIL](https://openaccess.thecvf.com/content/CVPR2022/html/Szeto_The_DEVIL_Is_in_the_Details_A_Diagnostic_Evaluation_Benchmark_CVPR_2022_paper.html)).
- [Corroborated] Commercial tools consistently expose masks, clean plates, tracking, batch work, and operator correction. VSR should make its existing guided path truthful and testable instead of pursuing one-click-only behavior.

## Reported Issues

### This repository

[Verified] The public tracker has 0 open issues and 0 open pull requests as of 2026-08-23. All 7 closed issues were read, along with enabled discussions.

- [Issue #7](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/7): the reported silent STTN fallback was fixed, but the separate request to apply a manual region with LaMa remains broken in the command bar and settings controller. RM-308 covers the traced current defect.
- [Issue #1](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/1): CUDA 12.8 and RTX 50-series setup code landed, but the issue closed without reporter confirmation and the public frozen bundle remains CPU-oriented. RM-315 treats this as validation and distribution debt, not a confirmed current GPU bug.
- [Issue #5](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/5): the first closure was premature, then queue selection and region propagation were fixed and moved into active workflow tests. Do not reopen unless those tests regress (`tests/test_gui_workflow_release.py`).
- [Issues #3](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/3) and [#4](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/4): RapidOCR packaging and file-dialog failures are fixed and covered by the current build configuration. No new item.
- [Issue #6](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/6): no sample, settings, or support bundle was provided. The report is not reproducible, so it does not justify a model change.
- [Issue #2](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/2): the malware allegation had no scanner signature or VirusTotal evidence. Unsigned-binary reputation remains blocked on a signing identity, but there is no evidence of malicious code.
- [Discussion #8](https://github.com/SysAdminDoc/VideoSubtitleRemover/discussions/8): a beginner walkthrough is the only unresolved concrete documentation request. It remains RM-306 in `Roadmap_Blocked.md` because recording and authenticated publishing are unavailable. It is not duplicated here.
- [Discussion #9](https://github.com/SysAdminDoc/VideoSubtitleRemover/discussions/9): the hardware and failure-feedback request has no replies. RM-315 should publish first-party measurements rather than wait for anecdotal submissions.

### External tracker and community signal

- [Corroborated] Focused removers repeatedly receive reports about low GPU use, OOM failures, unclear performance, blur, and output bitrate. These reports support RM-315 but are anecdotal, not cross-project benchmarks ([Yao #3](https://github.com/YaoFANGUK/video-subtitle-remover/issues/3), [Yao #200](https://github.com/YaoFANGUK/video-subtitle-remover/issues/200), [ProPainter #76](https://github.com/sczhou/ProPainter/issues/76), [VideoVanish discussion](https://news.ycombinator.com/item?id=45988018)).
- [Corroborated] Occlusion and foreground crossings are persistent failure modes in professional and community workflows. Existing mask correction and donor tools should remain visible; model replacement alone is not an adequate response ([Adobe community](https://community.adobe.com/t5/after-effects-discussions/content-aware-fill-issues-when-anything-passes-in-front-of-the-mask/td-p/14046912), [Stable Diffusion community](https://www.reddit.com/r/StableDiffusion/comments/1quw6ve/reliable_video_object_removal_inpainting_model/)).
- [Verified] Closed VSR reports show a recurring process risk: green unit tests have not always covered the exact GUI control users exercised. RM-308 therefore requires a command-bar workflow regression in the isolated GUI test environment, not only controller unit tests (`tests/test_gui_workflow_release.py`, commit history through 7ee0ce1).

## Security, Privacy, and Reliability

### Current strengths

- [Verified] A 2026-08-23 audit of the 32 installed release dependencies found no known vulnerability. Current floors include Python 3.13.15, ONNX Runtime 1.29.0 for CPU, OpenCV 5.0.0, Pillow 12.3.0, RapidOCR 3.9.2, idna 3.15, and PyInstaller 6.22.2 (`requirements.txt`, `dependency_profiles.json`, `build/release/3.38.0`).
- [Verified] ONNX metadata is bounded before parsing, remote-code adapters are restricted, normal outputs publish atomically, resume checkpoints validate source and configuration fingerprints, and support bundles redact paths and secrets (`backend/onnx_model_info.py`, `backend/remote_model_policy.py`, `backend/adapter_manifest.py`, `backend/atomic_replace.py`, `backend/resume_checkpoint.py`, `backend/support_bundle.py`).
- [Verified] Optional VLM network use is explicit and local processing remains the default. V3.38.0 added endpoint provenance and remote-VLM warnings (`backend/ocr_vlm.py`, `backend/batch_report.py`, `CHANGELOG.md`).

### Risks and missing guardrails

| Finding | Evidence and required control |
|---|---|
| Explicit stages can silently change semantics | `backend/inpainters_diffusion.py`, `backend/detection.py`, `backend/processor.py`, and `backend/post_restore.py` contain broad fallback or unchanged-output paths. RM-307 must make explicit selections fail closed and reserve cross-engine fallback for Auto. |
| Auto-fetched model identity is mutable | VACE uses an optional revision and accepts an unpinned snapshot. RM-312 must resolve and record an immutable commit plus artifact hashes before execution. Hugging Face's [download](https://huggingface.co/docs/huggingface_hub/guides/download) and [model security](https://huggingface.co/docs/hub/security-malware) guidance supports this boundary. |
| Existing output can be mislabeled | Existence-only skip paths can generate fresh sidecars for stale bytes. RM-311 must bind source, configuration, and output hashes before a skip is called successful. |
| Setup can report a partial environment as complete | `setup.py:568-656` silently switches dependency meaning after failure. RM-313 must expose a named locked profile, verify it, and leave a repairable failure state when no complete profile installs. |
| Same-user writes are not coordinated between processes | `gui/config.py:141-148` uses process-local locks and `gui/app.py:159` ignores the existing-mutex result. RM-314 must prevent or serialize concurrent persisted-state mutation. |
| Long processing does not hold the system awake | No power-request API is present. RM-316 must use `ES_SYSTEM_REQUIRED | ES_CONTINUOUS`, clear it on every exit path, never request `ES_DISPLAY_REQUIRED`, and remain a non-Windows no-op. |
| Optional network behavior is described too broadly | `README.md` says only update checks and opt-in crash reports make outbound requests, while VACE and faster-whisper can download weights (`backend/model_downloads.py`). RM-312 documentation must list every opt-in network path and the immutable model identity it records. |
| Security reporting is not documented | `.github/` has public issue forms but no **SECURITY.md**. This is useful follow-up work, but lower impact than the ten selected items because no private reporting channel has been established. |

### Advisory assessment

- [Verified] Current optional Torch floors are above fixes for [GHSA-53q9-r3pm-6pq6](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6) and [GHSA-63cw-57p8-fm3p](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p). Known hashes and model identity remain necessary because malicious checkpoints are a separate trust boundary.
- [Verified] PyInstaller 6.22.2 is above the [onefile privileged-extraction fix](https://github.com/pyinstaller/pyinstaller/security/advisories/GHSA-9fxf-4qw3-ghmr), and VSR ships onedir with an `asInvoker` installer.
- [Verified] Pillow 12.3.0, idna 3.15, OpenCV 5.0.0, and ONNX Runtime 1.29.0 meet the reviewed current advisory floors ([Pillow notes](https://pillow.readthedocs.io/en/stable/releasenotes/12.3.0.html), [idna advisory](https://github.com/kjd/idna/security/advisories/GHSA-65pc-fj4g-8rjx), [OpenCV CVE-2025-53644](https://nvd.nist.gov/vuln/detail/CVE-2025-53644), [ONNX Runtime 1.29.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.29.0)).
- [Needs live validation] FFmpeg 9.0.1 is current, but its published security matrix does not yet prove whether the pinned build contains the [CVE-2026-58049 fix](https://github.com/FFmpeg/FFmpeg/commit/f8d7795dcca36a4dd412e89cbd83e3dfec1e0d81). Release evidence should record the exact source revision.
- [Verified] CVE-2026-0994 concerns protobuf JSON parsing, not VSR's bounded ONNX wire parser. No VSR code calls `json_format` or `ParseDict`; a protobuf major-version jump is not justified without a transitive-call trace ([NVD](https://nvd.nist.gov/vuln/detail/CVE-2026-0994), [follow-up issue](https://github.com/protocolbuffers/protobuf/issues/26432)).

## Architecture Assessment

### Boundary improvements

- [Verified] Model selection is centralized in registries, but outcome semantics are not. RM-307 should add a shared requested-stage result contract carrying requested implementation, actual implementation, provider, fallback chain, failure class, and recovery hint. `auto` may choose alternatives; a named implementation may not silently cross engines (`backend/inpainter_registry.py`, `backend/device_provider.py`, `backend/detection.py`, `backend/post_restore.py`).
- [Verified] Fixed-region state is split between a command-bar string, settings toggles, persisted configuration, and processor behavior. RM-308 should establish one typed region mode with `automatic`, `manual-only`, and `guided-additive` semantics, then serialize it once (`gui/app.py`, `gui/settings_controller.py`, `gui/config.py`, `backend/processor.py`).
- [Verified] SRT text is detached from tracked detection geometry even though detection records already carry text and confidence. RM-310 should preserve recognized text on the track and compute cue consensus there, with `backend/_srt_mixin.py` limited to timing and serialization (`backend/detection.py`, `backend/tracking.py`, `backend/_srt_mixin.py`).
- [Verified] Checkpoint identity is stronger than completed-output identity. RM-311 should reuse the checkpoint fingerprint primitives in a versioned output-sidecar verifier instead of adding a second hashing design (`backend/resume_checkpoint.py`, `backend/batch_report.py`).

### Refactor candidates

- [Verified] `SubtitleRemover.process_video` is 1,169 lines, `RegionSelectorWindow.open` is 742 lines, `cli._run_processing` is 689 lines, and `ProcessingController._process_item` is 438 lines (`backend/processor.py:2683-3851`, `gui/region_controller.py:79`, `backend/cli.py:2035`, `gui/processing_controller.py:848`). `docs/architecture.md:438` still describes `process_video` as approximately 250 lines.
- The processor extraction is already recorded as blocked in `Roadmap_Blocked.md`, so it is not duplicated. RM-307, RM-308, RM-310, and RM-311 should extract only the policy boundaries required by their tests. Architecture documentation must be regenerated or corrected in the same commits.
- [Verified] Batch reports use fixed filenames and can overwrite prior evidence in a reused output directory (`backend/batch_report.py:390`). Timestamped or run-ID report retention is worthwhile, but it falls below the selected correctness and trust work.

### Test and documentation gaps

- No active test selects Manual region through the command bar and verifies the resulting processor command across all inpainting modes (`tests/test_gui_workflow_release.py`). RM-308 must add that isolated-desktop regression without using an active display.
- Failure-injection tests currently assert silent TBE fallback for a failed diffusion adapter (`tests/test_hardening_inpaint.py:298-370`). RM-307 must invert those expectations for explicit requests and cover OCR, segmentation, tracking, restoration, and provenance.
- The timing suite does not run an FFmpeg 9 fixture whose final frame has an irregular authoritative duration. RM-309 must add both parser unit tests and a real ffprobe integration test.
- The SRT suite lacks multilingual near-match consensus, grapheme-boundary, and no-second-OCR assertions. RM-310 must cover Latin, CJK, and RTL samples using [Unicode UAX 29](https://www.unicode.org/reports/tr29/).
- No multiprocessing test races settings or queue writes, and the mutex existence path is untested. RM-314 must exercise separate processes, not threads.
- Release evidence records packages and runtime inventory but not provider-labeled frozen inference or a public cold and warm performance matrix. RM-315 must generate both from machine-readable evidence.

### Category coverage decisions

- **Accessibility:** keyboard access, scaling, contrast, pseudo-localization, RTL, and MSAA metadata already have automated coverage. Native UIA and live Narrator or NVDA validation remain explicitly blocked in `Roadmap_Blocked.md`; no duplicate was added.
- **Internationalization:** gettext, RTL mirroring, and pseudo-localization exist. A real locale requires human translation and review, so it is not an autonomous engineering item.
- **Observability and testing:** logs, JSONL events, stage timings, crash capture, support bundles, and quality reports are strong. New tests and truthful provenance are embedded in RM-307 through RM-315.
- **Documentation:** the beginner video remains RM-306 and blocked. Documentation changes required by setup, network, provider, and architecture work are part of each acceptance contract rather than standalone cleanup.
- **Distribution and upgrades:** RM-313 and RM-315 cover setup profiles, runtime identity, frozen bundles, and upgrade evidence.
- **Plugins:** the trusted in-process registry and external-command adapter are adequate. Automatic third-party discovery would expand the code-execution and support boundary without demonstrated demand.
- **Mobile, cloud, collaboration, and multi-user services:** they conflict with the focused local Windows product. RM-314 addresses same-user process safety without adding accounts or synchronization.
- **Offline and migration:** default processing is local, optional downloads need clearer identity under RM-312, and versioned config migration is already strong (`backend/config_schema.py`, `gui/config.py`).

## Rejected Ideas

- **Default CLEAR, SVOR, VOID, EffectErase, DiffuEraser, VideoPainter, E2FGVI, or upstream ProPainter backend:** [Verified] current redistribution terms, model rights, dependency weight, or 12 GB consumer-GPU cost fail the product's default-lane constraints. Keep candidates benchmark-only ([CLEAR](https://github.com/silent-commit/CLEAR), [SVOR](https://github.com/xiaomi-research/SVOR), [VOID](https://github.com/netflix/void-model), [EffectErase](https://github.com/FudanCVL/EffectErase)).
- **SEDiT integration:** no public implementation, weights, and usable license were verified as of 2026-08-23 ([project](https://zheng222.github.io/SEDiT_project/)).
- **Object-WIPER backend:** the public repository exposes the TokSim metric rather than a complete removal backend and has no clear project license ([paper](https://openaccess.thecvf.com/content/CVPR2026/html/Kushwaha_Object-WIPER_Training-Free_Object_and_Associated_Effect_Removal_in_Videos_CVPR_2026_paper.html), [repository](https://github.com/sakshamsingh1/object_wiper)).
- **Replace modular OCR and tracking with one end-to-end video transformer:** recent papers support long-term association but do not establish better desktop cost or multilingual accuracy for this product. Improve consensus around existing detections first ([Sequential Transformer](https://openaccess.thecvf.com/content/WACV2024/html/Zhang_Sequential_Transformer_for_End-to-End_Video_Text_Detection_WACV_2024_paper.html)).
- **Make PP-OCRv6 the default from upstream metrics alone:** PaddleOCR states that v6 recognition metrics use a different internal evaluation set and are not directly comparable. Keep it opt-in until VSR's subtitle corpus proves a win ([PaddleOCR 3.7.0](https://github.com/PaddlePaddle/PaddleOCR/releases/tag/v3.7.0)).
- **Use human matting models as subtitle-alpha estimators:** MatAnyone and Robust Video Matting target people, not glyph opacity, outlines, glow, or antialiasing. A labeled subtitle-matte benchmark would be required first ([MatAnyone](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_MatAnyone_Stable_Video_Matting_with_Consistent_Memory_Propagation_CVPR_2025_paper.html), [RVM](https://github.com/PeterL1n/RobustVideoMatting)).
- **Average-FPS timing, H.264 alpha, or repair in a tone-mapped HDR proxy:** these contradict the governing media contracts. Keep integer clocks, codec-specific alpha, and linear-light repair ([RFC 9559](https://www.rfc-editor.org/rfc/rfc9559.html), [FFmpeg codecs](https://www.ffmpeg.org/ffmpeg-codecs.html), [BT.2100-3](https://www.itu.int/rec/R-REC-BT.2100-3-202502-I/en)).
- **Silent DirectML replacement with Windows ML:** DirectML is in sustained engineering, but managed provider updates change reproducibility. Any Windows ML work remains a separate hardware-gated experiment ([DirectML](https://learn.microsoft.com/en-us/windows/ai/directml/dml-get-started), [Windows ML](https://github.com/microsoft/WindowsML)).
- **Qt rewrite, hosted processing, mobile client, account sync, collaboration, or a plugin marketplace:** these increase platform, privacy, and support costs without tracker demand. Preserve the local Windows desktop focus.
- **GitHub build workflows, winget or Store distribution, and generic signing attestations:** repository policy requires local builds and direct release artifacts; a trusted code-signing identity remains the actual SmartScreen blocker.
- **Refile RM-297 through RM-305 or RM-306:** RM-297 through RM-305 shipped in v3.37.0. RM-306 remains recorded once in `Roadmap_Blocked.md`.

## Sources

### Repository and tracker

- https://github.com/SysAdminDoc/VideoSubtitleRemover
- https://github.com/SysAdminDoc/VideoSubtitleRemover/releases/tag/v3.38.0
- https://github.com/SysAdminDoc/VideoSubtitleRemover/issues
- https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/1
- https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/2
- https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/5
- https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/6
- https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/7
- https://github.com/SysAdminDoc/VideoSubtitleRemover/discussions/8
- https://github.com/SysAdminDoc/VideoSubtitleRemover/discussions/9

### Open source competitors and adjacent tools

- https://github.com/YaoFANGUK/video-subtitle-remover
- https://github.com/YaoFANGUK/video-subtitle-remover/issues/3
- https://github.com/YaoFANGUK/video-subtitle-remover/issues/64
- https://github.com/YaoFANGUK/video-subtitle-remover/issues/200
- https://github.com/KKenny0/videowipe
- https://github.com/KKenny0/videowipe/releases/tag/v0.8.0
- https://github.com/D-Ogi/WatermarkRemover-AI
- https://github.com/D-Ogi/WatermarkRemover-AI/issues/38
- https://github.com/D-Ogi/WatermarkRemover-AI/issues/48
- https://github.com/qwtoe/DeWatermark
- https://github.com/alpinist-GH/notebooklm-watermark-remover
- https://github.com/bakhtiersizhaev/seedance-watermark-remover
- https://github.com/hjunior29/video-text-remover
- https://github.com/sueun-dev/video-text-eraser
- https://github.com/Sanster/IOPaint
- https://github.com/geekyutao/Inpaint-Anything
- https://github.com/gaomingqi/Track-Anything
- https://github.com/sczhou/ProPainter
- https://github.com/sczhou/ProPainter/issues/76
- https://github.com/MCG-NKU/E2FGVI
- https://github.com/lixiaowen-xw/DiffuEraser
- https://github.com/lixiaowen-xw/DiffuEraser/issues/34
- https://github.com/TencentARC/VideoPainter
- https://github.com/bigD233/D2DF
- https://github.com/Kunbyte-AI/ROSE
- https://github.com/silent-commit/CLEAR
- https://github.com/xiaomi-research/SVOR
- https://github.com/netflix/void-model
- https://github.com/FudanCVL/EffectErase
- https://github.com/sakshamsingh1/object_wiper
- https://github.com/pq-yang/MatAnyone2
- https://github.com/PeterL1n/RobustVideoMatting
- https://github.com/SubtitleEdit/subtitleedit/blob/main/docs/features/video-ocr.md
- https://github.com/smacke/ffsubsync
- https://github.com/aperumetsr/VideoSubFinder

### Commercial and professional products

- https://helpx.adobe.com/after-effects/using/content-aware-fill.html
- https://www.adobe.com/products/aftereffects/plans.html
- https://www.blackmagicdesign.com/products/davinciresolve/studio
- https://borisfx.com/products/mocha-pro/
- https://borisfx.com/documentation/mocha/2026.5.0/
- https://www.foundry.com/products/nuke-family/nukex
- https://learn.foundry.com/nuke/content/reference_guide/air_nodes/copycat.html
- https://learn.foundry.com/nuke/content/comp_environment/smartpaint/smartvector.html
- https://runway.com/pricing
- https://help.runwayml.com/hc/en-us/articles/51683104370451-Creating-with-Edit-Studio
- https://filmora.wondershare.com/guide/ai-video-object-remover.html
- https://www.media.io/video-watermark-remover.html
- https://www.echosubs.com/
- https://www.subtitle-remover.com/

### Discovery lists and community reports

- https://github.com/suhwan-cho/awesome-video-inpainting
- https://github.com/zengyh1900/Awesome-Image-Inpainting
- https://github.com/topics/video-inpainting
- https://github.com/sitkevij/awesome-video
- https://github.com/brandonhimpfen/awesome-ffmpeg
- https://github.com/kba/awesome-ocr
- https://news.ycombinator.com/item?id=45988018
- https://www.reddit.com/r/StableDiffusion/comments/1quw6ve/reliable_video_object_removal_inpainting_model/
- https://community.adobe.com/t5/after-effects-discussions/content-aware-fill-issues-when-anything-passes-in-front-of-the-mask/td-p/14046912
- https://forum.blackmagicdesign.com/viewtopic.php?f=33&hilit=magic+mask&t=199707
- https://community.topazlabs.com/t/topaz-video-ai-7-2-0-1b/94506?page=2

### Standards, specifications, and platform APIs

- https://www.unicode.org/reports/tr29/
- https://www.unicode.org/reports/tr14/
- https://www.w3.org/TR/webvtt1/
- https://www.w3.org/TR/ttml2/
- https://www.w3.org/TR/ttml-imsc1.3/
- https://www.w3.org/TR/png-3/
- https://www.rfc-editor.org/rfc/rfc9043.html
- https://www.rfc-editor.org/rfc/rfc9559.html
- https://www.itu.int/rec/R-REC-BT.2100-3-202502-I/en
- https://www.itu.int/pub/R-REP-BT.2446-1-2021
- https://www.itu.int/pub/r-rep-bt.2390
- https://ffmpeg.org/ffmpeg.html
- https://ffmpeg.org/ffprobe.html
- https://ffmpeg.org/doxygen/trunk/ffprobe_8c_source.html
- https://www.ffmpeg.org/ffmpeg-codecs.html
- https://www.ffmpeg.org/ffmpeg-filters.html
- https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate
- https://learn.microsoft.com/en-us/windows/powertoys/awake
- https://learn.microsoft.com/en-us/windows/ai/directml/dml-get-started
- https://github.com/microsoft/WindowsML
- https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/samples
- https://learn.microsoft.com/en-us/windows/win32/medfound/uncompressed-video-media-types
- https://learn.microsoft.com/en-us/windows/win32/api/dxgicommon/ne-dxgicommon-dxgi_color_space_type

### Academic work and benchmarks

- https://openaccess.thecvf.com/content/WACV2022/html/Suvorov_Resolution-Robust_Large_Mask_Inpainting_With_Fourier_Convolutions_WACV_2022_paper.html
- https://github.com/sczhou/ProPainter
- https://github.com/MCG-NKU/E2FGVI
- https://github.com/researchmm/STTN
- https://github.com/ruiliu-ai/FuseFormer
- https://arxiv.org/abs/2603.21901
- https://zheng222.github.io/SEDiT_project/
- https://xiaomi-research.github.io/prove/
- https://openaccess.thecvf.com/content/CVPR2022/html/Szeto_The_DEVIL_Is_in_the_Details_A_Diagnostic_Evaluation_Benchmark_CVPR_2022_paper.html
- https://openaccess.thecvf.com/content/CVPR2026/html/Kushwaha_Object-WIPER_Training-Free_Object_and_Associated_Effect_Removal_in_Videos_CVPR_2026_paper.html
- https://yigitekin.github.io/BeyondMasks/
- https://openaccess.thecvf.com/content/WACV2024/html/Zhang_Sequential_Transformer_for_End-to-End_Video_Text_Detection_WACV_2024_paper.html
- https://openaccess.thecvf.com/content/CVPR2021/html/Feng_Semantic-Aware_Video_Text_Detection_CVPR_2021_paper.html
- https://openaccess.thecvf.com/content/CVPR2024/html/Huang_Bridging_the_Gap_Between_End-to-End_and_Two-Step_Text_Spotting_CVPR_2024_paper.html
- https://arxiv.org/abs/2503.04058
- https://arxiv.org/abs/2312.01938
- https://arxiv.org/abs/2405.19194
- https://openaccess.thecvf.com/content/CVPR2025/html/Yang_MatAnyone_Stable_Video_Matting_with_Consistent_Memory_Propagation_CVPR_2025_paper.html
- https://openaccess.thecvf.com/content/CVPR2026/html/Yang_MatAnyone_2_Scaling_Video_Matting_via_a_Learned_Quality_Evaluator_CVPR_2026_paper.html

### Dependency releases and security advisories

- https://www.python.org/downloads/release/python-31315/
- https://www.python.org/downloads/release/python-3147/
- https://blog.python.org/2026/08/python-31214-31116-31021/
- https://www.ffmpeg.org/download.html
- https://ffmpeg.org/security.html
- https://github.com/FFmpeg/FFmpeg/commit/f8d7795dcca36a4dd412e89cbd83e3dfec1e0d81
- https://nvd.nist.gov/vuln/detail/CVE-2026-58049
- https://github.com/opencv/opencv/releases/tag/5.0.0
- https://github.com/opencv/opencv/wiki/OpenCV-4-to-5-migration
- https://nvd.nist.gov/vuln/detail/CVE-2025-53644
- https://github.com/microsoft/onnxruntime/releases/tag/v1.29.0
- https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
- https://github.com/RapidAI/RapidOCR/releases/tag/v3.9.2
- https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/default_models.yaml
- https://github.com/PaddlePaddle/PaddleOCR/releases/tag/v3.7.0
- https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/module_usage/text_recognition.en.md
- https://pillow.readthedocs.io/en/stable/releasenotes/12.3.0.html
- https://pyinstaller.org/en/stable/CHANGES.html
- https://github.com/pyinstaller/pyinstaller/security/advisories/GHSA-9fxf-4qw3-ghmr
- https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6
- https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p
- https://github.com/kjd/idna/security/advisories/GHSA-65pc-fj4g-8rjx
- https://nvd.nist.gov/vuln/detail/CVE-2026-0994
- https://github.com/protocolbuffers/protobuf/issues/26432
- https://github.com/protocolbuffers/protobuf/pull/27007
- https://www.tcl-lang.org/software/tcltk/8.6.html
- https://www.tcl-lang.org/software/tcltk/9.0.html
- https://github.com/python/cpython/issues/124111
- https://huggingface.co/docs/huggingface_hub/guides/download
- https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download
- https://huggingface.co/docs/hub/security-malware
- https://huggingface.co/docs/hub/main/security-pickle

## Open Questions

1. **CUDA frozen-build envelope:** Can the existing portable layout include a CUDA 12.8 lane that passes package-size, cold-start, VRAM, and inference checks on the current 12 GB RTX 4070 SUPER without making CPU setup less clear? RM-315 requires measured evidence before release.
2. **SRT equivalence thresholds:** Which confidence weighting and grapheme-aware distance thresholds minimize cue fragmentation without merging genuinely changed text across Latin, CJK, and RTL subtitle samples? RM-310 must calibrate these against a checked-in deterministic corpus.
3. **FFmpeg source security identity:** Does the exact FFmpeg 9.0.1 binary shipped by the project include commit `f8d7795dcca36a4dd412e89cbd83e3dfec1e0d81` for CVE-2026-58049? The current product version alone cannot answer this.
4. **Accessibility evidence:** Does the packaged app expose collapsed Advanced controls and canvas operations correctly to Narrator and NVDA? This requires the isolated assistive-technology environment already recorded in `Roadmap_Blocked.md`.
