# Research: Video Subtitle Remover Pro

Evidence cutoff: 2026-08-21  
Repository state: v3.36.0 at [d79ac68](https://github.com/SysAdminDoc/VideoSubtitleRemover/commit/d79ac688974640b2aadef6d917215f3c82024b01)  
Research mode: repository and ecosystem analysis only. No feature implementation.

This document replaces every earlier research pass. It reflects the current source,
tests, documentation, issue tracker, discussions, released dependencies, and external
market as of the evidence cutoff.

## Executive Summary

Video Subtitle Remover Pro is already a broad local desktop production tool, not a
single-model demo. It combines a Windows GUI and CLI, automatic and manual masks,
temporal restoration, LaMa, resumable batches, translation, quality evidence, NLE
exports, PyInstaller packaging, an NSIS installer, and a Linux CPU container
([README.md](README.md), [backend/processor.py](backend/processor.py),
[gui/app.py](gui/app.py)). The closest focused open source tools remain narrower or
carry substantial license, memory, dependency, or maintenance constraints
([YaoFANGUK/video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover),
[VideoWipe](https://github.com/KKenny0/videowipe),
[WatermarkRemover-AI](https://github.com/D-Ogi/WatermarkRemover-AI),
[ProPainter](https://github.com/sczhou/ProPainter)).

The strongest opportunity is trust. Users need confidence that the selected model
really ran, the preview represents the part of the clip they are judging, rotated
text does not erase an oversized rectangle, HDR and variable-frame-rate media remain
faithful, and a finished job has measurable temporal quality. Professional tools
build this confidence through tracked masks, keyframes, reference frames, previews,
and review controls rather than one-click model claims
([Adobe Content-Aware Fill](https://helpx.adobe.com/after-effects/using/content-aware-fill.html),
[Mocha Pro](https://borisfx.com/documentation/mocha/2026.5.0/),
[Nuke SmartVector](https://learn.foundry.com/nuke/content/comp_environment/smartpaint/smartvector.html)).

This pass reviewed nine source classes and more than 70 direct sources. It harvested
150 raw ideas, removed shipped features and duplicates, then scored the remaining
candidates for product fit, impact, effort, regression risk, prerequisites, novelty,
and evidence strength. Ten additions survived:

The nine completed source classes were direct competitors, first-party and external
issue trackers, forums and community discussions, academic papers and benchmarks,
standards and official engineering guidance, security advisories, dependency
releases, commercial products and pricing, and curated discovery lists. The Sources
section records 72 numbered groups and 129 unique external URLs.

1. Correct the PyTorch advisory floor used by strict release verification.
2. Enforce a local-by-default privacy boundary for the optional VLM server.
3. Restore executable GUI workflow and accessibility release proof.
4. Make Test Cleanup use the selected moment and temporal context.
5. Preserve OCR quadrilaterals through tracking and mask creation.
6. Preserve exact rational timestamps for variable-frame-rate media.
7. Repair HDR regions in linear light without reducing them to 256 code values.
8. Add mask-local, motion-aware temporal and outside-mask color quality checks.
9. Inventory OpenCV's embedded FFmpeg and pin the container media runtime.
10. Publish the beginner workflow requested in discussion #8.

The two P0 items close verifiable safety gaps. The current dependency pins are safe,
but strict release verification approves affected PyTorch 2.6.0 through 2.9.1 because
it checks only the older 2.6.0 floor
([backend/release_verification.py:525](backend/release_verification.py#L525),
[GHSA-63cw-57p8-fm3p](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p)).
The optional llama.cpp OCR path defaults to loopback, yet its normalizer accepts any
HTTP or HTTPS host and supplies that endpoint to PaddleOCRVL
([backend/ocr_vlm.py:311](backend/ocr_vlm.py#L311),
[backend/ocr_vlm.py:490](backend/ocr_vlm.py#L490)). Remote use therefore needs an
explicit acknowledgement and encrypted transport.

No new inpainting backend is recommended. CLEAR now has public code and weights, so
the former claim that it is unreleased is stale
([CLEAR repository](https://github.com/silent-commit/CLEAR),
[CLEAR model card](https://huggingface.co/charlesw09/CLEAR-mask-free-video-subtitle-removal)).
It still belongs in a research-only benchmark lane until model rights, base-model
terms, consumer GPU cost, persistent large-text quality, and global color drift are
validated
([CLEAR issue #2](https://github.com/silent-commit/CLEAR/issues/2),
[CLEAR issue #5](https://github.com/silent-commit/CLEAR/issues/5)).

Evidence labels used below:

- **Verified:** confirmed in current code, an official source, or a direct tracker.
- **Corroborated:** supported by at least two independent sources or by code plus an
  external source.
- **Needs live validation:** a behavior observed in one local UI session or requiring
  physical hardware, assistive technology, or a real media corpus before a claim can
  be closed.

## Product Map

### Product boundary

The product is a Windows-first, local-first subtitle and overlay restoration
application with a scriptable backend. Supported delivery paths include the GUI,
CLI, portable ZIP, NSIS installer, and Linux CPU container
([README.md](README.md), [build_exe.bat](build_exe.bat),
[installer/vsr.nsi](installer/vsr.nsi), [Dockerfile](Dockerfile)). Automatic network
use is not the core workflow. Optional model and update paths are separately gated
([backend/remote_model_policy.py](backend/remote_model_policy.py),
[backend/update_check.py](backend/update_check.py)).

### Primary workflows

- Import one or more videos, images, folders, or watched folders; inspect a preview;
  choose automatic detection or a manual region; run Test Cleanup; then process and
  review output ([gui/app.py](gui/app.py),
  [gui/preview_controller.py](gui/preview_controller.py),
  [gui/region_controller.py](gui/region_controller.py)).
- Detect subtitle areas with RapidOCR, PaddleOCR, optional OCR engines, or an OpenCV
  fallback; convert detections to tracks and masks; choose temporal, LaMa, or
  conventional restoration; finish, encode, and emit evidence
  ([backend/detection.py](backend/detection.py),
  [backend/processor.py](backend/processor.py),
  [backend/batch_report.py](backend/batch_report.py)).
- Resume isolated jobs, reuse frozen mattes, correct masks, rerun failed ranges, and
  export SRT, WebVTT, FCPXML, EDL, mattes, and reports
  ([backend/job_worker.py](backend/job_worker.py),
  [backend/resume_checkpoint.py](backend/resume_checkpoint.py),
  [backend/_srt_mixin.py](backend/_srt_mixin.py),
  [backend/nle_sidecar.py](backend/nle_sidecar.py)).

### Likely users

The implemented workflows fit privacy-sensitive creators, localization operators,
archivists, batch processors, and editors who need local control. The project's own
tracker confirms demand for portable operation, Docker, GPU compatibility, manual
regions, and a beginner walkthrough
([issue #1](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/1),
[issue #3](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/3),
[issue #5](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/5),
[discussion #8](https://github.com/SysAdminDoc/VideoSubtitleRemover/discussions/8#discussioncomment-17848058)).
There is not enough public demand evidence to justify a cloud service, collaboration
suite, mobile app, or marketplace.

### Current strengths

- Local long-form processing, resumable jobs, deterministic plans, and frozen-matte
  reruns are stronger than the one-shot workflows documented by most focused
  removers ([README.md](README.md),
  [backend/job_worker.py](backend/job_worker.py)).
- The package already preserves audio, attachments, subtitles, HDR metadata, color
  tags, and variable-frame-rate output at the container level
  ([backend/io.py](backend/io.py), [backend/processor.py](backend/processor.py),
  [backend/release_verification.py](backend/release_verification.py)).
- Dependency profiles, model manifests, advisory checks, checksums, support bundles,
  and release evidence provide an unusually strong local software supply-chain
  baseline for this product category
  ([dependency_profiles.json](dependency_profiles.json),
  [backend/adapter_manifest.py](backend/adapter_manifest.py),
  [backend/security_checks.py](backend/security_checks.py)).
- Manual regions, scripted regions, reference fills, mask corrections, scene cuts,
  temporal aggregation, Poisson finishing, grain restoration, and translucent-text
  handling are already implemented. Re-filing them would create duplicate roadmap
  work ([backend/processor.py](backend/processor.py),
  [backend/reference_fill.py](backend/reference_fill.py),
  [backend/segmentation.py](backend/segmentation.py)).

### Current friction

- Test Cleanup opens frame zero instead of the selected moment, even though the GUI
  presents a current preview and the README describes a selected-frame workflow
  ([gui/preview_controller.py:704](gui/preview_controller.py#L704),
  [README.md:55](README.md#L55)).
- The low-resolution proxy helper describes fast planning but has no production
  caller ([backend/proxy_workflow.py:45](backend/proxy_workflow.py#L45)).
- OCR quadrilaterals are reduced to axis-aligned boxes before mask creation, which is
  a poor fit for rotated signs and diagonal overlays
  ([backend/detection.py:654](backend/detection.py#L654),
  [WatermarkRemover-AI issue #38](https://github.com/D-Ogi/WatermarkRemover-AI/issues/38)).
- The active GUI test is an import smoke test. Eighteen archived GUI modules are
  ignored at collection time, including tests for region propagation, scaling,
  localization, threading, and shutdown
  ([tests/test_gui_import_smoke.py](tests/test_gui_import_smoke.py),
  [tests/archive/conftest.py](tests/archive/conftest.py)).
- **Needs live validation:** in one Windows accessibility-tree inspection, controls
  inside the collapsed Advanced area remained discoverable. The panel is hidden with
  pack_forget while individual child handles retain MSAA annotations
  ([gui/settings_controller.py:99](gui/settings_controller.py#L99),
  [backend/a11y.py](backend/a11y.py)). Microsoft documents that programmatically
  hidden controls should be removed from the control view and that a complete,
  logical tab order requires direct testing
  ([UIA control element guidance](https://learn.microsoft.com/en-us/accessibility-tools-docs/items/wpf/control_iscontrolelement),
  [Windows accessibility testing](https://learn.microsoft.com/en-us/windows/apps/design/accessibility/accessibility-testing)).

## Competitive Landscape

### Focused open source tools

| Project | Verified signal | Implication for VSR |
|---|---|---|
| [YaoFANGUK/video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover) | The closest packaged lineage combines detection and inpainting. Users report low GPU utilization, multi-GPU contention, and blurred output in [#3](https://github.com/YaoFANGUK/video-subtitle-remover/issues/3), [#64](https://github.com/YaoFANGUK/video-subtitle-remover/issues/64), and [#200](https://github.com/YaoFANGUK/video-subtitle-remover/issues/200). | Keep provider visibility, bounded memory, and explicit quality evidence. Do not assume a larger model fixes workflow trust. |
| [VideoWipe](https://github.com/KKenny0/videowipe) | Release 0.8.0 expanded reviewable wipe planning and execution ([release](https://github.com/KKenny0/videowipe/releases/tag/v0.8.0)). | VSR already has plan and batch concepts. Improve preview fidelity and proof instead of copying another plan format. |
| [WatermarkRemover-AI](https://github.com/D-Ogi/WatermarkRemover-AI) | Its tracker documents diagonal-text failure and dependency pain in [#38](https://github.com/D-Ogi/WatermarkRemover-AI/issues/38) and [#48](https://github.com/D-Ogi/WatermarkRemover-AI/issues/48). | Preserve polygon geometry and avoid binding core UX to an archived image-only dependency. |
| [DeWatermark](https://github.com/qwtoe/DeWatermark), [NotebookLM watermark remover](https://github.com/alpinist-GH/notebooklm-watermark-remover), [Seedance remover](https://github.com/bakhtiersizhaev/seedance-watermark-remover) | Narrow tools gain reliability from known overlay locations, template unions, or target-specific heuristics. NotebookLM 0.2.4 specifically corrected temporal union behavior ([release](https://github.com/alpinist-GH/notebooklm-watermark-remover/releases/tag/v0.2.4)). | A reviewed static-logo discovery pilot remains useful, but broad subtitle trust and quality work has higher priority. |
| [video-text-remover](https://github.com/hjunior29/video-text-remover) and [video-text-eraser](https://github.com/sueun-dev/video-text-eraser) | Small tools expose simple masks and frame pipelines with much less packaging and recovery depth. | Preserve VSR's local production advantage. Borrow focused regression cases, not architecture. |
| [IOPaint](https://github.com/Sanster/IOPaint) | Strong local inpainting UX, but the repository is archived and centered on still images. | Do not make it a runtime dependency. |

### Research systems and model families

[LaMa](https://openaccess.thecvf.com/content/WACV2022/html/Suvorov_Resolution-Robust_Large_Mask_Inpainting_With_Fourier_Convolutions_WACV_2022_paper.html)
remains the practical permissive baseline already supported by VSR. More advanced
video models do not automatically fit a redistributable Windows desktop product:

- [ProPainter](https://github.com/sczhou/ProPainter) has strong temporal results but
  uses a noncommercial license and has continuing VRAM complaints such as
  [issue #76](https://github.com/sczhou/ProPainter/issues/76).
- [E2FGVI](https://github.com/MCG-NKU/E2FGVI) is CC BY-NC 4.0 and has a reported
  long-video flicker problem in
  [issue #49](https://github.com/MCG-NKU/E2FGVI/issues/49).
- [VideoPainter](https://github.com/TencentARC/VideoPainter), [D2DF](https://github.com/bigD233/D2DF),
  and [ROSE](https://github.com/Kunbyte-AI/ROSE) raise material GPU, model-size, or
  license costs. ROSE's maintainer states that memory optimization was not a focus
  ([issue #15](https://github.com/Kunbyte-AI/ROSE/issues/15)).
- [CLEAR](https://github.com/silent-commit/CLEAR) is now reproducible enough for a
  benchmark lane, but not for default shipping. Its model card says research use,
  it depends on a 1.3B Wan base, and its tracker records persistent large-font and
  global color-shift failures
  ([model card](https://huggingface.co/charlesw09/CLEAR-mask-free-video-subtitle-removal),
  [issue #2](https://github.com/silent-commit/CLEAR/issues/2),
  [issue #5](https://github.com/silent-commit/CLEAR/issues/5)).
- [SEDiT](https://arxiv.org/abs/2605.14894) is promising research, but no verified
  redistributable code and weight route was found by the evidence cutoff.

The most useful 2026 academic result for the current product is evaluation, not
another backend. [PROVE](https://arxiv.org/abs/2605.14534) reports that common global
metrics can disagree with perception and miss localized temporal defects. Its
region-conditioned temporal approach compares shared restored regions across nearby
frames, which directly addresses VSR's raw adjacent-frame ROI SSIM limitation
([backend/quality.py:252](backend/quality.py#L252)).
[VSR-Bench-400](https://openreview.net/pdf?id=MIRtxjuZF6),
[EffectErase](https://github.com/FudanCVL/EffectErase), [Object-WIPER](https://openaccess.thecvf.com/content/CVPR2026/html/Kushwaha_Object-WIPER_Training-Free_Object_and_Associated_Effect_Removal_in_Videos_CVPR_2026_paper.html),
and [BeyondMasks](https://yigitekin.github.io/BeyondMasks/) reinforce the need to
evaluate effects, boundaries, motion, and content preservation rather than treating
the hole alone as the quality target.

### Professional and commercial workflows

| Product | What its official workflow emphasizes | Relevant lesson |
|---|---|---|
| [Adobe After Effects Content-Aware Fill](https://helpx.adobe.com/after-effects/using/content-aware-fill.html) | Work areas, masks, fill methods, alpha expansion, and reference frames. Adobe documents 8, 16, and 32 bits per channel support ([help](https://helpx.adobe.com/ca/after-effects/desktop/remove-objects-from-your-videos/content-aware-fill.html)). | Make the judged time range and bit-depth behavior explicit. |
| [DaVinci Resolve Studio](https://www.blackmagicdesign.com/products/davinciresolve/studio) | Object removal is integrated with tracking, review, grading, and delivery. | Keep restoration inside a reviewable media workflow. |
| [Mocha Pro](https://borisfx.com/products/mocha-pro/) | Planar tracking, editable splines, remove modules, keyframes, and review. | Preserve editable geometry and temporal corrections. |
| [Nuke CopyCat](https://learn.foundry.com/nuke/content/reference_guide/air_nodes/copycat.html) and [SmartVector](https://learn.foundry.com/nuke/content/comp_environment/smartpaint/smartvector.html) | High-control learned cleanup and vector-guided propagation. | Confidence maps and correction frames matter more than a hidden one-click pass. |
| [Runway Edit Studio](https://help.runwayml.com/hc/en-us/articles/51683104370451-Creating-with-Edit-Studio) and [Filmora AI Object Remover](https://filmora.wondershare.com/guide/ai-video-object-remover.html) | Cloud or consumer flows prioritize speed and low-friction selection. | VSR should stay local and make its extra control understandable to beginners. |
| [EchoSubs](https://www.echosubs.com/), [Subtitle Remover](https://www.subtitle-remover.com/), and [Media.io](https://www.media.io/video-watermark-remover.html) | Hosted services emphasize upload-and-download convenience and metered plans. | Local privacy, long-form processing, and inspectable evidence remain VSR's useful contrast. |

Professional media QC products such as
[Interra BATON](https://www.interrasystems.com/file-based-qc.php),
[QScan](https://qscan.io/features), and
[Telestream Qualify](https://docs.telestream.dev/docs/qualify-user-guide)
show that delivery confidence depends on deterministic reports, severity, locations,
and reviewable failures. VSR should report the worst restored frame pair and color
drift region, not only one aggregate score.

### Community and trend signals

Community reports consistently describe occlusion, motion, flicker, memory, and
selection quality as the hard problems
([Stable Diffusion discussion](https://www.reddit.com/r/StableDiffusion/comments/1quw6ve/reliable_video_object_removal_inpainting_model/),
[Adobe occlusion discussion](https://community.adobe.com/t5/after-effects-discussions/content-aware-fill-issues-when-anything-passes-in-front-of-the-mask/td-p/14046912),
[Resolve feature discussion](https://forum.blackmagicdesign.com/viewtopic.php?f=33&hilit=magic+mask&t=199707),
[Topaz Video discussion](https://community.topazlabs.com/t/topaz-video-ai-7-2-0-1b/94506?page=2)).
Awesome lists and GitHub topic pages show a large research supply but do not resolve
desktop distribution rights or maintenance fit
([Awesome Image Inpainting](https://github.com/zengyh1900/Awesome-Image-Inpainting),
[video-inpainting topic](https://github.com/topics/video-inpainting),
[awesome-video](https://github.com/sitkevij/awesome-video),
[awesome-ffmpeg](https://github.com/brandonhimpfen/awesome-ffmpeg),
[awesome-ocr](https://github.com/kba/awesome-ocr)).

## Reported Issues

### This repository

The GitHub tracker had zero open issues and zero open pull requests on 2026-08-21.
Seven issues were closed
([issues](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues),
[pull requests](https://github.com/SysAdminDoc/VideoSubtitleRemover/pulls)).
Closed reports still provide useful regression evidence:

- [Issue #1](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/1) covered RTX
  50-series and TV-range color handling. The Blackwell lane and color-range
  preservation now exist. Keep those paths in regression tests.
- [Issue #2](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/2) alleged an
  infostealer without a signature or reproducible indicator. The unsigned ZIP,
  checksums, and SmartScreen guidance are now documented
  ([README.md](README.md)). There is no evidence for re-filing malware work.
- [Issue #3](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/3) asked for
  a working frozen build, portable delivery, and Docker. Those delivery paths ship
  now ([build_exe.bat](build_exe.bat), [Dockerfile](Dockerfile)).
- [Issue #4](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/4) was a
  missing file-dialog import and is fixed.
- [Issue #5](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/5) reported
  region selection and Inspect failures. A first closure did not satisfy the
  reporter, then a second fix landed. The production code now stores selected
  regions, but the corresponding interaction tests sit under the ignored archive
  ([tests/archive/test_gui_smoke.py](tests/archive/test_gui_smoke.py)).
- [Issue #6](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/6) contains
  no sample or reproducible detail. It cannot support a new implementation item.
- [Issue #7](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/7) reported
  identical output across model selections. The backend now fails closed when an
  inpainter is unavailable, with active tests
  ([backend/device_provider.py](backend/device_provider.py),
  [tests/test_device_provider.py](tests/test_device_provider.py)).

The only unresolved concrete request is a beginner video showing import,
translation, inpainting, and output
([discussion #8](https://github.com/SysAdminDoc/VideoSubtitleRemover/discussions/8#discussioncomment-17848058)).
[Discussion #9](https://github.com/SysAdminDoc/VideoSubtitleRemover/discussions/9)
has no replies and does not establish demand for another hardware lane.

### External trackers

External reports cluster around four actionable failure classes:

1. **Geometry:** diagonal text fails when a quadrilateral becomes a rectangle
   ([WatermarkRemover-AI #38](https://github.com/D-Ogi/WatermarkRemover-AI/issues/38)).
2. **Temporal quality:** blurred output, long-video flicker, occlusions, and persistent
   overlays remain common
   ([Yao #200](https://github.com/YaoFANGUK/video-subtitle-remover/issues/200),
   [E2FGVI #49](https://github.com/MCG-NKU/E2FGVI/issues/49),
   [CLEAR #2](https://github.com/silent-commit/CLEAR/issues/2)).
3. **Performance:** low utilization, shared GPU memory, and propagation memory are
   recurring complaints
   ([Yao #3](https://github.com/YaoFANGUK/video-subtitle-remover/issues/3),
   [Yao #64](https://github.com/YaoFANGUK/video-subtitle-remover/issues/64),
   [Track Anything #4](https://github.com/gaomingqi/Track-Anything/issues/4)).
4. **Dependency reliability:** downstream tools break when they depend on old,
   archived, or tightly pinned stacks
   ([WatermarkRemover-AI #48](https://github.com/D-Ogi/WatermarkRemover-AI/issues/48),
   [IOPaint](https://github.com/Sanster/IOPaint)).

These reports support polygon preservation, proxy planning, local quality metrics,
and conservative dependency policy. They do not justify bundling the affected
projects.

## Security Privacy Reliability

### Current security posture

The reviewed CPU environment had no confirmed reachable runtime CVE on 2026-08-21.
Python 3.13.15, OpenCV 5.0.0.93, Pillow 12.3.0, ONNX Runtime CPU 1.29.0, PyInstaller
6.22.2, NSIS 3.12, external FFmpeg 9.0.1, Requests 2.34.2, urllib3 2.7.0, and idna
3.15 are at or above their reviewed security floors
([Python 3.13.15](https://www.python.org/downloads/release/python-31315/),
[OpenCV 5.0.0](https://github.com/opencv/opencv/releases/tag/5.0.0),
[Pillow 12.3.0](https://pillow.readthedocs.io/en/stable/releasenotes/12.3.0.html),
[ONNX Runtime 1.29.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.29.0),
[PyInstaller 6.22.2](https://github.com/pyinstaller/pyinstaller/releases/tag/v6.22.2),
[NSIS 3.12](https://nsis.sourceforge.io/Docs/AppendixF.html#v3.12),
[FFmpeg downloads](https://ffmpeg.org/download.html)).
The dependency profile and package consistency checks also passed locally
([dependency_profiles.json](dependency_profiles.json),
[backend/dependency_profiles.py](backend/dependency_profiles.py)).

This result should not be weakened by false positives:

- OpenCV 5.0.0.93 is beyond the fix for
  [CVE-2025-53644](https://nvd.nist.gov/vuln/detail/CVE-2025-53644).
- FFmpeg 9.0.1 contains the RASC bounds fix associated with
  [CVE-2026-58049](https://nvd.nist.gov/vuln/detail/CVE-2026-58049) through
  [fix commit f8d7795](https://github.com/FFmpeg/FFmpeg/commit/f8d7795dcca36a4dd412e89cbd83e3dfec1e0d81).
- PyInstaller 6.22.2 is beyond
  [GHSA-9fxf-4qw3-ghmr](https://github.com/pyinstaller/pyinstaller/security/advisories/GHSA-9fxf-4qw3-ghmr),
  and the project ships an onedir, asInvoker build rather than the affected modern
  onefile path ([build_exe.bat](build_exe.bat)).
- The protobuf JSON recursion advisory does not map to VSR's direct binary ONNX
  parsing path ([backend/onnx_model_info.py](backend/onnx_model_info.py),
  [protobuf issue #26432](https://github.com/protocolbuffers/protobuf/issues/26432)).

### P0 release gate defect

Strict release verification rejects Torch below 2.6.0 because of CVE-2025-32434, but
does not reject 2.6.0 through 2.9.1
([backend/release_verification.py:525](backend/release_verification.py#L525)).
PyTorch's 2026 advisory states that versions before 2.10.0 are affected by
CVE-2026-24747
([GHSA-63cw-57p8-fm3p](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p)).
Current project profiles use 2.11.0 or 2.13.0, so installed builds are safe
([dependency_profiles.json](dependency_profiles.json)). The defect is the gate:
it can approve a future or locally altered release with a known affected version.

### P0 VLM endpoint boundary

The PaddleOCR-VL llama.cpp path is documented and named as a local server, with a
loopback default
([README.md:380](README.md#L380),
[backend/ocr_vlm.py:56](backend/ocr_vlm.py#L56)). Its URL normalizer strips whitespace
and a trailing slash but performs no host, credential, or transport policy check
([backend/ocr_vlm.py:311](backend/ocr_vlm.py#L311)). The reachability probe accepts
HTTP and HTTPS for any parsed host, and PaddleOCRVL receives that URL before its
predict call is given a temporary PNG of the frame
([backend/ocr_vlm.py:320](backend/ocr_vlm.py#L320),
[backend/ocr_vlm.py:490](backend/ocr_vlm.py#L490),
[backend/ocr_vlm.py:514](backend/ocr_vlm.py#L514)).

Loopback should remain automatic. Any non-loopback endpoint should require an
explicit remote-processing acknowledgement, reject URL credentials, require HTTPS,
resolve hostnames before use, and display that frame content can leave the machine.
Tests must cover IPv4 loopback, IPv6 loopback, localhost, mapped addresses, redirects,
DNS changes, malformed URLs, credentials, and probe bypass behavior.

### Native media inventory gap

OpenCV reports embedded FFmpeg libraries, including avcodec, while VSR's release
evidence validates only the separately installed FFmpeg executable
([backend/security_checks.py:262](backend/security_checks.py#L262),
[backend/release_verification.py:1388](backend/release_verification.py#L1388)).
Untrusted media is opened through cv2.VideoCapture
([backend/io.py:1477](backend/io.py#L1477)). This is an inventory blind spot, not a
confirmed vulnerability. Release evidence should record both media runtimes and
their build versions. The container also installs an unversioned distribution
FFmpeg package from an unpinned base image
([Dockerfile:1](Dockerfile#L1)).

### Optional execution lanes

- The declared Python range accepts any 3.11 through 3.14 patch, while later
  security checks enforce 3.11.16, 3.12.14, and 3.13.15 floors
  ([dependency_profiles.json](dependency_profiles.json),
  [backend/security_checks.py:108](backend/security_checks.py#L108),
  [CVE-2026-11940](https://nvd.nist.gov/vuln/detail/CVE-2026-11940)).
  Setup messaging should eventually enforce the same patch floors.
- CPU ONNX Runtime 1.29.0 includes malformed-model and external-data hardening. CUDA
  12 remains on 1.26.0 because newer PyPI GPU wheels moved to CUDA 13, and DirectML
  remains on its legacy package
  ([ONNX Runtime 1.29.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.29.0),
  [CUDA provider compatibility](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html),
  [DirectML provider status](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)).
  Custom or unhashed models should remain strongly gated on those older lanes.
- PaddleOCR 3.7.0 supports newer PaddlePaddle releases than the project's 3.0.0 pin
  ([PaddleOCR repository](https://github.com/PaddlePaddle/PaddleOCR),
  [dependency_profiles.json](dependency_profiles.json)). Requalification is useful,
  but it should not displace the verified safety and media-integrity work.

### Reliability gaps

- HDR frames are reduced from uint16 to uint8 by division by 257. Repaired pixels
  are expanded by multiplication by 257, so the masked result can contain only 256
  code values before later encoding
  ([backend/processor.py:1520](backend/processor.py#L1520)). Metadata preservation
  does not correct this pixel-precision loss. BT.2100 defines PQ and HLG transfer
  systems, and BT.2446 covers conversion practice
  ([ITU-R BT.2100-3](https://www.itu.int/rec/R-REC-BT.2100-3-202502-I/en),
  [ITU-R BT.2446-1](https://www.itu.int/pub/R-REP-BT.2446-1-2021)).
- VFR probing and manifests store decimal seconds rather than integer source
  timestamps plus a rational time base
  ([backend/io.py:52](backend/io.py#L52),
  [backend/processor.py:2779](backend/processor.py#L2779)). FFmpeg and ffprobe expose
  timestamp and time-base semantics directly
  ([FFmpeg documentation](https://ffmpeg.org/ffmpeg.html),
  [ffprobe documentation](https://ffmpeg.org/ffprobe.html),
  [Matroska RFC 9559](https://www.rfc-editor.org/rfc/rfc9559.html)).
- Temporal consistency is the mean raw SSIM between adjacent restored ROIs
  ([backend/quality.py:252](backend/quality.py#L252)). Legitimate camera or object
  motion can lower it, while a localized defect can disappear in the mean.
  [PROVE](https://arxiv.org/abs/2605.14534) directly supports region-conditioned,
  localized temporal evaluation.
- TBE uses one-way DIS or Farneback warping without a forward-backward occlusion
  confidence test
  ([backend/inpainters/_common.py:907](backend/inpainters/_common.py#L907)).
  Bidirectional consistency is an established optical-flow confidence principle
  ([OpenCV DIS](https://docs.opencv.org/5.0/main_modules/classcv_1_1DISOpticalFlow.html),
  [UnFlow](https://ojs.aaai.org/index.php/AAAI/article/view/12276)).
  It remains a follow-on pilot after local quality metrics can identify the failing
  clips.

## Architecture Assessment

### Local findings

1. **Release advisory enforcement is internally inconsistent.** Safe profile pins
   coexist with an obsolete minimum in strict verification
   ([dependency_profiles.json](dependency_profiles.json),
   [backend/release_verification.py:525](backend/release_verification.py#L525)).
2. **The optional local VLM boundary is descriptive, not enforced.** The README says
   local, while code accepts a remote plaintext host
   ([README.md:380](README.md#L380),
   [backend/ocr_vlm.py:311](backend/ocr_vlm.py#L311)).
3. **GUI proof regressed from behavioral tests to import smoke.** The archived suite
   contains relevant interactions, but collection ignores every archived module
   ([tests/archive/conftest.py](tests/archive/conftest.py),
   [tests/test_gui_import_smoke.py](tests/test_gui_import_smoke.py)).
4. **Test Cleanup is not time-faithful.** It reads the first decoded frame and runs a
   single-frame cleanup even for temporal modes
   ([gui/preview_controller.py:704](gui/preview_controller.py#L704)).
5. **Planning proxy infrastructure is dormant.** The cache and generation helper has
   no production caller ([backend/proxy_workflow.py](backend/proxy_workflow.py)).
6. **OCR geometry loses rotation.** Polygon-like results are converted to min/max
   boxes before later processing
   ([backend/ocr_vlm.py:376](backend/ocr_vlm.py#L376),
   [backend/detection.py:654](backend/detection.py#L654)).
7. **VFR precision is represented with floats.** Exact source ticks are discarded at
   the probe boundary ([backend/io.py:52](backend/io.py#L52)).
8. **HDR repair is an 8-bit operation inside a high-bit wrapper.** The source surface
   is uint16, but masked fill values are quantized through uint8
   ([backend/processor.py:1520](backend/processor.py#L1520)).
9. **Quality evidence is global and motion-sensitive.** Raw adjacent-frame SSIM does
   not isolate local restoration flicker from legitimate motion
   ([backend/quality.py:252](backend/quality.py#L252),
   [PROVE](https://arxiv.org/abs/2605.14534)).
10. **Two decoder stacks are shipped but only one is inventoried.** OpenCV's embedded
    FFmpeg processes untrusted inputs without equivalent release evidence
    ([backend/io.py:1477](backend/io.py#L1477),
    [backend/release_verification.py:1388](backend/release_verification.py#L1388)).

The main processing method is also very large, and architecture documentation still
describes it as roughly 250 lines
([backend/processor.py:2247](backend/processor.py#L2247),
[docs/architecture.md](docs/architecture.md)). A decomposition-only roadmap item was
not selected because it would consume risk without closing a user-visible defect.
The selected media-integrity work should establish smaller seams as part of normal
implementation.

### Candidate scoring

Scores use 1 as low and 5 as high. Effort and risk are costs, so lower is better.
Novelty measures whether the capability is absent rather than partly shipped.
Evidence combines current code, direct user reports, standards, and independent
external confirmation.

| Roadmap | Candidate | Fit | Impact | Effort | Risk | Dependencies | Novelty | Evidence | Tier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| RM-297 | Correct PyTorch release security gate | 5 | 5 | 1 | 1 | 1 | 5 | 5 | P0 |
| RM-298 | Enforce VLM endpoint privacy boundary | 5 | 5 | 2 | 2 | 1 | 5 | 5 | P0 |
| RM-299 | Restore GUI workflow and accessibility proof | 5 | 5 | 3 | 2 | 2 | 4 | 5 | P1 |
| RM-300 | Make Test Cleanup temporally representative | 5 | 5 | 3 | 2 | 2 | 5 | 5 | P1 |
| RM-301 | Preserve OCR polygon geometry | 5 | 4 | 3 | 3 | 2 | 5 | 5 | P1 |
| RM-302 | Preserve exact rational VFR timing | 5 | 4 | 4 | 4 | 2 | 5 | 4 | P1 |
| RM-303 | Preserve linear-light high-bit HDR repairs | 5 | 5 | 5 | 5 | 2 | 5 | 5 | P1 |
| RM-304 | Add local temporal and color-drift quality gates | 5 | 5 | 3 | 2 | 2 | 5 | 5 | P1 |
| RM-305 | Inventory both FFmpeg runtimes | 5 | 4 | 3 | 2 | 2 | 5 | 5 | P1 |
| RM-306 | Publish the requested beginner workflow | 5 | 3 | 2 | 1 | 1 | 5 | 5 | P2 |

### Recommended sequence

RM-297 and RM-298 should land first because they are small, verifiable safety
boundaries. RM-299 should follow because later UI work needs an active behavioral
test harness. RM-300 and RM-301 then improve what the user previews and selects.
RM-302 and RM-303 change media representations and require dedicated fixtures.
RM-304 should calibrate quality gates against those fixtures. RM-305 closes release
inventory evidence. RM-306 can be recorded after the resulting workflow is stable.

### Verification strategy

- Use synthetic fixtures for deterministic geometry, timestamps, HDR ramps, motion,
  flicker, and global color casts. Keep outside-mask pixels exact where the pipeline
  promises preservation.
- Add a small redistributable real-world corpus only when usage rights are recorded.
  [EffectErase's dataset](https://huggingface.co/datasets/FudanCVL/EffectErase/tree/main)
  is CC BY-NC 4.0, so it is suitable for research comparison but not an unrestricted
  distributable commercial fixture.
- Report worst cases with frame numbers, timestamps, mask overlays, and metric
  components. A single pass/fail average is insufficient
  ([PROVE](https://arxiv.org/abs/2605.14534),
  [Telestream Qualify](https://docs.telestream.dev/docs/qualify-user-guide)).
- Run GUI interaction, scaling, high-contrast, RTL, and accessibility-tree checks in
  the local release workflow. Do not restore GitHub Actions
  ([build_exe.bat](build_exe.bat),
  [tools/ui_scaling_probe.py](tools/ui_scaling_probe.py)).
- Mark Narrator and NVDA behavior **Needs live validation** until recorded on the
  packaged build. MSAA names alone do not prove native UIA control patterns
  ([Microsoft UIA provider guidance](https://learn.microsoft.com/en-us/windows/win32/winauto/uiauto-serversideprovider)).

## Rejected Ideas

### Already shipped or duplicate

Do not create new roadmap items for resumable isolated jobs, watch folders, matte
export, frozen-matte reruns, donor references, manual and scripted regions, mask
correction, scene cuts, quality reports, SRT, WebVTT, translation, FCPXML, EDL,
support bundles, checksums, local release evidence, DirectML fallback, CUDA
selection, lossless intermediates, audio preservation, HDR metadata, VFR output,
high-contrast styling, RTL scaffolding, or MSAA naming. Each exists in current code
or documentation
([README.md](README.md), [backend](backend), [gui](gui), [tests](tests)).
The selected HDR and VFR items correct precision inside existing features. They do
not re-file the container-level features.

### Model and backend proposals not selected

- Do not bundle ProPainter, E2FGVI, VideoPainter, DiffuEraser, ROSE, D2DF, or CLEAR
  as defaults. Current license, model-rights, VRAM, base-model, or dependency costs
  conflict with a redistributable local desktop application
  ([ProPainter](https://github.com/sczhou/ProPainter),
  [E2FGVI](https://github.com/MCG-NKU/E2FGVI),
  [VideoPainter](https://github.com/TencentARC/VideoPainter),
  [DiffuEraser](https://github.com/lixiaowen-xw/DiffuEraser),
  [ROSE](https://github.com/Kunbyte-AI/ROSE),
  [D2DF](https://github.com/bigD233/D2DF),
  [CLEAR model card](https://huggingface.co/charlesw09/CLEAR-mask-free-video-subtitle-removal)).
- CLEAR is no longer rejected as nonexistent. Keep a research-only adapter manifest
  and benchmark path until redistribution rights, consumer GPU cost, large-font
  persistence, and color drift pass written gates
  ([CLEAR repository](https://github.com/silent-commit/CLEAR),
  [CLEAR #2](https://github.com/silent-commit/CLEAR/issues/2),
  [CLEAR #5](https://github.com/silent-commit/CLEAR/issues/5)).
- Do not add RAFT, XMem, or Cutie before bidirectional DIS confidence and the new
  local quality metrics show that the existing OpenCV lane is inadequate
  ([OpenCV DIS](https://docs.opencv.org/5.0/main_modules/classcv_1_1DISOpticalFlow.html),
  [UnFlow](https://ojs.aaai.org/index.php/AAAI/article/view/12276)).
- Do not add a generic model-plugin marketplace. Loading arbitrary weights or remote
  code would weaken the existing manifest and immutable-source policy
  ([backend/adapter_manifest.py](backend/adapter_manifest.py),
  [backend/remote_model_policy.py](backend/remote_model_policy.py)).

### Product directions not selected

- Do not replace Tk with Qt or replace FFmpeg with Media Foundation. Both rewrites
  would put mature codecs, containers, attachments, timestamps, and current GUI
  behavior at risk without direct demand
  ([FFmpeg documentation](https://ffmpeg.org/ffmpeg.html),
  [Media Foundation D3D guidance](https://learn.microsoft.com/en-us/windows/win32/medfound/mf-source-reader-d3d-manager)).
- Do not add REST, Gradio, cloud OCR, cloud inpainting, account sync, collaboration,
  macOS, ROCm, mobile, winget, Microsoft Store, or a hosted service. The current
  evidence supports local Windows production and a Linux CPU CLI, not those product
  branches ([README.md](README.md), [issues](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues)).
- Do not restore GitHub Actions. The repository's release policy is local, and the
  current release script already gathers evidence
  ([build_exe.bat](build_exe.bat),
  [backend/release_verification.py](backend/release_verification.py)).
- Do not claim WCAG conformance, complete screen-reader support, or bitwise
  cross-provider determinism without live evidence. WCAG 2.2 is useful design
  guidance but does not prove a native Windows application's behavior
  ([WCAG 2.2](https://www.w3.org/TR/WCAG22/),
  [Windows accessibility testing](https://learn.microsoft.com/en-us/windows/apps/design/accessibility/accessibility-testing)).

### Good ideas below the ten-item cut

- Reviewed static-logo discovery remains useful after the existing synthetic
  benchmark gains a real licensed clip
  ([backend/static_logo_benchmark.py](backend/static_logo_benchmark.py),
  [tests/clips/manifest.json](tests/clips/manifest.json)).
- Bidirectional flow confidence is a strong follow-on if RM-304 identifies
  motion-boundary donor failures
  ([backend/inpainters/_common.py:907](backend/inpainters/_common.py#L907),
  [UnFlow](https://ojs.aaai.org/index.php/AAAI/article/view/12276)).
- Track-level OCR text consensus could improve SRT output, but it needs exact timing
  from RM-302 and a multilingual grapheme corpus first
  ([backend/_srt_mixin.py](backend/_srt_mixin.py),
  [Unicode UAX 29](https://unicode.org/reports/tr29/)).
- Output hashes, model hashes, provider options, and deterministic flags would
  strengthen the sidecar after media timing and decoder inventory are stable
  ([backend/batch_report.py](backend/batch_report.py),
  [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)).
- A Windows processing-time sleep request is low risk, but it has less direct
  evidence than the selected work
  ([SetThreadExecutionState](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate)).
- IMSC 1.3, OpenTimelineIO, and native UIA fragment providers are credible future
  paths. No direct user request currently outranks the selected work
  ([IMSC 1.3](https://www.w3.org/TR/ttml-imsc1.3/),
  [OpenTimelineIO](https://github.com/AcademySoftwareFoundation/OpenTimelineIO),
  [UIA provider guidance](https://learn.microsoft.com/en-us/windows/win32/winauto/uiauto-serversideprovider)).

## Sources

All external sources were accessed on 2026-08-21. Repository paths refer to commit
[d79ac68](https://github.com/SysAdminDoc/VideoSubtitleRemover/commit/d79ac688974640b2aadef6d917215f3c82024b01)
unless a later roadmap-only commit is stated.

### Repository and tracker

1. [README.md](README.md): product, setup, workflow, supported outputs, and privacy.
2. [backend/processor.py](backend/processor.py): core processing, HDR conversion,
   timing, restoration, and final assembly.
3. [backend/io.py](backend/io.py): ffprobe timing and OpenCV media input.
4. [backend/detection.py](backend/detection.py): OCR result normalization and boxes.
5. [backend/ocr_vlm.py](backend/ocr_vlm.py): optional VLM endpoint and temporary
   frame path.
6. [gui/preview_controller.py](gui/preview_controller.py): preview and Test Cleanup.
7. [backend/proxy_workflow.py](backend/proxy_workflow.py): dormant planning proxy.
8. [backend/quality.py](backend/quality.py) and
   [backend/_quality_mixin.py](backend/_quality_mixin.py): quality metrics and gates.
9. [backend/security_checks.py](backend/security_checks.py) and
   [backend/release_verification.py](backend/release_verification.py): advisory and
   release evidence.
10. [dependency_profiles.json](dependency_profiles.json): CPU, CUDA, and DirectML
    pins.
11. [backend/a11y.py](backend/a11y.py),
    [gui/settings_controller.py](gui/settings_controller.py), and
    [docs/architecture.md](docs/architecture.md): accessibility and UI claims.
12. [tests/archive/conftest.py](tests/archive/conftest.py),
    [tests/archive/test_gui_smoke.py](tests/archive/test_gui_smoke.py), and
    [tests/test_gui_import_smoke.py](tests/test_gui_import_smoke.py): GUI coverage.
13. [Issue tracker](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues),
    [pull requests](https://github.com/SysAdminDoc/VideoSubtitleRemover/pulls),
    [discussion #8](https://github.com/SysAdminDoc/VideoSubtitleRemover/discussions/8),
    and [discussion #9](https://github.com/SysAdminDoc/VideoSubtitleRemover/discussions/9).

### Focused projects and direct issue evidence

14. [YaoFANGUK/video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover),
    including [#3](https://github.com/YaoFANGUK/video-subtitle-remover/issues/3),
    [#64](https://github.com/YaoFANGUK/video-subtitle-remover/issues/64), and
    [#200](https://github.com/YaoFANGUK/video-subtitle-remover/issues/200).
15. [VideoWipe](https://github.com/KKenny0/videowipe) and
    [release 0.8.0](https://github.com/KKenny0/videowipe/releases/tag/v0.8.0).
16. [WatermarkRemover-AI](https://github.com/D-Ogi/WatermarkRemover-AI),
    [issue #38](https://github.com/D-Ogi/WatermarkRemover-AI/issues/38), and
    [issue #48](https://github.com/D-Ogi/WatermarkRemover-AI/issues/48).
17. [DeWatermark](https://github.com/qwtoe/DeWatermark).
18. [NotebookLM watermark remover](https://github.com/alpinist-GH/notebooklm-watermark-remover)
    and [release 0.2.4](https://github.com/alpinist-GH/notebooklm-watermark-remover/releases/tag/v0.2.4).
19. [Seedance watermark remover](https://github.com/bakhtiersizhaev/seedance-watermark-remover).
20. [video-text-remover](https://github.com/hjunior29/video-text-remover).
21. [video-text-eraser](https://github.com/sueun-dev/video-text-eraser).
22. [IOPaint](https://github.com/Sanster/IOPaint).
23. [Inpaint Anything](https://github.com/geekyutao/Inpaint-Anything).
24. [Track Anything](https://github.com/gaomingqi/Track-Anything) and
    [memory issue #4](https://github.com/gaomingqi/Track-Anything/issues/4).

### Academic work, benchmarks, and model projects

25. [LaMa WACV 2022](https://openaccess.thecvf.com/content/WACV2022/html/Suvorov_Resolution-Robust_Large_Mask_Inpainting_With_Fourier_Convolutions_WACV_2022_paper.html).
26. [ProPainter](https://github.com/sczhou/ProPainter) and
    [issue #76](https://github.com/sczhou/ProPainter/issues/76).
27. [E2FGVI](https://github.com/MCG-NKU/E2FGVI) and
    [issue #49](https://github.com/MCG-NKU/E2FGVI/issues/49).
28. [VideoPainter](https://github.com/TencentARC/VideoPainter).
29. [D2DF](https://github.com/bigD233/D2DF).
30. [ROSE](https://github.com/Kunbyte-AI/ROSE) and
    [issue #15](https://github.com/Kunbyte-AI/ROSE/issues/15).
31. [CLEAR paper](https://arxiv.org/abs/2603.21901),
    [code](https://github.com/silent-commit/CLEAR),
    [model card](https://huggingface.co/charlesw09/CLEAR-mask-free-video-subtitle-removal),
    [issue #2](https://github.com/silent-commit/CLEAR/issues/2), and
    [issue #5](https://github.com/silent-commit/CLEAR/issues/5).
32. [SEDiT](https://arxiv.org/abs/2605.14894).
33. [PROVE](https://arxiv.org/abs/2605.14534) and
    [project site](https://xiaomi-research.github.io/prove/).
34. [VSR-Bench-400](https://openreview.net/pdf?id=MIRtxjuZF6).
35. [EffectErase](https://github.com/FudanCVL/EffectErase) and
    [dataset](https://huggingface.co/datasets/FudanCVL/EffectErase/tree/main).
36. [Object-WIPER, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Kushwaha_Object-WIPER_Training-Free_Object_and_Associated_Effect_Removal_in_Videos_CVPR_2026_paper.html).
37. [BeyondMasks](https://yigitekin.github.io/BeyondMasks/).
38. [ViTEX benchmark](https://vitex-bench.github.io/).
39. [UnFlow](https://ojs.aaai.org/index.php/AAAI/article/view/12276) and
    [reference implementation](https://github.com/simonmeister/UnFlow).

### Standards and official engineering guidance

40. [ITU-R BT.2100-3](https://www.itu.int/rec/R-REC-BT.2100-3-202502-I/en).
41. [ITU-R BT.2446-1](https://www.itu.int/pub/R-REP-BT.2446-1-2021).
42. [FFmpeg documentation](https://ffmpeg.org/ffmpeg.html) and
    [ffprobe documentation](https://ffmpeg.org/ffprobe.html).
43. [Matroska specification, RFC 9559](https://www.rfc-editor.org/rfc/rfc9559.html).
44. [OpenCV DIS optical flow](https://docs.opencv.org/5.0/main_modules/classcv_1_1DISOpticalFlow.html).
45. [Microsoft UIA control element guidance](https://learn.microsoft.com/en-us/accessibility-tools-docs/items/wpf/control_iscontrolelement).
46. [Microsoft accessibility testing](https://learn.microsoft.com/en-us/windows/apps/design/accessibility/accessibility-testing).
47. [Microsoft UIA server-side providers](https://learn.microsoft.com/en-us/windows/win32/winauto/uiauto-serversideprovider).
48. [WCAG 2.2](https://www.w3.org/TR/WCAG22/).
49. [Unicode UAX 29](https://unicode.org/reports/tr29/) and
    [BCP 47, RFC 5646](https://www.rfc-editor.org/info/rfc5646).
50. [IMSC 1.3](https://www.w3.org/TR/ttml-imsc1.3/).
51. [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).
52. [Windows SetThreadExecutionState](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate).

### Security advisories and dependency releases

53. [PyTorch GHSA-63cw-57p8-fm3p](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p)
    and [GHSA-53q9-r3pm-6pq6](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6).
54. [Python 3.13.15](https://www.python.org/downloads/release/python-31315/),
    [Python security release announcement](https://blog.python.org/2026/08/python-31214-31116-31021/),
    and [CVE-2026-11940](https://nvd.nist.gov/vuln/detail/CVE-2026-11940).
55. [ONNX Runtime 1.29.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.29.0),
    [CUDA provider compatibility](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html),
    and [DirectML provider status](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html).
56. [OpenCV 5.0.0](https://github.com/opencv/opencv/releases/tag/5.0.0) and
    [CVE-2025-53644](https://nvd.nist.gov/vuln/detail/CVE-2025-53644).
57. [Pillow 12.3.0](https://pillow.readthedocs.io/en/stable/releasenotes/12.3.0.html)
    and [CVE-2026-54058](https://nvd.nist.gov/vuln/detail/CVE-2026-54058).
58. [FFmpeg downloads](https://ffmpeg.org/download.html),
    [CVE-2026-58049](https://nvd.nist.gov/vuln/detail/CVE-2026-58049), and
    [fix commit](https://github.com/FFmpeg/FFmpeg/commit/f8d7795dcca36a4dd412e89cbd83e3dfec1e0d81).
59. [PyInstaller 6.22.2](https://github.com/pyinstaller/pyinstaller/releases/tag/v6.22.2),
    [GHSA-9fxf-4qw3-ghmr](https://github.com/pyinstaller/pyinstaller/security/advisories/GHSA-9fxf-4qw3-ghmr),
    and [CVE-2025-59042 advisory](https://github.com/pyinstaller/pyinstaller/security/advisories/GHSA-p2xp-xx3r-mffc).
60. [NSIS 3.12 release notes](https://nsis.sourceforge.io/Docs/AppendixF.html#v3.12).
61. [Requests advisory](https://github.com/psf/requests/security/advisories/GHSA-gc5v-m9x4-r6x2),
    [urllib3 advisory](https://github.com/urllib3/urllib3/security/advisories/GHSA-qccp-gfcp-xxvc),
    and [idna advisory](https://github.com/kjd/idna/security/advisories/GHSA-65pc-fj4g-8rjx).
62. [RapidOCR 3.9.2](https://pypi.org/project/rapidocr/),
    [RapidOCR model hashes](https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/default_models.yaml),
    [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), and
    [Paddle installation matrix](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/index_cn.html).

### Professional tools, community reports, and discovery lists

63. [Adobe After Effects plans](https://www.adobe.com/products/aftereffects/plans.html)
    and [Content-Aware Fill](https://helpx.adobe.com/after-effects/using/content-aware-fill.html).
64. [DaVinci Resolve Studio](https://www.blackmagicdesign.com/products/davinciresolve/studio).
65. [Mocha Pro](https://borisfx.com/products/mocha-pro/) and
    [2026.5 documentation](https://borisfx.com/documentation/mocha/2026.5.0/).
66. [NukeX](https://www.foundry.com/products/nuke-family/nukex),
    [CopyCat](https://learn.foundry.com/nuke/content/reference_guide/air_nodes/copycat.html),
    and [SmartVector](https://learn.foundry.com/nuke/content/comp_environment/smartpaint/smartvector.html).
67. [Runway pricing](https://runway.com/pricing) and
    [Edit Studio](https://help.runwayml.com/hc/en-us/articles/51683104370451-Creating-with-Edit-Studio).
68. [Filmora AI Object Remover](https://filmora.wondershare.com/guide/ai-video-object-remover.html).
69. [EchoSubs](https://www.echosubs.com/),
    [Subtitle Remover](https://www.subtitle-remover.com/), and
    [Media.io](https://www.media.io/video-watermark-remover.html).
70. [Interra BATON](https://www.interrasystems.com/file-based-qc.php),
    [QScan](https://qscan.io/features), and
    [Telestream Qualify](https://docs.telestream.dev/docs/qualify-user-guide).
71. [Stable Diffusion community discussion](https://www.reddit.com/r/StableDiffusion/comments/1quw6ve/reliable_video_object_removal_inpainting_model/),
    [Adobe occlusion discussion](https://community.adobe.com/t5/after-effects-discussions/content-aware-fill-issues-when-anything-passes-in-front-of-the-mask/td-p/14046912),
    [Resolve discussion](https://forum.blackmagicdesign.com/viewtopic.php?f=33&hilit=magic+mask&t=199707),
    and [Topaz discussion](https://community.topazlabs.com/t/topaz-video-ai-7-2-0-1b/94506?page=2).
72. [Awesome Image Inpainting](https://github.com/zengyh1900/Awesome-Image-Inpainting),
    [video-inpainting topic](https://github.com/topics/video-inpainting),
    [awesome-video](https://github.com/sitkevij/awesome-video),
    [awesome-ffmpeg](https://github.com/brandonhimpfen/awesome-ffmpeg), and
    [awesome-ocr](https://github.com/kba/awesome-ocr).

## Open Questions

1. **VLM transport details:** PaddleOCRVL is given the configured server URL and a
   temporary frame PNG, but the dependency owns the final request body. Capture a
   loopback request in a test fixture to document exactly what leaves the process
   before finalizing user-facing privacy text
   ([backend/ocr_vlm.py:490](backend/ocr_vlm.py#L490)).
2. **Real accessibility behavior:** Does the packaged app expose collapsed Advanced
   controls to Narrator or NVDA, and are canvas controls operable without a mouse?
   The one observed tree is **Needs live validation**
   ([backend/a11y.py](backend/a11y.py),
   [Microsoft accessibility testing](https://learn.microsoft.com/en-us/windows/apps/design/accessibility/accessibility-testing)).
3. **HDR source policy:** Which combinations of PQ, HLG, mastering metadata, and
   missing or conflicting transfer tags should fail closed rather than use an
   operator-selected override
   ([ITU-R BT.2100-3](https://www.itu.int/rec/R-REC-BT.2100-3-202502-I/en))?
4. **VFR repair policy:** When source timestamps are missing, repeated, or
   non-monotonic, should VSR preserve, repair, or reject each class? Every repair
   needs a sidecar record
   ([ffprobe documentation](https://ffmpeg.org/ffprobe.html),
   [Matroska RFC 9559](https://www.rfc-editor.org/rfc/rfc9559.html)).
5. **Quality threshold calibration:** What false-positive rate is acceptable for
   intentional camera motion, cuts, lighting changes, and animated backgrounds?
   RM-304 needs a licensed calibration corpus, not only synthetic clips
   ([PROVE](https://arxiv.org/abs/2605.14534)).
6. **Polygon compatibility:** Which current OCR engines return true quadrilaterals
   reliably, and which serialized plan consumers require a backward-compatible box
   projection ([backend/detection.py](backend/detection.py),
   [backend/track_plan.py](backend/track_plan.py))?
7. **Embedded FFmpeg floor mapping:** OpenCV exposes library versions rather than a
   clean FFmpeg product version. RM-305 needs a maintained, cited mapping or a
   wheel-build provenance rule before it can fail a release
   ([backend/security_checks.py](backend/security_checks.py)).
8. **Tutorial hosting:** Discussion #8 establishes the content request but not the
   preferred host. The implementation should keep a versioned transcript and
   screenshots in the repository even if the video is hosted elsewhere
   ([discussion #8](https://github.com/SysAdminDoc/VideoSubtitleRemover/discussions/8#discussioncomment-17848058)).
