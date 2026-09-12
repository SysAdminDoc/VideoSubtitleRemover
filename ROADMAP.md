# Video Subtitle Remover Pro Roadmap

Actionable work only. Historical and completed roadmap material is archived in CHANGELOG.md; blocked work is kept in Roadmap_Blocked.md.

## Remaining work

Source research: RESEARCH.md, dated 2026-09-04. P0 safety work is ordered first.

Rejected on purpose (do not re-file): production ROSE/EraserDiT/VOID adapters, Qt rewrite,
REST/Gradio, Mac/ROCm, re-adding GitHub Actions, Sigstore/attestations for SmartScreen,
winget/Store submission, a "Copy CLI command" button (already at `gui/widgets.py:65`),
videowipe-style review tracks (already `backend/track_plan.py`), mask-ROI inference
(already `backend/inpainters/lama.py:510-568`), wipe compare (already
`gui/preview_controller.py:683-860`). Reasons in RESEARCH.md Rejected Ideas.

CLEAR now has released code and weights. Keep it research-only until redistribution
rights, consumer GPU cost, large-font quality, and color drift pass written gates.

The build host now carries an RTX 4070 SUPER (12 GB, driver 610.88). RM-322 re-triages
`Roadmap_Blocked.md` against it; do not re-file blocked GPU items separately.

Implementer index. Drain in this order. Blocked work is in Roadmap_Blocked.md.

## Research-Driven Additions

Items RM-353 onward were added by the 2026-09-04 research pass. Both open GitHub issues are in the
P0 block; neither is a defect in the code on main, and both are reachable from the packaged build.

### P0

### P1 (2026-09-04)

### P2 (2026-09-04)

- [ ] P2 | RM-366: Carry only the CUDA libraries the product loads, not all of torch
  Why: The NVIDIA portable ZIP is 2.119 GiB, past GitHub's 2 GiB asset ceiling, so v3.41.0 had to publish it as two parts a user joins by hand. The bundle is that size because it carries the whole PyTorch CUDA wheel purely as a carrier for the runtime ONNX Runtime loads cuBLAS and cuDNN out of.
  Evidence: measured in `dist/nvidia` on 2026-09-05: `torch_cpu.dll` 291.8 MiB, `torch_cuda.dll` 403.1 MiB, `cusolver64_12.dll` 120.6 MiB, `cusolverMg64_12.dll` 90.9 MiB and `nvrtc64_130_0.alt.dll` 86.8 MiB, none of which the ONNX inference path calls. Hiding the last three still passed `--frozen-provider-smoke` with `CUDAExecutionProvider` active and `fellBack: false`, but the same three hidden from `venv-nvidia` made `import torch` fail in `_load_dll_libraries`, because torch loads its CUDA set eagerly; `cufft64_12.dll` is required outright and `preload_dlls` names it on stderr. So the payload cannot be trimmed while torch is the carrier. The `nvidia-cublas-cu13` and `nvidia-cudnn-cu13` wheels ship those libraries without torch.
  Touches: `dependency_profiles.json`, `VideoSubtitleRemoverPro.spec`, `backend/onnxruntime_cuda.py`, `backend/release_staging.py`, README download matrix.
  Acceptance: The NVIDIA lane resolves the CUDA runtime from the `nvidia-*-cu13` wheels rather than from torch, or states in the profile manifest why it cannot. The staged portable ZIP is under 2 GiB, so `split_asset_for_upload` returns no parts and the release publishes one file per lane. The frozen provider smoke still reports `CUDAExecutionProvider` with `fellBack: false`, the opt-in GPU lane still passes, and any torch-dependent adapter that stops working is named in the manifest rather than discovered later.
  Complexity: L

- [ ] P2 | RM-360: Give a clip a pre-run verdict from a sampled dry run
  Why: The category's loudest and best-evidenced complaint is that automatic removal is unpredictable, so people burn long renders on clips that were never going to work. This product already computes everything needed to predict that and only reports it after the render finishes. Test cleanup previews one chosen timestamp, which answers a different question.
  Evidence: `backend/quality.py:766` computes `residual_text_score` per ROI frame and `backend/_quality_mixin.py:505-514` emits review spans, both post-encode; RM-300 made Test cleanup a before/current/after window around one preview timestamp, not an estimate over the clip; RM-325 already re-runs the detector over the repaired region, which is the per-sample verdict this needs; https://reccloud.com/ai-burned-in-subtitle-removal-large-video-archives.html and https://forum.videohelp.com/threads/418726-Is-there-a-way-to-remove-hardcoded-subtitles-without-cropping document the archive-scale unpredictability; no surveyed competitor offers a pre-run estimate.
  Touches: `backend/quality.py`, `backend/_quality_mixin.py`, `backend/cli.py`, `gui/quality_controller.py`, `gui/processing_controller.py`, `backend/batch_report.py`, `locale/vsr.pot`, tests.
  Acceptance: A dry-run mode samples N segments spread across the clip, repairs only those, scores them with the existing ROI and residual metrics plus RM-325 re-detection, and reports a verdict with its worst sampled segment, its timestamp, and the wall-clock estimate for the full run, in both the GUI and the CLI. Queueing a batch can run the dry pass first and mark items whose verdict is below a stated threshold rather than processing them silently. A test asserts the dry pass on a clip whose reference corpus result is known produces a verdict consistent with the full run's, and that a clip engineered to fail is flagged before the full render.
  Complexity: M

- [ ] P2 | RM-363: Ship presets for known generative-video marks
  Why: The fastest-growing demand in this space is removing the visible marks that video generators burn into their own output, and those marks have fixed positions and shapes, so they need a preset rather than a hand-drawn region. The preset mechanism, the region model and the logo intent all already exist.
  Evidence: https://github.com/wiltodelta/remove-ai-watermarks reached 5,410 stars between 2026-03-25 and 2026-09-03 on exactly this, and its video surface names Sora, Veo, Seedance, Doubao, Dola, Hailuo AI and Kling AI marks; `backend/presets.py:21` already ships `BUILTIN_PRESETS` with a logo and watermark intent and `parse_intent` at `:253`; `backend/track_plan.py` already carries per-track geometry that a preset could seed; the alpha-solve work for fully static translucent marks is RM-329, which this depends on for quality on semi-transparent marks.
  Touches: `backend/presets.py`, `backend/config.py`, `gui/layout_build.py`, `backend/cli.py`, `locale/vsr.pot`, `tests/clips/`, README, tests.
  Acceptance: A preset per supported generator seeds the region, intent and detection settings from the mark's documented placement and is selectable in the GUI and by name from the CLI. Each preset records the generator, the resolutions it was measured against, and the date, and a preset whose geometry has not been verified against a sample is not shipped. A synthetic clip per preset, composited with the mark at its stated placement, is repaired within the same residual threshold the reference corpus uses. Presets seed a region and never bypass user confirmation.
  Complexity: M

### P3 (2026-09-04)

- [ ] P3 | RM-365: Give pytest a checked-in configuration
  Why: The suite's entry point, its timeout and its markers live only in a batch file. A run started any other way collects from the wrong root, and a single hung test takes all 1,872 with it because nothing bounds one.
  Evidence: there is no `pytest.ini`, `setup.cfg`, `tox.ini`, or `conftest.py` at the repository root or in `tests/`; `pyproject.toml` carries only `[tool.ruff]`; `build_exe.bat` hardcodes `python -m pytest tests -q` and its inline comment explains that pytest is mandatory because twelve modules use bare module-level test functions that `unittest discover` collects zero tests from, a constraint recorded in a comment rather than in configuration; `tests/archive/conftest.py` sets `collect_ignore_glob = ["*.py"]` and is the only pytest configuration in the repository.
  Touches: `pyproject.toml`, `build_exe.bat`, README testing section.
  Acceptance: `[tool.pytest.ini_options]` in `pyproject.toml` sets `testpaths = ["tests"]`, a per-test timeout with the thread method, and any markers the suite uses, and records why pytest rather than unittest is required. A bare `python -m pytest` from the repository root collects the same test count as `python -m pytest tests -q`. A deliberately hung test fails on the timeout instead of stalling the run, and the suite completes. The declared timeout dependency is pinned in the build script's tooling install.
  Complexity: S

### P1

- [ ] P2 | RM-351: Benchmark a provider lane on footage the numbers can speak for
  Why: The committed provider evidence runs on a 160x96 sixteen-frame fixture, where the CUDA lane measures slightly slower than CPU because a fixed manual region does its work in CPU numpy rather than in ONNX inference. That is an honest measurement of the wrong thing: it cannot tell a user whether to download the CUDA bundle.
  Evidence: `docs/benchmarks/provider-benchmark-cpu.json` and `-nvidia.json` record 5.86 against 5.55 FPS cold on `tests/clips/static_dialogue.mkv`; the README says plainly that this is not a GPU recommendation. RM-342 already tracks acquiring redistributable real footage from https://huggingface.co/datasets/cyberagent/OTR (CC-BY-4.0) and https://github.com/YigitEkin/BeyondMasks (CC BY 4.0).
  Touches: `backend/provider_benchmark.py`, `docs/benchmarks/`, `tests/test_provider_benchmark.py`, README.
  Acceptance: The published evidence includes at least one clip at 720p or above, of at least ten seconds, driven with automatic detection so the inference paths actually run, on both the CPU and CUDA lanes. The README table reports that clip alongside the fixture and states which one a download decision should be based on. Clip licences are recorded beside the evidence.
  Complexity: M

### P2

- [ ] P2 | RM-352: Reduce the frozen blind-exception budget
  Why: RM-326 froze the count and fixed the sites it named, but 537 BLE001 and 127 S110 findings remain. Each is a place a failure is caught and not re-raised, so each is a place the product may be hiding something. The budget stops the number growing; it does not bring it down.
  Evidence: `tests/test_exception_budget.py` records the frozen counts measured on 2026-08-28 and fails if either rises; `ruff check backend gui scripts tools VideoSubtitleRemover.py --select BLE001,S110 --statistics` reports the current split. RM-326's own acceptance asked for an inline `noqa` and a one-line reason on every remaining site; 664 of those reasons cannot be written honestly in one pass, and writing them mechanically would produce exactly the mute-button pattern the item exists to remove, so the count was frozen instead and the reduction left here.
  Touches: `gui/app.py`, `gui/widgets.py`, `gui/processing_controller.py`, `gui/preview_controller.py`, `backend/io.py`, `backend/subprocess_policy.py`, `tests/test_exception_budget.py`, `pyproject.toml`.
  Acceptance: The budget in `tests/test_exception_budget.py` drops in every pass and never rises. Each removed site either narrows the clause to the exceptions that can actually occur, or logs and reports the failure rather than discarding it; a site that genuinely must swallow carries a `noqa` naming the reason. When the count reaches zero, `BLE001` and `S110` move into `pyproject.toml`'s `select` and this test is deleted.
  Complexity: L

- [ ] P2 | RM-327: Ship one real translation catalog and close the extraction hole
  Why: The i18n machinery is complete and gated at 90 percent coverage, but the only catalog is a hidden pseudo-locale, so the language picker offers System and English and the RTL feature has nothing to mirror; separately, strings routed through a local variable never reach the template at all.
  Evidence: `locale/` contains only `vsr.pot` (957 msgids, verified by `scripts/i18n_catalogs.py check` on 2026-08-27) and `qps-Ploc`, which `backend/i18n.py:160-180` hides unless `VSR_PSEUDO_LOCALE=1`; `gui/layout_build.py:1532-1535` therefore renders a two-entry picker; `gui/layout_build.py:1589-1590` advertises Arabic, Hebrew, Persian, and Urdu with no catalogs; `scripts/i18n_catalogs.py:99-121` inspects only call-site literals, so the 24 guidance strings assigned to locals at `gui/app.py:1259-1310`, the confirm and cancel labels at `gui/app.py:395-396`, `:2383-2384`, `:2770-2771`, `:2853-2854`, and the 11 OCR engine names at `gui/app.py:284-296` are absent from `vsr.pot`; https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/6 was filed in Chinese and the upstream user base is predominantly Chinese-speaking.
  Touches: `scripts/i18n_catalogs.py`, `locale/`, `gui/app.py`, `gui/layout_build.py`, `backend/i18n.py`, `tests/test_i18n_catalogs.py`, README translation section.
  Acceptance: The extraction lint detects strings assigned to a local and later passed to `tr`, `values=`, `confirm_label`, or `cancel_label`, and a test fails on a planted example. `vsr.pot` regenerates with the previously missing strings included. At least one real catalog ships above the 90 percent gate, chosen as Simplified Chinese, and appears in the picker on a stock build. A release probe renders the main window, settings, and one modal in that locale at 100 and 200 percent text scale without clipping.
  Complexity: M

- [ ] P2 | RM-329: Remove fully static semi-transparent overlays by solving for alpha
  Why: A watermark present in every frame has zero temporal exposure, so translucency unmixing never runs and the whole region falls to `cv2.inpaint`, which is exactly the fastest-growing request category and the case the current pipeline handles worst.
  Evidence: `backend/inpainters/_common.py:1303` gates `_unmix_translucent_regions` on `has_exposure.any()` and `:1305-1310` sends zero-exposure pixels to `_cv2_inpaint` first; `backend/inpainters/_common.py:277-296` already implements the closed-form two-endpoint solve for the exposed case; upstream 2026 demand is https://github.com/YaoFANGUK/video-subtitle-remover/issues/179, https://github.com/YaoFANGUK/video-subtitle-remover/issues/220, https://github.com/YaoFANGUK/video-subtitle-remover/issues/232, https://github.com/YaoFANGUK/video-subtitle-remover/issues/236; the multi-frame formulation is https://watermark-cvpr17.github.io/ (CVPR 2017), which jointly estimates the overlay, its alpha, and the clean backgrounds from many frames sharing one overlay, and the only clean community result reported for a synthetic AI watermark reversed the alpha blend algebraically rather than inpainting.
  Touches: `backend/inpainters/_common.py`, new overlay-estimation module, `backend/config.py`, `backend/cli.py`, `gui/layout_build.py`, reference clips, tests.
  Acceptance: A new opt-in mode estimates a per-pixel alpha and overlay color for a fixed region across N sampled frames with differing backgrounds, un-composites it, and falls back to the existing path with a recorded reason when the fit residual exceeds a stated tolerance. A synthetic reference clip that composites a known semi-transparent overlay at a known alpha is recovered to within a stated PSNR of the pre-composite source, and the same clip through the current static path is measurably worse. The estimated alpha and residual are recorded in provenance.
  Complexity: L

- [ ] P2 | RM-330: Measure whether stroke-level masks beat the current dilated polygons
  Why: The published evidence says mask shape, not the inpainter, sets removal quality, and the strongest permissively licensed stroke-level text segmenter is available while the product currently masks with dilated OCR polygons.
  Evidence: https://arxiv.org/abs/2511.22499 finds character-level masks beat minimal bounding-region masks and that quality is highly sensitive to mask profile; https://github.com/ymy-k/Hi-SAM is Apache-2.0 with public weights and reports 88.96 fgIoU on TextSeg and 84.86 on Total-Text with a stroke/word/line/paragraph hierarchy; current masking is polygon plus Lab-contrast dilation in `backend/detection_geometry.py` and `backend/mask_corrections.py`, with residue on outlined and shadowed text reported at https://github.com/YaoFANGUK/video-subtitle-remover/issues/80 and https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/6.
  Touches: `backend/segmentation.py`, `backend/detection_geometry.py`, `backend/mask_corrections.py`, benchmark tooling, `tests/clips/`, README algorithm section.
  Acceptance: A repeatable comparison runs the existing mask generator and a stroke-level generator over the reference clips plus the `shadow_outline` and `thick_font` cases, and records ROI PSNR, ROI SSIM, residual text score, and the RM-325 detector recall for both. The result is written to RESEARCH.md with the measured numbers, and the stroke path either becomes an opt-in mode with its licence and download recorded in `adapter_manifest.py`, or is rejected in RESEARCH.md with the measurement that killed it.
  Complexity: L

- [ ] P2 | RM-331: Decide PP-OCRv6 against this product's own subtitle corpus
  Why: The default is deliberately held at PP-OCRv5 mobile because upstream metrics are not comparable, but the deciding measurement has never been run, and the vendor reports a detection gain in exactly the half of OCR this product depends on.
  Evidence: `dependency_profiles.json` records the intentional exception keeping PP-OCRv5 mobile default with v6 tiers behind `--paddleocr-variant tiny|small|medium`; https://github.com/PaddlePaddle/PaddleOCR/releases/tag/v3.7.0 reports PP-OCRv6 medium at 34.5M parameters with +4.6 percent detection and +5.1 percent recognition over PP-OCRv5 server; rapidocr made PP-OCRv6 small its default in v3.9.0 on 2026-06-23 while `dependency_profiles.json` pins `rapidocr==3.9.2`, so the shipped default OCR may already be v6 and the paddle lane may be the one lagging.
  Touches: `backend/ocr_benchmark.py`, `backend/detection.py`, `backend/paddle_compat.py`, `tests/clips/`, `dependency_profiles.json`, README OCR section.
  Acceptance: `backend/ocr_benchmark.py` runs PP-OCRv5 mobile, PP-OCRv6 tiny, small, and medium, and the current rapidocr default over the reference clips and records detection precision, recall, F-score, box geometry fidelity, per-frame latency, and peak memory for each. The default is either changed with the measurement recorded in the manifest exception, or explicitly kept with the same measurement as the reason. The report also states which OCR generation the shipped rapidocr default actually loads.
  Complexity: M

- [ ] P2 | RM-335: Re-decide code signing on corrected SmartScreen evidence
  Why: The recorded reason for parking signing is that reputation is per-file-hash and resets each release even for signed publishers, which Microsoft's own documentation contradicts, and the cheapest managed option now costs ten dollars a month and admits individual developers.
  Evidence: `Roadmap_Blocked.md:202` records the per-file-hash reasoning; https://learn.microsoft.com/en-us/windows/security/operating-system-security/virus-and-threat-protection/microsoft-defender-smartscreen/ (updated 2026-04-25) states reputation checks apply to "a URL, a file, an app, or a certificate", and https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/smartscreen-reputation (updated 2026-08-17) states that unsigned files "must build reputation anew with every update" while signed reputation accumulates against the publisher certificate; the same page records that EV certificates no longer grant an instant bypass, so buy OV rather than paying the EV premium; https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/distribution-feature-status notes Smart App Control blocks unsigned executables outright rather than warning, which raises this above a cosmetic concern; the Azure Retail Prices API returned Basic Account at 9.99 USD per month with 5,000 included signatures on 2026-08-27; https://learn.microsoft.com/en-us/azure/artifact-signing/quickstart documents an individual-developer path for the United States and Canada with Microsoft Verified ID validation taking 1 to 20 business days; https://signpath.org/terms remains free for OSI-licensed projects but signs under SignPath's own certificate; https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/2 is the false-positive report this addresses.
  Touches: `Roadmap_Blocked.md`, `build_exe.bat`, `installer/vsr.nsi`, release verification, README download section.
  Acceptance: `Roadmap_Blocked.md` is corrected so it no longer states that signing does not help, and records the chosen path with its cost, eligibility, and the reason. If signing proceeds, `build_exe.bat` signs the installer, the portable payload executable, and the launcher; release verification asserts a valid signature and records the subject and thumbprint; and the README stops describing the artifacts as intentionally unsigned. If it does not proceed, the interim measures are implemented instead: bootloader rebuilt from source, UPX confirmed off, and each release submitted to https://www.microsoft.com/wdsi/filesubmission with the submission recorded in release evidence.
  Complexity: M

- [ ] P2 | RM-337: Split the README and remove the stale and mixed-ending content
  Why: The README is 104 KB, buries user install steps under release engineering, inlines generated reference tables, contradicts its own privacy promise, and carries mixed line endings that the working notes treat as a permanent editing hazard because the repository has no `.gitattributes`.
  Evidence: `README.md` is 1,649 lines with 820 CRLF and 829 LF-only lines measured on 2026-08-27 and no `.gitattributes` exists; `README.md:924-1120` and `:1174-1308` are about 330 lines of generated CLI and config reference; `README.md:119-363` mixes user install with release staging, SBOM derivation, reproducibility, and CVE triage; `README.md:1543` still reads `models/ # AI model weights (auto-downloaded)`, contradicting `README.md:26-27`; the six-anchor nav at `README.md:16` covers a document with more than thirty headings.
  Touches: `README.md`, `docs/`, `.gitattributes`, `scripts/generate_cli_reference.py`, `tests/test_documentation_drift.py`.
  Acceptance: A `.gitattributes` normalises text line endings and one deliberate commit does the conversion with no other change. The generated CLI and config references move to `docs/` with links from the README, the release-engineering and CVE-rationale sections move to `docs/`, and the README keeps overview, requirements, install, usage, algorithm comparison, and troubleshooting. The stale `models/` line is corrected. The nav reflects the remaining headings. The drift test covers the moved documents.
  Complexity: M

- [ ] P2 | RM-339: Show where the residue is, not just that there is some
  Why: The pipeline computes a per-frame residual score inside the mask and turns it into review spans, but the preview can only overlay detection boxes, so a user cannot see which pixels the gate is objecting to.
  Evidence: `backend/quality.py:766` computes `residual_text_score` per ROI frame and `backend/_quality_mixin.py:505-514` emits residual review spans; `gui/preview_controller.py:1491-1621` overlays detection boxes only; grepping `gui/` on 2026-08-27 found no difference view, heatmap, loupe, or onion-skin; no competitor surveyed exposes any per-frame quality signal to the user, and Blackmagic's candour about failure cases is the closest anything comes.
  Touches: `gui/preview_controller.py`, `gui/quality_controller.py`, `backend/quality.py`, `backend/_quality_mixin.py`, `locale/vsr.pot`.
  Acceptance: The preview offers a residue overlay that shades the repaired region by its per-pixel contribution to the residual score, plus a difference view against the source, both toggleable and both honoring the reduced-motion and high-contrast settings. Frames carrying a residual review span are reachable from the quality view in score order rather than only as a single worst frame. A release probe renders both overlays without clipping at 100 and 200 percent text scale.
  Complexity: M

- [ ] P2 | RM-345: Export extracted subtitles as positioned ASS
  Why: The product reads ASS for re-burn but can only write SRT and WebVTT, and FFmpeg's encoders for both drop position, so the extract half of the extract-translate-reburn workflow throws away the geometry the detector already produced.
  Evidence: `backend/cli.py:831` accepts an `.srt` or `.ass` file for `--restyle` while `--export-srt` and the WebVTT path in `backend/_srt_mixin.py` are the only writers; grepping `backend/` on 2026-08-27 found no ASS writer and no `LayoutRes` handling; FFmpeg's `srtenc.c` carries a standing TODO that subtitle position side data is never emitted and leaks `{\an8}` into the text, and `webvttenc.c` sets the colour, font, size, alignment, and move callbacks to NULL; ASS is the only text format Matroska carries losslessly per draft-ietf-cellar-codec-20 (2026-08-14); libass 0.17.0 added `LayoutResX`/`LayoutResY`, which separates the coordinate space detections are authored in from the render canvas; https://pypi.org/project/pysubs2/ 1.9.0 (2026-08-16) reads and writes ASS, SRT, WebVTT, and TTML from an ASS-shaped internal model.
  Touches: `backend/_srt_mixin.py`, `backend/cli.py`, `gui/layout_build.py`, `backend/config.py`, `locale/vsr.pot`, tests, README.
  Acceptance: A new export target writes ASS with `LayoutResX`/`LayoutResY` set from the source frame size, one `\pos` per cue derived from the tracked detection box, and alignment from the box position, alongside the existing SRT and WebVTT exports. Round-tripping the exported file through `--restyle` reproduces text at the original coordinates within a stated pixel tolerance on a reference clip. The SRT and WebVTT exports state in the UI and CLI help that they discard position.
  Complexity: M

- [x] P2 | RM-340: Fix multi-monitor DPI and dialog sizing
  Why: The launcher claims Per-Monitor V2 awareness but requests V1, and every dialog is sized from the primary monitor with hardcoded taskbar fractions, so dialogs mis-size on a secondary display.
  Evidence: `VideoSubtitleRemover.py:337-346` comments "Per-Monitor V2 first" and calls `windll.shcore.SetProcessDpiAwareness(2)`, which is `PROCESS_PER_MONITOR_DPI_AWARE`, while V2 requires `SetProcessDpiAwarenessContext(-4)`; `gui/dialog_layout.py:31-48` sizes from `root.winfo_screenwidth()` and `winfo_screenheight()` with 0.96 and 0.90 factors, and no call to `SPI_GETWORKAREA`, `MonitorFromWindow`, or `GetMonitorInfo` exists anywhere in the repository; `gui/app.py:153` sets a fixed `minsize(980, 720)` that ignores `text_scale_percent`; `gui/app.py:419` persists geometry but never `wm_state`, so a maximized window does not return maximized.
  Touches: `VideoSubtitleRemover.py`, `gui/dialog_layout.py`, `gui/app.py`, `gui/layout_responsive.py`, `gui/release_probe.py`.
  Acceptance: The process requests `DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2` and falls back through V1 and system awareness, with the comment matching the call. Dialogs size from the work area of the monitor that actually hosts the parent window. `minsize` scales with the text-scale setting so a 200 percent layout still fits a 1366x768 display. Maximized state is persisted and restored. The release probe covers a dialog opened on a secondary monitor of a different size.
  Complexity: M

- [ ] P2 | RM-342: Publish a reproducible comparison against the alternatives
  Why: There is no independent benchmark anywhere in this category, every visible comparison is vendor-owned, this project has the most complete measurement stack in the field and no third-party discussion at all, and the rights obstacle that blocked real benchmark media now has a licensed answer.
  Evidence: searches across Reddit, Hacker News, VideoHelp, Doom9, the 215-issue upstream tracker, and Chinese developer forums on 2026-08-27 returned zero mentions of this project outside its own repository, against 65 stars versus upstream's 12,575; `Roadmap_Blocked.md` parks replacing the placeholder benchmark clips on rights verification, and the corpus is ten synthetic 160x96 sixteen-frame clips; https://huggingface.co/datasets/cyberagent/OTR is CC-BY-4.0 with 89.3K overlay-text pairs and https://github.com/YigitEkin/BeyondMasks is a CC BY 4.0 ECCV 2026 removal benchmark pushed 2026-08-22; the product already computes PSNR, SSIM, ROI metrics, temporal and mask-local scores, and colour drift.
  Touches: `backend/ocr_benchmark.py`, `backend/mask_free_benchmark.py`, `backend/static_logo_benchmark.py`, `backend/reference_corpus.py`, `tests/clips/`, `docs/edge_case_corpus.md`, README.
  Acceptance: A documented, rerunnable harness scores this product against at least the upstream project and one other open source remover on a redistributable clip set drawn from the licensed sources above, reporting ROI PSNR, ROI SSIM, residual text score, RM-325 detector recall, temporal flicker, wall-clock time, and peak VRAM, with the clip licences recorded. Results and the exact command land in `docs/`, and the placeholder synthetic clips are either replaced or kept only as fast regression fixtures with that role stated.
  Complexity: L

### P3

- [ ] P3 | RM-347: Detect and report loss of embedded closed captions
  Why: CEA-608 and CEA-708 captions ride inside the video bitstream rather than as a separate stream, so the raw-frame pipeline destroys them with no warning, which is an accessibility regression the stream-mapping work cannot catch.
  Evidence: grepping `backend/` on 2026-08-27 found no reference to `a53cc`, CEA-608, CEA-708, or A53 side data anywhere; the pipeline decodes to raw frames and re-encodes through `backend/_encode_mixin.py`, so `AV_FRAME_DATA_A53_CC` never reaches the encoder; FFmpeg's `libx264` and `libx265` `a53cc` option defaults to on but its documentation states "Only the mpeg2 and h264 decoders provide these", so an HEVC source loses captions even in the best case; `backend/io.py:766` already probes subtitle streams and would not see these.
  Touches: `backend/io.py`, `backend/_encode_mixin.py`, `backend/_finalize_mixin.py`, `backend/batch_report.py`, `gui/quality_controller.py`, README limitations.
  Acceptance: Ingest probing detects embedded CEA-608 or CEA-708 data in the source and records its presence. When present, the run reports in the batch report, the quality view, and the CLI that embedded captions will not survive, and offers to extract them to a sidecar before processing. Where the source is H.264 or MPEG-2 and preservation is achievable, the captions are carried through and a test asserts the output still contains them. The README limitations section states the constraint.
  Complexity: M

- [ ] P3 | RM-344: Offer a light theme
  Why: Both shipped themes are dark, so a user who needs a light interface for daylight work or for a light-sensitivity reason has no option, and the project's own design rule asks for one where practical.
  Evidence: `gui/theme.py` defines only `apply_default_theme` at :167 and `apply_high_contrast_theme` at :115, both dark palettes; the token layer already abstracts every surface and text colour, and text tokens measure 7.78:1 to 17.23:1 so the palette is token-swappable rather than hardcoded.
  Touches: `gui/theme.py`, `gui/layout_build.py`, `gui/config.py`, `gui/release_probe.py`, `tests/test_gui_workflow_release.py`, `locale/vsr.pot`.
  Acceptance: A light palette ships as a third theme selectable beside the existing two, passes the RM-333 contrast gate on every composed token pair, and is exercised by the release probe alongside the default and high-contrast themes at 100 and 200 percent text scale. The selected theme persists across restarts.
  Complexity: M
