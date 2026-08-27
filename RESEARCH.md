# Research: Video Subtitle Remover Pro

Date: 2026-08-27. Replaces all prior research.

Repository state: v3.40.0 at [1f120d4](https://github.com/SysAdminDoc/VideoSubtitleRemover/commit/1f120d4), released 2026-08-27. Suite re-run locally on 2026-08-27 in the repo venv: 1587 passed, 1 skipped, 544 subtests, 184s.

Confidence labels:

- **Verified:** confirmed by a local probe, the v3.40.0 source, the tracker, or a primary vendor source read on 2026-08-27.
- **Corroborated:** supported by project evidence plus at least one independent primary source.
- **Needs live validation:** requires hardware, licensed media, or assistive technology not exercised in this pass.

## Executive Summary

Video Subtitle Remover Pro is a local-first Windows desktop and command-line tool for detecting, tracking, removing, exporting, translating, and reburning visible video text. RM-307 through RM-313 shipped between 2026-08-23 and 2026-08-27, so the execution-truth and reproducibility work from the previous research pass is done. The product is now, on evidence, the most complete open source remover in the field: it is the only one with polygon geometry, exact rational timing, HDR-aware repair, reviewable track plans, frozen mattes, per-job process isolation, a quality gate, and local release evidence. The upstream it forked from has been feature-frozen since 2026-04-08 and ships a CPU-only installer; the classical inpainting layer it depends on (STTN, LaMa, ProPainter, IOPaint) has not moved in over a year. Differentiation is now an engineering problem, not a modelling one.

The highest-value direction has shifted. The previous pass made every stage tell the truth about what it ran. This pass finds that the project cannot yet tell the truth about **how well** it ran, and that a class of users cannot run it at all. Both are fixable now that the build host has an RTX 4070 SUPER, which reactivates every GPU-gated entry in `Roadmap_Blocked.md`.

Priority opportunities:

1. **RM-317, non-ASCII path failure.** [Verified by local probe, 2026-08-27] `cv2.imwrite` returns False and `cv2.imread` returns None for any path containing non-ASCII characters under OpenCV 5.0.0.93 on Windows. `backend/safe_image.py` wraps `cv2.imread` and inherits the failure. Still-image cleanup (`backend/processor.py:2096-2102`), clean-reference plates (`backend/_clean_ref_mixin.py:195`), imported masks (`backend/_quality_mixin.py:109`), CLI image input (`backend/cli.py:625`) and frame-sequence export (`backend/io.py:2404`) all break for a user whose output path, filename, or Windows username contains CJK, Cyrillic, or accented Latin characters. Video decode is unaffected: `cv2.VideoCapture` uses the FFmpeg backend and handled a `中文/测试.mp4` path correctly in the same probe. This is upstream's longest-lived unfixed defect, reported five times across three years ([#71](https://github.com/YaoFANGUK/video-subtitle-remover/issues/71), [#111](https://github.com/YaoFANGUK/video-subtitle-remover/issues/111), [#174](https://github.com/YaoFANGUK/video-subtitle-remover/issues/174), [#248](https://github.com/YaoFANGUK/video-subtitle-remover/issues/248)), with the three-line community fix posted in 2024-12 and never merged.
2. **RM-318, the reference corpus does not apply the shipping quality gate.** [Verified by local run, 2026-08-27] `backend/reference_corpus.py:406-416` gates on a decoded frame hash plus PSNR and SSIM floors only. Running the corpus in the repo venv: **8 of 10 clips exceed `RESIDUAL_TEXT_SCORE_CEILING = 0.025`** (`backend/quality_gate.py:49`), `static_dialogue.mkv` by a factor of 32 at 0.808, every clip tags `Review` rather than passing, and all 10 report `passed: true`. The floors are self-referential snapshots of current output blessed at `value * (1 - 0.005)`, so the corpus can only detect a regression, never a quality failure. The product's own runtime gate would flag this material.
3. **RM-319, the NVIDIA lane is frozen on a channel that no longer receives torch releases.** [Verified against the PyTorch index, 2026-08-27] cu128 Windows cp313 wheels stop at `torch 2.11.0`; cu130 already carries `torch 2.13.0`. `dependency_profiles.json` pins the NVIDIA profile to torch 2.11.0 / torchvision 0.26.0 and records the reason correctly, but that pin now sits inside [CVE-2025-3000](https://github.com/advisories/GHSA-rrmf-rvhw-rf47) (`torch <= 2.12.1`, patched 2.13.0, GHSA severity low) and [CVE-2026-65918](https://nvd.nist.gov/vuln/detail/CVE-2026-65918) (torchvision through 0.28.0, CVSS 7.1) while the CPU and DirectML lanes have moved past the first. The manifest names "a coordinated ONNX Runtime CUDA-lane review" as the exit; both halves now exist (torch 2.13.0 cu130, onnxruntime-gpu 1.29.0 on CUDA 13) and the build host can verify them.
4. **RM-320, the embedded OpenCV FFmpeg is four major versions below the enforced floor.** [Verified by binary inspection, 2026-08-27] `venv/Lib/site-packages/cv2/opencv_videoio_ffmpeg500_64.dll` contains `Lavf61.7.100` and `Lavc61.19.100`, which is FFmpeg 7.1 (tagged 2024-09-29). `backend/security_checks.py:29` enforces `FFMPEG_SECURITY_FLOOR = (9, 0, 1)` on the external binary, and `OPENCV_FFMPEG_ADVISORY_RULES` is deliberately left empty pending "an advisory that maps a specific embedded build to an affected range" (`backend/security_checks.py:311-324`). That mapping is now available: OpenCV pins its 3rdparty FFmpeg to `n7.1`, and the ABI numbers confirm it. `cv2.VideoCapture` is still reachable at `backend/cli.py:637`, `backend/cli.py:1398`, `backend/io.py:736`, and `backend/inpainters_diffusion.py:655`, so untrusted media can reach a decoder that predates the entire 2026 CVE batch, including CVE-2026-8461 (MagicYUV heap OOB write from a plain file).
5. **RM-321, manual-region mode silently disables the headline algorithm.** [Verified] A static manual region yields `coverage == 0` for every masked pixel, so `backend/inpainters/_common.py:1290-1292` routes the whole band to `_cv2_inpaint` TELEA. TBE contributes nothing. The GUI meanwhile reports "Manual-only mask; automatic detection is off" in `Theme.SUCCESS` green (`gui/layout_helpers.py:177-181`). This is exactly the path [issue #7](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/7)'s reporter needed for static text overlays, and it is the path where LaMa should be routed instead.
6. **RM-322, there is no GPU verification lane at all.** [Verified] No test in the 1587-test suite is gated on CUDA, NVENC, or a provider probe; the only hardware skips are `shutil.which("ffmpeg")`. `Roadmap_Blocked.md` parks the `processor.py` frame-loop extraction, most of RM-248, and the D3D12 live encode check on "a GPU host", which now exists. RM-315 covers provider-labeled release artifacts and benchmarks; it does not cover repeatable correctness tests for provider selection, OOM recovery, and hardware encode.
7. **RM-325, the cheapest residue metric in the literature is one the product already has the model for.** [Corroborated] The standard scene-text-removal protocol runs a text detector over the *output* and reports recall: nonzero recall means text survived. VSR loads a detector for every job. `residual_text_score` (`backend/quality.py:99`) is a contrast heuristic that already produces review spans at runtime (`backend/_quality_mixin.py:505-514`), but nothing runs detection on the result. [PROVE](https://github.com/xiaomi-research/prove) (ACM MM 2026, Apache-2.0, [arXiv:2605.14534](https://arxiv.org/abs/2605.14534)) supplies the stronger localized metric when a heavier check is wanted.
8. **RM-327, i18n is a complete machine with nothing to run.** [Verified] `locale/vsr.pot` carries 957 msgids and the only catalog is the hidden `qps-Ploc` pseudo-locale, so `available_catalogs()` returns empty and the picker offers System and English (`gui/layout_build.py:1532-1535`). The RTL toggle advertises Arabic, Hebrew, Persian and Urdu (`gui/layout_build.py:1589-1590`) with zero catalogs to mirror. Separately the extraction lint only inspects call-site literals (`scripts/i18n_catalogs.py:99-121`), so strings routed through a local variable escape it: the 24 guidance strings at `gui/app.py:1259-1304`, seven confirm and cancel button labels, and eleven OCR engine names are all absent from the catalog.
9. **RM-329, the fastest-growing demand is the case temporal borrowing handles worst.** [Corroborated] 2026 tracker growth upstream is AI-generation watermarks: Doubao, Sora 2, generic corner marks ([#179](https://github.com/YaoFANGUK/video-subtitle-remover/issues/179), [#220](https://github.com/YaoFANGUK/video-subtitle-remover/issues/220), [#232](https://github.com/YaoFANGUK/video-subtitle-remover/issues/232), [#236](https://github.com/YaoFANGUK/video-subtitle-remover/issues/236)). These are static, semi-transparent, and present in every frame, so `_unmix_translucent_regions` never fires: it is gated on `has_exposure.any()` at `backend/inpainters/_common.py:1303`, and a fully static overlay has zero exposure. The one clean result reported anywhere in the community corpus solved it algebraically rather than by inpainting, reversing the alpha blend. That is the [Dekel et al. CVPR 2017](https://watermark-cvpr17.github.io/) multi-image matting formulation, unpublished for video, and it needs no GPU and no encumbered weights.
10. **RM-335, the recorded reason for parking code signing is wrong.** [Verified] `Roadmap_Blocked.md:202` parks signing because "SmartScreen reputation is per-file-hash and resets every release even for signed publishers." Microsoft's own documentation says reputation attaches to "a URL, a file, an app, **or a certificate**" ([SmartScreen docs](https://learn.microsoft.com/en-us/windows/security/operating-system-security/virus-and-threat-protection/microsoft-defender-smartscreen/), updated 2026-04-25), and the packaging guidance (updated 2026-08-17) states that unsigned files must build reputation anew with every update while signed reputation accumulates against the publisher certificate. Smart App Control **blocks** unsigned executables rather than warning about them. Azure Trusted Signing is now Azure Artifact Signing at **$9.99/month** (Azure Retail Prices API, 2026-08-27) and individual developers in the United States and Canada are now a documented first-class path. One correction in the other direction: EV certificates no longer grant an instant bypass, so the OV tier is the correct purchase.

## Product Map

### Core workflows

- Detect, track, and inpaint burned-in subtitles or watermarks from video, with automatic band detection, manual rectangle and polygon regions, timed spans, and keyframed moving regions.
- Batch and unattended operation: queue with per-item overrides, watch folder, `--pattern` globbing, checkpointed pause and resume, per-job child-process isolation, batch reports.
- Review workflows: track plans reviewed before any inpainting, mask correction with selective rerun, wipe compare, frozen approved mattes, quality reports with review spans.
- Extract, translate, and reburn: OCR to SRT or WebVTT, pluggable local translation command, restyle and reburn.
- NLE round-trip: EDL and FCPXML sidecars, lossless FFV1 or PNG matte interchange.

### User personas

- Windows desktop user cleaning hardsubs from downloaded or ripped media. Runs the installer, expects the GPU to be used.
- Archivist or batch operator running dozens of files a day unattended through the CLI or watch folder.
- Editor who wants the mask and timing back in an NLE rather than a finished file.

### Platforms and distribution

Windows 10 and 11, Python 3.11 to 3.14. Two artifacts per release, an NSIS installer and a portable ZIP, both intentionally unsigned, with SHA256SUMS, CycloneDX SBOM, pip-audit output, advisories, hidden-imports and verification JSON. A CPU-oriented Docker image exists for the CLI. No GitHub Actions, no winget, no Store, by repository policy.

### Key integrations and data flows

FFmpeg 9.0.1 as an external binary for demux, decode, encode, and mux. ONNX Runtime for RapidOCR and LaMa. Optional PaddleOCR, EasyOCR, Surya, OpenCV DNN. Optional torch paths behind environment gates. All model fetches are commit-pinned and hash-verified; the default OCR and cleanup path fetches nothing at runtime. Update check is opt-in and off by default; crash reporting needs two environment variables.

## Competitive Landscape

**[YaoFANGUK/video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover)** (Apache-2.0, 12,575 stars, last code commit 2026-04-08, 185 open issues). The upstream, and effectively frozen. Its v1.4.0 release ships **only** CPU builds (`VSR_v1.4.0_windows_x64_cpu_Setup.exe`, 46,961 downloads), and its largest issue cluster by comment volume is 31 issues and 82 comments of users asking why their GPU is idle. Learn: the demand is proven and unmet. Avoid: shipping one generic artifact whose provider is not in the filename, which v3.40.0 still does.

**[KKenny0/videowipe](https://github.com/KKenny0/videowipe)** (GPL-3.0, 44 stars, eight releases between 2026-05-22 and 2026-08-08). The only actively iterating competitor. Its differentiator is a preview-first track model, and its README positions explicitly against VSR. Learn: nothing structural, because VSR already has this and better in `backend/track_plan.py` with per-track keep flags, thumbnails and deterministic replay. Avoid: GPL, and source-only distribution with no PyPI presence.

**[SubtitleEdit](https://github.com/SubtitleEdit/subtitleedit)** (MIT, 13,980 stars, v5.1.0 2026-07-29, near-daily 5.2.0 betas). Not a competitor: it burns subtitles in and OCRs image-based subtitle formats, it does not remove hardsubs. Learn: `seconv`, a real CLI sharing one settings JSON with the GUI, and CrispEmbed, a local VLM OCR wrapper for hard cases. Avoid: nothing relevant.

**[Sanster/IOPaint](https://github.com/Sanster/IOPaint)** (23,349 stars, **archived**, last release 2024-11-23). The reference inpainting UI in this space is dead with no official successor. Learn: the whole classical layer is unmaintained, so waiting for a better backbone is not a strategy.

**EchoSubs** ([pricing](https://www.echosubs.com/tobuy)). The closest commercial analog: local-only desktop, ROI selection, batch queue, $69 lifetime. It meters "AI hours" on a product it advertises as 100% offline, caps the lifetime tier at 5 hours per month, and locks hardsub removal itself out of the free tier. It has no independent review footprint. Learn: its positioning attack on open source is "requires advanced Python and command-line knowledge", which is exactly what a working installer defeats. Avoid: metering the user's own GPU.

**GhostCut / JollyToday** ([pricing](https://jollytoday.com/vip/)). Credit-metered cloud, 4 to 6 credits per 30 seconds, so a 10-minute video costs roughly $5.20 to $7.80 at the $65/1,000 rate, and credits expire. Learn: that is the per-video price a local tool undercuts to zero. Avoid: expiring balances.

**HitPaw Video Object Remover** ($99.99 perpetual). Learn: it exposes its fill-mode ladder honestly, AI Model / Matte Filling / Color Filling / Smooth Filling / Gaussian Blur, and labels blur as the weak fallback rather than dressing it as AI. That is the model for RM-321's disclosure. Avoid: Trustpilot reports of roughly 2,200 credits (about $153) to process one episode, with credits consumed on previews.

**Runway** ([pricing](https://runwayml.com/pricing)). Aleph 2.0 is the only capability peer, removing objects and correcting shadows and reflections, at about 28 credits per second of footage, roughly $0.40 to $0.50 per second. Its terms grant training rights on inputs and outputs on every plan below Enterprise. Learn: the quality ceiling. Avoid: the entire cloud model, which is the axis where a local tool wins outright.

**DaVinci Resolve Studio** ($295 perpetual, [compare](https://www.blackmagicdesign.com/products/davinciresolve/compare)). Splits a static clone tool (Patch Replacer) from a tracked motion-aware one (Object Removal), and its own documentation says candidly that they do not work every time. Learn: both the two-tool split and the candor. Avoid: gating all AI behind a paid tier.

**Adobe After Effects Content-Aware Fill.** Learn: nothing to copy. One community report on an i9-13900KF with an RTX 3090 records a one-minute clip still rendering after 36 hours with the GPU idle. Avoid: the cloud round-trip, which sends the user's footage off the machine.

**VEED, Kapwing, Canva, Descript, Cleanup.pictures, Google Photos, Samsung.** All market AI object removal that is image-only for this purpose, or crop-and-cover for video. Samsung has publicly said video object erasure is a different proposition because the effect must be seamless per frame. Learn: the consumer field is far thinner than the marketing implies, and the whole subtitle-translation SaaS tier (Subly, Happy Scribe, Rev, Sonix, Maestra) cannot remove hardsubs at all, so translating a hardsubbed video today requires bolting two products together. VSR already does both halves.

**Research models.** [CLEAR](https://github.com/silent-commit/CLEAR) (Apache-2.0, weights public, mask-free, LoRA on Wan2.1 1.3B, +6.77 dB PSNR, but 4.86 s/frame at 720p) is the only 2026 model clearing every gate, and stays research-only per the existing roadmap decision. [SEDiT](https://zheng222.github.io/SEDiT_project/) is the product shape to watch: 31.59 dB and 4 seconds for 65 frames at 1080p, one step, but no code, no weights and no license. Everything else is either non-commercial ([ProPainter](https://github.com/sczhou/ProPainter) NTU S-Lab, [MiniMax-Remover](https://github.com/zibojia/MiniMax-Remover) CC-BY-NC, [EffectErase](https://github.com/FudanCVL/EffectErase) CC-BY-NC, [E2FGVI](https://github.com/MCG-NKU/E2FGVI) CC-BY-NC), unlicensed (GoMatching++, TMIM, UPOCR), or out of budget (VOID at 40 GB, EraserDiT above 60 GB). Correction worth recording: **DiffuEraser needs 33 GB for 720p**, not 12 GB; 12 GB buys 640x360, per its own repository README.

## Reported Issues

This repository has **zero open issues** and no open pull requests as of 2026-08-27. Seven issues are closed; two discussions are open.

Still actionable from closed threads:

- [#7](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/7), "all models produce the same results". The registry fail-open was fixed in `cfc4b03`, but the reporter's actual use case, static text overlays with a manually drawn region honored by LaMa for the whole clip, lands on the degraded path described in RM-321. Not closed in substance.
- [#3](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/3), "I would prefer an isolated portable installation, or at least Docker". Both now exist (portable ZIP, `Dockerfile`), but the Docker image is CPU-only and is not mentioned in the release body.
- [#1](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/1), RTX 50-series. The cu128 auto-detect shipped in v3.16.0 and was never confirmed on Blackwell hardware; RM-319 changes the target channel, so this needs re-checking rather than re-closing.
- [#2](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/2), "IT'S AN INFO STEALER". Correctly closed with no evidence supplied, but the title is permanently indexed against the repository name, and the underlying trigger (an unsigned PyInstaller bundle) is real and is what RM-335 addresses.
- [Discussion #8](https://github.com/SysAdminDoc/VideoSubtitleRemover/discussions/8), the only unresolved first-party request: a beginner asking for a walkthrough. Recorded as RM-306 in `Roadmap_Blocked.md`, blocked on an isolated recording environment.

Judged not actionable: [#6](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/6), residue on AI-generated live-action video, correctly closed as the capability limit of the current repair models, though RM-329 partially addresses the semi-transparent-overlay subset. [#4](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/4) and [#5](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/5) are fixed and covered by tests.

The reception signal worth acting on is that **five of five external reporters failed to get a working result on their first attempt** (launch failure, non-functional control, or output that ignored the selected model), and that there is **no third-party discussion of this project anywhere outside its own repository**. Searches across Reddit, Hacker News, VideoHelp, Doom9, the 215-issue upstream tracker and Chinese developer forums returned zero mentions of `SysAdminDoc/VideoSubtitleRemover` or "VideoSubtitleRemover Pro". At 65 stars against upstream's 12,575, distribution and first-run success are the binding constraints, not features.

The upstream tracker is the substitute for a user base. Its clusters, in demand order: GPU idle or undetected (31 issues, 82 comments), install and dependency failure (about 25), residue and ghosting (about 18), unusable wall-clock speed (13 issues, 38 comments), output blur and bitrate loss (7), region selection (13), output integrity such as dropped frames and A/V desync (11), RTX 50-series (8), non-NVIDIA lockout (18), non-ASCII paths (5, three years unfixed). VSR already answers most of these; it demonstrably does not answer the non-ASCII one.

## Security, Privacy, and Reliability

### Current strengths

No `shell=True` anywhere, enforced by policy at `backend/subprocess_policy.py:114` and an AST gate at `tests/test_source_hygiene.py:172`. No `pickle`, no `tarfile`/`zipfile` `extractall`, no `tempfile.mktemp`, no bare `except:`, no hardcoded absolute paths in production code. `torch.load` is called once, with `weights_only=True` (`backend/preprocess.py:93`). Remote model code execution is gated on a full 40-hex commit SHA (`backend/remote_model_policy.py`). The VLM endpoint has a genuine loopback-only SSRF guard with redirect revalidation (`backend/ocr_vlm.py:420-583`). The model-cache importer is transactional with per-member traversal, size, ratio and free-space ceilings. Telemetry is opt-in, self-hosted, and path-scrubbed. Git authorship is clean: no AI identities and no `Co-Authored-By` trailers in 668 commits.

### Risks and missing guardrails

- **Non-ASCII paths (RM-317).** Verified above. A hard `IOError` rather than silent corruption, but the feature is unusable and the message does not name the cause.
- **Embedded FFmpeg 7.1 (RM-320).** Verified above. The inventory exists; the classification is deliberately withheld for want of a mapping that now exists.
- **NVIDIA lane CVEs (RM-319).** Verified above. Neither is likely exploitable in VSR's use (no `torch.jit.script` of untrusted input, no torchvision GIF decoding), so this is a supply-chain and evidence problem rather than a live exploit.
- **The advisory feed cannot see three of the most relevant exposures.** [Verified] `CVE-2026-24747` is filed by GitHub against the pip package `pytorch`, not `torch`, so `gh api 'advisories?ecosystem=pip&affects=torch'` returns nothing and a pip-audit run keyed on `torch` reports a vulnerable 2.9.x pin as clean. ONNX Runtime publishes **zero** CVEs while shipping 54 security-fix bullets across 1.27, 1.28 and 1.29. OpenCV's bundled libavcodec is invisible to every SCA tool. The release evidence at `release-advisories.json` reports `total: 0`, which is true and not the same as safe.
- **126 broad `except Exception: pass` blocks in non-test code** (measured 2026-08-27). The worst is `backend/io.py:2159`, the FFmpeg stderr reader thread: an exception there silently truncates the diagnostic tail the error path later reports from. `pyproject.toml` selects only `E4,E7,E9,F`, so ruff's `BLE`, `B` and `S` families are off and nothing mechanically holds the count. Enabling them measures 417 `BLE001`, 127 `S110`, and 61 `B905`.
- **61 `zip()` calls without `strict=`**, including the inpainting hot path at `backend/inpainters/_common.py:644` (`zip(original, filled, masks)`) and `:1161` (`zip(frames, masks)`). A frames-to-masks length mismatch silently processes the shorter prefix instead of failing, which is precisely the silent-degradation class the last six months of commits have been eliminating.
- **The subprocess policy gate covers `backend/` only.** Eleven launch sites in `gui/` bypass it (`gui/job_supervisor.py:191`, `gui/utils.py:248`, `gui/utils.py:407`, `gui/quality_controller.py:596`, `gui/support_controller.py:180`, `gui/widgets.py:2441` and others), plus `setup.py` and `tools/`. None use `shell=True`, so this is defense in depth, not a live hole.
- **`hashlib.md5()` without `usedforsecurity=False`** at `backend/proxy_workflow.py:186`, used only as a cache fingerprint. It raises on a FIPS-mode host.
- **`exec()` on a user-supplied `.vpy`** at `backend/vapoursynth_bridge.py:88`. Inherent to the VapourSynth format and the user picks the file, but the file-picker copy does not say so.
- **No dependency-review staleness check.** `dependency_profiles.json` records `reviewedAt` and nothing ever ages it. protobuf shipped a **major** version (7.36.0) on 2026-08-20, the exact day of the last review.

### Advisory assessment (2026-08-27)

FFmpeg 9.0.1 remains the newest release; no 9.0.2 or 9.1 exists. Seven CVEs published 2026-08-19 (CVE-2026-75141 through 75147, top score 9.8 in `librist`) were **all fixed in 9.0.1 before disclosure**, and all seven are still `vulnStatus: Received` on NVD with zero version applicability data, so version-keyed scanners cannot resolve them. Pillow 12.3.0 closes a 14-CVE batch disclosed 2026-07-20 and matches the pin. onnxruntime 1.29.0 (2026-08-17) matches the CPU pin. PyInstaller 6.22.2 (2026-08-17) matches the build floor, and its onefile symlink fix does not apply because the build is onedir. NSIS 3.12 is current. CPython 3.13.15 is current.

### Recovery and rollback

Checkpointed pause and resume, per-job supervision, atomic output promotion, non-overwriting unique output paths, undo and redo in both editors. Missing: no undo for single-item queue removal, Clear completed, or queue sort; no reset-to-defaults across 140 settings; onboarding cannot be replayed once `onboarding_seen` is set at `gui/onboarding.py:214`.

## Architecture Assessment

### Boundary improvements

- `backend/processor.py` is 4,487 lines, 44% larger than the next file and the single hottest defect site in the commit history (timing, encode, mux, cancel and stage verification bugs all land there). The frame-loop extraction is already specified in `Roadmap_Blocked.md:156` and blocked only on a GPU host, which now exists. The mixin-extraction pattern is proven in this repo (`d9f88bb`, `cc9ffca`, `15b68b2`, `4d0bcea`).
- `gui/app.py` (3,113) has already shed builders to `layout_build.py` (2,524) and controllers to five mixins, yet still carries the Tk lifecycle logic that keeps producing teardown races and 24 of the broad exception swallows.
- `gui/widgets.py` (2,850) is one flat module of custom Tk widgets with 30 broad swallows; splitting per widget family would make each independently testable.

### Refactor candidates

`backend/cli.py` (2,880, 157 flags), `backend/io.py` (2,425), `gui/region_controller.py` (2,129), `backend/detection.py` (2,091), `gui/preview_controller.py` (2,020). None is urgent on its own; `processor.py` is.

### Test and documentation gaps

- **No GPU-gated test exists** (RM-322). The 1587-test suite exercises no CUDA path, no NVENC encode, no provider selection, no OOM recovery.
- **No non-ASCII path test exists.** Only `tests/test_nle_sidecar.py` touches non-ASCII, and that is sidecar text.
- The reference corpus is 10 synthetic 160x96 16-frame clips whose floors are snapshots of their own output (RM-318). `Roadmap_Blocked.md` parks replacing them on rights verification; [cyberagent/OTR](https://huggingface.co/datasets/cyberagent/OTR) (CC-BY-4.0, 89.3K overlay-text pairs) and [BeyondMasks](https://github.com/YigitEkin/BeyondMasks) (ECCV 2026, CC BY 4.0, pushed 2026-08-22) resolve the rights question for the first time.
- `README.md` is 104 KB and 1,649 lines, mixes maintainer-only release engineering into the user install path, inlines about 330 lines of generated CLI and config tables, has 820 CRLF against 829 LF lines with **no `.gitattributes` in the repository**, and contains one stale line at `README.md:1543` (`models/ # AI model weights (auto-downloaded)`) that contradicts the privacy promise at `README.md:26-27`. Human-voice compliance is otherwise clean: no em dashes, no en dashes, no rule-of-three tells.
- README and `gui/onboarding.py:106` promise drag-and-drop, but `tkinterdnd2` is commented out at `requirements.txt:114`, absent from `VideoSubtitleRemoverPro.spec`, and not installed in the repo venv, so no shipped build can do it.

### User-facing defects found by code audit

All [Verified] by reading the source on 2026-08-27. These are the source for RM-328, RM-338 through RM-341, RM-343 and RM-344.

- **First-run model download reports almost nothing** (RM-328): `gui/processing_controller.py:362-395` fires one 2.6 second toast, pins the queue row at 2 percent, and writes the reassuring detail text to the log only at `:387`. On a multi-gigabyte first fetch that reads as a hang, which is the most common beginner failure in this category.
- **The update notice cannot reach the release page** (RM-338): `gui/app.py:3015-3032` logs the URL and tells the user to look in the log panel, with the whole handler wrapped in a bare `except Exception: pass`.
- **The preview cannot show where the residue is** (RM-339): the pipeline computes a per-frame residual score inside the mask (`backend/quality.py:766`) and emits review spans from it (`backend/_quality_mixin.py:505-514`), but `gui/preview_controller.py:1491-1621` can only overlay detection boxes. Grepping `gui/` found no difference view, heatmap, loupe, or onion-skin. No competitor surveyed exposes any per-frame quality signal at all, so this is a differentiator rather than parity work.
- **Multi-monitor and DPI handling is one version behind its own comment** (RM-340): `VideoSubtitleRemover.py:337-346` says "Per-Monitor V2" and calls `SetProcessDpiAwareness(2)`, which is V1; `gui/dialog_layout.py:31-48` sizes every dialog from the primary monitor with hardcoded 0.96 and 0.90 taskbar factors, and no `SPI_GETWORKAREA`, `MonitorFromWindow`, or `GetMonitorInfo` call exists anywhere; `gui/app.py:153` fixes `minsize(980, 720)` regardless of text scale; `gui/app.py:419` persists geometry but never `wm_state`.
- **There is no way back from a broken configuration** (RM-341): no reset-to-defaults exists across 140 settings, `gui/onboarding.py:214` sets `onboarding_seen` as the dialog is built with no replay path, and queue removal, clear-completed, and sort all discard state with no undo.
- **Drag and drop is promised and cannot work** (RM-343): `tkinterdnd2` is commented out at `requirements.txt:114`, absent from `VideoSubtitleRemoverPro.spec`, and not installed in the repo venv, while `README.md:92`, `README.md:369`, and `gui/onboarding.py:106` all advertise it.
- **Both shipped themes are dark** (RM-344): `gui/theme.py` defines only `apply_default_theme` and `apply_high_contrast_theme`, against the project's own rule to include a light option where practical.

Two smaller items are recorded here and deliberately not filed. Log verbosity is fixed at `INFO` by `VideoSubtitleRemover.py:56-64` with no in-app control, which matters only when diagnosing a report and is partly served by the JSONL log and the support bundle. Toasts are a fixed 2,600 ms for every tone (`gui/widgets.py:1587`) with no dismiss or hover-to-pause, so an error message can disappear before it is read; the footer and log panel both retain it, so this is a polish item rather than a loss of information.

### Format and platform fidelity gaps

- **Subtitle export is lossy in the one dimension the detector is good at.** [Verified] The product accepts an `.ass` file for re-burn (`backend/cli.py:831`) but writes only SRT and WebVTT, and neither FFmpeg encoder emits position: `srtenc.c` carries a standing TODO that subtitle position side data is never written and leaks `{\an8}` into the text, and `webvttenc.c` nulls the colour, font, size, alignment, and move callbacks. ASS is the only text format Matroska carries losslessly (draft-ietf-cellar-codec-20, 2026-08-14), and libass 0.17.0 added `LayoutResX`/`LayoutResY` precisely so authored coordinates can be separated from the render canvas. This becomes RM-345. Worth recording for a future pass: ruby and vertical writing are not expressible in ASS at all, so faithful Japanese extraction would need TTML, and [IMSC Text Profile 1.3](https://www.w3.org/TR/ttml-imsc1.3/) became a W3C Recommendation on 2026-05-21 and drops the Image Profile.
- **The frozen build silently loses long-path support the source run has.** [Verified] No application manifest is referenced anywhere in `VideoSubtitleRemoverPro.spec` or `installer/vsr.nsi`, and no `longPathAware` declaration exists. Windows requires both the machine policy and a per-application manifest declaration, and CPython ships that declaration in its own `python.exe`, so the same code behaves differently frozen. No `SetCurrentProcessExplicitAppUserModelID` call exists either. This becomes RM-346.
- **Embedded closed captions are destroyed without warning.** [Verified] Grepping `backend/` found no reference to `a53cc`, CEA-608, CEA-708, or A53 side data. CEA captions live inside the video bitstream rather than as a mappable stream, so the raw-frame pipeline drops them, and FFmpeg only exports them from the MPEG-2 and H.264 decoders in any case, meaning an HEVC source cannot be preserved at all. This is an accessibility regression the existing stream-mapping work cannot catch, and becomes RM-347.

### Category coverage decisions

Migration and upgrade paths are covered (`backend/config_schema.py` schema v10, `migrate_gui_settings`, forward-compatible read-only mode). Multi-user is deliberately out of the product's shape and RM-314 already covers same-user concurrency. Plugin ecosystem is covered by `backend/inpainter_registry.py` without a marketplace, which the roadmap rejects. Offline resilience is strong: the default path fetches nothing. Observability is strong: rotating log, JSONL log, redacted support bundle, batch reports, execution provenance; the only gap is no verbosity control (`VideoSubtitleRemover.py:56-64` fixes `INFO`). Accessibility is genuinely engineered (MSAA annotation, live region, `tk_focusNext` probes across five text scales and three themes, reduced motion) and its remaining gap, real UIA providers, stays in `Roadmap_Blocked.md` because it needs a live screen reader and the standing rule forbids taking over the user's screen. Non-text contrast is the one a11y item that can be fixed headlessly and is filed as RM-333.

## Rejected Ideas

- **Alarm that VSR's ProPainter mode contaminates its MIT license with NTU S-Lab terms.** [Verified false] The mode is TBE plus LaMa residual and bundles no ProPainter weights; `README.md:397-399` discloses the borrowed name. Source: the OSS competitor sweep, which flagged it from the upstream license alone.
- **Add a "Copy CLI command" button, per upstream [#244](https://github.com/YaoFANGUK/video-subtitle-remover/issues/244) and [#246](https://github.com/YaoFANGUK/video-subtitle-remover/issues/246).** Already shipped at `gui/widgets.py:65` and `:2374`.
- **Adopt videowipe's preview-first track model.** Already shipped and richer: `backend/track_plan.py` with per-track keep flags, thumbnails, deterministic replay and mask-correction consumption.
- **Adopt YOSE's mask-aware compute so subtitle bands do not cost full-frame inference.** Already shipped: `backend/inpainters/lama.py:510-568` bounds tiling to the mask ROI and skips tiles with an empty mask, and `lama_tile_size` defaults to 512 so any 1080p frame takes that path.
- **Add a wipe compare and a worst-frame jump.** Both already exist (`gui/preview_controller.py:683-860`, `gui/quality_controller.py:271-301`), as do navigable mask review spans.
- **`B023` at `backend/ocr_fix.py:117`, "function does not bind loop variable `dst`".** False positive: `re.sub` consumes the lambda inside the same iteration.
- **Blackwell-only quantization (NVFP4, MXFP8) for the repair models.** Both require CUDA capability >= 10.0, which excludes every RTX 40-series card including the build host. [torchao/diffusers benchmark](https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/).
- **Nunchaku / SVDQuant for the inpainting model.** Supports no video models at all as of v1.2.0.
- **DiffuEraser as a consumer-GPU backend.** Its own README records 33 GB for 720p. Also inherits ProPainter weights.
- **GoMatching++ for video text tracking**, despite being the published answer to "what beats per-frame OCR plus IoU" at 7.3 GB and 11 FPS: the repository has **no LICENSE file**, so all rights are reserved. Reimplementing the LST-Matcher idea is legitimate; vendoring the code is not.
- **pyiqa for no-reference quality metrics.** PolyForm Noncommercial plus NTU S-Lab.
- **Replace `paddleocr` outright with `rapidocr`.** Tempting on footprint (26 MB against roughly 1.2 GB) and CVE count, but `dependency_profiles.json` already makes rapidocr the required default and paddleocr a reviewed opt-in, so the win is already banked. Removing the opt-in would drop PP-StructureV3 and PaddleOCR-VL.
- **Ship an in-app auto-updater.** The opt-in check is the correct privacy posture; the defect is that the release URL goes only to the log file, which RM-338 fixes without adding a downloader.
- **Refile RM-307 through RM-313.** Shipped between 2026-08-23 and 2026-08-27. RM-306, RM-314, RM-315 and RM-316 remain recorded once each.

## Sources

### Repository and tracker
- https://github.com/SysAdminDoc/VideoSubtitleRemover/releases/tag/v3.40.0
- https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/7
- https://github.com/SysAdminDoc/VideoSubtitleRemover/discussions/8
- https://github.com/YaoFANGUK/video-subtitle-remover
- https://github.com/YaoFANGUK/video-subtitle-remover/issues/96
- https://github.com/YaoFANGUK/video-subtitle-remover/issues/192
- https://github.com/YaoFANGUK/video-subtitle-remover/issues/240
- https://github.com/YaoFANGUK/video-subtitle-remover/issues/111

### Open source competitors and adjacent tools
- https://github.com/KKenny0/videowipe
- https://github.com/SubtitleEdit/subtitleedit
- https://github.com/Sanster/IOPaint
- https://github.com/silent-commit/CLEAR
- https://github.com/sczhou/ProPainter
- https://github.com/advimman/lama
- https://github.com/timminator/VideOCR

### Commercial products
- https://www.echosubs.com/tobuy
- https://jollytoday.com/vip/
- https://www.blackmagicdesign.com/products/davinciresolve/compare
- https://runwayml.com/pricing
- https://www.topazlabs.com/topaz-video-ai
- https://filmora.wondershare.com/guide/ai-video-object-remover.html
- https://docs.topazlabs.com/video-ai/reference-guide/importing-previewing-and-exporting

### Standards, specifications and platform APIs
- https://learn.microsoft.com/en-us/windows/security/operating-system-security/virus-and-threat-protection/microsoft-defender-smartscreen/
- https://learn.microsoft.com/en-us/azure/artifact-signing/quickstart
- https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/smartscreen-reputation
- https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation
- https://learn.microsoft.com/en-us/windows/win32/api/shobjidl_core/nf-shobjidl_core-setcurrentprocessexplicitappusermodelid
- https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate
- https://www.w3.org/TR/ttml-imsc1.3/
- https://datatracker.ietf.org/doc/draft-ietf-cellar-codec/
- https://github.com/libass/libass/wiki/ASS-File-Format-Guide
- https://pypi.org/project/pysubs2/
- https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
- https://pyinstaller.org/en/stable/bootloader-building.html
- https://www.microsoft.com/wdsi/filesubmission
- https://signpath.org/terms

### Academic work and benchmarks
- https://arxiv.org/abs/2605.14534
- https://github.com/xiaomi-research/prove
- https://arxiv.org/abs/2605.14894
- https://arxiv.org/abs/2603.21901
- https://arxiv.org/abs/2401.17904
- https://github.com/ymy-k/Hi-SAM
- https://arxiv.org/abs/2511.22499
- https://arxiv.org/abs/2506.21002
- https://github.com/takumi-yoshimatsu/ISTR
- https://watermark-cvpr17.github.io/
- https://arxiv.org/abs/2505.22228
- https://arxiv.org/abs/2501.00321
- https://huggingface.co/datasets/cyberagent/OTR
- https://github.com/YigitEkin/BeyondMasks

### Dependency releases and security advisories
- https://github.com/microsoft/onnxruntime/releases/tag/v1.27.0
- https://github.com/advisories/GHSA-63cw-57p8-fm3p
- https://github.com/advisories/GHSA-rrmf-rvhw-rf47
- https://nvd.nist.gov/vuln/detail/CVE-2026-65918
- https://nvd.nist.gov/vuln/detail/CVE-2026-24747
- https://ffmpeg.org/download.html
- https://github.com/PaddlePaddle/PaddleOCR/releases/tag/v3.7.0
- https://huggingface.co/blog/PaddlePaddle/pp-ocrv6
- https://rapidai.github.io/RapidOCRDocs/latest/install_usage/rapidocr/install/
- https://pyinstaller.org/en/stable/CHANGES.html
- https://download.pytorch.org/whl/cu130/torch/
- https://huggingface.co/docs/hub/security-pickle

## Open Questions

1. **CUDA 13 lane qualification.** Does `torch 2.13.0+cu130` plus `onnxruntime-gpu 1.29.0` pass VSR's provider smoke, reference-corpus hashes, and frozen-bundle size budget on the RTX 4070 SUPER, and does the resulting installer stay under the current artifact size? RM-319 cannot land without measured evidence, and the reference-corpus pixel hashes are bound to the reviewed profile so they will need re-blessing.
2. **Non-ASCII scope beyond image I/O.** Does the external adapter and diffusion temp-directory path (`backend/inpainters/external.py:230`, `backend/inpainters_diffusion.py:1254`) survive a `%TEMP%` that contains a non-ASCII username, and does the NSIS installer place the app correctly under such a profile? The local probe covered library behavior, not the installed product.
3. **Residual-text ceiling calibration on real media.** `RESIDUAL_TEXT_SCORE_CEILING = 0.025` was tuned against the synthetic corpus. Does it hold on redistributable 1080p footage, or does RM-318 need a per-resolution ceiling before the corpus can gate on it?
4. **Azure Artifact Signing eligibility.** The previously documented three-year-entity requirement is absent from the current documentation and the old URLs now 404. Whether it was dropped or relocated must be confirmed with Microsoft before RM-335 commits to the individual path.
