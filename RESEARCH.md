# Research: Video Subtitle Remover Pro

Date: 2026-09-04. Replaces all prior research. Supersedes the 2026-08-27 pass.

## Executive Summary

Video Subtitle Remover Pro is a Windows-first, fully local remover of burned-in subtitles and text
watermarks: a tkinter workbench over a multi-engine OCR cascade (RapidOCR, PaddleOCR, Surya,
EasyOCR, OpenCV DNN) and a three-family inpainting stack (Temporal Background Exposure, LaMa, and a
TBE plus LaMa hybrid), with a batch queue, crash-resume checkpoints, HDR and VFR preservation, SRT
and NLE export, and a release pipeline that refuses to stage an artifact it cannot prove. Against
the rest of the field it is not short of features or engineering rigour. It is short of a shipped
release. The last published build is v3.40.0 (2026-08-27); 29 commits and a 269-line `[Unreleased]`
CHANGELOG block sit above the tag. The two most valuable things in that block are separately built
CPU and CUDA artifacts, and the fix for a frozen CUDA build silently running on CPU, which is
exactly what both open user issues are about. The highest-value direction is therefore not a new
capability. It is closing the distance between what main can do, what the README says, and what a
person can actually download and run.

Top opportunities, in priority order:

1. Cut the lane-separated release. The README on main advertises an `-nvidia-` download that no
   GitHub Release carries, and issue #10 is a user hitting that exact wall.
2. Give LaMa a working weight-acquisition route. In the shipped build, selecting LaMa cannot succeed
   under any configuration. That is issue #11.
3. Make the algorithm picker say which engines can run before the run starts, instead of failing
   after the user commits.
4. Tell a CPU-lane user who has an NVIDIA card that a CUDA build exists. The app already knows both
   halves of that sentence and says neither.
5. Move every dependency profile to torch 2.14.0 and torchvision 0.29.0, closing CVE-2026-65918,
   which the repo's own profile notes record as unclosed.
6. Reach the support bundle from the failure that needs it. The bug template asks for one, and the
   reporter of issue #11 could not find it.
7. Bound the tiled inpainter's accumulators to the masked ROI instead of the full frame.
8. Give a clip a pre-run verdict from a sampled dry run. That is the answer to the field's loudest
   complaint, which is that automatic removal is unpredictable at archive scale.

## Product Map

Core workflows:

- **Single clip, reviewed.** Load video, set or confirm region, Test cleanup preview, run, inspect
  quality verdict and worst frame, export.
- **Batch queue.** Up to 500 items, per-item config, crash-resume checkpoints, verified skip of
  already-processed outputs against a v3 sidecar (source hash, normalised config, output hash).
- **Headless and scripted.** `python -m backend.cli` with a generated flag reference, plus
  watch-folder drain (`--watch <dir>`), track plans (`--plan-out` and `--plan-in`), and a Docker CPU
  runtime.
- **Extract rather than remove.** OCR-backed SRT and WebVTT export reusing tracked recognition, plus
  matte and NLE (EDL, FCPXML) interchange for people who would rather fix it in an editor.

Personas: a Windows prosumer processing their own footage through a GUI. That is the primary one,
and it is who the installer, onboarding, high-contrast theme and 100 to 200 percent scaling exist
for. Secondary: a scripted archive operator using the CLI and watch folder, and an editor who wants
the matte rather than the render.

Platforms and distribution: Windows x64 only for the shipped artifacts, built locally by
`build_exe.bat` (there is no CI, deliberately). Two lanes since 2026-08-28. CPU ships an NSIS
installer plus a portable ZIP at roughly 99 MB and 288 MB installed. NVIDIA ships a portable ZIP
only, 2.1 GB and 3.1 GB installed, because the 32-bit NSIS compiler dies past about 2 GB
(`backend/release_staging.py:71`). DirectML is a supported local setup profile that is deliberately
not published. A Linux CPU CLI exists as a Dockerfile.

Key integrations and data flows: FFmpeg 9.0.1 or newer as an external binary for probe, decode,
encode and mux, with a second and older FFmpeg 7.1 embedded inside the OpenCV wheel, which is what
`cv2.VideoCapture` actually decodes through; ONNX Runtime for OCR and LaMa inference; torch only
where a model needs it, and in the frozen NVIDIA bundle as the carrier of the CUDA runtime itself;
Hugging Face for opt-in model fetches, pinned to allowlisted repos at full commit SHAs with per-file
hash checks.

## Competitive Landscape

**YaoFANGUK/video-subtitle-remover** (Apache-2.0, 12,723 stars, last push 2026-06-30, 187 open
issues). The category's centre of gravity, and effectively feature-frozen. Learn: nothing technical
remains to take. Its issue tracker is the demand signal, not its code. Its largest cluster is
idle-GPU reports against CPU-only installers, which is precisely the failure this project is one
release away from having fixed and having not shipped. Avoid: an untriaged 187-issue tracker, and
shipping a CPU-only artifact while advertising GPU acceleration.

**KKenny0/videowipe** (GPL-3.0, 47 stars, last push 2026-08-31). The only genuinely fast mover, with
eight releases since 2026-07-30. Learn three things. Its releases publish a detection baseline
(remove Jaccard 0.239141, a boundary F score, and a 0.0 false-selection rate), not just a
restoration score, and it binds the evaluator source and detector weights into the report so a run
that changed either is rejected. Its v0.8.1 changed a safety default: persistent overlays across the
top of a video are kept unless explicitly selected, because a station logo is not a subtitle. Its
v0.8.0 attacked long and high-resolution video by shrinking inpainting segments and refusing to
retain full-resolution frames between model passes. Avoid: GPL-3.0, which is incompatible with this
project's MIT, and its issues-disabled posture, which throws away the demand signal.

**IOPaint** (Apache-2.0, 23,343 stars, archived since 2025-04-29). Learn: the reference for an
inpainting model registry with per-model download, hash and licence metadata, which this project has
in `backend/adapter_manifest.py` but does not wire to any actual fetch. Avoid: its fate. A model-zoo
tool with no maintainer is a dead tool, so every adapter added here needs a stated owner and a
verification path or it becomes the same liability.

**wiltodelta/remove-ai-watermarks** (Apache-2.0, 5,410 stars, created 2026-03-25, last push
2026-09-03). The fastest-growing adjacent project in the space, and the clearest demand signal of
2026: removing the visible marks that Sora, Veo, Kling, Seedance, Hailuo and Gemini burn into
generated video. Learn: a registry of known marks with fixed geometry, so a user picks a generator
instead of drawing a rectangle, and an explicit scope-and-safety document that says what the tool is
for. Avoid: its invisible-watermark and C2PA provenance-stripping half. That is a different product
with a different ethical posture, and this project should say so rather than drift into it.

**sczhou/ProPainter** (NOASSERTION, in practice NTU S-Lab 1.0 and non-commercial, 6,930 stars, last
push 2025-02-19). Learn: nothing new. The propagation-plus-transformer design is already the
conceptual parent of the TBE plus LaMa hybrid this project ships under the same name. Avoid:
shipping the real weights. The licence is incompatible with MIT redistribution, which is already
recorded in `Roadmap_Blocked.md` item 59, and the upstream repo has been static for over eighteen
months.

**GhostCut** (commercial, from about $0.1 per minute, plans from $9.9 a month). Learn: what the
market pays for is not the eraser. It is the pipeline around it. Batch of 100, an API, and OCR plus
ASR subtitle generation, translation and dubbing in one pass. The removal step is a commodity input
to a localisation workflow. Avoid: chasing that workflow. This is a local desktop tool, its
differentiator is that nothing leaves the machine, and adding a cloud pipeline would forfeit it.

**HitPaw, Media.io, Vmake, Kapwing, VEED, CapCut** (commercial SaaS). Learn: they all paywall
resolution and batch, and none of them expose a quality signal, so the user finds out by watching
the output. Avoid: their marketing claim of "fully automatic". The honest position, which this
project already takes in its README benchmark table, is a durable differentiator.

**DaVinci Resolve 21** (Blackmagic, Studio-only object removal). Learn: Magic Mask's new Render in
Place caches a tracked mask as an external matte linked back to the source, so the expensive
tracking pass is done once and reused. This project has the equivalent in `backend/track_plan.py`
and `backend/matte_interchange.py`, so the gap is discoverability, not capability. Avoid: assuming a
professional NLE is the competitor. Resolve's Object Removal is documented as working when a small
object crosses a stable surface, and it is not built for a 200-frame static caption band.

**Subtitle Edit** (GPL, active). Adjacent rather than competing: it does OCR-to-SRT extraction
extremely well and removes nothing. Learn: its OCR-fix-list model, which this project already
imports via `scripts/import_se_ocrfix.py`. Avoid: duplicating its editor.

## Reported Issues

The tracker is small, and both open issues are real, current, and reproducible from the code.

- **#10, "CUDA device requested but falls back to CPU" (2026-08-31, open, unlabelled).** The user's
  log shows `cuda:0` requested, then `no CUDA inference provider is available`, then RapidOCR
  running on CPU while `h264_nvenc` is detected, so an NVIDIA card is plainly present. This is not a
  bug in the code on main. It is the shipped artifact. `gh release view v3.40.0 --json assets` lists
  `VideoSubtitleRemoverPro-3.40.0-Setup.exe` and `-Windows-x64.zip` with no lane in either name, and
  the tag is dated 2026-08-27, one day before the lane split landed. Both fixes exist, unreleased:
  `backend/onnxruntime_cuda.py:20` hands `preload_dlls` the bundle's own `torch/lib`, because a
  frozen build has no `importlib.metadata` and ONNX Runtime could never find the CUDA libraries; and
  `backend/release_staging.py` now refuses an artifact whose measured provider disagrees with its
  filename. Meanwhile `README.md:128-131` on main already advertises the
  `VideoSubtitleRemoverPro-X.Y.Z-nvidia-*` download. The repository front page therefore promises a
  file the Releases page does not have. Filed as RM-353 and RM-356.

- **#11, "[Bug]: Lama doesnt work" (2026-09-04, open, `bug`).** v3.40.0 release package, Windows 11,
  NVIDIA, 4K MP4; other models work, LaMa fails with an error dialog. Traced and confirmed: every
  LaMa backend needs a weight file the shipped build does not have and offers no way to get.
  `backend/model_downloads.py:818-900` reports LaMa ONNX as `missing`, with the next action "Set
  VSR_LAMA_ONNX or place lama_fp32.onnx/lama.onnx in the app model cache", and the OpenCV DNN path
  wanting `VSR_OPENCV_LAMA`. `VideoSubtitleRemoverPro.spec:85-86` excludes `simple_lama_inpainting`
  from the bundle unless `VSR_ENABLE_PYTORCH_LAMA` was set at build time. There is no downloader for
  any of the three, so the failure is a certainty rather than a misconfiguration. This was traced by
  reading the shipped configuration rather than by running the released binary, so treat the exact
  dialog text as unconfirmed while the cause itself is verified. Since RM-307 made named engines
  fail closed, a typed failure with a recovery hint is the expected symptom. The materials
  for the fix already exist: `backend/adapter_manifest.py:93-124` carries the source URLs
  (`https://huggingface.co/Carve/LaMa-ONNX` and `https://huggingface.co/opencv/inpainting_lama`) and
  a pinned SHA-256 for every filename. The same reporter wrote "Could not create the bundle because
  I couldn't find the option": the bug template asks for a support bundle, and the only entry point
  is a ghost button at `gui/support_controller.py:873`. Filed as RM-354, RM-355 and RM-358.

- **Discussions #8 and #9** (both opened by the maintainer on 2026-07-30, one reply in total). The
  beginner walkthrough request in #8 is already tracked as RM-306 in `Roadmap_Blocked.md`, blocked
  on an isolated recording session. Not re-filed.

- **Judged not actionable.** There are no other open issues, no open pull requests, and no
  third-party contributions. The closed set was audited in the 2026-08-21 pass and remains closed
  correctly. The absence of a backlog is not a sign of health here: with 69 stars against upstream's
  12,723, this project has almost no user population to report anything, which is what RM-342 exists
  to change.

Cluster reading: both open issues, and the one usability complaint inside them, land on the same
module, which is the boundary between what the packaged artifact contains and what the interface
offers. That is where the next pass should spend its effort, and it is where P0 is aimed.

## Security, Privacy, and Reliability

- **CVE-2026-65918 (torchvision through 0.28.0, an out-of-bounds heap read in the GIF decoder's
  `read_from_tensor` callback) is now closable and is not closed.** `dependency_profiles.json:103`
  records the reasoning as of 2026-08-27: "cu130, cpu, and directml all top out at torchvision
  0.28.0, so every lane carries it equally until upstream publishes a fix." That is stale. Verified
  on 2026-09-04 against the PyTorch wheel indexes:
  `torchvision-0.29.0+cu130-cp313-cp313-win_amd64.whl` and the matching cpu wheel both exist,
  released 2026-09-02, past the affected range. Filed as RM-357.

- **torch 2.14.0 (2026-09-02) continues hardening the exact class that produced CVE-2026-24747.**
  Its security section includes "Always validate sparse-tensor invariants when loading with
  `torch.load(..., weights_only=True)` so malformed checkpoints cannot create tensors whose indices
  cause out-of-bounds reads" (#184750), plus `grid_sample` and `replication_pad` backward shape
  validation. This project loads a third-party `big-lama.pt` on the PyTorch LaMa path, so the class
  is directly reachable, and `torch-2.14.0+cu130-cp313-cp313-win_amd64.whl` is available. Filed as
  RM-357.

- **No newer ONNX Runtime.** 1.29.0 (2026-08-17) is still the latest on PyPI for both CPU and GPU,
  and `onnxruntime-directml` is still 1.24.4 (2026-03-17). The widening DirectML gap remains
  expected rather than drift, as `dependency_profiles.json:111` already records. No action.

- **No newer FFmpeg.** 9.0.1 (2026-08-12) is still the head of the 9.0 branch and there is no 9.0.2,
  so the enforced floor is correct. The unresolved item is unchanged and already tracked: OpenCV's
  Windows wheel embeds FFmpeg 7.1 regardless of which OpenCV is installed, so `cv2.VideoCapture`
  decodes user media through a build two major versions below the enforced floor. RM-348 in
  `Roadmap_Blocked.md` covers routing the 25 `cv2.VideoCapture` sites off it.

- **Full-frame float32 accumulators in every tiled inpaint path.** `backend/inpainters/lama.py:545`,
  `:665` and `:785` each allocate `weight_acc` at `(h, w)` and `color_acc` at `(h, w, 3)` in float32
  for every frame, then blend across the whole frame with a per-channel `np.where`, even though the
  tiles only ever cover the mask ROI computed a few lines earlier. At 3840x2160 that is about 133 MB
  of accumulator plus three 33 MB temporaries per frame, to repair a caption strip that is typically
  under 5 percent of the picture. Issue #11 is a 4K clip. videowipe shipped the same class of fix in
  its v0.8.0. Filed as RM-359.

- **Privacy posture is sound and should stay that way.** `docs/privacy-and-network.md` enumerates
  every optional download route and its offline alternative, VACE auto-fetch is pinned to
  `Wan-AI/Wan2.1-VACE-1.3B` at commit `574e6a744642ce3bee319afc31496b88bde8aac4` with per-file
  SHA-256 gating, and support bundles are redacted. Any LaMa fetcher added for RM-354 must inherit
  that contract: allowlisted host, pinned hash from `backend/model_hashes.py`, refusal when offline,
  and never automatic without consent.

- **No stated scope of lawful use.** Grepping README.md on 2026-09-04 for "lawful", "legal",
  "copyright", "you own" and "permission" returns nothing. A tool that removes watermarks, competing
  in a space where the 5,410-star neighbour ships `docs/legal-and-safety.md`, should say what it is
  for. Filed as RM-364.

- **Recovery.** Crash-resume checkpoints, queue undo, reset-settings and show-walkthrough
  affordances all exist (commit `a7a4adf`). Settings carry a schema version and a migration path at
  `gui/config.py:1073`. No gap found.

## Architecture Assessment

- **The processor decomposition is working and should finish.** `backend/processor.py` went from
  4,487 lines to 1,620 across commits `cebb276` and `f84ac01`, with byte-identical output verified
  on both lanes (`0f242793c470ebe0a4ac4f84555fd07a88a7c569d95a9ffd7181e6e64ffab28a` on
  `tests/clips/static_dialogue.mkv`). RM-349 tracks the remainder. Note the coupling:
  `tests/test_exception_logging.py` derives its file list from `backend/_*_mixin.py`, so any new
  processor code outside that naming pattern is silently uncovered.

- **The GUI is where the churn is and where the least is proven.** `gui/app.py` (3,283 lines),
  `gui/widgets.py` (2,859) and `gui/layout_build.py` (2,602) are the top three churn files after the
  docs, and the GUI layer out-churns the backend roughly two to one across the last 200 commits. The
  active GUI coverage is `tests/test_gui_workflow_release.py` plus the release probe, while 17 files
  and 191 tests sit inert in `tests/archive/` behind `collect_ignore_glob = ["*.py"]`. That is a
  deliberate call, but it means the highest-churn code has the thinnest safety net.

- **Engine availability is computed and then thrown away.** `backend/model_downloads.py:818-900`
  produces a per-engine record with `available`, `status`, `next_action` and `expected_files`.
  `gui/layout_build.py:604` builds the algorithm picker as
  `options=[(m.value, m.value) for m in InpaintMode]`, which is every enum member unconditionally,
  with no reference to that record. Wiring the existing record into the existing picker is a small
  change with an outsized effect. Filed as RM-355, shipped 2026-09-05.

  **Corrected 2026-09-05 while implementing it.** This paragraph originally said MiGAN was a second
  dead entry in the picker, and that two of five offered algorithms could not run. That was wrong,
  and the error was reading `backend/config.py:83-89` where `gui/config.py:184-188` governs: the two
  `InpaintMode` enums are deliberately separate, and the GUI one has four members with no MIGAN. The
  user-facing control is also not the picker at all but the command-bar combo at
  `gui/layout_build.py:164`, which maps Balanced, Motion, Detail and Temporal onto those same four.
  MiGAN is CLI-only, registered as a mode only when `VSR_MIGAN_ONNX` names a file
  (`backend/inpainters_onnx.py:428`), and `--mode migan` without it already fails closed naming the
  registered backends. LaMa was the only genuinely dead entry, which is still a P0 because it is the
  engine issue #11 was filed against.

- **The track plan defaults every track to removal.** `backend/track_plan.py:190` sets
  `"keep": False` on construction, so a station logo present for the whole runtime is treated
  identically to a caption that appears for two seconds. videowipe changed this default in v0.8.1
  after user reports. The plan format already carries per-track temporal ranges, so the information
  needed to distinguish the two is present and unused. Filed as RM-361.

- **The Docker runtime has never been executed.** The Dockerfile builds FFmpeg 9.0.1 from a
  SHA-256-verified source tarball on a digest-pinned `python:3.12-slim` base, which is careful work
  that is only ever reviewed statically, because the Linux engine was not running in any prior build
  session. It is also the project's only second environment: the entire 1,872-test suite runs on one
  Windows box, on one Python, with no CI by policy. Filed as RM-362.

- **Test configuration is entirely implicit.** There is no `pytest.ini`, no `conftest.py` at the
  root or in `tests/`, and `pyproject.toml` carries only `[tool.ruff]`. No `testpaths`, no
  `addopts`, no timeout, no registered markers, so invocation is always the positional
  `python -m pytest tests -q` written into `build_exe.bat`. A single hung test takes the whole
  1,872-test suite with it, and nothing in the repo records that the intended entry point is
  `tests`. Filed as RM-365.

- **Documentation gaps.** RM-337 already covers the 104 KB mixed-ending README. Beyond it, the
  README's release matrix at `:128-131` documents an artifact set that has never been published, and
  `docs/architecture.md` carries a hand-maintained version header that is one of four surfaces a
  version bump must touch, alongside `gui/config.py:139`, the README badge, and
  `installer/vsr.nsi:46-48`. That `.nsi` file is the only version string not derived from
  `APP_VERSION` and the only one that can silently fall behind.

### Categories reviewed and deliberately not filed

- **Accessibility.** MSAA dynamic annotation, focus traversal, the high-contrast theme, the 100 to
  200 percent scaling matrix and the packaged probes are all shipped. The two real remaining gaps,
  UIA providers for the custom Tk Canvas controls and live Narrator or NVDA evidence, are already in
  `Roadmap_Blocked.md` and blocked on an isolated screen-reader session, not on effort. RM-344's
  light theme and RM-340's multi-monitor DPI work remain the open a11y-adjacent items. Nothing new.
- **i18n and l10n.** RM-327 already carries both halves that matter: the extraction hole for strings
  assigned to locals, and the fact that no real catalog ships. Nothing to add.
- **Observability.** The support bundle, the FFmpeg command ring buffer, execution provenance and
  the batch report cover diagnosis adequately for a local desktop tool. The gap is reachability, not
  content, which is RM-358.
- **Multi-user and mobile.** Out of the product's shape. It is a single-user Windows desktop
  application with a per-user settings path, and there is no server, no account, and no phone.
- **Migration and upgrade.** `gui/config.py:1073` migrates settings by schema version, the v3
  sidecar handles output identity across versions, and `backend/update_check.py` reaches the release
  page. No gap found.
- **Plugin ecosystem.** `backend/inpainter_registry.py` and `RegisteredMode` already allow opt-in
  ONNX and diffusion backends to exist outside the core enum. Adequate for the number of adapters
  in play.
- **Distribution and packaging.** RM-335 (signing), RM-346 (long paths and AppUserModelID) and
  RM-337 (README split) already hold this ground. RM-353 adds the missing piece, which is actually
  publishing what the build system now produces.

## Rejected Ideas

- **Invisible-watermark and C2PA provenance stripping** (source: wiltodelta/remove-ai-watermarks,
  which ships `verify-openai-synthid`, `invisible` and `metadata` commands). Removing a
  cryptographic provenance manifest is a different product with a different ethical posture from
  repairing pixels a caption destroyed. Rejected on purpose, and RM-364 exists partly so the
  boundary is written down.
- **SEDiT adapter.** Re-checked on 2026-09-04: arXiv:2605.14894 v1, submitted 2026-05-14, no
  revisions, project page only, still no code and no weights. Unchanged from the prior pass, and it
  stays in `Roadmap_Blocked.md`.
- **DualEraser, formerly GenEraser, adapter** (arXiv:2605.30045, revised 2026-08-13). Targets object
  and effect removal such as smoke and light effects, and reports gains on ROSE and VOR-Eval, not on
  text. No weights referenced. Wrong problem, and no artifact to adapt.
- **SAM 3 and SAM 3.1 text-prompt segmentation.** Already recorded in `Roadmap_Blocked.md` items 67
  and 115, because the custom Meta SAM licence carries a redistribution clause incompatible with MIT
  redistribution. SAM 3.1 (2026-03-27) does not change the licence. Not re-filed.
- **A hosted or REST mode, and a translation or dubbing pipeline** (source: GhostCut's per-minute API
  workflow). Already on the ROADMAP "Rejected on purpose" list. Adding it would forfeit the one
  property no commercial competitor can match, which is that nothing leaves the machine.
- **GitHub Actions CI to get videowipe's Linux and multi-Python coverage.** Already on the "Rejected
  on purpose" list. RM-362 gets a second environment locally through the Docker runtime the project
  already ships, which is the same coverage without the policy conflict.
- **Copying videowipe's WipePlan review model or its editable pre-run regions.** Parity already
  exists in `backend/track_plan.py` (with `--plan-out` and `--plan-in`) and in the region and mask
  editors. What is worth taking from videowipe is the keep default, which is RM-361, and the
  detection baseline, which is already inside RM-331's acceptance.
- **A Magic-Mask-style render-in-place matte cache** (source: DaVinci Resolve 21). Parity exists in
  `backend/matte_interchange.py` and the frozen-matte manifests. Not a gap.
- **Bumping numpy past the `<2.5.0` pin**, with numpy 2.5.2 current. The pin is deliberate: the
  reference corpus hashes are only reproducible on the reviewed CPU profile with numpy 2.4.6 and
  OpenCV 5.0.0.93, so moving it invalidates every committed baseline for no benefit.
- **Bumping huggingface_hub past the 1.29.0 floor.** 1.30.0 (2026-09-03) adds
  `hf jobs scheduled ls` filtering flags and touches nothing this project uses. The 1.29.0 floor
  already captures the Xet connection-info cache fix that stops large downloads hitting a 429.

## Sources

Repository and tracker:
- https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/10
- https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/11
- https://github.com/SysAdminDoc/VideoSubtitleRemover/discussions/8
- https://github.com/SysAdminDoc/VideoSubtitleRemover/releases/tag/v3.40.0

Competitors and adjacent projects:
- https://github.com/YaoFANGUK/video-subtitle-remover
- https://github.com/KKenny0/videowipe/releases
- https://github.com/Sanster/IOPaint
- https://github.com/wiltodelta/remove-ai-watermarks
- https://github.com/sczhou/ProPainter
- https://github.com/lixiaowen-xw/DiffuEraser
- https://github.com/TencentARC/VideoPainter
- https://github.com/ymy-k/Hi-SAM
- https://github.com/geekyutao/Inpaint-Anything
- https://github.com/facebookresearch/sam3
- https://github.com/Agions/Distill
- https://github.com/sueun-dev/video-text-eraser
- https://github.com/topics/subtitle-removal
- https://github.com/topics/hardcoded-subtitles
- https://github.com/topics/video-inpainting
- https://github.com/zengyh1900/Awesome-Image-Inpainting

Commercial and market:
- https://jollytoday.com/subtitle-removal/
- https://coldiq.com/tools/ghostcut
- https://www.cgchannel.com/2026/06/blackmagic-design-releases-davinci-resolve-21-0/
- https://cutsio.com/blog/davinci-resolve-21-magic-mask-render-in-place
- https://www.videoproc.com/video-editor/remove-subtitles-from-mp4.htm
- https://reccloud.com/best-video-subtitle-remover-tools-2026.html
- https://reccloud.com/ai-burned-in-subtitle-removal-large-video-archives.html
- https://forum.videohelp.com/threads/418726-Is-there-a-way-to-remove-hardcoded-subtitles-without-cropping

Research:
- https://arxiv.org/abs/2605.14894
- https://arxiv.org/abs/2605.30045
- https://arxiv.org/html/2603.21901
- https://arxiv.org/abs/2511.22499
- https://arxiv.org/abs/2605.14534
- https://arxiv.org/pdf/2510.02787
- https://arxiv.org/html/2503.05639v1
- https://huggingface.co/datasets/cyberagent/OTR
- https://github.com/YigitEkin/BeyondMasks

Dependencies, security and platform:
- https://github.com/pytorch/pytorch/releases/tag/v2.14.0
- https://download.pytorch.org/whl/cu130/torch/
- https://download.pytorch.org/whl/cpu/torchvision/
- https://cve.threatint.com/CVE/CVE-2026-65918
- https://dailycve.com/pytorch-torchvision-out-of-bounds-heap-read-cve-2026-65918-high-dc-aug2026-1751/
- https://github.com/huggingface/huggingface_hub/releases/tag/v1.29.0
- https://github.com/huggingface/huggingface_hub/releases/tag/v1.30.0
- https://pypi.org/project/onnxruntime/
- https://pypi.org/project/onnxruntime-directml/
- https://pypi.org/project/easyocr/
- https://pypi.org/project/pysubs2/
- https://ffmpeg.org/download.html
- https://ffmpeg.org/security.html
- https://huggingface.co/Carve/LaMa-ONNX
- https://huggingface.co/opencv/inpainting_lama
- https://ai.meta.com/blog/segment-anything-model-3/
- https://docs.ultralytics.com/models/sam-3
- https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/smartscreen-reputation
- https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation
- https://github.com/PaddlePaddle/PaddleOCR/releases/tag/v3.7.0

## Open Questions

- **Does the maintainer want LaMa to remain a user-facing algorithm at all?** RM-354 assumes yes and
  fixes acquisition. The alternative is removing LaMa from the picker and keeping it only as the
  refinement stage inside the hybrid, which is cheaper and equally honest, and which would collapse
  RM-354 and RM-355 into a single deletion. This needs a product call, not more research.
- **Is a 3.1 GB NVIDIA ZIP acceptable as the recommended download, or should the CUDA runtime be
  fetched on first run?** RM-353 ships what exists. If the answer is fetch it, that is a different
  and larger item, and it interacts with the offline-first privacy promise.
- **Which single locale ships first for RM-327?** Simplified Chinese is assumed from the upstream
  user base, but nobody has asked this project's own users, and Discussion #9 exists for exactly
  that question with zero replies.
