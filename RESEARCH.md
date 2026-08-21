# Research -- Video Subtitle Remover Pro
Date: 2026-08-21 -- replaces all prior research.
Previous passes: 2026-08-20 (evening), 2026-08-20 (morning), 2026-08-11.

## Executive Summary

[Verified] VSR Pro at v3.35.0 (HEAD `8815b5e`) is a mature local hardsub remover:
TBE + LaMa, isolated jobs, provenance sidecars, a local release gate, and an empty
open-issue tracker. The 2026-08-20 evening pass already filed RM-277..289 (CPython
advisories, GUI import smoke, classified failures, i18n lint, quality pooling, HWND
annotation, donor-video, translation policy, FFmpeg command log, NSIS quiet uninstall,
reproducible-build envelope, PyInstaller 6.22.1). Those items are still open in code
and are **not re-filed here**. This pass's new work is the repo's own tracker (zero
open issues, seven closed) plus two facts that yesterday got wrong or never measured:
the sidecar still lies about which OCR engine ran, and CVE-2026-58049 is already in
FFmpeg 9.0.1.

Highest-value new directions:

1. [Verified] **The output sidecar's `engine` field is a package-presence probe, not
   the engine that ran.** `backend/batch_report.py:817-888` walks
   `importlib.util.find_spec` (`rapidocr` first) and writes that string even when
   `executionProvenance` on the same document records a different OCR stage. Issue #7
   was the user-visible form of this class of bug (GUI showed LaMa, STTN ran). The
   inpaint half was fixed (`backend/device_provider.py:181-190` now raises
   `InpainterUnavailableError`). The sidecar half was not.
2. [Verified] **CVE-2026-58049 is remediated on the current FFmpeg floor, but the app
   still reports it as unfixed on every "safe" 9.0.1 probe.** `n9.0.1`'s
   `libavcodec/rasc.c` history contains `f8d7795` (2026-07-21),
   "Check that 32-bit DLTA accesses stay within the row", cherry-picked from
   `11ff18a6`. NVD and GHSA-mjxr-6gqf-w78h (last updated 2026-08-11, the day before
   9.0.1) still list no patched versions -- that lag is why yesterday parked this.
   `tests/test_security_checks.py:115-116,164` locks the stale "unfixed" set in.
3. [Verified] **Fade-aware mask extension is still absent**, while it is the one
   WatermarkRemover-AI flag (`--fade-in`/`--fade-out`) VSR has no equivalent for, and
   the JET dehardsubbing guide names fading hardsubs as the case where diff-based
   masks fail. VSR's translucency unmix is steady-state alpha, not onset/offset.
4. [Verified] **README Detection / Troubleshooting copy still describes a 3.8-era
   picker.** `README.md:330-334` omits Surya and VLM from the pin list;
   `:351-358` still presents `VSR_VLM_OCR` as the activation path after v3.35.0 put
   those engines in the GUI picker; `:1260-1262` still says "Install PaddleOCR for
   best detection accuracy" while RapidOCR is priority 1.
5. [Verified] **This repo's issue tracker is empty of open bugs.** Demand lives in
   the seven closed issues and in YaoFANGUK/video-subtitle-remover's still-untriaged
   185-issue tracker. Do not invent features from an empty inbox.

## Product Map

- **Core workflows:** [Verified] queue images/videos/folders or watch a folder;
  automatic / scripted / keyframed / manual masks; reviewable track plans; isolated
  resumable batch; quality review with selective rerun; frozen-matte reruns.
- **Output workflows:** [Verified] payload-preserving encode, SRT/WebVTT export and
  translation, quality evidence, reproducibility sidecars, NLE interchange, mattes.
- **Personas:** [Likely] privacy-sensitive creators, archivists, localization
  operators, unattended batch users. Confirmed by issue #3 (portable/Docker) and
  discussion #9 (hardware-coverage ask) -- both from the maintainer, not a crowd.
- **Platforms:** [Verified] Windows 10/11 GUI+CLI, Python 3.11-3.14, unsigned
  PyInstaller onedir + NSIS 3.12 + portable ZIP, CPU / CUDA-12.8 / DirectML, Linux
  Docker CPU CLI. MIT. GitHub Actions removed (`c4a4617`).
- **Data flow:** [Verified] decode -> OCR cascade (RapidOCR > PaddleOCR > Surya >
  EasyOCR > OpenCV, plus opt-in VLM) -> tracking -> TBE/LaMa/cv2 -> finishing ->
  FFV1 -> final encode -> sidecar + quality report.

## Competitive Landscape

- **YaoFANGUK/video-subtitle-remover** (12,432 stars, last push 2026-06-30) --
  [Verified] Still dormant. 185 open issues, zero on `enhancement`/`help wanted`.
  Only post-2026-08-01 issue is #250 (2026-08-17, title-only, single-image crash).
  **Learn:** install/throughput still dominate that tracker. **Avoid:** copying #250
  without a VSR repro -- VSR's image path is covered by skip-detection tests.
- **KKenny0/videowipe** -- [Verified] Issues still disabled; last commit 2026-08-13.
  WipePlan equivalent shipped here as RM-275. **Avoid:** GPL-3.0.
- **silent-commit/CLEAR** -- [Verified] No commits since 2026-05-25. Apache-2.0 tag
  vs HF "research purposes only" / Wan community license. **Avoid:** shipping it.
- **Subtitle Edit 5.2.0-beta19** (2026-08-20) -- [Verified] Only live adjacent
  tracker. New: #13927 strip ASS tags before AI translate; #13923 Translation Mode.
  VSR re-burns `.ass` (`backend/post_restore.py:256`) but translates SRT/WebVTT, so
  #13927 is not a VSR gap. **Avoid:** Google Lens / cloud OCR.
- **D-Ogi/WatermarkRemover-AI** -- [Verified] Still the source for fade-in/out mask
  extension. Last push 2026-03-29. **Learn:** that flag. **Avoid:** watermark-of-the-month.
- **IOPaint** -- [Verified] Still archived. No successor.
- **Commercial (EchoSubs, HitPaw, Vmake)** -- [Verified] Unchanged: metered, no
  named models. Unmetered local batch remains the contrast.
- **LosslessCut** -- [Verified] One new issue (#3028 media keys). Last-FFmpeg-commands
  still the triage pattern (already RM-286).
- **D2DF (bigD233/D2DF)** -- [Verified] Correction of 2026-08-20: Apache-2.0 code and
  HF weights have existed since 2026-07-16, not paper-only. Still CogVideoX-5B-I2V.
  **Avoid:** filing it as actionable here; VRAM puts it with VOID/ROSE.

## Reported Issues

Tracker: `SysAdminDoc/VideoSubtitleRemover`. Issues enabled, discussions enabled, not
a fork. `gh issue list --state open` returned **[]** on 2026-08-21. Open PRs: none.
Discussions #8 (state of project, still titled v3.31.0) and #9 (hardware ask) are
maintainer posts; #9 has zero replies.

Worth knowing, all closed, none to re-open as bugs:

- **#7** (bug, closed 2026-07-29) -- every inpaint model produced identical output.
  Root cause: `create_inpainter` failed open to STTN. **Fixed** in `cfc4b03` /
  `backend/device_provider.py:181-190` (raises `InpainterUnavailableError`). The
  owner's follow-up enhancement ("draw a region and have LaMa honor it for the whole
  clip") is `sttn_skip_detection` + `subtitle_area`, already in the GUI. Residual
  work is the **sidecar `engine` lie** (item 1 above), not a silent STTN fallback.
- **#6** (closed 2026-07-29) -- AI-generated live-action, residue regardless of
  parameters. Empty body. Closed as current inpaint ceiling, not a silent-fallback
  confirm. Do not file "fix AI-video residue"; that is the blocked diffusion lane.
- **#5** (closed 2026-06-25, reporter said unresolved, then fixed in `8a2e29d`) --
  mask selection / Inspect empty state. Shipped in 3.17.2.
- **#4** (closed 2026-06-25) -- missing `filedialog` import in `gui/widgets.py`.
  Shipped (`335177f`).
- **#3** (closed 2026-06-25) -- frozen 3.16.0 would not load; reporter asked for
  portable/Docker. RapidOCR collect-data shipped; portable ZIP and Docker CPU CLI
  exist in 3.35.0.
- **#2** (closed 2026-06-03, not planned) -- "info stealer" AV heuristic. No
  signature or VirusTotal link. README already states the build is unsigned and
  SmartScreen may prompt (`README.md:95-96,119`). Do not re-file signing.
- **#1** (closed 2026-06-03) -- RTX 50-series / TV color range. Blackwell `cu128`
  path and `color_range` preservation shipped in 3.16.0. Torch 2.13 Windows wheels
  are **cu130 only** (no cu128); do not switch `setup.py:487` until the NVIDIA lane
  pin moves -- that is a GPU-host change, not this pass.

Stale / not acted on: discussion #8 still advertising 3.31.0 while the tag is
3.35.0 -- operator copy on GitHub, not a code item.

## Security, Privacy, and Reliability

- [Verified] **CVE-2026-58049 is in 9.0.1 and still listed as unfixed.**
  `backend/security_checks.py:58`, `backend/ffmpeg_profiles.py:191`,
  `tests/test_security_checks.py:115-116,164`. Same class of user-facing advisory
  error as RM-277. Floor stays 9.0.1; no 9.0.2 exists
  (ffmpeg.org/download.html, 2026-08-21).
- [Verified] **CPython advisory tuple still flat** -- RM-277, still open
  (`backend/security_checks.py:110-115`).
- [Verified] **PyInstaller floor still 6.10.0** -- RM-289. 6.22.2 shipped
  2026-08-17 (`#9508` onefile symlink false-positive). VSR is onedir
  (`VideoSubtitleRemoverPro.spec:91-109`); GHSA-9fxf-4qw3-ghmr still does not
  apply. Do not invent a new item; optionally raise RM-289's floor 6.22.1 -> 6.22.2.
- [Verified] **Default `onnxruntime-gpu` 1.29.0 on PyPI depends on CUDA 13 extras**
  (`nvidia-cuda-nvrtc~=13.0`). The NVIDIA-lane cap
  `onnxruntime-gpu>=1.26.0,<1.27.0` in `setup.py:41-44` remains the correct CUDA-12
  pin. Do not lift it because CPU ORT is 1.29.0.
- [Verified] **109 silent `except Exception: pass` sites remain.** New user-facing
  one: `_detection_engine_status` (`batch_report.py:825-826`). Surya auto-probe
  (`detection.py:593-594`) still swallows non-ImportError.
- [Verified] **External-code paths still contained** (VapourSynth `exec`, env-only
  `VSR_EXTERNAL_INPAINTER`). Recovery (atomic_replace, checkpoints, job objects)
  unchanged and adequate.

## Architecture Assessment

- [Verified] `process_video` frame loop and GPU adapter work stay in
  `Roadmap_Blocked.md`. `gui/region_controller.py:74-721` is still 648 lines; do not
  split it until RM-278 lands -- there is no collected GUI regression net.
- [Verified] Clean-reference is still still-plate-only -- RM-283.
- [Verified] Test holes beyond RM-278: `_detection_engine_status` has no test
  (sidecar tests only `assertIn("engine", sidecar)` at
  `tests/test_hardening_encode_io.py:1434`). `_clean_ref_mixin` / `_srt_mixin`
  remain lightly covered. `backend/ocr_variants.py` is adequately tested.
- [Verified] i18n: RM-280 covers `text=` sinks. Additional English-only plurals at
  `gui/region_controller.py:367-375,1610-1611,1676-1677` belong as a note on RM-280,
  not a second item.

## Rejected Ideas

- **Re-open #7 as a silent STTN fallback** -- measured; `create_inpainter` now fails
  closed. Source: issue #7 plus `backend/device_provider.py:181-190`.
- **File #6 as "better AI-video inpainting"** -- capability ceiling; CLEAR/ROSE/VOID
  remain blocked. Source: issue #6 close comment.
- **YaoFANGUK #250 single-image crash as a VSR bug** -- no VSR repro; image
  skip-detection path is tested. Source: upstream #250 (title-only).
- **Lift `onnxruntime-gpu` to 1.29.0** -- PyPI 1.29.0 GPU extra is CUDA 13.
  Source: pypi.org/project/onnxruntime-gpu/1.29.0/
- **Switch Blackwell install to cu130 / torch 2.13** -- 2.13 has no cu128 Windows
  wheels; current pin is 2.11.0+cu128. GPU-host change. Source: download.pytorch.org
  cu128 vs cu130 indexes; `setup.py:479-495`.
- **D2DF adapter** -- Apache-2.0 and weights exist (correction); CogVideoX-5B still
  exceeds consumer VRAM. Source: github.com/bigD233/D2DF.
- **Subtitle Edit #13927 ASS-tag stripping before translate** -- VSR does not
  AI-translate ASS cue internals. Source: backend/webvtt.py, post_restore.py.
- **CLEAR / ROSE / EraserDiT / VOID / SEDiT / EffectLearner / Qt / REST / Mac /
  GitHub Actions / Sigstore / winget submission / code signing** -- unchanged from
  2026-08-20; reasons stand. EffectLearner still "coming soon" (issue #1, 2026-08-08).
- **VapourSynth exec hardening, preset trust-gating, queue crash recovery** --
  already implemented. Re-verified 2026-08-21.

## Sources

### This repo
- https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/1
- https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/2
- https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/3
- https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/4
- https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/5
- https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/6
- https://github.com/SysAdminDoc/VideoSubtitleRemover/issues/7
- https://github.com/SysAdminDoc/VideoSubtitleRemover/discussions/8
- https://github.com/SysAdminDoc/VideoSubtitleRemover/discussions/9

### OSS landscape
- https://github.com/YaoFANGUK/video-subtitle-remover/issues/250
- https://github.com/silent-commit/CLEAR
- https://github.com/KKenny0/videowipe
- https://github.com/D-Ogi/WatermarkRemover-AI
- https://github.com/SubtitleEdit/subtitleedit/releases/tag/v5.2.0-beta19
- https://github.com/SubtitleEdit/subtitleedit/issues/13927
- https://github.com/bigD233/D2DF
- https://github.com/Kunbyte-AI/ROSE/issues/15
- https://github.com/mifi/lossless-cut/issues/3028
- https://jaded-encoding-thaumaturgy.github.io/JET-guide/master/filtering/situational/dehardsubbing/

### Security / dependencies
- https://github.com/FFmpeg/FFmpeg/commit/f8d7795dcca36a4dd412e89cbd83e3dfec1e0d81
- https://github.com/FFmpeg/FFmpeg/commits/n9.0.1/libavcodec/rasc.c
- https://github.com/advisories/GHSA-mjxr-6gqf-w78h
- https://nvd.nist.gov/vuln/detail/CVE-2026-58049
- https://ffmpeg.org/download.html
- https://github.com/pyinstaller/pyinstaller/releases/tag/v6.22.2
- https://github.com/pyinstaller/pyinstaller/security/advisories/GHSA-9fxf-4qw3-ghmr
- https://pypi.org/project/onnxruntime-gpu/1.29.0/
- https://github.com/microsoft/onnxruntime/releases
- https://osv.dev/vulnerability/CVE-2026-11940
- https://osv.dev/vulnerability/CVE-2026-2297

### Prior research (vault)
- `C:\Obsidian\Notes\Research\Subtitle Removal Ecosystem 2026-08-20.md`
- `C:\Obsidian\Notes\Research\Packaging, a11y and i18n mechanics 2026-08-20.md`

## Open Questions

- [Needs live validation] Does HWND annotation produce useful NVDA speech? Unchanged
  from 2026-08-20; still the sufficiency gate for RM-282.
- [Needs live validation] Donor-video alignment on real footage (RM-283). Unchanged.
- [Blocked on operator] Weblate / SignPath / discussion #8 version string. Not code.
- [Assumption] Whether 9.0.0 also contains `f8d7795`. It predates the 9.0.1 tag and
  sits on the 9.0 branch history (2026-07-21 vs 9.0.0 on 2026-08-04), so it likely
  does -- RM-291 does not need that answer, because the floor remains 9.0.1.
