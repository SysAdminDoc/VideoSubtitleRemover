# Research - Video Subtitle Remover Pro

## Executive Summary

This continuation pass re-checked the current tree, recent git history, the
existing ROADMAP, and external signals through June 14, 2026. Video Subtitle
Remover Pro is still strongest when it leans into its Windows-first,
local/offline workflow: OCR/VLM detection, TBE/LaMa/OpenCV DNN inpainting,
cache inspection, DirectML/CUDA/CPU fallbacks, FFmpeg remuxing, release
verification, and a polished tkinter UI are already ahead of most direct OSS
and cloud competitors.

The next high-value work is not another default model lane. The roadmap already
has enough detector, segmentation, and diffusion research items. The remaining
leverage is trust, release integrity, dependency drift control, and documentation
accuracy. This pass preserves the earlier backlog and adds four distinct gaps:
the optional GitHub Releases update check should follow REST API etiquette and
rate-limit handling; release workflow tooling should not execute mutable,
unverified downloads; architecture documentation is stale after the module split
and current codec/test additions; and release artifacts should carry a standard
SBOM/provenance evidence path.

Priority order from the combined audit:

- P1: remove the strict release ROADMAP version-marker requirement; verify
  release workflow executable downloads and action pinning policy; fix EDL
  non-ASCII output; add installer subprocess timeouts; move startup probes off
  the GUI thread; add first-run model-download progress.
- P2: validate PaddleOCR 3.7 / PP-OCRv6 compatibility; harden the startup update
  check; sync VVC docs; enforce source ASCII in CI; surface settings reset and
  all-unsupported drops; broaden core-pipeline tests; add dependency caps and
  diagnosable logging; refresh architecture docs.
- P3: add SBOM/provenance release evidence and keep ROSE/MiniMax-style
  inpainting adapters on the opt-in research bench.

## Current Product Map

- Core workflows: import files/folders, classify soft vs hard subtitles, OCR or
  VLM-detect text, generate masks, inpaint via AUTO/TBE/LaMa/registry modes,
  mux audio/subtitles, and produce optional batch, quality, and release reports.
- User personas: Windows creators repurposing social clips, archivists cleaning
  local libraries, localizers extracting/replacing subtitles, batch operators
  running overnight queues, and power users benchmarking local detectors and
  inpainters.
- Platforms and distribution: Python 3.10-3.13 source checkout, PyInstaller
  onedir build, NSIS installer, GitHub Actions release workflow, winget
  submission, and CUDA/DirectML/CPU execution paths.
- Key integrations: FFmpeg/ffprobe, RapidOCR, PaddleOCR, optional
  VLM/SAM/diffusion adapters, ONNX Runtime, OpenCV DNN, PyTorch, Windows
  taskbar/sound affordances, APPDATA settings/checkpoints/cache/logs, optional
  NLE sidecars.
- Non-goals that still look correct: cloud processing, telemetry by default,
  Docker as the primary user path, subscription mechanics, multi-user shared
  state, and bundled GPL detectors.

## Competitive Landscape

- **YaoFANGUK/video-subtitle-remover** remains the closest OSS reference. It
  validates broad packaging and user demand, but VSR's advantage is stronger
  release verification, local reliability work, and a clearer Windows GUI.
- **VideOCR** validates simple burned-in subtitle OCR workflows. VSR should
  borrow its simplicity signal, not its optional cloud OCR posture.
- **IOPaint / lama-cleaner** is archived but still useful as a model-zoo UX
  reference. VSR should keep explicit download/progress affordances and avoid
  binding default paths to unmaintained stacks.
- **ProPainter / DiffuEraser / MiniMax-Remover / ROSE / SVOR-class research**
  confirms that opt-in bench adapters and temporal metrics are worth tracking.
  These are too heavy or research-stage for default routes until weights,
  licenses, VRAM behavior, and failure modes are bounded.
- **GhostCut / RecCloud / HitPaw / Vmake / AniEraser / CreatOK** sell one-click,
  preview-first, browser/mobile subtitle removal. VSR already beats them on
  offline privacy and free batch work; it should continue borrowing progress
  transparency, clear preview confidence, and low-friction first-run guidance.
- **InpaintDelogo / VideoSubFinder / Subtitle Edit community threads** reinforce
  that users want detection, extraction, removal, and temporal refinement in one
  maintained tool rather than hand-built virtualdub/filter pipelines.

## Category Coverage Audit

- Security: dependency floors, advisory notes, cache inventory, adapter hash
  checks, and crash-report opt-in exist. Open gaps are release workflow
  executable verification, action pinning policy, update-check rate-limit
  handling, SBOM/provenance artifacts, and the libpng wheel blocker.
- Privacy/offline: strong. Processing stays local; cloud API defaults remain a
  rejected idea. The update check is optional but should be polite, bounded, and
  cache-aware.
- Accessibility: scaffolding exists, but real UIA/NVDA/Narrator status
  announcement work remains in ROADMAP #95.
- Internationalization and RTL: scaffolding exists, but gettext extraction and
  mirrored Arabic/Hebrew UI remain ROADMAP #97/#98. The EDL ASCII bug is a
  current i18n reliability defect.
- Observability: rotating logs, quality reports, batch reports, cache inspector,
  and release verification are strong. Weak spots are swallowed processing
  exceptions, silent settings resets, first-run model downloads, and update-check
  diagnostics.
- Testing: release and hardening coverage is broad. Missing coverage remains
  detection-cascade fallbacks, tracking single-frame misses, FFmpeg timeout
  fallback paths, non-ASCII NLE sidecars, source-ASCII invariant, PP-OCRv6
  payloads, and release workflow supply-chain assertions.
- Documentation: README is stale for VVC/H.266, and docs/architecture.md is
  stale for the GUI split, backend modules, test map, VVC, cache/update/security
  modules, and "current as of" commit marker.
- Distribution and upgrades: PyInstaller/NSIS/winget workflow exists, but the
  workflow still executes a mutable wingetcreate URL and stores pip evidence
  instead of a standard SBOM. The app update check has a timeout, but not ETag,
  API-version, User-Agent, or rate-limit backoff semantics.
- Plugin ecosystem: the in-process registry is the right constraint. Filesystem
  plugin auto-discovery can stay later because it increases support and security
  surface.
- Mobile and multi-user: mobile remains speculative; multi-user collaboration
  remains explicitly out of scope for a single-user offline desktop app.

## Security, Privacy, and Reliability Findings

- Verified: dependency floors still target the known high-risk image/model
  surfaces: requirements require OpenCV >= 4.12.0 and Pillow >= 12.2.0, and the
  docs reference torch >= 2.10.0. NVD/GitHub advisories support libpng 1.6.54
  and PyTorch 2.10.0 as relevant fixed points.
- Verified: ROADMAP #111 remains correctly blocked on opencv-python shipping a
  wheel with libpng >= 1.6.54. The current workflow warning is appropriate while
  that wheel is unavailable.
- Verified: strict release verification conflicts with roadmap hygiene because
  .github/workflows/build.yml requires ROADMAP.md to contain the release version
  even though ROADMAP is defined as a remaining-work backlog only.
- Verified: backend/nle_sidecar.py writes EDL sidecars as ASCII while embedding
  source and cleaned filenames, so CJK filenames can still raise
  UnicodeEncodeError.
- Verified: setup.py still has venv and pip subprocess calls without timeouts,
  while probe calls already carry timeouts.
- Verified: gui/app.py still runs GPU, AI-engine, and FFmpeg detection inline
  during MainWindow construction, so slow shell probes can delay first paint.
- Verified: corrupt settings reset to defaults with log-only feedback, and
  malformed saved regions are silently dropped.
- Verified: an all-unsupported drag-drop does not route through the import
  summary path, so the user gets no visible "skipped unsupported files" notice.
- Verified: PaddleOCR remains a live drift risk. Install paths float
  paddleocr>=3.0.0, README still describes PP-OCRv5, and PaddleOCR 3.7.0 /
  PP-OCRv6 shipped on June 11, 2026.
- Verified: VVC/H.266 is present in CLI, GUI, and encoder probing, but README
  still lists only H.264/H.265/AV1 in key user-facing places.
- New: backend/update_check.py makes a bounded Releases API request, but only
  sends Accept. GitHub's REST guidance calls for avoiding unnecessary polling,
  handling rate-limit headers, using conditional requests where appropriate, and
  not ignoring repeated errors. The local research pass hit unauthenticated API
  rate limits, which makes this practical rather than theoretical.
- New: .github/workflows/build.yml downloads wingetcreate.exe from a mutable
  aka.ms "latest" URL and executes it. Release tooling should be versioned and
  hash/signature verified before execution, and workflow actions should have a
  documented pinning/upgrade policy.
- New: release-verification.json records useful facts, but it is not a standard
  CycloneDX/SPDX SBOM and does not provide a provenance/attestation path for
  binaries. GitHub Actions supports SBOM attestations, with plan caveats for
  private repositories, so a local SBOM artifact plus optional attestation is the
  pragmatic next step.

## Architecture Assessment

- The backend module split is holding. New work should stay near existing
  backend modules: detection/OCR compatibility, IO/remux/encoder, cache
  inventory, update checks, inpaint registry, quality/reporting, and release
  verification tests.
- gui/app.py is still large and owns startup probes, import summaries, queue
  workflow, dialogs, processing orchestration, and report writing. Extract only
  when fixing a concrete defect; broad GUI rewrites are lower value than the
  current reliability backlog.
- backend/paddle_compat.py handles PaddleOCR 2.x/3.x constructor and result
  shapes, but it has no PP-OCRv6-specific evidence. Compatibility tests should
  land before changing default OCR assumptions.
- docs/architecture.md no longer matches the current tree. It still describes
  older GUI ownership, an older backend map, a narrow test map, and pre-VVC
  codec support. This is a contributor-risk issue because future work will waste
  time rediscovering the actual module boundaries.
- The release workflow is now a major product surface. It should be tested as
  code: no stale ROADMAP version markers, no mutable executable downloads, no
  missing SBOM, and explicit release-tool evidence.

## Gap Analysis Against Existing ROADMAP

Already covered and should not be duplicated:

- New detector/model lanes: Florence/Qwen, PaddleOCR-VL, PaddleOCR-VL via
  llama.cpp, SAM 3.1, SAM 2/3, MatAnyone, CoTracker, mask-free subtitle erasure.
- New diffusion/video inpainting lanes: ProPainter, DiffuEraser, VACE,
  VideoPainter, CoCoCo, EraserDiT, FloED, ROSE, MiniMax.
- Format and acceleration lanes: HDR/10-bit, AV1/VP9 verification, hardware
  decode, RIFE, OpenCV 5 DNN.
- Existing research-driven reliability items: EDL UTF-8, setup timeouts,
  off-main-thread probes, first-run model download progress, ROADMAP release
  marker removal, source-ASCII guard, settings corruption notice, unsupported
  drop feedback, core fallback tests, dependency caps, swallowed-exception
  logging, subprocess teardown, PaddleOCR 3.7 validation, VVC docs sync.

Distinct additions from this continuation pass:

- Harden the startup GitHub Releases update check with explicit headers,
  conditional requests, 304 handling, rate-limit backoff, and tests.
- Verify release workflow tooling and action supply-chain policy, especially the
  wingetcreate executable download and mutable "latest" URL.
- Refresh docs/architecture.md for the current module map, tests, VVC, and
  release/security/cache/update modules.
- Add a release SBOM/provenance evidence artifact, with GitHub attestation used
  only where repository plan and visibility support it.

## Rejected Ideas

- **Cloud API processing as default**: conflicts with offline/no-upload
  positioning and weakens the clearest differentiator against commercial tools.
- **Authenticated user token for the desktop update check**: overkill for an
  optional public-release check and creates credential-handling risk. Prefer
  cache, conditional requests, backoff, and bounded failure.
- **Make mask-free SEDiT/CLEAR default now**: promising, but not yet proven as a
  local, permissively licensed, bounded-dependency implementation.
- **Replace OCR cascade with VLM-only detection**: mismatches current speed and
  offline goals. Keep VLMs opt-in and benchmarked.
- **Add more diffusion adapters before reliability work**: the roadmap already
  has many heavy model lanes; current wins are release, docs, and fallback
  correctness.
- **Mandatory GitHub artifact attestations for every private release**: useful,
  but GitHub documents plan/visibility constraints. Generate a local SBOM first
  and add attestation when the repository plan supports it.
- **Rewrite the GUI in Tauri now**: possible long-term, but it does not address
  current release, dependency, startup, or documentation risks.
- **Bundle GPL detectors by default**: Surya and similar GPL components should
  remain opt-in to preserve the MIT distribution stance.

## Open Questions

- Does PaddleOCR 3.7.0 / PP-OCRv6 preserve the predict() payload keys currently
  parsed in backend/paddle_compat.py, or does it need another extractor branch?
- Should strict release verification replace the ROADMAP version check with a
  positive hygiene assertion that ROADMAP contains only incomplete work?
- Should VVC be documented as stable in README, or marked experimental until
  bundled/system FFmpeg libvvenc availability is recorded in release evidence?
- Where should the update-check ETag/Last-Modified state live: settings.json,
  a small cache file under APPDATA, or release-verification-style metadata?
- For private repository releases, should SBOM generation be mandatory while
  GitHub artifact attestation is optional based on plan support?

## Sources

OSS and adjacent tools:
- https://github.com/YaoFANGUK/video-subtitle-remover
- https://github.com/timminator/VideOCR
- https://github.com/Sanster/IOPaint
- https://github.com/sczhou/ProPainter
- https://github.com/lixiaowen-xw/diffueraser
- https://github.com/zibojia/MiniMax-Remover
- https://rose2025-inpaint.github.io/
- https://github.com/Purfview/InpaintDelogo
- https://github.com/JollyToday/GhostCut_Remove_Video_Text
- https://github.com/suhwan-cho/awesome-video-inpainting

Commercial and community:
- https://reccloud.com/remove-subtitle
- https://reccloud.com/best-video-subtitle-remover-tools-2026.html
- https://anieraser.media.io/remove-subtitles-from-video.html
- https://vmake.ai/remove-subtitles-from-video
- https://online.hitpaw.com/remove-subtitles-from-video.html
- https://www.creatok.ai/video-subtitle-remover
- https://www.reddit.com/r/handbrake/comments/1qbm1bs/what_is_the_easiest_way_to_remove_subtitles_from/
- https://github.com/SubtitleEdit/subtitleedit/discussions/9562

Models, standards, dependencies, advisories:
- https://github.com/PaddlePaddle/PaddleOCR/releases
- https://github.com/RapidAI/RapidOCR/releases
- https://github.com/microsoft/onnxruntime/releases
- https://github.com/opencv/opencv-python/issues/1186
- https://nvd.nist.gov/vuln/detail/CVE-2026-22801
- https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p
- https://github.com/fraunhoferhhi/vvenc/wiki/FFmpeg-Integration
- https://docs.github.com/en/rest/using-the-rest-api/best-practices-for-using-the-rest-api
- https://docs.github.com/en/actions/reference/security/secure-use
- https://docs.github.com/en/actions/how-tos/secure-your-work/use-artifact-attestations/use-artifact-attestations
- https://github.com/microsoft/winget-create/releases
- https://www.cisa.gov/resources-tools/resources/2025-minimum-elements-software-bill-materials-sbom
