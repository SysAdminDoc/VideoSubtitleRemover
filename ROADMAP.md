# Video Subtitle Remover Pro -- Roadmap

Remaining-work backlog only: every item below is NOT yet implemented.
Completed items are deleted from this file; history lives in CHANGELOG.md and git.

---

## Now

111. **CVE-2026-22801 libpng mitigation (remainder)** -- the runtime
     OpenCV build-info warning is implemented. Remaining work is blocked
     until opencv-python publishes a wheel with libpng >= 1.6.54.
     Priority: P1 security. Effort: S. Confidence: high.
     Acceptance criteria (remaining):
     - Pin is updated when a fixed opencv-python wheel is published.
     - CI `pip-audit` or equivalent catches this advisory.
     Source: https://github.com/opencv/opencv-python/issues/1186

---

## Next

### Detection

22. **Florence-2 / Qwen2.5-VL experimental detector** -- gated optional
    dependency; layout-aware, better multi-language, but heavy. Settings
    toggle for users wanting maximum accuracy on stylised text.
    Sources: https://qwenlm.github.io/blog/qwen2.5-vl/
    https://huggingface.co/microsoft/Florence-2-base
    https://github.com/opendatalab/OmniDocBench

23. **PaddleOCR-VL 0.9B detector tier** -- optional VLM-OCR with
    irregular-polygon bbox support; drop-in priority above PaddleOCR
    proper for CUDA users.
    Sources: https://huggingface.co/PaddlePaddle/PaddleOCR-VL
    https://arxiv.org/abs/2510.14528

113. **PaddleOCR-VL-1.5 llama.cpp detector tier** -- optional detector
     tier using PaddleOCR-VL via llama.cpp for CPU/edge deployment
     without CUDA; concrete implementation path for #22/#23.
     Priority: P2 detection. Effort: M. Confidence: medium-high.
     Acceptance criteria:
     - New tier in `backend/ocr_vlm.py` gated by `VSR_PADDLEOCR_VL=1`.
     - Works on CPU without CUDA dependency.
     - Falls back to standard PaddleOCR if llama.cpp is unavailable.
     Source: https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5

### Workflow / UX

31. **Subtitle-area drag-refinement in-GUI** -- replace the modal
    one-shot region selector with live-drag adjustment on the preview
    pane and instant mask re-render.

### Testing

54. **Reference-clip regression harness (remainder)** -- synthetic
    reference clips shipped; remaining: a `tests/clips/` corpus of 10-20
    edge-case clips (motion-heavy, karaoke, vertical JP, HDR, thin/thick
    fonts, dissolves) with a nightly GHA run comparing output hashes and
    PSNR/SSIM scores to committed baselines.

### Segmentation

115. **SAM 3.1 text-prompt mask refinement** -- optional mask refinement
     tier: after OCR detects subtitle regions, SAM 3 refines the mask to
     pixel-perfect boundaries using a text prompt. Gated behind
     `VSR_SAM3=1`.
     Priority: P2 segmentation. Effort: L. Confidence: medium.
     Acceptance criteria:
     - Opt-in refinement step; falls back to current dilation-based mask
       when SAM 3 is absent.
     - Benchmark mask precision vs current approach on reference clips.
     Dependencies: SAM 3 weights (~2 GB), torch runtime.
     Source: https://github.com/facebookresearch/sam3

---

## Later

### Real reference inpainters (opt-in scaffolds exist; full integrations remain)

59. **Real ProPainter** -- the actual `sczhou/ProPainter` reference with
    RAFT flow propagation, wired as `InpaintMode.PROPAINTER_REAL`; the
    TBE-based "ProPainter" stays the default fast path. ~400 MB weights
    on first use.
    Source: https://github.com/sczhou/ProPainter

60. **DiffuEraser** -- diffusion-based; beats ProPainter on content
    completeness and temporal consistency. Heavy (~5 GB weights, 8+ GB
    VRAM); opt-in.
    Sources: https://github.com/lixiaowen-xw/DiffuEraser
    https://arxiv.org/html/2501.10018

61. **Wan2.1-VACE** -- masked video-to-video editing; the 1.3B variant
    runs on consumer GPUs and beats ProPainter on motion coherence.
    Opt-in with auto-fetched weights.
    Sources: https://github.com/ali-vilab/VACE
    https://openaccess.thecvf.com/content/ICCV2025/papers/Jiang_VACE_All-in-One_Video_Creation_and_Editing_ICCV_2025_paper.pdf

62. **VideoPainter** -- dual-branch context encoder injecting background
    guidance into a pre-trained video diffusion transformer; handles
    videos over one minute with ID consistency.
    Source: https://arxiv.org/html/2503.05639v1

63. **CoCoCo** -- text-guided video inpainting with instance-aware region
    selector; research-bench alternative to VideoPainter.
    Source: https://github.com/zibojia/COCOCO

64. **EraserDiT** -- diffusion-transformer inpainting tuned for high-res
    subtitle removal; track through 2026, integrate once weights are
    stable.
    Source: https://arxiv.org/html/2506.12853v2

65. **FloED** -- optical-flow guided efficient diffusion; mid-weight
    alternative when VACE is overkill but TBE leaves residuals on fast
    pans.
    Source: https://github.com/NevSNev/FloED-main

### Mask / segmentation / matting

66. **SAM 2 mask refinement** -- click inside a detected subtitle region;
    SAM 2 promotes it to a clean object mask that follows the text,
    eliminating aggressive dilation. Memory propagation carries the seed
    through the clip.
    Source: https://github.com/facebookresearch/sam2

67. **SAM 3 text-prompt segmentation** -- natural-language "segment all
    burned-in text" without bounding boxes; effectively auto-mask.
    Sources: https://github.com/facebookresearch/sam3
    https://ai.meta.com/blog/segment-anything-model-3/

68. **MatAnyone 2** -- video matting with learned quality evaluator;
    best-in-class for tracking a thin mobile mask across a clip.
    Alternative mask generator for decorated subtitles.
    Source: https://github.com/pq-yang/MatAnyone2

69. **CoTracker3 point tracking** -- track arbitrary points through
    occlusion; lighter propagation primitive when SAM 2 is overkill.
    Sources: https://github.com/facebookresearch/co-tracker
    https://cotracker3.github.io/

### Acceleration

71. **PyNvVideoCodec hardware decode** -- replace `cv2.VideoCapture` for
    NVIDIA users; ~6x faster decode and frames stay on the GPU.
    Source: https://developer.nvidia.com/pynvvideocodec

72. **RIFE-interpolated fast mode** -- detect+inpaint every Nth frame and
    synthesise intermediates with Practical-RIFE from cleaned keyframes;
    2-4x throughput on smooth backgrounds, scene-cut fallback to
    nearer-frame duplicate.
    Source: https://github.com/hzwer/Practical-RIFE

### Format support

73. **10-bit / HDR pipeline** -- HDR *metadata passthrough* shipped;
    remaining: re-plumb processing as 16-bit numpy (current pipeline
    clamps HDR10/HLG/DV to 8-bit SDR) and output via
    `libx265 -pix_fmt yuv420p10le`. H.264 cannot encode HDR; HEVC or AV1
    only. Dolby Vision needs a `dovi_tool` round-trip.
    Source: https://codecalamity.com/encoding-uhd-4k-hdr10-videos-with-ffmpeg/

74. **AV1 + VP9 decode verification (remainder)** -- HEVC/AV1 egress
    shipped; remaining: verify AV1/VP9 decode across all codepaths and
    pair SVT-AV1 output with native film-grain synthesis.
    Source: https://trac.ffmpeg.org/wiki/Encode/AV1

76. **NLE round-trip** -- accept an EDL / XML from Premiere or DaVinci as
    input and emit a matching sidecar with the cleaned video substituted
    (the output sidecar writer exists; EDL/XML ingest does not).

### Distribution

### Simplification

117. **OpenCV 5 DNN for detection models** -- evaluate running RapidOCR's
     PP-OCR ONNX models through `cv2.dnn` instead of ONNX Runtime; if
     viable, the core detect+inpaint pipeline needs only OpenCV 5 + numpy.
     Priority: P2 simplification. Effort: L. Confidence: medium.
     Acceptance criteria:
     - PP-OCR detection and recognition models load and run via cv2.dnn
       with accuracy parity on reference clips.
     - ONNX Runtime path kept as fallback.
     Source: https://opencv.org/opencv-5/

---

## Under consideration (v5+)

Speculative research bench; not commitments.

81. **Plugin architecture** -- `backend/plugins/*.py` auto-discovery for
    custom detectors and inpainters (the in-process registry deliberately
    avoids filesystem discovery today).
    Source: https://github.com/YaoFANGUK/video-subtitle-remover

82. **Distributed / multi-GPU** -- split frame batches across GPUs with a
    coordinator handling the temporal window across device boundaries.

83. **Headless REST server mode** -- loopback-only local HTTP API (submit
    job, poll status, stream preview).

84. **Tauri / pywebview shell** -- webview-backed cross-platform GUI port
    over the existing Python core; CustomTkinter is the pure-Python
    alternative.
    Sources: https://v2.tauri.app/
    https://github.com/TomSchimansky/CustomTkinter

85. **Subtitle translation pipeline** -- after SRT export, translate via
    a local LLM (llama.cpp) and optionally re-burn translated subtitles
    into the cleaned video.

86. **Streaming mode** -- process a file while still being written (OBS,
    DVR) with a fixed N-second lag; requires a streaming temporal buffer
    instead of fixed batches.

87. **WebGPU / WASM fallback** -- compile the ONNX detect+inpaint path to
    `onnxruntime-web`; drag-and-drop page for locked-down environments.

88. **Font-aware inpainting** -- classify subtitle font style and apply
    per-class mask dilation and feather presets.

89. **Logo / watermark mode** -- persist the mask across the whole video
    with TBE using the full video as the temporal window.

90. **AI chat interface** -- optional side panel where a local LLM
    translates natural-language requests into `ProcessingConfig`
    mutations.

91. **Live performance dashboard** -- in-app panel with detection FPS,
    inpaint FPS, VRAM, disk I/O, and queue counts; pure local.

92. **macOS / Linux parity** -- three-OS GHA build matrix; blockers are
    Windows-specific pieces (ITaskbarList3, winsound, pythonw console
    hiding).

93. **Android port** -- Kotlin + Compose shell around the Python core via
    Chaquopy, with ffmpeg-kit for video I/O.
    Sources: https://github.com/arthenica/ffmpeg-kit
    https://chaquo.com/chaquopy/

94. **iOS / iPadOS port** -- long shot; BeeWare / Pyto-style bridge.

95. **Screen-reader support on Windows (UIA)** -- scaffold shipped;
    remaining: a custom UIA shim via pywin32/comtypes so NVDA/Narrator
    can announce status changes (existing Python a11y libs are
    Linux-only).
    Source: https://pypi.org/project/Tka11y/

97. **GUI localisation** -- scaffold shipped; remaining: gettext-based
    string externalisation with `.mo` files for the top ~10 languages.

98. **Right-to-left UI support** -- scaffold shipped; remaining: full
    Arabic/Hebrew mirroring of text, button order, and queue scroll.
    Depends on #97.

99. **Restyle mode** -- after removal + SRT export, re-burn a translated
    or restyled subtitle track via ffmpeg's `subtitles` filter with an
    `.ass` template.

100. **Watermark addition pipeline** -- burn a user-specified PNG
     watermark at a configurable corner with fade-in/out after cleaning.

101. **Mask-free subtitle erasure research lane** -- gated experimental
     adapter family for CLEAR-class and SEDiT-class models, separate from
     the mask-dependent diffusion scaffolds. Stays on the research bench
     until one model runs locally through a thin adapter and benchmarks
     well.
     Priority: P1 research, P2 product. Effort: L. Confidence: medium.
     Acceptance criteria:
     - A `mask_free` backend category accepting frames plus optional
       prompt/config, falling closed to AUTO/TBE when weights, licensing,
       VRAM, or upstream APIs are unavailable.
     - Adapter gated behind `VSR_EXPERIMENTAL_MASK_FREE=1`; never imports
       Wan/DiffSynth at module import time; verifies weight hashes;
       returns a structured "unavailable" reason instead of throwing.
     - Does not route through `_create_mask`; current OCR/TBE path stays
       default.
     - Benchmark vs AUTO and TBE/LaMa on reference clips with PSNR/SSIM
       plus the temporal metrics from #102, wall-clock s/frame, and peak
       VRAM.
     - Document license constraints before exposing any model in the GUI.
     Sources:
     https://huggingface.co/charlesw09/CLEAR-mask-free-video-subtitle-removal
     https://arxiv.org/abs/2603.21901
     https://arxiv.org/abs/2605.14894

---

## Explicitly not on the roadmap

- **Cloud API integrations** -- offline-first by design; local Whisper /
  Qwen / llama.cpp models only, no remote inference.
- **Docker / containerised builds** -- Windows-first PyInstaller binary;
  containers add friction for the dominant Windows user base.
- **Telemetry by default** -- crash reporting stays strict opt-in with
  scrubbed paths; no analytics or phone-home.
- **Subscription model** -- free single-binary MIT tool.
- **Multi-user / collaboration** -- single-user desktop tool; no shared
  state or concurrent editing.
- **GPL-licensed defaults** -- GPL deps (Surya, manga-image-translator)
  stay opt-in installs, never bundled, keeping the MIT licence clean.
- **Mocking the AI weights for "offline demos"** -- the zero-weight TBE
  path already works and is the demo.

---

## Research-Driven Additions

### P1 (root-cause reliability)

- [ ] P1 -- Off-main-thread startup hardware probes
  Why: the window does not paint for up to ~18s+ when nvidia-smi/ffmpeg are slow or absent, reading as a frozen app on launch.
  Evidence: `gui/app.py:137-139` calls `detect_gpu()`/`detect_ai_engines()`/`detect_ffmpeg()` inline in `MainWindow.__init__`; each shells out with 8-10s timeouts (`gui/utils.py:27,250`).
  Touches: `gui/app.py`, `gui/utils.py`
  Acceptance: the main window paints immediately with a "Detecting hardware..." state; GPU/engine/ffmpeg results are marshalled back via `root.after(0, ...)` and update the header chips when ready.
  Complexity: M

- [ ] P1 -- First-run model-download progress and guidance
  Why: EasyOCR / torch-LaMa / VLM weights download on first use with no progress UI; the app looks hung during a multi-hundred-MB fetch.
  Evidence: `backend/inpainters_onnx.py:17` documents a manual `huggingface-cli download`; EasyOCR/simple-lama fetch weights on first call with no in-app feedback; RESEARCH "Architecture Assessment".
  Touches: `gui/app.py`, `backend/inpainters/lama.py`, `backend/inpainters_onnx.py`, `backend/detection.py`
  Acceptance: first-use of any backend that fetches weights surfaces a determinate or indeterminate progress indicator + size estimate in the log panel/toast; failures show a clear retry message.
  Complexity: M

- [ ] P1 -- Remove ROADMAP version marker from strict release verification
  Why: strict releases currently require `ROADMAP.md` to contain the release tag, contradicting the mandatory rule that ROADMAP contains only incomplete work and no release/session log material.
  Evidence: `.github/workflows/build.yml:367-370` checks `@{ Path = "ROADMAP.md"; Pattern = "v$expectedVersion" }`; `ROADMAP.md` header says remaining-work backlog only; `AGENTS.md` repeats the same hygiene rule.
  Touches: `.github/workflows/build.yml`, `tests/test_release_workflow.py`
  Acceptance: release verification no longer requires a version marker in ROADMAP; tests assert README/CHANGELOG/APP_VERSION checks remain and ROADMAP hygiene is preserved.
  Complexity: S

### P2 (trust / test depth / hygiene)

- [ ] P2 -- CI/test guard for the source-ASCII invariant
  Why: "ALL .py files must be pure ASCII" is a CRITICAL project rule (CLAUDE.md) but nothing enforces it; a stray unicode char ships silently and only breaks downstream (PyInstaller bundling, arbitrary parsers).
  Evidence: no ASCII-check test in `tests/` and no step in `.github/workflows/build.yml` (verified); the hand-rolled ascii guarding in `backend/nle_sidecar.py` exists precisely because this class of bug is unguarded.
  Touches: `tests/`, `.github/workflows/build.yml`
  Acceptance: a pytest (or CI step) scans `**/*.py` and `*.bat` for bytes outside `\x00-\x7f` and fails listing offending file:line; passes on the current tree.
  Complexity: S

- [ ] P2 -- User-visible notice on settings.json corruption/reset
  Why: a malformed settings file silently resets all persisted state (geometry, panel layout, saved regions, user presets) with only a hidden-log warning, so users silently lose configuration.
  Evidence: `gui/config.py:621-633` returns default `ProcessingConfig()` on any load failure and logs to the rotating file only; malformed region rects are likewise dropped silently in `_coerce_rect`.
  Touches: `gui/config.py`, `gui/app.py`
  Acceptance: on a settings load/parse failure the app shows a startup toast/status ("Settings were corrupted and reset to defaults"); dropped region rects log a WARNING naming each discarded entry.
  Complexity: S

- [ ] P2 -- Drag-drop feedback when all dropped files are unsupported
  Why: dropping only unsupported files (e.g. `.exe`, `.txt`) does nothing at all -- no toast, no status -- reading as a broken drop target.
  Evidence: `gui/widgets.py:1204` `if valid:` early-returns before calling `on_drop`, bypassing the existing `_announce_import_summary` infra (`gui/app.py:3747`) which already handles the unsupported-count message.
  Touches: `gui/widgets.py`, `gui/app.py`
  Acceptance: an all-unsupported drop routes through `_announce_import_summary` (or equivalent) and surfaces "skipped N unsupported file(s)"; mixed drops report both added and skipped counts.
  Complexity: S

- [ ] P2 -- Core-pipeline unit tests (detection cascade, tracking, io fallbacks)
  Why: the large suite is integration/hardening-weighted; the modules most likely to break on a dependency bump have no dedicated unit coverage of their fallback branches.
  Evidence: no `tests/test_detection*.py`/`test_tracking*.py`/`test_io*.py`; `backend/detection.py` (cascade RapidOCR>Paddle>Surya>EasyOCR>cv2), `backend/tracking.py` (Kalman), `backend/io.py` (FFmpeg timeout/intermediate-writer fallback) untested in isolation.
  Touches: `tests/`
  Acceptance: parametrized unit tests cover the detection-cascade fallback order with engines mocked absent, the Kalman single-frame-miss recovery, and the io FFmpeg-timeout fallback path; all pass in CI.
  Complexity: M

- [ ] P2 -- Dependency upper bounds + version-cap CI check
  Why: requirements.txt has only lower bounds; a major bump (paddleocr 4.x, rapidocr 3.x) can change return shapes and silently break detection.
  Evidence: `requirements.txt` has no `,<` upper bounds (verified); CLAUDE.md already documents RapidOCR 1.x/2.x return-shape divergence handled in `_detect_rapid`.
  Touches: `requirements.txt`, `setup.py`, `.github/workflows/build.yml`
  Acceptance: OCR-engine deps carry documented major-version caps (`rapidocr>=2.0.0,<3.0.0`, `paddleocr>=3.0.0,<4.0.0`); a CI step fails if an installed engine exceeds the cap. numpy/opencv stay uncapped (documented upgrade path).
  Complexity: S

- [ ] P2 -- Diagnosable logging on swallowed processing-path exceptions
  Why: `except Exception: pass` in the processing/inpaint paths discards tracebacks needed for post-mortem when crash reporting is off (it is opt-in).
  Evidence: multiple swallowed blocks in `backend/processor.py` and `gui/app.py`; deliberate guards (e.g. `backend/detection.py:160-169` GPL Surya skip) must be left alone.
  Touches: `backend/processor.py`, `gui/app.py`, `backend/inpainters/*.py`
  Acceptance: swallowed exceptions on the processing/inpaint paths log at WARNING with `exc_info=True` and a short context tag; intentional control-flow guards are explicitly annotated and exempted.
  Complexity: S

- [ ] P2 -- Graceful in-flight subprocess teardown on app close
  Why: closing during processing can orphan the FFmpeg subprocess and leave worker/preview daemon threads mid-write.
  Evidence: `gui/app.py:215` `_on_close` exists but does not join worker/preview threads or terminate the active FFmpeg child.
  Touches: `gui/app.py`, `backend/io.py`/`backend/remux.py`
  Acceptance: on close-during-processing the active FFmpeg child is terminated and partial outputs cleaned up; preview/worker threads are signalled and joined with a short timeout before the process exits.
  Complexity: M

- [ ] P2 -- Validate PaddleOCR 3.7 / PP-OCRv6 compatibility
  Why: PaddleOCR 3.7.0 shipped PP-OCRv6 on 2026-06-11, while VSR floats `paddleocr>=3.0.0`; the next install can change OCR behavior without a VSR-side benchmark or parser test.
  Evidence: PaddleOCR release notes for v3.7.0 / PP-OCRv6; `requirements.txt:59`, `setup.py:348`, and `.github/workflows/build.yml:72` all install `paddleocr>=3.0.0`; `backend/paddle_compat.py` only documents 2.x/3.x shape compatibility and README still names PP-OCRv5.
  Touches: `backend/paddle_compat.py`, `backend/detection.py`, `tests/`, `README.md`, `requirements.txt`, `.github/workflows/build.yml`
  Acceptance: CI covers a mocked PP-OCRv6 `predict()` payload; README states the supported PaddleOCR tier accurately; dependency policy either pins a tested 3.x range or explicitly validates latest 3.x.
  Complexity: M

- [ ] P2 -- Sync VVC / H.266 support across README and release docs
  Why: VVC output is implemented in the CLI, GUI, and encoder probe, but user-facing docs still list only H.264/H.265/AV1, so users cannot discover the option or its FFmpeg/libvvenc requirement from README.
  Evidence: `backend/cli.py:308-310`, `gui/app.py:1692-1700`, and `backend/encoder.py:25-28` include `vvc`; `README.md:33`, `README.md:226`, and `README.md:276` still document only h264/h265/av1; VVenC FFmpeg docs require `--enable-libvvenc`.
  Touches: `README.md`, `.github/workflows/build.yml`, `tests/test_release_workflow.py`
  Acceptance: README feature list, CLI table, and configuration table include VVC/H.266 with the libvvenc caveat; release verification records whether the bundled/system FFmpeg exposes `libvvenc`.
  Complexity: S

### Later (research bench -- only if local, permissively licensed weights appear)

- [ ] P3 -- ROSE object-removal bench adapter
  Why: ROSE explicitly models removal side-effects (shadows, reflections), directly relevant to drop-shadowed/glowing subtitles that defeat simple masking -- the #5 community pain point.
  Evidence: ROSE "Remove Objects with Side Effects in Videos", https://arxiv.org/pdf/2508.18633; gap noted in RESEARCH community signal.
  Touches: `backend/inpainters_diffusion.py`, `backend/inpainter_registry.py`
  Acceptance: opt-in adapter (env-gated) registering like the existing diffusion scaffolds; falls closed to AUTO/TBE when weights/license/VRAM unavailable; benchmarked vs TBE/LaMa on the reference clips.
  Complexity: L

- [ ] P3 -- MiniMax-Remover lightweight bench adapter
  Why: a low-VRAM, consumer-GPU-friendly video object remover fits VSR's "runs on normal hardware" stance better than the 8-10 GB diffusion bench items.
  Evidence: MiniMax-Remover (minimax-optimization video object removal), surfaced in 2026 SOTA survey; RESEARCH Sources.
  Touches: `backend/inpainters_diffusion.py`, `backend/inpainter_registry.py`
  Acceptance: opt-in adapter behind an env gate; reports peak VRAM and s/frame in the quality report; falls closed to AUTO/TBE when unavailable.
  Complexity: L

### P1 - Reliability and Hardening

- [ ] P1 -- Harden startup GitHub Releases update check
  Why: the optional update check is bounded, but it still polls the GitHub
  Releases API without REST etiquette; rate-limit or API-shape failures are
  currently only swallowed/logged, and repeated launches can keep making the
  same avoidable request.
  Evidence: `backend/update_check.py` sends only an `Accept` header
  (`application/vnd.github+json`); GitHub REST best practices recommend
  explicit error/rate-limit handling and conditional requests where appropriate.
  The continuation research pass also hit unauthenticated GitHub API rate
  limits.
  Touches: `backend/update_check.py`, `gui/config.py`, `tests/test_hardening.py`
  Acceptance: the request sets a clear User-Agent and API-version header;
  ETag/Last-Modified state is stored under APPDATA or settings; 304 responses
  are treated as "no update"; 403/429 rate-limit headers back off without
  repeated retries; tests cover 200-newer, 304-not-modified, timeout, and
  rate-limited cases.
  Complexity: M
  Sources:
  https://docs.github.com/en/rest/using-the-rest-api/best-practices-for-using-the-rest-api

- [ ] P1 -- Verify release workflow downloaded tooling and action supply chain
  Why: release jobs execute installer and winget submission tooling, so mutable
  action tags and downloaded executables need stronger evidence than "latest"
  URLs before producing user-facing binaries.
  Evidence: `.github/workflows/build.yml` uses action major-version tags and
  downloads `wingetcreate.exe` from `https://aka.ms/wingetcreate/latest` before
  executing it; GitHub Actions secure-use guidance recommends hardening action
  and dependency trust boundaries.
  Touches: `.github/workflows/build.yml`, `tests/test_release_workflow.py`
  Acceptance: actions are pinned to reviewed SHAs or covered by a documented
  allowlist/update policy; wingetcreate is downloaded from a versioned release
  and Authenticode/hash verified before execution; release-verification records
  release-tool names, versions, and hashes; a workflow test rejects unchecked
  executable downloads from mutable "latest" URLs.
  Complexity: M
  Sources:
  https://docs.github.com/en/actions/reference/security/secure-use
  https://github.com/microsoft/winget-create/releases

- [ ] P1 -- Pin runtime remote-code model adapters
  Why: optional VLM and tracking adapters can execute remote repository code or moving refs, which weakens the local-first trust boundary.
  Evidence: `backend/ocr_vlm.py:99-100` uses `trust_remote_code=True`; `backend/segmentation.py:207` calls `torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")`; Hugging Face Auto Classes docs and PyTorch torch.hub docs both describe these as trusted-code boundaries.
  Touches: `backend/ocr_vlm.py`, `backend/segmentation.py`, `backend/adapter_manifest.py`, `backend/model_hashes.py`, `tests/`
  Acceptance: remote-code adapters require an approved local path or pinned revision; default runtime refuses unpinned remote-code loaders with an actionable message; release evidence records model repo/revision/path/hash; tests assert no `trust_remote_code=True` or `torch.hub.load()` path is reachable without the policy gate.
  Complexity: M

### P2 - Evidence Quality and Observability

- [ ] P2 -- Refresh architecture map after module split and release additions
  Why: stale architecture docs send contributors to the wrong files and make
  future research repeat basic tree discovery.
  Evidence: `docs/architecture.md` still reflects older GUI ownership, an older
  backend map, a narrow test map, pre-VVC codec support, and an old "current as
  of" marker; the current tree has `gui/app.py`, cache/update/security modules,
  VVC/H.266 paths, and broader release/hardening tests.
  Touches: `docs/architecture.md`, `README.md`
  Acceptance: architecture docs name the current GUI and backend module
  boundaries; include VVC/H.266, cache inventory, update checks, release
  verification, and current test families; remove stale monolith guidance and
  update the "current as of" marker.
  Complexity: S

- [ ] P2 -- Capture RapidOCR bundled-model provenance in release evidence
  Why: RapidOCR is the default OCR path, but its bundled ONNX assets are only audited for opsets today, not package version, file hashes, or asset provenance.
  Evidence: `backend/onnx_model_info.py:167-214` locates and audits RapidOCR model files; RapidOCR discussion #667 reports v3.x wheels depend on older v1.1.0 release assets.
  Touches: `backend/onnx_model_info.py`, `backend/model_hashes.py`, `.github/workflows/build.yml`, `tests/`
  Acceptance: release verification lists RapidOCR package name/version, discovered model filenames, SHA-256 hashes, and opsets; strict release fails or clearly warns when default RapidOCR assets are missing/unreadable; tests cover a mocked RapidOCR model directory.
  Complexity: M

- [ ] P2 -- Add GUI review worklist for quality-gate failures
  Why: batch quality gates can mark outputs as `review-needed`, but the GUI completion flow only offers a generic report open action, making failed-gate outputs easy to miss in large batches.
  Evidence: `backend/batch_report.py:37` defines `review-needed`; `backend/batch_report.py:140` sets that status on failed quality gates; `gui/app.py:3333-3349` exposes only `Open report` in the completion modal.
  Touches: `gui/app.py`, `gui/widgets.py`, `backend/batch_report.py`, `backend/quality_gate.py`
  Acceptance: when any batch item is `review-needed`, the completion dialog shows a review count and a primary review action; clicking it filters or focuses the queue on those items and opens the quality sheet/report for the first affected output; row actions expose `Open quality sheet` where available.
  Complexity: M

- [ ] P2 -- Resolve generated PowerShell launcher drift
  Why: setup generates a PowerShell launcher that is not part of the tracked/documented launcher set, so it can drift from release-tested launch behavior.
  Evidence: `setup.py:431-583` writes `Run_VSR_Pro.ps1`; tracked root launchers and README launcher instructions focus on `Run_VSR_Pro.bat` and `Run_VSR_Pro_Debug.bat`.
  Touches: `setup.py`, `README.md`, `.github/workflows/build.yml`, `tests/test_setup_bootstrap.py`
  Acceptance: either stop generating `Run_VSR_Pro.ps1` or track, package, document, and smoke-test it; release verification records the exact launcher files included in the bundle.
  Complexity: S

- [ ] P2 -- Schema-gate imported presets before applying settings
  Why: shared preset files can currently mutate any `ProcessingConfig` field with no preview or field allowlist, which can unexpectedly flip network, output, reporting, Whisper, or destructive workflow settings.
  Evidence: `gui/config.py:765-784` stores imported preset `fields` as-is; `gui/config.py:683-700` applies every imported key that exists on `ProcessingConfig`.
  Touches: `gui/config.py`, `backend/presets.py`, `gui/app.py`, `tests/test_hardening.py`
  Acceptance: imported presets are filtered through an explicit allowlist of safe tuning fields; rejected fields are reported in the import status; the GUI previews which settings will change before applying; tests cover malicious/unknown/path/network fields.
  Complexity: S

- [ ] P2 -- Add redacted support bundle and structured bug-report intake
  Why: users can open logs, but maintainers do not get a single privacy-preserving diagnostics package or issue form with the versions, ffmpeg/OpenCV/GPU facts, settings summary, and redacted logs needed to reproduce local media failures.
  Evidence: README links directly to generic GitHub issues; `.github/` only contains `workflows/build.yml`; `backend/crash_reporter.py` already scrubs paths and `backend/batch_report.py:161-168` already redacts report paths; GitHub issue forms support structured required fields; OWASP logging guidance warns that logs may contain sensitive data and should be protected/redacted.
  Touches: `gui/app.py`, `backend/crash_reporter.py`, `backend/batch_report.py`, `.github/ISSUE_TEMPLATE/bug_report.yml`, `README.md`, `tests/`
  Acceptance: GUI exposes "Create support bundle" and CLI exposes an equivalent command; the bundle contains app/Python/dependency/ffmpeg/OpenCV/GPU metadata, redacted settings, recent redacted logs, and optional batch report summaries without media paths or OCR text; GitHub bug form asks for the bundle and key reproduction fields; tests assert path/OCR redaction.
  Complexity: M

### P3 - Operational Maturity

- [ ] P3 -- Release SBOM and optional artifact provenance evidence
  Why: release verification records useful details, but users and maintainers do
  not get a standard CycloneDX/SPDX bill of materials or a provenance path for
  the Windows binaries.
  Evidence: `.github/workflows/build.yml` writes release-verification JSON and
  pip evidence but no SBOM artifact; GitHub supports SBOM attestations with
  repository-plan caveats, and CISA SBOM guidance treats component transparency
  as supply-chain risk evidence.
  Touches: `.github/workflows/build.yml`, `tests/test_release_workflow.py`
  Acceptance: release builds emit a CycloneDX or SPDX JSON SBOM for bundled
  Python/runtime dependencies; release-verification records the SBOM path and
  hash; strict release fails if the SBOM is missing; GitHub artifact attestation
  is added when repository visibility/plan supports it and skipped with a clear
  note otherwise.
  Complexity: M
  Sources:
  https://docs.github.com/en/actions/how-tos/secure-your-work/use-artifact-attestations/use-artifact-attestations
  https://www.cisa.gov/resources-tools/resources/2025-minimum-elements-software-bill-materials-sbom
