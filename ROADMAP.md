# Video Subtitle Remover Pro -- Roadmap

Remaining-work backlog only: every item below is NOT yet implemented.
Completed items are deleted from this file; history lives in CHANGELOG.md and git.

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
    fonts, dissolves) with a scheduled local/release run comparing output
    hashes and PSNR/SSIM scores to committed baselines.

---

## Later

### Real reference inpainters (opt-in scaffolds exist; full integrations remain)

61. **Wan2.1-VACE** -- masked video-to-video editing; the 1.3B variant
    runs on consumer GPUs and beats ProPainter on motion coherence.
    Opt-in with auto-fetched weights.
    Sources: https://github.com/ali-vilab/VACE
    https://openaccess.thecvf.com/content/ICCV2025/papers/Jiang_VACE_All-in-One_Video_Creation_and_Editing_ICCV_2025_paper.pdf

62. **VideoPainter** -- dual-branch context encoder injecting background
    guidance into a pre-trained video diffusion transformer; handles
    videos over one minute with ID consistency.
    Source: https://arxiv.org/html/2503.05639v1

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

### P1 -- Trust and release readiness

- [ ] P1 -- Rebuild local release verification after GitHub Actions removal
  Why: The workflow was removed in local-build mode, but most
  release-evidence assertions now skip when `.github/workflows/build.yml`
  is absent.
  Evidence: `c4a4617`, `tests/test_release_workflow.py`, `build_exe.bat`,
  `README.md`
  Touches: `build_exe.bat`, `tests/test_release_workflow.py`,
  `backend/onnx_model_info.py`, `backend/adapter_manifest.py`,
  `backend/remote_model_policy.py`
  Acceptance: A local release command emits `release-verification.json`,
  SBOM/dependency evidence, hidden-import inventory, launcher/doc version
  checks, RapidOCR/model-policy status, and smoke-launch status; tests pass
  without `.github/workflows/build.yml`.
  Complexity: M

- [ ] P1 -- Gate PyTorch LaMa hidden imports behind explicit opt-in
  Why: `build_exe.bat` always includes `--hidden-import
  simple_lama_inpainting` even though README says the PyTorch LaMa fallback
  is disabled unless `VSR_ENABLE_PYTORCH_LAMA=1`.
  Evidence: `build_exe.bat`, `README.md`, PyTorch CVE notes in
  `requirements.txt`
  Touches: `build_exe.bat`, `tests/test_release_workflow.py`,
  `tests/test_cv2dnn_lama.py`
  Acceptance: PyInstaller includes `simple_lama_inpainting` only when the
  opt-in env var is set and the module is importable; tests assert default
  bundles do not include the hidden import.
  Complexity: S

- [ ] P1 -- Add scaled mask-selector end-to-end regression coverage
  Why: Recent mask-selection fixes cover save flow, but the high-risk path
  still depends on canvas-to-image coordinate scaling, queued config
  snapshots, preloaded rectangles, and video/image input differences.
  Evidence: `8a2e29d`, `2afe087`, `gui/app.py` `_open_region_selector()`,
  `tests/test_gui_smoke.py`
  Touches: `gui/app.py`, `tests/test_gui_smoke.py`,
  `tests/test_hardening.py`
  Acceptance: A scripted GUI test opens a resized selector, preloads and
  draws multiple rectangles, clears/re-saves, verifies app config, selected
  queue item config, and mask-preview coordinates.
  Complexity: M

- [ ] P1 -- Gracefully classify corrupt or truncated video input
  Why: Corrupt media is a common support path for video tools; VSR should
  report a failed item with actionable copy instead of surfacing raw OpenCV
  or FFmpeg errors.
  Evidence: `backend/io.py`, `backend/processor.py`, VideoHelp community
  repair/removal threads
  Touches: `backend/io.py`, `backend/processor.py`,
  `tests/test_io_pipeline.py`, `tests/test_hardening.py`
  Acceptance: Zero-byte, truncated, and unsupported-codec fixtures produce
  failed queue records, concise log messages, and no leaked temp outputs.
  Complexity: M

### P2 -- Dependency, documentation, and UX hardening

- [ ] P2 -- Verify RapidOCR PP-OCRv6 packaging compatibility
  Why: RapidOCR 3.x can ship different PP-OCR model assets and package
  layout; PyInstaller data collection and detector parsing must stay
  compatible across the capped major range.
  Evidence: RapidOCR releases, `backend/detection.py`,
  `backend/paddle_compat.py`, `build_exe.bat`
  Touches: `backend/detection.py`, `backend/onnx_model_info.py`,
  `backend/dependency_caps.py`, `build_exe.bat`,
  `tests/test_detection_pipeline.py`
  Acceptance: RapidOCR 3.x detection works from source and PyInstaller
  bundle; release evidence lists bundled RapidOCR ONNX files and hashes.
  Complexity: M

- [ ] P2 -- Track fixed OpenCV/libpng wheel availability
  Why: VSR can warn about vulnerable bundled libpng, but cannot eliminate
  CVE-2026-22801 until opencv-python publishes wheels with libpng >= 1.6.54.
  Evidence: `backend/security_checks.py`, `requirements.txt`, NVD
  CVE-2026-22801, opencv-python release issue
  Touches: `backend/security_checks.py`, `requirements.txt`, `README.md`,
  `tests/test_hardening.py`
  Acceptance: When a fixed opencv-python wheel is available, dependency
  floor and warning tests are updated; until then the runtime warning and
  README guidance remain accurate.
  Complexity: S

- [ ] P2 -- Sync architecture docs with local-build and PyTorch opt-in truth
  Why: `docs/architecture.md` and repo working notes still mention removed
  GitHub Actions release paths and older automatic PyTorch fallback behavior.
  Evidence: `docs/architecture.md`, `CLAUDE.md`, `README.md`, `c4a4617`
  Touches: `docs/architecture.md`, `CLAUDE.md`
  Acceptance: Architecture docs describe local release verification, current
  LaMa priority order, and the explicit PyTorch opt-in gate without pointing
  agents at removed workflow files.
  Complexity: S

- [ ] P2 -- Add installed model and backend status panel
  Why: Competitors such as IOPaint reduce setup friction with visible model
  and backend state; VSR currently spreads this across startup chips, support
  bundles, logs, and README troubleshooting.
  Evidence: IOPaint, commercial subtitle-removal flows,
  `backend/model_downloads.py`, `backend/support_bundle.py`, `gui/app.py`
  Touches: `gui/app.py`, `backend/model_downloads.py`,
  `backend/support_bundle.py`, `backend/onnx_model_info.py`
  Acceptance: About/settings surface shows OCR/inpaint backends, required
  model files, provider availability, hash status, and next action without
  requiring a support bundle.
  Complexity: M

### P3 -- Research bench

- [ ] P3 -- Benchmark mask-free subtitle erasure before adapter work
  Why: CLEAR and SEDiT are subtitle-specific, but need VSR reference-clip
  evidence before becoming user-facing modes.
  Evidence: CLEAR Hugging Face release, SEDiT arXiv, `tests/clips/manifest.json`
  Touches: `tests/clips/manifest.json`, `backend/inpainters_diffusion.py`,
  `backend/adapter_manifest.py`
  Acceptance: Benchmark-only harness records runtime, artifact score, and
  subtitle-removal quality on licensed clips; no default dependency is added.
  Complexity: M

- [ ] P3 -- Add VOID as opt-in research adapter only if redistribution is safe
  Why: VOID is strong general video inpainting research, but it is not
  subtitle-specific and may be heavy/licensing-sensitive for default users.
  Evidence: VOID project, VSR adapter policy in `backend/adapter_manifest.py`
  Touches: `backend/inpainters_diffusion.py`,
  `backend/remote_model_policy.py`, `backend/adapter_manifest.py`
  Acceptance: `--mode void` registers only with `VSR_VOID=1`, verifies local
  weights through the adapter manifest, and falls back cleanly when absent.
  Complexity: L

- [ ] P3 -- Evaluate Windows ML as DirectML successor path
  Why: ONNX Runtime DirectML remains useful today, but Microsoft positions
  Windows ML as the forward local inference API; VSR should avoid locking new
  adapter work to a shrinking provider path.
  Evidence: ONNX Runtime DirectML docs, Microsoft Windows ML docs,
  `backend/inpainters_onnx.py`, `backend/detection.py`
  Touches: `backend/inpainters_onnx.py`, `backend/detection.py`,
  `backend/onnx_model_info.py`, `tests/test_hardening.py`
  Acceptance: A guarded provider probe proves whether Windows ML can run a
  small ONNX smoke model from Python on Windows; decision documented in
  README/ROADMAP before any migration.
  Complexity: M
