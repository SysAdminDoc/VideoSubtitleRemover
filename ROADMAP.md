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

### P0 -- Security and decode safety

- [ ] P0 -- Route untrusted PNG decoding around vulnerable OpenCV libpng
  Why: OpenCV/libpng currently only warns, while PNG source reads still pass
  through `cv2.imread` in GUI, CLI, processor, and frame-sequence paths.
  Evidence: `backend/security_checks.py`, `backend/processor.py:920`,
  `backend/io.py:642`, `gui/app.py:1256`, OpenCV issue #1186,
  NVD CVE-2026-22801, Pillow 12.2.0
  Touches: `backend/io.py`, `backend/processor.py`, `backend/cli.py`,
  `gui/app.py`, `tests/test_hardening.py`, `tests/test_io_pipeline.py`,
  `tests/test_gui_smoke.py`
  Acceptance: A single safe image-read helper routes PNG input through Pillow
  or fails closed whenever `opencv_libpng_status().vulnerable` is true; tests
  patch vulnerable/fixed states and assert no untrusted PNG path calls
  `cv2.imread` under the vulnerable state.
  Complexity: M

### P1 -- Trust and release readiness

- [ ] P1 -- Add advisory gating to local release evidence
  Why: Release evidence records dependencies and an SBOM, but strict builds do
  not fail on known high/critical advisories in bundled Python/native packages.
  Evidence: `backend/release_verification.py`, `requirements.txt`,
  ONNX Runtime v1.27.0, Pillow 12.2.0, NVD CVE-2025-32434,
  NVD CVE-2026-22801
  Touches: `backend/release_verification.py`, `build_exe.bat`,
  `requirements.txt`, `tests/test_release_workflow.py`, `README.md`
  Acceptance: `build_exe.bat` emits `release-advisories.json`; strict release
  mode fails on unallowed high/critical advisories; the OpenCV/libpng exception
  is explicit and removed automatically when runtime status reports a fixed
  bundled libpng.
  Complexity: M

- [ ] P1 -- Add FFmpeg capability profiles before long batches
  Why: VVC, VMAF, loudnorm, hardware encoders, and FFmpeg Whisper depend on the
  installed Windows FFmpeg build, but current probes are scattered and
  encoder-only.
  Evidence: `README.md`, `backend/support_bundle.py:run_self_test`,
  `backend/release_verification.py`, FFmpeg docs, VideoHelp community threads
  Touches: `backend/support_bundle.py`, `backend/release_verification.py`,
  `backend/cli.py`, `gui/app.py`, `README.md`, `tests/test_support_bundle.py`
  Acceptance: `--self-test` and the Help/backend panel report `basic`,
  `advanced_quality`, `speech_fallback`, and `modern_codec` profiles with exact
  missing filters/encoders; GUI warns before starting a batch whose selected
  options exceed the current FFmpeg profile.
  Complexity: M

- [ ] P1 -- Turn quality-gate remediation into retry configs
  Why: Review-needed outputs already carry ladder steps, but users still have to
  translate remediation prose into settings manually.
  Evidence: `backend/quality_gate.py`, `gui/app.py` review worklist and retry
  flow, commercial subtitle-removal task flows
  Touches: `backend/quality_gate.py`, `gui/app.py`, `gui/config.py`,
  `tests/test_gui_review_worklist.py`, `tests/test_hardening.py`
  Acceptance: A review-needed queue item exposes `Retry with suggested
  settings`; each ladder step mutates only the affected item's config, records
  the before/after config in the batch report, and requeues without changing
  unrelated items.
  Complexity: M

### P2 -- Dependency, documentation, and UX hardening

- [ ] P2 -- Add time-ranged manual subtitle regions
  Why: Manual masks are global today, so OCR-failure clips with changing
  subtitle placement require overbroad regions or repeated batches.
  Evidence: `gui/config.py` `subtitle_area` / `subtitle_areas`,
  `backend/processor.py` fixed-mask path, SubtitleEdit/RapidVideOCR extraction
  workflows, commercial timeline editors
  Touches: `gui/config.py`, `backend/config.py`, `gui/app.py`,
  `backend/processor.py`, `tests/test_gui_smoke.py`, `tests/test_hardening.py`
  Acceptance: Users can define multiple regions with optional start/end times;
  settings/preset/CLI config round-trip the new schema; processing applies the
  correct region per frame and falls back to current global behavior when no
  time spans are set.
  Complexity: L

- [ ] P2 -- Add verified portable model-cache export/import
  Why: The app can inspect model caches but cannot prepare an air-gapped or
  slow-network machine without re-downloading OCR, LaMa, or Whisper assets.
  Evidence: `backend/model_downloads.py`, `backend/cache_inventory.py`,
  `backend/model_hashes.py`, IOPaint model-management UX, offline-first product
  promise
  Touches: `backend/cache_inventory.py`, `backend/model_downloads.py`,
  `backend/model_hashes.py`, `backend/support_bundle.py`, `backend/cli.py`,
  `gui/app.py`, `tests/test_model_downloads.py`
  Acceptance: CLI and Help UI can export a zip with a manifest of known model
  files and hashes, import only verified files into the app model cache, reject
  path traversal/unknown executable code, and report missing optional assets
  afterward.
  Complexity: M

- [ ] P2 -- Expose RapidOCR OpenVINO as CPU/Intel OCR option
  Why: RapidOCR 3.9 and PaddleOCR 3.7 highlight PP-OCRv6/OpenVINO speedups,
  while VSR currently treats RapidOCR mainly as an ONNX Runtime path.
  Evidence: RapidOCR v3.9.0, PaddleOCR v3.7.0, `backend/detection.py`,
  `setup.py`, `backend/model_downloads.py`
  Touches: `backend/detection.py`, `backend/dependency_caps.py`, `setup.py`,
  `backend/support_bundle.py`, `tests/test_detection_pipeline.py`
  Acceptance: When OpenVINO dependencies are installed, VSR can select or
  auto-prefer RapidOCR OpenVINO on CPU/Intel systems, reports the active OCR
  engine/provider in backend status, and falls back to current RapidOCR/PaddleOCR
  paths if initialization fails.
  Complexity: M

- [ ] P2 -- Define NVIDIA ONNX Runtime CUDA provider migration path
  Why: ONNX Runtime v1.27 deprecates CUDA 12 packages and VSR's LaMa ONNX
  default needs clear CPU/CUDA/DirectML provider behavior on NVIDIA systems.
  Evidence: ONNX Runtime v1.27.0, `backend/inpainters_onnx.py`,
  `backend/onnx_model_info.py`, `setup.py`, `README.md`
  Touches: `backend/inpainters_onnx.py`, `backend/onnx_model_info.py`,
  `setup.py`, `backend/support_bundle.py`, `README.md`,
  `tests/test_onnx_model_info.py`
  Acceptance: Provider status distinguishes `onnxruntime`, `onnxruntime-gpu`
  CUDA12/CUDA13, and DirectML; setup/docs recommend the tested NVIDIA ONNX path;
  release evidence flags deprecated CUDA provider packages before shipping.
  Complexity: M

- [ ] P2 -- Sync language-support claims with engine reality
  Why: README overview still says 12-language support while current
  RapidOCR/PaddleOCR docs and feature bullets describe 50+ languages.
  Evidence: `README.md`, PaddleOCR v3.7.0, RapidOCR v3.9.0, `gui/config.py`
  Touches: `README.md`, `docs/architecture.md`, `gui/config.py`,
  `tests/test_source_hygiene.py`
  Acceptance: README/docs use one consistent language-support statement and the
  Help/backend status can distinguish app-curated GUI languages from
  engine-reported language capacity.
  Complexity: S

### P3 -- Research bench

- [ ] P3 -- Benchmark deterministic InpaintDelogo-style static-logo cleanup
  Why: Static transparent logos and channel bugs are common community
  complaints, and TBE has no temporal exposure when a fixed logo mask is present
  in every frame.
  Evidence: Purfview/InpaintDelogo, VideoHelp logo-removal threads,
  `backend/presets.py` logo intent, `CLAUDE.md` TBE fixed-mask gotcha
  Touches: `tests/clips/manifest.json`, `backend/presets.py`,
  `backend/inpainters/external.py`, `backend/quality.py`
  Acceptance: Benchmark-only harness compares current LaMa/cv2 static-logo
  cleanup against an InpaintDelogo-style deterministic path on licensed clips;
  no script execution or default dependency is added.
  Complexity: M

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
