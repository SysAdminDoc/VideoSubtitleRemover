# Video Subtitle Remover Pro -- Roadmap

Remaining-work backlog only: every item below is NOT yet implemented.
Completed items are deleted from this file; history lives in CHANGELOG.md and git.

---

## Now

106. **Release artifact verification gate (remainder)** -- strict mode
     (`release_quality=permissive|strict`), strict pip-audit/NSIS/signing,
     artifact hash manifest, bundled-doc checks, signature verification,
     and ffmpeg version recording are implemented.
     Priority: P0 release quality. Effort: S (remainder). Confidence: high.
     Acceptance criteria (remaining):
     - Execute the bundled GUI smoke path in strict mode when the release
       runner has a reliable interactive desktop session.

110. **OpenCV 5 DNN LaMa migration** -- migrate the LaMa inpainting
     backend from `simple-lama-inpainting` (PyTorch) to OpenCV 5's
     `cv2.dnn` LaMa runner; removes the PyTorch runtime dependency for
     that codepath and closes the `torch.load` CVE surface.
     Priority: P0 security + maintenance. Effort: M. Confidence: high.
     Acceptance criteria:
     - `LAMAInpainter` runs inference via `cv2.dnn.readNetFromONNX` using
       the `opencv/inpainting_lama` ONNX model.
     - Falls back to `simple-lama-inpainting` when opencv-python < 5.0;
       that package becomes an optional/fallback dependency.
     - A/B PSNR/SSIM parity with the PyTorch path (+/- 0.5 dB) on
       reference clips.
     - `requirements.txt` floor updated with documented upgrade path.
     - Tests mock the DNN path and verify fallback behavior.
     Risks: new DNN engine ONNX op coverage; fixed 512x512 input may need
     tiling for large frames.
     Sources: https://opencv.org/opencv-5/
     https://huggingface.co/opencv/inpainting_lama

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

### Input preprocessing

### Security

49. **Model-weight hash table expansion** -- the SHA-256 verifier in
    `backend/model_hashes.py` exists but only lists `big-lama.pt`;
    remaining: add vendored hash entries for every opt-in downloadable
    weight (real ProPainter, MI-GAN, LaMa-ONNX, etc.) so first-download
    verification actually covers them.

### Observability

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

### P1

### P2

- [ ] P2 - Add local cache inspector and cleanup command
  Why: Optional proxies, model weights, TensorRT engines, and Whisper assets can consume disk silently, but the app only exposes log/settings folders today.
  Evidence: `backend/proxy_workflow.py:10-29`; `backend/tensorrt_compile.py:1-31`; `backend/model_hashes.py:100-112`; `backend/whisper_fallback.py:24`
  Touches: `backend/cache_inventory.py`, `backend/cli.py`, `gui/app.py`, `README.md`, `tests/test_hardening.py`
  Acceptance: GUI and CLI report cache directories, byte sizes, and provenance labels; cleanup skips active-run files, leaves unknown user files alone, and records results in the log/status surface.
  Complexity: M

- [ ] P2 - Promote LaMa-ONNX from opt-in to default inpaint backend
  Why: IOPaint (the 23K-star LaMa ecosystem anchor) was archived in April 2025. simple-lama-inpainting's continued maintenance is uncertain. The existing LaMa-ONNX path (RM-25, opt-in via `VSR_LAMA_ONNX`) eliminates the `torch.load` CVE surface (CVE-2026-24747), removes the PyTorch runtime dependency for the LaMa codepath, and runs 3-5x faster via ONNX Runtime.
  Evidence: IOPaint archived at https://github.com/Sanster/IOPaint; simple-lama-inpainting uses torch.load internally; `backend/inpainters_onnx.py` already has a working LaMa-ONNX backend that registers via the inpainter registry; CVE-2026-24747 / CVE-2025-32434 affect torch.load in PyTorch <= 2.9.1; OpenCV 5 DNN LaMa (#110) is the longer-term exit but is CPU-only as of June 2026
  Touches: `backend/inpainters/lama.py`, `backend/inpainters_onnx.py`, `requirements.txt`, `setup.py`, `build_exe.bat`, `.github/workflows/build.yml`, `README.md`, `tests/test_hardening.py`
  Acceptance: The default LAMA mode uses ONNX Runtime with a bundled or auto-downloaded LaMa FP32 ONNX weight (SHA-256 verified via `backend/model_hashes.py`). simple-lama-inpainting becomes the fallback when ONNX Runtime is absent. A/B PSNR/SSIM parity with the PyTorch path on reference clips (+/- 0.5 dB). PyInstaller bundle includes the ONNX weight. torch is still needed for opt-in diffusion backends but no longer for the core LaMa path.
  Complexity: L


