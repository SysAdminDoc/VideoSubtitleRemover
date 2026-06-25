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

### P2 (trust / test depth / hygiene)

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

- [ ] P1 -- RapidOCR 3.x compatibility testing and cap lift
  Why: the `<3.0.0` pin blocks 9 months of improvements including a unified
  package, CoreML engine, and TensorRT support; v3.8.0+ changed the constructor
  and return types.
  Evidence: RapidOCR v3.9.0 released 2026-06-23; v3.8.0 breaking changes noted
  by Immich/docling-serve downstream consumers; RESEARCH open question.
  Touches: `requirements.txt`, `backend/detection.py` (`_detect_rapid`),
  `tests/test_detection_pipeline.py`
  Acceptance: `_detect_rapid` handles v2.x and v3.x constructor + return types;
  test covers both API shapes; cap changed to `<4.0.0`; no regression on
  reference clips.
  Complexity: M
  Source: https://github.com/RapidAI/RapidOCR/releases

- [ ] P1 -- PP-OCRv6 model evaluation and auto-selection
  Why: +5.1% recognition accuracy over PP-OCRv5, 5.2x CPU speedup, 50-language
  single model (no model switching); ships inside PaddleOCR 3.7+ within the
  existing pip floor.
  Evidence: PP-OCRv6 paper (arXiv 2606.13108) and PaddleOCR 3.7.0 release;
  benchmarks show 83.2% recognition accuracy at 34.5M params.
  Touches: `backend/detection.py` (`_detect_paddle`), `backend/paddle_compat.py`,
  `tests/test_detection_pipeline.py`
  Acceptance: detection pipeline auto-selects PP-OCRv6 medium model when
  PaddleOCR >= 3.7.0 is installed; accuracy validated on reference clips;
  fallback to existing models on older PaddleOCR.
  Complexity: M
  Source: https://arxiv.org/html/2606.13108

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

- [ ] P2 -- GitHub issue templates with support-bundle prompt
  Why: bug reports are unstructured and do not ask for the support bundle that
  already ships; structured issue forms reduce back-and-forth and surface
  reproducible details.
  Evidence: RESEARCH security/reliability section; `backend/support_bundle.py`
  exists but GitHub intake is a bare issue link; docs.github.com issue templates
  support structured required fields.
  Touches: `.github/ISSUE_TEMPLATE/bug_report.yml`,
  `.github/ISSUE_TEMPLATE/feature_request.yml`
  Acceptance: bug report form asks for VSR version, OS, GPU, support-bundle zip,
  and reproduction steps; feature request form asks for use case and expected
  behavior; both render correctly on GitHub.
  Complexity: S
  Source: https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests

- [ ] P2 -- TkinterDnD2 bundling or visible degradation feedback
  Why: native drag-drop silently fails without the optional `tkinterdnd2` package;
  the drop area degrades to click-only with no user-visible indication, causing
  confusion (issues #4/#5 reporter hit this).
  Evidence: `gui/widgets.py:1252-1259` wraps TkinterDnD2 in a silent try/except;
  issue #4 reporter could not drag files; EchoSubs and upstream both support
  drag-drop out of the box.
  Touches: `gui/widgets.py` (`DragDropFrame._setup_dnd`), `requirements.txt`,
  `setup.py`
  Acceptance: either TkinterDnD2 is installed by default (preferred), or the drop
  area shows a visible note when drag-drop is unavailable (e.g. "Drag-drop
  requires tkinterdnd2; click to browse").
  Complexity: S

- [ ] P2 -- FFmpeg 8 Whisper filter backend evaluation
  Why: FFmpeg 8 ships a built-in Whisper filter for live transcription, which
  could simplify or replace the project's `whisper_fallback.py` for subtitle-aware
  processing without requiring Python ML dependencies.
  Evidence: FFmpeg 8.0 "Huffman" release (2025-08-22) includes `whisper` filter;
  `backend/whisper_fallback.py` already has `--whisper-backend ffmpeg` support but
  performance/accuracy comparison against `faster-whisper` is not documented.
  Touches: `backend/whisper_fallback.py`, `docs/architecture.md`
  Acceptance: documented benchmark comparing FFmpeg Whisper filter vs
  faster-whisper on 3+ reference clips (accuracy, speed, VRAM); recommendation
  on default backend; architecture docs updated.
  Complexity: S

- [ ] P2 -- Document ProPainter name vs license distinction in user-facing docs
  Why: VSR's "ProPainter" mode is TBE+LaMa (MIT-clean), not the ICCV 2023
  ProPainter (NTU S-Lab non-commercial). CLAUDE.md documents this internally
  but user-facing docs and the README algorithm table do not.
  Evidence: ProPainter license is NTU S-Lab 1.0 (non-commercial only);
  competitive analysis flags this as the single biggest legal confusion risk
  for MIT-licensed tools; README Algorithm Comparison table says "ProPainter"
  without clarification.
  Touches: `README.md` (Algorithm Comparison table and credits section)
  Acceptance: README algorithm table and credits section explicitly note that
  VSR's ProPainter mode is a TBE+LaMa hybrid, not the ICCV 2023 ProPainter
  weights or code.
  Complexity: S

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
