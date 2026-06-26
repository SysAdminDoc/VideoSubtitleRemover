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

76. **NLE round-trip** -- accept an EDL / XML from Premiere or DaVinci as
    input and emit a matching sidecar with the cleaned video substituted
    (the output sidecar writer exists; EDL/XML ingest does not).

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

### P1 -- Trust and correctness

- [ ] P1 -- **Fix Toast list mutation during iteration**
  Why: `Toast._active` list is mutated by concurrent fade-out completions
  while `_position()` iterates it, causing skipped toasts or IndexError.
  Evidence: code audit of `gui/widgets.py` Toast._active list access
  Touches: `gui/widgets.py` Toast._position(), Toast._fade_step()
  Acceptance: Use a copy of the list in _position(); no crash on rapid
  toast stacking.
  Complexity: S

- [ ] P1 -- **Disable import buttons during batch processing**
  Why: DragDropFrame's `add_files_btn` and `add_folder_btn` in the empty
  queue state are never disabled by `set_enabled()`, allowing file addition
  mid-batch with race conditions against the queue lock.
  Evidence: code audit of `gui/widgets.py` DragDropFrame, `gui/app.py`
  _set_controls_enabled()
  Touches: `gui/app.py` _set_controls_enabled(), `gui/widgets.py`
  DragDropFrame
  Acceptance: Import buttons disabled during processing; re-enabled on
  completion/cancel.
  Complexity: S

- [ ] P1 -- **Clear SRT entries between process_video() calls**
  Why: `_srt_entries` is appended to but never cleared between consecutive
  calls to process_video(), producing contaminated SRT files when the same
  SubtitleRemover instance processes multiple videos.
  Evidence: code audit of `backend/processor.py` _srt_entries lifecycle
  Touches: `backend/processor.py` process_video()
  Acceptance: Each process_video() call starts with an empty SRT list;
  test confirms no cross-run contamination.
  Complexity: S

- [ ] P1 -- **Complete CLI command builder for all config fields**
  Why: `_build_cli_command()` emits only ~8 of 30+ ProcessingConfig
  fields. Users copying CLI commands from the GUI context menu get
  incomplete reproduction instructions. Missing: mask_feather_px,
  whisper_fallback, karaoke_grouping, remove_chyrons, detection_vertical,
  time_start, time_end, loudnorm_target, edge_ring_px, colour_tune_enable,
  phash_skip_enable, kalman_tracking, temporal_smooth_radius, export_srt,
  export_mask_video, output_frames, nle_sidecar, and more.
  Evidence: code audit of `gui/widgets.py:28-68` vs backend/config.py
  ProcessingConfig fields
  Touches: `gui/widgets.py` _build_cli_command()
  Acceptance: Every non-default ProcessingConfig field with a CLI
  counterpart appears in the generated command. Round-trip test confirms
  parity.
  Complexity: M

### P2 -- Robustness and validation

- [ ] P2 -- **Graceful handling of corrupt/truncated video input**
  Why: No graceful failure path for malformed MP4s, videos with missing
  codecs, or truncated files. cv2.VideoCapture silently returns empty
  reads, but ffprobe/ffmpeg crashes are unhandled.
  Evidence: test gap analysis -- no test for corrupt input
  Touches: `backend/io.py` _open_capture(), `backend/processor.py`
  process_video()
  Acceptance: Corrupt input produces a warning log and `failed` status
  instead of a traceback. Test with a zero-byte and a truncated file.
  Complexity: M

- [ ] P2 -- **Cancel elapsed timer on processing error**
  Why: `_elapsed_timer_id` is set on batch start but only cancelled on
  close confirmation. If processing errors out, the timer fires
  indefinitely until shutdown.
  Evidence: code audit of `gui/app.py` _start_elapsed_timer() and
  _on_processing_complete()
  Touches: `gui/app.py` _on_processing_complete(), _on_processing_error()
  Acceptance: Elapsed timer stops on any terminal processing state
  (complete, error, cancelled).
  Complexity: S

- [ ] P2 -- **Validate ProcessingConfig device string format**
  Why: `device` accepts any string (e.g., "gpu", "1"); invalid values
  fail late in CUDA init instead of at config parse time.
  Evidence: code audit of `backend/config.py` ProcessingConfig.device
  Touches: `backend/config.py` normalized()
  Acceptance: Only "cpu", "cuda:N", "directml" accepted; others raise
  or fall back to "cpu" with a warning.
  Complexity: S

- [ ] P2 -- **Warn on unknown fields in JSON config overlays**
  Why: CLI `--config` JSON overlays accept any key. Typos (e.g.,
  `detectino_lang`) silently become defaults, frustrating users who
  think their config applied.
  Evidence: code audit of `backend/cli.py` _load_json_config()
  Touches: `backend/cli.py` _load_json_config()
  Acceptance: Unknown field names logged as warnings with did-you-mean
  suggestions.
  Complexity: S

- [ ] P2 -- **Add `rapidocr` to support bundle dependency list**
  Why: `_DEPENDENCY_PACKAGES` in support_bundle.py includes
  `rapidocr-onnxruntime` but not `rapidocr`, so the primary OCR engine
  version is missing from bug reports.
  Evidence: code audit of `backend/support_bundle.py` line 24
  Touches: `backend/support_bundle.py` _DEPENDENCY_PACKAGES
  Acceptance: Support bundle captures rapidocr version when installed.
  Complexity: S

- [ ] P2 -- **Scrub work_directory from support bundles**
  Why: `_SENSITIVE_KEYS` does not include `work_directory` (new field
  from commit 38200b8), so user-chosen work directories could appear
  unscrubbed in support bundles.
  Evidence: code audit of `backend/support_bundle.py` _SENSITIVE_KEYS
  Touches: `backend/support_bundle.py` _SENSITIVE_KEYS
  Acceptance: work_directory redacted in support bundle output.
  Complexity: S

- [ ] P2 -- **Verify RapidOCR v3.9.0 PP-OCRv6 compatibility**
  Why: RapidOCR v3.9.0 (June 2026) defaults to PP-OCRv6 det/rec models
  with different filenames and doubled package size (15 MB to 29 MB).
  Model paths, config loading, and PyInstaller data collection may need
  updates.
  Evidence: RapidOCR v3.9.0 release notes (github.com/RapidAI/RapidOCR)
  Touches: `backend/detection.py` _build_rapidocr(), `build_exe.bat`
  collect-data, `backend/dependency_caps.py`, `.github/workflows/build.yml`
  Acceptance: Detection works identically with rapidocr 3.9.0; PyInstaller
  bundle includes the new model files; release CI passes.
  Complexity: M

### P2 -- Testing gaps

- [ ] P2 -- **Test full OCR cascade failure path**
  Why: No test verifies behavior when all OCR engines are absent AND the
  OpenCV fallback is used. The current cascade test patches individual
  engines but not the complete failure-to-fallback chain.
  Evidence: test gap analysis of `tests/test_detection_pipeline.py`
  Touches: `tests/test_detection_pipeline.py`
  Acceptance: Test with all engine modules nulled in sys.modules confirms
  OpenCV fallback returns boxes (or empty list with warning).
  Complexity: S

- [ ] P2 -- **Test temp file cleanup on processing exception**
  Why: `_cleanup_temp_output()` should fire when an exception occurs
  mid-inpaint, but no test verifies this.
  Evidence: test gap analysis
  Touches: `tests/test_io_pipeline.py` or new test file
  Acceptance: Simulated mid-processing exception leaves no temp files in
  the output directory.
  Complexity: S

- [ ] P2 -- **Test queue autosave/restore round-trip**
  Why: Queue autosave (commit 36737c4) has no dedicated test for the
  save-crash-restore cycle.
  Evidence: test gap analysis -- feature landed without round-trip test
  Touches: `tests/` (new test)
  Acceptance: Save queue state, clear, restore, verify items match.
  Complexity: S

- [ ] P2 -- **Test CLI numeric flag out-of-range behavior**
  Why: `--mask-dilate`, `--crf`, `--frame-skip` and other numeric CLI
  flags are parsed but never bounds-checked against ProcessingConfig
  constraints before reaching the normalizer.
  Evidence: code audit of `backend/cli.py` argparse definitions
  Touches: `tests/` (new test), optionally `backend/cli.py` for
  pre-normalization validation
  Acceptance: Out-of-range values are clamped with a warning, not
  silently accepted or crashed.
  Complexity: S

### Later (research bench -- only if local, permissively licensed weights appear)

- [ ] P3 -- **VOID video inpainting adapter (opt-in)**
  Why: VOID (Netflix, April 2026) is best-in-class open video inpainting,
  HuggingFace weights available, won 64.8% user preference vs
  Runway/ROSE/MiniMax/ProPainter. Handles object removal + physical
  interactions (shadows, reflections).
  Evidence: void-model.github.io, HuggingFace release
  Touches: `backend/inpainters_diffusion.py` (new registration),
  `backend/remote_model_policy.py` (new policy entry)
  Acceptance: `--mode void` available when VSR_VOID=1 and model weights
  present; falls back cleanly when absent.
  Complexity: L

