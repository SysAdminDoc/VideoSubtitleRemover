# VideoSubtitleRemover Roadmap

Living document. Tracks shipped features, near-term work, and longer-term
research directions. Items are ordered by value, not calendar date.

---

## Shipped

### v3.12.0 (current)

- **AUTO inpaint mode / per-frame quality tier** (roadmap #5 Inpainting)
  -- new `InpaintMode.AUTO` routes each TBE batch between temporal
  (STTN) and spatial (LaMa) inpainting based on a per-batch exposure
  score vs configurable threshold.
- **Keyframe-driven detection** (roadmap #1 Detection) -- ffprobe
  enumerates I-frames; OCR runs only at keyframes, Kalman propagates
  masks between. Falls back to pHash skip when ffprobe is missing.
- **Deinterlace on ingest** (roadmap #9 Preprocessing) -- ffprobe
  idet detects combing; ffmpeg yadif writes a temp progressive
  source. Auto + manual toggles.
- **PSNR / SSIM quality report** (roadmap #E Testing) -- 10-frame
  deterministic sample after each run. Pure-numpy SSIM, cv2 PSNR,
  no new deps. CLI + GUI toggle.

### v3.11.0

- **Glob + config-file batch CLI** (roadmap #8) --
  `--pattern`/`--out-dir`/`--config`; JSON overlay supports any
  ProcessingConfig field; mutual validation with `--input`.
- **Crash-resume checkpointing** (roadmap #9) -- SHA-256-fingerprinted
  `.done` markers under `%APPDATA%\VideoSubtitleRemoverPro\checkpoints\`.
  Input size/mtime is part of the key so re-encoded sources are
  correctly re-processed. `--no-resume` bypasses.
- **Preset JSON export / import** (roadmap #15 workflow) -- share
  tuned recipes as single JSON files with a `vsr_preset_format`
  version tag. Collisions with built-ins auto-rename.
- **Live processing preview** (roadmap #7) -- backend
  `on_preview_frame` callback pipes the latest inpainted frame
  into the preview pane at 15 FPS while a batch is running.

### v3.10.0

- **Kalman box tracking** (roadmap #7) -- constant-velocity filter
  per subtitle line smooths OCR jitter, fills single-frame misses,
  stabilises TBE coverage.
- **Perceptual-hash adaptive mask reuse** (roadmap #8) -- skip OCR
  on frames pHash-close to the last detected one; adapts to scene
  content rather than a fixed interval.
- **Colour-tuned mask expansion** (roadmap #2) -- Lab-space cluster
  split inside each detected box; extends the mask to cover serifs,
  drop shadows, decorative strokes.
- **Preset library** (roadmap #26) -- six built-ins (YouTube,
  Anime, Motion-heavy, TikTok, VHS, News chyron) plus user-saved
  presets persisted to `%APPDATA%\VideoSubtitleRemoverPro\presets.json`.

### v3.9.0

- **Scene-cut-aware TBE** (roadmap #11) -- histogram-based scene-cut
  detection inside the TBE batch; each segment aggregates
  independently so background is never mixed across a cut.
- **Flow-warped TBE** (roadmap #6) -- optional Farneback dense
  optical flow aligns every frame in a segment to a reference
  before aggregation. Dramatically cleaner on pans and zooms.
- **Edge-ring colour match** (roadmap #17) -- post-inpaint colour
  correction using a 1-pixel ring just outside the mask. Kills the
  faint colour seam on gradient backgrounds.
- **Auto subtitle-band detection** (roadmap #1) -- 30-frame probe
  + vertical clustering pins the dominant band as `subtitle_area`.
- **Multi-region masks** (roadmap #15) -- `subtitle_areas` field
  accepts a list of rects unioned with per-frame detections.
- **SRT sidecar export** (roadmap #12) -- OCR text collected per
  frame, collapsed into SRT cues with ~0.5s gap tolerance.
- **Debug mask video export** (roadmap #16) -- optional
  `.mask.mp4` of the per-frame binary mask.
- **Adaptive batch sizing** (roadmap #20) -- NVML free-VRAM probe
  scales `sttn_max_load_num` to [8, 512] based on available memory.

### v3.8.0

- **RapidOCR default detector** -- PP-OCR via ONNX Runtime. ~4-5x faster than
  paddlepaddle, no memory leaks, smaller install. Falls through to PaddleOCR
  / Surya / EasyOCR / OpenCV.
- **Temporal Background Exposure (TBE)** -- STTN mode now real video
  inpainting. For each masked pixel, samples neighbouring frames where the
  same pixel is unmasked and reconstructs the true background. Residual
  (always-masked) pixels fall back to cv2.
- **Hybrid ProPainter** -- TBE with higher coverage bar + LaMa residual
  refinement blended 65/35. ProPainter-tier quality on motion-heavy
  footage without the 10+ GB VRAM footprint.
- **Gaussian mask feathering** -- alpha-blended boundary compositing
  eliminates visible cut lines at the edge of the removal region. Applies
  to every inpaint path.
- **Engine badge** -- About dialog surfaces "Temporal BG (TBE)" as an
  always-available inpainting backend and reflects RapidOCR in the
  detection chain.

### v3.7.0

Premium polish pass -- design-token system, custom widgets
(ModernToggle / ModernSlider / SegmentedPicker), onboarding, toast
system, batch summary modal, taskbar progress. See CHANGELOG for the
full list.

### v3.6.0 and earlier

Surya detector, detection frame-skip, mask dilation, hardware encoding
(NVENC / QSV / AMF), PaddleOCR PP-OCRv5, DirectML support, time-range
processing, right-click mask preview, CLI flags. See CHANGELOG for
historical detail.

---

## Near-term (v3.9 -- v4.0)

Ordered by impact. Most of these are additive and do not require external
model-weight downloads.

### Detection

1. **Florence-2 / Qwen2-VL experimental detector** -- gated optional
   dependency. Better multi-lang, layout-aware, but heavy. Sits behind a
   settings toggle for users who want maximum accuracy on stylized text.

### Inpainting

2. **LaMa via ONNX Runtime** -- drop `simple-lama-inpainting` (PyTorch)
   in favour of `Carve/LaMa-ONNX`. 3-5x faster, 70% smaller install,
   CPU-friendly. Keep the PyTorch path as a fallback until ONNX is
   verified on all three GPU vendors.
3. **MI-GAN fast mode** -- optional mode for users who want single-frame
   inpainting at mobile-grade speed. ONNX-based, ~10ms per 512x512 on
   CPU. New `InpaintMode.MI_GAN` enum value.
### Temporal stability

### Workflow

4. **Whisper fallback for hard-coded video without clear text** --
    when OCR confidence drops below a threshold, call `faster-whisper`
    on the audio to generate a synthetic SRT and use its timing to
    mask the bottom-third region. Covers videos where the subtitle is
    anti-aliased past OCR's reliability floor.
### CLI / batch

### Input preprocessing

5. **Pre-detect denoise** -- noisy sources (VHS rips, low-light
    phone clips) starve OCR of contrast and leak flicker into TBE.
    An optional `FastDVDnet` pass on the detection frames (not the
    output) boosts OCR hit rate by ~15% without altering the final
    inpainted video.
    Reference: [m-tassano/fastdvdnet](https://github.com/m-tassano/fastdvdnet).
6. **PySceneDetect-backed scene splitter** -- swap our histogram
    scene-cut heuristic (roadmap #11) for `scenedetect`. Adaptive
    detector handles dissolves / fades / flashes correctly; the
    current threshold approach mis-fires on bright cuts.
    Reference: [Breakthrough/PySceneDetect](https://github.com/Breakthrough/PySceneDetect).
7. **Proxy-file workflow** -- render a low-res proxy for GUI
    preview and tuning; apply the final pass at full resolution
    only when the user commits. Matches the way Premiere / DaVinci
    handle 4K-8K source footage.
8. **Raw frame-sequence input** -- accept DPX / EXR / PNG sequences
    as input (and output) for VFX pipelines that never touch an mp4.
    Wrap the existing video codepath with a virtual "frames-as-video"
    adapter.

### Presets

9. **"Repeat last job" shortcut** -- one-click resubmit of the
    most recent settings on a new file. The queue already stores
    `ProcessingConfig` per item, so this is surfacing what's
    already persisted.

### Specialised OCR

10. **Manga / anime mode** -- route detection through
    `manga-ocr` (vertical Japanese) + `comic-text-detector` (irregular
    speech-bubble shapes). OCR that understands panel layout beats
    generic text detection on anime subs and sign translations.
    Reference: [kha-white/manga-ocr](https://github.com/kha-white/manga-ocr),
    [dmMaze/comic-text-detector](https://github.com/dmMaze/comic-text-detector).
11. **Karaoke / animated subtitle tracking** -- music-video karaoke
    captions morph per-syllable; the current detector sees them as
    constantly-new text. Treat the whole karaoke line as one moving
    object via optical-flow association and mask the union bounds.
12. **Vertical text mode** -- Japanese tategaki and classical
    Chinese read top-to-bottom. Pass a `vertical=True` flag through
    to the OCR backends that support it (RapidOCR, PaddleOCR) and
    rotate the detected boxes accordingly before masking.
13. **Chyron vs subtitle distinction** -- news graphics (station
    logos, persistent lower-thirds, breaking-news tickers) are
    conceptually different from dialogue subtitles. Auto-classify
    each detected box and let users toggle which categories to
    remove.

---

## Mid-term (v4.x)

Heavier integrations; most require model-weight downloads or extra
install steps.

### Real reference implementations

14. **Real ProPainter** (ICCV 2023) -- the actual `sczhou/ProPainter`
    reference with RAFT optical-flow propagation and the recurrent
    transformer. Needs `~400 MB` of model weights auto-downloaded on
    first use. Wire as a new enum value `InpaintMode.PROPAINTER_REAL`,
    keep the current TBE-based "ProPainter" as the default fast path.
    Reference: [sczhou/ProPainter](https://github.com/sczhou/ProPainter).
15. **DiffuEraser** (2025, diffusion) -- state-of-the-art for
    temporal-consistent video inpainting. Combines BrushNet +
    ProPainter + AnimateDiff. Very heavy (8+ GB VRAM, ~5 GB weights)
    -- ship as opt-in "DiffuEraser" mode for users with the hardware.
    Reference: [lixiaowen-xw/DiffuEraser](https://github.com/lixiaowen-xw/DiffuEraser).
16. **Wan2.1-VACE** (Alibaba, ICCV 2025) -- "all-in-one" video
    creation / editing. The 1.3 B-parameter variant is feasible on
    consumer GPUs and its inpainting mode beats ProPainter on motion
    coherence. Opt-in mode with weights auto-fetched from ModelScope.
    Reference: [ali-vilab/VACE](https://github.com/ali-vilab/VACE).
17. **EraserDiT** -- diffusion-transformer video inpainting tuned for
    high-res subtitle removal specifically. Too new to commit to;
    track through 2026 and integrate once weights are stable.
    Reference: [arXiv:2506.12853](https://arxiv.org/html/2506.12853v1).
18. **FloED** -- Optical-Flow guided Efficient Diffusion. Mid-weight
    alternative to DiffuEraser when VACE is overkill but TBE leaves
    visible residuals on fast pans.
    Reference: [NevSNev/FloED](https://github.com/NevSNev/FloED-main).

### Mask / segmentation

19. **SAM 2 mask refinement** -- user clicks inside the
    (already-detected) subtitle region, SAM 2 promotes that click
    into a clean object mask that follows the text exactly.
    Eliminates the need to dilate aggressively to catch serifs. SAM 2
    also has consistent memory propagation, so one seed click carries
    the mask through the whole clip.
    Reference: [facebookresearch/sam2](https://github.com/facebookresearch/sam2).
20. **SAM 3 text-prompt segmentation** -- SAM 3 (released Nov 2025)
    accepts natural-language prompts. One-click "segment all
    burned-in text" without any bounding boxes. Effectively auto-mask.
    Reference: [Meta SAM 3](https://ai.meta.com/sam2/).
21. **MatAnyone 2** (CVPR 2026) -- state-of-the-art video matting.
    Not directly a subtitle tool, but the same memory-propagation
    architecture is exactly what we want for tracking a thin, mobile
    mask (rolling subtitle line) across a clip. Use its trimap-free
    inference as an alternative mask generator for stylised/decorated
    subtitles that OCR+SAM can't clean up.
    Reference: [pq-yang/MatAnyone2](https://github.com/pq-yang/MatAnyone2).

### Acceleration

22. **TensorRT inpainter path** -- compile the LaMa ONNX model to a
    TensorRT engine on first GPU run. Expected ~2-3x further speedup
    on top of ONNX Runtime alone for NVIDIA hardware. Cache compiled
    engine under `%APPDATA%`.
23. **INT8 quantization of the OCR detector** -- RapidOCR ships FP32.
    Post-training quantization via ONNX Runtime Quantizer should cut
    detection cost ~50% on CPU with <0.5% accuracy loss.
24. **Batched LaMa inference** -- `simple-lama-inpainting` runs one
    frame at a time. Re-expose the underlying model and pipe full
    batches (our batch size is 30) through a single forward pass.
    Dominant speedup for LAMA and ProPainter-hybrid modes.
25. **PyNvVideoCodec hardware decode** -- replace `cv2.VideoCapture`
    with NVIDIA's PyNvVideoCodec for NVIDIA users. Measured ~6x
    faster decode (91 fps vs 15 fps on reference hardware), plus
    decoded frames live on the GPU already -- zero CPU-GPU copies
    when the subsequent detect/inpaint is also on GPU.
    Reference: [PyNvVideoCodec](https://developer.nvidia.com/pynvvideocodec).
26. **Prefetch / pipeline parallelism** -- read frame N+1 on a worker
    thread while frame N is being inpainted. The GIL is released in
    cv2/numpy/onnxruntime calls, so simple threading is enough.
    Current code is strictly serial -- easy win of ~20% on any
    hardware.
27. **RIFE-interpolated fast mode** -- detect+inpaint every Nth frame,
    synthesise intermediates with RIFE from the cleaned keyframes.
    Equivalent to 2-4x throughput when the background is smooth
    (dialogue scenes). Falls back gracefully on scene cuts (which
    RIFE handles by duplicating the nearer frame).
    Reference: [hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE).

### Format support

28. **10-bit / HDR pipeline** -- the current cv2 pipeline processes
   in 8-bit BGR, clamping HDR10 / HLG / Dolby Vision to SDR.
   Re-plumb as 16-bit numpy preserving `_Matrix` / `_Transfer` /
   `_Primaries` metadata; output via `ffmpeg -c:v libx265
   -pix_fmt yuv420p10le -color_primaries bt2020`. Biggest single
   quality win for anyone editing modern phone footage.
29. **AV1 + VP9 ingest/egress** -- YouTube's native web format is
   AV1, older YouTube and most WebM is VP9. Verify decode across
   all codepaths and expose `output_format=av1` (`libsvtav1` for
   speed, `libaom-av1` for quality).
30. **VapourSynth bridge** -- optional bridge that accepts a
   `.vpy` script as input and emits one as output. Lets advanced
   users slot VSR into a larger chain (QTGMC deinterlace, Waifu2x
   upscale, SMDegrain denoise).
   Reference: [vapoursynth/vapoursynth](https://github.com/vapoursynth/vapoursynth).
31. **NLE round-trip** -- when input is an EDL / XML from Premiere
   or DaVinci, emit a matching sidecar with the cleaned video
   substituted. Keeps VSR inside a post pipeline.

### Quality + testing

32. **Reference-clip regression harness** -- a `tests/clips/`
   directory of 10-20 short clips covering edge cases
   (motion-heavy, static, karaoke, vertical JP, HDR, thin font,
   thick font, dissolve cuts). Nightly GHA run compares output
   hashes + metric scores to committed baselines.
33. **Community edge-case corpus** -- GitHub Discussions thread
   where users submit 10-second problem clips + settings. We
   periodically fold the worst performers into the regression
   harness and ship baseline fixes.

### Audio

34. **Multi-track audio passthrough** -- mux all N input audio
   streams unchanged (today we merge only the first). DVD / Bluray
   rips routinely ship 3-5 language tracks; current code silently
   drops them.
35. **Loudness normalisation** -- optional `ffmpeg -af
   loudnorm=I=-16` pass to bring batches to a consistent EBU R128
   target. Useful for platform-specific output (YouTube -14 LUFS,
   Apple -16 LUFS, broadcast -23 LUFS).
36. **Audio-guided subtitle validation** -- cross-reference
   faster-whisper word timestamps against detected text boxes:
   boxes with no nearby speech are likely chyrons / captions, not
   dialogue. Feeds the chyron-vs-subtitle classifier.

### Post-processing

37. **Real-ESRGAN output upscale** -- optional "enhance" stage after
    inpainting. Users processing SD-era footage often want 2x / 4x
    upscale afterwards; bundling the step means one pipeline instead
    of hand-offs between tools. ONNX Runtime friendly, already fits
    our infra.
    Reference: [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).
38. **SwinIR restoration pass** -- alternative to Real-ESRGAN for
    footage where the subtitle removal leaves subtle blur (happens
    on dense text); SwinIR restores local detail better than generic
    upscalers.
    Reference: [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR).
39. **Film-grain re-synthesis** -- cinematic footage has uniform
    grain; inpainted regions come back grain-free and stand out on
    motion. A cheap additive-noise pass matched to the per-frame
    grain profile (estimated from an unmasked crop) re-integrates
    the inpainted region.

### UX

40. **Subtitle area drag-refinement in-GUI** -- current region
    selector is modal and one-shot. Allow live-drag adjustment on
    the preview pane with instant mask re-render, so the user sees
    the mask update as they nudge edges.
41. **Per-file overrides surface** -- every queue item already
    carries its own `ProcessingConfig`. Expose this in a themed
    "edit this item" popover so users can set different language /
    region / mode per file without cloning the global config.
42. **Drag-drop onto the app icon** -- register VSR as a file handler
    for the target video extensions; double-clicking an `.mp4`
    launches it straight into the queue.
43. **A/B flicker-scrubber** -- horizontal slider in the preview pane
    reveals original vs cleaned video, scrubbable with mouse drag.
    Makes it easy to spot residual flicker or colour mismatch before
    exporting.
44. **Quality self-test** -- for each finished file, sample 10 random
    frames, render the mask region side-by-side (original / cleaned)
    into a summary sheet, and show an inline "Quality: Good / Review"
    tag based on a simple SSIM threshold on unmasked regions.
    Flags batches that need re-runs without full manual review.

---

## Long-term (v5+)

Speculative. Anchored to projects that are under active research right
now and may not be practical to integrate for 12+ months.

45. **Plugin architecture** -- `backend/plugins/*.py` auto-discovery
    for custom detectors and inpainters. Lets us ship the core app
    as a thin runtime and iterate on model integrations
    independently.
46. **Distributed / multi-GPU** -- split the frame batch across GPUs,
    with a coordinator that handles the temporal window crossing
    device boundaries. Pays off for 4K / feature-length content.
47. **Headless REST server mode** -- `python VideoSubtitleRemover.py
    --serve 8888` exposes a local HTTP API (submit job, poll status,
    stream preview). Sets up the app for integration into other
    desktop tools and NAS workflows.
48. **Web build** -- port the GUI to a tauri / webview shell backed
    by the existing Python core. Keeps the native Windows feel while
    making macOS / Linux parity realistic.
49. **Subtitle translation pipeline** -- after SRT export, route
    through a local LLM (gemma3 / qwen3) for translation, then
    optionally re-burn translated subtitles into the cleaned video.
    End-to-end "re-dub in another language" workflow.
50. **Streaming mode** -- process a file while still being written
    (e.g. OBS recording, DVR capture). Tail the input and emit a
    cleaned output with a fixed N-second lag. Requires re-architecting
    the temporal window to be a streaming buffer rather than a
    fixed batch, but unlocks live-broadcast workflows.
51. **WebGPU / WASM fallback** -- for environments where installing
    Python is impossible (locked-down corporate, chromebooks),
    compile the ONNX inpaint + detect path to WebGPU via
    `onnxruntime-web`. Ship as a drag-and-drop web page. Not a
    replacement for the desktop app -- an escape hatch.
52. **Font-aware inpainting** -- classify the subtitle font style
    (Hanzi vs Latin vs Cyrillic vs decorative) via a small classifier
    on the detected boxes; apply per-class mask dilation and feather
    presets. Chinese characters need tighter masks than handwritten
    Latin karaoke captions.
53. **Logo / watermark mode** -- specialised mode that persists the
    mask across the entire video (static watermark case) with TBE
    using the full video as the temporal window rather than a
    30-frame batch. Subsumes the current "fixed subtitle area"
    workflow and extends it to full-video aggregation.

### Platforms

54. **Android port** -- Kotlin + Compose shell around the Python
   core via Chaquopy, with `ffmpeg-kit` for video I/O and NNAPI
   / Qualcomm AIP hooks for the inpainter. Detection runs on the
   device NPU if present (Tensor / Pixel Visual Core / QNN).
   Target: v1 lite supports image-only, v2 adds video.
   Reference: [arthenica/ffmpeg-kit](https://github.com/arthenica/ffmpeg-kit).
55. **macOS / Linux parity** -- the PyInstaller build is
   Windows-only; port the build workflow to a three-OS GHA
   matrix. tkinter ships on both platforms, FFmpeg and OpenCV
   work identically -- the blockers are Windows-specific pieces
   (taskbar progress, winsound beeps, console hiding).
56. **iOS / iPadOS port** -- long shot. Likely via a
   BeeWare / Pyto-style bridge rather than Chaquopy (which is
   Android-only). Realistic only once Apple silicon + NeuralEngine
   inpainting is plumbed.

### Accessibility + i18n

57. **Screen-reader support** -- wire tkinter widgets to Windows
   UI Automation (UIA) providers so NVDA / Narrator can announce
   status changes (items queued, progress, completion). Today
   none of the custom widgets are announceable.
58. **High-contrast theme variant** -- optional theme preset
   respecting Windows' `HighContrast` active scheme. The existing
   design-token system makes this cheap: swap the palette
   constants, keep the widget code.
59. **GUI localization** -- `gettext`-based string externalisation
   with `.mo` files for the top ~10 languages we already detect
   in subtitles (ironically the GUI itself is English-only).
   Community-contributable via Weblate or a simple PR flow.
60. **Right-to-left UI support** -- Arabic / Hebrew mirroring for
   text, button ordering, queue scroll direction. Secondary to
   #F.

### Workflows

61. **Restyle mode** -- after subtitle removal + SRT export,
   optionally re-burn a translated / restyled subtitle track at
   the same position using ffmpeg's `subtitles` filter with an
   .ass template. Users get "clean + re-dub-in-text" in one
   round-trip.
62. **Watermark addition pipeline** -- content-creator mode: after
   cleaning an input, burn a user-specified PNG watermark at a
   configurable corner with fade-in/out. Lets the tool also ship
   content for platforms that require branding.
63. **AI chat interface** -- optional side-panel wired to a local
   LLM (gemma3 / qwen3 via llama.cpp) that accepts natural
   language like "remove the bottom banner between 0:15 and 0:45,
   keep the song title top-right." LLM translates to
   `ProcessingConfig` mutations. Useful for power users with
   complex per-segment overrides.
64. **Live performance dashboard** -- in-app panel showing current
   detection FPS, inpaint FPS, GPU VRAM, disk I/O, queued /
   completed / failed counts. Pure local; no telemetry leaves
   the machine.

---

## Explicitly not on the roadmap

- **Cloud API integrations** (OpenAI Whisper API, Anthropic, etc.).
  VSR is offline-first by design. Local-only Whisper / Qwen are fine;
  remote APIs add friction and privacy cost.
- **Docker / containerised builds.** The tool is Windows-first and
  ships as a PyInstaller binary. Containers add install complexity
  for the 95% Windows user base.
- **Subscription model / telemetry.** The project is a free
  single-binary tool. No analytics, no phone-home.

---

## References

Research sources used to compile this roadmap, grouped by category:

### Subtitle removal / upstream
- [YaoFANGUK/video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover) -- upstream project

### Video inpainting
- [sczhou/ProPainter (ICCV 2023)](https://github.com/sczhou/ProPainter) -- real video inpainting reference
- [lixiaowen-xw/DiffuEraser](https://github.com/lixiaowen-xw/DiffuEraser) -- 2025 diffusion-based video inpainting
- [ali-vilab/VACE (ICCV 2025)](https://github.com/ali-vilab/VACE) -- Alibaba Wan2.1 all-in-one video edit
- [NevSNev/FloED](https://github.com/NevSNev/FloED-main) -- optical-flow guided diffusion inpainting
- [EraserDiT (arXiv:2506.12853)](https://arxiv.org/html/2506.12853v1) -- diffusion-transformer video inpainting
- [hitachinsk/FGT (ECCV 2022)](https://github.com/hitachinsk/FGT) -- flow-guided transformer

### Image inpainting
- [TencentARC/BrushNet (ECCV 2024)](https://github.com/TencentARC/BrushNet) -- plug-and-play dual-branch
- [advimman/lama (WACV 2022)](https://github.com/advimman/lama) -- LaMa reference
- [Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX) -- LaMa ONNX export
- [Picsart-AI-Research/MI-GAN (ICCV 2023)](https://github.com/Picsart-AI-Research/MI-GAN) -- mobile-fast inpainting
- [TurboFill (2025)](https://liangbinxie.github.io/projects/TurboFill/) -- few-step diffusion inpaint

### Text detection / OCR
- [RapidAI/RapidOCR](https://github.com/RapidAI/RapidOCR) -- PP-OCR ONNX runtime
- [PaddleOCR 3.0 Technical Report](https://arxiv.org/html/2507.05595v1)
- [Florence-2](https://huggingface.co/microsoft/Florence-2-base) -- VLM with OCR + grounding
- [Qwen2-VL](https://arxiv.org/html/2409.12191v1) -- VLM with bounding-box output
- [kha-white/manga-ocr](https://github.com/kha-white/manga-ocr) -- Japanese manga OCR
- [dmMaze/comic-text-detector](https://github.com/dmMaze/comic-text-detector) -- irregular bubble detection

### Segmentation / matting
- [facebookresearch/sam2](https://github.com/facebookresearch/sam2) -- SAM 2 video segmentation
- [Meta SAM 3](https://ai.meta.com/sam2/) -- text-promptable segmentation (Nov 2025)
- [pq-yang/MatAnyone2 (CVPR 2026)](https://github.com/pq-yang/MatAnyone2) -- video matting with memory propagation
- [PeterL1n/RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) -- real-time RVM

### Audio / transcription
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) -- 4x faster Whisper inference

### Acceleration
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) -- GPU inference acceleration
- [PyNvVideoCodec](https://developer.nvidia.com/pynvvideocodec) -- NVIDIA GPU-resident video decode
- [ONNX Runtime](https://onnxruntime.ai/) -- cross-vendor inference runtime

### Frame interpolation / super-resolution
- [hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE) -- real-time frame interpolation
- [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) -- video super-resolution
- [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR) -- image restoration

### Preprocessing / denoising
- [m-tassano/fastdvdnet](https://github.com/m-tassano/fastdvdnet) -- real-time video denoising
- [Breakthrough/PySceneDetect](https://github.com/Breakthrough/PySceneDetect) -- scene-cut detection
- [vapoursynth/vapoursynth](https://github.com/vapoursynth/vapoursynth) -- scriptable video processing

### Quality metrics / testing
- [chaofengc/IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) -- PSNR / SSIM / LPIPS toolbox
- [JunyaoHu/common_metrics_on_video_quality](https://github.com/JunyaoHu/common_metrics_on_video_quality) -- FVD + friends
- [richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity) -- LPIPS reference

### Mobile / platforms
- [arthenica/ffmpeg-kit](https://github.com/arthenica/ffmpeg-kit) -- cross-platform FFmpeg wrappers
- [Chaquopy](https://chaquo.com/chaquopy/) -- Python on Android

## Open-Source Research (Round 2)

### Related OSS Projects
- https://github.com/YaoFANGUK/video-subtitle-remover — reference STTN/LAMA/ProPainter pipeline
- https://github.com/YaoFANGUK/video-subtitle-extractor — companion OCR for SRT extraction
- https://github.com/JollyToday/GhostCut_Remove_Video_Text — multilingual OCR + inpainting (EN/CN/JA/KO/AR)
- https://github.com/rainwl/VideoRemoveText — lightweight LaMa + fixed ROI + color threshold
- https://github.com/Rats20/EraseSubtitles — multilingual text detection + inpainting research
- https://github.com/Keaneo/Scrubtitles — minimal script reference
- https://github.com/advimman/lama — upstream LaMa inpainting weights
- https://github.com/sczhou/ProPainter — upstream ProPainter (high-VRAM, best motion)
- https://github.com/researchmm/STTN — upstream STTN reference

### Features to Borrow
- Multi-backend inpaint switch: sttn-auto / sttn-det / lama / propainter / opencv (YaoFANGUK VSR)
- DirectML execution provider for AMD/Intel GPUs in addition to CUDA (YaoFANGUK VSR)
- Docker images keyed by hardware variant (CUDA 11.8/12.6/12.8, DirectML, CPU) (YaoFANGUK VSR)
- Paired extractor → remover workflow: OCR to SRT first, then remove (YaoFANGUK pair)
- Fixed-ROI + color-threshold fast path for speed when text location is stable (rainwl)
- Auto device pick: cuda → mps → cpu with CLI override (rainwl)
- Multilingual text detection module covering CJK + Arabic scripts (GhostCut, EraseSubtitles)
- Reference-frame picker UI so user can hint STTN which frames have clean background
- VRAM budget slider that down-selects available backends at runtime
- Batch mode for a folder of clips sharing the same subtitle ROI

### Patterns & Architectures Worth Studying
- Pluggable inpainter interface — each backend implements (detect, inpaint, dispose) so new models drop in (YaoFANGUK VSR)
- First-run weight download with checksum verify, cache under %LOCALAPPDATA% (rainwl, LaMa)
- ONNXRuntime + DirectML for non-NVIDIA GPUs rather than forcing CUDA (YaoFANGUK VSR)
- Frame-sequence temp cache as PNG chunks so a crashed run resumes from last completed frame
- Offload inference to a child process so GUI stays responsive and OOM doesn't kill the app
