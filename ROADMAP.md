# Video Subtitle Remover Pro -- Roadmap

> Living document. Shipped log + ordered backlog + research bench.
> Items ranked by value, not calendar date.
> Version line: **v3.13 planning cycle**, revised 2026-05-17.

Reading order:

1. [Shipped](#shipped) -- what is in the released binary today
2. [Now](#now-v313-in-flight) -- in-flight work in `Unreleased`
3. [Next](#next-v314--v40) -- ordered near-term backlog
4. [Later](#later-v4x) -- mid-term, usually needs new dependencies or weights
5. [Under consideration](#under-consideration-v5) -- speculative / research-bench
6. [Rejected](#explicitly-not-on-the-roadmap) -- consciously out of scope
7. [Themes](#themes) -- the categories every Now/Next item maps to
8. [Appendix: sources](#appendix-sources) -- every external reference

---

## Project philosophy (non-negotiable)

These constraints are inherited from the existing codebase and CLAUDE.md.
Every proposal below has been checked against them; misfits are explicitly
flagged in the [rejected](#explicitly-not-on-the-roadmap) section.

- **Offline-first.** No cloud APIs, no telemetry, no phone-home. Local
  Whisper / Qwen / LLM is fine; remote inference is not.
- **Single binary, no container.** Tool ships as a PyInstaller `--onedir`
  build with a first-run bootstrap batch script. Containers add friction for
  the dominant Windows user base.
- **Windows-first.** macOS / Linux ports are welcome but cannot block a
  Windows release.
- **MIT-clean.** GPL dependencies (Surya, manga-image-translator) are gated
  behind opt-in installs and never bundled in the default release.
- **Pure-ASCII source.** Every `.py` and `.bat` is plain ASCII so PyInstaller,
  signtool, and arbitrary terminals never trip on a stray Unicode character.
- **No mandatory model downloads.** Default install ships with what's needed
  for the AUTO / STTN / LAMA path. Heavier modes (real ProPainter,
  DiffuEraser, VACE) are opt-in and download weights on first use.
- **Never silently drop user data.** Audio passthrough, time ranges, subtitle
  area, presets, window geometry -- all persisted and restored.

---

## Shipped

### v3.12.0 (current release, 2026-04-17)

- **AUTO inpaint mode / per-batch quality tier** -- new `InpaintMode.AUTO`
  routes each TBE batch between temporal (TBE) and spatial (LaMa) inpainting
  based on a per-batch exposure score vs `auto_exposure_threshold` (default
  0.55). Lazy-loads LaMa only when routing calls for it.
- **Keyframe-driven detection** -- `ffprobe` enumerates I-frames; OCR runs only
  at keyframes, Kalman propagates masks between. Falls back to pHash skip when
  ffprobe is missing.
- **Deinterlace on ingest** -- `ffprobe idet` detects combing; `ffmpeg yadif`
  writes a temp progressive source. Auto + manual toggles. Fixes comb-artefact-
  induced OCR junk on DV / broadcast rips.
- **PSNR / SSIM quality report** -- 10-frame deterministic sample after each
  run. Pure-numpy SSIM, cv2 PSNR, no new dependency. CLI + GUI toggle.

### v3.11.0

- **Glob + config-file batch CLI** -- `--pattern` / `--out-dir` / `--config`;
  JSON overlay supports any `ProcessingConfig` field; mutual validation with
  `--input` / `--output`.
- **Crash-resume checkpointing** -- SHA-256-fingerprinted `.done` markers
  under `%APPDATA%\VideoSubtitleRemoverPro\checkpoints\`. Input size/mtime
  is part of the key so re-encoded sources are correctly re-processed.
  `--no-resume` bypasses.
- **Preset JSON export / import** -- share tuned recipes as a single JSON
  file with a `vsr_preset_format` version tag. Collisions with built-ins
  auto-rename.
- **Live processing preview** -- backend `on_preview_frame` callback pipes
  the latest inpainted frame into the preview pane at 15 FPS while a batch
  is running.

### v3.10.0

- **Kalman box tracking** -- constant-velocity filter per subtitle line
  smooths OCR jitter, fills single-frame misses, stabilises TBE coverage.
- **Perceptual-hash adaptive mask reuse** -- skip OCR on frames pHash-close
  to the last detected one; adapts to scene content rather than a fixed
  interval.
- **Colour-tuned mask expansion** -- Lab-space cluster split inside each
  detected box extends the mask to cover serifs, drop shadows, decorative
  strokes the OCR bbox clips.
- **Preset library** -- six built-ins (YouTube, Anime, Motion-heavy, TikTok,
  VHS, News chyron) plus user-saved presets persisted to
  `%APPDATA%\VideoSubtitleRemoverPro\presets.json`.

### v3.9.0

- **Scene-cut-aware TBE** -- histogram-based scene-cut detection inside the
  TBE batch; each segment aggregates independently.
- **Flow-warped TBE** -- optional Farneback dense optical flow aligns every
  frame in a segment to a reference before aggregation.
- **Edge-ring colour match** -- post-inpaint colour correction using a thin
  ring just outside the mask. Kills the faint colour seam on gradients.
- **Auto subtitle-band detection** -- 30-frame probe + vertical clustering
  pins the dominant band as `subtitle_area`.
- **Multi-region masks** -- `subtitle_areas` accepts a list of rects unioned
  with per-frame detections.
- **SRT sidecar export** -- OCR text collected per frame, collapsed into SRT
  cues with ~0.5s gap tolerance.
- **Debug mask video export** -- optional `.mask.mp4` of the per-frame binary
  mask.
- **Adaptive batch sizing** -- NVML free-VRAM probe scales `sttn_max_load_num`
  to [8, 512] based on available memory.

### v3.8.0

- **RapidOCR default detector** -- PP-OCR via ONNX Runtime, 4-5x faster than
  paddlepaddle, leak-free. Falls through to PaddleOCR / Surya / EasyOCR /
  OpenCV. Validates the published RapidOCR claim that ONNX path is
  memory-stable where the PaddlePaddle build is not
  ([PaddleOCR issue #11639](https://github.com/PaddlePaddle/PaddleOCR/issues/11639)).
- **Temporal Background Exposure (TBE)** -- STTN mode is real video
  inpainting: each masked pixel is reconstructed from neighbouring frames
  where the same pixel is unmasked.
- **Hybrid ProPainter** -- TBE with a higher coverage bar + LaMa residual
  refinement blended 65/35. ProPainter-tier quality on motion-heavy footage
  without the 10+ GB VRAM footprint.
- **Gaussian mask feathering** -- alpha-blended boundary compositing
  eliminates visible cut lines.
- **Engine badge** -- About dialog surfaces "Temporal BG (TBE)" as an
  always-available inpainting backend.

### v3.7.0

Premium polish pass -- design-token system, custom widgets (ModernToggle /
ModernSlider / SegmentedPicker), onboarding, toast system, batch summary
modal, Windows taskbar progress, first-run onboarding modal. See the
[CHANGELOG](./CHANGELOG.md) for the full per-widget list.

### v3.6.0 and earlier

Surya detector, detection frame-skip, mask dilation, hardware encoding
(NVENC / QSV / AMF), PaddleOCR PP-OCRv5, DirectML support, time-range
processing, right-click mask preview, CLI flags. See the
[CHANGELOG](./CHANGELOG.md) for historical detail.

---

## Now (v3.13, in flight)

These are the items already partially landed in the `Unreleased` section of
[CHANGELOG.md](./CHANGELOG.md). v3.13 is a hardening + UX-polish release;
the goal is to ship them as a single tag rather than dribbling each fix.

### Hardening (already landed on `main`)

1. **Shutdown race fix** -- `_shutdown_started` is set only after user
   confirms close-while-processing, so the racing `_on_processing_complete`
   callback no longer tears down the root window mid-dialog.
2. **Collision-proof queue item IDs** -- `uuid.uuid4()` replaces the
   millisecond-timestamp ID; safe when many files arrive in the same tick.
3. **NaN/inf-safe coercers** -- `_coerce_int` / `_coerce_float` reject
   non-finite floats and fall back to a default. Matches the backend guard.
4. **`from_dict` pre-sanitisation** -- `subtitle_area` and `subtitle_areas`
   in `ProcessingConfig.from_dict` go through `_coerce_rect` /
   `_coerce_rect_list` instead of raw `tuple()` calls.
5. **`_write_srt` zero-fps guard** -- replaces `fps or 30.0` with a real
   `fps > 1.0` check so near-zero but non-falsy fps values can't produce
   absurd SRT timestamps.
6. **`_load_json_config` 1 MB cap** -- rejects oversized settings files
   before parsing.
7. **`detect_ai_engines()` Surya guard** -- broadened from `ImportError` to
   `Exception` so a partially installed Surya doesn't crash engine probing.
8. **CLI Ctrl-C** -- clean exit code 130 instead of a raw traceback.

### UI/UX polish (already landed)

9. **Workflow step pills** (Import / Inspect / Run) now actually render and
   highlight the current stage.
10. **Section eyebrow labels** -- `_section_title(eyebrow=...)` actually
    renders the small-caps meta label above the title.
11. **Log badge order + pluralisation** -- "1 warning" / "2 warnings",
    correctly positioned between title and toggle.
12. **Tighter footer microcopy** -- 13-word line down to "Add files, review a
    sample frame, then start."
13. **Activity-log height** 5 -> 6 rows; progress-bar height parity at 5 px.

### Tests (already landed)

14. **`CoerceHardeningTests`** -- NaN/inf for `_coerce_int` / `_coerce_float`,
    non-iterable `subtitle_area` / `subtitle_areas` in `from_dict`.
15. **`BackendWriteSrtTests`** -- zero fps and near-zero fps fall back to 30.
16. **`LoadJsonConfigTests`** -- oversized config file rejected before parse.

### Still to do before v3.13 tag

17. **[x] Pin torch >= 2.10 to dodge CVE-2026-24747** -- `torch.load`
    weights_only RCE; 2.9.1 and earlier are vulnerable. Bumped in
    [requirements.txt](./requirements.txt),
    [setup.py](./setup.py), and the GHA build matrix. The torch-directml
    install path still pins torch 2.4.x because no patched torch-directml
    wheel exists yet; `setup.py` warns the user on that branch.
    Source: [GHSA-63cw-57p8-fm3p](https://github.com/advisories/GHSA-63cw-57p8-fm3p),
    [GHSA-53q9-r3pm-6pq6](https://github.com/advisories/GHSA-53q9-r3pm-6pq6) (CVE-2025-32434).
18. **[x] Pin Pillow >= 11.3.0** -- CVE-2026-25990 PSD-loader out-of-bounds
    write. Pinned in [requirements.txt](./requirements.txt) and the GHA
    install line.
    Source: [CVE-2026-25990](https://www.appsecure.security/vulnerability-database/cve-2026-25990/).
19. **[x] Pin opencv-python >= 4.12.0** -- CVE-2025-53644, uninitialised
    pointer in JPEG read. Pinned in [requirements.txt](./requirements.txt)
    and the GHA install line.
    Source: [NVD CVE-2025-53644](https://nvd.nist.gov/vuln/detail/cve-2025-53644).
20. **[x] Settings-schema versioning** -- `VSR_SETTINGS_FORMAT = 1` constant
    + `_migrate_settings()` shim in
    [VideoSubtitleRemover.py](./VideoSubtitleRemover.py); `to_dict()`
    stamps the version; `load_settings()` runs the migration before
    `from_dict`. Future-version files load as-is (no downgrade); legacy /
    garbage version fields are treated as 0 and stamped. Tests in
    [tests/test_hardening.py](./tests/test_hardening.py) `SettingsMigrationTests`.

---

## Next (v3.14 -- v4.0)

Ordered by impact within each theme. Most items are additive and do not
require model-weight downloads.

### Detection

21. **TransNetV2 deep scene-cut detector** -- replace (or fall back behind)
    our histogram heuristic with a real CNN. Histogram approach mis-fires
    on bright cuts and dissolves; TransNetV2 hits F1 ~0.80 vs PySceneDetect's
    ~0.6 on the SHOT dataset. Optional install; histogram path stays as the
    zero-dep default.
    Source: [TransNetV2](https://github.com/soCzech/TransNetV2),
    [AutoShot (NAS extension)](https://arxiv.org/abs/2304.06116).
22. **Florence-2 / Qwen2.5-VL experimental detector** -- gated optional
    dependency. Better multi-lang, layout-aware, but heavy. Settings toggle
    for users who want maximum accuracy on stylised text. Qwen2.5-VL leads
    the OmniDocBench leaderboard as of April 2026.
    Source: [Qwen2.5-VL blog](https://qwenlm.github.io/blog/qwen2.5-vl/),
    [OmniDocBench](https://github.com/opendatalab/OmniDocBench).
23. **PaddleOCR-VL 0.9B detector tier** -- alternative VLM-OCR with
    irregular-polygon bbox support; reportedly beats GPT-4o on OmniDocBench
    v1.5 at 94.5% accuracy. Drop-in priority above PaddleOCR proper for
    users on a CUDA box. Optional install.
    Source: [PaddleOCR-VL on HF](https://huggingface.co/PaddlePaddle/PaddleOCR-VL),
    [paper arXiv:2510.14528](https://arxiv.org/abs/2510.14528).
24. **Vertical-text mode** -- pass `vertical=True` through to RapidOCR /
    PaddleOCR for Japanese tategaki / classical Chinese; rotate boxes
    before masking. Cheap, big quality win on legacy CJK content.

### Inpainting

25. **LaMa via ONNX Runtime** -- drop `simple-lama-inpainting` (PyTorch) in
    favour of `Carve/LaMa-ONNX`. 3-5x faster, 70% smaller install, CPU-
    friendly. Keep the PyTorch path as a fallback until ONNX is verified
    on all three GPU vendors. Also closes the torch.load CVE blast radius.
    Source: [Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX).
26. **MI-GAN fast mode** -- new `InpaintMode.MI_GAN` for single-frame
    inpainting at mobile-grade speed. ONNX-based, ~10 ms per 512x512 on
    CPU per the ICCV 2023 paper. Great fallback for thumbnails / image
    queues / underpowered laptops.
    Source: [Picsart-AI-Research/MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN),
    [IOPaint MI-GAN model card](https://www.iopaint.com/models/erase/migan).
27. **Whisper fallback when OCR confidence floor is hit** -- when detected
    text confidence drops below a threshold, call `faster-whisper` on the
    audio to synthesise an SRT and use its timing to mask the bottom band.
    Covers videos where the subtitle is anti-aliased past OCR reliability.
    Source: [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

### Workflow

28. **"Repeat last job" shortcut** -- one-click resubmit of the most recent
    settings on a new file. The queue already stores `ProcessingConfig`
    per item, so this surfaces what's already persisted.
29. **Per-file overrides surface** -- every queue item already carries its
    own `ProcessingConfig`. Expose this in a themed popover so users can
    set different language / region / mode per file without cloning the
    global config.
30. **A/B flicker-scrubber in preview pane** -- horizontal slider reveals
    original vs cleaned video. Makes residual flicker / colour mismatch
    obvious before export.
31. **Subtitle-area drag-refinement in-GUI** -- current region selector is
    modal and one-shot; allow live-drag adjustment on the preview pane with
    instant mask re-render.

### Input preprocessing

32. **PySceneDetect-backed scene splitter** (zero-dep histogram stays as
    fallback) -- swaps the heuristic for the `scenedetect` AdaptiveDetector
    which handles dissolves and flashes correctly. Pairs with #21 for the
    deep-learning tier.
    Source: [Breakthrough/PySceneDetect](https://github.com/Breakthrough/PySceneDetect).
33. **Pre-detect denoise (FastDVDnet)** -- noisy sources (VHS rips, low-light
    phone clips) starve OCR of contrast and leak flicker into TBE. Optional
    FastDVDnet pass on the *detection* frames (not the output) boosts OCR
    hit rate by ~15%.
    Source: [m-tassano/fastdvdnet](https://github.com/m-tassano/fastdvdnet).
34. **Proxy-file workflow** -- render a low-res proxy for GUI preview and
    tuning; apply the final pass at full resolution only when the user
    commits. Matches the way Premiere / DaVinci handle 4K-8K source.
35. **[partial] Raw frame-sequence input** -- new
    `_FrameSequenceCapture` adapter walks a directory of PNG / JPG /
    BMP / TIFF / WebP files in sorted filename order and exposes the
    `cv2.VideoCapture` surface (`isOpened` / `read` / `set(POS_FRAMES)`
    / `get(FPS / WIDTH / HEIGHT / FRAME_COUNT)` / `release`). First
    frame fixes dimensions; mid-sequence size changes are letterboxed.
    `_open_capture` detects directory inputs and routes accordingly;
    `process_video` skips the audio merge when input is a directory.
    `--input-fps FPS` sets the synthesised stream rate (default 24).
    *Ingest only* for v3.13 -- output remains mp4. Sequence *output*
    (writing a directory of PNGs back out) is queued as a follow-up.

### CLI / batch

36. **[x] `--validate-config` dry-run** -- prints the resolved
    `ProcessingConfig` (after CLI flags + `--config` overlay normalisation)
    as JSON and exits 0 without instantiating the detector / inpainter.
    `--input` / `--output` are not required when this flag is set.
    Wired in [backend/processor.py](./backend/processor.py) `main()`.
37. **[x] `--skip-existing` toggle independent of checkpointing** -- skips
    any input whose output path already exists, regardless of the checkpoint
    store. Independent of `--no-resume`. Wired in `_process_one()` before
    the checkpoint check so the cheaper path runs first.

### Acceleration

38. **[x] OpenCV `CAP_PROP_HW_ACCELERATION` opt-in** -- new
    `_open_capture()` helper accepts `off` (default) / `auto` / `any` /
    `d3d11` / `vaapi` / `mfx`. Probes one frame after open; on empty-frame
    response (opencv/opencv#25185) silently re-opens with the software
    backend and warns. Wired into the main `process_video` decode site;
    the shorter scan sites (`detect_subtitle_band`, quality-report
    sampler) stay on software pending separate validation. CLI exposes
    `--decode-accel`.
    Source: [OpenCV hwaccel docs](https://docs.opencv.org/4.x/db/dc4/group__videoio__hwaccel.html).
39. **INT8 quantisation of the OCR detector** -- RapidOCR ships FP32.
    Post-training quantisation via ONNX Runtime Quantizer should cut
    detection cost ~50% on CPU with <0.5% accuracy loss.
40. **Batched LaMa inference** -- `simple-lama-inpainting` runs one frame
    at a time; re-expose the underlying model and pipe full batches (our
    batch size is 30) through a single forward pass. Dominant speedup for
    LAMA and ProPainter-hybrid.
41. **[x] Prefetch / pipeline parallelism** -- new `_PrefetchReader`
    wraps `cv2.VideoCapture` with a daemon worker thread feeding a
    bounded queue (default `max(8, batch_size * 2)`). Strict ownership:
    the wrapped cap is owned by the worker; main-thread `.set` / `.get`
    / `.read` against the raw cap would race. Cleanup goes through
    `reader.release()`, which sets a stop event, drains the queue so
    the worker isn't blocked on `put()`, joins the thread, then
    releases the underlying cap. The exception path in `process_video`
    routes through the same release so a mid-batch crash never leaks
    the thread. Toggle via `--no-prefetch`; queue size via
    `--prefetch-queue N`. Tests in
    [tests/test_hardening.py](./tests/test_hardening.py) `PrefetchReaderTests`
    cover ordered delivery, release-with-full-queue, and post-EOF
    idempotency.

### Specialised OCR

42. **Manga / anime mode** -- route detection through `manga-ocr` (vertical
    Japanese) + `comic-text-detector` (irregular speech-bubble shapes).
    OCR that understands panel layout beats generic text detection on
    anime subs and sign translations.
    Source: [kha-white/manga-ocr](https://github.com/kha-white/manga-ocr),
    [dmMaze/comic-text-detector](https://github.com/dmMaze/comic-text-detector).
43. **Karaoke / animated subtitle tracking** -- music-video karaoke
    captions morph per-syllable; the current detector sees them as
    constantly-new text. Treat the whole karaoke line as one moving object
    via optical-flow association and mask the union bounds.
44. **Chyron vs subtitle distinction** -- news graphics (station logos,
    persistent lower-thirds, breaking-news tickers) are conceptually
    different from dialogue subtitles. Auto-classify each detected box
    and let the user toggle which categories to remove.
45. **WhisperX-aided chyron classifier** -- pairs with #44: cross-reference
    `whisperx` word timestamps against detected boxes. Boxes with no
    nearby speech are likely chyrons / captions, not dialogue. Adds the
    speaker-diarised "who is talking when" signal that bare `faster-whisper`
    cannot give us.
    Source: [m-bain/whisperX](https://github.com/m-bain/whisperx).

### Audio

46. **[x] Multi-track audio passthrough** -- `_merge_audio` now uses
    `-map 1:a?` (all input audio streams) re-encoded to AAC. New
    `ProcessingConfig.multi_audio_passthrough` defaults to True; the
    legacy single-track behaviour is opt-in via `--single-audio`.
    Caveat: single-pass loudnorm still applies to the first selected
    stream only -- broadcast-grade multi-track loudnorm needs
    `-filter_complex` and is queued as follow-up.
47. **[x] Loudness normalisation** -- `ProcessingConfig.loudnorm_target`
    LUFS field (default 0.0 = off, range clamped to [-70, -5]).
    `_merge_audio` injects `-af loudnorm=I=<target>:TP=-1.5:LRA=11` during
    mux when set. CLI exposes `--loudnorm <LUFS>`. Single-pass for speed;
    broadcast-grade two-pass measure-then-apply can layer on top later.
    GUI control still pending -- this is CLI-only for v3.13.

### Security

48. **[x] Dependency vulnerability scan in CI** -- `pip-audit` step added
    to [.github/workflows/build.yml](./.github/workflows/build.yml) after
    the install step, so it sees exactly what PyInstaller will bundle.
    Non-fatal during the v3.13 transition (`continue-on-error: true`);
    flip to fail-on-vuln once the pin set in `requirements.txt` is
    stable. Catches future PyTorch / OpenCV / Pillow issues before they
    ship in the EXE.
49. **Model-weight hash verification on first download** -- when an opt-in
    mode (real ProPainter, MI-GAN, LaMa-ONNX, etc.) fetches weights, verify
    against a vendored SHA-256. Prevents silent supply-chain swaps and
    half-downloaded files that crash hours into a long render.

### Distribution

50. **Code-signed release** -- adopt Azure Trusted Signing (~$10/mo, no
    hardware token). EV no longer grants instant SmartScreen reputation
    since 2024, but standard signing still lets reputation accumulate
    against a stable publisher identity, eliminating the unrecognised-app
    SmartScreen prompt after a few hundred installs.
    Source: [Microsoft SmartScreen reputation docs](https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/smartscreen-reputation),
    [PyInstaller WinCodeSigning recipe](https://github.com/pyinstaller/pyinstaller/wiki/Recipe-Win-Code-Signing).
51. **NSIS or MSI installer** -- ship a proper installer instead of the
    raw zip. Adds start-menu / desktop shortcuts, registers a file handler
    for the target extensions (sets up #80), supports upgrade in-place.
    NSIS is enough today; MSIX remains blocked on Tauri / mainstream
    tooling.

### Observability

52. **Opt-in crash reporting via GlitchTip** -- self-hosted Sentry-API-
    compatible service runs in 512 MB RAM. Reports only stack trace +
    Python version + GPU vendor; never frame contents, never paths.
    Strict opt-in stays consistent with the offline-first philosophy
    (#1). Defaults off.
    Source: [GlitchTip vs Sentry](https://www.bugsink.com/blog/glitchtip-vs-sentry-vs-bugsink/),
    [GlitchTip self-hosted](https://hamedsh.medium.com/glitchtip-a-lightweight-sentry-alternative-903bceb3e105).
53. **[x] Structured JSON log option** -- new
    `backend.processor.JsonLineLogHandler` writes one JSON record per
    line (`ts` UTC ISO-8601, `level`, `logger`, `msg`, optional `exc`).
    The text log keeps writing in parallel; this handler is purely
    additive. CLI exposes `--json-log PATH`. GUI toggle still pending.
    Tests in
    [tests/test_hardening.py](./tests/test_hardening.py) `JsonLineLogHandlerTests`.

### Testing

54. **Reference-clip regression harness** -- a `tests/clips/` directory of
    10-20 short clips covering edge cases (motion-heavy, static, karaoke,
    vertical JP, HDR, thin font, thick font, dissolve cuts). Nightly GHA
    run compares output hashes + PSNR/SSIM (#3.12) scores to committed
    baselines.
55. **Community edge-case corpus** -- GitHub Discussions thread where users
    submit 10-second problem clips + the settings they tried. Worst
    performers get folded into the regression harness; baseline fixes
    ship as a release note.
56. **[x] Coerce / config fuzz pass** -- new `ConfigFuzzTests` in
    [tests/test_hardening.py](./tests/test_hardening.py) runs 1500
    deterministic random payloads through the GUI `ProcessingConfig.from_dict`
    + `.normalized()` and another 1500 through the backend
    `normalize_processing_config()`. Seeded RNG (no Hypothesis dep) walks
    the cross-product of (known field name) x (pathological value pool:
    None, "", NaN, inf, very large int, lists, dicts, bool, hex string).
    Post-conditions: never raises; numeric fields land in declared
    bounds; `decode_hw_accel` lands in the allowed token set;
    `loudnorm_target` is 0.0 or in [-70, -5].

### UX

57. **[x] Quality self-test sheet** -- new
    `SubtitleRemover._write_quality_sheet()` extends
    `_compute_quality_report` to render a side-by-side PNG next to
    the output (`<output>.qualitysheet.png`). Each sampled frame
    becomes one row (`original | cleaned`) with a caption
    (`PSNR / SSIM`); a header strip carries mean metrics + a
    `Good` / `Review` tag from the SSIM 0.95 threshold. New
    `ProcessingConfig.quality_report_sheet` auto-enables
    `quality_report` in normalisation so the overlay can't reach an
    inconsistent state. CLI exposes `--quality-sheet`. Tests in
    [tests/test_hardening.py](./tests/test_hardening.py) `QualitySheetTests`.
58. **Drag-drop onto the app icon** -- register VSR as a file handler for
    the target video extensions so double-clicking an `.mp4` launches it
    straight into the queue. Depends on the installer in #51.

---

## Later (v4.x)

Heavier integrations; most require model-weight downloads, extra install
steps, or a behavioural change in the existing pipeline.

### Real reference inpainters

59. **Real ProPainter** (ICCV 2023) -- the actual `sczhou/ProPainter`
    reference with RAFT optical-flow propagation and the recurrent
    transformer. ~400 MB of weights auto-downloaded on first use. Wire as
    `InpaintMode.PROPAINTER_REAL`, keep the TBE-based "ProPainter" as the
    default fast path. ProPainter outperforms prior art by ~1.5 dB PSNR.
    Source: [sczhou/ProPainter](https://github.com/sczhou/ProPainter).
60. **DiffuEraser** (2025) -- diffusion-based, beats ProPainter on content
    completeness and temporal consistency. Combines BrushNet, ProPainter
    and AnimateDiff. Heavy (~5 GB weights, 8+ GB VRAM); ship as opt-in.
    Source: [lixiaowen-xw/DiffuEraser](https://github.com/lixiaowen-xw/DiffuEraser),
    [arXiv:2501.10018](https://arxiv.org/html/2501.10018).
61. **Wan2.1-VACE** (Alibaba, ICCV 2025) -- "all-in-one" video creation /
    editing. The 1.3 B variant runs on consumer GPUs and its MV2V
    (masked video-to-video) mode beats ProPainter on motion coherence.
    Opt-in mode with weights auto-fetched from ModelScope / HF.
    Source: [ali-vilab/VACE](https://github.com/ali-vilab/VACE),
    [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Jiang_VACE_All-in-One_Video_Creation_and_Editing_ICCV_2025_paper.pdf).
62. **VideoPainter** -- dual-branch with a lightweight context encoder
    that injects background guidance into any pre-trained video diffusion
    transformer; handles videos exceeding one minute while keeping ID
    consistency. Plug-and-play alternative to DiffuEraser.
    Source: [VideoPainter paper](https://arxiv.org/html/2503.05639v1).
63. **CoCoCo** (AAAI 2025) -- text-guided video inpainting with an
    instance-aware region selector and 3D-full-attention motion block.
    Less mature than VideoPainter at maintaining ID consistency but a
    useful research bench item.
    Source: [zibojia/COCOCO](https://github.com/zibojia/COCOCO).
64. **EraserDiT** -- diffusion-transformer video inpainting tuned for
    high-res subtitle removal specifically. Too new to commit to; track
    through 2026 and integrate once weights are stable.
    Source: [arXiv:2506.12853](https://arxiv.org/html/2506.12853v2).
65. **FloED** -- Optical-Flow guided Efficient Diffusion. Mid-weight
    alternative to DiffuEraser when VACE is overkill but TBE leaves
    visible residuals on fast pans.
    Source: [NevSNev/FloED](https://github.com/NevSNev/FloED-main).

### Mask / segmentation / matting

66. **SAM 2 mask refinement** -- user clicks inside the (already-detected)
    subtitle region, SAM 2 promotes the click into a clean object mask
    that follows the text exactly. Eliminates aggressive dilation to
    catch serifs. Memory propagation carries the seed through the clip.
    Source: [facebookresearch/sam2](https://github.com/facebookresearch/sam2).
67. **SAM 3 text-prompt segmentation** -- SAM 3 (released 2025-11-19)
    accepts natural-language prompts. One-click "segment all burned-in
    text" without any bounding boxes. Effectively auto-mask, and they
    cite a 2x gain over prior PCS work.
    Source: [Meta SAM 3 release post](https://ai.meta.com/blog/segment-anything-model-3/),
    [facebookresearch/sam3](https://github.com/facebookresearch/sam3).
68. **MatAnyone 2** (CVPR 2026 Highlight) -- video matting with a learned
    quality evaluator + region-adaptive memory fusion. Best-in-class for
    tracking a thin, mobile mask (rolling subtitle line) across a clip.
    Use the trimap-free inference as an alternative mask generator for
    stylised / decorated subtitles OCR + SAM can't clean up.
    Source: [pq-yang/MatAnyone2](https://github.com/pq-yang/MatAnyone2).
69. **CoTracker3 point tracking** (ICCV 2025) -- track arbitrary points
    through occlusion across a clip; lighter than SAM 2 memory and trained
    on 1000x less data. Useful as a propagation primitive when SAM 2 is
    overkill (e.g. confirming a single karaoke caret across a frame).
    Source: [facebookresearch/co-tracker](https://github.com/facebookresearch/co-tracker),
    [project page](https://cotracker3.github.io/).

### Acceleration

70. **TensorRT inpainter path** -- compile the LaMa ONNX model to a
    TensorRT engine on first GPU run. Expected ~2-3x further speedup on
    top of ONNX Runtime alone for NVIDIA hardware. Cache compiled engine
    under `%APPDATA%`.
    Source: [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt).
71. **PyNvVideoCodec hardware decode** -- replace `cv2.VideoCapture` with
    NVIDIA's PyNvVideoCodec for NVIDIA users. Measured ~6x faster decode
    (91 fps vs 15 fps on reference hardware), plus decoded frames live on
    the GPU already -- zero CPU-GPU copies when the subsequent
    detect/inpaint is also on GPU. Depends on #38 for the CPU-side
    fallback path.
    Source: [PyNvVideoCodec](https://developer.nvidia.com/pynvvideocodec).
72. **RIFE-interpolated fast mode** -- detect+inpaint every Nth frame,
    synthesise intermediates with Practical-RIFE 4.26 from the cleaned
    keyframes. Equivalent to 2-4x throughput when the background is smooth
    (dialogue scenes). Scene-cut handling falls back to the nearer-frame
    duplicate.
    Source: [hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE).

### Format support

73. **10-bit / HDR pipeline** -- current cv2 pipeline processes in 8-bit
    BGR, clamping HDR10 / HLG / Dolby Vision to SDR. Re-plumb as 16-bit
    numpy preserving `_Matrix` / `_Transfer` / `_Primaries` metadata;
    output via `ffmpeg -c:v libx265 -pix_fmt yuv420p10le -color_primaries
    bt2020`. Note: H.264 cannot encode HDR; HEVC or AV1 only. Dolby Vision
    metadata still needs `dovi_tool` round-trip.
    Source: [HDR encoding guide (Code Calamity)](https://codecalamity.com/encoding-uhd-4k-hdr10-videos-with-ffmpeg/).
74. **AV1 + VP9 ingest/egress** -- YouTube's native web format is AV1,
    older YouTube and most WebM is VP9. Verify decode across all codepaths
    and expose `output_format=av1` (`libsvtav1` for speed, `libaom-av1` for
    quality). SVT-AV1 also offers native film-grain synthesis (#78).
75. **VapourSynth bridge** -- optional bridge that accepts a `.vpy` script
    as input and emits one as output. Lets advanced users slot VSR into a
    larger chain (QTGMC deinterlace, Waifu2x upscale, SMDegrain denoise).
    Source: [vapoursynth/vapoursynth](https://github.com/vapoursynth/vapoursynth).
76. **NLE round-trip** -- when input is an EDL / XML from Premiere or
    DaVinci, emit a matching sidecar with the cleaned video substituted.
    Keeps VSR inside a post pipeline.

### Post-processing

77. **SeedVR2 one-step video restoration** (ICLR 2026) -- diffusion-
    transformer trained with adversarial post-training; 16B parameters
    but a single sampling step makes it >4x faster than multi-step
    diffusion VR. Best-in-class quality on heavily degraded footage.
    Opt-in mode; heavy enough that we ship it as a "restore" stage rather
    than the default.
    Source: [IceClear/SeedVR2](https://github.com/IceClear/SeedVR2),
    [arXiv:2506.05301](https://arxiv.org/html/2506.05301v2).
78. **Real-ESRGAN output upscale** -- optional "enhance" stage after
    inpainting. Users processing SD-era footage often want 2x / 4x upscale
    afterwards; bundling the step means one pipeline instead of hand-offs
    between tools. ONNX Runtime friendly.
    Source: [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).
79. **SwinIR restoration pass** -- alternative to Real-ESRGAN for footage
    where the subtitle removal leaves subtle blur (dense text). SwinIR
    restores local detail better than generic upscalers.
    Source: [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR).
80. **Film-grain re-synthesis** -- cinematic footage has uniform grain;
    inpainted regions come back grain-free and stand out on motion. Two
    paths: (a) a cheap additive-noise pass matched to the per-frame grain
    profile estimated from an unmasked crop; (b) AV1's native film-grain
    synthesis when the output is SVT-AV1 (#74) so the encoder ships a
    grain table rather than wasting bitrate on actual noise.
    Source: [FFmpeg AV1 grain synthesis](https://trac.ffmpeg.org/wiki/Encode/AV1).

---

## Under consideration (v5+)

Speculative. Anchored to projects that are under active research right now
and may not be practical to integrate for 12+ months. Items here are NOT
commitments; they are the research bench.

81. **Plugin architecture** -- `backend/plugins/*.py` auto-discovery for
    custom detectors and inpainters. Lets us ship the core app as a thin
    runtime and iterate on model integrations independently. The pluggable
    inpainter interface in upstream YaoFANGUK/video-subtitle-remover is the
    closest existing reference.
82. **Distributed / multi-GPU** -- split the frame batch across GPUs with
    a coordinator that handles the temporal window crossing device
    boundaries. Pays off for 4K / feature-length.
83. **Headless REST server mode** -- `python VideoSubtitleRemover.py
    --serve 8888` exposes a local HTTP API (submit job, poll status,
    stream preview). Stays consistent with #1 because the server is
    loopback-only and the user can pull it through SSH; nothing leaves
    the machine.
84. **Tauri / pywebview shell for cross-platform GUI** -- port the GUI to
    a webview-backed shell using the existing Python core as a backend.
    Tauri 2.x ships a ~3 MB Windows binary using the OS WebView2 vs
    Electron's 150 MB+. Keeps the native Windows feel while making macOS
    / Linux parity realistic. CustomTkinter is an alternative if we want
    to stay in pure-Python land.
    Source: [Tauri 2.x](https://v2.tauri.app/),
    [TomSchimansky/CustomTkinter](https://github.com/TomSchimansky/CustomTkinter).
85. **Subtitle translation pipeline** -- after SRT export, route through a
    local LLM (gemma3 / qwen3 via llama.cpp) for translation, then
    optionally re-burn translated subtitles into the cleaned video. End-
    to-end "re-dub in another language" workflow. KrillinAI is the closest
    existing reference for a one-binary localisation pipeline (kept as
    inspiration; we will not bundle it).
86. **Streaming mode** -- process a file while still being written (OBS
    recording, DVR capture). Tail the input and emit a cleaned output
    with a fixed N-second lag. Requires re-architecting the temporal
    window from a fixed batch to a streaming buffer; unlocks live-broadcast.
87. **WebGPU / WASM fallback** -- compile the ONNX inpaint + detect path
    to WebGPU via `onnxruntime-web`. Ships as a drag-and-drop web page
    for locked-down environments where Python install is impossible
    (corporate VDIs, Chromebooks). Not a replacement for the desktop app
    -- an escape hatch.
88. **Font-aware inpainting** -- classify the subtitle font style (Hanzi
    vs Latin vs Cyrillic vs decorative) via a small classifier on the
    detected boxes; apply per-class mask dilation and feather presets.
    Hanzi needs tighter masks than handwritten Latin karaoke captions.
89. **Logo / watermark mode** -- specialised mode that persists the mask
    across the entire video (static watermark case) with TBE using the
    full video as the temporal window rather than a 30-frame batch.
    Subsumes the current "fixed subtitle area" workflow and extends it
    to full-video aggregation.
90. **AI chat interface** -- optional side-panel wired to a local LLM
    (gemma3 / qwen3 via llama.cpp) that accepts natural language like
    "remove the bottom banner between 0:15 and 0:45, keep the song title
    top-right." LLM translates to `ProcessingConfig` mutations.
91. **Live performance dashboard** -- in-app panel showing current
    detection FPS, inpaint FPS, GPU VRAM, disk I/O, queued / completed /
    failed counts. Pure local; no telemetry.

### Platforms

92. **macOS / Linux parity** -- the PyInstaller build is Windows-only;
    port the build workflow to a three-OS GHA matrix. tkinter ships on
    both, FFmpeg and OpenCV work identically; the blockers are Windows-
    specific pieces (taskbar progress via ITaskbarList3, winsound beeps,
    `pythonw.exe` console hiding).
93. **Android port** -- Kotlin + Compose shell around the Python core via
    Chaquopy, with `ffmpeg-kit` for video I/O and NNAPI / Qualcomm AIP /
    QNN hooks for the inpainter. v1 lite supports image-only; v2 adds
    video.
    Source: [arthenica/ffmpeg-kit](https://github.com/arthenica/ffmpeg-kit).
94. **iOS / iPadOS port** -- long shot. Likely via BeeWare / Pyto-style
    bridge rather than Chaquopy. Realistic only once Apple silicon +
    NeuralEngine inpainting is plumbed.

### Accessibility + i18n

95. **Screen-reader support on Windows (UIA)** -- wire tkinter widgets to
    Windows UI Automation providers so NVDA / Narrator can announce
    status changes (items queued, progress, completion). Today none of
    the custom widgets are announceable. Caveat: existing Python a11y
    libraries like Tka11y are Linux-only (AT-SPI), so this requires a
    custom UIA shim via `pywin32` / `comtypes` -- non-trivial.
    Source: [Tka11y on PyPI](https://pypi.org/project/Tka11y/) (notes the
    Windows MSAA gap).
96. **High-contrast theme variant** -- optional theme preset respecting
    Windows' `HighContrast` active scheme. The existing design-token
    system makes this cheap: swap the palette constants, keep the widget
    code.
97. **GUI localisation** -- `gettext`-based string externalisation with
    `.mo` files for the top ~10 languages we already detect in subtitles
    (the GUI itself is English-only today). Community-contributable via
    Weblate or a simple PR flow.
98. **Right-to-left UI support** -- Arabic / Hebrew mirroring for text,
    button ordering, queue scroll direction. Depends on #97.

### Workflows

99. **Restyle mode** -- after subtitle removal + SRT export, optionally
    re-burn a translated / restyled subtitle track at the same position
    using ffmpeg's `subtitles` filter with an `.ass` template. Users get
    "clean + re-dub-in-text" in one round-trip.
100. **Watermark addition pipeline** -- content-creator mode: after
     cleaning an input, burn a user-specified PNG watermark at a
     configurable corner with fade-in/out.

---

## Explicitly not on the roadmap

These have been considered and consciously rejected. Listed so the rejection
is durable and we do not silently resurrect them in a later cycle.

- **Cloud API integrations** (OpenAI Whisper API, Anthropic, Google Cloud
  Vision, etc.). VSR is offline-first by design (#1). Local-only Whisper /
  Qwen / gemma3 via llama.cpp are fine; remote APIs add friction and
  privacy cost. Hosted services like KrillinAI / EchoSubs AI are
  inspirations, not integration targets.
- **Docker / containerised builds.** The tool is Windows-first and ships
  as a PyInstaller binary. Containers add install complexity for the
  95% Windows user base. Upstream YaoFANGUK ships hardware-keyed Docker
  images; that is a reasonable choice for them but not for us.
- **Telemetry by default.** Even GlitchTip (#52) is strict opt-in and
  scrubs paths / frame contents. No analytics, no phone-home, no usage
  pings.
- **Subscription model.** Free single-binary, MIT. Commercial competitors
  (HitPaw, Vmake, EchoSubs) paywall what is increasingly free in OSS.
  Source: [EchoSubs 2026 comparison](https://echosubs.com/best-subtitle-remover-tools-2026)
  (cited as evidence the paywalled features map onto items we already
  ship, not as a target to imitate).
- **Multi-user / collaboration.** Single-user desktop tool; no shared
  state, no concurrent editing.
- **GPL-licensed defaults.** Surya (GPL) and manga-image-translator
  (GPL-3.0) are gated behind opt-in installs and never bundled. We keep
  the MIT licence clean.
- **Mocking the AI weights for "offline demos".** The TBE path already
  works with zero weights; that is the demo.

---

## Themes

Each Now/Next item maps to exactly one of these themes. If a future
proposal does not fit a theme, the theme list is wrong, not the proposal
-- update this section first.

| Theme | What it covers | Now/Next items |
|---|---|---|
| **Hardening** | Bug fixes, race conditions, input validation | 1-8, 14-19, 20, 48, 49, 56 |
| **UI/UX polish** | Microcopy, widget rendering, design-token consistency | 9-13, 28-31, 57 |
| **Detection** | OCR engines, scene-cut, layout-aware models | 21-24, 32, 33, 42-45 |
| **Inpainting** | TBE, LaMa, MI-GAN, AUTO routing | 25-27 |
| **CLI / batch** | Headless workflows, automation | 36, 37 |
| **Acceleration** | Decode, ONNX, batching, concurrency | 38-41 |
| **Audio** | Multi-track, loudness | 46, 47 |
| **Security** | CVE pinning, supply chain | 17-19, 48, 49 |
| **Distribution** | Code signing, installer | 50, 51, 58 |
| **Observability** | Crash reports, structured logs | 52, 53 |
| **Testing** | Regression, fuzz, edge-case corpus | 54-56 |
| **Migration** | Settings schema versioning | 20 |

Categories from the Phase-5 checklist that *intentionally* do not have
Now/Next items: **plugin ecosystem**, **mobile**, **multi-user / collab**,
**streaming**, **accessibility / i18n**. Each lives in
[Under consideration](#under-consideration-v5) with the reason: they need
either a non-trivial framework decision (Tauri vs CustomTkinter), a new
platform (Android), or a project-philosophy change (#1 / single-user).
None are forgotten; all are deferred deliberately.

---

## Appendix: sources

Every external claim above maps to a URL here. Grouped by category.

### Subtitle removal / OSS competitors

- [YaoFANGUK/video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover)
  -- upstream of this fork; pluggable inpainter interface, hardware-keyed
  Docker images, DirectML support
- [YaoFANGUK/video-subtitle-extractor](https://github.com/YaoFANGUK/video-subtitle-extractor)
  -- paired OCR-to-SRT extractor; inspiration for our SRT sidecar (#shipped v3.9)
- [timminator/VideOCR](https://github.com/timminator/VideOCR) -- GUI
  hardcoded-subtitle *extractor* (200+ langs, Google Lens hybrid mode)
- [JollyToday/GhostCut_Remove_Video_Text](https://github.com/JollyToday/GhostCut_Remove_Video_Text)
  -- multilingual OCR + inpainting (EN/CN/JA/KO/AR)
- [rainwl/VideoRemoveText](https://github.com/rainwl/VideoRemoveText) --
  lightweight LaMa + fixed-ROI + colour threshold; auto-device CUDA → MPS → CPU
- [Rats20/EraseSubtitles](https://github.com/Rats20/EraseSubtitles) --
  multilingual text detection + inpainting research
- [Keaneo/Scrubtitles](https://github.com/Keaneo/Scrubtitles) -- minimal
  script reference
- [zyddnys/manga-image-translator](https://github.com/zyddnys/manga-image-translator)
  -- text-detection + inpaint pipeline with text-erasure-only mode (GPL)
- [ogkalu2/comic-translate](https://github.com/ogkalu2/comic-translate) --
  AI comic / manga translator (multi-format)
- [PaddlePaddle/PaddleOCR issue #11639](https://github.com/PaddlePaddle/PaddleOCR/issues/11639)
  -- documents the long-running paddlepaddle memory-leak that justified our
  switch to RapidOCR (#shipped v3.8)

### Commercial competitors (referenced for feature spotting, not integration)

- [HitPaw Video Object Remover](https://www.hitpaw.com/hitpaw-object-remover.html)
- [Vmake AI Caption Remover](https://vmake.ai/caption-remover)
- [Media.io AnIeraser subtitle remover](https://anieraser.media.io/remove-subtitles-from-video.html)
- [EchoSubs Best Subtitle Remover Tools 2026](https://echosubs.com/best-subtitle-remover-tools-2026)
- [Filmora subtitle removal guide 2026](https://filmora.wondershare.com/video-editing-tips/remove-subtitles-from-video.html)

### Video inpainting research

- [advimman/LaMa](https://github.com/advimman/lama) (WACV 2022) -- LaMa
  reference paper
- [Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX) -- LaMa ONNX
  export (#25)
- [sczhou/ProPainter](https://github.com/sczhou/ProPainter) (ICCV 2023) --
  reference inpainter (#59)
- [Picsart-AI-Research/MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN)
  (ICCV 2023) -- mobile-fast inpainting (#26)
- [IOPaint MI-GAN model card](https://www.iopaint.com/models/erase/migan)
- [lixiaowen-xw/DiffuEraser](https://github.com/lixiaowen-xw/DiffuEraser) --
  2025 diffusion inpainting (#60)
- [arXiv:2501.10018 -- DiffuEraser technical report](https://arxiv.org/html/2501.10018)
- [ali-vilab/VACE](https://github.com/ali-vilab/VACE) (ICCV 2025) -- Wan2.1
  all-in-one video edit (#61)
- [ICCV 2025 VACE paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Jiang_VACE_All-in-One_Video_Creation_and_Editing_ICCV_2025_paper.pdf)
- [arXiv:2503.05639 -- VideoPainter](https://arxiv.org/html/2503.05639v1) (#62)
- [zibojia/COCOCO](https://github.com/zibojia/COCOCO) -- AAAI 2025 text-guided
  video inpainting (#63)
- [arXiv:2506.12853 -- EraserDiT](https://arxiv.org/html/2506.12853v2) (#64)
- [NevSNev/FloED](https://github.com/NevSNev/FloED-main) -- optical-flow
  diffusion inpainting (#65)
- [arXiv:2511.03272 -- long video inpainting (OHCD)](https://arxiv.org/html/2511.03272v1)
  -- adjacent research on overlapping high-order co-denoising
- [TencentARC/BrushNet](https://github.com/TencentARC/BrushNet) (ECCV 2024)
  -- plug-and-play dual-branch image inpainting
- [hitachinsk/FGT](https://github.com/hitachinsk/FGT) (ECCV 2022) -- flow-
  guided transformer reference

### Text detection / OCR

- [RapidAI/RapidOCR](https://github.com/RapidAI/RapidOCR) -- PP-OCR via
  ONNX Runtime; the engine we default to
- [PaddleOCR-VL 0.9B on Hugging Face](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)
  (#23)
- [arXiv:2510.14528 -- PaddleOCR-VL paper](https://arxiv.org/abs/2510.14528)
- [Florence-2 base](https://huggingface.co/microsoft/Florence-2-base) (#22)
- [Qwen2.5-VL blog](https://qwenlm.github.io/blog/qwen2.5-vl/) (#22)
- [opendatalab/OmniDocBench](https://github.com/opendatalab/OmniDocBench)
  -- evaluation suite that ranks the VLM-OCR tier
- [kha-white/manga-ocr](https://github.com/kha-white/manga-ocr) (#42)
- [dmMaze/comic-text-detector](https://github.com/dmMaze/comic-text-detector)
  (#42)

### Segmentation / matting / tracking

- [facebookresearch/sam2](https://github.com/facebookresearch/sam2) (#66)
- [facebookresearch/sam3](https://github.com/facebookresearch/sam3) (#67)
- [Meta SAM 3 release post (2025-11-19)](https://ai.meta.com/blog/segment-anything-model-3/)
- [pq-yang/MatAnyone](https://github.com/pq-yang/MatAnyone) (CVPR 2025)
- [pq-yang/MatAnyone2](https://github.com/pq-yang/MatAnyone2) (CVPR 2026
  Highlight) (#68)
- [facebookresearch/co-tracker](https://github.com/facebookresearch/co-tracker)
  -- CoTracker3, ICCV 2025 (#69)
- [CoTracker3 project page](https://cotracker3.github.io/)
- [PeterL1n/RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting)
  -- real-time RVM, kept as adjacent reference

### Audio / transcription

- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) (#27)
- [m-bain/whisperX](https://github.com/m-bain/whisperx) -- VAD + word-level
  alignment + diarisation on top of faster-whisper (#45)

### Acceleration

- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) (#70)
- [PyNvVideoCodec](https://developer.nvidia.com/pynvvideocodec) (#71)
- [ONNX Runtime DirectML EP](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)
- [microsoft/DirectML](https://github.com/microsoft/DirectML)
- [OpenCV hardware-accelerated video IO](https://docs.opencv.org/4.x/db/dc4/group__videoio__hwaccel.html)
  (#38)

### Preprocessing / scene detection

- [m-tassano/fastdvdnet](https://github.com/m-tassano/fastdvdnet) (#33)
- [Breakthrough/PySceneDetect](https://github.com/Breakthrough/PySceneDetect)
  (#32)
- [PySceneDetect docs (0.7)](https://www.scenedetect.com/docs/latest/)
- [soCzech/TransNetV2](https://github.com/soCzech/TransNetV2) (#21)
- [arXiv:2304.06116 -- AutoShot](https://arxiv.org/abs/2304.06116) (#21)
- [vapoursynth/vapoursynth](https://github.com/vapoursynth/vapoursynth) (#75)

### Frame interpolation / super-resolution / restoration

- [hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE) (#72)
- [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (#78)
- [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR) (#79)
- [IceClear/SeedVR2](https://github.com/IceClear/SeedVR2) (ICLR 2026) (#77)
- [arXiv:2506.05301 -- SeedVR2 paper](https://arxiv.org/html/2506.05301v2)
- [ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR) -- joint
  repo for SeedVR (CVPR 2025 Highlight) + SeedVR2

### Quality metrics / testing

- [chaofengc/IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) -- PSNR
  / SSIM / LPIPS toolbox (we use a pure-numpy SSIM today; this is the
  upgrade if we ever add LPIPS)
- [JunyaoHu/common_metrics_on_video_quality](https://github.com/JunyaoHu/common_metrics_on_video_quality)
  -- FVD + friends
- [richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity)
  -- LPIPS reference

### Mobile / platforms / GUI shells

- [arthenica/ffmpeg-kit](https://github.com/arthenica/ffmpeg-kit) (#93)
- [Chaquopy](https://chaquo.com/chaquopy/) (#93)
- [Tauri 2.x](https://v2.tauri.app/) (#84)
- [TomSchimansky/CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)
  (#84)
- [Tka11y on PyPI (Linux-only a11y caveat)](https://pypi.org/project/Tka11y/)
  (#95)

### Security

- [GHSA-63cw-57p8-fm3p / CVE-2026-24747 -- PyTorch weights_only RCE](https://github.com/advisories/GHSA-63cw-57p8-fm3p)
  (#17)
- [GHSA-53q9-r3pm-6pq6 / CVE-2025-32434 -- PyTorch torch.load RCE](https://github.com/advisories/GHSA-53q9-r3pm-6pq6)
  (#17)
- [CVE-2026-25990 -- Pillow PSD OOB write](https://www.appsecure.security/vulnerability-database/cve-2026-25990/)
  (#18)
- [NVD CVE-2025-53644 -- OpenCV JPEG read uninitialised pointer](https://nvd.nist.gov/vuln/detail/cve-2025-53644)
  (#19)
- [PyInstaller WinCodeSigning recipe](https://github.com/pyinstaller/pyinstaller/wiki/Recipe-Win-Code-Signing)
  (#50)
- [Microsoft SmartScreen reputation docs](https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/smartscreen-reputation)
  (#50)

### Observability

- [GlitchTip vs Sentry vs Bugsink](https://www.bugsink.com/blog/glitchtip-vs-sentry-vs-bugsink/)
  (#52)
- [GlitchTip lightweight Sentry alternative](https://hamedsh.medium.com/glitchtip-a-lightweight-sentry-alternative-903bceb3e105)
  (#52)

### Format / codec references

- [HDR10 / HEVC encoding with ffmpeg (Code Calamity)](https://codecalamity.com/encoding-uhd-4k-hdr10-videos-with-ffmpeg/)
  (#73)
- [FFmpeg AV1 encode + film-grain synthesis](https://trac.ffmpeg.org/wiki/Encode/AV1)
  (#74, #80)
- [Non-local means denoise FFMPEG filter](https://www.dirk-farin.net/projects/nlmeans/index.html)

### Awesome-lists harvested

- [zengyh1900/Awesome-Image-Inpainting](https://github.com/zengyh1900/Awesome-Image-Inpainting)
- [AlonzoLeeeooo/awesome-image-inpainting-studies](https://github.com/AlonzoLeeeooo/awesome-image-inpainting-studies)
- [liuzhen03/awesome-video-enhancement](https://github.com/liuzhen03/awesome-video-enhancement)
- [wanghaisheng/awesome-ocr](https://github.com/wanghaisheng/awesome-ocr)
- [ZumingHuang/awesome-ocr-resources](https://github.com/ZumingHuang/awesome-ocr-resources)
