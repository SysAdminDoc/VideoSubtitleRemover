

# Video Subtitle Remover Pro

<div align="center">

![Version](https://img.shields.io/badge/version-3.19.1-22c55e)
![Platform](https://img.shields.io/badge/platform-Windows-60a5fa)
![License](https://img.shields.io/badge/license-MIT-4ade80)
![Python](https://img.shields.io/badge/python-3.11--3.13%20CUDA-blue)

**Professional AI-powered tool for removing hard-coded subtitles from videos and images**

[Features](#features) | [Installation](#installation) | [Usage](#usage) | [Configuration](#configuration) | [CLI](#cli-usage) | [Troubleshooting](#troubleshooting)

</div>

---

## Overview

Video Subtitle Remover Pro uses real AI neural networks to remove hard-coded subtitles and text watermarks from videos and images. Unlike simple blur or crop methods, it intelligently fills in removed areas with content that matches the surrounding video.

Based on [YaoFANGUK/video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover), enhanced with a professional interface, real LaMa inpainting, multi-engine detection, and a 52-code language picker backed by broader OCR engine coverage.

## Features

- **Real Video Inpainting** -- Temporal Background Exposure (TBE) reconstructs the true background from neighbouring frames where the subtitle is absent. No external model weight downloads required.
- **Real AI Inpainting** -- LaMa neural network via ONNX Runtime (default, no torch dependency), OpenCV DNN weights, or an explicit PyTorch fallback opt-in
- **AUTO Inpaint Routing** -- Scene-cut-aware routing between STTN and ProPainter mode using temporal exposure and measured motion
- **Multi-Engine Detection** -- RapidOCR PP-OCRv6 through OpenCV 5 DNN, ONNX Runtime, or OpenVINO > PaddleOCR > Surya (GPL opt-in) > EasyOCR > threshold fallback (automatic)
- **Lossless Pipeline** -- FFV1 lossless intermediate (only the final encode is lossy) for noticeably cleaner outputs than the legacy mp4v intermediate
- **Modern Codec Output** -- Pick H.264 / H.265 / AV1 / VVC (H.266) from a dropdown; NVENC/QSV/AMF where available, libx265 / libsvtav1 software fallback, native SVT-AV1 film grain, and VVC when FFmpeg exposes `libvvenc`
- **Opt-in FFmpeg D3D12 Path** -- FFmpeg 8.1+ can upload and scale frames with D3D12 and encode H.264/H.265 only after a byte-valid driver smoke; advertised-but-broken codecs and runtime failures fall back through NVENC/QSV/AMF and software
- **Precise Multi-region Masks** -- Draw or select multiple rectangle/polygon regions, enter exact source-pixel coordinates and start/end seconds or frames, nudge with arrows, resize with Ctrl+arrows, and undo or redo edits
- **Moving Region Keyframes** -- Scrub to two or more frames, draw rectangle or polygon anchors, and interpolate the mask deterministically through the selected motion span
- **Confidence-Gated Clean Plates** -- Attach a same-size clean reference image to each timed rectangle, preview translation or homography alignment and per-frame color matching, and fall back to normal inpainting whenever alignment is uncertain
- **Quality-Directed Mask Correction** -- Review residual, flicker, and low-confidence frame spans; paint ordered add/subtract corrections with undo/redo; then rerun only the affected frames while reusing the prior cleaned output elsewhere
- **Lossless Matte Interchange** -- Export exact gray8 FFV1 or PNG-sequence masks with CFR/VFR timestamps, edit them externally, preview replace/add/subtract composition, and import them through strict manifest preflight
- **Erase, Translate, and Re-embed** -- Opt into one cleanup pass that accepts a translated SRT or sends OCR/Whisper/source-SRT cues to a pluggable local command, then burns the validated result with configurable ASS styling and hash-backed provenance
- **Inpaint Preview** -- "Test cleanup" runs detect + inpaint on the selected frame so you can A/B settings before committing
- **Cached Mask Tuning** -- Adjust mask dilation in the preview pane and see the composed result immediately without rerunning OCR
- **Seamless Boundaries** -- Gaussian alpha feathering at every inpaint boundary, no visible cut lines
- **Language Support** -- 52 selectable OCR language codes in the GUI, with installed OCR engines reporting broader capacity: RapidOCR 100+, PaddleOCR 106, Surya 90+ (GPL opt-in), and EasyOCR 80+; gettext catalogs in `locale/<BCP-47 tag>/LC_MESSAGES/vsr.mo` are packaged, preserve script/territory fallback, and follow the Windows interface locale
- **GPU Acceleration** -- NVIDIA CUDA, AMD/Intel DirectML through ONNX Runtime, hardware-decode hints (D3D11 / VAAPI / MFX), CPU fallback
- **Subtitle Region Selector** -- Scrub to any frame and draw one or more rectangles; use optional start/end seconds to save time-ranged manual masks
- **Live Region OCR Feedback** -- While drawing a rectangle, inspect detected text boxes and confidence before saving the region
- **Selected-Language Masks** -- Optionally remove only OCR boxes whose recognized script matches the chosen subtitle language, keeping unrelated on-screen text
- **Batch Processing** -- Queue files or drag entire folders; per-item cancellation plus safe pause/resume for long videos
- **Multi-track Audio + Loudness Normalisation** -- Pass through every audio track on Bluray rips; optional per-stream EBU R128 normalisation to LUFS targets (YouTube -14, Apple -16, broadcast -23)
- **Quality Self-Test** -- PSNR / SSIM report, optional FFmpeg/libvmaf VMAF score, ROI-cropped metrics for the inpaint region, and an optional side-by-side comparison PNG
- **Detection Efficiency Reports** -- Batch summaries show frames OCR'd versus skipped, skip reasons, unique regions, stage timings, and an optimization hint when OCR dominates
- **HDR Color Validation** -- Post-encode ffprobe checks record whether BT.2020/PQ/HLG and related color metadata were preserved in batch reports and output sidecars
- **CLI + Presets** -- `python -m backend.processor --pattern ... --preset "YouTube (default)"`; nine built-in presets + user presets persisted to `%APPDATA%`
- **Chyron vs Subtitle Filter** -- Keep persistent text (logos, lower-thirds) and remove dialogue, or vice versa
- **Karaoke Grouping** -- Per-syllable boxes fuse into a single line mask so highlighted lyrics do not leak through the gaps
- **Live Preview During Processing** -- 15 FPS throttled preview piped from the backend worker
- **Pre-batch ETA Estimate** -- 30-frame detect probe seeds the ETA so users see "about X left" from the very first frame
- **Pause/Resume Checkpointing** -- SHA-256 input fingerprint per file; finished files are skipped and paused videos resume from durable checkpoint frames
- **Backend Status** -- Help shows OCR/inpaint backends, language picker vs. engine capacity, ONNX/OpenCV providers, required model files, hash state, FFmpeg capability profiles, and the next setup action
- **Premium Dark UI** -- Cohesive design system with custom controls, rectangular status tiles, responsive workbench scrolling, taskbar progress, and onboarding
- **Settings Persistence** -- All knobs saved/restored between sessions; versioned schema with backfill migration
- **Release Tooling** -- Local PyInstaller/NSIS build scripts, dependency checks, support bundles, and winget-ready installer metadata

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 | Windows 11 |
| CPU | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| RAM | 8 GB | 16+ GB |
| GPU | Any (CPU mode) | NVIDIA RTX 2060+ (RTX 50-series supported via CUDA 12.8) |
| VRAM | - | 6+ GB |
| Python | 3.11 | 3.12 or 3.13 for CUDA |

## Installation

### Prebuilt Download (no setup)

Grab the latest standalone Windows x64 build from the
[Releases page](https://github.com/SysAdminDoc/VideoSubtitleRemover/releases/latest):
download `VideoSubtitleRemoverPro-vX.Y.Z-win-x64.zip`, extract anywhere, and run
`VideoSubtitleRemoverPro.exe` (or `Run_VSR_Pro.bat`). The build is unsigned, so
Windows SmartScreen may prompt -- choose **More info -> Run anyway**, and verify
the download against the published `.sha256` sidecar.

### Quick Install

1. **Download** or clone this repository
2. **Double-click** `Run_VSR_Pro.bat` — first run automatically:
   - Creates a virtual environment
   - Detects your GPU and installs appropriate packages
   - Shows a compact six-stage setup splash while the runtime is prepared
   - Installs the reviewed RapidOCR/ONNX runtime for the detected hardware
   - Launches the application
   - On later launches, verifies core packages and repairs a broken `venv`
     without stdin prompts
   - Use `Run_VSR_Pro_Debug.bat` for a visible troubleshooting console, or
     `Run_VSR_Pro.ps1` when you prefer launching from PowerShell

After the Windows Package Manager manifest is accepted, signed release
installers can also be installed with:

```powershell
winget install SysAdminDoc.VideoSubtitleRemoverPro
```

### Manual Install

```powershell
cd VideoSubtitleRemover

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Choose a reviewed profile: cpu, nvidia, or directml.
$profile = "cpu"

# Install PyTorch (Python 3.12/3.13 recommended for CUDA):
# NVIDIA RTX 20/30/40/50-series:
pip install torch>=2.10.0 torchvision>=0.25.0 --constraint "dependency_profiles/$profile.txt" --index-url https://download.pytorch.org/whl/cu128
# CPU:
pip install torch>=2.10.0 torchvision>=0.25.0 --constraint "dependency_profiles/$profile.txt" --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
pip install -r requirements.txt --constraint "dependency_profiles/$profile.txt"

# Run
python VideoSubtitleRemover.py
```

`python setup.py --profile auto` selects the reviewed CPU, NVIDIA, or DirectML
profile from detected hardware; pass a profile name explicitly for repeatable
CI or repair installs. Maintainers update `dependency_profiles.json`, run
`python -m backend.dependency_profiles update`, review the emitted diffs, and
then run `python -m backend.dependency_profiles check`. Generated constraint
and manifest SHA-256 values are included in release evidence. PaddleOCR,
EasyOCR, and legacy `simple-lama-inpainting` remain isolated opt-ins because
their OpenCV wheel ownership or NumPy caps conflict with the primary runtime.
Python 3.11 is the minimum supported interpreter because the security-reviewed
ONNX Runtime CPU/CUDA floor and pinned DirectML release do not provide Python
3.10 wheels.

### FFmpeg (Required for audio)

```powershell
winget install ffmpeg
```

Use **FFmpeg 8.1.2+ on the 8.1 branch, or 8.0.3+ on the 8.0 branch.** VSR
decodes untrusted media through FFmpeg, and builds `8.1.0-8.1.1` and
`8.0.0-8.0.2` predate the security backports for CVE-2026-8461 (MagicYUV heap
out-of-bounds write, RCE) and CVE-2026-30999. Older branches are outside VSR's
reviewed support policy; development snapshots and future branches remain
unknown until explicitly classified. The self-test, support bundle, and strict
release validation block vulnerable, unsupported, and unknown runtimes.

**Build toolchain floors:** the local build requires **PyInstaller >= 6.10.0**
(CVE-2025-59042 writable-CWD LPE) and the installer requires **NSIS >= 3.12**
(elevated Low IL temp-directory privilege-escalation hardening);
`installer/vsr.nsi` fails to compile on an older NSIS, and strict release
validation flags both.

Run `python -m backend.processor --self-test` to confirm the installed build's
`basic`, `advanced_quality`, `speech_fallback`, and `modern_codec` profiles.
Those profiles report missing filters such as `loudnorm`, `libvmaf`, or
`whisper`, missing encoders such as `libvvenc`, and OpenCV wheel ownership
before a long batch starts.

Run `python -m backend.cli --ocr-benchmark` to score the active OCR detector
(RapidOCR ships PP-OCRv6) on synthetic ground-truth subtitle fixtures --
detection recall plus per-frame latency -- and print JSON evidence. Any change
to the default detector should be gated on the `meets_floors` verdict (recall
>= 0.8); latency is reported as device-dependent evidence, not a hard gate.

Run `python -m backend.cli --inference-smoke` to prove the OCR and inpaint
backends actually execute: it pushes a generated text image and masked frame
through the detector and inpainter, printing the real engine / execution
provider (e.g. `RapidOCR`, `ONNX (CUDAExecutionProvider)`, or a `cv2`
fallback) and timing, and exits non-zero if a backend that loaded cannot run
inference. No model weights are downloaded; add `--gpu N` to test a CUDA
device.

### Validation

```powershell
python -m pip install ruff==0.15.20
python -m ruff check backend gui scripts VideoSubtitleRemover.py --no-cache
python scripts/generate_cli_reference.py
python scripts/i18n_catalogs.py check
python -m unittest discover -s tests -v
python -m backend.reference_corpus --json
python tools/local_smoke.py
```

`build_exe.bat` is the fail-closed local release command. It runs the Ruff
source-hygiene gate and complete unit suite, builds the PyInstaller folder,
compiles the production NSIS
installer plus a non-elevated extraction harness, smoke-tests every frozen
entry point and the extracted installer payload, runs the reference corpus,
audits the exact frozen Python components with `pip-audit`, and applies strict
runtime/advisory gates. It exits nonzero at the first failed stage.

The default frozen profile packages RapidOCR/ONNX and excludes the
multi-gigabyte PaddleOCR, EasyOCR, and PyTorch fallbacks. Set
`VSR_ENABLE_FULL_OCR=1` and/or `VSR_ENABLE_PYTORCH_LAMA=1` before the build to
include those optional runtimes intentionally. `sbom.cdx.json` is derived from
PyInstaller's `Analysis-00.toc`: required Python libraries and hashed native
files reflect the folder that actually ships, while PyInstaller and other
build tools are marked with excluded scope. `release-verification.json` and
`pip-audit.json` record the remaining release proof.

For an isolated CPU smoke without touching the Windows launcher, run the same
check in the local container recipe:

```powershell
docker build -t vsr-pro-smoke .
docker run --rm vsr-pro-smoke
```

The container path installs only the minimal CPU smoke dependencies (including
the canonical `onnxruntime>=1.25.0` security floor), records the resolved ONNX
Runtime version/providers, runs `python -m backend.processor --self-test`, then
processes a generated tiny image through the CLI with a fixed mask.

## Usage

1. **Launch** via `Run_VSR_Pro.bat`, `Run_VSR_Pro_Debug.bat`, or
   `Run_VSR_Pro.ps1`
2. **Import** -- Use the compact drop target to browse for files or folders,
   or drag media directly into the window
3. **Configure** -- Choose Auto, STTN, LAMA, or ProPainter in the right-side
   inspector; set a subtitle region and confirm the output location
4. **Open Advanced** when you need preset management, compute-device and
   language selection, workflow toggles, or expert tuning
5. **Inspect** -- Select a queue item to preview it, use **Review mask** to
   confirm detection, or draw a fixed subtitle band directly on the preview.
   The full selector supports exact rectangle or polygon coordinates,
   second/frame timing, arrow-key nudging, Ctrl+arrow resizing, and
   Ctrl+Z/Ctrl+Y history
6. **Process** -- Use **Start batch** at the lower right of the persistent
   queue and monitor per-file status and progress in place. Right-click a queue
   card, or press **Menu / Shift+F10** while it is focused, for per-item actions

### Algorithm Comparison

| Algorithm | Inpainting Engine | Speed | Quality | Best For |
|-----------|-------------------|-------|---------|----------|
| **STTN** | Temporal Background Exposure | Fastest | Great | Live-action video with changing subtitles (default) |
| LAMA | Neural (LaMa ONNX/OpenCV DNN; PyTorch opt-in) | Medium | Best still-frame | Images, animations, static backgrounds |
| ProPainter | TBE + LaMa refinement | Slowest | Best motion | Motion-heavy footage, thick/decorative text |
| Auto | Per-scene STTN / ProPainter routing | Adaptive | Adaptive | Mixed edits with both static dialogue and fast motion |

> All three modes now do real inpainting. STTN recovers the literal background from adjacent frames where the subtitle is absent -- this works because hard-coded subtitles are sparse in time, and the pixels behind them are revealed whenever the text changes or disappears. LAMA is a single-frame neural fill. ProPainter is a TBE + LaMa refinement hybrid -- it is **not** the ICCV 2023 ProPainter model or weights (which carry a non-commercial NTU S-Lab license). This implementation uses only MIT-licensed code.

### Detection Engines

The app automatically selects the best available engine. Advanced > Detection
can pin RapidOCR, OpenCV 5 DNN, PaddleOCR, EasyOCR, or the dependency-free
OpenCV fallback for comparison and reproducible runs; unavailable pinned
engines fall back safely instead of silently switching to another OCR model.
The same selector is available as `--ocr-engine` on the CLI:

Advanced > Detection also offers **Only remove the selected language**. It is
opt-in and requires recognized text from RapidOCR, PaddleOCR, or EasyOCR;
detection-only boxes are kept. Matching is by script family, so it can separate
Japanese/Cyrillic/Arabic/etc. overlays from Latin text, while Latin-script
languages such as English and French intentionally share one family.

| Priority | Engine | Install | Languages | Notes |
|----------|--------|---------|-----------|-------|
| 1 | **RapidOCR** (OpenCV/ONNX/OpenVINO PP-OCRv6) | `pip install "rapidocr>=2.0.0,<4.0.0"`; Intel: `pip install "openvino>=2025.0.0"` | 100+ | OpenCV 5 DNN is the dependency-light CPU path; accelerated providers remain available |
| 2 | PaddleOCR (reviewed opt-in) | `pip install "paddleocr==3.6.0" --constraint dependency_profiles/cpu.txt` in an isolated environment | 106 | High accuracy reference implementation; installs its own OpenCV wheel |
| 3 | Surya | `pip install surya-ocr` | 90+ | Layout-aware (GPL) |
| 4 | EasyOCR | `pip install "easyocr==1.7.2" --constraint dependency_profiles/cpu.txt` in an isolated environment | 80+ | Legacy fallback; installs its own OpenCV wheel |
| 5 | OpenCV fallback | Built-in | Any | Threshold-based |

Experimental VLM OCR tiers stay default-off. `VSR_VLM_OCR=florence2`,
`VSR_VLM_OCR=qwen25vl`, and `VSR_VLM_OCR=paddleocr-vl` try the heavier
transformer/PaddleOCR adapters before the table above. For CPU/edge
PaddleOCR-VL-1.5, start a local llama.cpp OpenAI-compatible server with the
GGUF model, then set `VSR_PADDLEOCR_VL=1`; use
`VSR_PADDLEOCR_VL_SERVER_URL` when the server is not at
`http://127.0.0.1:8080/v1`. If the server or PaddleOCRVL entrypoint is not
available, detection falls back to the normal cascade.

On NVIDIA systems, setup installs `onnxruntime-gpu>=1.25.0` for the tested
CUDA 12.x ONNX Runtime path; CUDA 13.x currently requires ONNX Runtime
nightly/custom wheels rather than the stable PyPI default. ONNX Runtime
`>=1.25.0` is required for the CPU and CUDA packages -- VSR runs untrusted
OCR/inpaint ONNX models through the runtime, and the self-test and strict
release validation flag older CPU/CUDA builds as a blocking security advisory.
Backend status and
release evidence distinguish `onnxruntime`, `onnxruntime-gpu`, CUDA package
channel, `onnxruntime-directml`, and the providers reported at runtime. On
AMD/Intel systems, setup preflights and installs the latest published/reviewed
DirectML wheel, `onnxruntime-directml==1.24.4`; incompatible Python/platform
combinations fail before the environment is changed and point to CPU or the
Windows ML audit. DirectML is in sustained engineering, with new Windows ONNX
Runtime feature development moving to Windows ML, so diagnostics and release
evidence report that lifecycle separately from CPU/CUDA security floors. On
Intel systems setup also tries `openvino>=2025.0.0` so RapidOCR can use its
OpenVINO engine for CPU/iGPU OCR acceleration. OpenCV 5 DNN runs RapidOCR's
bundled PP-OCRv6 detection and recognition models on CPU without ONNX Runtime;
`python -m backend.cli --ocr-benchmark --ocr-engine opencv-dnn` records recall,
latency, and resident-memory evidence. Set `VSR_RAPIDOCR_ENGINE=opencv` to
force that path, `VSR_RAPIDOCR_ENGINE=onnxruntime` to force ONNX Runtime, or
`VSR_RAPIDOCR_ENGINE=openvino` to request OpenVINO explicitly. When ONNX
Runtime reports `DmlExecutionProvider`,
RapidOCR is initialized with its DirectML provider settings; unsupported
RapidOCR versions or missing providers fall back to CPU automatically.
OpenVINO initialization failures also fall back to ONNX Runtime. RapidOCR
legacy tuple output and current structured object/dict output are both
normalized to the same axis-aligned detector boxes.
Opt-in ONNX inpainters inspect their model `opset_import` metadata before
creating a DirectML session; if the default ONNX opset is newer than DirectML's
supported ceiling, VSR uses the CPU provider instead of failing at session
creation.
Windows ML is currently audit-only, not a replacement for ONNX Runtime
DirectML. Run `python -m backend.processor --audit-windows-ml` on Windows to
check whether the Python bridge, Windows App SDK bootstrap, ONNX Runtime EP
device catalog, and a tiny ONNX identity-model smoke run are available. Until
that probe passes on real user machines and the default OCR/inpaint models are
benchmarked through the Windows ML path, VSR keeps DirectML as the AMD/Intel
GPU route.

Optional model paths such as `VSR_LAMA_ONNX`, `VSR_MIGAN_ONNX`,
`VSR_FASTDVDNET`, `VSR_TRANSNETV2`, `VSR_VACE_CKPT_DIR`, and
`VSR_VIDEOPAINTER_CKPT_DIR`, and `VSR_FLOED_WEIGHTS` are checked against a
local adapter manifest before loading. Known SHA-256 mismatches fall back
instead of deserializing the file. Legacy adapters without a pinned hash still
run, but new strict adapters can require a known hash unless
`VSR_ALLOW_UNVERIFIED_MODELS=1` is set and recorded in release evidence.
Local release evidence also writes `release-advisories.json`; strict mode
blocks unallowed high/critical dependency advisories. The reviewed OpenCV
5.0.0.93 wheel bundles libpng 1.6.57, so older vulnerable OpenCV builds no
longer receive a release exception.
Wan2.1-VACE is available as an opt-in registry mode: set `VSR_VACE=1`, install
the reviewed upstream `vace` package, then either set `VSR_VACE_CKPT_DIR` to a
local `Wan-AI/Wan2.1-VACE-1.3B` snapshot or set `VSR_VACE_AUTO_FETCH=1` with
`huggingface-hub` installed to fetch it into the app model cache.
VideoPainter is available only as a strict local research adapter: set
`VSR_VIDEOPAINTER=1`, review the upstream research/non-commercial and CogVideoX
license terms, set `VSR_VIDEOPAINTER_CKPT_DIR` to a local checkpoint root, set
`VSR_VIDEOPAINTER_COMMAND` to a local wrapper that accepts `--input-video`,
`--mask-video`, and `--output-video`, and opt in with
`VSR_ALLOW_UNVERIFIED_MODELS=1` for unpinned research weights.
FloED is available as a strict local research adapter: set `VSR_FLOED=1`, set
`VSR_FLOED_WEIGHTS` or `VSR_FLOED_CKPT_DIR` to a reviewed FloED checkpoint,
set `VSR_FLOED_COMMAND` to a local wrapper that accepts `--input-dir`,
`--mask-dir`, and `--output-dir`, and opt in with
`VSR_ALLOW_UNVERIFIED_MODELS=1` for unpinned research weights.
MatAnyone 2 is available as an opt-in mask refinement path for decorated or
thin subtitle masks: pass `--matanyone-refine`, set `VSR_MATANYONE=1`, install
the reviewed upstream `matanyone2` package, and set `VSR_MATANYONE_PATH` to a
local checkpoint or snapshot after reviewing the NTU S-Lab License 1.0 terms.
Unpinned PyTorch checkpoints require `VSR_ALLOW_UNVERIFIED_MODELS=1`; malformed
or missing alpha mattes fall back to the original OCR/SAM mask.
CoTracker3 can fill OCR-empty masks inside a video batch by propagating sparse
points from the nearest detected subtitle mask: pass `--cotracker-propagate`,
set `VSR_COTRACKER=1`, and set either `VSR_COTRACKER_REPO` to a reviewed local
co-tracker checkout or `VSR_COTRACKER_REF` to a full 40-character commit SHA
before any `torch.hub` load is allowed. Tags and branches are rejected because
they can move after review. Set `VSR_COTRACKER_MODE=online` only if you need
the online model; the default uses the offline CoTracker3 entrypoint.
VapourSynth `.vpy` input executes Python and therefore requires both
`VSR_VAPOURSYNTH=1` and `VSR_VAPOURSYNTH_SCRIPT_DIR` pointing to a reviewed
script directory. Scripts that resolve outside that directory are rejected,
including through symlinks.
NVIDIA users can request PyNvVideoCodec decode with `--decode-accel pynv`
or `--decode-accel nvdec` after installing NVIDIA's `PyNvVideoCodec` package.
The decoder uses GPU-backed surfaces when available, then converts to CPU BGR
frames for the current OpenCV/OCR/inpaint pipeline; missing packages or failed
opens fall back to software decode.
Smooth-background clips can trade precision for throughput with
`--rife-fast-stride N`: VSR inpaints keyframes every N frames, asks
Practical-RIFE to synthesize the skipped cleaned frames when `practical-rife`
is installed, and duplicates the nearer cleaned keyframe across scene cuts or
missing RIFE adapters.
The legacy `simple-lama-inpainting` PyTorch backend is disabled unless
`VSR_ENABLE_PYTORCH_LAMA=1` is set, because broken native torch wheels can
crash the GUI process during import. Its NumPy <2 cap also conflicts with the
primary OpenCV runtime, so use a separate legacy environment. Prefer `VSR_LAMA_ONNX` or
`VSR_OPENCV_LAMA` for automatic LaMa acceleration.

## CLI Usage

Process files from the command line:

```bash
python -m backend.processor -i input.mp4 -o output.mp4 -m lama --lang en --crf 20
```

For OCR-empty frames with speech, the optional Whisper fallback can
mask the bottom subtitle band. The default backend is `faster-whisper`;
FFmpeg 8 builds that include the `whisper` filter can instead use a
local whisper.cpp ggml model without Python ML dependencies:

```bash
python -m backend.processor -i input.mp4 -o output.mp4 --whisper-fallback --whisper-backend ffmpeg --ffmpeg-whisper-model C:\models\ggml-base.en.bin
```

The localization workflow can erase the original burned-in text and re-embed
a translated UTF-8 SRT in the same run. Supplying the translated captions is
the simplest deterministic path:

```powershell
python -m backend.processor -i input.mp4 -o localized.mp4 --translated-srt captions.es.srt --translation-style "FontSize=24,Outline=2"
```

To generate captions, provide a source SRT or let the existing OCR collection
(and then an enabled Whisper fallback) supply source cues. VSR invokes the
selected command directly without a shell and sends one bounded JSON document
on stdin; VSR does not include or contact a translation service. The chosen
command controls how cue text is handled:

```powershell
python -m backend.processor -i input.mp4 -o localized.mp4 --translate --translation-source-srt captions.en.srt --translation-source-lang en --translation-target-lang es --translation-command C:\tools\translate.py
```

The request schema is `vsr.translation_request.v1` with `sourceLanguage`,
`targetLanguage`, and `cues` entries containing `index` and `text`. The
command must return `vsr.translation_response.v1` with a `translations` array
in the same order and length. Timing and cue identifiers stay unchanged; empty,
malformed, oversized, or count-mismatched results fail the job. Generated
source and translated SRTs are saved beside the video. The reproducibility
sidecar records their names, SHA-256 hashes, provider, source kind, languages,
and final embed status without recording caption text. The workflow is off by
default and cannot be combined with the separate `--restyle` pass.

Embedded subtitle tracks can be inspected or remuxed without OCR, frame
decode, inpainting, or video re-encode:

```bash
python -m backend.processor -i input.mkv --soft-subtitle-dry-run
python -m backend.processor --pattern "inputs/*.mkv" --soft-subtitle-dry-run --soft-subtitle-plan-json soft-plan.json
python -m backend.processor -i input.mkv -o stripped.mkv --strip-soft-subtitles
```

When the input is a directory of images, `--output-frames` writes the cleaned
frames as individual PNGs instead of encoding a video:

```bash
python -m backend.processor -i frames_dir/ -o cleaned_dir/ --output-frames
```

In the GUI, queued videos with embedded subtitle tracks show a track summary;
right-click the item to fast strip, fast remux/keep, or continue with
burned-in cleanup.

Pattern batches and GUI batches write `vsr-batch-summary.json` and
`vsr-batch-summary.md` next to their outputs when they finish. The report
records each input, selected output path, codec/duration/subtitle preflight
data, source-aware output-quality warning, planned action, final status, and
elapsed time for skipped, checkpointed, paused, remuxed, processed, or failed
files.
They also break each item down by decode, OCR, mask, inpaint, encode, mux, and
quality-analysis time, with a run-level slowest-stage summary for diagnosing
slow hardware, OCR, model, or muxing bottlenecks.
Before processing, CLI and GUI batches compare source codec/resolution/bitrate
against the selected output codec and CRF; risky settings are shown as
preflight warnings, and the report records the safer recommendation plus that
the user continued after the warning. When quality reports are enabled, batch
summaries also include a `passed`, `review`, or `unknown` quality gate using
ROI metrics, a cheap residual-text score, and an adjacent-frame temporal
flicker score, plus any quality-sheet preview path for review-needed outputs.
A failed gate changes the batch row status to `review-needed`; skipped and
remux-only rows are marked `not_applicable`.
Review-needed queue items expose **Retry with suggested settings**, which
applies the quality gate's ladder step to that item only and records the
before/after retry config in the next batch report.
When the gate identifies residual text, adjacent-frame flicker, or a
low-confidence detection, **Correct mask** opens the flagged frame span in an
internal editor. Paint missing mask pixels or subtract over-masked pixels,
optionally propagate the stroke through the bounded span, and use undo/redo
before preparing the retry. VSR persists the ordered corrections with exact
frame bounds and, when the prior cleaned output is still available, reprocesses
only those ranges while copying the previously cleaned frames everywhere else.

Masks and soft alpha mattes can round-trip through an external compositor
without the old lossy `.mask.mp4` artifact. FFV1 writes
`<output>.mask.mkv`; PNG mode writes `<output>.mask/frame_########.png`.
Both formats include `<output>.mask.json` with exact source frame bounds,
CFR/VFR timestamps, durations, dimensions, and the export hash:

```bash
python -m backend.processor -i input.mp4 -o cleaned.mp4 --export-mask --mask-export-format ffv1
python -m backend.processor -i input.mp4 -o cleaned.mp4 --export-mask --mask-export-format png
python -m backend.processor -i input.mp4 -o revised.mp4 --import-mask cleaned.mask.json --mask-import-mode replace
```

Edit the referenced artifact while keeping the manifest beside it, then import
in `replace`, `add`, or `subtract` mode. VSR validates every frame, dimension,
frame count, timestamp, duration, and timing mode before processing begins.
The output reproducibility sidecar records the imported artifact's current
SHA-256, whether it differs from the exported hash, and the deterministic mask
composition order. **Review mask** shows that composed result before a run.

For static-camera overlays, a timed rectangle can use a deterministic clean
plate instead of estimated or neural pixels. Open **Set Region**, add and
select a timed rectangle, then choose a same-pixel-size clean image in the
**Clean reference** panel. Preview `Auto`, `Translation`, or `Homography`
alignment at the scrubbed frame, enable per-frame color matching, and set the
confidence floor. Auto prefers translation unless homography materially
improves the match. During processing, a plate is copied only where that
region intersects the finalized mask; low-confidence frames retain their mask
and go through the normal inpainter. Settings and queue snapshots retain the
plate assignment. Each output sidecar records the plate filename, SHA-256,
timed rectangle, alignment policy, confidence range/mean, method counts, color
delta, and accepted/fallback frame counts without exposing an absolute path in
the clean-reference evidence.

Long video runs can pause at safe frame-batch boundaries. In the GUI, click
**Pause batch** while processing; the current video writes checkpoint frames
under the selected work directory, or under
`%APPDATA%\VideoSubtitleRemoverPro\checkpoints\` when no work directory is set,
and returns to the queue as `Paused`. Starting the batch again resumes from the
first missing frame. In the CLI, press Ctrl-C once to request the same safe
pause; re-run the same command to resume. If the input, output path, frame
count, frame rate, size, or processing settings changed, VSR warns and restarts
that file from the beginning instead of trusting stale checkpoint frames.

### Reference Clip Contributions

Use the **Edge-case clip** GitHub issue form before adding real media to
`tests/clips/`. Real fixtures must be short, redistributable with this
MIT-licensed project, and manifest-backed with SHA-256, source URL, license
proof URL, retrieval date, rights confirmation, reproduction settings, and
metric floors. Good starting sources are NASA public-domain media, Library of
Congress public-domain media, Wikimedia Commons compatible-license files, or a
clip you shot and grant as CC0.

<!-- BEGIN GENERATED CLI REFERENCE -->
This table is generated from the live argparse actions and their category,
default, range, visibility, and deprecation metadata. Regenerate it with
`python scripts/generate_cli_reference.py --write`.

#### General

| Flag | Description | Default | Range/choices | Status |
|------|-------------|---------|---------------|--------|
| `-h`, `--help` | show this help message and exit | - | - | Public |

#### Inputs, batches, and reproducibility

| Flag | Description | Default | Range/choices | Status |
|------|-------------|---------|---------------|--------|
| `--input`, `-i` | Input file path | - | - | Public |
| `--output`, `-o` | Output file path | - | - | Public |
| `--pattern` | Glob pattern for batch mode (e.g. 'inputs/*.mp4') | - | - | Public |
| `--out-dir` | Output directory for batch mode | - | - | Public |
| `--config` | JSON config file (key=value pairs overriding CLI defaults) | - | - | Public |
| `--config-schema-version` | Canonical processing-config schema version for reproducible commands. | - | - | Public |
| `--set` | Override any canonical processing field; repeat for multiple values. | - | - | Public |
| `--preset` | Apply a built-in or user preset by name. | - | - | Public |
| `--list-presets` | Print every known preset and exit. | Off | - | Public |
| `--checkpoint-dir` | Checkpoint dir for crash-resume and pause/resume (default: %APPDATA%/.../checkpoints) | - | - | Public |
| `--work-dir` | Writable root for temporary, mask, checkpoint, and resume artifacts; falls back with a warning when unavailable. | - | - | Public |
| `--no-resume` | Ignore existing checkpoints and reprocess every file; pause checkpoints are still written for this run | Off | - | Public |
| `--start` | Start time in seconds | 0 | >=0 seconds | Public |
| `--end` | End time in seconds (0=full) | 0 | 0 or >= start | Public |
| `--nle-input` | Parse an EDL/FCPXML to extract time segments for processing. | - | - | Public |
| `--input-fps` | FPS for directory-of-images input. | 24.0 | 1..240 | Public |
| `--output-frames` | Write cleaned frames as individual PNGs instead of a video. | Off | - | Public |
| `--skip-existing` | Skip inputs whose output path already exists. | Off | - | Public |

#### Removal, detection, and masks

| Flag | Description | Default | Range/choices | Status |
|------|-------------|---------|---------------|--------|
| `--mode`, `-m` | Inpainting algorithm. | sttn | sttn \| lama \| propainter \| auto \| migan | Public |
| `--gpu`, `-g` | GPU device ID (-1 for CPU) | 0 | -1 or >=0 | Public |
| `--lang`, `-l` | Detection language | en | - | Public |
| `--language-filter` | Only mask OCR text matching the selected language's script. | Off | - | Public |
| `--skip-detection` | Skip automatic detection (STTN only) | Off | - | Public |
| `--fast` | Fast mode (LAMA only) | Off | - | Public |
| `--threshold` | Detection threshold (0.1-1.0) | 0.5 | 0.1..1.0 | Public |
| `--vertical` | Vertical-text mode (rotate frames 90 CCW before OCR). | Off | - | Public |
| `--frame-skip` | Reuse detection mask for N frames between detections | 0 | 0..240 frames | Public |
| `--mask-dilate` | Mask dilation in pixels (0=off) | 8 | 0..100 pixels | Public |
| `--confidence-dilate` | Scale mask dilation inversely with OCR confidence | Off | - | Public |
| `--mask-feather` | Gaussian edge feathering in pixels (0=off) | 4 | 0..100 pixels | Public |
| `--temporal-smooth` | Post-inpaint temporal smoothing radius for LaMa (0=off, 1-5) | 0 | 0..5 frames | Public |
| `--edge-ring` | Edge-ring colour match width in pixels (0=off) | 2 | 0..32 pixels | Public |
| `--flow-warp` | Farneback flow-warp TBE frames before aggregation | Off | - | Public |
| `--no-scene-split` | Disable scene-cut splitting inside TBE batches | Off | - | Public |
| `--pyscenedetect` | Prefer PySceneDetect AdaptiveDetector for scene cuts. | Off | - | Public |
| `--transnetv2` | Prefer TransNetV2 (deep CNN) for scene-cut detection. | Off | - | Public |
| `--denoise-detect` | Run a denoise pass on the detection-frame stream. | Off | - | Public |
| `--sam2-refine` | SAM 2 mask refinement of detected boxes. | Off | - | Public |
| `--matanyone-refine` | MatAnyone 2 alpha-matte refinement of masks. | Off | - | Public |
| `--cotracker-propagate` | Use CoTracker3 to fill OCR-empty masks in a batch. | Off | - | Public |
| `--no-tbe` | Disable Temporal Background Exposure (STTN/ProPainter use cv2) | Off | - | Public |
| `--no-adaptive-batch` | Disable VRAM-probe-driven batch sizing | Off | - | Public |
| `--temporal-mask-union` | Scene-cut-safe temporal mask stabilization: OR each frame's mask with a short trailing window (auto detection only) to retain pixels missed on single frames or moving overlays; resets at scene cuts | Off | - | Public |
| `--temporal-mask-window` | Trailing window size for --temporal-mask-union (1-15) | 3 | 1..15 frames | Public |
| `--auto-band` | Auto-detect the dominant subtitle band before processing | Off | - | Public |
| `--no-kalman` | Disable Kalman detection smoothing | Off | - | Public |
| `--no-phash` | Disable perceptual-hash adaptive mask reuse | Off | - | Public |
| `--phash-distance` | pHash Hamming distance threshold for mask reuse (0-64) | 4 | 0..64 | Public |
| `--colour-tune` | Grow the mask by dominant-colour match inside each box | Off | - | Public |
| `--colour-tolerance` | Lab-space colour distance tolerance for colour-tune | 25 | 0..255 | Public |
| `--auto-threshold` | AUTO-mode exposure threshold (0-1) | 0.55 | 0..1 | Public |
| `--keep-chyrons` | Leave persistent text (logos, lower-thirds, tickers). | Off | - | Public |
| `--keep-subtitles` | Leave non-persistent text (dialogue captions). | Off | - | Public |
| `--chyron-min-hits` | Kalman-track frame count to classify as chyron. | 90 | 1..100000 frames | Public |
| `--karaoke-grouping` | Fuse per-syllable OCR boxes on the same line. | Off | - | Public |
| `--karaoke-x-gap` | Max horizontal gap (px) between karaoke boxes. | 20 | 0..1024 pixels | Public |
| `--karaoke-y-overlap` | Min vertical overlap ratio for karaoke line fusion. | 0.5 | 0..1 | Public |

#### Speech and subtitle tracks

| Flag | Description | Default | Range/choices | Status |
|------|-------------|---------|---------------|--------|
| `--whisper-fallback` | Whisper-driven bottom-band default mask on OCR-empty frames. | Off | - | Public |
| `--whisper-backend` | Whisper fallback backend. | faster-whisper | faster-whisper \| ffmpeg | Public |
| `--restyle` | Re-burn an .srt or .ass subtitle file onto the cleaned output. | - | - | Public |
| `--restyle-style` | ASS force_style override for --restyle (e.g. 'FontSize=24,PrimaryColour=&H00FFFFFF'). | - | - | Public |
| `--translate` | Erase subtitles, translate a source SRT locally, and re-embed it. | Off | - | Public |
| `--translated-srt` | Validated UTF-8 SRT that is already translated; bypasses a provider. | - | - | Public |
| `--translation-source-srt` | Source-language SRT to translate; otherwise OCR/Whisper cues are used. | - | - | Public |
| `--translation-provider` | Registered local translation provider name (default: command). | command | - | Public |
| `--translation-source-lang` | Source language tag passed to the local translation provider. | auto | - | Public |
| `--translation-target-lang` | Required target language tag when generating translated subtitles. | - | - | Public |
| `--translation-command` | Local executable or Python script using the VSR translation JSON protocol. | - | - | Public |
| `--translation-style` | ASS force_style override for the translated subtitle burn pass. | - | - | Public |
| `--translation-timeout` | Timeout for the local translation provider command. | 300.0 | 5..3600 seconds | Public |
| `--whisper-model` | faster-whisper model size. | tiny | tiny \| base \| small \| medium \| large \| large-v2 \| large-v3 | Public |
| `--ffmpeg-whisper-model` | Path to a local whisper.cpp ggml model for --whisper-backend ffmpeg. | - | - | Public |
| `--ffmpeg-whisper-queue` | FFmpeg whisper filter queue size in seconds. | 3.0 | 0.02..3600 seconds | Public |
| `--ffmpeg-whisper-vad-model` | Path to a Silero VAD ONNX model for FFmpeg Whisper. | - | - | Public |
| `--ffmpeg-whisper-vad-threshold` | VAD confidence threshold (0.0-1.0, default 0.5). | 0.5 | 0..1 | Public |
| `--ffmpeg-whisper-min-speech` | Minimum speech duration for VAD segments (default 0). | 0.0 | 0..30 seconds | Public |
| `--export-srt` | Write an .srt sidecar with detected text | Off | - | Public |
| `--ocr-fix` | Apply a per-language OCR-fix replace list to the exported SRT text (built-in defaults plus %APPDATA%/VideoSubtitleRemoverPro/ocr_fix/{lang}.json). | Off | - | Public |
| `--soft-subtitle-dry-run` | Print embedded subtitle tracks and planned action, then exit. | Off | - | Public |
| `--soft-subtitle-plan-json` | Write soft-subtitle dry-run preflight details as JSON. | - | - | Public |
| `--strip-soft-subtitles` | Fast remux that removes embedded subtitle tracks without OCR. | Off | - | Public |
| `--keep-soft-subtitles` | Fast remux that keeps embedded subtitle tracks without OCR. | Off | - | Public |
| `--burned-in-only` | Ignore embedded subtitle tracks and run burned-in cleanup normally. | Off | - | Public |

#### Output and post-processing

| Flag | Description | Default | Range/choices | Status |
|------|-------------|---------|---------------|--------|
| `--no-audio` | Don't preserve audio | Off | - | Public |
| `--crf` | Output CRF quality (15-35) | 23 | 15..35 | Public |
| `--upscale` | Post-cleanup upscale (Real-ESRGAN). | 0 | 0 \| 2 \| 3 \| 4 | Public |
| `--no-color-preserve` | Do not re-tag the output with the source's color signalling. | Off | - | Public |
| `--nle-sidecar` | Emit an EDL or FCPXML sidecar next to the output. | off | off \| edl \| fcpxml | Public |
| `--swinir` | Post-cleanup SwinIR restoration pass. | Off | - | Public |
| `--seedvr2` | Post-cleanup SeedVR2 restoration pass. | Off | - | Public |
| `--film-grain` | Additive film grain after cleanup (0..0.5; 0 disables). | 0.0 | 0..0.5 | Public |
| `--watermark` | Burn a PNG watermark onto the output after cleanup. | - | - | Public |
| `--watermark-position` | Watermark corner position (default bottom-right). | bottom-right | top-left \| top-right \| bottom-left \| bottom-right \| center | Public |
| `--watermark-opacity` | Watermark opacity 0.0-1.0 (default 1.0). | 1.0 | 0..1 | Public |
| `--watermark-margin` | Watermark margin from edge in pixels (default 16). | 16 | 0..500 pixels | Public |
| `--no-hw-encode` | Disable hardware encoding (force libx264) | Off | - | Public |
| `--d3d12-accel` | Opt into FFmpeg 8.1+ D3D12 filters and encoding after a byte-valid runtime smoke; falls back automatically. | Off | - | Public |
| `--codec` | Output video codec (vvc requires FFmpeg with libvvenc). | h264 | h264 \| h265 \| av1 \| vvc | Public |
| `--export-mask` | Export a lossless grayscale matte plus timing manifest | Off | - | Public |
| `--mask-export-format` | Lossless matte export as FFV1 video or a PNG sequence. | ffv1 | ffv1 \| png | Public |
| `--import-mask` | Import an edited .mask.json timing manifest before inpainting. | - | - | Public |
| `--mask-import-mode` | Compose the imported matte after native mask generation. | replace | replace \| add \| subtract | Public |
| `--deinterlace` | Force ffmpeg yadif deinterlace before processing | Off | - | Public |
| `--no-deinterlace-detect` | Skip the automatic ffprobe interlacing detection | Off | - | Public |
| `--keyframe-detect` | OCR only at video I-frames (ffprobe-probed) | Off | - | Public |
| `--quality-report` | Compute PSNR/SSIM on a random frame sample after run | Off | - | Public |
| `--quality-sheet` | Render a side-by-side comparison PNG alongside the report. | Off | - | Public |
| `--loudnorm` | EBU R128 loudness target in LUFS. | 0.0 | 0 (off) or -70..-5 LUFS | Public |
| `--decode-accel` | Hardware-decode hint (OpenCV or PyNvVideoCodec). | off | off \| auto \| any \| d3d11 \| vaapi \| mfx \| pynv \| nvdec | Public |
| `--single-audio` | Mux only the first audio stream. | Off | - | Public |

#### Performance and recovery

| Flag | Description | Default | Range/choices | Status |
|------|-------------|---------|---------------|--------|
| `--rife-fast-stride` | Inpaint every Nth frame and synthesize skipped frames with Practical-RIFE (0 disables). | 0 | 0..60 frames | Public |
| `--max-retries` | Automatically re-attempt a batch item that fails with a transient error (GPU glitch, ffmpeg hiccup, timeout) up to N times with backoff (0=off, max 10) | 0 | 0..10 | Public |
| `--retry-backoff` | Base seconds between transient retries (0-600; each later attempt waits a multiple of this value) | 5.0 | 0..600 seconds | Public |
| `--no-prefetch` | Disable the worker-thread frame prefetcher. | Off | - | Public |
| `--prefetch-queue` | Bounded prefetch queue size in frames. | 0 | 0..512 frames | Public |

#### Diagnostics and automation

| Flag | Description | Default | Range/choices | Status |
|------|-------------|---------|---------------|--------|
| `--audit-onnx` | Audit all discoverable ONNX models for DirectML opset compatibility and exit. | Off | - | Public |
| `--audit-windows-ml` | Probe the Windows ML Python path with a tiny ONNX smoke model and exit. | Off | - | Public |
| `--scan-weights` | Scan cached model weights and verify SHA-256 against known hashes, then exit. | Off | - | Public |
| `--cache-info` | Print cache directory inventory with sizes and exit. | Off | - | Public |
| `--cache-clean` | Remove stale cache entries (checkpoints, proxies, TRT engines) and exit. | Off | - | Public |
| `--model-cache-export` | Write a portable model-cache zip with SHA-256 manifest and exit. | - | - | Public |
| `--model-cache-import` | Import a verified portable model-cache zip into the app model cache and exit. | - | - | Public |
| `--support-bundle` | Write a redacted diagnostics zip and exit. | - | - | Public |
| `--validate-config` | Print the resolved ProcessingConfig as JSON and exit. | Off | - | Public |
| `--self-test` | Probe OCR engines, inpaint backends, GPU providers, and codecs, then print results and exit. | Off | - | Public |
| `--inference-smoke` | Run a generated text image and masked frame through the OCR and inpaint backends to prove they actually execute (records provider/timing), then exit. No model downloads. Uses --gpu to pick the device. | Off | - | Public |
| `--ocr-benchmark` | Benchmark the active OCR detector on synthetic ground-truth subtitle fixtures (recall, latency, and memory) and print JSON evidence, then exit. Use --gpu to pick the device. Gate any default-detector swap on the meets_floors verdict. | Off | - | Public |
| `--ocr-engine` | Select the OCR detector for processing or --ocr-benchmark; auto uses the best available engine. | auto | auto \| rapidocr \| opencv-dnn \| paddleocr \| easyocr \| opencv | Public |
| `--dry-run` | Validate the run without encoding: probe each input, run detection on a few sampled frames, check the requested codec is available, and print a per-file plan, then exit. Combine with --json for machine output. | Off | - | Public |
| `--json` | Emit a machine-readable JSON result to stdout (the --dry-run plan, or the batch/file result). | Off | - | Public |
| `--auto-lang-probe` | Probe the first frame for script/language and print a suggestion, then exit. Requires -i. | Off | - | Public |
| `--intent` | Natural-language cleanup intent (e.g. 'remove subtitles', 'remove logo'). Prints config changes and exits. | - | - | Public |
| `--json-log` | Append a structured JSON-line log at PATH. | - | - | Public |

<!-- END GENERATED CLI REFERENCE -->

`--config` accepts the same manual region schema used by the GUI. Use
`subtitle_area` for one global rectangle, `subtitle_areas` for multiple global
rectangles, `subtitle_region_spans` for frame-time-specific masks, or
`subtitle_region_keyframes` for an interpolated moving rectangle/polygon:

```json
{
  "subtitle_region_spans": [
    {"rect": [80, 720, 1180, 820], "start": 0.0, "end": 14.5},
    {"rect": [120, 40, 900, 150], "start": 14.5, "end": 0.0}
  ],
  "sttn_skip_detection": true
}
```

Moving-region tracks use source-pixel coordinates and require at least two
same-shape anchors. Polygon anchors keep the same vertex count across the
track:

```json
{
  "subtitle_region_keyframes": [
    {
      "keyframes": [
        {"time": 2.0, "polygon": [80, 700, 420, 700, 420, 790, 80, 790]},
        {"time": 8.0, "polygon": [520, 680, 860, 680, 860, 770, 520, 770]}
      ]
    }
  ],
  "sttn_skip_detection": true
}
```

`end: 0.0` means the region stays active through the end of the processed
range. With `sttn_skip_detection` enabled, inactive timed ranges produce an
empty mask instead of reusing a previous manual mask.

Queue-card **Copy CLI command** output includes a schema version and repeatable
`--set FIELD=JSON` values for every non-default per-item processing control.
This keeps fields without a dedicated legacy flag reproducible too. Use
`--validate-config` to inspect the complete resolved canonical config.

## Configuration

Settings are stored in `%APPDATA%\VideoSubtitleRemoverPro\settings.json` and persist across sessions.

<!-- BEGIN GENERATED CONFIG REFERENCE -->
### Canonical processing fields

These fields are accepted by `--set FIELD=JSON` and JSON config overlays.
The table is generated directly from `ProcessingConfig` in registry order.

| Field | Type | Default |
|-------|------|---------|
| `mode` | `InpaintMode` | `sttn` |
| `device` | `str` | `cuda:0` |
| `sttn_skip_detection` | `bool` | `Off` |
| `sttn_neighbor_stride` | `int` | `10` |
| `sttn_reference_length` | `int` | `10` |
| `sttn_max_load_num` | `int` | `30` |
| `lama_super_fast` | `bool` | `Off` |
| `subtitle_area` | `Optional[Tuple[int, int, int, int]]` | `-` |
| `detection_threshold` | `float` | `0.5` |
| `detection_lang` | `str` | `en` |
| `detection_engine` | `str` | `auto` |
| `language_mask_filter` | `bool` | `Off` |
| `detection_frame_skip` | `int` | `0` |
| `detection_vertical` | `bool` | `Off` |
| `whisper_fallback` | `bool` | `Off` |
| `whisper_backend` | `str` | `faster-whisper` |
| `whisper_model_size` | `str` | `tiny` |
| `whisper_model_path` | `str` | `-` |
| `whisper_queue_seconds` | `float` | `3.0` |
| `whisper_vad_model` | `str` | `-` |
| `whisper_vad_threshold` | `float` | `0.5` |
| `whisper_min_speech_duration` | `float` | `0.0` |
| `upscale_factor` | `int` | `0` |
| `film_grain_strength` | `float` | `0.0` |
| `swinir_restore` | `bool` | `Off` |
| `seedvr2_restore` | `bool` | `Off` |
| `preserve_color_metadata` | `bool` | `On` |
| `watermark_image` | `str` | `-` |
| `watermark_position` | `str` | `bottom-right` |
| `watermark_opacity` | `float` | `1.0` |
| `watermark_margin` | `int` | `16` |
| `restyle_subtitle` | `str` | `-` |
| `restyle_style` | `str` | `-` |
| `translation_enabled` | `bool` | `Off` |
| `translation_srt` | `str` | `-` |
| `translation_source_srt` | `str` | `-` |
| `translation_provider` | `str` | `command` |
| `translation_source_lang` | `str` | `auto` |
| `translation_target_lang` | `str` | `-` |
| `translation_command` | `str` | `-` |
| `translation_style` | `str` | `-` |
| `translation_timeout_seconds` | `float` | `300.0` |
| `nle_sidecar` | `str` | `off` |
| `mask_dilate_px` | `int` | `8` |
| `mask_feather_px` | `int` | `4` |
| `confidence_weighted_dilation` | `bool` | `Off` |
| `confidence_dilation_scale` | `float` | `1.5` |
| `lama_tile_size` | `int` | `512` |
| `lama_tile_overlap` | `int` | `64` |
| `temporal_smooth_radius` | `int` | `0` |
| `tbe_enable` | `bool` | `On` |
| `tbe_min_coverage` | `int` | `3` |
| `tbe_use_median` | `bool` | `On` |
| `tbe_flow_warp` | `bool` | `Off` |
| `tbe_scene_cut_split` | `bool` | `On` |
| `tbe_scene_cut_threshold` | `float` | `0.35` |
| `tbe_scene_cut_use_pyscenedetect` | `bool` | `Off` |
| `tbe_scene_cut_use_transnetv2` | `bool` | `Off` |
| `detection_denoise` | `bool` | `Off` |
| `sam2_refine` | `bool` | `Off` |
| `matanyone_refine` | `bool` | `Off` |
| `cotracker_propagate` | `bool` | `Off` |
| `rife_fast_stride` | `int` | `0` |
| `edge_ring_px` | `int` | `2` |
| `subtitle_areas` | `Optional[List[Tuple[int, int, int, int]]]` | `-` |
| `subtitle_region_spans` | `Optional[List[dict]]` | `-` |
| `subtitle_region_keyframes` | `Optional[List[dict]]` | `-` |
| `manual_mask_corrections` | `Optional[List[dict]]` | `-` |
| `export_mask_video` | `bool` | `Off` |
| `mask_export_format` | `str` | `ffv1` |
| `mask_import_path` | `str` | `-` |
| `mask_import_mode` | `str` | `replace` |
| `export_srt` | `bool` | `Off` |
| `ocr_fix_enable` | `bool` | `Off` |
| `adaptive_batch` | `bool` | `On` |
| `gpu_oom_recovery` | `bool` | `On` |
| `batch_max_retries` | `int` | `0` |
| `batch_retry_backoff_seconds` | `float` | `5.0` |
| `temporal_mask_union` | `bool` | `Off` |
| `temporal_mask_window` | `int` | `3` |
| `auto_exposure_threshold` | `float` | `0.55` |
| `deinterlace` | `bool` | `Off` |
| `deinterlace_auto` | `bool` | `On` |
| `keyframe_detection` | `bool` | `Off` |
| `quality_report` | `bool` | `Off` |
| `kalman_tracking` | `bool` | `On` |
| `kalman_iou_threshold` | `float` | `0.3` |
| `kalman_max_age` | `int` | `2` |
| `phash_skip_enable` | `bool` | `On` |
| `phash_skip_distance` | `int` | `4` |
| `colour_tune_enable` | `bool` | `Off` |
| `colour_tune_tolerance` | `int` | `25` |
| `time_start` | `float` | `0.0` |
| `time_end` | `float` | `0.0` |
| `work_directory` | `str` | `-` |
| `preserve_audio` | `bool` | `On` |
| `output_format` | `str` | `mp4` |
| `output_quality` | `int` | `23` |
| `use_hw_encode` | `bool` | `On` |
| `d3d12_accel` | `bool` | `Off` |
| `output_frames` | `bool` | `Off` |
| `output_codec` | `str` | `h264` |
| `loudnorm_target` | `float` | `0.0` |
| `decode_hw_accel` | `str` | `off` |
| `multi_audio_passthrough` | `bool` | `On` |
| `prefetch_decode` | `bool` | `On` |
| `prefetch_queue_size` | `int` | `0` |
| `input_fps` | `float` | `24.0` |
| `quality_report_sheet` | `bool` | `Off` |
| `remove_subtitles` | `bool` | `On` |
| `remove_chyrons` | `bool` | `On` |
| `chyron_min_hits` | `int` | `90` |
| `karaoke_grouping` | `bool` | `Off` |
| `karaoke_x_gap_px` | `int` | `20` |
| `karaoke_y_overlap` | `float` | `0.5` |

<!-- END GENERATED CONFIG REFERENCE -->

### Advanced Settings

| Setting | Description | Default | Range |
|---------|-------------|---------|-------|
| Neighbor Stride | STTN temporal window | 10 | 5-30 |
| Reference Length | STTN reference frames | 10 | 5-30 |
| Max Load Frames | Batch size | 30 | 10-100 |
| CRF Quality | Output quality (lower=better) | 23 | 15-35 |
| Output Codec | H.264 / H.265 / AV1 / VVC (H.266) | h264 | h264/h265/av1/vvc; VVC requires FFmpeg with `libvvenc` |
| Frame Skip | Reuse detection mask for N frames | 0 | 0-10 |
| Mask Dilate | Expand detected regions (px) | 8 | 0-20 |
| Mask Feather | Soft alpha-blend at boundary (px) | 4 | 0-15 |
| Timed-region Clean Reference | Same-size clean plate with translation/homography preview, color matching, and confidence-gated inpaint fallback | None | Per timed rectangle |
| TBE Coverage | Min frames a pixel must be unmasked to trust its exposure | 3 | 1-10 |
| HW Encoding | Use NVENC/QSV/AMF if available | On | On/Off |
| FFmpeg D3D12 | Windows-only experimental upload, scale, deinterlace, and encode path with runtime validation and automatic fallback | Off | On/Off; FFmpeg 8.1+ |
| Localization | Re-embed a provided translated SRT or translate OCR/Whisper cues through a selected local command, with optional ASS `force_style` text | Off | UTF-8 SRT; source/target language tags; executable or Python script |
| HW Decode Hint | OpenCV/PyNvVideoCodec decode hint with software fallback | off | off/auto/d3d11/vaapi/mfx/pynv/nvdec |
| Loudness Target | EBU R128 LUFS target (0 = off) | 0 | 0 or -70..-5 |
| Multi-track Audio | Pass through every audio stream | On | On/Off |
| Quality Sheet | Side-by-side PNG next to output | Off | On/Off |
| Work Directory | Temporary, mask, checkpoint, and resume storage; write-tested before each batch | System temporary directory | Writable folder |
| Interface Text Size | Scale text and dependent controls; restart to apply | 100% | 100%-200% |

The D3D12 option stays off by default because advertised FFmpeg capabilities
do not prove that a display driver accepts a codec profile. Each selected
codec must first produce and re-read a complete 30-frame MP4. Processing then
uses D3D12 frame upload and `scale_d3d12`; interlaced SDR input also tries
`deinterlace_d3d12`. A failed smoke or processing command automatically moves
to the existing NVENC/QSV/AMF chain and then to the software encoder.

At 150% and 200%, the minimum 980x720 window switches to a compact, vertically
scrollable layout so actions stay keyboard reachable without horizontal
scrolling. The setting is under **Detailed controls** and applies to both the
default and high-contrast themes after restart.

The same panel offers a restart-applied interface language selector with
System, English, and every compiled catalog discovered under `locale/` or the
per-user `%APPDATA%\VideoSubtitleRemoverPro\locale\` directory. Translation
contributors can refresh the POT template, merge PO files, build the bundled
pseudo-locale, validate placeholders/plurals/UTF-8, compile MO files, and print
coverage in one deterministic command:

```powershell
python scripts/i18n_catalogs.py update
```

Use `python scripts/i18n_catalogs.py check` in review or CI; it fails when the
template, PO keys, pseudo-locale, or compiled catalogs drift.

## Troubleshooting

<details>
<summary><b>RTX 50-series (Blackwell): "no kernel image is available" or CPU-only</b></summary>

RTX 50-series cards (5070 / 5080 / 5090, compute capability sm_120) need
**CUDA 12.8** wheels, i.e. **PyTorch 2.7 or newer** from the `cu128` index.
The older `cu118` / `cu121` builds contain no Blackwell kernels and will
either raise `no kernel image is available for execution on the device`
or silently fall back to CPU.

`Run_VSR_Pro.bat` / `setup.py` now auto-detect 50-series cards and install
the `cu128` build. To fix an existing environment manually:

```powershell
.\venv\Scripts\activate
pip uninstall -y torch torchvision
pip install torch>=2.10.0 torchvision>=0.25.0 --index-url https://download.pytorch.org/whl/cu128
```

torch 2.7+ supports Python 3.9-3.13, so a recent Python is fine. If
PaddleOCR fails to load on Blackwell, detection automatically falls back
to RapidOCR (ONNX Runtime), which is GPU-generation agnostic.

</details>

<details>
<summary><b>Python 3.14 installs but NVIDIA CUDA is unavailable</b></summary>

PyTorch does not publish Windows CUDA wheels for Python 3.14 yet. If you
run setup with Python 3.14 and an NVIDIA GPU, setup stops before silently
installing a CPU-only torch build and recommends Python 3.12 or 3.13 for
GPU acceleration.

CPU-only use is still possible. Set `VSR_ALLOW_PY314_CPU=1` before
running setup if you explicitly accept slower CPU inference.

</details>

<details>
<summary><b>Colors shift / look washed out (TV vs full color range)</b></summary>

The upstream project re-encodes the output without carrying the source's
color signalling, so a **limited / TV-range (BT.601/709)** clip can come
back looking washed out or with shifted colors. This fork preserves the
source's `color_primaries`, `color_transfer`, `color_space`, and
**`color_range`** tags onto the final encode (`preserve_color_metadata`,
on by default; CLI `--no-color-preserve` to disable). Decoding is handled
by OpenCV's FFmpeg backend, which applies the correct YUV->RGB conversion
for the signalled range, and the same tags are re-applied on write so
players interpret the result the same way as the source.

For HDR10/HLG sources with color preservation enabled, VSR promotes the final
encode to an HDR-capable codec when needed (default H.264 becomes HEVC),
decodes a high-bit `bgr48le` source surface through FFmpeg when available, and
requests a 10-bit output surface (`yuv420p10le`) before re-applying the source
color tags. OCR and inpainting still operate on 8-bit BGR working copies, so
the cleaned subtitle pixels are derived from that model path, but unmasked HDR
pixels are kept from the high-bit source surface instead of being flattened
through an invalid 8-bit H.264 HDR encode. For standard SDR limited-range
content, colors are preserved. If you still see a mismatch, attach the
`ffprobe` color fields of your source to a bug report.

</details>

<details>
<summary><b>CUDA out of memory</b></summary>

- Reduce Max Load Frames in Advanced Settings
- Switch to LAMA mode (lower VRAM)
- Use CPU mode as fallback

</details>

<details>
<summary><b>No audio in output</b></summary>

- Install FFmpeg: `winget install ffmpeg`
- Ensure "Preserve original audio" is checked

</details>

<details>
<summary><b>Poor detection accuracy</b></summary>

- Try changing the detection language to match your subtitles
- Use "Set Region" to manually define the subtitle area
- Install PaddleOCR for best detection accuracy

</details>

<details>
<summary><b>Application won't start</b></summary>

- Ensure Python 3.11+ is installed; use Python 3.12 or 3.13 for NVIDIA CUDA
- Re-run a launcher to auto-repair a missing or broken `venv`, or run
  `python setup.py --repair` from the repo root for the same unattended repair
- Try `Run_VSR_Pro_Debug.bat` to keep the console open during startup, or
  `Run_VSR_Pro.ps1` from PowerShell to see setup/launch errors there
- Check the log file: `%APPDATA%\VideoSubtitleRemoverPro\vsr_pro.log`
- If the log or support bundle reports OpenCV's bundled libpng below
  `1.6.54`, upgrade to the reviewed `opencv-python>=5.0.0.93` wheel before
  opening untrusted PNG files or producing a release
- If self-test, backend status, or a support bundle reports multiple OpenCV
  wheels, run the printed `pip uninstall` command for every OpenCV variant,
  then reinstall one wheel, normally `opencv-python>=5.0.0.93`

</details>

### Log Files

- GUI activity panel (open it from the footer, then click "Open Log File" for
  the full log)
- File log: `%APPDATA%\VideoSubtitleRemoverPro\vsr_pro.log` (5MB rotating)
- About -> Support bundle saves a redacted `.zip` with runtime facts,
  dependency versions, settings summary, recent log lines, and batch report
  evidence, including per-stage timing summaries. CLI equivalent:
  `python -m backend.cli --support-bundle support.zip`
- About -> Model cache can export/import a portable cache bundle. CLI
  equivalents: `python -m backend.cli --model-cache-export models.zip`
  and `python -m backend.cli --model-cache-import models.zip`

## Project Structure

```
VideoSubtitleRemover/
|-- VideoSubtitleRemover.py   # Main GUI application
|-- Dockerfile                # Local CPU-only smoke container recipe
|-- .dockerignore             # Excludes build outputs, models, and venvs
|-- gui/
|   |-- app.py                # Tk construction and controller composition root
|   |-- region_controller.py  # Timed region/keyframe editor workflow
|   |-- settings_controller.py # Presets and detailed-settings behavior
|   |-- mask_correction_controller.py # Quality-directed mask corrections
|   |-- processing_controller.py  # Queue worker, pause/stop, reports, notify
|   |-- preview_controller.py     # Preview, A/B compare, inline region editor
|   |-- quality_controller.py     # Quality review, retry, report helpers
|   |-- support_controller.py     # Support bundle, model cache, About panels
|   |-- widgets.py            # Custom Tk controls
|   |-- config.py             # GUI config, queue state, presets
|   `-- theme.py              # Design tokens
|-- backend/
|   |-- __init__.py           # Module exports
|   |-- processor.py          # Legacy import/CLI compatibility shim
|   |-- detection.py          # OCR cascade and detector routing
|   |-- tracking.py           # Kalman, pHash, karaoke helpers
|   |-- io.py                 # Capture, ffprobe, intermediate writers
|   |-- cli.py                # Command-line entry point
|   |-- resume_checkpoint.py  # Durable pause/resume checkpoint helpers
|   |-- inpainters/           # Built-in STTN/LaMa/ProPainter/AUTO paths
|   |-- presets.py            # Shared preset library (GUI + CLI)
|   |-- adapter_manifest.py   # Optional model provenance and hash policy
|   `-- model_hashes.py       # Vendored SHA-256 weight hashes
|-- docs/
|   |-- architecture.md       # Pipeline map for new contributors
|   |-- edge_case_corpus.md   # Community regression-corpus guide
|   `-- archive/              # Retired audits and completed checklists
|-- ROADMAP.md                # Active incomplete work
|-- RESEARCH.md               # Current research synthesis
|-- setup.py                  # First-time environment setup
|-- Run_VSR_Pro.bat           # Windows launcher
|-- Run_VSR_Pro_Debug.bat     # Windows launcher with a visible console
|-- Run_VSR_Pro.ps1           # PowerShell launcher
|-- build_exe.bat             # PyInstaller build script
|-- requirements.txt          # Python dependencies
|-- tests/                    # Focused regression coverage for hardened paths
|-- tools/                    # Local developer smoke helpers
|-- .github/                  # Issue templates
|-- assets/                   # Application assets
|-- models/                   # AI model weights (auto-downloaded)
`-- output/                   # Default output location
```

See [docs/architecture.md](docs/architecture.md) for a walkthrough of
the detect -> tracker -> mask -> TBE -> refine -> mux pipeline and the
"add a new feature" checklist.

Planning entry points:
[ROADMAP.md](ROADMAP.md) for active incomplete work and
[RESEARCH.md](RESEARCH.md) for current research synthesis. Retired audits
and completed checklists live under [docs/archive/](docs/archive/).

## Credits

- Original project: [YaoFANGUK/video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover)
- LaMa inpainting: [simple-lama-inpainting](https://github.com/enesmsahin/simple-lama-inpainting)
- EasyOCR: [JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR)
- STTN: [Learning Joint Spatial-Temporal Transformations](https://arxiv.org/abs/2007.10247)
- ProPainter (research reference): [sczhou/ProPainter](https://github.com/sczhou/ProPainter) -- VSR's "ProPainter" mode is a TBE + LaMa hybrid inspired by the concept; it does not use the upstream ProPainter code or weights

## License

This project is licensed under the MIT License.

---

<div align="center">

**Video Subtitle Remover Pro** -- Built by SysAdminDoc

[Report Bug](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues) | [Request Feature](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues)

</div>
