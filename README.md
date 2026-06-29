

# Video Subtitle Remover Pro

<div align="center">

![Version](https://img.shields.io/badge/version-3.17.3-22c55e)
![Platform](https://img.shields.io/badge/platform-Windows-60a5fa)
![License](https://img.shields.io/badge/license-MIT-4ade80)
![Python](https://img.shields.io/badge/python-3.10--3.13%20CUDA-blue)

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
- **AUTO Inpaint Routing** -- Per-batch routing between TBE and LaMa based on exposure score
- **Multi-Engine Detection** -- RapidOCR (ONNX PP-OCR, 4-5x faster, leak-free) > PaddleOCR > Surya (GPL opt-in) > EasyOCR > OpenCV fallback chain (automatic)
- **Lossless Pipeline** -- FFV1 lossless intermediate (only the final encode is lossy) for noticeably cleaner outputs than the legacy mp4v intermediate
- **Modern Codec Output** -- Pick H.264 / H.265 / AV1 / VVC (H.266) from a dropdown; NVENC/QSV/AMF where available, libx265 / libsvtav1 software fallback, native SVT-AV1 film grain, and VVC when FFmpeg exposes `libvvenc`
- **Multi-region Masks** -- Draw multiple subtitle rects on a scrubbable video frame, optionally with start/end seconds for moving subtitle layouts
- **Inpaint Preview** -- "Test cleanup" runs detect + inpaint on the selected frame so you can A/B settings before committing
- **Seamless Boundaries** -- Gaussian alpha feathering at every inpaint boundary, no visible cut lines
- **Language Support** -- 52 selectable OCR language codes in the GUI, with installed OCR engines reporting broader capacity: RapidOCR 100+, PaddleOCR 106, Surya 90+ (GPL opt-in), and EasyOCR 80+; core GUI surfaces are wired for gettext catalogs dropped into `locale/<lang>/LC_MESSAGES/vsr.mo`
- **GPU Acceleration** -- NVIDIA CUDA, AMD/Intel DirectML through ONNX Runtime, hardware-decode hints (D3D11 / VAAPI / MFX), CPU fallback
- **Subtitle Region Selector** -- Scrub to any frame and draw one or more rectangles; use optional start/end seconds to save time-ranged manual masks
- **Batch Processing** -- Queue files or drag entire folders; per-item cancellation plus safe pause/resume for long videos
- **Multi-track Audio + Loudness Normalisation** -- Pass through every audio track on Bluray rips; optional per-stream EBU R128 normalisation to LUFS targets (YouTube -14, Apple -16, broadcast -23)
- **Quality Self-Test** -- PSNR / SSIM report, optional FFmpeg/libvmaf VMAF score, ROI-cropped metrics for the inpaint region, and an optional side-by-side comparison PNG
- **CLI + Presets** -- `python -m backend.processor --pattern ... --preset "YouTube (default)"`; six built-in presets + user presets persisted to `%APPDATA%`
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
| Python | 3.10 | 3.12 or 3.13 for CUDA |

## Installation

### Quick Install

1. **Download** or clone this repository
2. **Double-click** `Run_VSR_Pro.bat` — first run automatically:
   - Creates a virtual environment
   - Detects your GPU and installs appropriate packages
   - Installs PaddleOCR, EasyOCR, and LaMa inpainting
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

# Install PyTorch (choose one -- Python 3.12/3.13 recommended for CUDA):
# NVIDIA RTX 20/30/40/50-series:
pip install torch>=2.10.0 torchvision>=0.25.0 --index-url https://download.pytorch.org/whl/cu128
# CPU:
pip install torch>=2.10.0 torchvision>=0.25.0 --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
pip install -r requirements.txt

# Run
python VideoSubtitleRemover.py
```

### FFmpeg (Required for audio)

```powershell
winget install ffmpeg
```

Run `python -m backend.processor --self-test` to confirm the installed build's
`basic`, `advanced_quality`, `speech_fallback`, and `modern_codec` profiles.
Those profiles report missing filters such as `loudnorm`, `libvmaf`, or
`whisper`, missing encoders such as `libvvenc`, and OpenCV wheel ownership
before a long batch starts.

### Validation

```powershell
python -m unittest discover -s tests -v
python -m backend.reference_corpus --json
```

`build_exe.bat` also runs the committed reference corpus during local release
evidence generation and records the result in `release-verification.json`.

## Usage

1. **Launch** via `Run_VSR_Pro.bat`, `Run_VSR_Pro_Debug.bat`, or
   `Run_VSR_Pro.ps1`
2. **Add files** -- Click to browse, right-click for folders, or drag & drop
3. **Select algorithm** — LAMA (recommended), STTN, or ProPainter
4. **Set language** if subtitles are non-English
5. **Optionally set region** — select a queued item and drag on the preview for a fixed subtitle band, or use the settings card's Set Region action for multi-region and timed ranges
6. **Start Processing** and monitor progress
7. **Select a queue item** to preview it, use **Review mask** to confirm detection, and **double-click the preview** for a larger source frame

### Algorithm Comparison

| Algorithm | Inpainting Engine | Speed | Quality | Best For |
|-----------|-------------------|-------|---------|----------|
| **STTN** | Temporal Background Exposure | Fastest | Great | Live-action video with changing subtitles (default) |
| LAMA | Neural (LaMa ONNX/OpenCV DNN; PyTorch opt-in) | Medium | Best still-frame | Images, animations, static backgrounds |
| ProPainter | TBE + LaMa refinement | Slowest | Best motion | Motion-heavy footage, thick/decorative text |

> All three modes now do real inpainting. STTN recovers the literal background from adjacent frames where the subtitle is absent -- this works because hard-coded subtitles are sparse in time, and the pixels behind them are revealed whenever the text changes or disappears. LAMA is a single-frame neural fill. ProPainter is a TBE + LaMa refinement hybrid -- it is **not** the ICCV 2023 ProPainter model or weights (which carry a non-commercial NTU S-Lab license). This implementation uses only MIT-licensed code.

### Detection Engines

The app automatically selects the best available engine:

| Priority | Engine | Install | Languages | Notes |
|----------|--------|---------|-----------|-------|
| 1 | **RapidOCR** (ONNX/OpenVINO PP-OCR) | `pip install "rapidocr>=2.0.0,<4.0.0"`; Intel: `pip install "openvino>=2025.0.0"` | 100+ | ONNX Runtime by default; OpenVINO auto-preferred on CPU/Intel when installed |
| 2 | PaddleOCR (3.x, PP-OCRv6 default in 3.7) | `pip install "paddleocr>=3.0.0,<4.0.0"` | 106 | High accuracy reference implementation; PP-OCRv5/v6 result payloads are supported |
| 3 | Surya | `pip install surya-ocr` | 90+ | Layout-aware (GPL) |
| 4 | EasyOCR | `pip install easyocr` | 80+ | Legacy fallback |
| 5 | OpenCV fallback | Built-in | Any | Threshold-based |

Experimental VLM OCR tiers stay default-off. `VSR_VLM_OCR=florence2`,
`VSR_VLM_OCR=qwen25vl`, and `VSR_VLM_OCR=paddleocr-vl` try the heavier
transformer/PaddleOCR adapters before the table above. For CPU/edge
PaddleOCR-VL-1.5, start a local llama.cpp OpenAI-compatible server with the
GGUF model, then set `VSR_PADDLEOCR_VL=1`; use
`VSR_PADDLEOCR_VL_SERVER_URL` when the server is not at
`http://127.0.0.1:8080/v1`. If the server or PaddleOCRVL entrypoint is not
available, detection falls back to the normal cascade.

On NVIDIA systems, setup installs `onnxruntime-gpu>=1.21.0` for the tested
CUDA 12.x ONNX Runtime path; CUDA 13.x currently requires ONNX Runtime
nightly/custom wheels rather than the stable PyPI default. Backend status and
release evidence distinguish `onnxruntime`, `onnxruntime-gpu`, CUDA package
channel, `onnxruntime-directml`, and the providers reported at runtime. On
AMD/Intel systems, setup installs `onnxruntime-directml`; on Intel systems it
also tries `openvino>=2025.0.0` so RapidOCR can use its OpenVINO engine for
CPU/iGPU OCR acceleration. Set `VSR_RAPIDOCR_ENGINE=onnxruntime` to force the
default ONNX Runtime path or `VSR_RAPIDOCR_ENGINE=openvino` to request
OpenVINO explicitly. When ONNX Runtime reports `DmlExecutionProvider`,
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
blocks unallowed high/critical dependency advisories while keeping the current
OpenCV/libpng exception explicit until fixed wheels are available.
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
co-tracker checkout or `VSR_COTRACKER_REF` to a pinned commit/tag before any
`torch.hub` load is allowed. Set `VSR_COTRACKER_MODE=online` only if you need
the online model; the default uses the offline CoTracker3 entrypoint.
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
crash the GUI process during import. Prefer `VSR_LAMA_ONNX` or
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

Long video runs can pause at safe frame-batch boundaries. In the GUI, click
**Pause batch** while processing; the current video writes checkpoint frames
under `%APPDATA%\VideoSubtitleRemoverPro\checkpoints\` and returns to the queue
as `Paused`. Starting the batch again resumes from the first missing frame. In
the CLI, press Ctrl-C once to request the same safe pause; re-run the same
command to resume. If the input, output path, frame count, frame rate, size, or
processing settings changed, VSR warns and restarts that file from the
beginning instead of trusting stale checkpoint frames.

### Reference Clip Contributions

Use the **Edge-case clip** GitHub issue form before adding real media to
`tests/clips/`. Real fixtures must be short, redistributable with this
MIT-licensed project, and manifest-backed with SHA-256, source URL, license
proof URL, retrieval date, rights confirmation, reproduction settings, and
metric floors. Good starting sources are NASA public-domain media, Library of
Congress public-domain media, Wikimedia Commons compatible-license files, or a
clip you shot and grant as CC0.

| Flag | Description | Default |
|------|-------------|---------|
| `-i`, `--input` | Input file path | Required |
| `-o`, `--output` | Output file path | Required |
| `--pattern` | Glob pattern for batch (e.g. `inputs/*.mp4`) | - |
| `--out-dir` | Output directory for batch mode | - |
| `--config` | JSON config overlay | - |
| `--preset NAME` | Apply a built-in or user preset by name | - |
| `--list-presets` | List every preset and exit | - |
| `--checkpoint-dir` | Directory for done markers and pause/resume checkpoint frames | `%APPDATA%` app cache |
| `--no-resume` | Ignore existing checkpoints and reprocess files; this run still writes new pause checkpoints | Off |
| `-m`, `--mode` | Algorithm (sttn/lama/propainter/auto) | sttn |
| `--codec` | Output codec (h264/h265/av1/vvc; VVC requires FFmpeg with `libvvenc`) | h264 |
| `-g`, `--gpu` | GPU device ID (-1 for CPU) | 0 |
| `-l`, `--lang` | Detection language | en |
| `--crf` | Output quality (15-35, lower=better) | 23 |
| `--skip-detection` | Use manual region only | Off |
| `--fast` | LAMA fast mode | Off |
| `--no-audio` | Strip audio | Off |
| `--single-audio` | Mux only first audio stream | Off |
| `--loudnorm <LUFS>` | EBU R128 loudness target (0 disables) | 0 |
| `--frame-skip N` | Reuse mask for N frames (0=every frame) | 0 |
| `--rife-fast-stride N` | Inpaint keyframes and synthesize skipped frames with Practical-RIFE | 0 |
| `--mask-dilate N` | Expand masks by N pixels | 8 |
| `--no-hw-encode` | Force software encoding | Off |
| `--decode-accel` | HW decode hint (off/auto/d3d11/vaapi/mfx/pynv/nvdec) | off |
| `--keep-chyrons` | Leave persistent text (logos / lower-thirds) | Off |
| `--keep-subtitles` | Leave dialogue subtitles | Off |
| `--karaoke-grouping` | Fuse per-syllable boxes on the same line | Off |
| `--whisper-fallback` | Use Whisper timing to mask OCR-empty speech frames | Off |
| `--whisper-backend` | Whisper backend (`faster-whisper` or `ffmpeg`) | faster-whisper |
| `--whisper-model` | faster-whisper model size | tiny |
| `--ffmpeg-whisper-model` | Local whisper.cpp ggml model for FFmpeg Whisper | - |
| `--ffmpeg-whisper-queue` | FFmpeg whisper queue size in seconds | 3.0 |
| `--soft-subtitle-dry-run` | Print embedded subtitle tracks and planned action without loading OCR | Off |
| `--soft-subtitle-plan-json` | Write soft-subtitle dry-run preflight details as JSON | - |
| `--strip-soft-subtitles` | Stream-copy remux that removes embedded subtitle tracks | Off |
| `--keep-soft-subtitles` | Stream-copy remux that keeps embedded subtitle tracks | Off |
| `--burned-in-only` | Ignore embedded tracks and run visual cleanup normally | Off |
| `--quality-report` | Compute PSNR/SSIM and VMAF when libvmaf is available | Off |
| `--quality-sheet` | Side-by-side comparison PNG | Off |
| `--film-grain STRENGTH` | Add film grain after cleanup; AV1 software output uses SVT-AV1 native grain, other codecs use FFmpeg noise (0..0.5) | 0 |
| `--audit-onnx` | Audit all ONNX models for DirectML opset compatibility and exit | Off |
| `--audit-windows-ml` | Probe Windows ML Python bridge and tiny ONNX smoke inference | Off |
| `--scan-weights` | Scan cached model weights and verify SHA-256 against known hashes | Off |
| `--cache-info` | Print cache directory inventory with sizes and exit | Off |
| `--cache-clean` | Remove stale cache entries (checkpoints, proxies, TRT engines) | Off |
| `--model-cache-export PATH` | Write a portable model-cache zip with SHA-256 manifest | - |
| `--model-cache-import PATH` | Import a verified model-cache zip into the app model cache | - |
| `--support-bundle PATH` | Write a redacted diagnostics zip and exit | - |
| `--validate-config` | Print resolved config and exit | Off |
| `--self-test` | Probe OCR engines, GPU providers, codecs, and FFmpeg capability profiles, then exit | Off |
| `--auto-lang-probe` | Detect subtitle script/language from first frame and exit | Off |
| `--skip-existing` | Skip files whose output already exists | Off |
| `--no-prefetch` | Disable worker-thread frame prefetcher | Off |
| `--output-frames` | Write cleaned frames as individual PNGs instead of a video | Off |
| `--json-log PATH` | Append a structured JSON-line log | - |

`--config` accepts the same manual region schema used by the GUI. Use
`subtitle_area` for one global rectangle, `subtitle_areas` for multiple global
rectangles, or `subtitle_region_spans` for frame-time-specific masks:

```json
{
  "subtitle_region_spans": [
    {"rect": [80, 720, 1180, 820], "start": 0.0, "end": 14.5},
    {"rect": [120, 40, 900, 150], "start": 14.5, "end": 0.0}
  ],
  "sttn_skip_detection": true
}
```

`end: 0.0` means the region stays active through the end of the processed
range. With `sttn_skip_detection` enabled, inactive timed ranges produce an
empty mask instead of reusing a previous manual mask.

## Configuration

Settings are stored in `%APPDATA%\VideoSubtitleRemoverPro\settings.json` and persist across sessions.

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
| TBE Coverage | Min frames a pixel must be unmasked to trust its exposure | 3 | 1-10 |
| HW Encoding | Use NVENC/QSV/AMF if available | On | On/Off |
| HW Decode Hint | OpenCV/PyNvVideoCodec decode hint with software fallback | off | off/auto/d3d11/vaapi/mfx/pynv/nvdec |
| Loudness Target | EBU R128 LUFS target (0 = off) | 0 | 0 or -70..-5 |
| Multi-track Audio | Pass through every audio stream | On | On/Off |
| Quality Sheet | Side-by-side PNG next to output | Off | On/Off |

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

- Ensure Python 3.10+ is installed; use Python 3.12 or 3.13 for NVIDIA CUDA
- Re-run a launcher to auto-repair a missing or broken `venv`, or run
  `python setup.py --repair` from the repo root for the same unattended repair
- Try `Run_VSR_Pro_Debug.bat` to keep the console open during startup, or
  `Run_VSR_Pro.ps1` from PowerShell to see setup/launch errors there
- Check the log file: `%APPDATA%\VideoSubtitleRemoverPro\vsr_pro.log`
- If the log or support bundle reports OpenCV's bundled libpng below
  `1.6.54`, avoid opening untrusted PNG files. As of June 26, 2026,
  opencv-python still needs a fixed bundled-libpng wheel; update this
  guidance only when `security.opencv_libpng.vulnerable` reports `false`
- If self-test, backend status, or a support bundle reports multiple OpenCV
  wheels, run the printed `pip uninstall` command for every OpenCV variant,
  then reinstall one wheel, normally `opencv-python>=4.12.0`

</details>

### Log Files

- GUI log panel (collapsible, click "Open Log File" for full log)
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
|-- gui/
|   |-- app.py                # Main Tk shell and shared UI state
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
