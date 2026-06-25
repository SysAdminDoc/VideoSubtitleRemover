

# Video Subtitle Remover Pro

<div align="center">

![Version](https://img.shields.io/badge/version-3.17.1-22c55e)
![Platform](https://img.shields.io/badge/platform-Windows-60a5fa)
![License](https://img.shields.io/badge/license-MIT-4ade80)
![Python](https://img.shields.io/badge/python-3.10--3.13%20CUDA-blue)

**Professional AI-powered tool for removing hard-coded subtitles from videos and images**

[Features](#features) | [Installation](#installation) | [Usage](#usage) | [Configuration](#configuration) | [CLI](#cli-usage) | [Troubleshooting](#troubleshooting)

</div>

---

## Overview

Video Subtitle Remover Pro uses real AI neural networks to remove hard-coded subtitles and text watermarks from videos and images. Unlike simple blur or crop methods, it intelligently fills in removed areas with content that matches the surrounding video.

Based on [YaoFANGUK/video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover), enhanced with a professional interface, real LaMa inpainting, multi-engine detection, and 12-language support.

## Features

- **Real Video Inpainting** -- Temporal Background Exposure (TBE) reconstructs the true background from neighbouring frames where the subtitle is absent. No external model weight downloads required.
- **Real AI Inpainting** -- LaMa neural network via ONNX Runtime (default, no torch dependency) or simple-lama-inpainting (PyTorch fallback)
- **AUTO Inpaint Routing** -- Per-batch routing between TBE and LaMa based on exposure score
- **Multi-Engine Detection** -- RapidOCR (ONNX PP-OCR, 4-5x faster, leak-free) > PaddleOCR > Surya (GPL opt-in) > EasyOCR > OpenCV fallback chain (automatic)
- **Lossless Pipeline** -- FFV1 lossless intermediate (only the final encode is lossy) for noticeably cleaner outputs than the legacy mp4v intermediate
- **Modern Codec Output** -- Pick H.264 / H.265 / AV1 / VVC (H.266) from a dropdown; NVENC/QSV/AMF where available, libx265 / libsvtav1 software fallback, and VVC when FFmpeg exposes `libvvenc`
- **Multi-region Masks** -- Draw multiple subtitle rects on a scrubbable video frame; backend honours every rect
- **Inpaint Preview** -- "Test cleanup" runs detect + inpaint on the selected frame so you can A/B settings before committing
- **Seamless Boundaries** -- Gaussian alpha feathering at every inpaint boundary, no visible cut lines
- **~50 Language Support** -- English / Chinese / Japanese / Korean / European, plus Thai, Vietnamese, Polish, Greek, Ukrainian, Filipino, Hebrew, Czech, and more
- **GPU Acceleration** -- NVIDIA CUDA, AMD/Intel DirectML through ONNX Runtime, hardware-decode hints (D3D11 / VAAPI / MFX), CPU fallback
- **Subtitle Region Selector** -- Scrub to any frame and draw one or more rectangles
- **Batch Processing** -- Queue files or drag entire folders; per-item cancellation
- **Multi-track Audio + Loudness Normalisation** -- Pass through every audio track on Bluray rips; optional per-stream EBU R128 normalisation to LUFS targets (YouTube -14, Apple -16, broadcast -23)
- **Quality Self-Test** -- PSNR / SSIM report, optional FFmpeg/libvmaf VMAF score, ROI-cropped metrics for the inpaint region, and an optional side-by-side comparison PNG
- **CLI + Presets** -- `python -m backend.processor --pattern ... --preset "YouTube (default)"`; six built-in presets + user presets persisted to `%APPDATA%`
- **Chyron vs Subtitle Filter** -- Keep persistent text (logos, lower-thirds) and remove dialogue, or vice versa
- **Karaoke Grouping** -- Per-syllable boxes fuse into a single line mask so highlighted lyrics do not leak through the gaps
- **Live Preview During Processing** -- 15 FPS throttled preview piped from the backend worker
- **Pre-batch ETA Estimate** -- 30-frame detect probe seeds the ETA so users see "about X left" from the very first frame
- **Crash-Resume Checkpointing** -- SHA-256 input fingerprint per file; re-running a glob skips finished work
- **Premium Dark UI** -- Cohesive design system with custom controls, rectangular status tiles, responsive workbench scrolling, taskbar progress, and onboarding
- **Settings Persistence** -- All knobs saved/restored between sessions; versioned schema with backfill migration
- **CI/CD Releases** -- Automated Windows builds via GitHub Actions, pip-audit scan, strict artifact/version/dependency verification, and winget submission support

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

# Install PyTorch (choose one -- torch 2.7+ supports Python 3.9-3.13):
# NVIDIA RTX 20/30/40-series (Turing/Ampere/Ada):
pip install torch>=2.10.0 torchvision>=0.25.0 --index-url https://download.pytorch.org/whl/cu118
# NVIDIA RTX 50-series (Blackwell -- 5070/5080/5090, needs CUDA 12.8):
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

### Validation

```powershell
python -m unittest discover -s tests -v
```

## Usage

1. **Launch** via `Run_VSR_Pro.bat`, `Run_VSR_Pro_Debug.bat`, or
   `Run_VSR_Pro.ps1`
2. **Add files** -- Click to browse, right-click for folders, or drag & drop
3. **Select algorithm** — LAMA (recommended), STTN, or ProPainter
4. **Set language** if subtitles are non-English
5. **Optionally set region** — Click "Set Region" to draw a rectangle on the subtitle area
6. **Start Processing** and monitor progress
7. **Select a queue item** to preview it, use **Review mask** to confirm detection, and **double-click the preview** for a larger source frame

### Algorithm Comparison

| Algorithm | Inpainting Engine | Speed | Quality | Best For |
|-----------|-------------------|-------|---------|----------|
| **STTN** | Temporal Background Exposure | Fastest | Great | Live-action video with changing subtitles (default) |
| LAMA | Neural (LaMa ONNX or PyTorch) | Medium | Best still-frame | Images, animations, static backgrounds |
| ProPainter | TBE + LaMa refinement | Slowest | Best motion | Motion-heavy footage, thick/decorative text |

> All three modes now do real inpainting. STTN recovers the literal background from adjacent frames where the subtitle is absent -- this works because hard-coded subtitles are sparse in time, and the pixels behind them are revealed whenever the text changes or disappears. LAMA is a single-frame neural fill. ProPainter is a TBE + LaMa refinement hybrid -- it is **not** the ICCV 2023 ProPainter model or weights (which carry a non-commercial NTU S-Lab license). This implementation uses only MIT-licensed code.

### Detection Engines

The app automatically selects the best available engine:

| Priority | Engine | Install | Languages | Notes |
|----------|--------|---------|-----------|-------|
| 1 | **RapidOCR** (ONNX PP-OCR) | `pip install "rapidocr>=2.0.0,<4.0.0"` | 100+ | 4-5x faster than PaddleOCR, leak-free (default) |
| 2 | PaddleOCR (3.x, PP-OCRv6 default in 3.7) | `pip install "paddleocr>=3.0.0,<4.0.0"` | 106 | High accuracy reference implementation; PP-OCRv5/v6 result payloads are supported |
| 3 | Surya | `pip install surya-ocr` | 90+ | Layout-aware (GPL) |
| 4 | EasyOCR | `pip install easyocr` | 80+ | Legacy fallback |
| 5 | OpenCV fallback | Built-in | Any | Threshold-based |

On AMD/Intel systems, setup installs `onnxruntime-directml`. When ONNX
Runtime reports `DmlExecutionProvider`, RapidOCR is initialized with its
DirectML provider settings; unsupported RapidOCR versions or missing
providers fall back to CPU automatically. RapidOCR legacy tuple output and
current structured object/dict output are both normalized to the same
axis-aligned detector boxes. Opt-in ONNX inpainters inspect
their model `opset_import` metadata before creating a DirectML session; if
the default ONNX opset is newer than DirectML's supported ceiling, VSR uses
the CPU provider instead of failing at session creation.

Optional model paths such as `VSR_LAMA_ONNX`, `VSR_MIGAN_ONNX`,
`VSR_FASTDVDNET`, and `VSR_TRANSNETV2` are checked against a local adapter
manifest before loading. Known SHA-256 mismatches fall back instead of
deserializing the file. Legacy adapters without a pinned hash still run, but
new strict adapters can require a known hash unless
`VSR_ALLOW_UNVERIFIED_MODELS=1` is set and recorded in release evidence.

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
data, planned action, final status, and elapsed time for skipped,
checkpointed, remuxed, processed, or failed files. When quality reports are
enabled, batch summaries also include a `passed`, `review`, or `unknown`
quality gate using ROI metrics, a cheap residual-text score, and an
adjacent-frame temporal flicker score, plus any quality-sheet preview path for
review-needed outputs. A failed gate changes the batch row status to
`review-needed`; skipped and remux-only rows are marked `not_applicable`.

| Flag | Description | Default |
|------|-------------|---------|
| `-i`, `--input` | Input file path | Required |
| `-o`, `--output` | Output file path | Required |
| `--pattern` | Glob pattern for batch (e.g. `inputs/*.mp4`) | - |
| `--out-dir` | Output directory for batch mode | - |
| `--config` | JSON config overlay | - |
| `--preset NAME` | Apply a built-in or user preset by name | - |
| `--list-presets` | List every preset and exit | - |
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
| `--mask-dilate N` | Expand masks by N pixels | 8 |
| `--no-hw-encode` | Force software encoding | Off |
| `--decode-accel` | HW decode hint (off/auto/d3d11/vaapi/mfx) | off |
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
| `--audit-onnx` | Audit all ONNX models for DirectML opset compatibility and exit | Off |
| `--scan-weights` | Scan cached model weights and verify SHA-256 against known hashes | Off |
| `--cache-info` | Print cache directory inventory with sizes and exit | Off |
| `--cache-clean` | Remove stale cache entries (checkpoints, proxies, TRT engines) | Off |
| `--support-bundle PATH` | Write a redacted diagnostics zip and exit | - |
| `--validate-config` | Print resolved config and exit | Off |
| `--skip-existing` | Skip files whose output already exists | Off |
| `--no-prefetch` | Disable worker-thread frame prefetcher | Off |
| `--output-frames` | Write cleaned frames as individual PNGs instead of a video | Off |
| `--json-log PATH` | Append a structured JSON-line log | - |

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
| HW Decode Hint | cv2 HW-accel hint with software fallback | off | off/auto/d3d11/vaapi/mfx |
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

Note: the internal pixel pipeline is still 8-bit BGR, so true 10-bit HDR
sources are tone-mapped to SDR (the output is tagged correctly but not
10-bit). For standard SDR limited-range content, colors are preserved. If
you still see a mismatch, attach the `ffprobe` color fields of your source
to a bug report.

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
- Delete `venv` folder and re-run setup
- Try `Run_VSR_Pro_Debug.bat` to keep the console open during startup, or
  `Run_VSR_Pro.ps1` from PowerShell to see setup/launch errors there
- Check the log file: `%APPDATA%\VideoSubtitleRemoverPro\vsr_pro.log`
- If the log warns that OpenCV bundles libpng older than `1.6.54`, avoid
  opening untrusted PNG files until `opencv-python` ships a fixed wheel

</details>

### Log Files

- GUI log panel (collapsible, click "Open Log File" for full log)
- File log: `%APPDATA%\VideoSubtitleRemoverPro\vsr_pro.log` (5MB rotating)
- About -> Support bundle saves a redacted `.zip` with runtime facts,
  dependency versions, settings summary, recent log lines, and batch report
  evidence. CLI equivalent:
  `python -m backend.cli --support-bundle support.zip`

## Project Structure

```
VideoSubtitleRemover/
|-- VideoSubtitleRemover.py   # Main GUI application
|-- backend/
|   |-- __init__.py           # Module exports
|   |-- processor.py          # Legacy import/CLI compatibility shim
|   |-- detection.py          # OCR cascade and detector routing
|   |-- tracking.py           # Kalman, pHash, karaoke helpers
|   |-- io.py                 # Capture, ffprobe, intermediate writers
|   |-- cli.py                # Command-line entry point
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
|-- .github/workflows/
|   `-- build.yml             # CI/CD release workflow + pip-audit
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
