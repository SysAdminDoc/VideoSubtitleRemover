<!-- codex-branding:start -->
<p align="center"><img src="icon.png" width="128" alt="Video Subtitle Remover"></p>

<p align="center">
  <img alt="Version" src="https://img.shields.io/badge/version-preview-58A6FF?style=for-the-badge">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-4ade80?style=for-the-badge">
  <img alt="Platform" src="https://img.shields.io/badge/platform-Python%20GUI-58A6FF?style=for-the-badge">
</p>
<!-- codex-branding:end -->

# Video Subtitle Remover Pro

<div align="center">

![Version](https://img.shields.io/badge/version-3.7.0-22c55e)
![Platform](https://img.shields.io/badge/platform-Windows-60a5fa)
![License](https://img.shields.io/badge/license-Apache%202.0-red)
![Python](https://img.shields.io/badge/python-3.10+-blue)

**Professional AI-powered tool for removing hard-coded subtitles from videos and images**

[Features](#features) | [Installation](#installation) | [Usage](#usage) | [Configuration](#configuration) | [CLI](#cli-usage) | [Troubleshooting](#troubleshooting)

</div>

---

## Overview

Video Subtitle Remover Pro uses real AI neural networks to remove hard-coded subtitles and text watermarks from videos and images. Unlike simple blur or crop methods, it intelligently fills in removed areas with content that matches the surrounding video.

Based on [YaoFANGUK/video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover), enhanced with a professional interface, real LaMa inpainting, multi-engine detection, and 12-language support.

## Features

- **Real Video Inpainting** — Temporal Background Exposure (TBE) reconstructs the true background from neighbouring frames where the subtitle is absent. No external model weight downloads required.
- **Real AI Inpainting** — LaMa neural network for still-frame and residual refinement (via `simple-lama-inpainting`)
- **Multi-Engine Detection** — RapidOCR (ONNX PP-OCR, 4-5x faster, leak-free) > PaddleOCR > Surya > EasyOCR > OpenCV fallback chain (automatic)
- **Seamless Boundaries** — Gaussian alpha feathering at every inpaint boundary, no visible cut lines
- **12 Language Support** — English, Chinese, Japanese, Korean, French, German, Spanish, Portuguese, Russian, Arabic, Hindi, Italian
- **GPU Acceleration** — NVIDIA CUDA, AMD/Intel DirectML, and CPU fallback
- **Subtitle Region Selector** — Draw a rectangle on the first frame to target specific areas
- **Batch Processing** — Queue files or drag entire folders for automated processing
- **Before/After Preview** — Side-by-side comparison of completed items
- **Premium Dark UI** — Cohesive design system with custom sliders, toggles, and status chips
- **Audio Preservation** — Automatically preserves original audio via FFmpeg
- **Settings Persistence** — All settings saved/restored between sessions
- **CI/CD Releases** — Automated Windows builds via GitHub Actions

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 | Windows 11 |
| CPU | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| RAM | 8 GB | 16+ GB |
| GPU | Any (CPU mode) | NVIDIA RTX 2060+ |
| VRAM | - | 6+ GB |
| Python | 3.10 | 3.12 |

## Installation

### Quick Install

1. **Download** or clone this repository
2. **Double-click** `Run_VSR_Pro.bat` — first run automatically:
   - Creates a virtual environment
   - Detects your GPU and installs appropriate packages
   - Installs PaddleOCR, EasyOCR, and LaMa inpainting
   - Launches the application

### Manual Install

```powershell
cd VideoSubtitleRemover

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install PyTorch (choose one):
# NVIDIA:
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu118
# CPU:
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
pip install -r requirements.txt

# Run
python VideoSubtitleRemover.py
```

### FFmpeg (Required for audio)

```powershell
winget install ffmpeg
```

## Usage

1. **Launch** via `Run_VSR_Pro.bat`
2. **Add files** — Click to browse, right-click for folders, or drag & drop
3. **Select algorithm** — LAMA (recommended), STTN, or ProPainter
4. **Set language** if subtitles are non-English
5. **Optionally set region** — Click "Set Region" to draw a rectangle on the subtitle area
6. **Start Processing** and monitor progress
7. **Click filename** to preview, **double-click completed item** to open output

### Algorithm Comparison

| Algorithm | Inpainting Engine | Speed | Quality | Best For |
|-----------|-------------------|-------|---------|----------|
| **STTN** | Temporal Background Exposure | Fastest | Great | Live-action video with changing subtitles (default) |
| LAMA | Neural (LaMa) | Medium | Best still-frame | Images, animations, static backgrounds |
| ProPainter | TBE + LaMa refinement | Slowest | Best motion | Motion-heavy footage, thick/decorative text |

> All three modes now do real inpainting. STTN recovers the literal background from adjacent frames where the subtitle is absent -- this works because hard-coded subtitles are sparse in time, and the pixels behind them are revealed whenever the text changes or disappears. LAMA is a single-frame neural fill. ProPainter is a hybrid: TBE reconstructs the background, then LaMa refines any residual.

### Detection Engines

The app automatically selects the best available engine:

| Priority | Engine | Install | Languages | Notes |
|----------|--------|---------|-----------|-------|
| 1 | **RapidOCR** (ONNX PP-OCR) | `pip install rapidocr` | 100+ | 4-5x faster than PaddleOCR, leak-free (default) |
| 2 | PaddleOCR (PP-OCRv5) | `pip install paddleocr>=3.0.0` | 106 | High accuracy reference implementation |
| 3 | Surya | `pip install surya-ocr` | 90+ | Layout-aware (GPL) |
| 4 | EasyOCR | `pip install easyocr` | 80+ | Legacy fallback |
| 5 | OpenCV fallback | Built-in | Any | Threshold-based |

## CLI Usage

Process files from the command line:

```bash
python -m backend.processor -i input.mp4 -o output.mp4 -m lama --lang en --crf 20
```

| Flag | Description | Default |
|------|-------------|---------|
| `-i`, `--input` | Input file path | Required |
| `-o`, `--output` | Output file path | Required |
| `-m`, `--mode` | Algorithm (sttn/lama/propainter) | sttn |
| `-g`, `--gpu` | GPU device ID (-1 for CPU) | 0 |
| `-l`, `--lang` | Detection language | en |
| `--crf` | Output quality (15-35, lower=better) | 23 |
| `--skip-detection` | Use manual region only | Off |
| `--fast` | LAMA fast mode | Off |
| `--no-audio` | Strip audio | Off |
| `--frame-skip N` | Reuse mask for N frames (0=every frame) | 0 |
| `--mask-dilate N` | Expand masks by N pixels | 8 |
| `--no-hw-encode` | Force software encoding (libx264) | Off |

## Configuration

Settings are stored in `%APPDATA%\VideoSubtitleRemoverPro\settings.json` and persist across sessions.

### Advanced Settings

| Setting | Description | Default | Range |
|---------|-------------|---------|-------|
| Neighbor Stride | STTN temporal window | 10 | 5-30 |
| Reference Length | STTN reference frames | 10 | 5-30 |
| Max Load Frames | Batch size | 30 | 10-100 |
| CRF Quality | Output quality (lower=better) | 23 | 15-35 |
| Frame Skip | Reuse detection mask for N frames | 0 | 0-10 |
| Mask Dilate | Expand detected regions (px) | 8 | 0-20 |
| Mask Feather | Soft alpha-blend at boundary (px) | 4 | 0-15 |
| TBE Coverage | Min frames a pixel must be unmasked to trust its exposure | 3 | 1-10 |
| HW Encoding | Use NVENC/QSV/AMF if available | On | On/Off |

## Troubleshooting

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

- Ensure Python 3.10+ is installed
- Delete `venv` folder and re-run setup
- Check the log file: `%APPDATA%\VideoSubtitleRemoverPro\vsr_pro.log`

</details>

### Log Files

- GUI log panel (collapsible, click "Open Log File" for full log)
- File log: `%APPDATA%\VideoSubtitleRemoverPro\vsr_pro.log` (5MB rotating)

## Project Structure

```
VideoSubtitleRemover/
├── VideoSubtitleRemover.py   # Main GUI application
├── backend/
│   ├── __init__.py           # Module exports
│   └── processor.py          # Core processing (detection + inpainting)
├── setup.py                  # First-time environment setup
├── Run_VSR_Pro.bat           # Windows launcher
├── build_exe.bat             # PyInstaller build script
├── requirements.txt          # Python dependencies
├── .github/workflows/
│   └── build.yml             # CI/CD release workflow
├── assets/                   # Application assets
├── models/                   # AI model weights (auto-downloaded)
└── output/                   # Default output location
```

## Credits

- Original project: [YaoFANGUK/video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover)
- LaMa inpainting: [simple-lama-inpainting](https://github.com/enesmsahin/simple-lama-inpainting)
- EasyOCR: [JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR)
- STTN: [Learning Joint Spatial-Temporal Transformations](https://arxiv.org/abs/2007.10247)
- ProPainter: [sczhou/ProPainter](https://github.com/sczhou/ProPainter)

## License

This project is licensed under the Apache License 2.0.

---

<div align="center">

**Video Subtitle Remover Pro** -- Built by SysAdminDoc

[Report Bug](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues) | [Request Feature](https://github.com/SysAdminDoc/VideoSubtitleRemover/issues)

</div>
