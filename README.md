# Video Subtitle Remover Pro

<div align="center">

![Version](https://img.shields.io/badge/version-2.0.0-22c55e)
![Platform](https://img.shields.io/badge/platform-Windows-60a5fa)
![License](https://img.shields.io/badge/license-Apache%202.0-red)
![Python](https://img.shields.io/badge/python-3.10+-blue)

**Professional AI-powered tool for removing hard-coded subtitles from videos and images**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Configuration](#configuration) • [Troubleshooting](#troubleshooting)

</div>

---

## 📋 Overview

Video Subtitle Remover Pro is a professional Windows application that uses advanced AI algorithms to remove hard-coded subtitles and text watermarks from videos and images. Unlike simple blur or overlay methods, it intelligently fills in the removed areas with content that seamlessly matches the surrounding video.

Based on [YaoFANGUK/video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover), this version provides a polished, professional interface with enhanced features for Windows users.

## ✨ Features

- **🎯 AI-Powered Removal**: Uses advanced inpainting algorithms (STTN, LAMA, ProPainter) for high-quality subtitle removal
- **📹 Video & Image Support**: Process MP4, AVI, MKV, MOV, PNG, JPG, and more
- **🖥️ GPU Acceleration**: Supports NVIDIA CUDA, AMD/Intel DirectML, and CPU fallback
- **📂 Batch Processing**: Queue multiple files for automated processing
- **🎨 Modern Dark UI**: Professional interface with your preferred color scheme
- **⚙️ Customizable Settings**: Fine-tune algorithm parameters for optimal results
- **🔊 Audio Preservation**: Automatically preserves original audio in output videos
- **📊 Real-time Progress**: Track processing status with detailed feedback

## 💻 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 | Windows 11 |
| CPU | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| RAM | 8 GB | 16+ GB |
| GPU | Any (CPU mode) | NVIDIA RTX 2060+ |
| VRAM | - | 6+ GB |
| Storage | 2 GB | 10+ GB |
| Python | 3.10 | 3.12 |

### GPU Support

| GPU Type | Acceleration | Notes |
|----------|--------------|-------|
| NVIDIA (RTX 10xx-50xx) | CUDA | Best performance |
| AMD Radeon | DirectML | Good performance |
| Intel Arc/Iris | DirectML | Good performance |
| Integrated | CPU | Slower but works |

## 📦 Installation

### Quick Install

1. **Download** the latest release or clone this repository
2. **Double-click** `Run_VSR_Pro.bat` - it will automatically:
   - Create a Python virtual environment
   - Detect your GPU and install appropriate packages
   - Install all dependencies
   - Launch the application

### Manual Install

```powershell
# Clone or download the repository
cd video-subtitle-remover-pro

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install PyTorch (choose one):

# For NVIDIA GPUs:
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu118

# For AMD/Intel GPUs:
pip install torch==2.4.1 torchvision==0.19.1
pip install torch-directml==0.2.5.dev240914

# For CPU only:
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt

# Run the application
python VideoSubtitleRemover.py
```

### FFmpeg (Required for audio)

To preserve audio in processed videos, install FFmpeg:

```powershell
# Using winget (Windows 10+):
winget install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

## 🚀 Usage

### Basic Workflow

1. **Launch** the application via `Run_VSR_Pro.bat`
2. **Add files** by drag-and-drop or clicking the browse area
3. **Select algorithm** based on your content type:
   - **STTN**: Best for real-world videos (fast)
   - **LAMA**: Best for images and animations (quality)
   - **ProPainter**: Best for high-motion videos (slow)
4. **Configure settings** as needed
5. **Click "Start Processing"** and wait for completion
6. **Find outputs** in the `output` folder next to your source files

### Supported Formats

| Type | Input Formats |
|------|--------------|
| Video | MP4, AVI, MKV, MOV, WMV, FLV, WebM, M4V, MPEG |
| Image | JPG, JPEG, PNG, BMP, TIFF, WebP |

## ⚙️ Configuration

### Algorithm Comparison

| Algorithm | Speed | Quality | VRAM Usage | Best For |
|-----------|-------|---------|------------|----------|
| STTN | ⚡ Fast | ★★★★☆ | Moderate | Real-world videos |
| LAMA | 🔄 Medium | ★★★★★ | Low | Images, animations |
| ProPainter | 🐢 Slow | ★★★★★ | High | Motion-heavy videos |

### STTN Settings

| Setting | Description | Default | Range |
|---------|-------------|---------|-------|
| Skip Detection | Use manual region (faster) | Off | On/Off |
| Neighbor Stride | Temporal window size | 10 | 5-30 |
| Reference Length | Reference frame count | 10 | 5-30 |
| Max Load Frames | Batch size | 30 | 10-100 |

### Tips for Best Results

1. **For subtitles at the bottom**: Use STTN with skip detection enabled
2. **For anime/animation**: Use LAMA for best quality
3. **For fast-moving scenes**: Use ProPainter (requires high VRAM)
4. **For batch processing**: Use STTN with skip detection for speed
5. **For images**: LAMA provides the best single-frame results

## 🔧 Troubleshooting

### Common Issues

<details>
<summary><b>CUDA out of memory</b></summary>

- Reduce `Max Load Frames` in STTN settings
- Switch to LAMA mode (lower VRAM usage)
- Close other GPU-intensive applications
- Use CPU mode as fallback

</details>

<details>
<summary><b>Processing is slow</b></summary>

- Enable GPU acceleration if available
- Use STTN with `Skip Detection` enabled
- For LAMA, enable `Super Fast` mode
- Reduce video resolution before processing

</details>

<details>
<summary><b>Poor removal quality</b></summary>

- Try different algorithms (STTN → LAMA → ProPainter)
- Increase STTN parameters (Neighbor Stride, Reference Length)
- Disable LAMA Super Fast mode
- Ensure subtitle region is properly detected

</details>

<details>
<summary><b>No audio in output</b></summary>

- Install FFmpeg: `winget install ffmpeg`
- Ensure FFmpeg is in PATH
- Enable "Preserve original audio" option

</details>

<details>
<summary><b>Application won't start</b></summary>

- Ensure Python 3.10+ is installed
- Delete `venv` folder and re-run setup
- Check Windows Defender isn't blocking Python
- Run as Administrator

</details>

### Error Logs

Logs are displayed in the console window. For detailed debugging:

```powershell
# Run with verbose logging
python VideoSubtitleRemover.py --verbose 2>&1 | Tee-Object -FilePath debug.log
```

## 📁 Project Structure

```
video-subtitle-remover-pro/
├── VideoSubtitleRemover.py   # Main GUI application
├── setup.py                  # Installation script
├── Run_VSR_Pro.bat          # Windows launcher
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── backend/
│   └── processor.py          # Core processing logic
├── assets/                   # Application assets
├── models/                   # AI model weights
└── output/                   # Default output location
```

## 🤝 Credits

- Original project: [YaoFANGUK/video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover)
- STTN algorithm: [STTN Paper](https://arxiv.org/abs/2007.10247)
- LAMA algorithm: [LAMA Paper](https://arxiv.org/abs/2109.07161)
- ProPainter algorithm: [ProPainter Paper](https://arxiv.org/abs/2309.03897)

## 📄 License

This project is licensed under the Apache License 2.0 - see the original repository for details.

---

<div align="center">

**Video Subtitle Remover Pro** • Built with 💚 by Maven Imaging Tools

[Report Bug](https://github.com/YaoFANGUK/video-subtitle-remover/issues) • [Request Feature](https://github.com/YaoFANGUK/video-subtitle-remover/issues)

</div>
