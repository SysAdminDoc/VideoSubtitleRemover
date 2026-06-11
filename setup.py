"""
Video Subtitle Remover Pro - Setup Script
==========================================

This script helps set up the application environment on Windows.
Run: python setup.py
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

# Enable ANSI escape codes on Windows 10+
os.system('')
REQUIREMENTS_FILE = Path("requirements.txt")
PYTHON_CUDA_WHEEL_MAX = (3, 13)
PY314_CPU_OVERRIDE_ENV = "VSR_ALLOW_PY314_CPU"


def _windows_cuda_wheels_unavailable(version=None, system_name=None):
    """Return True when the current Python cannot install Windows CUDA wheels."""
    version = version or sys.version_info
    system_name = system_name or platform.system()
    return (
        system_name == "Windows"
        and (version.major, version.minor) > PYTHON_CUDA_WHEEL_MAX
    )


def _allow_py314_cpu_fallback():
    """Return True when the user explicitly accepts CPU-only setup."""
    return os.environ.get(PY314_CPU_OVERRIDE_ENV, "").strip().lower() in {
        "1", "true", "yes", "cpu"
    }


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_banner():
    """Print setup banner."""
    banner = """
+--------------------------------------------------------------+
|                                                              |
|          VIDEO SUBTITLE REMOVER PRO - SETUP                  |
|                                                              |
|          Professional AI-powered subtitle removal            |
|                                                              |
+--------------------------------------------------------------+
"""
    print(f"{Colors.GREEN}{banner}{Colors.END}")


def check_python():
    """Check Python version."""
    print(f"{Colors.BLUE}[1/6]{Colors.END} Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"{Colors.RED}ERROR: Python 3.10+ required. Found: {version.major}.{version.minor}{Colors.END}")
        return False
    
    print(f"  [OK] Python {version.major}.{version.minor}.{version.micro}")
    if _windows_cuda_wheels_unavailable():
        print(
            f"{Colors.YELLOW}  WARN: PyTorch does not publish Windows CUDA wheels for "
            f"Python {version.major}.{version.minor} yet.{Colors.END}"
        )
        print("  NVIDIA GPU acceleration needs Python 3.12 or 3.13.")
        print(
            f"  Set {PY314_CPU_OVERRIDE_ENV}=1 only if CPU-only setup is acceptable."
        )
    return True


def detect_gpu():
    """Detect available GPU."""
    print(f"\n{Colors.BLUE}[2/6]{Colors.END} Detecting GPU...")
    
    gpu_info = {
        "nvidia": False,
        "amd": False,
        "intel": False,
        "name": None,
        "cuda_version": None,
        "blackwell": False,
        "cuda_disabled_by_python": False
    }
    
    # Check NVIDIA
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_info["nvidia"] = True
            gpu_info["name"] = result.stdout.strip().split('\n')[0]

            # Blackwell (RTX 50-series, sm_120) needs CUDA 12.8 + torch 2.7+.
            # cu118/cu121 wheels carry no Blackwell kernels, so they error
            # ("no kernel image is available for execution on the device")
            # or silently fall back to CPU. Detect by name so the installer
            # can route these cards to the cu128 wheel index.
            name_lower = gpu_info["name"].lower()
            if any(model in name_lower for model in
                   (" 5050", " 5060", " 5070", " 5080", " 5090",
                    "rtx 50", "rtx pro 6000", "b100", "b200", "gb200")):
                gpu_info["blackwell"] = True

            # Get CUDA version
            result2 = subprocess.run(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=10
            )
            if result2.returncode == 0:
                driver = result2.stdout.strip().split('\n')[0]
                gpu_info["cuda_version"] = driver
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Check AMD/Intel via DirectX
    if not gpu_info["nvidia"]:
        try:
            result = subprocess.run(
                ['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                output = result.stdout.lower()
                if 'amd' in output or 'radeon' in output:
                    gpu_info["amd"] = True
                    lines = [l.strip() for l in result.stdout.split('\n') if 'amd' in l.lower() or 'radeon' in l.lower()]
                    gpu_info["name"] = lines[0] if lines else "AMD GPU"
                elif 'intel' in output:
                    gpu_info["intel"] = True
                    lines = [l.strip() for l in result.stdout.split('\n') if 'intel' in l.lower()]
                    gpu_info["name"] = lines[0] if lines else "Intel GPU"
        except Exception:
            pass

    if gpu_info["nvidia"]:
        print(f"  [OK] NVIDIA GPU detected: {gpu_info['name']}")
        print(f"    Driver version: {gpu_info['cuda_version']}")
        if gpu_info["blackwell"]:
            print(f"    Blackwell (RTX 50-series) detected -- using CUDA 12.8 wheels")
    elif gpu_info["amd"]:
        print(f"  [OK] AMD GPU detected: {gpu_info['name']}")
        print(f"    Will use DirectML")
    elif gpu_info["intel"]:
        print(f"  [OK] Intel GPU detected: {gpu_info['name']}")
        print(f"    Will use DirectML")
    else:
        print(f"  [WARN] No GPU detected, will use CPU mode")
    
    return gpu_info


def create_virtual_env():
    """Create virtual environment."""
    print(f"\n{Colors.BLUE}[3/6]{Colors.END} Creating virtual environment...")
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        print(f"  Virtual environment already exists")
        response = input(f"  Recreate? (y/N): ").strip().lower()
        if response == 'y':
            shutil.rmtree(venv_path)
        else:
            return True
    
    try:
        subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
        print(f"  [OK] Virtual environment created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}  ERROR: Failed to create virtual environment: {e}{Colors.END}")
        return False


def get_pip_command():
    """Get the pip command for the virtual environment."""
    if platform.system() == "Windows":
        return str(Path("venv/Scripts/pip.exe"))
    return str(Path("venv/bin/pip"))


def get_python_command():
    """Get the python command for the virtual environment."""
    if platform.system() == "Windows":
        return str(Path("venv/Scripts/python.exe"))
    return str(Path("venv/bin/python"))


def install_pytorch(gpu_info):
    """Install PyTorch based on GPU."""
    print(f"\n{Colors.BLUE}[4/6]{Colors.END} Installing PyTorch...")
    
    pip = get_pip_command()
    
    try:
        # torch >= 2.10.0 patches CVE-2026-24747 / CVE-2025-32434
        # (torch.load weights_only RCE in 2.9.1 and earlier).
        if gpu_info["nvidia"] and _windows_cuda_wheels_unavailable():
            version = sys.version_info
            gpu_info["cuda_disabled_by_python"] = True
            print(
                f"{Colors.RED}  ERROR: Python {version.major}.{version.minor} cannot "
                f"install Windows CUDA PyTorch wheels yet.{Colors.END}"
            )
            print("  Install Python 3.12 or 3.13 for NVIDIA GPU acceleration.")
            print(
                f"  To continue explicitly as CPU-only, set {PY314_CPU_OVERRIDE_ENV}=1 "
                "and rerun setup."
            )
            if not _allow_py314_cpu_fallback():
                return False
            print(f"{Colors.YELLOW}  WARN: Proceeding with CPU-only PyTorch by explicit override.{Colors.END}")
            subprocess.run([
                pip, 'install',
                'torch>=2.10.0', 'torchvision>=0.25.0',
                '--index-url', 'https://download.pytorch.org/whl/cpu'
            ], check=True)
        elif gpu_info["nvidia"] and gpu_info["blackwell"]:
            # Blackwell (RTX 50-series, sm_120) requires CUDA 12.8 wheels.
            # The cu128 index ships torch >= 2.7 with Blackwell kernels;
            # cu118/cu121 builds fail or fall back to CPU on these cards.
            print(f"  Installing PyTorch with CUDA 12.8 (Blackwell) support...")
            subprocess.run([
                pip, 'install',
                'torch>=2.10.0', 'torchvision>=0.25.0',
                '--index-url', 'https://download.pytorch.org/whl/cu128'
            ], check=True)
        elif gpu_info["nvidia"]:
            print(f"  Installing PyTorch with CUDA support...")
            subprocess.run([
                pip, 'install',
                'torch>=2.10.0', 'torchvision>=0.25.0',
                '--index-url', 'https://download.pytorch.org/whl/cu118'
            ], check=True)
        elif gpu_info["amd"] or gpu_info["intel"]:
            # torch-directml lags upstream torch; stay on the latest pair the
            # 0.2.5.dev240914 wheel was validated against. The DirectML
            # codepath does not exercise torch.load on untrusted files in our
            # pipeline, but we still warn users to upgrade once a patched
            # torch-directml ships.
            print(f"  Installing PyTorch with DirectML support...")
            print(f"{Colors.YELLOW}  WARN: torch-directml pins torch 2.4.x; CVE-2026-24747 fix is unavailable on this path.{Colors.END}")
            subprocess.run([pip, 'install', 'torch==2.4.1', 'torchvision==0.19.1'], check=True)
            subprocess.run([pip, 'install', 'torch-directml==0.2.5.dev240914'], check=True)
        else:
            print(f"  Installing PyTorch CPU version...")
            subprocess.run([
                pip, 'install',
                'torch>=2.10.0', 'torchvision>=0.25.0',
                '--index-url', 'https://download.pytorch.org/whl/cpu'
            ], check=True)
        
        print(f"  [OK] PyTorch installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}  ERROR: Failed to install PyTorch: {e}{Colors.END}")
        return False


def install_paddlepaddle(gpu_info):
    """Install PaddlePaddle based on GPU."""
    print(f"\n{Colors.BLUE}[5/6]{Colors.END} Installing PaddlePaddle...")
    
    pip = get_pip_command()
    
    try:
        if gpu_info["nvidia"] and gpu_info["blackwell"]:
            # Blackwell needs a CUDA 12.x PaddlePaddle build. cu126 is the
            # newest stable paddle index; the cu118 build has no sm_120
            # kernels. If PaddleOCR cannot load, detection automatically
            # falls back to RapidOCR (ONNX) which is GPU-agnostic.
            print(f"  Installing PaddlePaddle GPU (CUDA 12.6) version...")
            subprocess.run([
                pip, 'install', 'paddlepaddle-gpu==3.0.0',
                '-i', 'https://www.paddlepaddle.org.cn/packages/stable/cu126/'
            ], check=True)
        elif gpu_info["nvidia"]:
            print(f"  Installing PaddlePaddle GPU version...")
            subprocess.run([
                pip, 'install', 'paddlepaddle-gpu==3.0.0',
                '-i', 'https://www.paddlepaddle.org.cn/packages/stable/cu118/'
            ], check=True)
        else:
            print(f"  Installing PaddlePaddle CPU version...")
            subprocess.run([
                pip, 'install', 'paddlepaddle==3.0.0',
                '-i', 'https://www.paddlepaddle.org.cn/packages/stable/cpu/'
            ], check=True)
        
        print(f"  [OK] PaddlePaddle installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{Colors.YELLOW}  WARNING: PaddlePaddle installation failed: {e}{Colors.END}")
        print(f"  Text detection will use fallback method")
        return True


def install_dependencies():
    """Install remaining dependencies."""
    print(f"\n{Colors.BLUE}[6/6]{Colors.END} Installing other dependencies...")
    
    pip = get_pip_command()

    try:
        print("  Refreshing packaging tools...")
        subprocess.run([pip, 'install', '--upgrade', 'pip', 'setuptools', 'wheel'], check=True)

        if REQUIREMENTS_FILE.exists():
            print(f"  Installing dependencies from {REQUIREMENTS_FILE}...")
            try:
                subprocess.run([pip, 'install', '-r', str(REQUIREMENTS_FILE)], check=True)
                print(f"  [OK] Requirements installed")
                return True
            except subprocess.CalledProcessError:
                print(f"  Requirements install hit an optional dependency issue, falling back to the core stack...")

        core_packages = [
            'numpy>=1.21.0',
            'opencv-python>=4.12.0',
            'Pillow>=12.1.1',
            'rapidocr>=2.0.0',
            'easyocr>=1.7.0',
            'simple-lama-inpainting>=0.1.0',
        ]

        for package in core_packages:
            print(f"  Installing {package}...")
            subprocess.run([pip, 'install', package], check=True)

        try:
            subprocess.run([pip, 'install', 'paddleocr>=3.0.0'], check=True)
            print(f"  [OK] PaddleOCR installed")
        except subprocess.CalledProcessError:
            print(f"  Note: PaddleOCR skipped (RapidOCR / EasyOCR will be used instead)")

        print(f"  [OK] All dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}  ERROR: Failed to install dependencies: {e}{Colors.END}")
        return False


def check_ffmpeg():
    """Check if FFmpeg is available."""
    print(f"\n{Colors.BLUE}Checking FFmpeg...{Colors.END}")
    
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"  [OK] FFmpeg found: {version}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    print(f"{Colors.YELLOW}  [WARN] FFmpeg not found{Colors.END}")
    print(f"    Audio preservation requires FFmpeg.")
    print(f"    Download from: https://ffmpeg.org/download.html")
    print(f"    Or install with: winget install ffmpeg")
    return False


def create_launcher():
    """Create launcher batch files."""
    print(f"\n{Colors.BLUE}Creating launcher scripts...{Colors.END}")
    
    # Windows batch file
    batch_content = '''@echo off
setlocal EnableDelayedExpansion

title Video Subtitle Remover Pro

cd /d "%~dp0"

if not exist "venv\\Scripts\\python.exe" (
    echo.
    echo  ============================================================
    echo   VIDEO SUBTITLE REMOVER PRO
    echo  ============================================================
    echo.
    echo  First-time setup required.
    echo  Preparing the runtime and dependencies...
    echo.
    python -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 14) else 1)" >nul 2>nul
    if not errorlevel 1 (
        echo  WARNING: Python 3.14+ cannot install Windows CUDA PyTorch wheels.
        echo  Use Python 3.12 or 3.13 for NVIDIA GPU acceleration.
        echo  Set VSR_ALLOW_PY314_CPU=1 before launch only for CPU-only setup.
        echo.
    )
    python setup.py
    if errorlevel 1 (
        echo.
        echo  Setup did not complete. Review the messages above, then try again.
        pause
        exit /b 1
    )
)

echo Launching Video Subtitle Remover Pro...
if exist "venv\\Scripts\\pythonw.exe" (
    start "" "venv\\Scripts\\pythonw.exe" "VideoSubtitleRemover.py"
    exit /b 0
)

if exist "venv\\Scripts\\python.exe" (
    start "" "venv\\Scripts\\python.exe" "VideoSubtitleRemover.py"
    exit /b 0
)

echo.
echo  The Python runtime could not be found in the virtual environment.
echo  Re-run setup.py to repair the installation.
pause
exit /b 1
'''
    
    with open("Run_VSR_Pro.bat", "w") as f:
        f.write(batch_content)

    debug_batch_content = '''@echo off
setlocal EnableDelayedExpansion

title Video Subtitle Remover Pro (Debug)

cd /d "%~dp0"

if not exist "venv\\Scripts\\python.exe" (
    echo.
    echo  ============================================================
    echo   VIDEO SUBTITLE REMOVER PRO (DEBUG)
    echo  ============================================================
    echo.
    echo  First-time setup required.
    echo  Preparing the runtime and dependencies...
    echo.
    python -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 14) else 1)" >nul 2>nul
    if not errorlevel 1 (
        echo  WARNING: Python 3.14+ cannot install Windows CUDA PyTorch wheels.
        echo  Use Python 3.12 or 3.13 for NVIDIA GPU acceleration.
        echo  Set VSR_ALLOW_PY314_CPU=1 before launch only for CPU-only setup.
        echo.
    )
    python setup.py
    if errorlevel 1 (
        echo.
        echo  Setup did not complete. Review the messages above, then try again.
        pause
        exit /b 1
    )
)

call venv\\Scripts\\activate.bat
echo Launching Video Subtitle Remover Pro in debug mode...
echo The console will stay open after exit so you can review logs and tracebacks.
echo.
python VideoSubtitleRemover.py

pause
'''

    with open("Run_VSR_Pro_Debug.bat", "w") as f:
        f.write(debug_batch_content)
    
    # PowerShell script
    ps_content = '''# Video Subtitle Remover Pro Launcher
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".\\venv\\Scripts\\python.exe")) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host " VIDEO SUBTITLE REMOVER PRO" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "First-time setup required." -ForegroundColor Yellow
    Write-Host "Preparing the runtime and dependencies..." -ForegroundColor Yellow
    Write-Host ""
    python -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 14) else 1)" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "WARNING: Python 3.14+ cannot install Windows CUDA PyTorch wheels." -ForegroundColor Yellow
        Write-Host "Use Python 3.12 or 3.13 for NVIDIA GPU acceleration." -ForegroundColor Yellow
        Write-Host "Set VSR_ALLOW_PY314_CPU=1 before launch only for CPU-only setup." -ForegroundColor Yellow
        Write-Host ""
    }
    python setup.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "Setup did not complete. Review the messages above, then try again." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit $LASTEXITCODE
    }
}

if (Test-Path ".\\venv\\Scripts\\pythonw.exe") {
    Start-Process -FilePath ".\\venv\\Scripts\\pythonw.exe" -ArgumentList "VideoSubtitleRemover.py"
    exit 0
}

if (Test-Path ".\\venv\\Scripts\\python.exe") {
    Start-Process -FilePath ".\\venv\\Scripts\\python.exe" -ArgumentList "VideoSubtitleRemover.py"
    exit 0
}

Write-Host "The Python runtime could not be found in the virtual environment." -ForegroundColor Yellow
Read-Host "Press Enter to exit"
exit 1
'''

    with open("Run_VSR_Pro.ps1", "w") as f:
        f.write(ps_content)

    print(f"  [OK] Created Run_VSR_Pro.bat")
    print(f"  [OK] Created Run_VSR_Pro_Debug.bat")
    print(f"  [OK] Created Run_VSR_Pro.ps1")


def main():
    """Main setup function."""
    print_banner()
    
    if platform.system() != "Windows":
        print(f"{Colors.YELLOW}Note: This setup is optimized for Windows.{Colors.END}")
        print(f"For Linux/macOS, manual installation may be required.\n")
    
    # Step 1: Check Python
    if not check_python():
        sys.exit(1)
    
    # Step 2: Detect GPU
    gpu_info = detect_gpu()
    
    # Step 3: Create virtual environment
    if not create_virtual_env():
        sys.exit(1)
    
    # Step 4: Install PyTorch
    if not install_pytorch(gpu_info):
        sys.exit(1)
    
    # Step 5: Install PaddlePaddle
    install_paddlepaddle(gpu_info)
    
    # Step 6: Install other dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Check FFmpeg
    ffmpeg_ok = check_ffmpeg()
    
    # Create launcher
    create_launcher()
    
    # Done!
    print(f"\n{Colors.GREEN}{'='*60}{Colors.END}")
    print(f"{Colors.GREEN}  SETUP COMPLETE!{Colors.END}")
    print(f"{Colors.GREEN}{'='*60}{Colors.END}")
    print(f"\n  To run the application:")
    print(f"    * Double-click: {Colors.BOLD}Run_VSR_Pro.bat{Colors.END}")
    print(f"    * Troubleshooting: {Colors.BOLD}Run_VSR_Pro_Debug.bat{Colors.END}")
    print(f"    * Or run: {Colors.BOLD}python VideoSubtitleRemover.py{Colors.END}")
    print(f"\n  GPU Mode: ", end="")
    
    if gpu_info["nvidia"] and gpu_info.get("cuda_disabled_by_python"):
        print(f"{Colors.YELLOW}CPU (Python CUDA wheels unavailable){Colors.END}")
    elif gpu_info["nvidia"]:
        print(f"{Colors.GREEN}NVIDIA CUDA{Colors.END}")
    elif gpu_info["amd"] or gpu_info["intel"]:
        print(f"{Colors.GREEN}DirectML{Colors.END}")
    else:
        print(f"{Colors.YELLOW}CPU (slower){Colors.END}")
    if not ffmpeg_ok:
        print(f"\n  {Colors.YELLOW}FFmpeg is still missing.{Colors.END} Video outputs will work, but audio preservation stays unavailable until FFmpeg is installed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Setup cancelled.{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Setup failed: {e}{Colors.END}")
        sys.exit(1)
