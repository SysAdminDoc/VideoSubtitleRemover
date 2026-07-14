"""
Video Subtitle Remover Pro - Setup Script
==========================================

This script helps set up the application environment on Windows.
Run: python setup.py
"""

import os
import sys
import argparse
import subprocess
import platform
import shutil
import stat
from pathlib import Path

# Enable ANSI escape codes on Windows 10+
os.system('')
REQUIREMENTS_FILE = Path("requirements.txt")
PYTHON_CUDA_WHEEL_MAX = (3, 13)
PY314_CPU_OVERRIDE_ENV = "VSR_ALLOW_PY314_CPU"
VENV_CREATE_TIMEOUT_SECONDS = 600
PIP_INSTALL_TIMEOUT_SECONDS = 1800
DIRECTML_PACKAGE_VERSION = "1.24.4"
DIRECTML_PACKAGE_SPEC = f"onnxruntime-directml=={DIRECTML_PACKAGE_VERSION}"


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


def _run_setup_command(args, timeout_seconds, action):
    """Run a setup subprocess with a hard timeout and clear retry guidance."""
    try:
        return subprocess.run(args, check=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        minutes = max(1, timeout_seconds // 60)
        print(
            f"{Colors.RED}  ERROR: Timed out while {action} after "
            f"{minutes} minutes.{Colors.END}"
        )
        print("  Check your network, PyPI mirror, and antivirus scanner, then rerun setup.py.")
        print("  If a partial virtual environment was created, delete venv and retry.")
        raise


def _run_pip_install(args, action):
    """Run a pip install command with the standard installer timeout."""
    return _run_setup_command(args, PIP_INSTALL_TIMEOUT_SECONDS, action)


def _preflight_directml_distribution(pip):
    """Verify the reviewed DirectML wheel resolves before changing the venv."""
    print(f"  Preflighting {DIRECTML_PACKAGE_SPEC} wheel availability...")
    command = [
        pip,
        "install",
        "--dry-run",
        "--only-binary=:all:",
        "--no-deps",
        DIRECTML_PACKAGE_SPEC,
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=PIP_INSTALL_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        print(
            f"{Colors.RED}  ERROR: Timed out while checking the DirectML "
            f"wheel on PyPI.{Colors.END}"
        )
        print("  Check the network/PyPI mirror, then rerun setup.py.")
        return False
    if result.returncode == 0:
        print(f"  [OK] {DIRECTML_PACKAGE_SPEC} is available for this Python/platform")
        return True
    detail = (result.stderr or result.stdout or "no compatible wheel").strip()
    if detail:
        detail = detail.splitlines()[-1]
    print(
        f"{Colors.RED}  ERROR: {DIRECTML_PACKAGE_SPEC} is not available for "
        f"this Python/platform: {detail}{Colors.END}"
    )
    print(
        "  No packages were changed. Use the CPU setup path, install a supported "
        "Python 3.11-3.14 Windows environment, or evaluate Windows ML with "
        "`python -m backend.processor --audit-windows-ml`."
    )
    return False


def _is_reparse_point(path):
    """Return True for Windows junctions/symlinks without following targets."""
    try:
        attrs = getattr(os.lstat(path), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _is_repo_local_venv(path):
    """Only allow setup.py to remove the literal repo-local venv directory."""
    expected = os.path.normcase(os.path.abspath("venv"))
    actual = os.path.normcase(os.path.abspath(path))
    if actual != expected:
        return False
    return not (path.is_symlink() or _is_reparse_point(path))


def _remove_existing_venv(path):
    """Delete an existing venv only after path-boundary checks pass."""
    if not _is_repo_local_venv(path):
        print(
            f"{Colors.RED}  ERROR: Refusing to remove unsafe virtual "
            f"environment path: {path}{Colors.END}"
        )
        print("  Delete or rename the path manually, then rerun setup.py.")
        return False
    shutil.rmtree(path)
    return True


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


def create_virtual_env(repair=False):
    """Create virtual environment."""
    print(f"\n{Colors.BLUE}[3/6]{Colors.END} Creating virtual environment...")
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        if not repair:
            print("  Virtual environment already exists; keeping it.")
            print("  Run setup.py --repair to recreate the repo-local venv.")
            return True
        print("  Repair requested; recreating the repo-local virtual environment.")
        if not _remove_existing_venv(venv_path):
            return False
    
    try:
        _run_setup_command(
            [sys.executable, '-m', 'venv', 'venv'],
            VENV_CREATE_TIMEOUT_SECONDS,
            "creating the virtual environment",
        )
        print(f"  [OK] Virtual environment created")
        return True
    except subprocess.TimeoutExpired:
        return False
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
            _run_pip_install([
                pip, 'install',
                'torch>=2.10.0', 'torchvision>=0.25.0',
                '--index-url', 'https://download.pytorch.org/whl/cpu'
            ], "installing CPU PyTorch")
        elif gpu_info["nvidia"] and gpu_info["blackwell"]:
            # Blackwell (RTX 50-series, sm_120) requires CUDA 12.8 wheels.
            # The cu128 index also carries the current torch security floor.
            print(f"  Installing PyTorch with CUDA 12.8 (Blackwell) support...")
            _run_pip_install([
                pip, 'install',
                'torch>=2.10.0', 'torchvision>=0.25.0',
                '--index-url', 'https://download.pytorch.org/whl/cu128'
            ], "installing CUDA 12.8 PyTorch")
        elif gpu_info["nvidia"]:
            print(f"  Installing PyTorch with CUDA 12.8 support...")
            _run_pip_install([
                pip, 'install',
                'torch>=2.10.0', 'torchvision>=0.25.0',
                '--index-url', 'https://download.pytorch.org/whl/cu128'
            ], "installing CUDA 12.8 PyTorch")
        elif gpu_info["amd"] or gpu_info["intel"]:
            print(f"  Installing PyTorch CPU runtime for AMD/Intel fallback paths...")
            print(f"  DirectML acceleration is provided by ONNX Runtime, not torch-directml.")
            _run_pip_install([
                pip, 'install',
                'torch>=2.10.0', 'torchvision>=0.25.0',
                '--index-url', 'https://download.pytorch.org/whl/cpu'
            ], "installing CPU PyTorch")
        else:
            print(f"  Installing PyTorch CPU version...")
            _run_pip_install([
                pip, 'install',
                'torch>=2.10.0', 'torchvision>=0.25.0',
                '--index-url', 'https://download.pytorch.org/whl/cpu'
            ], "installing CPU PyTorch")
        
        print(f"  [OK] PyTorch installed")
        return True
    except subprocess.TimeoutExpired:
        return False
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
            _run_pip_install([
                pip, 'install', 'paddlepaddle-gpu==3.0.0',
                '-i', 'https://www.paddlepaddle.org.cn/packages/stable/cu126/'
            ], "installing CUDA 12.6 PaddlePaddle")
        elif gpu_info["nvidia"]:
            print(f"  Installing PaddlePaddle GPU version...")
            _run_pip_install([
                pip, 'install', 'paddlepaddle-gpu==3.0.0',
                '-i', 'https://www.paddlepaddle.org.cn/packages/stable/cu118/'
            ], "installing CUDA PaddlePaddle")
        else:
            print(f"  Installing PaddlePaddle CPU version...")
            _run_pip_install([
                pip, 'install', 'paddlepaddle==3.0.0',
                '-i', 'https://www.paddlepaddle.org.cn/packages/stable/cpu/'
            ], "installing CPU PaddlePaddle")
        
        print(f"  [OK] PaddlePaddle installed")
        return True
    except subprocess.TimeoutExpired:
        return False
    except subprocess.CalledProcessError as e:
        print(f"{Colors.YELLOW}  WARNING: PaddlePaddle installation failed: {e}{Colors.END}")
        print(f"  Text detection will use fallback method")
        return True


def install_dependencies(gpu_info=None):
    """Install remaining dependencies."""
    print(f"\n{Colors.BLUE}[6/6]{Colors.END} Installing other dependencies...")
    
    pip = get_pip_command()

    directml_requested = bool(
        gpu_info and (gpu_info.get("amd") or gpu_info.get("intel"))
    )
    if directml_requested and not _preflight_directml_distribution(pip):
        return False

    try:
        print("  Refreshing packaging tools...")
        _run_pip_install(
            [pip, 'install', '--upgrade', 'pip', 'setuptools<82', 'wheel'],
            "refreshing packaging tools",
        )

        installed_from_requirements = False
        if REQUIREMENTS_FILE.exists():
            print(f"  Installing dependencies from {REQUIREMENTS_FILE}...")
            try:
                _run_pip_install(
                    [pip, 'install', '-r', str(REQUIREMENTS_FILE)],
                    "installing requirements.txt",
                )
                print(f"  [OK] Requirements installed")
                installed_from_requirements = True
            except subprocess.CalledProcessError:
                print(f"  Requirements install hit an optional dependency issue, falling back to the core stack...")

        if not installed_from_requirements:
            core_packages = [
                'numpy>=1.21.0',
                'opencv-python>=4.12.0',
                'Pillow>=12.3.0',
                'rapidocr>=2.0.0,<4.0.0',
                'easyocr>=1.7.0',
                'simple-lama-inpainting>=0.1.0',
            ]

            for package in core_packages:
                print(f"  Installing {package}...")
                _run_pip_install([pip, 'install', package], f"installing {package}")

            try:
                _run_pip_install(
                    [pip, 'install', 'paddleocr>=3.0.0,<4.0.0'],
                    "installing PaddleOCR",
                )
                print(f"  [OK] PaddleOCR installed")
            except subprocess.CalledProcessError:
                print(f"  Note: PaddleOCR skipped (RapidOCR / EasyOCR will be used instead)")

        if directml_requested:
            print("  Installing ONNX Runtime DirectML provider...")
            _run_pip_install(
                [pip, 'install', DIRECTML_PACKAGE_SPEC],
                "installing ONNX Runtime DirectML",
            )
            print(f"  [OK] ONNX Runtime DirectML installed")
            if gpu_info.get("intel"):
                print("  Installing OpenVINO runtime for RapidOCR...")
                try:
                    _run_pip_install(
                        [pip, 'install', 'openvino>=2025.0.0'],
                        "installing OpenVINO runtime",
                    )
                    print(f"  [OK] OpenVINO runtime installed")
                except subprocess.CalledProcessError as exc:
                    print(f"{Colors.YELLOW}  WARNING: OpenVINO install failed: {exc}{Colors.END}")
                    print("  RapidOCR will use ONNX Runtime unless OpenVINO is installed manually.")
        elif gpu_info and gpu_info.get("nvidia") and not gpu_info.get("cuda_disabled_by_python"):
            print("  Installing ONNX Runtime CUDA provider...")
            print("  Stable PyPI onnxruntime-gpu is the CUDA 12.x path; CUDA 13 uses ONNX Runtime nightly/custom wheels.")
            try:
                _run_pip_install(
                    [pip, 'install', 'onnxruntime-gpu>=1.25.0'],
                    "installing ONNX Runtime CUDA",
                )
                print(f"  [OK] ONNX Runtime CUDA provider installed")
            except subprocess.CalledProcessError as exc:
                print(f"{Colors.YELLOW}  WARNING: ONNX Runtime CUDA install failed: {exc}{Colors.END}")
                print("  LaMa ONNX will use CPU/DirectML if available; PyTorch/Paddle paths are unchanged.")

        print(f"  [OK] All dependencies installed")
        return True
    except subprocess.TimeoutExpired:
        return False
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

:: Change to script directory
cd /d "%~dp0"

set "VSR_SETUP_REPAIR=0"

if not exist "venv\\Scripts\\python.exe" (
    set "VSR_SETUP_REPAIR=1"
) else (
    "venv\\Scripts\\python.exe" -c "import cv2, PIL, numpy" >nul 2>nul
    if errorlevel 1 set "VSR_SETUP_REPAIR=1"
)

if "%VSR_SETUP_REPAIR%"=="1" (
    echo.
    echo  ============================================================
    echo   VIDEO SUBTITLE REMOVER PRO
    echo  ============================================================
    echo.
    echo  Runtime setup or repair required.
    echo  Preparing the runtime and dependencies without prompts...
    echo.
    python -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 14) else 1)" >nul 2>nul
    if not errorlevel 1 (
        echo  WARNING: Python 3.14+ cannot install Windows CUDA PyTorch wheels.
        echo  Use Python 3.12 or 3.13 for NVIDIA GPU acceleration.
        echo  Set VSR_ALLOW_PY314_CPU=1 before launch only for CPU-only setup.
        echo.
    )
    python setup.py --repair
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

set "VSR_SETUP_REPAIR=0"

if not exist "venv\\Scripts\\python.exe" (
    set "VSR_SETUP_REPAIR=1"
) else (
    "venv\\Scripts\\python.exe" -c "import cv2, PIL, numpy" >nul 2>nul
    if errorlevel 1 set "VSR_SETUP_REPAIR=1"
)

if "%VSR_SETUP_REPAIR%"=="1" (
    echo.
    echo  ============================================================
    echo   VIDEO SUBTITLE REMOVER PRO (DEBUG)
    echo  ============================================================
    echo.
    echo  Runtime setup or repair required.
    echo  Preparing the runtime and dependencies without prompts...
    echo.
    python -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 14) else 1)" >nul 2>nul
    if not errorlevel 1 (
        echo  WARNING: Python 3.14+ cannot install Windows CUDA PyTorch wheels.
        echo  Use Python 3.12 or 3.13 for NVIDIA GPU acceleration.
        echo  Set VSR_ALLOW_PY314_CPU=1 before launch only for CPU-only setup.
        echo.
    )
    python setup.py --repair
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

$needsRepair = -not (Test-Path ".\\venv\\Scripts\\python.exe")
if (-not $needsRepair) {
    & ".\\venv\\Scripts\\python.exe" -c "import cv2, PIL, numpy" 1>$null 2>$null
    if ($LASTEXITCODE -ne 0) {
        $needsRepair = $true
    }
}

if ($needsRepair) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host " VIDEO SUBTITLE REMOVER PRO" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Runtime setup or repair required." -ForegroundColor Yellow
    Write-Host "Preparing the runtime and dependencies without prompts..." -ForegroundColor Yellow
    Write-Host ""
    python -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 14) else 1)" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "WARNING: Python 3.14+ cannot install Windows CUDA PyTorch wheels." -ForegroundColor Yellow
        Write-Host "Use Python 3.12 or 3.13 for NVIDIA GPU acceleration." -ForegroundColor Yellow
        Write-Host "Set VSR_ALLOW_PY314_CPU=1 before launch only for CPU-only setup." -ForegroundColor Yellow
        Write-Host ""
    }
    python setup.py --repair
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


def parse_setup_args(argv=None):
    """Parse setup command-line options."""
    parser = argparse.ArgumentParser(
        description="Prepare the Video Subtitle Remover Pro runtime."
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Recreate the repo-local venv after safety checks, without prompting.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Main setup function."""
    args = parse_setup_args(argv)
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
    if not create_virtual_env(repair=args.repair):
        sys.exit(1)
    
    # Step 4: Install PyTorch
    if not install_pytorch(gpu_info):
        sys.exit(1)
    
    # Step 5: Install PaddlePaddle
    if not install_paddlepaddle(gpu_info):
        sys.exit(1)
    
    # Step 6: Install other dependencies
    if not install_dependencies(gpu_info):
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
    print(f"    * PowerShell: {Colors.BOLD}.\\Run_VSR_Pro.ps1{Colors.END}")
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
