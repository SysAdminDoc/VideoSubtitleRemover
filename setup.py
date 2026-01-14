"""
Video Subtitle Remover Pro - Setup Script
==========================================

This script helps set up the application environment on Windows.
Run: python setup.py install
"""

import os
import sys
import subprocess
import platform
import urllib.request
import zipfile
import shutil
from pathlib import Path


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
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║          VIDEO SUBTITLE REMOVER PRO - SETUP                  ║
║                                                              ║
║          Professional AI-powered subtitle removal            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(f"{Colors.GREEN}{banner}{Colors.END}")


def check_python():
    """Check Python version."""
    print(f"{Colors.BLUE}[1/6]{Colors.END} Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"{Colors.RED}ERROR: Python 3.10+ required. Found: {version.major}.{version.minor}{Colors.END}")
        return False
    
    print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def detect_gpu():
    """Detect available GPU."""
    print(f"\n{Colors.BLUE}[2/6]{Colors.END} Detecting GPU...")
    
    gpu_info = {
        "nvidia": False,
        "amd": False,
        "intel": False,
        "name": None,
        "cuda_version": None
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
        except:
            pass
    
    if gpu_info["nvidia"]:
        print(f"  ✓ NVIDIA GPU detected: {gpu_info['name']}")
        print(f"    Driver version: {gpu_info['cuda_version']}")
    elif gpu_info["amd"]:
        print(f"  ✓ AMD GPU detected: {gpu_info['name']}")
        print(f"    Will use DirectML")
    elif gpu_info["intel"]:
        print(f"  ✓ Intel GPU detected: {gpu_info['name']}")
        print(f"    Will use DirectML")
    else:
        print(f"  ⚠ No GPU detected, will use CPU mode")
    
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
        print(f"  ✓ Virtual environment created")
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
        if gpu_info["nvidia"]:
            print(f"  Installing PyTorch with CUDA support...")
            subprocess.run([
                pip, 'install',
                'torch==2.7.0', 'torchvision==0.22.0',
                '--index-url', 'https://download.pytorch.org/whl/cu118'
            ], check=True)
        elif gpu_info["amd"] or gpu_info["intel"]:
            print(f"  Installing PyTorch with DirectML support...")
            subprocess.run([pip, 'install', 'torch==2.4.1', 'torchvision==0.19.1'], check=True)
            subprocess.run([pip, 'install', 'torch-directml==0.2.5.dev240914'], check=True)
        else:
            print(f"  Installing PyTorch CPU version...")
            subprocess.run([
                pip, 'install',
                'torch==2.7.0', 'torchvision==0.22.0',
                '--index-url', 'https://download.pytorch.org/whl/cpu'
            ], check=True)
        
        print(f"  ✓ PyTorch installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}  ERROR: Failed to install PyTorch: {e}{Colors.END}")
        return False


def install_paddlepaddle(gpu_info):
    """Install PaddlePaddle based on GPU."""
    print(f"\n{Colors.BLUE}[5/6]{Colors.END} Installing PaddlePaddle...")
    
    pip = get_pip_command()
    
    try:
        if gpu_info["nvidia"]:
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
        
        print(f"  ✓ PaddlePaddle installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{Colors.YELLOW}  WARNING: PaddlePaddle installation failed: {e}{Colors.END}")
        print(f"  Text detection will use fallback method")
        return True


def install_dependencies():
    """Install remaining dependencies."""
    print(f"\n{Colors.BLUE}[6/6]{Colors.END} Installing other dependencies...")
    
    pip = get_pip_command()
    
    packages = [
        'numpy>=1.21.0',
        'opencv-python>=4.5.0',
        'Pillow>=9.0.0',
        'tqdm>=4.64.0',
        'colorama>=0.4.5',
    ]
    
    try:
        # Install paddleocr (may fail if PaddlePaddle failed)
        try:
            subprocess.run([pip, 'install', 'paddleocr>=2.6.0'], check=True, capture_output=True)
        except:
            print(f"  Note: PaddleOCR installation skipped")
        
        for package in packages:
            subprocess.run([pip, 'install', package], check=True, capture_output=True)
        
        print(f"  ✓ Dependencies installed")
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
            print(f"  ✓ FFmpeg found: {version}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    print(f"{Colors.YELLOW}  ⚠ FFmpeg not found{Colors.END}")
    print(f"    Audio preservation requires FFmpeg.")
    print(f"    Download from: https://ffmpeg.org/download.html")
    print(f"    Or install with: winget install ffmpeg")
    return False


def create_launcher():
    """Create launcher batch files."""
    print(f"\n{Colors.BLUE}Creating launcher scripts...{Colors.END}")
    
    # Windows batch file
    batch_content = '''@echo off
title Video Subtitle Remover Pro
cd /d "%~dp0"
call venv\\Scripts\\activate.bat
python VideoSubtitleRemover.py
pause
'''
    
    with open("Run_VSR_Pro.bat", "w") as f:
        f.write(batch_content)
    
    # PowerShell script
    ps_content = '''# Video Subtitle Remover Pro Launcher
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
& .\\venv\\Scripts\\Activate.ps1
python VideoSubtitleRemover.py
Read-Host "Press Enter to exit"
'''
    
    with open("Run_VSR_Pro.ps1", "w") as f:
        f.write(ps_content)
    
    print(f"  ✓ Created Run_VSR_Pro.bat")
    print(f"  ✓ Created Run_VSR_Pro.ps1")


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
    check_ffmpeg()
    
    # Create launcher
    create_launcher()
    
    # Done!
    print(f"\n{Colors.GREEN}{'='*60}{Colors.END}")
    print(f"{Colors.GREEN}  SETUP COMPLETE!{Colors.END}")
    print(f"{Colors.GREEN}{'='*60}{Colors.END}")
    print(f"\n  To run the application:")
    print(f"    • Double-click: {Colors.BOLD}Run_VSR_Pro.bat{Colors.END}")
    print(f"    • Or run: {Colors.BOLD}python VideoSubtitleRemover.py{Colors.END}")
    print(f"\n  GPU Mode: ", end="")
    
    if gpu_info["nvidia"]:
        print(f"{Colors.GREEN}NVIDIA CUDA{Colors.END}")
    elif gpu_info["amd"] or gpu_info["intel"]:
        print(f"{Colors.GREEN}DirectML{Colors.END}")
    else:
        print(f"{Colors.YELLOW}CPU (slower){Colors.END}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Setup cancelled.{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Setup failed: {e}{Colors.END}")
        sys.exit(1)
