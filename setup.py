"""
Video Subtitle Remover Pro - Setup Script
==========================================

This script helps set up the application environment on Windows.
Run: python setup.py
"""

import os
import sys
import argparse
import datetime as _dt
import json
import subprocess
import platform
import shutil
import stat
import tempfile
from pathlib import Path

from backend.dependency_profiles import (
    SUPPORTED_PROFILES,
    ensure_profile_current,
    profile_capabilities,
    profile_required_packages,
    select_profile,
)

# Enable ANSI escape codes on Windows 10+
os.system('')
REQUIREMENTS_FILE = Path("requirements.txt")
PYTHON_CUDA_WHEEL_MAX = (3, 13)
PY314_CPU_OVERRIDE_ENV = "VSR_ALLOW_PY314_CPU"
VENV_CREATE_TIMEOUT_SECONDS = 600
PIP_INSTALL_TIMEOUT_SECONDS = 1800
MINIMUM_PYTHON = (3, 11)
DIRECTML_PACKAGE_VERSION = "1.24.4"
DIRECTML_PACKAGE_SPEC = f"onnxruntime-directml=={DIRECTML_PACKAGE_VERSION}"
# ONNX Runtime GPU wheels: from 1.27.0 (2026-06-15) the default
# onnxruntime-gpu PyPI wheel is the CUDA 13 build. RM-319 confirmed that by
# running 1.29.0 on a CUDA-13-less host, where it fails to load
# cublasLt64_13.dll and silently drops to CPU. The reviewed NVIDIA lane
# installs torch from the cu130 index, which supplies that runtime, so this
# floor moves to 1.27.0. CUDA 12 hosts install onnxruntime-gpu 1.26.x and
# cu128 torch manually (see the on-screen note in install_dependencies).
# Ref: github.com/microsoft/onnxruntime/releases/tag/v1.27.0
ONNXRUNTIME_GPU_MIN = "1.27.0"
ONNXRUNTIME_GPU_MAX_EXCLUSIVE = "1.30.0"
ONNXRUNTIME_GPU_SPEC = (
    f"onnxruntime-gpu>={ONNXRUNTIME_GPU_MIN},<{ONNXRUNTIME_GPU_MAX_EXCLUSIVE}"
)
# The CPU lane carries its own floor, which is the security floor rather
# than the GPU recommendation.
ONNXRUNTIME_CPU_MIN = "1.26.0"
ONNXRUNTIME_CPU_SPEC = f"onnxruntime>={ONNXRUNTIME_CPU_MIN}"
TORCH_MINIMUM = "2.13.0"
TORCHVISION_MINIMUM = "0.28.0"
TORCH_SPEC = f"torch>={TORCH_MINIMUM}"
TORCHVISION_SPEC = f"torchvision>={TORCHVISION_MINIMUM}"
SETUP_PROGRESS_ENV = "VSR_SETUP_PROGRESS_FILE"
SETUP_REPORT_SCHEMA = "vsr.setup_report.v1"
SETUP_REPORT_PATH = Path("venv/.vsr-setup-report.json")
PROFILE_VERIFY_REPORT_PATH = Path("venv/.vsr-profile-verification.json")
PROFILE_VERIFY_TIMEOUT_SECONDS = 300
_ACTIVE_SETUP_PROFILE = "auto"


def _setup_progress_path():
    """Return a launcher-owned temp status path, rejecting broader writes."""
    raw = os.environ.get(SETUP_PROGRESS_ENV, "").strip()
    if not raw:
        return None
    path = Path(raw)
    try:
        resolved = path.resolve()
        temp_root = Path(tempfile.gettempdir()).resolve()
    except OSError:
        return None
    if resolved.parent != temp_root:
        return None
    if not resolved.name.startswith("vsr-pro-setup-"):
        return None
    if resolved.suffix.lower() != ".status":
        return None
    return resolved


def write_setup_progress(message, percent, state="RUNNING"):
    """Atomically publish one bounded setup status for the optional splash."""
    path = _setup_progress_path()
    if path is None:
        return False
    safe_state = str(state).strip().upper()
    if safe_state not in {"RUNNING", "DONE", "ERROR"}:
        safe_state = "RUNNING"
    safe_message = " ".join(str(message).replace("|", " ").split())[:240]
    safe_percent = max(0, min(100, int(percent)))
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(
            f"{safe_state}|{safe_message}|{safe_percent}", encoding="utf-8")
        temporary.replace(path)
        return True
    except OSError:
        return False


def _dependency_profile_name(gpu_info=None):
    info = dict(gpu_info or {})
    return str(info.get("dependency_profile") or select_profile(info))


def _profile_constraint_args(gpu_info=None):
    name = _dependency_profile_name(gpu_info)
    return ["--constraint", str(ensure_profile_current(name))]


def _apply_profile_override(gpu_info, requested):
    """Apply an explicit deterministic install profile to detected hardware."""
    info = gpu_info
    if requested == "auto":
        info["dependency_profile"] = select_profile(info)
        return info
    detected_intel = bool(info.get("intel"))
    detected_amd = bool(info.get("amd"))
    detected_nvidia = bool(info.get("nvidia"))
    info["nvidia"] = requested == "nvidia"
    info["amd"] = requested == "directml" and (detected_amd or not detected_intel)
    info["intel"] = requested == "directml" and detected_intel
    info["blackwell"] = bool(
        requested == "nvidia" and detected_nvidia and info.get("blackwell"))
    info["cuda_disabled_by_python"] = False
    info["dependency_profile"] = requested
    return info


def _setup_repair_command(profile_name):
    profile = profile_name if profile_name in SUPPORTED_PROFILES else "auto"
    return f"python setup.py --repair --profile {profile}"


def write_setup_report(
    profile_name,
    status,
    *,
    stage="",
    message="",
    verification=None,
    path=None,
):
    """Write the last setup outcome atomically inside the repo venv."""
    target = Path(path) if path is not None else SETUP_REPORT_PATH
    if not target.parent.exists():
        return False
    try:
        capabilities = list(profile_capabilities(profile_name))
        required_packages = [
            dict(item) for item in profile_required_packages(profile_name)
        ]
    except (OSError, ValueError, KeyError):
        capabilities = []
        required_packages = []
    payload = {
        "schema": SETUP_REPORT_SCHEMA,
        "createdAt": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "profile": str(profile_name),
        "status": str(status),
        "stage": str(stage),
        "message": str(message),
        "repairCommand": _setup_repair_command(profile_name),
        "capabilities": capabilities,
        "requiredPackages": required_packages,
        "verification": (
            dict(verification) if isinstance(verification, dict) else None
        ),
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass
        return False
    return True


def _setup_failed(profile_name, stage, message, verification=None):
    repair = _setup_repair_command(profile_name)
    write_setup_progress("Setup failed. Review the console details.", 100, "ERROR")
    report_written = write_setup_report(
        profile_name,
        "failed",
        stage=stage,
        message=message,
        verification=verification,
    )
    print(f"{Colors.RED}  ERROR: {message}{Colors.END}")
    print(f"  Repair command: {repair}")
    if report_written:
        print(f"  Setup report: {SETUP_REPORT_PATH}")
    return 1


def _print_profile_contract(profile_name):
    required = profile_required_packages(profile_name)
    print(f"  Dependency profile: {profile_name}")
    print("  Locked required packages:")
    for item in required:
        print(f"    * {item['name']}=={item['expectedVersion']}")
    print("  Expected capabilities:")
    for capability in profile_capabilities(profile_name):
        print(f"    * {capability}")


def verify_installed_profile(profile_name, *, report_path=None):
    """Run the same locked-profile verifier used by source launchers."""
    target = (
        Path(report_path)
        if report_path is not None
        else PROFILE_VERIFY_REPORT_PATH
    )
    try:
        target.unlink(missing_ok=True)
    except OSError as exc:
        return {
            "schema": "vsr.dependency_profile_status.v1",
            "profile": profile_name,
            "valid": False,
            "errors": [f"Could not prepare the profile report: {exc}"],
        }
    command = [
        get_python_command(),
        "-m",
        "backend.dependency_profiles",
        "verify",
        "--profile",
        profile_name,
        "--output",
        str(target),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=PROFILE_VERIFY_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass
        return {
            "schema": "vsr.dependency_profile_status.v1",
            "profile": profile_name,
            "valid": False,
            "errors": [str(exc)],
        }
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError):
        payload = {
            "schema": "vsr.dependency_profile_status.v1",
            "profile": profile_name,
            "valid": False,
            "errors": [
                (result.stderr or result.stdout or "profile verifier returned no report").strip()
            ],
        }
    finally:
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass
    if result.returncode != 0:
        payload["valid"] = False
        if not payload.get("errors"):
            payload["errors"] = [
                f"profile verifier exited with code {result.returncode}"
            ]
    return payload


def _print_profile_verification(payload):
    print("  Verified required packages:")
    for item in payload.get("requiredPackages", []):
        if not isinstance(item, dict):
            continue
        print(
            f"    * {item.get('name')}=={item.get('installedVersion')} "
            f"(expected {item.get('expectedVersion')})"
        )
    smoke = payload.get("providerSmoke")
    if isinstance(smoke, dict):
        active = ", ".join(smoke.get("activeProviders") or []) or "none"
        print(f"  Provider inference: {active}")
    print("  Verified capabilities:")
    for capability in payload.get("capabilities", []):
        print(f"    * {capability}")


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
        *_profile_constraint_args({"amd": True, "dependency_profile": "directml"}),
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
    if (version.major, version.minor) < MINIMUM_PYTHON:
        print(f"{Colors.RED}ERROR: Python 3.11+ required. Found: {version.major}.{version.minor}{Colors.END}")
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

            # Blackwell (RTX 50-series, sm_120) needs CUDA 12.8 or newer.
            # cu118/cu121 wheels carry no Blackwell kernels, so they error
            # ("no kernel image is available for execution on the device")
            # or silently fall back to CPU. The reviewed lane is cu130,
            # which carries them; the flag stays because PaddlePaddle still
            # needs a separate index decision for these cards.
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
        controller_output = None
        # WMIC is absent on clean Windows 11 installations. CIM is the
        # supported probe; retain WMIC only as a compatibility fallback for
        # older images that do not expose PowerShell's CIM cmdlet.
        try:
            result = subprocess.run(
                [
                    "powershell", "-NoProfile", "-NonInteractive",
                    "-Command",
                    "Get-CimInstance Win32_VideoController | "
                    "Select-Object -ExpandProperty Name",
                ],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                controller_output = result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass

        if controller_output is None:
            try:
                result = subprocess.run(
                    ["wmic", "path", "win32_VideoController", "get", "name"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    controller_output = result.stdout
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                pass

        if controller_output is None:
            print(
                f"{Colors.YELLOW}  WARN: GPU probe was inconclusive "
                "(PowerShell CIM and WMIC were unavailable).{Colors.END}"
            )
        else:
            output = controller_output.lower()
            lines = [line.strip() for line in controller_output.splitlines() if line.strip()]
            if "amd" in output or "radeon" in output:
                gpu_info["amd"] = True
                gpu_info["name"] = next(
                    (line for line in lines
                     if "amd" in line.lower() or "radeon" in line.lower()),
                    "AMD GPU",
                )
            elif "intel" in output:
                gpu_info["intel"] = True
                gpu_info["name"] = next(
                    (line for line in lines if "intel" in line.lower()),
                    "Intel GPU",
                )

    if gpu_info["nvidia"]:
        print(f"  [OK] NVIDIA GPU detected: {gpu_info['name']}")
        print(f"    Driver version: {gpu_info['cuda_version']}")
        if gpu_info["blackwell"]:
            print(f"    Blackwell (RTX 50-series) detected -- using CUDA 13 wheels")
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
    profile_args = _profile_constraint_args(gpu_info)
    
    try:
        # torch >= 2.11.0 patches CVE-2026-24747 / CVE-2025-32434
        # (torch.load weights_only RCE in 2.9.1 and earlier).
        if gpu_info["nvidia"] and _windows_cuda_wheels_unavailable():
            version = sys.version_info
            gpu_info["cuda_disabled_by_python"] = True
            gpu_info["dependency_profile"] = "cpu"
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
                TORCH_SPEC, TORCHVISION_SPEC,
                *_profile_constraint_args(gpu_info),
                '--index-url', 'https://download.pytorch.org/whl/cpu'
            ], "installing CPU PyTorch")
        elif gpu_info["nvidia"]:
            # RM-319: the reviewed NVIDIA lane is CUDA 13. cu130 carries
            # Blackwell (sm_120) kernels like cu128 did, and unlike cu128 it
            # still publishes the current torch, so both card generations take
            # the same index. It also supplies the CUDA 13 runtime that the
            # default onnxruntime-gpu wheel loads.
            print(f"  Installing PyTorch with CUDA 13.0 support...")
            _run_pip_install([
                pip, 'install',
                TORCH_SPEC, TORCHVISION_SPEC,
                *profile_args,
                '--index-url', 'https://download.pytorch.org/whl/cu130'
            ], "installing CUDA 13.0 PyTorch")
        elif gpu_info["amd"] or gpu_info["intel"]:
            print(f"  Installing PyTorch CPU runtime for AMD/Intel fallback paths...")
            print(f"  DirectML acceleration is provided by ONNX Runtime, not torch-directml.")
            _run_pip_install([
                pip, 'install',
                TORCH_SPEC, TORCHVISION_SPEC,
                *profile_args,
                '--index-url', 'https://download.pytorch.org/whl/cpu'
            ], "installing CPU PyTorch")
        else:
            print(f"  Installing PyTorch CPU version...")
            _run_pip_install([
                pip, 'install',
                TORCH_SPEC, TORCHVISION_SPEC,
                *profile_args,
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
    profile_args = _profile_constraint_args(gpu_info)
    
    try:
        if gpu_info["nvidia"] and gpu_info["blackwell"]:
            # Blackwell needs a CUDA 12.x PaddlePaddle build. cu126 is the
            # newest stable paddle index; the cu118 build has no sm_120
            # kernels. If PaddleOCR cannot load, detection automatically
            # falls back to RapidOCR (ONNX) which is GPU-agnostic.
            print(f"  Installing PaddlePaddle GPU (CUDA 12.6) version...")
            _run_pip_install([
                pip, 'install', 'paddlepaddle-gpu==3.0.0',
                *profile_args,
                '-i', 'https://www.paddlepaddle.org.cn/packages/stable/cu126/'
            ], "installing CUDA 12.6 PaddlePaddle")
        elif gpu_info["nvidia"]:
            print(f"  Installing PaddlePaddle GPU version...")
            _run_pip_install([
                pip, 'install', 'paddlepaddle-gpu==3.0.0',
                *profile_args,
                '-i', 'https://www.paddlepaddle.org.cn/packages/stable/cu118/'
            ], "installing CUDA PaddlePaddle")
        else:
            print(f"  Installing PaddlePaddle CPU version...")
            _run_pip_install([
                pip, 'install', 'paddlepaddle==3.0.0',
                *profile_args,
                '-i', 'https://www.paddlepaddle.org.cn/packages/stable/cpu/'
            ], "installing CPU PaddlePaddle")
        
        print(f"  [OK] PaddlePaddle installed")
        return True
    except subprocess.TimeoutExpired:
        return False
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}  ERROR: PaddlePaddle installation failed: {e}{Colors.END}")
        return False


def install_dependencies(gpu_info=None):
    """Install the selected locked profile without partial fallback."""
    print(f"\n{Colors.BLUE}[6/6]{Colors.END} Installing other dependencies...")
    
    pip = get_pip_command()
    profile_args = _profile_constraint_args(gpu_info)

    directml_requested = bool(
        gpu_info and (gpu_info.get("amd") or gpu_info.get("intel"))
    )
    if directml_requested and not _preflight_directml_distribution(pip):
        return False
    if not REQUIREMENTS_FILE.is_file():
        print(
            f"{Colors.RED}  ERROR: Required dependency list is missing: "
            f"{REQUIREMENTS_FILE}{Colors.END}"
        )
        return False

    try:
        print("  Refreshing packaging tools...")
        _run_pip_install(
            [pip, 'install', '--upgrade', 'pip', 'setuptools<82', 'wheel'],
            "refreshing packaging tools",
        )

        print(f"  Installing dependencies from {REQUIREMENTS_FILE}...")
        _run_pip_install(
            [pip, 'install', '-r', str(REQUIREMENTS_FILE), *profile_args],
            "installing requirements.txt",
        )
        print("  [OK] Requirements install command completed")

        if directml_requested:
            print("  Installing ONNX Runtime DirectML provider...")
            _run_pip_install(
                [pip, 'install', DIRECTML_PACKAGE_SPEC, *profile_args],
                "installing ONNX Runtime DirectML",
            )
            print(f"  [OK] ONNX Runtime DirectML installed")
        elif gpu_info and gpu_info.get("nvidia") and not gpu_info.get("cuda_disabled_by_python"):
            print("  Installing ONNX Runtime CUDA provider...")
            print(f"  Installing {ONNXRUNTIME_GPU_SPEC} (CUDA 13 line).")
            print("  This needs the CUDA 13 runtime, which the cu130 PyTorch")
            print("  wheel installed above supplies. On a host that must stay")
            print("  on CUDA 12, install onnxruntime-gpu 1.26.x and cu128 torch")
            print("  manually per onnxruntime.ai/docs/install.")
            _run_pip_install(
                [pip, 'install', ONNXRUNTIME_GPU_SPEC, *profile_args],
                "installing ONNX Runtime CUDA",
            )
            print(f"  [OK] ONNX Runtime CUDA provider installed")
        else:
            print("  Installing ONNX Runtime CPU provider...")
            _run_pip_install(
                [pip, 'install', ONNXRUNTIME_CPU_SPEC, *profile_args],
                "installing ONNX Runtime CPU",
            )
            print("  [OK] ONNX Runtime CPU provider installed")

        print("  [OK] Locked dependency install finished; verification pending")
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
    "venv\\Scripts\\python.exe" -m backend.dependency_profiles verify >nul 2>nul
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
    set "VSR_SETUP_PROGRESS_FILE=%TEMP%\\vsr-pro-setup-!RANDOM!-!RANDOM!.status"
    >"!VSR_SETUP_PROGRESS_FILE!" echo RUNNING^|Preparing the local runtime...^|2
    where pythonw.exe >nul 2>nul
    if not errorlevel 1 (
        start "" /b pythonw.exe "scripts\\setup_splash.py" --progress-file "!VSR_SETUP_PROGRESS_FILE!"
    )
    python setup.py --repair
    if errorlevel 1 (
        >"!VSR_SETUP_PROGRESS_FILE!" echo ERROR^|Setup failed. Review the console details.^|100
        echo.
        echo  Setup did not complete. Review the messages above, then try again.
        pause
        del /q "!VSR_SETUP_PROGRESS_FILE!" >nul 2>nul
        exit /b 1
    )
    timeout /t 1 /nobreak >nul
    del /q "!VSR_SETUP_PROGRESS_FILE!" >nul 2>nul
    set "VSR_SETUP_PROGRESS_FILE="
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
    "venv\\Scripts\\python.exe" -m backend.dependency_profiles verify >nul 2>nul
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

function Invoke-VsrProbe {
    # PS 5.1 turns a native command's redirected stderr into error records,
    # and with $ErrorActionPreference = "Stop" the first one kills the
    # script -- which used to happen precisely when the venv was broken and
    # the repair branch below was the whole point. Drop to Continue for the
    # probe and report the exit code instead.
    param([string]$Exe, [string[]]$ProbeArgs)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        if (-not (Get-Command $Exe -ErrorAction SilentlyContinue) -and
            -not (Test-Path -LiteralPath $Exe)) {
            return 1
        }
        & $Exe @ProbeArgs 2>&1 | Out-Null
        if ($null -eq $LASTEXITCODE) { return 1 }
        return $LASTEXITCODE
    } catch {
        return 1
    } finally {
        $ErrorActionPreference = $previous
    }
}

$needsRepair = -not (Test-Path ".\\venv\\Scripts\\python.exe")
if (-not $needsRepair) {
    if ((Invoke-VsrProbe ".\\venv\\Scripts\\python.exe" @("-m", "backend.dependency_profiles", "verify")) -ne 0) {
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
    $py314 = Invoke-VsrProbe "python" @("-c", "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 14) else 1)")
    if ($py314 -eq 0) {
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
    parser.add_argument(
        "--profile",
        choices=("auto", *SUPPORTED_PROFILES),
        default="auto",
        help="Use a reviewed CPU, NVIDIA, or DirectML dependency profile.",
    )
    return parser.parse_args(argv)


def _anchor_working_directory():
    """Change into setup.py's own directory.

    Returns the directory that was current before the change so callers and
    tests can restore it. A failure to change directory is reported rather
    than silently ignored, because every later path depends on it.
    """
    previous = os.getcwd()
    target = os.path.dirname(os.path.abspath(__file__))
    if os.path.normcase(previous) == os.path.normcase(target):
        return previous
    try:
        os.chdir(target)
    except OSError as exc:
        print(f"{Colors.RED}Cannot enter {target}: {exc}{Colors.END}")
        raise
    print(f"{Colors.BLUE}Working directory: {target}{Colors.END}")
    return previous


def main(argv=None):
    """Main setup function."""
    global _ACTIVE_SETUP_PROFILE
    args = parse_setup_args(argv)
    # Every path below (venv/, requirements.txt, the generated launchers) is
    # relative, so running `python C:\\path\\to\\setup.py` from another
    # directory would bootstrap into the caller's cwd and silently skip
    # requirements.txt. Anchor to this file's directory first.
    _anchor_working_directory()
    write_setup_progress("Checking Python runtime...", 8)
    print_banner()
    
    if platform.system() != "Windows":
        print(f"{Colors.YELLOW}Note: This setup is optimized for Windows.{Colors.END}")
        print(f"For Linux/macOS, manual installation may be required.\n")
    
    # Step 1: Check Python
    if not check_python():
        return _setup_failed(
            "auto",
            "python_check",
            "The Python runtime does not meet setup requirements.",
        )
    
    # Step 2: Detect GPU
    write_setup_progress("Detecting graphics hardware...", 20)
    gpu_info = _apply_profile_override(detect_gpu(), args.profile)
    profile_name = _dependency_profile_name(gpu_info)
    _ACTIVE_SETUP_PROFILE = profile_name
    try:
        ensure_profile_current(profile_name)
    except RuntimeError as exc:
        return _setup_failed(profile_name, "profile_preflight", str(exc))
    _print_profile_contract(profile_name)
    
    # Step 3: Create virtual environment
    write_setup_progress("Creating the isolated runtime...", 35)
    if not create_virtual_env(repair=args.repair):
        return _setup_failed(
            profile_name,
            "virtual_environment",
            "The repo-local virtual environment could not be created.",
        )
    if not write_setup_report(
        profile_name,
        "running",
        stage="profile_install",
        message="Installing the selected locked dependency profile.",
    ):
        return _setup_failed(
            profile_name,
            "setup_report",
            "The setup report could not be written inside the virtual environment.",
        )
    
    # Step 4: Install PyTorch
    write_setup_progress("Installing the compute runtime...", 50)
    if not install_pytorch(gpu_info):
        partial_verification = verify_installed_profile(profile_name)
        return _setup_failed(
            profile_name,
            "pytorch_install",
            "The locked PyTorch runtime could not be installed.",
            partial_verification,
        )
    resolved_profile = _dependency_profile_name(gpu_info)
    if resolved_profile != profile_name:
        profile_name = resolved_profile
        _ACTIVE_SETUP_PROFILE = profile_name
        try:
            ensure_profile_current(profile_name)
        except RuntimeError as exc:
            return _setup_failed(profile_name, "profile_preflight", str(exc))
        print("  Python compatibility changed the resolved install profile.")
        _print_profile_contract(profile_name)
    
    # Step 5: Install the default RapidOCR/ONNX dependency set. PaddleOCR and
    # its PaddlePaddle runtime remain isolated opt-ins because they install a
    # competing OpenCV wheel into the environment.
    write_setup_progress("Installing OCR and application dependencies...", 72)
    if not install_dependencies(gpu_info):
        partial_verification = verify_installed_profile(profile_name)
        return _setup_failed(
            profile_name,
            "dependency_install",
            "The locked application dependency profile could not be installed.",
            partial_verification,
        )

    write_setup_progress("Verifying the installed dependency profile...", 88)
    verification = verify_installed_profile(profile_name)
    if verification.get("valid") is not True:
        errors = [str(item) for item in verification.get("errors", []) if item]
        detail = "; ".join(errors[:4]) or "the profile verifier rejected the runtime"
        return _setup_failed(
            profile_name,
            "profile_verification",
            f"Installed profile verification failed: {detail}",
            verification,
        )
    _print_profile_verification(verification)
    
    # Check FFmpeg
    write_setup_progress("Checking video tools...", 94)
    ffmpeg_ok = check_ffmpeg()
    
    # Create launcher
    create_launcher()
    if not write_setup_report(
        profile_name,
        "verified",
        stage="complete",
        message="The locked profile passed every runtime verification check.",
        verification=verification,
    ):
        return _setup_failed(
            profile_name,
            "setup_report",
            "The verified setup report could not be written.",
            verification,
        )
    write_setup_progress("Setup complete. Starting the application...", 100, "DONE")
    
    # Done!
    print(f"\n{Colors.GREEN}{'='*60}{Colors.END}")
    print(f"{Colors.GREEN}  SETUP COMPLETE!{Colors.END}")
    print(f"{Colors.GREEN}{'='*60}{Colors.END}")
    print(f"\n  Verified profile: {Colors.BOLD}{profile_name}{Colors.END}")
    print(f"  Setup report: {Colors.BOLD}{SETUP_REPORT_PATH}{Colors.END}")
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
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        _setup_failed(
            _ACTIVE_SETUP_PROFILE,
            "cancelled",
            "Setup was cancelled.",
        )
        print(f"\n{Colors.YELLOW}Setup cancelled.{Colors.END}")
        raise SystemExit(1)
    except Exception as e:
        _setup_failed(
            _ACTIVE_SETUP_PROFILE,
            "unexpected_error",
            str(e),
        )
        print(f"\n{Colors.RED}Setup failed: {e}{Colors.END}")
        raise SystemExit(1)
