@echo off
setlocal EnableDelayedExpansion

title Video Subtitle Remover Pro (Debug)

cd /d "%~dp0"

set "VSR_SETUP_REPAIR=0"

if not exist "venv\Scripts\python.exe" (
    set "VSR_SETUP_REPAIR=1"
) else (
    "venv\Scripts\python.exe" -c "import cv2, PIL, numpy" >nul 2>nul
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

call venv\Scripts\activate.bat
echo Launching Video Subtitle Remover Pro in debug mode...
echo The console will stay open after exit so you can review logs and tracebacks.
echo.
python VideoSubtitleRemover.py

pause
