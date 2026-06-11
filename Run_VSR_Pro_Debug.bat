@echo off
setlocal EnableDelayedExpansion

title Video Subtitle Remover Pro (Debug)

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
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

call venv\Scripts\activate.bat
echo Launching Video Subtitle Remover Pro in debug mode...
echo The console will stay open after exit so you can review logs and tracebacks.
echo.
python VideoSubtitleRemover.py

pause
