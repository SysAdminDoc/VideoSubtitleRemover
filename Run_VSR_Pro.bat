@echo off
setlocal EnableDelayedExpansion

title Video Subtitle Remover Pro

:: Change to script directory
cd /d "%~dp0"

:: Check if venv exists
if not exist "venv\Scripts\python.exe" (
    echo.
    echo  ╔═══════════════════════════════════════════════════════════╗
    echo  ║  VIDEO SUBTITLE REMOVER PRO                               ║
    echo  ╚═══════════════════════════════════════════════════════════╝
    echo.
    echo  First-time setup required!
    echo  Running setup.py...
    echo.
    python setup.py
    if errorlevel 1 (
        echo.
        echo  Setup failed. Please check the errors above.
        pause
        exit /b 1
    )
)

:: Activate virtual environment and run
call venv\Scripts\activate.bat
python VideoSubtitleRemover.py

pause
