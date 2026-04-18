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
