@echo off
title Building Video Subtitle Remover Pro EXE
cd /d "%~dp0"

echo.
echo  ╔═══════════════════════════════════════════════════════════╗
echo  ║  BUILDING VIDEO SUBTITLE REMOVER PRO                      ║
echo  ╚═══════════════════════════════════════════════════════════╝
echo.

:: Check for venv
if not exist "venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found.
    echo Run setup.py first to create the environment.
    pause
    exit /b 1
)

:: Activate venv
call venv\Scripts\activate.bat

:: Install PyInstaller if needed
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

:: Create assets folder if missing
if not exist "assets" mkdir assets

echo.
echo Building EXE (this may take several minutes)...
echo.

:: Build with PyInstaller - simplified for compatibility
pyinstaller --noconfirm ^
    --onedir ^
    --windowed ^
    --name "VideoSubtitleRemoverPro" ^
    --add-data "backend;backend" ^
    --hidden-import "PIL._tkinter_finder" ^
    --hidden-import "cv2" ^
    --hidden-import "numpy" ^
    --hidden-import "tkinter" ^
    --hidden-import "tkinter.ttk" ^
    --hidden-import "tkinter.filedialog" ^
    --hidden-import "tkinter.messagebox" ^
    VideoSubtitleRemover.py

if errorlevel 1 (
    echo.
    echo Build failed! Check errors above.
    pause
    exit /b 1
)

echo.
echo ═══════════════════════════════════════════════════════════════
echo  BUILD COMPLETE!
echo ═══════════════════════════════════════════════════════════════
echo.
echo  EXE Location: dist\VideoSubtitleRemoverPro\
echo.
echo  To distribute, zip the entire VideoSubtitleRemoverPro folder.
echo.
pause
