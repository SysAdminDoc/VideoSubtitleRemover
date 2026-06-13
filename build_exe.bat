@echo off
setlocal EnableDelayedExpansion

title Building Video Subtitle Remover Pro EXE
cd /d "%~dp0"

echo.
echo  ============================================================
echo   BUILDING VIDEO SUBTITLE REMOVER PRO
echo  ============================================================
echo.

set "PYTHON=venv\Scripts\python.exe"

:: Check for venv
if not exist "%PYTHON%" (
    echo ERROR: Virtual environment not found.
    echo Run setup.py first to create the environment.
    pause
    exit /b 1
)

:: Activate venv
call venv\Scripts\activate.bat

:: Install PyInstaller if needed
"%PYTHON%" -m pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    "%PYTHON%" -m pip install pyinstaller
    if errorlevel 1 (
        echo Failed to install PyInstaller.
        pause
        exit /b 1
    )
)

set "ICON_ARG="
if exist "icon.ico" set "ICON_ARG=--icon icon.ico"

set "DATA_ARGS=--add-data backend;backend"
if exist "assets" set "DATA_ARGS=%DATA_ARGS% --add-data assets;assets"
if exist "banner.png" set "DATA_ARGS=%DATA_ARGS% --add-data banner.png;."
if exist "icon.png" set "DATA_ARGS=%DATA_ARGS% --add-data icon.png;."
if exist "favicon.ico" set "DATA_ARGS=%DATA_ARGS% --add-data favicon.ico;."
if exist "icon.ico" set "DATA_ARGS=%DATA_ARGS% --add-data icon.ico;."
if exist "icons" set "DATA_ARGS=%DATA_ARGS% --add-data icons;icons"

set "HIDDEN_IMPORTS=--hidden-import PIL._tkinter_finder --hidden-import cv2 --hidden-import numpy --hidden-import tkinter --hidden-import tkinter.ttk --hidden-import tkinter.filedialog --hidden-import tkinter.messagebox --hidden-import simple_lama_inpainting"
echo Detecting optional runtime modules for packaging...
call :maybe_hidden_import rapidocr
call :maybe_hidden_import rapidocr_onnxruntime
call :maybe_hidden_import paddleocr
call :maybe_hidden_import easyocr

echo.
echo Building EXE (this may take several minutes)...
echo.

:: Build with PyInstaller
"%PYTHON%" -m PyInstaller --noconfirm ^
    --onedir ^
    --windowed ^
    %ICON_ARG% ^
    --name "VideoSubtitleRemoverPro" ^
    %DATA_ARGS% ^
    !HIDDEN_IMPORTS! ^
    VideoSubtitleRemover.py

if errorlevel 1 (
    echo.
    echo Build failed! Check errors above.
    pause
    exit /b 1
)

set "DIST_DIR=dist\VideoSubtitleRemoverPro"
if exist "!DIST_DIR!" (
    for %%F in (README.md LICENSE CHANGELOG.md) do (
        if exist "%%F" copy /Y "%%F" "!DIST_DIR!\%%F" >nul
    )
)

echo.
echo ============================================================
echo  BUILD COMPLETE!
echo ============================================================
echo.
echo  EXE Location: !DIST_DIR!\
echo  Bundle docs: README.md, LICENSE, CHANGELOG.md
echo.
echo  To distribute, zip the entire VideoSubtitleRemoverPro folder.
echo.
pause
exit /b 0

:maybe_hidden_import
"%PYTHON%" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec(r'%~1') else 1)" >nul 2>&1
if not errorlevel 1 (
    set "HIDDEN_IMPORTS=!HIDDEN_IMPORTS! --hidden-import %~1"
    echo   Including optional module: %~1
) else (
    echo   Optional module not installed, skipping: %~1
)
exit /b 0
