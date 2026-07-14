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
    exit /b 1
)

:: Activate venv
call venv\Scripts\activate.bat

:: Install/upgrade PyInstaller. >=6.10.0 carries the fix for CVE-2025-59042
:: (writable-CWD bootstrap local privilege escalation); older bootloaders let
:: an attacker inject Python via sys.path beside the frozen exe.
echo Ensuring release tooling...
"%PYTHON%" -m pip install "pyinstaller>=6.10.0" "pip-audit>=2.10.0"
if errorlevel 1 (
    echo Failed to install PyInstaller or pip-audit.
    exit /b 1
)

echo Checking reviewed dependency profiles...
"%PYTHON%" -m backend.dependency_profiles check
if errorlevel 1 (
    echo Dependency profile files are stale. Regenerate and review them first.
    exit /b 1
)

echo.
echo Running the complete test suite...
"%PYTHON%" -m unittest discover -s tests -q
if errorlevel 1 (
    echo Test suite failed; release build stopped.
    exit /b 1
)

set "ICON_ARG="
if exist "icon.ico" set "ICON_ARG=--icon icon.ico"
set "RUNTIME_HOOKS=--runtime-hook assets\runtime_hook_mp.py"

set "DATA_ARGS=--add-data backend;backend"
if exist "assets" set "DATA_ARGS=%DATA_ARGS% --add-data assets;assets"
if exist "banner.png" set "DATA_ARGS=%DATA_ARGS% --add-data banner.png;."
if exist "icon.png" set "DATA_ARGS=%DATA_ARGS% --add-data icon.png;."
if exist "favicon.ico" set "DATA_ARGS=%DATA_ARGS% --add-data favicon.ico;."
if exist "icon.ico" set "DATA_ARGS=%DATA_ARGS% --add-data icon.ico;."
if exist "icons" set "DATA_ARGS=%DATA_ARGS% --add-data icons;icons"
if exist "locale" set "DATA_ARGS=%DATA_ARGS% --add-data locale;locale"

set "HIDDEN_IMPORTS=--hidden-import PIL._tkinter_finder --hidden-import cv2 --hidden-import numpy --hidden-import tkinter --hidden-import tkinter.ttk --hidden-import tkinter.filedialog --hidden-import tkinter.messagebox"
set "EXCLUDES="
echo Detecting optional runtime modules for packaging...
call :maybe_hidden_import rapidocr
call :maybe_hidden_import rapidocr_onnxruntime
if /I "%VSR_ENABLE_FULL_OCR%"=="1" (
    call :maybe_hidden_import paddleocr
    call :maybe_hidden_import easyocr
) else (
    set "EXCLUDES=!EXCLUDES! --exclude-module paddle --exclude-module paddleocr --exclude-module easyocr"
    echo   Heavy PaddleOCR/EasyOCR fallbacks disabled; set VSR_ENABLE_FULL_OCR=1 to include them.
)
if /I "%VSR_ENABLE_PYTORCH_LAMA%"=="1" (
    call :maybe_hidden_import simple_lama_inpainting
) else (
    set "EXCLUDES=!EXCLUDES! --exclude-module simple_lama_inpainting"
    echo   PyTorch LaMa fallback disabled for packaging; set VSR_ENABLE_PYTORCH_LAMA=1 to include it.
)
if /I not "%VSR_ENABLE_FULL_OCR%"=="1" if /I not "%VSR_ENABLE_PYTORCH_LAMA%"=="1" (
    set "EXCLUDES=!EXCLUDES! --exclude-module torch --exclude-module torchvision"
    echo   PyTorch runtime disabled because no selected packaged feature requires it.
)

rem Collect data files for OCR packages.
set "COLLECT_DATA="
call :maybe_collect_data rapidocr
call :maybe_collect_data rapidocr_onnxruntime

echo.
echo Building EXE (this may take several minutes)...
echo.

:: Build with PyInstaller
"%PYTHON%" -m PyInstaller --noconfirm ^
    --onedir ^
    --windowed ^
    %ICON_ARG% ^
    --name "VideoSubtitleRemoverPro" ^
    !RUNTIME_HOOKS! ^
    %DATA_ARGS% ^
    !HIDDEN_IMPORTS! ^
    !EXCLUDES! ^
    !COLLECT_DATA! ^
    VideoSubtitleRemover.py

if errorlevel 1 (
    echo.
    echo Build failed! Check errors above.
    exit /b 1
)

set "DIST_DIR=dist\VideoSubtitleRemoverPro"
if exist "!DIST_DIR!" (
    for %%F in (README.md LICENSE CHANGELOG.md) do (
        if exist "%%F" copy /Y "%%F" "!DIST_DIR!\%%F" >nul
    )
    for %%F in (Run_VSR_Pro.bat Run_VSR_Pro_Debug.bat Run_VSR_Pro.ps1) do (
        if not exist "assets\frozen\%%F" (
            echo ERROR: Frozen launcher asset missing: assets\frozen\%%F
            exit /b 1
        )
        copy /Y "assets\frozen\%%F" "!DIST_DIR!\%%F" >nul
        if errorlevel 1 (
            echo ERROR: Failed to bundle frozen launcher: %%F
            exit /b 1
        )
    )
)

set "ANALYSIS_PATH=build\VideoSubtitleRemoverPro\Analysis-00.toc"
if not exist "!ANALYSIS_PATH!" (
    echo ERROR: PyInstaller analysis evidence missing: !ANALYSIS_PATH!
    exit /b 1
)

set "MAKENSIS="
for /f "delims=" %%I in ('where makensis.exe 2^>nul') do if not defined MAKENSIS set "MAKENSIS=%%I"
if not defined MAKENSIS if exist "%ProgramFiles(x86)%\NSIS\makensis.exe" set "MAKENSIS=%ProgramFiles(x86)%\NSIS\makensis.exe"
if not defined MAKENSIS if exist "%ProgramFiles%\NSIS\makensis.exe" set "MAKENSIS=%ProgramFiles%\NSIS\makensis.exe"
if not defined MAKENSIS (
    echo ERROR: NSIS 3.12 or newer is required to produce release artifacts.
    exit /b 1
)

set "RELEASE_DIR=!CD!\build\release"
if not exist "!RELEASE_DIR!" mkdir "!RELEASE_DIR!"
set "INSTALLER_PATH=!CD!\VideoSubtitleRemoverPro-Setup.exe"
set "INSTALLER_STAGE=!RELEASE_DIR!\VideoSubtitleRemoverPro-Setup.exe"
set "SMOKE_INSTALLER=!RELEASE_DIR!\VideoSubtitleRemoverPro-Smoke-Setup.exe"
set "SMOKE_INSTALL_DIR=!RELEASE_DIR!\installer-smoke"
if exist "!INSTALLER_PATH!" del /q "!INSTALLER_PATH!"

echo.
echo Compiling the production NSIS installer...
"!MAKENSIS!" "/DOUTPUT_DIR=!RELEASE_DIR!" "/DDIST_DIR=!CD!\!DIST_DIR!" installer\vsr.nsi
if errorlevel 1 exit /b 1

echo Compiling and extracting the non-elevated installer smoke harness...
"!MAKENSIS!" /DVSR_SMOKE_BUILD=1 "/DOUTPUT_DIR=!RELEASE_DIR!" "/DDIST_DIR=!CD!\!DIST_DIR!" installer\vsr.nsi
if errorlevel 1 exit /b 1
if exist "!SMOKE_INSTALL_DIR!" rmdir /s /q "!SMOKE_INSTALL_DIR!"
"!SMOKE_INSTALLER!" /S "/D=!SMOKE_INSTALL_DIR!"
if errorlevel 1 (
    echo ERROR: Installer smoke extraction failed.
    exit /b 1
)
if not exist "!SMOKE_INSTALL_DIR!\VideoSubtitleRemoverPro.exe" (
    echo ERROR: Installer smoke payload is missing the frozen executable.
    exit /b 1
)

echo.
echo Generating local release evidence...
"%PYTHON%" -m backend.release_verification ^
    --dist-dir "!DIST_DIR!" ^
    --analysis-path "!ANALYSIS_PATH!" ^
    --installer-path "!INSTALLER_STAGE!" ^
    --installer-smoke-executable "!SMOKE_INSTALL_DIR!\VideoSubtitleRemoverPro.exe" ^
    --hidden-imports "!HIDDEN_IMPORTS!" ^
    --runtime-hooks "!RUNTIME_HOOKS!" ^
    --excludes "!EXCLUDES!" ^
    --collect-data "!COLLECT_DATA!" ^
    --run-reference-corpus ^
    --run-dependency-audit ^
    --quality strict

if errorlevel 1 (
    echo.
    echo Release evidence generation failed.
    exit /b 1
)

copy /Y "!INSTALLER_STAGE!" "!INSTALLER_PATH!" >nul
if errorlevel 1 (
    echo ERROR: Strict proof passed but the installer could not be promoted.
    exit /b 1
)

echo.
echo ============================================================
echo  BUILD COMPLETE!
echo ============================================================
echo.
echo  EXE Location: !DIST_DIR!\
echo  Bundle docs: README.md, LICENSE, CHANGELOG.md
echo  Bundle launchers: Run_VSR_Pro.bat, Run_VSR_Pro_Debug.bat, Run_VSR_Pro.ps1
echo  Installer: !INSTALLER_PATH!
echo  Release evidence: release-verification.json, release-hidden-imports.json, release-advisories.json, pip-audit.json, sbom.cdx.json
echo.
echo  To distribute, zip the entire VideoSubtitleRemoverPro folder.
echo.
exit /b 0

:maybe_hidden_import
"%PYTHON%" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec(r'%~1') else 1)" >nul 2>&1
if errorlevel 1 goto hidden_import_skip
set "HIDDEN_IMPORTS=!HIDDEN_IMPORTS! --hidden-import %~1"
echo   Including optional module: %~1
exit /b 0
:hidden_import_skip
echo   Optional module not installed, skipping: %~1
exit /b 0

:maybe_collect_data
"%PYTHON%" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec(r'%~1') else 1)" >nul 2>&1
if errorlevel 1 goto collect_data_skip
set "COLLECT_DATA=!COLLECT_DATA! --collect-data %~1"
echo   Collecting data files for: %~1
exit /b 0
:collect_data_skip
echo   Optional data collection skipped (not installed): %~1
exit /b 0
