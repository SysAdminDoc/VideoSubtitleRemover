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

REM RM-288: pin the two settings that make a rebuild comparable. PyInstaller
REM stamps the PE timestamp from SOURCE_DATE_EPOCH; PYTHONHASHSEED fixes the
REM hash randomisation the build runs under. Both are recorded in the release
REM evidence. This does NOT make the build bit-for-bit reproducible -- it is
REM not path-invariant -- it makes two rebuilds comparable by content.
if not defined SOURCE_DATE_EPOCH set "SOURCE_DATE_EPOCH=1735689600"
if not defined PYTHONHASHSEED set "PYTHONHASHSEED=0"

:: Check for venv
if not exist "%PYTHON%" (
    echo ERROR: Virtual environment not found.
    echo Run setup.py first to create the environment.
    exit /b 1
)

:: Activate venv
call venv\Scripts\activate.bat

:: Install/upgrade PyInstaller. >=6.22.2 is the floor. 6.10.0 carried the fix
:: for CVE-2025-59042 (writable-CWD bootstrap local privilege escalation);
:: GHSA-9fxf-4qw3-ghmr was patched in 6.22.1 and does not apply to this
:: onedir/asInvoker build, so the raise is floor hygiene, not a live exposure.
echo Ensuring release tooling...
"%PYTHON%" -m pip install "pyinstaller>=6.22.2" "pip-audit>=2.10.0" "pytest>=9.0.0" "ruff==0.15.20"
if errorlevel 1 (
    echo Failed to install PyInstaller, pip-audit, pytest, or Ruff.
    exit /b 1
)

echo Checking reviewed dependency profiles...
"%PYTHON%" -m backend.dependency_profiles check
if errorlevel 1 (
    echo Dependency profile files are stale. Regenerate and review them first.
    exit /b 1
)

echo Checking Python source hygiene...
"%PYTHON%" -m ruff check backend gui scripts VideoSubtitleRemover.py --no-cache
if errorlevel 1 (
    echo Ruff found source violations; release build stopped.
    exit /b 1
)

echo Checking generated CLI and config reference...
"%PYTHON%" scripts\generate_cli_reference.py
if errorlevel 1 (
    echo README CLI/config reference drifted; release build stopped.
    exit /b 1
)

echo Checking the architecture module map...
"%PYTHON%" scripts\generate_architecture_map.py --check
if errorlevel 1 (
    echo Architecture module map drifted; release build stopped.
    exit /b 1
)

echo Checking gettext catalogs...
"%PYTHON%" scripts\i18n_catalogs.py check
if errorlevel 1 (
    echo Gettext catalogs drifted or failed validation; release build stopped.
    exit /b 1
)

echo.
echo Running the complete test suite...
:: pytest, not `unittest discover`: 12 test modules are written as bare
:: module-level test functions, which unittest imports and then collects
:: ZERO tests from -- silently, reporting OK. pytest collects those plus
:: every TestCase class, so its coverage is a strict superset.
"%PYTHON%" -m pytest tests -q
if errorlevel 1 (
    echo Test suite failed; release build stopped.
    exit /b 1
)

:: Opt-in proof for the committed spec itself. The normal release build below
:: has its own frozen smoke; this slower lane catches spec-only drift too.
if /I "%VSR_VERIFY_SPEC_BUILD%"=="1" (
    "%PYTHON%" -m scripts.frozen_build_smoke
    if errorlevel 1 (
        echo Committed PyInstaller spec smoke failed.
        exit /b 1
    )
)

set "RUNTIME_HOOKS=--runtime-hook assets\runtime_hook_mp.py"

:: The build is spec-driven, so data/icon arguments live in
:: VideoSubtitleRemoverPro.spec. The bookkeeping below exists only to record
:: what the spec did in the release evidence, so its feature gates MUST agree
:: with the spec's _enabled() helper, which accepts 1/true/yes/on.
call :truthy VSR_ENABLE_FULL_OCR FULL_OCR_ON
call :truthy VSR_ENABLE_PYTORCH_LAMA PYTORCH_LAMA_ON

set "HIDDEN_IMPORTS=--hidden-import PIL._tkinter_finder --hidden-import cv2 --collect-all numpy --hidden-import backend.opencv_ocr --hidden-import tkinter --hidden-import tkinter.ttk --hidden-import tkinter.filedialog --hidden-import tkinter.messagebox"
set "EXCLUDES="
echo Detecting optional runtime modules for packaging...
call :maybe_hidden_import rapidocr
call :maybe_hidden_import rapidocr_onnxruntime
if "!FULL_OCR_ON!"=="1" (
    call :maybe_hidden_import paddleocr
    call :maybe_hidden_import easyocr
) else (
    set "EXCLUDES=!EXCLUDES! --exclude-module paddle --exclude-module paddleocr --exclude-module easyocr"
    echo   Heavy PaddleOCR/EasyOCR fallbacks disabled; set VSR_ENABLE_FULL_OCR=1 to include them.
)
if "!PYTORCH_LAMA_ON!"=="1" (
    call :maybe_hidden_import simple_lama_inpainting
) else (
    set "EXCLUDES=!EXCLUDES! --exclude-module simple_lama_inpainting"
    echo   PyTorch LaMa fallback disabled for packaging; set VSR_ENABLE_PYTORCH_LAMA=1 to include it.
)
if not "!FULL_OCR_ON!"=="1" if not "!PYTORCH_LAMA_ON!"=="1" (
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

:: Build the production artifact from the reviewed, tracked spec. The spec
:: reads the same optional-profile environment gates used above.
"%PYTHON%" -m PyInstaller --noconfirm --clean VideoSubtitleRemoverPro.spec

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
:: NSIS requires /D= to be the LAST parameter, UNQUOTED, and takes the rest
:: of the command line literally. The previous Start-Process form happened to
:: satisfy that only because Windows PowerShell 5.1 joins -ArgumentList
:: without adding quotes (measured 2026-08-20 via GetCommandLineW); a shell
:: that quoted a spaced argument -- pwsh 7, or any future 5.1 fix -- would
:: silently extract to the default directory instead. Invoke it from cmd so
:: the tail is passed verbatim by design rather than by accident. The empty
:: first argument is the window title `start` would otherwise consume.
start "" /wait "!SMOKE_INSTALLER!" /S /D=!SMOKE_INSTALL_DIR!
if errorlevel 1 (
    echo ERROR: Installer smoke extraction failed.
    exit /b 1
)
if not exist "!SMOKE_INSTALL_DIR!\VideoSubtitleRemoverPro.exe" (
    echo ERROR: Installer smoke did not extract to the requested directory:
    echo        !SMOKE_INSTALL_DIR!
    echo        The installer reported success, so /D= was most likely not
    echo        honoured and the payload landed in the default location.
    echo        /D= must stay last and unquoted on the command line.
    exit /b 1
)

echo.
echo Generating local release evidence...
set "VSR_SMOKE_LOCALE=qps-Ploc"
"%PYTHON%" -m backend.release_verification ^
    --dist-dir "!DIST_DIR!" ^
    --evidence-dir "!RELEASE_DIR!" ^
    --analysis-path "!ANALYSIS_PATH!" ^
    --installer-path "!INSTALLER_STAGE!" ^
    --installer-smoke-executable "!SMOKE_INSTALL_DIR!\VideoSubtitleRemoverPro.exe" ^
    --hidden-imports "!HIDDEN_IMPORTS!" ^
    --runtime-hooks "!RUNTIME_HOOKS!" ^
    --excludes "!EXCLUDES!" ^
    --collect-data "!COLLECT_DATA!" ^
    --run-reference-corpus ^
    --run-dependency-audit ^
    --run-ui-release-probes ^
    --quality strict
set "RELEASE_EXIT=!ERRORLEVEL!"
set "VSR_SMOKE_LOCALE="

if not "!RELEASE_EXIT!"=="0" (
    echo.
    echo Release evidence generation failed.
    exit /b 1
)

:: Stage the whole release into a clean temporary directory, derive every
:: filename from APP_VERSION, hash exactly that set, and promote it in one
:: move. Nothing is published from the reusable scratch directory.
echo.
echo Staging the versioned release artifact set...
:: %PYTHON% is the space-free venv path; quoting it here makes cmd
:: strip the outer quote pair and mangle the whole command.
for /f "delims=" %%V in ('%PYTHON% -c "from gui.config import APP_VERSION; print(APP_VERSION)"') do set "APP_VERSION=%%V"
if not defined APP_VERSION (
    echo ERROR: Could not read APP_VERSION.
    exit /b 1
)
"%PYTHON%" -m backend.release_staging stage ^
    --version "!APP_VERSION!" ^
    --dist-dir "!DIST_DIR!" ^
    --installer-path "!INSTALLER_STAGE!" ^
    --evidence-dir "!RELEASE_DIR!" ^
    --release-root "!RELEASE_DIR!" ^
    --prune-stale
if errorlevel 1 (
    echo ERROR: Release staging failed; nothing was promoted.
    exit /b 1
)

"%PYTHON%" -m backend.release_staging verify ^
    --version "!APP_VERSION!" ^
    --release-root "!RELEASE_DIR!" >nul
if errorlevel 1 (
    echo ERROR: Promoted release set did not verify.
    exit /b 1
)

:: Promote from the verified versioned set, not from the scratch stage --
:: --prune-stale removes the loose pre-versioned copies.
copy /Y "!RELEASE_DIR!\!APP_VERSION!\VideoSubtitleRemoverPro-!APP_VERSION!-Setup.exe" "!INSTALLER_PATH!" >nul
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
echo  Release set: !RELEASE_DIR!\!APP_VERSION!\ (installer, portable ZIP, evidence, SHA256SUMS.txt)
echo.
echo  Publish with:
"%PYTHON%" -m backend.release_staging guidance --version "!APP_VERSION!"
echo.
exit /b 0

:truthy
:: Normalize an environment gate to 1/0 using the same accepted spellings as
:: VideoSubtitleRemoverPro.spec's _enabled(): 1, true, yes, on (any case).
:: %~1 = environment variable name, %~2 = output variable name.
set "_TRUTHY_VALUE=!%~1!"
set "%~2=0"
if /I "!_TRUTHY_VALUE!"=="1" set "%~2=1"
if /I "!_TRUTHY_VALUE!"=="true" set "%~2=1"
if /I "!_TRUTHY_VALUE!"=="yes" set "%~2=1"
if /I "!_TRUTHY_VALUE!"=="on" set "%~2=1"
set "_TRUTHY_VALUE="
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
