@echo off
setlocal EnableExtensions

title Video Subtitle Remover Pro (Frozen Diagnostics)
cd /d "%~dp0"
set "VSR_EXE=%~dp0VideoSubtitleRemoverPro.exe"

if not exist "%VSR_EXE%" (
    echo ERROR: VideoSubtitleRemoverPro.exe is missing from this folder.
    exit /b 1
)

echo Launching the frozen application and waiting for it to exit...
echo Diagnostic log: %APPDATA%\VideoSubtitleRemoverPro\vsr_pro.log
echo.
"%VSR_EXE%" %*
set "VSR_EXIT=%ERRORLEVEL%"

echo.
echo Video Subtitle Remover Pro exited with code %VSR_EXIT%.
if /I not "%VSR_LAUNCHER_SMOKE%"=="1" pause
exit /b %VSR_EXIT%
