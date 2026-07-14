@echo off
setlocal EnableExtensions

cd /d "%~dp0"
set "VSR_EXE=%~dp0VideoSubtitleRemoverPro.exe"

if not exist "%VSR_EXE%" (
    echo ERROR: VideoSubtitleRemoverPro.exe is missing from this folder.
    exit /b 1
)

if /I "%VSR_LAUNCHER_WAIT%"=="1" goto launch_wait

start "" "%VSR_EXE%" %*
exit /b %ERRORLEVEL%

:launch_wait
"%VSR_EXE%" %*
exit /b %ERRORLEVEL%
