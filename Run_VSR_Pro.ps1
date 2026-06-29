# Video Subtitle Remover Pro Launcher
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$needsRepair = -not (Test-Path ".\venv\Scripts\python.exe")
if (-not $needsRepair) {
    & ".\venv\Scripts\python.exe" -c "import cv2, PIL, numpy" 1>$null 2>$null
    if ($LASTEXITCODE -ne 0) {
        $needsRepair = $true
    }
}

if ($needsRepair) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host " VIDEO SUBTITLE REMOVER PRO" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Runtime setup or repair required." -ForegroundColor Yellow
    Write-Host "Preparing the runtime and dependencies without prompts..." -ForegroundColor Yellow
    Write-Host ""
    python -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 14) else 1)" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "WARNING: Python 3.14+ cannot install Windows CUDA PyTorch wheels." -ForegroundColor Yellow
        Write-Host "Use Python 3.12 or 3.13 for NVIDIA GPU acceleration." -ForegroundColor Yellow
        Write-Host "Set VSR_ALLOW_PY314_CPU=1 before launch only for CPU-only setup." -ForegroundColor Yellow
        Write-Host ""
    }
    python setup.py --repair
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "Setup did not complete. Review the messages above, then try again." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit $LASTEXITCODE
    }
}

if (Test-Path ".\venv\Scripts\pythonw.exe") {
    Start-Process -FilePath ".\venv\Scripts\pythonw.exe" -ArgumentList "VideoSubtitleRemover.py"
    exit 0
}

if (Test-Path ".\venv\Scripts\python.exe") {
    Start-Process -FilePath ".\venv\Scripts\python.exe" -ArgumentList "VideoSubtitleRemover.py"
    exit 0
}

Write-Host "The Python runtime could not be found in the virtual environment." -ForegroundColor Yellow
Read-Host "Press Enter to exit"
exit 1
