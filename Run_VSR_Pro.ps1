# Video Subtitle Remover Pro Launcher
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Invoke-VsrProbe {
    # PS 5.1 turns a native command's redirected stderr into error records,
    # and with $ErrorActionPreference = "Stop" the first one kills the
    # script -- which used to happen precisely when the venv was broken and
    # the repair branch below was the whole point. Drop to Continue for the
    # probe and report the exit code instead.
    param([string]$Exe, [string[]]$ProbeArgs)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        if (-not (Get-Command $Exe -ErrorAction SilentlyContinue) -and
            -not (Test-Path -LiteralPath $Exe)) {
            return 1
        }
        & $Exe @ProbeArgs 2>&1 | Out-Null
        if ($null -eq $LASTEXITCODE) { return 1 }
        return $LASTEXITCODE
    } catch {
        return 1
    } finally {
        $ErrorActionPreference = $previous
    }
}

$needsRepair = -not (Test-Path ".\venv\Scripts\python.exe")
if (-not $needsRepair) {
    if ((Invoke-VsrProbe ".\venv\Scripts\python.exe" @("-m", "backend.dependency_profiles", "verify")) -ne 0) {
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
    $py314 = Invoke-VsrProbe "python" @("-c", "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 14) else 1)")
    if ($py314 -eq 0) {
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
