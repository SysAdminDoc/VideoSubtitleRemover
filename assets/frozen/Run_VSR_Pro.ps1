# Native launcher for the frozen Video Subtitle Remover Pro distribution.
$ErrorActionPreference = "Stop"
$exe = Join-Path $PSScriptRoot "VideoSubtitleRemoverPro.exe"

if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    Write-Error "VideoSubtitleRemoverPro.exe is missing from this folder."
    exit 1
}

if ($env:VSR_LAUNCHER_WAIT -eq "1") {
    & $exe @args
    exit $LASTEXITCODE
}

Start-Process -FilePath $exe -ArgumentList $args -WorkingDirectory $PSScriptRoot
