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

if ($args.Count -gt 0) {
    # -ArgumentList rejects an empty collection (ValidateNotNullOrEmpty), so
    # the plain no-argument launch -- the normal case -- used to fail here.
    Start-Process -FilePath $exe -ArgumentList $args -WorkingDirectory $PSScriptRoot
} else {
    Start-Process -FilePath $exe -WorkingDirectory $PSScriptRoot
}
