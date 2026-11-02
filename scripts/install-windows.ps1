[CmdletBinding()]
param(
    [switch]$Cpu,
    [switch]$Recreate
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
Set-Location (Split-Path $PSScriptRoot -Parent)

# Prefer the user's current Python 3.13. Older supported environments remain usable.
$selectedPython = $null
$selectedMinor = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($minor in @(13, 12, 11)) {
        try {
            $candidate = & py "-3.$minor" -c 'import sys; print(sys.executable)' 2>$null
            if ($LASTEXITCODE -eq 0 -and $candidate) {
                $selectedPython = ($candidate | Select-Object -Last 1).Trim()
                $selectedMinor = $minor
                break
            }
        } catch { }
    }
}
if (-not $selectedPython -and (Get-Command python -ErrorAction SilentlyContinue)) {
    try {
        $candidate = & python -c 'import sys; print(sys.executable) if (3,11) <= sys.version_info[:2] < (3,14) else sys.exit(1)' 2>$null
        if ($LASTEXITCODE -eq 0 -and $candidate) {
            $selectedPython = ($candidate | Select-Object -Last 1).Trim()
            $selectedMinor = [int](& $selectedPython -c 'import sys; print(sys.version_info.minor)')
        }
    } catch { }
}
if (-not $selectedPython) {
    throw 'No supported Python found. Install standard Python 3.13 x64 from python.org (3.12 and 3.11 are also supported), then rerun this installer.'
}
& $selectedPython -c "import struct, sysconfig; raise SystemExit(0 if struct.calcsize('P') == 8 and not sysconfig.get_config_var('Py_GIL_DISABLED') else 1)"
if ($LASTEXITCODE -ne 0) { throw 'Use standard 64-bit Python 3.13 (not the free-threaded or 32-bit build), then rerun this installer.' }
Write-Host "Using Python 3.$selectedMinor at $selectedPython"
. (Join-Path $PSScriptRoot 'windows-paths.ps1')
$paths = Get-PeadPaths -ProjectRoot (Get-Location).Path
Initialize-PeadPaths -Paths $paths
$venvPython = $paths.Python
Write-Host "Managed environment: $($paths.EnvironmentDir)"
if (Test-Path '.venv') {
    Write-Host 'The checkout-local .venv is left untouched and will not be used.'
}
$lock = $null
try {
    try {
        $lock = [IO.File]::Open($paths.LockFile, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    } catch { throw 'Another PEAD installation may be running. Close it before retrying.' }
    $reuseEnvironment = (Test-PeadReady -Paths $paths) -and -not $Recreate
    if ($reuseEnvironment) {
        & $venvPython -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, $selectedMinor) else 1)"
        $reuseEnvironment = $LASTEXITCODE -eq 0
    }
    # A failure at any later step must leave the launcher blocked.
    if (Test-Path -LiteralPath $paths.ReadyFile) { Remove-Item -LiteralPath $paths.ReadyFile -Force }
    if ((Test-Path -LiteralPath $paths.EnvironmentDir) -and -not $reuseEnvironment) {
        $backup = $paths.EnvironmentDir + '-old-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
        Move-Item -LiteralPath $paths.EnvironmentDir -Destination $backup
        Write-Host "Preserved the previous managed environment as $backup"
    }
    & $selectedPython -m venv $paths.EnvironmentDir
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the short-path project environment. See the error above.' }
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'pip update failed; installation is incomplete and the launcher remains blocked.' }
    $torchBuild = 'cu126'
    if ($Cpu) { $torchBuild = 'cpu' }
    $torchIndex = "https://download.pytorch.org/whl/$torchBuild"
    & $venvPython -m pip install "torch==2.10.0+$torchBuild" "torchvision==0.25.0+$torchBuild" --index-url $torchIndex
    if ($LASTEXITCODE -ne 0) {
        throw "PyTorch installation failed in $($paths.EnvironmentDir). The installer already uses short package/temp paths; the environment is not marked ready. Read the pip error immediately above."
    }
    & $venvPython -m pip install -e '.[ml,dev]'
    if ($LASTEXITCODE -ne 0) { throw 'Application dependency installation failed. Read the pip error above; the environment is not ready.' }
    & $venvPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency verification failed. The launcher remains blocked; rerun the installer with -Recreate.' }
    & $venvPython -m pead.runtime --check
    if ($LASTEXITCODE -ne 0) { throw 'Model import/runtime verification failed. The launcher remains blocked; read the error above.' }
    $ready = [ordered]@{format=1; complete=$true; project=$paths.ProjectRoot; dependencies=$paths.DependencyHash; python="3.$selectedMinor"; build=$torchBuild}
    $temporaryMarker = $paths.ReadyFile + '.tmp'
    $ready | ConvertTo-Json | Set-Content -LiteralPath $temporaryMarker -Encoding UTF8
    Move-Item -LiteralPath $temporaryMarker -Destination $paths.ReadyFile -Force
    Write-Host 'INSTALLATION COMPLETE. Run scripts/start.ps1. Source images and datasets were not moved or changed.'
} finally {
    if ($lock) { $lock.Dispose() }
}
