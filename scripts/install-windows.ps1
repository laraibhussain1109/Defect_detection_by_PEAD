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
$venvPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (Test-Path '.venv') {
    $reuseEnvironment = $false
    if ((Test-Path $venvPython) -and -not $Recreate) {
        try {
            & $venvPython -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, $selectedMinor) else 1)"
            if ($LASTEXITCODE -eq 0) {
                & $venvPython -c 'from pead.runtime import validate_runtime; validate_runtime()' 2>$null
                if ($LASTEXITCODE -eq 0) {
                    & $venvPython -m pip check
                    $reuseEnvironment = $LASTEXITCODE -eq 0
                }
            }
        } catch { }
    }
    if (-not $reuseEnvironment) {
        $backup = '.venv-backup-' + [guid]::NewGuid().ToString('N')
        Move-Item -LiteralPath '.venv' -Destination $backup
        Write-Host "Preserved the old virtual environment as $backup"
    }
}
& $selectedPython -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Could not create the project virtual environment.' }
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip update failed.' }
$torchBuild = 'cu126'
if ($Cpu) { $torchBuild = 'cpu' }
$torchIndex = "https://download.pytorch.org/whl/$torchBuild"
& $venvPython -m pip install "torch==2.10.0+$torchBuild" "torchvision==0.25.0+$torchBuild" --index-url $torchIndex
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed. Check network access and the Python version shown above.' }
& $venvPython -m pip install -e '.[ml,dev]'
if ($LASTEXITCODE -ne 0) { throw 'Application installation failed.' }
& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Installed dependency check failed. Rerun this installer with -Recreate to rebuild the project environment.' }
& $venvPython -m pead.runtime --check
if ($LASTEXITCODE -ne 0) { throw 'Runtime verification failed. See the error above; the installer supports Python 3.13.' }
Write-Host 'Installation verified. Start with scripts/start.ps1. Your global Python installation and datasets were not changed.'
