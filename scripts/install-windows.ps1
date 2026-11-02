$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw 'Install Python 3.11 x64 from python.org with the Python launcher enabled, then run this installer again.'
}
& py -3.11 -c 'import sys; assert sys.version_info[:2] == (3, 11)'
if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 x64 is required. Install it from python.org, then run this installer again. The global Python 3.13 installation is not supported.' }
$venvPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (Test-Path '.venv') {
    $correctVersion = $false
    if (Test-Path $venvPython) {
        & $venvPython -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)'
        $correctVersion = $LASTEXITCODE -eq 0
    }
    if (-not $correctVersion) {
        $backup = '.venv-backup-' + [guid]::NewGuid().ToString('N')
        Move-Item -LiteralPath '.venv' -Destination $backup
        Write-Host "Preserved the incompatible virtual environment as $backup"
    }
}
& py -3.11 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Could not create the project Python 3.11 virtual environment.' }
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip update failed' }
& $venvPython -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
& $venvPython -m pip install -e '.[ml,dev]'
if ($LASTEXITCODE -ne 0) { throw 'Application installation failed' }
& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Installed dependency compatibility check failed.' }
& $venvPython -m pead.runtime --check
if ($LASTEXITCODE -ne 0) { throw 'Runtime verification failed. Use the project Python 3.11 environment, not global Python 3.13.' }
Write-Host 'Installation verified. Start the application with scripts/start.ps1. Administrator access is not required.'
