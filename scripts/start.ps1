$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
Set-Location (Split-Path $PSScriptRoot -Parent)
$venvPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    throw 'The project virtual environment is missing. Run: powershell -ExecutionPolicy Bypass -File scripts/install-windows.ps1'
}
& $venvPython -m pead.runtime --check
if ($LASTEXITCODE -ne 0) { throw 'Runtime verification failed. Run scripts/install-windows.ps1 to update the local environment, then restart using this launcher. Python 3.11-3.13 is supported.' }
& $venvPython -m uvicorn pead.app:app --host 127.0.0.1 --port 8000
if ($LASTEXITCODE -ne 0) { throw 'PEAD exited with an error; see the terminal output above.' }
