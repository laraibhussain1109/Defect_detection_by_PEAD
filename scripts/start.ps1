$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$venvPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    throw 'The project virtual environment is missing. Run: powershell -ExecutionPolicy Bypass -File scripts/install-windows.ps1'
}
& $venvPython -m pead.runtime --check
if ($LASTEXITCODE -ne 0) { throw 'Runtime verification failed. Run scripts/install-windows.ps1 to repair the local environment, then use this launcher. Do not run global Python 3.13.' }
& $venvPython -m uvicorn pead.app:app --host 127.0.0.1 --port 8000
if ($LASTEXITCODE -ne 0) { throw 'PEAD exited with an error; see the terminal output above.' }
