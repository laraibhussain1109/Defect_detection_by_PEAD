$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
Set-Location (Split-Path $PSScriptRoot -Parent)
. (Join-Path $PSScriptRoot 'windows-paths.ps1')
$paths = Get-PeadPaths -ProjectRoot (Get-Location).Path
if (-not (Test-PeadReady -Paths $paths)) {
    throw "Installation is incomplete or outdated at $($paths.EnvironmentDir). Run powershell -ExecutionPolicy Bypass -File scripts/install-windows.ps1 and wait for INSTALLATION COMPLETE. The launcher will not use the old .venv or global Python."
}
Initialize-PeadPaths -Paths $paths
$venvPython = $paths.Python
Write-Host "Starting with $venvPython"
& $venvPython -m pead.runtime --check
if ($LASTEXITCODE -ne 0) { throw 'Runtime verification failed. Rerun scripts/install-windows.ps1; the application was not started.' }
& $venvPython -m uvicorn pead.app:app --host 127.0.0.1 --port 8000
if ($LASTEXITCODE -ne 0) { throw 'PEAD exited with an error; see the terminal output above.' }
