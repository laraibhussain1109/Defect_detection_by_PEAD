# Execute with portable PowerShell on Linux. External Python/pip commands are
# controlled fixtures; the installer, launcher, and shared resolver are real.
# This verifies script behavior without administrator access or package downloads.
[CmdletBinding()]
param([string]$ProjectRoot = (Split-Path $PSScriptRoot -Parent))

$ErrorActionPreference = 'Stop'
$script:Assertions = 0
function Assert-True([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw "Assertion failed: $Message" }
    $script:Assertions += 1
}

$testRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('pead-ps-' + [guid]::NewGuid().ToString('N').Substring(0,8))
New-Item -ItemType Directory -Path $testRoot | Out-Null
$project = Join-Path $testRoot (('deep-checkout-' * 10) + 'repo')
New-Item -ItemType Directory -Path (Join-Path $project 'scripts') -Force | Out-Null
Copy-Item (Join-Path $ProjectRoot 'scripts/*.ps1') (Join-Path $project 'scripts')
Copy-Item (Join-Path $ProjectRoot 'pyproject.toml') (Join-Path $project 'pyproject.toml')
$env:USERPROFILE = Join-Path $testRoot 'u'
New-Item -ItemType Directory -Path $env:USERPROFILE -Force | Out-Null
$env:PEAD_RUNTIME_ROOT = $null
$env:PEAD_TEST_LOG = Join-Path $testRoot 'native-commands.log'
$basePython = Join-Path $testRoot 'python313.exe'
$env:PEAD_TEST_BASE_PYTHON = $basePython
$mockPython = @'
#!/bin/sh
set -eu
printf '%s\n' "$0 $*" >> "$PEAD_TEST_LOG"
if [ "${1-}" = "-c" ]; then
    case "${2-}" in
        *sys.version_info.minor*) printf '13\n';;
        *sys.executable*) printf '%s\n' "$0";;
    esac
    exit 0
fi
if [ "${1-}" = "-m" ] && [ "${2-}" = "venv" ]; then
    target="${3-}"
    mkdir -p "$target/Scripts"
    cp "$PEAD_TEST_BASE_PYTHON" "$target/Scripts/python.exe"
    chmod +x "$target/Scripts/python.exe"
    exit 0
fi
if [ "${1-}" = "-m" ] && [ "${2-}" = "pip" ]; then
    if [ "${3-}" = "install" ] && [ "${4-}" = "-e" ] && [ "${PEAD_TEST_FAIL_APP-}" = "1" ]; then
        printf 'ERROR: simulated anomalib/application installation failure\n' >&2
        exit 1
    fi
    if [ "${3-}" = "check" ] && [ "${PEAD_TEST_FAIL_CHECK-}" = "1" ]; then
        printf 'ERROR: simulated dependency verification failure\n' >&2
        exit 1
    fi
    case "$*" in
        *torch==*)
            if [ "${PEAD_TEST_FAIL_TORCH-}" = "1" ]; then
                printf 'ERROR: simulated package installation failure\n' >&2
                exit 1
            fi
            ;;
    esac
    exit 0
fi
if [ "${1-}" = "-m" ] && [ "${2-}" = "pead.runtime" ]; then
    if [ "${PEAD_TEST_FAIL_RUNTIME-}" = "1" ]; then
        printf 'Missing anomalib: simulated model import failure\n' >&2
        exit 1
    fi
    printf 'Runtime verified: Python 3.13 - fixture\n'
    exit 0
fi
if [ "${1-}" = "-m" ] && [ "${2-}" = "uvicorn" ]; then
    printf 'SERVER_LAUNCHED %s\n' "$0" >> "$PEAD_TEST_LOG"
    exit 0
fi
printf 'Unexpected fake Python invocation: %s\n' "$*" >&2
exit 2
'@
[System.IO.File]::WriteAllText($basePython, $mockPython + "`n", [System.Text.UTF8Encoding]::new($false))
& chmod +x $basePython
function global:py {
    param([Parameter(ValueFromRemainingArguments=$true)]$Arguments)
    if ($Arguments[0] -eq '-3.13') {
        $global:LASTEXITCODE = 0
        return $env:PEAD_TEST_BASE_PYTHON
    }
    $global:LASTEXITCODE = 1
}

try {
    . (Join-Path $project 'scripts/windows-paths.ps1')
    $paths = Get-PeadPaths -ProjectRoot $project
    Assert-True ($project.Length -ge 153) 'fixture reproduces the long checkout root'
    Assert-True ($paths.EnvironmentDir.Length -le 70) 'venv is short enough without long-path support'
    Assert-True (-not $paths.EnvironmentDir.StartsWith($project)) 'venv is outside the checkout'
    Assert-True ($paths.RuntimeRoot -eq (Join-Path $env:USERPROFILE '.pead')) 'default root is short and user writable'
    $same = Get-PeadPaths -ProjectRoot $project
    Assert-True ($same.Python -eq $paths.Python) 'resolver chooses the same interpreter repeatedly'

    $other = Join-Path $testRoot ('other/' + (Split-Path $project -Leaf))
    New-Item -ItemType Directory -Path $other -Force | Out-Null
    Copy-Item (Join-Path $ProjectRoot 'pyproject.toml') (Join-Path $other 'pyproject.toml')
    $different = Get-PeadPaths -ProjectRoot $other
    Assert-True ($different.EnvironmentDir -ne $paths.EnvironmentDir) 'same basenames have distinct managed environments'

    $env:PEAD_RUNTIME_ROOT = Join-Path $testRoot 'override'
    $overridden = Get-PeadPaths -ProjectRoot $project
    Assert-True ($overridden.RuntimeRoot -eq $env:PEAD_RUNTIME_ROOT) 'absolute runtime-root override is respected'
    $env:PEAD_RUNTIME_ROOT = 'relative-path'
    $rejected = $false
    try { Get-PeadPaths -ProjectRoot $project | Out-Null } catch { $rejected = $true }
    Assert-True $rejected 'relative runtime-root overrides are rejected'
    $env:PEAD_RUNTIME_ROOT = [System.IO.Path]::GetPathRoot($testRoot)
    $rejected = $false
    try { Get-PeadPaths -ProjectRoot $project | Out-Null } catch { $rejected = $true }
    Assert-True $rejected 'bare filesystem roots are rejected as runtime locations'
    $env:PEAD_RUNTIME_ROOT = Join-Path $testRoot ('too-long-' * 12)
    $longRoot = $env:PEAD_RUNTIME_ROOT
    $rejected = $false
    try { Get-PeadPaths -ProjectRoot $project | Out-Null } catch { $rejected = $true }
    Assert-True $rejected 'overlong runtime roots are rejected before native package operations'
    Assert-True (-not (Test-Path $longRoot)) 'rejected runtime-root override is never created'
    $env:PEAD_RUNTIME_ROOT = $null
    $knownRelative = 'Lib/site-packages/pkg_resources/tests/data/my-test-package_unpacked-egg/my_test_package-1.0-py3.7.egg'
    Assert-True ((Join-Path (Join-Path $project '.venv') $knownRelative).Length -gt 260) 'old package pathname reproduces the Windows long-path limit'
    Assert-True ((Join-Path $paths.EnvironmentDir $knownRelative).Length -lt 260) 'the same package pathname fits the external short environment'

    # Keep safely cached large wheels from the failed previous installation,
    # while never inheriting an overlong cache path into the repaired setup.
    $originalLocalAppData = $env:LOCALAPPDATA
    try {
        $env:LOCALAPPDATA = Join-Path $testRoot 'l'
        $shortCache = Join-Path $env:LOCALAPPDATA 'pip/Cache'
        New-Item -ItemType Directory -Path $shortCache -Force | Out-Null
        $cached = Get-PeadPaths -ProjectRoot $project
        Assert-True ($cached.CacheDir -eq $shortCache) 'an existing short pip cache is reused instead of downloading large wheels again'
        Assert-True ($cached.EnvironmentDir -eq $paths.EnvironmentDir) 'cache reuse does not change the managed interpreter location'
        $env:LOCALAPPDATA = Join-Path $testRoot ('long-appdata-' * 7)
        $overlongCache = Join-Path $env:LOCALAPPDATA 'pip/Cache'
        New-Item -ItemType Directory -Path $overlongCache -Force | Out-Null
        $uncached = Get-PeadPaths -ProjectRoot $project
        Assert-True ($uncached.CacheDir -eq (Join-Path $paths.RuntimeRoot 'c')) 'overlong existing pip caches fall back to the short managed cache'
    } finally {
        $env:LOCALAPPDATA = $originalLocalAppData
    }

    # Preserve a local environment left by an older/failed installer.
    $localSentinel = Join-Path $project '.venv/user-sentinel.txt'
    New-Item -ItemType Directory -Path (Split-Path $localSentinel -Parent) -Force | Out-Null
    Set-Content -Path $localSentinel -Value 'keep existing environment files'

    $env:PEAD_TEST_FAIL_TORCH = '1'
    $failed = $false
    try { & (Join-Path $project 'scripts/install-windows.ps1') -Cpu } catch { $failed = $true }
    Assert-True $failed 'failed pip installation is propagated'
    Assert-True (-not (Test-Path $paths.ReadyFile)) 'failed installation never gets a ready marker'
    Assert-True (Test-Path $localSentinel) 'old local .venv is preserved after failure'
    $startFailed = $false
    try { & (Join-Path $project 'scripts/start.ps1') } catch { $startFailed = $true }
    Assert-True $startFailed 'launcher refuses incomplete environment'
    Assert-True (-not ((Get-Content $env:PEAD_TEST_LOG -Raw) -match 'SERVER_LAUNCHED')) 'incomplete environment never launches Uvicorn'

    $partialSentinel = Join-Path $paths.EnvironmentDir 'partial-sentinel.txt'
    Set-Content -Path $partialSentinel -Value 'preserve failed managed environment'
    $env:PEAD_TEST_FAIL_TORCH = $null
    & (Join-Path $project 'scripts/install-windows.ps1') -Cpu
    Assert-True (Test-Path $paths.ReadyFile) 'successful installation marks the environment ready'
    Assert-True (Test-Path $paths.Python) 'ready environment has its own interpreter'
    Assert-True (Test-Path $localSentinel) 'old local .venv remains untouched after repair'
    $preserved = @(Get-ChildItem -Path $paths.RuntimeRoot -Recurse -Filter 'partial-sentinel.txt')
    Assert-True ($preserved.Count -eq 1) 'partial managed environment files are preserved during repair'

    # Every failure stage must revoke an existing success marker. This prevents
    # users launching after Torch partly installed but anomalib/imports failed.
    foreach ($failureFlag in @('PEAD_TEST_FAIL_APP','PEAD_TEST_FAIL_CHECK','PEAD_TEST_FAIL_RUNTIME')) {
        Assert-True (Test-Path $paths.ReadyFile) "a previously successful environment exists before $failureFlag"
        [Environment]::SetEnvironmentVariable($failureFlag, '1', 'Process')
        $failed = $false
        try { & (Join-Path $project 'scripts/install-windows.ps1') -Cpu } catch { $failed = $true }
        Assert-True $failed "$failureFlag is propagated as an installation failure"
        Assert-True (-not (Test-Path $paths.ReadyFile)) "$failureFlag revokes the previous ready marker"
        $startFailed = $false
        try { & (Join-Path $project 'scripts/start.ps1') } catch { $startFailed = $true }
        Assert-True $startFailed "launcher refuses the partial environment after $failureFlag"
        Assert-True (-not ((Get-Content $env:PEAD_TEST_LOG -Raw) -match 'SERVER_LAUNCHED')) "$failureFlag never falls back to a global or old Python server"
        [Environment]::SetEnvironmentVariable($failureFlag, $null, 'Process')
        & (Join-Path $project 'scripts/install-windows.ps1') -Cpu
        Assert-True (Test-Path $paths.ReadyFile) "successful repair after $failureFlag restores readiness"
        Assert-True (Test-Path $localSentinel) "source-local files remain untouched after $failureFlag"
    }

    $readyText = Get-Content -LiteralPath $paths.ReadyFile -Raw
    $readyData = $readyText | ConvertFrom-Json
    $readyData.project = $other
    $readyData | ConvertTo-Json | Set-Content -LiteralPath $paths.ReadyFile
    $rejected = $false
    try { & (Join-Path $project 'scripts/start.ps1') } catch { $rejected = $true }
    Assert-True $rejected 'launcher rejects ready markers from another project even on a hash collision'
    Set-Content -LiteralPath $paths.ReadyFile -Value $readyText
    $manifestPath = Join-Path $project 'pyproject.toml'
    $manifestText = Get-Content -LiteralPath $manifestPath -Raw
    Add-Content -LiteralPath $manifestPath -Value '# dependency configuration changed'
    $rejected = $false
    try { & (Join-Path $project 'scripts/start.ps1') } catch { $rejected = $true }
    Assert-True $rejected 'launcher rejects stale environments after dependency manifest changes'
    [System.IO.File]::WriteAllText($manifestPath, $manifestText, [System.Text.UTF8Encoding]::new($false))

    & (Join-Path $project 'scripts/start.ps1')
    $commands = Get-Content $env:PEAD_TEST_LOG -Raw
    Assert-True ($commands.Contains('SERVER_LAUNCHED ' + $paths.Python)) 'launcher uses the exact managed interpreter that installer prepared'
    Assert-True (-not $commands.Contains((Join-Path $project '.venv'))) 'native package commands never use the long local .venv'
    Assert-True ($env:TEMP -eq $paths.TempDir) 'pip temporary files use the shared short path'
    Assert-True ($env:PIP_CACHE_DIR -eq $paths.CacheDir) 'pip cache uses the shared short path'
    Assert-True ($env:PEAD_ASSETS_DIR -eq $paths.ModelAssetsDir) 'model assets inherit the shared short path'
    Write-Host "PASS: $script:Assertions PowerShell installer/launcher assertions"
} finally {
    Set-Location $ProjectRoot
    Remove-Item -LiteralPath $testRoot -Recurse -Force
}
