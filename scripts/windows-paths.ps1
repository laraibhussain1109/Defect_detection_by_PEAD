# Shared by install and start. Never put package files below a downloaded checkout.
function Get-PeadPaths {
    param([Parameter(Mandatory=$true)][string]$ProjectRoot)
    $project = [IO.Path]::GetFullPath($ProjectRoot).TrimEnd([char[]]@('\', '/'))
    $root = $env:PEAD_RUNTIME_ROOT
    if (-not $root) {
        $profilePath = $env:USERPROFILE
        if (-not $profilePath) { $profilePath = [Environment]::GetFolderPath('UserProfile') }
        if (-not $profilePath) { throw 'Cannot find your user profile. Set PEAD_RUNTIME_ROOT to a short, writable absolute folder.' }
        $root = Join-Path $profilePath '.pead'
    }
    if (-not [IO.Path]::IsPathRooted($root)) {
        throw 'PEAD_RUNTIME_ROOT must be an absolute path to a short, user-writable folder.'
    }
    $fullRoot = [IO.Path]::GetFullPath($root)
    if ($fullRoot.TrimEnd([char[]]@('\', '/')) -eq [IO.Path]::GetPathRoot($fullRoot).TrimEnd([char[]]@('\', '/'))) {
        throw 'PEAD_RUNTIME_ROOT must be a dedicated user-writable directory, not a drive or filesystem root.'
    }
    $root = $fullRoot.TrimEnd([char[]]@('\', '/'))
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $digest = $sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($project.ToUpperInvariant()))
        $id = ([BitConverter]::ToString($digest)).Replace('-', '').ToLowerInvariant().Substring(0, 12)
    } finally { $sha.Dispose() }
    $environment = Join-Path (Join-Path $root 'v') $id
    if ($environment.Length -gt 80) {
        throw "Managed Python path is too long ($($environment.Length) characters). Set PEAD_RUNTIME_ROOT to a shorter user-writable folder, then rerun. No Windows registry change is needed."
    }
    $manifest = Join-Path $project 'pyproject.toml'
    if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) { throw "Missing application manifest: $manifest" }
    $cache = Join-Path $root 'c'
    # Reuse already-downloaded multi-GB wheels when the normal Windows pip
    # cache is short. Never reuse the long checkout-local environment itself.
    if ($env:LOCALAPPDATA) {
        $existingCache = Join-Path $env:LOCALAPPDATA 'pip/Cache'
        if ($existingCache.Length -le 70 -and (Test-Path -LiteralPath $existingCache -PathType Container)) {
            $cache = $existingCache
        }
    }
    [pscustomobject]@{
        ProjectRoot = $project
        RuntimeRoot = $root
        EnvironmentDir = $environment
        Python = Join-Path (Join-Path $environment 'Scripts') 'python.exe'
        ReadyFile = Join-Path $environment 'pead-ready.json'
        DependencyHash = (Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash
        TempDir = Join-Path (Join-Path $root 't') $id
        CacheDir = $cache
        ModelAssetsDir = Join-Path $root 'a'
        LockFile = "$environment.lock"
    }
}

function Initialize-PeadPaths {
    param([Parameter(Mandatory=$true)]$Paths)
    foreach ($directory in @($Paths.RuntimeRoot, (Split-Path $Paths.EnvironmentDir -Parent), $Paths.TempDir, $Paths.CacheDir, $Paths.ModelAssetsDir)) {
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
    }
    # tempfile and pip inherit these short directories even during wheel builds.
    $env:TEMP = $Paths.TempDir
    $env:TMP = $Paths.TempDir
    $env:TMPDIR = $Paths.TempDir
    $env:PIP_CACHE_DIR = $Paths.CacheDir
    $env:PYTHONUTF8 = '1'
    if (-not $env:PEAD_ASSETS_DIR) { $env:PEAD_ASSETS_DIR = $Paths.ModelAssetsDir }
}

function Test-PeadReady {
    param([Parameter(Mandatory=$true)]$Paths)
    if (-not (Test-Path -LiteralPath $Paths.Python -PathType Leaf) -or
        -not (Test-Path -LiteralPath $Paths.ReadyFile -PathType Leaf)) { return $false }
    try {
        $ready = Get-Content -LiteralPath $Paths.ReadyFile -Raw | ConvertFrom-Json
        return ($ready.format -eq 1 -and $ready.complete -eq $true -and
                $ready.project -ieq $Paths.ProjectRoot -and $ready.dependencies -eq $Paths.DependencyHash)
    } catch { return $false }
}
