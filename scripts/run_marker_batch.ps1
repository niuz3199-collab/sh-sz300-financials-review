param(
    [Parameter(Mandatory = $true)]
    [string]$InputDir,
    [Parameter(Mandatory = $true)]
    [string]$OutputDir,
    [string]$ModelCacheDir = ".cache\\marker_models",
    [string]$TempDir = ".cache\\tmp_marker",
    [string]$MarkerExe,
    [int]$Workers = 1,
    [int]$MaxTasksPerWorker = 10,
    [switch]$SkipExisting = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$resolvedInputDir = if ([System.IO.Path]::IsPathRooted($InputDir)) {
    $InputDir
} else {
    Join-Path $repoRoot $InputDir
}
$resolvedOutputDir = if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir
} else {
    Join-Path $repoRoot $OutputDir
}
$resolvedModelCacheDir = if ([System.IO.Path]::IsPathRooted($ModelCacheDir)) {
    $ModelCacheDir
} else {
    Join-Path $repoRoot $ModelCacheDir
}
$resolvedTempDir = if ([System.IO.Path]::IsPathRooted($TempDir)) {
    $TempDir
} else {
    Join-Path $repoRoot $TempDir
}

New-Item -ItemType Directory -Force -Path $resolvedOutputDir | Out-Null
New-Item -ItemType Directory -Force -Path $resolvedModelCacheDir | Out-Null
New-Item -ItemType Directory -Force -Path $resolvedTempDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $resolvedOutputDir "marker_batch_${timestamp}.out.log"
$stderr = Join-Path $resolvedOutputDir "marker_batch_${timestamp}.err.log"
$launcherErr = Join-Path $resolvedOutputDir "marker_batch_${timestamp}.launcher.err.log"

$markerCandidates = @()
if ($MarkerExe) {
    if ([System.IO.Path]::IsPathRooted($MarkerExe)) {
        $markerCandidates += $MarkerExe
    } else {
        $markerCandidates += (Join-Path $repoRoot $MarkerExe)
    }
}
$markerCandidates += @(
    (Join-Path $env:USERPROFILE "miniconda3\\envs\\marker_gpu\\Scripts\\marker.exe"),
    (Join-Path $env:USERPROFILE ".local\\bin\\marker.exe")
)
$marker = $markerCandidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if (-not $marker) {
    throw "Could not locate marker.exe. Pass -MarkerExe explicitly."
}

$env:MODEL_CACHE_DIR = $resolvedModelCacheDir
$env:TEMP = $resolvedTempDir
$env:TMP = $resolvedTempDir

$args = @(
    $resolvedInputDir,
    "--output_dir", $resolvedOutputDir,
    "--output_format", "markdown",
    "--workers", "$Workers",
    "--max_tasks_per_worker", "$MaxTasksPerWorker"
)
if ($SkipExisting) {
    $args += "--skip_existing"
}

try {
    $proc = Start-Process `
        -FilePath $marker `
        -ArgumentList $args `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -Wait `
        -PassThru
    exit $proc.ExitCode
} catch {
    $_ | Out-File -FilePath $launcherErr -Encoding utf8
    exit 1
}
