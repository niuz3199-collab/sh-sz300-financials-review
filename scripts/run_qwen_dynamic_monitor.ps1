param(
    [string]$OutDir = ".cache\qwen_pdf_financials_v6_dynamic",
    [int]$IntervalSeconds = 30
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$resolvedOutDir = if ([System.IO.Path]::IsPathRooted($OutDir)) {
    $OutDir
} else {
    Join-Path $repoRoot $OutDir
}
New-Item -ItemType Directory -Force -Path $resolvedOutDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $resolvedOutDir "progress_monitor_${timestamp}.out.log"
$stderr = Join-Path $resolvedOutDir "progress_monitor_${timestamp}.err.log"
$launcherErr = Join-Path $resolvedOutDir "progress_monitor_${timestamp}.launcher.err.log"

$python = Join-Path $env:LocalAppData "Programs\Python\Python313\python.exe"
if (-not (Test-Path $python)) {
    $python = (Get-Command python.exe -ErrorAction Stop).Source
}

$statusFile = Join-Path $resolvedOutDir "progress_status.txt"
$jsonFile = Join-Path $resolvedOutDir "progress_status.json"
$args = @(
    "scripts\monitor_qwen_progress.py",
    "--base-dir", ".",
    "--out-dir", $resolvedOutDir,
    "--status-file", $statusFile,
    "--json-file", $jsonFile,
    "--watch",
    "--interval", "$IntervalSeconds",
    "--quiet"
)

try {
    $proc = Start-Process `
        -FilePath $python `
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
