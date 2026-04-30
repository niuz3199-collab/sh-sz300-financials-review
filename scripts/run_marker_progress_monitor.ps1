param(
    [string]$InputDir = ".cache\qwen_pdf_markdown_remaining\input_pdfs",
    [string]$OutputDir = ".cache\qwen_pdf_markdown_remaining\output_markdown",
    [int]$IntervalSeconds = 30
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
New-Item -ItemType Directory -Force -Path $resolvedOutputDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $resolvedOutputDir "marker_progress_monitor_${timestamp}.out.log"
$stderr = Join-Path $resolvedOutputDir "marker_progress_monitor_${timestamp}.err.log"
$launcherErr = Join-Path $resolvedOutputDir "marker_progress_monitor_${timestamp}.launcher.err.log"

$python = Join-Path $env:LocalAppData "Programs\Python\Python313\python.exe"
if (-not (Test-Path $python)) {
    $python = (Get-Command python.exe -ErrorAction Stop).Source
}

$statusFile = Join-Path $resolvedOutputDir "progress_status.txt"
$jsonFile = Join-Path $resolvedOutputDir "progress_status.json"
$args = @(
    "scripts\monitor_marker_progress.py",
    "--input-dir", $resolvedInputDir,
    "--output-dir", $resolvedOutputDir,
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
