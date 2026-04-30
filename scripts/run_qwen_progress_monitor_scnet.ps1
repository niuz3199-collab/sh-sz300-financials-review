Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$outDir = Join-Path $repoRoot ".cache\qwen_pdf_financials_v4_scnet"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $outDir "progress_monitor_${timestamp}.out.log"
$stderr = Join-Path $outDir "progress_monitor_${timestamp}.err.log"
$launcherErr = Join-Path $outDir "progress_monitor_${timestamp}.launcher.err.log"

$python = Join-Path $env:LocalAppData "Programs\Python\Python313\python.exe"
if (-not (Test-Path $python)) {
    $python = (Get-Command python.exe -ErrorAction Stop).Source
}

$args = @(
    "scripts\monitor_qwen_progress.py",
    "--base-dir", ".",
    "--out-dir", ".cache/qwen_pdf_financials_v4_scnet",
    "--status-file", ".cache/qwen_pdf_financials_v4_scnet/progress_status.txt",
    "--json-file", ".cache/qwen_pdf_financials_v4_scnet/progress_status.json",
    "--watch",
    "--interval", "30",
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
