param(
    [string]$OutDir = ".cache\qwen_pdf_financials_v6_dynamic",
    [string]$CsvName = "{year}_qwen_v6_dynamic.csv",
    [int]$StartYear = 2001,
    [int]$EndYear = 2025,
    [string]$WorkerId = "api",
    [string]$ApiBaseUrl = "https://naapi.cc/v1",
    [string]$ApiKeyEnv = "NAAPI_API_KEY",
    [string]$Model = "gpt-5.4",
    [int]$TimeoutSeconds = 2400,
    [int]$LeaseSeconds = 3000
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
$stdout = Join-Path $resolvedOutDir "worker_${WorkerId}_${timestamp}.out.log"
$stderr = Join-Path $resolvedOutDir "worker_${WorkerId}_${timestamp}.err.log"
$launcherErr = Join-Path $resolvedOutDir "worker_${WorkerId}_${timestamp}.launcher.err.log"

$python = Join-Path $env:LocalAppData "Programs\Python\Python313\python.exe"
if (-not (Test-Path $python)) {
    $python = (Get-Command python.exe -ErrorAction Stop).Source
}

$apiKeyValue = [Environment]::GetEnvironmentVariable($ApiKeyEnv)
if (-not $apiKeyValue) {
    throw "Environment variable $ApiKeyEnv is empty."
}

$pdfRoot = Get-ChildItem -Path $repoRoot -Directory |
    ForEach-Object {
        Get-ChildItem -Path $_.FullName -Directory -ErrorAction SilentlyContinue
    } |
    Where-Object {
        $_.Name -match "_fulltext$" -and
        (Test-Path (Join-Path $_.FullName "batch_download_log.csv"))
    } |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $pdfRoot) {
    throw "Could not locate the main *_fulltext PDF directory under $repoRoot"
}

$queueDbArg = Join-Path $resolvedOutDir "task_queue.sqlite"
$args = @(
    "scripts\qwen_dynamic_queue.py",
    "--base-dir", ".",
    "--pdf-root", $pdfRoot,
    "--out-dir", $resolvedOutDir,
    "--queue-db", $queueDbArg,
    "--csv-name", $CsvName,
    "--start-year", "$StartYear",
    "--end-year", "$EndYear",
    "--worker-id", $WorkerId,
    "--backend", "openai_text",
    "--api-base-url", $ApiBaseUrl,
    "--api-key-env", $ApiKeyEnv,
    "--model", $Model,
    "--timeout", "$TimeoutSeconds",
    "--lease-seconds", "$LeaseSeconds"
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
