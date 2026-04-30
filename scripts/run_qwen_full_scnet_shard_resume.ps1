Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$outDir = Join-Path $repoRoot ".cache\qwen_pdf_financials_v5_api_2013_2025"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $outDir "run_full_resume_${timestamp}.out.log"
$stderr = Join-Path $outDir "run_full_resume_${timestamp}.err.log"
$launcherErr = Join-Path $outDir "run_full_resume_${timestamp}.launcher.err.log"

$python = Join-Path $env:LocalAppData "Programs\Python\Python313\python.exe"
if (-not (Test-Path $python)) {
    $python = (Get-Command python.exe -ErrorAction Stop).Source
}

if (-not $env:SCNET_API_KEY) {
    throw "Environment variable SCNET_API_KEY is empty."
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

$args = @(
    "scripts\step6_extract_financials_qwen_pdf.py",
    "--base-dir", ".",
    "--pdf-root", $pdfRoot,
    "--out-dir", ".cache/qwen_pdf_financials_v5_api_2013_2025",
    "--csv-name", "{year}_qwen_v5_split.csv",
    "--start-year", "2013",
    "--end-year", "2025",
    "--backend", "openai_text",
    "--api-base-url", "https://api.scnet.cn/api/llm/v1",
    "--api-key-env", "SCNET_API_KEY",
    "--model", "Qwen3-235B-A22B",
    "--timeout", "2400",
    "--resume"
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
