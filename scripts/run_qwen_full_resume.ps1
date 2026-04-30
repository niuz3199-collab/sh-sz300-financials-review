Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$outDir = Join-Path $repoRoot ".cache\qwen_pdf_financials_v2"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $outDir "run_full_resume_${timestamp}.out.log"
$stderr = Join-Path $outDir "run_full_resume_${timestamp}.err.log"
$launcherErr = Join-Path $outDir "run_full_resume_${timestamp}.launcher.err.log"

$python = Join-Path $env:LocalAppData "Programs\Python\Python313\python.exe"
if (-not (Test-Path $python)) {
    $python = (Get-Command python.exe -ErrorAction Stop).Source
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

$csvName = Get-ChildItem -Path (Join-Path $repoRoot "2023") -File -Filter "*_qwen_v2.csv" |
    Where-Object { $_.Name -notmatch "test|fix" } |
    Select-Object -First 1 -ExpandProperty Name
if (-not $csvName) {
    $csvName = "{year}_qwen_v2.csv"
}

$args = @(
    "scripts\step6_extract_financials_qwen_pdf.py",
    "--base-dir", ".",
    "--pdf-root", $pdfRoot,
    "--out-dir", ".cache/qwen_pdf_financials_v2",
    "--csv-name", $csvName,
    "--start-year", "1997",
    "--end-year", "2025",
    "--model", "qwen3.5:9b",
    "--dpi", "200",
    "--timeout", "900",
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
