param(
    [string]$Model = "google/gemma-4-26b-a4b",
    [string]$ApiBaseUrl = "http://127.0.0.1:1234/v1",
    [string]$PartialCsv = ".tmp_gemma_markdown_repair_runner\partial_tasks_latest.csv",
    [string]$FulltextRoot = "",
    [string]$OutDir = ".tmp_gemma_pdf_hybrid_repair",
    [string]$YearCsvRoot = ".tmp_gemma_year_csvs_pdf_hybrid",
    [string]$PidFile = ".tmp_gemma_pdf_hybrid_runner\run.pid.txt",
    [string]$StatusFile = ".tmp_gemma_pdf_hybrid_runner\progress_status.txt",
    [string]$JsonFile = ".tmp_gemma_pdf_hybrid_runner\progress_status.json",
    [string[]]$Fields = @("parent_netprofit", "total_shares", "operating_cashflow", "capex"),
    [int]$StartYear = 2001,
    [int]$EndYear = 2025,
    [int]$Timeout = 240,
    [int]$MaxSnippets = 4,
    [int]$MaxChars = 12000,
    [int]$DpiCrop = 220,
    [int]$DpiPage = 150,
    [int]$Start = 0,
    [int]$Limit = 0,
    [switch]$ForceRerun
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$scriptPath = Join-Path $repoRoot "scripts\repair_gemma_pdf_hybrid_fields.py"

foreach ($pathValue in @($PidFile, $StatusFile, $JsonFile)) {
    if ($pathValue) {
        $resolved = if ([System.IO.Path]::IsPathRooted($pathValue)) { $pathValue } else { Join-Path $repoRoot $pathValue }
        $dir = Split-Path -Parent $resolved
        if ($dir) {
            New-Item -ItemType Directory -Force -Path $dir | Out-Null
        }
    }
}

$args = @(
    $scriptPath,
    "--base-dir", $repoRoot,
    "--partial-csv", $PartialCsv,
    "--out-dir", $OutDir,
    "--year-csv-root", $YearCsvRoot,
    "--model", $Model,
    "--api-base-url", $ApiBaseUrl,
    "--start-year", "$StartYear",
    "--end-year", "$EndYear",
    "--timeout", "$Timeout",
    "--max-snippets", "$MaxSnippets",
    "--max-chars", "$MaxChars",
    "--dpi-crop", "$DpiCrop",
    "--dpi-page", "$DpiPage",
    "--start", "$Start",
    "--limit", "$Limit",
    "--pid-file", $PidFile,
    "--status-file", $StatusFile,
    "--json-file", $JsonFile,
    "--fields"
)

$args += $Fields

if ($FulltextRoot) {
    $args += @("--fulltext-root", $FulltextRoot)
}

if ($ForceRerun) {
    $args += "--force-rerun"
}

Write-Host "[run] python $($args -join ' ')"
& python @args
exit $LASTEXITCODE
