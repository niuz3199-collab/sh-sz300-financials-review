param(
    [string]$Model = "",
    [string]$ApiBaseUrl = "",
    [string]$MarkdownRoot = "",
    [string]$OutDir = ".tmp_gemma_markdown_financials_full",
    [string]$YearCsvRoot = "",
    [string]$CsvName = "",
    [string]$PidFile = "",
    [string]$Statuses = "partial,error",
    [int]$Timeout = 0,
    [int]$StartYear = 0,
    [int]$EndYear = 0,
    [int]$MaxSnippets = 0,
    [int]$MaxCharsPerField = 0,
    [int]$MaxAttempts = 3,
    [int]$Start = 0,
    [int]$Limit = 0
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$scriptPath = Join-Path $repoRoot "scripts\repair_gemma_markdown_financials.py"

if ($PidFile) {
    $pidPath = if ([System.IO.Path]::IsPathRooted($PidFile)) { $PidFile } else { Join-Path $repoRoot $PidFile }
    $pidDir = Split-Path -Parent $pidPath
    if ($pidDir) {
        New-Item -ItemType Directory -Force -Path $pidDir | Out-Null
    }
    Set-Content -Path $pidPath -Value $PID -Encoding ASCII
}

$args = @(
    $scriptPath,
    "--base-dir", $repoRoot,
    "--out-dir", $OutDir,
    "--statuses", $Statuses,
    "--max-attempts", "$MaxAttempts",
    "--start", "$Start",
    "--limit", "$Limit"
)

if ($Model) { $args += @("--model", $Model) }
if ($ApiBaseUrl) { $args += @("--api-base-url", $ApiBaseUrl) }
if ($MarkdownRoot) { $args += @("--markdown-root", $MarkdownRoot) }
if ($YearCsvRoot) { $args += @("--year-csv-root", $YearCsvRoot) }
if ($CsvName) { $args += @("--csv-name", $CsvName) }
if ($Timeout -gt 0) { $args += @("--timeout", "$Timeout") }
if ($StartYear -gt 0) { $args += @("--start-year", "$StartYear") }
if ($EndYear -gt 0) { $args += @("--end-year", "$EndYear") }
if ($MaxSnippets -gt 0) { $args += @("--max-snippets", "$MaxSnippets") }
if ($MaxCharsPerField -gt 0) { $args += @("--max-chars-per-field", "$MaxCharsPerField") }

Write-Host "[run] python $($args -join ' ')"
& python @args
exit $LASTEXITCODE
