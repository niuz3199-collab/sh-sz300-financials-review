param(
    [string]$Model = "google/gemma-4-26b-a4b",
    [string]$ApiBaseUrl = "http://127.0.0.1:1234/v1",
    [string]$MarkdownRoot = ".cache\qwen_pdf_markdown_remaining\output_markdown",
    [string]$OutDir = ".cache\gemma_markdown_financials",
    [string]$YearCsvRoot = "",
    [string]$PidFile = "",
    [int]$StartYear = 2001,
    [int]$EndYear = 2025,
    [int]$Timeout = 180,
    [int]$MaxSnippets = 6,
    [int]$MaxCharsPerField = 28000,
    [int]$Start = 0,
    [int]$Limit = 0,
    [switch]$Resume,
    [switch]$Overwrite,
    [switch]$Debug
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$scriptPath = Join-Path $repoRoot "scripts\step6_extract_financials_from_markdown.py"

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
    "--markdown-root", $MarkdownRoot,
    "--out-dir", $OutDir,
    "--model", $Model,
    "--api-base-url", $ApiBaseUrl,
    "--start-year", "$StartYear",
    "--end-year", "$EndYear",
    "--timeout", "$Timeout",
    "--max-snippets", "$MaxSnippets",
    "--max-chars-per-field", "$MaxCharsPerField",
    "--start", "$Start",
    "--limit", "$Limit"
)

if ($YearCsvRoot) {
    $args += @("--year-csv-root", $YearCsvRoot)
}
if ($Resume) {
    $args += "--resume"
}
if ($Overwrite) {
    $args += "--overwrite"
}
if ($Debug) {
    $args += "--debug"
}

Write-Host "[run] python $($args -join ' ')"
& python @args
exit $LASTEXITCODE
