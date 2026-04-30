param(
    [string]$Model = "google/gemma-4-26b-a4b",
    [string]$ApiBaseUrl = "http://127.0.0.1:1234/v1",
    [string]$PartialCsv = ".tmp_gemma_markdown_repair_runner\partial_tasks_latest.csv",
    [string]$FulltextRoot = "",
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

$scriptPath = Join-Path $PSScriptRoot "run_gemma_pdf_hybrid_repair.ps1"

& $scriptPath `
    -Model $Model `
    -ApiBaseUrl $ApiBaseUrl `
    -PartialCsv $PartialCsv `
    -FulltextRoot $FulltextRoot `
    -OutDir ".tmp_gemma_pdf_hybrid_repair_v4" `
    -YearCsvRoot ".tmp_gemma_year_csvs_pdf_hybrid_v4" `
    -PidFile ".tmp_gemma_pdf_hybrid_runner_v4\run.pid.txt" `
    -StatusFile ".tmp_gemma_pdf_hybrid_runner_v4\progress_status.txt" `
    -JsonFile ".tmp_gemma_pdf_hybrid_runner_v4\progress_status.json" `
    -Fields $Fields `
    -StartYear $StartYear `
    -EndYear $EndYear `
    -Timeout $Timeout `
    -MaxSnippets $MaxSnippets `
    -MaxChars $MaxChars `
    -DpiCrop $DpiCrop `
    -DpiPage $DpiPage `
    -Start $Start `
    -Limit $Limit `
    -ForceRerun:$ForceRerun

exit $LASTEXITCODE
