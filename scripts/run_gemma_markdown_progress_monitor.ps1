param(
    [string]$MarkdownRoot = ".cache\qwen_pdf_markdown_remaining\output_markdown",
    [string]$OutDir = ".tmp_gemma_markdown_financials_full",
    [string]$PidFile = ".tmp_gemma_markdown_runner\run.pid.txt",
    [string]$StatusFile = ".tmp_gemma_markdown_runner\progress_status.txt",
    [string]$JsonFile = ".tmp_gemma_markdown_runner\progress_status.json",
    [int]$Interval = 60,
    [switch]$Once
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$scriptPath = Join-Path $repoRoot "scripts\monitor_gemma_markdown_progress.py"

$args = @(
    $scriptPath,
    "--markdown-root", $MarkdownRoot,
    "--out-dir", $OutDir,
    "--pid-file", $PidFile,
    "--status-file", $StatusFile,
    "--json-file", $JsonFile,
    "--interval", "$Interval"
)

if ($Once) {
    $args += "--once"
}

Write-Host "[run] python $($args -join ' ')"
& python @args
exit $LASTEXITCODE
