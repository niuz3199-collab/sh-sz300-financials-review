param(
    [int]$Interval = 60,
    [switch]$Once
)

$ErrorActionPreference = "Stop"

$scriptPath = Join-Path $PSScriptRoot "run_gemma_pdf_hybrid_repair_monitor.ps1"

& $scriptPath `
    -OutDir ".tmp_gemma_pdf_hybrid_repair_v4" `
    -PidFile ".tmp_gemma_pdf_hybrid_runner_v4\run.pid.txt" `
    -StatusFile ".tmp_gemma_pdf_hybrid_runner_v4\progress_status.txt" `
    -JsonFile ".tmp_gemma_pdf_hybrid_runner_v4\progress_status.json" `
    -Interval $Interval `
    -Once:$Once

exit $LASTEXITCODE
