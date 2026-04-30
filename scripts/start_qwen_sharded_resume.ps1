Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

if (-not $env:SCNET_API_KEY) {
    throw "Environment variable SCNET_API_KEY is empty."
}

$metaDir = Join-Path $repoRoot ".cache\qwen_pdf_financials_v5_sharded"
New-Item -ItemType Directory -Force -Path $metaDir | Out-Null

$local = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $repoRoot "scripts\run_qwen_full_local_shard_resume.ps1")) `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -PassThru

Start-Sleep -Seconds 2

$api = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $repoRoot "scripts\run_qwen_full_scnet_shard_resume.ps1")) `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -PassThru

Start-Sleep -Seconds 2

$monitor = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $repoRoot "scripts\run_qwen_progress_monitor_sharded.ps1")) `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -PassThru

$payload = [pscustomobject]@{
    local_wrapper_pid   = $local.Id
    api_wrapper_pid     = $api.Id
    monitor_wrapper_pid = $monitor.Id
    started_at          = (Get-Date).ToString("s")
}

$payload | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $metaDir "launcher_pids.json")
$payload | Format-List
