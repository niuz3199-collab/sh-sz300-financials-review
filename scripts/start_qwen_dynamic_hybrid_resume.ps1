Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

if (-not $env:NAAPI_API_KEY) {
    throw "Environment variable NAAPI_API_KEY is empty."
}
if (-not $env:SCNET_API_KEY) {
    throw "Environment variable SCNET_API_KEY is empty."
}

$outDir = Join-Path $repoRoot ".cache\qwen_pdf_financials_v6_dynamic_hybrid"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

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

$python = Join-Path $env:LocalAppData "Programs\Python\Python313\python.exe"
if (-not (Test-Path $python)) {
    $python = (Get-Command python.exe -ErrorAction Stop).Source
}

& $python "scripts\qwen_dynamic_queue.py" `
    --base-dir "." `
    --pdf-root $pdfRoot `
    --out-dir $outDir `
    --queue-db (Join-Path $outDir "task_queue.sqlite") `
    --csv-name "{year}_qwen_v6_dynamic_hybrid.csv" `
    --start-year 2001 `
    --end-year 2025 `
    --init-only

$local = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $repoRoot "scripts\run_qwen_dynamic_local_worker.ps1"),
        "-OutDir", $outDir,
        "-CsvName", "{year}_qwen_v6_dynamic_hybrid.csv",
        "-WorkerId", "local",
        "-Model", "qwen3.5:9b",
        "-Dpi", "200",
        "-OllamaNumCtx", "8192",
        "-TimeoutSeconds", "1200",
        "-LeaseSeconds", "1800"
    ) `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -PassThru

Start-Sleep -Seconds 2

$naapi = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $repoRoot "scripts\run_qwen_dynamic_api_worker.ps1"),
        "-OutDir", $outDir,
        "-CsvName", "{year}_qwen_v6_dynamic_hybrid.csv",
        "-WorkerId", "api_naapi",
        "-ApiBaseUrl", "https://naapi.cc/v1",
        "-ApiKeyEnv", "NAAPI_API_KEY",
        "-Model", "gpt-5.4",
        "-TimeoutSeconds", "2400",
        "-LeaseSeconds", "3000"
    ) `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -PassThru

Start-Sleep -Seconds 2

$scnetWorkers = @("api_scnet", "api_scnet2", "api_scnet3", "api_scnet4")
$scnets = @()
foreach ($workerId in $scnetWorkers) {
    $scnets += Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", (Join-Path $repoRoot "scripts\run_qwen_dynamic_api_worker.ps1"),
            "-OutDir", $outDir,
            "-CsvName", "{year}_qwen_v6_dynamic_hybrid.csv",
            "-WorkerId", $workerId,
            "-ApiBaseUrl", "https://api.scnet.cn/api/llm/v1",
            "-ApiKeyEnv", "SCNET_API_KEY",
            "-Model", "Qwen3-235B-A22B",
            "-TimeoutSeconds", "2400",
            "-LeaseSeconds", "3000"
        ) `
        -WorkingDirectory $repoRoot `
        -WindowStyle Hidden `
        -PassThru
    Start-Sleep -Seconds 2
}

$monitor = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $repoRoot "scripts\run_qwen_dynamic_monitor.ps1"),
        "-OutDir", $outDir,
        "-IntervalSeconds", "30"
    ) `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -PassThru

$payload = [pscustomobject]@{
    local_wrapper_pid   = $local.Id
    naapi_wrapper_pid   = $naapi.Id
    scnet_wrapper_pid   = $scnets[0].Id
    scnet_wrapper_pids  = @($scnets | ForEach-Object { $_.Id })
    monitor_wrapper_pid = $monitor.Id
    started_at          = (Get-Date).ToString("s")
}

$payload | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $outDir "launcher_pids.json")
$payload | Format-List
