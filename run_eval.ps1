# The paper's voice-gate numbers, regenerated from the recordings. One command; it changes no demo state.
#
#   powershell -ExecutionPolicy Bypass -File .\run_eval.ps1
#   powershell -ExecutionPolicy Bypass -File .\run_eval.ps1 -Sessions "S1,S2,S3" -Seed 11
#
# Embeds your Studio clips once (cached in data\eval_cache by clip hash), then writes paper\results_poc\<embedder>\
# (report.md, results.json, preregistration.json, scores.npz, tables\*.tex, figures\*). Those files are generated: never edit them.
# The laptop is also the training machine, so it checks the GPU and running python first and asks you to stop if busy.
param(
    [string]$Speaker = "S04",
    [string]$Sessions = "S1,S2,S3",
    [int]$Seed = 7,
    [ValidateSet("ecapa", "resnet", "wavlm")][string]$Embedder = "ecapa",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "backend"

$busy = Get-Process python -ErrorAction SilentlyContinue
if ($busy -and -not $Force) {
    Write-Host "python is already running ($($busy.Count) process(es)): a training job or a server? Stop it, or add -Force to go ahead anyway." -ForegroundColor Yellow
    exit 2
}
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpu = & nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
    Write-Host "GPU: $gpu"
}

& $py -m kavach.eval.poc_run --studio "data\studio\$Speaker" --sessions $Sessions --seed $Seed --embedder $Embedder
if ($LASTEXITCODE -ne 0) { Write-Host "STOPPED: the evaluation did not run (exit $LASTEXITCODE)." -ForegroundColor Red; exit $LASTEXITCODE }

Write-Host ""
Write-Host "Done. Read paper\results_poc\$Embedder\report.md (the limits are its first paragraph)." -ForegroundColor Green
# The whole paper (all models, the population benchmark, the synthetic-voice attacks, the IEEE PDF) is run_paper.ps1.
