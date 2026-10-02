# Start the KAVACH demo: FastAPI backend on :8000 and the UI on :3000.
#
#   powershell -ExecutionPolicy Bypass -File .\run_demo.ps1
#   powershell -ExecutionPolicy Bypass -File .\run_demo.ps1 -Seed      # rebuild the 12-speaker demo DB first
#   powershell -ExecutionPolicy Bypass -File .\run_demo.ps1 -Prefetch  # download all models first (once)
#
# The backend reads .env (Whisper `small`, the Gemini key); the demo switches
# below are set for this process only.
param([switch]$Seed, [switch]$Prefetch)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"

# Demo switches, set for this process only -- not in .env, which the test
# suite also reads (HANDOFF trap 4: a suite must not inherit its environment).
#  - Splice detection is off by default now (it rejected 167/168 genuine corpus
#    clips and no cue separates splices; `python -m kavach.calibrate_integrity`);
#    stated here so the demo does not depend on the default. Replay detection stays on.
#  - Load Whisper / ECAPA / LaBSE at start-up, not on the first login.
#  - The CSBG veto was fitted on dev and discarded by the offline runs (no FAR
#    reduction inside the 2% FRR budget); see Settings.csbg_veto_enabled.
$env:KAVACH_INTEGRITY_CHECK_SPLICE = "false"
$env:KAVACH_WARM_MODELS_ON_START = "true"
$env:KAVACH_CSBG_VETO_ENABLED = "false"
# The clone-attack demo: the bank holds the presenter's cloned answers and its
# routes return audio that SAYS the answer, so this build announces itself in
# /api/health (demoAttackBank). Only the presenter has agreed to be cloned.
$env:KAVACH_DEMO_ATTACK_BANK = "true"
$env:KAVACH_CLONE_VICTIMS = '["S04"]'
# The preflight (python -m kavach.demo_check --flows) drives failed logins against the presenter;
# this lets it clear the strikes afterwards so the owner starts the demo with a clean slate.
$env:KAVACH_DEMO_TOOLS = "true"

if ($Prefetch) {
    Write-Host "Downloading every model the demo needs (run once, on a good connection) ..." -ForegroundColor Cyan
    $env:PYTHONPATH = "backend"
    & $py -m kavach.prefetch
    Remove-Item Env:PYTHONPATH
}

if ($Seed) {
    Write-Host "Rebuilding the demo database from corpus_v2 + corpus_v3 ..." -ForegroundColor Cyan
    $env:PYTHONPATH = "backend"
    & $py -m kavach.seed_demo
    Remove-Item Env:PYTHONPATH
}

Write-Host "Starting backend on http://localhost:8000 ..." -ForegroundColor Cyan
$backend = Start-Process -FilePath $py `
    -ArgumentList "-m", "uvicorn", "kavach.api.app:app", "--port", "8000", "--app-dir", "backend" `
    -WorkingDirectory $root -PassThru -WindowStyle Minimized

Write-Host "Starting UI on http://localhost:3000 ..." -ForegroundColor Cyan
$ui = Start-Process -FilePath "npm.cmd" -ArgumentList "run", "dev" `
    -WorkingDirectory (Join-Path $root "kavach") -PassThru -WindowStyle Minimized

Start-Sleep -Seconds 6
Start-Process "http://localhost:3000/unlock"

Write-Host ""
Write-Host "Demo running. Models warm up in the background (about 20 s);" -ForegroundColor Green
Write-Host "the sidebar status turns green when the backend is ready."
Write-Host "Press Enter to stop both servers."
[void][Console]::ReadLine()

foreach ($p in @($backend, $ui)) {
    if ($p -and -not $p.HasExited) { taskkill /PID $p.Id /T /F | Out-Null }
}
Write-Host "Stopped."
