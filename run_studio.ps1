# Start a RECORDING session: the backend with the Studio enabled for the presenter, and the UI.
#
#   powershell -ExecutionPolicy Bypass -File .\run_studio.ps1
#   powershell -ExecutionPolicy Bypass -File .\run_studio.ps1 -VoiceOnly   # the short plan: ten-word clips only
#
# Separate from run_demo.ps1 on purpose: the demo build never accepts recordings into the dataset.
# The full plan loads no speech models (recording needs only the decoder), so it starts in seconds; -VoiceOnly loads Whisper to check each clip. It uses the
# real demo database, so the personal facts you enter on the Speakers page are the ones the demo asks about.
param([string]$Speaker = "S04", [switch]$VoiceOnly)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
# The full plan needs only the decoder, so no models. The quick plan also checks that you said the words,
# which needs Whisper: it loads at start (about 20 s) and the first check is then fast.
if (-not $VoiceOnly) { $env:KAVACH_OFFLINE = "1" } else { $env:KAVACH_WARM_MODELS_ON_START = "true" }
$env:KAVACH_STUDIO_ENABLED = "true"
$env:KAVACH_STUDIO_SPEAKERS = "[`"$Speaker`"]"
# Quick plan: ten-word clips only (about 4 minutes a sitting), the Unlock login's own task.
if ($VoiceOnly) { $env:KAVACH_STUDIO_VOICE_ONLY = "true" }

Write-Host "Starting backend on http://localhost:8000 (Studio enabled for $Speaker) ..." -ForegroundColor Cyan
$backend = Start-Process -FilePath $py `
    -ArgumentList "-m", "uvicorn", "kavach.api.app:app", "--port", "8000", "--app-dir", "backend" `
    -WorkingDirectory $root -PassThru -WindowStyle Minimized

Write-Host "Starting UI on http://localhost:3000 ..." -ForegroundColor Cyan
$ui = Start-Process -FilePath "npm.cmd" -ArgumentList "run", "dev" `
    -WorkingDirectory (Join-Path $root "kavach") -PassThru -WindowStyle Minimized

Start-Sleep -Seconds 6
Start-Process "http://localhost:3000/studio"

Write-Host ""
Write-Host "Recording Studio is open. Recordings are saved under data\studio\$Speaker (never leaves this machine)." -ForegroundColor Green
Write-Host "Press Enter here to stop both servers."
[void][Console]::ReadLine()

foreach ($p in @($backend, $ui)) {
    if ($p -and -not $p.HasExited) { taskkill /PID $p.Id /T /F | Out-Null }
}
Write-Host "Stopped."
