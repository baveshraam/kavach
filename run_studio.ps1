# Start a RECORDING session: the backend with the Studio enabled for the presenter, and the UI.
#
#   powershell -ExecutionPolicy Bypass -File .\run_studio.ps1
#
# Separate from run_demo.ps1 on purpose: the demo build never accepts recordings into the dataset.
# No speech models are loaded (recording needs only the decoder), so it starts in seconds. It uses the
# real demo database, so the personal facts you enter on the Speakers page are the ones the demo asks about.
param([string]$Speaker = "S04")

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$env:KAVACH_OFFLINE = "1"
$env:KAVACH_STUDIO_ENABLED = "true"
$env:KAVACH_STUDIO_SPEAKERS = "[`"$Speaker`"]"

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
