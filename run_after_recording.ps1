# After the Studio sessions are recorded: measure, calibrate, enrol. One command, in the right order.
#
#   powershell -ExecutionPolicy Bypass -File .\run_after_recording.ps1
#   powershell -ExecutionPolicy Bypass -File .\run_after_recording.ps1 -Sessions "S1,S2,S3" -NoEvidence
#
# Stop the Studio (run_studio.ps1) and the demo (run_demo.ps1) first: this writes the demo database.
# Steps, each of which must succeed before the next runs (see RECORDING_GUIDE.md for what each means):
#   1. the evidence report   S1-S3 enrol, the remaining sessions held out  -> paper\results_s04\report.md
#   2. the words check       does speech recognition hear your ten words?
#   3. the calibration       leave-one-session-out voice threshold          -> data\voice_policy.json
#   4. the enrolment         the demo voiceprint, from every session        (the database is backed up first)
# Then start run_demo.ps1 and run the preflight.
param(
    [string]$Speaker = "S04",
    [string]$Sessions = "S1,S2,S3,S4,S5",
    [switch]$NoEvidence,
    [switch]$DropOutliers
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "backend"

if (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue) {
    Write-Host "Something is listening on port 8000 (the Studio or the demo). Stop it first, then run this again." -ForegroundColor Yellow
    exit 2
}

function Step([string]$title, [scriptblock]$run) {
    Write-Host ""
    Write-Host "== $title" -ForegroundColor Cyan
    & $run
    if ($LASTEXITCODE -ne 0) { Write-Host "STOPPED: '$title' did not succeed (exit $LASTEXITCODE). Nothing after it was run." -ForegroundColor Red; exit $LASTEXITCODE }
}

$all = $Sessions.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
$studio = "data\studio\$Speaker"

if (-not $NoEvidence -and $all.Count -ge 4) {
    $enrol = ($all | Select-Object -First 3) -join ","
    $held = ($all | Select-Object -Skip 3) -join ","
    Step "1. Evidence: enrol on $enrol, hold out $held" {
        & $py -m kavach.eval.enrollee --studio $studio --enrol-sessions $enrol --test-sessions $held `
            --impostors data\corpus_v2\manifest.json --impostors data\corpus_v3\manifest.json --out paper\results_s04
    }
} else {
    Write-Host "(skipping the evidence report: it needs at least four sessions, or -NoEvidence was given)" -ForegroundColor DarkGray
}

Step "2. Does speech recognition hear your words?" { & $py -m kavach.studio.words_check --speaker $Speaker }

Step "3. Calibrate the voice threshold (leave-one-session-out over $Sessions)" {
    & $py -m kavach.calibrate_voice --studio $studio --sessions $Sessions --report paper\results_s04\calibration.md
}

Step "4. Enrol the demo voiceprint from $Sessions" {
    if ($DropOutliers) { & $py -m kavach.studio.enrol --speaker $Speaker --sessions $Sessions --drop-outliers }
    else { & $py -m kavach.studio.enrol --speaker $Speaker --sessions $Sessions }
}

Write-Host ""
Write-Host "Done. Now: run_demo.ps1, wait for the sidebar to turn green, then" -ForegroundColor Green
Write-Host "  `$env:PYTHONPATH='backend'; .\.venv\Scripts\python.exe -m kavach.demo_check --presenter $Speaker --flows"
Write-Host "and one live read-through on the real microphone, in the room you will present in."
