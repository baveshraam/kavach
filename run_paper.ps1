# Everything the paper reports, regenerated from the recordings, then the IEEE paper compiled. One command; it changes no demo state.
#
#   powershell -ExecutionPolicy Bypass -File .\run_paper.ps1
#   powershell -ExecutionPolicy Bypass -File .\run_paper.ps1 -PaperOnly      # rebuild tables, figures and the PDF from results already on disk
#
# Steps, each of which must succeed before the next runs (cohort embeddings are skipped when their files already exist):
#   1. cohort embeddings   LibriSpeech dev/test-clean, Tamil SLR65, the pilot corpus, for ECAPA, ResNet and WavLM-SV   (GPU)
#   2. single enrolee      leave-one-sitting-out study per model, plus the whole-clip probe run                         -> paper\results_poc\
#   3. population          many-enrollee LibriSpeech benchmark per model                                                -> paper\results_population\
#   4. synthetic voices    kNN-VC clones of the presenter from their OWN recordings + text-to-speech, scored per model   (GPU) -> paper\results_spoof\
#   5. the paper           numbers, tables, figures, then tectonic compiles paper\ieee\main.pdf and the project report paper\report\main.pdf
# The laptop is also the training machine: this checks the GPU and running python first. The only voice ever cloned is the presenter's own
# (KAVACH_CLONE_VICTIMS is set to S04 for step 4 only; the presenter asked for this evaluation).
param(
    [string]$Speaker = "S04",
    [string]$Sessions = "S1,S2,S3",
    [switch]$PaperOnly,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$py = Join-Path $root ".venv-clone\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = Join-Path $root ".venv\Scripts\python.exe" }
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "backend"
$models = @("ecapa", "resnet", "wavlm")

function Step([string]$title, [scriptblock]$run) {
    Write-Host ""
    Write-Host "== $title" -ForegroundColor Cyan
    & $run
    if ($LASTEXITCODE -ne 0) { Write-Host "STOPPED: '$title' did not succeed (exit $LASTEXITCODE). Nothing after it was run." -ForegroundColor Red; exit $LASTEXITCODE }
}

if (-not $PaperOnly) {
    $busy = Get-Process python -ErrorAction SilentlyContinue
    if ($busy -and -not $Force) {
        Write-Host "python is already running ($($busy.Count) process(es)): a training job or a server? Stop it, or add -Force to go ahead anyway." -ForegroundColor Yellow
        exit 2
    }
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { Write-Host ("GPU: " + (& nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader)) }

    # LibriSpeech test-clean goes into its own new directory, never into an existing one
    $test = "data\cohort\librispeech_test"
    if (-not (Test-Path "$test\LibriSpeech\test-clean")) {
        if (-not (Test-Path "data\cohort_dl\test-clean.tar.gz")) { Write-Host "data\cohort_dl\test-clean.tar.gz is missing (https://www.openslr.org/12)." -ForegroundColor Red; exit 2 }
        New-Item -ItemType Directory -Force $test | Out-Null
        tar -xzf "data\cohort_dl\test-clean.tar.gz" -C $test --no-same-owner
    }

    function Embed([string]$corpus, [string]$emb, [string]$out, [string]$file, [string[]]$more) {
        if (Test-Path (Join-Path $out $file)) { return }
        Step "embed $corpus with $emb -> $file" { & $py -m kavach.cohort --corpus $corpus --embedder $emb --device cuda --out $out @more }
    }
    foreach ($m in $models) {
        $d = if ($m -eq "ecapa") { "data\cohort\emb" } else { "data\cohort\emb_$m" }
        $t = "data\cohort\emb_test_$m"
        Embed "libri-test" $m $t "libri-test__$m.npz" @("--root", $test)
        Embed "libri-test" $m $t "libri-test__$m@5s.npz" @("--root", $test, "--chunk", "5")
        Embed "libri-dev" $m $d "libri-dev__$m.npz" @("--root", "data\cohort\librispeech")
        Embed "libri-dev" $m $d "libri-dev__$m@5s.npz" @("--root", "data\cohort\librispeech", "--chunk", "5")
        Embed "tamil-m" $m $d "tamil-m__$m@5s.npz" @("--root", "data\cohort\tamil", "--chunk", "5")
        Embed "tamil-f" $m $d "tamil-f__$m@5s.npz" @("--root", "data\cohort\tamil", "--chunk", "5")
        Embed "kavach12" $m $d "kavach12__$m@5s.npz" @("--from-cache", "data\cohort\emb\kavach12__ecapa.npz", "--chunk", "5")
    }

    foreach ($m in $models) {
        Step "single enrollee, $m" { & $py -m kavach.eval.poc_run --studio "data\studio\$Speaker" --sessions $Sessions --embedder $m --device cuda }
    }
    Step "single enrollee, whole-clip probes (the artefact), ecapa" {
        & $py -m kavach.eval.poc_run --studio "data\studio\$Speaker" --sessions $Sessions --embedder ecapa --device cuda --probe-seconds 0 --sweep 0 --out paper\results_poc_whole\ecapa
    }
    foreach ($m in $models) { Step "population, $m" { & $py -m kavach.eval.population_run --embedder $m } }

    $env:KAVACH_CLONE_VICTIMS = "[`"$Speaker`"]"
    Step "synthetic voices: generate (kNN-VC from $Speaker's own held-out sitting, text-to-speech)" {
        & $py -m kavach.eval.spoof_run generate --victim $Speaker --sessions $Sessions
    }
    foreach ($m in $models) { Step "synthetic voices: score, $m" { & $py -m kavach.eval.spoof_run score --victim $Speaker --embedder $m --device cuda } }
}

Step "paper: numbers, tables, figures" { & $py -m kavach.eval.paper_build }
Step "paper: compile paper\ieee\main.pdf" { Push-Location paper\ieee; try { tectonic main.tex } finally { Pop-Location } }
if (Test-Path paper\report\main.tex) {
    Step "project report: compile paper\report\main.pdf" { Push-Location paper\report; try { tectonic main.tex } finally { Pop-Location } }
}
Write-Host ""
Write-Host "Done. Open paper\ieee\main.pdf (the IEEE paper) and paper\report\main.pdf (the project report). Every number in them is a macro from paper\ieee\generated\numbers.tex." -ForegroundColor Green
