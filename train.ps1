param(
    [double]$Hours = 10,
    [string]$Data = 'data/lichess-full',
    [string]$Run = 'runs/main',
    [int]$Width = 512,
    [int]$BatchSize = 16384,
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path -LiteralPath "$Data/manifest.json")) {
    throw "Prepared dataset missing: $Data. See README preparation command."
}
$trainArgs = @('-u', '-m', 'scratch_chess.train', '--data', $Data, '--out', $Run, '--hours', "$Hours", '--width', "$Width", '--batch-size', "$BatchSize")
if ($Resume) { $trainArgs += @('--resume', "$Run/last.pt") }
& '.\.venv\Scripts\python.exe' @trainArgs
if ($LASTEXITCODE -ne 0) { throw 'Training failed; inspect the error above. Saved checkpoints remain available.' }
