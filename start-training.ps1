param(
    [double]$Hours = 10,
    [string]$Data = 'data/lichess-full',
    [string]$Run = 'runs/main',
    [int]$Width = 512,
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$runPath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $Run))
if (Test-Path -LiteralPath "$runPath/process.json") {
    $previous = Get-Content -LiteralPath "$runPath/process.json" -Raw | ConvertFrom-Json
    $process = Get-Process -Id $previous.pid -ErrorAction SilentlyContinue
    if ($process -and $process.StartTime.ToUniversalTime().ToString('o') -eq $previous.startTime) {
        throw "Training process $($previous.pid) is already running."
    }
}
if (-not (Test-Path -LiteralPath "$Data/manifest.json")) { throw "Dataset missing: $Data" }
if (-not $Resume -and (Test-Path -LiteralPath "$runPath/last.pt")) { throw 'Existing run: use -Resume or another -Run' }
if (Test-Path -LiteralPath "$runPath/STOP") { throw 'A STOP file exists; remove it before resuming.' }
New-Item -ItemType Directory -Force -Path $runPath | Out-Null
# Start-Process on Windows needs explicit quoting for paths containing spaces.
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$argumentText = '-u -m scratch_chess.train --data "' + $Data + '" --out "' + $Run + '" --hours ' + $Hours + ' --width ' + $Width
if ($Resume) { $argumentText += ' --resume "' + $Run + '/last.pt"' }
$process = Start-Process -FilePath $pythonPath -ArgumentList $argumentText -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput "$runPath/train.log" -RedirectStandardError "$runPath/train.err.log" -PassThru
@{ pid = $process.Id; startTime = $process.StartTime.ToUniversalTime().ToString('o'); run = $runPath } | ConvertTo-Json | Set-Content -LiteralPath "$runPath/process.json"
Write-Output "Training started (PID $($process.Id)). Log: $runPath/train.log"
Write-Output "Stop safely: create an empty file named STOP inside $runPath"

