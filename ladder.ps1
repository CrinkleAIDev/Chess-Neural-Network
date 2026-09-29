# Play one of our networks against Stockfish at several strength settings.
# Example: .\ladder.ps1 -Model runs\main\best.scn -Rounds 50
param(
    [string]$Model = 'runs\main\best.scn',
    [int[]]$Elo = @(1800, 2200, 2600),
    [string]$TC = '10+0.1',
    [int]$Rounds = 50,
    [int]$Concurrency = 10,
    [string]$Out = 'runs/ladder'
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$sf = 'tools\stockfish\sf_19\stockfish\stockfish-windows-x86-64-universal.exe'
if (-not (Test-Path $sf)) { throw "Stockfish missing: run python -m scratch_chess.fetch_teacher" }
New-Item -ItemType Directory -Force $Out | Out-Null
$engines = @('-engine', "cmd=native\scratchchess.exe", "args=--model $Model", 'name=scratch')
foreach ($e in $Elo) {
    $engines += @('-engine', "cmd=$sf", "name=sf$e", 'option.UCI_LimitStrength=true', "option.UCI_Elo=$e",
                  'option.Threads=1', 'option.Hash=64')
}
& 'tools\fastchess\fastchess-windows-x86-64\fastchess.exe' @engines `
    -tournament gauntlet -each "tc=$TC" -rounds $Rounds -games 2 -repeat -concurrency $Concurrency `
    -openings file=tools/books/8moves_v3.pgn format=pgn order=random `
    -pgnout "file=$Out/games.pgn" -recover -ratinginterval 50
