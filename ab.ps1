# A/B test two engine builds (or two networks) with fastchess SPRT.
# Example: .\ab.ps1 -New native\scratchchess.exe -Base native\baseline.exe -Name tt-cutoffs
param(
    [string]$New = 'native\scratchchess.exe',
    [string]$Base = 'native\baseline.exe',
    [string]$NewModel = 'runs\ready-check\best.scn',
    [string]$BaseModel = '',
    [string]$Name = 'ab',
    [string]$TC = '5+0.05',
    [int]$Concurrency = 10,
    [int]$Rounds = 1500,
    [double]$Elo1 = 10
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not $BaseModel) { $BaseModel = $NewModel }
New-Item -ItemType Directory -Force runs\ab | Out-Null
# Relative paths: fastchess splits engine args on spaces and this project path contains one.
$fastchess = 'tools\fastchess\fastchess-windows-x86-64\fastchess.exe'
& $fastchess `
    -engine "cmd=$New" "args=--model $NewModel" name=new `
    -engine "cmd=$Base" "args=--model $BaseModel" name=base `
    -each "tc=$TC" -rounds $Rounds -games 2 -repeat -concurrency $Concurrency `
    -openings file=tools/books/8moves_v3.pgn format=pgn order=random `
    -sprt elo0=0 "elo1=$Elo1" alpha=0.05 beta=0.05 `
    -pgnout "file=runs/ab/$Name.pgn" -recover -ratinginterval 50
