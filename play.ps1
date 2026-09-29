param([string]$Model = 'runs/main/best.npz', [double]$Seconds = 1)
Set-Location $PSScriptRoot
& '.\.venv\Scripts\python.exe' -m scratch_chess.play --model $Model --seconds $Seconds
