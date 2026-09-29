$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    # Reuse an existing CUDA torch installation when present; project dependencies
    # are installed in .venv. This is intentional and documented in README.
    python -m venv --system-site-packages .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create Python environment' }
}
& '.\.venv\Scripts\python.exe' -m pip install -e '.[dev]'
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& '.\.venv\Scripts\python.exe' -m scratch_chess.doctor
if ($LASTEXITCODE -ne 0) { throw 'CUDA verification failed. See README.' }
