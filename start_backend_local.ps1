$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot 'venv\Scripts\python.exe'

if (-not (Test-Path $pythonExe)) {
    throw "Python executable not found at $pythonExe. Create/activate venv first."
}

$env:APP_HOST = '127.0.0.1'
$env:APP_PORT = '5000'
$env:APP_DEBUG = '0'
$env:PYTHONDONTWRITEBYTECODE = '1'

$stalePyc = Join-Path $projectRoot 'app\__pycache__\app.cpython-313.pyc'
if (Test-Path $stalePyc) {
    Remove-Item $stalePyc -Force -ErrorAction SilentlyContinue
}

Write-Host "Starting backend on http://$($env:APP_HOST):$($env:APP_PORT)"
& $pythonExe (Join-Path $projectRoot 'app\app.py')
