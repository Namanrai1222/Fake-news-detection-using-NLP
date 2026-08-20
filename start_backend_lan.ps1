$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot 'venv\Scripts\python.exe'

if (-not (Test-Path $pythonExe)) {
    throw "Python executable not found at $pythonExe. Create/activate venv first."
}

$env:APP_HOST = '0.0.0.0'
$env:APP_PORT = '5000'
$env:APP_DEBUG = '0'
$env:PYTHONDONTWRITEBYTECODE = '1'

$stalePyc = Join-Path $projectRoot 'app\__pycache__\app.cpython-313.pyc'
if (Test-Path $stalePyc) {
    Remove-Item $stalePyc -Force -ErrorAction SilentlyContinue
}

Write-Host "Starting backend on all interfaces (LAN) at port $($env:APP_PORT)"
Write-Host "Use this machine IP from other devices, e.g. http://192.168.x.x:$($env:APP_PORT)"
& $pythonExe (Join-Path $projectRoot 'app\app.py')
