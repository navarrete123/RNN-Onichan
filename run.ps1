$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $scriptDir ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Error "No se encontro Python en '$pythonExe'. Activa o recrea la .venv del proyecto antes de ejecutar este script."
    exit 1
}

& $pythonExe (Join-Path $scriptDir "main.py") @args
