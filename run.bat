@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=%SCRIPT_DIR%.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
  echo No se encontro Python en "%PYTHON_EXE%".
  echo Activa o recrea la .venv del proyecto antes de ejecutar este script.
  exit /b 1
)

"%PYTHON_EXE%" "%SCRIPT_DIR%main.py" %*
endlocal
