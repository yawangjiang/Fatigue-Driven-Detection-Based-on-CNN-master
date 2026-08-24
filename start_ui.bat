@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_EXE=D:\Miniconda3\python.exe"

if exist "%~dp0fatigue_models.local.bat" call "%~dp0fatigue_models.local.bat"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] fatigue_cpu Python was not found:
    echo %PYTHON_EXE%
    pause
    exit /b 1
)

"%PYTHON_EXE%" fatigue_ui.py
if errorlevel 1 pause
