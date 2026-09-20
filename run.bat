@echo off
rem Lab 1 launcher for Windows.
rem Creates a virtual environment, installs dependencies, runs the lab and builds the report.
rem NOTE: this file is intentionally ASCII-only -- cmd.exe mis-parses batch files
rem that mix "chcp 65001" with non-ASCII characters.
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1

set PY=py -3
%PY% -c "import sys" >nul 2>&1
if errorlevel 1 set PY=python
%PY% -c "import sys" >nul 2>&1
if errorlevel 1 (
    echo Python 3.10+ not found. Install it from python.org and run again.
    pause
    exit /b 1
)

if not exist ".venv" (
    echo [1/4] Creating virtual environment .venv ...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"

echo [2/4] Installing dependencies ...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies. Check your internet connection.
    pause
    exit /b 1
)

echo [3/4] Running the lab (Iris, Wine, Penguins) ...
python "lab1\code\main.py" --dataset all
if errorlevel 1 (
    echo The lab script failed.
    pause
    exit /b 1
)

echo [4/4] Building the report ...
python "lab1\report\build_report.py"

echo.
echo Done. Tables: lab1\code\results  Figures: lab1\report\images  Report: the .docx file in lab1\report\
pause
