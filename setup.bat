@echo off
cd /d "%~dp0"
echo ===== Parking Entrance System - one-time setup =====
echo.

set PY=python
where python >nul 2>nul || set PY=py -3
%PY% -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3,10) else 1)" >nul 2>nul
if errorlevel 1 (
  echo Python 3.10 or newer was not found.
  echo Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH", then run setup.bat again.
  pause
  exit /b 1
)

if not exist venv\Scripts\python.exe (
  echo Creating a private Python environment ...
  %PY% -m venv venv
  if errorlevel 1 ( echo Could not create the environment. & pause & exit /b 1 )
)

call venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo.
echo Installing required packages (internet needed for this step only) ...
pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Installation failed. Check your internet connection and run setup.bat again.
  pause
  exit /b 1
)

echo.
echo ===== Setup finished. =====
echo Next steps:
echo   1. Open config.yaml in Notepad and put in your two camera addresses.
echo   2. Make sure the MongoDB service is running.
echo   3. Double-click check.bat to test, then run.bat to start.
pause
