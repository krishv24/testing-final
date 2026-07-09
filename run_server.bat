@echo off
echo ===================================================
echo Starting Hierarchical AI Meteorologist Server
echo ===================================================

:: 1. Check if venv exists, otherwise create it
if not exist "venv" (
    echo Creating virtual environment venv...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create venv. Please ensure Python is installed and added to PATH.
        pause
        exit /b 1
    )
)

:: 2. Install/update requirements
echo Installing and updating requirements...
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\pip.exe install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements.
    pause
    exit /b 1
)

:: 3. Start uvicorn server
echo Starting the development server...
venv\Scripts\python.exe -m uvicorn app.main:app --reload
