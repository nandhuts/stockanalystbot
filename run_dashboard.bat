@echo off
title AI Stock Advisor Web Dashboard Launcher
cd /d "%~dp0"
echo =====================================================================
echo 🖥️ Starting AI Stock Advisor Web Console Dashboard...
echo =====================================================================
echo.

REM Setup Python module search paths
set PYTHONPATH=.;src

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment (.venv) not found!
    echo Please ensure Python 3.12 is installed and setup before launching.
    pause
    exit /b
)

REM Activate environment and launch streamlit app
call .venv\Scripts\activate.bat
streamlit run src/ai_stock_advisor/dashboard/app.py
pause
