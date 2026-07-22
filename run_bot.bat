@echo off
title AI Stock Advisor Telegram Bot Launcher
cd /d "%~dp0"
echo =====================================================================
echo 📈 Starting AI Stock Advisor Telegram Bot...
echo =====================================================================
echo Active Bot Token: 8678345510:AAHXgqFp4eFpolgqUaIKT5WIHX9QxfBTBCM
echo Active Chat ID:   2104870862
echo =====================================================================
echo.

# Setup Python module search paths
set PYTHONPATH=.;src

# Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment (.venv) not found!
    echo Please ensure Python 3.12 is installed and setup before launching.
    pause
    exit /b
)

# Activate environment and launch bot client
call .venv\Scripts\activate.bat
python src/ai_stock_advisor/telegram/bot.py
pause
