@echo off
title Sepsis Early Warning System Dashboard
echo --------------------------------------------------
echo [1/2] Activating Clinical Environment...
echo --------------------------------------------------
cd /d "%~dp0"
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo [WARNING] Virtual environment not found. Attempting to run with system python...
)

echo --------------------------------------------------
echo [2/2] Launching ICU Dashboard (v6.0)...
echo --------------------------------------------------
streamlit run streamlit_app/app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Dashboard failed to start.
    echo Possible reasons:
    echo 1. Dependencies not installed (Run: pip install -r requirements.txt)
    echo 2. app.py missing from streamlit_app folder
    pause
)
