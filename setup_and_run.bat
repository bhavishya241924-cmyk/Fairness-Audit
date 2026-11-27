@echo off
title First-Time Setup — Fairness Toolkit
color 0B
cls

echo.
echo  =====================================================
echo   FIRST-TIME SETUP — Fairness Audit Toolkit
echo  =====================================================
echo.
echo  This will:
echo    1. Check Python installation
echo    2. Delete any old virtual environment
echo    3. Create a fresh virtual environment
echo    4. Install all required packages
echo.
echo  After this, just double-click run.bat every time.
echo.
pause

:: ── Python check ──────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [ERROR] Python is not installed or not in PATH.
    echo.
    echo  Steps to fix:
    echo    1. Download Python 3.10+ from https://www.python.org/downloads/
    echo    2. During install, tick "Add Python to PATH"
    echo    3. Restart this setup.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do echo  [OK] Found %%i

:: ── Clean previous venv ────────────────────────────────────
if exist "venv" (
    echo.
    echo  [SETUP] Removing old virtual environment...
    rmdir /s /q venv
)

:: ── Create fresh venv ─────────────────────────────────────
echo  [SETUP] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo  [ERROR] venv creation failed.
    pause
    exit /b 1
)
echo  [OK] Virtual environment created.

:: ── Activate ──────────────────────────────────────────────
call venv\Scripts\activate.bat

:: ── Upgrade pip ───────────────────────────────────────────
echo.
echo  [SETUP] Upgrading pip...
python -m pip install --upgrade pip --quiet

:: ── Install packages ──────────────────────────────────────
echo  [SETUP] Installing packages (3-5 minutes)...
echo         streamlit, pandas, scikit-learn, fairlearn, plotly, reportlab ...
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo  [ERROR] Some packages failed to install.
    echo  Check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo  =====================================================
echo   SETUP COMPLETE!
echo  =====================================================
echo.
echo  To start the app anytime, double-click:
echo       run.bat
echo.
echo  Starting the app now...
echo.

streamlit run app.py --server.headless false

pause
