@echo off
title Fairness Toolkit — Launcher
color 0A
cls

echo.
echo  =====================================================
echo   Algorithmic Fairness Audit ^& Bias Mitigation Toolkit
echo  =====================================================
echo.

:: ── Check Python ──────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found.
    echo  Install from https://www.python.org/downloads/
    echo  Make sure to tick "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo  [INFO] Python version:
python --version
echo.

:: ── Fix PowerShell execution policy silently ──────────────
powershell -Command "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force" >nul 2>&1

:: ── Create venv if it doesn't exist ───────────────────────
if not exist "venv\Scripts\python.exe" (
    echo  [SETUP] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo  [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo  [OK] Virtual environment created.
    echo.
    echo  [SETUP] Upgrading pip...
    venv\Scripts\python.exe -m pip install --upgrade pip --quiet

    echo  [SETUP] Installing core packages (numpy, pandas, scikit-learn)...
    venv\Scripts\python.exe -m pip install --only-binary=:all: numpy pandas scikit-learn
    if errorlevel 1 (
        echo  [RETRY] Trying pinned versions...
        venv\Scripts\python.exe -m pip install "numpy==1.26.4" "pandas==2.2.1" "scikit-learn==1.4.1" --only-binary=:all:
        if errorlevel 1 (
            echo  [RETRY] Trying older stable versions...
            venv\Scripts\python.exe -m pip install "numpy==1.24.3" "pandas==2.0.3" "scikit-learn==1.3.2" --only-binary=:all:
        )
    )

    echo  [SETUP] Installing UI packages (streamlit, plotly, matplotlib)...
    venv\Scripts\python.exe -m pip install --only-binary=:all: streamlit plotly matplotlib seaborn joblib
    if errorlevel 1 (
        venv\Scripts\python.exe -m pip install streamlit plotly matplotlib seaborn joblib
    )

    echo  [SETUP] Installing fairlearn and reportlab...
    venv\Scripts\python.exe -m pip install fairlearn reportlab
    if errorlevel 1 (
        echo  [WARN] fairlearn install had issues, trying again...
        venv\Scripts\python.exe -m pip install fairlearn --no-build-isolation
    )

    echo.
    echo  [OK] All packages installed successfully.
    echo.
) else (
    echo  [OK] Virtual environment found. Skipping install.
    echo.
)

:: ── Launch Streamlit using venv python directly ────────────
echo  [OK] Starting Algorithmic Fairness Toolkit...
echo  [OK] Browser will open at http://localhost:8501
echo.
echo  -------------------------------------------------------
echo  Press Ctrl+C in this window to STOP the server.
echo  -------------------------------------------------------
echo.

venv\Scripts\python.exe -m streamlit run app.py --server.headless false

echo.
echo  [INFO] Server stopped.
pause
