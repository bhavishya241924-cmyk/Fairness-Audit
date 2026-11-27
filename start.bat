@echo off
title Fairness Toolkit
cd /d "%~dp0"
venv\Scripts\python.exe -m streamlit run app.py
pause
