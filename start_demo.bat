@echo off
title Logistics AI Copilot - Demo
cd /d "%~dp0"

echo ============================================================
echo   Logistics AI Copilot ^& MLOps Platform
echo   Starting demo server...
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo Run first:  python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

if not exist "models\registry.json" (
    echo [SETUP] First run detected - generating data and training models...
    .venv\Scripts\python scripts\generate_data.py
    .venv\Scripts\python scripts\train_model.py
    .venv\Scripts\python scripts\ingest_docs.py
    echo.
)

echo [OK] Opening browser at http://localhost:8000
echo [OK] Press Ctrl+C to stop the server
echo.
start "" http://localhost:8000
.venv\Scripts\python -m uvicorn src.api.main:app --port 8000
