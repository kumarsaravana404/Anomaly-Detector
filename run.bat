@echo off
echo ========================================
echo Starting Login Anomaly Detection System
echo ========================================
echo.
echo Server will start at http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

REM Check if virtual environment exists
if exist .venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
)

python app.py
pause
