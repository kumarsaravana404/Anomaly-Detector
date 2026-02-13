@echo off
echo ========================================
echo Login Anomaly Detection System Setup
echo ========================================
echo.

REM Check if virtual environment exists
if exist .venv\Scripts\activate.bat (
    echo Virtual environment found. Activating...
    call .venv\Scripts\activate.bat
    echo.
    echo Installing dependencies in virtual environment...
    pip install flask numpy pandas scikit-learn matplotlib seaborn joblib
) else (
    echo Installing dependencies globally...
    pip install flask numpy pandas scikit-learn matplotlib seaborn joblib
)

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To run the application:
echo   Double-click run.bat
echo   OR
echo   python app.py
echo.
echo Then open your browser to:
echo   http://localhost:5000
echo.
pause
