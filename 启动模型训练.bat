@echo off
chcp 65001 >nul
echo ========================================
echo Blind Road Detection System - Training Center
echo ========================================
echo.
echo Starting Integrated Training Interface...
echo.
echo Features:
echo - Blind road obstacle detection annotation
echo - Environment detection annotation tool (24 classes)
echo - Model training and validation
echo - Data preparation and export
echo.
echo Starting...

python model_training_interface.py

if errorlevel 1 (
    echo.
    echo ERROR: Failed to start, please check Python environment
    echo.
    echo Common solutions:
    echo 1. Ensure Python 3.7+ is installed
    echo 2. Install PyQt5: pip install PyQt5
    echo 3. Install OpenCV: pip install opencv-python
    echo 4. Install NumPy: pip install numpy
    echo 5. Install ultralytics: pip install ultralytics
    echo.
    pause
) else (
    echo.
    echo SUCCESS: Integrated training interface exited normally
)

pause
