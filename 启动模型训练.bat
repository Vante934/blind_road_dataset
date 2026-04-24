@echo off
chcp 65001 >nul
echo ========================================
echo Blind Road Detection System - Enhanced Trainer
echo ========================================
echo.
echo Starting Enhanced Training Interface...
echo.
echo Workflow:
echo Step 1: Data Preparation (Using this tool)
echo - Input: Street view images
echo - Annotation: Blind road, manhole covers, static obstacles
echo - Smart Data Health Check: Detect and clean bad data, invalid annotations, and tiny objects
echo - Class Balance Visualization: Identify sample imbalance with histogram
echo - Augmentation Strategy Config: Configure Mosaic, MixUp, HSV, Flip
echo - Output: Cleaned and balanced dataset
echo.
echo Step 2: Model Training
echo - Model Selection: YOLOv8n/s/m
echo - Hyperparameter Control: ImgSize (640/1280), Epochs, Batch Size
echo - Training Process: Monitor progress and metrics in real-time
echo - Output Model: Model_A (YOLO-Seg) - Prevents falls
echo.
echo Step 3: Model Evaluation
echo - Confusion Matrix Visualization: Diagnose model errors with heatmap
echo - Advanced Metrics Radar: Multi-dimensional evaluation (mAP, Recall, Precision)
echo - Bad Case Gallery: Identify common failure patterns with visualization
echo - Confidence Threshold Optimization: Find optimal balance between precision and recall
echo.
echo Step 4: Dynamic/Sound Data Production (Using Label Studio)
echo - Input: Short videos with sound
echo - Annotation: Moving vehicles, horn sounds
echo - Output Models: Model_B (YOLO-Detect) + Model_C (Audio-Cls) - Prevents accidents
echo.
echo Step 5: Code Integration (In system)
echo - Combine all models for comprehensive safety
echo.
echo Features:
echo - Blind road and static obstacle annotation
echo - Smart data health check with automatic cleaning
echo - Class balance analysis with histogram visualization
echo - Configurable data augmentation (Mosaic, MixUp, HSV, Flip)
echo - Hyperparameter tuning interface with real-time feedback
echo - Advanced model evaluation with confusion matrix and heatmap
echo - Bad case analysis and visualization gallery
echo - Camera detection with voice announcements and obstacle visualization
echo - Confidence threshold optimization for deployment
echo.
echo Starting...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ and add it to PATH
    echo.
    pause
    exit /b 1
)

REM Check for required dependencies
echo Checking dependencies...

python -c "import PyQt5" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing PyQt5...
    pip install PyQt5
)

python -c "import cv2" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing OpenCV...
    pip install opencv-python
)

python -c "import numpy" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing NumPy...
    pip install numpy
)

python -c "import ultralytics" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Ultralytics...
    pip install ultralytics
)

python -c "import pyttsx3" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing pyttsx3...
    pip install pyttsx3
)

python -c "import matplotlib" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Matplotlib...
    pip install matplotlib
)

python -c "import seaborn" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Seaborn...
    pip install seaborn
)

python -c "import sklearn" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing scikit-learn...
    pip install scikit-learn
)

echo.
echo All dependencies are ready, starting application...
echo.

REM Start the application
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
    echo 6. Install pyttsx3: pip install pyttsx3
    echo 7. Install Matplotlib: pip install matplotlib
    echo 8. Install Seaborn: pip install seaborn
    echo 9. Install scikit-learn: pip install scikit-learn
    echo.
    pause
) else (
    echo.
    echo SUCCESS: Enhanced training interface exited normally
)

pause