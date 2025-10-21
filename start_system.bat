@echo off
chcp 65001 >nul
echo ========================================
echo       Blind Road Detection System
echo ========================================
echo.

:: Check Python environment
echo [1/6] Checking Python environment...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found, please install Python 3.8+
    pause
    exit /b 1
)
echo Python environment OK

:: Check ADB tools
echo [2/6] Checking ADB tools...
adb version >nul 2>&1
if errorlevel 1 (
    echo Warning: ADB not found, please ensure Android SDK is installed
    echo Please add ADB to system PATH
)

:: Check connected devices
echo [3/6] Checking connected Android devices...
adb devices | findstr "device$" >nul
if errorlevel 1 (
    echo Warning: No connected Android device detected
    echo Please ensure:
    echo 1. Phone connected via USB
    echo 2. Developer options and USB debugging enabled
    echo 3. USB debugging authorized on phone
    echo.
    set /p choice="Continue to start PC server? (y/n): "
    if /i not "%choice%"=="y" (
        echo Startup cancelled
        pause
        exit /b 1
    )
) else (
    echo Android device detected
)

:: Start PC server
echo [4/6] Starting PC server...
echo Starting server, please wait...
start "PC Server" python pc_server.py

:: Wait for server startup
timeout /t 3 /nobreak >nul

:: Start debug interface
echo [5/6] Starting debug interface...
if exist "debug_interface.py" (
    start "Debug Interface" python debug_interface.py
)

:: Start realtime display (if device connected)
echo [6/6] Starting realtime display...
if exist "mobile_realtime_display.py" (
    start "Realtime Display" python mobile_realtime_display.py
)

echo.
echo ========================================
echo       System startup completed!
echo ========================================
echo.
echo You can now:
echo 1. Monitor system status in debug interface
echo 2. Open "Blind Road Detection" app on phone
echo 3. Click "Start Detection" for realtime detection
echo 4. View detection results in debug interface
echo 5. Manage devices and train models in debug interface
echo.
echo System integrated with blind road detection and navigation!
echo.
echo Press any key to exit...
pause >nul
