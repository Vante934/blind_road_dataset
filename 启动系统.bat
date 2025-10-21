@echo off
chcp 65001 >nul
echo ========================================
echo Blind Road Detection System - Environment Detection
echo ========================================

cd /d %~dp0

echo Starting system...
python core/two_point_annotator.py

if errorlevel 1 (
    echo ERROR: Failed to start, please check Python environment
    pause
) else (
    echo System exited normally
)

pause