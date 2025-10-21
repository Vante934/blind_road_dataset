@echo off
chcp 65001 >nul
echo ========================================
echo 语音播报系统启动器
echo ========================================
echo.

echo 正在检查Python环境...
python --version
if %errorlevel% neq 0 (
    echo 错误: Python未安装或未添加到PATH
    pause
    exit /b 1
)

echo.
echo 正在检查依赖包...
python -c "import edge_tts, pyttsx3, pygame, PyQt5" 2>nul
if %errorlevel% neq 0 (
    echo 正在安装语音系统依赖包...
    pip install edge-tts pyttsx3 pygame PyQt5 azure-cognitiveservices-speech
    if %errorlevel% neq 0 (
        echo 错误: 依赖包安装失败
        pause
        exit /b 1
    )
)

echo.
echo 选择启动模式:
echo 1. 语音库管理器 (GUI界面)
echo 2. 语音系统测试
echo 3. 简化语音测试
echo 4. 退出
echo.

set /p choice=请输入选择 (1-4): 

if "%choice%"=="1" (
    echo 启动语音库管理器...
    python simplified_voice_manager.py
) else if "%choice%"=="2" (
    echo 启动语音系统测试...
    python test_voice_system.py
) else if "%choice%"=="3" (
    echo 启动简化语音测试...
    python test_voice_gui.py
) else if "%choice%"=="4" (
    echo 退出
    exit /b 0
) else (
    echo 无效选择，请重新运行
    pause
    exit /b 1
)

echo.
echo 程序执行完成
pause


