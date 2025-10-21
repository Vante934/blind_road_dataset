@echo off
chcp 65001 >nul
echo ========================================
echo 盲道检测系统 - 开发环境快速配置
echo ========================================
echo.

echo 正在检查Python环境...
python --version
if errorlevel 1 (
    echo 错误: Python未安装或未添加到PATH
    echo 请访问 https://www.python.org/downloads/ 下载安装Python
    pause
    exit /b 1
)

echo.
echo 正在创建虚拟环境...
python -m venv venv
if errorlevel 1 (
    echo 错误: 创建虚拟环境失败
    pause
    exit /b 1
)

echo.
echo 正在激活虚拟环境...
call venv\Scripts\activate.bat

echo.
echo 正在升级pip...
python -m pip install --upgrade pip

echo.
echo 正在安装依赖包...
pip install -r requirements.txt
if errorlevel 1 (
    echo 警告: 部分依赖安装失败，请手动检查
)

echo.
echo 正在检查Android环境...
where adb >nul 2>&1
if errorlevel 1 (
    echo 警告: Android SDK未配置，请安装Android Studio
    echo 访问: https://developer.android.com/studio
) else (
    echo Android SDK已配置
)

echo.
echo 正在运行基础测试...
python test_basic.py
if errorlevel 1 (
    echo 警告: 基础测试失败
)

echo.
echo ========================================
echo 环境配置完成！
echo ========================================
echo.
echo 下一步操作:
echo 1. 激活虚拟环境: venv\Scripts\activate
echo 2. 运行系统: .\启动系统.bat
echo 3. 运行训练: .\启动模型训练.bat
echo.
pause
