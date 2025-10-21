@echo off
chcp 65001 >nul
echo ========================================
echo 🎯 语音库管理系统启动器
echo ========================================
echo.
echo 💡 功能说明:
echo    • 管理四种语音提示类型
echo    • 支持数据集自动分析
echo    • 可调整大小的分区界面
echo    • 自动生成子分类标识
echo.

:: 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: 未找到Python，请先安装Python 3.7+
    echo 下载地址: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo ✅ Python环境检查通过

:: 检查必要的依赖包
echo.
echo 🔍 正在检查依赖包...

python -c "import PyQt5" >nul 2>&1
if errorlevel 1 (
    echo ⚠️ 正在安装PyQt5...
    pip install PyQt5
    if errorlevel 1 (
        echo ❌ PyQt5安装失败，请手动安装: pip install PyQt5
        echo.
        pause
        exit /b 1
    )
    echo ✅ PyQt5安装成功
) else (
    echo ✅ PyQt5已安装
)

python -c "import pyttsx3" >nul 2>&1
if errorlevel 1 (
    echo ⚠️ 正在安装pyttsx3...
    pip install pyttsx3
    if errorlevel 1 (
        echo ❌ pyttsx3安装失败，请手动安装: pip install pyttsx3
        echo.
        pause
        exit /b 1
    )
    echo ✅ pyttsx3安装成功
) else (
    echo ✅ pyttsx3已安装
)

echo.
echo ✅ 所有依赖包检查完成

:: 启动语音库管理系统
echo.
echo 🚀 正在启动语音库管理系统...
echo.
echo 📋 使用提示:
echo    • 左侧: 选择主要分类，管理子分类和文本
echo    • 中间: 编辑文本内容，测试语音效果
echo    • 右侧: 分析外部数据集，自动导入
echo    • 分区: 可拖拽调整各区域大小
echo.

python voice_library_manager_gui.py

if errorlevel 1 (
    echo.
    echo ❌ 语音库管理系统启动失败
    echo 请检查错误信息并重试
    echo.
    pause
    exit /b 1
)

echo.
echo ✅ 语音库管理系统已关闭
echo 感谢使用！
pause
