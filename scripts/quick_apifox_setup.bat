@echo off
chcp 65001 >nul
echo ============================================================
echo 🚀 盲道检测系统 - Apifox快速配置脚本
echo ============================================================
echo.

echo 📍 当前目录: %CD%
echo.

echo [步骤1] 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python未安装或未添加到PATH
    pause
    exit /b 1
)
echo ✅ Python环境正常
echo.

echo [步骤2] 检查依赖包...
python -c "import requests" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  requests包未安装，正在安装...
    pip install requests
)
echo ✅ 依赖包正常
echo.

echo [步骤3] 检查服务器状态...
python -c "import requests; requests.get('http://localhost:8082/status', timeout=2)" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  服务器未运行
    echo.
    echo 请先启动服务器：
    echo   1. 打开新的命令行窗口
    echo   2. 运行: python start_complete_server.py
    echo   3. 然后再次运行此脚本
    echo.
    pause
    exit /b 1
)
echo ✅ 服务器运行正常
echo.

echo [步骤4] 导出OpenAPI规范...
cd /d %~dp0..
python apifox_config/export_openapi.py

if errorlevel 1 (
    echo.
    echo ❌ 导出失败
    pause
    exit /b 1
)

echo.
echo ============================================================
echo ✅ 配置完成！
echo ============================================================
echo.
echo 📖 下一步操作：
echo   1. 打开Apifox应用
echo   2. 创建新项目
echo   3. 导入文件: apifox_config\blind_road_api.json
echo   4. 开始测试！
echo.
echo 详细说明请查看: apifox_config\README.md
echo.
pause



