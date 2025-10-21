@echo off
chcp 65001 >nul
echo ========================================
echo 盲道检测云服务器自动训练部署系统
echo ========================================
echo.
echo 功能特性:
echo - 自动下载和准备数据集
echo - 支持多种YOLO模型架构
echo - 超参数自动优化
echo - 云服务器自动部署
echo - 实时训练监控
echo - 自动结果下载
echo.
echo 开始执行...

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python未安装或未添加到PATH
    echo 请先安装Python 3.8+
    pause
    exit /b 1
)

REM 显示Python版本信息
echo 📋 Python版本信息:
python --version
echo.

echo ✅ Python环境检查通过

REM 安装依赖包
echo 📦 安装依赖包...
echo 使用简化版依赖包以确保Python 3.12兼容性...

REM 先升级pip
echo 🔧 升级pip...
python -m pip install --upgrade pip

REM 安装核心依赖
echo 📦 安装核心依赖包...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
pip install opencv-python
pip install numpy pandas matplotlib seaborn
pip install tqdm rich loguru
pip install PyYAML requests
pip install Pillow

if errorlevel 1 (
    echo ❌ 核心依赖安装失败，尝试安装基础依赖...
    pip install torch torchvision ultralytics opencv-python numpy pandas matplotlib seaborn PyYAML requests
    if errorlevel 1 (
        echo ❌ 基础依赖安装失败
        pause
        exit /b 1
    )
)

echo ✅ 依赖包安装完成

REM 创建必要目录
echo 📁 创建项目目录...
if not exist "datasets" mkdir datasets
if not exist "results" mkdir results
if not exist "models" mkdir models
if not exist "logs" mkdir logs

echo ✅ 目录创建完成

REM 下载数据集
echo 📥 下载和准备数据集...
python dataset_downloader.py
if errorlevel 1 (
    echo ⚠️ 数据集准备失败，但继续执行...
    echo 您可以稍后手动运行: python dataset_downloader.py
)

echo ✅ 数据集准备完成

REM 检查云服务配置
echo ☁️ 检查云服务配置...
if not exist "cloud_config.yaml" (
    echo ⚠️ 云服务配置文件不存在，将使用默认配置
)

if not exist "training_config.yaml" (
    echo ⚠️ 训练配置文件不存在，将使用默认配置
)

echo ✅ 配置文件检查完成

REM 选择运行模式
echo.
echo 请选择运行模式:
echo 1. 快速测试 (使用现有数据集，5分钟完成)
echo 2. 简化版训练 (推荐，兼容Python 3.13)
echo 3. 完整版本地训练 (需要更多依赖)
echo 4. 云服务器训练 (推荐用于正式训练)
echo 5. 仅准备数据集
echo 6. 仅下载现有结果
echo 7. 修复依赖问题
echo.
set /p choice="请输入选择 (1-7): "

if "%choice%"=="1" (
    echo 🚀 开始快速测试...
    python 快速测试.py
    if errorlevel 1 (
        echo ❌ 快速测试失败
        pause
        exit /b 1
    )
    echo ✅ 快速测试完成
    goto :end
)

if "%choice%"=="2" (
    echo 🏠 开始简化版训练...
    python 简化训练脚本.py
    if errorlevel 1 (
        echo ❌ 简化版训练失败
        pause
        exit /b 1
    )
    echo ✅ 简化版训练完成
    goto :end
)

if "%choice%"=="3" (
    echo 🏠 开始完整版本地训练...
    python advanced_training_system.py
    if errorlevel 1 (
        echo ❌ 完整版训练失败
        pause
        exit /b 1
    )
    echo ✅ 完整版训练完成
    goto :end
)

if "%choice%"=="4" (
    echo ☁️ 开始云服务器部署和训练...
    
    REM 检查AWS凭证
    echo 检查AWS凭证...
    aws sts get-caller-identity >nul 2>&1
    if errorlevel 1 (
        echo ⚠️ AWS凭证未配置，请先配置AWS CLI
        echo 运行: aws configure
        pause
        exit /b 1
    )
    
    echo ✅ AWS凭证检查通过
    
    REM 执行云部署
    python cloud_deployment.py
    if errorlevel 1 (
        echo ❌ 云服务器部署失败
        pause
        exit /b 1
    )
    echo ✅ 云服务器训练完成
    goto :end
)

if "%choice%"=="5" (
    echo 📊 仅准备数据集...
    python dataset_downloader.py
    echo ✅ 数据集准备完成
    goto :end
)

if "%choice%"=="6" (
    echo 📥 下载现有结果...
    echo 请手动运行: python cloud_deployment.py
    echo 然后选择下载结果选项
    goto :end
)

if "%choice%"=="7" (
    echo 🔧 修复依赖问题...
    call 修复依赖问题.bat
    goto :end
)

echo ❌ 无效选择
pause
exit /b 1

:end
echo.
echo ========================================
echo 执行完成！
echo ========================================
echo.
echo 结果文件位置:
echo - 训练结果: results/
echo - 模型文件: models/
echo - 日志文件: logs/
echo - 数据集: datasets/
echo.
echo 查看训练日志:
echo - 本地训练: logs/training.log
echo - 云服务器: 通过SSH连接查看
echo.
echo 如需帮助，请查看项目文档或联系开发者
echo.
pause

