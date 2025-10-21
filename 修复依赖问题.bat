@echo off
chcp 65001 >nul
echo ========================================
echo 盲道检测系统 - 依赖问题修复工具
echo ========================================
echo.
echo 检测到Python 3.13兼容性问题，正在修复...
echo.

REM 检查Python版本
python --version
echo.

REM 升级pip
echo 🔧 升级pip...
python -m pip install --upgrade pip

REM 安装核心依赖
echo 📦 安装核心依赖包...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo ⚠️ CUDA版本安装失败，尝试CPU版本...
    pip install torch torchvision torchaudio
)

pip install ultralytics
pip install opencv-python
pip install numpy pandas matplotlib seaborn
pip install tqdm rich loguru
pip install PyYAML requests

echo.
echo ✅ 核心依赖安装完成

REM 安装可选依赖
echo 📦 安装可选依赖包...
pip install boto3 paramiko wandb mlflow tensorboard
pip install optuna hyperopt
pip install albumentations imgaug
pip install onnx onnxruntime

echo.
echo ✅ 可选依赖安装完成

REM 测试导入
echo 🧪 测试关键模块导入...
python -c "import torch; print('PyTorch版本:', torch.__version__)"
python -c "import ultralytics; print('Ultralytics版本:', ultralytics.__version__)"
python -c "import cv2; print('OpenCV版本:', cv2.__version__)"
python -c "import numpy; print('NumPy版本:', numpy.__version__)"

echo.
echo ✅ 依赖修复完成！
echo.
echo 现在可以运行: 一键训练部署.bat
echo.
pause









