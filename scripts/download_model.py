#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动下载YOLO模型脚本
"""

import os
import urllib.request
import zipfile
import shutil
from pathlib import Path

# 模型下载地址
MODEL_URL = "https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt"
MODEL_NAME = "best.pt"
MODEL_DIR = Path("models")

def download_model():
    """下载YOLO模型"""
    # 创建模型目录
    MODEL_DIR.mkdir(exist_ok=True)
    
    # 目标文件路径
    model_path = MODEL_DIR / MODEL_NAME
    
    # 检查模型是否已存在
    if model_path.exists():
        print(f"✅ 模型文件 {model_path} 已存在，跳过下载")
        return True
    
    print(f"🔄 正在下载模型文件: {MODEL_URL}")
    print(f"📁 保存路径: {model_path}")
    
    try:
        # 下载模型文件
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print(f"✅ 模型下载成功: {model_path}")
        return True
    except Exception as e:
        print(f"❌ 模型下载失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("自动下载YOLO模型")
    print("=" * 60)
    
    # 下载模型
    success = download_model()
    
    if success:
        print("\n✅ 模型准备完成，可以运行测试了")
        print("\n运行测试命令:")
        print("  python tests\test_detector.py")
    else:
        print("\n❌ 模型下载失败，请检查网络连接")

if __name__ == "__main__":
    main()
