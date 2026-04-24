#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/setup_test_environment.py
自动准备测试环境
"""

from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np


def create_models_dir():
    """创建models目录"""
    models_dir = Path("../models")
    models_dir.mkdir(exist_ok=True)
    print("✅ models目录已创建")
    return models_dir


def download_yolo_model(models_dir):
    """下载YOLO预训练模型"""
    model_path = models_dir / "best.pt"
    yolo8n_path = models_dir / "yolov8n.pt"
    
    if model_path.exists():
        print(f"✅ 模型已存在: {model_path}")
        return model_path
    
    print("⏬ 正在下载YOLOv8n预训练模型...")
    try:
        # 下载yolov8n模型
        model = YOLO('yolov8n.pt')  # 下载最小的模型
        model.save(str(yolo8n_path))
        print(f"✅ 模型已下载到: {yolo8n_path}")
        
        # 复制到best.pt
        import shutil
        shutil.copy2(str(yolo8n_path), str(model_path))
        print(f"✅ 模型已复制到: {model_path}")
        
        return model_path
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None


def create_test_images():
    """创建测试图片"""
    test_images_dir = Path("test_images")
    test_images_dir.mkdir(exist_ok=True)
    
    # 创建几张测试图片
    for i in range(3):
        # 创建随机图像
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 添加一些形状
        cv2.rectangle(img, (100+i*50, 100), (200+i*50, 200), (0, 255, 0), -1)
        cv2.circle(img, (400, 300+i*30), 50, (255, 0, 0), -1)
        
        # 保存
        img_path = test_images_dir / f"test{i+1}.jpg"
        cv2.imwrite(str(img_path), img)
        print(f"✅ 测试图片已创建: {img_path}")
    
    return test_images_dir


def create_data_dir():
    """创建data目录"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    print("✅ data目录已创建")
    return data_dir


def main():
    """主函数"""
    print("="*60)
    print("测试环境准备工具")
    print("="*60)
    
    print("\n1. 创建必要的目录...")
    models_dir = create_models_dir()
    data_dir = create_data_dir()
    
    print("\n2. 下载YOLO模型...")
    model_path = download_yolo_model(models_dir)
    
    print("\n3. 创建测试图片...")
    test_images_dir = create_test_images()
    
    print("\n" + "="*60)
    print("✅ 测试环境准备完成！")
    print("="*60)
    print("\n现在可以运行测试了：")
    print("  python tests/test_detector.py")
    print("  python tests/test_database.py")
    print("="*60)


if __name__ == "__main__":
    main()
