#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据准备脚本
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.download_model import download_model

def create_test_images():
    """创建测试图像"""
    # 创建测试图像目录
    test_image_dir = Path("data", "images")
    test_image_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建测试图像
    test_image_path = test_image_dir / "image_001.jpg"
    
    if not test_image_path.exists():
        # 创建一个640x640的测试图像
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        # 添加一些简单的形状作为测试目标
        cv2.rectangle(test_image, (100, 100), (200, 200), (0, 255, 0), 2)
        cv2.circle(test_image, (320, 320), 50, (0, 0, 255), 2)
        cv2.line(test_image, (400, 100), (500, 200), (255, 0, 0), 2)
        
        # 保存测试图像
        cv2.imwrite(str(test_image_path), test_image)
        print(f"✅ 创建测试图像: {test_image_path}")
    else:
        print(f"✅ 测试图像 {test_image_path} 已存在，跳过创建")

def main():
    """主函数"""
    print("=" * 60)
    print("测试数据准备脚本")
    print("=" * 60)
    
    # 下载模型
    print("\n1. 下载YOLO模型")
    download_model()
    
    # 创建测试图像
    print("\n2. 创建测试图像")
    create_test_images()
    
    print("\n" + "=" * 60)
    print("✅ 测试数据准备完成")
    print("\n运行测试命令:")
    print("  python tests\test_detector.py")
    print("  python tests\test_database.py")

if __name__ == "__main__":
    main()
