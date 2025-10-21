#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试脚本 - 使用现有数据集进行训练
"""

import os
import sys
from pathlib import Path

def main():
    print("🚀 盲道检测快速测试")
    print("=" * 40)
    
    # 检查现有数据集
    dataset_path = "datasets/yolo_format/blind_road_damage"
    if not os.path.exists(dataset_path):
        print(f"❌ 数据集不存在: {dataset_path}")
        print("正在创建测试数据集...")
        
        # 创建测试数据集
        os.makedirs(dataset_path, exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "labels"), exist_ok=True)
        
        # 创建简单的测试数据
        import cv2
        import numpy as np
        
        for i in range(10):
            # 创建测试图像
            img = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
            img_path = os.path.join(dataset_path, "images", f"test_{i:03d}.jpg")
            cv2.imwrite(img_path, img)
            
            # 创建测试标注
            label_path = os.path.join(dataset_path, "labels", f"test_{i:03d}.txt")
            with open(label_path, 'w') as f:
                f.write("0 0.5 0.5 0.2 0.2\n")  # 一个简单的标注
        
        # 创建数据集配置
        dataset_config = {
            'path': os.path.abspath(dataset_path),
            'train': 'images',
            'val': 'images',
            'test': 'images',
            'nc': 1,
            'names': ['blind_road']
        }
        
        import yaml
        with open(os.path.join(dataset_path, "dataset.yaml"), 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ 测试数据集创建完成: {dataset_path}")
    
    print(f"✅ 找到数据集: {dataset_path}")
    
    # 检查数据集结构
    images_dir = os.path.join(dataset_path, "images")
    labels_dir = os.path.join(dataset_path, "labels")
    yaml_file = os.path.join(dataset_path, "dataset.yaml")
    
    if not os.path.exists(images_dir):
        print(f"❌ 图像目录不存在: {images_dir}")
        return
    
    if not os.path.exists(labels_dir):
        print(f"❌ 标注目录不存在: {labels_dir}")
        return
    
    if not os.path.exists(yaml_file):
        print(f"❌ 配置文件不存在: {yaml_file}")
        return
    
    print("✅ 数据集结构完整")
    
    # 统计文件数量
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    
    print(f"📊 数据集统计:")
    print(f"   图像文件: {len(image_files)}")
    print(f"   标注文件: {len(label_files)}")
    
    if len(image_files) == 0:
        print("❌ 没有找到图像文件")
        return
    
    # 开始训练
    print("\n🎯 开始训练...")
    
    try:
        from ultralytics import YOLO
        
        # 加载模型
        model = YOLO('yolov8n.pt')
        print("✅ 模型加载成功")
        
        # 训练参数
        train_args = {
            'data': yaml_file,
            'epochs': 5,  # 减少训练轮数用于快速测试
            'batch': 4,   # 减少批次大小
            'imgsz': 416, # 减少图像尺寸
            'device': 'cpu',  # 使用CPU
            'project': 'results/quick_test',
            'name': 'blind_road_test',
            'save': True,
            'patience': 3,
            'verbose': True
        }
        
        print(f"训练参数: {train_args}")
        
        # 开始训练
        results = model.train(**train_args)
        
        print("✅ 训练完成！")
        print(f"结果保存在: results/quick_test/blind_road_test/")
        
        # 测试模型
        print("\n🧪 测试模型...")
        model_path = "results/quick_test/blind_road_test/weights/best.pt"
        
        if os.path.exists(model_path):
            test_model = YOLO(model_path)
            
            # 使用第一张图像进行测试
            test_image = os.path.join(images_dir, image_files[0])
            print(f"测试图像: {test_image}")
            
            results = test_model(test_image)
            print(f"✅ 模型测试成功，检测到 {len(results[0].boxes) if results[0].boxes is not None else 0} 个对象")
        else:
            print("❌ 模型文件不存在")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

