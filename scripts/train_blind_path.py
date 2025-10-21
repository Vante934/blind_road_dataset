#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
盲道检测YOLOv8训练脚本
"""

import os
import sys
from pathlib import Path

def check_dependencies():
    """检查依赖"""
    try:
        import ultralytics
        print("✅ ultralytics 已安装")
    except ImportError:
        print("❌ 请先安装 ultralytics: pip install ultralytics")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch 已安装: {torch.__version__}")
        
        # 检查CUDA可用性
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用: {torch.cuda.device_count()} 个GPU")
            return "cuda"
        else:
            print("⚠️  CUDA 不可用，将使用CPU训练")
            return "cpu"
            
    except ImportError:
        print("❌ 请先安装 PyTorch: pip install torch")
        return False

def download_yolo_model():
    """下载YOLOv8模型"""
    try:
        from ultralytics import YOLO
        
        # 检查模型是否已存在
        if os.path.exists('yolov8n.pt'):
            print("✅ YOLOv8n 模型已存在")
            return True
        
        print("📥 正在下载 YOLOv8n 模型...")
        model = YOLO('yolov8n.pt')  # 这会自动下载模型
        print("✅ YOLOv8n 模型下载完成")
        return True
        
    except Exception as e:
        print(f"❌ 下载模型失败: {e}")
        return False

def train_blind_path_detector(device_type="cpu"):
    """训练盲道检测模型"""
    
    print("🚀 开始训练盲道检测模型...")
    
    # 检查数据集
    dataset_path = "yolo_dataset/dataset.yaml"
    if not os.path.exists(dataset_path):
        print(f"❌ 数据集配置文件不存在: {dataset_path}")
        return False
    
    # 检查图像和标签
    images_dir = "yolo_dataset/images"
    labels_dir = "yolo_dataset/labels"
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print("❌ 数据集目录不完整")
        return False
    
    image_count = len([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])
    label_count = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
    
    print(f"📊 数据集信息:")
    print(f"  - 图像数量: {image_count}")
    print(f"  - 标签数量: {label_count}")
    
    if image_count == 0 or label_count == 0:
        print("❌ 数据集为空")
        return False
    
    try:
        from ultralytics import YOLO
        
    # 加载预训练模型
        print("📥 加载 YOLOv8n 模型...")
        model = YOLO('yolov8n.pt')
        
        # 根据设备类型调整训练参数
        if device_type == "cpu":
            print("🖥️  使用CPU训练（速度较慢，请耐心等待）")
            batch_size = 4  # CPU训练使用更小的批次
            epochs = 30     # 减少训练轮数
        else:
            print("🚀 使用GPU训练")
            batch_size = 8
            epochs = 50
    
    # 训练参数
        print("🎯 开始训练...")
    results = model.train(
            data=dataset_path,           # 数据集配置文件
            epochs=epochs,               # 训练轮数
            imgsz=640,                   # 图像尺寸
            batch=batch_size,            # 批次大小
            name='blind_path_detector',  # 实验名称
            patience=10,                 # 早停耐心值
            save=True,                   # 保存模型
            device=device_type,          # 使用指定设备
            verbose=True,                # 详细输出
            plots=True,                  # 生成训练图表
            save_period=10               # 每10轮保存一次
        )
        
        print("✅ 训练完成！")
        print(f"📁 模型保存在: runs/detect/blind_path_detector/")
        
        # 显示训练结果
        if hasattr(results, 'results_dict'):
            print(f"📊 训练结果:")
            for key, value in results.results_dict.items():
                if isinstance(value, float):
                    print(f"  - {key}: {value:.4f}")
                else:
                    print(f"  - {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("盲道检测模型训练工具")
    print("=" * 50)
    
    # 1. 检查依赖和设备
    device_type = check_dependencies()
    if not device_type:
        return
    
    # 2. 下载模型
    if not download_yolo_model():
        return
    
    # 3. 开始训练
    success = train_blind_path_detector(device_type)
    
    if success:
        print("\n🎉 训练成功完成！")
        print("📁 模型文件位置: runs/detect/blind_path_detector/weights/best.pt")
        print("🚀 可以使用训练好的模型进行盲道检测了！")
    else:
        print("\n❌ 训练失败，请检查错误信息")

if __name__ == "__main__":
    main()
