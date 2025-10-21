#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版盲道检测训练脚本
专门针对Python 3.13兼容性优化
"""

import os
import sys
import json
import time
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """检查依赖包"""
    required_packages = [
        'torch', 'torchvision', 'ultralytics', 'cv2', 'numpy', 
        'pandas', 'matplotlib', 'yaml', 'tqdm', 'PIL'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'yaml':
                import yaml
            elif package == 'PIL':
                from PIL import Image
            else:
                __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} 未安装")
    
    if missing_packages:
        print(f"\n缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install " + " ".join(missing_packages))
        return False
    
    return True

def create_simple_dataset():
    """创建简单的测试数据集"""
    print("📊 创建简单测试数据集...")
    
    # 使用现有的数据集
    existing_dataset = Path("datasets/yolo_format/blind_road_damage")
    if existing_dataset.exists():
        print(f"✅ 使用现有数据集: {existing_dataset}")
        return str(existing_dataset)
    
    # 如果现有数据集不存在，创建简单测试数据集
    dataset_dir = Path("datasets/simple_test")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建YOLO格式目录
    (dataset_dir / "images").mkdir(exist_ok=True)
    (dataset_dir / "labels").mkdir(exist_ok=True)
    
    # 创建一些测试图像和标注
    import cv2
    import numpy as np
    
    print("🎨 生成测试图像...")
    for i in range(10):  # 创建10张测试图像
        # 创建随机图像
        img = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
        
        # 保存图像
        img_path = dataset_dir / "images" / f"test_{i:03d}.jpg"
        cv2.imwrite(str(img_path), img)
        
        # 创建对应的标注文件
        label_path = dataset_dir / "labels" / f"test_{i:03d}.txt"
        with open(label_path, 'w') as f:
            # 添加一个随机标注
            if np.random.random() > 0.5:  # 50%概率有标注
                class_id = np.random.randint(0, 3)
                x_center = np.random.uniform(0.2, 0.8)
                y_center = np.random.uniform(0.2, 0.8)
                width = np.random.uniform(0.1, 0.3)
                height = np.random.uniform(0.1, 0.3)
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    # 创建数据集配置
    dataset_config = {
        'path': str(dataset_dir.absolute()),
        'train': 'images',
        'val': 'images',
        'test': 'images',
        'nc': 3,
        'names': ['正常盲道', '损坏盲道', '障碍物']
    }
    
    with open(dataset_dir / "dataset.yaml", 'w', encoding='utf-8') as f:
        import yaml
        yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ 数据集配置已创建: {dataset_dir / 'dataset.yaml'}")
    return str(dataset_dir)

def train_simple_model(dataset_path):
    """训练简单模型"""
    print("🚀 开始训练简单模型...")
    
    try:
        from ultralytics import YOLO
        
        # 加载预训练模型
        model = YOLO('yolov8n.pt')
        print("✅ 预训练模型加载成功")
        
        # 训练参数
        train_args = {
            'data': os.path.join(dataset_path, 'dataset.yaml'),
            'epochs': 10,  # 减少训练轮数用于测试
            'batch': 8,    # 减少批次大小
            'imgsz': 416,  # 减少图像尺寸
            'device': 'cpu',  # 使用CPU避免GPU问题
            'project': 'results/simple_training',
            'name': 'blind_road_detection',
            'save': True,
            'patience': 5,
            'verbose': True
        }
        
        print("开始训练...")
        print(f"训练参数: {train_args}")
        
        # 开始训练
        results = model.train(**train_args)
        
        print("✅ 训练完成！")
        print(f"结果保存在: results/simple_training/")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        return False

def test_model():
    """测试模型"""
    print("🧪 测试模型...")
    
    try:
        from ultralytics import YOLO
        import cv2
        import numpy as np
        
        # 加载训练好的模型
        model_path = "results/simple_training/blind_road_detection/weights/best.pt"
        
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return False
        
        model = YOLO(model_path)
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
        
        # 进行预测
        results = model(test_image)
        
        print("✅ 模型测试成功")
        print(f"检测结果: {len(results[0].boxes) if results[0].boxes is not None else 0} 个对象")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🎯 简化版盲道检测训练系统")
    print("=" * 50)
    print("专门针对Python 3.13兼容性优化")
    print()
    
    # 1. 检查依赖
    print("步骤1: 检查依赖包...")
    if not check_dependencies():
        print("\n❌ 依赖检查失败，请先安装必要的包")
        print("运行命令: pip install torch torchvision ultralytics opencv-python numpy pandas matplotlib pyyaml tqdm")
        return
    
    print("✅ 依赖检查通过")
    
    # 2. 创建数据集
    print("\n步骤2: 准备数据集...")
    dataset_path = create_simple_dataset()
    
    # 3. 训练模型
    print("\n步骤3: 训练模型...")
    if not train_simple_model(dataset_path):
        print("❌ 训练失败")
        return
    
    # 4. 测试模型
    print("\n步骤4: 测试模型...")
    if not test_model():
        print("❌ 测试失败")
        return
    
    print("\n🎉 简化版训练完成！")
    print("\n结果文件:")
    print("- 训练结果: results/simple_training/")
    print("- 最佳模型: results/simple_training/blind_road_detection/weights/best.pt")
    print("- 训练日志: results/simple_training/blind_road_detection/")
    
    print("\n💡 下一步建议:")
    print("1. 准备真实数据集替换测试数据")
    print("2. 调整训练参数获得更好效果")
    print("3. 使用GPU加速训练")
    print("4. 进行模型优化和部署")

if __name__ == "__main__":
    main()
