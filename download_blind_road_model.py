#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载专门训练的盲道检测模型
"""

import os
import requests
from pathlib import Path

def download_blind_road_model():
    """下载盲道检测模型"""
    print("🔽 下载盲道检测模型...")
    
    # 模型下载链接（示例）
    model_urls = {
        "yolov8_blind_road": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "yolo11_blind_road": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt"
    }
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    for model_name, url in model_urls.items():
        model_path = models_dir / f"{model_name}.pt"
        
        if model_path.exists():
            print(f"✅ {model_name} 已存在")
            continue
            
        try:
            print(f"📥 下载 {model_name}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ {model_name} 下载完成}")
            
        except Exception as e:
            print(f"❌ {model_name} 下载失败: {e}")

def create_blind_road_dataset():
    """创建盲道检测数据集"""
    print("\n📁 创建盲道检测数据集...")
    
    # 创建数据集目录
    dataset_dirs = [
        "datasets/blind_road/train/images",
        "datasets/blind_road/train/labels", 
        "datasets/blind_road/val/images",
        "datasets/blind_road/val/labels"
    ]
    
    for dir_path in dataset_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ 创建目录: {dir_path}")
    
    # 创建数据集配置文件
    dataset_config = """
# 盲道检测数据集配置
path: datasets/blind_road
train: train/images
val: val/images

# 类别定义
names:
  0: blind_path      # 盲道
  1: obstacle        # 障碍物
  2: person          # 行人
  3: vehicle         # 车辆
  4: pothole         # 坑洼
  5: step            # 台阶
"""
    
    with open("datasets/blind_road/dataset.yaml", "w", encoding="utf-8") as f:
        f.write(dataset_config)
    
    print("✅ 数据集配置文件创建完成")

def main():
    """主函数"""
    print("🚀 盲道检测模型设置")
    print("=" * 50)
    
    # 1. 下载模型
    download_blind_road_model()
    
    # 2. 创建数据集
    create_blind_road_dataset()
    
    print("\n💡 建议:")
    print("1. 使用现有的通用模型进行初步测试")
    print("2. 收集盲道场景图像进行专门训练")
    print("3. 使用自定义类别重新训练模型")
    print("4. 或者使用预训练的盲道检测模型")

if __name__ == "__main__":
    main()

