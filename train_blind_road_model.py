#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
盲道检测模型训练脚本
"""

from ultralytics import YOLO
import os

def train_blind_road_model():
    """训练盲道检测模型"""
    print("🚀 开始训练盲道检测模型...")
    
    # 检查数据集
    dataset_path = "datasets/blind_road/dataset.yaml"
    if not os.path.exists(dataset_path):
        print(f"❌ 数据集配置文件不存在: {dataset_path}")
        return
    
    # 加载预训练模型
    model = YOLO('models/yolo11n.pt')  # 使用YOLO11n作为基础
    
    # 训练参数
    results = model.train(
        data=dataset_path,
        epochs=100,
        imgsz=640,
        batch=16,
        device='cuda' if os.system('nvidia-smi') == 0 else 'cpu',
        project='results/blind_road_training',
        name='blind_road_detection',
        save=True,
        save_period=10,
        patience=20,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        augment=True,
        mixup=0.0,
        copy_paste=0.0,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        val=True,
        plots=True
    )
    
    print("✅ 训练完成！")
    print(f"📁 结果保存在: results/blind_road_training/blind_road_detection/")
    
    return results

if __name__ == "__main__":
    train_blind_road_model()

