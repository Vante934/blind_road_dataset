#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建适合通用模型测试的数据集
使用包含person, car, bicycle等通用目标的图像
"""

import os
import cv2
import numpy as np
import glob
import shutil
from pathlib import Path

def create_coco_style_test_data():
    """创建COCO风格的测试数据"""
    print("🎯 创建适合通用模型测试的数据集...")
    
    # 创建测试目录
    test_dir = Path("datasets/test/coco_style")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找包含通用目标的图像
    source_dirs = [
        "data/images",
        "data/Blind_DataSet",
        "data/Environment_DataSet"
    ]
    
    target_classes = ['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck']
    selected_images = []
    
    print("🔍 搜索包含通用目标的图像...")
    
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            continue
            
        # 获取所有图像文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            pattern = os.path.join(source_dir, f"**/{ext}")
            image_files.extend(glob.glob(pattern, recursive=True))
        
        print(f"📁 从 {source_dir} 找到 {len(image_files)} 张图像")
        
        # 选择一些图像（假设它们可能包含通用目标）
        selected = image_files[:20]  # 选择前20张
        selected_images.extend(selected)
    
    # 复制选中的图像
    copied_count = 0
    for i, img_path in enumerate(selected_images):
        try:
            filename = f"test_{i+1:03d}.jpg"
            dest_path = test_dir / filename
            shutil.copy2(img_path, dest_path)
            copied_count += 1
        except Exception as e:
            print(f"⚠️ 复制失败 {img_path}: {e}")
    
    print(f"✅ 复制了 {copied_count} 张图像到 {test_dir}")
    return test_dir

def create_synthetic_test_data():
    """创建合成测试数据"""
    print("\n🎨 创建合成测试数据...")
    
    test_dir = Path("datasets/test/synthetic")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建包含明显目标的合成图像
    for i in range(10):
        # 创建白色背景
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # 添加不同的目标
        if i % 3 == 0:
            # 人物轮廓
            cv2.rectangle(img, (200, 150), (250, 400), (0, 0, 0), -1)  # 身体
            cv2.circle(img, (225, 120), 30, (0, 0, 0), -1)  # 头部
        elif i % 3 == 1:
            # 汽车轮廓
            cv2.rectangle(img, (150, 200), (450, 300), (0, 0, 0), -1)  # 车身
            cv2.circle(img, (200, 320), 25, (0, 0, 0), -1)  # 轮子
            cv2.circle(img, (400, 320), 25, (0, 0, 0), -1)  # 轮子
        else:
            # 自行车轮廓
            cv2.circle(img, (200, 300), 30, (0, 0, 0), -1)  # 前轮
            cv2.circle(img, (400, 300), 30, (0, 0, 0), -1)  # 后轮
            cv2.line(img, (200, 300), (400, 300), (0, 0, 0), 3)  # 车架
        
        # 保存图像
        filename = f"synthetic_{i+1:02d}.jpg"
        filepath = test_dir / filename
        cv2.imwrite(str(filepath), img)
        print(f"✅ 创建合成图像: {filename}")
    
    return test_dir

def test_with_proper_data():
    """使用合适的数据测试模型"""
    print("\n🧪 使用合适数据测试模型...")
    
    try:
        from ultralytics import YOLO
        
        # 加载模型
        model = YOLO('models/yolo11n.pt')
        print("✅ 模型加载成功")
        
        # 测试合成数据
        synthetic_dir = Path("datasets/test/synthetic")
        if synthetic_dir.exists():
            print(f"\n📁 测试合成数据: {synthetic_dir}")
            test_images = list(synthetic_dir.glob("*.jpg"))
            
            total_detections = 0
            for img_path in test_images[:5]:  # 测试前5张
                print(f"\n🖼️ 测试: {img_path.name}")
                
                # 执行检测
                results = model(str(img_path))
                
                detections = 0
                for result in results:
                    if result.boxes is not None:
                        detections = len(result.boxes)
                        total_detections += detections
                        
                        if detections > 0:
                            print(f"✅ 检测到 {detections} 个目标")
                            for j, box in enumerate(result.boxes):
                                conf = box.conf[0].item()
                                cls = int(box.cls[0].item())
                                cls_name = model.names[cls]
                                print(f"   目标 {j+1}: {cls_name} (置信度: {conf:.3f})")
                        else:
                            print("❌ 未检测到目标")
            
            print(f"\n📊 合成数据测试结果: {total_detections} 个目标")
        
        # 测试真实数据
        coco_dir = Path("datasets/test/coco_style")
        if coco_dir.exists():
            print(f"\n📁 测试真实数据: {coco_dir}")
            test_images = list(coco_dir.glob("*.jpg"))
            
            total_detections = 0
            for img_path in test_images[:3]:  # 测试前3张
                print(f"\n🖼️ 测试: {img_path.name}")
                
                # 执行检测
                results = model(str(img_path))
                
                detections = 0
                for result in results:
                    if result.boxes is not None:
                        detections = len(result.boxes)
                        total_detections += detections
                        
                        if detections > 0:
                            print(f"✅ 检测到 {detections} 个目标")
                            for j, box in enumerate(result.boxes):
                                conf = box.conf[0].item()
                                cls = int(box.cls[0].item())
                                cls_name = model.names[cls]
                                print(f"   目标 {j+1}: {cls_name} (置信度: {conf:.3f})")
                        else:
                            print("❌ 未检测到目标")
            
            print(f"\n📊 真实数据测试结果: {total_detections} 个目标")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def main():
    """主函数"""
    print("🔍 创建适合的测试数据")
    print("=" * 50)
    
    # 1. 创建合成测试数据
    synthetic_dir = create_synthetic_test_data()
    
    # 2. 创建真实测试数据
    coco_dir = create_coco_style_test_data()
    
    # 3. 测试模型
    test_with_proper_data()
    
    print("\n💡 建议:")
    print("1. 使用合成数据验证模型基本功能")
    print("2. 使用包含通用目标的真实图像")
    print("3. 不要期望通用模型检测盲道相关目标")
    print("4. 如需盲道检测，需要专门训练的模型")

if __name__ == "__main__":
    main()

