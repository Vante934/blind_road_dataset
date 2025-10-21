#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型检测能力检查脚本
检查模型是否能检测到目标，并分析检测结果
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import glob

def check_model_detection():
    """检查模型检测能力"""
    print("🔍 检查模型检测能力...")
    
    # 加载模型
    try:
        model = YOLO('models/yolo11n.pt')
        print("✅ YOLO11n 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 获取测试图像
    test_images = glob.glob("datasets/test/standard/*.jpg")[:5]  # 测试前5张图像
    
    if not test_images:
        print("❌ 没有找到测试图像")
        return
    
    print(f"📁 找到 {len(test_images)} 张测试图像")
    
    total_detections = 0
    images_with_detections = 0
    
    for i, img_path in enumerate(test_images):
        print(f"\n🖼️ 测试图像 {i+1}: {os.path.basename(img_path)}")
        
        try:
            # 加载图像
            image = cv2.imread(img_path)
            if image is None:
                print(f"⚠️ 无法加载图像: {img_path}")
                continue
            
            # 执行检测
            results = model(image)
            
            # 分析结果
            detections = 0
            for result in results:
                if result.boxes is not None:
                    detections = len(result.boxes)
                    total_detections += detections
                    
                    if detections > 0:
                        images_with_detections += 1
                        print(f"✅ 检测到 {detections} 个目标")
                        
                        # 显示检测详情
                        for j, box in enumerate(result.boxes):
                            conf = box.conf[0].item()
                            cls = int(box.cls[0].item())
                            print(f"   目标 {j+1}: 类别={cls}, 置信度={conf:.3f}")
                    else:
                        print("❌ 未检测到目标")
            
        except Exception as e:
            print(f"❌ 处理图像失败: {e}")
    
    # 统计结果
    print(f"\n📊 检测统计:")
    print(f"   总检测数: {total_detections}")
    print(f"   有检测的图像: {images_with_detections}/{len(test_images)}")
    print(f"   检测率: {images_with_detections/len(test_images)*100:.1f}%")
    
    if total_detections == 0:
        print("\n⚠️ 未检测到任何目标，可能原因:")
        print("1. 模型未针对盲道场景训练")
        print("2. 置信度阈值过高")
        print("3. 图像中确实没有可检测的目标")
        print("4. 模型类别与图像内容不匹配")
        
        print("\n💡 建议:")
        print("1. 使用专门训练的盲道检测模型")
        print("2. 降低置信度阈值")
        print("3. 检查图像内容是否包含可检测目标")
        print("4. 使用包含明显目标的测试图像")

def check_model_classes():
    """检查模型类别"""
    print("\n🏷️ 检查模型类别...")
    
    try:
        model = YOLO('models/yolo11n.pt')
        
        # 获取模型信息
        print(f"模型名称: {model.model_name if hasattr(model, 'model_name') else 'YOLO11n'}")
        
        # 尝试获取类别信息
        if hasattr(model, 'names'):
            print(f"模型类别数: {len(model.names)}")
            print("支持的类别:")
            for i, name in model.names.items():
                print(f"  {i}: {name}")
        else:
            print("⚠️ 无法获取模型类别信息")
            
    except Exception as e:
        print(f"❌ 检查模型类别失败: {e}")

def create_test_image_with_objects():
    """创建包含明显目标的测试图像"""
    print("\n🎨 创建测试图像...")
    
    try:
        # 创建包含明显目标的图像
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255  # 白色背景
        
        # 绘制一些明显的目标
        cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), -1)  # 绿色矩形
        cv2.circle(img, (400, 300), 50, (255, 0, 0), -1)  # 蓝色圆形
        cv2.rectangle(img, (300, 150), (400, 250), (0, 0, 255), -1)  # 红色矩形
        
        # 保存测试图像
        test_path = "datasets/test/standard/test_objects.jpg"
        cv2.imwrite(test_path, img)
        print(f"✅ 创建测试图像: {test_path}")
        
        return test_path
        
    except Exception as e:
        print(f"❌ 创建测试图像失败: {e}")
        return None

def main():
    """主函数"""
    print("🔍 模型检测能力分析")
    print("=" * 50)
    
    # 1. 检查模型类别
    check_model_classes()
    
    # 2. 检查检测能力
    check_model_detection()
    
    # 3. 创建测试图像
    test_img = create_test_image_with_objects()
    
    if test_img:
        print(f"\n🧪 使用新创建的测试图像重新检测...")
        try:
            model = YOLO('models/yolo11n.pt')
            image = cv2.imread(test_img)
            results = model(image)
            
            total_det = 0
            for result in results:
                if result.boxes is not None:
                    total_det += len(result.boxes)
            
            print(f"新测试图像检测结果: {total_det} 个目标")
            
        except Exception as e:
            print(f"❌ 重新检测失败: {e}")

if __name__ == "__main__":
    main()

