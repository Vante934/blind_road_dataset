#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强盲道检测器测试脚本
"""

import sys
import os
import numpy as np
import cv2
import time

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_blind_road_detector import EnhancedBlindRoadDetector

def create_test_image():
    """创建测试图像"""
    # 创建640x480的测试图像
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 添加背景
    image[:] = (50, 50, 50)  # 深灰色背景
    
    # 添加模拟盲道（水平）
    cv2.rectangle(image, (50, 200), (590, 250), (128, 128, 128), -1)
    cv2.rectangle(image, (50, 260), (590, 270), (100, 100, 100), -1)
    
    # 添加模拟盲道（垂直）
    cv2.rectangle(image, (300, 50), (350, 400), (128, 128, 128), -1)
    cv2.rectangle(image, (290, 50), (300, 400), (100, 100, 100), -1)
    
    # 添加模拟障碍物
    cv2.rectangle(image, (400, 150), (450, 200), (0, 0, 255), -1)  # 红色障碍物
    cv2.circle(image, (150, 350), 30, (0, 255, 0), -1)  # 绿色圆形障碍物
    
    # 添加一些噪声
    noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
    image = cv2.add(image, noise)
    
    return image

def test_detector_performance():
    """测试检测器性能"""
    print("=" * 60)
    print("🚀 增强盲道检测器性能测试")
    print("=" * 60)
    
    # 初始化检测器
    print("\n[初始化] 正在初始化检测器...")
    detector = EnhancedBlindRoadDetector()
    
    if not detector.model:
        print("❌ 检测器初始化失败")
        return False
    
    print("✅ 检测器初始化成功")
    
    # 创建测试图像
    print("\n[准备] 创建测试图像...")
    test_image = create_test_image()
    print(f"✅ 测试图像创建完成: {test_image.shape}")
    
    # 保存测试图像
    cv2.imwrite("test_detection_image.jpg", test_image)
    print("✅ 测试图像已保存: test_detection_image.jpg")
    
    # 执行多次检测测试性能
    print("\n[测试] 执行性能测试...")
    num_tests = 5
    processing_times = []
    
    for i in range(num_tests):
        start_time = time.time()
        result = detector.detect(test_image)
        end_time = time.time()
        
        processing_time = end_time - start_time
        processing_times.append(processing_time)
        
        print(f"  测试 {i+1}: {processing_time:.3f}s")
    
    # 计算统计信息
    avg_time = sum(processing_times) / len(processing_times)
    min_time = min(processing_times)
    max_time = max(processing_times)
    
    print(f"\n[性能] 平均处理时间: {avg_time:.3f}s")
    print(f"[性能] 最快处理时间: {min_time:.3f}s")
    print(f"[性能] 最慢处理时间: {max_time:.3f}s")
    
    # 分析最后一次检测结果
    print("\n[分析] 检测结果分析...")
    result = detector.detect(test_image)
    
    detections = result['detections']
    analysis = result['analysis']
    
    print(f"[结果] 检测到 {len(detections)} 个对象")
    
    # 显示盲道检测结果
    blind_path = analysis['blind_path']
    if blind_path.detected:
        print(f"[盲道] 检测到盲道:")
        print(f"  - 置信度: {blind_path.confidence:.3f}")
        print(f"  - 类型: {blind_path.path_type}")
        print(f"  - 方向: {blind_path.direction}")
        print(f"  - 条件: {blind_path.condition}")
        print(f"  - 宽度: {blind_path.width:.1f}px")
        print(f"  - 中心: {blind_path.center}")
    else:
        print("[盲道] 未检测到盲道")
    
    # 显示障碍物检测结果
    obstacle = analysis['obstacle']
    if obstacle.detected:
        print(f"[障碍物] 检测到障碍物:")
        print(f"  - 置信度: {obstacle.confidence:.3f}")
        print(f"  - 类型: {obstacle.obstacle_type}")
        print(f"  - 距离估算: {obstacle.distance_estimate:.1f}m")
        print(f"  - 严重程度: {obstacle.severity}")
        print(f"  - 中心: {obstacle.center}")
    else:
        print("[障碍物] 未检测到障碍物")
    
    # 显示统计信息
    stats = detector.get_stats()
    print(f"\n[统计] 检测统计:")
    print(f"  - 总检测数: {stats['total_detections']}")
    print(f"  - 盲道检测数: {stats['blind_path_detections']}")
    print(f"  - 障碍物检测数: {stats['obstacle_detections']}")
    
    return True

def test_real_image():
    """测试真实图像"""
    print("\n" + "=" * 60)
    print("📸 真实图像测试")
    print("=" * 60)
    
    # 检查是否有真实图像
    test_images_dir = "data/runs/detect/predict"
    if os.path.exists(test_images_dir):
        image_files = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if image_files:
            print(f"[发现] 找到 {len(image_files)} 个测试图像")
            
            # 初始化检测器
            detector = EnhancedBlindRoadDetector()
            if not detector.model:
                print("❌ 检测器初始化失败")
                return
            
            # 测试前几个图像
            for i, image_file in enumerate(image_files[:3]):
                image_path = os.path.join(test_images_dir, image_file)
                print(f"\n[测试] 处理图像: {image_file}")
                
                # 读取图像
                image = cv2.imread(image_path)
                if image is None:
                    print(f"❌ 无法读取图像: {image_file}")
                    continue
                
                # 执行检测
                start_time = time.time()
                result = detector.detect(image)
                end_time = time.time()
                
                processing_time = end_time - start_time
                print(f"[结果] 处理时间: {processing_time:.3f}s")
                
                # 分析结果
                analysis = result['analysis']
                blind_path = analysis['blind_path']
                obstacle = analysis['obstacle']
                
                print(f"[盲道] {'检测到' if blind_path.detected else '未检测到'}")
                print(f"[障碍物] {'检测到' if obstacle.detected else '未检测到'}")
                
        else:
            print("[信息] 未找到测试图像")
    else:
        print("[信息] 测试图像目录不存在")

def main():
    """主函数"""
    print("=" * 60)
    print("🧪 增强盲道检测器测试")
    print("=" * 60)
    
    # 测试检测器性能
    success = test_detector_performance()
    
    if success:
        # 测试真实图像
        test_real_image()
        
        print("\n" + "=" * 60)
        print("🎉 测试完成！")
        print("=" * 60)
    else:
        print("\n❌ 测试失败")

if __name__ == "__main__":
    main()
