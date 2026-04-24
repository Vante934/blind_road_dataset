#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_detector.py - 修复版
增加了更好的错误处理和Mock数据
"""

import sys
sys.path.append(".")

from modules.detector import get_detector, BlindRoadDetector
from pathlib import Path
import cv2
import numpy as np
import pytest


class TestDetector:
    """检测器测试类"""
    
    @classmethod
    def setup_class(cls):
        """测试前的准备工作"""
        # 检查模型文件
        model_path = Path("models/best.pt")
        if not model_path.exists():
            pytest.skip(f"⚠️ 模型文件不存在: {model_path}\n"
                       f"请运行以下命令之一：\n"
                       f"1. python scripts/setup_test_environment.py  # 下载预训练模型\n"
                       f"2. 将你训练的模型复制到 models/best.pt")
    
    def test_detector_initialization(self):
        """测试1: 检测器初始化"""
        print("\n" + "="*60)
        print("测试1: 检测器初始化")
        print("="*60)
        
        try:
            detector = get_detector("models/best.pt", device="cpu")
            assert detector is not None
            assert detector.model is not None
            print("✅ 检测器初始化成功")
        except Exception as e:
            pytest.fail(f"❌ 初始化失败: {e}")
    
    def test_detect_from_bytes(self):
        """测试2: 从字节流检测"""
        print("\n" + "="*60)
        print("测试2: 从字节流检测")
        print("="*60)
        
        # 创建一个简单的测试图像（纯色图）
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[:] = (100, 150, 200)  # 填充颜色
        
        # 添加一些形状（模拟物体）
        cv2.rectangle(test_image, (100, 100), (200, 200), (0, 255, 0), -1)
        cv2.circle(test_image, (400, 300), 50, (255, 0, 0), -1)
        
        # 编码为JPEG
        success, encoded = cv2.imencode('.jpg', test_image)
        assert success, "图像编码失败"
        
        image_bytes = encoded.tobytes()
        
        # 调用检测器
        detector = get_detector()
        result = detector.detect_from_bytes(image_bytes)
        
        # 验证返回格式
        assert result["success"] == True, "检测未成功"
        assert "obstacles" in result, "缺少obstacles字段"
        assert "processing_time_ms" in result, "缺少processing_time_ms字段"
        assert "blind_road_detected" in result, "缺少blind_road_detected字段"
        
        print(f"✅ 检测成功")
        print(f"   处理时间: {result['processing_time_ms']:.1f}ms")
        print(f"   障碍物数量: {len(result['obstacles'])}")
        print(f"   盲道检测: {result['blind_road_detected']}")
    
    def test_with_real_image(self):
        """测试3: 真实图像检测（如果存在）"""
        print("\n" + "="*60)
        print("测试3: 真实图像检测")
        print("="*60)
        
        # 尝试多个可能的测试图片路径
        test_paths = [
            "test_images/test1.jpg",
            "test_images/test.jpg",
            "data/test.jpg",
            "tests/test_image.jpg"
        ]
        
        test_image_path = None
        for path in test_paths:
            if Path(path).exists():
                test_image_path = path
                break
        
        if test_image_path is None:
            print("⚠️ 未找到测试图片，跳过此测试")
            print("   提示：可以放一张测试图片到以下任意路径：")
            for path in test_paths:
                print(f"   - {path}")
            pytest.skip("未找到测试图片")
            return
        
        print(f"📸 使用测试图片: {test_image_path}")
        
        with open(test_image_path, "rb") as f:
            image_bytes = f.read()
        
        detector = get_detector()
        result = detector.detect_from_bytes(image_bytes, conf_threshold=0.3)
        
        print(f"✅ 检测完成")
        print(f"   障碍物数量: {len(result['obstacles'])}")
        print(f"   盲道检测: {result['blind_road_detected']}")
        print(f"   盲道状态: {result['blind_road_status']}")
        print(f"   语音提示: {result['voice_alert']}")
        
        # 打印检测到的障碍物详情
        if result['obstacles']:
            print("\n   检测详情:")
            for i, obs in enumerate(result['obstacles'], 1):
                print(f"   {i}. {obs['class_name']} - "
                      f"置信度:{obs['confidence']:.2f} - "
                      f"方向:{obs['direction']} - "
                      f"距离:{obs['distance_estimate']}m")
    
    def test_performance(self):
        """测试4: 性能测试"""
        print("\n" + "="*60)
        print("测试4: 性能测试（10次检测）")
        print("="*60)
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        success, encoded = cv2.imencode('.jpg', test_image)
        image_bytes = encoded.tobytes()
        
        detector = get_detector()
        times = []
        
        # 预热（第一次推理会慢一些）
        detector.detect_from_bytes(image_bytes)
        
        # 正式测试
        for i in range(10):
            result = detector.detect_from_bytes(image_bytes)
            times.append(result['processing_time_ms'])
            print(f"   第{i+1}次: {result['processing_time_ms']:.1f}ms")
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n📊 性能统计:")
        print(f"   平均时间: {avg_time:.1f}ms")
        print(f"   最快: {min_time:.1f}ms")
        print(f"   最慢: {max_time:.1f}ms")
        
        # 性能要求：CPU模式下应该<500ms
        if avg_time < 200:
            print(f"   ✅ 性能优秀 (<200ms)")
        elif avg_time < 500:
            print(f"   ✅ 性能合格 (<500ms)")
        else:
            print(f"   ⚠️ 性能较慢 (>{avg_time:.0f}ms)")
            print(f"   建议：使用GPU或更小的模型（yolov8n）")


if __name__ == "__main__":
    # 运行测试
    print("="*60)
    print("YOLO检测器单元测试")
    print("="*60)
    
    pytest.main([__file__, "-v", "-s"])

