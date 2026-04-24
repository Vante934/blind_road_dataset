#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强盲道检测集成模块 - 第8天核心功能开发
集成真实YOLO模型，优化检测性能和准确性
"""

import os
import sys
import cv2
import numpy as np
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import torch
from PIL import Image
import base64
import io
import yaml

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """检测结果数据类"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    area: float
    center: Tuple[int, int]

@dataclass
class BlindPathInfo:
    """盲道信息"""
    detected: bool
    confidence: float
    bbox: Tuple[int, int, int, int]
    path_type: str
    direction: str
    condition: str
    width: float
    center: Tuple[int, int]

@dataclass
class ObstacleInfo:
    """障碍物信息"""
    detected: bool
    confidence: float
    bbox: Tuple[int, int, int, int]
    obstacle_type: str
    distance_estimate: float
    severity: str
    center: Tuple[int, int]

class EnhancedBlindRoadDetector:
    """增强盲道检测器"""
    
    def __init__(self, model_path: str = "models/yolov8n.pt", config_path: str = "data/yolo_dataset/dataset.yaml"):
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.class_names = []
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        self.input_size = (640, 640)
        
        # 检测统计
        self.detection_stats = {
            "total_detections": 0,
            "blind_path_detections": 0,
            "obstacle_detections": 0,
            "false_positives": 0
        }
        
        # 初始化模型
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化检测模型"""
        try:
            # 检查模型文件
            if not os.path.exists(self.model_path):
                logger.error(f"模型文件不存在: {self.model_path}")
                return False
            
            # 加载YOLO模型
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                logger.info(f"✅ YOLO模型加载成功: {self.model_path}")
            except ImportError:
                logger.error("ultralytics模块未安装，请运行: pip install ultralytics")
                return False
            
            # 加载类别名称
            self._load_class_names()
            
            # 预热模型
            self._warmup_model()
            
            logger.info("✅ 增强盲道检测器初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {e}")
            return False
    
    def _load_class_names(self):
        """加载类别名称"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    self.class_names = config.get('names', [])
            else:
                # 默认类别名称
                self.class_names = [
                    'blind_path',      # 盲道
                    'obstacle',        # 障碍物
                    'person',          # 行人
                    'vehicle',         # 车辆
                    'construction'     # 施工区域
                ]
            
            logger.info(f"✅ 加载类别名称: {len(self.class_names)} 个类别")
            
        except Exception as e:
            logger.error(f"❌ 加载类别名称失败: {e}")
            self.class_names = ['blind_path', 'obstacle', 'person', 'vehicle', 'construction']
    
    def _warmup_model(self):
        """预热模型"""
        try:
            # 创建虚拟输入进行预热
            dummy_input = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
            self.detect(dummy_input)
            logger.info("✅ 模型预热完成")
        except Exception as e:
            logger.warning(f"⚠️ 模型预热失败: {e}")
    
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """执行检测"""
        try:
            start_time = time.time()
            
            # 预处理图像
            processed_image = self._preprocess_image(image)
            
            # 执行检测
            results = self.model(processed_image, conf=self.confidence_threshold, iou=self.nms_threshold)
            
            # 后处理结果
            detection_results = self._postprocess_results(results[0], image.shape)
            
            # 分析检测结果
            analysis = self._analyze_detections(detection_results)
            
            # 更新统计
            self._update_stats(detection_results)
            
            processing_time = time.time() - start_time
            
            return {
                "detections": detection_results,
                "analysis": analysis,
                "processing_time": processing_time,
                "image_shape": image.shape,
                "model_info": {
                    "model_path": self.model_path,
                    "confidence_threshold": self.confidence_threshold,
                    "nms_threshold": self.nms_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 检测失败: {e}")
            return self._get_empty_result()
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """预处理图像"""
        try:
            # 调整图像大小
            resized = cv2.resize(image, self.input_size)
            
            # 归一化
            normalized = resized.astype(np.float32) / 255.0
            
            return normalized
            
        except Exception as e:
            logger.error(f"❌ 图像预处理失败: {e}")
            return image
    
    def _postprocess_results(self, result, original_shape: Tuple[int, int, int]) -> List[DetectionResult]:
        """后处理检测结果"""
        detections = []
        
        try:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                # 缩放坐标到原始图像尺寸
                h, w = original_shape[:2]
                scale_x = w / self.input_size[0]
                scale_y = h / self.input_size[1]
                
                for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                    # 缩放边界框
                    x1, y1, x2, y2 = box
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    
                    # 计算面积和中心点
                    area = (x2 - x1) * (y2 - y1)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    # 获取类别名称
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    
                    detection = DetectionResult(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=float(conf),
                        bbox=(x1, y1, x2, y2),
                        area=area,
                        center=center
                    )
                    
                    detections.append(detection)
            
        except Exception as e:
            logger.error(f"❌ 结果后处理失败: {e}")
        
        return detections
    
    def _analyze_detections(self, detections: List[DetectionResult]) -> Dict[str, Any]:
        """分析检测结果"""
        analysis = {
            "blind_path": BlindPathInfo(
                detected=False,
                confidence=0.0,
                bbox=(0, 0, 0, 0),
                path_type="unknown",
                direction="unknown",
                condition="unknown",
                width=0.0,
                center=(0, 0)
            ),
            "obstacle": ObstacleInfo(
                detected=False,
                confidence=0.0,
                bbox=(0, 0, 0, 0),
                obstacle_type="none",
                distance_estimate=0.0,
                severity="low",
                center=(0, 0)
            ),
            "summary": {
                "total_objects": len(detections),
                "blind_paths": 0,
                "obstacles": 0,
                "other_objects": 0
            }
        }
        
        try:
            for detection in detections:
                if detection.class_name == "blind_path":
                    analysis["blind_path"] = self._analyze_blind_path(detection)
                    analysis["summary"]["blind_paths"] += 1
                elif detection.class_name in ["obstacle", "person", "vehicle", "construction"]:
                    if not analysis["obstacle"].detected or detection.confidence > analysis["obstacle"].confidence:
                        analysis["obstacle"] = self._analyze_obstacle(detection)
                    analysis["summary"]["obstacles"] += 1
                else:
                    analysis["summary"]["other_objects"] += 1
            
        except Exception as e:
            logger.error(f"❌ 检测分析失败: {e}")
        
        return analysis
    
    def _analyze_blind_path(self, detection: DetectionResult) -> BlindPathInfo:
        """分析盲道检测结果"""
        x1, y1, x2, y2 = detection.bbox
        width = x2 - x1
        height = y2 - y1
        
        # 判断方向
        direction = "horizontal" if width > height else "vertical"
        
        # 判断条件（基于置信度）
        if detection.confidence > 0.8:
            condition = "excellent"
        elif detection.confidence > 0.6:
            condition = "good"
        elif detection.confidence > 0.4:
            condition = "fair"
        else:
            condition = "poor"
        
        # 判断类型（基于面积和形状）
        aspect_ratio = width / height if height > 0 else 1
        if aspect_ratio > 2:
            path_type = "tactile_paving"
        elif aspect_ratio < 0.5:
            path_type = "guiding_strip"
        else:
            path_type = "warning_surface"
        
        return BlindPathInfo(
            detected=True,
            confidence=detection.confidence,
            bbox=detection.bbox,
            path_type=path_type,
            direction=direction,
            condition=condition,
            width=width,
            center=detection.center
        )
    
    def _analyze_obstacle(self, detection: DetectionResult) -> ObstacleInfo:
        """分析障碍物检测结果"""
        x1, y1, x2, y2 = detection.bbox
        area = detection.area
        
        # 估算距离（基于面积）
        if area > 50000:  # 大物体，可能很近
            distance_estimate = 2.0
        elif area > 20000:  # 中等物体
            distance_estimate = 5.0
        elif area > 5000:   # 小物体
            distance_estimate = 10.0
        else:  # 很小的物体
            distance_estimate = 15.0
        
        # 判断严重程度
        if detection.confidence > 0.8 and area > 20000:
            severity = "high"
        elif detection.confidence > 0.6 and area > 10000:
            severity = "medium"
        else:
            severity = "low"
        
        return ObstacleInfo(
            detected=True,
            confidence=detection.confidence,
            bbox=detection.bbox,
            obstacle_type=detection.class_name,
            distance_estimate=distance_estimate,
            severity=severity,
            center=detection.center
        )
    
    def _update_stats(self, detections: List[DetectionResult]):
        """更新检测统计"""
        self.detection_stats["total_detections"] += len(detections)
        
        for detection in detections:
            if detection.class_name == "blind_path":
                self.detection_stats["blind_path_detections"] += 1
            elif detection.class_name in ["obstacle", "person", "vehicle", "construction"]:
                self.detection_stats["obstacle_detections"] += 1
    
    def _get_empty_result(self) -> Dict[str, Any]:
        """获取空结果"""
        return {
            "detections": [],
            "analysis": {
                "blind_path": BlindPathInfo(
                    detected=False,
                    confidence=0.0,
                    bbox=(0, 0, 0, 0),
                    path_type="unknown",
                    direction="unknown",
                    condition="unknown",
                    width=0.0,
                    center=(0, 0)
                ),
                "obstacle": ObstacleInfo(
                    detected=False,
                    confidence=0.0,
                    bbox=(0, 0, 0, 0),
                    obstacle_type="none",
                    distance_estimate=0.0,
                    severity="low",
                    center=(0, 0)
                ),
                "summary": {
                    "total_objects": 0,
                    "blind_paths": 0,
                    "obstacles": 0,
                    "other_objects": 0
                }
            },
            "processing_time": 0.0,
            "image_shape": (0, 0, 0),
            "model_info": {
                "model_path": self.model_path,
                "confidence_threshold": self.confidence_threshold,
                "nms_threshold": self.nms_threshold
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取检测统计"""
        return self.detection_stats.copy()
    
    def update_config(self, confidence_threshold: float = None, nms_threshold: float = None):
        """更新配置"""
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        if nms_threshold is not None:
            self.nms_threshold = nms_threshold
        
        logger.info(f"✅ 配置更新: conf={self.confidence_threshold}, nms={self.nms_threshold}")

def test_enhanced_detector():
    """测试增强检测器"""
    print("=" * 60)
    print("🧪 测试增强盲道检测器")
    print("=" * 60)
    
    # 初始化检测器
    detector = EnhancedBlindRoadDetector()
    
    if not detector.model:
        print("❌ 检测器初始化失败")
        return
    
    # 创建测试图像
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 添加一些测试形状
    cv2.rectangle(test_image, (100, 200), (300, 250), (128, 128, 128), -1)  # 模拟盲道
    cv2.rectangle(test_image, (400, 150), (450, 200), (0, 0, 255), -1)     # 模拟障碍物
    
    # 执行检测
    print("\n[测试] 执行检测...")
    result = detector.detect(test_image)
    
    # 显示结果
    print(f"[结果] 处理时间: {result['processing_time']:.3f}s")
    print(f"[结果] 检测到 {len(result['detections'])} 个对象")
    
    analysis = result['analysis']
    print(f"[结果] 盲道检测: {'是' if analysis['blind_path'].detected else '否'}")
    print(f"[结果] 障碍物检测: {'是' if analysis['obstacle'].detected else '否'}")
    
    # 显示统计
    stats = detector.get_stats()
    print(f"\n[统计] 总检测数: {stats['total_detections']}")
    print(f"[统计] 盲道检测数: {stats['blind_path_detections']}")
    print(f"[统计] 障碍物检测数: {stats['obstacle_detections']}")
    
    print("\n✅ 增强检测器测试完成")

if __name__ == "__main__":
    test_enhanced_detector()
