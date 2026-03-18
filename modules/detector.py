#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modules/detector.py
盲道与障碍物检测器 - 为FastAPI后端提供检测服务
"""

from ultralytics import YOLO
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
import time
from pathlib import Path


class BlindRoadDetector:
    """
    盲道与障碍物检测器
    
    使用说明：
        detector = BlindRoadDetector("models/best.pt")
        result = detector.detect_from_bytes(image_bytes)
    """
    
    def __init__(self, model_path: str = "models/best.pt", device: str = "cpu"):
        """
        初始化检测器
        
        Args:
            model_path: YOLO模型权重文件路径
            device: 'cpu' 或 'cuda' (GPU)
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        print(f"🔄 正在加载YOLO模型: {model_path}")
        self.model = YOLO(model_path)
        self.device = device
        print(f"✅ YOLO模型加载完成 (设备: {device})")
        
        # ==================
        # 类别配置（根据你的训练模型调整）
        # ==================
        self.class_names = {
            0: "blind_road",      # 盲道
            1: "person",          # 行人
            2: "bicycle",         # 自行车
            3: "car",             # 汽车
            4: "pole",            # 电线杆/障碍柱
            5: "trash_bin",       # 垃圾桶
            6: "construction",    # 施工障碍
            7: "step",            # 台阶
            8: "pothole",         # 坑洞
            # ⚠️ 根据你实际训练的类别修改
        }
        
        # 中文名称映射（用于语音播报）
        self.chinese_names = {
            "blind_road": "盲道",
            "person": "行人",
            "bicycle": "自行车",
            "car": "车辆",
            "pole": "障碍柱",
            "trash_bin": "垃圾桶",
            "construction": "施工障碍",
            "step": "台阶",
            "pothole": "坑洞",
        }
        
        # 危险等级配置
        self.danger_levels = {
            "person": "medium",
            "bicycle": "high",
            "car": "high",
            "pole": "high",
            "trash_bin": "medium",
            "construction": "high",
            "step": "high",
            "pothole": "high",
            "blind_road": "low",
        }
        
        # 距离估算参考高度（米）
        self.reference_heights = {
            "person": 1.7,
            "bicycle": 1.0,
            "car": 1.5,
            "pole": 3.0,
            "trash_bin": 0.8,
            "construction": 1.0,
            "step": 0.15,
            "pothole": 0.3,
        }
        
        # 图像区域划分
        self.LEFT_THRESHOLD = 0.33
        self.RIGHT_THRESHOLD = 0.67
    
    def detect_from_bytes(self, image_bytes: bytes, conf_threshold: float = 0.5) -> Dict:
        """
        从图像字节流进行检测（FastAPI上传的文件）
        
        Args:
            image_bytes: JPEG/PNG图像的字节数据
            conf_threshold: 置信度阈值
            
        Returns:
            {
                "success": bool,
                "obstacles": [...],
                "blind_road_detected": bool,
                "blind_road_status": str,
                "voice_alert": str,
                "processing_time_ms": float
            }
        """
        start_time = time.time()
        
        # 解码图像
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {
                "success": False,
                "obstacles": [],
                "blind_road_detected": False,
                "blind_road_status": None,
                "voice_alert": "图像解码失败",
                "processing_time_ms": 0,
                "error": "无法解码图像"
            }
        
        return self._detect_internal(image, conf_threshold, start_time)
    
    def detect_from_frame(self, frame: np.ndarray, conf_threshold: float = 0.5) -> Dict:
        """
        从OpenCV帧进行检测（如果你有实时视频流）
        
        Args:
            frame: BGR格式的numpy数组
            conf_threshold: 置信度阈值
            
        Returns:
            同 detect_from_bytes
        """
        return self._detect_internal(frame, conf_threshold, time.time())
    
    def _detect_internal(self, image: np.ndarray, conf_threshold: float, start_time: float) -> Dict:
        """
        核心检测逻辑
        """
        h, w = image.shape[:2]
        
        # 运行YOLO推理
        results = self.model(
            image,
            conf=conf_threshold,
            device=self.device,
            verbose=False  # 不打印YOLO日志
        )
        
        obstacles = []
        blind_road_boxes = []  # 存储所有盲道检测框
        voice_alerts = []
        
        # 解析检测结果
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            
            for box in boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # 归一化坐标（0-1）
                bbox_normalized = [x1/w, y1/h, x2/w, y2/h]
                
                # 获取类别名
                class_name = self.class_names.get(cls_id, f"unknown_{cls_id}")
                
                # 计算中心点位置
                center_x = (x1 + x2) / 2 / w
                center_y = (y1 + y2) / 2 / h
                
                # 判断方向
                if center_x < self.LEFT_THRESHOLD:
                    direction = "left"
                elif center_x > self.RIGHT_THRESHOLD:
                    direction = "right"
                else:
                    direction = "center"
                
                # 估算距离
                obj_height_ratio = (y2 - y1) / h
                distance_estimate = self._estimate_distance(obj_height_ratio, class_name)
                
                # 危险等级
                danger_level = self.danger_levels.get(class_name, "medium")
                
                if class_name == "blind_road":
                    # 盲道检测
                    blind_road_boxes.append({
                        "center_x": center_x,
                        "center_y": center_y,
                        "bbox": bbox_normalized
                    })
                else:
                    # 障碍物
                    obstacle = {
                        "class_name": class_name,
                        "confidence": round(confidence, 3),
                        "bbox": [round(v, 4) for v in bbox_normalized],
                        "distance_estimate": round(distance_estimate, 1) if distance_estimate else None,
                        "direction": direction,
                        "danger_level": danger_level
                    }
                    obstacles.append(obstacle)
                    
                    # 生成语音提示
                    alert = self._generate_voice_alert(
                        class_name, direction, distance_estimate, danger_level
                    )
                    if alert:
                        voice_alerts.append(alert)
        
        # 分析盲道状态
        blind_road_status, blind_road_alert = self._analyze_blind_road(blind_road_boxes)
        if blind_road_alert:
            voice_alerts.insert(0, blind_road_alert)  # 盲道提示优先级最高
        
        # 合并语音提示
        voice_alert = ".".join(voice_alerts) if voice_alerts else None
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "obstacles": obstacles,
            "blind_road_detected": len(blind_road_boxes) > 0,
            "blind_road_status": blind_road_status,
            "voice_alert": voice_alert,
            "processing_time_ms": round(processing_time, 1)
        }
    
    def _estimate_distance(self, height_ratio: float, class_name: str) -> Optional[float]:
        """
        基于目标在图像中的高度比例估算距离
        
        公式: distance ≈ (ref_height * focal_factor) / height_ratio
        """
        ref_height = self.reference_heights.get(class_name)
        if ref_height is None or height_ratio < 0.01:
            return None
        
        # focal_factor 需要根据实际摄像头标定
        # 这里用经验值，你可以根据实际测试调整
        focal_factor = 0.8
        estimated = (ref_height * focal_factor) / height_ratio
        
        # 限制在合理范围（0.5米 - 20米）
        return max(0.5, min(estimated, 20.0))
    
    def _generate_voice_alert(
        self, 
        class_name: str, 
        direction: str, 
        distance: Optional[float], 
        danger_level: str
    ) -> Optional[str]:
        """
        生成语音提示文本
        """
        if distance is None:
            return None
        
        direction_text = {
            "left": "左方",
            "center": "前方",
            "right": "右方"
        }
        
        obj_name = self.chinese_names.get(class_name, "障碍物")
        dir_text = direction_text[direction]
        
        # 根据危险等级和距离生成不同的提示
        if danger_level == "high" and distance < 3:
            return f"注意！{dir_text}{distance:.0f}米处有{obj_name}"
        elif danger_level == "high" and distance < 5:
            return f"{dir_text}有{obj_name}，距离{distance:.0f}米"
        elif danger_level == "medium" and distance < 2:
            return f"{dir_text}有{obj_name}，请注意"
        
        return None
    
    def _analyze_blind_road(self, blind_road_boxes: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
        """
        分析盲道状态
        
        Returns:
            (status, voice_alert)
            status: "on_track" | "deviated_left" | "deviated_right" | "lost" | None
        """
        if not blind_road_boxes:
            return None, "未检测到盲道"
        
        # 取置信度最高或面积最大的盲道框
        main_blind_road = blind_road_boxes[0]  # 简化处理，取第一个
        
        center_x = main_blind_road["center_x"]
        
        # 判断盲道位置
        if 0.35 < center_x < 0.65:
            # 盲道在中心区域，说明在正道上
            return "on_track", None
        elif center_x <= 0.35:
            # 盲道在左边，说明用户偏右了
            return "deviated_right", "您已偏离盲道，请向左调整"
        else:
            # 盲道在右边，说明用户偏左了
            return "deviated_left", "您已偏离盲道，请向右调整"
    
    def batch_detect(self, images: List[np.ndarray]) -> List[Dict]:
        """
        批量检测（如果需要处理多帧）
        
        Args:
            images: 图像数组列表
            
        Returns:
            检测结果列表
        """
        return [self.detect_from_frame(img) for img in images]


# ==================
# 全局单例（避免重复加载模型）
# ==================
_detector_instance: Optional[BlindRoadDetector] = None

def get_detector(
    model_path: str = "models/best.pt",
    device: str = "cpu"
) -> BlindRoadDetector:
    """
    获取检测器单例
    
    这是B成员调用的主要接口：
        from modules.detector import get_detector
        detector = get_detector()
        result = detector.detect_from_bytes(image_bytes)
    """
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = BlindRoadDetector(model_path, device)
    return _detector_instance


# ==================
# 便捷函数（可选）
# ==================
def quick_detect(image_bytes: bytes) -> Dict:
    """
    快速检测函数（自动使用单例）
    
    使用示例：
        from modules.detector import quick_detect
        result = quick_detect(image_bytes)
    """
    detector = get_detector()
    return detector.detect_from_bytes(image_bytes)


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("YOLO检测器测试")
    print("=" * 60)
    
    # 加载模型
    detector = get_detector("models/best.pt", device="cpu")
    
    # 测试图片
    test_image_path = "test_images/test1.jpg"
    if Path(test_image_path).exists():
        with open(test_image_path, "rb") as f:
            image_bytes = f.read()
        
        result = detector.detect_from_bytes(image_bytes)
        
        print(f"\n✅ 检测成功")
        print(f"处理时间: {result['processing_time_ms']:.1f}ms")
        print(f"检测到障碍物数量: {len(result['obstacles'])}")
        print(f"盲道检测: {result['blind_road_detected']}")
        print(f"盲道状态: {result['blind_road_status']}")
        print(f"语音提示: {result['voice_alert']}")
        
        if result['obstacles']:
            print("\n障碍物详情:")
            for i, obs in enumerate(result['obstacles'], 1):
                print(f"  {i}. {obs['class_name']} - "
                      f"{obs['direction']} - "
                      f"{obs['distance_estimate']}米 - "
                      f"危险等级:{obs['danger_level']}")
    else:
        print(f"⚠️ 测试图片不存在: {test_image_path}")
        print("请准备一张测试图片到 test_images/test1.jpg")
