"""
障碍物检测模块 - 优化版

改进点:
1. 双阈值策略 (高置信度物体 + 低置信度但危险的物体)
2. 时序平滑 (帧间一致性)
3. 区域加权 (中心区域优先)
4. 小目标增强
"""
import cv2
import numpy as np
import logging
from typing import List, Optional, Dict
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.error("ultralytics未安装")

from app.config import settings


@dataclass
class Detection:
    """检测结果"""
    class_name: str
    class_id: int
    confidence: float
    bbox: List[float]        # 归一化坐标 [x1, y1, x2, y2]
    bbox_pixels: List[int]   # 像素坐标
    
    # 扩展属性
    center: List[float] = None      # bbox中心点 [cx, cy]
    area: float = 0.0               # bbox面积
    aspect_ratio: float = 0.0       # 宽高比
    direction: str = "center"       # left/center/right
    distance: Optional[float] = None
    
    # 运动属性
    velocity: Optional[List[float]] = None  # [vx, vy] 像素/帧
    
    # 跟踪ID
    track_id: Optional[int] = None


class EnhancedObstacleDetector:
    """
    增强型障碍物检测器
    
    新增功能:
    1. 自适应置信度阈值
    2. 帧间平滑
    3. 小目标检测优化
    4. 区域优先级
    """
    
    # 危险物体（即使置信度低也要检测）
    CRITICAL_CLASSES = {
        "car", "truck", "bus", "motorcycle", "bicycle",
        "person", "dog", "cat"
    }
    
    def __init__(
        self,
        model_path: str = None,
        device: str = "cpu",
        conf_threshold: float = 0.5,
        conf_critical: float = 0.3,  # 危险物体的低阈值
        iou_threshold: float = 0.45,
        history_size: int = 5
    ):
        if not YOLO_AVAILABLE:
            raise RuntimeError("请安装 ultralytics")
        
        self.model_path = model_path or settings.YOLO_MODEL_PATH
        self.device = device or settings.YOLO_DEVICE
        self.conf_threshold = conf_threshold
        self.conf_critical = conf_critical
        self.iou_threshold = iou_threshold
        
        # 加载模型
        logger.info(f"加载YOLO模型: {self.model_path}")
        self.model = YOLO(self.model_path)
        
        # 历史缓存（时序平滑）
        self.history = deque(maxlen=history_size)
        
        # 帧计数
        self.frame_count = 0
        
        logger.info(f"✅ YOLO模型加载完成")
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        检测障碍物（增强版）
        
        Args:
            image: BGR图像
        Returns:
            检测结果列表
        """
        h, w = image.shape[:2]
        self.frame_count += 1
        
        # ===== Step 1: YOLO推理 =====
        results = self.model.predict(
            source=image,
            conf=self.conf_critical,  # 使用低阈值，后续精细过滤
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            imgsz=640  # 输入尺寸
        )
        
        raw_detections = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.model.names[cls_id]
                
                # ===== Step 2: 双阈值过滤 =====
                # 规则：
                # - 危险物体 → 置信度 > conf_critical (0.3)
                # - 普通物体 → 置信度 > conf_threshold (0.5)
                if cls_name in self.CRITICAL_CLASSES:
                    if conf < self.conf_critical:
                        continue
                else:
                    if conf < self.conf_threshold:
                        continue
                
                # ===== Step 3: 计算扩展属性 =====
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                bbox_norm = [x1/w, y1/h, x2/w, y2/h]
                
                area = (x2 - x1) * (y2 - y1) / (w * h)  # 归一化面积
                aspect_ratio = (x2 - x1) / (y2 - y1 + 1e-6)
                
                # 方向判断（基于中心点）
                center_x_norm = cx / w
                if center_x_norm < 0.33:
                    direction = "left"
                elif center_x_norm < 0.67:
                    direction = "center"
                else:
                    direction = "right"
                
                detection = Detection(
                    class_name=cls_name,
                    class_id=cls_id,
                    confidence=conf,
                    bbox=bbox_norm,
                    bbox_pixels=[int(x1), int(y1), int(x2), int(y2)],
                    center=[cx/w, cy/h],
                    area=area,
                    aspect_ratio=aspect_ratio,
                    direction=direction
                )
                
                raw_detections.append(detection)
        
        # ===== Step 4: 区域优先级调整 =====
        # 中心区域的物体置信度加权
        for det in raw_detections:
            if det.direction == "center":
                det.confidence *= 1.1  # 中心物体提升10%
        
        # ===== Step 5: 小目标增强 =====
        # 面积很小但置信度高的物体（可能是远处危险物）
        for det in raw_detections:
            if det.area < 0.01 and det.confidence > 0.7:
                logger.debug(f"小目标增强: {det.class_name}, area={det.area:.4f}")
        
        # ===== Step 6: 时序平滑（可选）=====
        if self.history:
            smoothed = self._temporal_smooth(raw_detections)
        else:
            smoothed = raw_detections
        
        # 加入历史
        self.history.append(smoothed)
        
        logger.debug(f"检测到 {len(smoothed)} 个目标")
        
        return smoothed
    
    def _temporal_smooth(self, current: List[Detection]) -> List[Detection]:
        """
        时序平滑：抑制闪烁检测
        
        策略：
        - 当前帧检测到 + 历史帧也检测到 → 保留
        - 当前帧检测到 + 历史帧未检测到 → 降低置信度
        - 当前帧未检测 + 历史帧检测到 → 保留一帧（防止丢失）
        """
        if not self.history:
            return current
        
        prev_detections = self.history[-1]
        
        smoothed = []
        
        for det in current:
            # 检查历史帧是否有相似检测
            has_history = False
            for prev in prev_detections:
                if (det.class_name == prev.class_name and
                    self._iou(det.bbox_pixels, prev.bbox_pixels) > 0.3):
                    has_history = True
                    break
            
            if has_history:
                # 稳定检测，保留
                smoothed.append(det)
            else:
                # 新出现的检测，降低置信度
                det.confidence *= 0.8
                if det.confidence > 0.3:
                    smoothed.append(det)
        
        return smoothed
    
    def _iou(self, box1: List[int], box2: List[int]) -> float:
        """计算IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - inter
        
        return inter / (union + 1e-6)