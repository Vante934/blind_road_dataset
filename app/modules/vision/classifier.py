"""
障碍物分类器 - 增强版

改进点:
1. 基于运动向量的动静分类
2. 地面异常专用检测
3. 危险等级细化评估
"""
import logging
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

from app.modules.vision.detector import Detection

logger = logging.getLogger(__name__)


class ObstacleType(str, Enum):
    DYNAMIC = "dynamic"
    STATIC = "static"
    GROUND_HAZARD = "ground"


@dataclass
class ClassifiedObstacle:
    """分类后的障碍物"""
    detection: Detection
    obstacle_type: ObstacleType
    danger_level: float          # 0-1
    priority: int                # 1-5
    description: str
    
    is_moving: bool = False
    need_trajectory: bool = False
    
    # 新增
    movement_vector: Optional[List[float]] = None  # [vx, vy]
    predicted_position: Optional[List[float]] = None


class EnhancedObstacleClassifier:
    """
    增强型障碍物分类器
    
    新增功能:
    1. 运动检测（光流法）
    2. 地面异常专用规则
    3. 上下文感知（多物体关系）
    """
    
    # 分类规则（扩展版）
    CLASSIFICATION_RULES = {
        # ===== 动态障碍物（高危） =====
        "car": {
            "type": ObstacleType.DYNAMIC,
            "danger": 0.9,
            "priority": 5,
            "desc": "汽车",
            "track": True,
            "speed_threshold": 0.01  # 运动阈值（像素/帧）
        },
        "truck": {
            "type": ObstacleType.DYNAMIC,
            "danger": 0.95,
            "priority": 5,
            "desc": "卡车",
            "track": True,
            "speed_threshold": 0.01
        },
        "bus": {
            "type": ObstacleType.DYNAMIC,
            "danger": 0.95,
            "priority": 5,
            "desc": "公交车",
            "track": True,
            "speed_threshold": 0.01
        },
        "motorcycle": {
            "type": ObstacleType.DYNAMIC,
            "danger": 0.85,
            "priority": 5,
            "desc": "摩托车",
            "track": True,
            "speed_threshold": 0.02
        },
        "bicycle": {
            "type": ObstacleType.DYNAMIC,
            "danger": 0.7,
            "priority": 4,
            "desc": "自行车",
            "track": True,
            "speed_threshold": 0.015
        },
        "person": {
            "type": ObstacleType.DYNAMIC,
            "danger": 0.6,
            "priority": 4,
            "desc": "行人",
            "track": True,
            "speed_threshold": 0.01
        },
        "dog": {
            "type": ObstacleType.DYNAMIC,
            "danger": 0.5,
            "priority": 3,
            "desc": "动物",
            "track": True,
            "speed_threshold": 0.02
        },
        "cat": {
            "type": ObstacleType.DYNAMIC,
            "danger": 0.4,
            "priority": 3,
            "desc": "动物",
            "track": True,
            "speed_threshold": 0.02
        },
        
        # ===== 静态障碍物 =====
        "traffic_light": {
            "type": ObstacleType.STATIC,
            "danger": 0.3,
            "priority": 2,
            "desc": "红绿灯",
            "track": False
        },
        "stop_sign": {
            "type": ObstacleType.STATIC,
            "danger": 0.4,
            "priority": 3,
            "desc": "停止标志",
            "track": False
        },
        "fire_hydrant": {
            "type": ObstacleType.STATIC,
            "danger": 0.6,
            "priority": 3,
            "desc": "消防栓",
            "track": False
        },
        "bench": {
            "type": ObstacleType.STATIC,
            "danger": 0.5,
            "priority": 2,
            "desc": "长椅",
            "track": False
        },
        "chair": {
            "type": ObstacleType.STATIC,
            "danger": 0.5,
            "priority": 2,
            "desc": "椅子",
            "track": False
        },
        
        # ===== 地面异常（特殊处理） =====
        "pothole": {
            "type": ObstacleType.GROUND_HAZARD,
            "danger": 0.75,
            "priority": 4,
            "desc": "坑洞",
            "track": False
        },
        "stairs": {
            "type": ObstacleType.GROUND_HAZARD,
            "danger": 0.85,
            "priority": 5,
            "desc": "台阶",
            "track": False
        },
        "curb": {
            "type": ObstacleType.GROUND_HAZARD,
            "danger": 0.65,
            "priority": 3,
            "desc": "路缘",
            "track": False
        },
    }
    
    def __init__(self):
        self.prev_detections: List[Detection] = []
    
    def classify(
        self, 
        detections: List[Detection],
        image: np.ndarray = None  # 可选：用于运动分析
    ) -> List[ClassifiedObstacle]:
        """
        分类检测结果
        
        Args:
            detections: 当前帧检测结果
            image: 当前帧图像（可选，用于运动分析）
        Returns:
            分类后的障碍物列表
        """
        classified = []
        
        for det in detections:
            # ===== Step 1: 基础分类 =====
            rule = self.CLASSIFICATION_RULES.get(
                det.class_name,
                {  # 默认规则
                    "type": ObstacleType.STATIC,
                    "danger": 0.5,
                    "priority": 2,
                    "desc": det.class_name,
                    "track": False,
                    "speed_threshold": 0.01
                }
            )
            
            # ===== Step 2: 运动检测（如果有历史帧）=====
            is_moving = False
            movement_vector = None
            
            if self.prev_detections:
                movement = self._detect_movement(det, self.prev_detections)
                if movement:
                    movement_vector = movement
                    speed = np.linalg.norm(movement)
                    
                    # 判断是否运动
                    if speed > rule.get("speed_threshold", 0.01):
                        is_moving = True
                        
                        # 动态调整类型
                        if rule["type"] == ObstacleType.STATIC:
                            # 静态物体在运动 → 可能是误分类
                            logger.debug(f"{det.class_name} 检测到运动，重新分类为动态")
                            rule = {**rule, "type": ObstacleType.DYNAMIC, "track": True}
            
            # ===== Step 3: 地面异常检测（基于位置）=====
            if self._is_ground_level(det):
                # bbox底部在画面下方 → 可能是地面异常
                if rule["type"] == ObstacleType.STATIC:
                    logger.debug(f"{det.class_name} 位于地面，可能是地面异常")
                    rule = {**rule, "type": ObstacleType.GROUND_HAZARD}
            
            # ===== Step 4: 危险等级调整 =====
            danger_level = rule["danger"]
            
            # 根据距离调整
            if det.distance:
                if det.distance < 1.0:
                    danger_level *= 1.3
                elif det.distance < 2.0:
                    danger_level *= 1.1
            
            # 根据方向调整
            if det.direction == "center":
                danger_level *= 1.2
            
            # 根据运动调整
            if is_moving:
                danger_level *= 1.15
            
            danger_level = min(danger_level, 1.0)
            
            # ===== Step 5: 构建分类结果 =====
            obstacle = ClassifiedObstacle(
                detection=det,
                obstacle_type=rule["type"],
                danger_level=danger_level,
                priority=rule["priority"],
                description=rule["desc"],
                is_moving=is_moving or rule.get("track", False),
                need_trajectory=rule.get("track", False),
                movement_vector=movement_vector
            )
            
            classified.append(obstacle)
        
        # 保存当前帧用于下次比较
        self.prev_detections = detections
        
        # 按优先级排序
        classified.sort(key=lambda x: (x.priority, x.danger_level), reverse=True)
        
        return classified
    
    def _detect_movement(
        self, 
        current: Detection, 
        prev_list: List[Detection]
    ) -> Optional[List[float]]:
        """
        检测物体运动
        
        策略：找到上一帧中最匹配的bbox，计算位移
        """
        best_match = None
        best_iou = 0.0
        
        for prev in prev_list:
            if prev.class_name != current.class_name:
                continue
            
            iou = self._calc_iou(current.bbox, prev.bbox)
            if iou > best_iou:
                best_iou = iou
                best_match = prev
        
        if best_match and best_iou > 0.3:
            # 计算中心点位移
            dx = current.center[0] - best_match.center[0]
            dy = current.center[1] - best_match.center[1]
            return [dx, dy]
        
        return None
    
    def _is_ground_level(self, det: Detection) -> bool:
        """判断是否在地面高度"""
        # bbox底部y坐标 > 0.7 认为是地面附近
        bottom_y = det.bbox[3]
        return bottom_y > 0.7
    
    def _calc_iou(self, box1: List[float], box2: List[float]) -> float:
        """计算IoU（归一化坐标）"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - inter
        
        return inter / (union + 1e-6)
    
    def filter_by_distance(
        self, 
        obstacles: List[ClassifiedObstacle], 
        max_distance: float = 5.0
    ) -> List[ClassifiedObstacle]:
        """过滤距离"""
        return [
            obs for obs in obstacles
            if obs.detection.distance is None or obs.detection.distance <= max_distance
        ]