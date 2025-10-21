#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
两点模式和拖拽模式标注工具
功能：
1. 两点模式：点击两个点，自动生成直线
2. 拖拽模式：点击起点，拖拽到终点，形成直线
3. 摄像头实时检测和语音播报
4. 轨迹预测：盲道识别 + 动态障碍物轨迹预测 + 碰撞风险评估
"""

import sys
import os
import cv2
import json
import time
import glob
import requests
import base64
import subprocess
import threading
import numpy as np
import math
from pathlib import Path
from collections import deque
from typing import Dict, List, Tuple, Optional

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QMessageBox, QListWidget, QGroupBox, QSpinBox,
                             QFrame, QSplitter, QTextEdit, QListWidgetItem,
                             QShortcut, QDialog, QGridLayout, QProgressBar,
                             QComboBox, QSlider, QStatusBar, QCheckBox)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QRect, QTimer, QThread
from PyQt5.QtGui import QPixmap, QImage, QMouseEvent, QPainter, QPen, QColor, QFont, QKeySequence

# 安全导入，防止依赖缺失导致程序崩溃
print("正在检查依赖...")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ ultralytics 导入成功")
except ImportError as e:
    YOLO_AVAILABLE = False
    print(f"❌ ultralytics 导入失败: {e}")
    print("请运行: pip install ultralytics")

try:
    from modules.trajectory_predictor import TrajectoryPredictor
    TRAJECTORY_PREDICTOR_AVAILABLE = True
    print("✅ 轨迹预测模块导入成功")
except ImportError as e:
    TRAJECTORY_PREDICTOR_AVAILABLE = False
    print(f"⚠️ 轨迹预测模块导入失败: {e}")

try:
    import sys
    import os
    # 添加项目根目录到路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    
    from modules.environment_detector import EnvironmentDetector
    ENVIRONMENT_DETECTOR_AVAILABLE = True
    print("✅ 环境检测模块导入成功")
except ImportError as e:
    ENVIRONMENT_DETECTOR_AVAILABLE = False
    print(f"⚠️ 环境检测模块导入失败: {e}")

try:
    from modules.voice_library import VoiceLibrary
    VOICE_LIBRARY_AVAILABLE = True
    print("✅ 语音库模块导入成功")
except ImportError as e:
    VOICE_LIBRARY_AVAILABLE = False
    print(f"⚠️ 语音库模块导入失败: {e}")

# ==================== 轨迹预测模块 ====================
class BlindPathDetector:
    """盲道检测器"""
    
    def __init__(self):
        self.path_history = deque(maxlen=50)
        self.path_center = None
        self.path_width = 0
        self.confidence = 0.0
        
    def detect_blind_path(self, frame: np.ndarray) -> Dict:
        """检测盲道"""
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 应用高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 筛选可能的盲道轮廓
        blind_path_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # 面积阈值
                # 计算轮廓的边界矩形
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # 盲道通常是长条形的
                if aspect_ratio > 2.0 and w > 50:
                    blind_path_contours.append(contour)
        
        if blind_path_contours:
            # 选择最大的盲道轮廓
            largest_contour = max(blind_path_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # 计算盲道中心
            center_x = x + w // 2
            center_y = y + h // 2
            
            self.path_center = (center_x, center_y)
            self.path_width = w
            self.confidence = min(1.0, cv2.contourArea(largest_contour) / 10000)
            
            return {
                'center': (center_x, center_y),
                'width': w,
                'confidence': self.confidence,
                'detected': True
            }
        
        return {'detected': False, 'confidence': 0.0}
    
    def predict_path_trajectory(self, steps: int = 10) -> List[Tuple[int, int]]:
        """预测盲道轨迹"""
        if len(self.path_history) < 3:
            return []
        
        # 获取最近的位置点
        recent_points = list(self.path_history)[-3:]
        
        # 计算平均速度
        if len(recent_points) >= 2:
            dx = recent_points[-1][0] - recent_points[0][0]
            dy = recent_points[-1][1] - recent_points[0][1]
            dt = recent_points[-1][2] - recent_points[0][2]
            
            if dt > 0:
                vx = dx / dt
                vy = dy / dt
                
                # 预测未来位置
                predicted_points = []
                current_point = (recent_points[-1][0], recent_points[-1][1])
                
                for i in range(1, steps + 1):
                    next_x = int(current_point[0] + vx * i * 0.1)  # 0.1秒间隔
                    next_y = int(current_point[1] + vy * i * 0.1)
                    predicted_points.append((next_x, next_y))
                
                return predicted_points
        
        return []

class MotionPredictor:
    """运动预测器"""
    
    def __init__(self, prediction_steps: int = 5):
        self.trajectories = {}  # object_id -> deque of (x, y, timestamp)
        self.prediction_steps = prediction_steps
        
    def update_trajectory(self, object_id: int, position: Tuple[int, int], timestamp: float):
        """更新物体轨迹"""
        if object_id not in self.trajectories:
            self.trajectories[object_id] = deque(maxlen=20)
        
        self.trajectories[object_id].append((position[0], position[1], timestamp))
    
    def predict_trajectory(self, object_id: int) -> List[Tuple[int, int]]:
        """预测物体轨迹"""
        if object_id not in self.trajectories or len(self.trajectories[object_id]) < 3:
            return []
        
        trajectory = list(self.trajectories[object_id])
        
        # 计算速度
        if len(trajectory) >= 2:
            dx = trajectory[-1][0] - trajectory[0][0]
            dy = trajectory[-1][1] - trajectory[0][1]
            dt = trajectory[-1][2] - trajectory[0][2]
            
            if dt > 0:
                vx = dx / dt
                vy = dy / dt
                
                # 预测未来位置
                predicted_points = []
                current_point = (trajectory[-1][0], trajectory[-1][1])
                
                for i in range(1, self.prediction_steps + 1):
                    next_x = int(current_point[0] + vx * i * 0.1)
                    next_y = int(current_point[1] + vy * i * 0.1)
                    predicted_points.append((next_x, next_y))
                
                return predicted_points
        
        return []
    
    def calculate_collision_risk(self, object_id: int, user_position: Tuple[int, int], 
                               prediction_steps: int = 5) -> float:
        """计算碰撞风险"""
        predicted_trajectory = self.predict_trajectory(object_id)
        if not predicted_trajectory:
            return 0.0
        
        # 计算预测轨迹与用户位置的最小距离
        min_distance = float('inf')
        for point in predicted_trajectory:
            distance = math.sqrt((point[0] - user_position[0])**2 + (point[1] - user_position[1])**2)
            min_distance = min(min_distance, distance)
        
        # 距离越近，风险越高
        if min_distance < 50:  # 50像素内为高风险
            return 1.0
        elif min_distance < 100:  # 100像素内为中风险
            return 0.5
        else:
            return 0.1

class EnhancedTracker:
    """增强跟踪器"""
    
    def __init__(self, max_disappeared: int = 30, max_distance: int = 50):
        self.next_object_id = 0
        self.objects = {}  # object_id -> (centroid, class_id, disappeared_count)
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def register(self, centroid: Tuple[int, int], class_id: int = 0):
        """注册新物体"""
        self.objects[self.next_object_id] = (centroid, class_id, 0)
        self.next_object_id += 1
    
    def deregister(self, object_id: int):
        """注销物体"""
        if object_id in self.objects:
            del self.objects[object_id]
    
    def update(self, detections: List[List]) -> List[Dict]:
        """更新跟踪状态"""
        if len(detections) == 0:
            # 没有检测到物体，增加消失计数
            for object_id in list(self.objects.keys()):
                centroid, class_id, disappeared = self.objects[object_id]
                disappeared += 1
                if disappeared > self.max_disappeared:
                    self.deregister(object_id)
                else:
                    self.objects[object_id] = (centroid, class_id, disappeared)
            return []
        
        # 初始化物体中心点数组
        input_centroids = []
        for detection in detections:
            centroid = self.get_centroid(detection)
            input_centroids.append(centroid)
        
        # 如果没有跟踪的物体，注册所有检测到的物体
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], detections[i][5] if len(detections[i]) > 5 else 0)
        else:
            # 获取当前跟踪的物体ID和中心点
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[object_id][0] for object_id in object_ids]
            
            # 计算距离矩阵
            distances = np.zeros((len(object_ids), len(input_centroids)))
            for i in range(len(object_ids)):
                for j in range(len(input_centroids)):
                    distances[i, j] = self.calculate_distance(object_centroids[i], input_centroids[j])
            
            # 使用匈牙利算法进行匹配
            from scipy.optimize import linear_sum_assignment
            try:
                row_indices, col_indices = linear_sum_assignment(distances)
                
                # 处理匹配结果
                for row, col in zip(row_indices, col_indices):
                    if distances[row, col] < self.max_distance:
                        # 更新现有物体
                        object_id = object_ids[row]
                        self.objects[object_id] = (input_centroids[col], detections[col][5] if len(detections[col]) > 5 else 0, 0)
                    else:
                        # 距离太远，注册为新物体
                        self.register(input_centroids[col], detections[col][5] if len(detections[col]) > 5 else 0)
                
                # 处理未匹配的检测结果
                unmatched_cols = set(range(len(input_centroids))) - set(col_indices)
                for col in unmatched_cols:
                    self.register(input_centroids[col], detections[col][5] if len(detections[col]) > 5 else 0)
                
                # 处理未匹配的跟踪物体
                unmatched_rows = set(range(len(object_ids))) - set(row_indices)
                for row in unmatched_rows:
                    object_id = object_ids[row]
                    centroid, class_id, disappeared = self.objects[object_id]
                    disappeared += 1
                    if disappeared > self.max_disappeared:
                        self.deregister(object_id)
                    else:
                        self.objects[object_id] = (centroid, class_id, disappeared)
                        
            except ImportError:
                # 如果没有scipy，使用简单的最近邻匹配
                for i, input_centroid in enumerate(input_centroids):
                    min_distance = float('inf')
                    min_index = -1
                    
                    for j, object_centroid in enumerate(object_centroids):
                        distance = self.calculate_distance(object_centroid, input_centroid)
                        if distance < min_distance and distance < self.max_distance:
                            min_distance = distance
                            min_index = j
                    
                    if min_index != -1:
                        # 更新现有物体
                        object_id = object_ids[min_index]
                        self.objects[object_id] = (input_centroid, detections[i][5] if len(detections[i]) > 5 else 0, 0)
                    else:
                        # 注册为新物体
                        self.register(input_centroid, detections[i][5] if len(detections[i]) > 5 else 0)
        
        # 返回跟踪结果
        tracked_objects = []
        for object_id, (centroid, class_id, disappeared) in self.objects.items():
            if disappeared == 0:  # 只返回当前帧检测到的物体
                tracked_objects.append({
                    'id': object_id,
                    'centroid': centroid,
                    'class_id': class_id,
                    'bbox': self.find_bbox_for_centroid(centroid, detections)
                })
        
        return tracked_objects
    
    def find_bbox_for_centroid(self, centroid: Tuple[int, int], detections: List[List]) -> List:
        """根据中心点找到对应的边界框"""
        for detection in detections:
            if len(detection) >= 4:
                x1, y1, x2, y2 = detection[:4]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                if abs(center_x - centroid[0]) < 10 and abs(center_y - centroid[1]) < 10:
                    return detection[:4]
        return [0, 0, 0, 0]
    
    def get_centroid(self, detection: List) -> Tuple[int, int]:
        """计算检测框的中心点"""
        if len(detection) >= 4:
            x1, y1, x2, y2 = detection[:4]
            return (int((x1 + x2) / 2), int((y1 + y2) / 2))
        return (0, 0)
    
    def calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """计算两点间距离"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def get_collision_risks(self, user_position: Tuple[int, int]) -> Dict[int, float]:
        """获取所有物体的碰撞风险"""
        risks = {}
        for object_id, (centroid, class_id, disappeared) in self.objects.items():
            if disappeared == 0:
                distance = self.calculate_distance(centroid, user_position)
                if distance < 100:
                    risk = max(0, 1 - distance / 100)
                    risks[object_id] = risk
        return risks

class TrajectoryPredictor:
    """轨迹预测器主类"""
    
    def __init__(self):
        self.blind_path_detector = BlindPathDetector()
        self.motion_predictor = MotionPredictor()
        self.tracker = EnhancedTracker()
        self.user_position = (320, 240)  # 默认用户位置（屏幕中心）
        
    def process_frame(self, frame: np.ndarray, detections: List[List]) -> Dict:
        """处理单帧图像"""
        # 检测盲道
        blind_path_info = self.blind_path_detector.detect_blind_path(frame)
        
        # 跟踪物体
        tracked_objects = self.tracker.update(detections)
        
        # 更新运动预测器
        current_time = time.time()
        for obj in tracked_objects:
            self.motion_predictor.update_trajectory(obj['id'], obj['centroid'], current_time)
        
        # 计算碰撞风险
        collision_risks = self.tracker.get_collision_risks(self.user_position)
        
        # 生成警告信息
        warnings = self.generate_warnings(tracked_objects, collision_risks, blind_path_info)
        
        return {
            'blind_path': blind_path_info,
            'tracked_objects': tracked_objects,
            'collision_risks': collision_risks,
            'warnings': warnings,
            'safety_guidance': self.get_safety_guidance()
        }
    
    def generate_warnings(self, tracked_objects: List[Dict], collision_risks: Dict[int, float], 
                         blind_path_info: Optional[Dict]) -> List[str]:
        """生成警告信息"""
        warnings = []
        
        # 检查高碰撞风险的物体
        for object_id, risk in collision_risks.items():
            if risk > 0.7:
                obj = next((obj for obj in tracked_objects if obj['id'] == object_id), None)
                if obj:
                    class_name = self.get_class_name(obj['class_id'])
                    direction = self.get_direction(self.user_position[0], obj['centroid'][0])
                    warnings.append(f"⚠️ 高风险：{direction}有{class_name}，碰撞风险{risk:.1%}")
        
        # 检查盲道状态
        if blind_path_info:
            if blind_path_info['confidence'] < 0.5:
                warnings.append("⚠️ 盲道识别置信度较低，请小心")
        else:
            warnings.append("⚠️ 未检测到盲道，请谨慎前行")
        
        return warnings
    
    def get_class_name(self, class_id: int) -> str:
        """获取类别名称"""
        return CLASS_INFO.get(class_id, {}).get('name', f'物体{class_id}')
    
    def get_direction(self, x1: int, x2: int) -> str:
        """获取方向描述"""
        diff = x2 - x1
        if abs(diff) < 50:
            return "正前方"
        elif diff > 0:
            return "右前方"
        else:
            return "左前方"
    
    def calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """计算距离"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def update_user_position(self, position: Tuple[int, int]):
        """更新用户位置"""
        self.user_position = position
    
    def get_safety_guidance(self, frame_center: Tuple[int, int] = (320, 240)) -> str:
        """获取安全指导"""
        return "请沿盲道前行，注意周围障碍物"

# ==================== 语音管理模块 ====================
class VoiceLibrary:
    """语音库管理"""
    
    def __init__(self, config_file: str = "configs/voice_config.json"):
        self.config_file = config_file
        self.obstacle_types = {}
        self.class_mapping = {}
        self.special_scenarios = {}
        self.load_config()
    
    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.obstacle_types = config.get('obstacle_types', {})
                self.class_mapping = config.get('class_mapping', {})
                self.special_scenarios = config.get('special_scenarios', {})
        except Exception as e:
            print(f"⚠️ 加载语音配置失败: {e}")
            # 使用默认配置
            self.obstacle_types = {}
            self.class_mapping = {}
            self.special_scenarios = {}
    
    def get_obstacle_info(self, class_id: int) -> Dict:
        """获取障碍物信息"""
        class_id_str = str(class_id)
        if class_id_str in self.class_mapping:
            path = self.class_mapping[class_id_str]
            current = self.obstacle_types
            for key in path:
                if key in current:
                    current = current[key]
                else:
                    return {}
            return current
        return {}
    
    def generate_message(self, class_id: int, distance: float, direction: str, risk_level: str = "near") -> str:
        """生成语音消息"""
        obstacle_info = self.get_obstacle_info(class_id)
        if obstacle_info and 'templates' in obstacle_info:
            templates = obstacle_info['templates']
            if risk_level in templates:
                template = templates[risk_level]
                return template.format(distance=f"{distance:.1f}", direction=direction)
        
        # 使用默认模板
        return f"前方{distance:.1f}米{direction}有障碍物"
    
    def get_special_message(self, scenario: str) -> str:
        """获取特殊场景消息"""
        return self.special_scenarios.get(scenario, "")

# 百度语音API配置
BAIDU_APP_ID = '119634292'
BAIDU_API_KEY = 'w978fA2S7PJmUy4IEvlGqxfx'
BAIDU_SECRET_KEY = 'ZeTBNN1UYQRL1kaDEEImHm07Y09jgaRc'

# COCO数据集类别映射（YOLO默认检测类别）
COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus",
    6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant",
    11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat",
    16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant",
    21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella",
    26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis",
    31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove",
    36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
    56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 60: "dining table",
    61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote",
    66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster",
    71: "sink", 72: "refrigerator", 73: "book", 74: "clock", 75: "vase",
    76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush"
}

# 盲道导航障碍物类型标签库（完整版）
# 依据ISO 23599国际无障碍标准与GB50763国内规范制定
BLIND_ROAD_CLASSES = {
    # 静态障碍物 - 地面障碍
    0: {"name": "井盖凸起", "color": (0, 255, 0), "category": "static_ground_protrusion", "risk_level": 3},
    1: {"name": "路面修补凸包", "color": (0, 255, 128), "category": "static_ground_protrusion", "risk_level": 2},
    2: {"name": "减速带", "color": (0, 255, 255), "category": "static_ground_protrusion", "risk_level": 2},
    3: {"name": "轨道凸起", "color": (0, 0, 255), "category": "static_ground_protrusion", "risk_level": 3},
    4: {"name": "排水沟", "color": (255, 0, 0), "category": "static_ground_depression", "risk_level": 3},
    5: {"name": "坑洞", "color": (255, 0, 128), "category": "static_ground_depression", "risk_level": 4},
    6: {"name": "缺失地砖", "color": (255, 0, 255), "category": "static_ground_depression", "risk_level": 2},
    7: {"name": "施工凹槽", "color": (255, 128, 0), "category": "static_ground_depression", "risk_level": 4},
    8: {"name": "积水滩", "color": (255, 128, 128), "category": "static_ground_surface", "risk_level": 2},
    9: {"name": "油污区", "color": (255, 128, 255), "category": "static_ground_surface", "risk_level": 3},
    10: {"name": "光滑大理石地面", "color": (128, 0, 255), "category": "static_ground_surface", "risk_level": 2},
    11: {"name": "落叶堆积", "color": (128, 128, 255), "category": "static_ground_surface", "risk_level": 1},
    
    # 静态障碍物 - 固定设施
    12: {"name": "路灯杆", "color": (0, 128, 255), "category": "static_facility_street", "risk_level": 3},
    13: {"name": "公交站牌", "color": (128, 255, 0), "category": "static_facility_street", "risk_level": 2},
    14: {"name": "垃圾桶", "color": (128, 255, 128), "category": "static_facility_street", "risk_level": 2},
    15: {"name": "自行车架", "color": (128, 255, 255), "category": "static_facility_street", "risk_level": 2},
    16: {"name": "报刊亭", "color": (255, 128, 255), "category": "static_facility_commercial", "risk_level": 2},
    17: {"name": "冰淇淋车", "color": (255, 255, 128), "category": "static_facility_commercial", "risk_level": 2},
    18: {"name": "临时展台", "color": (255, 255, 255), "category": "static_facility_commercial", "risk_level": 2},
    19: {"name": "自动售货机", "color": (64, 64, 64), "category": "static_facility_commercial", "risk_level": 2},
    20: {"name": "消防栓", "color": (128, 64, 64), "category": "static_facility_public", "risk_level": 3},
    21: {"name": "电箱", "color": (64, 128, 64), "category": "static_facility_public", "risk_level": 3},
    22: {"name": "邮筒", "color": (64, 64, 128), "category": "static_facility_public", "risk_level": 2},
    23: {"name": "AED设备箱", "color": (128, 128, 64), "category": "static_facility_public", "risk_level": 2},
    
    # 动态障碍物 - 行人相关
    24: {"name": "站立行人", "color": (64, 128, 128), "category": "dynamic_pedestrian_individual", "risk_level": 2},
    25: {"name": "奔跑儿童", "color": (192, 64, 64), "category": "dynamic_pedestrian_individual", "risk_level": 4},
    26: {"name": "滑板少年", "color": (64, 192, 64), "category": "dynamic_pedestrian_individual", "risk_level": 3},
    27: {"name": "低头族", "color": (64, 64, 192), "category": "dynamic_pedestrian_individual", "risk_level": 2},
    28: {"name": "排队人群", "color": (192, 192, 64), "category": "dynamic_pedestrian_group", "risk_level": 2},
    29: {"name": "旅游团", "color": (64, 192, 192), "category": "dynamic_pedestrian_group", "risk_level": 2},
    30: {"name": "广场舞群体", "color": (192, 64, 192), "category": "dynamic_pedestrian_group", "risk_level": 2},
    31: {"name": "抗议集会", "color": (192, 192, 192), "category": "dynamic_pedestrian_group", "risk_level": 3},
    32: {"name": "轮椅使用者", "color": (32, 32, 32), "category": "dynamic_pedestrian_special", "risk_level": 3},
    33: {"name": "导盲犬", "color": (96, 32, 32), "category": "dynamic_pedestrian_special", "risk_level": 3},
    34: {"name": "拄拐行人", "color": (32, 96, 32), "category": "dynamic_pedestrian_special", "risk_level": 3},
    35: {"name": "婴儿车", "color": (32, 32, 96), "category": "dynamic_pedestrian_special", "risk_level": 3},
    
    # 动态障碍物 - 交通工具
    36: {"name": "共享单车", "color": (96, 96, 32), "category": "dynamic_vehicle_non_motorized", "risk_level": 2},
    37: {"name": "电动自行车", "color": (32, 96, 96), "category": "dynamic_vehicle_non_motorized", "risk_level": 3},
    38: {"name": "三轮车", "color": (96, 32, 96), "category": "dynamic_vehicle_non_motorized", "risk_level": 2},
    39: {"name": "平衡车", "color": (96, 96, 96), "category": "dynamic_vehicle_non_motorized", "risk_level": 3},
    40: {"name": "违停汽车", "color": (160, 48, 48), "category": "dynamic_vehicle_motorized", "risk_level": 3},
    41: {"name": "送货卡车", "color": (48, 160, 48), "category": "dynamic_vehicle_motorized", "risk_level": 3},
    42: {"name": "紧急车辆", "color": (48, 48, 160), "category": "dynamic_vehicle_motorized", "risk_level": 4},
    43: {"name": "移动餐车", "color": (160, 160, 48), "category": "dynamic_vehicle_motorized", "risk_level": 2},
    44: {"name": "手推车", "color": (48, 160, 160), "category": "dynamic_vehicle_micro", "risk_level": 2},
    45: {"name": "行李车", "color": (160, 48, 160), "category": "dynamic_vehicle_micro", "risk_level": 2},
    46: {"name": "超市购物车", "color": (160, 160, 160), "category": "dynamic_vehicle_micro", "risk_level": 2},
    47: {"name": "平板拖车", "color": (80, 24, 24), "category": "dynamic_vehicle_micro", "risk_level": 2},
    
    # 动态障碍物 - 动物类
    48: {"name": "未栓绳犬只", "color": (24, 80, 24), "category": "dynamic_animal_pet", "risk_level": 4},
    49: {"name": "猫", "color": (24, 24, 80), "category": "dynamic_animal_pet", "risk_level": 2},
    50: {"name": "鸽子群", "color": (80, 80, 24), "category": "dynamic_animal_pet", "risk_level": 1},
    51: {"name": "流浪动物", "color": (24, 80, 80), "category": "dynamic_animal_pet", "risk_level": 2},
    52: {"name": "导盲犬工作", "color": (80, 24, 80), "category": "dynamic_animal_working", "risk_level": 3},
    53: {"name": "警犬", "color": (80, 80, 80), "category": "dynamic_animal_working", "risk_level": 3},
    54: {"name": "马术用马", "color": (40, 12, 12), "category": "dynamic_animal_working", "risk_level": 3},
    
    # 日常高频障碍 - 商业活动
    55: {"name": "早餐摊", "color": (12, 40, 12), "category": "daily_commercial_stall", "risk_level": 2},
    56: {"name": "夜市摊位", "color": (12, 12, 40), "category": "daily_commercial_stall", "risk_level": 2},
    57: {"name": "促销展台", "color": (40, 40, 12), "category": "daily_commercial_stall", "risk_level": 2},
    58: {"name": "流动花车", "color": (12, 40, 40), "category": "daily_commercial_stall", "risk_level": 2},
    59: {"name": "快递堆放", "color": (40, 12, 40), "category": "daily_commercial_goods", "risk_level": 2},
    60: {"name": "货品装卸", "color": (40, 40, 40), "category": "daily_commercial_goods", "risk_level": 2},
    61: {"name": "啤酒箱", "color": (20, 6, 6), "category": "daily_commercial_goods", "risk_level": 2},
    62: {"name": "蔬菜筐", "color": (6, 20, 6), "category": "daily_commercial_goods", "risk_level": 2},
    
    # 日常高频障碍 - 临时性障碍
    63: {"name": "婚礼拱门", "color": (6, 6, 20), "category": "daily_temporary_activity", "risk_level": 2},
    64: {"name": "拍摄器材", "color": (20, 20, 6), "category": "daily_temporary_activity", "risk_level": 2},
    65: {"name": "临时舞台", "color": (6, 20, 20), "category": "daily_temporary_activity", "risk_level": 2},
    66: {"name": "充气城堡", "color": (20, 6, 20), "category": "daily_temporary_activity", "risk_level": 2},
    67: {"name": "折断树枝", "color": (20, 20, 20), "category": "daily_temporary_natural", "risk_level": 2},
    68: {"name": "冰面", "color": (10, 3, 3), "category": "daily_temporary_natural", "risk_level": 3},
    69: {"name": "积雪堆", "color": (3, 10, 3), "category": "daily_temporary_natural", "risk_level": 2},
    70: {"name": "沙尘堆积", "color": (3, 3, 10), "category": "daily_temporary_natural", "risk_level": 1},
    
    # 日常高频障碍 - 特殊场景
    71: {"name": "地铁闸机", "color": (10, 10, 3), "category": "daily_special_transport", "risk_level": 2},
    72: {"name": "安检设备", "color": (3, 10, 10), "category": "daily_special_transport", "risk_level": 2},
    73: {"name": "公交卡机", "color": (10, 3, 10), "category": "daily_special_transport", "risk_level": 2},
    74: {"name": "共享单车停放区", "color": (10, 10, 10), "category": "daily_special_transport", "risk_level": 2},
    75: {"name": "ATM机", "color": (5, 1, 1), "category": "daily_special_service", "risk_level": 2},
    76: {"name": "充电桩", "color": (1, 5, 1), "category": "daily_special_service", "risk_level": 2},
    77: {"name": "快递柜", "color": (1, 1, 5), "category": "daily_special_service", "risk_level": 2},
    78: {"name": "体重秤", "color": (5, 5, 1), "category": "daily_special_service", "risk_level": 1},
    
    # 建筑障碍 - 设计缺陷
    79: {"name": "过窄盲道", "color": (1, 5, 5), "category": "architectural_design_defect", "risk_level": 3},
    80: {"name": "直角转弯", "color": (5, 1, 5), "category": "architectural_design_defect", "risk_level": 2},
    81: {"name": "突然断头", "color": (5, 5, 5), "category": "architectural_design_defect", "risk_level": 4},
    82: {"name": "绿化带侵占", "color": (2, 0, 0), "category": "architectural_design_defect", "risk_level": 2},
    83: {"name": "过矮扶手", "color": (0, 2, 0), "category": "architectural_design_defect", "risk_level": 2},
    84: {"name": "反光玻璃幕墙", "color": (0, 0, 2), "category": "architectural_design_defect", "risk_level": 2},
    85: {"name": "旋转门无辅助把手", "color": (2, 2, 0), "category": "architectural_design_defect", "risk_level": 3},
    
    # 建筑障碍 - 施工相关
    86: {"name": "水泥搅拌车", "color": (0, 2, 2), "category": "architectural_construction", "risk_level": 3},
    87: {"name": "脚手架", "color": (2, 0, 2), "category": "architectural_construction", "risk_level": 3},
    88: {"name": "建材堆放", "color": (2, 2, 2), "category": "architectural_construction", "risk_level": 2},
    89: {"name": "钻孔机", "color": (1, 0, 0), "category": "architectural_construction", "risk_level": 3},
    90: {"name": "临时电线", "color": (0, 1, 0), "category": "architectural_construction", "risk_level": 3},
    91: {"name": "测量标桩", "color": (0, 0, 1), "category": "architectural_construction", "risk_level": 2},
    92: {"name": "探坑", "color": (1, 1, 0), "category": "architectural_construction", "risk_level": 3},
    93: {"name": "围挡延伸", "color": (0, 1, 1), "category": "architectural_construction", "risk_level": 2},
    
    # 建筑障碍 - 特殊建筑结构
    94: {"name": "悬挂灯笼", "color": (1, 0, 1), "category": "architectural_special", "risk_level": 2},
    95: {"name": "监控杆", "color": (1, 1, 1), "category": "architectural_special", "risk_level": 2},
    96: {"name": "吊装装饰", "color": (0, 0, 0), "category": "architectural_special", "risk_level": 2},
    97: {"name": "横幅绳索", "color": (255, 255, 255), "category": "architectural_special", "risk_level": 2},
    98: {"name": "通风井", "color": (128, 128, 128), "category": "architectural_special", "risk_level": 3},
    99: {"name": "地铁出口", "color": (64, 64, 64), "category": "architectural_special", "risk_level": 2},
    100: {"name": "地下车库坡道", "color": (32, 32, 32), "category": "architectural_special", "risk_level": 3}
}

DEFAULT_COLOR = (128, 128, 128)

class SimpleVoiceSystem:
    """简化的语音系统 - 支持GUI内嵌播放"""
    def __init__(self):
        self.access_token = None
        self.get_access_token()
        self.is_enabled = True
        self.last_speak_time = 0
        self.last_speak_text = ""  # 记录上次播报的文本
        self.speak_lock = threading.Lock()
        self.media_player = None
        self.init_media_player()
    
    def init_media_player(self):
        """初始化媒体播放器"""
        try:
            from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
            from PyQt5.QtCore import QUrl
            self.media_player = QMediaPlayer()
            self.media_player.setVolume(80)
            print("✅ GUI媒体播放器初始化成功")
        except Exception as e:
            print(f"⚠️ GUI媒体播放器初始化失败: {e}")
            self.media_player = None
    
    def get_access_token(self):
        """获取百度API访问令牌"""
        try:
            url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={BAIDU_API_KEY}&client_secret={BAIDU_SECRET_KEY}"
            response = requests.get(url)
            if response.status_code == 200:
                result = response.json()
                self.access_token = result.get('access_token')
                print(f"✅ 百度API访问令牌获取成功")
                return True
            else:
                print(f"❌ 获取访问令牌失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 获取访问令牌出错: {e}")
            return False
    
    def speak(self, text):
        """语音播报（简化版）"""
        if not self.is_enabled or not self.access_token:
            return
        
        # 防重复机制：相同文本不重复播报
        if text == self.last_speak_text:
            return
        
        # 简单的冷却机制（增加冷却时间，避免语音重叠）
        current_time = time.time()
        if current_time - self.last_speak_time < 2.0:  # 2秒冷却，避免语音重叠
            return
        
        self.last_speak_time = current_time
        self.last_speak_text = text  # 记录本次播报的文本
        
        def speak_thread():
            try:
                with self.speak_lock:
                    # 生成语音文件
                    temp_file = self.generate_speech(text)
                    if temp_file and os.path.exists(temp_file):
                        # 播放音频
                        self.play_audio(temp_file)
                        # 延迟删除文件
                        threading.Timer(3.0, lambda: self.cleanup_file(temp_file)).start()
            except Exception as e:
                print(f"❌ 语音播报错误: {e}")
        
        threading.Thread(target=speak_thread, daemon=True).start()
    
    def generate_speech(self, text):
        """生成语音文件"""
        try:
            import tempfile
            import uuid
            
            # 使用UUID生成唯一文件名，避免冲突
            unique_id = str(uuid.uuid4())[:8]
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, f'blind_road_{unique_id}.mp3')
            
            # 确保文件不存在
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            url = "https://tsn.baidu.com/text2audio"
            params = {
                'tok': self.access_token,
                'tex': text,
                'per': 0,  # 女声
                'spd': 5,  # 语速
                'pit': 5,  # 音调
                'vol': 5,  # 音量
                'cuid': 'blind_road_detector',
                'ctp': 1,
                'lan': 'zh'
            }
            
            response = requests.post(url, data=params)
            if response.status_code == 200:
                with open(temp_file, 'wb') as f:
                    f.write(response.content)
                print(f"✅ 语音文件生成: {temp_file}")
                return temp_file
            else:
                print(f"❌ 语音合成失败: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ 生成语音文件错误: {e}")
            return None
    
    def play_audio(self, audio_file):
        """播放音频文件 - GUI内嵌播放"""
        try:
            if self.media_player and os.path.exists(audio_file):
                # 使用GUI内嵌媒体播放器播放
                from PyQt5.QtCore import QUrl
                from PyQt5.QtMultimedia import QMediaContent
                
                file_path = os.path.abspath(audio_file)
                url = QUrl.fromLocalFile(file_path)
                content = QMediaContent(url)
                self.media_player.setMedia(content)
                self.media_player.play()
                print(f"🎵 GUI内嵌播放音频: {audio_file}")
            else:
                # 备用方案：使用系统默认播放器
                import platform
                if platform.system() == "Windows":
                    subprocess.Popen(['start', audio_file], shell=True)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.Popen(['open', audio_file])
                else:  # Linux
                    subprocess.Popen(['xdg-open', audio_file])
                print(f"🎵 系统播放器播放音频: {audio_file}")
        except Exception as e:
            print(f"❌ 播放音频错误: {e}")
            # 最后的备用方案
            try:
                import platform
                if platform.system() == "Windows":
                    subprocess.Popen(['start', audio_file], shell=True)
                print(f"🎵 备用播放器播放音频: {audio_file}")
            except:
                print(f"❌ 所有播放方式都失败")
    
    def cleanup_file(self, file_path):
        """清理临时文件"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️ 清理临时文件: {file_path}")
        except:
            pass
    
    def enable(self):
        """启用语音"""
        self.is_enabled = True
        print("🔊 语音系统已启用")
    
    def disable(self):
        """禁用语音"""
        self.is_enabled = False
        print("🔇 语音系统已禁用")

# 使用COCO类别作为默认检测类别
CLASS_INFO = COCO_CLASSES
DEFAULT_COLOR = (128, 128, 128)  # 未知类别用灰色

class TwoPointAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.annotations = []
        self.current_image_path = None
        self.image_files = []
        self.current_image_index = -1
        
        # 绘制状态
        self.drawing_mode = "two_point"  # "two_point" 或 "drag"
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.temp_points = []  # 临时存储两点模式的点
        self.drag_start = None  # 拖拽模式的起点
        
        # 界面设置
        self.font_size = 12
        self.line_width = 3
        self.point_size = 8
        
        # 图像缩放信息
        self.original_size = None  # 原始图像尺寸
        self.display_size = None   # 显示尺寸
        self.scale_factor = 1.0    # 缩放因子
        
        # 轨迹预测系统
        if TRAJECTORY_PREDICTOR_AVAILABLE:
            self.trajectory_predictor = TrajectoryPredictor()
            print("✅ 轨迹预测系统已集成")
        else:
            self.trajectory_predictor = None
            print("⚠️ 轨迹预测系统不可用")
        
        # 环境检测系统
        if ENVIRONMENT_DETECTOR_AVAILABLE:
            try:
                self.env_detector = EnvironmentDetector()
                print("✅ 环境检测系统已集成")
            except Exception as e:
                print(f"⚠️ 环境检测系统初始化失败: {e}")
                self.env_detector = None
        else:
            self.env_detector = None
            print("⚠️ 环境检测系统不可用")
        
        # 语音播报系统
        self.voice_enabled = True
        self.voice_mode = "简洁模式"  # 简洁模式/详细模式/静默模式
        self.volume = 80
        self.last_voice_time = 0
        self.voice_cooldown = 2.0  # 语音播报冷却时间
        import queue
        self.voice_queue = queue.Queue()  # 语音播报队列
        self.current_voice_priority = 0  # 当前播报优先级
        
        # 检测精度设置
        self.detection_accuracy = "高精度"
        self.detection_confidence_threshold = 0.3
        self.detection_nms_threshold = 0.4
        
        # 障碍物变化检测
        self.previous_detections = []  # 上一帧的检测结果
        self.detection_change_threshold = 0.3  # 变化阈值
        self.last_announcement_time = 0  # 上次播报时间
        self.announcement_cooldown = 3.0  # 播报冷却时间
        
        # 初始化语音合成
        self.init_voice_synthesis()
        
        # 初始化语音播报系统
        self.init_voice_system()
        
        # 初始化YOLO模型
        self.init_yolo_model()
        
        # 初始化语音状态显示
        QTimer.singleShot(1000, self.update_voice_status)
    
    def init_voice_synthesis(self):
        """初始化语音合成"""
        try:
            import pyttsx3
            
            # 测试语音合成是否可用
            test_tts = pyttsx3.init()
            test_tts.stop()
            
            print("✅ 语音合成引擎初始化成功")
            self.voice_synthesis_available = True
            
        except ImportError:
            print("⚠️ pyttsx3未安装，使用控制台输出代替语音播报")
            self.voice_synthesis_available = False
        except Exception as e:
            print(f"⚠️ 语音合成初始化失败: {e}")
            self.voice_synthesis_available = False
    
    def init_voice_system(self):
        """初始化语音播报系统"""
        try:
            import threading
            import queue
            import time
            
            # 初始化语音队列和状态管理
            self.voice_queue = queue.PriorityQueue()
            self.voice_thread = None
            self.is_voice_playing = False
            self.voice_lock = threading.Lock()
            self.global_tts_engine = None
            
            # 启动语音工作线程
            self.start_voice_worker()
            
            print("✅ 语音播报系统初始化成功")
            
        except Exception as e:
            print(f"⚠️ 语音播报系统初始化失败: {e}")
            self.voice_synthesis_available = False
    
    def init_yolo_model(self):
        """初始化YOLO模型"""
        try:
            if YOLO_AVAILABLE:
                from ultralytics import YOLO
                # 尝试加载预训练模型
                self.yolo_model = YOLO('yolov8n.pt')  # 使用nano版本，速度更快
                print("✅ YOLO模型加载成功")
                self.yolo_available = True
            else:
                print("⚠️ YOLO不可用，使用模拟检测")
                self.yolo_model = None
                self.yolo_available = False
        except Exception as e:
            print(f"⚠️ YOLO模型加载失败: {e}")
            self.yolo_model = None
            self.yolo_available = False
        
    def initUI(self):
        self.setWindowTitle('盲道障碍检测系统 - 增强版 v2.0')
        self.setGeometry(100, 100, 1920, 1080)
        
        # 设置大字体
        self.setStyleSheet("""
            QMainWindow { font-size: 14px; }
            QPushButton { font-size: 14px; padding: 8px 12px; min-height: 28px; }
            QLabel { font-size: 14px; }
            QGroupBox { font-size: 13px; font-weight: bold; }
            QListWidget { font-size: 13px; }
            QSplitter::handle { background-color: #ddd; }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局 - 垂直分割
        main_layout = QVBoxLayout(central_widget)
        
        # 创建主分割器（水平）
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # 左侧控制面板
        control_panel = self.create_control_panel()
        control_panel.setMinimumWidth(300)
        control_panel.setMaximumWidth(500)
        main_splitter.addWidget(control_panel)
        
        # 中间摄像头检测区域（主要区域）
        camera_panel = self.create_camera_panel()
        camera_panel.setMinimumWidth(800)
        main_splitter.addWidget(camera_panel)
        
        # 摄像头检测相关变量
        self.camera_active = False
        self.cap = None
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_frame)
        
        # 右侧轨迹预测面板
        analysis_panel = self.create_analysis_panel()
        analysis_panel.setMinimumWidth(300)
        analysis_panel.setMaximumWidth(500)
        main_splitter.addWidget(analysis_panel)
        
        # 设置分割器比例：控制面板:摄像头:分析面板 = 1:4:1
        main_splitter.setSizes([300, 1600, 300])
        
        # 底部状态栏
        self.create_status_bar()
        
        # 快捷键撤回
        undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        undo_shortcut.activated.connect(self.undo_last_annotation)
        
    def create_control_panel(self):
        """创建控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 标题
        title_label = QLabel("盲道障碍检测系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 8px; color: #333; background-color: #e3f2fd; padding: 10px; border-radius: 8px;")
        layout.addWidget(title_label)
        
        # 摄像头控制组
        camera_control_group = QGroupBox("摄像头控制")
        camera_control_layout = QVBoxLayout(camera_control_group)
        
        # 摄像头开关
        self.camera_start_btn = QPushButton("📹 开启摄像头检测")
        self.camera_start_btn.clicked.connect(self.toggle_camera_detection)
        self.camera_start_btn.setStyleSheet("QPushButton { background-color: #4caf50; color: white; font-weight: bold; padding: 12px; font-size: 16px; }")
        camera_control_layout.addWidget(self.camera_start_btn)
        
        # 显示控制
        display_control_layout = QHBoxLayout()
        
        self.show_detection_btn = QPushButton("检测框")
        self.show_detection_btn.setCheckable(True)
        self.show_detection_btn.setChecked(True)
        self.show_detection_btn.clicked.connect(self.toggle_show_detection)
        self.show_detection_btn.setStyleSheet("QPushButton { background-color: #4caf50; color: white; font-weight: bold; padding: 8px; }")
        display_control_layout.addWidget(self.show_detection_btn)
        
        self.show_trajectory_btn = QPushButton("轨迹")
        self.show_trajectory_btn.setCheckable(True)
        self.show_trajectory_btn.setChecked(True)
        self.show_trajectory_btn.clicked.connect(self.toggle_show_trajectory)
        self.show_trajectory_btn.setStyleSheet("QPushButton { background-color: #4caf50; color: white; font-weight: bold; padding: 8px; }")
        display_control_layout.addWidget(self.show_trajectory_btn)
        
        camera_control_layout.addLayout(display_control_layout)
        layout.addWidget(camera_control_group)
        
        # 语音播报控制组
        voice_group = QGroupBox("语音播报控制")
        voice_layout = QVBoxLayout(voice_group)
        
        # 语音模式选择
        self.voice_mode_combo = QComboBox()
        self.voice_mode_combo.addItems(["简洁模式", "详细模式", "静默模式"])
        self.voice_mode_combo.setCurrentText("简洁模式")
        self.voice_mode_combo.currentTextChanged.connect(self.change_voice_mode)
        voice_layout.addWidget(QLabel("播报模式:"))
        voice_layout.addWidget(self.voice_mode_combo)
        
        # 语音播报开关
        self.voice_enabled_btn = QPushButton("🔊 语音播报: 开启")
        self.voice_enabled_btn.setCheckable(True)
        self.voice_enabled_btn.setChecked(True)
        self.voice_enabled_btn.clicked.connect(self.toggle_voice)
        self.voice_enabled_btn.setStyleSheet("QPushButton { background-color: #4caf50; color: white; font-weight: bold; }")
        voice_layout.addWidget(self.voice_enabled_btn)
        
        # 音量控制
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("音量:"))
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        self.volume_slider.valueChanged.connect(self.change_volume)
        volume_layout.addWidget(self.volume_slider)
        self.volume_label = QLabel("80%")
        volume_layout.addWidget(self.volume_label)
        voice_layout.addLayout(volume_layout)
        
        layout.addWidget(voice_group)
        
        # 环境检测控制组
        env_group = QGroupBox("环境检测控制")
        env_layout = QVBoxLayout(env_group)
        
        # 环境检测开关
        self.env_detection_btn = QPushButton("🌍 环境检测: 开启")
        self.env_detection_btn.setCheckable(True)
        self.env_detection_btn.setChecked(True)
        self.env_detection_btn.clicked.connect(self.toggle_environment_detection)
        self.env_detection_btn.setStyleSheet("QPushButton { background-color: #ff9800; color: white; font-weight: bold; padding: 10px; font-size: 14px; }")
        env_layout.addWidget(self.env_detection_btn)
        
        # 环境检测状态标志
        self.env_detection_enabled = True
        
        # 环境检测模式选择
        self.env_mode_combo = QComboBox()
        self.env_mode_combo.addItems(["标准模式", "高精度模式", "快速模式"])
        self.env_mode_combo.setCurrentText("标准模式")
        self.env_mode_combo.currentTextChanged.connect(self.change_env_mode)
        env_layout.addWidget(QLabel("检测模式:"))
        env_layout.addWidget(self.env_mode_combo)
        
        # 环境检测测试按钮
        self.env_test_btn = QPushButton("🧪 环境检测测试")
        self.env_test_btn.clicked.connect(self.test_environment_detection)
        self.env_test_btn.setStyleSheet("QPushButton { background-color: #9c27b0; color: white; font-weight: bold; padding: 8px; }")
        env_layout.addWidget(self.env_test_btn)
        
        # 语音播报环境按钮
        self.voice_env_btn = QPushButton("🔊 语音播报环境")
        self.voice_env_btn.clicked.connect(self.voice_announce_environment)
        self.voice_env_btn.setStyleSheet("QPushButton { background-color: #e91e63; color: white; font-weight: bold; padding: 8px; }")
        env_layout.addWidget(self.voice_env_btn)
        
        layout.addWidget(env_group)
        
        # 环境检测结果显示组
        env_result_group = QGroupBox("环境检测结果")
        env_result_layout = QVBoxLayout(env_result_group)
        
        # 环境安全状态
        self.env_safety_label = QLabel("环境安全: 未检测")
        self.env_safety_label.setStyleSheet("font-weight: bold; color: #666; padding: 8px; background-color: #f5f5f5; border-radius: 5px;")
        env_result_layout.addWidget(self.env_safety_label)
        
        # 安全评分
        self.env_score_label = QLabel("安全评分: --")
        self.env_score_label.setStyleSheet("font-weight: bold; color: #666; padding: 8px; background-color: #f5f5f5; border-radius: 5px;")
        env_result_layout.addWidget(self.env_score_label)
        
        # 环境详情
        self.env_details_text = QTextEdit()
        self.env_details_text.setMaximumHeight(120)
        self.env_details_text.setReadOnly(True)
        self.env_details_text.setPlaceholderText("环境检测详情将显示在这里...")
        self.env_details_text.setStyleSheet("background-color: #f9f9f9; border: 1px solid #ddd; padding: 5px; font-size: 12px;")
        env_result_layout.addWidget(QLabel("环境详情:"))
        env_result_layout.addWidget(self.env_details_text)
        
        # 警告信息
        self.env_warnings_text = QTextEdit()
        self.env_warnings_text.setMaximumHeight(80)
        self.env_warnings_text.setReadOnly(True)
        self.env_warnings_text.setPlaceholderText("警告信息将显示在这里...")
        self.env_warnings_text.setStyleSheet("background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 5px; font-size: 12px;")
        env_result_layout.addWidget(QLabel("警告信息:"))
        env_result_layout.addWidget(self.env_warnings_text)
        
        layout.addWidget(env_result_group)
        
        # 手机测试按钮
        self.mobile_test_btn = QPushButton("📱 启动手机测试")
        self.mobile_test_btn.clicked.connect(self.start_mobile_test)
        self.mobile_test_btn.setStyleSheet("QPushButton { background-color: #2196f3; color: white; font-weight: bold; font-size: 16px; padding: 12px; }")
        layout.addWidget(self.mobile_test_btn)
        
        # 语音播报状态显示
        self.voice_status_display = QLabel("语音播报: 已启用")
        self.voice_status_display.setStyleSheet("color: #4caf50; font-weight: bold; font-size: 14px; padding: 8px;")
        layout.addWidget(self.voice_status_display)
        
        layout.addStretch()
        return panel
    
    def create_camera_panel(self):
        """创建摄像头检测面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 摄像头控制按钮
        control_layout = QHBoxLayout()
        
        self.camera_start_btn = QPushButton("📹 开启摄像头检测")
        self.camera_start_btn.clicked.connect(self.toggle_camera_detection)
        self.camera_start_btn.setStyleSheet("QPushButton { background-color: #4caf50; color: white; font-weight: bold; padding: 12px; font-size: 16px; }")
        control_layout.addWidget(self.camera_start_btn)
        
        self.show_detection_btn = QPushButton("显示检测框")
        self.show_detection_btn.setCheckable(True)
        self.show_detection_btn.setChecked(True)
        self.show_detection_btn.clicked.connect(self.toggle_show_detection)
        self.show_detection_btn.setStyleSheet("QPushButton { background-color: #4caf50; color: white; font-weight: bold; padding: 8px; }")
        control_layout.addWidget(self.show_detection_btn)
        
        self.show_trajectory_btn = QPushButton("显示轨迹")
        self.show_trajectory_btn.setCheckable(True)
        self.show_trajectory_btn.setChecked(True)
        self.show_trajectory_btn.clicked.connect(self.toggle_show_trajectory)
        self.show_trajectory_btn.setStyleSheet("QPushButton { background-color: #4caf50; color: white; font-weight: bold; padding: 8px; }")
        control_layout.addWidget(self.show_trajectory_btn)
        
        control_layout.addStretch()
        
        # 检测状态显示
        self.camera_status_label = QLabel("摄像头状态: 未启动")
        self.camera_status_label.setStyleSheet("color: #666; padding: 8px; background-color: #f0f0f0; border-radius: 5px; font-size: 14px;")
        control_layout.addWidget(self.camera_status_label)
        
        layout.addLayout(control_layout)
        
        # 摄像头显示区域
        self.camera_display = QLabel("点击'开启摄像头检测'开始实时检测")
        self.camera_display.setMinimumSize(800, 600)
        self.camera_display.setStyleSheet("""
            QLabel {
                border: 2px solid #ddd;
                border-radius: 10px;
                background-color: #f9f9f9;
                color: #666;
                font-size: 16px;
                padding: 20px;
            }
        """)
        self.camera_display.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.camera_display)
        
        # 检测状态信息
        status_layout = QHBoxLayout()
        
        self.detection_status_label = QLabel("检测状态: 未启动")
        self.detection_status_label.setStyleSheet("color: #666; padding: 5px; background-color: #f0f0f0; border-radius: 3px;")
        status_layout.addWidget(self.detection_status_label)
        
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("color: #4caf50; padding: 5px; background-color: #e8f5e8; border-radius: 3px;")
        status_layout.addWidget(self.fps_label)
        
        self.detection_count_label = QLabel("检测数量: 0")
        self.detection_count_label.setStyleSheet("color: #2196f3; padding: 5px; background-color: #e3f2fd; border-radius: 3px;")
        status_layout.addWidget(self.detection_count_label)
        
        layout.addLayout(status_layout)
        
        return panel
    
    def create_analysis_panel(self):
        """创建轨迹预测分析面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 轨迹预测组
        trajectory_group = QGroupBox("轨迹预测")
        trajectory_layout = QVBoxLayout(trajectory_group)
        
        # 动态障碍物轨迹
        self.dynamic_objects_label = QLabel("动态障碍物轨迹:")
        self.dynamic_objects_label.setStyleSheet("font-weight: bold; color: #333;")
        trajectory_layout.addWidget(self.dynamic_objects_label)
        
        self.dynamic_objects_list = QTextEdit()
        self.dynamic_objects_list.setMaximumHeight(100)
        self.dynamic_objects_list.setReadOnly(True)
        self.dynamic_objects_list.setStyleSheet("background-color: #f5f5f5; border: 1px solid #ddd; padding: 5px;")
        trajectory_layout.addWidget(self.dynamic_objects_list)
        
        # 用户轨迹建议
        self.user_trajectory_label = QLabel("用户轨迹建议:")
        self.user_trajectory_label.setStyleSheet("font-weight: bold; color: #333;")
        trajectory_layout.addWidget(self.user_trajectory_label)
        
        self.user_trajectory_list = QTextEdit()
        self.user_trajectory_list.setMaximumHeight(100)
        self.user_trajectory_list.setReadOnly(True)
        self.user_trajectory_list.setStyleSheet("background-color: #f5f5f5; border: 1px solid #ddd; padding: 5px;")
        trajectory_layout.addWidget(self.user_trajectory_list)
        
        # 轨迹播报文本
        self.trajectory_voice_text = QTextEdit()
        self.trajectory_voice_text.setMaximumHeight(60)
        self.trajectory_voice_text.setReadOnly(True)
        self.trajectory_voice_text.setPlaceholderText("轨迹预测语音播报内容将显示在这里...")
        self.trajectory_voice_text.setStyleSheet("background-color: #d1ecf1; border: 1px solid #bee5eb; padding: 5px;")
        trajectory_layout.addWidget(QLabel("轨迹播报:"))
        trajectory_layout.addWidget(self.trajectory_voice_text)
        
        layout.addWidget(trajectory_group)
        
        # 语音播报组
        voice_group = QGroupBox("语音播报状态")
        voice_layout = QVBoxLayout(voice_group)
        
        self.voice_status_label = QLabel("播报状态: 待机")
        self.voice_status_label.setStyleSheet("font-weight: bold; color: #666;")
        voice_layout.addWidget(self.voice_status_label)
        
        self.last_voice_label = QLabel("最后播报: 无")
        self.last_voice_label.setStyleSheet("color: #666;")
        voice_layout.addWidget(self.last_voice_label)
        
        layout.addWidget(voice_group)
        
        layout.addStretch()
        return panel
    
    def create_status_bar(self):
        """创建状态栏"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # 系统状态
        self.system_status_label = QLabel("系统就绪")
        self.status_bar.addWidget(self.system_status_label)
        
        # 检测状态
        self.detection_status_bar_label = QLabel("检测: 未启动")
        self.status_bar.addWidget(self.detection_status_bar_label)
        
        # 语音播报状态
        self.voice_status_bar_label = QLabel("语音: 静默")
        self.status_bar.addWidget(self.voice_status_bar_label)
        
        # 环境风险评估
        self.env_risk_bar_label = QLabel("环境: 安全")
        self.status_bar.addWidget(self.env_risk_bar_label)
        
        # 轨迹预测状态
        self.trajectory_status_bar_label = QLabel("轨迹: 未预测")
        self.status_bar.addPermanentWidget(self.trajectory_status_bar_label)
        
        # 时间显示
        self.time_label = QLabel("")
        self.status_bar.addPermanentWidget(self.time_label)
        
        # 更新时间显示
        self.update_time_timer = QTimer()
        self.update_time_timer.timeout.connect(self.update_time_display)
        self.update_time_timer.start(1000)  # 每秒更新一次
        
    def create_display_panel(self):
        """创建显示面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 状态栏
        status_layout = QHBoxLayout()
        self.status_label = QLabel("请选择图像进行标注")
        self.status_label.setStyleSheet("font-size: 14px; color: #666; padding: 5px; background-color: #f0f0f0; border-radius: 3px;")
        status_layout.addWidget(self.status_label)
        
        self.mode_label = QLabel("当前模式: 两点模式")
        self.mode_label.setStyleSheet("font-size: 14px; color: #2196f3; padding: 5px; background-color: #e3f2fd; border-radius: 3px;")
        status_layout.addWidget(self.mode_label)
        
        layout.addLayout(status_layout)
        
        # 图像显示区域
        self.image_display = TwoPointImageLabel("请选择图像进行标注")
        self.image_display.setMinimumSize(900, 700)
        self.image_display.mouse_pressed.connect(self.on_mouse_pressed)
        self.image_display.mouse_moved.connect(self.on_mouse_moved)
        self.image_display.mouse_released.connect(self.on_mouse_released)
        layout.addWidget(self.image_display)
        
        return panel
        
    def set_drawing_mode(self, mode):
        """设置绘制模式"""
        self.drawing_mode = mode
        
        if mode == "two_point":
            self.two_point_mode_btn.setChecked(True)
            self.drag_mode_btn.setChecked(False)
            self.box_mode_btn.setChecked(False)
            self.mode_label.setText("当前模式: 两点模式")
            self.mode_label.setStyleSheet("font-size: 14px; color: #2196f3; padding: 5px; background-color: #e3f2fd; border-radius: 3px;")
            self.clear_temp_points()
        elif mode == "drag":
            self.two_point_mode_btn.setChecked(False)
            self.drag_mode_btn.setChecked(True)
            self.box_mode_btn.setChecked(False)
            self.mode_label.setText("当前模式: 拖拽模式")
            self.mode_label.setStyleSheet("font-size: 14px; color: #ff9800; padding: 5px; background-color: #fff3e0; border-radius: 3px;")
            self.clear_temp_points()
        elif mode == "box":
            self.two_point_mode_btn.setChecked(False)
            self.drag_mode_btn.setChecked(False)
            self.box_mode_btn.setChecked(True)
            self.mode_label.setText("当前模式: 框选障碍物")
            self.mode_label.setStyleSheet("font-size: 14px; color: #ff9800; padding: 5px; background-color: #fff3e0; border-radius: 3px;")
            self.clear_temp_points()
            
    def clear_temp_points(self):
        """清除临时点"""
        self.temp_points = []
        self.drag_start = None
        self.end_point = None
        self.box_start = None
        self.box_end = None
        self.update_display()
        
    def convert_display_to_image_coords(self, display_x, display_y):
        """将显示坐标转换为图像坐标"""
        # 优化坐标映射，减少漂移
        if not self.display_size or not self.original_size:
            return display_x, display_y
        disp_w, disp_h = self.display_size
        orig_w, orig_h = self.original_size
        # 计算缩放比例和偏移
        scale = min(disp_w / orig_w, disp_h / orig_h)
        pad_x = (disp_w - orig_w * scale) / 2
        pad_y = (disp_h - orig_h * scale) / 2
        x = int((display_x - pad_x) / scale)
        y = int((display_y - pad_y) / scale)
        x = max(0, min(orig_w - 1, x))
        y = max(0, min(orig_h - 1, y))
        return x, y
        
    def select_multiple_images(self):
        """选择多个图像文件"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择多个图像文件", "", "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff)"
        )
        
        if file_paths:
            self.image_files = sorted(file_paths)
            self.current_image_index = 0
            self.load_image_by_index(0)
            self.update_file_info()
            self.enable_navigation()
            
    def select_single_image(self):
        """选择单张图像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择单张图像", "", "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff)"
        )
        
        if file_path:
            self.image_files = [file_path]
            self.current_image_index = 0
            self.load_image_by_index(0)
            self.update_file_info()
            self.enable_navigation()
            
    def load_from_images_folder(self):
        """从images文件夹加载"""
        images_dir = "images"
        if not os.path.exists(images_dir):
            QMessageBox.warning(self, "警告", "images文件夹不存在")
            return
            
        # 查找所有图像文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        self.image_files = []
        
        for ext in image_extensions:
            self.image_files.extend(glob.glob(os.path.join(images_dir, ext)))
            self.image_files.extend(glob.glob(os.path.join(images_dir, ext.upper())))
        
        self.image_files.sort()
        
        if self.image_files:
            self.current_image_index = 0
            self.load_image_by_index(0)
            self.update_file_info()
            self.enable_navigation()
            QMessageBox.information(self, "成功", f"从images文件夹加载了 {len(self.image_files)} 张图像")
        else:
            QMessageBox.warning(self, "警告", "images文件夹中没有找到图像文件")
            
    def update_file_info(self):
        """更新文件信息"""
        if self.image_files:
            self.file_info_label.setText(f"已加载 {len(self.image_files)} 张图像")
            self.file_info_label.setStyleSheet("color: #4caf50; padding: 5px; background-color: #e8f5e8; border-radius: 3px;")
        else:
            self.file_info_label.setText("未选择文件")
            self.file_info_label.setStyleSheet("color: #666; padding: 5px; background-color: #f5f5f5; border-radius: 3px;")
            
    def enable_navigation(self):
        """启用导航按钮"""
        self.prev_btn.setEnabled(len(self.image_files) > 1)
        self.next_btn.setEnabled(len(self.image_files) > 1)
        
    def load_image_by_index(self, index):
        """根据索引加载图像"""
        if 0 <= index < len(self.image_files):
            self.current_image_index = index
            image_path = self.image_files[index]
            self.load_image_from_path(image_path)
            self.update_image_info()
            
    def load_image_from_path(self, image_path):
        """从路径加载图像"""
        if os.path.exists(image_path):
            self.current_image_path = image_path
            self.original_image = cv2.imread(image_path)
            
            # 记录原始图像尺寸
            self.original_size = (self.original_image.shape[1], self.original_image.shape[0])
            
            self.display_image = self.original_image.copy()
            self.annotations = []
            self.clear_temp_points()
            self.update_display()
            self.annotation_list.clear()
            
            # 更新窗口标题
            self.setWindowTitle(f'两点模式和拖拽模式标注工具 - {os.path.basename(image_path)}')
            self.status_label.setText(f"当前图像: {os.path.basename(image_path)}")
            
            # 尝试加载已有标注
            self.load_existing_annotations()
            
            # 自动进行环境检测和轨迹预测
            self.auto_analyze_image()
            
    def load_previous_image(self):
        """加载上一张图像"""
        if self.current_image_index > 0:
            self.load_image_by_index(self.current_image_index - 1)
            
    def load_next_image(self):
        """加载下一张图像"""
        if self.current_image_index < len(self.image_files) - 1:
            self.load_image_by_index(self.current_image_index + 1)
    
    def auto_analyze_image(self):
        """自动分析图像"""
        if self.original_image is None:
            return
        
        # 在后台线程中执行分析，避免界面卡顿
        import threading
        
        def analyze():
            try:
                # 环境检测
                env_result = self.analyze_environment(self.original_image)
                
                # 轨迹预测
                trajectory_result = self.predict_trajectory(self.original_image)
                
                print("✅ 图像分析完成")
            except Exception as e:
                print(f"❌ 图像分析失败: {e}")
        
        # 启动后台分析线程
        analysis_thread = threading.Thread(target=analyze)
        analysis_thread.daemon = True
        analysis_thread.start()
    
    def change_voice_mode(self, mode):
        """改变语音播报模式"""
        self.voice_mode = mode
        print(f"语音播报模式已切换为: {mode}")
        
        if mode == "静默模式":
            self.voice_enabled_btn.setText("🔇 语音播报: 关闭")
            self.voice_enabled_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; }")
        else:
            self.voice_enabled_btn.setText("🔊 语音播报: 开启")
            self.voice_enabled_btn.setStyleSheet("QPushButton { background-color: #4caf50; color: white; font-weight: bold; }")
        
        # 更新语音状态显示
        self.update_voice_status()
    
    def toggle_voice(self):
        """切换语音播报开关"""
        self.voice_enabled = not self.voice_enabled
        if self.voice_enabled:
            self.voice_enabled_btn.setText("🔊 语音播报: 开启")
            self.voice_enabled_btn.setStyleSheet("QPushButton { background-color: #4caf50; color: white; font-weight: bold; }")
        else:
            self.voice_enabled_btn.setText("🔇 语音播报: 关闭")
            self.voice_enabled_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; }")
        
        # 更新语音状态显示
        self.update_voice_status()
    
    def change_volume(self, value):
        """改变音量"""
        self.volume = value
        self.volume_label.setText(f"{value}%")
    
    def update_time_display(self):
        """更新时间显示"""
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.setText(current_time)
    
    def voice_announce(self, message, priority=1, category="info"):
        """语音播报 - 优化版本，支持连续完整播报"""
        if not self.voice_enabled or self.voice_mode == "静默模式":
            return
        
        current_time = time.time()
        
        # 检查冷却时间（根据优先级调整）
        cooldown_time = 0.5 if priority >= 4 else 1.0 if priority >= 2 else 2.0
        if current_time - self.last_voice_time < cooldown_time:
            return
        
        # 检查优先级
        if priority < self.current_voice_priority:
            return
        
        # 根据模式过滤消息
        if self.voice_mode == "简洁模式" and priority < 3:
            return
        
        # 播报消息
        print(f"🔊 语音播报 [{category}]: {message}")
        
        # 更新播报时间
        self.last_voice_time = current_time
        
        # 实际语音合成
        if self.voice_synthesis_available:
            try:
                # 创建语音播报任务
                self.schedule_voice_task(message, priority)
                
            except Exception as e:
                print(f"⚠️ 语音播报调度失败: {e}")
        
        # 更新界面
        self.last_voice_label.setText(f"最后播报: {message}")
        self.voice_status_label.setText(f"播报状态: {category}")
        
        # 更新状态栏
        self.voice_status_bar_label.setText(f"语音: {category}")
        if category == "紧急":
            self.voice_status_bar_label.setStyleSheet("color: #d32f2f; font-weight: bold;")
        elif category == "警告":
            self.voice_status_bar_label.setStyleSheet("color: #f57c00; font-weight: bold;")
        elif category == "检测":
            self.voice_status_bar_label.setStyleSheet("color: #2196f3; font-weight: bold;")
        else:
            self.voice_status_bar_label.setStyleSheet("color: #4caf50;")
    
    def schedule_voice_task(self, message, priority=1):
        """调度语音播报任务 - 支持队列管理和连续播报"""
        try:
            import time
            
            # 检查语音系统是否已初始化
            if not hasattr(self, 'voice_queue'):
                print("⚠️ 语音系统未初始化，跳过播报")
                return
            
            # 将语音任务加入队列（优先级越高，数字越小）
            self.voice_queue.put((priority, time.time(), message))
            
        except Exception as e:
            print(f"⚠️ 语音任务调度失败: {e}")
    
    def start_voice_worker(self):
        """启动语音工作线程"""
        def voice_worker():
            import queue
            while True:
                try:
                    # 从队列获取语音任务
                    priority, timestamp, message = self.voice_queue.get(timeout=1)
                    
                    with self.voice_lock:
                        if self.is_voice_playing:
                            # 如果正在播放，等待当前播放完成
                            continue
                        
                        self.is_voice_playing = True
                    
                    # 执行语音播报
                    self.execute_voice_playback(message)
                    
                    with self.voice_lock:
                        self.is_voice_playing = False
                    
                    # 标记任务完成
                    self.voice_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"⚠️ 语音工作线程错误: {e}")
                    with self.voice_lock:
                        self.is_voice_playing = False
        
        # 启动语音工作线程
        self.voice_thread = threading.Thread(target=voice_worker, daemon=True)
        self.voice_thread.start()
        print("✅ 语音工作线程已启动")
    
    def execute_voice_playback(self, message):
        """执行语音播报"""
        try:
            import pyttsx3
            
            # 创建或重用语音引擎
            if self.global_tts_engine is None:
                self.global_tts_engine = pyttsx3.init()
                
                # 设置语音参数
                self.global_tts_engine.setProperty('rate', 150)
                self.global_tts_engine.setProperty('volume', self.volume / 100.0)
                
                # 尝试设置中文语音
                voices = self.global_tts_engine.getProperty('voices')
                if voices:
                    for voice in voices:
                        voice_name = voice.name.lower()
                        voice_id = voice.id.lower()
                        if any(keyword in voice_name for keyword in ['chinese', 'zh', 'mandarin', '中文']):
                            self.global_tts_engine.setProperty('voice', voice.id)
                            print(f"✅ 使用中文语音: {voice.name}")
                            break
                
                # 设置语音完成回调
                def on_finish(utterance_id):
                    print(f"✅ 语音播报完成: {utterance_id}")
                
                def on_error(utterance_id):
                    print(f"❌ 语音播报错误: {utterance_id}")
                
                self.global_tts_engine.connect('finished-utterance', on_finish)
                self.global_tts_engine.connect('error', on_error)
            
            # 播报消息
            utterance_id = f"voice_{int(time.time() * 1000)}"
            self.global_tts_engine.say(message, utterance_id)
            self.global_tts_engine.runAndWait()
            
        except Exception as e:
            print(f"⚠️ 语音播报执行失败: {e}")
            # 如果全局引擎有问题，尝试重新创建
            try:
                if self.global_tts_engine:
                    self.global_tts_engine.stop()
                self.global_tts_engine = None
            except:
                pass
    
    def reset_voice_priority(self):
        """重置语音优先级"""
        self.current_voice_priority = 0
    
    def open_annotation_tool(self):
        """打开盲道标注工具窗口"""
        try:
            from core.annotation_window import AnnotationWindow
            self.annotation_window = AnnotationWindow(self)
            self.annotation_window.show()
        except ImportError:
            QMessageBox.information(self, "提示", "盲道标注工具功能正在开发中...")
    
    def open_training_interface(self):
        """打开模型训练界面"""
        try:
            from model_training_interface import ModelTrainingInterface
            self.training_window = ModelTrainingInterface()
            self.training_window.show()
        except ImportError:
            QMessageBox.information(self, "提示", "模型训练界面功能正在开发中...")
            
    def update_image_info(self):
        """更新图像信息"""
        if self.image_files:
            self.image_info_label.setText(f"{self.current_image_index + 1}/{len(self.image_files)}")
            
    def load_existing_annotations(self):
        """加载已有标注"""
        if not self.current_image_path:
            return
            
        image_name = os.path.basename(self.current_image_path)
        annotation_file = f"annotations/{image_name.replace('.', '_')}_annotations.json"
        
        if os.path.exists(annotation_file):
            try:
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.annotations = data.get('annotations', [])
                self.update_display()
                self.update_annotation_list()
                
                self.status_label.setText(f"已加载 {len(self.annotations)} 个标注")
            except Exception as e:
                print(f"加载标注失败: {e}")
                
    def on_mouse_pressed(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.LeftButton and self.current_image_path:
            if self.drawing_mode == "two_point":
                image_x, image_y = self.convert_display_to_image_coords(event.pos().x(), event.pos().y())
                self.temp_points.append((image_x, image_y))
                if len(self.temp_points) == 2:
                    annotation = {
                        'start': self.temp_points[0],
                        'end': self.temp_points[1],
                        'type': 'blind_path_line'
                    }
                    self.annotations.append(annotation)
                    self.temp_points = []
                    self.update_annotation_list()
                self.update_display()
            elif self.drawing_mode == "drag":
                image_x, image_y = self.convert_display_to_image_coords(event.pos().x(), event.pos().y())
                self.drawing = True
                self.drag_start = (image_x, image_y)
                self.end_point = event.pos()
            elif self.drawing_mode == "box":
                image_x, image_y = self.convert_display_to_image_coords(event.pos().x(), event.pos().y())
                self.drawing = True
                self.box_start = (image_x, image_y)
                self.box_end = (image_x, image_y)
        self.update_display()
                
    def on_mouse_moved(self, event):
        """鼠标移动事件"""
        if self.drawing and self.current_image_path:
            if self.drawing_mode == "drag":
                self.end_point = event.pos()
            elif self.drawing_mode == "box":
                image_x, image_y = self.convert_display_to_image_coords(event.pos().x(), event.pos().y())
                self.box_end = (image_x, image_y)
            self.update_display()
            
    def on_mouse_released(self, event):
        """鼠标释放事件"""
        if self.drawing and self.current_image_path and event.button() == Qt.LeftButton:
            if self.drawing_mode == "drag":
                image_x, image_y = self.convert_display_to_image_coords(event.pos().x(), event.pos().y())
                self.drawing = False
                if self.drag_start:
                    annotation = {
                        'start': self.drag_start,
                        'end': (image_x, image_y),
                        'type': 'blind_path_line'
                    }
                    self.annotations.append(annotation)
                    self.update_annotation_list()
                    self.drag_start = None
                    self.end_point = None
                self.update_display()
            elif self.drawing_mode == "box":
                image_x, image_y = self.convert_display_to_image_coords(event.pos().x(), event.pos().y())
                self.drawing = False
                if self.box_start:
                    x_min, y_min = self.box_start
                    x_max, y_max = image_x, image_y
                    x_min, x_max = min(x_min, x_max), max(x_min, x_max)
                    y_min, y_max = min(y_min, y_max), max(y_min, y_max)
                    annotation = {
                        'type': 'obstacle_box',
                        'box': (x_min, y_min, x_max, y_max)
                    }
                    self.annotations.append(annotation)
                    self.update_annotation_list()
                    self.box_start = None
                    self.box_end = None
                self.update_display()
            
    def update_display(self):
        """更新显示"""
        if not hasattr(self, 'original_image'):
            return
            
        # 复制原图
        self.display_image = self.original_image.copy()
        
        # 绘制临时点（两点模式）
        for i, point in enumerate(self.temp_points):
            x, y = point
            cv2.circle(self.display_image, (x, y), self.point_size, (255, 0, 0), -1)
            cv2.putText(self.display_image, f"P{i+1}", (x+5, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # 如果有两个点，显示连接线
            if len(self.temp_points) == 2 and i == 0:
                next_point = self.temp_points[1]
                cv2.line(self.display_image, (x, y), next_point, (255, 0, 0), 2)
        
        # 绘制拖拽线（拖拽模式）
        if self.drawing and self.drag_start and self.end_point:
            # 转换终点坐标
            end_x, end_y = self.convert_display_to_image_coords(self.end_point.x(), self.end_point.y())
            cv2.line(self.display_image, 
                    self.drag_start,
                    (end_x, end_y),
                    (255, 0, 0), 2)
        
        # 绘制障碍物框
        for i, annotation in enumerate(self.annotations):
            if annotation['type'] == 'obstacle_box':
                x_min, y_min, x_max, y_max = annotation['box']
                cv2.rectangle(self.display_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), self.line_width)
                cv2.putText(self.display_image, f"B{i+1}", (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        # 绘制正在拖拽的障碍物框
        if self.drawing and self.drawing_mode == "box" and self.box_start and self.box_end:
            x1, y1 = self.box_start
            x2, y2 = self.box_end
            cv2.rectangle(self.display_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # 绘制已保存的线段标注
        for i, annotation in enumerate(self.annotations):
            if annotation['type'] == 'blind_path_line':
                start = annotation['start']
                end = annotation['end']
                cv2.line(self.display_image, start, end, (0, 255, 0), self.line_width)
                cv2.putText(self.display_image, f"L{i+1}", start, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 显示图像
        height, width, channel = self.display_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(self.display_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        scaled_pixmap = pixmap.scaled(
            self.image_display.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        # 记录显示尺寸
        self.display_size = (scaled_pixmap.width(), scaled_pixmap.height())
        
        self.image_display.setPixmap(scaled_pixmap)
        self.image_display.setStyleSheet("border: none;")
        
    def update_annotation_list(self):
        """更新标注列表"""
        self.annotation_list.clear()
        
        # 添加线段
        for i, annotation in enumerate(self.annotations):
            if annotation['type'] == 'blind_path_line':
                start = annotation['start']
                end = annotation['end']
                self.annotation_list.addItem(f"盲道线段 {i+1}: ({start[0]},{start[1]}) -> ({end[0]},{end[1]})")
            elif annotation['type'] == 'obstacle_box':
                x_min, y_min, x_max, y_max = annotation['box']
                self.annotation_list.addItem(f"障碍物框 {i+1}: ({x_min},{y_min}) -> ({x_max},{y_max})")
                
    def save_annotations(self):
        """保存标注"""
        if not self.current_image_path:
            QMessageBox.warning(self, "警告", "没有加载图像")
            return
            
        # 保存标注数据
        image_name = os.path.basename(self.current_image_path)
        annotation_data = {
            'image_path': self.current_image_path,
            'annotations': self.annotations,
            'timestamp': time.time()
        }
        
        # 创建标注目录
        os.makedirs('annotations', exist_ok=True)
        
        # 保存为JSON文件
        annotation_file = f"annotations/{image_name.replace('.', '_')}_annotations.json"
        with open(annotation_file, 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, ensure_ascii=False, indent=2)
            
        QMessageBox.information(self, "成功", f"标注已保存到: {annotation_file}")
        
        # 如果有下一张图像，询问是否跳转
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            reply = QMessageBox.question(self, "继续标注", 
                                       "是否跳转到下一张图像继续标注？",
                                       QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.load_next_image()
                
        # 保存为YOLO格式txt（只保存障碍物框，类别1）
        yolo_label_dir = r"yolo_dataset/labels"
        os.makedirs(yolo_label_dir, exist_ok=True)
        txt_name = os.path.splitext(os.path.basename(self.current_image_path))[0] + ".txt"
        txt_path = os.path.join(yolo_label_dir, txt_name)
        img = cv2.imread(self.current_image_path)
        h, w = img.shape[:2]
        with open(txt_path, 'w') as f:
            for ann in self.annotations:
                if ann['type'] == 'obstacle_box':
                    x_min, y_min, x_max, y_max = ann['box']
                    # 转YOLO格式
                    x_center = (x_min + x_max) / 2 / w
                    y_center = (y_min + y_max) / 2 / h
                    bw = (x_max - x_min) / w
                    bh = (y_max - y_min) / h
                    f.write(f"1 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")
                
    def clear_annotations(self):
        """清除当前标注"""
        self.annotations = []
        self.clear_temp_points()
        self.annotation_list.clear()
        if hasattr(self, 'original_image'):
            self.update_display()
            
    def delete_selected_annotation(self):
        """删除选中的标注"""
        current_row = self.annotation_list.currentRow()
        if current_row >= 0 and current_row < len(self.annotations):
            del self.annotations[current_row]
            self.update_display()
            self.update_annotation_list()

    def undo_last_annotation(self):
        if self.annotations:
            self.annotations.pop()
            self.update_annotation_list()
            self.update_display()

    def analyze_environment(self, frame):
        """分析环境"""
        if not self.env_detector:
            return None
        
        try:
            # 获取当前帧的检测结果
            detections = self.detect_objects_in_frame(frame)
            
            # 转换检测格式
            detection_objects = []
            for detection in detections:
                if 'bbox' in detection:
                    detection_objects.append({
                        'bbox': detection['bbox'],
                        'confidence': detection.get('confidence', 0.8),
                        'class': detection.get('class_id', 0)
                    })
            
            # 执行环境检测
            env_result = self.env_detector.detect_environment(frame, detection_objects)
            
            # 实时语音播报环境信息
            if env_result and self.voice_enabled:
                self.real_time_environment_voice_announce(env_result, detections)
            
            return env_result
        except Exception as e:
            print(f"环境检测失败: {e}")
            return None
    
    def predict_trajectory(self, frame):
        """预测轨迹"""
        if not self.trajectory_predictor:
            return None
        
        try:
            # 转换检测格式
            detections = []
            for annotation in self.annotations:
                if 'bbox' in annotation:
                    bbox = annotation['bbox']
                    detections.append([
                        bbox[0], bbox[1], bbox[2], bbox[3],  # x1, y1, x2, y2
                        0.8,  # confidence
                        0     # class
                    ])
            
            # 执行轨迹预测
            result = self.trajectory_predictor.process_frame(frame, detections)
            
            # 更新界面显示
            self.update_trajectory_display(result)
            
            return result
        except Exception as e:
            print(f"轨迹预测失败: {e}")
            return None
    
    def update_environment_display(self, env_result, detections):
        """更新环境检测显示"""
        if not env_result:
            return
        
        # 更新环境安全状态
        overall_safety = env_result.get('overall_safety_level', 'safe')
        safety_score = env_result.get('safety_score', 1.0)
        safety_percentage = int(safety_score * 100)
        
        # 环境安全状态显示
        if overall_safety == 'high_risk':
            self.env_safety_label.setText(f"环境安全: 高风险")
            self.env_safety_label.setStyleSheet("font-weight: bold; color: #d32f2f; padding: 8px; background-color: #ffebee; border-radius: 5px;")
        elif overall_safety == 'medium_risk':
            self.env_safety_label.setText(f"环境安全: 中等风险")
            self.env_safety_label.setStyleSheet("font-weight: bold; color: #f57c00; padding: 8px; background-color: #fff3e0; border-radius: 5px;")
        else:
            self.env_safety_label.setText(f"环境安全: 安全")
            self.env_safety_label.setStyleSheet("font-weight: bold; color: #4caf50; padding: 8px; background-color: #e8f5e8; border-radius: 5px;")
        
        # 更新安全评分
        self.env_score_label.setText(f"安全评分: {safety_percentage}%")
        if safety_percentage >= 80:
            self.env_score_label.setStyleSheet("font-weight: bold; color: #4caf50; padding: 8px; background-color: #e8f5e8; border-radius: 5px;")
        elif safety_percentage >= 60:
            self.env_score_label.setStyleSheet("font-weight: bold; color: #f57c00; padding: 8px; background-color: #fff3e0; border-radius: 5px;")
        else:
            self.env_score_label.setStyleSheet("font-weight: bold; color: #d32f2f; padding: 8px; background-color: #ffebee; border-radius: 5px;")
        
        # 显示详细环境信息
        env_details = self.get_environment_details(env_result, detections)
        if env_details:
            self.env_details_text.setPlainText(env_details)
        else:
            self.env_details_text.setPlainText("暂无环境检测详情")
        
        # 更新警告信息
        warnings = env_result.get('warnings', [])
        emergency_alerts = env_result.get('emergency_alerts', [])
        
        warning_text = ""
        if emergency_alerts:
            warning_text += "🚨 紧急警报:\n"
            for alert in emergency_alerts[:3]:
                warning_text += f"• {alert}\n"
        
        if warnings:
            if warning_text:
                warning_text += "\n"
            warning_text += "⚠️ 警告信息:\n"
            for warning in warnings[:3]:
                warning_text += f"• {warning}\n"
        
        if warning_text:
            self.env_warnings_text.setPlainText(warning_text)
        else:
            self.env_warnings_text.setPlainText("暂无警告信息")
        
        # 生成环境播报文本
        env_voice_content = self.generate_environment_voice_content(detections, env_result)
        
        # 语音播报
        if env_voice_content:
            priority = 5 if overall_safety == 'high_risk' else 3 if overall_safety == 'medium_risk' else 2
            category = "紧急" if overall_safety == 'high_risk' else "警告" if overall_safety == 'medium_risk' else "检测"
            self.voice_announce(env_voice_content, priority=priority, category=category)
        
        # 更新状态栏环境风险评估
        if overall_safety == 'high_risk':
            self.env_risk_bar_label.setText("环境: 高风险")
            self.env_risk_bar_label.setStyleSheet("color: #d32f2f; font-weight: bold;")
        elif overall_safety == 'medium_risk':
            self.env_risk_bar_label.setText("环境: 中等风险")
            self.env_risk_bar_label.setStyleSheet("color: #f57c00; font-weight: bold;")
        else:
            self.env_risk_bar_label.setText("环境: 安全")
    
    def get_environment_details(self, env_result, detections=None):
        """获取环境检测详细信息"""
        details = []
        
        # 检测到的环境事物
        if detections:
            details.append("🔍 检测到的环境事物:")
            for i, detection in enumerate(detections, 1):
                obj_type = detection.get('class_name', '未知物体')
                confidence = detection.get('confidence', 0)
                bbox = detection.get('bbox', [0, 0, 0, 0])
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                
                # 判断物体位置
                position = self.get_object_position(x1, y1, x2, y2)
                
                details.append(f"  {i}. {obj_type} ({position}, 置信度: {confidence:.2f}, 大小: {width}x{height})")
            
            if not detections:
                details.append("  暂无检测到的环境事物")
        else:
            details.append("🔍 检测到的环境事物: 暂无")
        
        # 天气条件
        weather_info = env_result.get('weather_conditions')
        if weather_info:
            weather_type = weather_info.get('weather_type', 'clear')
            visibility = weather_info.get('visibility_level', 'good')
            safety_impact = weather_info.get('safety_impact', 'low')
            details.append(f"\n🌤️ 天气条件: {weather_type} (能见度: {visibility}, 影响: {safety_impact})")
        
        # 光照条件
        lighting_info = env_result.get('lighting_conditions')
        if lighting_info:
            lighting_level = lighting_info.get('lighting_level', 'normal')
            visibility_quality = lighting_info.get('visibility_quality', 'good')
            safety_impact = lighting_info.get('safety_impact', 'low')
            details.append(f"💡 光照条件: {lighting_level} (质量: {visibility_quality}, 影响: {safety_impact})")
        
        # 路面条件
        surface_info = env_result.get('surface_conditions')
        if surface_info:
            surface_type = surface_info.get('surface_type', 'smooth')
            safety_level = surface_info.get('safety_level', 'safe')
            walking_difficulty = surface_info.get('walking_difficulty', 'easy')
            details.append(f"🛣️ 路面条件: {surface_type} (安全: {safety_level}, 难度: {walking_difficulty})")
        
        # 施工区域
        construction_info = env_result.get('construction_zone')
        if construction_info and construction_info.get('is_construction_zone'):
            zone_type = construction_info.get('zone_type', 'unknown')
            safety_level = construction_info.get('safety_level', 'safe')
            confidence = construction_info.get('confidence', 0)
            details.append(f"🚧 施工区域: {zone_type} (安全: {safety_level}, 置信度: {confidence:.2f})")
        
        # 十字路口
        intersection_info = env_result.get('intersection')
        if intersection_info and intersection_info.get('is_intersection'):
            traffic_light = intersection_info.get('traffic_light_state', 'unknown')
            crosswalk = intersection_info.get('crosswalk_detected', False)
            details.append(f"🚦 十字路口: 交通灯({traffic_light}), 斑马线({'是' if crosswalk else '否'})")
        
        # 拥挤程度
        crowd_info = env_result.get('crowd_density')
        if crowd_info:
            density_level = crowd_info.get('density_level', 'low')
            navigation_difficulty = crowd_info.get('navigation_difficulty', 'easy')
            details.append(f"👥 拥挤程度: {density_level} (导航难度: {navigation_difficulty})")
        
        return "\n".join(details) if details else "暂无环境检测详情"
    
    def generate_environment_voice_content(self, detections, env_result):
        """生成环境检测语音播报内容"""
        content_parts = []
        
        # 检测到的物体播报
        if detections:
            dynamic_objects = [d for d in detections if d.get('class_id') == 1]  # 动态障碍
            static_objects = [d for d in detections if d.get('class_id') == 0]  # 静态障碍
            ground_hazards = [d for d in detections if d.get('class_id') == 2]  # 地面危险
            
            if dynamic_objects:
                for obj in dynamic_objects:
                    obj_name = obj.get('class_name', '物体')
                    bbox = obj.get('bbox', [0, 0, 0, 0])
                    x1, y1, x2, y2 = bbox
                    position = self.get_object_position(x1, y1, x2, y2)
                    content_parts.append(f"前方{position}检测到动态障碍物{obj_name}")
            
            if ground_hazards:
                for obj in ground_hazards:
                    obj_name = obj.get('class_name', '物体')
                    bbox = obj.get('bbox', [0, 0, 0, 0])
                    x1, y1, x2, y2 = bbox
                    position = self.get_object_position(x1, y1, x2, y2)
                    content_parts.append(f"前方{position}检测到地面危险{obj_name}")
            
            if static_objects:
                for obj in static_objects:
                    obj_name = obj.get('class_name', '物体')
                    bbox = obj.get('bbox', [0, 0, 0, 0])
                    x1, y1, x2, y2 = bbox
                    position = self.get_object_position(x1, y1, x2, y2)
                    content_parts.append(f"前方{position}检测到静态障碍物{obj_name}")
        
        # 环境风险播报
        overall_safety = env_result.get('overall_safety_level', 'safe')
        safety_score = env_result.get('safety_score', 1.0)
        safety_percentage = int(safety_score * 100)
        
        if overall_safety == 'high_risk':
            content_parts.append(f"环境高风险，安全评分{safety_percentage}%，请小心")
        elif overall_safety == 'medium_risk':
            content_parts.append(f"环境中等风险，安全评分{safety_percentage}%，请注意")
        else:
            content_parts.append(f"环境安全，安全评分{safety_percentage}%")
        
        # 天气条件播报
        weather_info = env_result.get('weather_conditions')
        if weather_info:
            weather_type = weather_info.get('weather_type', 'clear')
            if weather_type == 'rain':
                content_parts.append("检测到雨天，路面可能湿滑")
            elif weather_type == 'fog':
                content_parts.append("检测到雾天，能见度较低")
            elif weather_type == 'snow':
                content_parts.append("检测到雪天，路面可能结冰")
        
        # 光照条件播报
        lighting_info = env_result.get('lighting_conditions')
        if lighting_info:
            lighting_level = lighting_info.get('lighting_level', 'normal')
            if lighting_level == 'very_dark':
                content_parts.append("环境很暗，请小心行走")
            elif lighting_level == 'dark':
                content_parts.append("环境较暗，请注意安全")
            elif lighting_level == 'very_bright':
                content_parts.append("环境很亮，可能影响视线")
        
        # 路面条件播报
        surface_info = env_result.get('surface_conditions')
        if surface_info:
            surface_type = surface_info.get('surface_type', 'smooth')
            if surface_type == 'wet':
                content_parts.append("路面湿滑，请小心行走")
            elif surface_type == 'uneven':
                content_parts.append("路面不平，请注意脚下")
            elif surface_type == 'rough':
                content_parts.append("路面粗糙，行走时请注意")
        
        # 施工区域播报
        construction_info = env_result.get('construction_zone')
        if construction_info and construction_info.get('is_construction_zone'):
            content_parts.append("前方施工区域，请绕行")
        
        # 十字路口播报
        intersection_info = env_result.get('intersection')
        if intersection_info and intersection_info.get('is_intersection'):
            content_parts.append("前方十字路口，请注意交通信号")
        
        # 拥挤程度播报
        crowd_info = env_result.get('crowd_density')
        if crowd_info:
            density_level = crowd_info.get('density_level', 'low')
            if density_level == 'high':
                content_parts.append("人群拥挤，请小心避让")
            elif density_level == 'medium':
                content_parts.append("人群较多，请注意安全")
        
        # 紧急警报播报
        emergency_alerts = env_result.get('emergency_alerts', [])
        if emergency_alerts:
            content_parts.append("紧急警报：" + "；".join(emergency_alerts[:2]))  # 最多播报2个紧急警报
        
        # 环境建议播报
        guidance = env_result.get('navigation_guidance', [])
        if guidance:
            content_parts.extend(guidance[:2])  # 最多播报2个建议
        
        return "；".join(content_parts) if content_parts else "环境安全"
    
    def get_object_position(self, x1, y1, x2, y2):
        """获取物体位置描述"""
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        if center_x < 200:
            return "左侧"
        elif center_x > 440:
            return "右侧"
        elif center_y < 150:
            return "上方"
        elif center_y > 330:
            return "下方"
        else:
            return "中央"
    
    def format_environment_details(self, env_result):
        """格式化环境详情"""
        details = []
        
        # 施工区域信息
        construction = env_result.get('construction_zone')
        if construction and construction.get('is_construction_zone'):
            details.append(f"施工区域: {construction.get('zone_type', 'unknown')}")
            details.append(f"安全等级: {construction.get('safety_level', 'unknown')}")
            if construction.get('bypass_path'):
                details.append("建议绕行")
        
        # 路口信息
        intersection = env_result.get('intersection')
        if intersection and intersection.get('is_intersection'):
            details.append(f"路口检测: 是")
            if intersection.get('traffic_light_state') != 'unknown':
                details.append(f"交通灯: {intersection.get('traffic_light_state')}")
            if intersection.get('crosswalk_detected'):
                details.append("斑马线: 检测到")
        
        # 拥挤程度信息
        crowd = env_result.get('crowd_density')
        if crowd:
            details.append(f"拥挤程度: {crowd.get('density_level', 'unknown')}")
            details.append(f"导航难度: {crowd.get('navigation_difficulty', 'unknown')}")
        
        return '\n'.join(details) if details else "环境检测正常"
    
    def update_trajectory_display(self, result):
        """更新轨迹预测显示"""
        if not result:
            return
        
        # 更新轨迹状态
        if result.get('blind_path'):
            blind_path = result['blind_path']
            confidence = blind_path.get('confidence', 0)
            if confidence > 0.7:
                self.trajectory_status_label.setText("轨迹状态: 盲道清晰")
                self.trajectory_status_label.setStyleSheet("color: #4caf50; padding: 8px; background-color: #e8f5e8; border-radius: 5px; font-size: 14px;")
                self.voice_announce("盲道清晰，建议沿盲道前进", priority=1, category="指导")
            elif confidence > 0.4:
                self.trajectory_status_label.setText("轨迹状态: 盲道部分可见")
                self.trajectory_status_label.setStyleSheet("color: #ff9800; padding: 8px; background-color: #fff3e0; border-radius: 5px; font-size: 14px;")
                self.voice_announce("盲道部分可见，请谨慎前进", priority=2, category="警告")
            else:
                self.trajectory_status_label.setText("轨迹状态: 盲道不清晰")
                self.trajectory_status_label.setStyleSheet("color: #f44336; padding: 8px; background-color: #ffebee; border-radius: 5px; font-size: 14px;")
                self.voice_announce("盲道不清晰，建议使用其他导航方式", priority=3, category="警告")
        else:
            self.trajectory_status_label.setText("轨迹状态: 未检测到盲道")
            self.trajectory_status_label.setStyleSheet("color: #666; padding: 8px; background-color: #f5f5f5; border-radius: 5px; font-size: 14px;")
        
        # 更新碰撞风险
        collision_risks = result.get('collision_risks', {})
        if collision_risks:
            max_risk = max(collision_risks.values()) if collision_risks else 0
            if max_risk > 0.7:
                self.collision_risk_label.setText("碰撞风险: 高")
                self.collision_risk_label.setStyleSheet("color: #d32f2f; padding: 8px; background-color: #ffebee; border-radius: 5px; font-size: 14px;")
                self.voice_announce("检测到高风险障碍物，请立即停止", priority=5, category="紧急")
            elif max_risk > 0.4:
                self.collision_risk_label.setText("碰撞风险: 中")
                self.collision_risk_label.setStyleSheet("color: #f57c00; padding: 8px; background-color: #fff3e0; border-radius: 5px; font-size: 14px;")
                self.voice_announce("存在中等风险障碍物，请减速慢行", priority=3, category="警告")
            else:
                self.collision_risk_label.setText("碰撞风险: 低")
                self.collision_risk_label.setStyleSheet("color: #4caf50; padding: 8px; background-color: #e8f5e8; border-radius: 5px; font-size: 14px;")
        else:
            self.collision_risk_label.setText("碰撞风险: 低")
            self.collision_risk_label.setStyleSheet("color: #4caf50; padding: 8px; background-color: #e8f5e8; border-radius: 5px; font-size: 14px;")
        
        # 更新安全建议
        recommendations = result.get('safety_recommendations', [])
        if recommendations:
            self.safety_recommendations_label.setText(f"安全建议: {', '.join(recommendations)}")
            self.safety_recommendations_label.setStyleSheet("color: #ff9800; padding: 8px; background-color: #fff3e0; border-radius: 5px; font-size: 14px;")
            # 播报安全建议
            for rec in recommendations:
                self.voice_announce(rec, priority=2, category="建议")
        else:
            self.safety_recommendations_label.setText("安全建议: 无")
            self.safety_recommendations_label.setStyleSheet("color: #666; padding: 8px; background-color: #f5f5f5; border-radius: 5px; font-size: 14px;")
        
        # 更新轨迹详情
        trajectory_details = self.format_trajectory_details(result)
        self.trajectory_details_text.setPlainText(trajectory_details)
        
        # 更新状态栏轨迹预测状态
        if result.get('blind_path'):
            confidence = result['blind_path'].get('confidence', 0)
            if confidence > 0.7:
                self.trajectory_status_bar_label.setText("轨迹: 清晰")
                self.trajectory_status_bar_label.setStyleSheet("color: #4caf50;")
            elif confidence > 0.4:
                self.trajectory_status_bar_label.setText("轨迹: 模糊")
                self.trajectory_status_bar_label.setStyleSheet("color: #f57c00;")
            else:
                self.trajectory_status_bar_label.setText("轨迹: 不清晰")
                self.trajectory_status_bar_label.setStyleSheet("color: #d32f2f;")
        else:
            self.trajectory_status_bar_label.setText("轨迹: 未检测")
            self.trajectory_status_bar_label.setStyleSheet("color: #666;")
    
    def format_trajectory_details(self, result):
        """格式化轨迹详情"""
        details = []
        
        # 盲道信息
        blind_path = result.get('blind_path')
        if blind_path:
            details.append(f"盲道置信度: {blind_path.get('confidence', 0):.2f}")
            if blind_path.get('predicted_trajectory'):
                details.append(f"预测轨迹点: {len(blind_path['predicted_trajectory'])}个")
        
        # 跟踪目标信息
        tracked_objects = result.get('tracked_objects', [])
        if tracked_objects:
            details.append(f"跟踪目标: {len(tracked_objects)}个")
            for obj in tracked_objects:
                obj_id = obj.get('id', 'unknown')
                class_id = obj.get('class_id', 0)
                details.append(f"  目标{obj_id}: 类别{class_id}")
        
        # 碰撞风险详情
        collision_risks = result.get('collision_risks', {})
        if collision_risks:
            high_risk_count = sum(1 for risk in collision_risks.values() if risk > 0.7)
            if high_risk_count > 0:
                details.append(f"高风险目标: {high_risk_count}个")
        
        return '\n'.join(details) if details else "轨迹预测正常"
    
    def toggle_camera_detection(self):
        """切换摄像头检测状态"""
        if not self.camera_active:
            self.start_camera_detection()
        else:
            self.stop_camera_detection()
    
    def start_camera_detection(self):
        """开始摄像头检测"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.warning(self, "错误", "无法打开摄像头，请检查摄像头权限")
                return
            
            self.camera_active = True
            self.camera_start_btn.setText("📹 停止摄像头检测")
            self.camera_start_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 12px; font-size: 16px; }")
            self.camera_status_label.setText("摄像头状态: 运行中")
            self.camera_status_label.setStyleSheet("color: #4caf50; padding: 8px; background-color: #e8f5e8; border-radius: 5px; font-size: 14px;")
            self.detection_status_label.setText("检测状态: 检测中")
            self.detection_status_label.setStyleSheet("color: #4caf50; padding: 5px; background-color: #e8f5e8; border-radius: 3px;")
            
            # 启动定时器
            self.camera_timer.start(33)  # 约30FPS
            
            # 更新状态栏
            self.detection_status_bar_label.setText("检测: 运行中")
            self.detection_status_bar_label.setStyleSheet("color: #4caf50; font-weight: bold;")
            
            # 语音播报
            self.voice_announce("摄像头检测已启动", priority=1, category="系统")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动摄像头检测失败: {e}")
    
    def stop_camera_detection(self):
        """停止摄像头检测"""
        self.camera_active = False
        self.camera_timer.stop()
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.camera_start_btn.setText("📹 开启摄像头检测")
        self.camera_start_btn.setStyleSheet("QPushButton { background-color: #4caf50; color: white; font-weight: bold; padding: 12px; font-size: 16px; }")
        self.camera_status_label.setText("摄像头状态: 已停止")
        self.camera_status_label.setStyleSheet("color: #666; padding: 8px; background-color: #f0f0f0; border-radius: 5px; font-size: 14px;")
        self.detection_status_label.setText("检测状态: 已停止")
        self.detection_status_label.setStyleSheet("color: #666; padding: 5px; background-color: #f0f0f0; border-radius: 3px;")
        
        self.camera_display.setText("摄像头检测已停止")
        
        # 更新状态栏
        self.detection_status_bar_label.setText("检测: 已停止")
        self.detection_status_bar_label.setStyleSheet("color: #666;")
        
        # 语音播报
        self.voice_announce("摄像头检测已停止", priority=1, category="系统")
    
    def update_camera_frame(self):
        """更新摄像头帧"""
        if not self.cap or not self.camera_active:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # 进行物体检测
        detections = self.detect_objects_in_frame(frame)
        
        # 环境检测和轨迹预测
        env_result = None
        trajectory_result = None
        
        # 实时环境检测
        if self.env_detector and self.env_detection_enabled:
            env_result = self.analyze_environment(frame)
        
        if self.trajectory_predictor:
            trajectory_result = self.predict_trajectory(frame)
        
        # 绘制检测结果
        if self.show_detection_btn.isChecked():
            frame = self.draw_detection_boxes(frame, detections)
        
        if self.show_trajectory_btn.isChecked() and trajectory_result:
            frame = self.draw_trajectory_predictions(frame, trajectory_result)
        
        # 显示帧
        self.display_camera_frame(frame)
        
        # 更新检测信息
        self.update_detection_info(detections, env_result, trajectory_result)
        
        # 检测障碍物变化
        detection_changes = self.detect_obstacle_changes(detections)
        
        # 更新环境检测和轨迹预测显示
        if env_result:
            self.update_environment_display(env_result, detections)
        else:
            # 如果没有环境检测结果，显示默认状态
            self.env_safety_label.setText("环境安全: 检测中...")
            self.env_safety_label.setStyleSheet("font-weight: bold; color: #666; padding: 8px; background-color: #f5f5f5; border-radius: 5px;")
            self.env_score_label.setText("安全评分: --")
            self.env_score_label.setStyleSheet("font-weight: bold; color: #666; padding: 8px; background-color: #f5f5f5; border-radius: 5px;")
            self.env_details_text.setPlainText("环境检测中，请稍候...")
            self.env_warnings_text.setPlainText("暂无警告信息")
        
        if trajectory_result:
            self.update_trajectory_display(trajectory_result, detections)
        
        # 根据变化进行语音播报
        if detection_changes:
            self.announce_detection_changes(detection_changes)
        
        # 更新上一帧检测结果
        self.previous_detections = detections.copy()
    
    def detect_obstacle_changes(self, current_detections):
        """检测障碍物变化"""
        changes = {
            'new_obstacles': [],      # 新出现的障碍物
            'disappeared_obstacles': [],  # 消失的障碍物
            'moved_obstacles': [],    # 移动的障碍物
            'changed_obstacles': []   # 变化的障碍物
        }
        
        current_time = time.time()
        
        # 检查冷却时间
        if current_time - self.last_announcement_time < self.announcement_cooldown:
            return changes
        
        # 如果没有上一帧数据，跳过变化检测
        if not self.previous_detections:
            return changes
        
        # 检测新出现的障碍物
        for current_obj in current_detections:
            is_new = True
            for prev_obj in self.previous_detections:
                if self.is_same_object(current_obj, prev_obj):
                    is_new = False
                    break
            if is_new:
                changes['new_obstacles'].append(current_obj)
        
        # 检测消失的障碍物
        for prev_obj in self.previous_detections:
            still_exists = False
            for current_obj in current_detections:
                if self.is_same_object(prev_obj, current_obj):
                    still_exists = True
                    break
            if not still_exists:
                changes['disappeared_obstacles'].append(prev_obj)
        
        # 检测移动的障碍物
        for current_obj in current_detections:
            for prev_obj in self.previous_detections:
                if self.is_same_object(current_obj, prev_obj):
                    if self.has_moved_significantly(current_obj, prev_obj):
                        changes['moved_obstacles'].append({
                            'previous': prev_obj,
                            'current': current_obj
                        })
                    break
        
        # 如果有任何变化，更新播报时间
        if any(changes.values()):
            self.last_announcement_time = current_time
        
        return changes
    
    def is_same_object(self, obj1, obj2):
        """判断是否为同一个物体"""
        # 基于类别和位置判断
        if obj1.get('class_name') != obj2.get('class_name'):
            return False
        
        # 计算边界框重叠度
        bbox1 = obj1.get('bbox', [0, 0, 0, 0])
        bbox2 = obj2.get('bbox', [0, 0, 0, 0])
        
        overlap = self.calculate_bbox_overlap(bbox1, bbox2)
        return overlap > 0.3  # 重叠度阈值
    
    def calculate_bbox_overlap(self, bbox1, bbox2):
        """计算两个边界框的重叠度"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 计算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def has_moved_significantly(self, current_obj, prev_obj):
        """判断物体是否显著移动"""
        bbox1 = current_obj.get('bbox', [0, 0, 0, 0])
        bbox2 = prev_obj.get('bbox', [0, 0, 0, 0])
        
        # 计算中心点距离
        center1_x = (bbox1[0] + bbox1[2]) / 2
        center1_y = (bbox1[1] + bbox1[3]) / 2
        center2_x = (bbox2[0] + bbox2[2]) / 2
        center2_y = (bbox2[1] + bbox2[3]) / 2
        
        distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
        
        # 移动距离阈值（像素）
        return distance > 50
    
    def announce_detection_changes(self, changes):
        """播报检测变化"""
        current_time = time.time()
        
        # 播报新出现的障碍物
        for obj in changes['new_obstacles']:
            obj_name = obj.get('class_name', '物体')
            bbox = obj.get('bbox', [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            position = self.get_object_position(x1, y1, x2, y2)
            class_id = obj.get('class_id', 0)
            
            if class_id == 1:  # 动态障碍
                message = f"{position}出现动态障碍物{obj_name}，请注意"
                self.voice_announce(message, priority=4, category="检测")
            elif class_id == 2:  # 地面危险
                message = f"{position}出现地面危险{obj_name}，建议绕行"
                self.voice_announce(message, priority=3, category="检测")
            else:  # 静态障碍
                message = f"{position}出现静态障碍物{obj_name}，注意避让"
                self.voice_announce(message, priority=2, category="检测")
        
        # 播报消失的障碍物
        for obj in changes['disappeared_obstacles']:
            obj_name = obj.get('class_name', '物体')
            bbox = obj.get('bbox', [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            position = self.get_object_position(x1, y1, x2, y2)
            class_id = obj.get('class_id', 0)
            
            if class_id == 1:  # 动态障碍
                message = f"{position}的动态障碍物{obj_name}已离开"
                self.voice_announce(message, priority=1, category="信息")
        
        # 播报移动的障碍物
        for move_info in changes['moved_obstacles']:
            current_obj = move_info['current']
            prev_obj = move_info['previous']
            obj_name = current_obj.get('class_name', '物体')
            
            # 计算移动方向
            bbox1 = current_obj.get('bbox', [0, 0, 0, 0])
            bbox2 = prev_obj.get('bbox', [0, 0, 0, 0])
            
            center1_x = (bbox1[0] + bbox1[2]) / 2
            center2_x = (bbox2[0] + bbox2[2]) / 2
            
            if center1_x > center2_x + 20:
                direction = "向右移动"
            elif center1_x < center2_x - 20:
                direction = "向左移动"
            else:
                direction = "位置变化"
            
            message = f"动态障碍物{obj_name}{direction}"
            self.voice_announce(message, priority=3, category="检测")
    
    def detect_objects_in_frame(self, frame):
        """检测帧中的物体"""
        detections = []
        
        try:
            # 使用YOLO模型进行检测
            if self.yolo_available and self.yolo_model:
                # 运行YOLO检测
                results = self.yolo_model(frame, conf=self.detection_confidence_threshold, 
                                        iou=self.detection_nms_threshold, verbose=False)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # 获取边界框坐标
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().numpy())
                            
                            # 根据类别ID确定类别名称
                            class_names = {
                                0: '人',          # person
                                1: '自行车',      # bicycle
                                2: '汽车',        # car
                                3: '摩托车',      # motorcycle
                                4: '公交车',      # bus
                                5: '卡车',        # truck
                                6: '交通灯',      # traffic_light
                                7: '停车标志',    # stop_sign
                                8: '长椅',        # bench
                                9: '椅子',        # chair
                                10: '桌子',       # table
                                11: '瓶子',       # bottle
                                12: '杯子',       # cup
                                13: '笔记本电脑', # laptop
                                14: '书',         # book
                                15: '剪刀',       # scissors
                                16: '泰迪熊',     # teddy_bear
                                17: '吹风机',     # hair_drier
                                18: '牙刷',       # toothbrush
                                19: '垃圾桶',     # trash_can
                                20: '花盆',       # potted_plant
                                21: '沙发',       # couch
                                22: '床',         # bed
                                23: '电视',       # tv
                                24: '手机',       # cell_phone
                                25: '键盘',       # keyboard
                                26: '鼠标',       # mouse
                                27: '遥控器',     # remote
                                28: '时钟',       # clock
                                29: '花瓶',       # vase
                                30: '滑板',       # skateboard
                                31: '冲浪板',     # surfboard
                                32: '网球拍',     # tennis_racket
                                33: '棒球',       # baseball
                                34: '棒球棒',     # baseball_bat
                                35: '棒球手套',   # baseball_glove
                                36: '滑板车',     # skateboard
                                37: '滑板',       # skateboard
                                38: '滑板',       # skateboard
                                39: '滑板'        # skateboard
                            }
                            
                            class_name = class_names.get(class_id, 'unknown')
                            
                            # 根据类别确定障碍物类型
                            if class_id in [0]:  # 人
                                obstacle_type = 1  # 动态障碍
                            elif class_id in [1, 2, 3, 4, 5]:  # 车辆
                                obstacle_type = 1  # 动态障碍
                            elif class_id in [8, 9, 10]:  # 长椅、椅子、桌子
                                obstacle_type = 0  # 静态障碍
                            else:
                                obstacle_type = 0  # 静态障碍
                            
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'class_name': class_name,
                                'confidence': float(confidence),
                                'class_id': obstacle_type
                            })
            
            else:
                # 如果没有YOLO模型，使用模拟检测结果
                if np.random.random() > 0.8:  # 降低检测频率，提高精度
                    h, w = frame.shape[:2]
                    x1 = int(w * 0.2)
                    y1 = int(h * 0.3)
                    x2 = int(w * 0.4)
                    y2 = int(h * 0.7)
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'class_name': 'obstacle',
                        'confidence': 0.85,
                        'class_id': 0  # 0: 静态障碍, 1: 动态障碍, 2: 地面危险
                    })
        
        except Exception as e:
            print(f"⚠️ 物体检测失败: {e}")
        
        return detections
    
    def draw_detection_boxes(self, frame, detections):
        """绘制检测框"""
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            class_id = detection.get('class_id', 0)
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # 根据类别选择颜色
            colors = {
                0: (0, 255, 0),    # 绿色 - 静态障碍
                1: (0, 0, 255),    # 红色 - 动态障碍
                2: (255, 0, 0),    # 蓝色 - 地面危险
            }
            color = colors.get(class_id, (255, 255, 0))
            
            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def draw_trajectory_predictions(self, frame, trajectory_result):
        """绘制轨迹预测"""
        # 绘制盲道轨迹
        if trajectory_result.get('blind_path') and trajectory_result['blind_path'].get('predicted_trajectory'):
            trajectory = trajectory_result['blind_path']['predicted_trajectory']
            for i, point in enumerate(trajectory):
                cv2.circle(frame, point, 3, (0, 255, 255), -1)
                if i > 0:
                    cv2.line(frame, trajectory[i-1], point, (0, 255, 255), 2)
        
        # 绘制目标轨迹
        for obj in trajectory_result.get('tracked_objects', []):
            if obj.get('predicted_trajectory'):
                trajectory = obj['predicted_trajectory']
                for i, point in enumerate(trajectory):
                    cv2.circle(frame, point, 2, (255, 0, 255), -1)
                    if i > 0:
                        cv2.line(frame, trajectory[i-1], point, (255, 0, 255), 1)
        
        return frame
    
    def display_camera_frame(self, frame):
        """显示摄像头帧"""
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.camera_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.camera_display.setPixmap(scaled_pixmap)
    
    def update_detection_info(self, detections, env_result, trajectory_result):
        """更新检测信息"""
        # 更新检测数量
        self.detection_count_label.setText(f"检测数量: {len(detections)}")
        
        # 更新FPS（简单计算）
        current_time = time.time()
        if not hasattr(self, 'last_fps_time'):
            self.last_fps_time = current_time
            self.frame_count = 0
        else:
            self.frame_count += 1
            if current_time - self.last_fps_time >= 1.0:
                fps = self.frame_count / (current_time - self.last_fps_time)
                self.fps_label.setText(f"FPS: {fps:.1f}")
                self.last_fps_time = current_time
                self.frame_count = 0
    
    def announce_detection_results(self, detections, env_result, trajectory_result):
        """语音播报检测结果"""
        # 播报检测到的障碍物
        for detection in detections:
            class_name = detection['class_name']
            confidence = detection['confidence']
            class_id = detection.get('class_id', 0)
            bbox = detection.get('bbox', [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            
            # 判断物体位置
            center_x = (x1 + x2) / 2
            if center_x < 200:
                position = "左侧"
            elif center_x > 440:
                position = "右侧"
            else:
                position = "前方"
            
            # 根据类别确定优先级和播报内容
            if class_id == 1:  # 动态障碍
                priority = 4
                message = f"{position}检测到动态障碍物{class_name}，请注意避让"
            elif class_id == 2:  # 地面危险
                priority = 3
                message = f"{position}检测到地面危险{class_name}，建议绕行"
            else:  # 静态障碍
                priority = 2
                message = f"{position}检测到静态障碍物{class_name}，注意避让"
            
            self.voice_announce(message, priority=priority, category="检测")
        
        # 播报环境风险评估
        if env_result:
            overall_safety = env_result.get('overall_safety_level', 'safe')
            if overall_safety == 'high_risk':
                self.voice_announce("环境风险极高，请立即停止前进", priority=5, category="紧急")
            elif overall_safety == 'medium_risk':
                self.voice_announce("环境存在风险，请提高警惕", priority=3, category="警告")
        
        # 播报轨迹预测提示
        if trajectory_result:
            blind_path = trajectory_result.get('blind_path')
            if blind_path:
                confidence = blind_path.get('confidence', 0)
                if confidence > 0.7:
                    self.voice_announce("盲道清晰，建议沿盲道前进", priority=1, category="指导")
                elif confidence > 0.4:
                    self.voice_announce("盲道部分可见，请谨慎前进", priority=2, category="警告")
                else:
                    self.voice_announce("盲道不清晰，建议使用其他导航方式", priority=3, category="警告")
            
            # 播报碰撞风险
            collision_risks = trajectory_result.get('collision_risks', {})
            if collision_risks:
                max_risk = max(collision_risks.values())
                if max_risk > 0.7:
                    self.voice_announce("检测到高风险障碍物，请立即停止", priority=5, category="紧急")
                elif max_risk > 0.4:
                    self.voice_announce("存在中等风险障碍物，请减速慢行", priority=3, category="警告")
    
    def toggle_show_detection(self):
        """切换显示检测框"""
        pass  # 状态已在update_camera_frame中使用
    
    def toggle_show_trajectory(self):
        """切换显示轨迹"""
        pass  # 状态已在update_camera_frame中使用
    
    def change_detection_accuracy(self, accuracy):
        """改变检测精度"""
        self.detection_accuracy = accuracy
        print(f"检测精度已切换为: {accuracy}")
        
        # 根据精度调整检测参数
        if accuracy == "高精度":
            self.detection_confidence_threshold = 0.3
            self.detection_nms_threshold = 0.4
        elif accuracy == "平衡":
            self.detection_confidence_threshold = 0.5
            self.detection_nms_threshold = 0.5
        else:  # 快速
            self.detection_confidence_threshold = 0.7
            self.detection_nms_threshold = 0.6
    
    def toggle_show_detection_switch(self):
        """切换显示检测框开关"""
        if self.show_boxes_switch.isChecked():
            self.show_boxes_switch.setText("显示检测框")
            self.show_boxes_switch.setStyleSheet("QPushButton { background-color: #4caf50; color: white; font-weight: bold; padding: 8px; }")
        else:
            self.show_boxes_switch.setText("显示检测框")
            self.show_boxes_switch.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 8px; }")
    
    def toggle_show_trajectory_switch(self):
        """切换显示轨迹开关"""
        if self.show_trajectory_switch.isChecked():
            self.show_trajectory_switch.setText("显示轨迹")
            self.show_trajectory_switch.setStyleSheet("QPushButton { background-color: #4caf50; color: white; font-weight: bold; padding: 8px; }")
        else:
            self.show_trajectory_switch.setText("显示轨迹")
            self.show_trajectory_switch.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 8px; }")
    
    def toggle_show_environment_switch(self):
        """切换环境检测开关"""
        if self.show_environment_switch.isChecked():
            self.show_environment_switch.setText("环境检测")
            self.show_environment_switch.setStyleSheet("QPushButton { background-color: #4caf50; color: white; font-weight: bold; padding: 8px; }")
        else:
            self.show_environment_switch.setText("环境检测")
            self.show_environment_switch.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 8px; }")
    
    def start_mobile_test(self):
        """启动手机测试功能"""
        try:
            from mobile_app_android import MobileApp
            mobile_app = MobileApp()
            if mobile_app.initialize():
                print("📱 手机测试功能已启动")
                self.voice_announce("手机测试功能已启动", priority=1, category="系统")
            else:
                QMessageBox.warning(self, "错误", "手机测试功能启动失败")
        except ImportError:
            QMessageBox.information(self, "提示", "手机测试功能正在开发中...")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动手机测试失败: {e}")
    
    def toggle_environment_detection(self):
        """切换环境检测开关"""
        if self.env_detection_btn.isChecked():
            self.env_detection_btn.setText("🌍 环境检测: 开启")
            self.env_detection_btn.setStyleSheet("QPushButton { background-color: #ff9800; color: white; font-weight: bold; padding: 10px; font-size: 14px; }")
            self.env_detection_enabled = True
            self.voice_announce("环境检测已开启", priority=1, category="系统")
        else:
            self.env_detection_btn.setText("🌍 环境检测: 关闭")
            self.env_detection_btn.setStyleSheet("QPushButton { background-color: #666; color: white; font-weight: bold; padding: 10px; font-size: 14px; }")
            self.env_detection_enabled = False
            self.voice_announce("环境检测已关闭", priority=1, category="系统")
    
    def change_env_mode(self, mode):
        """改变环境检测模式"""
        self.voice_announce(f"环境检测模式已切换为{mode}", priority=1, category="系统")
        print(f"环境检测模式切换为: {mode}")
    
    def test_environment_detection(self):
        """测试环境检测功能"""
        try:
            if not self.env_detector:
                QMessageBox.warning(self, "警告", "环境检测模块未加载")
                return
            
            # 创建测试图像
            import numpy as np
            test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
            mock_detections = [{'bbox': [100, 100, 200, 200], 'confidence': 0.8, 'class': 0}]
            
            # 执行环境检测
            import time
            start_time = time.time()
            result = self.env_detector.detect_environment(test_frame, mock_detections)
            detection_time = time.time() - start_time
            
            # 显示结果
            overall_safety = result.get('overall_safety_level', 'unknown')
            safety_score = result.get('safety_score', 0)
            safety_percentage = int(safety_score * 100)
            
            result_text = f"""环境检测测试结果:
            
检测耗时: {detection_time:.3f}秒
安全等级: {overall_safety}
安全评分: {safety_percentage}%"""
            
            # 添加详细信息
            weather_info = result.get('weather_conditions')
            if weather_info:
                weather_type = weather_info.get('weather_type', 'unknown')
                result_text += f"\n天气条件: {weather_type}"
            
            lighting_info = result.get('lighting_conditions')
            if lighting_info:
                lighting_level = lighting_info.get('lighting_level', 'unknown')
                result_text += f"\n光照条件: {lighting_level}"
            
            surface_info = result.get('surface_conditions')
            if surface_info:
                surface_type = surface_info.get('surface_type', 'unknown')
                result_text += f"\n路面条件: {surface_type}"
            
            warnings = result.get('warnings', [])
            if warnings:
                result_text += f"\n警告数量: {len(warnings)}"
                for i, warning in enumerate(warnings[:3], 1):
                    result_text += f"\n  {i}. {warning}"
            
            # 更新环境检测结果显示
            self.update_environment_display(result, mock_detections)
            
            QMessageBox.information(self, "环境检测测试", result_text)
            self.voice_announce(f"环境检测测试完成，安全等级{overall_safety}，安全评分{safety_percentage}%", priority=2, category="测试")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"环境检测测试失败: {e}")
    
    def voice_announce_environment(self):
        """语音播报当前环境信息"""
        try:
            if not self.env_detector:
                QMessageBox.warning(self, "警告", "环境检测模块未加载")
                return
            
            # 获取当前摄像头帧
            if not self.cap or not self.camera_active:
                QMessageBox.warning(self, "警告", "请先开启摄像头检测")
                return
            
            ret, frame = self.cap.read()
            if not ret:
                QMessageBox.warning(self, "警告", "无法获取摄像头图像")
                return
            
            # 进行环境检测
            detections = self.detect_objects_in_frame(frame)
            env_result = self.analyze_environment(frame)
            
            if not env_result:
                QMessageBox.warning(self, "警告", "环境检测失败")
                return
            
            # 生成环境语音播报内容
            voice_content = self.generate_environment_voice_content(detections, env_result)
            
            if voice_content:
                # 语音播报
                overall_safety = env_result.get('overall_safety_level', 'safe')
                priority = 5 if overall_safety == 'high_risk' else 3 if overall_safety == 'medium_risk' else 2
                category = "紧急" if overall_safety == 'high_risk' else "警告" if overall_safety == 'medium_risk' else "检测"
                self.voice_announce(voice_content, priority=priority, category=category)
                
                # 显示播报内容
                QMessageBox.information(self, "语音播报环境", f"正在播报环境信息:\n\n{voice_content}")
            else:
                QMessageBox.information(self, "语音播报环境", "当前环境安全，无需特殊播报")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"语音播报环境失败: {e}")
    
    def real_time_environment_voice_announce(self, env_result, detections):
        """实时环境语音播报"""
        try:
            # 检查播报冷却时间
            current_time = time.time()
            if current_time - self.last_voice_time < self.voice_cooldown:
                return
            
            # 获取环境安全等级
            overall_safety = env_result.get('overall_safety_level', 'safe')
            safety_score = env_result.get('safety_score', 1.0)
            safety_percentage = int(safety_score * 100)
            
            # 根据安全等级决定播报内容
            if overall_safety == 'high_risk':
                # 高风险环境，立即播报
                voice_content = f"环境高风险，安全评分{safety_percentage}%，请立即停止前进"
                priority = 5
                category = "紧急"
                self.voice_announce(voice_content, priority=priority, category=category)
                self.last_voice_time = current_time
                
            elif overall_safety == 'medium_risk':
                # 中等风险环境，播报警告
                voice_content = f"环境中等风险，安全评分{safety_percentage}%，请小心前行"
                priority = 3
                category = "警告"
                self.voice_announce(voice_content, priority=priority, category=category)
                self.last_voice_time = current_time
                
            else:
                # 安全环境，定期播报状态
                if current_time - self.last_voice_time > 10:  # 每10秒播报一次安全状态
                    voice_content = f"环境安全，安全评分{safety_percentage}%"
                    priority = 1
                    category = "检测"
                    self.voice_announce(voice_content, priority=priority, category=category)
                    self.last_voice_time = current_time
            
            # 检测到重要环境事物时播报
            if detections:
                for detection in detections:
                    obj_type = detection.get('class_name', '未知物体')
                    confidence = detection.get('confidence', 0)
                    
                    # 只播报高置信度的检测结果
                    if confidence > 0.7:
                        bbox = detection.get('bbox', [0, 0, 0, 0])
                        x1, y1, x2, y2 = bbox
                        position = self.get_object_position(x1, y1, x2, y2)
                        
                        # 根据物体类型决定播报内容
                        if obj_type in ['人', 'person', '行人']:
                            voice_content = f"前方{position}检测到行人"
                        elif obj_type in ['车', 'car', '车辆']:
                            voice_content = f"前方{position}检测到车辆"
                        elif obj_type in ['障碍物', 'obstacle']:
                            voice_content = f"前方{position}检测到障碍物"
                        else:
                            voice_content = f"前方{position}检测到{obj_type}"
                        
                        # 播报检测到的物体
                        self.voice_announce(voice_content, priority=2, category="检测")
                        self.last_voice_time = current_time
                        break  # 一次只播报一个物体，避免重复
            
        except Exception as e:
            print(f"实时环境语音播报失败: {e}")
    
    def update_voice_status(self):
        """更新语音播报状态显示"""
        if self.voice_enabled:
            if self.voice_mode == "静默模式":
                self.voice_status_display.setText("语音播报: 静默模式")
                self.voice_status_display.setStyleSheet("color: #666; font-weight: bold; font-size: 14px; padding: 8px;")
            elif self.voice_mode == "简洁模式":
                self.voice_status_display.setText("语音播报: 简洁模式")
                self.voice_status_display.setStyleSheet("color: #ff9800; font-weight: bold; font-size: 14px; padding: 8px;")
            else:  # 详细模式
                self.voice_status_display.setText("语音播报: 详细模式")
                self.voice_status_display.setStyleSheet("color: #4caf50; font-weight: bold; font-size: 14px; padding: 8px;")
        else:
            self.voice_status_display.setText("语音播报: 已禁用")
            self.voice_status_display.setStyleSheet("color: #f44336; font-weight: bold; font-size: 14px; padding: 8px;")
    
    def open_camera_detect(self):
        """保留原有方法以兼容性"""
        self.toggle_camera_detection()

class TwoPointImageLabel(QLabel):
    """两点模式图像标签，支持鼠标事件"""
    mouse_pressed = pyqtSignal(QMouseEvent)
    mouse_moved = pyqtSignal(QMouseEvent)
    mouse_released = pyqtSignal(QMouseEvent)
    
    def __init__(self, text="请选择图像进行标注"):
        super().__init__(text)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #cccccc;
                border-radius: 10px;
                background-color: #f9f9f9;
                color: #666666;
                font-size: 16px;
                padding: 20px;
            }
            QLabel:hover {
                border-color: #0078d4;
                background-color: #f0f8ff;
            }
        """)
        
    def mousePressEvent(self, event: QMouseEvent):
        self.mouse_pressed.emit(event)
        
    def mouseMoveEvent(self, event: QMouseEvent):
        self.mouse_moved.emit(event)
        
    def mouseReleaseEvent(self, event: QMouseEvent):
        self.mouse_released.emit(event)

# 增强版摄像头检测窗口
class EnhancedCameraDetectWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("增强版摄像头实时检测")
        self.setGeometry(100, 100, 1400, 1000)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
        
        # 检测状态
        self.is_detecting = False
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # 检测结果
        self.current_detections = []
        self.current_environment = None
        self.current_trajectory = None
        
        self.init_ui()
        
    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)
        
        # 控制面板
        control_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("开始检测")
        self.start_btn.clicked.connect(self.toggle_detection)
        self.start_btn.setStyleSheet("QPushButton { background-color: #4caf50; color: white; font-weight: bold; padding: 10px; }")
        control_layout.addWidget(self.start_btn)
        
        self.show_boxes_btn = QPushButton("显示检测框")
        self.show_boxes_btn.setCheckable(True)
        self.show_boxes_btn.setChecked(True)
        self.show_boxes_btn.clicked.connect(self.toggle_show_boxes)
        control_layout.addWidget(self.show_boxes_btn)
        
        self.show_trajectory_btn = QPushButton("显示轨迹")
        self.show_trajectory_btn.setCheckable(True)
        self.show_trajectory_btn.setChecked(True)
        self.show_trajectory_btn.clicked.connect(self.toggle_show_trajectory)
        control_layout.addWidget(self.show_trajectory_btn)
        
        control_layout.addStretch()
        
        # 状态显示
        self.status_label = QLabel("状态: 未启动")
        self.status_label.setStyleSheet("color: #666; padding: 5px; background-color: #f0f0f0; border-radius: 3px;")
        control_layout.addWidget(self.status_label)
        
        layout.addLayout(control_layout)
        
        # 主显示区域
        main_layout = QHBoxLayout()
        
        # 摄像头显示区域
        self.camera_label = QLabel("点击'开始检测'启动摄像头")
        self.camera_label.setMinimumSize(800, 600)
        self.camera_label.setStyleSheet("""
            QLabel {
                border: 2px solid #ddd;
                border-radius: 10px;
                background-color: #f9f9f9;
                color: #666;
                font-size: 16px;
                padding: 20px;
            }
        """)
        self.camera_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.camera_label)
        
        # 检测信息面板
        info_panel = QWidget()
        info_panel.setMaximumWidth(300)
        info_layout = QVBoxLayout(info_panel)
        
        # 检测结果
        detection_group = QGroupBox("检测结果")
        detection_layout = QVBoxLayout(detection_group)
        
        self.detection_count_label = QLabel("检测数量: 0")
        detection_layout.addWidget(self.detection_count_label)
        
        self.detection_list = QListWidget()
        self.detection_list.setMaximumHeight(150)
        detection_layout.addWidget(self.detection_list)
        
        info_layout.addWidget(detection_group)
        
        # 环境信息
        env_group = QGroupBox("环境信息")
        env_layout = QVBoxLayout(env_group)
        
        self.env_status_label = QLabel("环境: 未检测")
        env_layout.addWidget(self.env_status_label)
        
        self.env_warnings_label = QLabel("警告: 无")
        env_layout.addWidget(self.env_warnings_label)
        
        info_layout.addWidget(env_group)
        
        # 轨迹信息
        trajectory_group = QGroupBox("轨迹信息")
        trajectory_layout = QVBoxLayout(trajectory_group)
        
        self.trajectory_status_label = QLabel("轨迹: 未预测")
        trajectory_layout.addWidget(self.trajectory_status_label)
        
        self.collision_risk_label = QLabel("碰撞风险: 低")
        trajectory_layout.addWidget(self.collision_risk_label)
        
        info_layout.addWidget(trajectory_group)
        
        main_layout.addWidget(info_panel)
        layout.addLayout(main_layout)
        
    def toggle_detection(self):
        """切换检测状态"""
        if not self.is_detecting:
            self.start_detection()
        else:
            self.stop_detection()
    
    def start_detection(self):
        """开始检测"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.warning(self, "错误", "无法打开摄像头")
                return
            
            self.is_detecting = True
            self.start_btn.setText("停止检测")
            self.start_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 10px; }")
            self.status_label.setText("状态: 检测中")
            self.status_label.setStyleSheet("color: #4caf50; padding: 5px; background-color: #e8f5e8; border-radius: 3px;")
            
            self.timer.start(33)  # 约30FPS
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动检测失败: {e}")
    
    def stop_detection(self):
        """停止检测"""
        self.is_detecting = False
        self.timer.stop()
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.start_btn.setText("开始检测")
        self.start_btn.setStyleSheet("QPushButton { background-color: #4caf50; color: white; font-weight: bold; padding: 10px; }")
        self.status_label.setText("状态: 已停止")
        self.status_label.setStyleSheet("color: #666; padding: 5px; background-color: #f0f0f0; border-radius: 3px;")
        
        self.camera_label.setText("检测已停止")
    
    def update_frame(self):
        """更新帧"""
        if not self.cap or not self.is_detecting:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # 进行检测
        detections = self.detect_objects(frame)
        
        # 环境检测和轨迹预测
        if self.parent and self.parent.env_detector and self.parent.trajectory_predictor:
            # 环境检测
            env_result = self.parent.analyze_environment(frame)
            if env_result:
                self.current_environment = env_result
                self.update_environment_display(env_result)
            
            # 轨迹预测
            trajectory_result = self.parent.predict_trajectory(frame)
            if trajectory_result:
                self.current_trajectory = trajectory_result
                self.update_trajectory_display(trajectory_result)
        
        # 绘制检测结果
        if self.show_boxes_btn.isChecked():
            frame = self.draw_detections(frame, detections)
        
        if self.show_trajectory_btn.isChecked() and self.current_trajectory:
            frame = self.draw_trajectory(frame, self.current_trajectory)
        
        # 显示帧
        self.display_frame(frame)
        
        # 更新检测信息
        self.update_detection_info(detections)
    
    def detect_objects(self, frame):
        """检测物体"""
        # 这里应该调用实际的检测模型
        # 暂时返回模拟结果
        detections = []
        
        # 模拟检测结果
        if np.random.random() > 0.7:
            h, w = frame.shape[:2]
            x1 = int(w * 0.2)
            y1 = int(h * 0.3)
            x2 = int(w * 0.4)
            y2 = int(h * 0.7)
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'class_name': 'obstacle',
                'confidence': 0.85,
                'class_id': 0
            })
        
        return detections
    
    def draw_detections(self, frame, detections):
        """绘制检测框"""
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            class_id = detection.get('class_id', 0)
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # 根据类别选择颜色
            colors = {
                0: (0, 255, 0),    # 绿色 - 静态障碍
                1: (0, 0, 255),    # 红色 - 动态障碍
                2: (255, 0, 0),    # 蓝色 - 地面危险
            }
            color = colors.get(class_id, (255, 255, 0))
            
            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def draw_trajectory(self, frame, trajectory_result):
        """绘制轨迹预测"""
        # 绘制盲道轨迹
        if trajectory_result.get('blind_path') and trajectory_result['blind_path'].get('predicted_trajectory'):
            trajectory = trajectory_result['blind_path']['predicted_trajectory']
            for i, point in enumerate(trajectory):
                cv2.circle(frame, point, 3, (0, 255, 255), -1)
                if i > 0:
                    cv2.line(frame, trajectory[i-1], point, (0, 255, 255), 2)
        
        # 绘制目标轨迹
        for obj in trajectory_result.get('tracked_objects', []):
            if obj.get('predicted_trajectory'):
                trajectory = obj['predicted_trajectory']
                for i, point in enumerate(trajectory):
                    cv2.circle(frame, point, 2, (255, 0, 255), -1)
                    if i > 0:
                        cv2.line(frame, trajectory[i-1], point, (255, 0, 255), 1)
        
        return frame
    
    def display_frame(self, frame):
        """显示帧"""
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.camera_label.setPixmap(scaled_pixmap)
    
    def update_detection_info(self, detections):
        """更新检测信息"""
        self.current_detections = detections
        self.detection_count_label.setText(f"检测数量: {len(detections)}")
        
        # 更新检测列表
        self.detection_list.clear()
        for i, detection in enumerate(detections):
            class_name = detection['class_name']
            confidence = detection['confidence']
            item_text = f"{i+1}. {class_name} ({confidence:.2f})"
            self.detection_list.addItem(item_text)
    
    def update_environment_display(self, env_result):
        """更新环境显示"""
        overall_safety = env_result.get('overall_safety_level', 'safe')
        if overall_safety == 'high_risk':
            self.env_status_label.setText("环境: 高风险")
            self.env_status_label.setStyleSheet("color: #d32f2f;")
        elif overall_safety == 'medium_risk':
            self.env_status_label.setText("环境: 中等风险")
            self.env_status_label.setStyleSheet("color: #f57c00;")
        else:
            self.env_status_label.setText("环境: 安全")
            self.env_status_label.setStyleSheet("color: #4caf50;")
        
        warnings = env_result.get('warnings', [])
        if warnings:
            self.env_warnings_label.setText(f"警告: {len(warnings)}个")
            self.env_warnings_label.setStyleSheet("color: #d32f2f;")
        else:
            self.env_warnings_label.setText("警告: 无")
            self.env_warnings_label.setStyleSheet("color: #4caf50;")
    
    def update_trajectory_display(self, result, detections):
        """更新轨迹显示"""
        # 更新动态障碍物轨迹
        dynamic_objects = [d for d in detections if d.get('class_id') == 1]  # 动态障碍
        if dynamic_objects:
            trajectory_text = ""
            for i, obj in enumerate(dynamic_objects, 1):
                obj_name = obj.get('class_name', '物体')
                bbox = obj.get('bbox', [0, 0, 0, 0])
                x1, y1, x2, y2 = bbox
                position = self.get_object_position(x1, y1, x2, y2)
                
                # 预测下一步行为
                predicted_action = self.predict_object_behavior(obj)
                trajectory_text += f"{i}. {obj_name} ({position}) - 预测行为: {predicted_action}\n"
            
            self.dynamic_objects_list.setPlainText(trajectory_text)
        else:
            self.dynamic_objects_list.setPlainText("无动态障碍物")
        
        # 更新用户轨迹建议
        user_suggestions = self.generate_user_trajectory_suggestions(detections, result)
        self.user_trajectory_list.setPlainText(user_suggestions)
        
        # 生成轨迹播报文本
        trajectory_voice_content = self.generate_trajectory_voice_content(detections, result)
        self.trajectory_voice_text.setPlainText(trajectory_voice_content)
        
        # 语音播报
        if trajectory_voice_content:
            self.voice_announce(trajectory_voice_content, priority=3, category="轨迹")
        
        # 更新轨迹状态
        if result.get('blind_path'):
            confidence = result['blind_path'].get('confidence', 0)
            if confidence > 0.7:
                self.trajectory_status_label.setText("轨迹: 清晰")
                self.trajectory_status_label.setStyleSheet("color: #4caf50;")
            else:
                self.trajectory_status_label.setText("轨迹: 模糊")
                self.trajectory_status_label.setStyleSheet("color: #ff9800;")
        else:
            self.trajectory_status_label.setText("轨迹: 未检测")
            self.trajectory_status_label.setStyleSheet("color: #666;")
        
        # 更新碰撞风险
        collision_risks = result.get('collision_risks', {})
        if collision_risks:
            max_risk = max(collision_risks.values())
            if max_risk > 0.7:
                self.collision_risk_label.setText("碰撞风险: 高")
                self.collision_risk_label.setStyleSheet("color: #d32f2f;")
            elif max_risk > 0.4:
                self.collision_risk_label.setText("碰撞风险: 中")
                self.collision_risk_label.setStyleSheet("color: #f57c00;")
            else:
                self.collision_risk_label.setText("碰撞风险: 低")
                self.collision_risk_label.setStyleSheet("color: #4caf50;")
        else:
            self.collision_risk_label.setText("碰撞风险: 低")
            self.collision_risk_label.setStyleSheet("color: #4caf50;")
    
    def predict_object_behavior(self, obj):
        """预测物体行为"""
        obj_name = obj.get('class_name', '物体')
        bbox = obj.get('bbox', [0, 0, 0, 0])
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # 根据物体类型和位置预测行为
        if obj_name in ['person', '人']:
            if center_x < 200:
                return "可能向左移动"
            elif center_x > 440:
                return "可能向右移动"
            else:
                return "可能继续直行"
        elif obj_name in ['car', '汽车', 'truck', '卡车', 'bus', '公交车']:
            if center_x < 200:
                return "可能向左转向"
            elif center_x > 440:
                return "可能向右转向"
            else:
                return "可能继续直行"
        elif obj_name in ['bicycle', '自行车', 'motorcycle', '摩托车']:
            return "可能快速移动，注意避让"
        else:
            return "位置相对稳定"
    
    def generate_user_trajectory_suggestions(self, detections, result):
        """生成用户轨迹建议"""
        suggestions = []
        
        # 分析动态障碍物
        dynamic_objects = [d for d in detections if d.get('class_id') == 1]
        if dynamic_objects:
            for obj in dynamic_objects:
                obj_name = obj.get('class_name', '物体')
                bbox = obj.get('bbox', [0, 0, 0, 0])
                x1, y1, x2, y2 = bbox
                position = self.get_object_position(x1, y1, x2, y2)
                
                if position == "左侧":
                    suggestions.append(f"建议向右偏移，避开左侧{obj_name}")
                elif position == "右侧":
                    suggestions.append(f"建议向左偏移，避开右侧{obj_name}")
                elif position == "中央":
                    suggestions.append(f"建议减速或停止，前方有{obj_name}")
                else:
                    suggestions.append(f"注意{position}的{obj_name}")
        
        # 分析静态障碍物
        static_objects = [d for d in detections if d.get('class_id') == 0]
        if static_objects:
            for obj in static_objects:
                obj_name = obj.get('class_name', '物体')
                bbox = obj.get('bbox', [0, 0, 0, 0])
                x1, y1, x2, y2 = bbox
                position = self.get_object_position(x1, y1, x2, y2)
                
                if position == "左侧":
                    suggestions.append(f"建议向右绕行，避开左侧{obj_name}")
                elif position == "右侧":
                    suggestions.append(f"建议向左绕行，避开右侧{obj_name}")
                elif position == "中央":
                    suggestions.append(f"建议寻找绕行路径，前方有{obj_name}")
        
        # 分析地面危险
        ground_hazards = [d for d in detections if d.get('class_id') == 2]
        if ground_hazards:
            for obj in ground_hazards:
                obj_name = obj.get('class_name', '物体')
                bbox = obj.get('bbox', [0, 0, 0, 0])
                x1, y1, x2, y2 = bbox
                position = self.get_object_position(x1, y1, x2, y2)
                suggestions.append(f"注意{position}的地面危险{obj_name}，建议绕行")
        
        if not suggestions:
            suggestions.append("路径清晰，可以正常前进")
        
        return "\n".join(suggestions)
    
    def generate_trajectory_voice_content(self, detections, result):
        """生成轨迹预测语音播报内容"""
        content_parts = []
        
        # 动态障碍物轨迹播报
        dynamic_objects = [d for d in detections if d.get('class_id') == 1]
        if dynamic_objects:
            for obj in dynamic_objects:
                obj_name = obj.get('class_name', '物体')
                bbox = obj.get('bbox', [0, 0, 0, 0])
                x1, y1, x2, y2 = bbox
                position = self.get_object_position(x1, y1, x2, y2)
                predicted_action = self.predict_object_behavior(obj)
                content_parts.append(f"前方{position}的{obj_name}，{predicted_action}")
        
        # 用户轨迹建议播报
        user_suggestions = self.generate_user_trajectory_suggestions(detections, result)
        if user_suggestions and "路径清晰" not in user_suggestions:
            # 只播报最重要的建议
            first_suggestion = user_suggestions.split('\n')[0]
            content_parts.append(first_suggestion)
        
        return "；".join(content_parts) if content_parts else "路径安全"
    
    def toggle_show_boxes(self):
        """切换显示检测框"""
        pass  # 状态已在update_frame中使用
    
    def toggle_show_trajectory(self):
        """切换显示轨迹"""
        pass  # 状态已在update_frame中使用
    
    def closeEvent(self, event):
        """关闭事件"""
        self.stop_detection()
        event.accept()

# 新增：摄像头检测窗口
class CameraDetectWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("摄像头实时检测")
        self.setGeometry(100, 100, 1280, 960)
        self.layout = QVBoxLayout(self)
        
        # 摄像头显示区域
        self.image_label = QLabel("正在打开摄像头...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.layout.addWidget(self.image_label)
        
        # 状态显示区域
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                padding: 5px;
                font-size: 14px;
                color: #333;
            }
        """)
        self.layout.addWidget(self.status_label)
        
        # 语音控制面板
        self.create_voice_control_panel()
        
        # 初始化变量
        self.timer = None
        self.cap = None
        self.model = None
        self.baidu_client = None
        self.tts_lock = None
        self.last_alert_time = 0
        self.is_running = False
        
        # 语音相关
        self.voice_system = None
        
        # 语音库
        self.voice_library = None
        if VOICE_LIBRARY_AVAILABLE:
            self.voice_library = VoiceLibrary()
            print("✅ 语音库初始化成功")
        else:
            print("⚠️ 语音库不可用，使用默认语音提示")
        
        # 距离变化检测
        self.last_distance = None
        self.last_direction = None
        self.last_label = None
        
        # 轨迹预测
        self.trajectory_predictor = None
        if TRAJECTORY_PREDICTOR_AVAILABLE:
            self.trajectory_predictor = TrajectoryPredictor()
            print("✅ 轨迹预测模块初始化成功")
        else:
            print("⚠️ 轨迹预测模块不可用，轨迹预测功能将不可用")
        
        # 初始化摄像头
        self.init_camera()
        
        # 初始化模型
        self.load_model()
        
        # 初始化语音API
        self.init_voice_api()
        
        # 启动定时器
        self.start_timer()
    
    def create_voice_control_panel(self):
        """创建语音控制面板"""
        # 创建主控制面板
        control_panel = QHBoxLayout()
        
        # 语音控制面板
        voice_panel = QGroupBox("语音控制")
        voice_panel.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #0078d4;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        voice_layout = QHBoxLayout()
        
        # 语音开关按钮
        self.voice_toggle_btn = QPushButton("语音播报: 开启")
        self.voice_toggle_btn.setCheckable(True)
        self.voice_toggle_btn.setChecked(True)
        self.voice_toggle_btn.clicked.connect(self.toggle_voice)
        self.voice_toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #28a745;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
        """)
        
        # 音量控制
        volume_label = QLabel("音量:")
        self.volume_slider = QSpinBox()
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        self.volume_slider.valueChanged.connect(self.change_volume)
        
        # 测试语音按钮
        test_voice_btn = QPushButton("测试语音")
        test_voice_btn.clicked.connect(self.test_voice_system)
        test_voice_btn.setStyleSheet("""
            QPushButton {
                background-color: #ffc107;
                color: #333;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:pressed {
                background-color: #e0a800;
            }
        """)
        
        # 语音状态显示
        self.voice_status_label = QLabel("语音系统就绪")
        self.voice_status_label.setStyleSheet("""
            QLabel {
                color: #28a745;
                font-weight: bold;
                padding: 5px;
            }
        """)
        
        voice_layout.addWidget(self.voice_toggle_btn)
        voice_layout.addWidget(volume_label)
        voice_layout.addWidget(self.volume_slider)
        voice_layout.addWidget(test_voice_btn)
        voice_layout.addWidget(self.voice_status_label)
        voice_layout.addStretch()
        
        voice_panel.setLayout(voice_layout)
        
        # 轨迹预测控制面板
        trajectory_panel = QGroupBox("轨迹预测控制")
        trajectory_panel.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #28a745;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        trajectory_layout = QHBoxLayout()
        
        # 轨迹预测开关
        self.trajectory_toggle_btn = QPushButton("轨迹预测: 开启")
        self.trajectory_toggle_btn.setCheckable(True)
        self.trajectory_toggle_btn.setChecked(True)
        self.trajectory_toggle_btn.clicked.connect(self.toggle_trajectory)
        self.trajectory_toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #28a745;
            }
            QPushButton:!checked {
                background-color: #6c757d;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
        """)
        
        # 轨迹预测状态显示
        self.trajectory_status_label = QLabel("轨迹预测就绪")
        self.trajectory_status_label.setStyleSheet("""
            QLabel {
                color: #28a745;
                font-weight: bold;
                padding: 5px;
            }
        """)
        
        # 盲道状态显示
        self.blind_path_status_label = QLabel("盲道: 未检测")
        self.blind_path_status_label.setStyleSheet("""
            QLabel {
                color: #6c757d;
                font-weight: bold;
                padding: 5px;
            }
        """)
        
        # 碰撞风险显示
        self.collision_risk_label = QLabel("碰撞风险: 低")
        self.collision_risk_label.setStyleSheet("""
            QLabel {
                color: #28a745;
                font-weight: bold;
                padding: 5px;
            }
        """)
        
        trajectory_layout.addWidget(self.trajectory_toggle_btn)
        trajectory_layout.addWidget(self.trajectory_status_label)
        trajectory_layout.addWidget(self.blind_path_status_label)
        trajectory_layout.addWidget(self.collision_risk_label)
        trajectory_layout.addStretch()
        
        trajectory_panel.setLayout(trajectory_layout)
        
        # 将两个面板添加到主控制面板
        control_panel.addWidget(voice_panel)
        control_panel.addWidget(trajectory_panel)
        
        # 将主控制面板添加到布局
        self.layout.addLayout(control_panel)

    def init_camera(self):
        """初始化摄像头"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                # 尝试其他摄像头索引
                for i in range(1, 5):
                    self.cap = cv2.VideoCapture(i)
                    if self.cap.isOpened():
                        break
                
                if not self.cap.isOpened():
                    raise Exception("无法打开摄像头")
                    
            self.status_label.setText("摄像头初始化成功")
        except Exception as e:
            self.status_label.setText(f"摄像头初始化失败: {e}")
            QMessageBox.warning(self, "摄像头错误", f"无法打开摄像头: {e}")

    def init_voice_api(self):
        """初始化语音API"""
        try:
            print("🔧 开始初始化语音API...")
            
            # 初始化简化语音系统
            self.voice_system = SimpleVoiceSystem()
            print("✅ 语音系统初始化成功")
            
            self.status_label.setText("语音API初始化成功")
            print("✅ 语音系统初始化完成")
            
            # 测试语音功能
            self.test_voice_system()
            
        except Exception as e:
            self.status_label.setText(f"语音API初始化失败: {e}")
            print(f"❌ 语音系统初始化失败: {e}")
    
    def test_voice_system(self):
        """测试语音系统"""
        try:
            print("🧪 开始测试语音系统...")
            test_text = "语音系统测试成功"
            self.speak(test_text)
            self.voice_status_label.setText("语音测试中...")
            self.voice_status_label.setStyleSheet("color: #ffc107; font-weight: bold; padding: 5px;")
            print("✅ 语音测试完成")
        except Exception as e:
            print(f"❌ 语音测试失败: {e}")
            self.voice_status_label.setText("语音测试失败")
            self.voice_status_label.setStyleSheet("color: #dc3545; font-weight: bold; padding: 5px;")
    
    def toggle_voice(self):
        """切换语音播报开关"""
        if self.voice_toggle_btn.isChecked():
            self.voice_toggle_btn.setText("语音播报: 开启")
            self.voice_toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #28a745;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:pressed {
                    background-color: #1e7e34;
                }
            """)
            self.voice_status_label.setText("语音播报已开启")
            self.voice_status_label.setStyleSheet("color: #28a745; font-weight: bold; padding: 5px;")
        else:
            self.voice_toggle_btn.setText("语音播报: 关闭")
            self.voice_toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #dc3545;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:pressed {
                    background-color: #c82333;
                }
            """)
            self.voice_status_label.setText("语音播报已关闭")
            self.voice_status_label.setStyleSheet("color: #dc3545; font-weight: bold; padding: 5px;")
    
    def change_volume(self, value):
        """改变音量"""
        if self.media_player:
            self.media_player.setVolume(value)
            print(f"🔊 音量设置为: {value}%")

    def start_timer(self):
        """启动定时器"""
        try:
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)  # 30ms间隔
            self.is_running = True
        except Exception as e:
            self.status_label.setText(f"定时器启动失败: {e}")

    def speak(self, text):
        """语音播报"""
        print(f"🎤 speak方法被调用，文本: {text}")
        
        # 检查语音开关
        if not self.voice_toggle_btn.isChecked():
            print("🔇 语音播报已关闭")
            return
            
        if not self.voice_system or not text:
            print("❌ 语音播报失败: 语音系统或文本为空")
            return
        
        print(f"✅ 准备播报语音: {text}")
        
        # 直接调用语音系统
        self.voice_system.speak(text)
        
        # 更新状态显示
        self.voice_status_label.setText("正在播报...")
        self.voice_status_label.setStyleSheet("color: #0078d4; font-weight: bold; padding: 5px;")
        
        # 延迟恢复状态
        QTimer.singleShot(2000, self.reset_voice_status)
    
    def reset_voice_status(self):
        """重置语音状态"""
        if self.voice_toggle_btn.isChecked():
            self.voice_status_label.setText("语音播报已开启")
            self.voice_status_label.setStyleSheet("color: #28a745; font-weight: bold; padding: 5px;")
        else:
            self.voice_status_label.setText("语音播报已关闭")
            self.voice_status_label.setStyleSheet("color: #dc3545; font-weight: bold; padding: 5px;")
    
    def toggle_voice(self):
        """切换语音播报开关"""
        if self.voice_toggle_btn.isChecked():
            self.voice_toggle_btn.setText("语音播报: 开启")
            self.voice_toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #28a745;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:pressed {
                    background-color: #1e7e34;
                }
            """)
            if self.voice_system:
                self.voice_system.enable()
            self.voice_status_label.setText("语音播报已开启")
            self.voice_status_label.setStyleSheet("color: #28a745; font-weight: bold; padding: 5px;")
        else:
            self.voice_toggle_btn.setText("语音播报: 关闭")
            self.voice_toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #dc3545;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:pressed {
                    background-color: #c82333;
                }
            """)
            if self.voice_system:
                self.voice_system.disable()
            self.voice_status_label.setText("语音播报已关闭")
            self.voice_status_label.setStyleSheet("color: #dc3545; font-weight: bold; padding: 5px;")
    
    def change_volume(self, value):
        """改变音量（简化版）"""
        print(f"🔊 音量设置为: {value}%")
    
    def toggle_trajectory(self):
        """切换轨迹预测开关"""
        if self.trajectory_toggle_btn.isChecked():
            self.trajectory_toggle_btn.setText("轨迹预测: 开启")
            self.trajectory_toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #28a745;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:pressed {
                    background-color: #1e7e34;
                }
            """)
            self.trajectory_status_label.setText("轨迹预测已开启")
            self.trajectory_status_label.setStyleSheet("color: #28a745; font-weight: bold; padding: 5px;")
        else:
            self.trajectory_toggle_btn.setText("轨迹预测: 关闭")
            self.trajectory_toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #6c757d;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:pressed {
                    background-color: #545b62;
                }
            """)
            self.trajectory_status_label.setText("轨迹预测已关闭")
            self.trajectory_status_label.setStyleSheet("color: #6c757d; font-weight: bold; padding: 5px;")
    
    def update_trajectory_ui(self, prediction_result):
        """更新轨迹预测UI显示"""
        if not prediction_result:
            return
        
        # 更新盲道状态
        blind_path_info = prediction_result.get('blind_path_info')
        if blind_path_info and blind_path_info.get('detected'):
            self.blind_path_status_label.setText("盲道: 已检测")
            self.blind_path_status_label.setStyleSheet("color: #28a745; font-weight: bold; padding: 5px;")
        else:
            self.blind_path_status_label.setText("盲道: 未检测")
            self.blind_path_status_label.setStyleSheet("color: #6c757d; font-weight: bold; padding: 5px;")
        
        # 更新碰撞风险
        collision_risks = prediction_result.get('collision_risks', {})
        if collision_risks:
            max_risk = max(collision_risks.values()) if collision_risks.values() else 0
            if max_risk > 0.7:
                risk_text = "碰撞风险: 高"
                risk_color = "#dc3545"
            elif max_risk > 0.4:
                risk_text = "碰撞风险: 中"
                risk_color = "#ffc107"
            else:
                risk_text = "碰撞风险: 低"
                risk_color = "#28a745"
            
            self.collision_risk_label.setText(risk_text)
            self.collision_risk_label.setStyleSheet(f"color: {risk_color}; font-weight: bold; padding: 5px;")
        else:
            self.collision_risk_label.setText("碰撞风险: 低")
            self.collision_risk_label.setStyleSheet("color: #28a745; font-weight: bold; padding: 5px;")
    
    def handle_voice_command(self, command):
        """处理语音指令"""
        try:
            command = command.lower()
            print(f"🎤 收到语音指令: {command}")
            
            if "停止" in command or "暂停" in command:
                self.is_running = False
                self.timer.stop()
                self.status_label.setText("检测已停止")
                self.speak("检测已停止")
                
            elif "开始" in command or "启动" in command:
                self.is_running = True
                self.timer.start(30)
                self.status_label.setText("检测已启动")
                self.speak("检测已启动")
                
            elif "距离" in command:
                self.speak("当前检测距离为3到5米")
                
            elif "帮助" in command or "说明" in command:
                self.speak("可用指令：开始检测、停止检测、距离信息、帮助说明")
                
            else:
                self.speak(f"收到指令：{command}")
                
        except Exception as e:
            print(f"❌ 处理语音指令失败: {e}")

    def load_model(self):
        """加载YOLO模型"""
        if not YOLO_AVAILABLE:
            self.status_label.setText("YOLO库不可用")
            return
            
        try:
            # 优先使用本地模型文件
            model_paths = [
                'models/yolov8n.pt',  # 优先使用models目录下的模型
                'yolov8n.pt',         # 当前目录下的模型
                'models/yolo11n.pt'   # 备用模型
            ]
            
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path:
                print(f"📥 加载本地模型: {model_path}")
                self.model = YOLO(model_path)
                self.status_label.setText("模型加载成功")
            else:
                print("⚠️ 本地模型文件不存在，尝试下载...")
                self.status_label.setText("正在下载模型...")
                self.model = YOLO('yolov8n.pt')  # 这会自动下载
                self.status_label.setText("模型下载并加载成功")
                
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            self.status_label.setText(f"模型加载失败: {e}")
            self.model = None

    def estimate_distance(self, box, known_height=1.5, focal_length=700):
        """距离估算"""
        try:
            pixel_height = abs(box[3] - box[1])
            if pixel_height == 0:
                return 99
            distance = (known_height * focal_length) / pixel_height
            return max(0.1, min(99, distance))  # 限制距离范围
        except:
            return 99

    def update_frame(self):
        """更新摄像头帧 - 增强版（集成轨迹预测）"""
        if not self.is_running or not self.cap or not self.cap.isOpened():
            return
            
        try:
            ret, frame = self.cap.read()
            if not ret:
                self.status_label.setText("无法获取摄像头帧")
                return
                
            if self.model:
                # Step 1: YOLOv8检测
                results = self.model(frame)
                detections = []
                
                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confs = result.boxes.conf.cpu().numpy()
                        clss = result.boxes.cls.cpu().numpy()
                        
                        for box, conf, cls in zip(boxes, confs, clss):
                            if conf > 0.5:  # 置信度阈值
                                x1, y1, x2, y2 = box
                                detections.append([x1, y1, x2, y2, conf, int(cls)])
                
                # Step 2: 轨迹预测处理（如果启用）
                prediction_result = None
                if (self.trajectory_predictor and 
                    self.trajectory_toggle_btn.isChecked() and 
                    TRAJECTORY_PREDICTOR_AVAILABLE):
                    try:
                        prediction_result = self.trajectory_predictor.process_frame(frame, detections)
                        # 更新轨迹预测UI
                        self.update_trajectory_ui(prediction_result)
                    except Exception as e:
                        print(f"轨迹预测处理失败: {e}")
                
                # Step 3: 绘制检测结果和轨迹
                frame = self.draw_detections_and_trajectories(frame, detections, prediction_result)
                
                # Step 4: 传统检测预警（保持原有功能）
                alert_info = None
                for detection in detections:
                    try:
                        x1, y1, x2, y2, conf, cls = detection
                        b = [int(x1), int(y1), int(x2), int(y2)]
                        
                        # 获取类别信息
                        if cls in CLASS_INFO:
                            label_name = CLASS_INFO[cls]
                            # 为不同类别设置不同颜色
                            if cls == 0:  # person
                                color = (0, 255, 0)  # 绿色
                            elif cls == 39:  # bottle
                                color = (255, 0, 0)  # 红色
                            elif cls == 38:  # tennis racket
                                color = (0, 0, 255)  # 蓝色
                            elif cls == 41:  # cup
                                color = (255, 255, 0)  # 青色
                            elif cls == 71:  # sink
                                color = (255, 0, 255)  # 紫色
                            elif cls == 78:  # toothbrush
                                color = (0, 255, 255)  # 黄色
                            else:
                                color = DEFAULT_COLOR
                        else:
                            label_name = f"class{cls}"
                            color = DEFAULT_COLOR
                        
                        # 距离估算
                        distance = self.estimate_distance(b)
                        
                        # 计算方位
                        frame_width = frame.shape[1]
                        center_x = (b[0] + b[2]) / 2
                        position_ratio = center_x / frame_width
                        
                        if position_ratio < 0.33:
                            direction = "左侧"
                        elif position_ratio > 0.67:
                            direction = "右侧"
                        else:
                            direction = "正前方"
                        
                        # 只对最近的障碍物预警
                        if alert_info is None or distance < alert_info["distance"]:
                            alert_info = {
                                "cls": cls, 
                                "distance": distance, 
                                "label_name": label_name,
                                "direction": direction
                            }
                    except Exception as e:
                        print(f"处理检测框时出错: {e}")
                        continue
                
                # 预警分级
                if alert_info:
                    self.on_detected(alert_info["cls"], alert_info["distance"], alert_info["label_name"], alert_info["direction"])
                else:
                    self.status_label.setText("")
            
            # 显示图像
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_img).scaled(
                    self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(pixmap)
            except Exception as e:
                print(f"显示图像时出错: {e}")
                
        except Exception as e:
            print(f"更新帧时出错: {e}")
    

    
    def draw_detections_and_trajectories(self, frame, detections, prediction_result):
        """绘制检测结果和轨迹"""
        try:
            # 绘制检测框
            for detection in detections:
                x1, y1, x2, y2, conf, cls = detection
                b = [int(x1), int(y1), int(x2), int(y2)]
                
                # 获取类别信息
                if cls in CLASS_INFO:
                    label_name = CLASS_INFO[cls]
                    # 为不同类别设置不同颜色
                    if cls == 0:  # person
                        color = (0, 255, 0)  # 绿色
                    elif cls == 39:  # bottle
                        color = (255, 0, 0)  # 红色
                    elif cls == 38:  # tennis racket
                        color = (0, 0, 255)  # 蓝色
                    elif cls == 41:  # cup
                        color = (255, 255, 0)  # 青色
                    elif cls == 71:  # sink
                        color = (255, 0, 255)  # 紫色
                    elif cls == 78:  # toothbrush
                        color = (0, 255, 255)  # 黄色
                    else:
                        color = DEFAULT_COLOR
                else:
                    label_name = f"class{cls}"
                    color = DEFAULT_COLOR
                
                # 绘制检测框
                cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, 2)
                label = f"{label_name} {conf:.2f}"
                cv2.putText(frame, label, (b[0], max(b[1]-10, 0)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 绘制轨迹预测结果
            if prediction_result:
                # 绘制盲道轮廓
                blind_path_info = prediction_result.get('blind_path_info')
                if blind_path_info and blind_path_info.get('detected'):
                    contour = blind_path_info.get('contour')
                    if contour is not None:
                        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                        # 绘制盲道中心线
                        center = blind_path_info.get('center')
                        if center:
                            cv2.circle(frame, center, 5, (0, 255, 0), -1)
                
                # 绘制跟踪对象和轨迹
                tracked_objects = prediction_result.get('tracked_objects', [])
                for obj in tracked_objects:
                    obj_id = obj.get('id')
                    bbox = obj.get('bbox')
                    trajectory = obj.get('trajectory', [])
                    
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        # 绘制跟踪框
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        cv2.putText(frame, f"ID:{obj_id}", (int(x1), int(y1)-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    # 绘制历史轨迹
                    if len(trajectory) > 1:
                        for i in range(1, len(trajectory)):
                            pt1 = trajectory[i-1]
                            pt2 = trajectory[i]
                            cv2.line(frame, pt1, pt2, (255, 255, 0), 2)
                
                # 绘制预测轨迹
                predicted_trajectories = prediction_result.get('predicted_trajectories', {})
                for obj_id, trajectory in predicted_trajectories.items():
                    if len(trajectory) > 1:
                        for i in range(1, len(trajectory)):
                            pt1 = trajectory[i-1]
                            pt2 = trajectory[i]
                            cv2.line(frame, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)
                
                # 绘制碰撞风险警告
                collision_risks = prediction_result.get('collision_risks', {})
                for obj_id, risk in collision_risks.items():
                    if risk > 0.7:  # 高风险
                        # 找到对应的跟踪对象
                        for obj in tracked_objects:
                            if obj.get('id') == obj_id:
                                bbox = obj.get('bbox')
                                if bbox:
                                    x1, y1, x2, y2 = bbox
                                    # 绘制警告框
                                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                                    cv2.putText(frame, "HIGH RISK", (int(x1), int(y1)-30), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                break
            
        except Exception as e:
            print(f"绘制检测结果和轨迹时出错: {e}")
        
        return frame

    def on_detected(self, cls, distance, label_name, direction):
        """检测到障碍物时的处理 - 增强版智能语音库系统"""
        try:
                            # 获取障碍物的详细信息
                obstacle_info = self.get_obstacle_info(cls, label_name)
                category = obstacle_info.get('category', 'unknown')
                obstacle_type = obstacle_info.get('type', label_name)
                risk_level = obstacle_info.get('risk_level', 1)
                
                print(f"🔍 检测到障碍物: {label_name} ({obstacle_type}), 类别: {category}, 距离: {distance:.1f}米, 方位: {direction}")
                print(f"🔧 语音开关状态: {self.voice_toggle_btn.isChecked()}")
                
                # 检查是否有显著变化（距离变化超过0.5米或方位变化）
                distance_changed = (self.last_distance is None or 
                                  abs(distance - self.last_distance) > 0.5)
                direction_changed = (self.last_direction != direction)
                label_changed = (self.last_label != label_name)
                
                # 更新记录
                self.last_distance = distance
                self.last_direction = direction
                self.last_label = label_name
                
                # 只有在有显著变化时才播报语音
                should_speak = distance_changed or direction_changed or label_changed
                
                # 使用智能语音提示系统
                if should_speak:
                    msg = self.generate_smart_message(obstacle_type, distance, direction, category, risk_level)
                    self.status_label.setText(msg)
                    print(f"📢 智能语音提示: {msg}")
                    self.speak(msg)
                    
        except Exception as e:
            print(f"❌ 处理检测结果时出错: {e}")
    
    def get_obstacle_info(self, cls, label_name):
        """获取障碍物的详细信息"""
        try:
            # 从COCO类别获取基本信息
            if cls in CLASS_INFO:
                label_name = CLASS_INFO[cls]
                # 根据类别设置风险等级和分类
                if cls == 0:  # person
                    category = "dynamic_pedestrian"
                    risk_level = 3
                elif cls in [39, 41, 71, 78]:  # bottle, cup, sink, toothbrush
                    category = "static_object"
                    risk_level = 2
                elif cls == 38:  # tennis racket
                    category = "sports_equipment"
                    risk_level = 2
                else:
                    category = "general_object"
                    risk_level = 1
            else:
                label_name = f"class{cls}"
                category = "unknown"
                risk_level = 1
            
            # 从voice_config.json获取更详细的映射
            if hasattr(self, 'voice_library') and self.voice_library:
                class_mapping = self.voice_library.class_mapping
                mapping = class_mapping.get(str(cls), [])
                if len(mapping) >= 2:
                    category = mapping[0]
                    obstacle_type = mapping[1]
                else:
                    obstacle_type = label_name
            else:
                obstacle_type = label_name
            
            return {
                'category': category,
                'type': label_name,
                'name': label_name,
                'class_id': cls,
                'risk_level': risk_level
            }
        except Exception as e:
            print(f"❌ 获取障碍物信息失败: {e}")
            return {
                'category': 'unknown',
                'type': label_name,
                'name': label_name,
                'class_id': cls,
                'risk_level': 1
            }
    
    def generate_smart_message(self, obstacle_type, distance, direction, category, risk_level):
        """生成智能语音消息"""
        try:
            # 根据风险等级和距离生成不同的消息
            if risk_level >= 4:  # 高风险
                if distance < 1.0:
                    return f"紧急！{direction}{distance:.1f}米有{obstacle_type}，立即停止！"
                elif distance < 2.0:
                    return f"危险！{direction}{distance:.1f}米有{obstacle_type}，立即减速！"
                else:
                    return f"注意！{direction}{distance:.1f}米有{obstacle_type}，请小心"
            
            elif risk_level == 3:  # 中高风险
                if distance < 1.5:
                    return f"小心！{direction}{distance:.1f}米有{obstacle_type}，请减速"
                elif distance < 3.0:
                    return f"{direction}{distance:.1f}米有{obstacle_type}，请注意"
                else:
                    return f"前方{distance:.1f}米{direction}有{obstacle_type}"
            
            elif risk_level == 2:  # 中等风险
                if distance < 2.0:
                    return f"{direction}{distance:.1f}米有{obstacle_type}，请注意"
                else:
                    return f"前方{distance:.1f}米{direction}有{obstacle_type}"
            
            else:  # 低风险
                if distance < 1.0:
                    return f"{direction}{distance:.1f}米有{obstacle_type}"
                else:
                    return f"前方{distance:.1f}米{direction}有{obstacle_type}"
                    
        except Exception as e:
            print(f"❌ 生成智能消息失败: {e}")
            return f"{direction}{distance:.1f}米有{obstacle_type}"
    


    def closeEvent(self, event):
        """关闭事件处理"""
        try:
            self.is_running = False
            
            # 停止定时器
            if self.timer:
                self.timer.stop()
            
            # 释放摄像头
            if self.cap and self.cap.isOpened():
                self.cap.release()
            
            # 清理语音系统
            if self.voice_system:
                self.voice_system.disable()
                print("✅ 语音系统已停止")
                
            print("摄像头检测窗口已关闭，资源已清理")
        except Exception as e:
            print(f"关闭窗口时出错: {e}")
        finally:
            event.accept()

def main():
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    # 创建主窗口
    window = TwoPointAnnotator()
    window.show()
    
    print("两点模式和拖拽模式标注工具已启动！")
    print("新功能：")
    print("1. 两点模式：点击两个点，自动生成直线")
    print("2. 拖拽模式：点击起点，拖拽到终点，形成直线")
    print("3. 支持清除临时点重新开始")
    print("4. 支持你的原始图片")
    print("5. 修复了坐标漂移问题")
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 