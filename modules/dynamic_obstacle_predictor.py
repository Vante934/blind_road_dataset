#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态障碍物预测模块
结合目标检测和轨迹预测，实现动态障碍物的运动预测和碰撞风险评估
"""

import numpy as np
import cv2
import math
from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass
from enum import Enum
import time

class ObstacleType(Enum):
    """障碍物类型枚举"""
    PERSON = "person"
    VEHICLE = "vehicle"
    BICYCLE = "bicycle"
    MOTORCYCLE = "motorcycle"
    UNKNOWN = "unknown"

class MovementState(Enum):
    """运动状态枚举"""
    STATIC = "static"
    MOVING = "moving"
    FAST_MOVING = "fast_moving"
    STOPPING = "stopping"

@dataclass
class Obstacle:
    """障碍物数据结构"""
    id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[float, float]      # (cx, cy)
    obstacle_type: ObstacleType
    confidence: float
    timestamp: float
    
    # 运动信息
    velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy)
    speed: float = 0.0
    direction: float = 0.0  # 角度，弧度
    movement_state: MovementState = MovementState.STATIC
    
    # 轨迹历史
    trajectory: deque = None
    max_trajectory_length: int = 50
    
    def __post_init__(self):
        if self.trajectory is None:
            self.trajectory = deque(maxlen=self.max_trajectory_length)
        self.trajectory.append((self.center, self.timestamp))

@dataclass
class Prediction:
    """预测结果数据结构"""
    obstacle_id: int
    predicted_positions: List[Tuple[float, float]]  # 预测位置列表
    predicted_times: List[float]                   # 对应时间
    collision_risk: float                          # 碰撞风险 (0-1)
    time_to_collision: Optional[float]             # 碰撞时间（秒）
    recommended_action: str                        # 建议行动

class DynamicObstaclePredictor:
    """动态障碍物预测器"""
    
    def __init__(self, 
                 prediction_horizon: float = 3.0,  # 预测时间范围（秒）
                 time_step: float = 0.1,           # 时间步长（秒）
                 collision_threshold: float = 100.0,  # 碰撞距离阈值（像素）
                 velocity_threshold: float = 5.0,     # 运动速度阈值（像素/秒）
                 max_tracking_distance: float = 200.0):  # 最大跟踪距离（像素）
        
        self.prediction_horizon = prediction_horizon
        self.time_step = time_step
        self.collision_threshold = collision_threshold
        self.velocity_threshold = velocity_threshold
        self.max_tracking_distance = max_tracking_distance
        
        # 障碍物跟踪
        self.obstacles: Dict[int, Obstacle] = {}
        self.next_obstacle_id = 0
        
        # 预测参数
        self.prediction_steps = int(prediction_horizon / time_step)
        
        # 卡尔曼滤波器参数
        self.kalman_filters: Dict[int, cv2.KalmanFilter] = {}
        
    def update_obstacles(self, detections: List[Dict]) -> List[Obstacle]:
        """更新障碍物信息"""
        current_time = time.time()
        current_obstacles = []
        
        # 为每个检测创建或更新障碍物
        for detection in detections:
            bbox = detection['bbox']
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            
            # 尝试匹配现有障碍物
            matched_obstacle = self._match_obstacle(center, current_time)
            
            if matched_obstacle:
                # 更新现有障碍物
                self._update_obstacle(matched_obstacle, bbox, center, detection, current_time)
            else:
                # 创建新障碍物
                obstacle = self._create_obstacle(bbox, center, detection, current_time)
                self.obstacles[obstacle.id] = obstacle
            
            current_obstacles.append(self.obstacles[matched_obstacle.id] if matched_obstacle else obstacle)
        
        # 清理长时间未更新的障碍物
        self._cleanup_old_obstacles(current_time)
        
        return current_obstacles
    
    def _match_obstacle(self, center: Tuple[float, float], current_time: float) -> Optional[Obstacle]:
        """匹配现有障碍物"""
        best_match = None
        min_distance = float('inf')
        
        for obstacle in self.obstacles.values():
            # 计算距离
            distance = math.sqrt(
                (center[0] - obstacle.center[0])**2 + 
                (center[1] - obstacle.center[1])**2
            )
            
            # 检查时间差（避免匹配过旧的障碍物）
            time_diff = current_time - obstacle.timestamp
            if time_diff > 1.0:  # 超过1秒未更新
                continue
                
            # 检查距离阈值
            if distance < self.max_tracking_distance and distance < min_distance:
                min_distance = distance
                best_match = obstacle
        
        return best_match
    
    def _create_obstacle(self, bbox: Tuple[int, int, int, int], 
                        center: Tuple[float, float], 
                        detection: Dict, 
                        current_time: float) -> Obstacle:
        """创建新障碍物"""
        obstacle_type = self._get_obstacle_type(detection.get('class_name', 'unknown'))
        
        obstacle = Obstacle(
            id=self.next_obstacle_id,
            bbox=bbox,
            center=center,
            obstacle_type=obstacle_type,
            confidence=detection.get('confidence', 0.0),
            timestamp=current_time
        )
        
        # 初始化卡尔曼滤波器
        self._init_kalman_filter(obstacle.id)
        
        self.next_obstacle_id += 1
        return obstacle
    
    def _update_obstacle(self, obstacle: Obstacle, bbox: Tuple[int, int, int, int],
                        center: Tuple[float, float], detection: Dict, current_time: float):
        """更新障碍物信息"""
        # 计算速度
        time_diff = current_time - obstacle.timestamp
        if time_diff > 0:
            velocity = (
                (center[0] - obstacle.center[0]) / time_diff,
                (center[1] - obstacle.center[1]) / time_diff
            )
            speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
            direction = math.atan2(velocity[1], velocity[0])
            
            # 更新障碍物信息
            obstacle.bbox = bbox
            obstacle.center = center
            obstacle.velocity = velocity
            obstacle.speed = speed
            obstacle.direction = direction
            obstacle.timestamp = current_time
            obstacle.confidence = detection.get('confidence', obstacle.confidence)
            
            # 更新运动状态
            obstacle.movement_state = self._determine_movement_state(speed)
            
            # 添加到轨迹
            obstacle.trajectory.append((center, current_time))
            
            # 更新卡尔曼滤波器
            self._update_kalman_filter(obstacle.id, center, velocity)
    
    def _get_obstacle_type(self, class_name: str) -> ObstacleType:
        """根据类别名称获取障碍物类型"""
        class_name = class_name.lower()
        if class_name in ['person', 'people']:
            return ObstacleType.PERSON
        elif class_name in ['car', 'truck', 'bus']:
            return ObstacleType.VEHICLE
        elif class_name == 'bicycle':
            return ObstacleType.BICYCLE
        elif class_name == 'motorcycle':
            return ObstacleType.MOTORCYCLE
        else:
            return ObstacleType.UNKNOWN
    
    def _determine_movement_state(self, speed: float) -> MovementState:
        """确定运动状态"""
        if speed < 1.0:
            return MovementState.STATIC
        elif speed < self.velocity_threshold:
            return MovementState.MOVING
        elif speed < self.velocity_threshold * 2:
            return MovementState.FAST_MOVING
        else:
            return MovementState.STOPPING
    
    def _init_kalman_filter(self, obstacle_id: int):
        """初始化卡尔曼滤波器"""
        kf = cv2.KalmanFilter(4, 2)  # 4个状态变量(x,y,vx,vy)，2个观测变量(x,y)
        
        # 状态转移矩阵
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # 观测矩阵
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # 过程噪声协方差
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        
        # 观测噪声协方差
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        
        # 误差协方差
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        
        self.kalman_filters[obstacle_id] = kf
    
    def _update_kalman_filter(self, obstacle_id: int, center: Tuple[float, float], 
                             velocity: Tuple[float, float]):
        """更新卡尔曼滤波器"""
        if obstacle_id not in self.kalman_filters:
            return
        
        kf = self.kalman_filters[obstacle_id]
        
        # 预测
        kf.predict()
        
        # 更新
        measurement = np.array([[center[0]], [center[1]]], dtype=np.float32)
        kf.correct(measurement)
    
    def _cleanup_old_obstacles(self, current_time: float):
        """清理长时间未更新的障碍物"""
        timeout = 2.0  # 2秒超时
        to_remove = []
        
        for obstacle_id, obstacle in self.obstacles.items():
            if current_time - obstacle.timestamp > timeout:
                to_remove.append(obstacle_id)
        
        for obstacle_id in to_remove:
            del self.obstacles[obstacle_id]
            if obstacle_id in self.kalman_filters:
                del self.kalman_filters[obstacle_id]
    
    def predict_obstacle_trajectories(self, user_position: Tuple[float, float] = None) -> List[Prediction]:
        """预测障碍物轨迹"""
        predictions = []
        
        for obstacle in self.obstacles.values():
            if len(obstacle.trajectory) < 3:  # 需要至少3个点才能预测
                continue
            
            # 使用卡尔曼滤波器预测
            predicted_positions = self._predict_with_kalman(obstacle.id)
            
            if not predicted_positions:
                continue
            
            # 计算碰撞风险
            collision_risk = 0.0
            time_to_collision = None
            
            if user_position:
                collision_risk, time_to_collision = self._calculate_collision_risk(
                    predicted_positions, user_position
                )
            
            # 生成建议行动
            recommended_action = self._generate_recommendation(
                obstacle, collision_risk, time_to_collision
            )
            
            prediction = Prediction(
                obstacle_id=obstacle.id,
                predicted_positions=predicted_positions,
                predicted_times=[i * self.time_step for i in range(len(predicted_positions))],
                collision_risk=collision_risk,
                time_to_collision=time_to_collision,
                recommended_action=recommended_action
            )
            
            predictions.append(prediction)
        
        return predictions
    
    def _predict_with_kalman(self, obstacle_id: int) -> List[Tuple[float, float]]:
        """使用卡尔曼滤波器预测轨迹"""
        if obstacle_id not in self.kalman_filters:
            return []
        
        kf = self.kalman_filters[obstacle_id]
        predicted_positions = []
        
        # 保存当前状态
        current_state = kf.statePre.copy()
        
        for step in range(self.prediction_steps):
            # 预测下一步
            kf.predict()
            state = kf.statePre
            
            # 提取位置
            x = state[0, 0]
            y = state[1, 0]
            predicted_positions.append((x, y))
        
        # 恢复状态
        kf.statePre = current_state
        
        return predicted_positions
    
    def _calculate_collision_risk(self, predicted_positions: List[Tuple[float, float]], 
                                 user_position: Tuple[float, float]) -> Tuple[float, Optional[float]]:
        """计算碰撞风险"""
        min_distance = float('inf')
        collision_time = None
        
        for i, (x, y) in enumerate(predicted_positions):
            distance = math.sqrt(
                (x - user_position[0])**2 + 
                (y - user_position[1])**2
            )
            
            if distance < min_distance:
                min_distance = distance
                collision_time = i * self.time_step
        
        # 计算碰撞风险（距离越近，风险越高）
        if min_distance < self.collision_threshold:
            collision_risk = max(0.0, 1.0 - min_distance / self.collision_threshold)
        else:
            collision_risk = 0.0
        
        return collision_risk, collision_time if collision_risk > 0.5 else None
    
    def _generate_recommendation(self, obstacle: Obstacle, collision_risk: float, 
                                time_to_collision: Optional[float]) -> str:
        """生成建议行动"""
        if collision_risk > 0.8:
            return f"⚠️ 高风险！{time_to_collision:.1f}秒后可能碰撞，请立即避让"
        elif collision_risk > 0.5:
            return f"⚠️ 中等风险，{time_to_collision:.1f}秒后接近，建议减速"
        elif collision_risk > 0.2:
            return f"ℹ️ 低风险，注意观察{obstacle.obstacle_type.value}"
        else:
            return "✅ 安全，无障碍物威胁"
    
    def get_obstacle_statistics(self) -> Dict:
        """获取障碍物统计信息"""
        total_obstacles = len(self.obstacles)
        moving_obstacles = sum(1 for obs in self.obstacles.values() 
                             if obs.movement_state != MovementState.STATIC)
        
        obstacle_types = {}
        for obstacle in self.obstacles.values():
            obs_type = obstacle.obstacle_type.value
            obstacle_types[obs_type] = obstacle_types.get(obs_type, 0) + 1
        
        return {
            'total_obstacles': total_obstacles,
            'moving_obstacles': moving_obstacles,
            'static_obstacles': total_obstacles - moving_obstacles,
            'obstacle_types': obstacle_types,
            'average_speed': np.mean([obs.speed for obs in self.obstacles.values()]) if self.obstacles else 0.0
        }
    
    def visualize_predictions(self, frame: np.ndarray, predictions: List[Prediction]) -> np.ndarray:
        """可视化预测结果"""
        vis_frame = frame.copy()
        
        for prediction in predictions:
            obstacle = self.obstacles.get(prediction.obstacle_id)
            if not obstacle:
                continue
            
            # 绘制当前位置
            cv2.circle(vis_frame, (int(obstacle.center[0]), int(obstacle.center[1])), 
                      5, (0, 255, 0), -1)
            
            # 绘制预测轨迹
            if len(prediction.predicted_positions) > 1:
                points = []
                for pos in prediction.predicted_positions:
                    points.append((int(pos[0]), int(pos[1])))
                
                # 绘制轨迹线
                for i in range(len(points) - 1):
                    color_intensity = int(255 * (1 - i / len(points)))  # 逐渐变淡
                    cv2.line(vis_frame, points[i], points[i+1], 
                            (0, color_intensity, 255), 2)
            
            # 绘制碰撞风险
            if prediction.collision_risk > 0.5:
                # 绘制警告圆圈
                radius = int(self.collision_threshold * prediction.collision_risk)
                cv2.circle(vis_frame, (int(obstacle.center[0]), int(obstacle.center[1])), 
                          radius, (0, 0, 255), 2)
                
                # 添加文本
                text = f"Risk: {prediction.collision_risk:.2f}"
                cv2.putText(vis_frame, text, 
                           (int(obstacle.center[0]), int(obstacle.center[1]) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return vis_frame


if __name__ == "__main__":
    # 测试代码
    predictor = DynamicObstaclePredictor()
    
    # 模拟检测结果
    detections = [
        {
            'bbox': (100, 100, 200, 200),
            'class_name': 'person',
            'confidence': 0.9
        },
        {
            'bbox': (300, 150, 400, 250),
            'class_name': 'car',
            'confidence': 0.8
        }
    ]
    
    # 更新障碍物
    obstacles = predictor.update_obstacles(detections)
    print(f"检测到 {len(obstacles)} 个障碍物")
    
    # 预测轨迹
    predictions = predictor.predict_obstacle_trajectories()
    print(f"生成了 {len(predictions)} 个预测")
    
    # 获取统计信息
    stats = predictor.get_obstacle_statistics()
    print(f"统计信息: {stats}")

