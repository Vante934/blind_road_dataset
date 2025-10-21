#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轨迹预测系统
使用光流分析技术预测障碍物运动轨迹
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from collections import deque
import math

@dataclass
class TrajectoryPoint:
    """轨迹点"""
    x: float
    y: float
    timestamp: float
    confidence: float = 1.0

@dataclass
class TrajectoryPrediction:
    """轨迹预测结果"""
    object_id: str
    current_position: Tuple[float, float]
    predicted_position: Tuple[float, float]
    velocity: Tuple[float, float]
    acceleration: Tuple[float, float]
    collision_risk: float
    time_to_collision: float
    predicted_trajectory: List[Tuple[float, float]]

class OpticalFlowTracker:
    """光流跟踪器"""
    
    def __init__(self, max_corners=100, quality_level=0.01, min_distance=10):
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        
        # Lucas-Kanade光流参数
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # 特征检测器
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=3
        )
        
        # 跟踪历史
        self.track_history = {}
        self.max_history_length = 20
    
    def detect_features(self, frame: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """检测特征点"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        corners = cv2.goodFeaturesToTrack(
            gray,
            mask=mask,
            **self.feature_params
        )
        
        return corners
    
    def track_features(self, old_gray: np.ndarray, new_gray: np.ndarray, 
                      old_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """跟踪特征点"""
        if old_points is None or len(old_points) == 0:
            return None, None, None
        
        # 计算光流
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            old_gray, new_gray, old_points, None, **self.lk_params
        )
        
        # 选择好的点
        good_old = old_points[status == 1]
        good_new = new_points[status == 1]
        
        return good_old, good_new, status
    
    def update_track_history(self, object_id: str, points: np.ndarray, timestamp: float):
        """更新跟踪历史"""
        if object_id not in self.track_history:
            self.track_history[object_id] = deque(maxlen=self.max_history_length)
        
        for point in points:
            self.track_history[object_id].append(
                TrajectoryPoint(point[0], point[1], timestamp)
            )

class TrajectoryPredictor:
    """轨迹预测器"""
    
    def __init__(self):
        self.optical_flow_tracker = OpticalFlowTracker()
        self.object_trajectories = {}
        self.prediction_horizon = 2.0  # 预测时间范围（秒）
        self.collision_threshold = 1.0  # 碰撞风险阈值（米）
        
        # 预测参数
        self.min_trajectory_points = 3
        self.max_velocity = 10.0  # 最大速度（米/秒）
        self.prediction_steps = 10
    
    def update_trajectory(self, object_id: str, bbox: Tuple[float, float, float, float], 
                         timestamp: float, frame: np.ndarray = None):
        """更新对象轨迹"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        if object_id not in self.object_trajectories:
            self.object_trajectories[object_id] = deque(maxlen=20)
        
        self.object_trajectories[object_id].append(
            TrajectoryPoint(center_x, center_y, timestamp)
        )
    
    def predict_trajectory(self, object_id: str) -> Optional[TrajectoryPrediction]:
        """预测对象轨迹"""
        if object_id not in self.object_trajectories:
            return None
        
        trajectory = list(self.object_trajectories[object_id])
        if len(trajectory) < self.min_trajectory_points:
            return None
        
        # 计算速度和加速度
        velocity = self._calculate_velocity(trajectory)
        acceleration = self._calculate_acceleration(trajectory)
        
        # 获取当前位置
        current_pos = trajectory[-1]
        current_position = (current_pos.x, current_pos.y)
        
        # 预测未来位置
        predicted_position, predicted_trajectory = self._predict_future_position(
            current_position, velocity, acceleration
        )
        
        # 计算碰撞风险
        collision_risk = self._calculate_collision_risk(
            current_position, predicted_position, velocity
        )
        
        # 计算碰撞时间
        time_to_collision = self._calculate_time_to_collision(
            current_position, predicted_position, velocity
        )
        
        return TrajectoryPrediction(
            object_id=object_id,
            current_position=current_position,
            predicted_position=predicted_position,
            velocity=velocity,
            acceleration=acceleration,
            collision_risk=collision_risk,
            time_to_collision=time_to_collision,
            predicted_trajectory=predicted_trajectory
        )
    
    def _calculate_velocity(self, trajectory: List[TrajectoryPoint]) -> Tuple[float, float]:
        """计算速度"""
        if len(trajectory) < 2:
            return (0.0, 0.0)
        
        # 使用最近的点计算速度
        recent_points = trajectory[-min(5, len(trajectory)):]
        
        if len(recent_points) < 2:
            return (0.0, 0.0)
        
        # 计算平均速度
        total_dx = 0.0
        total_dy = 0.0
        total_dt = 0.0
        
        for i in range(1, len(recent_points)):
            dx = recent_points[i].x - recent_points[i-1].x
            dy = recent_points[i].y - recent_points[i-1].y
            dt = recent_points[i].timestamp - recent_points[i-1].timestamp
            
            if dt > 0:
                total_dx += dx / dt
                total_dy += dy / dt
                total_dt += 1
        
        if total_dt > 0:
            avg_vx = total_dx / total_dt
            avg_vy = total_dy / total_dt
        else:
            avg_vx = avg_vy = 0.0
        
        # 限制最大速度
        speed = math.sqrt(avg_vx**2 + avg_vy**2)
        if speed > self.max_velocity:
            scale = self.max_velocity / speed
            avg_vx *= scale
            avg_vy *= scale
        
        return (avg_vx, avg_vy)
    
    def _calculate_acceleration(self, trajectory: List[TrajectoryPoint]) -> Tuple[float, float]:
        """计算加速度"""
        if len(trajectory) < 3:
            return (0.0, 0.0)
        
        # 使用最近的点计算加速度
        recent_points = trajectory[-min(5, len(trajectory)):]
        
        if len(recent_points) < 3:
            return (0.0, 0.0)
        
        # 计算速度变化
        velocities = []
        for i in range(1, len(recent_points)):
            dx = recent_points[i].x - recent_points[i-1].x
            dy = recent_points[i].y - recent_points[i-1].y
            dt = recent_points[i].timestamp - recent_points[i-1].timestamp
            
            if dt > 0:
                vx = dx / dt
                vy = dy / dt
                velocities.append((vx, vy, recent_points[i].timestamp))
        
        if len(velocities) < 2:
            return (0.0, 0.0)
        
        # 计算加速度
        total_ax = 0.0
        total_ay = 0.0
        total_dt = 0.0
        
        for i in range(1, len(velocities)):
            dvx = velocities[i][0] - velocities[i-1][0]
            dvy = velocities[i][1] - velocities[i-1][1]
            dt = velocities[i][2] - velocities[i-1][2]
            
            if dt > 0:
                total_ax += dvx / dt
                total_ay += dvy / dt
                total_dt += 1
        
        if total_dt > 0:
            avg_ax = total_ax / total_dt
            avg_ay = total_ay / total_dt
        else:
            avg_ax = avg_ay = 0.0
        
        return (avg_ax, avg_ay)
    
    def _predict_future_position(self, current_position: Tuple[float, float], 
                                velocity: Tuple[float, float], 
                                acceleration: Tuple[float, float]) -> Tuple[Tuple[float, float], List[Tuple[float, float]]]:
        """预测未来位置"""
        x, y = current_position
        vx, vy = velocity
        ax, ay = acceleration
        
        # 预测轨迹点
        predicted_trajectory = []
        dt = self.prediction_horizon / self.prediction_steps
        
        for i in range(self.prediction_steps + 1):
            t = i * dt
            
            # 使用运动学方程: x = x0 + v0*t + 0.5*a*t^2
            pred_x = x + vx * t + 0.5 * ax * t**2
            pred_y = y + vy * t + 0.5 * ay * t**2
            
            predicted_trajectory.append((pred_x, pred_y))
        
        # 最终预测位置
        predicted_position = predicted_trajectory[-1]
        
        return predicted_position, predicted_trajectory
    
    def _calculate_collision_risk(self, current_position: Tuple[float, float], 
                                predicted_position: Tuple[float, float], 
                                velocity: Tuple[float, float]) -> float:
        """计算碰撞风险"""
        # 基于速度和距离的简单碰撞风险评估
        speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
        
        # 速度越高，风险越大
        speed_risk = min(speed / self.max_velocity, 1.0)
        
        # 基于预测位置的风险（这里简化处理）
        # 实际应用中应该考虑用户位置和路径
        position_risk = 0.5  # 默认中等风险
        
        # 综合风险
        collision_risk = (speed_risk * 0.7 + position_risk * 0.3)
        
        return min(collision_risk, 1.0)
    
    def _calculate_time_to_collision(self, current_position: Tuple[float, float], 
                                   predicted_position: Tuple[float, float], 
                                   velocity: Tuple[float, float]) -> float:
        """计算碰撞时间"""
        speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
        
        if speed < 0.1:  # 速度太慢
            return float('inf')
        
        # 简化的碰撞时间计算
        # 实际应用中应该考虑具体的碰撞几何
        distance = math.sqrt(
            (predicted_position[0] - current_position[0])**2 + 
            (predicted_position[1] - current_position[1])**2
        )
        
        time_to_collision = distance / speed
        
        return time_to_collision
    
    def get_all_predictions(self) -> List[TrajectoryPrediction]:
        """获取所有对象的预测"""
        predictions = []
        
        for object_id in self.object_trajectories:
            prediction = self.predict_trajectory(object_id)
            if prediction:
                predictions.append(prediction)
        
        return predictions
    
    def cleanup_old_trajectories(self, current_time: float, max_age: float = 5.0):
        """清理旧轨迹"""
        objects_to_remove = []
        
        for object_id, trajectory in self.object_trajectories.items():
            if trajectory and current_time - trajectory[-1].timestamp > max_age:
                objects_to_remove.append(object_id)
        
        for object_id in objects_to_remove:
            del self.object_trajectories[object_id]

class TrajectoryVisualizer:
    """轨迹可视化器"""
    
    def __init__(self):
        self.trajectory_colors = {
            'low_risk': (0, 255, 0),      # 绿色 - 低风险
            'medium_risk': (0, 255, 255), # 黄色 - 中风险
            'high_risk': (0, 0, 255),     # 红色 - 高风险
        }
    
    def draw_trajectory(self, frame: np.ndarray, prediction: TrajectoryPrediction) -> np.ndarray:
        """绘制轨迹"""
        if not prediction.predicted_trajectory:
            return frame
        
        # 确定风险等级颜色
        if prediction.collision_risk < 0.3:
            color = self.trajectory_colors['low_risk']
        elif prediction.collision_risk < 0.7:
            color = self.trajectory_colors['medium_risk']
        else:
            color = self.trajectory_colors['high_risk']
        
        # 绘制预测轨迹
        points = np.array(prediction.predicted_trajectory, np.int32)
        if len(points) > 1:
            cv2.polylines(frame, [points], False, color, 2)
        
        # 绘制当前位置
        current_pos = tuple(map(int, prediction.current_position))
        cv2.circle(frame, current_pos, 5, color, -1)
        
        # 绘制预测位置
        predicted_pos = tuple(map(int, prediction.predicted_position))
        cv2.circle(frame, predicted_pos, 8, color, 2)
        
        # 绘制速度箭头
        self._draw_velocity_arrow(frame, prediction.current_position, prediction.velocity, color)
        
        # 绘制风险信息
        self._draw_risk_info(frame, prediction, color)
        
        return frame
    
    def _draw_velocity_arrow(self, frame: np.ndarray, position: Tuple[float, float], 
                           velocity: Tuple[float, float], color: Tuple[int, int, int]):
        """绘制速度箭头"""
        x, y = position
        vx, vy = velocity
        
        # 计算箭头长度
        arrow_length = min(50, math.sqrt(vx**2 + vy**2) * 10)
        
        if arrow_length > 5:
            # 计算箭头终点
            end_x = int(x + vx * arrow_length)
            end_y = int(y + vy * arrow_length)
            
            # 绘制箭头
            cv2.arrowedLine(frame, (int(x), int(y)), (end_x, end_y), color, 2)
    
    def _draw_risk_info(self, frame: np.ndarray, prediction: TrajectoryPrediction, 
                       color: Tuple[int, int, int]):
        """绘制风险信息"""
        x, y = prediction.current_position
        
        # 风险等级文本
        if prediction.collision_risk < 0.3:
            risk_text = "低风险"
        elif prediction.collision_risk < 0.7:
            risk_text = "中风险"
        else:
            risk_text = "高风险"
        
        # 碰撞时间文本
        if prediction.time_to_collision < float('inf'):
            ttc_text = f"TTC: {prediction.time_to_collision:.1f}s"
        else:
            ttc_text = "TTC: ∞"
        
        # 绘制文本背景
        text_y = int(y) - 20
        cv2.rectangle(frame, (int(x) - 5, text_y - 20), (int(x) + 100, text_y + 5), (0, 0, 0), -1)
        
        # 绘制文本
        cv2.putText(frame, risk_text, (int(x), text_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, ttc_text, (int(x), text_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

# 使用示例
if __name__ == "__main__":
    # 创建轨迹预测器
    predictor = TrajectoryPredictor()
    visualizer = TrajectoryVisualizer()
    
    # 模拟轨迹数据
    current_time = time.time()
    
    # 添加一些模拟轨迹点
    for i in range(10):
        x = 100 + i * 10
        y = 200 + i * 5
        predictor.update_trajectory("object_1", (x-10, y-10, x+10, y+10), current_time + i * 0.1)
    
    # 进行预测
    prediction = predictor.predict_trajectory("object_1")
    
    if prediction:
        print(f"预测结果:")
        print(f"  当前位置: {prediction.current_position}")
        print(f"  预测位置: {prediction.predicted_position}")
        print(f"  速度: {prediction.velocity}")
        print(f"  加速度: {prediction.acceleration}")
        print(f"  碰撞风险: {prediction.collision_risk:.2f}")
        print(f"  碰撞时间: {prediction.time_to_collision:.2f}s")


