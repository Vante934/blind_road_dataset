#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
精确定位模块
实现GPS定位优化和多传感器融合
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import math

@dataclass
class GPSData:
    """GPS数据"""
    latitude: float
    longitude: float
    altitude: float = 0.0
    accuracy: float = 10.0  # 米
    speed: float = 0.0  # 米/秒
    bearing: float = 0.0  # 度
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class SensorData:
    """传感器数据"""
    accelerometer: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, z (m/s²)
    gyroscope: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, z (rad/s)
    magnetometer: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, z (μT)
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class PreciseLocation:
    """精确定位结果"""
    latitude: float
    longitude: float
    altitude: float
    accuracy: float  # 米
    confidence: float  # 0-1
    timestamp: datetime
    source: str  # 'gps', 'fused', 'estimated'


class KalmanFilter:
    """卡尔曼滤波器 - 用于GPS数据平滑"""
    
    def __init__(self, process_noise=0.1, measurement_noise=5.0):
        """
        初始化卡尔曼滤波器
        
        Args:
            process_noise: 过程噪声（位置变化的不确定性）
            measurement_noise: 测量噪声（GPS测量的不确定性）
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # 状态向量: [latitude, longitude, velocity_lat, velocity_lng]
        self.state = np.zeros(4)
        self.covariance = np.eye(4) * 1000  # 初始不确定性很大
        
        self.initialized = False
    
    def update(self, measurement: Tuple[float, float], dt: float = 1.0) -> Tuple[float, float]:
        """
        更新滤波器
        
        Args:
            measurement: (latitude, longitude) 测量值
            dt: 时间间隔（秒）
        
        Returns:
            (filtered_latitude, filtered_longitude) 滤波后的位置
        """
        if not self.initialized:
            self.state[0] = measurement[0]
            self.state[1] = measurement[1]
            self.initialized = True
            return measurement
        
        # 预测步骤
        # 状态转移矩阵（假设匀速运动）
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # 预测状态
        predicted_state = F @ self.state
        
        # 预测协方差
        Q = np.eye(4) * self.process_noise
        predicted_covariance = F @ self.covariance @ F.T + Q
        
        # 更新步骤
        # 观测矩阵（只观测位置）
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # 测量残差
        measurement_residual = np.array(measurement) - H @ predicted_state
        
        # 残差协方差
        R = np.eye(2) * self.measurement_noise
        S = H @ predicted_covariance @ H.T + R
        
        # 卡尔曼增益
        K = predicted_covariance @ H.T @ np.linalg.inv(S)
        
        # 更新状态
        self.state = predicted_state + K @ measurement_residual
        
        # 更新协方差
        self.covariance = (np.eye(4) - K @ H) @ predicted_covariance
        
        return (self.state[0], self.state[1])


class PreciseLocationTracker:
    """精确定位追踪器"""
    
    def __init__(self):
        self.kalman_filter = KalmanFilter(process_noise=0.1, measurement_noise=5.0)
        self.last_location: Optional[PreciseLocation] = None
        self.location_history: List[PreciseLocation] = []
        self.max_history = 100
    
    def update(self, gps_data: GPSData, sensor_data: Optional[SensorData] = None) -> PreciseLocation:
        """
        更新位置
        
        Args:
            gps_data: GPS数据
            sensor_data: 传感器数据（可选）
        
        Returns:
            精确定位结果
        """
        # 1. GPS数据预处理
        filtered_lat, filtered_lng = self.kalman_filter.update(
            (gps_data.latitude, gps_data.longitude)
        )
        
        # 2. 传感器融合（如果有传感器数据）
        if sensor_data:
            # 使用传感器数据辅助定位
            fused_location = self._fuse_sensors(
                filtered_lat, filtered_lng,
                gps_data, sensor_data
            )
            final_lat, final_lng = fused_location
            confidence = 0.9
            accuracy = max(3.0, gps_data.accuracy * 0.7)  # 传感器融合可以提高精度
        else:
            final_lat, final_lng = filtered_lat, filtered_lng
            confidence = 0.7
            accuracy = gps_data.accuracy
        
        # 3. 历史数据验证
        if self.last_location:
            # 检查位置变化是否合理
            distance = self._calculate_distance(
                self.last_location.latitude, self.last_location.longitude,
                final_lat, final_lng
            )
            
            # 如果位置变化过大，可能是GPS跳变，降低置信度
            if distance > 50:  # 50米
                confidence *= 0.5
                accuracy *= 1.5
        
        # 4. 创建精确定位结果
        precise_location = PreciseLocation(
            latitude=final_lat,
            longitude=final_lng,
            altitude=gps_data.altitude,
            accuracy=accuracy,
            confidence=confidence,
            timestamp=datetime.now(),
            source='fused' if sensor_data else 'gps'
        )
        
        # 5. 更新历史
        self.last_location = precise_location
        self.location_history.append(precise_location)
        if len(self.location_history) > self.max_history:
            self.location_history.pop(0)
        
        return precise_location
    
    def _fuse_sensors(self, lat: float, lng: float, 
                     gps_data: GPSData, sensor_data: SensorData) -> Tuple[float, float]:
        """
        传感器融合
        
        Args:
            lat, lng: GPS位置
            gps_data: GPS数据
            sensor_data: 传感器数据
        
        Returns:
            融合后的位置 (lat, lng)
        """
        # 简单的传感器融合算法
        # 使用加速度计估计位移
        
        acc_x, acc_y, acc_z = sensor_data.accelerometer
        
        # 如果加速度很小，说明可能是静止状态
        acc_magnitude = math.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        
        if acc_magnitude < 0.5:  # 静止状态
            # 保持当前位置
            return (lat, lng)
        
        # 如果有速度信息，可以用于位置估计
        if gps_data.speed > 0:
            # 使用速度和方向估计位置
            # 这里简化处理，实际应该使用更复杂的算法
            dt = 1.0  # 假设1秒
            distance = gps_data.speed * dt
            
            # 将距离转换为经纬度偏移（简化计算）
            lat_offset = distance * math.cos(math.radians(gps_data.bearing)) / 111320.0
            lng_offset = distance * math.sin(math.radians(gps_data.bearing)) / (111320.0 * math.cos(math.radians(lat)))
            
            return (lat + lat_offset, lng + lng_offset)
        
        return (lat, lng)
    
    def _calculate_distance(self, lat1: float, lng1: float, 
                           lat2: float, lng2: float) -> float:
        """
        计算两点之间的距离（米）
        使用Haversine公式
        """
        R = 6371000  # 地球半径（米）
        
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lng2 - lng1)
        
        a = (math.sin(delta_phi / 2)**2 +
             math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def get_current_location(self) -> Optional[PreciseLocation]:
        """获取当前位置"""
        return self.last_location
    
    def get_location_history(self) -> List[PreciseLocation]:
        """获取位置历史"""
        return self.location_history.copy()


# 全局定位追踪器实例
_location_tracker = None

def get_location_tracker() -> PreciseLocationTracker:
    """获取定位追踪器实例（单例模式）"""
    global _location_tracker
    if _location_tracker is None:
        _location_tracker = PreciseLocationTracker()
    return _location_tracker


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("精确定位模块测试")
    print("=" * 60)
    
    tracker = PreciseLocationTracker()
    
    # 模拟GPS数据
    gps_data = GPSData(
        latitude=39.9042,
        longitude=116.4074,
        accuracy=10.0,
        speed=1.5,
        bearing=45.0
    )
    
    # 模拟传感器数据
    sensor_data = SensorData(
        accelerometer=(0.1, 0.2, 9.8),
        gyroscope=(0.01, 0.02, 0.01)
    )
    
    # 更新位置
    location = tracker.update(gps_data, sensor_data)
    print(f"✅ 定位成功:")
    print(f"   位置: ({location.latitude:.6f}, {location.longitude:.6f})")
    print(f"   精度: {location.accuracy:.2f}米")
    print(f"   置信度: {location.confidence:.2f}")
    print(f"   来源: {location.source}")
    
    print("=" * 60)
