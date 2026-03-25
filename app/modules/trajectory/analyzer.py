"""
轨迹分析器

功能：
1. 维护历史轨迹
2. 计算速度、加速度
3. 预测碰撞时间(TTC)
"""
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field


class TrendType:
    """运动趋势类型"""
    APPROACHING = "approaching"  # 接近
    RECEDING = "receding"      # 远离
    STATIONARY = "stationary"    # 静止


@dataclass
class TrajectoryResult:
    """轨迹分析结果"""
    trend: str  # 运动趋势
    speed: float  # 速度 (m/s)
    ttc: float  # 碰撞时间 (s)
    trajectory_danger_score: float  # 轨迹危险评分 (0-1)


class TrajectoryAnalyzer:
    """轨迹分析器"""
    
    def __init__(self, window_size: int = 20):
        """
        初始化轨迹分析器
        
        Args:
            window_size: 历史数据窗口大小
        """
        self.window_size = window_size
        self.trajectories: Dict[str, List[Dict]] = {}  # {object_id: [{distance, timestamp}]}
    
    def update(self, device_id: str, distance: float, timestamp: float, direction: str) -> TrajectoryResult:
        """
        更新轨迹并分析
        
        Args:
            device_id: 设备或物体ID
            distance: 距离 (m)
            timestamp: 时间戳
            direction: 方向
        Returns:
            TrajectoryResult: 轨迹分析结果
        """
        # 初始化轨迹数据
        if device_id not in self.trajectories:
            self.trajectories[device_id] = []
        
        # 添加新数据点
        self.trajectories[device_id].append({
            'distance': distance,
            'timestamp': timestamp,
            'direction': direction
        })
        
        # 保持窗口大小
        if len(self.trajectories[device_id]) > self.window_size:
            self.trajectories[device_id] = self.trajectories[device_id][-self.window_size:]
        
        # 分析轨迹
        return self._analyze_trajectory(self.trajectories[device_id])
    
    def _analyze_trajectory(self, trajectory: List[Dict]) -> TrajectoryResult:
        """
        分析轨迹数据
        
        Args:
            trajectory: 轨迹数据列表
        Returns:
            TrajectoryResult: 分析结果
        """
        if len(trajectory) < 2:
            # 数据不足，返回默认值
            return TrajectoryResult(
                trend=TrendType.STATIONARY,
                speed=0.0,
                ttc=float('inf'),
                trajectory_danger_score=0.0
            )
        
        # 计算速度
        recent = trajectory[-5:]  # 最近5个点
        if len(recent) < 2:
            recent = trajectory
        
        speed = 0.0
        total_time = 0.0
        for i in range(1, len(recent)):
            dt = recent[i]['timestamp'] - recent[i-1]['timestamp']
            if dt > 0:
                dd = abs(recent[i]['distance'] - recent[i-1]['distance'])
                speed += dd / (dt / 1000)  # 转换为秒
                total_time += 1
        
        if total_time > 0:
            speed /= total_time
        
        # 计算趋势
        first_dist = trajectory[0]['distance']
        last_dist = trajectory[-1]['distance']
        
        if abs(last_dist - first_dist) < 0.1:
            trend = TrendType.STATIONARY
        elif last_dist < first_dist:
            trend = TrendType.APPROACHING
        else:
            trend = TrendType.RECEDING
        
        # 计算TTC
        ttc = float('inf')
        if trend == TrendType.APPROACHING and speed > 0.1:
            ttc = last_dist / speed
        
        # 计算危险评分
        danger_score = 0.0
        if trend == TrendType.APPROACHING:
            # 距离越近、速度越快、TTC越短，危险越高
            if last_dist < 0.5:
                danger_score = 1.0
            elif last_dist < 1.5:
                danger_score = 0.8
            elif last_dist < 3.0:
                danger_score = 0.6
            
            # TTC因素
            if ttc < 1.0:
                danger_score = 1.0
            elif ttc < 2.0:
                danger_score = max(danger_score, 0.8)
            elif ttc < 3.0:
                danger_score = max(danger_score, 0.6)
        
        return TrajectoryResult(
            trend=trend,
            speed=speed,
            ttc=ttc,
            trajectory_danger_score=danger_score
        )