"""
轨迹预测 - 卡尔曼滤波器

实现:
1. 卡尔曼滤波状态估计
2. 速度/加速度计算
3. TTC预测
4. 轨迹外推
"""
import logging
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
from collections import deque
import time

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryState:
    """轨迹状态"""
    # 位置
    x: float              # 水平位置(归一化)
    y: float              # 垂直位置(归一化)
    
    # 速度
    vx: float             # 水平速度
    vy: float             # 垂直速度
    
    # 加速度
    ax: float = 0.0
    ay: float = 0.0
    
    # 协方差（不确定性）
    covariance: np.ndarray = None
    
    # 时间戳
    timestamp: float = 0.0


@dataclass
class TrajectoryPrediction:
    """轨迹预测结果"""
    current_state: TrajectoryState
    
    # 运动信息
    speed: float              # 速度(m/s)
    acceleration: float       # 加速度(m/s²)
    direction: str            # approaching/receding/stationary
    
    # 预测
    ttc: Optional[float]      # 碰撞时间(秒)
    predicted_positions: List[Tuple[float, float]]  # 未来位置序列
    
    # 危险评估
    danger_score: float       # 0-1
    confidence: float         # 预测置信度


class KalmanTracker:
    """
    卡尔曼滤波器轨迹跟踪
    
    状态向量: X = [x, y, vx, vy, ax, ay]^T
    
    状态转移方程:
    X(k) = F·X(k-1) + w
    
    观测方程:
    Z(k) = H·X(k) + v
    
    其中:
    - F: 状态转移矩阵
    - H: 观测矩阵
    - w: 过程噪声
    - v: 观测噪声
    """
    
    def __init__(
        self,
        process_noise: float = 0.01,      # 过程噪声
        measurement_noise: float = 0.1,   # 观测噪声
        pixel_to_meter: float = 0.01,     # 像素→米转换系数
    ):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.pixel_to_meter = pixel_to_meter
        
        # 状态
        self.state: Optional[TrajectoryState] = None
        
        # 历史轨迹
        self.history: deque = deque(maxlen=30)
        
        # 初始化标志
        self.initialized = False
    
    def update(
        self, 
        measurement: Tuple[float, float],  # (x, y) 归一化坐标
        timestamp: float,
        distance: Optional[float] = None   # 实际距离(米)，用于校准
    ) -> TrajectoryPrediction:
        """
        更新轨迹
        
        Args:
            measurement: 观测位置 (x, y)
            timestamp: 时间戳(秒)
            distance: 实际距离(米)
        Returns:
            预测结果
        """
        
        if not self.initialized:
            # 初始化
            self._initialize(measurement, timestamp)
            return self._build_prediction()
        
        # 计算时间间隔
        dt = timestamp - self.state.timestamp
        if dt <= 0:
            logger.warning("时间戳异常")
            return self._build_prediction()
        
        # ===== Kalman Filter =====
        
        # 1. 预测步骤
        predicted_state = self._predict(dt)
        
        # 2. 更新步骤
        updated_state = self._update(predicted_state, measurement)
        
        # 3. 保存状态
        updated_state.timestamp = timestamp
        self.state = updated_state
        
        # 4. 加入历史
        self.history.append((timestamp, measurement[0], measurement[1]))
        
        # 5. 构建预测
        prediction = self._build_prediction(distance)
        
        return prediction
    
    def _initialize(self, measurement: Tuple[float, float], timestamp: float):
        """初始化状态"""
        self.state = TrajectoryState(
            x=measurement[0],
            y=measurement[1],
            vx=0.0,
            vy=0.0,
            ax=0.0,
            ay=0.0,
            timestamp=timestamp,
            covariance=np.eye(6) * 1.0  # 初始协方差
        )
        self.initialized = True
        logger.debug("卡尔曼滤波器已初始化")
    
    def _predict(self, dt: float) -> TrajectoryState:
        """
        预测步骤
        
        状态转移方程（匀加速运动）:
        x(k) = x(k-1) + vx·dt + 0.5·ax·dt²
        y(k) = y(k-1) + vy·dt + 0.5·ay·dt²
        vx(k) = vx(k-1) + ax·dt
        vy(k) = vy(k-1) + ay·dt
        ax(k) = ax(k-1)
        ay(k) = ay(k-1)
        """
        
        # 状态转移矩阵 F
        F = np.array([
            [1, 0, dt, 0,  0.5*dt**2, 0],
            [0, 1, 0,  dt, 0,         0.5*dt**2],
            [0, 0, 1,  0,  dt,        0],
            [0, 0, 0,  1,  0,         dt],
            [0, 0, 0,  0,  1,         0],
            [0, 0, 0,  0,  0,         1]
        ])
        
        # 当前状态向量
        X = np.array([
            self.state.x,
            self.state.y,
            self.state.vx,
            self.state.vy,
            self.state.ax,
            self.state.ay
        ])
        
        # 预测
        X_pred = F @ X
        
        # 协方差预测
        Q = np.eye(6) * self.process_noise  # 过程噪声协方差
        P_pred = F @ self.state.covariance @ F.T + Q
        
        # 构建预测状态
        predicted = TrajectoryState(
            x=X_pred[0],
            y=X_pred[1],
            vx=X_pred[2],
            vy=X_pred[3],
            ax=X_pred[4],
            ay=X_pred[5],
            covariance=P_pred,
            timestamp=self.state.timestamp
        )
        
        return predicted
    
    def _update(
        self, 
        predicted: TrajectoryState, 
        measurement: Tuple[float, float]
    ) -> TrajectoryState:
        """
        更新步骤（融合观测）
        
        卡尔曼增益:
        K = P·H^T·(H·P·H^T + R)^(-1)
        
        状态更新:
        X = X_pred + K·(Z - H·X_pred)
        
        协方差更新:
        P = (I - K·H)·P_pred
        """
        
        # 观测矩阵 H（只观测位置）
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])
        
        # 观测向量
        Z = np.array([measurement[0], measurement[1]])
        
        # 预测状态向量
        X_pred = np.array([
            predicted.x,
            predicted.y,
            predicted.vx,
            predicted.vy,
            predicted.ax,
            predicted.ay
        ])
        
        # 观测噪声协方差
        R = np.eye(2) * self.measurement_noise
        
        # 卡尔曼增益
        S = H @ predicted.covariance @ H.T + R
        K = predicted.covariance @ H.T @ np.linalg.inv(S)
        
        # 创新（残差）
        innovation = Z - H @ X_pred
        
        # 状态更新
        X_updated = X_pred + K @ innovation
        
        # 协方差更新
        P_updated = (np.eye(6) - K @ H) @ predicted.covariance
        
        # 构建更新状态
        updated = TrajectoryState(
            x=X_updated[0],
            y=X_updated[1],
            vx=X_updated[2],
            vy=X_updated[3],
            ax=X_updated[4],
            ay=X_updated[5],
            covariance=P_updated,
            timestamp=predicted.timestamp
        )
        
        return updated
    
    def _build_prediction(self, distance: Optional[float] = None) -> TrajectoryPrediction:
        """构建预测结果"""
        
        if not self.state:
            return TrajectoryPrediction(
                current_state=None,
                speed=0,
                acceleration=0,
                direction="stationary",
                ttc=None,
                predicted_positions=[],
                danger_score=0,
                confidence=0
            )
        
        # ===== 速度计算 =====
        # 像素速度 → 实际速度
        vx_pixel = self.state.vx
        vy_pixel = self.state.vy
        
        speed_pixel = np.sqrt(vx_pixel**2 + vy_pixel**2)
        speed_meter = speed_pixel * self.pixel_to_meter
        
        # ===== 加速度计算 =====
        ax_pixel = self.state.ax
        ay_pixel = self.state.ay
        
        accel_pixel = np.sqrt(ax_pixel**2 + ay_pixel**2)
        accel_meter = accel_pixel * self.pixel_to_meter
        
        # ===== 运动方向判断 =====
        # 基于y方向速度（y增大 → 靠近）
        if vy_pixel > 0.01:
            direction = "approaching"
        elif vy_pixel < -0.01:
            direction = "receding"
        else:
            direction = "stationary"
        
        # ===== TTC计算 =====
        ttc = None
        
        if distance and direction == "approaching":
            if speed_meter > 0.1:
                # 简单模型: TTC = distance / speed
                ttc = distance / speed_meter
                
                # 考虑加速度的修正
                if accel_meter > 0.1:
                    # TTC = (-v + sqrt(v² + 2ad)) / a
                    discriminant = speed_meter**2 + 2 * accel_meter * distance
                    if discriminant > 0:
                        ttc = (-speed_meter + np.sqrt(discriminant)) / accel_meter
                
                # 合理性检查
                if ttc < 0 or ttc > 30:
                    ttc = None
        
        # ===== 轨迹外推（预测未来1秒内的位置）=====
        predicted_positions = []
        
        for t in np.linspace(0.1, 1.0, 10):
            future_x = self.state.x + self.state.vx * t + 0.5 * self.state.ax * t**2
            future_y = self.state.y + self.state.vy * t + 0.5 * self.state.ay * t**2
            predicted_positions.append((future_x, future_y))
        
        # ===== 危险评分 =====
        danger_score = self._calculate_danger(
            speed_meter, accel_meter, ttc, direction, distance
        )
        
        # ===== 预测置信度（基于协方差） =====
        if self.state.covariance is not None:
            position_variance = self.state.covariance[0, 0] + self.state.covariance[1, 1]
            confidence = 1.0 / (1.0 + position_variance)
        else:
            confidence = 0.5
        
        return TrajectoryPrediction(
            current_state=self.state,
            speed=round(speed_meter, 3),
            acceleration=round(accel_meter, 3),
            direction=direction,
            ttc=round(ttc, 2) if ttc else None,
            predicted_positions=predicted_positions,
            danger_score=round(danger_score, 3),
            confidence=round(confidence, 3)
        )
    
    def _calculate_danger(
        self,
        speed: float,
        accel: float,
        ttc: Optional[float],
        direction: str,
        distance: Optional[float]
    ) -> float:
        """计算危险评分"""
        
        if direction == "receding":
            return 0.0
        
        if direction == "stationary":
            return 0.1
        
        score = 0.0
        
        # TTC评分（权重最大）
        if ttc is not None:
            if ttc <= 1.0:
                score += 0.6
            elif ttc <= 2.0:
                score += 0.4
            elif ttc <= 3.0:
                score += 0.25
            elif ttc <= 5.0:
                score += 0.15
        
        # 速度评分
        if speed > 3.0:
            score += 0.2
        elif speed > 1.5:
            score += 0.12
        elif speed > 0.5:
            score += 0.06
        
        # 加速度评分
        if accel > 1.0:
            score += 0.15
        elif accel > 0.3:
            score += 0.08
        
        # 距离评分
        if distance:
            if distance < 1.0:
                score += 0.05
        
        return min(score, 1.0)