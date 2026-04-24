"""
路径规划器 - 增强版

改进点:
1. 占用栅格地图构建（BEV视角）
2. 动态障碍物预测轨迹考虑
3. 安全裕度计算
4. 多方案对比
"""
import logging
import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Direction(str, Enum):
    """行走方向"""
    STOP = "stop"
    FORWARD = "forward"
    LEFT = "left"
    RIGHT = "right"
    SLOW = "slow"
    BACK = "back"


@dataclass
class RouteOption:
    """路线选项"""
    direction: Direction
    safety_score: float      # 安全评分 0-1
    clearance: float         # 通行余量(米)
    reason: str              # 原因说明
    priority: int            # 优先级 1-5


@dataclass
class RoutePlan:
    """路径规划结果"""
    recommended: RouteOption         # 推荐方案
    alternatives: List[RouteOption]  # 备选方案
    tts_instruction: str             # 语音指令
    visual_hint: Optional[Dict] = None  # 可视化提示（给前端）


class OccupancyGrid:
    """
    占用栅格地图
    
    俯视图（BEV）网格划分:
    
         Left    Center   Right
        ┌───────┬────────┬───────┐
    Far │   L3  │   C3   │  R3   │  (3-5米)
        ├───────┼────────┼───────┤
    Mid │   L2  │   C2   │  R2   │  (1.5-3米)
        ├───────┼────────┼───────┤
    Near│   L1  │   C1   │  R1   │  (0-1.5米)
        └───────┴────────┴───────┘
          (-1,0)  (0,0)   (1,0)   ← 用户位置
    
    每个格子状态:
    - FREE (0): 空闲
    - OCCUPIED (1): 被占用
    - UNCERTAIN (0.5): 不确定
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (3, 3),  # (横向, 纵向)
        cell_size: float = 1.5,  # 每格大小(米)
    ):
        self.grid_size = grid_size
        self.cell_size = cell_size
        
        # 初始化网格（全部空闲）
        self.grid = np.zeros(grid_size)
        
        # 置信度网格
        self.confidence = np.ones(grid_size) * 0.5
    
    def update(
        self, 
        obstacles: List[Dict],
        trajectories: List[Dict] = None
    ):
        """
        更新占用网格
        
        Args:
            obstacles: 障碍物列表
            trajectories: 轨迹预测列表
        """
        # 重置
        self.grid.fill(0)
        self.confidence.fill(0.5)
        
        for obs in obstacles:
            self._mark_obstacle(obs)
        
        # 如果有轨迹预测，标记未来位置
        if trajectories:
            for traj in trajectories:
                self._mark_trajectory(traj)
    
    def _mark_obstacle(self, obs: Dict):
        """标记障碍物占用"""
        distance = obs.get("distance", 5.0)
        direction = obs.get("direction", "center")
        obs_type = obs.get("type", "static")
        
        # 距离 → 纵向索引
        if distance < 1.5:
            row = 0  # Near
        elif distance < 3.0:
            row = 1  # Mid
        else:
            row = 2  # Far
        
        # 方向 → 横向索引
        if direction == "left":
            col = 0
        elif direction == "center":
            col = 1
        else:  # right
            col = 2
        
        # 标记占用
        if row < self.grid_size[1] and col < self.grid_size[0]:
            self.grid[col, row] = 1.0
            
            # 置信度（动态障碍物置信度更高）
            if obs_type == "dynamic":
                self.confidence[col, row] = 0.9
            else:
                self.confidence[col, row] = 0.8
            
            # 扩散到相邻格子（安全裕度）
            self._spread_occupation(col, row, 0.5)
    
    def _mark_trajectory(self, traj: Dict):
        """标记轨迹预测的未来位置"""
        predicted_positions = traj.get("predicted_positions", [])
        
        for pos in predicted_positions:
            # pos是归一化坐标 [x, y]
            # 转换到网格坐标
            # 简化映射：x → 横向，y → 纵向
            
            # x: -0.33左, 0中, 0.33右
            if pos[0] < -0.2:
                col = 0
            elif pos[0] < 0.2:
                col = 1
            else:
                col = 2
            
            # y: 0.3近, 0.5中, 0.7远
            if pos[1] > 0.6:
                row = 0
            elif pos[1] > 0.4:
                row = 1
            else:
                row = 2
            
            # 标记（较低置信度，因为是预测）
            if row < self.grid_size[1] and col < self.grid_size[0]:
                self.grid[col, row] = max(self.grid[col, row], 0.7)
                self.confidence[col, row] = max(self.confidence[col, row], 0.6)
    
    def _spread_occupation(self, col: int, row: int, value: float):
        """扩散占用（安全边界）"""
        # 向相邻格子扩散
        for dc in [-1, 0, 1]:
            for dr in [-1, 0, 1]:
                if dc == 0 and dr == 0:
                    continue
                
                nc, nr = col + dc, row + dr
                
                if 0 <= nc < self.grid_size[0] and 0 <= nr < self.grid_size[1]:
                    # 只扩散到空闲格子
                    if self.grid[nc, nr] < value:
                        self.grid[nc, nr] = value
                        self.confidence[nc, nr] = 0.5
    
    def is_free(self, col: int, row: int, threshold: float = 0.3) -> bool:
        """检查格子是否空闲"""
        if not (0 <= col < self.grid_size[0] and 0 <= row < self.grid_size[1]):
            return False
        
        return self.grid[col, row] < threshold
    
    def get_zone_status(self, zone: str) -> Dict:
        """
        获取区域状态
        
        Args:
            zone: "left", "center", "right"
        Returns:
            状态字典
        """
        col_map = {"left": 0, "center": 1, "right": 2}
        col = col_map.get(zone, 1)
        
        # 统计该列的占用情况
        occupied_cells = 0
        total_cells = self.grid_size[1]
        
        for row in range(total_cells):
            if self.grid[col, row] > 0.3:
                occupied_cells += 1
        
        occupation_rate = occupied_cells / total_cells
        
        # 最近障碍物距离
        min_distance = None
        for row in range(total_cells):
            if self.grid[col, row] > 0.5:
                # row → 距离
                if row == 0:
                    dist = 0.75  # Near中心
                elif row == 1:
                    dist = 2.25  # Mid中心
                else:
                    dist = 4.0   # Far中心
                
                if min_distance is None or dist < min_distance:
                    min_distance = dist
        
        return {
            "zone": zone,
            "occupation_rate": occupation_rate,
            "is_free": occupation_rate < 0.33,
            "is_blocked": occupation_rate > 0.66,
            "min_distance": min_distance,
            "clearance": min_distance if min_distance else 5.0
        }


class EnhancedRoutePlanner:
    """
    增强型路径规划器
    
    策略:
    1. 构建占用栅格
    2. 评估各方向安全性
    3. 考虑动态障碍物趋势
    4. 生成多方案建议
    """
    
    def __init__(self):
        self.grid = OccupancyGrid()
    
    def plan(
        self,
        obstacles: List[Dict],
        trajectories: List[Dict] = None,
        current_direction: str = "forward"
    ) -> RoutePlan:
        """
        规划路径
        
        Args:
            obstacles: 障碍物列表
            trajectories: 轨迹预测列表
            current_direction: 当前行进方向
        Returns:
            路径规划结果
        """
        
        # ===== Step 1: 更新占用网格 =====
        self.grid.update(obstacles, trajectories)
        
        # ===== Step 2: 评估各区域状态 =====
        left_status = self.grid.get_zone_status("left")
        center_status = self.grid.get_zone_status("center")
        right_status = self.grid.get_zone_status("right")
        
        logger.debug(
            f"区域状态: Left={left_status['occupation_rate']:.2f}, "
            f"Center={center_status['occupation_rate']:.2f}, "
            f"Right={right_status['occupation_rate']:.2f}"
        )
        
        # ===== Step 3: 生成候选方案 =====
        options = []
        
        # 方案1: 直行
        if center_status["is_free"]:
            options.append(RouteOption(
                direction=Direction.FORWARD,
                safety_score=self._calc_safety_score(center_status, obstacles, "center"),
                clearance=center_status["clearance"],
                reason="前方道路畅通",
                priority=3
            ))
        
        # 方案2: 左转
        if left_status["is_free"]:
            options.append(RouteOption(
                direction=Direction.LEFT,
                safety_score=self._calc_safety_score(left_status, obstacles, "left"),
                clearance=left_status["clearance"],
                reason=f"左侧通行空间{left_status['clearance']:.1f}米",
                priority=2
            ))
        
        # 方案3: 右转
        if right_status["is_free"]:
            options.append(RouteOption(
                direction=Direction.RIGHT,
                safety_score=self._calc_safety_score(right_status, obstacles, "right"),
                clearance=right_status["clearance"],
                reason=f"右侧通行空间{right_status['clearance']:.1f}米",
                priority=2
            ))
        
        # 方案4: 减速
        if not center_status["is_blocked"]:
            options.append(RouteOption(
                direction=Direction.SLOW,
                safety_score=0.6,
                clearance=center_status["clearance"],
                reason="前方有障碍物，建议减速通过",
                priority=1
            ))
        
        # 方案5: 停止（兜底）
        options.append(RouteOption(
            direction=Direction.STOP,
            safety_score=0.3,
            clearance=0.0,
            reason="前方拥挤，请停止前进",
            priority=5 if not options else 1
        ))
        
        # ===== Step 4: 特殊规则调整 =====
        options = self._apply_special_rules(
            options, obstacles, trajectories, 
            left_status, center_status, right_status
        )
        
        # ===== Step 5: 选择最佳方案 =====
        if not options:
            # 无有效方案，默认停止
            recommended = RouteOption(
                direction=Direction.STOP,
                safety_score=0.2,
                clearance=0.0,
                reason="无安全通行路径",
                priority=5
            )
            alternatives = []
        else:
            # 按安全评分排序
            options.sort(key=lambda x: (x.safety_score, x.clearance), reverse=True)
            recommended = options[0]
            alternatives = options[1:4]  # 最多3个备选
        
        # ===== Step 6: 生成语音指令 =====
        tts = self._generate_instruction(recommended, obstacles)
        
        # ===== Step 7: 可视化提示（给前端） =====
        visual_hint = {
            "grid": self.grid.grid.tolist(),
            "recommended_direction": recommended.direction.value,
            "arrow_angle": self._direction_to_angle(recommended.direction),
            "clearance_zones": {
                "left": left_status["clearance"],
                "center": center_status["clearance"],
                "right": right_status["clearance"]
            }
        }
        
        return RoutePlan(
            recommended=recommended,
            alternatives=alternatives,
            tts_instruction=tts,
            visual_hint=visual_hint
        )
    
    def _calc_safety_score(
        self, 
        zone_status: Dict, 
        obstacles: List[Dict],
        zone: str
    ) -> float:
        """计算安全评分"""
        
        # 基础分（基于占用率）
        base_score = 1.0 - zone_status["occupation_rate"]
        
        # 通行余量加成
        clearance = zone_status["clearance"]
        if clearance > 3.0:
            clearance_bonus = 0.2
        elif clearance > 2.0:
            clearance_bonus = 0.1
        elif clearance > 1.0:
            clearance_bonus = 0.05
        else:
            clearance_bonus = 0.0
        
        # 动态障碍物惩罚
        dynamic_penalty = 0.0
        for obs in obstacles:
            if obs.get("direction") == zone and obs.get("type") == "dynamic":
                if obs.get("distance", 5.0) < 2.0:
                    dynamic_penalty += 0.15
        
        # 地面异常惩罚
        ground_penalty = 0.0
        for obs in obstacles:
            if obs.get("direction") == zone and obs.get("type") == "ground":
                ground_penalty += 0.1
        
        total_score = base_score + clearance_bonus - dynamic_penalty - ground_penalty
        
        return max(0.0, min(1.0, total_score))
    
    def _apply_special_rules(
        self,
        options: List[RouteOption],
        obstacles: List[Dict],
        trajectories: List[Dict],
        left_status: Dict,
        center_status: Dict,
        right_status: Dict
    ) -> List[RouteOption]:
        """应用特殊规则"""
        
        # 规则1: 如果有快速接近的动态障碍物，禁止该方向
        if trajectories:
            for traj in trajectories:
                ttc = traj.get("ttc")
                direction = traj.get("direction")
                obj_dir = traj.get("object_direction", "center")  # 假设轨迹包含物体方向
                
                if ttc and ttc < 2.0 and direction == "approaching":
                    # 移除该方向的选项
                    if obj_dir == "left":
                        options = [o for o in options if o.direction != Direction.LEFT]
                    elif obj_dir == "right":
                        options = [o for o in options if o.direction != Direction.RIGHT]
                    elif obj_dir == "center":
                        options = [o for o in options if o.direction != Direction.FORWARD]
        
        # 规则2: 如果两侧都不安全，强制停止
        if left_status["is_blocked"] and right_status["is_blocked"]:
            # 移除左右选项，提升停止优先级
            options = [o for o in options if o.direction not in [Direction.LEFT, Direction.RIGHT]]
            
            for opt in options:
                if opt.direction == Direction.STOP:
                    opt.priority = 5
                    opt.reason = "左右两侧均不可通行，请停止"
        
        # 规则3: 如果中心有地面异常，优先绕行
        center_ground_hazards = [
            o for o in obstacles
            if o.get("direction") == "center" and o.get("type") == "ground"
        ]
        if center_ground_hazards:
            for opt in options:
                if opt.direction == Direction.FORWARD:
                    opt.safety_score *= 0.5  # 降低直行安全性
                    opt.reason = "前方有地面异常，建议绕行"
        
        # 规则4: 优先保持当前方向（如果安全）
        # （避免频繁变向）
        # 这里简化处理
        
        return options
    
    def _generate_instruction(
        self, 
        option: RouteOption, 
        obstacles: List[Dict]
    ) -> str:
        """生成语音指令"""
        
        direction_map = {
            Direction.STOP: "请立即停止",
            Direction.FORWARD: "可以继续前进",
            Direction.LEFT: "请向左绕行",
            Direction.RIGHT: "请向右绕行",
            Direction.SLOW: "请放慢脚步",
            Direction.BACK: "请后退"
        }
        
        action = direction_map.get(option.direction, "请注意")
        
        # 添加原因
        if option.reason:
            instruction = f"{action}，{option.reason}。"
        else:
            instruction = f"{action}。"
        
        # 添加距离信息（如果有）
        if option.clearance > 0 and option.clearance < 3.0:
            instruction = instruction.rstrip("。")
            instruction += f"，通行空间约{option.clearance:.1f}米。"
        
        return instruction
    
    def _direction_to_angle(self, direction: Direction) -> int:
        """方向转角度（用于前端箭头显示）"""
        angle_map = {
            Direction.FORWARD: 0,
            Direction.LEFT: -45,
            Direction.RIGHT: 45,
            Direction.BACK: 180,
            Direction.STOP: 0,
            Direction.SLOW: 0
        }
        return angle_map.get(direction, 0)