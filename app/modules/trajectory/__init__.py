"""
轨迹分析模块

功能：
1. 维护历史轨迹
2. 计算速度、加速度
3. 预测碰撞时间(TTC)
"""
import logging
import time
from typing import Dict, Any
from app.modules import BaseModule, ModuleResult
from app.config import settings

logger = logging.getLogger(__name__)

if settings.MODULE_TRAJECTORY_ENABLED:
    try:
        from .analyzer import TrajectoryAnalyzer
        TRAJECTORY_AVAILABLE = True
    except ImportError as e:
        logger.error(f"轨迹模块依赖缺失: {e}")
        TRAJECTORY_AVAILABLE = False
else:
    TRAJECTORY_AVAILABLE = False


class TrajectoryModule(BaseModule):
    """
    轨迹分析模块
    
    输入格式:
    {
        "device_id": "device_001",
        "obstacles": [...],  # 来自视觉模块的障碍物列表
        "distance": 2.0,     # ToF距离（兜底）
        "timestamp": 1234567890
    }
    
    输出格式:
    {
        "trajectories": [
            {
                "object_id": "device_001_person_center",
                "trend": "approaching",
                "speed": 1.2,
                "ttc": 1.5,
                "danger_score": 0.7
            }
        ]
    }
    """
    
    def __init__(self):
        super().__init__("trajectory")
        
        if TRAJECTORY_AVAILABLE and settings.MODULE_TRAJECTORY_ENABLED:
            self.analyzer = TrajectoryAnalyzer(
                window_size=settings.TRAJECTORY_WINDOW_SIZE
            )
            logger.info("✅ 轨迹分析模块初始化成功")
        else:
            self.analyzer = None
            logger.info("⚪ 轨迹分析模块未启用")
    
    async def process(self, input_data: Dict[str, Any]) -> ModuleResult:
        """分析轨迹"""
        
        if not self.enabled or not TRAJECTORY_AVAILABLE or not self.analyzer:
            return ModuleResult(
                module_name=self.module_name,
                success=True,
                data={"trajectories": []},
                metadata={"enabled": False}
            )
        
        try:
            device_id = input_data.get("device_id", "unknown")
            timestamp = input_data.get("timestamp", time.time() * 1000)
            
            trajectories = []
            
            # 对每个动态障碍物分析轨迹
            obstacles = input_data.get("obstacles", [])
            for obs in obstacles:
                if obs.get("type") == "dynamic" and obs.get("distance"):
                    obj_id = f"{device_id}_{obs['class']}_{obs['direction']}"
                    
                    result = self.analyzer.update(
                        device_id=obj_id,
                        distance=obs["distance"],
                        timestamp=timestamp,
                        direction=obs["direction"]
                    )
                    
                    trajectories.append({
                        "object_id": obj_id,
                        "object_class": obs["class"],
                        "distance": obs["distance"],
                        "direction": obs["direction"],
                        "trend": result.trend,
                        "speed": result.speed,
                        "ttc": result.ttc,
                        "danger_score": result.trajectory_danger_score
                    })
            
            # 如果有ToF距离但没有视觉障碍物，也分析
            if not trajectories and input_data.get("distance"):
                result = self.analyzer.update(
                    device_id=device_id,
                    distance=input_data["distance"],
                    timestamp=timestamp,
                    direction=input_data.get("direction", "rear")
                )
                
                direction = input_data.get("direction", "rear")
                trajectories.append({
                    "object_id": device_id,
                    "object_class": "unknown",
                    "distance": input_data["distance"],
                    "direction": direction,
                    "trend": result.trend,
                    "speed": result.speed,
                    "ttc": result.ttc,
                    "danger_score": result.trajectory_danger_score
                })
            
            return ModuleResult(
                module_name=self.module_name,
                success=True,
                data={"trajectories": trajectories},
                metadata={"count": len(trajectories)}
            )
        
        except Exception as e:
            logger.error(f"轨迹分析异常: {e}", exc_info=True)
            return ModuleResult(
                module_name=self.module_name,
                success=False,
                data={"trajectories": []},
                error=str(e)
            )


# 单例导出
trajectory_module = TrajectoryModule()