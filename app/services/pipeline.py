"""
处理流程 - 集成版

整合所有模块，实现端到端的盲道导航系统处理流程
"""
import logging
from typing import Dict, Any, Optional

from app.config import settings
from app.models.schemas import SensorData, PipelineResponse
from app.services.integrated_pipeline import IntegratedPipeline

logger = logging.getLogger(__name__)

# 全局Pipeline实例
pipeline = None


def init_pipeline():
    """初始化Pipeline"""
    global pipeline
    if pipeline is None:
        pipeline = IntegratedPipeline()
    return pipeline


async def process_sensor_data(sensor_data: SensorData) -> PipelineResponse:
    """
    处理传感器数据
    
    Args:
        sensor_data: 传感器数据
    Returns:
        处理结果
    """
    try:
        # 初始化Pipeline
        pipeline = init_pipeline()
        
        # 处理数据
        result = await pipeline.process(sensor_data)
        
        # 构建响应
        response = PipelineResponse(
            success=result.success,
            message="处理成功" if result.success else f"处理失败: {result.error}",
            data={
                "obstacles": result.vision_obstacles,
                "sound": result.sound_result,
                "trajectories": result.trajectories,
                "warning": result.warning_decision,
                "route": result.route_plan,
                "processing_time": result.processing_time,
                "total_time": result.total_time
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"处理异常: {e}")
        return PipelineResponse(
            success=False,
            message=f"处理失败: {str(e)}",
            data={}
        )


async def get_system_status() -> Dict[str, Any]:
    """
    获取系统状态
    
    Returns:
        系统状态
    """
    return {
        "modules": {
            "vision": settings.MODULE_VISION_ENABLED,
            "trajectory": settings.MODULE_TRAJECTORY_ENABLED,
            "route_planning": settings.MODULE_ROUTE_PLANNING_ENABLED
        },
        "pipeline_initialized": pipeline is not None
    }