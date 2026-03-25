"""
完整流程集成

将所有模块串联，实现端到端处理
"""
import logging
import time
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from app.config import settings
from app.models.schemas import SensorData

# 导入各模块
from app.modules.vision.detector import EnhancedObstacleDetector
from app.modules.vision.classifier import EnhancedObstacleClassifier
from app.modules.vision.depth_estimator import DepthEstimator
from app.modules.audio.enhanced_classifier import EnhancedSoundClassifier
from app.modules.trajectory.kalman_tracker import KalmanTracker
from app.core.fusion_engine import BayesianFusionEngine, FusionInput
from app.services.route_planner import EnhancedRoutePlanner

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """流程结果"""
    # 感知结果
    vision_obstacles: list = field(default_factory=list)
    sound_result: Optional[Dict] = None
    trajectories: list = field(default_factory=list)
    
    # 决策结果
    warning_decision: Optional[Dict] = None
    route_plan: Optional[Dict] = None
    
    # 性能指标
    processing_time: Dict[str, float] = field(default_factory=dict)
    total_time: float = 0.0
    
    # 状态
    success: bool = True
    error: Optional[str] = None


class IntegratedPipeline:
    """
    完整集成流程
    
    流程:
    前端数据 → 视觉检测 → 障碍物分类 → 声音识别 → 
    轨迹预测 → 融合决策 → 路径规划 → 返回指令
    """
    
    def __init__(self):
        # 初始化各模块
        self._init_modules()
        
        # 多目标跟踪器字典 {object_id: KalmanTracker}
        self.trackers: Dict[str, KalmanTracker] = {}
    
    def _init_modules(self):
        """初始化模块"""
        logger.info("初始化完整流程组件...")
        
        # 视觉模块
        if settings.MODULE_VISION_ENABLED:
            self.vision_detector = EnhancedObstacleDetector()
            self.vision_classifier = EnhancedObstacleClassifier()
            self.depth_estimator = DepthEstimator()
        else:
            self.vision_detector = None
            self.vision_classifier = None
            self.depth_estimator = None
        
        # 音频模块（始终启用）
        self.sound_classifier = EnhancedSoundClassifier()
        
        # 决策引擎
        self.fusion_engine = BayesianFusionEngine()
        
        # 路径规划器
        if settings.MODULE_ROUTE_PLANNING_ENABLED:
            self.route_planner = EnhancedRoutePlanner()
        else:
            self.route_planner = None
        
        logger.info("✅ 所有组件初始化完成")
    
    async def process(self, sensor_data: SensorData) -> PipelineResult:
        """
        主处理流程
        
        Args:
            sensor_data: 传感器数据
        Returns:
            流程结果
        """
        start_time = time.time()
        result = PipelineResult()
        
        try:
            # ===== 阶段1: 视觉检测 =====
            if sensor_data.video_frame and self.vision_detector:
                t0 = time.time()
                
                # 解码图像
                import base64
                import cv2
                import numpy as np
                
                img_bytes = base64.b64decode(sensor_data.video_frame)
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                # 检测
                detections = self.vision_detector.detect(image)
                
                # 深度估算
                for det in detections:
                    det.distance = self.depth_estimator.estimate_distance(
                        det, 
                        tof_distance=sensor_data.tof_distance,
                        image=image
                    )
                
                # 分类
                obstacles = self.vision_classifier.classify(detections, image)
                
                # 转换为字典
                result.vision_obstacles = [
                    {
                        "class": obs.description,
                        "confidence": obs.detection.confidence,
                        "distance": obs.detection.distance,
                        "direction": obs.detection.direction,
                        "type": obs.obstacle_type.value,
                        "danger_level": obs.danger_level,
                        "bbox": obs.detection.bbox,
                        "is_moving": obs.is_moving
                    }
                    for obs in obstacles
                ]
                
                result.processing_time["vision"] = time.time() - t0
                logger.debug(f"视觉检测: {len(result.vision_obstacles)} 个障碍物")
            
            # ===== 阶段2: 声音识别 =====
            if sensor_data.audio_data:
                t0 = time.time()
                
                import base64
                audio_bytes = base64.b64decode(sensor_data.audio_data.audio_base64)
                
                # 分类
                sound_result = self.sound_classifier.classify(
                    audio_data=audio_bytes,
                    asr_text=None,  # 这里可以接入ASR
                    sample_rate=sensor_data.audio_data.sample_rate
                )
                
                # 音量分析
                volume_info = self.sound_classifier.classify_by_volume(audio_bytes)
                
                if sound_result:
                    result.sound_result = {
                        "sound_type": sound_result.sound_type,
                        "sound_label": sound_result.sound_label,
                        "confidence": sound_result.confidence,
                        "danger_score": sound_result.danger_score,
                        "urgency": sound_result.urgency,
                        "volume_info": volume_info
                    }
                
                result.processing_time["audio"] = time.time() - t0
                logger.debug(f"声音识别: {sound_result.sound_label if sound_result else 'None'}")
            
            # ===== 阶段3: 轨迹预测 =====
            if settings.MODULE_TRAJECTORY_ENABLED and result.vision_obstacles:
                t0 = time.time()
                
                trajectories = []
                
                for obs in result.vision_obstacles:
                    # 只跟踪动态障碍物
                    if not obs.get("is_moving"):
                        continue
                    
                    if not obs.get("distance"):
                        continue
                    
                    # 构建跟踪ID
                    track_id = f"{sensor_data.device_id}_{obs['class']}_{obs['direction']}"
                    
                    # 获取或创建跟踪器
                    if track_id not in self.trackers:
                        self.trackers[track_id] = KalmanTracker()
                    
                    tracker = self.trackers[track_id]
                    
                    # 更新轨迹（使用bbox中心）
                    bbox_center = (
                        (obs["bbox"][0] + obs["bbox"][2]) / 2,
                        (obs["bbox"][1] + obs["bbox"][3]) / 2
                    )
                    
                    prediction = tracker.update(
                        measurement=bbox_center,
                        timestamp=sensor_data.timestamp / 1000.0,  # 转秒
                        distance=obs["distance"]
                    )
                    
                    trajectories.append({
                        "object_id": track_id,
                        "object_class": obs["class"],
                        "speed": prediction.speed,
                        "acceleration": prediction.acceleration,
                        "direction": prediction.direction,
                        "ttc": prediction.ttc,
                        "danger_score": prediction.danger_score,
                        "confidence": prediction.confidence,
                        "predicted_positions": prediction.predicted_positions,
                        "object_direction": obs["direction"]
                    })
                
                result.trajectories = trajectories
                result.processing_time["trajectory"] = time.time() - t0
                logger.debug(f"轨迹预测: {len(trajectories)} 条轨迹")
            
            # ===== 阶段4: 融合决策 =====
            t0 = time.time()
            
            fusion_input = FusionInput(
                obstacles=result.vision_obstacles,
                trajectories=result.trajectories,
                sound_classification=result.sound_result,
                volume_info=result.sound_result.get("volume_info") if result.sound_result else None,
                distance=sensor_data.tof_distance,
                direction="center"  # 默认前方
            )
            
            warning_decision = self.fusion_engine.decide(fusion_input)
            
            result.warning_decision = {
                "warning_level": warning_decision.warning_level,
                "warning_level_name": warning_decision.warning_level_name,
                "confidence": warning_decision.confidence,
                "tts_text": warning_decision.tts_text,
                "vibration_intensity": warning_decision.vibration_intensity,
                "vibration_pattern": warning_decision.vibration_pattern,
                "primary_threat": warning_decision.primary_threat,
                "threat_breakdown": warning_decision.threat_breakdown,
                "timestamp": warning_decision.timestamp
            }
            
            result.processing_time["fusion"] = time.time() - t0
            
            # ===== 阶段5: 路径规划 =====
            if self.route_planner:
                t0 = time.time()
                
                route_plan = self.route_planner.plan(
                    obstacles=result.vision_obstacles,
                    trajectories=result.trajectories
                )
                
                result.route_plan = {
                    "recommended": {
                        "direction": route_plan.recommended.direction.value,
                        "safety_score": route_plan.recommended.safety_score,
                        "clearance": route_plan.recommended.clearance,
                        "reason": route_plan.recommended.reason,
                        "priority": route_plan.recommended.priority
                    },
                    "alternatives": [
                        {
                            "direction": alt.direction.value,
                            "safety_score": alt.safety_score,
                            "clearance": alt.clearance,
                            "reason": alt.reason,
                            "priority": alt.priority
                        }
                        for alt in route_plan.alternatives
                    ],
                    "tts_instruction": route_plan.tts_instruction,
                    "visual_hint": route_plan.visual_hint
                }
                
                result.processing_time["route"] = time.time() - t0
            
            # 计算总处理时间
            result.total_time = time.time() - start_time
            logger.info(f"总处理时间: {result.total_time:.3f}秒")
            
        except Exception as e:
            logger.error(f"处理过程异常: {e}")
            result.success = False
            result.error = str(e)
        
        return result


# 全局单例
integrated_pipeline = IntegratedPipeline()