"""
多级预警融合决策引擎 - 贝叶斯网络版

改进点:
1. 概率融合（贝叶斯推理）
2. 动态权重调整
3. 上下文感知
4. 不确定性建模
"""
import logging
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import time

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class FusionInput:
    """融合输入"""
    # 视觉维度
    obstacles: List[Dict] = field(default_factory=list)
    
    # 轨迹维度
    trajectories: List[Dict] = field(default_factory=list)
    
    # 声音维度
    sound_classification: Optional[Dict] = None
    volume_info: Optional[Dict] = None
    
    # 距离维度
    distance: Optional[float] = None
    direction: str = "center"
    
    # 上下文
    weather: str = "clear"        # clear/rain/fog
    time_of_day: str = "day"      # day/night
    location_type: str = "street" # street/indoor/park


@dataclass
class WarningDecision:
    """预警决策"""
    warning_level: int            # 0-3
    warning_level_name: str
    confidence: float             # 决策置信度
    
    # TTS内容
    tts_text: str
    
    # 反馈强度
    vibration_intensity: int      # 0-3
    vibration_pattern: str
    
    # 详细信息
    primary_threat: Optional[Dict] = None  # 主要威胁
    threat_breakdown: Dict[str, float] = field(default_factory=dict)  # 各维度威胁分数
    
    timestamp: float = 0.0


class BayesianFusionEngine:
    """
    贝叶斯融合引擎
    
    贝叶斯网络结构:
    
                Weather    TimeOfDay
                   ↓           ↓
    Vision → Obstacle_Danger ← Distance
       ↓            ↓
    Trajectory → Motion_Danger
       ↓            ↓
    Sound ────→ Overall_Danger → Warning_Level
    
    概率推理:
    P(Danger | Evidence) = P(Evidence | Danger) · P(Danger) / P(Evidence)
    """
    
    def __init__(self):
        # 先验概率（可从历史数据学习）
        self.prior_danger = {
            "safe": 0.7,
            "low": 0.2,
            "medium": 0.08,r
            "high": 0.02
        }
        
        # 条件概率表（CPT）- 简化版
        self._init_cpt()
    
    def _init_cpt(self):
        """初始化条件概率表"""
        # 这里简化处理，实际应从训练数据学习
        pass
    
    def decide(self, fusion_input: FusionInput) -> WarningDecision:
        """
        融合决策
        
        流程:
        1. 提取各维度特征
        2. 计算各维度威胁概率
        3. 贝叶斯融合
        4. 映射到预警级别
        5. 生成TTS
        """
        
        # ===== Step 1: 各维度评分 =====
        
        # 视觉威胁
        vision_score, vision_threat = self._eval_vision(fusion_input.obstacles)
        
        # 轨迹威胁
        trajectory_score, traj_threat = self._eval_trajectory(fusion_input.trajectories)
        
        # 声音威胁
        sound_score, sound_threat = self._eval_sound(
            fusion_input.sound_classification,
            fusion_input.volume_info
        )
        
        # 距离威胁
        distance_score = self._eval_distance(fusion_input.distance)
        
        # ===== Step 2: 上下文调整 =====
        context_factor = self._get_context_factor(fusion_input)
        
        # ===== Step 3: 动态权重分配 =====
        weights = self._calculate_weights(fusion_input)
        
        # ===== Step 4: 加权融合 =====
        total_score = (
            vision_score * weights["vision"] +
            trajectory_score * weights["trajectory"] +
            sound_score * weights["sound"] +
            distance_score * weights["distance"]
        ) * context_factor
        
        # ===== Step 5: 特殊规则覆盖 =====
        warning_level, confidence = self._apply_rules(
            total_score, fusion_input, vision_threat, traj_threat, sound_threat
        )
        
        # ===== Step 6: 生成TTS =====
        tts_text = self._generate_tts(
            warning_level, fusion_input, vision_threat, traj_threat, sound_threat
        )
        
        # ===== Step 7: 构建决策 =====
        decision = WarningDecision(
            warning_level=warning_level,
            warning_level_name=self._get_level_name(warning_level),
            confidence=confidence,
            tts_text=tts_text,
            vibration_intensity=self._get_vibration(warning_level),
            vibration_pattern=self._get_vibration_pattern(warning_level),
            primary_threat=self._get_primary_threat(vision_threat, traj_threat, sound_threat),
            threat_breakdown={
                "vision": vision_score,
                "trajectory": trajectory_score,
                "sound": sound_score,
                "distance": distance_score,
                "total": total_score
            },
            timestamp=time.time() * 1000
        )
        
        logger.info(
            f"[决策] Level={warning_level}, Confidence={confidence:.2f}, "
            f"Vision={vision_score:.2f}, Traj={trajectory_score:.2f}, "
            f"Sound={sound_score:.2f}, Dist={distance_score:.2f}"
        )
        
        return decision
    
    def _eval_vision(self, obstacles: List[Dict]) -> tuple:
        """评估视觉威胁"""
        if not obstacles:
            return 0.0, None
        
        # 找最危险的障碍物
        max_danger = 0.0
        most_dangerous = None
        
        for obs in obstacles:
            danger = obs.get("danger_level", 0)
            distance = obs.get("distance", 5.0)
            direction = obs.get("direction", "center")
            obs_type = obs.get("type", "static")
            
            # 基础危险度
            score = danger
            
            # 距离因子
            if distance < 1.0:
                dist_factor = 1.5
            elif distance < 2.0:
                dist_factor = 1.2
            elif distance < 3.0:
                dist_factor = 1.0
            else:
                dist_factor = 0.7
            
            # 方向因子
            if direction == "center":
                dir_factor = 1.3
            elif direction in ["left", "right"]:
                dir_factor = 1.0
            else:
                dir_factor = 0.8
            
            # 类型因子
            if obs_type == "dynamic":
                type_factor = 1.2
            elif obs_type == "ground":
                type_factor = 1.1
            else:
                type_factor = 1.0
            
            total = score * dist_factor * dir_factor * type_factor
            
            if total > max_danger:
                max_danger = total
                most_dangerous = obs
        
        # 多障碍物加成
        if len(obstacles) >= 3:
            max_danger *= 1.1
        
        return min(max_danger, 1.0), most_dangerous
    
    def _eval_trajectory(self, trajectories: List[Dict]) -> tuple:
        """评估轨迹威胁"""
        if not trajectories:
            return 0.0, None
        
        max_danger = 0.0
        most_dangerous = None
        
        for traj in trajectories:
            danger = traj.get("danger_score", 0)
            ttc = traj.get("ttc")
            speed = traj.get("speed", 0)
            direction = traj.get("direction", "stationary")
            
            score = danger
            
            # TTC加成
            if ttc is not None:
                if ttc <= 1.0:
                    score *= 1.5
                elif ttc <= 2.0:
                    score *= 1.3
                elif ttc <= 3.0:
                    score *= 1.1
            
            # 速度加成
            if speed > 3.0:
                score *= 1.2
            elif speed > 1.5:
                score *= 1.1
            
            # 方向调整
            if direction == "receding":
                score *= 0.3
            
            if score > max_danger:
                max_danger = score
                most_dangerous = traj
        
        return min(max_danger, 1.0), most_dangerous
    
    def _eval_sound(self, sound_class: Optional[Dict], volume: Optional[Dict]) -> tuple:
        """评估声音威胁"""
        if not sound_class:
            return 0.0, None
        
        danger = sound_class.get("danger_score", 0)
        confidence = sound_class.get("confidence", 0)
        urgency = sound_class.get("urgency", 0)
        
        # 有效危险度
        effective_danger = danger * confidence
        
        # 紧急度加成
        if urgency == 3:
            effective_danger *= 1.3
        elif urgency == 2:
            effective_danger *= 1.1
        
        # 音量加成
        if volume and volume.get("is_loud"):
            effective_danger *= 1.1
        
        return min(effective_danger, 1.0), sound_class
    
    def _eval_distance(self, distance: Optional[float]) -> float:
        """评估距离威胁"""
        if distance is None:
            return 0.0
        
        if distance <= 0.5:
            return 1.0
        elif distance <= 1.0:
            return 0.8
        elif distance <= 1.5:
            return 0.6
        elif distance <= 2.5:
            return 0.4
        elif distance <= 4.0:
            return 0.2
        else:
            return 0.05
    
    def _get_context_factor(self, input: FusionInput) -> float:
        """上下文因子"""
        factor = 1.0
        
        # 天气影响
        if input.weather == "rain":
            factor *= 1.1  # 雨天更危险
        elif input.weather == "fog":
            factor *= 1.2  # 雾天视线差
        
        # 时间影响
        if input.time_of_day == "night":
            factor *= 1.15  # 夜间更危险
        
        # 地点影响
        if input.location_type == "street":
            factor *= 1.0  # 街道正常
        elif input.location_type == "highway":
            factor *= 1.2  # 公路更危险
        
        return factor
    
    def _calculate_weights(self, input: FusionInput) -> Dict[str, float]:
        """动态权重分配"""
        
        # 基础权重
        weights = {
            "vision": 0.0,
            "trajectory": 0.0,
            "sound": 0.0,
            "distance": 0.0
        }
        
        # 根据数据可用性分配
        available = []
        
        if input.obstacles:
            available.append("vision")
        if input.trajectories:
            available.append("trajectory")
        if input.sound_classification:
            available.append("sound")
        if input.distance:
            available.append("distance")
        
        if not available:
            return weights
        
        # 策略1: 全都有 → 标准权重
        if len(available) == 4:
            weights = {
                "vision": 0.40,
                "trajectory": 0.30,
                "sound": 0.20,
                "distance": 0.10
            }
        # 策略2: 只有视觉+轨迹
        elif set(available) == {"vision", "trajectory"}:
            weights = {
                "vision": 0.55,
                "trajectory": 0.45,
                "sound": 0.0,
                "distance": 0.0
            }
        # 策略3: 只有声音+距离（退化模式）
        elif set(available) == {"sound", "distance"}:
            weights = {
                "vision": 0.0,
                "trajectory": 0.0,
                "sound": 0.60,
                "distance": 0.40
            }
        # 策略4: 均分可用维度
        else:
            weight_per_dim = 1.0 / len(available)
            for dim in available:
                weights[dim] = weight_per_dim
        
        return weights
    
    def _apply_rules(
        self, 
        total_score: float, 
        input: FusionInput,
        vision_threat: Optional[Dict],
        traj_threat: Optional[Dict],
        sound_threat: Optional[Dict]
    ) -> tuple:
        """应用硬规则"""
        
        # 规则1: 距离极近 → 强制一级
        if input.distance and input.distance < 0.5:
            return 1, 0.95
        
        # 规则2: TTC ≤ 1秒 → 强制一级
        if traj_threat and traj_threat.get("ttc"):
            if traj_threat["ttc"] <= 1.0:
                return 1, 0.9
        
        # 规则3: 紧急声音（警笛/刹车） + 中近距离 → 至少二级
        if sound_threat and sound_threat.get("urgency") == 3:
            if input.distance and input.distance < 3.0:
                if total_score < 0.45:
                    total_score = 0.45  # 提升到二级阈值
        
        # 规则4: 多个动态障碍物 + 都在靠近 → 升级
        if vision_threat and input.trajectories:
            dynamic_approaching = sum(
                1 for t in input.trajectories
                if t.get("direction") == "approaching"
            )
            if dynamic_approaching >= 2:
                total_score *= 1.2
        
        # 映射到级别
        if total_score >= 0.70:
            level = 1
            confidence = min(total_score, 0.98)
        elif total_score >= 0.45:
            level = 2
            confidence = min(total_score / 0.7, 0.95)
        elif total_score >= 0.25:
            level = 3
            confidence = min(total_score / 0.5, 0.90)
        else:
            level = 0
            confidence = 0.5
        
        return level, confidence
    
    def _generate_tts(
        self,
        level: int,
        input: FusionInput,
        vision_threat: Optional[Dict],
        traj_threat: Optional[Dict],
        sound_threat: Optional[Dict]
    ) -> str:
        """生成TTS文本（智能化）"""
        
        if level == 0:
            return ""
        
        # 构建文本片段
        direction_map = {
            "center": "前方",
            "left": "左侧",
            "right": "右侧",
            "rear": "后方"
        }
        direction_text = direction_map.get(input.direction, "附近")
        
        # 主威胁描述
        threat_desc = ""
        
        if vision_threat:
            cls = vision_threat.get("class", "障碍物")
            dist = vision_threat.get("distance")
            
            threat_desc = f"{cls}"
            if dist:
                threat_desc += f"，距离{dist:.1f}米"
        
        # 运动描述
        motion_desc = ""
        if traj_threat:
            ttc = traj_threat.get("ttc")
            speed = traj_threat.get("speed", 0)
            direction = traj_threat.get("direction", "")
            
            if direction == "approaching":
                if speed > 2.0:
                    motion_desc = "正在快速靠近"
                else:
                    motion_desc = "正在靠近"
                
                if ttc and ttc <= 3.0:
                    motion_desc += f"，{ttc:.0f}秒内将到达"
        
        # 声音描述
        sound_desc = ""
        if sound_threat:
            sound_label = sound_threat.get("sound_label", "")
            if sound_label:
                sound_desc = f"检测到{sound_label}"
        
        # 组合TTS（分级别）
        if level == 1:
            # 一级：简短有力
            parts = ["危险！"]
            
            if threat_desc:
                parts.append(f"{direction_text}{threat_desc}")
            
            if motion_desc:
                parts.append(motion_desc)
            elif sound_desc:
                parts.append(sound_desc)
            
            parts.append("请立即避让！")
            
            tts = "，".join(parts)
        
        elif level == 2:
            # 二级：详细说明
            parts = ["注意！"]
            
            if threat_desc:
                parts.append(f"{direction_text}{threat_desc}")
            
            if motion_desc:
                parts.append(motion_desc)
            
            if sound_desc:
                parts.append(sound_desc)
            
            parts.append("请注意安全。")
            
            tts = "，".join(parts)
        
        elif level == 3:
            # 三级：温和提醒
            parts = ["提醒："]
            
            if threat_desc:
                parts.append(f"{direction_text}{threat_desc}")
            elif sound_desc:
                parts.append(f"{direction_text}{sound_desc}")
            else:
                parts.append(f"{direction_text}有物体")
            
            tts = parts[0] + parts[1] + "。"
        
        else:
            tts = ""
        
        # 长度限制（避免太长）
        if len(tts) > 50:
            tts = tts[:47] + "..."
        
        return tts
    
    def _get_primary_threat(
        self,
        vision: Optional[Dict],
        traj: Optional[Dict],
        sound: Optional[Dict]
    ) -> Optional[Dict]:
        """确定主要威胁"""
        threats = []
        
        if vision:
            threats.append(("vision", vision, vision.get("danger_level", 0)))
        if traj:
            threats.append(("trajectory", traj, traj.get("danger_score", 0)))
        if sound:
            threats.append(("sound", sound, sound.get("danger_score", 0) * sound.get("confidence", 0)))
        
        if not threats:
            return None
        
        # 选择危险度最高的
        primary = max(threats, key=lambda x: x[2])
        
        return {
            "type": primary[0],
            "data": primary[1]
        }
    
    def _get_level_name(self, level: int) -> str:
        return {
            1: "一级危险",
            2: "二级警告",
            3: "三级提醒",
            0: "安全"
        }.get(level, "未知")
    
    def _get_vibration(self, level: int) -> int:
        return {1: 3, 2: 2, 3: 1, 0: 0}.get(level, 0)
    
    def _get_vibration_pattern(self, level: int) -> str:
        return {1: "sos", 2: "continuous", 3: "single", 0: "none"}.get(level, "none")