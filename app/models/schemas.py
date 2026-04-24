"""
数据模型定义
"""
from pydantic import BaseModel
from typing import Optional, List, Dict


class AudioData(BaseModel):
    """音频数据模型"""
    audio_base64: str
    audio_format: str
    sample_rate: int


class SensorData(BaseModel):
    """传感器数据模型"""
    device_id: str
    timestamp: float
    tof_distance: Optional[float] = None
    tof_direction: Optional[str] = "rear"
    audio_data: Optional[AudioData] = None
    video_frame: Optional[str] = None  # Base64编码的图像


class WarningCommand(BaseModel):
    """预警指令模型"""
    type: str
    warning_level: int
    warning_level_name: str
    tts_text: str
    vibration_intensity: int
    vibration_pattern: str
    distance: Optional[float] = None
    direction: Optional[str] = None
    timestamp: float


class PipelineResponse(BaseModel):
    """Pipeline响应模型"""
    success: bool
    message: str
    data: Dict