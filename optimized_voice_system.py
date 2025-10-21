#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import threading
import queue
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoicePriority(Enum):
    """语音优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    EMERGENCY = 4

class VoiceCategory(Enum):
    """语音类别"""
    OBSTACLE = "obstacle"
    ENVIRONMENT = "environment"
    WEATHER = "weather"
    SAFETY = "safety"
    EMERGENCY = "emergency"

@dataclass
class VoiceMessage:
    """语音消息"""
    text: str
    priority: VoicePriority
    category: VoiceCategory
    timestamp: float
    content_hash: str

class OptimizedVoiceSystem:
    """优化的语音播报系统"""
    
    def __init__(self, config_file: str = "configs/optimized_voice_config.json"):
        self.config_file = config_file
        self.is_enabled = True
        self.volume = 0.8
        self.rate = 1.0
        self.voice_queue = queue.PriorityQueue()
        self.is_playing = False
        self.tts_engine = None
        
        # 去重机制
        self.recent_messages = {}  # 最近播报的消息
        self.message_cooldown = 3.0  # 消息冷却时间（秒）
        self.max_recent_messages = 10  # 最多保存的最近消息数
        
        # 分段播报
        self.max_message_length = 50  # 单条消息最大长度
        self.segment_delay = 1.0  # 分段播报间隔
        
        # 初始化语音引擎
        self._init_voice_engine()
        
        # 启动语音播放线程
        self._start_voice_thread()
    
    def _init_voice_engine(self):
        """初始化语音引擎"""
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            
            # 设置中文语音
            voices = self.tts_engine.getProperty('voices')
            for voice in voices:
                if 'chinese' in voice.name.lower() or 'zh' in voice.id.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
            
            self.tts_engine.setProperty('rate', int(150 * self.rate))
            self.tts_engine.setProperty('volume', self.volume)
            logger.info("✅ 优化语音引擎初始化成功")
            
        except Exception as e:
            logger.error(f"❌ 语音引擎初始化失败: {e}")
            self.tts_engine = None
    
    def _start_voice_thread(self):
        """启动语音播放线程"""
        self.voice_thread = threading.Thread(target=self._voice_worker, daemon=True)
        self.voice_thread.start()
    
    def _voice_worker(self):
        """语音播放工作线程"""
        while True:
            try:
                if not self.is_enabled:
                    time.sleep(0.1)
                    continue
                
                # 获取语音消息
                priority, text, message = self.voice_queue.get(timeout=1.0)
                
                if message is None:  # 停止信号
                    break
                
                self.is_playing = True
                self._play_voice_message(message)
                self.is_playing = False
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"语音播放错误: {e}")
                self.is_playing = False
    
    def _play_voice_message(self, message: VoiceMessage):
        """播放语音消息"""
        try:
            if self.tts_engine:
                # 创建新的引擎实例避免线程冲突
                import pyttsx3
                temp_engine = pyttsx3.init()
                
                # 设置中文语音
                voices = temp_engine.getProperty('voices')
                for voice in voices:
                    if 'chinese' in voice.name.lower() or 'zh' in voice.id.lower():
                        temp_engine.setProperty('voice', voice.id)
                        break
                
                temp_engine.setProperty('rate', int(150 * self.rate))
                temp_engine.setProperty('volume', self.volume)
                temp_engine.say(message.text)
                temp_engine.runAndWait()
            else:
                print(f"🔊 语音播报: {message.text}")
        except Exception as e:
            logger.error(f"播放语音失败: {e}")
            print(f"🔊 语音播报: {message.text}")
    
    def _generate_content_hash(self, text: str) -> str:
        """生成内容哈希值用于去重"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _is_duplicate_message(self, text: str) -> bool:
        """检查是否为重复消息"""
        content_hash = self._generate_content_hash(text)
        current_time = time.time()
        
        # 检查是否在冷却时间内
        if content_hash in self.recent_messages:
            last_time = self.recent_messages[content_hash]
            if current_time - last_time < self.message_cooldown:
                return True
        
        # 更新最近消息记录
        self.recent_messages[content_hash] = current_time
        
        # 清理过期的消息记录
        if len(self.recent_messages) > self.max_recent_messages:
            # 移除最旧的消息
            oldest_hash = min(self.recent_messages.keys(), 
                            key=lambda k: self.recent_messages[k])
            del self.recent_messages[oldest_hash]
        
        return False
    
    def _split_long_message(self, text: str) -> List[str]:
        """分割长消息"""
        if len(text) <= self.max_message_length:
            return [text]
        
        # 按分号分割
        segments = text.split('；')
        result = []
        current_segment = ""
        
        for segment in segments:
            if len(current_segment + segment) <= self.max_message_length:
                if current_segment:
                    current_segment += "；" + segment
                else:
                    current_segment = segment
            else:
                if current_segment:
                    result.append(current_segment)
                current_segment = segment
        
        if current_segment:
            result.append(current_segment)
        
        return result
    
    def speak(self, text: str, priority: VoicePriority = VoicePriority.NORMAL, 
              category: VoiceCategory = VoiceCategory.ENVIRONMENT):
        """播放语音（优化版）"""
        if not self.is_enabled or not text:
            return
        
        # 检查重复消息
        if self._is_duplicate_message(text):
            logger.debug(f"跳过重复消息: {text}")
            return
        
        # 分割长消息
        segments = self._split_long_message(text)
        
        for i, segment in enumerate(segments):
            if not segment.strip():
                continue
            
            # 创建语音消息
            message = VoiceMessage(
                text=segment.strip(),
                priority=priority,
                category=category,
                timestamp=time.time(),
                content_hash=self._generate_content_hash(segment)
            )
            
            # 添加到队列
            self.voice_queue.put((priority.value, message.text, message))
            
            # 分段播报间隔
            if i < len(segments) - 1:
                time.sleep(self.segment_delay)
    
    def speak_obstacle(self, obstacle_type: str, distance: float, direction: str, 
                      risk_level: str = "medium"):
        """播报障碍物信息（优化版）"""
        if distance > 5.0:
            return  # 距离过远不播报
        
        if risk_level == "high":
            text = f"危险！{direction}{distance:.1f}米有{obstacle_type}，请立即停止"
            priority = VoicePriority.EMERGENCY
        elif risk_level == "medium":
            text = f"注意！{direction}{distance:.1f}米有{obstacle_type}，请减速"
            priority = VoicePriority.HIGH
        else:
            text = f"前方{distance:.1f}米{direction}有{obstacle_type}，请注意"
            priority = VoicePriority.NORMAL
        
        self.speak(text, priority, VoiceCategory.OBSTACLE)
    
    def speak_environment(self, safety_level: str, safety_score: float):
        """播报环境信息（优化版）"""
        safety_percentage = int(safety_score * 100)
        
        if safety_level == "high_risk":
            text = f"环境高风险，安全评分{safety_percentage}%，请小心"
            priority = VoicePriority.HIGH
        elif safety_level == "medium_risk":
            text = f"环境中等风险，安全评分{safety_percentage}%，请注意"
            priority = VoicePriority.NORMAL
        else:
            text = f"环境安全，安全评分{safety_percentage}%"
            priority = VoicePriority.LOW
        
        self.speak(text, priority, VoiceCategory.ENVIRONMENT)
    
    def speak_weather(self, weather_type: str):
        """播报天气信息（优化版）"""
        weather_messages = {
            "rain": "检测到雨天，路面可能湿滑",
            "fog": "检测到雾天，能见度较低",
            "snow": "检测到雪天，路面可能结冰"
        }
        
        if weather_type in weather_messages:
            self.speak(weather_messages[weather_type], VoicePriority.NORMAL, VoiceCategory.WEATHER)
    
    def speak_emergency(self, emergency_type: str):
        """播报紧急信息（优化版）"""
        emergency_messages = {
            "construction": "前方施工区域，请绕行",
            "intersection": "前方十字路口，请注意交通信号",
            "crowd": "人群较多，请注意安全"
        }
        
        if emergency_type in emergency_messages:
            self.speak(emergency_messages[emergency_type], VoicePriority.HIGH, VoiceCategory.EMERGENCY)
    
    def speak_smart_summary(self, detections: List[Dict], env_result: Dict):
        """智能摘要播报（优化版）"""
        # 只播报最重要的信息
        important_messages = []
        
        # 1. 障碍物信息（只播报最近的）
        if detections:
            closest_obstacle = min(detections, key=lambda x: x.get('distance', 999))
            obstacle_type = closest_obstacle.get('class_name', '物体')
            distance = closest_obstacle.get('distance', 0)
            direction = closest_obstacle.get('direction', '前方')
            
            if distance <= 5.0:
                if distance <= 1.0:
                    important_messages.append(f"危险！{direction}{distance:.1f}米有{obstacle_type}")
                elif distance <= 3.0:
                    important_messages.append(f"注意！{direction}{distance:.1f}米有{obstacle_type}")
                else:
                    important_messages.append(f"前方{distance:.1f}米{direction}有{obstacle_type}")
        
        # 2. 环境安全信息
        safety_level = env_result.get('overall_safety_level', 'safe')
        safety_score = env_result.get('safety_score', 1.0)
        safety_percentage = int(safety_score * 100)
        
        if safety_level == "high_risk":
            important_messages.append(f"环境高风险，安全评分{safety_percentage}%")
        elif safety_level == "medium_risk":
            important_messages.append(f"环境中等风险，安全评分{safety_percentage}%")
        
        # 3. 紧急信息（只播报最重要的）
        emergency_alerts = env_result.get('emergency_alerts', [])
        if emergency_alerts:
            important_messages.append(emergency_alerts[0])  # 只播报第一个紧急警报
        
        # 播报重要信息
        for message in important_messages[:2]:  # 最多播报2条重要信息
            if message:
                self.speak(message, VoicePriority.HIGH, VoiceCategory.SAFETY)
    
    def set_voice_enabled(self, enabled: bool):
        """设置语音开关"""
        self.is_enabled = enabled
        logger.info(f"优化语音系统 {'启用' if enabled else '禁用'}")
    
    def set_volume(self, volume: float):
        """设置音量"""
        self.volume = max(0.0, min(1.0, volume))
        logger.info(f"音量设置为: {self.volume}")
    
    def set_rate(self, rate: float):
        """设置语速"""
        self.rate = max(0.5, min(2.0, rate))
        logger.info(f"语速设置为: {self.rate}")
    
    def get_status(self) -> Dict:
        """获取系统状态"""
        return {
            'enabled': self.is_enabled,
            'playing': self.is_playing,
            'queue_size': self.voice_queue.qsize(),
            'volume': self.volume,
            'rate': self.rate,
            'recent_messages_count': len(self.recent_messages)
        }


# 使用示例
if __name__ == "__main__":
    # 创建优化语音系统
    voice_system = OptimizedVoiceSystem()
    
    # 测试优化语音播报
    print("🎯 测试优化语音播报系统")
    print("=" * 50)
    
    # 测试1: 基本语音播放
    voice_system.speak("优化语音系统测试开始")
    time.sleep(2)
    
    # 测试2: 障碍物播报
    voice_system.speak_obstacle("行人", 2.5, "前方", "medium")
    time.sleep(2)
    
    # 测试3: 环境播报
    voice_system.speak_environment("medium_risk", 0.6)
    time.sleep(2)
    
    # 测试4: 重复消息去重
    voice_system.speak("这是一条测试消息")
    voice_system.speak("这是一条测试消息")  # 应该被去重
    time.sleep(2)
    
    # 测试5: 智能摘要
    mock_detections = [
        {"class_name": "行人", "distance": 2.0, "direction": "前方"},
        {"class_name": "车辆", "distance": 4.0, "direction": "左侧"}
    ]
    mock_env_result = {
        "overall_safety_level": "medium_risk",
        "safety_score": 0.6,
        "emergency_alerts": ["前方施工区域，请绕行"]
    }
    
    voice_system.speak_smart_summary(mock_detections, mock_env_result)
    time.sleep(3)
    
    # 显示状态
    status = voice_system.get_status()
    print(f"系统状态: {status}")
    
    print("✅ 优化语音播报系统测试完成")
