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

class FixedVoiceSystem:
    """修复的语音播报系统"""
    
    def __init__(self):
        self.is_enabled = True
        self.volume = 0.8
        self.rate = 1.0
        self.voice_queue = queue.PriorityQueue()
        self.is_playing = False
        self.tts_engine = None
        
        # 实时播报控制
        self.last_voice_time = 0
        self.voice_interval = 2.0  # 语音播报间隔
        self.max_message_length = 30  # 最大消息长度
        
        # 去重机制
        self.recent_messages = {}
        self.message_cooldown = 3.0
        
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
            logger.info("✅ 修复语音引擎初始化成功")
            
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
    
    def _play_voice_message(self, message):
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
                temp_engine.say(message)
                temp_engine.runAndWait()
                print(f"🔊 语音播报成功: {message}")
            else:
                print(f"🔊 语音播报: {message}")
        except Exception as e:
            logger.error(f"播放语音失败: {e}")
            print(f"🔊 语音播报: {message}")
    
    def _is_duplicate_message(self, text: str) -> bool:
        """检查是否为重复消息"""
        content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        current_time = time.time()
        
        # 检查是否在冷却时间内
        if content_hash in self.recent_messages:
            last_time = self.recent_messages[content_hash]
            if current_time - last_time < self.message_cooldown:
                return True
        
        # 更新最近消息记录
        self.recent_messages[content_hash] = current_time
        
        # 清理过期的消息记录
        if len(self.recent_messages) > 10:
            oldest_hash = min(self.recent_messages.keys(), 
                            key=lambda k: self.recent_messages[k])
            del self.recent_messages[oldest_hash]
        
        return False
    
    def _simplify_message(self, text: str) -> str:
        """简化消息内容"""
        if len(text) <= self.max_message_length:
            return text
        
        # 按分号分割，取最重要的部分
        segments = text.split('；')
        important_segments = []
        
        for segment in segments:
            if any(keyword in segment for keyword in ['危险', '注意', '停止', '绕行', '高风险']):
                important_segments.append(segment)
                if len('；'.join(important_segments)) > self.max_message_length:
                    break
        
        if important_segments:
            return '；'.join(important_segments)
        else:
            # 如果没有重要信息，取前30个字符
            return text[:self.max_message_length]
    
    def speak(self, text: str, priority: VoicePriority = VoicePriority.NORMAL):
        """播放语音（修复版）"""
        if not self.is_enabled or not text:
            return
        
        # 检查冷却时间
        current_time = time.time()
        if current_time - self.last_voice_time < self.voice_interval:
            return
        
        # 检查重复消息
        if self._is_duplicate_message(text):
            logger.debug(f"跳过重复消息: {text}")
            return
        
        # 简化消息内容
        simplified_text = self._simplify_message(text)
        
        # 添加到队列
        self.voice_queue.put((priority.value, simplified_text, simplified_text))
        self.last_voice_time = current_time
    
    def speak_obstacle_alert(self, obstacle_type: str, distance: float, direction: str):
        """播报障碍物警报（修复版）"""
        if distance > 5.0:
            return
        
        if distance <= 1.0:
            text = f"危险！{direction}{distance:.1f}米有{obstacle_type}"
            priority = VoicePriority.EMERGENCY
        elif distance <= 3.0:
            text = f"注意！{direction}{distance:.1f}米有{obstacle_type}"
            priority = VoicePriority.HIGH
        else:
            text = f"前方{distance:.1f}米{direction}有{obstacle_type}"
            priority = VoicePriority.NORMAL
        
        self.speak(text, priority)
    
    def speak_environment_alert(self, safety_level: str, safety_score: float):
        """播报环境警报（修复版）"""
        safety_percentage = int(safety_score * 100)
        
        if safety_level == "high_risk":
            text = f"环境高风险，安全评分{safety_percentage}%"
            priority = VoicePriority.HIGH
        elif safety_level == "medium_risk":
            text = f"环境中等风险，安全评分{safety_percentage}%"
            priority = VoicePriority.NORMAL
        else:
            text = f"环境安全，安全评分{safety_percentage}%"
            priority = VoicePriority.LOW
        
        self.speak(text, priority)
    
    def speak_emergency_alert(self, alert_type: str):
        """播报紧急警报（修复版）"""
        emergency_messages = {
            "construction": "前方施工区域，请绕行",
            "intersection": "前方十字路口，请注意交通信号",
            "crowd": "人群较多，请注意安全"
        }
        
        if alert_type in emergency_messages:
            self.speak(emergency_messages[alert_type], VoicePriority.HIGH)
    
    def process_environment_detection(self, detections, env_result):
        """处理环境检测结果（修复版）"""
        if not self.is_enabled:
            return
        
        # 只播报最重要的信息
        important_messages = []
        
        # 1. 障碍物信息（只播报最近的）
        if detections:
            sorted_detections = sorted(detections, key=lambda x: x.get('distance', 999))
            closest_obstacle = sorted_detections[0]
            
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
        
        # 3. 紧急信息（只播报最重要的一个）
        emergency_alerts = env_result.get('emergency_alerts', [])
        if emergency_alerts:
            important_messages.append(emergency_alerts[0])
        
        # 播报重要信息（最多1条）
        if important_messages:
            self.speak(important_messages[0], VoicePriority.HIGH)
    
    def set_voice_enabled(self, enabled: bool):
        """设置语音开关"""
        self.is_enabled = enabled
        logger.info(f"修复语音系统 {'启用' if enabled else '禁用'}")
    
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
            'last_voice_time': self.last_voice_time
        }


# 使用示例
if __name__ == "__main__":
    # 创建修复语音系统
    voice_system = FixedVoiceSystem()
    
    # 测试修复语音播报
    print("🎯 测试修复语音播报系统")
    print("=" * 50)
    
    # 测试1: 基本语音播放
    voice_system.speak("修复语音系统测试开始")
    time.sleep(2)
    
    # 测试2: 障碍物播报
    voice_system.speak_obstacle_alert("行人", 2.0, "前方")
    time.sleep(2)
    
    # 测试3: 环境播报
    voice_system.speak_environment_alert("medium_risk", 0.6)
    time.sleep(2)
    
    # 测试4: 紧急警报
    voice_system.speak_emergency_alert("construction")
    time.sleep(2)
    
    # 测试5: 模拟环境检测结果处理
    mock_detections = [
        {"class_name": "行人", "distance": 2.0, "direction": "前方"},
        {"class_name": "车辆", "distance": 4.0, "direction": "左侧"}
    ]
    mock_env_result = {
        "overall_safety_level": "medium_risk",
        "safety_score": 0.6,
        "emergency_alerts": ["前方施工区域，请绕行"],
        "weather_conditions": {"weather_type": "snow"}
    }
    
    voice_system.process_environment_detection(mock_detections, mock_env_result)
    time.sleep(3)
    
    # 显示状态
    status = voice_system.get_status()
    print(f"系统状态: {status}")
    
    print("✅ 修复语音播报系统测试完成")

