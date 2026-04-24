#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音导航系统 - 核心语音导航功能
"""

import os
import json
import time
import threading
from typing import Dict, Any, Optional, List
import pyttsx3
import speech_recognition as sr

class VoiceNavigator:
    """语音导航器"""
    
    def __init__(self, config_path: str = None):
        """初始化语音导航器"""
        self.config = self.load_config(config_path)
        self.tts_engine = None
        self.recognizer = None
        self.microphone = None
        self.is_listening = False
        self.is_speaking = False
        
        self.initialize_voice_engine()
        self.initialize_speech_recognition()
    
    def load_config(self, config_path: str = None) -> Dict[str, Any]:
        """加载语音配置"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 默认配置
            return {
                'language': 'zh-CN',
                'voice_speed': 1.0,
                'volume': 0.8,
                'voice_engine': 'pyttsx3',
                'speech_recognition': 'speech_recognition',
                'voice_library': {
                    'obstacle_detected': '前方发现障碍物，请注意安全',
                    'blind_path_detected': '检测到盲道，可以安全通行',
                    'path_clear': '前方道路正常',
                    'turn_left': '请向左转',
                    'turn_right': '请向右转',
                    'go_straight': '请直行',
                    'stop': '请停止'
                }
            }
    
    def initialize_voice_engine(self):
        """初始化语音合成引擎"""
        try:
            self.tts_engine = pyttsx3.init()
            
            # 设置语音参数
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # 尝试找到中文语音
                for voice in voices:
                    if 'chinese' in voice.name.lower() or 'zh' in voice.id.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
            self.tts_engine.setProperty('rate', int(200 * self.config['voice_speed']))
            self.tts_engine.setProperty('volume', self.config['volume'])
            
            print("✅ 语音合成引擎初始化成功")
            
        except Exception as e:
            print(f"❌ 语音合成引擎初始化失败: {e}")
    
    def initialize_speech_recognition(self):
        """初始化语音识别"""
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # 调整麦克风环境噪声
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
            
            print("✅ 语音识别引擎初始化成功")
            
        except Exception as e:
            print(f"❌ 语音识别引擎初始化失败: {e}")
    
    def speak(self, text: str, blocking: bool = False):
        """语音播报"""
        if not self.tts_engine:
            print("❌ 语音合成引擎未初始化")
            return False
        
        try:
            if blocking:
                self.is_speaking = True
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                self.is_speaking = False
            else:
                # 非阻塞模式
                threading.Thread(
                    target=self._speak_thread,
                    args=(text,),
                    daemon=True
                ).start()
            
            print(f"🔊 语音播报: {text}")
            return True
            
        except Exception as e:
            print(f"❌ 语音播报失败: {e}")
            return False
    
    def _speak_thread(self, text: str):
        """语音播报线程"""
        self.is_speaking = True
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"❌ 语音播报线程出错: {e}")
        finally:
            self.is_speaking = False
    
    def listen(self, timeout: float = 5.0) -> Optional[str]:
        """语音识别"""
        if not self.recognizer or not self.microphone:
            print("❌ 语音识别引擎未初始化")
            return None
        
        try:
            with self.microphone as source:
                print("🎤 请说话...")
                audio = self.recognizer.listen(source, timeout=timeout)
            
            print("🔄 正在识别...")
            text = self.recognizer.recognize_google(
                audio, 
                language=self.config['language']
            )
            
            print(f"👂 识别结果: {text}")
            return text
            
        except sr.WaitTimeoutError:
            print("⏰ 语音识别超时")
            return None
        except sr.UnknownValueError:
            print("❓ 无法识别语音")
            return None
        except Exception as e:
            print(f"❌ 语音识别失败: {e}")
            return None
    
    def start_listening(self, callback: callable = None):
        """开始持续监听"""
        if self.is_listening:
            print("⚠️ 已经在监听中")
            return
        
        self.is_listening = True
        
        def listen_loop():
            while self.is_listening:
                try:
                    text = self.listen(timeout=1.0)
                    if text and callback:
                        callback(text)
                except Exception as e:
                    print(f"❌ 监听循环出错: {e}")
                time.sleep(0.1)
        
        threading.Thread(target=listen_loop, daemon=True).start()
        print("🎤 开始持续监听")
    
    def stop_listening(self):
        """停止监听"""
        self.is_listening = False
        print("🛑 停止监听")
    
    def generate_navigation_instruction(self, detection_data: Dict[str, Any]) -> str:
        """根据检测结果生成导航指令"""
        voice_library = self.config['voice_library']
        
        if detection_data.get('has_obstacle', False):
            obstacle_type = detection_data.get('obstacle_type', '障碍物')
            return f"前方发现{obstacle_type}，请注意安全"
        elif detection_data.get('has_blind_path', False):
            return voice_library['blind_path_detected']
        else:
            return voice_library['path_clear']
    
    def process_voice_command(self, command: str) -> Dict[str, Any]:
        """处理语音命令"""
        command = command.lower().strip()
        
        # 简单的命令识别
        if '开始' in command or 'start' in command:
            return {'action': 'start_detection', 'message': '开始检测'}
        elif '停止' in command or 'stop' in command:
            return {'action': 'stop_detection', 'message': '停止检测'}
        elif '状态' in command or 'status' in command:
            return {'action': 'check_status', 'message': '检查状态'}
        elif '帮助' in command or 'help' in command:
            return {'action': 'show_help', 'message': '显示帮助'}
        else:
            return {'action': 'unknown', 'message': '未知命令'}
    
    def cleanup(self):
        """清理资源"""
        self.stop_listening()
        if self.tts_engine:
            self.tts_engine.stop()
        print("🧹 语音导航器资源已清理")

def main():
    """测试语音导航系统"""
    print("=" * 50)
    print("🎤 语音导航系统测试")
    print("=" * 50)
    
    navigator = VoiceNavigator()
    
    try:
        # 测试语音播报
        print("测试语音播报...")
        navigator.speak("语音导航系统测试开始")
        
        # 测试语音识别
        print("测试语音识别...")
        text = navigator.listen(timeout=5.0)
        if text:
            print(f"识别到: {text}")
            # 处理命令
            result = navigator.process_voice_command(text)
            print(f"处理结果: {result}")
        
        # 测试导航指令生成
        test_data = {'has_obstacle': True, 'obstacle_type': '石头'}
        instruction = navigator.generate_navigation_instruction(test_data)
        print(f"导航指令: {instruction}")
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断测试")
    finally:
        navigator.cleanup()

if __name__ == "__main__":
    main()

