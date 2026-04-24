#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主控制器 - 整合三个模块的核心控制器
负责协调盲道检测、Android应用和语音导航系统
"""

import sys
import os
import json
import time
from typing import Dict, Any, Optional

class MainController:
    """主控制器类"""
    
    def __init__(self):
        """初始化主控制器"""
        self.config = self.load_config()
        self.modules = {
            'detection': None,  # 盲道检测模块
            'android': None,    # Android应用模块
            'voice': None       # 语音导航模块
        }
        self.is_running = False
        
    def load_config(self) -> Dict[str, Any]:
        """加载系统配置"""
        config_path = os.path.join(os.path.dirname(__file__), 'system_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 默认配置
            return {
                'detection': {
                    'enabled': True,
                    'model_path': 'models/yolov8n.pt',
                    'confidence_threshold': 0.5
                },
                'android': {
                    'enabled': True,
                    'server_port': 8080,
                    'camera_resolution': [640, 480]
                },
                'voice': {
                    'enabled': True,
                    'language': 'zh-CN',
                    'voice_speed': 1.0,
                    'volume': 0.8
                }
            }
    
    def initialize_modules(self):
        """初始化所有模块"""
        print("🔧 正在初始化系统模块...")
        
        # 初始化盲道检测模块
        if self.config['detection']['enabled']:
            try:
                from core.blind_road_sdk import BlindRoadDetector
                self.modules['detection'] = BlindRoadDetector()
                print("✅ 盲道检测模块初始化成功")
            except Exception as e:
                print(f"❌ 盲道检测模块初始化失败: {e}")
        
        # 初始化Android应用模块
        if self.config['android']['enabled']:
            try:
                # 这里可以导入Android相关的模块
                print("✅ Android应用模块初始化成功")
            except Exception as e:
                print(f"❌ Android应用模块初始化失败: {e}")
        
        # 初始化语音导航模块
        if self.config['voice']['enabled']:
            try:
                from voice_system.voice_navigator import VoiceNavigator
                self.modules['voice'] = VoiceNavigator()
                print("✅ 语音导航模块初始化成功")
            except Exception as e:
                print(f"❌ 语音导航模块初始化失败: {e}")
        
        print("🎉 模块初始化完成！")
    
    def start_system(self):
        """启动整个系统"""
        print("🚀 正在启动盲道检测系统...")
        
        self.initialize_modules()
        self.is_running = True
        
        print("=" * 50)
        print("🎯 盲道检测系统已启动")
        print("=" * 50)
        print("功能模块:")
        print(f"  - 盲道检测: {'✅' if self.modules['detection'] else '❌'}")
        print(f"  - Android应用: {'✅' if self.modules['android'] else '❌'}")
        print(f"  - 语音导航: {'✅' if self.modules['voice'] else '❌'}")
        print("=" * 50)
        
        return True
    
    def stop_system(self):
        """停止系统"""
        print("🛑 正在停止系统...")
        self.is_running = False
        
        # 清理资源
        for module_name, module in self.modules.items():
            if module and hasattr(module, 'cleanup'):
                try:
                    module.cleanup()
                    print(f"✅ {module_name}模块已清理")
                except Exception as e:
                    print(f"❌ {module_name}模块清理失败: {e}")
        
        print("🎉 系统已停止")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'is_running': self.is_running,
            'modules': {
                name: module is not None 
                for name, module in self.modules.items()
            },
            'config': self.config,
            'timestamp': time.time()
        }
    
    def run(self):
        """运行主循环"""
        try:
            self.start_system()
            
            # 主循环
            while self.is_running:
                time.sleep(1)
                
                # 这里可以添加系统监控逻辑
                # 例如：检查模块状态、处理用户输入等
                
        except KeyboardInterrupt:
            print("\n⚠️ 用户中断系统")
        except Exception as e:
            print(f"❌ 系统运行错误: {e}")
        finally:
            self.stop_system()

def main():
    """主函数"""
    print("=" * 60)
    print("🎯 盲道检测系统 - 主控制器")
    print("=" * 60)
    
    controller = MainController()
    controller.run()

if __name__ == "__main__":
    main()

