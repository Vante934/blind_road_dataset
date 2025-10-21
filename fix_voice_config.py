#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os

def fix_voice_config():
    """修复语音配置文件"""
    print("🔧 修复语音配置文件")
    print("=" * 50)
    
    # 创建简化的语音配置
    voice_config = {
        "obstacle_types": {
            "person": {"name": "行人", "priority": "high"},
            "car": {"name": "车辆", "priority": "high"},
            "bicycle": {"name": "自行车", "priority": "medium"},
            "pothole": {"name": "坑洞", "priority": "high"},
            "construction": {"name": "施工区域", "priority": "high"},
            "trash_bin": {"name": "垃圾桶", "priority": "medium"},
            "street_light": {"name": "路灯杆", "priority": "medium"},
            "tree": {"name": "树木", "priority": "medium"},
            "sign": {"name": "标志牌", "priority": "low"}
        },
        "warning_templates": {
            "high_risk": "危险！{object_name}接近，请立即停止！",
            "medium_risk": "注意！{object_name}在附近，请减速",
            "low_risk": "前方有{object_name}，请注意"
        },
        "voice_settings": {
            "rate": 1.0,
            "volume": 0.8,
            "voice": "zh-CN-XiaoxiaoNeural"
        },
        "azure_speech_key": "",
        "azure_speech_region": "eastus"
    }
    
    # 确保configs目录存在
    os.makedirs("configs", exist_ok=True)
    
    # 保存配置文件
    config_file = "configs/voice_config.json"
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(voice_config, f, ensure_ascii=False, indent=2)
        print(f"✅ 语音配置文件已修复: {config_file}")
    except Exception as e:
        print(f"❌ 保存配置文件失败: {e}")
        return False
    
    # 验证配置文件
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
        
        # 检查关键字段
        required_fields = ['obstacle_types', 'warning_templates', 'voice_settings']
        for field in required_fields:
            if field not in loaded_config:
                print(f"❌ 缺少字段: {field}")
                return False
        
        print("✅ 配置文件验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 配置文件验证失败: {e}")
        return False

def test_voice_config():
    """测试语音配置"""
    print("\n🧪 测试语音配置")
    print("=" * 50)
    
    try:
        from blind_road_sdk import VoiceSystem
        
        # 创建语音系统
        voice_system = VoiceSystem()
        
        # 测试障碍物消息生成
        test_cases = [
            ("person", 0.8, "高风险行人"),
            ("car", 0.6, "中风险车辆"),
            ("bicycle", 0.4, "低风险自行车")
        ]
        
        for obstacle_type, risk_level, description in test_cases:
            print(f"测试: {description}")
            try:
                message = voice_system.generate_message(obstacle_type, risk_level)
                print(f"生成消息: {message}")
                
                # 测试语音播放
                voice_system.speak(message)
                print("✅ 语音播放成功")
                
            except Exception as e:
                print(f"❌ 测试失败: {e}")
        
        print("✅ 语音配置测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 语音配置测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 语音配置修复工具")
    print("=" * 60)
    
    # 修复配置文件
    if fix_voice_config():
        print("\n📋 配置文件修复完成")
        
        # 测试配置
        if test_voice_config():
            print("\n🎉 语音配置修复成功！")
            print("\n💡 现在可以正常使用语音播报功能了")
        else:
            print("\n⚠️ 语音配置测试失败，请检查系统")
    else:
        print("\n❌ 语音配置修复失败")

if __name__ == "__main__":
    main()

