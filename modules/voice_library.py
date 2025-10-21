# -*- coding: utf-8 -*-
"""
盲道障碍检测语音库系统
提供智能的障碍物语音提示模板
支持从配置文件读取自定义语音文本
"""

import json
import os

class VoiceLibrary:
    """语音库系统"""
    
    def __init__(self, config_file=None):
        if config_file is None:
            # 尝试从不同位置查找配置文件
            possible_paths = [
                "voice_config.json",
                "configs/voice_config.json",
                os.path.join(os.path.dirname(__file__), "..", "configs", "voice_config.json")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    config_file = path
                    break
            
            if config_file is None:
                config_file = "voice_config.json"  # 默认文件名
        self.config_file = config_file
        self.obstacle_types = {}
        self.class_mapping = {}
        self.generic_templates = {}
        self.special_scenarios = {}
        
        # 加载配置文件
        self.load_config()
    
    def load_config(self):
        """从配置文件加载语音库"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                self.obstacle_types = config.get('obstacle_types', {})
                self.class_mapping = config.get('class_mapping', {})
                self.generic_templates = config.get('generic_templates', {})
                self.special_scenarios = config.get('special_scenarios', {})
                
                print(f"✅ 语音库配置文件加载成功: {self.config_file}")
            else:
                print(f"⚠️ 配置文件不存在: {self.config_file}，使用默认配置")
                self.create_default_config()
        except Exception as e:
            print(f"❌ 加载配置文件失败: {e}，使用默认配置")
            self.create_default_config()
    
    def create_default_config(self):
        """创建默认配置"""
        self.obstacle_types = {
            # 静态障碍物
            "static": {
                "vehicle": {
                    "name": "车辆",
                    "templates": {
                        "far": "前方{distance}米{direction}有停放车辆",
                        "near": "{direction}{distance}米有车辆，请减速",
                        "danger": "危险！{direction}{distance}米有车辆，立即减速！",
                        "emergency": "危险！立即停止！{direction}{distance}米有车辆"
                    }
                }
            },
            # 动态障碍物
            "dynamic": {
                "person": {
                    "name": "行人",
                    "templates": {
                        "far": "前方{distance}米{direction}有行人",
                        "near": "{direction}{distance}米有行人，请减速",
                        "danger": "危险！{direction}{distance}米有行人，立即减速！",
                        "emergency": "危险！立即停止！{direction}{distance}米有行人"
                    }
                }
            },
            # 地面异常
            "ground": {
                "pothole": {
                    "name": "坑洼",
                    "templates": {
                        "far": "前方{distance}米{direction}有坑洼",
                        "near": "{direction}{distance}米有坑洼，请减速",
                        "danger": "危险！{direction}{distance}米有坑洼，立即减速！",
                        "emergency": "危险！立即停止！{direction}{distance}米有深坑"
                    }
                }
            },
            # 环境声音
            "env_sound": {
                "car_horn": {
                    "name": "汽车鸣笛",
                    "templates": {
                        "far": "前方有汽车鸣笛，请注意安全",
                        "near": "附近有汽车鸣笛，请注意避让",
                        "danger": "危险！汽车鸣笛声接近，请警惕！",
                        "emergency": "危险！立即停止！汽车鸣笛声非常近！"
                    }
                },
                "construction_noise": {
                    "name": "施工噪音",
                    "templates": {
                        "far": "前方有施工噪音，请注意",
                        "near": "附近有施工噪音，请减速",
                        "danger": "危险！施工噪音接近，请警惕！",
                        "emergency": "危险！立即停止！施工噪音非常近！"
                    }
                },
                "alarm": {
                    "name": "警报声",
                    "templates": {
                        "far": "前方有警报声，请注意安全",
                        "near": "附近有警报声，请警惕",
                        "danger": "危险！警报声接近，请注意！",
                        "emergency": "危险！立即停止！警报声非常近！"
                    }
                }
            }
        }
        
        self.class_mapping = {
            "0": ["static", "vehicle"],
            "1": ["dynamic", "person"],
            "2": ["ground", "pothole"]
        }
        
        self.generic_templates = {
            "far": "前方{distance}米{direction}有障碍物",
            "near": "{direction}{distance}米有障碍物，请减速",
            "danger": "危险！{direction}{distance}米有障碍物，立即减速！",
            "emergency": "危险！立即停止！{direction}{distance}米有障碍物"
        }
        
        self.special_scenarios = {
            "multiple_obstacles": "前方发现多个障碍物，请小心",
            "narrow_passage": "通道狭窄，请减速通过"
        }
    
    def save_config(self):
        """保存配置到文件"""
        try:
            config = {
                'obstacle_types': self.obstacle_types,
                'class_mapping': self.class_mapping,
                'generic_templates': self.generic_templates,
                'special_scenarios': self.special_scenarios
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 语音库配置已保存到: {self.config_file}")
            return True
        except Exception as e:
            print(f"❌ 保存配置文件失败: {e}")
            return False
    
    def get_obstacle_info(self, class_id, class_name=None):
        """根据类别ID获取障碍物信息"""
        class_id_str = str(class_id)
        
        if class_id_str in self.class_mapping:
            mapping = self.class_mapping[class_id_str]
            if isinstance(mapping, list) and len(mapping) >= 2:
                category, obstacle_type = mapping[0], mapping[-1]
                if (category in self.obstacle_types and 
                    obstacle_type in self.obstacle_types[category]):
                    return self.obstacle_types[category][obstacle_type]
        
        # 未知类别，返回通用信息
        return {
            "name": class_name or f"障碍物{class_id}",
            "templates": self.generic_templates
        }
    
    def generate_voice_message(self, class_id, distance, direction, class_name=None):
        """生成语音播报消息"""
        obstacle_info = self.get_obstacle_info(class_id, class_name)
        templates = obstacle_info["templates"]
        
        # 根据距离选择模板
        if distance > 5:
            return None  # 距离过远，不播报
        elif distance > 3:
            template = templates.get("far", "前方{distance}米{direction}有障碍物")
        elif distance > 1:
            template = templates.get("near", "{direction}{distance}米有障碍物，请减速")
        elif distance > 0.5:
            template = templates.get("danger", "危险！{direction}{distance}米有障碍物，立即减速！")
        else:
            template = templates.get("emergency", "危险！立即停止！{direction}{distance}米有障碍物")
        
        # 格式化消息
        message = template.format(
            distance=f"{distance:.1f}",
            direction=direction
        )
        
        return message
    
    def get_special_scenario_message(self, scenario_type):
        """获取特殊场景消息"""
        return self.special_scenarios.get(scenario_type, "请注意安全")
    
    def add_custom_obstacle(self, category, obstacle_type, name, templates):
        """添加自定义障碍物类型"""
        if category not in self.obstacle_types:
            self.obstacle_types[category] = {}
        
        self.obstacle_types[category][obstacle_type] = {
            "name": name,
            "templates": templates
        }
        print(f"✅ 添加障碍物类型: {category}.{obstacle_type} -> {name}")
    
    def update_class_mapping(self, class_id, category, obstacle_type):
        """更新类别映射"""
        class_id_str = str(class_id)
        self.class_mapping[class_id_str] = [category, obstacle_type]
        print(f"✅ 更新类别映射: class{class_id} -> {category}.{obstacle_type}")
    
    def add_class_mapping(self, class_id, category, obstacle_type):
        """添加类别映射"""
        self.update_class_mapping(class_id, category, obstacle_type)
    
    def remove_class_mapping(self, class_id):
        """移除类别映射"""
        class_id_str = str(class_id)
        if class_id_str in self.class_mapping:
            del self.class_mapping[class_id_str]
            print(f"✅ 移除类别映射: class{class_id}")
    
    def update_obstacle_template(self, category, obstacle_type, distance_level, template):
        """更新障碍物语音模板"""
        if (category in self.obstacle_types and 
            obstacle_type in self.obstacle_types[category]):
            self.obstacle_types[category][obstacle_type]["templates"][distance_level] = template
            print(f"✅ 更新模板: {category}.{obstacle_type}.{distance_level}")
        else:
            print(f"❌ 障碍物类型不存在: {category}.{obstacle_type}")
    
    def add_special_scenario(self, scenario_name, message):
        """添加特殊场景消息"""
        self.special_scenarios[scenario_name] = message
        print(f"✅ 添加特殊场景: {scenario_name}")
    
    def reload_config(self):
        """重新加载配置文件"""
        self.load_config()
        print("✅ 语音库配置已重新加载")
    
    def get_all_obstacles(self):
        """获取所有障碍物类型"""
        result = []
        for category, obstacles in self.obstacle_types.items():
            for obstacle_type, info in obstacles.items():
                result.append({
                    'category': category,
                    'type': obstacle_type,
                    'name': info['name']
                })
        return result
    
    def get_all_class_mappings(self):
        """获取所有类别映射"""
        return self.class_mapping.copy()

# 创建全局语音库实例
voice_library = VoiceLibrary()

# 示例使用
if __name__ == "__main__":
    # 测试语音库
    lib = VoiceLibrary()
    
    # 测试不同距离的语音播报
    test_cases = [
        (0, 4.5, "正前方"),  # 车辆，远距离
        (1, 2.0, "左侧"),    # 行人，近距离
        (2, 0.8, "右侧"),    # 坑洼，危险距离
        (3, 0.3, "正前方"),  # 摊位，紧急距离
    ]
    
    for class_id, distance, direction in test_cases:
        message = lib.generate_voice_message(class_id, distance, direction)
        print(f"类别{class_id}, 距离{distance}米, 方位{direction}: {message}") 