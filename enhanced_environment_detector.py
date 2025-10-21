#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版环境检测器
集成基于深度学习的环境事物检测和基于规则的环境分析
"""

import cv2
import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional
from collections import deque
import os

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO不可用，将使用基于规则的检测")

class EnhancedEnvironmentDetector:
    """增强版环境检测器"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.class_names = []
        self.class_info = {}
        self.detection_history = deque(maxlen=50)
        
        # 加载类别信息
        self.load_class_info()
        
        # 初始化深度学习模型
        if YOLO_AVAILABLE and model_path and os.path.exists(model_path):
            self.load_detection_model(model_path)
        else:
            print("⚠️ 使用基于规则的环境检测")
        
        # 初始化基于规则的检测器
        self.init_rule_based_detectors()
    
    def load_class_info(self):
        """加载类别信息"""
        try:
            with open('environment_annotation_classes.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                classes = data['environment_annotation_classes']
                
                self.class_names = []
                self.class_info = {}
                
                for category, items in classes.items():
                    for item_id, info in items.items():
                        self.class_names.append(info['name'])
                        self.class_info[info['id']] = {
                            'name': info['name'],
                            'category': category,
                            'safety_impact': info['safety_impact'],
                            'priority': info['detection_priority']
                        }
                
                print(f"✅ 加载了 {len(self.class_names)} 个环境类别")
        except Exception as e:
            print(f"❌ 加载类别信息失败: {e}")
            # 使用默认类别
            self.class_names = ['雨滴', '湿润表面', '雾颗粒', '雪块', '阴影区域', '强光点', '暗角', 
                              '裂缝', '坑洞', '台阶', '不平整路面', '施工标志', '安全锥', '施工围栏', 
                              '施工机械', '交通信号灯', '斑马线', '停车标志', '让行标志', '树木', 
                              '街道设施', '电线杆', '自行车']
            self.class_info = {}
    
    def load_detection_model(self, model_path: str):
        """加载深度学习检测模型"""
        try:
            self.model = YOLO(model_path)
            print(f"✅ 加载环境检测模型: {model_path}")
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            self.model = None
    
    def init_rule_based_detectors(self):
        """初始化基于规则的检测器"""
        # 这里可以集成之前的环境检测模块
        try:
            from modules.environment_detector import EnvironmentDetector
            self.rule_based_detector = EnvironmentDetector()
            print("✅ 初始化基于规则的环境检测器")
        except ImportError:
            print("⚠️ 基于规则的环境检测器不可用")
            self.rule_based_detector = None
    
    def detect_environment_objects(self, frame: np.ndarray) -> List[Dict]:
        """检测环境物体"""
        detections = []
        
        if self.model:
            # 使用深度学习模型检测
            try:
                results = self.model(frame, conf=0.3, iou=0.5)
                
                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confs = result.boxes.conf.cpu().numpy()
                        clss = result.boxes.cls.cpu().numpy()
                        
                        for box, conf, cls in zip(boxes, confs, clss):
                            class_id = int(cls)
                            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                            
                            detection = {
                                'bbox': box.tolist(),
                                'confidence': float(conf),
                                'class_id': class_id,
                                'class_name': class_name,
                                'detection_method': 'deep_learning'
                            }
                            
                            # 添加类别信息
                            if class_id in self.class_info:
                                detection.update(self.class_info[class_id])
                            
                            detections.append(detection)
            except Exception as e:
                print(f"❌ 深度学习检测失败: {e}")
        
        return detections
    
    def detect_environment_conditions(self, frame: np.ndarray) -> Dict:
        """检测环境条件（基于规则）"""
        if self.rule_based_detector:
            # 转换检测格式
            detection_objects = []
            for detection in self.detect_environment_objects(frame):
                detection_objects.append({
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'class': detection['class_id']
                })
            
            # 使用基于规则的检测器
            return self.rule_based_detector.detect_environment(frame, detection_objects)
        else:
            return {
                'overall_safety_level': 'safe',
                'safety_score': 1.0,
                'warnings': [],
                'emergency_alerts': []
            }
    
    def analyze_detection_results(self, detections: List[Dict]) -> Dict:
        """分析检测结果"""
        analysis = {
            'total_objects': len(detections),
            'high_priority_objects': [],
            'safety_risks': [],
            'environmental_factors': {
                'weather_related': [],
                'lighting_related': [],
                'surface_related': [],
                'construction_related': [],
                'traffic_related': [],
                'obstacle_related': []
            },
            'recommendations': []
        }
        
        for detection in detections:
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # 按优先级分类
            if detection.get('priority', 3) <= 1:
                analysis['high_priority_objects'].append(detection)
            
            # 按类别分类
            category = detection.get('category', 'obstacle_related')
            if category in analysis['environmental_factors']:
                analysis['environmental_factors'][category].append(detection)
            
            # 安全风险评估
            safety_impact = detection.get('safety_impact', 'low')
            if safety_impact in ['high', 'very_high']:
                analysis['safety_risks'].append({
                    'object': class_name,
                    'confidence': confidence,
                    'impact': safety_impact,
                    'bbox': detection['bbox']
                })
        
        # 生成建议
        analysis['recommendations'] = self.generate_recommendations(analysis)
        
        return analysis
    
    def generate_recommendations(self, analysis: Dict) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于高优先级物体
        if analysis['high_priority_objects']:
            recommendations.append("检测到高优先级环境物体，请特别注意")
        
        # 基于安全风险
        if analysis['safety_risks']:
            high_risk_count = sum(1 for risk in analysis['safety_risks'] if risk['impact'] == 'very_high')
            if high_risk_count > 0:
                recommendations.append("检测到高风险环境因素，建议暂停前进")
            else:
                recommendations.append("检测到中等风险环境因素，请谨慎前行")
        
        # 基于环境因素
        if analysis['environmental_factors']['weather_related']:
            recommendations.append("检测到天气相关因素，请注意天气变化")
        
        if analysis['environmental_factors']['construction_related']:
            recommendations.append("检测到施工相关物体，建议绕行")
        
        if analysis['environmental_factors']['traffic_related']:
            recommendations.append("检测到交通相关设施，请注意交通信号")
        
        return recommendations
    
    def detect_environment(self, frame: np.ndarray, gps_data: Optional[Dict] = None) -> Dict:
        """综合环境检测"""
        result = {
            'timestamp': time.time(),
            'detection_method': 'hybrid',
            'objects': [],
            'environmental_conditions': {},
            'analysis': {},
            'overall_safety_level': 'safe',
            'safety_score': 1.0,
            'warnings': [],
            'emergency_alerts': [],
            'recommendations': []
        }
        
        # 1. 检测环境物体
        objects = self.detect_environment_objects(frame)
        result['objects'] = objects
        
        # 2. 检测环境条件
        conditions = self.detect_environment_conditions(frame)
        result['environmental_conditions'] = conditions
        
        # 3. 分析检测结果
        analysis = self.analyze_detection_results(objects)
        result['analysis'] = analysis
        
        # 4. 综合安全评估
        safety_level, safety_score = self.assess_overall_safety(objects, conditions, analysis)
        result['overall_safety_level'] = safety_level
        result['safety_score'] = safety_score
        
        # 5. 生成警告和建议
        warnings, emergency_alerts = self.generate_warnings_and_alerts(objects, conditions, analysis)
        result['warnings'] = warnings
        result['emergency_alerts'] = emergency_alerts
        result['recommendations'] = analysis['recommendations']
        
        # 6. 记录检测历史
        self.detection_history.append(result)
        
        return result
    
    def assess_overall_safety(self, objects: List[Dict], conditions: Dict, analysis: Dict) -> Tuple[str, float]:
        """评估整体安全性"""
        safety_factors = []
        
        # 基于检测物体的安全评估
        for obj in objects:
            safety_impact = obj.get('safety_impact', 'low')
            confidence = obj.get('confidence', 0.5)
            priority = obj.get('priority', 3)
            
            if safety_impact == 'very_high':
                safety_factors.append(0.1 * confidence)
            elif safety_impact == 'high':
                safety_factors.append(0.3 * confidence)
            elif safety_impact == 'medium':
                safety_factors.append(0.6 * confidence)
            else:
                safety_factors.append(0.9 * confidence)
        
        # 基于环境条件的安全评估
        if conditions.get('safety_score'):
            safety_factors.append(conditions['safety_score'])
        
        # 计算综合安全分数
        if safety_factors:
            final_score = sum(safety_factors) / len(safety_factors)
        else:
            final_score = 1.0
        
        # 确定安全等级
        if final_score < 0.3:
            level = 'high_risk'
        elif final_score < 0.6:
            level = 'medium_risk'
        else:
            level = 'safe'
        
        return level, final_score
    
    def generate_warnings_and_alerts(self, objects: List[Dict], conditions: Dict, analysis: Dict) -> Tuple[List[str], List[str]]:
        """生成警告和紧急警报"""
        warnings = []
        emergency_alerts = []
        
        # 基于检测物体的警告
        for obj in objects:
            class_name = obj['class_name']
            confidence = obj['confidence']
            safety_impact = obj.get('safety_impact', 'low')
            
            if safety_impact == 'very_high':
                emergency_alerts.append(f"🚨 检测到高风险物体: {class_name} (置信度: {confidence:.2f})")
            elif safety_impact == 'high':
                warnings.append(f"⚠️ 检测到高风险物体: {class_name} (置信度: {confidence:.2f})")
            elif safety_impact == 'medium':
                warnings.append(f"⚠️ 检测到中等风险物体: {class_name}")
        
        # 基于环境条件的警告
        if conditions.get('warnings'):
            warnings.extend(conditions['warnings'])
        if conditions.get('emergency_alerts'):
            emergency_alerts.extend(conditions['emergency_alerts'])
        
        return warnings, emergency_alerts
    
    def get_detection_statistics(self) -> Dict:
        """获取检测统计信息"""
        if not self.detection_history:
            return {}
        
        recent_detections = list(self.detection_history)[-10:]  # 最近10次检测
        
        stats = {
            'total_detections': len(recent_detections),
            'avg_safety_score': np.mean([d['safety_score'] for d in recent_detections]),
            'safety_level_distribution': {},
            'object_counts': {},
            'warning_counts': {
                'warnings': 0,
                'emergency_alerts': 0
            }
        }
        
        # 统计安全等级分布
        for detection in recent_detections:
            level = detection['overall_safety_level']
            stats['safety_level_distribution'][level] = stats['safety_level_distribution'].get(level, 0) + 1
        
        # 统计物体检测
        for detection in recent_detections:
            for obj in detection['objects']:
                class_name = obj['class_name']
                stats['object_counts'][class_name] = stats['object_counts'].get(class_name, 0) + 1
        
        # 统计警告
        for detection in recent_detections:
            stats['warning_counts']['warnings'] += len(detection['warnings'])
            stats['warning_counts']['emergency_alerts'] += len(detection['emergency_alerts'])
        
        return stats

# 使用示例
if __name__ == "__main__":
    # 创建增强版环境检测器
    detector = EnhancedEnvironmentDetector()
    
    # 模拟检测
    test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    # 执行检测
    result = detector.detect_environment(test_frame)
    
    # 打印结果
    print("环境检测结果:")
    print(f"检测方法: {result['detection_method']}")
    print(f"检测物体数: {len(result['objects'])}")
    print(f"安全等级: {result['overall_safety_level']}")
    print(f"安全评分: {result['safety_score']:.2f}")
    print(f"警告数: {len(result['warnings'])}")
    print(f"紧急警报数: {len(result['emergency_alerts'])}")







