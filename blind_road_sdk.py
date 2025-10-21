#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
盲道障碍检测SDK - 测试版
集成所有功能：检测、预测、语音、标注、训练
"""

import os
import sys
import cv2
import numpy as np
import json
import time
import threading
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import deque
import math

# ==================== 依赖检查 ====================
def check_dependencies():
    """检查所有依赖"""
    dependencies = {
        'ultralytics': 'pip install ultralytics',
        'torch': 'pip install torch',
        'opencv-python': 'pip install opencv-python',
        'numpy': 'pip install numpy',
        'requests': 'pip install requests',
        'PyQt5': 'pip install PyQt5'
    }
    
    missing = []
    for dep, install_cmd in dependencies.items():
        try:
            __import__(dep)
            print(f"✅ {dep} 已安装")
        except ImportError:
            print(f"❌ {dep} 未安装，请运行: {install_cmd}")
            missing.append(dep)
    
    return len(missing) == 0

# ==================== 核心检测模块 ====================
class BlindRoadDetector:
    """盲道检测核心模块"""
    
    def __init__(self, model_path: str = "models/yolov8n.pt"):
        self.model_path = model_path
        self.model = None
        self.is_initialized = False
        self.detection_history = deque(maxlen=100)
        
    def initialize(self) -> bool:
        """初始化检测模型"""
        try:
            from ultralytics import YOLO
            
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
            else:
                print(f"⚠️ 模型文件不存在: {self.model_path}，使用默认模型")
                self.model = YOLO('yolov8n.pt')
            
            self.is_initialized = True
            print("✅ 盲道检测模型初始化成功")
            return True
            
        except Exception as e:
            print(f"❌ 模型初始化失败: {e}")
            return False
    
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Dict]:
        """执行检测"""
        if not self.is_initialized:
            if not self.initialize():
                return []
        
        try:
            results = self.model(frame, conf=conf_threshold, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf),
                            'class_id': cls,
                            'class_name': self.get_class_name(cls)
                        })
            
            # 添加到历史记录
            self.detection_history.append({
                'timestamp': time.time(),
                'detections': detections
            })
            
            return detections
            
        except Exception as e:
            print(f"❌ 检测失败: {e}")
            return []
    
    def get_class_name(self, class_id: int) -> str:
        """获取类别名称"""
        class_names = {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
            4: "airplane", 5: "bus", 6: "train", 7: "truck",
            8: "boat", 9: "traffic light", 10: "fire hydrant"
        }
        return class_names.get(class_id, f"unknown_{class_id}")

# ==================== 轨迹预测模块 ====================
class TrajectoryPredictor:
    """轨迹预测模块"""
    
    def __init__(self):
        self.tracked_objects = {}
        self.path_history = deque(maxlen=50)
        self.user_position = (320, 240)
        
    def update_user_position(self, position: Tuple[int, int]):
        """更新用户位置"""
        self.user_position = position
    
    def track_objects(self, detections: List[Dict]) -> List[Dict]:
        """跟踪检测到的物体"""
        current_time = time.time()
        tracked = []
        
        for detection in detections:
            bbox = detection['bbox']
            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            
            object_id = detection.get('object_id', len(self.tracked_objects))
            
            if object_id not in self.tracked_objects:
                self.tracked_objects[object_id] = {
                    'history': deque(maxlen=10),
                    'class_id': detection['class_id'],
                    'class_name': detection['class_name']
                }
            
            self.tracked_objects[object_id]['history'].append({
                'position': center,
                'timestamp': current_time,
                'bbox': bbox
            })
            
            tracked.append({
                'object_id': object_id,
                'position': center,
                'bbox': bbox,
                'class_id': detection['class_id'],
                'class_name': detection['class_name'],
                'velocity': self.calculate_velocity(object_id)
            })
        
        return tracked
    
    def calculate_velocity(self, object_id: int) -> Tuple[float, float]:
        """计算物体速度"""
        if object_id not in self.tracked_objects:
            return (0.0, 0.0)
        
        history = self.tracked_objects[object_id]['history']
        if len(history) < 2:
            return (0.0, 0.0)
        
        pos1 = history[-1]['position']
        pos2 = history[-2]['position']
        time_diff = history[-1]['timestamp'] - history[-2]['timestamp']
        
        if time_diff > 0:
            vx = (pos1[0] - pos2[0]) / time_diff
            vy = (pos1[1] - pos2[1]) / time_diff
            return (vx, vy)
        
        return (0.0, 0.0)
    
    def predict_collision_risk(self, tracked_objects: List[Dict]) -> Dict[int, float]:
        """预测碰撞风险"""
        risks = {}
        
        for obj in tracked_objects:
            object_id = obj['object_id']
            position = obj['position']
            velocity = obj['velocity']
            
            distance = math.sqrt((position[0] - self.user_position[0])**2 + 
                               (position[1] - self.user_position[1])**2)
            
            if distance < 100:
                risk = max(0.0, 1.0 - distance / 100.0)
                speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
                if speed > 10:
                    risk *= 1.5
                risks[object_id] = min(1.0, risk)
            else:
                risks[object_id] = 0.0
        
        return risks
    
    def generate_warnings(self, tracked_objects: List[Dict], collision_risks: Dict[int, float]) -> List[str]:
        """生成警告信息"""
        warnings = []
        
        for obj in tracked_objects:
            object_id = obj['object_id']
            risk = collision_risks.get(object_id, 0.0)
            
            if risk > 0.7:
                warnings.append(f"危险！{obj['class_name']}接近，风险等级：{risk:.2f}")
            elif risk > 0.3:
                warnings.append(f"注意！{obj['class_name']}在附近，风险等级：{risk:.2f}")
        
        return warnings

# ==================== 语音系统模块 ====================
class VoiceSystem:
    """语音系统模块"""
    
    def __init__(self, config_file: str = "configs/voice_config.json"):
        self.config_file = config_file
        self.is_enabled = True
        self.volume = 0.8
        self.rate = 1.0
        self.voice_config = self.load_voice_config()
        self._init_voice_engine()
        
    def load_voice_config(self) -> Dict:
        """加载语音配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ 加载语音配置失败: {e}")
        
        return {
            'obstacle_types': {
                'person': {'name': '行人', 'priority': 'high'},
                'car': {'name': '车辆', 'priority': 'high'},
                'bicycle': {'name': '自行车', 'priority': 'medium'},
                'pothole': {'name': '坑洞', 'priority': 'high'},
                'construction': {'name': '施工区域', 'priority': 'high'}
            },
            'warning_templates': {
                'high_risk': '危险！{object_name}接近，请立即停止！',
                'medium_risk': '注意！{object_name}在附近，请减速',
                'low_risk': '前方有{object_name}，请注意'
            }
        }
    
    def generate_message(self, class_name: str, risk_level: float) -> str:
        """生成语音消息"""
        obstacle_info = self.voice_config['obstacle_types'].get(class_name, {'name': class_name})
        object_name = obstacle_info['name']
        
        if risk_level > 0.7:
            template = self.voice_config['warning_templates']['high_risk']
        elif risk_level > 0.3:
            template = self.voice_config['warning_templates']['medium_risk']
        else:
            template = self.voice_config['warning_templates']['low_risk']
        
        return template.format(object_name=object_name)
    
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
            print("✅ 语音引擎初始化成功")
            
        except Exception as e:
            print(f"⚠️ 语音引擎初始化失败: {e}")
            self.tts_engine = None
    
    def speak(self, message: str):
        """播放语音（稳定版）"""
        if not self.is_enabled:
            return
        
        try:
            # 优先使用pyttsx3（最稳定）
            if self.tts_engine:
                self.tts_engine.setProperty('rate', int(150 * self.rate))
                self.tts_engine.setProperty('volume', self.volume)
                self.tts_engine.say(message)
                self.tts_engine.runAndWait()
                return
            
            # 降级到控制台输出
            print(f"🔊 语音播报: {message}")
            
        except Exception as e:
            print(f"❌ 语音播放失败: {e}")
            print(f"🔊 语音播报: {message}")
    
    
    def enable(self):
        """启用语音"""
        self.is_enabled = True
        print("✅ 语音系统已启用")
    
    def disable(self):
        """禁用语音"""
        self.is_enabled = False
        print("🔇 语音系统已禁用")
    
    def set_volume(self, volume: float):
        """设置音量"""
        self.volume = max(0.0, min(1.0, volume))
        print(f"🔊 音量设置为: {self.volume:.1f}")
    
    def set_rate(self, rate: float):
        """设置语速"""
        self.rate = max(0.5, min(2.0, rate))
        print(f"⚡ 语速设置为: {self.rate:.1f}")

# ==================== 标注工具模块 ====================
class AnnotationTool:
    """标注工具模块"""
    
    def __init__(self):
        self.annotations = []
        self.current_image = None
        self.annotation_file = None
        
    def load_image(self, image_path: str) -> bool:
        """加载图像"""
        try:
            self.current_image = cv2.imread(image_path)
            if self.current_image is None:
                return False
            
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            self.annotation_file = f"data/annotations/{image_name}_annotations.json"
            
            self.load_annotations()
            return True
            
        except Exception as e:
            print(f"❌ 加载图像失败: {e}")
            return False
    
    def load_annotations(self):
        """加载现有标注"""
        if os.path.exists(self.annotation_file):
            try:
                with open(self.annotation_file, 'r', encoding='utf-8') as f:
                    self.annotations = json.load(f)
                print(f"✅ 加载标注文件: {self.annotation_file}")
            except Exception as e:
                print(f"⚠️ 加载标注失败: {e}")
                self.annotations = []
        else:
            self.annotations = []
    
    def add_annotation(self, x1: int, y1: int, x2: int, y2: int, class_name: str = "obstacle"):
        """添加标注"""
        annotation = {
            'id': len(self.annotations),
            'bbox': [x1, y1, x2, y2],
            'class_name': class_name,
            'timestamp': time.time()
        }
        self.annotations.append(annotation)
        print(f"✅ 添加标注: {class_name} at ({x1}, {y1}, {x2}, {y2})")
    
    def save_annotations(self) -> bool:
        """保存标注"""
        if not self.annotation_file:
            return False
        
        try:
            os.makedirs(os.path.dirname(self.annotation_file), exist_ok=True)
            with open(self.annotation_file, 'w', encoding='utf-8') as f:
                json.dump(self.annotations, f, ensure_ascii=False, indent=2)
            print(f"✅ 保存标注到: {self.annotation_file}")
            return True
        except Exception as e:
            print(f"❌ 保存标注失败: {e}")
            return False

# ==================== 训练模块 ====================
class ModelTrainer:
    """模型训练模块"""
    
    def __init__(self):
        self.training_config = {
            'model': 'yolov8n.pt',
            'data': 'data/yolo_dataset/dataset.yaml',
            'epochs': 100,
            'batch_size': 16,
            'imgsz': 640,
            'device': 'auto'
        }
        self.training_callback = None
        
    def set_training_callback(self, callback: Callable):
        """设置训练回调函数"""
        self.training_callback = callback
    
    def start_training(self, dataset_yaml: str = None, **kwargs) -> bool:
        """开始训练"""
        try:
            from ultralytics import YOLO
            
            config = self.training_config.copy()
            if dataset_yaml:
                config['data'] = dataset_yaml
            config.update(kwargs)
            
            print("🚀 开始训练模型...")
            print(f"📋 训练配置: {config}")
            
            model = YOLO(config['model'])
            
            results = model.train(
                data=config['data'],
                epochs=config['epochs'],
                batch=config['batch_size'],
                imgsz=config['imgsz'],
                device=config['device']
            )
            
            print("✅ 训练完成")
            return True
            
        except Exception as e:
            print(f"❌ 训练失败: {e}")
            return False
    
    def evaluate_model(self, model_path: str, test_data: str) -> Dict:
        """评估模型"""
        try:
            from ultralytics import YOLO
            
            model = YOLO(model_path)
            results = model.val(data=test_data)
            
            metrics = {
                'mAP50': results.box.map50,
                'mAP50-95': results.box.map,
                'precision': results.box.mp,
                'recall': results.box.mr
            }
            
            print(f"📊 模型评估结果:")
            for metric, value in metrics.items():
                print(f"  - {metric}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            print(f"❌ 模型评估失败: {e}")
            return {}

# ==================== 主SDK类 ====================
class BlindRoadSDK:
    """盲道障碍检测SDK主类"""
    
    def __init__(self):
        self.detector = BlindRoadDetector()
        self.trajectory_predictor = TrajectoryPredictor()
        self.voice_system = VoiceSystem()
        self.annotation_tool = AnnotationTool()
        self.model_trainer = ModelTrainer()
        
        self.is_initialized = False
        self.camera = None
        self.is_camera_active = False
        
    def initialize(self) -> bool:
        """初始化SDK"""
        print("🔧 初始化盲道障碍检测SDK...")
        
        if not check_dependencies():
            print("❌ 依赖检查失败")
            return False
        
        if not self.detector.initialize():
            print("❌ 检测器初始化失败")
            return False
        
        self.is_initialized = True
        print("✅ SDK初始化成功")
        return True
    
    def start_camera(self, camera_id: int = 0) -> bool:
        """启动摄像头"""
        if not self.is_initialized:
            print("❌ SDK未初始化")
            return False
        
        try:
            self.camera = cv2.VideoCapture(camera_id)
            if not self.camera.isOpened():
                print("❌ 无法打开摄像头")
                return False
            
            self.is_camera_active = True
            print("✅ 摄像头启动成功")
            return True
            
        except Exception as e:
            print(f"❌ 启动摄像头失败: {e}")
            return False
    
    def stop_camera(self):
        """停止摄像头"""
        if self.camera:
            self.camera.release()
        self.is_camera_active = False
        print("🛑 摄像头已停止")
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """处理单帧图像"""
        if not self.is_initialized:
            return {}
        
        detections = self.detector.detect(frame)
        tracked_objects = self.trajectory_predictor.track_objects(detections)
        collision_risks = self.trajectory_predictor.predict_collision_risk(tracked_objects)
        warnings = self.trajectory_predictor.generate_warnings(tracked_objects, collision_risks)
        
        for warning in warnings:
            self.voice_system.speak(warning)
        
        return {
            'detections': detections,
            'tracked_objects': tracked_objects,
            'collision_risks': collision_risks,
            'warnings': warnings
        }
    
    def get_camera_frame(self) -> Optional[np.ndarray]:
        """获取摄像头帧"""
        if not self.is_camera_active or not self.camera:
            return None
        
        ret, frame = self.camera.read()
        if ret:
            return frame
        return None
    
    def run_detection_loop(self, callback: Callable = None):
        """运行检测循环"""
        if not self.is_camera_active:
            print("❌ 摄像头未启动")
            return
        
        print("🔄 开始检测循环...")
        
        try:
            while self.is_camera_active:
                frame = self.get_camera_frame()
                if frame is None:
                    continue
                
                result = self.process_frame(frame)
                
                if callback:
                    callback(frame, result)
                
                cv2.imshow('Blind Road Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("🛑 检测循环被中断")
        finally:
            cv2.destroyAllWindows()
            self.stop_camera()
    
    def train_model(self, dataset_path: str, **kwargs) -> bool:
        """训练模型"""
        print("🎯 开始模型训练...")
        return self.model_trainer.start_training(dataset_path, **kwargs)
    
    def evaluate_model(self, model_path: str, test_data: str) -> Dict:
        """评估模型"""
        return self.model_trainer.evaluate_model(model_path, test_data)
    
    def annotate_image(self, image_path: str) -> bool:
        """标注图像"""
        return self.annotation_tool.load_image(image_path)
    
    def add_annotation(self, x1: int, y1: int, x2: int, y2: int, class_name: str = "obstacle"):
        """添加标注"""
        self.annotation_tool.add_annotation(x1, y1, x2, y2, class_name)
    
    def save_annotations(self) -> bool:
        """保存标注"""
        return self.annotation_tool.save_annotations()
    
    def set_voice_enabled(self, enabled: bool):
        """设置语音开关"""
        if enabled:
            self.voice_system.enable()
        else:
            self.voice_system.disable()
    
    def set_voice_volume(self, volume: float):
        """设置语音音量"""
        self.voice_system.set_volume(volume)
    
    def get_status(self) -> Dict:
        """获取SDK状态"""
        return {
            'initialized': self.is_initialized,
            'camera_active': self.is_camera_active,
            'voice_enabled': self.voice_system.is_enabled,
            'voice_volume': self.voice_system.volume,
            'detection_count': len(self.detector.detection_history)
        }
    
    def cleanup(self):
        """清理资源"""
        self.stop_camera()
        print("🧹 SDK资源已清理")

# ==================== 使用示例 ====================
def example_usage():
    """使用示例"""
    print("🚀 盲道障碍检测SDK使用示例")
    
    sdk = BlindRoadSDK()
    
    if not sdk.initialize():
        print("❌ SDK初始化失败")
        return
    
    sdk.set_voice_enabled(True)
    sdk.set_voice_volume(0.8)
    
    if sdk.start_camera():
        def detection_callback(frame, result):
            print(f"检测到 {len(result['detections'])} 个物体")
            if result['warnings']:
                print(f"警告: {result['warnings']}")
        
        sdk.run_detection_loop(detection_callback)
    
    sdk.cleanup()

if __name__ == "__main__":
    example_usage() 