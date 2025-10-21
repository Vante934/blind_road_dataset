#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成障碍物检测和预测系统
结合YOLO检测、轨迹预测和风险评估
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO
from modules.dynamic_obstacle_predictor import DynamicObstaclePredictor, Prediction
from modules.trajectory_predictor import TrajectoryPredictor

class IntegratedObstacleSystem:
    """集成障碍物检测和预测系统"""
    
    def __init__(self, 
                 model_path: str = "models/yolo11n.pt",
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4):
        
        # 初始化YOLO模型
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # 初始化动态障碍物预测器
        self.obstacle_predictor = DynamicObstaclePredictor(
            prediction_horizon=3.0,
            time_step=0.1,
            collision_threshold=100.0,
            velocity_threshold=5.0
        )
        
        # 初始化轨迹预测器（如果可用）
        try:
            self.trajectory_predictor = TrajectoryPredictor()
            self.trajectory_available = True
        except:
            self.trajectory_predictor = None
            self.trajectory_available = False
        
        # 用户位置跟踪
        self.user_position = None
        self.user_trajectory = []
        
        # 系统状态
        self.is_running = False
        self.frame_count = 0
        
    def detect_obstacles(self, frame: np.ndarray) -> List[Dict]:
        """检测障碍物"""
        # 使用YOLO进行目标检测
        results = self.model(frame, conf=self.confidence_threshold, iou=self.nms_threshold)
        
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # 获取置信度和类别
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    # 过滤掉不需要的类别
                    if class_name in ['person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle']:
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'class_name': class_name,
                            'confidence': float(confidence),
                            'class_id': class_id
                        })
        
        return detections
    
    def update_user_position(self, position: Tuple[float, float]):
        """更新用户位置"""
        self.user_position = position
        self.user_trajectory.append((position, time.time()))
        
        # 保持轨迹历史长度
        if len(self.user_trajectory) > 100:
            self.user_trajectory = self.user_trajectory[-100:]
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """处理单帧图像"""
        self.frame_count += 1
        start_time = time.time()
        
        # 1. 检测障碍物
        detections = self.detect_obstacles(frame)
        
        # 2. 更新障碍物跟踪
        obstacles = self.obstacle_predictor.update_obstacles(detections)
        
        # 3. 预测障碍物轨迹
        predictions = self.obstacle_predictor.predict_obstacle_trajectories(self.user_position)
        
        # 4. 计算处理时间
        processing_time = time.time() - start_time
        
        # 5. 生成警告信息
        warnings = self._generate_warnings(predictions)
        
        # 6. 获取统计信息
        stats = self.obstacle_predictor.get_obstacle_statistics()
        
        return {
            'detections': detections,
            'obstacles': obstacles,
            'predictions': predictions,
            'warnings': warnings,
            'statistics': stats,
            'processing_time': processing_time,
            'frame_count': self.frame_count
        }
    
    def _generate_warnings(self, predictions: List[Prediction]) -> List[str]:
        """生成警告信息"""
        warnings = []
        
        for prediction in predictions:
            if prediction.collision_risk > 0.8:
                warnings.append(f"🚨 紧急警告：障碍物ID {prediction.obstacle_id} 高风险碰撞！")
            elif prediction.collision_risk > 0.5:
                warnings.append(f"⚠️ 注意：障碍物ID {prediction.obstacle_id} 中等风险")
            
            if prediction.time_to_collision and prediction.time_to_collision < 2.0:
                warnings.append(f"⏰ 碰撞倒计时：{prediction.time_to_collision:.1f}秒")
        
        return warnings
    
    def visualize_results(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """可视化检测和预测结果"""
        vis_frame = frame.copy()
        
        # 绘制检测框
        for detection in results['detections']:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # 绘制边界框
            color = self._get_class_color(class_name)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(vis_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 绘制预测轨迹
        vis_frame = self.obstacle_predictor.visualize_predictions(vis_frame, results['predictions'])
        
        # 绘制用户位置
        if self.user_position:
            cv2.circle(vis_frame, (int(self.user_position[0]), int(self.user_position[1])), 
                      10, (255, 0, 0), -1)
            cv2.putText(vis_frame, "User", 
                       (int(self.user_position[0]) + 15, int(self.user_position[1])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # 绘制警告信息
        y_offset = 30
        for warning in results['warnings']:
            cv2.putText(vis_frame, warning, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 25
        
        # 绘制统计信息
        stats_text = f"障碍物: {results['statistics']['total_obstacles']} | " \
                    f"运动: {results['statistics']['moving_obstacles']} | " \
                    f"处理时间: {results['processing_time']*1000:.1f}ms"
        cv2.putText(vis_frame, stats_text, (10, vis_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_frame
    
    def _get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """获取类别对应的颜色"""
        colors = {
            'person': (0, 255, 0),      # 绿色
            'car': (255, 0, 0),        # 蓝色
            'truck': (0, 0, 255),      # 红色
            'bus': (0, 255, 255),      # 黄色
            'bicycle': (255, 0, 255),  # 紫色
            'motorcycle': (255, 255, 0) # 青色
        }
        return colors.get(class_name, (128, 128, 128))
    
    def get_system_status(self) -> Dict:
        """获取系统状态"""
        return {
            'is_running': self.is_running,
            'frame_count': self.frame_count,
            'user_position': self.user_position,
            'trajectory_available': self.trajectory_available,
            'model_loaded': self.model is not None,
            'obstacle_predictor_ready': self.obstacle_predictor is not None
        }
    
    def start_system(self):
        """启动系统"""
        self.is_running = True
        self.frame_count = 0
        print("🚀 集成障碍物检测系统已启动")
    
    def stop_system(self):
        """停止系统"""
        self.is_running = False
        print("⏹️ 集成障碍物检测系统已停止")
    
    def reset_system(self):
        """重置系统"""
        self.obstacle_predictor = DynamicObstaclePredictor()
        self.user_position = None
        self.user_trajectory = []
        self.frame_count = 0
        print("🔄 系统已重置")


class ObstacleDetectionGUI:
    """障碍物检测GUI界面"""
    
    def __init__(self):
        self.system = IntegratedObstacleSystem()
        self.cap = None
        self.is_running = False
        
    def start_camera(self, camera_id: int = 0):
        """启动摄像头"""
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise Exception(f"无法打开摄像头 {camera_id}")
        
        self.system.start_system()
        self.is_running = True
        
        print(f"📹 摄像头 {camera_id} 已启动")
    
    def process_camera_feed(self):
        """处理摄像头数据流"""
        if not self.cap or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # 处理帧
        results = self.system.process_frame(frame)
        
        # 可视化结果
        vis_frame = self.system.visualize_results(frame, results)
        
        return vis_frame, results
    
    def stop_camera(self):
        """停止摄像头"""
        if self.cap:
            self.cap.release()
        self.system.stop_system()
        self.is_running = False
        print("📹 摄像头已停止")
    
    def update_user_position(self, x: int, y: int):
        """更新用户位置（鼠标点击）"""
        self.system.update_user_position((float(x), float(y)))
        print(f"📍 用户位置更新: ({x}, {y})")


if __name__ == "__main__":
    # 测试代码
    gui = ObstacleDetectionGUI()
    
    try:
        # 启动摄像头
        gui.start_camera(0)
        
        while True:
            # 处理摄像头数据
            result = gui.process_camera_feed()
            if result is None:
                break
            
            vis_frame, results = result
            
            # 显示结果
            cv2.imshow('Dynamic Obstacle Detection', vis_frame)
            
            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                gui.system.reset_system()
            elif key == ord('s'):
                # 保存当前帧
                cv2.imwrite(f'frame_{gui.system.frame_count}.jpg', vis_frame)
                print(f"💾 保存帧: frame_{gui.system.frame_count}.jpg")
    
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断")
    except Exception as e:
        print(f"❌ 错误: {e}")
    finally:
        gui.stop_camera()
        cv2.destroyAllWindows()

