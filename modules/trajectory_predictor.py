# -*- coding: utf-8 -*-
"""
轨迹预测模块
集成盲道识别、动态障碍物跟踪和轨迹预测功能
"""

import cv2
import numpy as np
import math
import time
from collections import deque
from typing import List, Tuple, Dict, Optional
import json
import threading

class BlindPathDetector:
    """盲道检测器"""
    
    def __init__(self):
        self.path_history = deque(maxlen=50)
        self.path_center = None
        self.path_width = 0
        self.confidence = 0.0
        
    def detect_blind_path(self, frame: np.ndarray) -> Dict:
        """检测盲道"""
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 应用高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 筛选可能的盲道轮廓
        blind_path_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # 面积阈值
                # 计算轮廓的边界矩形
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # 盲道通常是长条形的
                if aspect_ratio > 2.0 and w > 50:
                    blind_path_contours.append(contour)
        
        if blind_path_contours:
            # 选择最大的盲道轮廓
            largest_contour = max(blind_path_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # 计算盲道中心
            center_x = x + w // 2
            center_y = y + h // 2
            
            self.path_center = (center_x, center_y)
            self.path_width = w
            self.confidence = min(1.0, cv2.contourArea(largest_contour) / 10000)
            
            # 添加到历史记录
            self.path_history.append((center_x, center_y, time.time()))
            
            return {
                'center': (center_x, center_y),
                'width': w,
                'height': h,
                'confidence': self.confidence,
                'contour': largest_contour
            }
        
        return None
    
    def predict_path_trajectory(self, steps: int = 10) -> List[Tuple[int, int]]:
        """预测盲道轨迹"""
        if len(self.path_history) < 3:
            return []
        
        # 获取最近的位置点
        recent_points = list(self.path_history)[-3:]
        
        # 计算平均速度
        if len(recent_points) >= 2:
            dx = recent_points[-1][0] - recent_points[0][0]
            dy = recent_points[-1][1] - recent_points[0][1]
            dt = recent_points[-1][2] - recent_points[0][2]
            
            if dt > 0:
                vx = dx / dt
                vy = dy / dt
                
                # 预测未来位置
                predicted_points = []
                current_point = (recent_points[-1][0], recent_points[-1][1])
                
                for i in range(1, steps + 1):
                    predicted_x = int(current_point[0] + vx * i * 0.1)  # 时间步长0.1秒
                    predicted_y = int(current_point[1] + vy * i * 0.1)
                    predicted_points.append((predicted_x, predicted_y))
                
                return predicted_points
        
        return []

class MotionPredictor:
    """运动预测器"""
    
    def __init__(self, prediction_steps: int = 5):
        self.prediction_steps = prediction_steps
        self.object_trajectories = {}  # {object_id: deque of positions}
        self.velocity_history = {}     # {object_id: deque of velocities}
        self.max_trajectory_length = 30
        
    def update_trajectory(self, object_id: int, position: Tuple[int, int], timestamp: float):
        """更新目标轨迹"""
        if object_id not in self.object_trajectories:
            self.object_trajectories[object_id] = deque(maxlen=self.max_trajectory_length)
            self.velocity_history[object_id] = deque(maxlen=self.max_trajectory_length)
        
        # 添加位置
        self.object_trajectories[object_id].append((position[0], position[1], timestamp))
        
        # 计算速度
        if len(self.object_trajectories[object_id]) >= 2:
            prev_pos = self.object_trajectories[object_id][-2]
            curr_pos = self.object_trajectories[object_id][-1]
            
            dt = curr_pos[2] - prev_pos[2]
            if dt > 0:
                vx = (curr_pos[0] - prev_pos[0]) / dt
                vy = (curr_pos[1] - prev_pos[1]) / dt
                self.velocity_history[object_id].append((vx, vy, timestamp))
    
    def predict_trajectory(self, object_id: int) -> List[Tuple[int, int]]:
        """预测目标轨迹"""
        if object_id not in self.object_trajectories:
            return []
        
        trajectory = self.object_trajectories[object_id]
        if len(trajectory) < 3:
            return []
        
        # 获取最近的位置和速度
        recent_positions = list(trajectory)[-3:]
        recent_velocities = list(self.velocity_history[object_id])[-3:] if object_id in self.velocity_history else []
        
        if len(recent_velocities) == 0:
            return []
        
        # 计算平均速度
        avg_vx = sum(v[0] for v in recent_velocities) / len(recent_velocities)
        avg_vy = sum(v[1] for v in recent_velocities) / len(recent_velocities)
        
        # 预测未来位置
        predicted_points = []
        current_pos = recent_positions[-1]
        
        for i in range(1, self.prediction_steps + 1):
            predicted_x = int(current_pos[0] + avg_vx * i * 0.1)  # 时间步长0.1秒
            predicted_y = int(current_pos[1] + avg_vy * i * 0.1)
            predicted_points.append((predicted_x, predicted_y))
        
        return predicted_points
    
    def calculate_collision_risk(self, object_id: int, user_position: Tuple[int, int], 
                               prediction_steps: int = 5) -> float:
        """计算碰撞风险"""
        predicted_trajectory = self.predict_trajectory(object_id)
        if not predicted_trajectory:
            return 0.0
        
        # 计算预测轨迹与用户位置的最小距离
        min_distance = float('inf')
        for point in predicted_trajectory:
            distance = math.sqrt((point[0] - user_position[0])**2 + (point[1] - user_position[1])**2)
            min_distance = min(min_distance, distance)
        
        # 将距离转换为风险值（距离越小，风险越高）
        if min_distance < 50:  # 危险距离阈值
            risk = 1.0 - (min_distance / 50.0)
        else:
            risk = 0.0
        
        return risk

class EnhancedTracker:
    """增强版目标跟踪器"""
    
    def __init__(self, max_disappeared: int = 30, max_distance: int = 50):
        self.next_object_id = 0
        self.objects = {}  # {object_id: (centroid, last_seen, trajectory, class_id)}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.motion_predictor = MotionPredictor()
        
    def register(self, centroid: Tuple[int, int], class_id: int = 0):
        """注册新目标"""
        self.objects[self.next_object_id] = (centroid, time.time(), deque([centroid], maxlen=30), class_id)
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        
        # 初始化运动预测器
        self.motion_predictor.update_trajectory(self.next_object_id - 1, centroid, time.time())
    
    def deregister(self, object_id: int):
        """注销目标"""
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]
    
    def update(self, detections: List[List]) -> List[Dict]:
        """更新跟踪状态"""
        if len(detections) == 0:
            # 没有检测到目标，增加消失计数
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return []
        
        if len(self.objects) == 0:
            # 没有跟踪的目标，注册所有检测到的目标
            for detection in detections:
                centroid = self.get_centroid(detection)
                class_id = int(detection[5]) if len(detection) > 5 else 0
                self.register(centroid, class_id)
        else:
            # 匹配现有目标
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[object_id][0] for object_id in object_ids]
            
            # 计算距离矩阵
            distances = []
            for detection in detections:
                detection_centroid = self.get_centroid(detection)
                row = []
                for object_centroid in object_centroids:
                    distance = self.calculate_distance(detection_centroid, object_centroid)
                    row.append(distance)
                distances.append(row)
            
            # 简单的最近邻匹配
            used_detections = set()
            used_objects = set()
            
            for i, detection in enumerate(detections):
                if i in used_detections:
                    continue
                    
                min_distance = float('inf')
                min_object_id = None
                
                for j, object_id in enumerate(object_ids):
                    if j in used_objects:
                        continue
                    
                    if distances[i][j] < min_distance and distances[i][j] < self.max_distance:
                        min_distance = distances[i][j]
                        min_object_id = object_id
                
                if min_object_id is not None:
                    # 更新目标位置
                    centroid = self.get_centroid(detection)
                    last_seen, trajectory, class_id = self.objects[min_object_id][1:]
                    trajectory.append(centroid)
                    self.objects[min_object_id] = (centroid, time.time(), trajectory, class_id)
                    self.disappeared[min_object_id] = 0
                    
                    # 更新运动预测器
                    self.motion_predictor.update_trajectory(min_object_id, centroid, time.time())
                    
                    used_detections.add(i)
                    used_objects.add(object_ids.index(min_object_id))
            
            # 处理未匹配的检测（新目标）
            for i, detection in enumerate(detections):
                if i not in used_detections:
                    centroid = self.get_centroid(detection)
                    class_id = int(detection[5]) if len(detection) > 5 else 0
                    self.register(centroid, class_id)
            
            # 处理未匹配的目标（消失的目标）
            for j, object_id in enumerate(object_ids):
                if j not in used_objects:
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
        
        # 返回当前跟踪的目标
        tracked_objects = []
        for object_id, (centroid, last_seen, trajectory, class_id) in self.objects.items():
            if self.disappeared[object_id] <= self.max_disappeared:
                # 预测轨迹
                predicted_trajectory = self.motion_predictor.predict_trajectory(object_id)
                
                tracked_objects.append({
                    'id': object_id,
                    'centroid': centroid,
                    'trajectory': list(trajectory),
                    'predicted_trajectory': predicted_trajectory,
                    'last_seen': last_seen,
                    'class_id': class_id
                })
        
        return tracked_objects
    
    def get_centroid(self, detection: List) -> Tuple[int, int]:
        """获取检测框的中心点"""
        x1, y1, x2, y2 = detection[:4]
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    def calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """计算两点间距离"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def get_collision_risks(self, user_position: Tuple[int, int]) -> Dict[int, float]:
        """获取所有目标的碰撞风险"""
        risks = {}
        for object_id in self.objects.keys():
            risk = self.motion_predictor.calculate_collision_risk(object_id, user_position)
            risks[object_id] = risk
        return risks

class TrajectoryPredictor:
    """轨迹预测主模块"""
    
    def __init__(self):
        self.blind_path_detector = BlindPathDetector()
        self.tracker = EnhancedTracker()
        self.user_position = (320, 240)  # 假设用户位置在帧中心
        self.last_warning_time = 0
        self.warning_cooldown = 3.0  # 警告冷却时间
        
        # 导入环境检测模块
        try:
            from modules.environment_detector import EnvironmentDetector
            self.env_detector = EnvironmentDetector()
            self.env_detection_available = True
            print("✅ 环境检测模块已集成")
        except ImportError:
            self.env_detector = None
            self.env_detection_available = False
            print("⚠️ 环境检测模块不可用")
        
    def process_frame(self, frame: np.ndarray, detections: List[List], gps_data: Optional[Dict] = None) -> Dict:
        """处理单帧图像"""
        result = {
            'blind_path': None,
            'tracked_objects': [],
            'collision_risks': {},
            'warnings': [],
            'environment_analysis': None,
            'navigation_guidance': [],
            'safety_recommendations': []
        }
        
        # 1. 检测盲道
        blind_path_info = self.blind_path_detector.detect_blind_path(frame)
        if blind_path_info:
            result['blind_path'] = blind_path_info
            result['blind_path']['predicted_trajectory'] = self.blind_path_detector.predict_path_trajectory()
        
        # 2. 更新目标跟踪
        tracked_objects = self.tracker.update(detections)
        result['tracked_objects'] = tracked_objects
        
        # 3. 计算碰撞风险
        collision_risks = self.tracker.get_collision_risks(self.user_position)
        result['collision_risks'] = collision_risks
        
        # 4. 环境检测（如果可用）
        if self.env_detection_available and self.env_detector:
            # 转换检测格式
            detection_objects = []
            for detection in detections:
                if len(detection) >= 6:
                    detection_objects.append({
                        'bbox': detection[:4],
                        'confidence': detection[4],
                        'class': int(detection[5])
                    })
            
            env_result = self.env_detector.detect_environment(frame, detection_objects, gps_data)
            result['environment_analysis'] = env_result
            
            # 添加环境安全信息
            result['safety_score'] = env_result.get('safety_score', 1.0)
            result['overall_safety_level'] = env_result.get('overall_safety_level', 'safe')
            
            # 合并导航指导
            result['navigation_guidance'].extend(env_result.get('navigation_guidance', []))
            
            # 合并警告和紧急警报
            result['warnings'].extend(env_result.get('warnings', []))
            result['emergency_alerts'] = env_result.get('emergency_alerts', [])
            
            # 添加环境检测详情
            if env_result.get('weather_conditions'):
                result['weather_info'] = env_result['weather_conditions']
            if env_result.get('lighting_conditions'):
                result['lighting_info'] = env_result['lighting_conditions']
            if env_result.get('surface_conditions'):
                result['surface_info'] = env_result['surface_conditions']
        
        # 5. 生成警告
        warnings = self.generate_warnings(tracked_objects, collision_risks, blind_path_info)
        result['warnings'].extend(warnings)
        
        # 6. 生成安全建议
        result['safety_recommendations'] = self.generate_safety_recommendations(result)
        
        return result
    
    def generate_warnings(self, tracked_objects: List[Dict], collision_risks: Dict[int, float], 
                         blind_path_info: Optional[Dict]) -> List[str]:
        """生成警告信息"""
        warnings = []
        current_time = time.time()
        
        # 检查冷却时间
        if current_time - self.last_warning_time < self.warning_cooldown:
            return warnings
        
        # 检查高碰撞风险的目标
        for object_id, risk in collision_risks.items():
            if risk > 0.7:  # 高风险阈值
                # 找到对应的目标信息
                target_info = next((obj for obj in tracked_objects if obj['id'] == object_id), None)
                if target_info:
                    class_id = target_info.get('class_id', 0)
                    class_name = self.get_class_name(class_id)
                    
                    # 计算距离和方向
                    centroid = target_info['centroid']
                    distance = self.calculate_distance(centroid, self.user_position)
                    direction = self.get_direction(centroid[0], self.user_position[0])
                    
                    warning_msg = f"危险！{direction}{int(distance/10)}米有{class_name}，碰撞风险高！"
                    warnings.append(warning_msg)
        
        # 检查盲道偏离
        if blind_path_info and blind_path_info['confidence'] > 0.5:
            path_center = blind_path_info['center']
            distance_from_path = abs(path_center[0] - self.user_position[0])
            
            if distance_from_path > 100:  # 偏离盲道阈值
                warnings.append("警告！您已偏离盲道，请回到盲道上")
        
        # 更新最后警告时间
        if warnings:
            self.last_warning_time = current_time
        
        return warnings
    
    def get_class_name(self, class_id: int) -> str:
        """获取类别名称"""
        names = ["人", "车", "障碍物", "坑洼", "其他"]
        return names[class_id % len(names)]
    
    def get_direction(self, x1: int, x2: int) -> str:
        """获取方向"""
        center = 320  # 帧中心
        if x1 < center - 50:
            return "左侧"
        elif x1 > center + 50:
            return "右侧"
        else:
            return "前方"
    
    def calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """计算两点间距离"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def update_user_position(self, position: Tuple[int, int]):
        """更新用户位置"""
        self.user_position = position
    
    def get_safety_guidance(self, frame_center: Tuple[int, int] = (320, 240)) -> str:
        """获取安全指导"""
        # 分析当前环境，提供安全指导
        tracked_objects = self.tracker.objects
        
        if not tracked_objects:
            return "环境安全，可以正常前进"
        
        # 计算最近的障碍物
        min_distance = float('inf')
        nearest_object = None
        
        for object_id, (centroid, _, _, class_id) in tracked_objects.items():
            distance = self.calculate_distance(centroid, frame_center)
            if distance < min_distance:
                min_distance = distance
                nearest_object = (object_id, centroid, class_id)
        
        if nearest_object and min_distance < 100:
            object_id, centroid, class_id = nearest_object
            class_name = self.get_class_name(class_id)
            direction = self.get_direction(centroid[0], frame_center[0])
            
            if min_distance < 50:
                return f"紧急！{direction}有{class_name}，请立即停止！"
            else:
                return f"注意！{direction}{int(min_distance/10)}米有{class_name}，请减速"
        
        return "环境相对安全，请保持警惕"
    
    def generate_safety_recommendations(self, result: Dict) -> List[str]:
        """生成安全建议"""
        recommendations = []
        
        # 基于盲道检测的建议
        if result.get('blind_path'):
            blind_path = result['blind_path']
            if blind_path.get('confidence', 0) > 0.7:
                recommendations.append("盲道清晰，建议沿盲道前进")
            elif blind_path.get('confidence', 0) > 0.4:
                recommendations.append("盲道部分可见，请谨慎前进")
            else:
                recommendations.append("盲道不清晰，建议使用其他导航方式")
        
        # 基于碰撞风险的建议
        collision_risks = result.get('collision_risks', {})
        high_risk_objects = [obj_id for obj_id, risk in collision_risks.items() if risk > 0.7]
        if high_risk_objects:
            recommendations.append(f"检测到{len(high_risk_objects)}个高风险障碍物，请立即停止")
        elif any(risk > 0.4 for risk in collision_risks.values()):
            recommendations.append("存在中等风险障碍物，请减速慢行")
        
        # 基于环境分析的建议
        env_analysis = result.get('environment_analysis')
        if env_analysis:
            overall_safety = env_analysis.get('overall_safety_level', 'safe')
            if overall_safety == 'high_risk':
                recommendations.append("环境风险极高，建议暂停导航")
            elif overall_safety == 'medium_risk':
                recommendations.append("环境存在风险，请提高警惕")
            
            # 施工区域建议
            construction = env_analysis.get('construction_zone')
            if construction and construction.get('is_construction_zone'):
                if construction.get('bypass_path'):
                    recommendations.append("检测到施工区域，建议绕行")
                else:
                    recommendations.append("前方施工区域，请谨慎通行")
            
            # 路口建议
            intersection = env_analysis.get('intersection')
            if intersection and intersection.get('is_intersection'):
                if intersection.get('traffic_light_state') == 'red':
                    recommendations.append("路口红灯，请等待")
                elif intersection.get('traffic_light_state') == 'green':
                    recommendations.append("路口绿灯，可以通行")
                else:
                    recommendations.append("前方路口，请确认安全后通行")
            
            # 拥挤场所建议
            crowd = env_analysis.get('crowd_density')
            if crowd and crowd.get('density_level') in ['high', 'very_high']:
                recommendations.append("环境拥挤，建议选择相对安全的路径")
        
        return recommendations

# 使用示例
if __name__ == "__main__":
    # 创建轨迹预测器
    predictor = TrajectoryPredictor()
    
    # 模拟检测结果
    mock_detections = [
        [100, 100, 150, 200, 0.8, 0],  # 人
        [300, 150, 400, 250, 0.9, 1],  # 车
    ]
    
    # 模拟处理帧
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = predictor.process_frame(mock_frame, mock_detections)
    
    print("处理结果:", json.dumps(result, indent=2, ensure_ascii=False)) 