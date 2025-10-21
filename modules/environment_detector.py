# -*- coding: utf-8 -*-
"""
环境检测模块
功能：
1. 施工区域识别：检测施工标志、围栏、机械设备
2. 十字路口检测：识别交通信号灯、斑马线、路口标志
3. 拥挤场所分析：检测人群密度、障碍物分布
"""

import cv2
import numpy as np
import math
import time
from typing import Dict, List, Tuple, Optional
from collections import deque
import json

class ConstructionZoneDetector:
    """施工区域检测器"""
    
    def __init__(self):
        self.construction_keywords = [
            '施工', '建设', '维修', '工程', '危险', '禁止通行'
        ]
        self.construction_colors = [
            (0, 0, 255),    # 红色 - 危险标志
            (0, 255, 255),  # 黄色 - 警告标志
            (128, 128, 128) # 灰色 - 围栏
        ]
        self.detection_history = deque(maxlen=30)
        
    def detect_construction_zone(self, frame: np.ndarray) -> Dict:
        """检测施工区域"""
        result = {
            'is_construction_zone': False,
            'confidence': 0.0,
            'zone_type': 'none',
            'safety_level': 'safe',
            'detected_objects': [],
            'bypass_path': None
        }
        
        # 1. 检测施工标志和围栏
        construction_objects = self._detect_construction_objects(frame)
        
        # 2. 检测橙色/黄色安全设备
        safety_equipment = self._detect_safety_equipment(frame)
        
        # 3. 检测施工机械
        machinery = self._detect_construction_machinery(frame)
        
        # 4. 分析整体施工区域
        if construction_objects or safety_equipment or machinery:
            result['is_construction_zone'] = True
            result['detected_objects'] = construction_objects + safety_equipment + machinery
            
            # 计算置信度
            confidence = self._calculate_construction_confidence(
                construction_objects, safety_equipment, machinery
            )
            result['confidence'] = confidence
            
            # 确定区域类型和安全等级
            result['zone_type'] = self._classify_construction_zone(
                construction_objects, safety_equipment, machinery
            )
            result['safety_level'] = self._assess_safety_level(confidence)
            
            # 生成绕行路径
            result['bypass_path'] = self._generate_bypass_path(frame, result['detected_objects'])
        
        # 记录检测历史
        self.detection_history.append({
            'timestamp': time.time(),
            'is_construction': result['is_construction_zone'],
            'confidence': result['confidence']
        })
        
        return result
    
    def _detect_construction_objects(self, frame: np.ndarray) -> List[Dict]:
        """检测施工相关物体"""
        objects = []
        
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 检测红色区域（危险标志）
        red_lower = np.array([0, 50, 50])
        red_upper = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        
        # 检测黄色区域（警告标志）
        yellow_lower = np.array([20, 50, 50])
        yellow_upper = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        
        # 合并掩码
        construction_mask = cv2.bitwise_or(red_mask, yellow_mask)
        
        # 查找轮廓
        contours, _ = cv2.findContours(construction_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # 面积阈值
                x, y, w, h = cv2.boundingRect(contour)
                
                # 计算形状特征
                aspect_ratio = w / h if h > 0 else 0
                extent = area / (w * h) if w * h > 0 else 0
                
                # 判断是否为施工标志
                if aspect_ratio > 0.5 and extent > 0.3:
                    objects.append({
                        'type': 'construction_sign',
                        'bbox': [x, y, x+w, y+h],
                        'confidence': min(1.0, area / 5000),
                        'color': 'red' if area > 1000 else 'yellow'
                    })
        
        return objects
    
    def _detect_safety_equipment(self, frame: np.ndarray) -> List[Dict]:
        """检测安全设备"""
        objects = []
        
        # 检测橙色安全锥
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        orange_lower = np.array([10, 100, 100])
        orange_upper = np.array([25, 255, 255])
        orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
        
        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 200 < area < 2000:  # 安全锥的典型面积范围
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                
                # 安全锥通常是高大于宽
                if aspect_ratio > 1.2:
                    objects.append({
                        'type': 'safety_cone',
                        'bbox': [x, y, x+w, y+h],
                        'confidence': min(1.0, area / 1000)
                    })
        
        return objects
    
    def _detect_construction_machinery(self, frame: np.ndarray) -> List[Dict]:
        """检测施工机械"""
        objects = []
        
        # 使用边缘检测识别大型机械
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 查找大型轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # 大型机械的面积阈值
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # 机械通常是矩形或接近矩形
                if 0.5 < aspect_ratio < 3.0:
                    objects.append({
                        'type': 'construction_machinery',
                        'bbox': [x, y, x+w, y+h],
                        'confidence': min(1.0, area / 20000)
                    })
        
        return objects
    
    def _calculate_construction_confidence(self, signs: List, equipment: List, machinery: List) -> float:
        """计算施工区域检测置信度"""
        total_confidence = 0.0
        total_weight = 0.0
        
        # 施工标志权重最高
        for sign in signs:
            total_confidence += sign['confidence'] * 0.5
            total_weight += 0.5
        
        # 安全设备权重中等
        for equip in equipment:
            total_confidence += equip['confidence'] * 0.3
            total_weight += 0.3
        
        # 机械权重较低
        for mach in machinery:
            total_confidence += mach['confidence'] * 0.2
            total_weight += 0.2
        
        return total_confidence / total_weight if total_weight > 0 else 0.0
    
    def _classify_construction_zone(self, signs: List, equipment: List, machinery: List) -> str:
        """分类施工区域类型"""
        if machinery:
            return 'heavy_construction'
        elif signs and equipment:
            return 'road_work'
        elif signs:
            return 'maintenance'
        else:
            return 'unknown'
    
    def _assess_safety_level(self, confidence: float) -> str:
        """评估安全等级"""
        if confidence > 0.8:
            return 'high_risk'
        elif confidence > 0.5:
            return 'medium_risk'
        elif confidence > 0.2:
            return 'low_risk'
        else:
            return 'safe'
    
    def _generate_bypass_path(self, frame: np.ndarray, objects: List[Dict]) -> Optional[List[Tuple[int, int]]]:
        """生成绕行路径"""
        if not objects:
            return None
        
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # 分析障碍物分布
        left_obstacles = []
        right_obstacles = []
        
        for obj in objects:
            bbox = obj['bbox']
            obj_center_x = (bbox[0] + bbox[2]) // 2
            if obj_center_x < center_x:
                left_obstacles.append(obj)
            else:
                right_obstacles.append(obj)
        
        # 选择障碍物较少的一侧作为绕行路径
        if len(left_obstacles) < len(right_obstacles):
            # 左侧绕行
            return [(center_x - 100, center_y), (center_x - 150, center_y + 50)]
        else:
            # 右侧绕行
            return [(center_x + 100, center_y), (center_x + 150, center_y + 50)]

class IntersectionDetector:
    """十字路口检测器"""
    
    def __init__(self):
        self.traffic_light_colors = {
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'green': (0, 255, 0)
        }
        self.crosswalk_pattern = None
        self.detection_history = deque(maxlen=20)
        
    def detect_intersection(self, frame: np.ndarray, gps_data: Optional[Dict] = None) -> Dict:
        """检测十字路口"""
        result = {
            'is_intersection': False,
            'confidence': 0.0,
            'traffic_light_state': 'unknown',
            'crosswalk_detected': False,
            'crossing_guidance': None,
            'safety_recommendation': 'proceed_with_caution'
        }
        
        # 1. 检测交通信号灯
        traffic_light = self._detect_traffic_light(frame)
        
        # 2. 检测斑马线
        crosswalk = self._detect_crosswalk(frame)
        
        # 3. 检测路口标志
        intersection_signs = self._detect_intersection_signs(frame)
        
        # 4. 结合GPS数据判断
        gps_intersection = self._analyze_gps_data(gps_data) if gps_data else False
        
        # 综合判断是否为路口
        intersection_indicators = [traffic_light, crosswalk, intersection_signs, gps_intersection]
        positive_indicators = sum(1 for indicator in intersection_indicators if indicator)
        
        if positive_indicators >= 2:  # 至少2个指标表明是路口
            result['is_intersection'] = True
            result['confidence'] = positive_indicators / 4.0
            
            # 更新交通灯状态
            if traffic_light:
                result['traffic_light_state'] = traffic_light.get('state', 'unknown')
            
            # 更新斑马线检测
            result['crosswalk_detected'] = bool(crosswalk)
            
            # 生成过街指导
            result['crossing_guidance'] = self._generate_crossing_guidance(
                traffic_light, crosswalk, gps_data
            )
            
            # 生成安全建议
            result['safety_recommendation'] = self._generate_safety_recommendation(
                traffic_light, crosswalk
            )
        
        # 记录检测历史
        self.detection_history.append({
            'timestamp': time.time(),
            'is_intersection': result['is_intersection'],
            'confidence': result['confidence']
        })
        
        return result
    
    def _detect_traffic_light(self, frame: np.ndarray) -> Optional[Dict]:
        """检测交通信号灯"""
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 检测红色（红灯）
        red_lower = np.array([0, 50, 50])
        red_upper = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        
        # 检测黄色（黄灯）
        yellow_lower = np.array([20, 50, 50])
        yellow_upper = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        
        # 检测绿色（绿灯）
        green_lower = np.array([40, 50, 50])
        green_upper = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # 查找圆形区域（信号灯通常是圆形）
        for mask, color in [(red_mask, 'red'), (yellow_mask, 'yellow'), (green_mask, 'green')]:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 2000:  # 信号灯的典型面积
                    # 检查是否为圆形
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.7:  # 接近圆形
                            return {
                                'state': color,
                                'confidence': min(1.0, area / 1000),
                                'position': cv2.boundingRect(contour)
                            }
        
        return None
    
    def _detect_crosswalk(self, frame: np.ndarray) -> Optional[Dict]:
        """检测斑马线"""
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 使用霍夫变换检测直线
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            # 分析平行线（斑马线的特征）
            horizontal_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = math.atan2(y2 - y1, x2 - x1) * 180 / np.pi
                if abs(angle) < 15:  # 水平线
                    horizontal_lines.append(line[0])
            
            # 检查是否有足够的平行线
            if len(horizontal_lines) >= 3:
                # 计算线条间距
                y_coords = []
                for line in horizontal_lines:
                    y_coords.append((line[1] + line[3]) / 2)
                
                y_coords.sort()
                distances = [y_coords[i+1] - y_coords[i] for i in range(len(y_coords)-1)]
                
                # 检查间距是否相对均匀（斑马线特征）
                if distances and max(distances) - min(distances) < 20:
                    return {
                        'detected': True,
                        'confidence': min(1.0, len(horizontal_lines) / 10),
                        'line_count': len(horizontal_lines)
                    }
        
        return None
    
    def _detect_intersection_signs(self, frame: np.ndarray) -> bool:
        """检测路口标志"""
        # 这里可以添加更复杂的标志识别逻辑
        # 目前使用简单的形状检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 查找矩形轮廓（标志通常是矩形）
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 1000 < area < 10000:  # 标志的典型面积
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # 标志通常是矩形
                if 0.5 < aspect_ratio < 2.0:
                    return True
        
        return False
    
    def _analyze_gps_data(self, gps_data: Dict) -> bool:
        """分析GPS数据判断是否在路口附近"""
        # 这里可以添加GPS数据分析逻辑
        # 例如检查是否在已知的路口坐标附近
        return False
    
    def _generate_crossing_guidance(self, traffic_light: Optional[Dict], 
                                  crosswalk: Optional[Dict], 
                                  gps_data: Optional[Dict]) -> str:
        """生成过街指导"""
        if traffic_light:
            state = traffic_light.get('state', 'unknown')
            if state == 'red':
                return "红灯亮起，请等待绿灯"
            elif state == 'yellow':
                return "黄灯闪烁，请准备过街"
            elif state == 'green':
                return "绿灯亮起，可以安全过街"
        
        if crosswalk:
            return "检测到斑马线，请确认安全后过街"
        
        return "前方路口，请确认安全后过街"
    
    def _generate_safety_recommendation(self, traffic_light: Optional[Dict], 
                                      crosswalk: Optional[Dict]) -> str:
        """生成安全建议"""
        if traffic_light and traffic_light.get('state') == 'red':
            return "high_alert"
        elif traffic_light and traffic_light.get('state') == 'yellow':
            return "caution"
        elif crosswalk:
            return "proceed_with_caution"
        else:
            return "normal"

class CrowdDensityAnalyzer:
    """拥挤场所分析器"""
    
    def __init__(self):
        self.density_thresholds = {
            'low': 0.1,
            'medium': 0.3,
            'high': 0.6,
            'very_high': 0.8
        }
        self.obstacle_density_history = deque(maxlen=30)
        
    def analyze_crowd_density(self, frame: np.ndarray, detections: List[Dict]) -> Dict:
        """分析拥挤程度"""
        result = {
            'density_level': 'low',
            'density_score': 0.0,
            'obstacle_distribution': 'sparse',
            'navigation_difficulty': 'easy',
            'filtered_obstacles': [],
            'recommended_path': None
        }
        
        if not detections:
            return result
        
        # 1. 计算障碍物密度
        density_score = self._calculate_obstacle_density(frame, detections)
        result['density_score'] = density_score
        
        # 2. 确定密度等级
        result['density_level'] = self._classify_density_level(density_score)
        
        # 3. 分析障碍物分布
        result['obstacle_distribution'] = self._analyze_obstacle_distribution(detections)
        
        # 4. 评估导航难度
        result['navigation_difficulty'] = self._assess_navigation_difficulty(
            density_score, result['obstacle_distribution']
        )
        
        # 5. 过滤密集障碍物
        result['filtered_obstacles'] = self._filter_dense_obstacles(detections, density_score)
        
        # 6. 推荐路径
        result['recommended_path'] = self._recommend_navigation_path(
            frame, result['filtered_obstacles']
        )
        
        # 记录历史
        self.obstacle_density_history.append({
            'timestamp': time.time(),
            'density_score': density_score,
            'obstacle_count': len(detections)
        })
        
        return result
    
    def _calculate_obstacle_density(self, frame: np.ndarray, detections: List[Dict]) -> float:
        """计算障碍物密度"""
        if not detections:
            return 0.0
        
        frame_area = frame.shape[0] * frame.shape[1]
        total_obstacle_area = 0
        
        for detection in detections:
            if 'bbox' in detection:
                bbox = detection['bbox']
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                total_obstacle_area += area
        
        return total_obstacle_area / frame_area
    
    def _classify_density_level(self, density_score: float) -> str:
        """分类密度等级"""
        if density_score >= self.density_thresholds['very_high']:
            return 'very_high'
        elif density_score >= self.density_thresholds['high']:
            return 'high'
        elif density_score >= self.density_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _analyze_obstacle_distribution(self, detections: List[Dict]) -> str:
        """分析障碍物分布"""
        if not detections:
            return 'sparse'
        
        # 分析障碍物在图像中的分布
        left_obstacles = 0
        right_obstacles = 0
        center_obstacles = 0
        
        frame_width = 640  # 假设帧宽度
        left_boundary = frame_width // 3
        right_boundary = 2 * frame_width // 3
        
        for detection in detections:
            if 'bbox' in detection:
                bbox = detection['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                
                if center_x < left_boundary:
                    left_obstacles += 1
                elif center_x > right_boundary:
                    right_obstacles += 1
                else:
                    center_obstacles += 1
        
        total_obstacles = left_obstacles + right_obstacles + center_obstacles
        
        if total_obstacles == 0:
            return 'sparse'
        
        # 判断分布模式
        if center_obstacles / total_obstacles > 0.5:
            return 'concentrated'
        elif abs(left_obstacles - right_obstacles) / total_obstacles < 0.2:
            return 'balanced'
        else:
            return 'unbalanced'
    
    def _assess_navigation_difficulty(self, density_score: float, distribution: str) -> str:
        """评估导航难度"""
        if density_score > 0.6 or distribution == 'concentrated':
            return 'very_difficult'
        elif density_score > 0.3 or distribution == 'unbalanced':
            return 'difficult'
        elif density_score > 0.1:
            return 'moderate'
        else:
            return 'easy'
    
    def _filter_dense_obstacles(self, detections: List[Dict], density_score: float) -> List[Dict]:
        """过滤密集障碍物"""
        if density_score < 0.3:  # 低密度，不过滤
            return detections
        
        # 根据密度调整过滤策略
        if density_score > 0.6:  # 高密度，只保留最重要的障碍物
            # 按置信度排序，保留前50%
            sorted_detections = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)
            return sorted_detections[:len(sorted_detections)//2]
        else:  # 中等密度，保留75%
            sorted_detections = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)
            return sorted_detections[:int(len(sorted_detections) * 0.75)]
    
    def _recommend_navigation_path(self, frame: np.ndarray, obstacles: List[Dict]) -> Optional[List[Tuple[int, int]]]:
        """推荐导航路径"""
        if not obstacles:
            return None
        
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # 分析障碍物分布，找到相对安全的路径
        left_obstacles = []
        right_obstacles = []
        
        for obstacle in obstacles:
            if 'bbox' in obstacle:
                bbox = obstacle['bbox']
                obj_center_x = (bbox[0] + bbox[2]) / 2
                if obj_center_x < center_x:
                    left_obstacles.append(obstacle)
                else:
                    right_obstacles.append(obstacle)
        
        # 选择障碍物较少的一侧
        if len(left_obstacles) < len(right_obstacles):
            # 推荐左侧路径
            return [(center_x - 80, center_y), (center_x - 120, center_y + 30)]
        else:
            # 推荐右侧路径
            return [(center_x + 80, center_y), (center_x + 120, center_y + 30)]

class WeatherConditionDetector:
    """天气条件检测器"""
    
    def __init__(self):
        self.weather_keywords = ['雨', '雪', '雾', '风', '冰', '湿滑']
        self.visibility_threshold = 0.3  # 能见度阈值
        self.detection_history = deque(maxlen=20)
        
    def detect_weather_conditions(self, frame: np.ndarray) -> Dict:
        """检测天气条件"""
        result = {
            'weather_type': 'clear',
            'visibility_level': 'good',
            'safety_impact': 'low',
            'recommendations': []
        }
        
        # 1. 检测雨天条件
        rain_detection = self._detect_rain_conditions(frame)
        
        # 2. 检测雾天条件
        fog_detection = self._detect_fog_conditions(frame)
        
        # 3. 检测雪天条件
        snow_detection = self._detect_snow_conditions(frame)
        
        # 4. 综合评估
        weather_conditions = [rain_detection, fog_detection, snow_detection]
        active_conditions = [cond for cond in weather_conditions if cond['detected']]
        
        if active_conditions:
            # 选择最严重的天气条件
            primary_condition = max(active_conditions, key=lambda x: x['severity'])
            result['weather_type'] = primary_condition['type']
            result['visibility_level'] = primary_condition['visibility']
            result['safety_impact'] = primary_condition['safety_impact']
            result['recommendations'] = primary_condition['recommendations']
        
        return result
    
    def _detect_rain_conditions(self, frame: np.ndarray) -> Dict:
        """检测雨天条件"""
        # 检测水滴和湿润表面
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测高对比度边缘（水滴特征）
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 检测湿润表面（低对比度区域）
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        contrast = np.std(blurred)
        
        detected = edge_density > 0.1 and contrast < 30
        
        return {
            'detected': detected,
            'type': 'rain',
            'severity': edge_density * 2,
            'visibility': 'poor' if edge_density > 0.15 else 'moderate',
            'safety_impact': 'high' if edge_density > 0.15 else 'medium',
            'recommendations': ['路面湿滑，请减速慢行', '注意防滑'] if detected else []
        }
    
    def _detect_fog_conditions(self, frame: np.ndarray) -> Dict:
        """检测雾天条件"""
        # 检测整体亮度和对比度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # 雾天特征：高亮度、低对比度
        detected = mean_brightness > 150 and contrast < 25
        
        return {
            'detected': detected,
            'type': 'fog',
            'severity': (mean_brightness - 150) / 100,
            'visibility': 'very_poor' if contrast < 15 else 'poor',
            'safety_impact': 'very_high' if contrast < 15 else 'high',
            'recommendations': ['能见度低，请谨慎前行', '建议暂停导航'] if detected else []
        }
    
    def _detect_snow_conditions(self, frame: np.ndarray) -> Dict:
        """检测雪天条件"""
        # 检测白色区域和雪花特征
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 检测白色区域（雪）
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        
        white_ratio = np.sum(white_mask > 0) / white_mask.size
        detected = white_ratio > 0.3
        
        return {
            'detected': detected,
            'type': 'snow',
            'severity': white_ratio,
            'visibility': 'poor' if white_ratio > 0.5 else 'moderate',
            'safety_impact': 'high' if white_ratio > 0.5 else 'medium',
            'recommendations': ['路面可能有积雪，请小心行走'] if detected else []
        }

class LightingConditionDetector:
    """光照条件检测器"""
    
    def __init__(self):
        self.brightness_thresholds = {
            'very_dark': 50,
            'dark': 100,
            'normal': 150,
            'bright': 200
        }
        self.detection_history = deque(maxlen=20)
        
    def detect_lighting_conditions(self, frame: np.ndarray) -> Dict:
        """检测光照条件"""
        result = {
            'lighting_level': 'normal',
            'visibility_quality': 'good',
            'safety_impact': 'low',
            'recommendations': []
        }
        
        # 计算整体亮度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # 计算对比度
        contrast = np.std(gray)
        
        # 检测光照等级
        if mean_brightness < self.brightness_thresholds['very_dark']:
            result['lighting_level'] = 'very_dark'
            result['visibility_quality'] = 'very_poor'
            result['safety_impact'] = 'very_high'
            result['recommendations'] = ['环境过暗，建议使用照明设备']
        elif mean_brightness < self.brightness_thresholds['dark']:
            result['lighting_level'] = 'dark'
            result['visibility_quality'] = 'poor'
            result['safety_impact'] = 'high'
            result['recommendations'] = ['环境较暗，请谨慎前行']
        elif mean_brightness > self.brightness_thresholds['bright']:
            result['lighting_level'] = 'bright'
            result['visibility_quality'] = 'good' if contrast > 30 else 'poor'
            result['safety_impact'] = 'low' if contrast > 30 else 'medium'
            if contrast <= 30:
                result['recommendations'] = ['强光环境，注意阴影区域']
        else:
            result['lighting_level'] = 'normal'
            result['visibility_quality'] = 'good' if contrast > 25 else 'moderate'
            result['safety_impact'] = 'low' if contrast > 25 else 'medium'
        
        return result

class SurfaceConditionDetector:
    """路面条件检测器"""
    
    def __init__(self):
        self.surface_types = ['smooth', 'rough', 'wet', 'icy', 'uneven']
        self.detection_history = deque(maxlen=20)
        
    def detect_surface_conditions(self, frame: np.ndarray) -> Dict:
        """检测路面条件"""
        result = {
            'surface_type': 'smooth',
            'safety_level': 'safe',
            'walking_difficulty': 'easy',
            'recommendations': []
        }
        
        # 1. 检测路面纹理
        texture_analysis = self._analyze_surface_texture(frame)
        
        # 2. 检测湿滑表面
        wet_detection = self._detect_wet_surfaces(frame)
        
        # 3. 检测不平整路面
        uneven_detection = self._detect_uneven_surfaces(frame)
        
        # 综合评估
        if wet_detection['detected']:
            result['surface_type'] = 'wet'
            result['safety_level'] = 'caution'
            result['walking_difficulty'] = 'moderate'
            result['recommendations'] = ['路面湿滑，请小心行走']
        elif uneven_detection['detected']:
            result['surface_type'] = 'uneven'
            result['safety_level'] = 'caution'
            result['walking_difficulty'] = 'difficult'
            result['recommendations'] = ['路面不平，请放慢脚步']
        elif texture_analysis['roughness'] > 0.3:
            result['surface_type'] = 'rough'
            result['safety_level'] = 'moderate'
            result['walking_difficulty'] = 'moderate'
            result['recommendations'] = ['路面较粗糙，注意脚下']
        
        return result
    
    def _analyze_surface_texture(self, frame: np.ndarray) -> Dict:
        """分析路面纹理"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 使用拉普拉斯算子检测纹理
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        roughness = np.var(laplacian)
        
        return {
            'roughness': min(1.0, roughness / 1000),
            'texture_density': np.sum(laplacian > 50) / laplacian.size
        }
    
    def _detect_wet_surfaces(self, frame: np.ndarray) -> Dict:
        """检测湿滑表面"""
        # 检测反射和湿润特征
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 检测高亮度区域（反射）
        bright_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
        bright_ratio = np.sum(bright_mask > 0) / bright_mask.size
        
        # 检测低对比度区域（湿润表面）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray)
        
        detected = bright_ratio > 0.1 and contrast < 40
        
        return {
            'detected': detected,
            'confidence': bright_ratio * 2,
            'wetness_level': 'high' if bright_ratio > 0.2 else 'moderate'
        }
    
    def _detect_uneven_surfaces(self, frame: np.ndarray) -> Dict:
        """检测不平整路面"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 使用边缘检测识别不规则形状
        edges = cv2.Canny(gray, 50, 150)
        
        # 检测垂直边缘（台阶、裂缝等）
        vertical_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        vertical_edges = cv2.filter2D(edges, -1, vertical_kernel)
        
        vertical_density = np.sum(vertical_edges > 50) / vertical_edges.size
        detected = vertical_density > 0.05
        
        return {
            'detected': detected,
            'confidence': vertical_density * 10,
            'unevenness_level': 'high' if vertical_density > 0.1 else 'moderate'
        }

class EnvironmentDetector:
    """环境检测主模块"""
    
    def __init__(self):
        self.construction_detector = ConstructionZoneDetector()
        self.intersection_detector = IntersectionDetector()
        self.crowd_analyzer = CrowdDensityAnalyzer()
        self.weather_detector = WeatherConditionDetector()
        self.lighting_detector = LightingConditionDetector()
        self.surface_detector = SurfaceConditionDetector()
        self.detection_history = deque(maxlen=50)
        
    def detect_environment(self, frame: np.ndarray, detections: List[Dict], 
                          gps_data: Optional[Dict] = None) -> Dict:
        """综合环境检测"""
        result = {
            'timestamp': time.time(),
            'construction_zone': None,
            'intersection': None,
            'crowd_density': None,
            'weather_conditions': None,
            'lighting_conditions': None,
            'surface_conditions': None,
            'overall_safety_level': 'safe',
            'safety_score': 1.0,  # 0-1，1为最安全
            'navigation_guidance': [],
            'warnings': [],
            'emergency_alerts': []
        }
        
        # 1. 施工区域检测
        construction_result = self.construction_detector.detect_construction_zone(frame)
        if construction_result['is_construction_zone']:
            result['construction_zone'] = construction_result
        
        # 2. 十字路口检测
        intersection_result = self.intersection_detector.detect_intersection(frame, gps_data)
        if intersection_result['is_intersection']:
            result['intersection'] = intersection_result
        
        # 3. 拥挤场所分析
        crowd_result = self.crowd_analyzer.analyze_crowd_density(frame, detections)
        result['crowd_density'] = crowd_result
        
        # 4. 天气条件检测
        weather_result = self.weather_detector.detect_weather_conditions(frame)
        if weather_result['weather_type'] != 'clear':
            result['weather_conditions'] = weather_result
        
        # 5. 光照条件检测
        lighting_result = self.lighting_detector.detect_lighting_conditions(frame)
        if lighting_result['lighting_level'] != 'normal':
            result['lighting_conditions'] = lighting_result
        
        # 6. 路面条件检测
        surface_result = self.surface_detector.detect_surface_conditions(frame)
        if surface_result['surface_type'] != 'smooth':
            result['surface_conditions'] = surface_result
        
        # 7. 综合安全评估
        safety_assessment = self._comprehensive_safety_assessment(
            construction_result, intersection_result, crowd_result,
            weather_result, lighting_result, surface_result
        )
        result['overall_safety_level'] = safety_assessment['level']
        result['safety_score'] = safety_assessment['score']
        
        # 8. 生成导航指导
        result['navigation_guidance'] = self._generate_comprehensive_guidance(
            construction_result, intersection_result, crowd_result,
            weather_result, lighting_result, surface_result
        )
        
        # 9. 生成警告和紧急警报
        warnings_and_alerts = self._generate_comprehensive_warnings(
            construction_result, intersection_result, crowd_result,
            weather_result, lighting_result, surface_result
        )
        result['warnings'] = warnings_and_alerts['warnings']
        result['emergency_alerts'] = warnings_and_alerts['emergency_alerts']
        
        # 记录检测历史
        self.detection_history.append(result)
        
        return result
    
    def _comprehensive_safety_assessment(self, construction: Dict, intersection: Dict, 
                                        crowd: Dict, weather: Dict, lighting: Dict, 
                                        surface: Dict) -> Dict:
        """综合安全评估"""
        safety_factors = []
        
        # 1. 施工区域安全评分
        if construction.get('is_construction_zone'):
            if construction['safety_level'] == 'high_risk':
                safety_factors.append({'factor': 'construction', 'score': 0.2, 'weight': 0.3})
            elif construction['safety_level'] == 'medium_risk':
                safety_factors.append({'factor': 'construction', 'score': 0.5, 'weight': 0.3})
            else:
                safety_factors.append({'factor': 'construction', 'score': 0.8, 'weight': 0.3})
        
        # 2. 路口安全评分
        if intersection.get('is_intersection'):
            if intersection['safety_recommendation'] == 'high_alert':
                safety_factors.append({'factor': 'intersection', 'score': 0.1, 'weight': 0.25})
            elif intersection['safety_recommendation'] == 'caution':
                safety_factors.append({'factor': 'intersection', 'score': 0.4, 'weight': 0.25})
            else:
                safety_factors.append({'factor': 'intersection', 'score': 0.7, 'weight': 0.25})
        
        # 3. 拥挤程度安全评分
        if crowd.get('density_level') == 'very_high':
            safety_factors.append({'factor': 'crowd', 'score': 0.2, 'weight': 0.2})
        elif crowd.get('density_level') == 'high':
            safety_factors.append({'factor': 'crowd', 'score': 0.4, 'weight': 0.2})
        elif crowd.get('density_level') == 'medium':
            safety_factors.append({'factor': 'crowd', 'score': 0.6, 'weight': 0.2})
        else:
            safety_factors.append({'factor': 'crowd', 'score': 0.9, 'weight': 0.2})
        
        # 4. 天气条件安全评分
        if weather.get('weather_type') != 'clear':
            if weather['safety_impact'] == 'very_high':
                safety_factors.append({'factor': 'weather', 'score': 0.1, 'weight': 0.15})
            elif weather['safety_impact'] == 'high':
                safety_factors.append({'factor': 'weather', 'score': 0.3, 'weight': 0.15})
            elif weather['safety_impact'] == 'medium':
                safety_factors.append({'factor': 'weather', 'score': 0.6, 'weight': 0.15})
            else:
                safety_factors.append({'factor': 'weather', 'score': 0.8, 'weight': 0.15})
        
        # 5. 光照条件安全评分
        if lighting.get('lighting_level') != 'normal':
            if lighting['safety_impact'] == 'very_high':
                safety_factors.append({'factor': 'lighting', 'score': 0.2, 'weight': 0.1})
            elif lighting['safety_impact'] == 'high':
                safety_factors.append({'factor': 'lighting', 'score': 0.4, 'weight': 0.1})
            elif lighting['safety_impact'] == 'medium':
                safety_factors.append({'factor': 'lighting', 'score': 0.7, 'weight': 0.1})
            else:
                safety_factors.append({'factor': 'lighting', 'score': 0.9, 'weight': 0.1})
        
        # 6. 路面条件安全评分
        if surface.get('surface_type') != 'smooth':
            if surface['safety_level'] == 'caution':
                safety_factors.append({'factor': 'surface', 'score': 0.5, 'weight': 0.1})
            elif surface['safety_level'] == 'moderate':
                safety_factors.append({'factor': 'surface', 'score': 0.7, 'weight': 0.1})
            else:
                safety_factors.append({'factor': 'surface', 'score': 0.9, 'weight': 0.1})
        
        # 计算加权平均安全分数
        if not safety_factors:
            return {'level': 'safe', 'score': 1.0, 'factors': []}
        
        total_weighted_score = sum(factor['score'] * factor['weight'] for factor in safety_factors)
        total_weight = sum(factor['weight'] for factor in safety_factors)
        final_score = total_weighted_score / total_weight if total_weight > 0 else 1.0
        
        # 确定安全等级
        if final_score < 0.3:
            level = 'high_risk'
        elif final_score < 0.6:
            level = 'medium_risk'
        else:
            level = 'safe'
        
        return {
            'level': level,
            'score': final_score,
            'factors': safety_factors
        }
    
    def _assess_overall_safety(self, construction: Dict, intersection: Dict, crowd: Dict) -> str:
        """评估整体安全等级（兼容旧版本）"""
        safety_scores = []
        
        # 施工区域安全评分
        if construction.get('is_construction_zone'):
            if construction['safety_level'] == 'high_risk':
                safety_scores.append(0.2)
            elif construction['safety_level'] == 'medium_risk':
                safety_scores.append(0.5)
            else:
                safety_scores.append(0.8)
        
        # 路口安全评分
        if intersection.get('is_intersection'):
            if intersection['safety_recommendation'] == 'high_alert':
                safety_scores.append(0.1)
            elif intersection['safety_recommendation'] == 'caution':
                safety_scores.append(0.4)
            else:
                safety_scores.append(0.7)
        
        # 拥挤程度安全评分
        if crowd.get('density_level') == 'very_high':
            safety_scores.append(0.2)
        elif crowd.get('density_level') == 'high':
            safety_scores.append(0.4)
        elif crowd.get('density_level') == 'medium':
            safety_scores.append(0.6)
        else:
            safety_scores.append(0.9)
        
        if not safety_scores:
            return 'safe'
        
        avg_safety = sum(safety_scores) / len(safety_scores)
        
        if avg_safety < 0.3:
            return 'high_risk'
        elif avg_safety < 0.6:
            return 'medium_risk'
        else:
            return 'safe'
    
    def _generate_comprehensive_guidance(self, construction: Dict, intersection: Dict, 
                                        crowd: Dict, weather: Dict, lighting: Dict, 
                                        surface: Dict) -> List[str]:
        """生成综合导航指导"""
        guidance = []
        
        # 施工区域指导
        if construction.get('is_construction_zone'):
            if construction.get('bypass_path'):
                guidance.append("检测到施工区域，建议绕行")
            else:
                guidance.append("前方施工区域，请谨慎通行")
        
        # 路口指导
        if intersection.get('is_intersection'):
            if intersection.get('crossing_guidance'):
                guidance.append(intersection['crossing_guidance'])
        
        # 拥挤场所指导
        if crowd.get('density_level') in ['high', 'very_high']:
            if crowd.get('recommended_path'):
                guidance.append("环境拥挤，建议选择相对安全的路径")
            else:
                guidance.append("环境拥挤，请减速慢行")
        
        # 天气条件指导
        if weather.get('weather_type') != 'clear':
            guidance.extend(weather.get('recommendations', []))
        
        # 光照条件指导
        if lighting.get('lighting_level') != 'normal':
            guidance.extend(lighting.get('recommendations', []))
        
        # 路面条件指导
        if surface.get('surface_type') != 'smooth':
            guidance.extend(surface.get('recommendations', []))
        
        return guidance
    
    def _generate_navigation_guidance(self, construction: Dict, intersection: Dict, crowd: Dict) -> List[str]:
        """生成导航指导（兼容旧版本）"""
        guidance = []
        
        # 施工区域指导
        if construction.get('is_construction_zone'):
            if construction.get('bypass_path'):
                guidance.append("检测到施工区域，建议绕行")
            else:
                guidance.append("前方施工区域，请谨慎通行")
        
        # 路口指导
        if intersection.get('is_intersection'):
            if intersection.get('crossing_guidance'):
                guidance.append(intersection['crossing_guidance'])
        
        # 拥挤场所指导
        if crowd.get('density_level') in ['high', 'very_high']:
            if crowd.get('recommended_path'):
                guidance.append("环境拥挤，建议选择相对安全的路径")
            else:
                guidance.append("环境拥挤，请减速慢行")
        
        return guidance
    
    def _generate_comprehensive_warnings(self, construction: Dict, intersection: Dict, 
                                        crowd: Dict, weather: Dict, lighting: Dict, 
                                        surface: Dict) -> Dict:
        """生成综合警告和紧急警报"""
        warnings = []
        emergency_alerts = []
        
        # 施工区域警告
        if construction.get('is_construction_zone') and construction['safety_level'] == 'high_risk':
            emergency_alerts.append("🚨 高风险施工区域，请立即停止前进")
        elif construction.get('is_construction_zone'):
            warnings.append("⚠️ 检测到施工区域，请谨慎通行")
        
        # 路口警告
        if intersection.get('is_intersection') and intersection['safety_recommendation'] == 'high_alert':
            emergency_alerts.append("🚨 路口红灯，禁止通行")
        elif intersection.get('is_intersection'):
            warnings.append("⚠️ 前方路口，请注意交通信号")
        
        # 拥挤场所警告
        if crowd.get('density_level') == 'very_high':
            emergency_alerts.append("🚨 环境极度拥挤，建议暂停导航")
        elif crowd.get('density_level') == 'high':
            warnings.append("⚠️ 环境拥挤，请减速慢行")
        
        # 天气条件警告
        if weather.get('weather_type') != 'clear':
            if weather['safety_impact'] == 'very_high':
                emergency_alerts.append(f"🚨 恶劣天气条件：{weather['weather_type']}，建议暂停导航")
            elif weather['safety_impact'] == 'high':
                warnings.append(f"⚠️ 天气条件不佳：{weather['weather_type']}，请谨慎前行")
            else:
                warnings.append(f"⚠️ 天气条件：{weather['weather_type']}，请注意安全")
        
        # 光照条件警告
        if lighting.get('lighting_level') == 'very_dark':
            emergency_alerts.append("🚨 环境过暗，建议使用照明设备或暂停导航")
        elif lighting.get('lighting_level') == 'dark':
            warnings.append("⚠️ 环境较暗，请谨慎前行")
        elif lighting.get('lighting_level') == 'bright':
            warnings.append("⚠️ 强光环境，注意阴影区域")
        
        # 路面条件警告
        if surface.get('surface_type') == 'wet':
            warnings.append("⚠️ 路面湿滑，请小心行走")
        elif surface.get('surface_type') == 'uneven':
            warnings.append("⚠️ 路面不平，请放慢脚步")
        elif surface.get('surface_type') == 'rough':
            warnings.append("⚠️ 路面较粗糙，注意脚下")
        
        return {
            'warnings': warnings,
            'emergency_alerts': emergency_alerts
        }
    
    def _generate_warnings(self, construction: Dict, intersection: Dict, crowd: Dict) -> List[str]:
        """生成警告信息（兼容旧版本）"""
        warnings = []
        
        # 施工区域警告
        if construction.get('is_construction_zone') and construction['safety_level'] == 'high_risk':
            warnings.append("⚠️ 高风险施工区域，请立即停止前进")
        
        # 路口警告
        if intersection.get('is_intersection') and intersection['safety_recommendation'] == 'high_alert':
            warnings.append("⚠️ 路口红灯，禁止通行")
        
        # 拥挤场所警告
        if crowd.get('density_level') == 'very_high':
            warnings.append("⚠️ 环境极度拥挤，建议暂停导航")
        
        return warnings

# 使用示例
if __name__ == "__main__":
    # 创建环境检测器
    env_detector = EnvironmentDetector()
    
    # 模拟检测结果
    mock_detections = [
        {'bbox': [100, 100, 150, 200], 'confidence': 0.8, 'class': 'person'},
        {'bbox': [300, 150, 400, 250], 'confidence': 0.9, 'class': 'obstacle'},
    ]
    
    # 模拟处理帧
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = env_detector.detect_environment(mock_frame, mock_detections)
    
    print("环境检测结果:", json.dumps(result, indent=2, ensure_ascii=False))

