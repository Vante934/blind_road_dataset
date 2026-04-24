#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强导航系统集成模块 - 第9天核心功能开发
实现路径规划、导航算法和实时指导
"""

import os
import sys
import json
import time
import math
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NavigationMode(Enum):
    """导航模式"""
    WALKING = "walking"
    WHEELCHAIR = "wheelchair"
    GUIDE_DOG = "guide_dog"
    ASSISTIVE_TECH = "assistive_tech"

class PathType(Enum):
    """路径类型"""
    BLIND_PATH = "blind_path"
    SIDEWALK = "sidewalk"
    CROSSWALK = "crosswalk"
    STAIRS = "stairs"
    RAMP = "ramp"
    ELEVATOR = "elevator"

class ObstacleSeverity(Enum):
    """障碍物严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Location:
    """位置信息"""
    latitude: float
    longitude: float
    altitude: float = 0.0
    accuracy: float = 5.0  # 米
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class PathSegment:
    """路径段"""
    start_point: Location
    end_point: Location
    path_type: PathType
    length: float
    direction: float  # 角度，0-360度
    obstacles: List[Dict[str, Any]] = None
    accessibility_score: float = 1.0  # 0-1，1表示完全无障碍
    
    def __post_init__(self):
        if self.obstacles is None:
            self.obstacles = []

@dataclass
class NavigationInstruction:
    """导航指令"""
    instruction_type: str  # "turn_left", "turn_right", "go_straight", "stop", "warning"
    direction: float
    distance: float
    description: str
    voice_message: str
    priority: int = 1  # 1-5，5最高优先级
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class RouteInfo:
    """路线信息"""
    route_id: str
    start_location: Location
    end_location: Location
    total_distance: float
    estimated_time: float  # 分钟
    path_segments: List[PathSegment]
    instructions: List[NavigationInstruction]
    accessibility_score: float
    difficulty_level: int  # 1-5
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class EnhancedNavigationSystem:
    """增强导航系统"""
    
    def __init__(self, config_path: str = "navigation_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # 导航状态
        self.current_location = None
        self.current_route = None
        self.navigation_mode = NavigationMode.WALKING
        self.is_navigating = False
        
        # 路径规划参数
        self.max_route_distance = 5000  # 米
        self.preferred_path_types = [PathType.BLIND_PATH, PathType.SIDEWALK]
        self.avoid_obstacles = True
        
        # 统计信息
        self.navigation_stats = {
            "routes_planned": 0,
            "routes_completed": 0,
            "total_distance": 0.0,
            "total_time": 0.0,
            "obstacles_avoided": 0
        }
        
        # 初始化地图数据
        self.map_data = self._load_map_data()
        
        logger.info("✅ 增强导航系统初始化成功")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        default_config = {
            "navigation": {
                "max_distance": 5000,
                "preferred_path_types": ["blind_path", "sidewalk"],
                "avoid_obstacles": True,
                "voice_guidance": True,
                "haptic_feedback": True
            },
            "pathfinding": {
                "algorithm": "a_star",
                "heuristic_weight": 1.0,
                "max_iterations": 10000
            },
            "accessibility": {
                "min_width": 0.8,  # 米
                "max_slope": 0.05,  # 5%
                "avoid_stairs": True,
                "prefer_ramps": True
            }
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # 合并默认配置
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                        elif isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                if sub_key not in config[key]:
                                    config[key][sub_key] = sub_value
                    return config
            else:
                # 保存默认配置
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2, ensure_ascii=False)
                return default_config
                
        except Exception as e:
            logger.error(f"❌ 加载配置失败: {e}")
            return default_config
    
    def _load_map_data(self) -> Dict[str, Any]:
        """加载地图数据"""
        # 这里应该加载真实的地图数据
        # 目前使用模拟数据
        return {
            "nodes": [
                {"id": 1, "lat": 39.9042, "lng": 116.4074, "type": "intersection"},
                {"id": 2, "lat": 39.9045, "lng": 116.4077, "type": "landmark"},
                {"id": 3, "lat": 39.9048, "lng": 116.4080, "type": "intersection"},
            ],
            "edges": [
                {"from": 1, "to": 2, "type": "blind_path", "length": 100, "accessibility": 0.9},
                {"from": 2, "to": 3, "type": "sidewalk", "length": 150, "accessibility": 0.8},
            ],
            "obstacles": [
                {"lat": 39.9043, "lng": 116.4075, "type": "construction", "severity": "medium"},
            ]
        }
    
    def plan_route(self, start_location: Location, end_location: Location, 
                   mode: NavigationMode = None) -> Optional[RouteInfo]:
        """规划路线"""
        try:
            if mode:
                self.navigation_mode = mode
            
            logger.info(f"开始规划路线: {start_location} -> {end_location}")
            
            # 计算直线距离
            distance = self._calculate_distance(start_location, end_location)
            
            if distance > self.max_route_distance:
                logger.warning(f"路线距离过长: {distance}m > {self.max_route_distance}m")
                return None
            
            # 使用A*算法规划路径
            path_segments = self._plan_path_segments(start_location, end_location)
            
            if not path_segments:
                logger.warning("无法找到可行路径")
                return None
            
            # 生成导航指令
            instructions = self._generate_instructions(path_segments)
            
            # 计算路线统计
            total_distance = sum(segment.length for segment in path_segments)
            estimated_time = self._estimate_travel_time(total_distance, path_segments)
            accessibility_score = self._calculate_accessibility_score(path_segments)
            difficulty_level = self._calculate_difficulty_level(path_segments)
            
            # 创建路线信息
            route = RouteInfo(
                route_id=f"route_{int(time.time())}",
                start_location=start_location,
                end_location=end_location,
                total_distance=total_distance,
                estimated_time=estimated_time,
                path_segments=path_segments,
                instructions=instructions,
                accessibility_score=accessibility_score,
                difficulty_level=difficulty_level
            )
            
            # 更新统计
            self.navigation_stats["routes_planned"] += 1
            
            logger.info(f"✅ 路线规划完成: {len(path_segments)}段路径, {total_distance:.1f}m, {estimated_time:.1f}分钟")
            
            return route
            
        except Exception as e:
            logger.error(f"❌ 路线规划失败: {e}")
            return None
    
    def _plan_path_segments(self, start: Location, end: Location) -> List[PathSegment]:
        """规划路径段"""
        segments = []
        
        try:
            # 简化的路径规划算法
            # 在实际应用中，这里应该使用更复杂的路径规划算法
            
            # 计算中间点
            mid_lat = (start.latitude + end.latitude) / 2
            mid_lng = (start.longitude + end.longitude) / 2
            mid_point = Location(mid_lat, mid_lng)
            
            # 创建路径段
            # 第一段：起点到中点
            segment1 = PathSegment(
                start_point=start,
                end_point=mid_point,
                path_type=PathType.BLIND_PATH,
                length=self._calculate_distance(start, mid_point),
                direction=self._calculate_bearing(start, mid_point),
                accessibility_score=0.9
            )
            
            # 第二段：中点到终点
            segment2 = PathSegment(
                start_point=mid_point,
                end_point=end,
                path_type=PathType.SIDEWALK,
                length=self._calculate_distance(mid_point, end),
                direction=self._calculate_bearing(mid_point, end),
                accessibility_score=0.8
            )
            
            segments = [segment1, segment2]
            
        except Exception as e:
            logger.error(f"❌ 路径段规划失败: {e}")
        
        return segments
    
    def _generate_instructions(self, segments: List[PathSegment]) -> List[NavigationInstruction]:
        """生成导航指令"""
        instructions = []
        
        try:
            for i, segment in enumerate(segments):
                # 开始指令
                if i == 0:
                    start_instruction = NavigationInstruction(
                        instruction_type="start",
                        direction=segment.direction,
                        distance=0,
                        description="开始导航",
                        voice_message="开始导航，请跟随语音指引",
                        priority=5
                    )
                    instructions.append(start_instruction)
                
                # 直行指令
                straight_instruction = NavigationInstruction(
                    instruction_type="go_straight",
                    direction=segment.direction,
                    distance=segment.length,
                    description=f"直行{segment.length:.0f}米",
                    voice_message=f"直行{segment.length:.0f}米",
                    priority=3
                )
                instructions.append(straight_instruction)
                
                # 转弯指令（如果有下一段）
                if i < len(segments) - 1:
                    next_segment = segments[i + 1]
                    turn_angle = self._calculate_turn_angle(segment.direction, next_segment.direction)
                    
                    if abs(turn_angle) > 15:  # 需要转弯
                        turn_type = "turn_left" if turn_angle > 0 else "turn_right"
                        turn_instruction = NavigationInstruction(
                            instruction_type=turn_type,
                            direction=next_segment.direction,
                            distance=0,
                            description=f"{'左转' if turn_angle > 0 else '右转'}{abs(turn_angle):.0f}度",
                            voice_message=f"{'左转' if turn_angle > 0 else '右转'}{abs(turn_angle):.0f}度",
                            priority=4
                        )
                        instructions.append(turn_instruction)
                
                # 结束指令
                if i == len(segments) - 1:
                    end_instruction = NavigationInstruction(
                        instruction_type="arrive",
                        direction=segment.direction,
                        distance=0,
                        description="到达目的地",
                        voice_message="已到达目的地",
                        priority=5
                    )
                    instructions.append(end_instruction)
            
        except Exception as e:
            logger.error(f"❌ 生成导航指令失败: {e}")
        
        return instructions
    
    def start_navigation(self, route: RouteInfo) -> bool:
        """开始导航"""
        try:
            if not route or not route.path_segments:
                logger.error("无效的路线")
                return False
            
            self.current_route = route
            self.is_navigating = True
            
            logger.info(f"✅ 开始导航: {route.route_id}")
            logger.info(f"路线信息: {route.total_distance:.1f}m, 预计{route.estimated_time:.1f}分钟")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 开始导航失败: {e}")
            return False
    
    def update_location(self, location: Location) -> Optional[NavigationInstruction]:
        """更新位置并获取当前指令"""
        try:
            if not self.is_navigating or not self.current_route:
                return None
            
            self.current_location = location
            
            # 找到最近的路径段
            current_segment = self._find_current_segment(location)
            if not current_segment:
                return None
            
            # 找到对应的指令
            current_instruction = self._find_current_instruction(current_segment)
            
            return current_instruction
            
        except Exception as e:
            logger.error(f"❌ 更新位置失败: {e}")
            return None
    
    def stop_navigation(self) -> bool:
        """停止导航"""
        try:
            if self.is_navigating:
                self.is_navigating = False
                self.navigation_stats["routes_completed"] += 1
                
                if self.current_route:
                    self.navigation_stats["total_distance"] += self.current_route.total_distance
                    self.navigation_stats["total_time"] += self.current_route.estimated_time
                
                logger.info("✅ 导航已停止")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ 停止导航失败: {e}")
            return False
    
    def _calculate_distance(self, loc1: Location, loc2: Location) -> float:
        """计算两点间距离（米）"""
        # 使用Haversine公式
        R = 6371000  # 地球半径（米）
        
        lat1_rad = math.radians(loc1.latitude)
        lat2_rad = math.radians(loc2.latitude)
        delta_lat = math.radians(loc2.latitude - loc1.latitude)
        delta_lng = math.radians(loc2.longitude - loc1.longitude)
        
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(delta_lng / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def _calculate_bearing(self, start: Location, end: Location) -> float:
        """计算方位角（度）"""
        lat1_rad = math.radians(start.latitude)
        lat2_rad = math.radians(end.latitude)
        delta_lng = math.radians(end.longitude - start.longitude)
        
        y = math.sin(delta_lng) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lng))
        
        bearing = math.atan2(y, x)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360
        
        return bearing
    
    def _calculate_turn_angle(self, from_bearing: float, to_bearing: float) -> float:
        """计算转弯角度"""
        angle = to_bearing - from_bearing
        if angle > 180:
            angle -= 360
        elif angle < -180:
            angle += 360
        return angle
    
    def _estimate_travel_time(self, distance: float, segments: List[PathSegment]) -> float:
        """估算旅行时间（分钟）"""
        # 基础速度（米/分钟）
        base_speed = 80  # 约4.8km/h
        
        # 根据路径类型调整速度
        speed_multiplier = 1.0
        for segment in segments:
            if segment.path_type == PathType.BLIND_PATH:
                speed_multiplier *= 1.0  # 盲道速度正常
            elif segment.path_type == PathType.SIDEWALK:
                speed_multiplier *= 0.9  # 人行道稍慢
            elif segment.path_type == PathType.CROSSWALK:
                speed_multiplier *= 0.8  # 人行横道更慢
            elif segment.path_type == PathType.STAIRS:
                speed_multiplier *= 0.6  # 楼梯很慢
        
        # 根据无障碍程度调整
        avg_accessibility = sum(segment.accessibility_score for segment in segments) / len(segments)
        speed_multiplier *= avg_accessibility
        
        return distance / (base_speed * speed_multiplier)
    
    def _calculate_accessibility_score(self, segments: List[PathSegment]) -> float:
        """计算无障碍评分"""
        if not segments:
            return 0.0
        
        total_score = sum(segment.accessibility_score for segment in segments)
        return total_score / len(segments)
    
    def _calculate_difficulty_level(self, segments: List[PathSegment]) -> int:
        """计算难度等级（1-5）"""
        if not segments:
            return 5
        
        # 基于无障碍评分计算难度
        avg_accessibility = self._calculate_accessibility_score(segments)
        
        if avg_accessibility >= 0.9:
            return 1  # 很容易
        elif avg_accessibility >= 0.8:
            return 2  # 容易
        elif avg_accessibility >= 0.6:
            return 3  # 中等
        elif avg_accessibility >= 0.4:
            return 4  # 困难
        else:
            return 5  # 很困难
    
    def _find_current_segment(self, location: Location) -> Optional[PathSegment]:
        """找到当前位置对应的路径段"""
        if not self.current_route:
            return None
        
        min_distance = float('inf')
        current_segment = None
        
        for segment in self.current_route.path_segments:
            # 计算到路径段的距离
            distance = self._point_to_line_distance(location, segment.start_point, segment.end_point)
            
            if distance < min_distance:
                min_distance = distance
                current_segment = segment
        
        return current_segment
    
    def _point_to_line_distance(self, point: Location, line_start: Location, line_end: Location) -> float:
        """计算点到线段的距离"""
        # 简化的点到线段距离计算
        return min(
            self._calculate_distance(point, line_start),
            self._calculate_distance(point, line_end)
        )
    
    def _find_current_instruction(self, segment: PathSegment) -> Optional[NavigationInstruction]:
        """找到当前路径段对应的指令"""
        if not self.current_route:
            return None
        
        # 找到对应的指令
        for instruction in self.current_route.instructions:
            if instruction.direction == segment.direction:
                return instruction
        
        return None
    
    def get_navigation_status(self) -> Dict[str, Any]:
        """获取导航状态"""
        return {
            "is_navigating": self.is_navigating,
            "current_location": {
                "latitude": self.current_location.latitude if self.current_location else None,
                "longitude": self.current_location.longitude if self.current_location else None,
                "accuracy": self.current_location.accuracy if self.current_location else None
            } if self.current_location else None,
            "current_route": {
                "route_id": self.current_route.route_id if self.current_route else None,
                "total_distance": self.current_route.total_distance if self.current_route else None,
                "estimated_time": self.current_route.estimated_time if self.current_route else None,
                "accessibility_score": self.current_route.accessibility_score if self.current_route else None
            } if self.current_route else None,
            "navigation_mode": self.navigation_mode.value,
            "stats": self.navigation_stats
        }

def test_navigation_system():
    """测试导航系统"""
    print("=" * 60)
    print("🧭 测试增强导航系统")
    print("=" * 60)
    
    # 初始化导航系统
    nav_system = EnhancedNavigationSystem()
    
    # 创建测试位置
    start_location = Location(39.9042, 116.4074)  # 北京天安门
    end_location = Location(39.9048, 116.4080)     # 附近位置
    
    print(f"\n[测试] 规划路线:")
    print(f"起点: {start_location.latitude}, {start_location.longitude}")
    print(f"终点: {end_location.latitude}, {end_location.longitude}")
    
    # 规划路线
    route = nav_system.plan_route(start_location, end_location)
    
    if route:
        print(f"\n[结果] 路线规划成功:")
        print(f"路线ID: {route.route_id}")
        print(f"总距离: {route.total_distance:.1f}米")
        print(f"预计时间: {route.estimated_time:.1f}分钟")
        print(f"无障碍评分: {route.accessibility_score:.2f}")
        print(f"难度等级: {route.difficulty_level}")
        print(f"路径段数: {len(route.path_segments)}")
        print(f"指令数: {len(route.instructions)}")
        
        # 显示路径段信息
        print(f"\n[路径段] 详细信息:")
        for i, segment in enumerate(route.path_segments):
            print(f"  段{i+1}: {segment.path_type.value} - {segment.length:.1f}m - 方向{segment.direction:.1f}°")
        
        # 显示导航指令
        print(f"\n[指令] 导航指令:")
        for i, instruction in enumerate(route.instructions):
            print(f"  {i+1}. {instruction.description} - {instruction.voice_message}")
        
        # 开始导航
        print(f"\n[测试] 开始导航...")
        if nav_system.start_navigation(route):
            print("✅ 导航开始成功")
            
            # 模拟位置更新
            print(f"\n[测试] 模拟位置更新...")
            mid_location = Location(39.9045, 116.4077)  # 中间位置
            instruction = nav_system.update_location(mid_location)
            
            if instruction:
                print(f"[指令] 当前指令: {instruction.description}")
                print(f"[语音] {instruction.voice_message}")
            
            # 停止导航
            print(f"\n[测试] 停止导航...")
            if nav_system.stop_navigation():
                print("✅ 导航停止成功")
        
        # 显示统计信息
        stats = nav_system.get_navigation_status()
        print(f"\n[统计] 导航统计:")
        print(f"  规划路线数: {stats['stats']['routes_planned']}")
        print(f"  完成路线数: {stats['stats']['routes_completed']}")
        print(f"  总距离: {stats['stats']['total_distance']:.1f}米")
        print(f"  总时间: {stats['stats']['total_time']:.1f}分钟")
        
    else:
        print("❌ 路线规划失败")
    
    print("\n✅ 导航系统测试完成")

if __name__ == "__main__":
    test_navigation_system()
