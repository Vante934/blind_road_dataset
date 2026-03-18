#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库服务封装模块
提供用户数据库操作和检测记录管理功能
"""

from typing import List, Dict, Optional
from datetime import datetime
from models.database import DatabaseManager, User, Detection, NavigationRecord

class DatabaseService:
    """数据库服务类"""
    
    def __init__(self, database_url: str = None):
        """
        初始化数据库服务
        
        Args:
            database_url: 数据库连接URL
        """
        self.db_manager = DatabaseManager(database_url)
    
    # 用户相关操作
    def get_user_by_username(self, username: str) -> Optional[User]:
        """
        根据用户名获取用户信息
        
        Args:
            username: 用户名
            
        Returns:
            User对象或None
        """
        session = self.db_manager.get_session()
        try:
            return session.query(User).filter_by(username=username).first()
        finally:
            session.close()
    
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """
        根据用户ID获取用户信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            User对象或None
        """
        session = self.db_manager.get_session()
        try:
            return session.query(User).filter_by(id=user_id).first()
        finally:
            session.close()
    
    def create_user(self, username: str, email: str, password_hash: str) -> User:
        """
        创建新用户
        
        Args:
            username: 用户名
            email: 邮箱
            password_hash: 密码哈希
            
        Returns:
            创建的User对象
        """
        session = self.db_manager.get_session()
        try:
            # 检查用户名是否已存在
            existing_user = session.query(User).filter_by(username=username).first()
            if existing_user:
                raise ValueError(f"用户名 '{username}' 已存在")
            
            # 创建新用户
            user = User(
                username=username,
                email=email,
                password_hash=password_hash,
                is_active=True
            )
            session.add(user)
            session.commit()
            session.refresh(user)
            return user
        finally:
            session.close()
    
    def update_user(self, user_id: int, **kwargs) -> Optional[User]:
        """
        更新用户信息
        
        Args:
            user_id: 用户ID
            **kwargs: 要更新的字段
            
        Returns:
            更新后的User对象或None
        """
        session = self.db_manager.get_session()
        try:
            user = session.query(User).filter_by(id=user_id).first()
            if not user:
                return None
            
            # 更新字段
            for key, value in kwargs.items():
                if hasattr(user, key):
                    setattr(user, key, value)
            
            session.commit()
            session.refresh(user)
            return user
        finally:
            session.close()
    
    def delete_user(self, user_id: int) -> bool:
        """
        删除用户
        
        Args:
            user_id: 用户ID
            
        Returns:
            是否删除成功
        """
        session = self.db_manager.get_session()
        try:
            user = session.query(User).filter_by(id=user_id).first()
            if not user:
                return False
            
            session.delete(user)
            session.commit()
            return True
        finally:
            session.close()
    
    def get_all_users(self) -> List[User]:
        """
        获取所有用户
        
        Returns:
            用户列表
        """
        session = self.db_manager.get_session()
        try:
            return session.query(User).all()
        finally:
            session.close()
    
    # 检测记录相关操作
    def save_detection(self, user_id: int, image_path: str, detection_result: List[Dict], 
                     detection_type: str, confidence: float = None) -> Detection:
        """
        保存检测记录
        
        Args:
            user_id: 用户ID
            image_path: 图像路径
            detection_result: 检测结果
            detection_type: 检测类型
            confidence: 置信度（如果为None，则自动计算）
            
        Returns:
            创建的Detection对象
        """
        session = self.db_manager.get_session()
        try:
            # 计算置信度（如果未提供）
            if confidence is None and detection_result:
                # 确保detection_result中的元素是字典
                confidence_values = []
                for d in detection_result:
                    if isinstance(d, dict):
                        confidence_values.append(d.get('confidence', 0))
                if confidence_values:
                    confidence = max(confidence_values)
                else:
                    confidence = 0
            
            # 创建检测记录
            detection = Detection(
                user_id=user_id,
                image_path=image_path,
                detection_result=detection_result,
                detection_type=detection_type,
                confidence=confidence or 0
            )
            session.add(detection)
            session.commit()
            session.refresh(detection)
            return detection
        finally:
            session.close()
    
    def get_user_detections(self, user_id: int, limit: int = 100) -> List[Detection]:
        """
        获取用户的检测记录
        
        Args:
            user_id: 用户ID
            limit: 最大返回数量
            
        Returns:
            检测记录列表
        """
        session = self.db_manager.get_session()
        try:
            return session.query(Detection).filter_by(user_id=user_id).order_by(Detection.created_at.desc()).limit(limit).all()
        finally:
            session.close()
    
    def get_detection_by_id(self, detection_id: int) -> Optional[Detection]:
        """
        根据ID获取检测记录
        
        Args:
            detection_id: 检测记录ID
            
        Returns:
            Detection对象或None
        """
        session = self.db_manager.get_session()
        try:
            return session.query(Detection).filter_by(id=detection_id).first()
        finally:
            session.close()
    
    def get_detection_statistics(self, user_id: int) -> Dict:
        """
        获取用户的检测统计信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            统计信息字典
        """
        session = self.db_manager.get_session()
        try:
            # 获取用户的所有检测记录
            detections = session.query(Detection).filter_by(user_id=user_id).all()
            
            # 计算统计信息
            total_detections = len(detections)
            total_targets = sum(len(d.detection_result) for d in detections if d.detection_result)
            
            # 按类型统计
            type_counts = {}
            for detection in detections:
                if detection.detection_type:
                    type_counts[detection.detection_type] = type_counts.get(detection.detection_type, 0) + 1
            
            # 计算平均置信度
            confidences = [d.confidence for d in detections if d.confidence]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'total_detections': total_detections,
                'total_targets': total_targets,
                'type_counts': type_counts,
                'avg_confidence': avg_confidence,
                'user_id': user_id
            }
        finally:
            session.close()
    
    def delete_detection(self, detection_id: int) -> bool:
        """
        删除检测记录
        
        Args:
            detection_id: 检测记录ID
            
        Returns:
            是否删除成功
        """
        session = self.db_manager.get_session()
        try:
            detection = session.query(Detection).filter_by(id=detection_id).first()
            if not detection:
                return False
            
            session.delete(detection)
            session.commit()
            return True
        finally:
            session.close()
    
    # 导航记录相关操作
    def save_navigation_record(self, user_id: int, start_latitude: float, start_longitude: float, 
                             end_latitude: float, end_longitude: float, route_data: Dict, 
                             distance: float = None, duration: int = None, 
                             navigation_mode: str = 'walking', status: str = 'completed') -> NavigationRecord:
        """
        保存导航记录
        
        Args:
            user_id: 用户ID
            start_latitude: 起始纬度
            start_longitude: 起始经度
            end_latitude: 结束纬度
            end_longitude: 结束经度
            route_data: 路径数据
            distance: 距离（米）
            duration: 持续时间（秒）
            navigation_mode: 导航模式
            status: 状态
            
        Returns:
            创建的NavigationRecord对象
        """
        session = self.db_manager.get_session()
        try:
            record = NavigationRecord(
                user_id=user_id,
                start_latitude=start_latitude,
                start_longitude=start_longitude,
                end_latitude=end_latitude,
                end_longitude=end_longitude,
                route_data=route_data,
                distance=distance,
                duration=duration,
                navigation_mode=navigation_mode,
                status=status
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return record
        finally:
            session.close()
    
    def get_user_navigation_records(self, user_id: int, limit: int = 100) -> List[NavigationRecord]:
        """
        获取用户的导航记录
        
        Args:
            user_id: 用户ID
            limit: 最大返回数量
            
        Returns:
            导航记录列表
        """
        session = self.db_manager.get_session()
        try:
            return session.query(NavigationRecord).filter_by(user_id=user_id).order_by(NavigationRecord.created_at.desc()).limit(limit).all()
        finally:
            session.close()

# 全局数据库服务实例
_default_db_service = None

def get_database_service(database_url: str = None) -> DatabaseService:
    """
    获取数据库服务实例（单例模式）
    
    Args:
        database_url: 数据库连接URL
        
    Returns:
        DatabaseService实例
    """
    global _default_db_service
    
    if _default_db_service is None:
        _default_db_service = DatabaseService(database_url)
    
    return _default_db_service

if __name__ == "__main__":
    # 测试代码
    try:
        db_service = get_database_service()
        print("✅ 数据库服务初始化成功")
        
        # 测试用户操作
        users = db_service.get_all_users()
        print(f"✅ 总用户数: {len(users)}")
        for user in users:
            print(f"  - {user.username} ({user.email})")
        
        # 测试检测记录操作
        if users:
            user_id = users[0].id
            stats = db_service.get_detection_statistics(user_id)
            print(f"\n✅ 用户 {user_id} 的检测统计:")
            print(f"  总检测次数: {stats['total_detections']}")
            print(f"  总目标数: {stats['total_targets']}")
            print(f"  平均置信度: {stats['avg_confidence']:.2f}")
            print(f"  检测类型分布: {stats['type_counts']}")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
