#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库模型定义
使用SQLAlchemy ORM实现数据库操作
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import json
import os

Base = declarative_base()

class User(Base):
    """用户表"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # 关系
    detections = relationship("Detection", back_populates="user")
    navigation_records = relationship("NavigationRecord", back_populates="user")
    devices = relationship("Device", back_populates="user")

class Detection(Base):
    """检测记录表"""
    __tablename__ = 'detections'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    image_path = Column(String(255))
    image_data = Column(Text)  # Base64编码的图像数据
    detection_result = Column(JSON)  # 检测结果JSON
    confidence = Column(Float, default=0.0)
    detection_type = Column(String(50))  # 'blind_path', 'obstacle', 'environment'
    processing_time = Column(Float)  # 处理时间（秒）
    created_at = Column(DateTime, default=datetime.now, index=True)
    
    # 关系
    user = relationship("User", back_populates="detections")

class NavigationRecord(Base):
    """导航记录表"""
    __tablename__ = 'navigation_records'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    start_latitude = Column(Float, nullable=False)
    start_longitude = Column(Float, nullable=False)
    end_latitude = Column(Float, nullable=False)
    end_longitude = Column(Float, nullable=False)
    route_data = Column(JSON)  # 路径数据
    distance = Column(Float)  # 距离（米）
    duration = Column(Integer)  # 持续时间（秒）
    navigation_mode = Column(String(50))  # 'walking', 'wheelchair', etc.
    status = Column(String(50))  # 'completed', 'cancelled', 'in_progress'
    created_at = Column(DateTime, default=datetime.now, index=True)
    completed_at = Column(DateTime)
    
    # 关系
    user = relationship("User", back_populates="navigation_records")

class Device(Base):
    """设备表"""
    __tablename__ = 'devices'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    device_id = Column(String(100), unique=True, nullable=False, index=True)
    device_type = Column(String(50))  # 'android', 'ios', 'web'
    device_info = Column(JSON)  # 设备信息JSON
    capabilities = Column(JSON)  # 设备能力JSON
    last_seen = Column(DateTime, default=datetime.now)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    
    # 关系
    user = relationship("User", back_populates="devices")

class MapData(Base):
    """地图数据表"""
    __tablename__ = 'map_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    latitude = Column(Float, nullable=False, index=True)
    longitude = Column(Float, nullable=False, index=True)
    location_type = Column(String(50))  # 'blind_path', 'sidewalk', 'crosswalk', etc.
    path_type = Column(String(50))
    accessibility_score = Column(Float, default=1.0)  # 0-1，无障碍评分
    width = Column(Float)  # 宽度（米）
    obstacles = Column(JSON)  # 障碍物信息
    map_metadata = Column(JSON)  # 其他元数据
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

class VoiceLog(Base):
    """语音日志表"""
    __tablename__ = 'voice_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True, index=True)
    text = Column(Text, nullable=False)
    voice_type = Column(String(50))
    audio_path = Column(String(255))
    duration = Column(Float)  # 音频时长（秒）
    created_at = Column(DateTime, default=datetime.now, index=True)

class SystemLog(Base):
    """系统日志表"""
    __tablename__ = 'system_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    log_level = Column(String(20))  # 'INFO', 'WARNING', 'ERROR'
    module = Column(String(50))  # 'detection', 'navigation', 'voice'
    message = Column(Text)
    details = Column(JSON)
    created_at = Column(DateTime, default=datetime.now, index=True)


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, database_url: str = None):
        """
        初始化数据库管理器
        
        Args:
            database_url: 数据库连接URL
                SQLite: 'sqlite:///blind_road.db'
                PostgreSQL: 'postgresql://user:password@localhost/dbname'
        """
        if database_url is None:
            # 默认使用SQLite
            db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'blind_road.db')
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            database_url = f'sqlite:///{db_path}'
        
        self.engine = create_engine(
            database_url,
            echo=False,  # 设置为True可以看到SQL语句
            pool_pre_ping=True,  # 连接池预检查
            connect_args={'check_same_thread': False} if 'sqlite' in database_url else {}
        )
        
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """创建所有表（先删除再创建）"""
        # 先删除所有表，确保干净的环境
        Base.metadata.drop_all(self.engine)
        # 再创建所有表
        Base.metadata.create_all(self.engine)
        print("✅ 数据库表创建成功")
    
    def drop_tables(self):
        """删除所有表（谨慎使用）"""
        Base.metadata.drop_all(self.engine)
        print("⚠️ 所有数据库表已删除")
    
    def get_session(self):
        """获取数据库会话"""
        return self.SessionLocal()
    
    def init_database(self):
        """初始化数据库"""
        self.create_tables()
        print("✅ 数据库初始化完成")


# 全局数据库管理器实例
db_manager = None

def get_db_manager(database_url: str = None):
    """获取数据库管理器实例（单例模式）"""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager(database_url)
    return db_manager

def get_db_session():
    """获取数据库会话（用于依赖注入）"""
    db = get_db_manager().get_session()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    # 测试数据库初始化
    print("=" * 60)
    print("数据库初始化测试")
    print("=" * 60)
    
    manager = DatabaseManager()
    manager.init_database()
    
    # 测试插入数据
    session = manager.get_session()
    try:
        # 创建测试用户
        test_user = User(
            username="test_user",
            email="test@example.com",
            password_hash="hashed_password_here"
        )
        session.add(test_user)
        session.commit()
        print("✅ 测试用户创建成功")
        
        # 查询用户
        user = session.query(User).filter_by(username="test_user").first()
        if user:
            print(f"✅ 查询成功: {user.username}, {user.email}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        session.rollback()
    finally:
        session.close()
    
    print("=" * 60)
    print("数据库测试完成")
    print("=" * 60)
