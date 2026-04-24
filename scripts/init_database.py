#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库初始化脚本
用于创建数据库表和初始化数据
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import DatabaseManager, User, MapData
from datetime import datetime

def init_database():
    """初始化数据库"""
    print("=" * 60)
    print("开始初始化数据库...")
    print("=" * 60)
    
    # 创建数据库管理器
    manager = DatabaseManager()
    
    # 创建所有表
    manager.create_tables()
    
    # 初始化默认数据
    session = manager.get_session()
    try:
        # 检查是否已有默认用户
        admin_user = session.query(User).filter_by(username="admin").first()
        if not admin_user:
            # 创建默认管理员用户
            admin_user = User(
                username="admin",
                email="admin@blindroad.com",
                password_hash="admin123",  # 实际应用中应该使用哈希
                is_active=True
            )
            session.add(admin_user)
            print("✅ 创建默认管理员用户")
        
        # 初始化示例地图数据（可选）
        map_count = session.query(MapData).count()
        if map_count == 0:
            # 添加示例地图数据
            sample_map_data = [
                MapData(
                    latitude=39.9042,
                    longitude=116.4074,
                    location_type="blind_path",
                    path_type="tactile_paving",
                    accessibility_score=0.9,
                    width=0.6,
                    obstacles=[]
                ),
                MapData(
                    latitude=39.9045,
                    longitude=116.4077,
                    location_type="sidewalk",
                    path_type="concrete",
                    accessibility_score=0.8,
                    width=1.5,
                    obstacles=[]
                ),
            ]
            session.add_all(sample_map_data)
            print(f"✅ 添加 {len(sample_map_data)} 条示例地图数据")
        
        session.commit()
        print("✅ 数据库初始化完成")
        
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        session.rollback()
        raise
    finally:
        session.close()
    
    print("=" * 60)

if __name__ == "__main__":
    init_database()
