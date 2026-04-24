#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_database.py - 完全修复版
解决了数据污染和连接未关闭的问题
"""

import sys
sys.path.append(".")

from modules.database_service import DatabaseService
from models.database import DatabaseManager, User, Detection, NavigationRecord
import tempfile
import pytest
from pathlib import Path
import time
import gc


class TestDatabaseService:
    """数据库服务测试类"""
    
    @classmethod
    def setup_class(cls):
        """测试前的准备 - 创建临时数据库"""
        # 创建临时数据库文件
        cls.temp_db = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.db', 
            delete=False  # 不自动删除，我们手动控制
        )
        cls.temp_db_path = cls.temp_db.name
        cls.temp_db.close()
        
        cls.db_url = f"sqlite:///{cls.temp_db_path}"
        print(f"\n📁 临时数据库: {cls.temp_db_path}")
        
        # 创建数据库服务实例
        cls.db_service = DatabaseService(cls.db_url)
        
        # 创建数据库表
        db_manager = DatabaseManager(cls.db_url)
        db_manager.create_tables()
        
        print("✅ 测试数据库已创建")
    
    @classmethod
    def teardown_class(cls):
        """测试后的清理 - 删除临时数据库"""
        try:
            # 强制关闭所有连接
            if hasattr(cls, 'db_service'):
                # 关闭引擎
                cls.db_service.db_manager.engine.dispose()
                
                # 等待连接完全关闭
                time.sleep(0.5)
                
                # 强制垃圾回收
                del cls.db_service
                gc.collect()
                time.sleep(0.5)
            
            # 删除临时文件
            if hasattr(cls, 'temp_db_path'):
                temp_path = Path(cls.temp_db_path)
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                        print("✅ 临时数据库已删除")
                    except PermissionError as e:
                        print(f"⚠️ 临时数据库删除失败（文件被占用），系统会自动清理: {e}")
        except Exception as e:
            print(f"⚠️ 清理时发生错误: {e}")
    
    def setup_method(self, method):
        """每个测试方法执行前的准备"""
        # 清理所有表的数据（保留表结构）
        session = self.db_service.db_manager.get_session()
        try:
            session.query(Detection).delete()
            session.query(NavigationRecord).delete()
            session.query(User).delete()
            session.commit()
            print(f"\n🧹 测试前清理数据完成")
        except Exception as e:
            session.rollback()
            print(f"⚠️ 清理数据时出错: {e}")
        finally:
            session.close()
    
    def teardown_method(self, method):
        """每个测试方法执行后的清理"""
        # 确保session都被关闭
        gc.collect()  # 强制垃圾回收
    
    # =========================================================================
    # 用户管理测试
    # =========================================================================
    
    def test_create_user(self):
        """测试1: 创建用户"""
        print("\n" + "="*60)
        print("测试1: 创建用户")
        print("="*60)
        
        user = self.db_service.create_user(
            username="test_user_1",
            email="test1@example.com",
            password_hash="hashed_password1"
        )
        
        assert user is not None, "创建用户失败"
        assert user.username == "test_user_1"
        assert user.email == "test1@example.com"
        
        print(f"✅ 用户创建成功: {user.username}")
    
    def test_create_duplicate_user(self):
        """测试2: 创建重复用户（应该失败）"""
        print("\n" + "="*60)
        print("测试2: 创建重复用户")
        print("="*60)
        
        # 先创建一个用户
        user1 = self.db_service.create_user(
            username="test_user_2",
            email="test2@example.com",
            password_hash="hashed_password2"
        )
        assert user1 is not None
        print(f"   第一个用户创建成功: {user1.username}")
        
        # 尝试创建同名用户
        try:
            user2 = self.db_service.create_user(
                username="test_user_2",  # 同名
                email="test2_another@example.com",
                password_hash="hashed_password3"
            )
            assert user2 is None, "重复用户应该创建失败"
        except ValueError as e:
            print(f"✅ 正确拒绝了重复用户名: {e}")
    
    def test_get_user_by_username(self):
        """测试3: 根据用户名获取用户"""
        print("\n" + "="*60)
        print("测试3: 根据用户名获取用户")
        print("="*60)
        
        # 先创建用户
        self.db_service.create_user(
            username="test_user_3",
            email="test3@example.com",
            password_hash="hashed_password3"
        )
        
        # 根据用户名获取用户
        user = self.db_service.get_user_by_username("test_user_3")
        
        # 验证用户获取成功
        assert user is not None, "获取用户失败"
        assert user.username == "test_user_3"
        assert user.email == "test3@example.com"
        
        print(f"✅ 获取用户成功: {user.username}")
    
    def test_get_user_by_id(self):
        """测试4: 根据ID获取用户"""
        print("\n" + "="*60)
        print("测试4: 根据ID获取用户")
        print("="*60)
        
        # 创建用户
        created_user = self.db_service.create_user(
            username="test_user_4",
            email="test4@example.com",
            password_hash="hashed_password4"
        )
        
        user_id = created_user.id
        
        # 获取用户
        user = self.db_service.get_user_by_id(user_id)
        assert user is not None, "获取用户失败"
        assert user.id == user_id
        assert user.username == "test_user_4"
        
        print(f"✅ 获取用户成功: {user.username}")
        
        # 获取不存在的用户
        user_not_exist = self.db_service.get_user_by_id(99999)
        assert user_not_exist is None, "不存在的用户应该返回None"
        print("✅ 不存在的用户返回None（符合预期）")
    
    def test_update_user(self):
        """测试5: 更新用户信息"""
        print("\n" + "="*60)
        print("测试5: 更新用户信息")
        print("="*60)
        
        # 先创建用户
        user = self.db_service.create_user(
            username="test_user_5",
            email="test5@example.com",
            password_hash="hashed_password5"
        )
        user_id = user.id
        
        # 更新用户信息
        updated_user = self.db_service.update_user(
            user_id,
            email="updated@example.com"
        )
        
        # 验证用户更新成功
        assert updated_user is not None
        assert updated_user.id == user_id
        assert updated_user.email == "updated@example.com"
        
        print(f"✅ 用户更新成功: {updated_user.username}")
    
    def test_delete_user(self):
        """测试6: 删除用户"""
        print("\n" + "="*60)
        print("测试6: 删除用户")
        print("="*60)
        
        # 先创建用户
        user = self.db_service.create_user(
            username="test_user_6",
            email="test6@example.com",
            password_hash="hashed_password6"
        )
        user_id = user.id
        
        # 删除用户
        result = self.db_service.delete_user(user_id)
        
        # 验证用户删除成功
        assert result, "删除用户失败"
        
        # 验证用户已不存在
        deleted_user = self.db_service.get_user_by_id(user_id)
        assert deleted_user is None, "用户应该已被删除"
        
        print("✅ 用户删除成功")
    
    # =========================================================================
    # 检测记录测试
    # =========================================================================
    
    def test_save_detection(self):
        """测试7: 保存检测记录"""
        print("\n" + "="*60)
        print("测试7: 保存检测记录")
        print("="*60)
        
        # 先创建用户
        user = self.db_service.create_user(
            username="test_user_7",
            email="test7@example.com",
            password_hash="hashed_password7"
        )
        user_id = user.id
        
        # 模拟检测结果
        detection_result = [
            {"class_name": "person", "confidence": 0.85},
            {"class_name": "car", "confidence": 0.72}
        ]
        
        # 保存记录
        detection = self.db_service.save_detection(
            user_id=user_id,
            image_path="test/image.jpg",
            detection_result=detection_result,
            detection_type="blind_road"
        )
        
        assert detection is not None, "保存检测记录失败"
        assert detection.user_id == user_id
        assert detection.image_path == "test/image.jpg"
        assert detection.detection_type == "blind_road"
        
        print("✅ 检测记录保存成功")
    
    def test_get_user_detections(self):
        """测试8: 获取检测历史"""
        print("\n" + "="*60)
        print("测试8: 获取检测历史")
        print("="*60)
        
        # 创建用户
        user = self.db_service.create_user(
            username="test_user_8",
            email="test8@example.com",
            password_hash="hashed_password8"
        )
        user_id = user.id
        
        # 保存3条检测记录
        for i in range(3):
            detection_result = [
                {"class_name": f"class_{i}", "confidence": 0.8 + i * 0.05}
            ]
            self.db_service.save_detection(
                user_id=user_id,
                image_path=f"test/image_{i}.jpg",
                detection_result=detection_result,
                detection_type="blind_road"
            )
            time.sleep(0.01)  # 确保时间戳不同
        
        # 获取历史记录
        detections = self.db_service.get_user_detections(user_id, limit=10)
        
        assert len(detections) == 3, f"应该有3条记录，实际{len(detections)}条"
        print(f"✅ 获取到{len(detections)}条历史记录")
    
    def test_get_detection_statistics(self):
        """测试9: 获取检测统计信息"""
        print("\n" + "="*60)
        print("测试9: 获取检测统计信息")
        print("="*60)
        
        # 先创建用户
        user = self.db_service.create_user(
            username="test_user_9",
            email="test9@example.com",
            password_hash="hashed_password9"
        )
        user_id = user.id
        
        # 保存几条检测记录
        for i in range(2):
            detection_result = [
                {"class_name": f"class_{i}", "confidence": 0.8 + i * 0.1}
            ]
            
            self.db_service.save_detection(
                user_id=user_id,
                image_path=f"test/image_{i}.jpg",
                detection_result=detection_result,
                detection_type="blind_road"
            )
        
        # 获取检测统计信息
        stats = self.db_service.get_detection_statistics(user_id)
        
        # 验证统计信息
        assert isinstance(stats, dict)
        assert stats['user_id'] == user_id
        assert stats['total_detections'] == 2
        assert stats['total_targets'] == 2
        assert stats['type_counts']['blind_road'] == 2
        
        print("✅ 检测统计信息获取成功")
    
    # =========================================================================
    # 综合测试
    # =========================================================================
    
    def test_full_workflow(self):
        """测试10: 完整工作流"""
        print("\n" + "="*60)
        print("测试10: 完整工作流模拟")
        print("="*60)
        
        # 1. 创建用户
        user = self.db_service.create_user(
            username="workflow_user",
            email="workflow@example.com",
            password_hash="hashed_password"
        )
        assert user is not None
        print(f"   1. 用户创建: {user.username}")
        
        user_id = user.id
        
        # 2. 保存检测记录
        for i in range(3):
            detection_result = [
                {"class_name": "person"} if i == 1 else []
            ]
            
            self.db_service.save_detection(
                user_id=user_id,
                image_path=f"test/image_{i}.jpg",
                detection_result=detection_result,
                detection_type="blind_road"
            )
            time.sleep(0.01)
        
        print(f"   2. 保存了3次检测记录")
        
        # 3. 查询历史
        history = self.db_service.get_user_detections(user_id)
        print(f"   3. 查询历史: {len(history)}条检测记录")
        
        # 4. 获取统计信息
        stats = self.db_service.get_detection_statistics(user_id)
        print(f"   4. 统计信息: {stats['total_detections']}次检测")
        
        print("\n✅ 完整工作流测试通过")


if __name__ == "__main__":
    # 运行测试
    print("="*60)
    print("数据库服务单元测试")
    print("="*60)
    
    # 使用pytest运行
    pytest.main([__file__, "-v", "-s", "--tb=short"])

