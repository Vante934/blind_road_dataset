#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据验证模型
用于数据验证和序列化
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime

# 用户相关模型
class UserBase(BaseModel):
    """用户基础模型"""
    username: str = Field(..., min_length=3, max_length=50, description="用户名")
    email: str = Field(..., description="邮箱")

class UserCreate(UserBase):
    """用户创建模型"""
    password: str = Field(..., min_length=6, description="密码")

class UserUpdate(BaseModel):
    """用户更新模型"""
    username: Optional[str] = Field(None, min_length=3, max_length=50, description="用户名")
    email: Optional[str] = Field(None, description="邮箱")
    password: Optional[str] = Field(None, min_length=6, description="密码")
    is_active: Optional[bool] = Field(None, description="是否激活")

class UserResponse(UserBase):
    """用户响应模型"""
    id: int = Field(..., description="用户ID")
    is_active: bool = Field(..., description="是否激活")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    
    class Config:
        orm_mode = True

# 检测相关模型
class DetectionBase(BaseModel):
    """检测基础模型"""
    user_id: int = Field(..., description="用户ID")
    image_path: str = Field(..., description="图像路径")
    detection_result: List[Dict[str, Any]] = Field(..., description="检测结果")
    detection_type: str = Field(..., description="检测类型")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="置信度")

class DetectionCreate(DetectionBase):
    """检测创建模型"""
    pass

class DetectionResponse(DetectionBase):
    """检测响应模型"""
    id: int = Field(..., description="检测记录ID")
    created_at: datetime = Field(..., description="创建时间")
    
    class Config:
        orm_mode = True

class DetectionStatistics(BaseModel):
    """检测统计模型"""
    total_detections: int = Field(..., description="总检测次数")
    total_targets: int = Field(..., description="总目标数")
    type_counts: Dict[str, int] = Field(..., description="检测类型分布")
    avg_confidence: float = Field(..., ge=0, le=1, description="平均置信度")
    user_id: int = Field(..., description="用户ID")

# 导航相关模型
class NavigationBase(BaseModel):
    """导航基础模型"""
    user_id: int = Field(..., description="用户ID")
    start_latitude: float = Field(..., ge=-90, le=90, description="起始纬度")
    start_longitude: float = Field(..., ge=-180, le=180, description="起始经度")
    end_latitude: float = Field(..., ge=-90, le=90, description="结束纬度")
    end_longitude: float = Field(..., ge=-180, le=180, description="结束经度")
    route_data: Dict[str, Any] = Field(..., description="路径数据")
    distance: Optional[float] = Field(None, ge=0, description="距离（米）")
    duration: Optional[int] = Field(None, ge=0, description="持续时间（秒）")
    navigation_mode: str = Field(default="walking", description="导航模式")
    status: str = Field(default="completed", description="状态")

class NavigationCreate(NavigationBase):
    """导航创建模型"""
    pass

class NavigationResponse(NavigationBase):
    """导航响应模型"""
    id: int = Field(..., description="导航记录ID")
    created_at: datetime = Field(..., description="创建时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    
    class Config:
        orm_mode = True

# 检测结果模型
class DetectionResult(BaseModel):
    """检测结果模型"""
    bbox: List[int] = Field(..., description="检测框坐标 [x1, y1, x2, y2]")
    class_id: int = Field(..., description="类别ID")
    class_name: str = Field(..., description="类别名称")
    confidence: float = Field(..., ge=0, le=1, description="置信度")
    size: Dict[str, int] = Field(..., description="目标大小 {width, height}")

# 错误响应模型
class ErrorResponse(BaseModel):
    """错误响应模型"""
    error: str = Field(..., description="错误信息")
    code: int = Field(..., description="错误代码")

# 成功响应模型
class SuccessResponse(BaseModel):
    """成功响应模型"""
    message: str = Field(..., description="成功信息")
    data: Optional[Any] = Field(None, description="附加数据")

# 登录相关模型
class LoginRequest(BaseModel):
    """登录请求模型"""
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")

class LoginResponse(BaseModel):
    """登录响应模型"""
    access_token: str = Field(..., description="访问令牌")
    token_type: str = Field(default="bearer", description="令牌类型")
    user: UserResponse = Field(..., description="用户信息")

# 配置相关模型
class ConfigModel(BaseModel):
    """配置模型"""
    model_path: str = Field(..., description="模型路径")
    conf_threshold: float = Field(default=0.25, ge=0, le=1, description="置信度阈值")
    image_size: int = Field(default=640, description="图像大小")
    max_detections: int = Field(default=100, ge=1, description="最大检测数量")
    use_gpu: bool = Field(default=True, description="是否使用GPU")
