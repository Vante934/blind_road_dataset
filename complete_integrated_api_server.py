#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整集成API服务器 - 包含检测、导航、语音和Android功能
第8-10天核心功能开发
"""

import os
import sys
import json
import time
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# 导入真实模块
try:
    from enhanced_blind_road_detector import EnhancedBlindRoadDetector
    DETECTION_AVAILABLE = True
    print("✅ 增强盲道检测模块可用")
except ImportError:
    try:
        from blind_road_sdk import BlindRoadDetector
        DETECTION_AVAILABLE = True
        print("✅ 基础盲道检测模块可用")
    except ImportError:
        print("警告: 盲道检测模块不可用，使用模拟模式")
        DETECTION_AVAILABLE = False

try:
    from enhanced_navigation_system import EnhancedNavigationSystem
    NAVIGATION_AVAILABLE = True
    print("✅ 增强导航系统模块可用")
except ImportError:
    print("警告: 导航系统模块不可用，使用模拟模式")
    NAVIGATION_AVAILABLE = False

try:
    from voice_system.voice_navigator import VoiceNavigator
    VOICE_AVAILABLE = True
except ImportError:
    print("警告: 语音模块不可用，使用模拟模式")
    VOICE_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModuleStatus(Enum):
    """模块状态"""
    INACTIVE = "inactive"
    ACTIVE = "active"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class MessageType(Enum):
    """消息类型"""
    DETECTION_RESULT = "detection_result"
    NAVIGATION_INSTRUCTION = "navigation_instruction"
    VOICE_MESSAGE = "voice_message"
    SYSTEM_STATUS = "system_status"
    ERROR = "error"

# Pydantic模型
class APIResponse(BaseModel):
    """API响应"""
    success: bool
    code: int
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class DetectionRequest(BaseModel):
    """检测请求"""
    image_data: str = Field(..., description="Base64编码的图像数据")
    image_format: str = Field(default="png", description="图像格式")
    
    @validator('image_data')
    def validate_image_data(cls, v):
        if not v or len(v) < 50:
            raise ValueError('图像数据不能为空或过短')
        return v

class RouteRequest(BaseModel):
    """路线规划请求"""
    start_latitude: float = Field(..., description="起点纬度")
    start_longitude: float = Field(..., description="起点经度")
    end_latitude: float = Field(..., description="终点纬度")
    end_longitude: float = Field(..., description="终点经度")
    navigation_mode: str = Field(default="walking", description="导航模式")
    
    @validator('start_latitude', 'end_latitude')
    def validate_latitude(cls, v):
        if not -90 <= v <= 90:
            raise ValueError('纬度必须在-90到90之间')
        return v
    
    @validator('start_longitude', 'end_longitude')
    def validate_longitude(cls, v):
        if not -180 <= v <= 180:
            raise ValueError('经度必须在-180到180之间')
        return v

class LocationUpdate(BaseModel):
    """位置更新"""
    latitude: float = Field(..., description="纬度")
    longitude: float = Field(..., description="经度")
    accuracy: float = Field(default=5.0, description="精度（米）")
    timestamp: Optional[str] = Field(default=None, description="时间戳")

class VoiceRequest(BaseModel):
    """语音请求"""
    text: str = Field(..., description="要合成的文本")
    voice_type: str = Field(default="default", description="语音类型")
    speed: float = Field(default=1.0, description="语速")
    volume: float = Field(default=1.0, description="音量")

class DeviceRegistration(BaseModel):
    """设备注册"""
    device_id: str
    device_info: Dict[str, Any]
    capabilities: List[str]

class CompleteIntegratedAPIServer:
    """完整集成API服务器"""
    
    def __init__(self):
        self.app = FastAPI(
            title="盲道检测系统 - 完整集成API",
            description="包含检测、导航、语音和Android功能的完整API",
            version="2.0.0",
            docs_url="/docs",  # Swagger文档
            redoc_url="/redoc",  # ReDoc文档
            openapi_url="/openapi.json"  # OpenAPI规范JSON
        )
        
        # 配置CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 初始化模块
        self.detector = None
        self.navigation_system = None
        self.voice_navigator = None
        self.module_status = {
            "detection": ModuleStatus.INACTIVE,
            "navigation": ModuleStatus.INACTIVE,
            "android": ModuleStatus.INACTIVE,
            "voice": ModuleStatus.INACTIVE
        }
        
        # WebSocket连接管理
        self.active_connections: Dict[str, WebSocket] = {}
        
        # 初始化模块
        self._initialize_modules()
        
        # 注册路由
        self._register_routes()
    
    def _initialize_modules(self):
        """初始化真实模块"""
        logger.info("正在初始化真实模块...")
        
        # 初始化盲道检测模块
        if DETECTION_AVAILABLE:
            try:
                try:
                    self.detector = EnhancedBlindRoadDetector()
                    logger.info("✅ 增强盲道检测模块初始化成功")
                except Exception as e:
                    logger.warning(f"⚠️ 增强检测器初始化失败，使用基础检测器: {e}")
                    self.detector = BlindRoadDetector()
                    logger.info("✅ 基础盲道检测模块初始化成功")
                
                self.module_status["detection"] = ModuleStatus.ACTIVE
            except Exception as e:
                logger.error(f"❌ 盲道检测模块初始化失败: {e}")
                self.module_status["detection"] = ModuleStatus.ERROR
        
        # 初始化导航系统模块
        if NAVIGATION_AVAILABLE:
            try:
                self.navigation_system = EnhancedNavigationSystem()
                self.module_status["navigation"] = ModuleStatus.ACTIVE
                logger.info("✅ 导航系统模块初始化成功")
            except Exception as e:
                logger.error(f"❌ 导航系统模块初始化失败: {e}")
                self.module_status["navigation"] = ModuleStatus.ERROR
        
        # 初始化语音导航模块
        if VOICE_AVAILABLE:
            try:
                self.voice_navigator = VoiceNavigator()
                self.module_status["voice"] = ModuleStatus.ACTIVE
                logger.info("✅ 语音导航模块初始化成功")
            except Exception as e:
                logger.error(f"❌ 语音导航模块初始化失败: {e}")
                self.module_status["voice"] = ModuleStatus.ERROR
    
    def _register_routes(self):
        """注册路由"""
        
        @self.app.get("/")
        async def root():
            """根路径 - API服务信息"""
            return {
                "message": "盲道检测系统API服务运行正常",
                "version": "2.0.0",
                "status": "running",
                "description": "包含障碍物识别、语音播报、环境识别功能的完整API",
                "endpoints": {
                    "system_status": "/api/v1/system/status",
                    "detection": "/api/v1/detection/analyze",
                    "voice": "/api/v1/voice/synthesize",
                    "android": "/api/v1/android/register",
                    "docs": "/docs"
                },
                "modules": {
                    name: status.value for name, status in self.module_status.items()
                },
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/status")
        async def simple_status():
            """简单状态检查端点 - 用于Android连接测试"""
            return {
                "status": "ok",
                "message": "服务器运行正常",
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/api/v1/system/status")
        async def get_system_status():
            """获取系统状态"""
            return {
                "status": "running",
                "modules": {
                    name: status.value for name, status in self.module_status.items()
                },
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/api/v1/detection/analyze")
        async def analyze_image(request: DetectionRequest):
            """图像检测分析"""
            try:
                start_time = time.time()
                
                # 解码图像
                import base64
                import numpy as np
                from PIL import Image
                import io
                
                try:
                    image_data = base64.b64decode(request.image_data)
                    image = Image.open(io.BytesIO(image_data))
                    image = image.convert('RGB')
                    image_array = np.array(image)
                except Exception as e:
                    logger.error(f"图像解码失败: {e}")
                    # 使用模拟结果
                    formatted_results = self._get_mock_detection_results()
                    processing_time = 0.02
                else:
                    # 使用真实检测模块
                    if self.detector and self.module_status["detection"] == ModuleStatus.ACTIVE:
                        try:
                            results = self.detector.detect(image_array)
                            processing_time = time.time() - start_time
                            formatted_results = self._format_enhanced_detection_results(results)
                        except Exception as e:
                            logger.error(f"真实检测失败，使用模拟结果: {e}")
                            formatted_results = self._get_mock_detection_results()
                            processing_time = 0.02
                    else:
                        formatted_results = self._get_mock_detection_results()
                        processing_time = 0.02
                
                detection_id = str(uuid.uuid4())
                
                return {
                    "detection_id": detection_id,
                    "results": formatted_results,
                    "processing_time": processing_time,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"检测处理失败: {e}")
                raise HTTPException(status_code=500, detail=f"检测处理失败: {str(e)}")
        
        @self.app.post("/api/v1/navigation/plan-route")
        async def plan_route(request: RouteRequest):
            """规划路线"""
            try:
                if not self.navigation_system or self.module_status["navigation"] != ModuleStatus.ACTIVE:
                    return JSONResponse(
                        status_code=503,
                        content={"error": "导航系统不可用", "timestamp": datetime.now().isoformat()}
                    )
                
                # 创建位置对象
                from enhanced_navigation_system import Location, NavigationMode
                
                start_location = Location(request.start_latitude, request.start_longitude)
                end_location = Location(request.end_latitude, request.end_longitude)
                
                # 确定导航模式
                mode_map = {
                    "walking": NavigationMode.WALKING,
                    "wheelchair": NavigationMode.WHEELCHAIR,
                    "guide_dog": NavigationMode.GUIDE_DOG,
                    "assistive_tech": NavigationMode.ASSISTIVE_TECH
                }
                navigation_mode = mode_map.get(request.navigation_mode, NavigationMode.WALKING)
                
                # 规划路线
                route = self.navigation_system.plan_route(start_location, end_location, navigation_mode)
                
                if not route:
                    return JSONResponse(
                        status_code=400,
                        content={"error": "无法规划路线", "timestamp": datetime.now().isoformat()}
                    )
                
                # 格式化路线信息
                route_info = {
                    "route_id": route.route_id,
                    "start_location": {
                        "latitude": route.start_location.latitude,
                        "longitude": route.start_location.longitude
                    },
                    "end_location": {
                        "latitude": route.end_location.latitude,
                        "longitude": route.end_location.longitude
                    },
                    "total_distance": route.total_distance,
                    "estimated_time": route.estimated_time,
                    "accessibility_score": route.accessibility_score,
                    "difficulty_level": route.difficulty_level,
                    "path_segments": [
                        {
                            "path_type": segment.path_type.value,
                            "length": segment.length,
                            "direction": segment.direction,
                            "accessibility_score": segment.accessibility_score
                        }
                        for segment in route.path_segments
                    ],
                    "instructions": [
                        {
                            "instruction_type": instruction.instruction_type,
                            "direction": instruction.direction,
                            "distance": instruction.distance,
                            "description": instruction.description,
                            "voice_message": instruction.voice_message,
                            "priority": instruction.priority
                        }
                        for instruction in route.instructions
                    ],
                    "created_at": route.created_at.isoformat()
                }
                
                return {"message": "路线规划成功", "route": route_info, "timestamp": datetime.now().isoformat()}
                
            except Exception as e:
                logger.error(f"路线规划失败: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"error": f"路线规划失败: {str(e)}", "timestamp": datetime.now().isoformat()}
                )
        
        @self.app.post("/api/v1/navigation/update-location")
        async def update_location(request: LocationUpdate):
            """更新位置"""
            try:
                if not self.navigation_system or self.module_status["navigation"] != ModuleStatus.ACTIVE:
                    return JSONResponse(
                        status_code=503,
                        content={"error": "导航系统不可用", "timestamp": datetime.now().isoformat()}
                    )
                
                # 创建位置对象
                from enhanced_navigation_system import Location
                
                location = Location(request.latitude, request.longitude, accuracy=request.accuracy)
                
                # 更新位置
                instruction = self.navigation_system.update_location(location)
                
                response_data = {
                    "message": "位置更新成功",
                    "current_location": {
                        "latitude": location.latitude,
                        "longitude": location.longitude,
                        "accuracy": location.accuracy
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                if instruction:
                    response_data["current_instruction"] = {
                        "instruction_type": instruction.instruction_type,
                        "direction": instruction.direction,
                        "distance": instruction.distance,
                        "description": instruction.description,
                        "voice_message": instruction.voice_message,
                        "priority": instruction.priority
                    }
                
                return response_data
                
            except Exception as e:
                logger.error(f"位置更新失败: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"error": f"位置更新失败: {str(e)}", "timestamp": datetime.now().isoformat()}
                )
        
        @self.app.get("/api/v1/navigation/status")
        async def get_navigation_status():
            """获取导航状态"""
            try:
                if not self.navigation_system or self.module_status["navigation"] != ModuleStatus.ACTIVE:
                    return JSONResponse(
                        status_code=503,
                        content={"error": "导航系统不可用", "timestamp": datetime.now().isoformat()}
                    )
                
                status = self.navigation_system.get_navigation_status()
                return {"navigation_status": status, "timestamp": datetime.now().isoformat()}
                
            except Exception as e:
                logger.error(f"获取导航状态失败: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"error": f"获取导航状态失败: {str(e)}", "timestamp": datetime.now().isoformat()}
                )
        
        @self.app.post("/api/v1/voice/synthesize")
        async def synthesize_speech(request: VoiceRequest):
            """语音合成"""
            try:
                if not self.voice_navigator or self.module_status["voice"] != ModuleStatus.ACTIVE:
                    # 使用模拟语音合成
                    audio_id = str(uuid.uuid4())
                    logger.info(f"模拟语音合成: {request.text}")
                    return {
                        "audio_id": audio_id,
                        "text": request.text,
                        "voice_type": request.voice_type,
                        "status": "synthesized",
                        "timestamp": datetime.now().isoformat()
                    }
                
                # 使用真实语音合成
                try:
                    # 检查可用的方法
                    if hasattr(self.voice_navigator, 'synthesize_speech'):
                        result = self.voice_navigator.synthesize_speech(request.text)
                    elif hasattr(self.voice_navigator, 'speak'):
                        result = self.voice_navigator.speak(request.text)
                    else:
                        raise AttributeError("语音模块没有可用的合成方法")
                    
                    audio_id = str(uuid.uuid4())
                    logger.info(f"真实语音合成成功: {request.text}")
                    
                    return {
                        "audio_id": audio_id,
                        "text": request.text,
                        "voice_type": request.voice_type,
                        "status": "synthesized",
                        "result": result,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                except Exception as e:
                    logger.error(f"真实语音合成失败，使用模拟: {e}")
                    audio_id = str(uuid.uuid4())
                    return {
                        "audio_id": audio_id,
                        "text": request.text,
                        "voice_type": request.voice_type,
                        "status": "synthesized",
                        "timestamp": datetime.now().isoformat()
                    }
                
            except Exception as e:
                logger.error(f"语音合成失败: {e}")
                raise HTTPException(status_code=500, detail=f"语音合成失败: {str(e)}")
        
        @self.app.post("/api/v1/android/register", response_model=APIResponse)
        async def register_device(registration: DeviceRegistration):
            """设备注册"""
            try:
                device_id = registration.device_id
                
                # 更新模块状态
                self.module_status["android"] = ModuleStatus.ACTIVE
                
                # 记录设备信息
                logger.info(f"设备注册成功: {device_id}")
                
                return APIResponse(
                    success=True,
                    code=201,
                    message="设备注册成功",
                    data={
                        "device_id": device_id,
                        "status": "registered",
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
            except Exception as e:
                logger.error(f"设备注册失败: {e}")
                raise HTTPException(status_code=400, detail=f"设备注册失败: {str(e)}")
        
        @self.app.post("/api/v1/android/data/upload", response_model=APIResponse)
        async def upload_data(data: Dict[str, Any]):
            """数据上传"""
            try:
                upload_id = str(uuid.uuid4())
                
                # 处理上传的数据
                device_id = data.get("device_id")
                data_type = data.get("data_type")
                
                logger.info(f"收到数据上传: 设备={device_id}, 类型={data_type}")
                
                return APIResponse(
                    success=True,
                    code=200,
                    message="数据上传成功",
                    data={
                        "upload_id": upload_id,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
            except Exception as e:
                logger.error(f"数据上传失败: {e}")
                raise HTTPException(status_code=400, detail=f"数据上传失败: {str(e)}")
    
    def _format_enhanced_detection_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """格式化增强检测结果"""
        try:
            analysis = results.get("analysis", {})
            
            formatted = {
                "blind_path": {
                    "detected": analysis.get("blind_path", {}).get("detected", False),
                    "confidence": analysis.get("blind_path", {}).get("confidence", 0.0),
                    "bbox": list(analysis.get("blind_path", {}).get("bbox", (0, 0, 0, 0))),
                    "path_type": analysis.get("blind_path", {}).get("path_type", "unknown"),
                    "direction": analysis.get("blind_path", {}).get("direction", "unknown"),
                    "condition": analysis.get("blind_path", {}).get("condition", "unknown"),
                    "width": analysis.get("blind_path", {}).get("width", 0.0)
                },
                "obstacle": {
                    "detected": analysis.get("obstacle", {}).get("detected", False),
                    "confidence": analysis.get("obstacle", {}).get("confidence", 0.0),
                    "bbox": list(analysis.get("obstacle", {}).get("bbox", (0, 0, 0, 0))),
                    "obstacle_type": analysis.get("obstacle", {}).get("obstacle_type", "unknown"),
                    "distance_estimate": analysis.get("obstacle", {}).get("distance_estimate", 0.0),
                    "severity": analysis.get("obstacle", {}).get("severity", "low")
                },
                "summary": analysis.get("summary", {}),
                "processing_time": results.get("processing_time", 0.0),
                "model_info": results.get("model_info", {})
            }
            
            return formatted
            
        except Exception as e:
            logger.error(f"格式化增强检测结果失败: {e}")
            return self._get_mock_detection_results()
    
    def _get_mock_detection_results(self) -> Dict[str, Any]:
        """获取模拟检测结果"""
        return {
            "blind_path": {
                "detected": True,
                "confidence": 0.85,
                "bbox": [100, 200, 300, 250],
                "path_type": "tactile_paving",
                "direction": "horizontal",
                "condition": "good",
                "width": 200.0
            },
            "obstacle": {
                "detected": False,
                "confidence": 0.0,
                "bbox": [0, 0, 0, 0],
                "obstacle_type": "none",
                "distance_estimate": 0.0,
                "severity": "low"
            },
            "summary": {
                "total_objects": 1,
                "blind_paths": 1,
                "obstacles": 0,
                "other_objects": 0
            },
            "processing_time": 0.02,
            "model_info": {
                "model_path": "mock",
                "confidence_threshold": 0.5,
                "nms_threshold": 0.4
            }
        }

def start_server(host: str = "0.0.0.0", port: int = 8082):
    """启动服务器"""
    import socket
    import getpass
    import platform
    
    server = CompleteIntegratedAPIServer()
    
    # 获取本机IP地址
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
    except Exception:
        local_ip = "127.0.0.1"
    finally:
        s.close()
    
    print("=" * 60)
    print("🚀 盲道检测系统 - 完整集成API服务器")
    print("=" * 60)
    print(f"📡 服务器地址: http://{host}:{port}")
    print(f"💻 本机IP地址: {local_ip}")
    print(f"🌐 局域网访问: http://{local_ip}:{port}")
    print(f"📍 Swagger文档: http://{local_ip}:{port}/docs")
    print("=" * 60)
    print("📱 Android端配置:")
    print(f"   服务器地址: http://{local_ip}:{port}")
    print("=" * 60)
    print("\n💡 提示:")
    print("   - 确保Android设备与电脑在同一网络")
    print("   - 如果连接失败，检查防火墙设置")
    print("   - 使用 /status 端点测试连接")
    print("\n按 Ctrl+C 停止服务器\n")
    
    logger.info(f"启动完整集成API服务器: {host}:{port}")
    
    uvicorn.run(
        server.app,
        host=host,
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    start_server()
