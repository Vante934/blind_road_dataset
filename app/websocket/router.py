"""
WebSocket 路由

🔄 修改策略：
- 保持原有消息格式和处理逻辑（✅ 已联调通过）
- 增加新的调用点，调用 sensing_pipeline
- 通过配置开关控制是否使用新流程
"""
import json
import logging
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import settings
from app.websocket.connection_manager import manager
from app.services.pipeline import init_pipeline, process_sensor_data
from app.models.schemas import SensorData, AudioData
from app.core.warning_engine import WarningEngine  # 原有引擎
from app.models.schemas import SensorData

logger = logging.getLogger(__name__)
router = APIRouter()

# ===== 原有组件（保持不变）=====
warning_engine = WarningEngine()


@router.websocket("/ws/{device_id}")
async def websocket_endpoint(websocket: WebSocket, device_id: str):
    """
    WebSocket 主端点
    
    🔄 兼容性保证：
    - 配置全关 → 完全走旧流程（✅ 保证联调成功的功能）
    - 配置开启 → 走新流程（模块化处理）
    """
    
    await manager.connect(websocket, device_id)
    
    # 连接确认（🆕 增加服务端能力声明）
    await websocket.send_json({
        "type": "connected",
        "data": {
            "message": f"设备 {device_id} 连接成功",
            "server_time": time.time() * 1000,
            # 🆕 告知前端支持的功能
            "capabilities": {
                "vision": settings.MODULE_VISION_ENABLED,
                "trajectory": settings.MODULE_TRAJECTORY_ENABLED,
                "audio": True,  # 始终支持
            }
        }
    })
    
    try:
        while True:
            raw_message = await websocket.receive_text()
            message = json.loads(raw_message)
            
            msg_type = message.get("type", "")
            msg_data = message.get("data", {})
            
            if msg_type == "heartbeat":
                await websocket.send_json({
                    "type": "heartbeat_ack",
                    "data": {"server_time": time.time() * 1000}
                })
                
            elif msg_type == "sensor_data":
                # 🔄 核心处理点
                await _process_sensor_data_v2(websocket, device_id, msg_data)
                
            else:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": f"未知消息类型: {msg_type}"}
                })
    
    except WebSocketDisconnect:
        logger.info(f"设备 {device_id} 断开连接")
    except Exception as e:
        logger.error(f"WebSocket异常: {device_id}, error={e}")
    finally:
        manager.disconnect(device_id)


async def _process_sensor_data_v2(websocket: WebSocket, device_id: str, data: dict):
    """
    🔄 传感器数据处理 V2
    
    策略：
    - 检测是否有新字段（video_frame）且功能开启 → 走新流程
    - 否则 → 走旧流程（保证兼容）
    """
    try:
        # ===== 检测是否使用新流程 =====
        use_new_pipeline = (
            settings.MODULE_VISION_ENABLED or 
            settings.MODULE_TRAJECTORY_ENABLED
        )
        
        if use_new_pipeline:
            # 🆕 新流程：模块化处理
            await _process_with_pipeline(websocket, device_id, data)
        else:
            # ✅ 旧流程：保持原有逻辑（已联调通过）
            await _process_legacy(websocket, device_id, data)
    
    except Exception as e:
        logger.error(f"处理异常: {e}", exc_info=True)
        await websocket.send_json({
            "type": "error",
            "data": {"message": str(e)}
        })


async def _process_with_pipeline(websocket: WebSocket, device_id: str, data: dict):
    """
    🆕 新流程：使用感知流程编排器
    """
    # Step 1: 构造传感器数据
    audio_data = None
    if data.get("audio_data"):
        audio_data = AudioData(
            audio_base64=data.get("audio_data").get("audio_base64"),
            audio_format=data.get("audio_data").get("audio_format", "pcm"),
            sample_rate=data.get("audio_data").get("sample_rate", 16000)
        )
    
    sensor_data = SensorData(
        device_id=device_id,
        timestamp=data.get("timestamp", time.time() * 1000),
        tof_distance=data.get("tof_distance") or data.get("distance"),
        tof_direction=data.get("tof_direction", "rear"),
        audio_data=audio_data,
        video_frame=data.get("video_frame")
    )
    
    # Step 2: 执行感知流程
    pipeline_response = await process_sensor_data(sensor_data)
    
    # Step 3: 处理响应
    if pipeline_response.success:
        data = pipeline_response.data
        
        # Step 4: 构建响应
        if data.get("warning"):
            warning = data.get("warning")
            response = {
                "type": "warning",
                "data": {
                    # ✅ 旧版必须字段
                    "warning_level": warning.get("warning_level"),
                    "warning_level_name": warning.get("warning_level_name"),
                    "tts_text": warning.get("tts_text"),
                    "vibration_intensity": warning.get("vibration_intensity"),
                    "vibration_pattern": warning.get("vibration_pattern"),
                    "distance": data.get("warning").get("primary_threat", {}).get("data", {}).get("distance"),
                    "direction": data.get("warning").get("primary_threat", {}).get("data", {}).get("direction"),
                    "timestamp": time.time() * 1000,
                    
                    # 🆕 新增字段（旧前端忽略）
                    "obstacles_info": data.get("obstacles", [])[:3],  # 前3个
                    "sound_info": data.get("sound"),
                    "performance": {
                        "total_time": round(data.get("total_time", 0) * 1000, 1),
                        "module_times": {k: round(v*1000, 1) for k, v in data.get("processing_time", {}).items()}
                    }
                }
            }
            
            await websocket.send_json(response)
        else:
            await websocket.send_json({
                "type": "status",
                "data": {
                    "status": "safe",
                    "obstacles_count": len(data.get("obstacles", [])),
                    "timestamp": time.time() * 1000
                }
            })
    else:
        await websocket.send_json({
            "type": "error",
            "data": {"message": pipeline_response.message}
        })


async def _process_legacy(websocket: WebSocket, device_id: str, data: dict):
    """
    ✅ 旧流程：完全保持原有逻辑（已联调通过的代码）
    
    这里放你们之前联调通过的代码，一个字都不改
    """
    # 🔄 这里复制你们之前的 _process_sensor_data 函数的内容
    # 保持100%不变
    
    sensor_data = SensorData(**{**data, "device_id": device_id})
    
    # ... 你们原有的音频识别逻辑 ...
    # ... 你们原有的预警决策逻辑 ...
    # ... 你们原有的下发逻辑 ...
    
    # 示例（替换成你们的实际代码）:
    warning_command = await warning_engine.process_legacy(sensor_data)
    
    if warning_command:
        await websocket.send_json({
            "type": "warning",
            "data": warning_command.model_dump()
        })
    else:
        await websocket.send_json({
            "type": "status",
            "data": {
                "status": "safe",
                "timestamp": time.time() * 1000
            }
        })