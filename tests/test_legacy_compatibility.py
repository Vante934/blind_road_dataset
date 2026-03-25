"""
验证旧功能完全正常
"""
import asyncio
import websockets
import json
import time


async def test_legacy_flow():
    """
    测试旧流程（所有新模块关闭）
    
    .env 配置:
    MODULE_VISION_ENABLED=false
    MODULE_TRAJECTORY_ENABLED=false
    """
    uri = "ws://localhost:8000/ws/test_legacy"
    
    async with websockets.connect(uri) as ws:
        # 1. 连接确认
        response = json.loads(await ws.recv())
        print("✅ 连接确认:", response["data"]["message"])
        assert response["type"] == "connected"
        
        # 2. 心跳
        await ws.send(json.dumps({"type": "heartbeat", "data": {}}))
        response = json.loads(await ws.recv())
        assert response["type"] == "heartbeat_ack"
        print("✅ 心跳正常")
        
        # 3. 发送ToF距离数据（旧格式）
        await ws.send(json.dumps({
            "type": "sensor_data",
            "data": {
                "device_id": "test_legacy",
                "timestamp": time.time() * 1000,
                "tof_distance": 0.3,  # 很近，应触发预警
                "tof_direction": "rear"
            }
        }))
        
        response = json.loads(await ws.recv())
        print("✅ 预警响应:", response)
        
        assert response["type"] == "warning"
        assert response["data"]["warning_level"] == 1
        assert "危险" in response["data"]["tts_text"]
        
        print("✅ 旧功能测试通过！")


if __name__ == "__main__":
    asyncio.run(test_legacy_flow())