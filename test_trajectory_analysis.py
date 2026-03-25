# tests/test_trajectory_analysis.py
"""
测试轨迹分析模块
"""
import asyncio
import websockets
import json
import time

async def test_trajectory_analysis():
    """
    测试轨迹分析功能
    
    .env 配置:
    MODULE_TRAJECTORY_ENABLED=true
    """
    uri = "ws://localhost:8000/ws/test_trajectory"
    
    async with websockets.connect(uri) as ws:
        # 1. 连接确认
        response = json.loads(await ws.recv())
        print("✅ 连接确认:", response["data"]["message"])
        print("✅ 服务端能力:", response["data"]["capabilities"])
        assert response["type"] == "connected"
        
        # 2. 发送连续距离数据（模拟物体接近）
        distances = [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5]
        
        for d in distances:
            await ws.send(json.dumps({
                "type": "sensor_data",
                "data": {
                    "device_id": "test_trajectory",
                    "timestamp": time.time() * 1000,
                    "tof_distance": d,
                    "tof_direction": "rear"
                }
            }))
            
            response = json.loads(await ws.recv())
            
            if response["type"] == "warning":
                print(f"距离{d}m → 预警: {response['data']['tts_text']}")
                print(f"   轨迹数据: {response['data'].get('obstacles_info', '无')}")
            else:
                print(f"距离{d}m → 状态: {response['data']['status']}")
            
            await asyncio.sleep(0.5)
        
        print("✅ 轨迹分析测试完成！")

if __name__ == "__main__":
    asyncio.run(test_trajectory_analysis())