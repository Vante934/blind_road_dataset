import asyncio
import json
import websockets
import time

async def test_trajectory_analysis():
    uri = "ws://localhost:8000/ws/test_device"
    
    async with websockets.connect(uri) as websocket:
        # 接收连接响应
        response = await websocket.recv()
        print("Connected:", response)
        
        # 模拟物体接近的场景
        distances = [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5]
        
        for distance in distances:
            # 发送距离数据
            message = {
                "type": "sensor_data",
                "data": {
                    "distance": distance,
                    "angle": 0.0,
                    "sensor_id": "ultrasonic_front",
                    "timestamp": time.time() * 1000
                }
            }
            await websocket.send(json.dumps(message))
            print(f"Sent: distance={distance}m")
            
            # 接收系统响应
            response = await websocket.recv()
            print("Received:", response)
            
            # 等待一段时间模拟真实场景
            time.sleep(0.5)

if __name__ == "__main__":
    print("Testing trajectory analysis functionality...")
    print("Make sure the server is running on ws://localhost:8000/ws")
    print("Press Ctrl+C to stop the test")
    
    try:
        asyncio.run(test_trajectory_analysis())
    except KeyboardInterrupt:
        print("Test stopped by user")
    except Exception as e:
        print(f"Error: {e}")
