#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
前后端连接测试脚本
用于测试前后端API连接是否正常
"""

import requests
import json
import time
from datetime import datetime

# API配置
API_BASE_URL = "http://localhost:8000"
API_V1_URL = f"{API_BASE_URL}/api/v1"

def test_connection():
    """测试基础连接"""
    print("=" * 60)
    print("测试1: 基础连接测试")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=5)
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.json()}")
        
        if response.status_code == 200:
            print("✅ 基础连接成功")
            return True
        else:
            print("❌ 基础连接失败")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器，请确保后端服务已启动")
        return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False

def test_system_status():
    """测试系统状态接口"""
    print("\n" + "=" * 60)
    print("测试2: 系统状态接口")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_V1_URL}/system/status", timeout=5)
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        
        if response.status_code == 200:
            print("✅ 系统状态接口正常")
            return True
        else:
            print("❌ 系统状态接口异常")
            return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False

def test_detection_api():
    """测试检测接口"""
    print("\n" + "=" * 60)
    print("测试3: 检测接口（需要图像数据）")
    print("=" * 60)
    
    # 这里使用一个简单的测试图像（Base64编码的最小PNG）
    # 实际使用时应该使用真实的图像
    test_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    try:
        payload = {
            "image_data": test_image_base64,
            "image_format": "png"
        }
        
        response = requests.post(
            f"{API_V1_URL}/detection/analyze",
            json=payload,
            timeout=10
        )
        
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        
        if response.status_code == 200:
            print("✅ 检测接口正常")
            return True
        else:
            print("❌ 检测接口异常")
            return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False

def test_cors():
    """测试CORS配置"""
    print("\n" + "=" * 60)
    print("测试4: CORS配置测试")
    print("=" * 60)
    
    try:
        # 模拟前端请求
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Content-Type"
        }
        
        response = requests.options(f"{API_V1_URL}/system/status", headers=headers, timeout=5)
        
        print(f"状态码: {response.status_code}")
        print(f"CORS头信息:")
        for key, value in response.headers.items():
            if 'access-control' in key.lower():
                print(f"  {key}: {value}")
        
        if 'access-control-allow-origin' in response.headers:
            print("✅ CORS配置正常")
            return True
        else:
            print("⚠️ CORS配置可能有问题")
            return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False

def test_websocket():
    """测试WebSocket连接（需要websocket库）"""
    print("\n" + "=" * 60)
    print("测试5: WebSocket连接测试")
    print("=" * 60)
    
    try:
        import websocket
        import threading
        
        ws_url = API_BASE_URL.replace("http", "ws") + "/ws"
        print(f"WebSocket URL: {ws_url}")
        
        received_messages = []
        
        def on_message(ws, message):
            print(f"收到消息: {message}")
            received_messages.append(message)
        
        def on_error(ws, error):
            print(f"WebSocket错误: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            print("WebSocket连接关闭")
        
        def on_open(ws):
            print("✅ WebSocket连接成功")
            # 发送测试消息
            test_message = {
                "type": "test",
                "message": "测试消息"
            }
            ws.send(json.dumps(test_message))
        
        ws = websocket.WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        # 在单独线程中运行
        wst = threading.Thread(target=ws.run_forever)
        wst.daemon = True
        wst.start()
        
        # 等待3秒
        time.sleep(3)
        
        ws.close()
        
        if len(received_messages) > 0:
            print("✅ WebSocket通信正常")
            return True
        else:
            print("⚠️ WebSocket可能有问题")
            return False
            
    except ImportError:
        print("⚠️ websocket-client库未安装，跳过WebSocket测试")
        print("   安装命令: pip install websocket-client")
        return None
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False

def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("前后端连接测试")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    results = []
    
    # 执行测试
    results.append(("基础连接", test_connection()))
    results.append(("系统状态", test_system_status()))
    results.append(("CORS配置", test_cors()))
    results.append(("检测接口", test_detection_api()))
    ws_result = test_websocket()
    if ws_result is not None:
        results.append(("WebSocket", ws_result))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {passed}个通过, {failed}个失败")
    
    if failed == 0:
        print("\n🎉 所有测试通过！前后端连接正常")
    else:
        print(f"\n⚠️ 有{failed}个测试失败，请检查相关配置")

if __name__ == "__main__":
    main()
