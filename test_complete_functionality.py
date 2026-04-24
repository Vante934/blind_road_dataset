#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整功能测试脚本 - 第8-10天核心功能开发测试
"""

import requests
import json
import time
import base64
import numpy as np
from PIL import Image
import io

def create_test_image():
    """创建测试图像"""
    # 创建640x480的测试图像
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 添加背景
    image[:] = (50, 50, 50)  # 深灰色背景
    
    # 添加模拟盲道（水平）
    image[200:250, 50:590] = (128, 128, 128)  # 盲道
    image[260:270, 50:590] = (100, 100, 100)   # 盲道边缘
    
    # 添加模拟障碍物
    image[150:200, 400:450] = (0, 0, 255)  # 红色障碍物
    
    # 转换为PIL图像并编码
    pil_image = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return image_data

def test_system_status():
    """测试系统状态"""
    print("\n[测试] 系统状态接口...")
    try:
        response = requests.get("http://localhost:8082/api/v1/system/status")
        if response.status_code == 200:
            data = response.json()
            print(f"[成功] 系统状态: {data['status']}")
            print(f"[模块] 检测: {data['modules']['detection']}")
            print(f"[模块] 导航: {data['modules']['navigation']}")
            print(f"[模块] 语音: {data['modules']['voice']}")
            print(f"[模块] Android: {data['modules']['android']}")
            return True
        else:
            print(f"[失败] 状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"[失败] 错误: {e}")
        return False

def test_detection():
    """测试检测功能"""
    print("\n[测试] 检测分析接口...")
    try:
        image_data = create_test_image()
        
        payload = {
            "image_data": image_data,
            "image_format": "png"
        }
        
        response = requests.post("http://localhost:8082/api/v1/detection/analyze", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            results = data['results']
            
            print(f"[成功] 检测完成: {data['processing_time']:.3f}s")
            print(f"[盲道] 检测到: {results['blind_path']['detected']}")
            print(f"[盲道] 置信度: {results['blind_path']['confidence']:.3f}")
            print(f"[盲道] 类型: {results['blind_path']['path_type']}")
            print(f"[障碍物] 检测到: {results['obstacle']['detected']}")
            print(f"[总结] 总对象: {results['summary']['total_objects']}")
            
            return True
        else:
            print(f"[失败] 状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"[失败] 错误: {e}")
        return False

def test_navigation():
    """测试导航功能"""
    print("\n[测试] 导航系统接口...")
    
    # 测试路线规划
    print("\n[子测试] 路线规划...")
    try:
        payload = {
            "start_latitude": 39.9042,
            "start_longitude": 116.4074,
            "end_latitude": 39.9048,
            "end_longitude": 116.4080,
            "navigation_mode": "walking"
        }
        
        response = requests.post("http://localhost:8082/api/v1/navigation/plan-route", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            route = data['route']
            
            print(f"[成功] 路线规划: {route['route_id']}")
            print(f"[距离] 总距离: {route['total_distance']:.1f}米")
            print(f"[时间] 预计时间: {route['estimated_time']:.1f}分钟")
            print(f"[无障碍] 评分: {route['accessibility_score']:.2f}")
            print(f"[难度] 等级: {route['difficulty_level']}")
            print(f"[路径段] 数量: {len(route['path_segments'])}")
            print(f"[指令] 数量: {len(route['instructions'])}")
            
            # 测试位置更新
            print("\n[子测试] 位置更新...")
            location_payload = {
                "latitude": 39.9045,
                "longitude": 116.4077,
                "accuracy": 5.0
            }
            
            location_response = requests.post("http://localhost:8082/api/v1/navigation/update-location", json=location_payload)
            
            if location_response.status_code == 200:
                location_data = location_response.json()
                print(f"[成功] 位置更新: {location_data['message']}")
                
                if 'current_instruction' in location_data:
                    instruction = location_data['current_instruction']
                    print(f"[指令] 当前指令: {instruction['description']}")
                    print(f"[语音] {instruction['voice_message']}")
                
                return True
            else:
                print(f"[失败] 位置更新状态码: {location_response.status_code}")
                return False
        else:
            print(f"[失败] 路线规划状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"[失败] 错误: {e}")
        return False

def test_voice():
    """测试语音功能"""
    print("\n[测试] 语音合成接口...")
    try:
        payload = {
            "text": "前方发现盲道，可以安全通行",
            "voice_type": "default",
            "speed": 1.0,
            "volume": 1.0
        }
        
        response = requests.post("http://localhost:8082/api/v1/voice/synthesize", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"[成功] 语音合成: {data['audio_id']}")
            print(f"[文本] {data['text']}")
            print(f"[状态] {data['status']}")
            return True
        else:
            print(f"[失败] 状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"[失败] 错误: {e}")
        return False

def test_android():
    """测试Android功能"""
    print("\n[测试] Android模块接口...")
    
    # 测试设备注册
    print("\n[子测试] 设备注册...")
    try:
        import uuid
        device_id = f"test_device_{int(time.time())}"
        
        payload = {
            "device_id": device_id,
            "device_info": {
                "model": "Test Device",
                "os_version": "Android 12",
                "app_version": "1.0.0"
            },
            "capabilities": ["camera", "gps", "voice"]
        }
        
        response = requests.post("http://localhost:8082/api/v1/android/register", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"[成功] 设备注册: {data['data']['device_id']}")
            
            # 测试数据上传
            print("\n[子测试] 数据上传...")
            upload_payload = {
                "device_id": device_id,
                "data_type": "detection_result",
                "data": {
                    "timestamp": time.time(),
                    "location": {"lat": 39.9042, "lng": 116.4074},
                    "detection_results": {"blind_path_detected": True}
                }
            }
            
            upload_response = requests.post("http://localhost:8082/api/v1/android/data/upload", json=upload_payload)
            
            if upload_response.status_code == 200:
                upload_data = upload_response.json()
                print(f"[成功] 数据上传: {upload_data['data']['upload_id']}")
                return True
            else:
                print(f"[失败] 数据上传状态码: {upload_response.status_code}")
                return False
        else:
            print(f"[失败] 设备注册状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"[失败] 错误: {e}")
        return False

def test_navigation_status():
    """测试导航状态"""
    print("\n[测试] 导航状态接口...")
    try:
        response = requests.get("http://localhost:8082/api/v1/navigation/status")
        
        if response.status_code == 200:
            data = response.json()
            nav_status = data['navigation_status']
            
            print(f"[成功] 导航状态获取成功")
            print(f"[导航中] {nav_status['is_navigating']}")
            print(f"[模式] {nav_status['navigation_mode']}")
            
            stats = nav_status['stats']
            print(f"[统计] 规划路线: {stats['routes_planned']}")
            print(f"[统计] 完成路线: {stats['routes_completed']}")
            print(f"[统计] 总距离: {stats['total_distance']:.1f}米")
            
            return True
        else:
            print(f"[失败] 状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"[失败] 错误: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("🧪 完整功能测试 - 第8-10天核心功能开发")
    print("=" * 60)
    
    # 等待服务器启动
    print("\n[等待] 等待API服务器启动...")
    time.sleep(2)
    
    # 执行测试
    tests = [
        ("系统状态", test_system_status),
        ("检测功能", test_detection),
        ("导航功能", test_navigation),
        ("语音功能", test_voice),
        ("Android功能", test_android),
        ("导航状态", test_navigation_status)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"[异常] {test_name}测试异常: {e}")
            results.append((test_name, False))
    
    # 显示测试结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 测试通过")
    print(f"成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 所有测试通过！核心功能开发完成！")
    else:
        print(f"\n⚠️ 有 {total-passed} 个测试失败，需要进一步优化")

if __name__ == "__main__":
    main()
