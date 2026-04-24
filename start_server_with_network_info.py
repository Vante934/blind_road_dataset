#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动服务器并显示网络配置信息
"""

import sys
import os
import socket
import platform

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_local_ip():
    """获取本机IP地址"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def show_network_info(port=8082):
    """显示网络信息"""
    local_ip = get_local_ip()
    
    print("\n" + "=" * 60)
    print("🌐 网络配置信息")
    print("=" * 60)
    print(f"💻 本机IP地址: {local_ip}")
    print(f"📍 系统平台: {platform.system()} {platform.release()}")
    print(f"🌐 主机名: {socket.gethostname()}")
    print(f"🔌 监听端口: {port}")
    print("=" * 60)
    print(f"\n📱 Android端配置:")
    print(f"   服务器地址: http://{local_ip}:{port}")
    print(f"   完整URL: http://{local_ip}:{port}/status")
    print("=" * 60)
    print(f"\n🌐 API端点:")
    print(f"   • 根路径: http://{local_ip}:{port}/")
    print(f"   • 状态检查: http://{local_ip}:{port}/status")
    print(f"   • 系统状态: http://{local_ip}:{port}/api/v1/system/status")
    print(f"   • 检测接口: http://{local_ip}:{port}/api/v1/detection/analyze")
    print(f"   • 语音接口: http://{local_ip}:{port}/api/v1/voice/synthesize")
    print(f"   • 设备注册: http://{local_ip}:{port}/api/v1/android/register")
    print(f"   • API文档: http://{local_ip}:{port}/docs")
    print("=" * 60)
    
    print(f"\n💡 提示:")
    print(f"   1. 确保Android设备与电脑在同一WiFi网络")
    print(f"   2. 如果连接失败，检查Windows防火墙设置")
    print(f"   3. 可以在浏览器中访问 http://{local_ip}:{port}/status 测试")
    print(f"   4. 使用 Ctrl+C 停止服务器")
    print("=" * 60 + "\n")
    
    return local_ip

if __name__ == "__main__":
    from complete_integrated_api_server import start_server
    
    # 显示网络信息
    local_ip = show_network_info(8082)
    
    # 启动服务器
    print("🚀 正在启动服务器...\n")
    try:
        start_server()
    except KeyboardInterrupt:
        print("\n\n[停止] 服务器已停止")
    except Exception as e:
        print(f"\n\n[错误] 服务器启动失败: {e}")
        sys.exit(1)
