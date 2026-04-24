#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出OpenAPI规范到Apifox
用于前后端联调
"""

import json
import requests
import sys
import os
from pathlib import Path

def export_openapi_spec(server_url: str = "http://localhost:8082", output_file: str = "blind_road_api.json"):
    """
    从运行中的服务器导出OpenAPI规范
    
    Args:
        server_url: 服务器地址
        output_file: 输出文件名
    """
    try:
        print(f"📡 正在从 {server_url} 导出OpenAPI规范...")
        
        # 获取OpenAPI JSON
        openapi_url = f"{server_url}/openapi.json"
        response = requests.get(openapi_url, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ 获取OpenAPI规范失败: HTTP {response.status_code}")
            print(f"提示: 请确保服务器正在运行")
            return False
        
        # 保存到文件
        openapi_data = response.json()
        
        # 确保输出目录存在
        output_dir = Path("apifox_config")
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(openapi_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ OpenAPI规范已导出到: {output_path}")
        print(f"📋 API端点数量: {len(openapi_data.get('paths', {}))}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print(f"❌ 连接失败: 无法连接到 {server_url}")
        print("💡 请确保服务器正在运行")
        print("   运行命令: python start_complete_server.py")
        return False
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        return False

def create_apifox_config():
    """创建Apifox配置文件"""
    
    config = {
        "project": {
            "name": "盲道检测系统",
            "description": "包含盲道检测、导航、语音和Android功能的完整API",
            "version": "2.0.0"
        },
        "environments": [
            {
                "name": "开发环境",
                "base_url": "http://localhost:8082",
                "variables": {
                    "api_version": "v1",
                    "device_id": "test_device_001"
                }
            },
            {
                "name": "测试环境",
                "base_url": "http://192.168.1.100:8082",
                "variables": {
                    "api_version": "v1",
                    "device_id": "test_device_002"
                }
            },
            {
                "name": "生产环境",
                "base_url": "https://api.yourdomain.com",
                "variables": {
                    "api_version": "v1",
                    "device_id": "prod_device"
                }
            }
        ],
        "test_scenarios": [
            {
                "name": "完整检测流程",
                "steps": [
                    {
                        "method": "POST",
                        "endpoint": "/api/v1/android/register",
                        "description": "设备注册"
                    },
                    {
                        "method": "POST",
                        "endpoint": "/api/v1/detection/analyze",
                        "description": "图像检测"
                    },
                    {
                        "method": "POST",
                        "endpoint": "/api/v1/voice/synthesize",
                        "description": "语音合成"
                    }
                ]
            }
        ]
    }
    
    # 保存配置文件
    output_dir = Path("apifox_config")
    output_dir.mkdir(exist_ok=True)
    
    config_path = output_dir / "apifox_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Apifox配置文件已创建: {config_path}")
    return True

def print_import_guide():
    """打印Apifox导入指南"""
    print("\n" + "=" * 60)
    print("📖 Apifox导入指南")
    print("=" * 60)
    print("\n1. 打开Apifox应用")
    print("2. 创建新项目或选择现有项目")
    print("3. 点击【导入】→【OpenAPI】")
    print("4. 选择文件: apifox_config/blind_road_api.json")
    print("5. 配置环境变量（参考 apifox_config/apifox_config.json）")
    print("\n完成！现在您可以开始测试API接口了。")
    print("=" * 60)

def main():
    """主函数"""
    print("=" * 60)
    print("🚀 盲道检测系统 - Apifox导出工具")
    print("=" * 60)
    
    # 检查服务器是否运行
    server_url = "http://localhost:8082"
    
    try:
        response = requests.get(f"{server_url}/status", timeout=5)
        if response.status_code == 200:
            print(f"✅ 服务器运行正常")
        else:
            print(f"⚠️ 服务器响应异常")
    except:
        print(f"❌ 无法连接到服务器")
        print(f"\n💡 请先启动服务器:")
        print(f"   python start_complete_server.py")
        choice = input("\n是否继续？（y/n）: ")
        if choice.lower() != 'y':
            return
    
    # 导出OpenAPI规范
    success = export_openapi_spec(server_url)
    
    if success:
        # 创建配置文件
        create_apifox_config()
        
        # 打印导入指南
        print_import_guide()
        
        print("\n✅ 所有文件已准备就绪！")
    else:
        print("\n❌ 导出失败，请检查服务器状态")

if __name__ == "__main__":
    main()



