#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试API服务器根路径
"""

import requests
import json

def test_api_root():
    """测试API根路径"""
    try:
        print("测试API根路径...")
        response = requests.get("http://localhost:8082/")
        
        print(f"状态码: {response.status_code}")
        print(f"响应内容: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"API版本: {data.get('version')}")
            print(f"服务状态: {data.get('status')}")
            print(f"可用端点: {list(data.get('endpoints', {}).keys())}")
        else:
            print("❌ API根路径访问失败")
            
    except Exception as e:
        print(f"❌ 连接失败: {e}")

def test_system_status():
    """测试系统状态端点"""
    try:
        print("\n测试系统状态端点...")
        response = requests.get("http://localhost:8082/api/v1/system/status")
        
        print(f"状态码: {response.status_code}")
        print(f"响应内容: {response.text}")
        
        if response.status_code == 200:
            print("✅ 系统状态端点正常")
        else:
            print("❌ 系统状态端点访问失败")
            
    except Exception as e:
        print(f"❌ 连接失败: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("API服务器测试")
    print("=" * 50)
    
    test_api_root()
    test_system_status()
    
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)
