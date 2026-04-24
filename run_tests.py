#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行接口测试脚本
"""

import subprocess
import time
import sys
import os

def start_api_server():
    """启动API服务器"""
    print("[启动] 启动API服务器...")
    try:
        # 启动API服务器进程
        process = subprocess.Popen([
            sys.executable, "start_api_server.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 等待服务器启动
        time.sleep(5)
        
        return process
    except Exception as e:
        print(f"[错误] 启动API服务器失败: {e}")
        return None

def run_tests():
    """运行测试"""
    print("[测试] 运行接口测试...")
    try:
        result = subprocess.run([
            sys.executable, "interface_testing.py"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("错误输出:", result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"[错误] 运行测试失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("盲道检测系统接口测试")
    print("=" * 60)
    
    # 启动API服务器
    server_process = start_api_server()
    if not server_process:
        print("[错误] 无法启动API服务器，测试终止")
        return
    
    try:
        # 运行测试
        success = run_tests()
        
        if success:
            print("\n[成功] 所有测试通过！")
        else:
            print("\n[失败] 部分测试失败")
    
    finally:
        # 停止API服务器
        if server_process:
            print("\n[停止] 停止API服务器...")
            server_process.terminate()
            server_process.wait()

if __name__ == "__main__":
    main()
