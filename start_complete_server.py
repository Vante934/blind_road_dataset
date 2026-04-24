#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动完整集成API服务器
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from complete_integrated_api_server import start_server

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 盲道检测系统 - 完整集成API服务器")
    print("=" * 60)
    
    try:
        start_server()
    except KeyboardInterrupt:
        print("\n[停止] 服务器已停止")
    except Exception as e:
        print(f"\n❌ 服务器启动失败: {e}")
        sys.exit(1)
