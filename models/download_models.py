#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型下载脚本
自动下载YOLOv8预训练模型
"""

import os
import sys
import urllib.request
from pathlib import Path

# 模型下载URL
MODEL_URLS = {
    'yolov8n.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
    'yolov8s.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
    'yolov8m.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
    'yolov8l.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt',
    'yolov8x.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt',
}

def download_file(url, filepath):
    """下载文件"""
    try:
        print(f"正在下载: {os.path.basename(filepath)}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"✅ 下载完成: {filepath}")
        return True
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def main():
    """主函数"""
    # 获取当前脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = script_dir
    
    # 确保models目录存在
    os.makedirs(models_dir, exist_ok=True)
    
    print("=" * 50)
    print("YOLOv8 模型下载工具")
    print("=" * 50)
    print(f"模型保存目录: {models_dir}")
    print()
    
    # 显示可用模型
    print("可用模型:")
    for i, model_name in enumerate(MODEL_URLS.keys(), 1):
        filepath = os.path.join(models_dir, model_name)
        exists = "✓" if os.path.exists(filepath) else " "
        size_info = ""
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            size_info = f" ({size:.1f} MB)"
        print(f"  {exists} {i}. {model_name}{size_info}")
    print()
    
    # 选择要下载的模型
    print("请选择要下载的模型:")
    print("  1. yolov8n.pt (推荐，最小最快，~6MB)")
    print("  2. yolov8s.pt (~22MB)")
    print("  3. yolov8m.pt (~52MB)")
    print("  4. yolov8l.pt (~88MB)")
    print("  5. yolov8x.pt (~136MB)")
    print("  6. 下载所有模型")
    print("  0. 退出")
    print()
    
    choice = input("请输入选项 (0-6): ").strip()
    
    if choice == '0':
        print("退出")
        return
    
    models_to_download = []
    
    if choice == '1':
        models_to_download = ['yolov8n.pt']
    elif choice == '2':
        models_to_download = ['yolov8s.pt']
    elif choice == '3':
        models_to_download = ['yolov8m.pt']
    elif choice == '4':
        models_to_download = ['yolov8l.pt']
    elif choice == '5':
        models_to_download = ['yolov8x.pt']
    elif choice == '6':
        models_to_download = list(MODEL_URLS.keys())
    else:
        print("无效选项")
        return
    
    # 下载模型
    success_count = 0
    for model_name in models_to_download:
        filepath = os.path.join(models_dir, model_name)
        
        # 检查文件是否已存在
        if os.path.exists(filepath):
            overwrite = input(f"{model_name} 已存在，是否覆盖? (y/n): ").strip().lower()
            if overwrite != 'y':
                print(f"跳过: {model_name}")
                continue
        
        url = MODEL_URLS[model_name]
        if download_file(url, filepath):
            success_count += 1
    
    print()
    print("=" * 50)
    print(f"下载完成: {success_count}/{len(models_to_download)} 个模型")
    print("=" * 50)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断下载")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
