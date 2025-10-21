#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Android部署脚本
自动将盲道障碍检测应用打包成APK
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_requirements():
    """检查部署环境"""
    print("🔍 检查部署环境...")
    
    # 检查Python版本
    if sys.version_info < (3, 7):
        print("❌ 需要Python 3.7或更高版本")
        return False
    
    # 检查buildozer
    try:
        subprocess.run(["buildozer", "--version"], check=True, capture_output=True)
        print("✅ buildozer已安装")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ buildozer未安装，请运行: pip install buildozer")
        return False
    
    # 检查Android SDK
    android_home = os.environ.get('ANDROID_HOME')
    if not android_home:
        print("⚠️ ANDROID_HOME环境变量未设置")
        print("请安装Android SDK并设置ANDROID_HOME")
        return False
    
    print("✅ 部署环境检查完成")
    return True

def prepare_files():
    """准备部署文件"""
    print("📁 准备部署文件...")
    
    # 创建部署目录
    deploy_dir = Path("android_deploy")
    deploy_dir.mkdir(exist_ok=True)
    
    # 复制必要文件
    files_to_copy = [
        "mobile_app.py",
        "voice_library.py", 
        "voice_config.json",
        "buildozer.spec",
        "requirements.txt"
    ]
    
    for file in files_to_copy:
        if Path(file).exists():
            shutil.copy2(file, deploy_dir)
            print(f"✅ 复制: {file}")
        else:
            print(f"⚠️ 文件不存在: {file}")
    
    # 复制模型文件
    model_files = [
        "runs/detect/train5/weights/best.pt",
        "yolov8n.pt"
    ]
    
    for model in model_files:
        if Path(model).exists():
            model_dir = deploy_dir / "models"
            model_dir.mkdir(exist_ok=True)
            shutil.copy2(model, model_dir)
            print(f"✅ 复制模型: {model}")
        else:
            print(f"⚠️ 模型文件不存在: {model}")
    
    # 创建图标文件（如果不存在）
    icon_file = deploy_dir / "icon.png"
    if not icon_file.exists():
        create_default_icon(icon_file)
    
    print("✅ 文件准备完成")
    return deploy_dir

def create_default_icon(icon_path):
    """创建默认图标"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # 创建512x512的图标
        img = Image.new('RGBA', (512, 512), (255, 255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # 绘制简单的图标
        draw.ellipse([100, 100, 412, 412], fill=(0, 150, 255, 255))
        draw.text((256, 256), "盲", fill=(255, 255, 255, 255), anchor="mm")
        
        img.save(icon_path)
        print("✅ 创建默认图标")
    except ImportError:
        print("⚠️ PIL未安装，跳过图标创建")

def build_apk(deploy_dir):
    """构建APK"""
    print("🔨 开始构建APK...")
    
    os.chdir(deploy_dir)
    
    try:
        # 清理之前的构建
        subprocess.run(["buildozer", "android", "clean"], check=True)
        print("✅ 清理完成")
        
        # 构建APK
        print("⏳ 构建中，请耐心等待...")
        result = subprocess.run(["buildozer", "android", "debug"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ APK构建成功！")
            
            # 查找生成的APK文件
            apk_files = list(Path("bin").glob("*.apk"))
            if apk_files:
                apk_path = apk_files[0]
                print(f"📱 APK文件: {apk_path.absolute()}")
                return apk_path
        else:
            print("❌ APK构建失败")
            print("错误信息:")
            print(result.stderr)
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 构建过程出错: {e}")
        return None
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        return None

def install_apk(apk_path):
    """安装APK到设备"""
    print("📱 安装APK到设备...")
    
    try:
        # 检查设备连接
        devices = subprocess.run(["adb", "devices"], 
                               capture_output=True, text=True)
        
        if "device" not in devices.stdout:
            print("❌ 未检测到Android设备")
            print("请确保设备已连接并启用USB调试")
            return False
        
        # 安装APK
        result = subprocess.run(["adb", "install", "-r", str(apk_path)], 
                              capture_output=True, text=True)
        
        if "Success" in result.stdout:
            print("✅ APK安装成功！")
            return True
        else:
            print("❌ APK安装失败")
            print(result.stderr)
            return False
            
    except FileNotFoundError:
        print("❌ adb未找到，请安装Android SDK")
        return False
    except Exception as e:
        print(f"❌ 安装过程出错: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始Android部署...")
    
    # 检查环境
    if not check_requirements():
        return
    
    # 准备文件
    deploy_dir = prepare_files()
    if not deploy_dir:
        return
    
    # 构建APK
    apk_path = build_apk(deploy_dir)
    if not apk_path:
        return
    
    # 询问是否安装
    install = input("是否安装APK到设备？(y/n): ").lower().strip()
    if install == 'y':
        install_apk(apk_path)
    
    print("🎉 部署完成！")

if __name__ == "__main__":
    main() 