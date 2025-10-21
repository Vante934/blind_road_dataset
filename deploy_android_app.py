#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Android应用部署脚本
自动构建、安装和配置Android应用
"""

import os
import sys
import subprocess
import json
import time
import shutil
from typing import Dict, List, Optional

class AndroidAppDeployer:
    """Android应用部署器"""
    
    def __init__(self):
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.android_project_dir = os.path.join(self.project_root, "android_app")
        self.build_output_dir = os.path.join(self.android_project_dir, "app", "build", "outputs")
        self.apk_path = None
        
    def check_environment(self) -> bool:
        """检查Android开发环境"""
        print("🔍 检查Android开发环境...")
        
        # 检查Android SDK
        if not self.check_android_sdk():
            print("❌ Android SDK未找到")
            return False
        
        # 检查Gradle
        if not self.check_gradle():
            print("❌ Gradle未找到")
            return False
        
        # 检查Java
        if not self.check_java():
            print("❌ Java未找到")
            return False
        
        print("✅ Android开发环境检查通过")
        return True
    
    def check_android_sdk(self) -> bool:
        """检查Android SDK"""
        try:
            # 检查ANDROID_HOME环境变量
            android_home = os.environ.get('ANDROID_HOME')
            if android_home and os.path.exists(android_home):
                print(f"✅ Android SDK: {android_home}")
                return True
            
            # 检查ANDROID_SDK_ROOT环境变量
            android_sdk_root = os.environ.get('ANDROID_SDK_ROOT')
            if android_sdk_root and os.path.exists(android_sdk_root):
                print(f"✅ Android SDK: {android_sdk_root}")
                return True
            
            # 检查默认路径
            default_paths = [
                os.path.expanduser("~/Android/Sdk"),
                os.path.expandvars("C:\\Users\\%USERNAME%\\AppData\\Local\\Android\\Sdk"),
                "C:\\Android\\Sdk",
                "C:\\Program Files\\Android\\Sdk",
                "C:\\Program Files (x86)\\Android\\Sdk",
                "/usr/local/android-sdk"
            ]
            
            for path in default_paths:
                if path and os.path.exists(path):
                    print(f"✅ Android SDK: {path}")
                    return True
            
            print("❌ 未找到Android SDK")
            print("请先运行: python setup_android_environment.py")
            return False
            
        except Exception as e:
            print(f"❌ 检查Android SDK失败: {e}")
            return False
    
    def check_gradle(self) -> bool:
        """检查Gradle"""
        try:
            result = subprocess.run(['gradle', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Gradle可用")
                return True
            return False
        except FileNotFoundError:
            return False
    
    def check_java(self) -> bool:
        """检查Java"""
        try:
            result = subprocess.run(['java', '-version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Java可用")
                return True
            return False
        except FileNotFoundError:
            return False
    
    def build_apk(self) -> bool:
        """构建APK"""
        print("🔨 开始构建APK...")
        
        try:
            # 切换到Android项目目录
            os.chdir(self.android_project_dir)
            
            # 清理构建
            print("🧹 清理构建文件...")
            subprocess.run(['gradle', 'clean'], check=True)
            
            # 构建Debug APK
            print("🔨 构建Debug APK...")
            subprocess.run(['gradle', 'assembleDebug'], check=True)
            
            # 查找APK文件
            apk_dir = os.path.join(self.build_output_dir, "apk", "debug")
            if os.path.exists(apk_dir):
                apk_files = [f for f in os.listdir(apk_dir) if f.endswith('.apk')]
                if apk_files:
                    self.apk_path = os.path.join(apk_dir, apk_files[0])
                    print(f"✅ APK构建成功: {self.apk_path}")
                    return True
            
            print("❌ 未找到APK文件")
            return False
            
        except subprocess.CalledProcessError as e:
            print(f"❌ APK构建失败: {e}")
            return False
        except Exception as e:
            print(f"❌ APK构建出错: {e}")
            return False
        finally:
            # 返回项目根目录
            os.chdir(self.project_root)
    
    def install_apk(self, device_id: str = None) -> bool:
        """安装APK到设备"""
        if not self.apk_path or not os.path.exists(self.apk_path):
            print("❌ APK文件不存在")
            return False
        
        print(f"📱 安装APK到设备...")
        
        try:
            # 获取设备列表
            devices = self.get_connected_devices()
            if not devices:
                print("❌ 没有找到已连接的设备")
                return False
            
            target_device = device_id or devices[0]
            print(f"📱 目标设备: {target_device}")
            
            # 安装APK
            cmd = ['adb', '-s', target_device, 'install', '-r', self.apk_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and "Success" in result.stdout:
                print("✅ APK安装成功")
                return True
            else:
                print(f"❌ APK安装失败: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ 安装APK失败: {e}")
            return False
    
    def get_connected_devices(self) -> List[str]:
        """获取已连接的设备列表"""
        try:
            result = subprocess.run(['adb', 'devices'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]
                devices = []
                for line in lines:
                    if line.strip() and '\tdevice' in line:
                        device_id = line.split('\t')[0]
                        devices.append(device_id)
                return devices
        except Exception as e:
            print(f"❌ 获取设备列表失败: {e}")
        return []
    
    def configure_app(self) -> bool:
        """配置应用"""
        print("⚙️ 配置应用...")
        
        try:
            # 创建配置文件
            config = {
                "server_url": "http://10.82.209.144:8080",
                "model_path": "models/best.pt",
                "detection_confidence": 0.5,
                "data_collection_interval": 1000,
                "auto_update": True
            }
            
            config_path = os.path.join(self.android_project_dir, "app", "src", "main", "assets", "config.json")
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            print("✅ 应用配置完成")
            return True
            
        except Exception as e:
            print(f"❌ 配置应用失败: {e}")
            return False
    
    def copy_models(self) -> bool:
        """复制模型文件到Android项目"""
        print("📁 复制模型文件...")
        
        try:
            # 源模型目录
            source_models_dir = os.path.join(self.project_root, "models")
            if not os.path.exists(source_models_dir):
                print("⚠️ 源模型目录不存在")
                return True
            
            # 目标模型目录
            target_models_dir = os.path.join(self.android_project_dir, "app", "src", "main", "assets", "models")
            os.makedirs(target_models_dir, exist_ok=True)
            
            # 复制模型文件
            model_files = [f for f in os.listdir(source_models_dir) if f.endswith('.pt')]
            for model_file in model_files:
                source_path = os.path.join(source_models_dir, model_file)
                target_path = os.path.join(target_models_dir, model_file)
                shutil.copy2(source_path, target_path)
                print(f"📁 复制模型: {model_file}")
            
            print("✅ 模型文件复制完成")
            return True
            
        except Exception as e:
            print(f"❌ 复制模型文件失败: {e}")
            return False
    
    def start_app(self, device_id: str = None) -> bool:
        """启动应用"""
        print("🚀 启动应用...")
        
        try:
            devices = self.get_connected_devices()
            if not devices:
                print("❌ 没有找到已连接的设备")
                return False
            
            target_device = device_id or devices[0]
            package_name = "com.blindroad.detector"
            
            # 启动应用
            cmd = ['adb', '-s', target_device, 'shell', 'am', 'start', '-n', 
                   f"{package_name}/.MainActivity"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ 应用启动成功")
                return True
            else:
                print(f"❌ 应用启动失败: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ 启动应用失败: {e}")
            return False
    
    def deploy(self, device_id: str = None) -> bool:
        """完整部署流程"""
        print("🚀 开始Android应用部署")
        print("=" * 50)
        
        # 检查环境
        if not self.check_environment():
            return False
        
        # 配置应用
        if not self.configure_app():
            return False
        
        # 复制模型文件
        if not self.copy_models():
            return False
        
        # 构建APK
        if not self.build_apk():
            return False
        
        # 安装APK
        if not self.install_apk(device_id):
            return False
        
        # 启动应用
        if not self.start_app(device_id):
            return False
        
        print("✅ Android应用部署完成！")
        return True
    
    def create_deployment_script(self) -> bool:
        """创建部署脚本"""
        print("📝 创建部署脚本...")
        
        try:
            script_content = '''#!/bin/bash
# Android应用自动部署脚本

echo "🚀 开始Android应用部署..."

# 检查ADB
if ! command -v adb &> /dev/null; then
    echo "❌ ADB未找到，请安装Android SDK"
    exit 1
fi

# 检查设备连接
echo "📱 检查设备连接..."
devices=$(adb devices | grep -v "List" | grep "device" | wc -l)
if [ $devices -eq 0 ]; then
    echo "❌ 没有找到已连接的设备"
    exit 1
fi

echo "✅ 找到 $devices 个设备"

# 构建APK
echo "🔨 构建APK..."
cd android_app
./gradlew clean assembleDebug

if [ $? -ne 0 ]; then
    echo "❌ APK构建失败"
    exit 1
fi

# 安装APK
echo "📱 安装APK..."
APK_PATH="app/build/outputs/apk/debug/app-debug.apk"
if [ -f "$APK_PATH" ]; then
    adb install -r "$APK_PATH"
    if [ $? -eq 0 ]; then
        echo "✅ APK安装成功"
    else
        echo "❌ APK安装失败"
        exit 1
    fi
else
    echo "❌ APK文件不存在: $APK_PATH"
    exit 1
fi

# 启动应用
echo "🚀 启动应用..."
adb shell am start -n com.blindroad.detector/.MainActivity

echo "✅ 部署完成！"
'''
            
            script_path = os.path.join(self.project_root, "deploy_android.sh")
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            # 设置执行权限
            os.chmod(script_path, 0o755)
            
            print(f"✅ 部署脚本已创建: {script_path}")
            return True
            
        except Exception as e:
            print(f"❌ 创建部署脚本失败: {e}")
            return False

def main():
    """主函数"""
    deployer = AndroidAppDeployer()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "deploy":
            device_id = sys.argv[2] if len(sys.argv) > 2 else None
            deployer.deploy(device_id)
        elif command == "build":
            deployer.build_apk()
        elif command == "install":
            device_id = sys.argv[2] if len(sys.argv) > 2 else None
            deployer.install_apk(device_id)
        elif command == "script":
            deployer.create_deployment_script()
        else:
            print("用法: python deploy_android_app.py [deploy|build|install|script] [device_id]")
    else:
        # 默认执行完整部署
        deployer.deploy()

if __name__ == "__main__":
    main() 