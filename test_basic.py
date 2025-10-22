#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础功能测试脚本
用于验证开发环境是否正确配置
"""

import sys
import os
import subprocess
import importlib

def test_python_version():
    """测试Python版本"""
    print("🐍 测试Python版本...")
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python版本过低，需要3.7+")
        return False
    else:
        print("✅ Python版本符合要求")
        return True

def test_required_packages():
    """测试必需包"""
    print("\n📦 测试必需包...")
    
    required_packages = [
        'numpy', 'opencv-python', 'torch', 'torchvision', 
        'ultralytics', 'PyQt5', 'matplotlib', 'PIL'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                importlib.import_module('cv2')
            elif package == 'PyQt5':
                importlib.import_module('PyQt5')
            else:
                importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  缺少包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    else:
        print("✅ 所有必需包已安装")
        return True

def test_project_structure():
    """测试项目结构"""
    print("\n📁 测试项目结构...")
    
    required_files = [
        'requirements.txt',
        '启动系统.bat',
        '启动模型训练.bat',
        'blind_road_sdk.py',
        'pc_server.py'
    ]
    
    required_dirs = [
        'modules',
        'data',
        'android_app',
        'configs',
        'integration',
        'voice_system',
        'docs'
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"✅ {file}")
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_dirs.append(dir_name)
        else:
            print(f"✅ {dir_name}/")
    
    if missing_files or missing_dirs:
        print(f"\n⚠️  缺少文件: {missing_files}")
        print(f"⚠️  缺少目录: {missing_dirs}")
        return False
    else:
        print("✅ 项目结构完整")
        return True

def test_android_environment():
    """测试Android环境"""
    print("\n🤖 测试Android环境...")
    
    try:
        result = subprocess.run(['adb', 'version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Android SDK已配置")
            return True
        else:
            print("❌ Android SDK未配置")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Android SDK未安装或未添加到PATH")
        print("请安装Android Studio: https://developer.android.com/studio")
        return False

def test_git_environment():
    """测试Git环境"""
    print("\n🔧 测试Git环境...")
    
    try:
        result = subprocess.run(['git', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Git已安装")
            return True
        else:
            print("❌ Git未安装")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Git未安装")
        print("请安装Git: https://git-scm.com/")
        return False

def test_integration_modules():
    """测试整合模块"""
    print("\n🔗 测试整合模块...")
    
    integration_files = [
        'integration/main_controller.py',
        'integration/module_communication.py',
        'integration/system_config.json'
    ]
    
    voice_files = [
        'voice_system/voice_navigator.py',
        'voice_system/voice_config.json'
    ]
    
    missing_files = []
    
    for file in integration_files + voice_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"✅ {file}")
    
    if missing_files:
        print(f"\n⚠️  缺少整合模块文件: {missing_files}")
        return False
    else:
        print("✅ 整合模块完整")
        return True

def main():
    """主测试函数"""
    print("=" * 50)
    print("🧪 盲道检测系统 - 开发环境测试")
    print("=" * 50)
    
    tests = [
        ("Python版本", test_python_version),
        ("必需包", test_required_packages),
        ("项目结构", test_project_structure),
        ("Android环境", test_android_environment),
        ("Git环境", test_git_environment),
        ("整合模块", test_integration_modules)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name}测试出错: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！开发环境配置正确")
        print("\n🚀 可以开始开发了！")
        print("运行命令:")
        print("  - 启动系统: .\\启动系统.bat")
        print("  - 启动训练: .\\启动模型训练.bat")
        print("  - 启动语音: .\\启动语音系统.bat")
    else:
        print("⚠️  部分测试失败，请检查配置")
        print("\n📖 详细配置指南: docs/开发环境配置.md")
    
    print("=" * 50)
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
