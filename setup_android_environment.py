#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Android环境配置脚本
帮助用户设置Android SDK和相关开发工具
"""

import os
import sys
import subprocess
import json
import platform
from pathlib import Path

class AndroidEnvironmentSetup:
    def __init__(self):
        self.system = platform.system()
        self.home_dir = Path.home()
        self.android_sdk_path = None
        self.adb_path = None
        self.gradle_path = None
        
    def check_java(self):
        """检查Java环境"""
        print("🔍 检查Java环境...")
        try:
            result = subprocess.run(['java', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ Java环境正常")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        print("❌ Java未安装或未添加到PATH")
        print("请安装Java JDK 8或更高版本")
        return False
    
    def find_android_sdk(self):
        """查找Android SDK路径"""
        print("🔍 查找Android SDK...")
        
        # 常见的Android SDK路径
        possible_paths = []
        
        if self.system == "Windows":
            possible_paths = [
                "C:\\Users\\%USERNAME%\\AppData\\Local\\Android\\Sdk",
                "C:\\Android\\Sdk",
                "C:\\Program Files\\Android\\Sdk",
                "C:\\Program Files (x86)\\Android\\Sdk",
                os.path.expandvars("%ANDROID_HOME%"),
                os.path.expandvars("%ANDROID_SDK_ROOT%")
            ]
        elif self.system == "Darwin":  # macOS
            possible_paths = [
                str(self.home_dir / "Library/Android/sdk"),
                str(self.home_dir / "Android/Sdk"),
                "/usr/local/android-sdk",
                os.path.expandvars("$ANDROID_HOME"),
                os.path.expandvars("$ANDROID_SDK_ROOT")
            ]
        else:  # Linux
            possible_paths = [
                str(self.home_dir / "Android/Sdk"),
                "/usr/local/android-sdk",
                "/opt/android-sdk",
                os.path.expandvars("$ANDROID_HOME"),
                os.path.expandvars("$ANDROID_SDK_ROOT")
            ]
        
        for path in possible_paths:
            if path and os.path.exists(path):
                sdk_path = Path(path)
                if (sdk_path / "platform-tools" / "adb").exists():
                    self.android_sdk_path = sdk_path
                    self.adb_path = sdk_path / "platform-tools" / "adb"
                    print(f"✅ 找到Android SDK: {sdk_path}")
                    return True
        
        print("❌ 未找到Android SDK")
        return False
    
    def find_gradle(self):
        """查找Gradle"""
        print("🔍 查找Gradle...")
        
        # 检查PATH中的gradle
        try:
            result = subprocess.run(['gradle', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ Gradle已安装")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # 检查Android Studio中的gradle
        if self.android_sdk_path:
            gradle_wrapper = self.android_sdk_path / "gradle" / "gradle-7.5" / "bin" / "gradle"
            if gradle_wrapper.exists():
                self.gradle_path = gradle_wrapper
                print(f"✅ 找到Gradle: {gradle_wrapper}")
                return True
        
        print("❌ 未找到Gradle")
        return False
    
    def setup_environment_variables(self):
        """设置环境变量"""
        print("🔧 设置环境变量...")
        
        if not self.android_sdk_path:
            print("❌ 无法设置环境变量：未找到Android SDK")
            return False
        
        # 创建环境变量设置脚本
        if self.system == "Windows":
            self.create_windows_env_script()
        else:
            self.create_unix_env_script()
        
        print("✅ 环境变量脚本已创建")
        return True
    
    def create_windows_env_script(self):
        """创建Windows环境变量设置脚本"""
        script_content = f"""@echo off
echo 设置Android开发环境变量...

set ANDROID_HOME={self.android_sdk_path}
set ANDROID_SDK_ROOT={self.android_sdk_path}
set PATH=%ANDROID_HOME%\\platform-tools;%ANDROID_HOME%\\tools;%ANDROID_HOME%\\tools\\bin;%PATH%

echo Android SDK路径: %ANDROID_HOME%
echo 请将此脚本添加到系统环境变量或手动运行
echo.
echo 测试ADB连接:
adb version
"""
        
        with open("setup_android_env.bat", "w", encoding="utf-8") as f:
            f.write(script_content)
    
    def create_unix_env_script(self):
        """创建Unix环境变量设置脚本"""
        script_content = f"""#!/bin/bash
echo "设置Android开发环境变量..."

export ANDROID_HOME={self.android_sdk_path}
export ANDROID_SDK_ROOT={self.android_sdk_path}
export PATH=$ANDROID_HOME/platform-tools:$ANDROID_HOME/tools:$ANDROID_HOME/tools/bin:$PATH

echo "Android SDK路径: $ANDROID_HOME"
echo "请将此脚本添加到 ~/.bashrc 或 ~/.zshrc"
echo ""
echo "测试ADB连接:"
adb version
"""
        
        with open("setup_android_env.sh", "w", encoding="utf-8") as f:
            f.write(script_content)
        
        # 设置执行权限
        os.chmod("setup_android_env.sh", 0o755)
    
    def download_android_sdk(self):
        """下载Android SDK（如果未找到）"""
        print("📥 准备下载Android SDK...")
        
        if self.system == "Windows":
            print("请手动下载Android Studio:")
            print("https://developer.android.com/studio")
            print("安装后，Android SDK通常位于:")
            print("C:\\Users\\%USERNAME%\\AppData\\Local\\Android\\Sdk")
        else:
            print("请手动下载Android SDK:")
            print("https://developer.android.com/studio#command-tools")
            print("或安装Android Studio")
    
    def create_manual_setup_guide(self):
        """创建手动设置指南"""
        guide_content = """# Android开发环境手动设置指南

## 方法1：安装Android Studio（推荐）

1. 访问 https://developer.android.com/studio
2. 下载并安装Android Studio
3. 启动Android Studio，完成初始设置
4. 在SDK Manager中安装必要的SDK组件

## 方法2：仅安装Android SDK

### Windows
1. 下载Android SDK Command-line Tools
2. 解压到 C:\\Android\\Sdk
3. 设置环境变量：
   - ANDROID_HOME = C:\\Android\\Sdk
   - PATH += %ANDROID_HOME%\\platform-tools

### macOS/Linux
1. 下载Android SDK Command-line Tools
2. 解压到 ~/Android/Sdk
3. 设置环境变量：
   ```bash
   export ANDROID_HOME=~/Android/Sdk
   export PATH=$ANDROID_HOME/platform-tools:$PATH
   ```

## 验证安装

运行以下命令验证：
```bash
adb version
```

## 常见问题

1. **ADB命令未找到**
   - 检查环境变量设置
   - 重启终端/命令提示符

2. **权限问题（Linux/macOS）**
   - 确保adb有执行权限
   - 可能需要sudo权限

3. **Android Studio找不到SDK**
   - 在Android Studio中设置SDK路径
   - File > Settings > Appearance & Behavior > System Settings > Android SDK
"""
        
        with open("Android环境设置指南.md", "w", encoding="utf-8") as f:
            f.write(guide_content)
    
    def run(self):
        """运行环境设置"""
        print("🚀 Android开发环境配置工具")
        print("=" * 50)
        
        # 检查Java
        java_ok = self.check_java()
        
        # 查找Android SDK
        sdk_ok = self.find_android_sdk()
        
        # 查找Gradle
        gradle_ok = self.find_gradle()
        
        print("\n" + "=" * 50)
        print("📊 环境检查结果:")
        print(f"Java: {'✅' if java_ok else '❌'}")
        print(f"Android SDK: {'✅' if sdk_ok else '❌'}")
        print(f"Gradle: {'✅' if gradle_ok else '❌'}")
        
        if sdk_ok and gradle_ok:
            print("\n🎉 环境配置完成！")
            self.setup_environment_variables()
            print("\n现在可以运行:")
            print("python deploy_android_app.py deploy")
        else:
            print("\n⚠️ 环境配置不完整")
            if not sdk_ok:
                print("\n需要安装Android SDK:")
                self.download_android_sdk()
                self.create_manual_setup_guide()
            
            if not gradle_ok:
                print("\n需要安装Gradle:")
                print("建议通过Android Studio安装")
        
        print("\n📖 详细设置指南已保存到: Android环境设置指南.md")

def main():
    setup = AndroidEnvironmentSetup()
    setup.run()

if __name__ == "__main__":
    main()

