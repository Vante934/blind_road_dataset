# Android SDK 问题解决指南

## 问题描述
Android Studio显示错误："Your Android SDK is missing, out of date or corrupted"

## 解决方案

### 方法一：通过Android Studio修复（推荐）

1. **点击"Open SDK Manager"按钮**
   - 在错误对话框中点击"Open SDK Manager"
   - 或者通过菜单：Tools → SDK Manager

2. **安装Android SDK**
   - 在"SDK Platforms"标签页中：
     - 勾选最新的Android版本（推荐API 34）
     - 勾选Android 7.0 (API 24) - 最低支持版本
   - 在"SDK Tools"标签页中：
     - 勾选"Android SDK Build-Tools"
     - 勾选"Android SDK Platform-Tools"
     - 勾选"Android SDK Tools"
     - 勾选"Android Emulator"（如果需要模拟器）

3. **点击"Apply"或"OK"**
   - 等待下载和安装完成

### 方法二：手动配置SDK路径

1. **打开SDK Manager**
   - File → Settings → Appearance & Behavior → System Settings → Android SDK

2. **设置SDK路径**
   - 如果SDK已安装但路径错误，点击"Edit"
   - 设置正确的SDK路径（通常是：`C:\Users\用户名\AppData\Local\Android\Sdk`）

3. **验证安装**
   - 确保SDK路径正确
   - 确保必要的组件已安装

### 方法三：重新安装Android Studio

如果上述方法无效，可以重新安装：

1. **卸载当前版本**
   - 通过控制面板卸载Android Studio
   - 删除SDK文件夹（可选）

2. **下载最新版本**
   - 访问：https://developer.android.com/studio
   - 下载最新版本

3. **重新安装**
   - 按照安装向导进行
   - 确保勾选"Android SDK"选项

## 验证修复

### 1. 检查SDK状态
```bash
# 在命令行中检查
adb version
```

### 2. 在Android Studio中验证
- File → Project Structure → SDK Location
- 确保SDK路径正确显示

### 3. 创建测试项目
- 创建新的Empty Activity项目
- 检查是否能正常构建

## 常见问题解决

### 问题1：网络连接问题
**解决方案：**
- 使用VPN或代理
- 配置Android Studio代理设置
- 使用国内镜像源

### 问题2：权限问题
**解决方案：**
- 以管理员身份运行Android Studio
- 检查SDK文件夹权限

### 问题3：磁盘空间不足
**解决方案：**
- 清理磁盘空间
- 移动SDK到其他磁盘

## 针对您的盲人项目

### 1. 确保SDK版本兼容
```kotlin
// 在build.gradle.kts中确保版本兼容
android {
    compileSdk = 34
    defaultConfig {
        minSdk = 24
        targetSdk = 34
    }
}
```

### 2. 安装必要的组件
- Android SDK Platform 34
- Android SDK Build-Tools 34.0.0
- Android SDK Platform-Tools
- Android Emulator（用于测试）

### 3. 配置NDK（用于机器学习）
- 在SDK Manager中安装NDK
- 确保版本兼容TensorFlow Lite

## 快速修复脚本

### Windows批处理脚本
```batch
@echo off
echo 正在修复Android SDK问题...

REM 设置环境变量
set ANDROID_HOME=C:\Users\%USERNAME%\AppData\Local\Android\Sdk
set PATH=%PATH%;%ANDROID_HOME%\tools;%ANDROID_HOME%\platform-tools

REM 检查SDK状态
echo 检查SDK状态...
adb version

echo 修复完成！
pause
```

### PowerShell脚本
```powershell
# 设置环境变量
$env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"
$env:PATH += ";$env:ANDROID_HOME\tools;$env:ANDROID_HOME\platform-tools"

# 检查SDK状态
Write-Host "检查SDK状态..."
adb version
```

## 后续步骤

1. **修复SDK问题后**
2. **重新打开您的盲人项目**
3. **同步Gradle文件**
4. **测试连接功能**

## 联系支持

如果问题仍然存在，请提供：
- Android Studio版本
- 操作系统版本
- 错误日志
- SDK安装路径

