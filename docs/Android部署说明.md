# 盲道障碍检测 - Android部署说明

## 概述
本指南将帮助你将盲道障碍检测应用部署到Android设备上进行测试。

## 部署方案

### 方案1：Android APK打包（推荐）
将Python应用转换为Android APK，直接在手机上运行。

### 方案2：Web应用 + 手机浏览器
将应用转换为Web应用，通过手机浏览器访问。

### 方案3：Flutter/React Native重写
用跨平台框架重写应用。

## 环境准备

### 1. 安装必要工具

#### 1.1 安装buildozer
```bash
pip install buildozer
```

#### 1.2 安装Android SDK
1. 下载Android Studio: https://developer.android.com/studio
2. 安装Android SDK
3. 设置环境变量：
   - Windows: 设置 `ANDROID_HOME` 为SDK路径
   - Linux/Mac: 在 `~/.bashrc` 中添加 `export ANDROID_HOME=/path/to/android/sdk`

#### 1.3 安装Java JDK
```bash
# Ubuntu/Debian
sudo apt-get install openjdk-8-jdk

# Windows/Mac
# 下载并安装Oracle JDK 8
```

### 2. 检查环境
运行部署脚本检查环境：
```bash
python deploy_android.py
```

## 快速部署

### 方法1：使用自动化脚本
1. 双击运行 `启动Android部署.bat`
2. 按照提示操作
3. 等待APK生成完成

### 方法2：手动部署
1. 准备文件：
   ```bash
   python deploy_android.py
   ```

2. 构建APK：
   ```bash
   cd android_deploy
   buildozer android debug
   ```

3. 安装到设备：
   ```bash
   adb install bin/blindroaddetector-1.0-debug.apk
   ```

## 文件说明

### 核心文件
- `mobile_app.py`: 移动端应用主文件
- `voice_library.py`: 语音库系统
- `voice_config.json`: 语音配置文件
- `buildozer.spec`: Android构建配置
- `deploy_android.py`: 自动化部署脚本

### 模型文件
- `runs/detect/train5/weights/best.pt`: 你的自定义模型
- `yolov8n.pt`: 默认YOLO模型

## 功能特性

### 移动端优化
- 简化的UI界面，适合手机屏幕
- 触摸友好的按钮设计
- 实时摄像头检测
- 智能语音播报

### 语音系统
- 百度语音API集成
- 多级语音预警
- 环境声音识别
- 可自定义语音库

### 检测功能
- 实时障碍物检测
- 距离估算
- 方向识别
- 多类别支持

## 测试指南

### 1. 基础功能测试
1. 启动应用
2. 点击"开始检测"
3. 测试语音开关
4. 测试语音播报

### 2. 检测功能测试
1. 对准不同障碍物
2. 测试距离检测
3. 测试方向识别
4. 测试语音提示

### 3. 性能测试
1. 测试帧率
2. 测试内存使用
3. 测试电池消耗
4. 测试稳定性

## 常见问题

### Q1: buildozer安装失败
A: 确保使用Python 3.7+，并安装必要的依赖：
```bash
pip install --upgrade pip
pip install buildozer
```

### Q2: Android SDK路径错误
A: 检查环境变量设置：
```bash
echo $ANDROID_HOME  # Linux/Mac
echo %ANDROID_HOME% # Windows
```

### Q3: APK构建失败
A: 检查依赖和权限：
1. 确保所有Python包都已安装
2. 检查网络连接
3. 查看详细错误日志

### Q4: 应用崩溃
A: 检查日志：
```bash
adb logcat | grep blindroaddetector
```

## 性能优化

### 1. 模型优化
- 使用更小的模型
- 量化模型权重
- 使用TensorRT加速

### 2. 代码优化
- 减少不必要的计算
- 优化内存使用
- 使用多线程处理

### 3. 电池优化
- 降低检测频率
- 优化摄像头使用
- 减少网络请求

## 后续开发

### 1. 功能扩展
- 添加更多障碍物类型
- 支持更多语音选项
- 添加用户设置界面

### 2. 性能提升
- 使用更先进的模型
- 优化检测算法
- 改进用户体验

### 3. 平台扩展
- 支持iOS
- 开发Web版本
- 创建桌面应用

## 技术支持

如遇到问题，请：
1. 查看错误日志
2. 检查环境配置
3. 参考常见问题
4. 联系技术支持

## 更新日志

### v1.0
- 初始版本
- 基础检测功能
- 语音播报系统
- Android部署支持 