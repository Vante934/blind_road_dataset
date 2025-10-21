# Android Studio 安装和Hello World项目创建指南

## 第一步：下载Android Studio

1. 访问官方下载页面：https://developer.android.google.cn/studio
2. 点击"Download Android Studio"按钮
3. 选择Windows版本下载（约1GB）

## 第二步：安装Android Studio

1. 双击下载的`.exe`文件
2. 按照安装向导完成安装：
   - 选择安装组件（保持默认选择）
   - 选择安装路径（建议默认路径）
   - 等待安装完成

## 第三步：首次配置

1. 启动Android Studio
2. 选择"Do not import settings"
3. 选择"Standard"安装类型
4. 接受许可协议
5. 等待SDK组件下载完成

## 第四步：创建Hello World项目

### 4.1 创建新项目
1. 在欢迎界面点击"Start a new Android Studio project"
2. 选择"Empty Activity"模板
3. 点击"Next"

### 4.2 配置项目
1. **Name**: HelloWorld
2. **Package name**: com.example.helloworld
3. **Save location**: 选择你想要的保存位置
4. **Language**: 选择Kotlin（推荐）或Java
5. **Minimum SDK**: API 21 (Android 5.0) 或更高
6. 点击"Finish"

### 4.3 等待项目构建
- Android Studio会自动下载必要的依赖
- 等待Gradle构建完成

## 第五步：运行Hello World应用

### 5.1 创建虚拟设备（模拟器）
1. 点击工具栏的"Device Manager"图标
2. 点击"Create Device"
3. 选择"Phone" > "Pixel 4"（或其他设备）
4. 选择系统镜像（建议选择最新的API级别）
5. 点击"Next" > "Finish"

### 5.2 运行应用
1. 确保虚拟设备已启动
2. 点击工具栏的绿色"Run"按钮（▶️）
3. 选择你的虚拟设备
4. 等待应用编译和安装
5. 应用会在模拟器中启动，显示"Hello World!"

## 第六步：修改Hello World文本

1. 在项目结构中，打开`app/src/main/res/layout/activity_main.xml`
2. 找到`android:text="Hello World!"`这一行
3. 修改为你想要的文本，例如：
   ```xml
   android:text="你好，世界！"
   ```
4. 保存文件
5. 再次点击"Run"按钮查看更改

## 常见问题解决

### 问题1：SDK下载失败
- 检查网络连接
- 尝试使用VPN
- 手动下载SDK组件

### 问题2：模拟器启动失败
- 确保启用了虚拟化技术（在BIOS中）
- 检查Windows功能中的Hyper-V设置

### 问题3：项目构建失败
- 检查网络连接
- 尝试"File" > "Sync Project with Gradle Files"
- 清理项目："Build" > "Clean Project"

## 下一步学习建议

1. 学习Android基础概念：Activity、Layout、View
2. 学习Kotlin/Java语言基础
3. 学习Android UI组件
4. 学习事件处理
5. 学习数据存储

## 有用的资源

- Android官方文档：https://developer.android.google.cn/docs
- Kotlin官方文档：https://kotlinlang.org/docs/
- Android开发者社区：https://developer.android.google.cn/community











