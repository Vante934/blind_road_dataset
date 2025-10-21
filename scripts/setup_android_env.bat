@echo off
chcp 65001 >nul
echo 🛠️ Android SDK环境配置工具
echo.

echo 请按照以下步骤操作：
echo.
echo 1. 下载并安装Android Studio
echo    - 访问: https://developer.android.com/studio
echo    - 下载Windows版本
echo    - 安装时选择"Custom"模式
echo    - 确保选中"Android SDK"和"Android Virtual Device"
echo.
echo 2. 找到Android SDK路径
echo    - 默认路径: C:\Users\%USERNAME%\AppData\Local\Android\Sdk
echo    - 或在Android Studio中查看: File -> Settings -> Appearance & Behavior -> System Settings -> Android SDK
echo.
echo 3. 设置环境变量
echo    - 右键"此电脑" -> 属性 -> 高级系统设置 -> 环境变量
echo    - 在"系统变量"中新建 ANDROID_HOME
echo    - 值设为你的Android SDK路径
echo.
echo 4. 添加到PATH
echo    - 在"系统变量"中找到Path
echo    - 添加以下路径：
echo      %ANDROID_HOME%\platform-tools
echo      %ANDROID_HOME%\tools
echo      %ANDROID_HOME%\tools\bin
echo.
echo 5. 重启命令提示符
echo.
echo 完成后运行: python deploy_android.py
echo.
pause 