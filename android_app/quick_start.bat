@echo off
echo ========================================
echo 盲道检测系统 - Android快速启动
echo ========================================

echo.
echo 1. 检查Android Studio环境...
where adb >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ 未找到ADB，请确保Android SDK已正确安装
    pause
    exit /b 1
)
echo ✅ ADB环境检查通过

echo.
echo 2. 检查设备连接...
adb devices
echo.

echo 3. 构建APK...
call gradlew assembleDebug
if %errorlevel% neq 0 (
    echo ❌ 构建失败
    pause
    exit /b 1
)
echo ✅ APK构建成功

echo.
echo 4. 安装到设备...
adb install -r app\build\outputs\apk\debug\app-debug.apk
if %errorlevel% neq 0 (
    echo ❌ 安装失败
    pause
    exit /b 1
)
echo ✅ 应用安装成功

echo.
echo 5. 启动应用...
adb shell am start -n com.blindroad.detector/.ConnectionTestActivity
echo ✅ 应用已启动

echo.
echo ========================================
echo 部署完成！
echo 请在手机上测试连接功能
echo ========================================
pause

