@echo off
echo ========================================
echo Android SDK 问题修复工具
echo ========================================

echo.
echo 1. 检查Android Studio安装...
where studio64.exe >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ 未找到Android Studio，请先安装Android Studio
    echo 下载地址: https://developer.android.com/studio
    pause
    exit /b 1
)
echo ✅ Android Studio已安装

echo.
echo 2. 设置环境变量...
set ANDROID_HOME=%LOCALAPPDATA%\Android\Sdk
set PATH=%PATH%;%ANDROID_HOME%\tools;%ANDROID_HOME%\platform-tools
echo ✅ 环境变量已设置

echo.
echo 3. 检查SDK状态...
if exist "%ANDROID_HOME%" (
    echo ✅ SDK目录存在: %ANDROID_HOME%
) else (
    echo ❌ SDK目录不存在: %ANDROID_HOME%
    echo 请通过Android Studio安装SDK
    pause
    exit /b 1
)

echo.
echo 4. 检查ADB...
adb version >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ ADB不可用，请安装Android SDK Platform-Tools
) else (
    echo ✅ ADB可用
)

echo.
echo 5. 检查必要组件...
if exist "%ANDROID_HOME%\platforms\android-34" (
    echo ✅ Android API 34 已安装
) else (
    echo ❌ Android API 34 未安装
)

if exist "%ANDROID_HOME%\build-tools" (
    echo ✅ Build Tools 已安装
) else (
    echo ❌ Build Tools 未安装
)

echo.
echo ========================================
echo 修复建议:
echo 1. 打开Android Studio
echo 2. 点击 "Open SDK Manager"
echo 3. 安装以下组件:
echo    - Android SDK Platform 34
echo    - Android SDK Build-Tools
echo    - Android SDK Platform-Tools
echo    - Android Emulator (可选)
echo 4. 点击 "Apply" 开始安装
echo ========================================

echo.
echo 是否现在打开Android Studio? (Y/N)
set /p choice=
if /i "%choice%"=="Y" (
    start "" "studio64.exe"
)

pause

