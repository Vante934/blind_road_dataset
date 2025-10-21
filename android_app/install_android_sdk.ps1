# Android SDK 手动安装脚本
# 请以管理员身份运行PowerShell

Write-Host "开始安装Android SDK..." -ForegroundColor Green

# 设置SDK安装路径
$SDK_PATH = "D:\SoftWare\AndroidStudio\sdk"
$SDK_TOOLS_URL = "https://dl.google.com/android/repository/commandlinetools-win-11076708_latest.zip"

# 创建SDK目录
if (!(Test-Path $SDK_PATH)) {
    New-Item -ItemType Directory -Path $SDK_PATH -Force
    Write-Host "创建SDK目录: $SDK_PATH" -ForegroundColor Yellow
}

# 下载SDK命令行工具
$toolsZip = "$env:TEMP\android-commandlinetools.zip"
Write-Host "下载Android SDK命令行工具..." -ForegroundColor Yellow

try {
    Invoke-WebRequest -Uri $SDK_TOOLS_URL -OutFile $toolsZip
    Write-Host "下载完成" -ForegroundColor Green
} catch {
    Write-Host "下载失败: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# 解压到SDK目录
Write-Host "解压SDK工具..." -ForegroundColor Yellow
Expand-Archive -Path $toolsZip -DestinationPath "$SDK_PATH\temp" -Force

# 移动文件到正确位置
$cmdlineToolsPath = "$SDK_PATH\cmdline-tools\latest"
if (!(Test-Path $cmdlineToolsPath)) {
    New-Item -ItemType Directory -Path $cmdlineToolsPath -Force
}

Move-Item -Path "$SDK_PATH\temp\cmdline-tools\*" -Destination $cmdlineToolsPath -Force
Remove-Item -Path "$SDK_PATH\temp" -Recurse -Force

# 设置环境变量
Write-Host "设置环境变量..." -ForegroundColor Yellow
$env:ANDROID_HOME = $SDK_PATH
$env:PATH += ";$SDK_PATH\cmdline-tools\latest\bin;$SDK_PATH\platform-tools"

# 安装SDK组件
Write-Host "安装Android SDK组件..." -ForegroundColor Yellow
& "$SDK_PATH\cmdline-tools\latest\bin\sdkmanager.bat" --sdk_root=$SDK_PATH "platform-tools" "platforms;android-34" "build-tools;34.0.0"

Write-Host "Android SDK安装完成！" -ForegroundColor Green
Write-Host "SDK路径: $SDK_PATH" -ForegroundColor Cyan
Write-Host "请重启Android Studio以识别新的SDK路径" -ForegroundColor Yellow

# 清理临时文件
Remove-Item -Path $toolsZip -Force











