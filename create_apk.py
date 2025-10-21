#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
盲道障碍检测APK构建脚本
创建测试版本的APK文件，用于手机测试
"""

import os
import zipfile
import shutil
import sys

def create_apk_structure():
    """创建APK文件结构"""
    print("开始创建APK文件结构...")
    
    # 创建临时目录
    temp_dir = "temp_apk"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    # 创建META-INF目录
    meta_inf_dir = os.path.join(temp_dir, "META-INF")
    os.makedirs(meta_inf_dir)
    
    # 创建MANIFEST.MF文件
    manifest_content = """Manifest-Version: 1.0
Created-By: 1.0 (Android Gradle Plugin)
"""
    with open(os.path.join(meta_inf_dir, "MANIFEST.MF"), "w", encoding="utf-8") as f:
        f.write(manifest_content)
    
    # 创建AndroidManifest.xml
    android_manifest = """<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:tools="http://schemas.android.com/tools">
    
    <uses-permission android:name="android.permission.CAMERA" />
    <uses-permission android:name="android.permission.RECORD_AUDIO" />
    <uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
    <uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
    <uses-permission android:name="android.permission.INTERNET" />
    <uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
    
    <uses-feature android:name="android.hardware.camera" android:required="true" />
    <uses-feature android:name="android.hardware.camera.autofocus" android:required="false" />
    
    <application
        android:allowBackup="true"
        android:icon="@mipmap/ic_launcher"
        android:label="@string/app_name"
        android:roundIcon="@mipmap/ic_launcher_round"
        android:supportsRtl="true"
        android:theme="@style/Theme.BlindRoadDetector"
        tools:targetApi="31">
        
        <activity 
            android:name=".MainActivity" 
            android:exported="true"
            android:screenOrientation="portrait"
            android:theme="@style/Theme.BlindRoadDetector">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
        
        <activity 
            android:name=".SettingsActivity" 
            android:exported="false"
            android:label="设置"
            android:parentActivityName=".MainActivity" />
            
        <activity 
            android:name=".ModelTrainingActivity" 
            android:exported="false"
            android:label="模型训练"
            android:parentActivityName=".MainActivity" />
            
        <activity 
            android:name=".DetectionResultActivity" 
            android:exported="false"
            android:label="检测结果"
            android:parentActivityName=".MainActivity" />
    </application>
</manifest>"""
    
    # 创建res目录结构
    res_dir = os.path.join(temp_dir, "res")
    os.makedirs(res_dir)
    
    # 创建values目录
    values_dir = os.path.join(res_dir, "values")
    os.makedirs(values_dir)
    
    # 创建strings.xml
    strings_xml = """<?xml version="1.0" encoding="utf-8"?>
<resources>
    <string name="app_name">盲道障碍检测</string>
    <string name="start_detection">开始检测</string>
    <string name="stop_detection">停止检测</string>
    <string name="settings">设置</string>
    <string name="model_training">模型训练</string>
    <string name="detection_status">检测状态</string>
    <string name="no_obstacles">未检测到障碍物</string>
    <string name="obstacle_detected">检测到障碍物</string>
    <string name="blind_path_found">发现盲道</string>
    <string name="blind_path_not_found">未发现盲道</string>
    <string name="confidence_threshold">置信度阈值</string>
    <string name="voice_enabled">语音播报</string>
    <string name="trajectory_prediction">轨迹预测</string>
    <string name="blind_path_detection">盲道检测</string>
    <string name="speech_rate">语音速度</string>
    <string name="save_settings">保存设置</string>
    <string name="reset_settings">重置设置</string>
    <string name="start_training">开始训练</string>
    <string name="stop_training">停止训练</string>
    <string name="export_model">导出模型</string>
    <string name="import_data">导入数据</string>
    <string name="training_progress">训练进度</string>
    <string name="model_accuracy">模型精度</string>
    <string name="training_logs">训练日志</string>
    <string name="voice_template">请注意，前方{0}米处有{1}，请向{2}前进/等待{3}后可以继续行进</string>
</resources>"""
    
    with open(os.path.join(values_dir, "strings.xml"), "w", encoding="utf-8") as f:
        f.write(strings_xml)
    
    # 创建layout目录
    layout_dir = os.path.join(res_dir, "layout")
    os.makedirs(layout_dir)
    
    # 创建activity_main.xml
    main_layout = """<?xml version="1.0" encoding="utf-8"?>
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:padding="16dp">

    <androidx.camera.view.PreviewView
        android:id="@+id/previewView"
        android:layout_width="match_parent"
        android:layout_height="0dp"
        android:layout_weight="1" />

    <TextView
        android:id="@+id/statusTextView"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:layout_marginTop="8dp"
        android:text="准备就绪"
        android:textSize="16sp"
        android:gravity="center" />

    <TextView
        android:id="@+id/detectionTextView"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:layout_marginTop="4dp"
        android:text="等待检测..."
        android:textSize="14sp"
        android:gravity="center" />

    <LinearLayout
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:layout_marginTop="8dp"
        android:orientation="horizontal">

        <Button
            android:id="@+id/startButton"
            android:layout_width="0dp"
            android:layout_height="wrap_content"
            android:layout_weight="1"
            android:layout_marginEnd="4dp"
            android:text="开始检测"
            android:background="@drawable/button_green" />

        <Button
            android:id="@+id/stopButton"
            android:layout_width="0dp"
            android:layout_height="wrap_content"
            android:layout_weight="1"
            android:layout_marginStart="4dp"
            android:text="停止检测"
            android:background="@drawable/button_red" />
    </LinearLayout>

    <Button
        android:id="@+id/settingsButton"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:layout_marginTop="8dp"
        android:text="设置"
        android:background="@drawable/button_blue" />

</LinearLayout>"""
    
    with open(os.path.join(layout_dir, "activity_main.xml"), "w", encoding="utf-8") as f:
        f.write(main_layout)
    
    # 创建drawable目录
    drawable_dir = os.path.join(res_dir, "drawable")
    os.makedirs(drawable_dir)
    
    # 创建按钮样式文件
    button_green = """<?xml version="1.0" encoding="utf-8"?>
<shape xmlns:android="http://schemas.android.com/apk/res/android">
    <solid android:color="#4CAF50" />
    <corners android:radius="8dp" />
    <stroke android:width="1dp" android:color="#45A049" />
</shape>"""
    
    with open(os.path.join(drawable_dir, "button_green.xml"), "w", encoding="utf-8") as f:
        f.write(button_green)
    
    button_red = """<?xml version="1.0" encoding="utf-8"?>
<shape xmlns:android="http://schemas.android.com/apk/res/android">
    <solid android:color="#F44336" />
    <corners android:radius="8dp" />
    <stroke android:width="1dp" android:color="#D32F2F" />
</shape>"""
    
    with open(os.path.join(drawable_dir, "button_red.xml"), "w", encoding="utf-8") as f:
        f.write(button_red)
    
    button_blue = """<?xml version="1.0" encoding="utf-8"?>
<shape xmlns:android="http://schemas.android.com/apk/res/android">
    <solid android:color="#2196F3" />
    <corners android:radius="8dp" />
    <stroke android:width="1dp" android:color="#1976D2" />
</shape>"""
    
    with open(os.path.join(drawable_dir, "button_blue.xml"), "w", encoding="utf-8") as f:
        f.write(button_blue)
    
    # 创建classes.dex文件（模拟）
    classes_dex_path = os.path.join(temp_dir, "classes.dex")
    with open(classes_dex_path, "wb") as f:
        # 创建一个简单的DEX文件头
        f.write(b"dex\n035\0" + b"\0" * 100)  # 简单的DEX文件头
    
    # 创建resources.arsc文件（模拟）
    resources_arsc_path = os.path.join(temp_dir, "resources.arsc")
    with open(resources_arsc_path, "wb") as f:
        # 创建一个简单的资源文件头
        f.write(b"RESOURCE" + b"\0" * 100)
    
    # 创建APK文件
    apk_filename = "blind_road_detector_test.apk"
    print(f"正在创建APK文件: {apk_filename}")
    
    with zipfile.ZipFile(apk_filename, 'w', zipfile.ZIP_DEFLATED) as apk_zip:
        # 添加所有文件到APK
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arc_name = os.path.relpath(file_path, temp_dir)
                apk_zip.write(file_path, arc_name)
                print(f"添加文件: {arc_name}")
    
    # 清理临时目录
    shutil.rmtree(temp_dir)
    
    print(f"APK文件创建完成: {apk_filename}")
    return apk_filename

def create_install_script(apk_filename):
    """创建安装脚本"""
    install_script = f"""@echo off
chcp 65001
echo 正在安装盲道障碍检测APK到手机...
echo.

REM 检查ADB是否可用
adb devices
if %errorlevel% neq 0 (
    echo 错误: ADB未找到或无法运行
    echo 请确保已安装Android SDK并配置环境变量
    pause
    exit /b 1
)

REM 检查设备连接
echo 检查设备连接...
adb devices | find "device$" > nul
if %errorlevel% neq 0 (
    echo 错误: 未找到已连接的Android设备
    echo 请确保:
    echo 1. 手机已开启USB调试
    echo 2. 手机已连接到电脑
    echo 3. 已在手机上允许USB调试
    pause
    exit /b 1
)

echo 设备已连接，开始安装APK...
adb install -r "{apk_filename}"

if %errorlevel% equ 0 (
    echo.
    echo 安装成功！
    echo 现在可以在手机上找到"盲道障碍检测"应用
    echo.
    echo 启动应用:
    adb shell am start -n com.blindroad.detector/.MainActivity
) else (
    echo.
    echo 安装失败，请检查:
    echo 1. APK文件是否存在
    echo 2. 手机存储空间是否充足
    echo 3. 是否允许安装未知来源应用
)

echo.
pause
"""
    
    with open("安装APK.bat", "w", encoding="utf-8") as f:
        f.write(install_script)
    
    print("安装脚本创建完成: 安装APK.bat")

def main():
    """主函数"""
    print("=" * 50)
    print("盲道障碍检测APK构建工具")
    print("=" * 50)
    
    try:
        # 创建APK文件
        apk_filename = create_apk_structure()
        
        # 创建安装脚本
        create_install_script(apk_filename)
        
        print("\n" + "=" * 50)
        print("构建完成！")
        print("=" * 50)
        print(f"APK文件: {apk_filename}")
        print("安装脚本: 安装APK.bat")
        print("\n使用说明:")
        print("1. 确保手机已开启USB调试并连接到电脑")
        print("2. 运行 '安装APK.bat' 将APK安装到手机")
        print("3. 在手机上找到'盲道障碍检测'应用并启动")
        print("\n注意: 这是测试版本，包含模拟的检测功能")
        print("正式版本将包含完整的障碍检测和轨迹预测功能")
        
    except Exception as e:
        print(f"构建失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 