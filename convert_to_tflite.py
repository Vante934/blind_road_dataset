#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将YOLOv8 PyTorch模型转换为TensorFlow Lite格式
用于Android部署
"""

import torch
import os
from ultralytics import YOLO

def convert_yolo_to_tflite():
    """将YOLOv8模型转换为TensorFlow Lite格式"""
    
    # 模型路径
    pt_model_path = "models/yolo11n.pt"
    tflite_model_path = "app/src/main/assets/yolov8n.tflite"
    
    print("🔄 开始转换YOLOv8模型到TensorFlow Lite格式...")
    
    try:
        # 加载YOLOv8模型
        print("📥 加载YOLOv8模型...")
        model = YOLO(pt_model_path)
        
        # 导出为TensorFlow Lite格式
        print("🔄 导出为TensorFlow Lite格式...")
        model.export(
            format='tflite',
            imgsz=640,
            optimize=True,
            int8=False,  # 使用FP32精度，确保兼容性
            dynamic=False,
            simplify=True,
            opset=None,
            workspace=4,
            nms=True
        )
        
        # 查找生成的tflite文件
        tflite_files = []
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith('.tflite'):
                    tflite_files.append(os.path.join(root, file))
        
        if tflite_files:
            # 使用最新生成的tflite文件
            latest_tflite = max(tflite_files, key=os.path.getctime)
            print(f"📁 找到生成的TFLite文件: {latest_tflite}")
            
            # 复制到Android assets目录
            import shutil
            shutil.copy2(latest_tflite, tflite_model_path)
            print(f"✅ 模型已复制到: {tflite_model_path}")
            
            # 检查文件大小
            file_size = os.path.getsize(tflite_model_path)
            print(f"📊 模型文件大小: {file_size / (1024*1024):.2f} MB")
            
        else:
            print("❌ 未找到生成的TFLite文件")
            return False
            
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return False
    
    print("✅ 模型转换完成！")
    return True

if __name__ == "__main__":
    success = convert_yolo_to_tflite()
    if success:
        print("\n🎉 YOLOv8模型已成功转换为TensorFlow Lite格式并复制到Android项目！")
        print("📱 现在可以在Android应用中使用该模型进行检测了。")
    else:
        print("\n❌ 模型转换失败，请检查错误信息。")



