#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
盲道标注数据分析工具
功能：
1. 统计标注数据
2. 转换为YOLO格式
3. 准备训练数据集
"""

import os
import json
import glob
import cv2
import numpy as np
from pathlib import Path

class BlindPathDataAnalyzer:
    def __init__(self):
        self.annotations_dir = "annotations"
        self.images_dir = "images"
        self.output_dir = "yolo_dataset"
        
    def analyze_annotations(self):
        """分析所有标注数据"""
        print("=== 盲道标注数据分析 ===")
        
        # 获取所有标注文件
        annotation_files = glob.glob(os.path.join(self.annotations_dir, "*_annotations.json"))
        
        if not annotation_files:
            print("❌ 没有找到标注文件")
            return
            
        print(f"📁 找到 {len(annotation_files)} 个标注文件")
        
        total_lines = 0
        total_images = 0
        lines_per_image = []
        
        for ann_file in annotation_files:
            try:
                with open(ann_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                annotations = data.get('annotations', [])
                lines_count = len([ann for ann in annotations if ann['type'] == 'blind_path_line'])
                
                total_lines += lines_count
                total_images += 1
                lines_per_image.append(lines_count)
                
                print(f"  📄 {os.path.basename(ann_file)}: {lines_count} 条线段")
                
            except Exception as e:
                print(f"  ❌ 读取 {ann_file} 失败: {e}")
        
        print(f"\n📊 统计结果:")
        print(f"  总图像数: {total_images}")
        print(f"  总线段数: {total_lines}")
        print(f"  平均每张图像线段数: {total_lines/total_images:.1f}")
        print(f"  最多线段数: {max(lines_per_image)}")
        print(f"  最少线段数: {min(lines_per_image)}")
        
        return total_images, total_lines, lines_per_image
        
    def convert_to_yolo_format(self):
        """转换为YOLO格式"""
        print("\n=== 转换为YOLO格式 ===")
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "labels"), exist_ok=True)
        
        # 获取所有标注文件
        annotation_files = glob.glob(os.path.join(self.annotations_dir, "*_annotations.json"))
        
        converted_count = 0
        
        for ann_file in annotation_files:
            try:
                with open(ann_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                image_path = data.get('image_path', '')
                if not image_path or not os.path.exists(image_path):
                    print(f"  ❌ 图像文件不存在: {image_path}")
                    continue
                
                # 读取图像获取尺寸
                image = cv2.imread(image_path)
                if image is None:
                    print(f"  ❌ 无法读取图像: {image_path}")
                    continue
                
                height, width = image.shape[:2]
                
                # 生成YOLO标签文件
                image_name = os.path.basename(image_path)
                label_name = image_name.replace('.jpg', '.txt').replace('.png', '.txt')
                label_path = os.path.join(self.output_dir, "labels", label_name)
                
                yolo_lines = []
                annotations = data.get('annotations', [])
                
                for ann in annotations:
                    if ann['type'] == 'blind_path_line':
                        start = ann['start']
                        end = ann['end']
                        
                        # 计算线段中点作为目标点
                        center_x = (start[0] + end[0]) / 2
                        center_y = (start[1] + end[1]) / 2
                        
                        # 计算线段长度作为目标大小
                        length = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                        
                        # 转换为YOLO格式 (x_center, y_center, width, height) 归一化
                        x_center = center_x / width
                        y_center = center_y / height
                        w = min(length / width, 1.0)  # 宽度归一化
                        h = min(length / height, 1.0)  # 高度归一化
                        
                        # 类别ID: 0 = 盲道线段
                        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
                
                # 保存YOLO标签文件
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))
                
                # 复制图像文件
                output_image_path = os.path.join(self.output_dir, "images", image_name)
                cv2.imwrite(output_image_path, image)
                
                converted_count += 1
                print(f"  ✅ {image_name}: {len(yolo_lines)} 个目标")
                
            except Exception as e:
                print(f"  ❌ 转换 {ann_file} 失败: {e}")
        
        print(f"\n✅ 转换完成: {converted_count} 个文件")
        return converted_count
        
    def create_dataset_config(self):
        """创建数据集配置文件"""
        print("\n=== 创建数据集配置 ===")
        
        # 创建YOLO数据集配置文件
        config_content = """# YOLO盲道检测数据集配置
path: ./yolo_dataset  # 数据集根目录
train: images  # 训练图像目录
val: images    # 验证图像目录

# 类别数量和名称
nc: 1  # 类别数量
names: ['blind_path']  # 类别名称

# 数据集信息
# 盲道检测数据集
# 包含盲道线段标注
"""
        
        config_path = os.path.join(self.output_dir, "dataset.yaml")
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print(f"✅ 配置文件已创建: {config_path}")
        
    def create_training_script(self):
        """创建训练脚本"""
        print("\n=== 创建训练脚本 ===")
        
        script_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
盲道检测YOLOv8训练脚本
"""

from ultralytics import YOLO
import os

def train_blind_path_detector():
    """训练盲道检测模型"""
    
    # 加载预训练模型
    model = YOLO('yolov8n.pt')  # 使用YOLOv8 nano模型
    
    # 训练参数
    results = model.train(
        data='yolo_dataset/dataset.yaml',  # 数据集配置文件
        epochs=100,                        # 训练轮数
        imgsz=640,                         # 图像尺寸
        batch=16,                          # 批次大小
        name='blind_path_detector',        # 实验名称
        patience=20,                       # 早停耐心值
        save=True,                         # 保存模型
        device='auto'                      # 自动选择设备
    )
    
    print("训练完成！")
    return results

if __name__ == "__main__":
    train_blind_path_detector()
'''
        
        script_path = "train_blind_path.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        print(f"✅ 训练脚本已创建: {script_path}")
        
    def run_full_analysis(self):
        """运行完整分析"""
        print("🚀 开始盲道标注数据分析...")
        
        # 1. 分析标注数据
        total_images, total_lines, lines_per_image = self.analyze_annotations()
        
        # 2. 转换为YOLO格式
        converted_count = self.convert_to_yolo_format()
        
        # 3. 创建数据集配置
        self.create_dataset_config()
        
        # 4. 创建训练脚本
        self.create_training_script()
        
        print(f"\n🎉 分析完成！")
        print(f"📊 数据集统计:")
        print(f"  - 图像数量: {total_images}")
        print(f"  - 标注线段: {total_lines}")
        print(f"  - 转换成功: {converted_count}")
        print(f"\n📁 输出目录: {self.output_dir}")
        print(f"📄 训练脚本: train_blind_path.py")
        print(f"\n🚀 下一步: 运行 python train_blind_path.py 开始训练")

def main():
    analyzer = BlindPathDataAnalyzer()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main() 