#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境检测训练数据准备脚本
将标注数据转换为YOLO格式，准备训练数据
"""

import os
import json
import shutil
import cv2
import numpy as np
from pathlib import Path

class EnvironmentTrainingDataPrep:
    """环境检测训练数据准备类"""
    
    def __init__(self):
        self.annotation_dir = "data/environment_annotations"
        self.image_dir = "data/environment_images"
        self.output_dir = "data/yolo_environment_dataset"
        self.classes_file = "data/yolo_environment_dataset/classes.txt"
        
        # 环境类别定义
        self.environment_classes = {
            0: "晴天", 1: "雨天", 2: "雪天", 3: "雾天",
            4: "明亮", 5: "正常", 6: "昏暗", 7: "黑暗",
            8: "平整", 9: "湿滑", 10: "结冰", 11: "坑洼",
            12: "施工", 13: "盲道", 14: "人行道", 15: "路口",
            16: "施工区", 17: "停车场", 18: "护栏", 19: "警示牌",
            20: "红绿灯", 21: "斑马线", 22: "无障碍设施", 23: "其他"
        }
        
    def prepare_training_data(self):
        """准备训练数据"""
        print("🚀 开始准备环境检测训练数据...")
        
        # 创建输出目录
        self.create_output_directories()
        
        # 创建类别文件
        self.create_classes_file()
        
        # 转换标注数据
        self.convert_annotations()
        
        # 创建数据集配置文件
        self.create_dataset_config()
        
        print("✅ 训练数据准备完成！")
        print(f"📁 输出目录: {self.output_dir}")
        print(f"📋 类别文件: {self.classes_file}")
        
    def create_output_directories(self):
        """创建输出目录"""
        directories = [
            self.output_dir,
            os.path.join(self.output_dir, "images"),
            os.path.join(self.output_dir, "labels"),
            os.path.join(self.output_dir, "train"),
            os.path.join(self.output_dir, "val"),
            os.path.join(self.output_dir, "test")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"📁 创建目录: {directory}")
            
    def create_classes_file(self):
        """创建类别文件"""
        with open(self.classes_file, 'w', encoding='utf-8') as f:
            for class_id, class_name in self.environment_classes.items():
                f.write(f"{class_name}\n")
        print(f"📋 创建类别文件: {self.classes_file}")
        
    def convert_annotations(self):
        """转换标注数据"""
        if not os.path.exists(self.annotation_dir):
            print(f"❌ 标注目录不存在: {self.annotation_dir}")
            return
            
        annotation_files = [f for f in os.listdir(self.annotation_dir) if f.endswith('.json')]
        
        if not annotation_files:
            print(f"❌ 未找到标注文件: {self.annotation_dir}")
            return
            
        print(f"📊 找到 {len(annotation_files)} 个标注文件")
        
        for annotation_file in annotation_files:
            self.convert_single_annotation(annotation_file)
            
    def convert_single_annotation(self, annotation_file):
        """转换单个标注文件"""
        annotation_path = os.path.join(self.annotation_dir, annotation_file)
        
        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                annotation_data = json.load(f)
                
            image_path = annotation_data.get('image_path', '')
            annotations = annotation_data.get('annotations', [])
            
            if not os.path.exists(image_path):
                print(f"⚠️ 图像文件不存在: {image_path}")
                return
                
            # 获取图像尺寸
            image = cv2.imread(image_path)
            if image is None:
                print(f"⚠️ 无法读取图像: {image_path}")
                return
                
            height, width = image.shape[:2]
            
            # 复制图像到输出目录
            image_filename = os.path.basename(image_path)
            output_image_path = os.path.join(self.output_dir, "images", image_filename)
            shutil.copy2(image_path, output_image_path)
            
            # 创建YOLO格式的标注文件
            label_filename = os.path.splitext(image_filename)[0] + '.txt'
            label_path = os.path.join(self.output_dir, "labels", label_filename)
            
            with open(label_path, 'w', encoding='utf-8') as f:
                for annotation in annotations:
                    if 'bbox' in annotation and 'class_id' in annotation:
                        x1, y1, x2, y2 = annotation['bbox']
                        class_id = annotation['class_id']
                        
                        # 转换为YOLO格式 (归一化坐标)
                        x_center = (x1 + x2) / 2 / width
                        y_center = (y1 + y2) / 2 / height
                        bbox_width = (x2 - x1) / width
                        bbox_height = (y2 - y1) / height
                        
                        # 确保坐标在有效范围内
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        bbox_width = max(0, min(1, bbox_width))
                        bbox_height = max(0, min(1, bbox_height))
                        
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
                        
            print(f"✅ 转换完成: {image_filename}")
            
        except Exception as e:
            print(f"❌ 转换失败 {annotation_file}: {e}")
            
    def create_dataset_config(self):
        """创建数据集配置文件"""
        config = {
            "path": os.path.abspath(self.output_dir),
            "train": "images",
            "val": "images",
            "test": "images",
            "nc": len(self.environment_classes),
            "names": list(self.environment_classes.values())
        }
        
        config_path = os.path.join(self.output_dir, "dataset.yaml")
        with open(config_path, 'w', encoding='utf-8') as f:
            import yaml
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
        print(f"📋 创建数据集配置: {config_path}")
        
    def split_dataset(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """分割数据集"""
        if not os.path.exists(os.path.join(self.output_dir, "images")):
            print("❌ 图像目录不存在，请先运行数据转换")
            return
            
        image_files = [f for f in os.listdir(os.path.join(self.output_dir, "images")) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not image_files:
            print("❌ 未找到图像文件")
            return
            
        # 随机打乱
        import random
        random.shuffle(image_files)
        
        # 计算分割点
        total = len(image_files)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        
        # 分割文件
        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]
        
        # 复制文件到对应目录
        self.copy_files_to_split(train_files, "train")
        self.copy_files_to_split(val_files, "val")
        self.copy_files_to_split(test_files, "test")
        
        print(f"📊 数据集分割完成:")
        print(f"   训练集: {len(train_files)} 张图像")
        print(f"   验证集: {len(val_files)} 张图像")
        print(f"   测试集: {len(test_files)} 张图像")
        
    def copy_files_to_split(self, files, split_name):
        """复制文件到分割目录"""
        split_dir = os.path.join(self.output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        for filename in files:
            # 复制图像
            src_image = os.path.join(self.output_dir, "images", filename)
            dst_image = os.path.join(split_dir, filename)
            shutil.copy2(src_image, dst_image)
            
            # 复制标注
            label_filename = os.path.splitext(filename)[0] + '.txt'
            src_label = os.path.join(self.output_dir, "labels", label_filename)
            dst_label = os.path.join(split_dir, label_filename)
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
                
    def generate_statistics(self):
        """生成数据集统计信息"""
        if not os.path.exists(os.path.join(self.output_dir, "labels")):
            print("❌ 标注目录不存在")
            return
            
        label_files = [f for f in os.listdir(os.path.join(self.output_dir, "labels")) 
                      if f.endswith('.txt')]
        
        if not label_files:
            print("❌ 未找到标注文件")
            return
            
        # 统计每个类别的数量
        class_counts = {class_id: 0 for class_id in self.environment_classes.keys()}
        total_annotations = 0
        
        for label_file in label_files:
            label_path = os.path.join(self.output_dir, "labels", label_file)
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.strip().split()[0])
                        if class_id in class_counts:
                            class_counts[class_id] += 1
                            total_annotations += 1
                            
        # 生成统计报告
        stats_path = os.path.join(self.output_dir, "dataset_statistics.txt")
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("环境检测数据集统计报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"总图像数: {len(label_files)}\n")
            f.write(f"总标注数: {total_annotations}\n")
            f.write(f"平均每张图像标注数: {total_annotations/len(label_files):.2f}\n\n")
            
            f.write("各类别标注数量:\n")
            f.write("-" * 30 + "\n")
            for class_id, count in class_counts.items():
                class_name = self.environment_classes[class_id]
                percentage = count / total_annotations * 100 if total_annotations > 0 else 0
                f.write(f"{class_id:2d}. {class_name:10s}: {count:4d} ({percentage:5.1f}%)\n")
                
        print(f"📊 统计报告已生成: {stats_path}")

def main():
    """主函数"""
    prep = EnvironmentTrainingDataPrep()
    
    print("🎯 环境检测训练数据准备工具")
    print("=" * 50)
    
    # 准备训练数据
    prep.prepare_training_data()
    
    # 分割数据集
    prep.split_dataset()
    
    # 生成统计信息
    prep.generate_statistics()
    
    print("\n🎉 所有任务完成！")

if __name__ == "__main__":
    main()







