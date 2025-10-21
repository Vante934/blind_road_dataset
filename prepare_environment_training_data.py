#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境检测训练数据准备脚本
将环境事物标注数据转换为YOLO训练格式
"""

import os
import json
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

class EnvironmentTrainingDataPreparer:
    """环境检测训练数据准备器"""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.class_mapping = {}
        self.stats = {
            'total_images': 0,
            'total_annotations': 0,
            'class_counts': {},
            'conversion_errors': 0
        }
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'images').mkdir(exist_ok=True)
        (self.output_dir / 'labels').mkdir(exist_ok=True)
        
        # 加载类别映射
        self.load_class_mapping()
    
    def load_class_mapping(self):
        """加载类别映射"""
        try:
            with open('environment_annotation_classes.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                classes = data['environment_annotation_classes']
                
                class_id = 0
                for category, items in classes.items():
                    for item_id, info in items.items():
                        self.class_mapping[info['name']] = class_id
                        class_id += 1
                
                print(f"✅ 加载了 {len(self.class_mapping)} 个类别映射")
        except Exception as e:
            print(f"❌ 加载类别映射失败: {e}")
            # 使用默认映射
            default_classes = ['雨滴', '湿润表面', '雾颗粒', '雪块', '阴影区域', '强光点', '暗角', 
                             '裂缝', '坑洞', '台阶', '不平整路面', '施工标志', '安全锥', '施工围栏', 
                             '施工机械', '交通信号灯', '斑马线', '停车标志', '让行标志', '树木', 
                             '街道设施', '电线杆', '自行车']
            self.class_mapping = {name: i for i, name in enumerate(default_classes)}
    
    def process_annotation_file(self, annotation_file: Path) -> bool:
        """处理单个标注文件"""
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 获取图像信息
            image_size = data.get('image_size')
            if not image_size:
                print(f"⚠️ 跳过 {annotation_file}: 缺少图像尺寸信息")
                return False
            
            img_height, img_width = image_size
            
            # 获取标注信息
            annotations = data.get('annotations', [])
            if not annotations:
                print(f"⚠️ 跳过 {annotation_file}: 没有标注数据")
                return False
            
            # 生成输出文件名
            base_name = annotation_file.stem
            image_file = self.input_dir / f"{base_name}.jpg"
            if not image_file.exists():
                image_file = self.input_dir / f"{base_name}.png"
            if not image_file.exists():
                print(f"⚠️ 跳过 {annotation_file}: 找不到对应的图像文件")
                return False
            
            # 复制图像文件
            output_image = self.output_dir / 'images' / f"{base_name}.jpg"
            shutil.copy2(image_file, output_image)
            
            # 转换标注格式
            yolo_annotations = []
            for annotation in annotations:
                class_name = annotation['class_name']
                if class_name not in self.class_mapping:
                    print(f"⚠️ 未知类别: {class_name}")
                    continue
                
                class_id = self.class_mapping[class_name]
                bbox = annotation['bbox']  # [x1, y1, x2, y2] 归一化坐标
                
                # 转换为YOLO格式 (class_id x_center y_center width height)
                x_center = (bbox[0] + bbox[2]) / 2
                y_center = (bbox[1] + bbox[3]) / 2
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                
                yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                
                # 更新统计
                self.stats['class_counts'][class_name] = self.stats['class_counts'].get(class_name, 0) + 1
            
            # 保存YOLO格式标注
            label_file = self.output_dir / 'labels' / f"{base_name}.txt"
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            
            self.stats['total_images'] += 1
            self.stats['total_annotations'] += len(yolo_annotations)
            
            return True
            
        except Exception as e:
            print(f"❌ 处理 {annotation_file} 时出错: {e}")
            self.stats['conversion_errors'] += 1
            return False
    
    def create_dataset_yaml(self):
        """创建数据集配置文件"""
        yaml_content = f"""# 环境检测数据集配置
path: {self.output_dir.absolute()}
train: images
val: images

# 类别定义
nc: {len(self.class_mapping)}
names: {list(self.class_mapping.keys())}

# 类别详细信息
class_info:
"""
        
        # 添加类别详细信息
        for class_name, class_id in self.class_mapping.items():
            yaml_content += f"  {class_id}: {class_name}\n"
        
        # 保存YAML文件
        yaml_file = self.output_dir / 'dataset.yaml'
        with open(yaml_file, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        print(f"✅ 创建数据集配置文件: {yaml_file}")
    
    def create_training_script(self):
        """创建训练脚本"""
        script_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境检测模型训练脚本
"""

from ultralytics import YOLO
import os

def train_environment_detection_model():
    """训练环境检测模型"""
    
    # 加载预训练模型
    model = YOLO('yolov8n.pt')  # 使用YOLOv8 nano版本
    
    # 训练参数
    training_args = {{
        'data': '{self.output_dir.absolute()}/dataset.yaml',
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'device': 'cpu',  # 如果有GPU，改为 'cuda'
        'workers': 4,
        'project': 'environment_detection',
        'name': 'environment_model',
        'save': True,
        'save_period': 10,
        'cache': True,
        'patience': 20,
        'lr0': 0.01,
        'lrf': 0.1,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 2.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'plots': True,
        'verbose': True
    }}
    
    print("🚀 开始训练环境检测模型...")
    print(f"📊 数据集路径: {{training_args['data']}}")
    print(f"🎯 训练轮数: {{training_args['epochs']}}")
    print(f"📏 图像尺寸: {{training_args['imgsz']}}")
    print(f"📦 批次大小: {{training_args['batch']}}")
    
    # 开始训练
    results = model.train(**training_args)
    
    print("✅ 训练完成!")
    print(f"📁 模型保存在: {{results.save_dir}}")
    
    # 验证模型
    print("🔍 验证模型性能...")
    val_results = model.val()
    
    return results, val_results

if __name__ == "__main__":
    train_environment_detection_model()
'''
        
        script_file = self.output_dir / 'train_environment_model.py'
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # 设置执行权限
        os.chmod(script_file, 0o755)
        
        print(f"✅ 创建训练脚本: {script_file}")
    
    def process_all_annotations(self):
        """处理所有标注文件"""
        print(f"🔍 开始处理标注文件...")
        print(f"📁 输入目录: {self.input_dir}")
        print(f"📁 输出目录: {self.output_dir}")
        
        # 查找所有标注文件
        annotation_files = list(self.input_dir.glob("*.json"))
        
        if not annotation_files:
            print("❌ 没有找到标注文件")
            return
        
        print(f"📄 找到 {len(annotation_files)} 个标注文件")
        
        # 处理每个标注文件
        processed_count = 0
        for annotation_file in annotation_files:
            if self.process_annotation_file(annotation_file):
                processed_count += 1
                print(f"✅ 处理完成: {annotation_file.name}")
            else:
                print(f"❌ 处理失败: {annotation_file.name}")
        
        print(f"\n📊 处理完成: {processed_count}/{len(annotation_files)} 个文件")
        
        # 创建配置文件
        self.create_dataset_yaml()
        self.create_training_script()
        
        # 打印统计信息
        self.print_statistics()
    
    def print_statistics(self):
        """打印统计信息"""
        print("\n📊 数据准备统计:")
        print(f"  总图像数: {self.stats['total_images']}")
        print(f"  总标注数: {self.stats['total_annotations']}")
        print(f"  转换错误: {self.stats['conversion_errors']}")
        
        print("\n📈 类别分布:")
        for class_name, count in sorted(self.stats['class_counts'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {count}")
        
        print(f"\n📁 输出目录: {self.output_dir}")
        print(f"  📷 图像目录: {self.output_dir}/images")
        print(f"  🏷️ 标签目录: {self.output_dir}/labels")
        print(f"  ⚙️ 配置文件: {self.output_dir}/dataset.yaml")
        print(f"  🚀 训练脚本: {self.output_dir}/train_environment_model.py")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='环境检测训练数据准备工具')
    parser.add_argument('--input', '-i', required=True, help='输入目录（包含图像和标注文件）')
    parser.add_argument('--output', '-o', required=True, help='输出目录（YOLO格式数据集）')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input):
        print(f"❌ 输入目录不存在: {args.input}")
        return
    
    # 创建数据准备器
    preparer = EnvironmentTrainingDataPreparer(args.input, args.output)
    
    # 处理所有标注
    preparer.process_all_annotations()
    
    print("\n🎉 数据准备完成！")
    print("💡 下一步:")
    print("  1. 检查生成的数据集")
    print("  2. 运行训练脚本: python train_environment_model.py")
    print("  3. 评估模型性能")
    print("  4. 集成到盲道检测系统")

if __name__ == "__main__":
    main()







