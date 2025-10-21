#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境检测模型训练脚本
使用YOLOv8训练环境检测模型
"""

import os
import sys
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns

class EnvironmentModelTrainer:
    """环境检测模型训练类"""
    
    def __init__(self):
        self.dataset_path = "data/yolo_environment_dataset"
        self.model_path = "models/environment_detector.pt"
        self.results_dir = "results/environment_training"
        
        # 创建结果目录
        os.makedirs(self.results_dir, exist_ok=True)
        
    def train_model(self, epochs=100, batch_size=16, img_size=640):
        """训练环境检测模型"""
        print("🚀 开始训练环境检测模型...")
        
        # 检查数据集
        if not self.check_dataset():
            return False
            
        # 加载预训练模型
        model = YOLO('yolov8n.pt')  # 使用YOLOv8n作为基础模型
        
        # 训练参数
        train_args = {
            'data': os.path.join(self.dataset_path, 'dataset.yaml'),
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'project': self.results_dir,
            'name': 'environment_detection',
            'save': True,
            'save_period': 10,
            'patience': 20,
            'lr0': 0.01,
            'lrf': 0.01,
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
            'source': None,
            'vid_stride': 1,
            'stream_buffer': False,
            'visualize': False,
            'augment': True,
            'agnostic_nms': False,
            'classes': None,
            'retina_masks': False,
            'format': 'torchscript',
            'keras': False,
            'optimize': False,
            'int8': False,
            'dynamic': False,
            'simplify': False,
            'opset': None,
            'workspace': 4,
            'nms': False,
            'lr_scheduler': 'auto',
            'cos_lr': False,
            'close_mosaic': 10,
            'resume': False,
            'amp': True,
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'multi_scale': False,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'split': 'val',
            'save_json': False,
            'save_hybrid': False,
            'conf': None,
            'iou': 0.7,
            'max_det': 300,
            'half': False,
            'dnn': False,
            'vid_stride': 1,
            'stream_buffer': False,
            'visualize': False,
            'augment': False,
            'agnostic_nms': False,
            'classes': None,
            'retina_masks': False,
            'boxes': True,
            'format': 'torchscript',
            'keras': False,
            'optimize': False,
            'int8': False,
            'dynamic': False,
            'simplify': False,
            'opset': None,
            'workspace': 4,
            'nms': False,
            'lr_scheduler': 'auto',
            'cos_lr': False,
            'close_mosaic': 10,
            'resume': False,
            'amp': True,
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'multi_scale': False,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'split': 'val',
            'save_json': False,
            'save_hybrid': False,
            'conf': None,
            'iou': 0.7,
            'max_det': 300,
            'half': False,
            'dnn': False,
            'vid_stride': 1,
            'stream_buffer': False,
            'visualize': False,
            'augment': False,
            'agnostic_nms': False,
            'classes': None,
            'retina_masks': False,
            'boxes': True
        }
        
        try:
            # 开始训练
            results = model.train(**train_args)
            
            # 保存模型
            model.save(self.model_path)
            print(f"✅ 模型已保存到: {self.model_path}")
            
            # 生成训练报告
            self.generate_training_report(results)
            
            return True
            
        except Exception as e:
            print(f"❌ 训练失败: {e}")
            return False
            
    def check_dataset(self):
        """检查数据集"""
        dataset_yaml = os.path.join(self.dataset_path, 'dataset.yaml')
        
        if not os.path.exists(dataset_yaml):
            print(f"❌ 数据集配置文件不存在: {dataset_yaml}")
            return False
            
        # 检查图像和标注文件
        images_dir = os.path.join(self.dataset_path, 'images')
        labels_dir = os.path.join(self.dataset_path, 'labels')
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"❌ 图像或标注目录不存在")
            return False
            
        # 统计文件数量
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        
        print(f"📊 数据集统计:")
        print(f"   图像文件: {len(image_files)}")
        print(f"   标注文件: {len(label_files)}")
        
        if len(image_files) == 0:
            print("❌ 未找到图像文件")
            return False
            
        if len(label_files) == 0:
            print("❌ 未找到标注文件")
            return False
            
        return True
        
    def generate_training_report(self, results):
        """生成训练报告"""
        report_path = os.path.join(self.results_dir, 'training_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("环境检测模型训练报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"训练完成时间: {results.timestamp}\n")
            f.write(f"训练轮数: {results.epochs}\n")
            f.write(f"批次大小: {results.batch_size}\n")
            f.write(f"图像尺寸: {results.imgsz}\n")
            f.write(f"设备: {results.device}\n\n")
            
            f.write("训练结果:\n")
            f.write("-" * 30 + "\n")
            f.write(f"最佳mAP50: {results.best_fitness:.4f}\n")
            f.write(f"最终mAP50: {results.fitness:.4f}\n")
            f.write(f"最佳epoch: {results.best_epoch}\n\n")
            
            f.write("模型性能指标:\n")
            f.write("-" * 30 + "\n")
            if hasattr(results, 'metrics'):
                for metric, value in results.metrics.items():
                    f.write(f"{metric}: {value:.4f}\n")
                    
        print(f"📊 训练报告已生成: {report_path}")
        
    def validate_model(self, model_path=None):
        """验证模型"""
        if model_path is None:
            model_path = self.model_path
            
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return False
            
        print("🔍 开始验证模型...")
        
        try:
            # 加载模型
            model = YOLO(model_path)
            
            # 验证模型
            results = model.val(data=os.path.join(self.dataset_path, 'dataset.yaml'))
            
            print("✅ 模型验证完成")
            print(f"mAP50: {results.fitness:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型验证失败: {e}")
            return False
            
    def test_model(self, image_path, model_path=None):
        """测试模型"""
        if model_path is None:
            model_path = self.model_path
            
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return False
            
        if not os.path.exists(image_path):
            print(f"❌ 图像文件不存在: {image_path}")
            return False
            
        print(f"🔍 测试模型: {image_path}")
        
        try:
            # 加载模型
            model = YOLO(model_path)
            
            # 进行预测
            results = model(image_path)
            
            # 显示结果
            for result in results:
                print(f"检测到 {len(result.boxes)} 个对象")
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    print(f"  类别: {class_id}, 置信度: {confidence:.3f}")
                    
            return True
            
        except Exception as e:
            print(f"❌ 模型测试失败: {e}")
            return False

def main():
    """主函数"""
    trainer = EnvironmentModelTrainer()
    
    print("🎯 环境检测模型训练工具")
    print("=" * 50)
    
    # 检查数据集
    if not trainer.check_dataset():
        print("❌ 数据集检查失败，请先准备训练数据")
        return
        
    # 训练模型
    success = trainer.train_model(epochs=50, batch_size=8, img_size=640)
    
    if success:
        print("✅ 模型训练完成")
        
        # 验证模型
        trainer.validate_model()
        
        # 测试模型（如果有测试图像）
        test_image = "data/environment_images/test.jpg"
        if os.path.exists(test_image):
            trainer.test_model(test_image)
    else:
        print("❌ 模型训练失败")

if __name__ == "__main__":
    main()







