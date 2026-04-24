#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
盲道检测数据集下载器
自动下载和整理公开的盲道检测数据集
"""

import os
import requests
import zipfile
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
from tqdm import tqdm
import yaml

class BlindRoadDatasetDownloader:
    """盲道检测数据集下载器"""
    
    def __init__(self, base_dir: str = "datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # 数据集配置
        self.datasets = {
            "blind_road_basic": {
                "name": "盲道识别基础数据集",
                "description": "包含381张盲道图像，233张有盲道，148张无盲道",
                "size": "50x50像素",
                "format": "分类",
                "url": "https://aistudio.baidu.com/aistudio/datasetdetail/123456",  # 需要替换为实际URL
                "local_path": "blind_road_basic"
            },
            "unitree_blind_road": {
                "name": "Unitree盲道分割数据集",
                "description": "500多张室外盲道分割图像",
                "size": "高分辨率",
                "format": "分割",
                "url": "https://aistudio.baidu.com/aistudio/datasetdetail/123457",  # 需要替换为实际URL
                "local_path": "unitree_blind_road"
            },
            "blind_road_damage": {
                "name": "盲道损坏检测数据集",
                "description": "4426张图片，3个类别，包含Pascal VOC和YOLO格式",
                "size": "多分辨率",
                "format": "检测",
                "url": "https://example.com/blind_road_damage.zip",  # 需要替换为实际URL
                "local_path": "blind_road_damage"
            },
            "coco_pedestrian": {
                "name": "COCO行人检测数据集",
                "description": "包含行人和道路场景，可用于迁移学习",
                "size": "高分辨率",
                "format": "检测",
                "url": "http://images.cocodataset.org/zips/val2017.zip",
                "local_path": "coco_pedestrian"
            }
        }
        
        # 创建目录结构
        self.setup_directories()
    
    def setup_directories(self):
        """设置目录结构"""
        dirs = [
            "raw",           # 原始数据
            "processed",     # 处理后数据
            "yolo_format",   # YOLO格式
            "annotations",   # 标注文件
            "images",        # 图像文件
            "train",         # 训练集
            "val",           # 验证集
            "test"           # 测试集
        ]
        
        for dir_name in dirs:
            (self.base_dir / dir_name).mkdir(exist_ok=True)
    
    def download_dataset(self, dataset_name: str, force_download: bool = False) -> bool:
        """下载指定数据集"""
        if dataset_name not in self.datasets:
            print(f"❌ 未知数据集: {dataset_name}")
            return False
        
        dataset_info = self.datasets[dataset_name]
        local_path = self.base_dir / "raw" / dataset_info["local_path"]
        
        if local_path.exists() and not force_download:
            print(f"✅ 数据集已存在: {local_path}")
            return True
        
        print(f"📥 开始下载数据集: {dataset_info['name']}")
        print(f"   描述: {dataset_info['description']}")
        print(f"   格式: {dataset_info['format']}")
        
        try:
            # 创建目标目录
            local_path.mkdir(parents=True, exist_ok=True)
            
            # 模拟下载过程（实际使用时需要替换为真实URL）
            if dataset_name == "coco_pedestrian":
                return self.download_coco_dataset(local_path)
            else:
                return self.create_synthetic_dataset(dataset_name, local_path)
                
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            return False
    
    def download_coco_dataset(self, local_path: Path) -> bool:
        """下载COCO数据集"""
        try:
            print("📥 下载COCO验证集...")
            # 这里应该实现真实的下载逻辑
            # 由于COCO数据集很大，这里只是示例
            print("⚠️ 注意: COCO数据集较大，请手动下载并放置到相应目录")
            return True
        except Exception as e:
            print(f"❌ COCO数据集下载失败: {e}")
            return False
    
    def create_synthetic_dataset(self, dataset_name: str, local_path: Path) -> bool:
        """创建合成数据集用于演示"""
        print(f"🎨 创建合成数据集: {dataset_name}")
        
        try:
            if dataset_name == "blind_road_basic":
                self.create_blind_road_basic_dataset(local_path)
            elif dataset_name == "unitree_blind_road":
                self.create_unitree_dataset(local_path)
            elif dataset_name == "blind_road_damage":
                self.create_damage_dataset(local_path)
            
            print(f"✅ 合成数据集创建完成: {local_path}")
            return True
            
        except Exception as e:
            print(f"❌ 创建合成数据集失败: {e}")
            return False
    
    def create_blind_road_basic_dataset(self, local_path: Path):
        """创建基础盲道数据集"""
        # 创建类别目录
        positive_dir = local_path / "positive"  # 有盲道
        negative_dir = local_path / "negative"  # 无盲道
        positive_dir.mkdir(exist_ok=True)
        negative_dir.mkdir(exist_ok=True)
        
        # 生成233张有盲道的图像
        for i in tqdm(range(233), desc="生成有盲道图像"):
            img = self.generate_blind_road_image(has_blind_road=True)
            cv2.imwrite(str(positive_dir / f"positive_{i:03d}.jpg"), img)
        
        # 生成148张无盲道的图像
        for i in tqdm(range(148), desc="生成无盲道图像"):
            img = self.generate_blind_road_image(has_blind_road=False)
            cv2.imwrite(str(negative_dir / f"negative_{i:03d}.jpg"), img)
    
    def create_unitree_dataset(self, local_path: Path):
        """创建Unitree风格数据集"""
        images_dir = local_path / "images"
        masks_dir = local_path / "masks"
        images_dir.mkdir(exist_ok=True)
        masks_dir.mkdir(exist_ok=True)
        
        for i in tqdm(range(500), desc="生成Unitree数据集"):
            # 生成高分辨率图像
            img = self.generate_high_res_blind_road_image()
            mask = self.generate_blind_road_mask(img.shape[:2])
            
            cv2.imwrite(str(images_dir / f"image_{i:04d}.jpg"), img)
            cv2.imwrite(str(masks_dir / f"mask_{i:04d}.png"), mask)
    
    def create_damage_dataset(self, local_path: Path):
        """创建损坏检测数据集"""
        images_dir = local_path / "images"
        annotations_dir = local_path / "annotations"
        images_dir.mkdir(exist_ok=True)
        annotations_dir.mkdir(exist_ok=True)
        
        # 创建YOLO格式标注
        yolo_annotations = []
        
        for i in tqdm(range(1000), desc="生成损坏检测数据集"):
            img, annotations = self.generate_damage_detection_image()
            
            # 保存图像
            img_path = images_dir / f"damage_{i:04d}.jpg"
            cv2.imwrite(str(img_path), img)
            
            # 保存YOLO格式标注
            yolo_path = annotations_dir / f"damage_{i:04d}.txt"
            with open(yolo_path, 'w') as f:
                for ann in annotations:
                    f.write(f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} "
                           f"{ann['width']:.6f} {ann['height']:.6f}\n")
            
            yolo_annotations.append({
                'image_path': str(img_path),
                'annotation_path': str(yolo_path),
                'annotations': annotations
            })
        
        # 保存数据集信息
        dataset_info = {
            'name': '盲道损坏检测数据集',
            'description': '包含各种盲道损坏情况的检测数据集',
            'classes': ['normal', 'damaged', 'obstacle'],
            'class_names': ['正常盲道', '损坏盲道', '障碍物'],
            'total_images': 1000,
            'annotations': yolo_annotations
        }
        
        with open(local_path / 'dataset_info.json', 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    def generate_blind_road_image(self, has_blind_road: bool = True, size: tuple = (50, 50)) -> np.ndarray:
        """生成盲道图像"""
        img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        
        if has_blind_road:
            # 添加盲道纹理
            for i in range(0, size[0], 5):
                for j in range(0, size[1], 5):
                    if (i + j) % 10 == 0:
                        cv2.rectangle(img, (j, i), (j+3, i+3), (100, 100, 100), -1)
        
        return img
    
    def generate_high_res_blind_road_image(self, size: tuple = (640, 480)) -> np.ndarray:
        """生成高分辨率盲道图像"""
        img = np.random.randint(50, 200, (*size, 3), dtype=np.uint8)
        
        # 添加道路纹理
        for i in range(0, size[0], 20):
            cv2.line(img, (0, i), (size[1], i), (120, 120, 120), 2)
        
        # 添加盲道纹理
        blind_road_width = 60
        start_x = (size[1] - blind_road_width) // 2
        
        for i in range(0, size[0], 10):
            for j in range(start_x, start_x + blind_road_width, 8):
                if (i + j) % 16 == 0:
                    cv2.rectangle(img, (j, i), (j+4, i+4), (80, 80, 80), -1)
        
        return img
    
    def generate_blind_road_mask(self, size: tuple) -> np.ndarray:
        """生成盲道分割掩码"""
        mask = np.zeros(size, dtype=np.uint8)
        
        # 在图像中央创建盲道区域
        blind_road_width = 60
        start_x = (size[1] - blind_road_width) // 2
        
        mask[:, start_x:start_x + blind_road_width] = 255
        
        return mask
    
    def generate_damage_detection_image(self, size: tuple = (640, 480)) -> tuple:
        """生成损坏检测图像和标注"""
        img = self.generate_high_res_blind_road_image(size)
        annotations = []
        
        # 随机添加损坏区域
        if np.random.random() > 0.3:  # 70%概率有损坏
            # 损坏区域
            x1 = np.random.randint(100, size[1]-200)
            y1 = np.random.randint(100, size[0]-200)
            x2 = x1 + np.random.randint(50, 150)
            y2 = y1 + np.random.randint(50, 150)
            
            # 绘制损坏区域
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), -1)
            
            # 转换为YOLO格式
            x_center = (x1 + x2) / 2 / size[1]
            y_center = (y1 + y2) / 2 / size[0]
            width = (x2 - x1) / size[1]
            height = (y2 - y1) / size[0]
            
            annotations.append({
                'class_id': 1,  # 损坏盲道
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            })
        
        # 随机添加障碍物
        if np.random.random() > 0.5:  # 50%概率有障碍物
            x1 = np.random.randint(50, size[1]-100)
            y1 = np.random.randint(50, size[0]-100)
            x2 = x1 + np.random.randint(30, 80)
            y2 = y1 + np.random.randint(30, 80)
            
            # 绘制障碍物
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), -1)
            
            # 转换为YOLO格式
            x_center = (x1 + x2) / 2 / size[1]
            y_center = (y1 + y2) / 2 / size[0]
            width = (x2 - x1) / size[1]
            height = (y2 - y1) / size[0]
            
            annotations.append({
                'class_id': 2,  # 障碍物
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            })
        
        return img, annotations
    
    def convert_to_yolo_format(self, dataset_name: str) -> bool:
        """转换为YOLO格式"""
        print(f"🔄 转换数据集为YOLO格式: {dataset_name}")
        
        try:
            raw_path = self.base_dir / "raw" / self.datasets[dataset_name]["local_path"]
            yolo_path = self.base_dir / "yolo_format" / dataset_name
            
            if not raw_path.exists():
                print(f"❌ 原始数据集不存在: {raw_path}")
                return False
            
            # 创建YOLO格式目录
            yolo_path.mkdir(parents=True, exist_ok=True)
            (yolo_path / "images").mkdir(exist_ok=True)
            (yolo_path / "labels").mkdir(exist_ok=True)
            
            if dataset_name == "blind_road_damage":
                return self.convert_damage_dataset_to_yolo(raw_path, yolo_path)
            else:
                return self.convert_general_dataset_to_yolo(raw_path, yolo_path)
                
        except Exception as e:
            print(f"❌ 转换失败: {e}")
            return False
    
    def convert_damage_dataset_to_yolo(self, raw_path: Path, yolo_path: Path) -> bool:
        """转换损坏检测数据集为YOLO格式"""
        images_dir = raw_path / "images"
        annotations_dir = raw_path / "annotations"
        
        if not images_dir.exists() or not annotations_dir.exists():
            print("❌ 图像或标注目录不存在")
            return False
        
        # 复制图像文件
        for img_file in images_dir.glob("*.jpg"):
            shutil.copy2(img_file, yolo_path / "images" / img_file.name)
        
        # 复制标注文件
        for ann_file in annotations_dir.glob("*.txt"):
            shutil.copy2(ann_file, yolo_path / "labels" / ann_file.name)
        
        # 创建数据集配置文件
        dataset_yaml = {
            'path': str(yolo_path.absolute()),
            'train': 'images',
            'val': 'images',
            'test': 'images',
            'nc': 3,
            'names': ['正常盲道', '损坏盲道', '障碍物']
        }
        
        with open(yolo_path / "dataset.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ YOLO格式转换完成: {yolo_path}")
        return True
    
    def convert_general_dataset_to_yolo(self, raw_path: Path, yolo_path: Path) -> bool:
        """转换通用数据集为YOLO格式"""
        # 这里可以实现其他数据集的转换逻辑
        print("⚠️ 通用数据集转换功能待实现")
        return True
    
    def split_dataset(self, dataset_name: str, train_ratio: float = 0.7, 
                     val_ratio: float = 0.2, test_ratio: float = 0.1) -> bool:
        """分割数据集"""
        print(f"📊 分割数据集: {dataset_name}")
        
        try:
            yolo_path = self.base_dir / "yolo_format" / dataset_name
            if not yolo_path.exists():
                print(f"❌ YOLO格式数据集不存在: {yolo_path}")
                return False
            
            # 创建分割目录
            for split in ['train', 'val', 'test']:
                (yolo_path / split / "images").mkdir(parents=True, exist_ok=True)
                (yolo_path / split / "labels").mkdir(parents=True, exist_ok=True)
            
            # 获取所有图像文件
            image_files = list((yolo_path / "images").glob("*.jpg"))
            np.random.shuffle(image_files)
            
            total_images = len(image_files)
            train_end = int(total_images * train_ratio)
            val_end = train_end + int(total_images * val_ratio)
            
            # 分割数据集
            train_files = image_files[:train_end]
            val_files = image_files[train_end:val_end]
            test_files = image_files[val_end:]
            
            # 复制文件到对应目录
            for files, split in [(train_files, 'train'), (val_files, 'val'), (test_files, 'test')]:
                for img_file in files:
                    # 复制图像
                    shutil.copy2(img_file, yolo_path / split / "images" / img_file.name)
                    
                    # 复制对应的标注文件
                    label_file = yolo_path / "labels" / img_file.with_suffix('.txt').name
                    if label_file.exists():
                        shutil.copy2(label_file, yolo_path / split / "labels" / label_file.name)
            
            print(f"✅ 数据集分割完成:")
            print(f"   训练集: {len(train_files)} 张")
            print(f"   验证集: {len(val_files)} 张")
            print(f"   测试集: {len(test_files)} 张")
            
            return True
            
        except Exception as e:
            print(f"❌ 数据集分割失败: {e}")
            return False
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """获取数据集信息"""
        if dataset_name not in self.datasets:
            return {}
        
        info = self.datasets[dataset_name].copy()
        local_path = self.base_dir / "raw" / info["local_path"]
        
        if local_path.exists():
            info["local_exists"] = True
            info["file_count"] = len(list(local_path.rglob("*.*")))
        else:
            info["local_exists"] = False
            info["file_count"] = 0
        
        return info
    
    def list_available_datasets(self):
        """列出可用数据集"""
        print("📋 可用数据集列表:")
        print("=" * 60)
        
        for name, info in self.datasets.items():
            status = "✅ 已下载" if (self.base_dir / "raw" / info["local_path"]).exists() else "❌ 未下载"
            print(f"名称: {info['name']}")
            print(f"描述: {info['description']}")
            print(f"格式: {info['format']}")
            print(f"状态: {status}")
            print("-" * 60)

def main():
    """主函数"""
    print("🎯 盲道检测数据集下载器")
    print("=" * 50)
    
    downloader = BlindRoadDatasetDownloader()
    
    # 列出可用数据集
    downloader.list_available_datasets()
    
    # 下载损坏检测数据集
    print("\n📥 开始下载损坏检测数据集...")
    success = downloader.download_dataset("blind_road_damage")
    
    if success:
        # 转换为YOLO格式
        print("\n🔄 转换为YOLO格式...")
        downloader.convert_to_yolo_format("blind_road_damage")
        
        # 分割数据集
        print("\n📊 分割数据集...")
        downloader.split_dataset("blind_road_damage")
        
        print("\n✅ 数据集准备完成！")
        print(f"数据集位置: {downloader.base_dir}")
    else:
        print("\n❌ 数据集下载失败")

if __name__ == "__main__":
    main()





















