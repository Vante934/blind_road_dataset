#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练优化脚本
用于提升模型精度到85%以上
"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
import yaml

def optimize_training_config():
    """优化训练配置"""
    
    print("=" * 60)
    print("模型训练优化配置")
    print("=" * 60)
    
    # 检查数据集
    dataset_path = "datasets/blind_road/dataset.yaml"
    if not os.path.exists(dataset_path):
        print(f"❌ 数据集配置文件不存在: {dataset_path}")
        print("请先准备数据集")
        return False
    
    # 读取数据集配置
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)
    
    print(f"✅ 数据集配置: {dataset_path}")
    print(f"   类别数量: {dataset_config.get('nc', '未知')}")
    print(f"   类别名称: {dataset_config.get('names', [])}")
    
    # 选择模型（使用更大的模型以提高精度）
    model_size = "yolov8m.pt"  # 使用medium模型，平衡速度和精度
    # 如果需要更高精度，可以使用 yolov8l.pt 或 yolov8x.pt
    
    print(f"\n📦 使用模型: {model_size}")
    
    # 加载预训练模型
    try:
        model = YOLO(model_size)
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("尝试使用yolov8n.pt...")
        model = YOLO('yolov8n.pt')
    
    # 优化后的训练参数
    training_params = {
        'data': dataset_path,
        'epochs': 200,  # 增加训练轮数
        'imgsz': 640,
        'batch': 16,  # 根据GPU内存调整
        'device': 'cuda' if os.system('nvidia-smi') == 0 else 'cpu',
        'project': 'results/blind_road_training',
        'name': 'optimized_blind_road_detection',
        'save': True,
        'save_period': 10,  # 每10个epoch保存一次
        'patience': 50,  # 早停耐心值
        'lr0': 0.001,  # 初始学习率（降低以提高稳定性）
        'lrf': 0.01,  # 最终学习率
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,  # 边界框损失权重
        'cls': 0.5,  # 分类损失权重
        'dfl': 1.5,  # DFL损失权重
        
        # 数据增强（重要！）
        'augment': True,
        'hsv_h': 0.015,  # 色调增强
        'hsv_s': 0.7,  # 饱和度增强
        'hsv_v': 0.4,  # 明度增强
        'degrees': 10.0,  # 旋转角度
        'translate': 0.1,  # 平移
        'scale': 0.5,  # 缩放
        'shear': 2.0,  # 剪切
        'perspective': 0.0,  # 透视变换
        'flipud': 0.0,  # 上下翻转
        'fliplr': 0.5,  # 左右翻转
        'mosaic': 1.0,  # Mosaic增强
        'mixup': 0.1,  # Mixup增强
        'copy_paste': 0.1,  # Copy-Paste增强
        
        # 验证
        'val': True,
        'plots': True,  # 生成图表
        
        # 其他优化
        'optimizer': 'AdamW',  # 使用AdamW优化器
        'verbose': True,
        'seed': 42,  # 随机种子
        'deterministic': True,  # 确定性训练
        'single_cls': False,  # 多类别
        'rect': False,  # 矩形训练
        'cos_lr': True,  # 余弦学习率调度
        'close_mosaic': 10,  # 最后10个epoch关闭mosaic
        'resume': False,  # 是否恢复训练
        'amp': True,  # 自动混合精度
        'fraction': 1.0,  # 使用全部数据
        'profile': False,  # 性能分析
        'freeze': None,  # 冻结层数
        'multi_scale': False,  # 多尺度训练
    }
    
    print("\n📋 训练参数:")
    for key, value in training_params.items():
        print(f"   {key}: {value}")
    
    # 确认开始训练
    print("\n" + "=" * 60)
    response = input("是否开始训练？(y/n): ")
    if response.lower() != 'y':
        print("训练已取消")
        return False
    
    print("\n🚀 开始训练...")
    print("=" * 60)
    
    try:
        # 开始训练
        results = model.train(**training_params)
        
        print("\n" + "=" * 60)
        print("✅ 训练完成！")
        print("=" * 60)
        
        # 显示结果
        if hasattr(results, 'results_dict'):
            print("\n📊 训练结果:")
            for key, value in results.results_dict.items():
                print(f"   {key}: {value}")
        
        # 最佳模型路径
        best_model_path = os.path.join(
            training_params['project'],
            training_params['name'],
            'weights',
            'best.pt'
        )
        
        if os.path.exists(best_model_path):
            print(f"\n🏆 最佳模型保存在: {best_model_path}")
            
            # 验证模型
            print("\n🔍 验证最佳模型...")
            best_model = YOLO(best_model_path)
            metrics = best_model.val(data=dataset_path)
            
            print("\n📈 验证指标:")
            if hasattr(metrics, 'results_dict'):
                for key, value in metrics.results_dict.items():
                    print(f"   {key}: {value}")
            
            # 检查精度
            if hasattr(metrics, 'results_dict'):
                mAP50 = metrics.results_dict.get('metrics/mAP50(B)', 0)
                mAP50_95 = metrics.results_dict.get('metrics/mAP50-95(B)', 0)
                
                print(f"\n🎯 模型精度:")
                print(f"   mAP50: {mAP50:.4f}")
                print(f"   mAP50-95: {mAP50_95:.4f}")
                
                if mAP50 >= 0.85:
                    print("✅ 模型精度达到目标（85%以上）！")
                else:
                    print("⚠️ 模型精度未达到目标，建议:")
                    print("   1. 增加训练数据")
                    print("   2. 使用更大的模型（yolov8l或yolov8x）")
                    print("   3. 调整超参数")
                    print("   4. 检查数据质量")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_dataset_quality():
    """检查数据集质量"""
    print("\n" + "=" * 60)
    print("数据集质量检查")
    print("=" * 60)
    
    dataset_path = "datasets/blind_road"
    train_images = Path(dataset_path) / "train" / "images"
    train_labels = Path(dataset_path) / "train" / "labels"
    val_images = Path(dataset_path) / "val" / "images"
    val_labels = Path(dataset_path) / "val" / "labels"
    
    train_img_count = len(list(train_images.glob("*"))) if train_images.exists() else 0
    train_label_count = len(list(train_labels.glob("*"))) if train_labels.exists() else 0
    val_img_count = len(list(val_images.glob("*"))) if val_images.exists() else 0
    val_label_count = len(list(val_labels.glob("*"))) if val_labels.exists() else 0
    
    print(f"训练集:")
    print(f"  图像数量: {train_img_count}")
    print(f"  标注数量: {train_label_count}")
    
    print(f"\n验证集:")
    print(f"  图像数量: {val_img_count}")
    print(f"  标注数量: {val_label_count}")
    
    # 检查数据平衡
    if train_img_count > 0:
        ratio = val_img_count / train_img_count
        print(f"\n验证集比例: {ratio:.2%}")
        if ratio < 0.1:
            print("⚠️ 验证集比例过低，建议至少10%")
        elif ratio > 0.3:
            print("⚠️ 验证集比例过高，建议10-20%")
        else:
            print("✅ 验证集比例合理")
    
    # 检查数据量
    total_images = train_img_count + val_img_count
    print(f"\n总数据量: {total_images}张")
    
    if total_images < 500:
        print("⚠️ 数据量较少，建议至少500张图像")
        print("   建议:")
        print("   1. 增加数据采集")
        print("   2. 使用数据增强")
        print("   3. 使用预训练模型")
    elif total_images < 1000:
        print("✅ 数据量基本满足要求")
        print("   建议增加到1000张以上以获得更好效果")
    else:
        print("✅ 数据量充足")
    
    return {
        'train_images': train_img_count,
        'train_labels': train_label_count,
        'val_images': val_img_count,
        'val_labels': val_label_count,
        'total': total_images
    }

if __name__ == "__main__":
    print("=" * 60)
    print("模型训练优化工具")
    print("=" * 60)
    
    # 检查数据集
    dataset_info = check_dataset_quality()
    
    # 开始优化训练
    if dataset_info['total'] > 0:
        optimize_training_config()
    else:
        print("\n❌ 未找到数据集，请先准备数据集")
        print("数据集路径: datasets/blind_road/")
