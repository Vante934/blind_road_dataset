#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据设置脚本
帮助用户快速设置模型测试所需的数据集
"""

import os
import shutil
import glob
from pathlib import Path

def setup_test_directories():
    """创建测试目录结构"""
    print("🔧 创建测试目录结构...")
    
    test_dirs = [
        "datasets/test/standard",
        "datasets/test/challenging", 
        "datasets/test/real_world"
    ]
    
    for dir_path in test_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ 创建目录: {dir_path}")

def copy_existing_images():
    """复制现有图像到测试目录"""
    print("\n📁 复制现有图像到测试目录...")
    
    # 查找项目中的图像文件
    image_sources = [
        "data/images",
        "data/Blind_DataSet", 
        "data/Environment_DataSet",
        "data/environment_images"
    ]
    
    total_copied = 0
    
    for source_dir in image_sources:
        if os.path.exists(source_dir):
            print(f"📂 从 {source_dir} 复制图像...")
            
            # 查找图像文件
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            image_files = []
            
            for ext in image_extensions:
                pattern = os.path.join(source_dir, f"**/{ext}")
                image_files.extend(glob.glob(pattern, recursive=True))
            
            if image_files:
                # 复制到标准测试集
                copied_count = 0
                for img_file in image_files[:50]:  # 限制数量
                    try:
                        filename = os.path.basename(img_file)
                        dest_path = os.path.join("datasets/test/standard", filename)
                        shutil.copy2(img_file, dest_path)
                        copied_count += 1
                    except Exception as e:
                        print(f"⚠️ 复制失败 {img_file}: {e}")
                
                print(f"✅ 复制了 {copied_count} 张图像到标准测试集")
                total_copied += copied_count
            else:
                print(f"⚠️ {source_dir} 中没有找到图像文件")
    
    return total_copied

def create_sample_images():
    """创建示例图像（如果没有任何图像）"""
    print("\n🎨 创建示例图像...")
    
    try:
        import cv2
        import numpy as np
        
        # 创建一些示例图像
        for i in range(5):
            # 创建随机图像
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # 添加一些简单的形状
            cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), -1)
            cv2.circle(img, (400, 300), 50, (255, 0, 0), -1)
            
            # 保存图像
            filename = f"sample_image_{i+1}.jpg"
            filepath = os.path.join("datasets/test/standard", filename)
            cv2.imwrite(filepath, img)
            print(f"✅ 创建示例图像: {filename}")
            
    except ImportError:
        print("⚠️ OpenCV 不可用，无法创建示例图像")
    except Exception as e:
        print(f"⚠️ 创建示例图像失败: {e}")

def check_test_data():
    """检查测试数据状态"""
    print("\n📊 检查测试数据状态...")
    
    test_dirs = [
        "datasets/test/standard",
        "datasets/test/challenging",
        "datasets/test/real_world"
    ]
    
    total_images = 0
    
    for dir_path in test_dirs:
        if os.path.exists(dir_path):
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            image_files = []
            
            for ext in image_extensions:
                pattern = os.path.join(dir_path, f"**/{ext}")
                image_files.extend(glob.glob(pattern, recursive=True))
            
            count = len(image_files)
            total_images += count
            print(f"📁 {dir_path}: {count} 张图像")
        else:
            print(f"❌ {dir_path}: 目录不存在")
    
    print(f"\n📈 总计: {total_images} 张测试图像")
    
    if total_images == 0:
        print("\n⚠️ 没有找到测试图像！")
        print("建议:")
        print("1. 将您的图像文件复制到 datasets/test/standard/ 目录")
        print("2. 或者使用 '自定义路径' 功能选择包含图像的文件夹")
        print("3. 支持的图像格式: JPG, JPEG, PNG, BMP")
    else:
        print("✅ 测试数据准备完成！")

def main():
    """主函数"""
    print("🚀 盲道检测模型测试数据设置")
    print("=" * 50)
    
    # 1. 创建目录结构
    setup_test_directories()
    
    # 2. 复制现有图像
    copied_count = copy_existing_images()
    
    # 3. 如果没有图像，创建示例
    if copied_count == 0:
        create_sample_images()
    
    # 4. 检查数据状态
    check_test_data()
    
    print("\n" + "=" * 50)
    print("🎉 设置完成！")
    print("\n📝 使用说明:")
    print("1. 运行 python model_training_interface.py")
    print("2. 点击 '模型测试' 标签页")
    print("3. 选择数据集和模型")
    print("4. 点击 '开始测试' 按钮")
    print("\n💡 提示:")
    print("- 可以使用 '自定义路径' 按钮选择其他文件夹")
    print("- 支持 JPG, JPEG, PNG, BMP 格式的图像")
    print("- 建议使用包含盲道场景的图像进行测试")

if __name__ == "__main__":
    main()

