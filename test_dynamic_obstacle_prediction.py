#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态障碍物预测功能测试脚本
演示障碍物检测、跟踪和轨迹预测功能
"""

import cv2
import numpy as np
import time
import sys
import os
from typing import List, Tuple

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from modules.integrated_obstacle_system import IntegratedObstacleSystem, ObstacleDetectionGUI
from modules.dynamic_obstacle_predictor import DynamicObstaclePredictor

def test_obstacle_predictor():
    """测试障碍物预测器"""
    print("🧪 测试障碍物预测器...")
    
    predictor = DynamicObstaclePredictor()
    
    # 模拟检测结果序列
    detection_sequences = [
        # 第1帧
        [
            {'bbox': (100, 100, 200, 200), 'class_name': 'person', 'confidence': 0.9},
            {'bbox': (300, 150, 400, 250), 'class_name': 'car', 'confidence': 0.8}
        ],
        # 第2帧（移动后的位置）
        [
            {'bbox': (110, 105, 210, 205), 'class_name': 'person', 'confidence': 0.9},
            {'bbox': (320, 160, 420, 260), 'class_name': 'car', 'confidence': 0.8}
        ],
        # 第3帧
        [
            {'bbox': (120, 110, 220, 210), 'class_name': 'person', 'confidence': 0.9},
            {'bbox': (340, 170, 440, 270), 'class_name': 'car', 'confidence': 0.8}
        ]
    ]
    
    print("📊 处理检测序列...")
    for i, detections in enumerate(detection_sequences):
        print(f"\n帧 {i+1}:")
        
        # 更新障碍物
        obstacles = predictor.update_obstacles(detections)
        print(f"  检测到 {len(obstacles)} 个障碍物")
        
        for obstacle in obstacles:
            print(f"    障碍物 {obstacle.id}: {obstacle.obstacle_type.value}, "
                  f"速度: {obstacle.speed:.2f} px/s, "
                  f"状态: {obstacle.movement_state.value}")
        
        # 预测轨迹
        predictions = predictor.predict_obstacle_trajectories()
        print(f"  生成了 {len(predictions)} 个预测")
        
        for prediction in predictions:
            print(f"    预测 {prediction.obstacle_id}: "
                  f"风险 {prediction.collision_risk:.2f}, "
                  f"建议: {prediction.recommended_action}")
    
    # 获取统计信息
    stats = predictor.get_obstacle_statistics()
    print(f"\n📈 统计信息: {stats}")
    
    return predictor

def test_integrated_system():
    """测试集成系统"""
    print("\n🔧 测试集成障碍物检测系统...")
    
    system = IntegratedObstacleSystem()
    
    # 创建测试图像
    test_image = create_test_image()
    
    # 处理图像
    results = system.process_frame(test_image)
    
    print(f"✅ 处理完成:")
    print(f"  检测到 {len(results['detections'])} 个目标")
    print(f"  跟踪到 {len(results['obstacles'])} 个障碍物")
    print(f"  生成 {len(results['predictions'])} 个预测")
    print(f"  处理时间: {results['processing_time']*1000:.1f}ms")
    
    if results['warnings']:
        print(f"⚠️ 警告信息:")
        for warning in results['warnings']:
            print(f"    {warning}")
    
    return system, results

def create_test_image() -> np.ndarray:
    """创建测试图像"""
    # 创建白色背景
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # 绘制一些目标
    # 人物
    cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), -1)
    cv2.circle(img, (150, 80), 20, (0, 255, 0), -1)  # 头部
    
    # 汽车
    cv2.rectangle(img, (300, 150), (450, 250), (255, 0, 0), -1)
    cv2.circle(img, (320, 270), 15, (255, 0, 0), -1)  # 轮子
    cv2.circle(img, (430, 270), 15, (255, 0, 0), -1)  # 轮子
    
    # 自行车
    cv2.circle(img, (200, 300), 25, (0, 0, 255), -1)  # 前轮
    cv2.circle(img, (350, 300), 25, (0, 0, 255), -1)  # 后轮
    cv2.line(img, (200, 300), (350, 300), (0, 0, 255), 3)  # 车架
    
    return img

def demo_camera_detection():
    """演示摄像头检测"""
    print("\n📹 启动摄像头检测演示...")
    
    gui = ObstacleDetectionGUI()
    
    try:
        # 启动摄像头
        gui.start_camera(0)
        
        print("🎮 控制说明:")
        print("  'q' - 退出")
        print("  'r' - 重置系统")
        print("  's' - 保存当前帧")
        print("  鼠标点击 - 设置用户位置")
        
        frame_count = 0
        while True:
            # 处理摄像头数据
            result = gui.process_camera_feed()
            if result is None:
                print("❌ 无法读取摄像头数据")
                break
            
            vis_frame, results = result
            frame_count += 1
            
            # 显示结果
            cv2.imshow('Dynamic Obstacle Detection', vis_frame)
            
            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                gui.system.reset_system()
                print("🔄 系统已重置")
            elif key == ord('s'):
                filename = f'detection_frame_{frame_count}.jpg'
                cv2.imwrite(filename, vis_frame)
                print(f"💾 保存帧: {filename}")
            
            # 每30帧显示一次统计信息
            if frame_count % 30 == 0:
                stats = results['statistics']
                print(f"📊 帧 {frame_count}: {stats['total_obstacles']} 个障碍物, "
                      f"{stats['moving_obstacles']} 个运动, "
                      f"处理时间: {results['processing_time']*1000:.1f}ms")
    
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断")
    except Exception as e:
        print(f"❌ 错误: {e}")
    finally:
        gui.stop_camera()
        cv2.destroyAllWindows()

def test_trajectory_prediction():
    """测试轨迹预测功能"""
    print("\n🎯 测试轨迹预测功能...")
    
    predictor = DynamicObstaclePredictor()
    
    # 模拟一个移动的障碍物
    positions = [
        (100, 100), (110, 105), (120, 110), (130, 115), (140, 120)
    ]
    
    for i, pos in enumerate(positions):
        detections = [{
            'bbox': (pos[0]-50, pos[1]-50, pos[0]+50, pos[1]+50),
            'class_name': 'person',
            'confidence': 0.9
        }]
        
        obstacles = predictor.update_obstacles(detections)
        predictions = predictor.predict_obstacle_trajectories()
        
        print(f"位置 {i+1}: {pos}")
        if predictions:
            pred = predictions[0]
            print(f"  预测位置数: {len(pred.predicted_positions)}")
            print(f"  碰撞风险: {pred.collision_risk:.2f}")
            print(f"  建议: {pred.recommended_action}")
        
        time.sleep(0.1)  # 模拟时间间隔

def create_prediction_visualization():
    """创建预测可视化"""
    print("\n🎨 创建预测可视化...")
    
    # 创建测试图像
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # 绘制网格
    for i in range(0, 640, 50):
        cv2.line(img, (i, 0), (i, 480), (200, 200, 200), 1)
    for i in range(0, 480, 50):
        cv2.line(img, (0, i), (640, i), (200, 200, 200), 1)
    
    # 绘制坐标轴
    cv2.line(img, (0, 240), (640, 240), (0, 0, 0), 2)  # X轴
    cv2.line(img, (320, 0), (320, 480), (0, 0, 0), 2)  # Y轴
    
    # 添加标题
    cv2.putText(img, "Dynamic Obstacle Prediction Demo", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # 保存图像
    cv2.imwrite("prediction_demo.jpg", img)
    print("✅ 预测演示图像已保存: prediction_demo.jpg")

def main():
    """主函数"""
    print("🚀 动态障碍物预测功能测试")
    print("=" * 50)
    
    # 1. 测试障碍物预测器
    predictor = test_obstacle_predictor()
    
    # 2. 测试集成系统
    system, results = test_integrated_system()
    
    # 3. 测试轨迹预测
    test_trajectory_prediction()
    
    # 4. 创建可视化
    create_prediction_visualization()
    
    # 5. 询问是否启动摄像头演示
    print("\n" + "=" * 50)
    choice = input("是否启动摄像头检测演示？(y/n): ").lower().strip()
    
    if choice == 'y':
        demo_camera_detection()
    else:
        print("📝 测试完成！")
        print("\n💡 功能特点:")
        print("✅ 实时障碍物检测和跟踪")
        print("✅ 动态轨迹预测")
        print("✅ 碰撞风险评估")
        print("✅ 智能预警系统")
        print("✅ 可视化结果展示")

if __name__ == "__main__":
    main()

