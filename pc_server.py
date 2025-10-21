#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电脑端服务器
接收手机数据、管理训练、推送模型更新
"""

import os
import json
import time
import threading
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from datetime import datetime
import shutil

app = Flask(__name__)
CORS(app)

class PCServer:
    """电脑端服务器主类"""
    
    def __init__(self):
        self.port = 8080
        self.host = "0.0.0.0"
        self.collected_data = []
        self.devices = {}
        self.model_version = "1.0.0"
        self.training_queue = []
        self.is_training = False
        
        # 创建必要目录
        os.makedirs("mobile_data", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("training_data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
    def start_server(self):
        """启动服务器"""
        print(f"🖥️ 启动电脑端服务器: http://{self.host}:{self.port}")
        app.run(host=self.host, port=self.port, debug=False)
    
    def add_device(self, device_id):
        """添加设备"""
        if device_id not in self.devices:
            self.devices[device_id] = {
                'connected_time': time.time(),
                'last_heartbeat': time.time(),
                'data_count': 0,
                'model_version': self.model_version
            }
            print(f"📱 新设备连接: {device_id}")
    
    def update_device_heartbeat(self, device_id):
        """更新设备心跳"""
        if device_id in self.devices:
            self.devices[device_id]['last_heartbeat'] = time.time()
    
    def save_detection_data(self, data):
        """保存检测数据"""
        try:
            device_id = data.get('device_id')
            timestamp = data.get('timestamp')
            
            # 保存数据
            filename = f"mobile_data/detection_{device_id}_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # 更新设备信息
            self.add_device(device_id)
            self.update_device_heartbeat(device_id)
            self.devices[device_id]['data_count'] += 1
            
            print(f"📊 保存检测数据: {device_id}")
            
        except Exception as e:
            print(f"❌ 保存检测数据失败: {e}")
    
    def save_training_data(self, data):
        """保存训练数据"""
        try:
            device_id = data.get('device_id')
            training_data = data.get('data', [])
            
            if not training_data:
                return
            
            # 创建设备数据目录
            device_dir = f"training_data/{device_id}"
            os.makedirs(device_dir, exist_ok=True)
            
            # 保存图像和标注
            for item in training_data:
                image_path = item.get('image_path')
                if image_path and os.path.exists(image_path):
                    # 复制图像到训练数据目录
                    new_image_path = f"{device_dir}/frame_{item['frame_id']}.jpg"
                    shutil.copy2(image_path, new_image_path)
                    
                    # 保存标注信息
                    annotation_path = f"{device_dir}/frame_{item['frame_id']}.json"
                    with open(annotation_path, 'w', encoding='utf-8') as f:
                        json.dump(item, f, ensure_ascii=False, indent=2)
            
            print(f"📊 保存训练数据: {device_id}, {len(training_data)} 帧")
            
            # 添加到训练队列
            self.add_to_training_queue(device_id)
            
        except Exception as e:
            print(f"❌ 保存训练数据失败: {e}")
    
    def add_to_training_queue(self, device_id):
        """添加到训练队列"""
        if device_id not in self.training_queue:
            self.training_queue.append(device_id)
            print(f"🎯 添加到训练队列: {device_id}")
            
            # 如果当前没有训练，开始训练
            if not self.is_training:
                self.start_training()
    
    def start_training(self):
        """开始训练"""
        if self.is_training:
            return
        
        self.is_training = True
        print("🎯 开始模型训练...")
        
        # 在后台线程中运行训练
        training_thread = threading.Thread(target=self.run_training)
        training_thread.daemon = True
        training_thread.start()
    
    def run_training(self):
        """运行训练"""
        try:
            while self.training_queue and self.is_training:
                device_id = self.training_queue.pop(0)
                
                print(f"🎯 训练设备数据: {device_id}")
                
                # 准备训练数据
                self.prepare_training_data(device_id)
                
                # 运行训练
                success = self.train_model(device_id)
                
                if success:
                    # 更新模型版本
                    self.model_version = f"{self.model_version}.{int(time.time())}"
                    
                    # 推送模型更新
                    self.push_model_update(device_id)
                
                time.sleep(1)  # 避免过于频繁的训练
            
            self.is_training = False
            print("✅ 训练完成")
            
        except Exception as e:
            print(f"❌ 训练失败: {e}")
            self.is_training = False
    
    def prepare_training_data(self, device_id):
        """准备训练数据"""
        try:
            device_dir = f"training_data/{device_id}"
            if not os.path.exists(device_dir):
                return False
            
            # 创建YOLO格式数据集
            yolo_dir = f"training_data/{device_id}_yolo"
            os.makedirs(yolo_dir, exist_ok=True)
            os.makedirs(f"{yolo_dir}/images", exist_ok=True)
            os.makedirs(f"{yolo_dir}/labels", exist_ok=True)
            
            # 转换数据格式
            image_files = [f for f in os.listdir(device_dir) if f.endswith('.jpg')]
            
            for image_file in image_files:
                image_path = f"{device_dir}/{image_file}"
                annotation_file = image_file.replace('.jpg', '.json')
                annotation_path = f"{device_dir}/{annotation_file}"
                
                if os.path.exists(annotation_path):
                    # 读取标注信息
                    with open(annotation_path, 'r', encoding='utf-8') as f:
                        annotation = json.load(f)
                    
                    # 复制图像
                    new_image_path = f"{yolo_dir}/images/{image_file}"
                    shutil.copy2(image_path, new_image_path)
                    
                    # 转换标注格式
                    self.convert_to_yolo_format(annotation, f"{yolo_dir}/labels/{image_file.replace('.jpg', '.txt')}")
            
            print(f"✅ 准备训练数据完成: {len(image_files)} 张图像")
            return True
            
        except Exception as e:
            print(f"❌ 准备训练数据失败: {e}")
            return False
    
    def convert_to_yolo_format(self, annotation, output_path):
        """转换为YOLO格式"""
        try:
            detections = annotation.get('detections', [])
            
            with open(output_path, 'w') as f:
                for detection in detections:
                    bbox = detection['bbox']
                    class_name = detection['class_name']
                    
                    # 转换为YOLO格式 (class_id x_center y_center width height)
                    # 这里需要根据您的类别映射进行调整
                    class_id = 0  # 默认类别ID
                    if class_name == 'obstacle':
                        class_id = 0
                    elif class_name == 'person':
                        class_id = 1
                    
                    # 计算中心点和宽高（归一化）
                    x_center = (bbox[0] + bbox[2]) / 2 / 640  # 假设图像宽度640
                    y_center = (bbox[1] + bbox[3]) / 2 / 480  # 假设图像高度480
                    width = (bbox[2] - bbox[0]) / 640
                    height = (bbox[3] - bbox[1]) / 480
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    
        except Exception as e:
            print(f"❌ 转换YOLO格式失败: {e}")
    
    def train_model(self, device_id):
        """训练模型"""
        try:
            print(f"🎯 开始训练模型: {device_id}")
            
            # 这里应该调用您的训练脚本
            # 暂时模拟训练过程
            time.sleep(5)  # 模拟训练时间
            
            # 生成新的模型文件
            new_model_path = f"models/best_{device_id}_{int(time.time())}.pt"
            
            # 复制现有模型作为新模型（实际应该训练生成）
            if os.path.exists("models/best.pt"):
                shutil.copy2("models/best.pt", new_model_path)
            
            print(f"✅ 模型训练完成: {new_model_path}")
            return True
            
        except Exception as e:
            print(f"❌ 模型训练失败: {e}")
            return False
    
    def push_model_update(self, device_id):
        """推送模型更新"""
        try:
            # 这里应该通知设备有新模型可用
            print(f"📤 推送模型更新到设备: {device_id}")
            
            # 更新设备模型版本
            if device_id in self.devices:
                self.devices[device_id]['model_version'] = self.model_version
            
        except Exception as e:
            print(f"❌ 推送模型更新失败: {e}")

# 全局服务器实例
server = PCServer()

# Flask路由
@app.route('/detection_data', methods=['POST'])
def receive_detection_data():
    """接收检测数据"""
    try:
        data = request.json
        server.save_detection_data(data)
        return jsonify({'status': 'success', 'model_update_available': False})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/training_data', methods=['POST'])
def receive_training_data():
    """接收训练数据"""
    try:
        data = request.json
        server.save_training_data(data)
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/model_status', methods=['GET'])
def get_model_status():
    """获取模型状态"""
    try:
        device_id = request.args.get('device_id', '')
        
        if device_id in server.devices:
            current_version = server.devices[device_id]['model_version']
            if current_version != server.model_version:
                return jsonify({
                    'update_available': True,
                    'version': server.model_version,
                    'download_url': f'/download_model/{server.model_version}'
                })
        
        return jsonify({'update_available': False})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/download_model/<version>', methods=['GET'])
def download_model(version):
    """下载模型"""
    try:
        model_path = f"models/best_{version}.pt"
        if os.path.exists(model_path):
            return send_file(model_path, as_attachment=True)
        else:
            return jsonify({'status': 'error', 'message': 'Model not found'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/commands', methods=['GET'])
def get_commands():
    """获取指令"""
    try:
        device_id = request.args.get('device_id', '')
        
        # 这里可以根据需要返回指令
        commands = []
        
        return jsonify(commands)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/status', methods=['GET'])
def get_server_status():
    """获取服务器状态"""
    try:
        status = {
            'server_time': datetime.now().isoformat(),
            'model_version': server.model_version,
            'connected_devices': len(server.devices),
            'is_training': server.is_training,
            'training_queue': len(server.training_queue),
            'devices': server.devices
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/start_training', methods=['POST'])
def start_training():
    """手动开始训练"""
    try:
        data = request.json
        device_id = data.get('device_id')
        
        if device_id:
            server.add_to_training_queue(device_id)
            return jsonify({'status': 'success', 'message': f'Training started for {device_id}'})
        else:
            return jsonify({'status': 'error', 'message': 'Device ID required'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

def main():
    """主函数"""
    print("🖥️ 盲道障碍检测 - 电脑端服务器")
    print("=" * 50)
    
    try:
        server.start_server()
    except KeyboardInterrupt:
        print("\n👋 服务器退出")

if __name__ == "__main__":
    main() 