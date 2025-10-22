#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块通信 - 处理三个模块之间的通信和协调
"""

import json
import socket
import threading
import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

@dataclass
class Message:
    """消息数据类"""
    sender: str
    receiver: str
    message_type: str
    data: Dict[str, Any]
    timestamp: float

class ModuleCommunicator:
    """模块通信器"""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.socket = None
        self.clients = {}
        self.message_handlers = {}
        self.is_running = False
        
    def register_handler(self, message_type: str, handler: Callable):
        """注册消息处理器"""
        self.message_handlers[message_type] = handler
    
    def start_server(self):
        """启动通信服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(('localhost', self.port))
            self.socket.listen(5)
            self.is_running = True
            
            print(f"🌐 通信服务器已启动，端口: {self.port}")
            
            # 启动客户端连接处理线程
            threading.Thread(target=self._handle_connections, daemon=True).start()
            
        except Exception as e:
            print(f"❌ 通信服务器启动失败: {e}")
    
    def _handle_connections(self):
        """处理客户端连接"""
        while self.is_running:
            try:
                client_socket, address = self.socket.accept()
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address),
                    daemon=True
                )
                client_thread.start()
            except Exception as e:
                if self.is_running:
                    print(f"❌ 处理连接时出错: {e}")
    
    def _handle_client(self, client_socket: socket.socket, address):
        """处理单个客户端"""
        try:
            while self.is_running:
                data = client_socket.recv(1024)
                if not data:
                    break
                
                message_data = json.loads(data.decode('utf-8'))
                message = Message(
                    sender=message_data.get('sender', 'unknown'),
                    receiver=message_data.get('receiver', 'all'),
                    message_type=message_data.get('type', 'unknown'),
                    data=message_data.get('data', {}),
                    timestamp=time.time()
                )
                
                self._process_message(message)
                
        except Exception as e:
            print(f"❌ 处理客户端消息时出错: {e}")
        finally:
            client_socket.close()
    
    def _process_message(self, message: Message):
        """处理接收到的消息"""
        print(f"📨 收到消息: {message.sender} -> {message.receiver} ({message.message_type})")
        
        # 调用注册的处理器
        if message.message_type in self.message_handlers:
            try:
                self.message_handlers[message.message_type](message)
            except Exception as e:
                print(f"❌ 处理消息时出错: {e}")
    
    def send_message(self, receiver: str, message_type: str, data: Dict[str, Any]):
        """发送消息"""
        message = Message(
            sender='main_controller',
            receiver=receiver,
            message_type=message_type,
            data=data,
            timestamp=time.time()
        )
        
        # 这里可以实现具体的发送逻辑
        print(f"📤 发送消息: {message.sender} -> {message.receiver} ({message.message_type})")
    
    def stop_server(self):
        """停止通信服务器"""
        self.is_running = False
        if self.socket:
            self.socket.close()
        print("🛑 通信服务器已停止")

class DetectionToVoiceBridge:
    """检测结果到语音的桥接器"""
    
    def __init__(self, communicator: ModuleCommunicator):
        self.communicator = communicator
        self.register_handlers()
    
    def register_handlers(self):
        """注册消息处理器"""
        self.communicator.register_handler('detection_result', self.handle_detection_result)
        self.communicator.register_handler('voice_request', self.handle_voice_request)
    
    def handle_detection_result(self, message: Message):
        """处理检测结果"""
        detection_data = message.data
        
        # 生成语音指令
        voice_instruction = self.generate_voice_instruction(detection_data)
        
        # 发送语音指令
        self.communicator.send_message(
            receiver='voice_system',
            message_type='voice_instruction',
            data={'instruction': voice_instruction}
        )
    
    def generate_voice_instruction(self, detection_data: Dict[str, Any]) -> str:
        """根据检测结果生成语音指令"""
        if detection_data.get('has_obstacle', False):
            obstacle_type = detection_data.get('obstacle_type', '障碍物')
            return f"前方发现{obstacle_type}，请注意安全"
        elif detection_data.get('has_blind_path', False):
            return "检测到盲道，可以安全通行"
        else:
            return "前方道路正常"
    
    def handle_voice_request(self, message: Message):
        """处理语音请求"""
        request_data = message.data
        request_type = request_data.get('type', 'unknown')
        
        if request_type == 'status_check':
            # 检查检测状态
            self.communicator.send_message(
                receiver='detection_system',
                message_type='status_request',
                data={'request_id': request_data.get('request_id')}
            )

def main():
    """测试通信模块"""
    print("=" * 50)
    print("🌐 模块通信测试")
    print("=" * 50)
    
    communicator = ModuleCommunicator()
    bridge = DetectionToVoiceBridge(communicator)
    
    try:
        communicator.start_server()
        
        # 模拟消息处理
        time.sleep(5)
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
    finally:
        communicator.stop_server()

if __name__ == "__main__":
    main()
