"""
WebSocket连接管理器
"""
import logging
from typing import Dict, List
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    WebSocket连接管理器
    
    功能：
    1. 管理WebSocket连接
    2. 发送消息到特定设备
    3. 广播消息到所有设备
    """
    
    def __init__(self):
        # {device_id: WebSocket}
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, device_id: str):
        """
        建立WebSocket连接
        
        Args:
            websocket: WebSocket连接对象
            device_id: 设备ID
        """
        await websocket.accept()
        self.active_connections[device_id] = websocket
        logger.info(f"设备 {device_id} 连接成功")
    
    def disconnect(self, device_id: str):
        """
        断开WebSocket连接
        
        Args:
            device_id: 设备ID
        """
        if device_id in self.active_connections:
            del self.active_connections[device_id]
            logger.info(f"设备 {device_id} 断开连接")
    
    async def send_personal_message(self, message: dict, device_id: str):
        """
        发送消息到特定设备
        
        Args:
            message: 消息内容
            device_id: 设备ID
        """
        if device_id in self.active_connections:
            try:
                await self.active_connections[device_id].send_json(message)
            except Exception as e:
                logger.error(f"发送消息失败: {e}")
                self.disconnect(device_id)
    
    async def broadcast(self, message: dict):
        """
        广播消息到所有设备
        
        Args:
            message: 消息内容
        """
        disconnected = []
        for device_id, connection in self.active_connections.items():
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"广播消息失败: {e}")
                disconnected.append(device_id)
        
        # 清理断开的连接
        for device_id in disconnected:
            self.disconnect(device_id)


# 单例
try:
    manager
except NameError:
    manager = ConnectionManager()