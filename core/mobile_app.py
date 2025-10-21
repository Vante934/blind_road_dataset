# -*- coding: utf-8 -*-
"""
盲道障碍检测移动端应用
简化版本，专门用于Android部署
"""

import sys
import os
import cv2
import numpy as np
import json
import threading
import time
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
import requests
import base64
import tempfile
import uuid

# 导入语音库
try:
    from voice_library import VoiceLibrary
    VOICE_LIBRARY_AVAILABLE = True
    print("✅ 语音库导入成功")
except ImportError:
    VOICE_LIBRARY_AVAILABLE = False
    print("⚠️ 语音库导入失败，将使用默认语音提示")

# 导入简化的语音系统
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from simple_voice_system import voice_system
    SIMPLE_VOICE_AVAILABLE = True
    print("✅ 简化语音系统导入成功")
except ImportError as e:
    SIMPLE_VOICE_AVAILABLE = False
    print(f"⚠️ 简化语音系统导入失败: {e}")

# 导入轨迹预测系统
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from trajectory_predictor import TrajectoryPredictor, TrajectoryVisualizer
    TRAJECTORY_PREDICTION_AVAILABLE = True
    print("✅ 轨迹预测系统导入成功")
except ImportError as e:
    TRAJECTORY_PREDICTION_AVAILABLE = False
    print(f"⚠️ 轨迹预测系统导入失败: {e}")

# 导入YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLO导入成功")
except ImportError:
    YOLO_AVAILABLE = False
    print("❌ YOLO导入失败")

class BaiduTTS:
    """百度语音合成"""
    
    def __init__(self, app_id, api_key, secret_key):
        self.app_id = app_id
        self.api_key = api_key
        self.secret_key = secret_key
        self.access_token = None
        self.token_expire_time = 0
        
    def get_access_token(self):
        """获取访问令牌"""
        if self.access_token and time.time() < self.token_expire_time:
            return self.access_token
            
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.secret_key
        }
        
        try:
            response = requests.post(url, params=params)
            result = response.json()
            
            if "access_token" in result:
                self.access_token = result["access_token"]
                self.token_expire_time = time.time() + result["expires_in"] - 60
                print("✅ 百度语音令牌获取成功")
                return self.access_token
            else:
                print(f"❌ 获取百度语音令牌失败: {result}")
                return None
        except Exception as e:
            print(f"❌ 获取百度语音令牌异常: {e}")
            return None
    
    def text_to_speech(self, text, output_file=None):
        """文本转语音"""
        token = self.get_access_token()
        if not token:
            return False
            
        url = "https://tsn.baidu.com/text2audio"
        
        data = {
            "tex": text,
            "tok": token,
            "cuid": "blind_road_detector",
            "ctp": "1",
            "lan": "zh",
            "spd": "5",
            "pit": "5",
            "vol": "5",
            "per": "0"
        }
        
        try:
            response = requests.post(url, data=data)
            
            if response.status_code == 200 and response.headers.get("Content-Type", "").startswith("audio"):
                if output_file is None:
                    output_file = os.path.join(tempfile.gettempdir(), f"blind_road_audio_{uuid.uuid4().hex[:8]}.mp3")
                
                with open(output_file, "wb") as f:
                    f.write(response.content)
                
                print(f"✅ 语音合成成功: {output_file}")
                return output_file
            else:
                print(f"❌ 语音合成失败: {response.text}")
                return False
        except Exception as e:
            print(f"❌ 语音合成异常: {e}")
            return False

class SimpleVoiceSystem:
    """简化语音系统 - 优化版本，支持连续完整播报"""
    
    def __init__(self):
        self.media_player = None
        self.last_speak_text = ""
        self.speak_cooldown = 1.0  # 减少冷却时间
        self.last_speak_time = 0
        self.init_media_player()
        
        # 百度语音配置
        self.baidu_client = None
        self.init_baidu_tts()
        
        # 语音队列管理
        self.voice_queue = []
        self.is_playing = False
        self.voice_lock = threading.Lock()
        
        # 启动语音处理线程
        self.start_voice_processor()
    
    def init_media_player(self):
        """初始化媒体播放器"""
        try:
            self.media_player = QMediaPlayer()
            print("✅ 媒体播放器初始化成功")
        except Exception as e:
            print(f"❌ 媒体播放器初始化失败: {e}")
            self.media_player = None
    
    def init_baidu_tts(self):
        """初始化百度语音"""
        try:
            # 百度语音API配置
            app_id = "119634292"
            api_key = "w978fA2S7PJmUy4IEvlGqxfx"
            secret_key = "ZeTBNN1UYQRL1kaDEEImHm07Y09jgaRc"
            
            self.baidu_client = BaiduTTS(app_id, api_key, secret_key)
            print("✅ 百度语音初始化成功")
        except Exception as e:
            print(f"❌ 百度语音初始化失败: {e}")
            self.baidu_client = None
    
    def start_voice_processor(self):
        """启动语音处理线程"""
        def voice_processor():
            while True:
                try:
                    with self.voice_lock:
                        if self.voice_queue and not self.is_playing:
                            text = self.voice_queue.pop(0)
                            self.is_playing = True
                        else:
                            text = None
                    
                    if text:
                        self.execute_speech(text)
                        with self.voice_lock:
                            self.is_playing = False
                    else:
                        time.sleep(0.1)  # 短暂等待
                        
                except Exception as e:
                    print(f"⚠️ 语音处理线程错误: {e}")
                    with self.voice_lock:
                        self.is_playing = False
        
        voice_thread = threading.Thread(target=voice_processor, daemon=True)
        voice_thread.start()
        print("✅ 语音处理线程已启动")
    
    def speak(self, text, priority=1):
        """语音播报 - 支持优先级和队列管理"""
        if not text:
            return
        
        # 检查冷却时间（根据优先级调整）
        current_time = time.time()
        cooldown = 0.5 if priority >= 3 else 1.0
        if current_time - self.last_speak_time < cooldown:
            return
        
        # 检查是否重复
        if text == self.last_speak_text:
            return
        
        self.last_speak_text = text
        self.last_speak_time = current_time
        
        print(f"🔊 语音播报 (优先级{priority}): {text}")
        
        # 添加到语音队列
        with self.voice_lock:
            if priority >= 3:
                # 高优先级，插入到队列前面
                self.voice_queue.insert(0, text)
            else:
                # 普通优先级，添加到队列末尾
                self.voice_queue.append(text)
    
    def execute_speech(self, text):
        """执行语音播报"""
        try:
            # 使用百度语音合成
            if self.baidu_client:
                audio_file = self.baidu_client.text_to_speech(text)
                if audio_file:
                    self.play_audio(audio_file)
                    return
            
            # 备用方案：使用pyttsx3
            try:
                import pyttsx3
                tts = pyttsx3.init()
                tts.setProperty('rate', 150)
                tts.say(text)
                tts.runAndWait()
                tts.stop()
                print(f"✅ 备用语音播报成功: {text}")
            except Exception as e:
                print(f"⚠️ 备用语音播报失败: {e}")
                
        except Exception as e:
            print(f"❌ 语音播报执行失败: {e}")
    
    def play_audio(self, audio_file):
        """播放音频"""
        if not audio_file or not os.path.exists(audio_file):
            return
            
        try:
            if self.media_player:
                # GUI内嵌播放
                url = QUrl.fromLocalFile(audio_file)
                content = QMediaContent(url)
                self.media_player.setMedia(content)
                self.media_player.play()
                print(f"✅ 音频播放成功: {audio_file}")
            else:
                # 备用方案：系统播放器
                import subprocess
                subprocess.Popen(['start', audio_file], shell=True)
                print(f"✅ 系统播放器播放: {audio_file}")
        except Exception as e:
            print(f"❌ 音频播放失败: {e}")

class MobileDetectWindow(QMainWindow):
    """移动端检测窗口"""
    
    def __init__(self):
        super().__init__()
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # YOLO模型
        self.model = None
        self.init_yolo_model()
        
        # 语音系统
        if SIMPLE_VOICE_AVAILABLE:
            self.voice_system = voice_system
            print("✅ 使用简化语音系统")
        else:
            self.voice_system = SimpleVoiceSystem()
            print("⚠️ 使用备用语音系统")
        
        # 语音库
        self.voice_library = None
        if VOICE_LIBRARY_AVAILABLE:
            self.voice_library = VoiceLibrary()
            print("✅ 语音库初始化成功")
        
        # 状态跟踪
        self.last_distance = None
        self.last_direction = None
        self.last_label = None
        
        # 轨迹预测系统
        if TRAJECTORY_PREDICTION_AVAILABLE:
            self.trajectory_predictor = TrajectoryPredictor()
            self.trajectory_visualizer = TrajectoryVisualizer()
            print("✅ 轨迹预测系统初始化成功")
        else:
            self.trajectory_predictor = None
            self.trajectory_visualizer = None
            print("⚠️ 轨迹预测系统不可用")
        
        self.init_ui()
    
    def init_yolo_model(self):
        """初始化YOLO模型"""
        try:
            model_path = "runs/detect/train5/weights/best.pt"
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                print(f"✅ 加载自定义模型: {model_path}")
            else:
                self.model = YOLO("yolov8n.pt")
                print("✅ 加载默认模型: yolov8n.pt")
        except Exception as e:
            print(f"❌ YOLO模型加载失败: {e}")
            self.model = None
    
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("盲道障碍检测 - 移动端")
        self.setGeometry(100, 100, 800, 600)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        layout = QVBoxLayout(central_widget)
        
        # 摄像头显示区域
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("border: 2px solid gray;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setText("点击开始检测")
        layout.addWidget(self.camera_label)
        
        # 控制按钮
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("开始检测")
        self.start_btn.clicked.connect(self.toggle_camera)
        button_layout.addWidget(self.start_btn)
        
        self.voice_btn = QPushButton("语音开关")
        self.voice_btn.setCheckable(True)
        self.voice_btn.setChecked(True)
        button_layout.addWidget(self.voice_btn)
        
        self.test_btn = QPushButton("测试语音")
        self.test_btn.clicked.connect(self.test_voice)
        button_layout.addWidget(self.test_btn)
        
        layout.addLayout(button_layout)
        
        # 状态显示
        self.status_label = QLabel("准备就绪")
        self.status_label.setStyleSheet("font-size: 14px; color: blue;")
        layout.addWidget(self.status_label)
        
        # 检测信息显示
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("font-size: 12px; color: green;")
        layout.addWidget(self.info_label)
    
    def toggle_camera(self):
        """切换摄像头状态"""
        if self.cap is None or not self.cap.isOpened():
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """启动摄像头"""
        try:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.timer.start(30)  # 30ms间隔
                self.start_btn.setText("停止检测")
                self.status_label.setText("检测中...")
                print("✅ 摄像头启动成功")
            else:
                self.status_label.setText("摄像头启动失败")
                print("❌ 摄像头启动失败")
        except Exception as e:
            self.status_label.setText(f"摄像头错误: {e}")
            print(f"❌ 摄像头错误: {e}")
    
    def stop_camera(self):
        """停止摄像头"""
        if self.timer.isActive():
            self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_btn.setText("开始检测")
        self.status_label.setText("已停止")
        self.camera_label.setText("点击开始检测")
        print("✅ 摄像头已停止")
    
    def update_frame(self):
        """更新摄像头帧"""
        if not self.cap or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # YOLO检测
        if self.model:
            results = self.model(frame, verbose=False)
            current_time = time.time()
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # 获取检测结果
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # 显示所有检测结果，不管置信度
                        print(f"🔍 检测到: {self.get_class_name(cls)} 置信度: {conf:.2f}")
                        
                        if conf > 0.1:  # 极低置信度阈值，确保显示所有检测
                            # 绘制边界框
                            color = self.get_class_color(cls)
                            thickness = max(4, int(6 * conf))  # 进一步增加线条粗细
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                            
                            # 绘制标签背景
                            label_text = f"{self.get_class_name(cls)} {conf:.2f}"
                            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                            
                            # 标签背景
                            cv2.rectangle(frame, (int(x1), int(y1) - label_size[1] - 10), 
                                        (int(x1) + label_size[0] + 10, int(y1)), color, -1)
                            
                            # 绘制标签文本
                            cv2.putText(frame, label_text, (int(x1) + 5, int(y1) - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                            
                            # 计算距离和方向
                            distance = self.estimate_distance(y2 - y1)
                            direction = self.get_direction(x1, x2, frame.shape[1])
                            
                            # 获取标签名称
                            label_name = self.get_class_name(cls)
                            
                            # 绘制标签背景
                            label_text = f"{label_name} {conf:.2f}"
                            distance_text = f"{distance:.1f}m"
                            direction_text = direction
                            
                            # 计算文本尺寸
                            font_scale = 0.6
                            font_thickness = 2
                            (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                            (dist_w, dist_h), _ = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                            
                            # 绘制标签背景
                            label_y = int(y1) - 10
                            if label_y < label_h:
                                label_y = int(y1) + label_h + 10
                            
                            # 标签背景
                            cv2.rectangle(frame, (int(x1), label_y - label_h - 5), 
                                        (int(x1) + max(label_w, dist_w) + 10, label_y + 5), color, -1)
                            
                            # 绘制文本
                            cv2.putText(frame, label_text, (int(x1) + 5, label_y - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
                            
                            # 距离文本
                            cv2.putText(frame, distance_text, (int(x1) + 5, label_y + dist_h), 
                                      cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
                            
                            # 绘制方向指示器
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            self.draw_direction_indicator(frame, center_x, center_y, direction, color)
                            
                            # 绘制置信度条
                            self.draw_confidence_bar(frame, int(x1), int(y1), int(x2), int(y2), conf, color)
                            
                            # 轨迹预测
                            if self.trajectory_predictor:
                                object_id = f"obj_{i}_{cls}_{int(current_time*1000)}"
                                bbox = (int(x1), int(y1), int(x2), int(y2))
                                self.trajectory_predictor.update_trajectory(object_id, bbox, current_time)
                                
                                # 获取轨迹预测
                                prediction = self.trajectory_predictor.predict_trajectory(object_id)
                                if prediction:
                                    # 绘制轨迹预测
                                    frame = self.trajectory_visualizer.draw_trajectory(frame, prediction)
                                    
                                    # 基于轨迹预测的语音播报
                                    if prediction.collision_risk > 0.7:
                                        risk_msg = f"警告！{label_name}轨迹预测显示高风险，建议停止"
                                        if self.voice_btn.isChecked():
                                            self.voice_system.speak_urgent(risk_msg)
                                    elif prediction.collision_risk > 0.4:
                                        risk_msg = f"注意！{label_name}轨迹预测显示中风险，请减速"
                                        if self.voice_btn.isChecked():
                                            self.voice_system.speak(risk_msg)
                                    
                                    # 更新状态显示
                                    self.status_label.setText(f"轨迹预测: 风险{prediction.collision_risk:.2f}, TTC:{prediction.time_to_collision:.1f}s")
                                    
                                    # 清理旧轨迹
                                    self.trajectory_predictor.cleanup_old_trajectories(current_time, max_age=3.0)
                            
                            # 语音播报
                            if self.voice_btn.isChecked():
                                self.on_detected(cls, distance, label_name, direction)
        
        # 转换为Qt格式并显示
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # 缩放以适应显示区域
        scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio)
        self.camera_label.setPixmap(scaled_pixmap)
    
    def get_class_color(self, cls):
        """获取类别颜色"""
        colors = [
            (0, 255, 0),    # 绿色 - 静态障碍
            (255, 0, 0),    # 红色 - 动态障碍
            (0, 0, 255),    # 蓝色 - 地面异常
            (255, 255, 0),  # 黄色 - 环境声音
        ]
        return colors[cls % len(colors)]
    
    def get_class_name(self, cls):
        """获取类别名称"""
        names = ["车辆", "行人", "坑洼", "摊位", "垃圾桶", "宠物", "台阶", "斜坡", "积水", "家具", "移动车辆"]
        return names[cls] if cls < len(names) else f"障碍物{cls}"
    
    def estimate_distance(self, box_height):
        """估算距离（基于边界框高度）"""
        # 简化的距离估算
        focal_length = 1000  # 焦距
        real_height = 1.0    # 假设真实高度1米
        distance = (focal_length * real_height) / box_height
        return max(0.1, min(10.0, distance))  # 限制在0.1-10米范围内
    
    def get_direction(self, x1, x2, frame_width):
        """获取方向"""
        center_x = (x1 + x2) / 2
        if center_x < frame_width * 0.4:
            return "左侧"
        elif center_x > frame_width * 0.6:
            return "右侧"
        else:
            return "正前方"
    
    def draw_direction_indicator(self, frame, center_x, center_y, direction, color):
        """绘制方向指示器"""
        arrow_size = 15
        if direction == "左侧":
            # 左箭头
            points = np.array([
                [center_x - arrow_size, center_y],
                [center_x, center_y - arrow_size//2],
                [center_x, center_y + arrow_size//2]
            ], np.int32)
        elif direction == "右侧":
            # 右箭头
            points = np.array([
                [center_x + arrow_size, center_y],
                [center_x, center_y - arrow_size//2],
                [center_x, center_y + arrow_size//2]
            ], np.int32)
        else:
            # 上箭头（正前方）
            points = np.array([
                [center_x, center_y - arrow_size],
                [center_x - arrow_size//2, center_y],
                [center_x + arrow_size//2, center_y]
            ], np.int32)
        
        cv2.fillPoly(frame, [points], color)
    
    def draw_confidence_bar(self, frame, x1, y1, x2, y2, confidence, color):
        """绘制置信度条"""
        bar_width = 4
        bar_height = int((y2 - y1) * confidence)
        bar_x = x2 + 5
        bar_y = y2 - bar_height
        
        # 绘制背景
        cv2.rectangle(frame, (bar_x, y1), (bar_x + bar_width, y2), (100, 100, 100), -1)
        
        # 绘制置信度条
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, y2), color, -1)
        
        # 绘制置信度文本
        conf_text = f"{confidence:.2f}"
        cv2.putText(frame, conf_text, (bar_x + bar_width + 2, bar_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def on_detected(self, cls, distance, label_name, direction):
        """检测到障碍物时的处理"""
        try:
            print(f"🔍 检测到障碍物: {label_name}, 距离: {distance:.1f}米, 方位: {direction}")
            
            # 检查是否有显著变化
            distance_changed = (self.last_distance is None or 
                              abs(distance - self.last_distance) > 0.5)
            direction_changed = (self.last_direction != direction)
            label_changed = (self.last_label != label_name)
            
            should_speak = distance_changed or direction_changed or label_changed
            
            # 直接使用修复的语音系统
            if should_speak:
                if distance > 5:
                    self.status_label.setText("")
                    print("📏 距离过远，不播报")
                elif distance > 3:
                    msg = f"前方{distance:.1f}米{direction}有{label_name}"
                    self.status_label.setText(msg)
                    self.info_label.setText(f"检测: {label_name} | 距离: {distance:.1f}m | 方向: {direction}")
                    print(f"📢 语音提示: {msg}")
                    self.voice_system.speak(msg)
                elif distance > 1:
                    msg = f"{direction}{distance:.1f}米有{label_name}，请减速"
                    self.status_label.setText(msg)
                    self.info_label.setText(f"检测: {label_name} | 距离: {distance:.1f}m | 方向: {direction}")
                    print(f"📢 语音提示: {msg}")
                    self.voice_system.speak(msg)
                else:
                    msg = f"危险！立即停止！{direction}{distance:.1f}米有{label_name}"
                    self.status_label.setText(msg)
                    self.info_label.setText(f"检测: {label_name} | 距离: {distance:.1f}m | 方向: {direction}")
                    print(f"📢 语音提示: {msg}")
                    self.voice_system.speak_urgent(msg)
            
            # 更新状态
            self.last_distance = distance
            self.last_direction = direction
            self.last_label = label_name
            
        except Exception as e:
            print(f"❌ 处理检测结果时出错: {e}")
    
    def test_voice(self):
        """测试语音"""
        test_msg = "语音系统测试成功"
        self.status_label.setText(test_msg)
        self.voice_system.speak(test_msg)
        print(f"🔊 语音测试: {test_msg}")
    
    def closeEvent(self, event):
        """关闭事件"""
        self.stop_camera()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = MobileDetectWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 